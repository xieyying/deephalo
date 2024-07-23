import pyopenms as oms
import pandas as pd
import numpy as np
from scipy.spatial import KDTree
import tensorflow as tf


def set_para(fun_para,para_list):
    """
    Set the parameters for the feature detection process
    """
    for i in range(len(para_list)):
        fun_para.setValue(para_list[i][0], para_list[i][1])
    return fun_para


class FeatureDetection:
    """
    Extract features from mzML files using pyopenms
    """
    def __init__(self, file, pars):
        self.file = file
        self.pars = pars
        self.exp = oms.MSExperiment()   
        self.mass_traces = []
        self.mass_traces_deconvol = []
        self.feature_map = oms.FeatureMap()
        self.chrom_out = []

    def load_file(self):
        """Load the mzML file"""
        oms.MzMLFile().load(self.file, self.exp)

    def get_dataframes(self):
        """Get the ion dataframes and filter them"""
        self.ion_df = self.exp.get_ion_df()
        filter_threshold = next((item[1] for item in self.pars.mass_trace_detection if item[0] == 'noise_threshold_int'), None)
        self.ion_df = self.ion_df[self.ion_df['inty'] > filter_threshold] 

    def mass_trace_detection(self):
        """Detect the mass traces"""
        mtd = oms.MassTraceDetection()
        mtd_par = mtd.getDefaults()
        set_para(mtd_par,self.pars.mass_trace_detection)
        mtd.setParameters(mtd_par)
        mtd.run(self.exp, self.mass_traces, 0)


    def elution_peak_detection(self):
        """Detect the elution peaks"""
        epd = oms.ElutionPeakDetection()
        epd_par = epd.getDefaults()
        set_para(epd_par,self.pars.elution_peak_detection)
        epd.setParameters(epd_par)
        epd.detectPeaks(self.mass_traces, self.mass_traces_deconvol)    

    def feature_detection(self):
        """Detect the features"""
        ffm = oms.FeatureFindingMetabo()
        ffm_par = ffm.getDefaults()
        set_para(ffm_par,self.pars.feature_detection)
        ffm.setParameters(ffm_par)
        ffm.run(self.mass_traces_deconvol, self.feature_map, self.chrom_out)
        self.feature_map.setUniqueIds()
        self.feature_map.setPrimaryMSRunPath([self.file.encode()])
    
    def run(self):
        """Run the feature detection process"""
        self.load_file()
        self.get_dataframes()
        self.mass_trace_detection()
        self.elution_peak_detection()
        self.feature_detection()
        return self

class FeatureMapProcessor(FeatureDetection):
    """
    Extract isotope patterns based on the featuremap information and original data output by pyopenms。
    These isotope patterns include isotope patterns based on features and isotope patterns for each scan within the feature
    
    """
    def __init__(self, file, pars):
        super().__init__(file, pars)
        self.df_feature = pd.DataFrame()
        self.df_feature_flatten = pd.DataFrame()
        self.charge_f = []
        self.feature_id = []
        self.feature_id_flatten = []
        self.masstrace_intensity = []
        self.masstrace_centroid_mz = []
        self.masstrace_lable = []

    def _process_feature(self, feature):
        """Process each feature"""
        self.masstrace_intensity.append(feature.getMetaValue("masstrace_intensity") )
        self.masstrace_centroid_mz.append((feature.getMetaValue("masstrace_centroid_mz")) )
        self.masstrace_lable.append(feature.getMetaValue("label").split("_")) 
        self.feature_id.append(str(feature.getUniqueId()))
        convex_hulls = feature.getConvexHulls()
        self._process_convex_hulls(convex_hulls, feature)

    def _process_convex_hulls(self, convex_hulls, feature):
        """Process each convex hull"""
        for i in range(len(convex_hulls)):
            if i ==0:
                df_trace1 = pd.DataFrame(convex_hulls[i].getHullPoints(),columns=['rt','mz_m0'])
                df_trace1 = df_trace1.drop_duplicates(subset=['rt'],keep='first').sort_values(by='rt')
            else:
                hull = convex_hulls[i]
                hull_points = hull.getHullPoints()
                hull_points = np.array(hull_points)
                df_trace = pd.DataFrame(hull_points,columns=['rt',f'mz_m{i}'])
                df_trace = df_trace.drop_duplicates(subset=['rt'],keep='first').sort_values(by='rt')
                df_trace1 = pd.merge(df_trace1, df_trace, on='rt', how='outer')
                df_trace1 = df_trace1.dropna()
        df_trace1 = df_trace1.melt(id_vars='rt', var_name='mz_type', value_name='mz')
        self.df_feature_flatten = pd.concat([self.df_feature_flatten,df_trace1])
        self.charge_f.extend ([feature.getCharge() for _ in range(len(df_trace1))])
        self.feature_id_flatten.extend ([str(feature.getUniqueId()) for _ in range(len(df_trace1))])

    def _transform_to_dataframe(self):
        """Transform the data to a dataframe"""
        self.df_feature_flatten['charge_f'] = self.charge_f
        self.df_feature_flatten['feature_id_flatten'] = self.feature_id_flatten
        self.df_feature["feature_id"] = self.feature_id
        self.df_feature["masstrace_intensity"] = self.masstrace_intensity
        self.df_feature["masstrace_centroid_mz"] = self.masstrace_centroid_mz

    def _merge_feature_df(self):
        """Merge the feature dataframes"""
        self.df_feature = pd.merge(self.df_feature, self.df_feature_, left_on='feature_id', right_on='feature_id', how='left')

    def _add_intensity(self):
        """Add intensity to the dataframe"""
        mz_error = 0.01  # 为了保证寻找的数据正确，给一个很小的数，固定即可，不需要变动
        rt_error = 0.01  # 为了保证寻找的数据正确，给一个很小的数，固定即可，不需要变动
        tree = KDTree(self.ion_df[['RT', 'mz']].values)
        distances, indices = tree.query(self.df_feature_flatten[['rt', 'mz']], distance_upper_bound=np.sqrt(mz_error**2 + rt_error**2 ))
        valid_indices = indices[distances != np.inf]
        self.df_feature_flatten['inty'] = self.ion_df.iloc[valid_indices]['inty'].values
        self.df_feature_flatten['mz_raw_data'] = self.ion_df.iloc[valid_indices]['mz'].values  # 只是用于过程中检查找到的数据对不对，目前核对正确，正式版可以删除
        self.df_feature_flatten['rt_raw_data'] = self.ion_df.iloc[valid_indices]['RT'].values  ## 只是用于过程中检查找到的数据对不对，目前核对正确，正式版可以删除

    def _grouping_df_based_on_feature_plus_scan(self):
        """Group the dataframe based on FeaturePlusScan"""
        self.df_scan = self.df_feature_flatten.groupby(['feature_id_flatten', 'rt']).apply(lambda x: pd.Series({
            'mz_list': x.sort_values('mz_type')['mz'].tolist(),
            'inty_list': x.sort_values('mz_type')['inty'].tolist(),
            'charge': x.sort_values('mz_type')['charge_f'].tolist()[0],
        })).reset_index()
        self.df_scan['mz_list'], self.df_scan['inty_list'] = zip(*self.df_scan.apply(lambda row: zip(*sorted(zip(row['mz_list'], row['inty_list']))), axis=1))
        mz =[i[0] for i in self.df_scan['mz_list']]
        self.df_scan['mz'] = mz*self.df_scan['charge']-(self.df_scan['charge']-1)*1.007276466812
        
    def process(self):
        """Process the feature map"""
        self.df_feature_ = self.feature_map.get_df()
        self.df_feature_ = self.df_feature_.reset_index()
        for feature in self.feature_map:
            if feature.getMetaValue("num_of_masstraces") >=self.pars.FeatureMapProcessor_min_num_of_masstraces and feature.getIntensity() >= self.pars.FeatureMapProcessor_min_feature_int:
                self._process_feature(feature)
        self._transform_to_dataframe()
        self._merge_feature_df()
        self._add_intensity()
        self._grouping_df_based_on_feature_plus_scan()
        return self.df_feature, self.df_scan  
        # self.df_feature 包括以feature 为基础的isotope patterns（mz为controid mz，intensity为总和（但具体细节待确认）
        # self.df_scan 包括以scan 为基础的isotope patterns (intensity来自原始数据mz来自featuremap.convexHulls)
        # 这两个表格可以直接变换为model的输入形式，分别得出以feature整体为基础的预测结果，以及以scan为基础的预测结果
        # 最后可以根据feature_id综合连个结果计算H_score



    