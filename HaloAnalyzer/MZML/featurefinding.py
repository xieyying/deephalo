import pyopenms as oms
import pandas as pd
import numpy as np
from scipy.spatial import KDTree
import tensorflow as tf
from collections import Counter
import pyteomics.mgf as mgf

class FeatureDetection:
    """
    Extract features from mzML files using pyopenms
    """
    def __init__(self, file):
        self.file = file
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
        self.ion_df = self.ion_df[self.ion_df['inty'] > 200.0]
        self.ion_df.to_csv(r'C:\Users\xyy\Desktop\test\tem\ion_df.csv', index=False)
        

    def mass_trace_detection(self):
        """Detect the mass traces"""
        mtd = oms.MassTraceDetection()
        mtd_par = mtd.getDefaults()
        print('mass_trace_detection parameters:')
        print(mtd_par)
        mtd_par.setValue("mass_error_ppm", 20.0)
        mtd_par.setValue('min_trace_length', 3.0)
        mtd_par.setValue("noise_threshold_int", 200.0)
        mtd_par.setValue("chrom_peak_snr",3.0)
        # mtd_par.setValue('quant_method', 'median')
        mtd_par.setValue('reestimate_mt_sd', 'true')
        mtd_par.setValue('trace_termination_outliers', 5)
        mtd_par.setValue('trace_termination_criterion', 'outlier')
        #In 'sample_rate' mode, trace extension in both directions stops if ratio of found peaks versus visited spectra falls below the 'min_sample_rate' threshold."
        mtd_par.setValue('min_sample_rate', 0.5)
        #'Minimum fraction of scans along the mass trace that must contain a peak
        
        mtd.setParameters(mtd_par)
        mtd.run(self.exp, self.mass_traces, 0)
        print(len(self.mass_traces))
        # for i in range(len(self.mass_traces)):
        #     trace = self.mass_traces[i]
        #     print(f"Trace {i}:")
        #     print(f"Centroid MZ: {trace.getCentroidMZ()}")
        #     print(f"Centroid RT: {trace.getCentroidRT()}")
        #     # print(f"Max Intensity: {trace.getMaxIntensity()}")
        #     print(f"Size: {trace.getSize()}")
        #     print("--------------------")    

    def elution_peak_detection(self):
        """Detect the elution peaks"""
        epd = oms.ElutionPeakDetection()
        epd_par = epd.getDefaults()
        print('elution_peak_detection parameters:')
        print(epd_par)
        epd_par.setValue("width_filtering", "fixed")
        epd_par.setValue('chrom_fwhm', 3.0)

        epd.setParameters(epd_par)
        epd.detectPeaks(self.mass_traces, self.mass_traces_deconvol)
        print(len(self.mass_traces_deconvol))
        # for i in range(len(self.mass_traces_deconvol)):
        #     trace = self.mass_traces_deconvol[i]
        #     print(f"Trace {i}:")
        #     print(f"Centroid MZ: {trace.getCentroidMZ()}")
        #     print(f"Centroid RT: {trace.getCentroidRT()}")
        #     # print(f"Max Intensity: {trace.getMaxIntensity()}")
        #     print(f"Size: {trace.getSize()}")
        #     print("--------------------")   


    def feature_detection(self):
        """Detect the features"""
        ffm = oms.FeatureFindingMetabo()
        ffm_par = ffm.getDefaults()
        print('feature_detection parameters:')
        print(ffm_par)
        ffm_par.setValue("local_rt_range", 10.0)
        ffm_par.setValue("local_mz_range", 12.0)
        ffm_par.setValue("chrom_fwhm", 3.0)
        ffm_par.setValue('enable_RT_filtering', 'true')
        ffm_par.setValue('mz_scoring_by_elements', 'true')
        ffm_par.setValue("elements", "CHNOPSClBrFeBSeIF")
        # ffm_par.setValue("report_chromatograms", 'true')
        ffm_par.setValue('isotope_filtering_model','none')
        ffm_par.setValue("remove_single_traces", "true")
        ffm_par.setValue("report_convex_hulls", 'true')
        ffm.setParameters(ffm_par)
        ffm.run(self.mass_traces_deconvol, self.feature_map, self.chrom_out)
        self.feature_map.setUniqueIds()
        self.feature_map.setPrimaryMSRunPath([self.file.encode()])
        # print(len(self.feature_map))
        print(self.feature_map.get_df())
        print(len(self.feature_map.get_df()))
        # 提取xml文件
        oms.FeatureXMLFile().store(r'C:\Users\xyy\Desktop\test\tem\feature_map.featureXML', self.feature_map)
        self.feature_map.get_df().to_csv(r'C:\Users\xyy\Desktop\test\tem\feature_map.csv', index=False)

        
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
    def __init__(self, file):
        super().__init__(file)
        self.df_feature = pd.DataFrame()
        self.df_feature_flatten = pd.DataFrame()
        self.charge_f = []
        self.feature_id = []
        self.feature_id_flatten = []
        self.masstrace_intensity = []
        self.masstrace_centroid_mz = []
        self.masstrace_lable = []

    def process(self):
        """Process the feature map"""
        self.df_feature_ = self.feature_map.get_df()
        self.df_feature_ = self.df_feature_.reset_index()
        for feature in self.feature_map:
            if feature.getMetaValue("num_of_masstraces") >2 and feature.getIntensity() > 1000:
                self._process_feature(feature)
        self._transform_to_dataframe()
        self._merge_feature_df()
        self._add_intensity()
        self._grouping_df_based_on_FeaturePlusScan()
        return self.df_feature, self.df_scan  
        # self.df_feature 包括以feature 为基础的isotope patterns（mz为controid mz，intensity为总和（但具体细节待确认）
        # self.df_scan 包括以scan 为基础的isotope patterns (intensity来自原始数据mz来自featuremap.convexHulls)
        # 这两个表格可以直接变换为model的输入形式，分别得出以feature整体为基础的预测结果，以及以scan为基础的预测结果
        # 最后可以根据feature_id综合连个结果计算H_score

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
        # print("________________")
        # print(self.df_feature_flatten)
        """Add intensity to the dataframe"""
        mz_error = 0.1
        rt_error = 0.1
        tree = KDTree(self.ion_df[['RT', 'mz']].values)
        distances, indices = tree.query(self.df_feature_flatten[['rt', 'mz']], distance_upper_bound=np.sqrt(mz_error**2 + rt_error**2 ))
        valid_indices = indices[distances != np.inf]
        self.df_feature_flatten['inty'] = self.ion_df.iloc[valid_indices]['inty'].values
        self.df_feature_flatten['mz_raw_data'] = self.ion_df.iloc[valid_indices]['mz'].values  # 只是用于过程中检查找到的数据对不对，目前核对正确，正式版可以删除
        self.df_feature_flatten['rt_raw_data'] = self.ion_df.iloc[valid_indices]['RT'].values  ## 只是用于过程中检查找到的数据对不对，目前核对正确，正式版可以删除

    def _grouping_df_based_on_FeaturePlusScan(self):
        """Group the dataframe based on FeaturePlusScan"""
        self.df_scan = self.df_feature_flatten.groupby(['feature_id_flatten', 'rt']).apply(lambda x: pd.Series({
            'mz_list': x.sort_values('mz_type')['mz'].tolist(),
            'inty_list': x.sort_values('mz_type')['inty'].tolist(),
            'charge': x.sort_values('mz_type')['charge_f'].tolist()[0],
        })).reset_index()
        #将mz_list和inty_list按照mz的大小排序
        self.df_scan['mz_list'], self.df_scan['inty_list'] = zip(*self.df_scan.apply(lambda row: zip(*sorted(zip(row['mz_list'], row['inty_list']))), axis=1))


def featureFinding(file):
    """
    Find features from mzML data
    return two DataFrame. One containing feature based isotope patterns. The other contain scan based isotope patterns
    """
    df = FeatureMapProcessor(file)
    feature = df.run()
    df_f,df_scan = feature.process()
    return df_f,df_scan

def isotope_processing(df, mz_list_name = 'mz_list', inty_list_name = "inty_list"):
    """
    Process DataFrame and make it ready for halo model inputs
    """
    # get the mz_list and inty_list
    mz_list = df[mz_list_name].values
    m2_m1 = [i[2] - i[1] for i in mz_list]
    m1_m0 = [i[1] - i[0] for i in mz_list]
    
    # Ensure all lists in inty_list have 7 elements
    inty_list = [list(i) + [0]*(7-len(i)) for i in df[inty_list_name].tolist()]
    
    # Convert inty_list to a DataFrame
    inty_df = pd.DataFrame(inty_list)
    
    # Normalize each row by its max value
    inty_df = inty_df.div(inty_df.max(axis=1), axis=0)
    
    # Assign new columns to df
    df['m2_m1'] = m2_m1*df['charge']
    df['m1_m0'] = m1_m0*df['charge']
    for i in range(7):
        df[f'p{i}_int'] = inty_df[i].values
    return df

def add_predict(df,model_path,features_list):
    """
    Add prediction result based on DNN Halo model
    """
    # Load the TensorFlow model
    clf = tf.keras.models.load_model(model_path)
    # Load the features
    querys = df[features_list].values
    querys = querys.astype('float32')
    # Predict the features
    res = clf.predict(querys)
    classes_pred = np.argmax(res, axis=1)
    # Add the prediction results to df_features
    df.loc[:, 'class_pred'] = classes_pred
    return df

def calculate_zig_zag(I):
    """
    Calculate the ZigZag score based on the classification results of all scans in an ROI
    """
    # Calculate the maximum and minimum values of I
    Imax= max(I)
    Imin = min(I)
    N = len(I) 
    total = 0
    # Calculate the ZigZag score for I
    for n in range(1,N-1):
        term = (2 * I[n] - I[n - 1] - I[n + 1])**2 
        total += term
    zigzag = total/(N*(Imax-Imin)**2)
    # Convert the ZigZag score to a percentage
    score = (4-8/N-zigzag)/(4-8/N)*100
    return score

def roi_scan_based_halo_evaluation(I):
    """
    Determine the probability of an ROI being a halo based on the classification results of all scans in the ROI
    """
    # Get the common classes in the ROI
    com_class = list(Counter(I).keys())
    counter = Counter(I)
    # Calculate the ratio of 0,1,2 in I
    scan_based_halo_ratio = sum(1 for i in I if i in {0, 1, 2}) / len(I)

    # Determine the halo classification for the ROI
    if any(i in com_class for i in [0, 1, 2]):
        scan_based_halo_class = 'halo'
        if len(com_class) == 1:
            scan_based_halo_score = 100
            scan_based_halo_sub_score = 100
            scan_based_halo_sub_class = com_class[0]
        else:
            if {0, 1, 2}.issuperset(set(com_class)):
                scan_based_halo_score = 100
                scan_based_halo_sub_class =max(counter.items(), key=lambda x: x[1])[0]
                scan_based_halo_sub_class_ratio = counter[scan_based_halo_sub_class] / len(I)
                if len(I) > 2:
                    scan_based_halo_sub_score = calculate_zig_zag(I) * scan_based_halo_sub_class_ratio
                else:
                    scan_based_halo_sub_score = scan_based_halo_ratio
            else:
                I_new = [1 if i in [0,1,2] else 0 for i in I]
                if len(I) > 2:
                    scan_based_halo_score = calculate_zig_zag(I_new) * scan_based_halo_ratio
                else:
                    scan_based_halo_score = scan_based_halo_ratio
                scan_based_halo_sub_class = "None"
                scan_based_halo_sub_score = "None"
    else:
        scan_based_halo_class = 'non-halo'
        scan_based_halo_score = 0
        scan_based_halo_sub_class = 'None'
        scan_based_halo_sub_score = 'None'

    return scan_based_halo_class,scan_based_halo_score,scan_based_halo_sub_class,scan_based_halo_sub_score,scan_based_halo_ratio

def haloEvalution(df_f,df_scan):
    """
    Evaluate the probability of features based on both feature isotope patterns and scan based isotope patterns
    """
    for i in df_scan['feature_id_flatten'].unique():
        I = df_scan[df_scan['feature_id_flatten'] == i]['class_pred'].tolist()
        scan_based_halo_class,scan_based_halo_score,scan_based_halo_sub_class,scan_based_halo_sub_score,scan_based_halo_ratio = roi_scan_based_halo_evaluation(I)
        
        df_scan.loc[df_scan['feature_id_flatten'] == i,'scan_based_halo_class'] = scan_based_halo_class
        df_scan.loc[df_scan['feature_id_flatten'] == i,'scan_based_halo_score'] = scan_based_halo_score
        df_scan.loc[df_scan['feature_id_flatten'] == i,'scan_based_halo_sub_class'] = scan_based_halo_sub_class
        df_scan.loc[df_scan['feature_id_flatten'] == i,'scan_based_halo_sub_score'] = scan_based_halo_sub_score
        df_scan.loc[df_scan['feature_id_flatten'] == i,'scan_based_halo_ratio'] = scan_based_halo_ratio
        
        df_f.loc[df_f['feature_id'] == i,'scan_based_halo_class'] = scan_based_halo_class
        df_f.loc[df_f['feature_id'] == i,'scan_based_halo_score'] = scan_based_halo_score
        df_f.loc[df_f['feature_id'] == i,'scan_based_halo_sub_class'] = scan_based_halo_sub_class
        df_f.loc[df_f['feature_id'] == i,'scan_based_halo_sub_score'] = scan_based_halo_sub_score
        df_f.loc[df_f['feature_id'] == i,'scan_based_halo_ratio'] = scan_based_halo_ratio
        
    feature_based_halo_score = df_f['class_pred'].apply(lambda x: 1 if x in [0,1,2] else 0)
    df_f['feature_based_halo_score'] = feature_based_halo_score
    df_f['H_score'] = (df_f['scan_based_halo_score'])/300 + (df_f['scan_based_halo_ratio'])/3 + (df_f['feature_based_halo_score'])/3 
    
    return df_f,df_scan

if __name__ == '__main__':
    feature_list = [
        "p0_int",
        "p1_int",
        "p2_int",
        "p3_int",
        "p4_int",
        "p5_int",
        "m2_m1",
        "m1_m0",        
    ]

    file = r'D:\workissues\manuscript\halo_mining\HaloAnalyzer\Simulated_LC_MS\LC_MSMS_data_from_papers\1820_molecules_for_smiter.mzML'

    model_path = r'C:\Users\xyy\Desktop\python\HaloAnalyzer_training\022_six_dataset_openms\trained_models\pick_halo_ann.h5'

    df_f,df_scan = featureFinding(file)
    df_feature_for_model_input = isotope_processing(df_f,'masstrace_centroid_mz','masstrace_intensity')
    # print(df_feature_for_model_input)
    df_scan_for_model_input =isotope_processing(df_scan,'mz_list','inty_list')
    # print(df_scan_for_model_input)
    
    df_f = add_predict(df_feature_for_model_input,model_path, feature_list)
    df_scan = add_predict(df_scan_for_model_input,model_path, feature_list)
    # print(df_f)
    # print(df_scan)
    
    df_f_result,df_scan_result = haloEvalution(df_f,df_scan)
    
    # # save some pseudo spectra
    # mgf_spectra = []
    # df_scan_result['m0'] = df_scan_result['mz_list'].apply(lambda x: x[0])
   
    # df_for_mgf = df_scan_result[(df_scan_result['m0'] >839.1) & (df_scan_result['m0'] < 839.5)]
    # print('len',len(df_for_mgf))
    # for i in range(len(df_for_mgf)):
       
    #     spectrum_ = {}
    #     mz = df_for_mgf.iloc[i]['mz_list']
    #     inty = df_for_mgf.iloc[i]['inty_list']
    #     charge = df_for_mgf.iloc[i]['charge']
    #     spectrum_['params'] = {'pepmass': mz[0], 'charge': 1,'formula':'C10H10O2','compound_name':'pseudo_spectra'}
    #     spectrum_['m/z array'] = np.array(mz)
    #     spectrum_['intensity array'] = np.array(inty)
    #     mgf_spectra.append(spectrum_)
    # mgf.write(mgf_spectra, r'C:\Users\xyy\Desktop\test\tem\pseudo_spectra.mgf')
    
    
    df_f_result.to_csv(r'C:\Users\xyy\Desktop\test\tem\df_f_result.csv', index=False)
    df_scan_result.to_csv(r'C:\Users\xyy\Desktop\test\tem\df_scan_result.csv', index=False)
    # print(df_f)
    print( "----------------")
    