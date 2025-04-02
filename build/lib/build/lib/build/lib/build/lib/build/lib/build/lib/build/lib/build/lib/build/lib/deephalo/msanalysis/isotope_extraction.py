import concurrent.futures
import pyopenms as oms
import pandas as pd
import numpy as np
from scipy.spatial import KDTree
import tensorflow as tf

class FeatureDetection:
    """Extract features from mzML files using pyopenms"""
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
        set_para(mtd_par, self.pars.mass_trace_detection)
        mtd.setParameters(mtd_par)
        mtd.run(self.exp, self.mass_traces, 0)

    def elution_peak_detection(self):
        """Detect the elution peaks"""
        epd = oms.ElutionPeakDetection()
        epd_par = epd.getDefaults()
        set_para(epd_par, self.pars.elution_peak_detection)
        epd.setParameters(epd_par)
        epd.detectPeaks(self.mass_traces, self.mass_traces_deconvol)

    def feature_detection(self):
        """Detect the features"""
        ffm = oms.FeatureFindingMetabo()
        ffm_par = ffm.getDefaults()
        set_para(ffm_par, self.pars.feature_detection)
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
        print('feature detection is done')
        return self

class FeatureMapProcessor(FeatureDetection):
    """Extract isotope patterns based on the featuremap information and original data output by pyopenms"""
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
        self.average_intensity = []
        self.df_feature_hulls_intensities = []

    def _process_feature(self, feature,label):
        """Process each feature"""
        masstrace_intensity = feature.getMetaValue("masstrace_intensity")[label:]
        masstrace_centroid_mz = feature.getMetaValue("masstrace_centroid_mz")[label:]
        masstrace_lable = feature.getMetaValue("label").split("_")
        feature_id = str(feature.getUniqueId())
        convex_hulls = feature.getConvexHulls()[label:]
        df_feature_flatten, charge_f, feature_id_flatten = self._process_convex_hulls(convex_hulls, feature)
        return masstrace_intensity, masstrace_centroid_mz, masstrace_lable, feature_id, df_feature_flatten, charge_f, feature_id_flatten

    def _process_convex_hulls(self, convex_hulls, feature):
        """Process each convex hull"""
        df_trace1 = None
        for i in range(len(convex_hulls)):
            if i == 0:
                df_trace1 = pd.DataFrame(convex_hulls[i].getHullPoints(), columns=['RT', 'mz_m0'])
                df_trace1 = df_trace1.drop_duplicates(subset=['RT'], keep='first').sort_values(by='RT')
            else:
                hull = convex_hulls[i]
                hull_points = hull.getHullPoints()
                hull_points = np.array(hull_points)
                df_trace = pd.DataFrame(hull_points, columns=['RT', f'mz_m{i}'])
                df_trace = df_trace.drop_duplicates(subset=['RT'], keep='first').sort_values(by='RT')
                df_trace1 = pd.merge(df_trace1, df_trace, on='RT', how='outer')
                df_trace1 = df_trace1.dropna()
        df_trace1 = df_trace1.melt(id_vars='RT', var_name='mz_type', value_name='mz')
        charge_f = [feature.getCharge() for _ in range(len(df_trace1))]
        feature_id_flatten = [str(feature.getUniqueId()) for _ in range(len(df_trace1))]
        return df_trace1, charge_f, feature_id_flatten

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
        distances, indices = tree.query(self.df_feature_flatten[['RT', 'mz']], distance_upper_bound=np.sqrt(mz_error**2 + rt_error**2))
        valid_indices = indices[distances != np.inf]
        self.df_feature_flatten['inty'] = self.ion_df.iloc[valid_indices]['inty'].values
        self.df_feature_flatten['mz_raw_data'] = self.ion_df.iloc[valid_indices]['mz'].values  # 只是用于过程中检查找到的数据对不对，目前核对正确，正式版可以删除
        self.df_feature_flatten['rt_raw_data'] = self.ion_df.iloc[valid_indices]['RT'].values  ## 只是用于过程中检查找到的数据对不对，目前核对正确，正式版可以删除

    def _grouping_df_based_on_feature_plus_scan(self):
        """Group the dataframe based on FeaturePlusScan"""
        # Group by 'feature_id_flatten' and 'RT'
        grouped = self.df_feature_flatten.groupby(['feature_id_flatten', 'RT'])
        
        # Aggregate the required columns
        agg_df = grouped.agg({
            'mz': list,
            'inty': list,
            'charge_f': 'first'
        }).reset_index()
        
        # Sort mz and inty lists
        agg_df['mz_list'] = agg_df['mz'].apply(lambda x: np.array(x))
        agg_df['inty_list'] = agg_df['inty'].apply(lambda x: np.array(x))
        
        # Sort mz_list and inty_list together
        agg_df['sorted_mz_inty'] = agg_df.apply(lambda row: sorted(zip(row['mz_list'], row['inty_list'])), axis=1)
        agg_df['mz_list'] = agg_df['sorted_mz_inty'].apply(lambda x: [i[0] for i in x])
        agg_df['inty_list'] = agg_df['sorted_mz_inty'].apply(lambda x: [i[1] for i in x])
        
        # Extract charge
        agg_df['charge'] = agg_df['charge_f']
        
        # Drop intermediate columns
        agg_df = agg_df.drop(columns=['mz', 'inty', 'sorted_mz_inty', 'charge_f'])
        
        self.df_scan = agg_df
        mz = [i[0] for i in self.df_scan['mz_list']]
        self.df_scan['mz'] = mz * self.df_scan['charge'] - (self.df_scan['charge'] - 1) * 1.007276466812

    def process_feature_batch(self, features, labels):
        """Process a batch of features and return the results"""
        batch_results = []
        for i in range(len(features)):
            if labels[i] is not None:
                result = self._process_feature(features[i], labels[i])
                batch_results.append(result)
                
        return batch_results

    def process(self):
        """Process the feature map"""
        self.df_feature_ = self.feature_map.get_df()
        self.df_feature_ = self.df_feature_.reset_index()
        
        print('feature_map:', len(self.df_feature_))
        # Filter features first
        filtered_features = [
            feature for feature in self.feature_map
            if feature.getMetaValue("num_of_masstraces") >= 3 and
               feature.getIntensity() >= self.pars.FeatureMapProcessor_min_feature_int and
               len(feature.getConvexHulls()[0].getHullPoints()) > self.pars.FeatureMapProcessor_min_scan_number and 
                ((feature.getMetaValue("masstrace_centroid_mz")[0])*(feature.getCharge()) <= 2000)]
        print(self.pars.FeatureMapProcessor_use_mass_difference)
        if self.pars.FeatureMapProcessor_use_mass_difference == 'none': 
            labels = [0 for _ in filtered_features]
        else:
            labels = [feature_classifier(feature.getMetaValue("masstrace_centroid_mz"), feature.getCharge()) for feature in filtered_features]
            print('labels:', len(labels))

            
        
        iso_1_2_features = [
            (str(feature.getUniqueId()), feature.getMetaValue("masstrace_intensity"), feature.getMetaValue("masstrace_centroid_mz")) for feature in self.feature_map
            if 3>feature.getMetaValue("num_of_masstraces") >= self.pars.FeatureMapProcessor_min_num_of_masstraces and
               feature.getIntensity() >= self.pars.FeatureMapProcessor_min_feature_int and
               len(feature.getConvexHulls()[0].getHullPoints()) > self.pars.FeatureMapProcessor_min_scan_number
        ]
    
        # Create a DataFrame with feature_id and masstrace_intensity
        iso_1_2_features_df = pd.DataFrame(iso_1_2_features, columns=['feature_id', 'masstrace_intensity', 'masstrace_centroid_mz'])

        # Filter the original DataFrame and merge with the new DataFrame
        self.iso_12_features_df = self.df_feature_[self.df_feature_['feature_id'].isin(iso_1_2_features_df['feature_id'])]
        self.iso_12_features_df = self.iso_12_features_df.merge(iso_1_2_features_df, on='feature_id', how='left')

        if filtered_features == [] and len(iso_1_2_features) == 0:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        elif filtered_features == [] and len(iso_1_2_features) != 0:
            return pd.DataFrame(), pd.DataFrame(), self.iso_12_features_df
            
        # Process features in batches
        batch_size = 100  # Adjust batch size as needed
        batches = [filtered_features[i:i + batch_size] for i in range(0, len(filtered_features), batch_size)]
        labels  = [labels[i:i + batch_size] for i in range(0, len(labels), batch_size)]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(self.process_feature_batch, batches, labels))

        masstrace_intensity = []
        masstrace_centroid_mz = []
        masstrace_lable = []
        feature_id = []
        df_feature_flatten_list = []
        charge_f = []
        feature_id_flatten = []

        for batch_results in results:
            for result in batch_results:
                masstrace_intensity.append(result[0])
                masstrace_centroid_mz.append(result[1])
                masstrace_lable.append(result[2])
                feature_id.append(result[3])
                df_feature_flatten_list.append(result[4])
                charge_f.extend(result[5])
                feature_id_flatten.extend(result[6])

        # Concatenate dataframes once
        self.df_feature_flatten = pd.concat(df_feature_flatten_list, ignore_index=True)
        self.masstrace_intensity = masstrace_intensity
        self.masstrace_centroid_mz = masstrace_centroid_mz
        self.masstrace_lable = masstrace_lable
        self.feature_id = feature_id
        self.charge_f = charge_f
        self.feature_id_flatten = feature_id_flatten
        
        self._transform_to_dataframe()
        self._merge_feature_df()
        self._add_intensity()
        self._grouping_df_based_on_feature_plus_scan()
        return self.df_feature, self.df_scan, self.iso_12_features_df


def feature_classifier(masstrace_centroid_mz: list,charge:int) -> int:
    """
    Classify features based on the mass difference between the centroid m/z values of the masstrace.

    Args:
        masstrace_centroid_mz (list): List of centroid m/z values.

    Returns:
        int: Classification label (0, 1, 2, 3) or None if no classification criteria are met.
    """
    if not isinstance(masstrace_centroid_mz, list):
        raise TypeError("masstrace_centroid_mz must be a list of m/z values.")
    
    masstrace_centroid_mz = np.array(masstrace_centroid_mz) * charge
    # Calculate mass differences (up to the first 5 differences)
    md = np.diff(masstrace_centroid_mz[:6])  # Limits to first 6 m/z values

    # Define acceptable ranges
    range1 = (0.9964, 1.0096)
    range2 = (0.9888, 1.0081)

    # Class 0
    if range1[0] < md[0] <= range1[1] and range2[0] < md[1] <= range2[1]:
        return 0

    # Class 1
    if len(md) >= 3:
        if range1[0] <= md[1] <= range1[1] and range2[0] < md[2] <= range2[1]:
            return 1

    # Class 2
    if len(md) >= 4:
        if range1[0] <= md[2] <= range1[1] and range2[0] < md[3] <= range2[1]:
            return 2

    # Class 3
    if len(md) >= 5:
        if range1[0] <= md[3] <= range1[1] and range2[0] < md[4] <= range2[1]:
            return 3

    # If none of the conditions are met
    # print("No classification criteria met.masstrace_centroid_mz:",masstrace_centroid_mz)
    return None

def set_para(fun_para, para_list):
    """Set the parameters for the feature detection process"""
    for i in range(len(para_list)):
        fun_para.setValue(para_list[i][0], para_list[i][1])
    return fun_para
