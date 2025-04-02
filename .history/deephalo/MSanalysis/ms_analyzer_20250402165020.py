import pandas as pd
import numpy as np
from .isotope_extraction import FeatureMapProcessor,set_para
import os
import pyopenms as oms
from mzml2gnps.methods import pipeline
from .halogen_scorer import HalogenScorer


def isotope_extracting(file,para):
    """
    Find features from mzML data
    return two DataFrame. One containing feature based isotope patterns. The other contain scan based isotope patterns
    """
    feature = FeatureMapProcessor(file,para)
    tem = feature.run()
    df_f,df_scan,iso_12_df = tem.process()

    return df_f,df_scan,tem.feature_map,iso_12_df

def isotope_processing(df, mz_list_name = 'mz_list', inty_list_name = "inty_list"):
    """
    Process DataFrame and make it ready for halo model inputs
    """
    
    # get the mz_list and inty_list
    mz_list = df[mz_list_name].values
    
    m2_m1 = [i[2] - i[1] if len(i)>2 else 1.0012 for i in mz_list ]
    m1_m0 = [i[1] - i[0] if len(i)>=2 else 1.0012 for i in mz_list ]
    # m1_m0 = [i[1] - i[0] for i in mz_list]
    
    # Assign new columns to df
    df['m2_m1'] = m2_m1*df['charge']
    df['m1_m0'] = m1_m0*df['charge']
    
    # Ensure all lists in inty_list have 7 elements
    inty_list = [list(i) + [0]*(7-len(i)) for i in df[inty_list_name].tolist()]
    # Convert inty_list to a DataFrame
    inty_df = pd.DataFrame(inty_list)
    
    # Normalize each row by its max value
    inty_df = inty_df.div(inty_df.max(axis=1), axis=0)
    for i in range(7):
        df[f'p{i}_int'] = inty_df[i].values

    return df

def anormal_isotope_pattern_detection(EPM_dense_output_model, ADM, df_feature,features_list):
    """
    EPM_dense_output_model: Element prediction model dense output model
    ADM: Anomaly detection model
    df_feature: DataFrame containing feature based isotope patterns
    features_list: List of features to be used for the model
    """

    querys_input = df_feature[features_list].values
    autoencoder_input = EPM_dense_output_model.predict(querys_input)
    reconstructed_X = ADM.predict(autoencoder_input)
    reconstruction_error = np.mean(np.power(autoencoder_input - reconstructed_X, 2), axis=1)
    df_feature.loc[:, 'reconstruction_error'] = reconstruction_error
    
    return df_feature

def add_predict(df,EPM,features_list):
    """
    Add prediction result based on DNN Halo model (Element prediction model,EPM)
    """
    # Load the features
    querys = df[features_list].values
    querys = querys.astype('float32')
    # Predict the features
    res = EPM.predict(querys)
    classes_pred = np.argmax(res, axis=1)
    # Add the prediction results to df_features
    df_ = df.copy()
    df_.loc[:, 'Feature_based_prediction'] = classes_pred

    return df_

def halo_scoring_(df_f_,df_scan_):
    """
    Evaluate the probability of features based on both feature isotope patterns and scan based isotope patterns
    """
    df_f = df_f_.copy()
    df_scan = df_scan_.copy()
    for i in df_scan['feature_id_flatten'].unique():
        I = df_scan[df_scan['feature_id_flatten'] == i]['Feature_based_prediction'].tolist()
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
        
    feature_based_halo_score = df_f['Feature_based_prediction'].apply(lambda x: 1 if x in [0,1,2] else 0)
    df_f['feature_based_halo_score'] = feature_based_halo_score
    df_f['H_score'] = (df_f['scan_based_halo_score'])/300 + (df_f['scan_based_halo_ratio'])/3 + (df_f['feature_based_halo_score'])/3 
    df_scan['H_score'] = (df_scan['scan_based_halo_score'])/300 + (df_scan['scan_based_halo_ratio'])/3 + (df_scan['Feature_based_prediction'].apply(lambda x: 1 if x in [0,1,2] else 0))/3
    
    return df_f,df_scan

def halo_scoring(df_f_, df_scan_):
    """
    Evaluate the probability of features based on both feature isotope patterns and scan based isotope patterns
    """
    df_f = df_f_.copy()
    df_scan = df_scan_.copy()
  
    # Create a dictionary to store the results
    results = {
        'feature_id_flatten': [],
        'h_score': [],
        'sub_class': [],
        'r_halo': [],
        'z_score': [],
        'f_score': []
    }

    for i in df_scan['feature_id_flatten'].unique():
        scan_predictions = df_scan[df_scan['feature_id_flatten'] == i]['Feature_based_prediction'].tolist()
        feature_prediction = df_f[df_f['feature_id'] == i]['Feature_based_prediction'].value()
        h_score, sub_class, r_halo, z_score, f_score = HalogenScorer().process_feature(scan_predictions, feature_prediction)

        
        results['feature_id_flatten'].append(i)
        results['h_score'].append(h_score)
        results['sub_class'].append(sub_class)
        results['r_halo'].append(r_halo)
        results['z_score'].append(z_score)
        results['f_score'].append(f_score)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Merge results with df_scan and df_f
    df_scan = df_scan.merge(results_df, on='feature_id_flatten', how='left')
    df_f = df_f.merge(results_df.rename(columns={'feature_id_flatten': 'feature_id'}), on='feature_id', how='left')

    return df_f, df_scan

def get_blank_paths(para):
    """
    Get blank paths from para.args_blank
    """

    # If args.blank is a directory, get all files in the directory
    if os.path.isdir(para.args_blank):
        blank_paths = [os.path.join(para.args_blank, f) for f in os.listdir(para.args_blank) if f.endswith('.mzML') or f.endswith('.mzml')]
    else:
        # If args.blank is a single file, convert it to a list
        blank_paths = [para.args_blank]
    return blank_paths

def create_blank(para):
    """
    Create a blank for feature subtraction
    """
    #处理blank mzml文件
    #如果args.blank为文件夹，则blank_paths为文件夹下所有文件，否则为单个文件，都转化为列表
    if para.args_blank is not None:
        blank_paths = get_blank_paths(para)
        if len(blank_paths) == 0:
            print('No blank files found in the specified directory.')
            blank_paths = None
    else:
        blank_paths = None

    blank_feature_maps = []
    for file in blank_paths:
        try:
            blank_feature_maps.append(isotope_extracting(file,para)[2])
        except Exception as e:
            print(f'Encounter error {e} when processing the file: {file}')
            with open('./error_files.txt', 'a') as f:
                f.write(file+'\n')
    return blank_feature_maps

def substract_blank(feature_maps,pars):
    feature_grouper = oms.FeatureGroupingAlgorithmKD()
    g_parameters = feature_grouper.getParameters()
    set_para(g_parameters,pars.feature_grouping)
    feature_grouper.setParameters(g_parameters)
    
    consensus_map = oms.ConsensusMap()
    file_descriptions = consensus_map.getColumnHeaders()
    files = []
    for i, feature_map in enumerate(feature_maps):
        file_description = file_descriptions.get(i, oms.ColumnHeader())
        file_name = os.path.basename(feature_map.getMetaValue("spectra_data")[0].decode())
        files.append(file_name)
        file_description.filename = file_name
        file_description.size = feature_map.size()
        file_descriptions[i] = file_description

    feature_grouper.group(feature_maps, consensus_map)
    consensus_map.setColumnHeaders(file_descriptions)
    consensus_map.setUniqueIds()
    # oms.ConsensusXMLFile().store("FeatureMatrix.consensusXML", consensus_map)

    df = consensus_map.get_df()
    
    for i in range(len(files[:-1])):
        df = df[df[files[i]] <= 0.001]
    
    return df

def ms2_extraction(file,df):
    """
    Extract MS2 data from mzML file
    """
    precmz = df['mz'].tolist() if 'mz' in df.columns else None
    rt = df['RT'].tolist() if 'RT' in df.columns else None
    RTstart = df['RTstart'].tolist() if 'RTstart' in df.columns else None
    RTend = df['RTend'].tolist() if 'RTend' in df.columns else None
    ms2_output = r'./result/ms2_output'
    if not os.path.exists(ms2_output):  # Check if directory exists
        os.makedirs(ms2_output)  # Create the directory if it does not exist
    output = os.path.join(ms2_output, os.path.basename(file))
    pipeline(file, output, precmz, rt, RTstart, RTend, precmz_tolerance=20,rt_tolerance=0.5,precinty_thre=10,correct=True, merge=False)
    
def flow_base(file,pars, EPM, EPM_dense_output_model, ADM, blank=None,ms2=None):
    """
    The main workflow for feature finding, isotope processing, prediction, and evaluation
    EPM: Element prediction model
    EPM_dense_output_model: Element prediction model dense output model
    ADM: Anomaly detection model
    """
    
    # Extract features and scan data
    df_f, df_scan, feature_map_,iso_12_df = isotope_extracting(file, pars)
    
    if blank is not None and df_f.shape[0] > 0:
        blank.append(feature_map_)
        df_f_ = substract_blank(blank, pars)

        # Extract rows in df_f that have the same RT and mz as in df_f_
        df_f = df_f[df_f['RT'].isin(df_f_['RT'])]
        df_f = df_f[df_f['mz'].isin(df_f_['mz'])]

    if df_f.shape[0] == 0 and iso_12_df.shape[0] == 0:
        return pd.DataFrame(), pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    elif df_f.shape[0] == 0 and iso_12_df.shape[0] > 0:
        return pd.DataFrame(), pd.DataFrame(),iso_12_df,pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    
    # Process isotope features
    df_feature_for_model_input = isotope_processing(df_f, 'masstrace_centroid_mz', 'masstrace_intensity')
    df_feature_for_model_input = anormal_isotope_pattern_detection(EPM_dense_output_model, ADM, df_feature_for_model_input,pars.features_list)
    # Filter the results 
   
    if df_feature_for_model_input.shape[0] == 0 and iso_12_df.shape[0] == 0:
        return pd.DataFrame(), pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    elif df_feature_for_model_input.shape[0] == 0 and iso_12_df.shape[0] > 0:
        return pd.DataFrame(), pd.DataFrame(),iso_12_df,pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    
    # Filter scan data
    df_scan = df_scan.loc[df_scan['feature_id_flatten'].isin(df_feature_for_model_input['feature_id'])]

    # Process isotope features for scan data
    df_scan_for_model_input = isotope_processing(df_scan, 'mz_list', 'inty_list')
    df_scan_for_model_input = anormal_isotope_pattern_detection(EPM_dense_output_model, ADM, df_scan_for_model_input,pars.features_list)
    # Filter the results based on the reconstruction error
    # df_scan_for_model_input = df_scan_for_model_input[df_scan_for_model_input['reconstruction_error'] 
                                                    #   <= 0.000345] #Anomaly detection threshold 100%: 0.00034502154449000955

    if df_scan_for_model_input.shape[0] == 0 and iso_12_df.shape[0] == 0:
        return pd.DataFrame(), pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    elif df_scan_for_model_input.shape[0] == 0 and iso_12_df.shape[0] > 0:
        return pd.DataFrame(), pd.DataFrame(),iso_12_df,pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    
    # Add prediction results
    df_f = add_predict(df_feature_for_model_input, EPM, pars.features_list)
    df_scan = add_predict(df_scan_for_model_input, EPM, pars.features_list)

    # Evaluate results
    df_f_result, df_scan_result = halo_scoring(df_f, df_scan)
    
    # Seperate the results based on the Feature_based_prediction
    df_Se = df_f_result[df_f_result['Feature_based_prediction']==3]
    df_B = df_f_result[df_f_result['Feature_based_prediction']==4]
    df_Fe = df_f_result[df_f_result['Feature_based_prediction']==5]
    
    # Filter high score AND the reconstruction error results for halo group
    df_f_result = df_f_result[df_f_result['reconstruction_error'] <= pars.FeatureFilter_Anomaly_detection_threshold]
    df_f_result = df_f_result[df_f_result['H_score'] >= pars.FeatureFilter_H_score_threshold] 
    df_scan_result = df_scan_result.loc[df_scan_result['feature_id_flatten'].isin(df_f_result['feature_id'])]
    
    # Extract MS2 data
    if ms2 is not None and df_f_result.shape[0] > 0:
        ms2_extraction(file, df_f_result)
    
    return df_f_result, df_scan_result,iso_12_df,df_Se,df_B,df_Fe

class MSAnalyzer:
    """
    Main class for MS analysis workflow.
    """
    def __init__(self,file,para) -> None:
        self.file = file
        self.para = para
        self.EPM = None
        self.EPM_dense_output_model = None
        self.ADM = None

    def work_flow(self, EPM = None, EPM_dense_output_model = None, ADM = None, blank=None, ms2=None):
        return flow_base(self.file,self.para, EPM, EPM_dense_output_model, ADM, blank=blank,ms2=ms2)   

if __name__ == '__main__':
    pass

    
    