import pandas as pd
import numpy as np
import tensorflow as tf
from collections import Counter
from .methods_feature_finding import FeatureMapProcessor,set_para
import os
import keras
import pyopenms as oms
from mzml2gnps.methods import *


def feature_finding(file,para):
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

def load_trained_model(model_path, layer_name):
    """
    加载已训练的模型并提取指定层的输出。

    Args:
        model_path (str): 已训练模型的文件路径。
        layer_name (str): 要提取的层的名称。

    Returns:
        Model: 提取指定层输出的子模型。
    """
    # 加载已训练的模型
    model = keras.models.load_model(model_path)
    
    # 确认层名称正确
    try:
        target_layer = model.get_layer(layer_name)
    except ValueError:
        print(f"层名称 '{layer_name}' 未找到。请检查模型的层名称。")
        for i, layer in enumerate(model.layers):
            print(f"{i}: {layer.name}")
        raise
    
    # 创建子模型
    units_1_output_model = keras.models.Model(inputs=model.input, outputs=target_layer.output)
    return units_1_output_model

def anormal_isotope_pattern_detection(df_feature):
    querys_2 = df_feature[['p0_int','p1_int','p2_int','p3_int','p4_int']]
    querys_2 = querys_2.values.astype('float32')
    anomy_sco = tf.keras.models.load_model('.\autoencoder_5_peak_intensities.h5')
    reconstructed_X = anomy_sco.predict(querys_2)
    reconstruction_error = np.mean(np.power(querys_2 - reconstructed_X, 2), axis=1)
    df_feature.loc[:, 'reconstruction_error'] = reconstruction_error

    return df_feature

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
    df_ = df.copy()
    df_.loc[:, 'class_pred'] = classes_pred
    # anomaly detection
    return df_

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
            if len(I) >= 2:
                scan_based_halo_score = 100
                scan_based_halo_sub_score = 100
                scan_based_halo_sub_class = com_class[0]
            else:
                scan_based_halo_score = 0
                scan_based_halo_sub_class = "None"
                scan_based_halo_sub_score = "None"
            
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

def halo_evalution_(df_f_,df_scan_):
    """
    Evaluate the probability of features based on both feature isotope patterns and scan based isotope patterns
    """
    df_f = df_f_.copy()
    df_scan = df_scan_.copy()
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
    df_scan['H_score'] = (df_scan['scan_based_halo_score'])/300 + (df_scan['scan_based_halo_ratio'])/3 + (df_scan['class_pred'].apply(lambda x: 1 if x in [0,1,2] else 0))/3
    
    return df_f,df_scan

def halo_evalution(df_f_, df_scan_):
    """
    Evaluate the probability of features based on both feature isotope patterns and scan based isotope patterns
    """
    df_f = df_f_.copy()
    df_scan = df_scan_.copy()

    # Create a dictionary to store the results
    results = {
        'feature_id_flatten': [],
        'scan_based_halo_class': [],
        'scan_based_halo_score': [],
        'scan_based_halo_sub_class': [],
        'scan_based_halo_sub_score': [],
        'scan_based_halo_ratio': []
    }

    for i in df_scan['feature_id_flatten'].unique():
        I = df_scan[df_scan['feature_id_flatten'] == i]['class_pred'].tolist()
        scan_based_halo_class, scan_based_halo_score, scan_based_halo_sub_class, scan_based_halo_sub_score, scan_based_halo_ratio = roi_scan_based_halo_evaluation(I)
        
        results['feature_id_flatten'].append(i)
        results['scan_based_halo_class'].append(scan_based_halo_class)
        results['scan_based_halo_score'].append(scan_based_halo_score)
        results['scan_based_halo_sub_class'].append(scan_based_halo_sub_class)
        results['scan_based_halo_sub_score'].append(scan_based_halo_sub_score)
        results['scan_based_halo_ratio'].append(scan_based_halo_ratio)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Merge results with df_scan and df_f
    df_scan = df_scan.merge(results_df, on='feature_id_flatten', how='left')
    df_f = df_f.merge(results_df.rename(columns={'feature_id_flatten': 'feature_id'}), on='feature_id', how='left')

    # Calculate feature_based_halo_score
    df_f['feature_based_halo_score'] = df_f['class_pred'].apply(lambda x: 1 if x in [0, 1, 2] else 0)

    # Calculate H_score
    df_f['H_score'] = (df_f['scan_based_halo_score']) / 300 + (df_f['scan_based_halo_ratio']) / 3 + (df_f['feature_based_halo_score']) / 3
    df_scan['H_score'] = (df_scan['scan_based_halo_score']) / 300 + (df_scan['scan_based_halo_ratio']) / 3 + (df_scan['class_pred'].apply(lambda x: 1 if x in [0, 1, 2] else 0)) / 3

    return df_f, df_scan

def create_blank(args,para):
    """
    Create a blank for feature subtraction
    """
    #处理blank mzml文件
    #如果args.blank为文件夹，则blank_paths为文件夹下所有文件，否则为单个文件，都转化为列表
    if args.blank is not None:
        if os.path.isdir(args.blank):
            #利用walk遍历文件夹下所有文件
            blank_paths = []
            for root, dirs, files in os.walk(args.blank):
                for file in files:
                    if file.endswith('.mzML'):
                        blank_paths.append(os.path.join(root, file))
        else:
            blank_paths = [args.blank]
    else:
        blank_paths = None

    blank_feature_maps = []
    for file in blank_paths:
        try:
            blank_feature_maps.append(feature_finding(file,para)[2])
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
    pipline(file, output, precmz, rt, RTstart, RTend, precmz_tolerance=20,rt_tolerance=0.5,precinty_thre=10,correct=True, merge=False)
    
def flow_base(file,model_path,pars,blank=None,ms2=None):
    """
    The main workflow for feature finding, isotope processing, prediction, and evaluation
    """
    
    # Extract features and scan data
    df_f, df_scan, feature_map_,iso_12_df = feature_finding(file, pars)
    
    if blank is not None and df_f.shape[0] > 0:
        blank.append(feature_map_)
        df_f_ = substract_blank(blank, pars)

        # Extract rows in df_f that have the same RT and mz as in df_f_
        df_f = df_f[df_f['RT'].isin(df_f_['RT'])]
        df_f = df_f[df_f['mz'].isin(df_f_['mz'])]

    if df_f.shape[0] == 0 and iso_12_df.shape[0] == 0:
        return pd.DataFrame(), pd.DataFrame(),pd.DataFrame()
    elif df_f.shape[0] == 0 and iso_12_df.shape[0] > 0:
        return pd.DataFrame(), pd.DataFrame(),iso_12_df
    
    # Process isotope features
    df_feature_for_model_input = isotope_processing(df_f, 'masstrace_centroid_mz', 'masstrace_intensity')
    df_feature_for_model_input = anormal_isotope_pattern_detection(df_feature_for_model_input)
       
    if df_feature_for_model_input.shape[0] == 0 and iso_12_df.shape[0] == 0:
        return pd.DataFrame(), pd.DataFrame(),pd.DataFrame()
    elif df_feature_for_model_input.shape[0] == 0 and iso_12_df.shape[0] > 0:
        return pd.DataFrame(), pd.DataFrame(),iso_12_df
    
    # Filter scan data
    df_scan = df_scan.loc[df_scan['feature_id_flatten'].isin(df_feature_for_model_input['feature_id'])]

    # Process isotope features for scan data
    df_scan_for_model_input = isotope_processing(df_scan, 'mz_list', 'inty_list')
    df_scan_for_model_input = anormal_isotope_pattern_detection(df_scan_for_model_input)

    if df_scan_for_model_input.shape[0] == 0 and iso_12_df.shape[0] == 0:
        return pd.DataFrame(), pd.DataFrame(),pd.DataFrame()
    elif df_scan_for_model_input.shape[0] == 0 and iso_12_df.shape[0] > 0:
        return pd.DataFrame(), pd.DataFrame(),iso_12_df
    
    # Add prediction results
    df_f = add_predict(df_feature_for_model_input, model_path, pars.features_list)
    df_scan = add_predict(df_scan_for_model_input, model_path, pars.features_list)

    # Evaluate results
    df_f_result, df_scan_result = halo_evalution(df_f, df_scan)
    
    # Seperate the results based on the class_pred
    df_Se = df_f_result[df_f_result['class_pred']==3]
    df_B = df_f_result[df_f_result['class_pred']==4]
    df_Fe = df_f_result[df_f_result['class_pred']==5]
    
    # Filter high score results for halo group
    df_f_result = df_f_result[df_f_result['H_score'] >= pars.FeatureFilter_H_score_threshold] 
    df_scan_result = df_scan_result.loc[df_scan_result['feature_id_flatten'].isin(df_f_result['feature_id'])]
    
    # Extract MS2 data
    if ms2 is not None and df_f_result.shape[0] > 0:
        ms2_extraction(file, df_f_result)

    return df_f_result, df_scan_result,iso_12_df,df_Se,df_B,df_Fe

if __name__ == '__main__':
    pass
    # file = r'C:\Users\xq75\Desktop\Test Folder\xcms_test\Vancomycin.mzML'
    # model_path = r'C:\Users\xq75\Desktop\p_test\trained_models\pick_halo_ann.h5'
    # pars= run_parameters()

    # df_f_result,df_scan_result = flow_base(file,model_path,pars)
    # df_f_result.to_csv(r'C:\Users\xq75\Desktop\oms\df_f_result.csv', index=False)
    # df_scan_result.to_csv(r'C:\Users\xq75\Desktop\oms\df_scan_result.csv', index=False)

    
    