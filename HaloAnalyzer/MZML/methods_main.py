import pandas as pd
import numpy as np
import tensorflow as tf
from collections import Counter
from HaloAnalyzer.parameters import run_parameters
from .methods_feature_finding import FeatureMapProcessor

def featureFinding(file,para):
    """
    Find features from mzML data
    return two DataFrame. One containing feature based isotope patterns. The other contain scan based isotope patterns
    """
    feature = FeatureMapProcessor(file,para)
    df_f,df_scan = feature.run().process()
    return df_f,df_scan

def isotope_processing(df, mz_list_name = 'mz_list', inty_list_name = "inty_list"):
    """
    Process DataFrame and make it ready for halo model inputs
    """
    # get the mz_list and inty_list
    mz_list = df[mz_list_name].values
    m2_m1 = [i[2] - i[1] for i in mz_list]
    m1_m0 = [i[1] - i[0] for i in mz_list]
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

def flow_base(file,model_path,pars):
    """
    The main workflow for feature finding, isotope processing, prediction, and evaluation
    """
    df_f,df_scan = featureFinding(file,pars)
    df_feature_for_model_input = isotope_processing(df_f,'masstrace_centroid_mz','masstrace_intensity')
    df_scan_for_model_input =isotope_processing(df_scan,'mz_list','inty_list')
    df_f = add_predict(df_feature_for_model_input,model_path, pars.features_list)
    df_scan = add_predict(df_scan_for_model_input,model_path, pars.features_list)
    df_f_result,df_scan_result = haloEvalution(df_f,df_scan)
    return df_f_result,df_scan_result

if __name__ == '__main__':

    file = r'C:\Users\xq75\Desktop\Test Folder\xcms_test\Vancomycin.mzML'
    model_path = r'C:\Users\xq75\Desktop\p_test\trained_models\pick_halo_ann.h5'
    pars= run_parameters()

    df_f_result,df_scan_result = workflow(file,model_path,pars)
    df_f_result.to_csv(r'C:\Users\xq75\Desktop\oms\df_f_result.csv', index=False)
    df_scan_result.to_csv(r'C:\Users\xq75\Desktop\oms\df_scan_result.csv', index=False)

    
    