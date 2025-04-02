#import packages
import os
import pandas as pd
import shutil
import copy
from .dataset.training_dataset_generator import Dataset,Datasets
from .model.model_training import ElementPredicitonModel,HyperparameterSearch
from .msanalysis.ms_analyzer import MSAnalyzer
from .msanalysis.ms_analyzer import create_blank,get_blank_paths
import pyopenms as oms
import importlib.resources
from tensorflow import keras
from .dereplication.database_processing import DereplicationDataset
from .dereplication.dereplication2networks import add_deephalo_results_to_graphml
from .dereplication.dereplication_ms1 import dereplicationms1
import tomli_w


def pipeline_dataset(para) -> None:
    """
    Generate datasets for training and testing.

    Parameters:
    para: object, contains dataset configurations including input data, column names, and dataset types.
    """
    # Ensure the dataset directory exists
    os.makedirs('./dataset', exist_ok=True)
    
    # Load datasets
    datasets = []
    print("Loading datasets:", para.datasets)
    for data_path, formula_column in para.datasets:
        dataset = Dataset(data_path, formula_column).data
        datasets.append(dataset)
    
    # Merge datasets
    merged_data = Datasets(datasets)
    
    # Generate datasets for each specified type
    for dataset_type in para.type_list:
        print(f"Processing dataset type: {dataset_type}")
        merged_data.work_flow(para, dataset_type)

#Model Pipeline
def pipeline_model(para) -> None:
    # pass
    # """
    # train model and save model
    
    # Args:
    # mode: str, default 'manual', optional 'manual','search','feature_importance','read_feature_importance'
    # """
    os.makedirs('./trained_models',exist_ok=True)
    
    # #根据配置文件选择训练数据
    paths = ['./dataset/base.csv']
    if para.use_fe_data == 'True':
        paths.append('./dataset/Fe.csv')
    if para.use_b_data == 'True':
        paths.append('./dataset/B.csv')
    if para.use_se_data == 'True':
        paths.append('./dataset/Se.csv')
    if para.use_hydroisomer_data == 'True':
        paths.append('./dataset/hydro.csv')
    para.paths = paths
    para.test_path = './dataset/test.csv'

    if para.args_mode == 'manual':
        ElementPredicitonModel(para).work_flow()
    elif para.args_mode == 'search':
        HyperparameterSearch(para).perform_search()

def load_trained_model():
    """
    Load a trained model and extract the output of a specified layer.

    Returns:
        EPM: The full element prediction model
        EPM_dense_output_model: A sub-model that outputs the specified layer's output
        ADM_model: The anomaly detection model
    """
    # Load the trained model using importlib.resources
    EPM_path = importlib.resources.files('deephalo') / 'assets' / 'pretrained_models' / 'isonn_element_prediction_model.h5'
    EPM = keras.models.load_model(EPM_path)

    # Ensure the layer name is correct
    try:
        target_layer = EPM.get_layer('dense')
    except ValueError:
        raise ValueError("Layer name 'dense' not found. Please check the model's layer names.")
    
    # Create a sub-model
    EPM_dense_output_model = keras.models.Model(inputs=EPM.input, outputs=target_layer.output)
    
    # Anomaly Detection model
    ADM_path = importlib.resources.files('deephalo') / 'assets' / 'pretrained_models' / 'anomaly_detection_model.h5'
    ADM_model = keras.models.load_model(ADM_path)
    
    return EPM, EPM_dense_output_model, ADM_model

def process_file(file, para,  EPM, EPM_dense_output_model, ADM, blank=None,ms2=None):
    """
    Function to process a single .mzML file, now correctly receiving `para` and other arguments.
    """
    # Assuming MSAnalyzer uses `para` for its configuration
    df_f_result, df_scan_result,iso_12_df,df_Se,df_B,df_Fe= MSAnalyzer(os.path.join(para.args_input, file), para).work_flow(EPM, EPM_dense_output_model, ADM, blank=blank,ms2=ms2)

    # Save results or further processing
    if df_f_result.shape[0] > 0:
        if file.lower().endswith(('.mzml')):
            df_f_result.to_csv(os.path.join('./result/halo', os.path.basename(file).replace('.mzML', '_feature.csv').replace('.mzml', '_feature.csv')), index=False)
            df_scan_result.to_csv(os.path.join('./result/halo', os.path.basename(file).replace('.mzML', '_scan.csv').replace('.mzml', '_scan.csv')), index=False)

    # if df_Se.shape[0] > 0:
    #     df_Se.to_csv(os.path.join('./result/Se', os.path.basename(file).lower().replace('.mzml', '_Se.csv')), index=False)
    # if df_B.shape[0] > 0:
    #     df_B.to_csv(os.path.join('./result/B', os.path.basename(file).lower().replace('.mzml', '_B.csv')), index=False)
    # if df_Fe.shape[0] > 0:
    #     df_Fe.to_csv(os.path.join('./result/Fe', os.path.basename(file).lower().replace('.mzml', '_Fe.csv')), index=False)
    if iso_12_df.shape[0] > 0:
        iso_12_df.to_csv(os.path.join('./result/iso_12', os.path.basename(file).replace('.mzML', '_1_or_2_iso.csv').replace('.mzml', '_1_or_2_iso.csv')), index=False)
    print(f'Processed {file}')

def pipeline_ms_analyze(para):
    
    EPM_model, EPM_dense_output_model, ADM_model = load_trained_model()
    # account for the processing of files
    n = 0
    os.makedirs('./result/halo',exist_ok=True)
    # os.makedirs('./result/Se',exist_ok=True)
    # os.makedirs('./result/B',exist_ok=True)
    # os.makedirs('./result/Fe',exist_ok=True)
    os.makedirs('./result/iso_12',exist_ok=True)
    # save config file 
    try:
        with open('./result/config.toml', 'wb') as f:
            tomli_w.dump(para.config, f)
    except Exception as e:
        print(f"Warning: Could not save config file: {str(e)}")
        
    ms2_ = True if para.args_ms2 else None
         
    if para.args_blank is not None:
        os.makedirs('./result/blank',exist_ok=True)
        blank_featurexml_path = './result/blank'

        if os.listdir(blank_featurexml_path) and not para.args_overwrite_blank:
            blank_ =[]
            for file in os.listdir(blank_featurexml_path):
                b = oms.FeatureMap()
                print(f"Loading blank feature_map: {file}")
                oms.FeatureXMLFile().load(os.path.join(blank_featurexml_path,file), b)
                blank_.append(b)
        else:
            shutil.rmtree(blank_featurexml_path)
            os.makedirs(blank_featurexml_path) 
            print("Creating a new blank feature_map.")
            blank_ = create_blank(para)
            for b in blank_:
                file_name = os.path.basename(b.getMetaValue("spectra_data")[0].decode()).replace('.mzML','_blank.featureXML').replace('.mzml','_blank.featureXML')
                oms.FeatureXMLFile().store(os.path.join(blank_featurexml_path,file_name), b)
        blank_files = get_blank_paths(para)
        for blank_file in blank_files:
            n += 1
            print(f"Processing blank file: {blank_file}")
            try:
                process_file(blank_file, para, EPM_model, EPM_dense_output_model, ADM_model, blank=None,ms2=ms2_)
            except Exception as e:
                print(f'Encounter error {e} when processing the blank file: {blank_file}')
                with open(r'./result/error_files.txt', 'a') as f:
                    f.write(blank_file+'\n')
            print(f'Processed blank file: {blank_file}')    
    else:
        blank_ = None
        
    if os.path.isfile(para.args_input):
        process_file(para.args_input, para, EPM_model, EPM_dense_output_model, ADM_model, blank=blank_,ms2=ms2_)
   
    elif os.path.isdir(para.args_input):
        
        files_to_process = []
        for root, dirs, files in os.walk(para.args_input):
            for file in files:
                if file.lower().endswith(('.mzml')):
                    files_to_process.append(os.path.join(root, file))
        print(f'Found {files_to_process} files to process.')
        for file in files_to_process:
            print(f'Processing {file}')
            try:
                process_file(file, para, EPM_model, EPM_dense_output_model, ADM_model,blank=copy.deepcopy(blank_),ms2=ms2_)
                n += 1
                print(f'Processed {n} files in total.')
            except Exception as e:
                print(f'Encounter error {e} when processing the file: {file}')
                with open(r'./result/error_files.txt', 'a') as f:
                    f.write(file+'\n')
    else:
        print('Invalid input path')
        return

def pipeline_dereplication(para):
    """
    Perform dereplication using GNPS output and/or user database.
    """
    #MS1 dereplication
    # If a user database is provided, prepare the dereplication database
    if para.args_user_database:
        if 'DeepHalo_dereplication_ready_database' in str(para.args_user_database):
            user_dereplication_database = pd.read_csv(para.args_user_database, low_memory=False).dropna(subset=['M+H'])
        else:
            print('Processing user database...(this may take a while)')
            user_dereplication_database = DereplicationDataset(para.args_user_database, 'formula').work_flow()
            ready_db_path = str(para.args_user_database).rsplit('.', 1)[0] + "_DeepHalo_dereplication_ready_database.csv"
            user_dereplication_database.to_csv(ready_db_path, index=False)
            print(f"User database has been processed and saved as {ready_db_path}")
        dereplication_database = {'user_database': user_dereplication_database}
        dereplication_folder = dereplicationms1(para, dereplication_database)
    else:
        dereplication_folder = dereplicationms1(para, None)
    
    # If a GNPS analysis folder is provided, integrate the dereplication and analysis results
    # into the GNPS file and output a new network file
    if para.args_GNPS_folder != None:
        if para.args_user_database == None:
            dereplication_folder = dereplication_folder
        add_deephalo_results_to_graphml(para.args_GNPS_folder, dereplication_folder)
        print('The results have been added to the GNPS file ending with "_adding_DeepHalo_results.graphml"')
        print("_______________________")
    print('Element pattern classifications (0-7):')
    print('0: Complex halogens (Cl₃+, Br₂+, or mixed Cl/Br)')
    print('1: Medium halogen patterns (Cl₃/Br)')
    print('2: Simple halogen patterns (Cl/Cl₂)')
    print('3: Selenium compounds')
    print('4: Boron compounds')
    print('5: Iron compounds') 
    print('6: CHONFPSNa compounds')
    print('7: Overlapping isotope patterns')
if __file__ == '__main__':
    pass

