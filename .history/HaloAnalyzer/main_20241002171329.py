#import packages
import os
import shutil
import copy
from .Dataset.my_dataset import Dataset,Datasets
from .Model.my_model import MyModel,my_search
from .MZML.my_mzml import MyMzml
from HaloAnalyzer.parameters import RunParameters
from .model_test import path_check
from .MZML.methods_main import create_blank
import pyopenms as oms
from multiprocessing import Pool
from functools import partial




#Dataset Pipeline
def pipeline_dataset(para) -> None:
    """
    generate dataset for training and testing
    """
    path_check('./dataset')
    datas = []
    for data_df, col_formula in para.datasets:
        datas.append(Dataset(data_df,col_formula).data)
    raw_data = Datasets(datas)
    type_list = para.type_list
    for type in type_list:
        raw_data.work_flow(para,type)

#Model Pipeline
def pipeline_model(args,para) -> None:
    # pass
    # """
    # train model and save model
    
    # Args:
    # mode: str, default 'manual', optional 'manual','search','feature_importance','read_feature_importance'
    # """
    path_check('./trained_models')
    
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

    if args.mode == 'manual':
        model = MyModel(para)
        model.work_flow()
    elif args.mode == 'search':
        model = my_search(para)


def process_file(file, args, para, blank=None,ms2=None):
    """
    Function to process a single .mzML file, now correctly receiving `para` and other arguments.
    """
    # Assuming MyMzml uses `para` for its configuration
    df_f_result, df_scan_result,df_no_halo = MyMzml(os.path.join(args.input, file), para).work_flow(blank=blank,ms2=ms2)
    
    # Save results or further processing
    if df_f_result.shape[0] > 0:
        if file.endswith('.mzML') or file.endswith('.mzml'):
            df_f_result.to_csv(os.path.join('./result', os.path.basename(file).replace('.mzML', '_feature.csv').replace('.mzml', '_feature.csv')), index=False)
            df_no_halo.to_csv(os.path.join('./result', os.path.basename(file).replace('.mzML', '_no_halo.csv').replace('.mzml', '_no_halo.csv')), index=False)
            df_scan_result.to_csv(os.path.join('./result', os.path.basename(file).replace('.mzML', '_scan.csv').replace('.mzml', '_scan.csv')), index=False)
    print(f'Processed {file}')

def pipeline_analyze_mzml(args,para):

    if args.blank is not None:
        path_check('./result/blank')
        blank_featurexml_path = './result/blank'

        if os.listdir(blank_featurexml_path) and not args.overwrite_blank:
            blank_ =[]
            for file in os.listdir(blank_featurexml_path):
                b = oms.FeatureMap()
                print(f"Loading blank feature_map: {file}")
                oms.FeatureXMLFile().load(os.path.join(blank_featurexml_path,file), b)
                blank_.append(b)
        else:
            shutil.rmtree(blank_featurexml_path)
            path_check(blank_featurexml_path) 
            print("Creating a new blank feature_map.")
            blank_ = create_blank(args,para)
            for b in blank_:
                file_name = os.path.basename(b.getMetaValue("spectra_data")[0].decode()).replace('.mzML','_blank.featureXML')
                oms.FeatureXMLFile().store(os.path.join(blank_featurexml_path,file_name), b)
    else:
        blank_ = None
        
    ms2_ = True if args.ms2 else None
    
    if os.path.isfile(args.input):
        process_file(args.input, args, para, blank=blank_,ms2=ms2_)
    elif os.path.isdir(args.input):
        files_to_process = []
        for root, dirs, files in os.walk(args.input):
            for file in files:
                if file.endswith('.mzML') or file.endswith('.mzml'):
                    files_to_process.append(os.path.join(root, file))

        for file in files_to_process:
            print(f'Processing {file}')
            # try:
            process_file(file, args, para, blank=copy.deepcopy(blank_),ms2=ms2_)
            # except Exception as e:
            #     print(f'Encounter error {e} when processing the file: {file}')
            #     with open('./error_files.txt', 'a') as f:
            #         f.write(file+'\n')

    else:
        print('Invalid input path')
        return

if __file__ == '__main__':
    pass

