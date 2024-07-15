#import packages
import os
import shutil
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


def process_file(file, args, para, blank,ms2):
    """
    Function to process a single .mzML file, now correctly receiving `para` and other arguments.
    """
    # Assuming MyMzml uses `para` for its configuration
    df_f_result, df_scan_result = MyMzml(os.path.join(args.input, file), para).work_flow(blank=blank,ms2=ms2)
    # Save results or further processing
    df_f_result.to_csv(os.path.join('./result', os.path.basename(file).replace('.mzML','_feature.csv')), index=False)
    df_scan_result.to_csv(os.path.join('./result', os.path.basename(file).replace('.mzML','_scan.csv')), index=False)
    # Assuming saving or further processing logic here
    print(f'Processed {file}')
    # return f'Processed {file}'

def pipeline_analyze_mzml(args,para):

    path_check('./result')
    blank_featurexml_path = r'./result/blank'

    if args.blank is not None :
        if os.path.exists(blank_featurexml_path):
            blank =[]
            for file in os.listdir(blank_featurexml_path):
                b = oms.FeatureMap()
                print(f"Loading blank feature_map: {file}")
                oms.FeatureXMLFile().load(os.path.join(blank_featurexml_path,file), b)
                blank.append(b)
        else:
            path_check(blank_featurexml_path)
            print(f"Blank feature_map not found: {blank_featurexml_path}. Creating a new blank feature_map path.")
            
            blank = create_blank(args,para)
            for b in blank:
                file_name = os.path.basename(b.getMetaValue("spectra_data")[0].decode()).replace('.mzML','_blank.featureXML')
                oms.FeatureXMLFile().store(os.path.join(blank_featurexml_path,file_name), b)
    else:
        blank = None
    if args.ms2 is not None:
        ms2 = True
    else:
        ms2 = None
    #如果args.input是文件
    if os.path.isfile(args.input):
        process_file(args.input, args, para, blank,ms2)
    #如果args.input是文件夹
    elif os.path.isdir(args.input):
        # Process all files in the directory with multiprocessing Pool
        files_to_process = [file for file in os.listdir(args.input) if file.endswith('.mzML')]
        pool = Pool(4)
        fun = partial(process_file, args=args, para=para, blank=blank,ms2=ms2)
        pool.map(fun, [file for file in files_to_process])

    else:
        print('Invalid input path')
        return

if __file__ == '__main__':
    pass

