#import packages
import os
import shutil
from .Dataset.my_dataset import Dataset,Datasets
from .Model.my_model import MyModel,my_search
from .MZML.my_mzml import MyMzml
from HaloAnalyzer.parameters import RunParameters
from .model_test import path_check
from .MZML.methods_main import create_blank
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


def pipeline_analyze_mzml(args,para):
    path_check('./result')
    if args.blank is not None :
        blank = create_blank(args,para)
    else:
        blank = None

    df_f_result,df_scan_result = MyMzml(args.input,para).work_flow(blank=blank)
    df_f_result.to_csv('./result/df_f_result.csv', index=False)
    df_scan_result.to_csv('./result/df_scan_result.csv', index=False)


if __file__ == '__main__':
    pass

