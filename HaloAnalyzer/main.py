#import packages
import os
import shutil
from .Dataset.my_dataset import dataset,datasets
from .Model.my_model import my_model,my_search
from .MZML.my_mzml import my_mzml
from .parameters import run_parameters 

#Dataset Pipeline
def pipeline_dataset(type_list = ['base','Fe','B','Se','hydro']) -> None:
    """
    generate dataset for training and testing
    """
    para = run_parameters()
    datas = []
    for data_df, col_formula in para.datasets:
        datas.append(dataset(data_df,col_formula).data)
    raw_data = datasets(datas)
    type_list = ['base']
    for type in type_list:
        raw_data.work_flow(para,type)

#Model Pipeline
def pipeline_model(mode = 'manual') -> None:
    # pass
    # """
    # train model and save model
    
    # Args:
    # mode: str, default 'manual', optional 'manual','search','feature_importance','read_feature_importance'
    # """
    para = run_parameters()
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
    
    test_path = './dataset/test.csv'

    #定义训练参数
    model_para = {'batch_size': para.train_batch,
                  'epochs': para.epochs,
                  'features': para.features_list,
                  'paths': paths,
                  'test_path': test_path,
                  'classes': para.classes,
                  'weight': para.classes_weight,
                  'learning_rate': para.learning_rate,}
    if mode == 'manual':
        model = my_model(model_para)
        model.work_flow()
    elif mode == 'search':
        model = my_search(model_para)


def pipeline_analyze_mzml(args):
    pass


if __file__ == '__main__':
    pass

