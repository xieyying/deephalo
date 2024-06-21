#import packages
import os
import shutil
from .Dataset.my_dataset import Dataset,Datasets
from .Model.my_model import MyModel,my_search
from .MZML.my_mzml import MyMzml
from HaloAnalyzer.parameters import RunParameters

#Dataset Pipeline
def pipeline_dataset() -> None:
    """
    generate dataset for training and testing
    """
    para = RunParameters()
    datas = []
    for data_df, col_formula in para.datasets:
        datas.append(Dataset(data_df,col_formula).data)
    raw_data = Datasets(datas)
    type_list = para.type_list
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
    para = RunParameters()
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
        model = MyModel(model_para)
        model.work_flow()
    elif mode == 'search':
        model = my_search(model_para)


def pipeline_analyze_mzml(args):
    #检查./result是否存在，不存在则创建
    if not os.path.exists('./result'):
        os.makedirs('./result')
    para = RunParameters()
    df_f_result,df_scan_result = MyMzml(args.input,para).work_flow()
    df_f_result.to_csv('./result/df_f_result.csv', index=False)
    df_scan_result.to_csv('./result/df_scan_result.csv', index=False)


if __file__ == '__main__':
    pass

