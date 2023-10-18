#import packages
from .Dataset.my_dataset import dataset,datasets
from .Model.my_model import my_model
from .MZML.analysis import analysis_by_asari
from .parameters import run_parameters

def load_config():
    return run_parameters()
     

def pipeline_dataset():
    para = load_config()
    datas = []
    for data in para.datasets:
        datas.append(dataset(data[0],data[1]).data)
    raw_data = datasets(datas)
    raw_data.work_flow(para.mz_start,para.mz_end,para.elements_list,'base')
    raw_data.work_flow(para.mz_start,para.mz_end,para.elements_list,'noise',repeats=para.repeat_for_noise)
    raw_data.work_flow(para.mz_start,para.mz_end,para.elements_list,'Fe')
    raw_data.work_flow(para.mz_start,para.mz_end,para.elements_list,'hydro',rates=para.rate_for_hydro)
    raw_data.work_flow(para.mz_start,para.mz_end,para.elements_list,'hydro2',rates=para.rate_for_hydro2)
    raw_data.work_flow(para.mz_start,para.mz_end,para.elements_list,'hydro3',rates=para.rate_for_hydro3)
    # data.data_statistics_customized()

#Model Pipeline
def pipeline_model():
    #加载参数
    para = load_config()

    #根据配置文件选择训练数据
    paths = ['./dataset/base.csv']
    if para.use_fe_data == 'True':
        paths.append('./dataset/Fe.csv')
    if para.use_hydroisomer_data == 'True':
        paths.append('./dataset/hydro.csv')
        paths.append('./dataset/hydro2.csv')
        paths.append('./dataset/hydro3.csv')
    if para.use_noise_data == 'True':
        paths.append('./dataset/noise.csv')

    #定义训练参数
    model_para = {'batch_size': para.train_batch,
                  'epochs': para.epochs,
                  'features': para.features_list,
                  'paths': paths,
                  'base_classes': para.base_classes,
                  'sub_classes': para.sub_classes,
                  'hydro_classes': para.hydro_classes,
                  'base_weight': para.base_weight,
                  'sub_weight': para.sub_weight,
                  'hydroisomer_weight': para.hydroisomer_weight,
                  'learning_rate': para.learning_rate,}
    model = my_model(model_para)
    model.work_flow()

#Find Halo Pipeline
def pipeline_find_halo(mzml_path):
    para = load_config()
    a = analysis_by_asari(mzml_path,para)
    a.asari_workflow()


