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
    raw_data.work_flow(para.mz_start,para.mz_end,para.elements_list,'noise')
    raw_data.work_flow(para.mz_start,para.mz_end,para.elements_list,'Fe')
    raw_data.work_flow(para.mz_start,para.mz_end,para.elements_list,'hydro',[0.33,0.66])
    raw_data.work_flow(para.mz_start,para.mz_end,para.elements_list,'hydro2',[0.99])
    raw_data.work_flow(para.mz_start,para.mz_end,para.elements_list,'hydro3',[1.32,1.65])
    # data.data_statistics_customized()

#Model Pipeline
def pipeline_model():
    para = load_config()
    model_para = {'batch_size': para.train_batch,
                  'epochs': para.epochs,
                  'features': para.features_list,
                  'paths': ['./dataset/base.csv',
                            './dataset/Fe.csv',
                            './dataset/hydro.csv',
                            './dataset/hydro2.csv',
                            './dataset/hydro3.csv'
                         ],
                  'base_classes': para.base_classes,
                  'sub_classes': para.sub_classes,
                  'hydro_classes': para.hydro_classes}
    model = my_model(model_para)
    model.work_flow()

#Find Halo Pipeline
def pipeline_find_halo(mzml_path):
    para = load_config()
    a = analysis_by_asari(mzml_path,para)
    a.asari_workflow()


