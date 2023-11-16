#import packages
import os
from .Dataset.my_dataset import dataset,datasets
from .Model.my_model import my_model
from .MZML.my_mzml import my_mzml
from .parameters import run_parameters
from .MZML.methons import extract_ms2_of_rois
def load_config():
    return run_parameters()
     

def pipeline_dataset():
    para = load_config()
    datas = []
    for data in para.datasets:
        datas.append(dataset(data[0],data[1]).data)
    raw_data = datasets(datas)
    raw_data.work_flow(para.mz_start,para.mz_end,para.elements_list,'base')
    # raw_data.work_flow(para.mz_start,para.mz_end,para.elements_list,'noise',repeats=para.repeat_for_noise)
    raw_data.work_flow(para.mz_start,para.mz_end,para.elements_list,'Fe')
    raw_data.work_flow(para.mz_start,para.mz_end,para.elements_list,'B')
    raw_data.work_flow(para.mz_start,para.mz_end,para.elements_list,'Se')
    raw_data.work_flow(para.mz_start,para.mz_end,para.elements_list,'hydro',rates=para.rate_for_hydro)

    # data.data_statistics_customized()

#Model Pipeline
def pipeline_model():
    #加载参数
    para = load_config()

    #根据配置文件选择训练数据
    paths = ['./dataset/base1.csv']
    if para.use_fe_data == 'True':
        paths.append('./dataset/Fe.csv')
    if para.use_b_data == 'True':
        paths.append('./dataset/B.csv')
    if para.use_se_data == 'True':
        paths.append('./dataset/Se.csv')        
    if para.use_hydroisomer_data == 'True':
        paths.append('./dataset/hydro.csv')
    if para.use_noise_data == 'True':
        paths.append('./dataset/noise.csv')

    #定义训练参数
    model_para = {'batch_size': para.train_batch,
                  'epochs': para.epochs,
                  'features': para.features_list,
                  'paths': paths,
                  'classes': para.classes,
                  'weight': para.classes_weight,
                  'learning_rate': para.learning_rate,}
    model = my_model(model_para)
    model.work_flow()

#Find Halo Pipeline
def pipeline_find_halo(mzml_path):
    para = load_config()
    mzml_para = {'path':mzml_path,
                 'feature_list':para.features_list,
                 'asari':para.asari_dict,
                 'mzml':para.mzml_dict,}
    data = my_mzml(mzml_para)
    data.work_flow()


#extract ms2 of rois
def pipeline_extract_ms2_of_rois(mzml_path,project_path,rois:list):
    save_halo_evaluation = os.path.normpath(project_path +r'./test_mzml_prediction/halo_evaluation.csv')
    output_path = os.path.normpath(project_path +r'./test_mzml_prediction/select_roi_ms2.mgf')
    extract_ms2_of_rois(mzml_path,save_halo_evaluation,output_path,rois)
    

    
                 
if __file__ == '__main__':
    p1 = r'C:\Users\xq75\Desktop\test'
    p2 = r'./test_mzml_prediction/halo_evaluation.csv'
    p3 = os.path.join(p1,p2)
    print(p3)

