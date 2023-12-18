#import packages
import os
from .Dataset.my_dataset import dataset,datasets
from .Model.my_model import my_model
from .MZML.my_mzml import my_mzml
from .parameters import run_parameters
from .MZML.methods import extract_ms2_of_rois

#load config file
def load_config() -> dict:
    """load config file"""
    return run_parameters()
     
#Dataset Pipeline
def pipeline_dataset() -> None:
    """generate dataset for training and testing"""
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
def pipeline_model() -> None:
    """train model and save model"""
    #加载参数
    para = load_config()

    #根据配置文件选择训练数据
    paths = ['./dataset/base.csv']
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
def pipeline_find_halo(mzml_path) -> None:
    """find halo in mzml file"""
    para = load_config()
    mzml_para = {'path':mzml_path,
                 'feature_list':para.features_list,
                 'asari':para.asari_dict,
                 'mzml':para.mzml_dict,}
    data = my_mzml(mzml_para)
    data.work_flow()


def batch_find_halo(folder_path) -> None:
    """find halo in mzml files"""
    para = load_config()

    # get all mzml files
    mzml_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.mzML'):
                mzml_files.append(os.path.join(root, file))
    for f in mzml_files:
        mzml_para = {'path':f,
                    'feature_list':para.features_list,
                    'asari':para.asari_dict,
                    'mzml':para.mzml_dict,}
        data = my_mzml(mzml_para)
        data.work_flow()

#extract ms2 of rois
def pipeline_extract_ms2_of_rois(mzml_path,project_path,rois:list):
    """extract ms2 of rois"""
    save_halo_evaluation = os.path.normpath(project_path +r'/test_mzml_prediction/'+  mzml_path.split('.')[0].split('\\')[-1] +'_halo_evaluation.csv')
    output_path = os.path.normpath(project_path +r'/test_mzml_prediction/'+  mzml_path.split('.')[0].split('\\')[-1] +'_select_roi_ms2.mgf')
    extract_ms2_of_rois(mzml_path,save_halo_evaluation,output_path,rois)
    


if __file__ == '__main__':
    pass

