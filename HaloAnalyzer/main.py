#import packages
import os
import shutil
from .Dataset.my_dataset import dataset,datasets
from .Model.my_model import my_model,my_search,KerasModelWrapper
from .MZML.my_mzml import my_mzml
from .parameters import run_parameters
import logging

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
    # raw_data.work_flow(para.mz_start,para.mz_end,para.elements_list,'base')
    # raw_data.work_flow(para.mz_start,para.mz_end,para.elements_list,'noise',repeats=para.repeat_for_noise)
    # raw_data.work_flow(para.mz_start,para.mz_end,para.elements_list,'Fe')
    # raw_data.work_flow(para.mz_start,para.mz_end,para.elements_list,'B')
    # raw_data.work_flow(para.mz_start,para.mz_end,para.elements_list,'Se')
    raw_data.work_flow(para.mz_start,para.mz_end,para.elements_list,'hydro',rates=para.rate_for_hydro)
    # data.data_statistics_customized()

#Model Pipeline
def pipeline_model(mode = 'manual') -> None:
    """train model and save model"""
    #加载参数
    para = load_config()

    #根据配置文件选择训练数据
    paths = ['./dataset/base89to6.csv']
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
    if para.use_overloaded_data == 'True':
        paths.append('./dataset/NPatlas_COCONUT_base6789_b1234_adding_10_intensity.csv')
    
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
    elif mode == 'feature_importance':
        model = KerasModelWrapper(model_para)
        model.work_flow()
    elif mode == 'read_feature_importance':
        model = KerasModelWrapper(model_para)
        model.read_feature_importance()

def get_parameters(mzml_path):
    para = load_config()
    mzml_para = {'path':mzml_path,
                'feature_list':para.features_list,
                'asari':para.asari_dict,
                'mzml':para.mzml_dict,}
    return mzml_para

def blank_analyze_mzml(blank_path)-> None:
    """analyze blank mzml file"""
    para = load_config()
    blank_files = []
    # Create the folder
    folder_path = './test_mzml_prediction/blank'

    # If the folder exists, delete it
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    # Create the folder
    os.mkdir(folder_path)
    for root, dirs, files in os.walk(blank_path):
        for file in files:
            if file.endswith('.mzML') or file.endswith('.mzml') or file.endswith('.mzXML') or file.endswith('.mzxml'):
                blank_files.append(os.path.join(root, file))
    b_f = 0 # blank文件的个数
    b_p = 0 # blank文件处理成功的个数
    b_e = 0 # blank文件处理失败的个数
    MS1_scan_num = 0 # Total number of MS1 scans processed
    MS2_scan_num = 0 # Total number of MS2 scans processed
    feature_num = 0 # Total number of features extracted
    
    for f in blank_files:
        b_f += 1
        mzml_para = {'path':f,
                    'feature_list':para.features_list,
                    'asari':para.asari_dict,
                    'mzml':para.mzml_dict,}
       
        mzml_para['mzml']['ROI_identify_method'] = 'peak_only'  # 为了尽可能去除空白文件中的feature，blank_analyze_mzml中ROI_identify_method均设为'peak_only'
        
        try:
            b_p += 1
            data = my_mzml(mzml_para)
            feature_num_= data.blank_analyses()
            feature_num += feature_num_ 
        except: 
            print('Encounter error in dealing with :',f)
        
            b_e += 1
            pass

        feature_combined_num = data.blank_combine()
        logging.info(f"Total blank files: {b_f}, Successfully dealt with blank files: {b_p}, Blank files encountered with errors: {b_e}")
        logging.info(f"Total number of MS1 scans processed: {MS1_scan_num}, Total number of MS2 scans processed: {MS2_scan_num}")
        logging.info(f"Total number of features extracted: {feature_num}, Total number of features combined: {feature_combined_num}")
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        logging.getLogger('').addHandler(console)
        
#Find Halo Pipeline
def pipeline_find_halo_no_blank(mzml_path) -> None:
    """find halo in mzml file"""
    n, h, total_MS1_scan_num, total_MS2_scan_num, feature_num, h_f = my_mzml(get_parameters(mzml_path)).work_flow_no_blank()
    return n, h, total_MS1_scan_num, total_MS2_scan_num, feature_num, h_f

def batch_find_halo_no_blank(folder_path) -> None:
    """find halo in mzml files"""
    # get all mzml files
    mzml_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.mzML') or file.endswith('.mzml') or file.endswith('.mzXML') or file.endswith('.mzxml'):
                mzml_files.append(os.path.join(root, file))
    error_file = []
    t = 0 # Total number of files processed
    d = 0 # Number of files processed successfully
    e = 0 # Number of files that encountered errors during processing
    n = 0 # Number of files without halogenates
    h = 0 # Number of files containing halogenates
    MS1_scan_num = 0 # Total number of MS1 scans processed
    MS2_scan_num = 0 # Total number of MS2 scans processed
    feature_num = 0 # Total number of features extracted
    h_f = 0 # Total number of features containing halogenates
    #错误文件
    for f in mzml_files:
        t += 1
        try:
            n_,h_,MS1_scan_num_,MS2_scan_num_,feature_num_,h_f_ = pipeline_find_halo_no_blank(f)
            logging.info(f"Number of MS1 scans processed: {MS1_scan_num_}, Number of MS2 scans processed: {MS2_scan_num_}")
            logging.info(f"Number of features extracted: {feature_num_}, Number of features containing halogenates: {h_f_}")
            print('-----------------')
            MS1_scan_num += MS1_scan_num_
            MS2_scan_num += MS2_scan_num_
            feature_num += feature_num_
            h_f += h_f_
            d += 1
            if n_:
                n += 1
            elif h_:
                h += 1  
        except:
            print('Encounter error in dealing with :',f)
            e += 1
            error_file.append(f)
            pass
    logging.info(f"Total files: {t}, Successfully dealt with files: {d}, Files encountered with errors: {e}, No halo files: {n}, Halo files: {h}")
    logging.info(f"Total number of MS1 scans processed: {MS1_scan_num}, Total number of MS2 scans processed: {MS2_scan_num}")
    logging.info(f"Total number of features extracted: {feature_num}", f"Total number of features containing halogenates: {h_f}")
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    with open(r'test_mzml_prediction/error_file.txt','w') as f:
       for i in error_file:
          f.write(i+'\n')  

def pipeline_find_halo_substrate_blank(mzml_path) -> None:
    """find halo in mzml file"""
    n, h, total_MS1_scan_num, total_MS2_scan_num, feature_num, h_f = my_mzml(get_parameters(mzml_path)).work_flow_subtract_blank()
    return n, h, total_MS1_scan_num, total_MS2_scan_num, feature_num, h_f

def batch_find_halo_substrate_blank(folder_path) -> None:
    """find halo in mzml files"""
    MS1_scan_num = 0 # Total number of MS1 scans processed
    MS2_scan_num = 0 # Total number of MS2 scans processed
    total_feature_num = 0 # Total number of features extracted
    halo_feature_num = 0 # Total number of features containing halogenates
    
    # get all mzml files
    mzml_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.mzML') or file.endswith('.mzml') or file.endswith('.mzXML') or file.endswith('.mzxml'):
                mzml_files.append(os.path.join(root, file))
    #错误文件
    error_file = []
    t = 0 # Total number of files processed
    d = 0 # Number of files processed successfully
    e = 0 # Number of files that encountered errors during processing
    n = 0 # Number of files without halogenates
    h = 0 # Number of files containing halogenates
    
    for f in mzml_files:
        t += 1
        try:
            n_, h_, MS1_scan_num_, MS2_scan_num_, feature_num_, h_f = pipeline_find_halo_substrate_blank(f)
            logging.info(f"Number of MS1 scans processed: {MS1_scan_num_}, Number of MS2 scans processed: {MS2_scan_num_}")
            logging.info(f"Number of features extracted: {feature_num_}, Number of features containing halogenates: {h_f}")
            logging.info('-----------------')

            d += 1
            if n_:
                n += 1
            elif h_:
                h += 1
            MS1_scan_num += MS1_scan_num_
            MS2_scan_num += MS2_scan_num_
            total_feature_num += feature_num_
            halo_feature_num += h_f
        except:
            print('Encounter error in dealing with :',f)
            print('-----------------')
            e += 1
            error_file.append(f)
            pass
    logging.info(f"Total files: {t}, Successfully dealt with files: {d}, Files encountered with errors: {e}, No halo files: {n}, Halo files: {h}")
    logging.info(f"Total number of MS1 scans processed: {MS1_scan_num}, Total number of MS2 scans processed: {MS2_scan_num}")
    logging.info(f"Total number of features extracted: {total_feature_num}, Total number of features containing halogenates: {halo_feature_num}")
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    # 错误文件写入
    with open(r'test_mzml_prediction/error_file.txt','w') as f:
       for i in error_file:
          f.write(i+'\n')

if __file__ == '__main__':
    pass

