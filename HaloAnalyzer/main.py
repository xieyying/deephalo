#import packages
import os
import shutil
from .Dataset.my_dataset import dataset,datasets
from .Model.my_model import my_model,my_search
from .MZML.my_mzml import my_mzml
from .parameters import run_parameters
import logging

#load config file
def load_config() -> dict:
    """
    load config file
    
    Returns:
    class: run_parameters
    """
    return run_parameters()
     
#Dataset Pipeline
def pipeline_dataset() -> None:
    """
    generate dataset for training and testing
    """
    para = load_config()
    datas = []
    for data in para.datasets:
        datas.append(dataset(data[0],data[1]).data)
    raw_data = datasets(datas)
    raw_data.work_flow(para.mz_start,para.mz_end,para.elements_list,'base')
    raw_data.work_flow(para.mz_start,para.mz_end,para.elements_list,'Fe')
    raw_data.work_flow(para.mz_start,para.mz_end,para.elements_list,'B')
    raw_data.work_flow(para.mz_start,para.mz_end,para.elements_list,'Se')
    raw_data.work_flow(para.mz_start,para.mz_end,para.elements_list,'hydro',rates=para.rate_for_hydro)


#Model Pipeline
def pipeline_model(mode = 'manual') -> None:
    """
    train model and save model
    
    Args:
    mode: str, default 'manual', optional 'manual','search','feature_importance','read_feature_importance'
    """
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
    # elif mode == 'feature_importance':#是否还要保留
    #     model = KerasModelWrapper(model_para)
    #     model.work_flow()
    # elif mode == 'read_feature_importance':#是否还要保留
    #     model = KerasModelWrapper(model_para)
    #     model.read_feature_importance()

def get_parameters(mzml_path):
    """
    get parameters for processing mzml file
    
    Args:
    mzml_path: str, path of mzml file

    Returns:
    dict: parameters for processing mzml file
    mzml_para = {'path':mzml_path,
                'feature_list':para.features_list,
                'mzml':para.mzml_dict,}
    """
    para = load_config()
    mzml_para = {'path':mzml_path,
                'feature_list':para.features_list,
                # 'asari':para.asari_dict,
                'mzml':para.mzml_dict,}
    return mzml_para

def blank_analyze_mzml(blank_path)-> None:
    """
    analyze blank mzml file, extract features and combine features
    
    Args:
    blank_path: str, path of blank mzml file
    """
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
                    # 'asari':para.asari_dict,
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
    """
    find halo in mzml file
    
    Args:
    mzml_path: str, path of mzml file

    Returns:
    int: n, Number of files without halogenates
    int: h, Number of files containing halogenates
    int: total_MS1_scan_num, Total number of MS1 scans processed
    int: total_MS2_scan_num, Total number of MS2 scans processed
    int: feature_num, Total number of features extracted
    int: h_f, Total number of features containing halogenates

    """
    n, h, total_MS1_scan_num, total_MS2_scan_num, feature_num, h_f = my_mzml(get_parameters(mzml_path)).work_flow_no_blank()
    return n, h, total_MS1_scan_num, total_MS2_scan_num, feature_num, h_f

def batch_find_halo_no_blank(folder_path) -> None:
    """
    find halo in mzml files, not subtract blank
    
    Args:
    folder_path: str, path of folder containing mzml files

    Returns:
    int: n, Number of files without halogenates
    int: h, Number of files containing halogenates
    int: total_MS1_scan_num, Total number of MS1 scans processed
    int: total_MS2_scan_num, Total number of MS2 scans processed
    int: feature_num, Total number of features extracted
    int: h_f, Total number of features containing halogenates
    
    """
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
    """
    find halo in mzml file
    
    Args:
    mzml_path: str, path of mzml file
    
    Returns:
    int: n, Number of files without halogenates
    int: h, Number of files containing halogenates
    int: total_MS1_scan_num, Total number of MS1 scans processed
    int: total_MS2_scan_num, Total number of MS2 scans processed
    int: feature_num, Total number of features extracted
    int: h_f, Total number of features containing halogenates
    """
    n, h, total_MS1_scan_num, total_MS2_scan_num, feature_num, h_f = my_mzml(get_parameters(mzml_path)).work_flow_subtract_blank()
    return n, h, total_MS1_scan_num, total_MS2_scan_num, feature_num, h_f

def batch_find_halo_substrate_blank(folder_path) -> None:
    """
    find halo in mzml files, subtract blank
    
    Args:
    folder_path: str, path of folder containing mzml file s

    Returns:
    int: n, Number of files without halogenates
    int: h, Number of files containing halogenates
    int: total_MS1_scan_num, Total number of MS1 scans processed
    int: total_MS2_scan_num, Total number of MS2 scans processed
    int: feature_num, Total number of features extracted
    int: h_f, Total number of features containing halogenates

    """
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

def pipeline_analyze_mzml(args):
    # Check if the directory exists, if not, create it
    if not os.path.exists('./test_mzml_prediction'):
        os.makedirs('./test_mzml_prediction')
    # Check if the file exists, if so, remove it
    if os.path.exists('./test_mzml_prediction/run.txt'):
        os.remove('./test_mzml_prediction/run.log')
    # Now you can safely set up your logging
    logging.basicConfig(filename='./test_mzml_prediction/run.log', level=logging.INFO)

    blank_path = args.blank
    mzml_path = args.input
    #扣除blank的情况
    if blank_path != None:
        #如果存在./test_mzml_prediction/blank/merged_blank_halo.csv则不再进行blank_analyze_mzml
        if not os.path.exists('./test_mzml_prediction/blank/merged_blank_halo.csv') or args.overwrite_blank:
            blank_analyze_mzml(blank_path)
        #根据mzml_path的类型选择不同的分析方式，文件夹则批量处理，文件则单个处理
        if os.path.isdir(mzml_path):
            batch_find_halo_substrate_blank(mzml_path)
        elif os.path.isfile(mzml_path):
            pipeline_find_halo_substrate_blank(mzml_path)
        else:
            print("Please specify a mzML file or a folder containing mzML files to analyze.")
    #不扣除blank的情况
    else:
        if os.path.isdir(mzml_path):
            batch_find_halo_no_blank(mzml_path)
        elif os.path.isfile(mzml_path):
            pipeline_find_halo_no_blank(mzml_path)
        else:
            print("Please specify a mzML file or a folder containing mzML files to analyze.")

    # with open(r'test_mzml_prediction/log.txt','w') as f: f.write(mzml_path)

def pipeline_viz_result():
    #更新config中的vis_path
    #     parameters = run_parameters()
    #     c = parameters.config
    #     c['visualization']['path'] = args.project
    #     parameters.update(c)
    #     # 运行vis.py
    #     vis_path = importlib_resources.files('HaloAnalyzer') / 'vis.py'
    #     print(vis_path)
    #     os.system('python -m streamlit run %s' %vis_path)
    # elif args.run == 'extract_ms2':
    #     mzml_path = args.input
    #     project_path = args.project
    #     rois_list = args.list_rois
    #     if mzml_path == None:
    #         print("Please specify a mzML file to analyze.")
    #     if rois_list == None:
    #         print("Please specify a list of rois to extract ms2 spectra.")
    #     if project_path == None:
    #         print("Please specify a project path.")
    #     if mzml_path != None and rois_list != None and project_path != None:
    #         pipeline_extract_ms2_of_rois(mzml_path,project_path,rois_list)
if __file__ == '__main__':
    pass

