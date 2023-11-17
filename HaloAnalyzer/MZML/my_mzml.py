#import相关模块
import os
import pandas as pd
import tensorflow as tf
from .methods import load_mzml_file,asari_ROI_identify,get_tic,get_calc_targets,find_isotopologues,ms2ms1_linked_ROI_identify,\
                    add_predict,add_is_halo_isotopes,halo_evaluation
from pyteomics import mzml ,mgf

class my_mzml:
    def __init__(self,para) -> None:
        self.path = para['path']
        self.feature_list = para['feature_list']
        self.asari_dict = para['asari']
        self.mzml_dict = para['mzml']
        self.model_path = r'./trained_models/pick_halo_ann.h5'
        self.save_tic =  r'./test_mzml_prediction/tic.csv'
        self.save_rois = r'./test_mzml_prediction/rois.csv'
        self.save_isotopolgues = r'./test_mzml_prediction/isotopolgues.csv'
        self.save_halo_evaluation = r'./test_mzml_prediction/halo_evaluation.csv'
        self.save_mgf = r'./test_mzml_prediction/roi_ms2.mgf'

        
    #加载数据
    def load_mzml_data(self):
        self.mzml_data =load_mzml_file(self.path)
        self.mzml_data_all = load_mzml_file(self.path,level='all')
    #保存原始tic数据
    def save_tic_spectra(self):
        df_tic = get_tic(self.mzml_data)
        df_tic.to_csv(self.save_tic,index=False)

    #分析数据
    #ROI的识别：asari或ms2_linked others
    def ROI_identify(self):
        method = self.mzml_dict['ROI_identify_method']
        if method == 'asari':
            self.df_rois = asari_ROI_identify(self.path,self.asari_dict)

        elif method == 'ms1ms2_linked':
            self.df_rois = ms2ms1_linked_ROI_identify(self.mzml_data_all,self.mzml_dict)

    #对ROI进行特征提取
    def extract_features(self):
        df1 = get_calc_targets(self.df_rois)
        df_isotopologues = find_isotopologues(df1,self.mzml_data)
        #对isotopologue进行预测
        df_isotopologues = add_predict(df_isotopologues,self.model_path,self.feature_list)
        #添加is_halo_isotopes判断结果
        df_isotopologues = add_is_halo_isotopes(df_isotopologues)
        #保存isotopologues到self.df_isotopologues
        self.df_isotopologues = df_isotopologues
  
    #对ROI进行halo评估
    def rois_evaluation(self):
        self.halo_evaluation = halo_evaluation(self.df_isotopologues.copy())


    def save_result(self):
        self.df_rois.to_csv(self.save_rois,index=False)
        self.df_isotopologues.to_csv(self.save_isotopolgues,index=False)
        self.halo_evaluation.to_csv(self.save_halo_evaluation,index=False)

    def work_flow(self):
        if not os.path.exists('./test_mzml_prediction'):
            os.mkdir('./test_mzml_prediction')
        self.load_mzml_data()
        self.save_tic_spectra()
        self.ROI_identify()
        self.extract_features()
        self.rois_evaluation()
        self.save_result()


        
                
if __name__ == "__main__":
    path = r'E:\XinBackup\source_data\mzmls\Vancomycin.mzML'
    features_list = [
    "new_a0_ints",
    "new_a1_ints",
    "new_a2_ints",
    "new_a3_ints",
    "new_a2_a1_10"]
    
    t = my_mzml(path,features_list) 
    t.work_flow()
    t.extract_ms2_of_rois([1,])
