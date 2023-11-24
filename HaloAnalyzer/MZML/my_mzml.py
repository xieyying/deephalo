#import相关模块
import os
import pandas as pd
import tensorflow as tf
from .methods import load_mzml_file,asari_ROI_identify,get_tic,get_calc_targets,find_isotopologues,ms2ms1_linked_ROI_identify,\
                    add_predict,add_is_halo_isotopes,halo_evaluation
from pyteomics import mzml ,mgf

class my_mzml:
    """自定义mzml类，包含了mzml数据的加载，ROI的识别，特征提取，halo评估等方法"""
    def __init__(self,para) -> None:
        self.path = para['path']
        print(self.path)
        self.feature_list = para['feature_list']
        self.asari_dict = para['asari']
        self.mzml_dict = para['mzml']
        self.model_path = r'./trained_models/pick_halo_ann.h5'
        self.save_tic =  r'./test_mzml_prediction/' + para['path'].split('.mzML')[0].split('\\')[-1] + '_tic.csv'
        self.save_rois = r'./test_mzml_prediction/' + para['path'].split('.mzML')[0].split('\\')[-1] + '_rois.csv'
        self.save_isotopolgues = r'./test_mzml_prediction/'+  para['path'].split('.mzML')[0].split('\\')[-1] + '_isotopolgues.csv'
        self.save_halo_evaluation = r'./test_mzml_prediction/' +  para['path'].split('.mzML')[0].split('\\')[-1] + '_halo_evaluation.csv'
        self.save_mgf = r'./test_mzml_prediction/' +  para['path'].split('.mzML')[0].split('\\')[-1] +'_roi_ms2.mgf'

        
    #加载数据
    def load_mzml_data(self):
        """加载mzml数据"""
        #MS1数据
        self.mzml_data =load_mzml_file(self.path)
        #全部数据
        self.mzml_data_all = load_mzml_file(self.path,level='all')
    
    #保存原始tic数据
    def save_tic_spectra(self):
        """保存原始tic数据"""
        df_tic = get_tic(self.mzml_data)
        df_tic.to_csv(self.save_tic,index=False)

    #分析数据
    def ROI_identify(self):
        """ROI的识别：asari或ms2_linked others"""
        method = self.mzml_dict['ROI_identify_method']
        if method == 'asari':
            self.df_rois = asari_ROI_identify(self.path,self.asari_dict)
        elif method == 'ms1ms2_linked':
            self.df_rois = ms2ms1_linked_ROI_identify(self.mzml_data_all,self.mzml_dict)

    def extract_features(self):
        """对ROI进行特征提取"""
        df1 = get_calc_targets(self.df_rois)
        df_isotopologues = find_isotopologues(df1,self.mzml_data,self.mzml_dict)
        #对isotopologue进行预测
        df_isotopologues = add_predict(df_isotopologues,self.model_path,self.feature_list)
        #添加is_halo_isotopes判断结果
        df_isotopologues = add_is_halo_isotopes(df_isotopologues)
        #保存is_halo_isotopes 为1的isotopologues到self.df_isotopologues
        self.df_isotopologues = df_isotopologues[df_isotopologues['is_halo_isotopes']==1]
        # self.df_isotopologues = df_isotopologues
  
    def rois_evaluation(self):
        """对ROI进行halo评估"""
        self.halo_evaluation = halo_evaluation(self.df_isotopologues.copy())


    def save_result(self):
        """保存结果"""
        # self.df_rois.to_csv('roi.csv',index=False)
        self.df_isotopologues.to_csv(self.save_isotopolgues,index=False)
        # self.halo_evaluation.to_csv('halo.csv',index=False)
        #找到self.halo_evaluation和self.df_rois中相同的roi_id
        df = pd.merge(self.halo_evaluation,self.df_rois,on='id_roi')

        #为方便查看，更改df的列顺序
        #将df中的'mz'列至于第2列
        cols = list(df)
        cols.insert(1,cols.pop(cols.index('mz')))
        df = df.loc[:,cols]
        #过滤掉df中counter_list_x中元素少于self.mzml_dict['min_element_sum']的行
        df = df[df['counter_list_x'].map(lambda x: len(x)) >= self.mzml_dict['min_element_sum']]
        df.to_csv(self.save_halo_evaluation,index=False)


    def work_flow(self):
        import time
        start = time.time()
        """mzml数据处理流程"""
        if not os.path.exists('./test_mzml_prediction'):
            os.mkdir('./test_mzml_prediction')
        self.load_mzml_data()
        time_load = time.time()
        print('load mzml data cost time: ',time_load-start)
        
        self.save_tic_spectra()
        time_tic = time.time()
        print('save tic cost time: ',time_tic-time_load)

        self.ROI_identify()
        time_roi = time.time()
        print('ROI identify cost time: ',time_roi-time_tic)
        self.extract_features()
        time_feature = time.time()
        print('extract features cost time: ',time_feature-time_roi)
        self.rois_evaluation()
        time_evaluation = time.time()
        print('evaluation cost time: ',time_evaluation-time_feature)
        self.save_result()
        time_save = time.time()
        print('save result cost time: ',time_save-time_evaluation)


        
                
if __name__ == "__main__":
    pass