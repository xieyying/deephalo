#import相关模块
import os
import pandas as pd
import tensorflow as tf
from .methods import load_mzml_file,asari_ROI_identify,get_calc_targets,find_isotopologues,ms2ms1_linked_ROI_identify,\
                    add_predict,add_is_halo_isotopes,halo_evaluation,correct_isotopic_peaks
from pyteomics import mzml ,mgf
from ..model_test import timeit
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
    @timeit
    def load_mzml_data(self):
        """加载mzml数据"""
        #MS1数据
        self.mzml_data_all,self.mzml_data =load_mzml_file(self.path)

    
    #分析数据
    @timeit
    def ROI_identify(self):
        """ROI的识别：asari或ms2_linked others"""
        method = self.mzml_dict['ROI_identify_method']
        if method == 'MS':
            self.df_rois = asari_ROI_identify(self.path,self.asari_dict)
        elif method == 'DDA':
            self.df_rois = ms2ms1_linked_ROI_identify(self.mzml_data_all,self.mzml_dict,self.path)
    @timeit
    def extract_features(self):
        """对ROI进行特征提取"""
        df1 = get_calc_targets(self.df_rois)
        df_isotopologues = find_isotopologues(df1,self.mzml_data,self.mzml_dict)
        # correct df_isotopologues
        df_isotopologues = correct_isotopic_peaks(df_isotopologues)
        #保存df_isotopologues
        df_isotopologues.to_csv(self.save_isotopolgues,index=False)
        #添加is_halo_isotopes判断结果
        df_isotopologues = add_is_halo_isotopes(df_isotopologues)
        #保存is_halo_isotopes 为1的isotopologues到self.df_isotopologues
        df_isotopologues = df_isotopologues[df_isotopologues['is_halo_isotopes']==1]
        self.df_isotopologues = df_isotopologues.copy()
        #对isotopologue进行预测
        df_isotopologues = add_predict(self.df_isotopologues,self.model_path,self.feature_list)

    @timeit
    def rois_evaluation(self):
        """对ROI进行halo评估"""
        df = self.df_isotopologues.copy()
        # df = df[df['counter_list_x'].map(lambda x: len(x)) >= self.mzml_dict['min_element_sum']]
        self.halo_evaluation = halo_evaluation(df)

    @timeit
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

        """mzml数据处理流程"""
        if not os.path.exists('./test_mzml_prediction'):
            os.mkdir('./test_mzml_prediction')
        self.load_mzml_data()
        self.ROI_identify()
        self.extract_features()
        self.rois_evaluation()
        self.save_result()



        
                
if __name__ == "__main__":
    pass