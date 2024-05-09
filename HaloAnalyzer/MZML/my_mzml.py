#import相关模块
import os
import pandas as pd

import tensorflow as tf
from .methods import load_mzml_file,asari_ROI_identify,get_calc_targets,find_isotopologues,ms2ms1_linked_ROI_identify,\
                      add_predict,add_is_isotopes,halo_evaluation,correct_isotopic_peaks,merge_close_values,blank_subtract,extract_ms2_of_rois
from pyteomics import mzml ,mgf
from ..model_test import timeit
from .test_method import get_ROIs
class my_mzml:
    """自定义mzml类，包含了mzml数据的加载，ROI的识别，特征提取，halo评估等方法"""
    def __init__(self,para) -> None:
        self.path = para['path']
        print(self.path)
        self.feature_list = para['feature_list']
        self.asari_dict = para['asari']
        self.mzml_dict = para['mzml']
        self.model_path = r'./trained_models/pick_halo_ann.h5'
        self.save_tic =  r'./test_mzml_prediction/' + para['path'].split('.mz')[0].split('\\')[-1] + '_tic.csv'
        self.save_rois = r'./test_mzml_prediction/' + para['path'].split('.mz')[0].split('\\')[-1] + '_rois.csv'
        self.save_isotopolgues = r'./test_mzml_prediction/'+  para['path'].split('.mz')[0].split('\\')[-1] + '_isotopolgues.csv'
        self.save_halo_evaluation = r'./test_mzml_prediction/' +  para['path'].split('.mz')[0].split('\\')[-1] + '_halo_evaluation.csv'
        self.save_blank_halo_evaluation = r'./test_mzml_prediction/blank/' +  para['path'].split('.mz')[0].split('\\')[-1] + '_halo_evaluation.csv'
        self.save_mgf = r'./test_mzml_prediction/mgf/' +  para['path'].split('.mz')[0].split('\\')[-1] +'_roi_ms2.mgf'
        
    #加载数据
    @timeit
    def load_mzml_data(self):
        """加载mzml数据"""
        #MS1数据
        self.mzml_data_all,self.mzml_data =load_mzml_file(self.path,self.mzml_dict)
        self.total_MS1_scan_num = len(self.mzml_data)
        self.total_MS2_scan_num = len(self.mzml_data_all) - len(self.mzml_data)
    #分析数据
    @timeit
    def ROI_identify(self):
        """ROI的识别：asari或ms2_linked others"""
        method = self.mzml_dict['ROI_identify_method']
        if method == 'MS':
            self.df_rois = asari_ROI_identify(self.path,self.asari_dict)
        elif method == 'DDA':
            self.df_rois, self.ms2ms1_linked_df = ms2ms1_linked_ROI_identify(self.mzml_data_all,self.mzml_dict,self.path)
            # self.df_rois.to_csv(r'C:\Users\xyy\Desktop\DDA_roi.csv',index=False)
        elif method == 'peak_only':
            self.df_rois = get_ROIs(self.path)
           
            # self.df_rois.to_csv(r'C:\Users\xyy\Desktop\peak_only.csv',index=False)
            
    @timeit
    def extract_features(self):
        """对ROI进行特征提取"""

        df1 = get_calc_targets(self.df_rois)
        df1.to_csv(r'C:\Users\xyy\Desktop\after_get_calc_targets.csv',index=False)
        df_isotopologues = find_isotopologues(df1,self.mzml_data,self.mzml_dict)
        # correct df_isotopologues
        df_isotopologues = correct_isotopic_peaks(df_isotopologues)
     
        #添加is_isotopes判断结果
        df_isotopologues = add_is_isotopes(df_isotopologues)
        #保存is_isotopes为1的isotopologues到self.df_isotopologues
        df_isotopologues = df_isotopologues[df_isotopologues['is_isotopes']==1]
        df_isotopologues = df_isotopologues.groupby('id_roi').filter(lambda x: len(x) >= 3)  # TODO: change this threshold
        self.df_isotopologues = df_isotopologues.copy()
        #预测
        self.df_isotopologues = add_predict(self.df_isotopologues,self.model_path,self.feature_list)

    @timeit
    def rois_evaluation(self):
        """对ROI进行halo评估"""
        df = self.df_isotopologues.copy()
        #对df进行预测        
        self.halo_evaluation,self.roi_mean_for_pred,self.df_roi_total_for_prediction = halo_evaluation(df)
        self.feature_num = len(self.halo_evaluation)
        roi_mean_prediction = add_predict(self.roi_mean_for_pred,self.model_path,self.feature_list)
        roi_total_prediction = add_predict(self.df_roi_total_for_prediction,self.model_path,self.feature_list)
        #添加基于平均强度、平均分子量，以及总强度和平均分子量的预测结果
        self.halo_evaluation['roi_mean_pred'] = roi_mean_prediction['class_pred']
        self.halo_evaluation['roi_total_pred'] = roi_total_prediction['class_pred']
        #将df_rois和halo_evaluation合并
        self.merge_df = pd.merge(self.halo_evaluation,self.df_rois,on='id_roi')
        #为方便查看，更改df的列顺序
        #将df中的'mz'列至于第2列,将'charge'列至于第3列
        cols = ['id_roi','mz', 'charge'] + [col for col in self.merge_df.columns if col not in ['id_roi','mz', 'charge']]
        self.merge_df = self.merge_df.reindex(columns=cols)
        
        roi_mean_halo_score = self.merge_df['roi_mean_pred']
   
        # 如果roi_mean_halo_score为0，1，2则为1，如果为其他则为0
        roi_mean_halo_score = roi_mean_halo_score.apply(lambda x: 1 if x in [0,1,2] else 0)
        #计算H-score
        self.merge_df['H-score'] = (self.merge_df['scan_based_halo_score'])/300 + (roi_mean_halo_score)/3 + (self.merge_df['scan_based_halo_ratio'])/3
        self.merge_df.to_csv(r'C:\Users\xyy\Desktop\halo_before_filter.csv',index=False)
        
    @timeit
    def filter_result(self,min_element_sum=3,H_score=0.8):
        """保存结果"""
        #筛选出scan_based_halo_class_list长度大于等于min_element_sum的结果
        self.merge_df = self.merge_df[self.merge_df['scan_based_halo_class_list'].apply(len) >= min_element_sum]
        self.merge_df = self.merge_df[self.merge_df['precursor_ints_sum'] > 1e6]  # TODO: change this threshold
        
        # 去除precursor_ints中所有的值都小于1e5的结果（意味着feature中每个scan的强度均小于1e5），
        # 为了去除一些低强度化合物，强度太低的scan的二级质谱噪音影响太大，不利于后续的分析
        mask = self.merge_df['precursor_ints'].apply(lambda x: not all(i < 5e4 for i in x))   # TODO: change this threshold
        self.merge_df = self.merge_df[mask]
                
        #筛选H-score大于阈值的结果
        self.df = self.merge_df[self.merge_df['H-score']>=H_score]
        
    def common_work_flow(self):
        self.load_mzml_data()
        self.ROI_identify()
        self.extract_features()
        self.rois_evaluation()

    def blank_analyses(self):
        """blank_subtract"""
        self.common_work_flow()
        # 为尽可能去除空白，保留H-score大于0的结果
        self.merge_df = self.merge_df[self.merge_df['H-score']>0] 
        self.merge_df.to_csv(self.save_blank_halo_evaluation,index=False)
        return self.feature_num
    
    def blank_combine(self):
        """blank_combine"""
        # 合并./test_mzml_prediction/blank中所有csv文件，如果各个csv文件中mz相差小于100 ppm，RTmean相差小于0.5 min，则合并为一个mz，其平均值为各个mz的平均值，平均RTmean为各个RT的平均值
        files = os.listdir('./test_mzml_prediction/blank')
        df_blank_merge = pd.DataFrame()
       
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join('./test_mzml_prediction/blank', file)
                df_blank_merge_ = pd.read_csv(file_path)
                df_blank_merge = pd.concat([df_blank_merge, df_blank_merge_], ignore_index=True)

        self.df_blank_merge = merge_close_values(df_blank_merge)
        self.df_blank_merge= self.df_blank_merge.reset_index(drop=True)
        self.df_blank_merge.to_csv('./test_mzml_prediction/blank/merged_blank_halo.csv', index=False)
        return len(self.df_blank_merge)
        
    @timeit
    def work_flow_subtract_blank(self,H_score=0.8):
        """mzml数据处理流程"""
        if not os.path.exists('./test_mzml_prediction'):
            os.mkdir('./test_mzml_prediction')
        self.common_work_flow() 
        self.filter_result(int(self.mzml_dict['min_element_sum']),H_score)
        #统计
        n = 0 # 没找到halo化合物的文件数
        h = 0 # 找到halo化合物的文件数
        h_f = 0 # halo化合物的个数
        if len(self.df) == 0:
            n += 1
            h_f = 0
            print('没有找到halo化合物')
        else:
            df_blank = pd.read_csv('./test_mzml_prediction/blank/merged_blank_halo.csv')
            df_blank_subtract = blank_subtract(self.df,df_blank)
            if len(df_blank_subtract) == 0:
                n += 1
                h_f = 0
                print('没有找到halo化合物')
            else:
                h += 1
                df_blank_subtract.to_csv(self.save_halo_evaluation,index=False)
                h_f = len(df_blank_subtract)
                if self.mzml_dict['ROI_identify_method'] == 'DDA':
                    if not os.path.exists('./test_mzml_prediction/mgf'):
                        os.mkdir('./test_mzml_prediction/mgf')
                    extract_ms2_of_rois(self.mzml_data_all,df_blank_subtract,self.ms2ms1_linked_df,self.save_mgf)
        return n,h,self.total_MS1_scan_num,self.total_MS2_scan_num,self.feature_num,h_f
                             
    def work_flow_no_blank(self,H_score=0.8):
        """mzml数据处理流程"""
        if not os.path.exists('./test_mzml_prediction'):
            os.mkdir('./test_mzml_prediction')
        self.common_work_flow() 
        self.filter_result(int(self.mzml_dict['min_element_sum']),H_score)
        n = 0
        h = 0
        h_f = 0
        if len(self.df) == 0:
            n += 1
            print('没有找到halo化合物')
            h_f = 0
            
        else:
            self.df.to_csv(self.save_halo_evaluation,index=False)
            print('已保存')
            h += 1
            h_f = len(self.df)
            if self.mzml_dict['ROI_identify_method'] == 'DDA':
                if not os.path.exists('./test_mzml_prediction/mgf'):
                    os.mkdir('./test_mzml_prediction/mgf')
                extract_ms2_of_rois(self.mzml_data_all,self.df,self.ms2ms1_linked_df,self.save_mgf)
        return n, h, self.total_MS1_scan_num, self.total_MS2_scan_num, self.feature_num, h_f
    
if __name__ == "__main__":
    pass