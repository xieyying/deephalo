#import相关模块
import os
import pandas as pd
import tensorflow as tf
from .methods import load_mzml_file,asari_ROI_identify,get_calc_targets,find_isotopologues,ms2ms1_linked_ROI_identify,\
                      add_predict,add_is_isotopes,halo_evaluation,correct_isotopic_peaks,merge_close_values,blank_subtract
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
        self.save_mgf = r'./test_mzml_prediction/' +  para['path'].split('.mz')[0].split('\\')[-1] +'_roi_ms2.mgf'


        
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
            # self.df_rois.to_csv(self.save_rois,index=False)
        elif method == 'peak_only':
            self.df_rois = get_ROIs(self.path)
            
    @timeit
    def extract_features(self):
        """对ROI进行特征提取"""
        df1 = get_calc_targets(self.df_rois)
        df_isotopologues = find_isotopologues(df1,self.mzml_data,self.mzml_dict)
        # df_isotopologues.to_csv(r'C:\Users\xyy\Desktop\after_find.csv',index=False)
        # correct df_isotopologues
        df_isotopologues = correct_isotopic_peaks(df_isotopologues)
        # df_isotopologues.to_csv(r'C:\Users\xyy\Desktop\after_correct.csv',index=False)
     
        #添加is_isotopes判断结果
        df_isotopologues = add_is_isotopes(df_isotopologues)
        #保存is_isotopes 为1的isotopologues到self.df_isotopologues
        df_isotopologues = df_isotopologues[df_isotopologues['is_isotopes']==1]
        df_isotopologues = df_isotopologues.groupby('id_roi').filter(lambda x: len(x) >= 3)
        # df_isotopologues.to_csv(r'C:\Users\xyy\Desktop\after_correct.csv',index=False)
     
        self.df_isotopologues = df_isotopologues.copy()
        #对isotopologue进行预测
        self.df_isotopologues = add_predict(self.df_isotopologues,self.model_path,self.feature_list)

    @timeit
    def rois_evaluation(self):
        """对ROI进行halo评估"""
        df = self.df_isotopologues.copy()
        #对df进行预测        
        self.halo_evaluation,self.roi_mean_for_pred,self.df_roi_total_for_prediction = halo_evaluation(df)
        roi_mean_prediction = add_predict(self.roi_mean_for_pred,self.model_path,self.feature_list)
        roi_total_prediction = add_predict(self.df_roi_total_for_prediction,self.model_path,self.feature_list)
        # roi_mean_prediction.to_csv(r'C:\Users\xyy\Desktop\python\HaloAnalyzer_training\020_main_5\test_mzml_prediction\roitest.csv',index=False)
        
        self.halo_evaluation['roi_mean_pred'] = roi_mean_prediction['class_pred']
        self.halo_evaluation['roi_total_pred'] = roi_total_prediction['class_pred']

    @timeit
    def filter_result(self,H_score=0.6):
        """保存结果"""
        # self.df_rois.to_csv('roi.csv',index=False)
        # self.df_isotopologues.to_csv(self.save_isotopolgues,index=False)
      
        #找到self.halo_evaluation和self.df_rois中相同的roi_id
        df = pd.merge(self.halo_evaluation,self.df_rois,on='id_roi')

        #为方便查看，更改df的列顺序
        #将df中的'mz'列至于第2列
        cols = list(df)
        cols.insert(1,cols.pop(cols.index('mz')))
        df = df.loc[:,cols]
        roi_mean_halo_score = df['roi_mean_pred']
        
        # 如果roi_mean_halo_score为0，1，2则为1，如果为其他则为0
        roi_mean_halo_score = roi_mean_halo_score.apply(lambda x: 1 if x in [0,1,2] else 0)

        # #Se
        # roi_mean_halo_score = roi_mean_halo_score.apply(lambda x: 1 if x in [3] else 0)

        #计算H-score
        df['H-score'] = (df['scan_based_halo_score'])/300 + (roi_mean_halo_score)/3 + (df['scan_based_halo_ratio'])/3
        #筛选H-score大于阈值的结果
        self.df = df[df['H-score']>=H_score]

    def common_work_flow(self):
        self.load_mzml_data()
        self.ROI_identify()
        self.extract_features()
        self.rois_evaluation()

    def blank_analyses(self,H_score=0.1):
        """blank_subtract"""
        if not os.path.exists('./test_mzml_prediction/blank'):
            os.mkdir('./test_mzml_prediction/blank')
        self.common_work_flow()
        self.filter_result(H_score)
        if len(self.df) == 0:
            print('没有找到halo化合物')
            return
        self.df.to_csv(self.save_blank_halo_evaluation,index=False)
    def blank_combine(self):
        """blank_combine"""
        # 合并./test_mzml_prediction/blank中所有csv文件，如果各个csv文件中mz相差小于10 ppm，RTmean相差小于0.5 min，则合并为一个mz，其平均值为各个mz的平均值，平均RTmean为各个RT的平均值
        files = os.listdir('./test_mzml_prediction/blank')
        df_blank_merge = pd.DataFrame()
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join('./test_mzml_prediction/blank', file)
                df_blank_merge_ = pd.read_csv(file_path)
                df_blank_merge = pd.concat([df_blank_merge, df_blank_merge_], ignore_index=True)

        self.df_blank_merge = df_blank_merge.drop_duplicates()
        self.df_blank_merge = merge_close_values(self.df_blank_merge)
        self.df_blank_merge= self.df_blank_merge.reset_index(drop=True)

        self.df_blank_merge.to_csv('./test_mzml_prediction/blank/merged_blank_halo.csv', index=False)

    def work_flow_subtract_blank(self,H_score=0.8):
        """mzml数据处理流程"""
        # 没找到halo化合物的文件
        # logger = logging.getLogger(__name__) 
        if not os.path.exists('./test_mzml_prediction'):
            os.mkdir('./test_mzml_prediction')
        self.common_work_flow() 
        self.filter_result(H_score)
        if len(self.df) == 0:
            print('没有找到halo化合物')
    
            return
        else:
            df_blank = pd.read_csv('./test_mzml_prediction/blank/merged_blank_halo.csv')
            df_blank_subtract = blank_subtract(self.df,df_blank)
            if len(df_blank_subtract) == 0:
                print('没有找到halo化合物')
                # logger.info(self.path,'没有找到halo化合物')
                return
            else:
                df_blank_subtract.to_csv(self.save_halo_evaluation,index=False)
                # logger.info(self.path,'找到halo化合物')
    def work_flow_no_blank(self,H_score=0.8):
        """mzml数据处理流程"""
        if not os.path.exists('./test_mzml_prediction'):
            os.mkdir('./test_mzml_prediction')
        self.common_work_flow() 
        self.filter_result(H_score)
        if len(self.df) == 0:
            print('没有找到halo化合物')
            return
        else:
            self.df.to_csv(self.save_halo_evaluation,index=False)
    
    def work_flow_given_roi(self,roi_df):
        """mzml数据处理流程"""
        if not os.path.exists('./test_mzml_prediction'):
            os.mkdir('./test_mzml_prediction')

        self.load_mzml_data()
        self.df_rois = pd.read_csv(self.save_rois)
        self.extract_features()
        self.rois_evaluation()
        


                
if __name__ == "__main__":
    pass