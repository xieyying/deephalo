import pandas as pd
import os
from pyteomics import mzml
from .methods import judge_charge
from .methods import feature_extractor,correct_df_charge
import tensorflow as tf
from .base import mzml_base
from .methods import process_spectrum

class analysis_by_asari:
    def __init__(self,path,para) -> None:
        #找到test_mzml文件夹中的mzml文件
        if not os.path.exists(r'./test_mzml_prediction'):
            os.makedirs(r'./test_mzml_prediction')
        self.mzml_path = path
        self.save_tic =  r'./test_mzml_prediction/tic.csv'
        self.save_target = r'./test_mzml_prediction/target.csv'
        self.save_features = r'./test_mzml_prediction/features.csv'
        self.model_path = r'./trained_models/pick_halo_ann.h5'
        # self.features = para.features_list
        self.para = para
        # self.min_prominence_threshold = para.min_prominence_threshold
        # self.mz_tolerance = para.mz_tolerance
        # self.asari_min_intensity = para.asari_min_intensity
        # self.min_timepoints = para.min_timepoints
        # self.min_peak_height = para.min_peak_height

        #mzml参数
        # self.mzml_min_intensity = para.mzml_min_intensity



    def get_mzml_data(self):
        #提取质谱信息
        ms1_spectra = mzml_base.load_mzml_file(self.mzml_path,1)
        print('ms1_spectra:',len(ms1_spectra))

        #获取ms1_spectra中的scan，total ion current
        scan = [i for i in range(len(ms1_spectra))]
        rt = [s['scanList']['scan'][0]['scan start time']*60 for s in ms1_spectra]
        tic = [s['total ion current'] for s in ms1_spectra]

        df = pd.DataFrame({'rt':rt,'tic':tic,'scan':scan,})
        df.to_csv(self.save_tic,index=False)

        #获得features
        df = feature_extractor(self.mzml_path,self.para)
        df.to_csv(self.save_features,index=False)

        #提取ms1_spectra中的质谱信息
        #每张质谱中rt在rtime_left和rtime_right之间的质谱信息，mz在mz-0.02和mz+0.02之间的质谱信息
        #质谱信息包括scan，rt，mz，intensity
        #质谱信息存入新的dataframe

        df2 = pd.DataFrame()
        df2 = process_spectrum(ms1_spectra,df,df2,self.para.mzml_min_intensity)

        #将df2存入csv文件
        # df2.to_csv(self.save_target,index=False)
        df2 = correct_df_charge(df2)
        self.raw_df = df2
    
    def add_label(self):
        #读取数据
        df = self.raw_df
        #提取出new_df_filted中的A1-A0,A2-A1,B-2_relative_intensity,B-1_relative_intensity,A-1_relative_intensity,A-2_relative_intensity列的数值转为numpy数组
        querys = df[self.para.features_list].values
        #将querys中的数据类型转换成float32
        querys = querys.astype('float32')
        clf = tf.keras.models.load_model(self.model_path)
        res = clf.predict(querys)
        base = tf.math.argmax(res[0],1).numpy()
        sub = tf.math.argmax(res[1],1).numpy()
        hydro = tf.math.argmax(res[2],1).numpy()
        #将base,sub,hydro结果转为字符，保存到new_df_filted中的class列
        base = ["'"+str(i)+"'" for i in base]
        sub = ["'"+str(i)+"'" for i in sub]
        #将hydro转为字符，0变为'hyro',1变为'non-hydro'
        hydro = ["'hydro'" if i==0 else "'non-hydro'" for i in hydro]
        # hydro = ["'"+str(i)+"'" for i in hydro]
        df['base_class'] = base
        df['sub_class'] = sub
        df['hydro_class'] = hydro

        #保存到csv文件
        df.to_csv(self.save_target,index=False)

                        
    def asari_workflow(self):

        self.get_mzml_data()
        self.add_label()


if __name__ == '__main__':
    pass