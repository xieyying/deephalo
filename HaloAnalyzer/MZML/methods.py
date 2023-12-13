from pyteomics import mzml ,mgf,mzxml
import pandas as pd
from .methods_sub import feature_extractor,fliter_mzml_data,get_mz_max,get_charge,get_isotopic_peaks,ROIs,MS1_MS2_connected,is_halo_isotopes,roi_halo_evaluation,my_data
import numpy as np
import tensorflow as tf
import time,os
from ..Dataset.methods_sub import mass_spectrum_calc_2
from ..model_test import timeit

# 禁用TensorFlow的日志信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

#读取mzml文件
def load_mzml_file(path,level=1):
    #如果path是mzml文件，则用pyteomics.mzml读取
    #如果path是mzxml文件，则用pyteomics.mzxml读取
    spectra = my_data(path)
    spectra.run()
    level_spectra = spectra.get_by_level(level)
    return spectra.data,level_spectra


#用asari方法提取ROI
def asari_ROI_identify(path,para):
    #获得features
    df_features = feature_extractor(path,para)
    #为df_features添加roi_id列
    df_features['id_roi'] = df_features.index

    return df_features

def ms2ms1_linked_ROI_identify(spectra,mzml_dict,path):
    #获取path的文件后缀
    source = path.split('.')[-1]
    df = MS1_MS2_connected(spectra, mzml_dict,source)

    df.to_csv('roi0.csv')
    #将df中的每一行转为一个字典
    t = df.to_dict('records')

    rois = ROIs()
    for i in range(len(t)):
        rois.update(t[i])

    pd.DataFrame(rois.rois).to_csv('roi_no_sorted.csv')
    rois.merge()

    rois.filter(mzml_dict['min_element_roi']) # 这个参数应设置为可调参数

    df = rois.get_roi_df()
    #将mz_mean列名改为mz
    df = df.rename(columns={'mz_mean':'mz'})
    df.to_csv('roi-ckeck.csv')

    return df
@timeit
def get_calc_targets(df_rois):
    #scan最小值为left_base中的最小值，最大值为right_base中的最大值
    scan_min = df_rois['left_base'].min()
    right_base = df_rois['right_base'].max()
    #scan的范围为left_base和right_base之间的所有值
    scan_range = np.arange(scan_min,right_base+1)
    #若scan在left_base和right_base之间，则将mz添加到该scan的mz_list中
    df1 = pd.DataFrame()
    for scan in scan_range:
        mz_list = []
        roi_list = []
        for i in range(len(df_rois)):
            if df_rois.loc[i,'left_base'] <=scan<=df_rois.loc[i,'right_base']:
                mz_list.append(df_rois.loc[i,'mz'])
                roi_list.append(df_rois.loc[i,'id_roi'])
        df1 = pd.concat([df1, pd.Series({'scan':scan,'mz_list':mz_list,'roi_list':roi_list})], axis=1)
    df1 = df1.T
    #删除df1中的mz_list为[]的行
    df1 = df1[df1['mz_list'].map(lambda x: len(x)) > 0]
    df1 = df1.reset_index(drop=True)
    df1.to_csv('calc_targets.csv')
    return df1     

def find_isotopologues(df1,mzml_data,mzml_dict):
    df = pd.DataFrame()
    for i in range(len(df1)):
        scan_id = df1['scan'][i]
        rt,mz,intensity = fliter_mzml_data(mzml_data.iloc[scan_id],min_intensity=mzml_dict['min_intensity'])
        rt = rt/60
        for j in range(len(df1['mz_list'][i])):

            target_roi = df1['roi_list'][i][j]
            target_mz = df1['mz_list'][i][j]
            dict_base = {'scan':scan_id,'RT':rt,'id_roi':target_roi,'target_mz':target_mz}
            try:
                dict_mz_max = get_mz_max(mz,intensity,target_mz)#需要修正误差范围
                if dict_mz_max['mz_max1'] == dict_mz_max['mz_max2']:
                    charge = get_charge(dict_mz_max['mz_list2'],dict_mz_max['ints_list2'],dict_mz_max['intensity_max2'])
                    dict_isotoplogues = get_isotopic_peaks(dict_mz_max['mz_max2'],dict_mz_max['mz_list2'],dict_mz_max['ints_list2'],charge)
                else:
                    #charge为0，a0,a1,a2,a3,b1,b2均为0
                    charge = 0
                    dict_isotoplogues = {'mz_b3':0,'ints_b3':0,'mz_b2':0,'ints_b2':0,'mz_b1':0,'ints_b1':0,'mz_a0':0,'ints_a0':0,'mz_a1':0,'ints_a1':0,'mz_a2':0,'ints_a2':0,'mz_a3':0,'ints_a3':0}
  
                dict_new = mass_spectrum_calc_2(dict_isotoplogues)
                if charge != 0:
                    #合并dict_base,dict_mz_max,dict_isotoplogues
                    dict_all = dict_base.copy()
                    dict_all.update(dict_mz_max)
                    dict_all.update(dict_isotoplogues)
                    dict_all.update(dict_new)
                    df = pd.concat([df, pd.Series(dict_all)], axis=1)
            except:
                pass

    df = df.T
    # df.to_csv('iso_0.csv')
    return df

def add_predict(df,model_path,features_list):
    
    #加载tf模型
    clf = tf.keras.models.load_model(model_path)
    #加载特征
    querys = df[features_list].values
    querys = querys.astype('float32')
    #对特征进行预测
    res = clf.predict(querys)
    # classes = tf.math.argmax(res[0],1).numpy()
    classes_pred = np.argmax(res, axis=1)
    #将预测结果添加到df_features中
    df.loc[:, 'class_pred'] = classes_pred

    return df

def add_is_halo_isotopes(df):
    is_halo_isotopes_list = []
    #提取每行的ints_b2,ints_b1,ints_a1,ints_a2,ints_a3
    for i in range(len(df)):
        ints_b3 = df['ints_b3'].iloc[i]
        ints_b2 = df['ints_b2'].iloc[i]
        ints_b1 = df['ints_b1'].iloc[i]
        ints_a0 = df['ints_a0'].iloc[i]
        ints_a1 = df['ints_a1'].iloc[i]
        ints_a2 = df['ints_a2'].iloc[i]
        ints_a3 = df['ints_a3'].iloc[i]
        is_halo_isotopes_list.append(is_halo_isotopes(ints_b3,ints_b2,ints_b1,ints_a0,ints_a1,ints_a2,ints_a3))
    df['is_halo_isotopes'] = is_halo_isotopes_list
    
    return df

def halo_evaluation(df):
    #以此将self.df_features中的数据按照roi进行分组
    df_evaluation = pd.DataFrame()
    #循环self.df_features中的每一行，将target_roi相同的scan，class_pred分别整合到一个list中
    #获取self.df_features中的target_roi列的唯一值
    target_roi_list = df['id_roi'].unique()
    for id in target_roi_list:
        #获取target_roi为id的行
        df_ = df[df['id_roi']==id]
        #获取该行的scan列
        counter_list = df_['scan'].tolist()    
        #获取该行的class_pred列
        halo_class_list = df_['class_pred'].tolist()
        #每个scan中的目标precursor,即同位素峰中最强的一个
        MS1_precursor = df_['mz_a0'].tolist()
        #roi中每个scan中提取的一组同位素峰的mz
        isotope_mz = df_[['mz_b3','mz_b2','mz_b1','mz_a0','mz_a1','mz_a2','mz_a3']].values
        #roi中每个scan中提取的一组同位素峰的intensity
        isotope_ints = df_[['ints_b3','ints_b2','ints_b1','ints_a0','ints_a1','ints_a2','ints_a3']].values
        #roi中每个scan中提取的一组同位素峰对应的mz的平均值
        isotope_mz_mean = isotope_mz.mean(axis=0)
        #roi中每个scan中提取的一组同位素峰对应的峰强度的平均值
        isotope_ints_mean = isotope_ints.mean(axis=0)

        #将isotope_mz_mean和isotope_ints_mean合并为一个字典
        isotope_mz_mean_dict = dict(zip(['mz_b3','mz_b2','mz_b1','mz_a0','mz_a1','mz_a2','mz_a3'],isotope_mz_mean))
        isotope_ints_mean_dict = dict(zip(['ints_b3','ints_b2','ints_b1','ints_a0','ints_a1','ints_a2','ints_a3'],isotope_ints_mean))
        #将isotope_mz_mean_dict和isotope_ints_mean_dict合并为一个字典
        isotope_mz_mean_dict.update(isotope_ints_mean_dict)
        roi_new_features = mass_spectrum_calc_2(isotope_mz_mean_dict)

        #roi中每个scan中提取的一组同位素峰对应的mz的总和
        isotope_mz_total = isotope_mz.sum(axis=1)
        #roi中每个scan中提取的一组同位素峰对应的峰强度的总和
        isotope_ints_total = isotope_ints.sum(axis=1)
        

        halo_class,halo_score,halo_sub_class,halo_sub_score = roi_halo_evaluation(halo_class_list)
        #获取该行的RT
        rt = df_['RT'].tolist()
        rt_left = min(rt)
        rt_right = max(rt)
        #获取改行的所有intensity_max1之和
        precursor_ints_sum  = sum(df_['intensity_max1'].tolist())

        #添加到df_evaluation中的新行
        df_evaluation = pd.concat([df_evaluation, pd.Series({'id_roi':id,'precursor_ints_sum':precursor_ints_sum,'RT_left':rt_left,'RT_right':rt_right,'counter_list':counter_list,'halo_class_list':halo_class_list,'halo_class':halo_class,'halo_score':halo_score,'halo_sub_class':halo_sub_class,'halo_sub_score':halo_sub_score,
                                                             'MS1_precursor':MS1_precursor,'isotope_mz':isotope_mz,'isotope_ints':isotope_ints,'isotope_mz_mean':isotope_mz_mean,'isotope_ints_mean':isotope_ints_mean,'isotope_mz_total':isotope_mz_total,'isotope_ints_total':isotope_ints_total,
                                                                'roi_new_features':roi_new_features})], axis=1)
    df_evaluation = df_evaluation.T
    # df_evaluation = df_evaluation.reset_index(drop=True)
    return df_evaluation

def extract_ms2_of_rois(mzml_path,halo_evaluation_path,out_path,rois:list):
    mgf_spectra = []
    data = mzml.read(mzml_path,use_index=True,read_schema=True)
    df_halo_evaluation = pd.read_csv(halo_evaluation_path)
    for id in rois:
        #获取self.df_socres中的target_roi为id的行
        df = df_halo_evaluation[df_halo_evaluation['id_roi']==id]
        #获取该行的counter_list列
        counter_list = df['counter_list_x'].tolist()[0]
        #将counter_list由str转为list
        counter_list = eval(counter_list)
        scans = data.get_by_indexes(counter_list)
        for scan in scans:
            #获取spectra中除了'm/z array', 'intensity array'以外的所有信息存入params
            params = {k: scan[k] for k in scan if k not in ['m/z array', 'intensity array']}
            mz = scan['m/z array']
            ints= scan['intensity array']
            #组成mgf格式的字典
            mgf_dict = { 'params':params,'m/z array':mz,'intensity array':ints}
            mgf_spectra.append(mgf_dict)

    mgf.write(mgf_spectra, out_path)