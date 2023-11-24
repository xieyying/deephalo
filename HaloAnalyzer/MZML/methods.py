from pyteomics import mzml ,mgf
import pandas as pd
from .methods_sub import feature_extractor,fliter_mzml_data,get_mz_max,get_charge,get_isotoplogues,ROIs,MS1_MS2_connected,is_halo_isotopes,roi_halo_evaluation
import numpy as np
import tensorflow as tf
import time
from ..Dataset.methods_sub import mass_spectrum_calc_2


#读取mzml文件
def load_mzml_file(path,level=1):
    if level != 'all':
        spectra = mzml.read(path,use_index=True,read_schema=True)
        level_spectra = [s for s in spectra if s.get('ms level') == level]
        return level_spectra 
    else:
        spectra = mzml.read(path,use_index=True,read_schema=True)
        return spectra
          

#获取ms1_spectra中的scan，total ion current
def get_tic(mzml_data):
    scan = [i for i in range(len(mzml_data))]
    rt = [s['scanList']['scan'][0]['scan start time']*60 for s in mzml_data]
    tic = [s['total ion current'] for s in mzml_data]
    df_tic = pd.DataFrame({'rt':rt,'tic':tic,'scan':scan,})
    return df_tic

#用asari方法提取ROI
def asari_ROI_identify(path,para):
    #获得features
    df_features = feature_extractor(path,para)
    #为df_features添加roi_id列
    df_features['id_roi'] = df_features.index

    return df_features

def ms2ms1_linked_ROI_identify(spectra,mzml_dict):
    start = time.time()
    df = MS1_MS2_connected(spectra, mzml_dict)
    connect_time = time.time()
    # df.to_csv('roi0.csv')
    #将df中的每一行转为一个字典
    t = df.to_dict('records')

    rois = ROIs()
    for i in range(len(t)):
        rois.update(t[i])
    update_time = time.time()
    pd.DataFrame(rois.rois).to_csv('roi_no_sorted.csv')
    rois.merge()
    merge_time = time.time()
    rois.filter(mzml_dict['min_element_roi']) # 这个参数应设置为可调参数
    filter_time = time.time()
    df = rois.get_roi_df()
    #将mz_mean列名改为mz
    df = df.rename(columns={'mz_mean':'mz'})
    df.to_csv('roi-ckeck.csv')
    save_time = time.time()
    print('connect_time',connect_time   - start)
    print('update_time',update_time     - connect_time)
    print('merge_time',merge_time       - update_time)
    print('filter_time',filter_time     - merge_time)
    print('save_time',save_time         - filter_time)
    print('total_time',save_time        - start)

    return df

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
    # df1.to_csv('calc_targets.csv')
    return df1     

def  find_isotopologues(df1,mzml_data,mzml_dict):
    df = pd.DataFrame()
    for i in range(len(df1)):
        scan_id = df1['scan'][i]
        rt,mz,intensity = fliter_mzml_data(mzml_data[scan_id],min_intensity=mzml_dict['min_intensity'])
        rt = rt/60
        for j in range(len(df1['mz_list'][i])):

            target_roi = df1['roi_list'][i][j]
            target_mz = df1['mz_list'][i][j]
            dict_base = {'scan':scan_id,'RT':rt,'id_roi':target_roi,'target_mz':target_mz}
            try:
                dict_mz_max = get_mz_max(mz,intensity,target_mz)#需要修正误差范围
                if dict_mz_max['mz_max1'] == dict_mz_max['mz_max2']:
                    charge = get_charge(dict_mz_max['mz_list2'],dict_mz_max['ints_list2'],dict_mz_max['intensity_max2'])
                    dict_isotoplogues = get_isotoplogues(dict_mz_max['mz_max2'],dict_mz_max['mz_list2'],dict_mz_max['ints_list2'],charge)
                else:
                    #charge为0，a0,a1,a2,a3,b1,b2均为0
                    charge = 0
                    dict_isotoplogues = {'mz_b3':0,'ints_b3':0,'mz_b2':0,'ints_b2':0,'mz_b1':0,'ints_b1':0,'mz_a0':0,'ints_a0':0,'mz_a1':0,'ints_a1':0,'mz_a2':0,'ints_a2':0,'mz_a3':0,'ints_a3':0}
                #应该将函数的返回值修改为字典，可以避免重复修改此处代码    
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
    df['class_pred'] = classes_pred

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
        halo_class,halo_score,halo_sub_class,halo_sub_score = roi_halo_evaluation(halo_class_list)
        #获取该行的RT
        rt = df_['RT'].tolist()
        rt_left = min(rt)
        rt_right = max(rt)
        #获取改行的所有intensity_max1之和
        precursor_ints_sum  = sum(df_['intensity_max1'].tolist())

        #添加到df_evaluation中的新行
        df_evaluation = pd.concat([df_evaluation, pd.Series({'id_roi':id,'precursor_ints_sum':precursor_ints_sum,'RT_left':rt_left,'RT_right':rt_right,'counter_list':counter_list,'halo_class_list':halo_class_list,'halo_class':halo_class,'halo_score':halo_score,'halo_sub_class':halo_sub_class,'halo_sub_score':halo_sub_score})], axis=1)
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