from pyteomics import mzml ,mgf,mzxml
import pandas as pd
from .methods_sub import feature_extractor,fliter_mzml_data,get_mz_max,get_charge,get_isotopic_peaks,ROIs,MS1_MS2_connected,is_isotopes,roi_scan_based_halo_evaluation,my_data,filter_isotopic_peaks
import numpy as np
import tensorflow as tf
import time,os
from ..Dataset.methods_sub import mass_spectrum_calc
from ..model_test import timeit
from multiprocessing import Pool

# 禁用TensorFlow的日志信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

#读取mzml文件
def load_mzml_file(path,mzml_dict,level=1):
    #如果path是mzxml文件，则用pyteomics.mzxml读取
    spectra = my_data(path)
    spectra.run(mzml_dict['min_intensity'])
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
    df1 = MS1_MS2_connected(spectra, mzml_dict,source)
    # df.to_csv(r'C:\Users\xyy\Desktop\ms1_ms2_connected.csv',index=False)

    # df.to_csv('roi0.csv')
    #将df中的每一行转为一个字典
    t = df1.to_dict('records')

    rois = ROIs()
    for i in range(len(t)):
        rois.update(t[i])

    # pd.DataFrame(rois.rois).to_csv('roi_no_sorted.csv')
    rois.merge()

    rois.filter(mzml_dict['min_element_roi'])

    df = rois.get_roi_df()
    #将mz_mean列名改为mz
    df = df.rename(columns={'mz_mean':'mz'})
    # df.to_csv('roi-ckeck.csv')
    return df, df1

def process_scan(scan, df_rois):
    mz_list = []
    roi_list = []
    for i in range(len(df_rois)):
        if df_rois.iloc[i]['left_base'] <= scan <= df_rois.iloc[i]['right_base']:
            mz_list.append(df_rois.iloc[i]['mz'])
            roi_list.append(df_rois.iloc[i]['id_roi'])
    return {'scan':scan,'mz_list':mz_list,'roi_list':roi_list}

def get_calc_targets(df_rois, n_jobs=4): # n_jobs应该在终端设置为可调，目前还没设置(设置在congfig中了)
    scan_min = df_rois['left_base'].min()
    right_base = df_rois['right_base'].max()
    scan_range = np.arange(scan_min,right_base+1)

    with Pool(n_jobs) as p:
        results = p.starmap(process_scan, [(scan, df_rois) for scan in scan_range])

    df1 = pd.DataFrame(results)
    df1 = df1[df1['mz_list'].map(lambda x: len(x)) > 0]
    df1 = df1.reset_index(drop=True)
    # df1.to_csv('calc_targets.csv')
    return df1

def process_row(i, df1, mzml_data, mzml_dict):
    scan_id = int(df1['scan'][i])
    rt,mz,intensity = fliter_mzml_data(mzml_data.iloc[scan_id],min_intensity=mzml_dict['min_intensity'])
    rt = rt/60
    result = []
    for j in range(len(df1['mz_list'][i])):
        target_roi = df1['roi_list'][i][j]
        target_mz = df1['mz_list'][i][j]
        dict_base = {'scan':scan_id,'RT':rt,'id_roi':target_roi,'target_mz':target_mz}
        try:
            dict_mz_max = get_mz_max(mz,intensity,target_mz)
            if dict_mz_max['mz_max1'] == dict_mz_max['mz_max2'] and dict_mz_max['intensity_max2'] >=10000: # 这个参数应设置为可调参数
                charge = get_charge(dict_mz_max['mz_list2'],dict_mz_max['ints_list2'],dict_mz_max['intensity_max2'])
                dict_isotoplogues = get_isotopic_peaks(dict_mz_max['mz_max2'],dict_mz_max['mz_list2'],dict_mz_max['ints_list2'],charge)
            else:
                charge = 0
                # dict_isotoplogues = {'mz_b3':0,'ints_b_3':0,'mz_b_2':0,'ints_b_2':0,'mz_b_1':0,'ints_b_1':0,'mz_b0':0,'ints_b0':0,'mz_b1':0,'ints_b1':0,'mz_b2':0,'ints_b2':0,'mz_b3':0,'ints_b3':0,
                #                      'ints_b_3_raw':0,'ints_b_2_raw':0,'ints_b_1_raw':0,'ints_b0_raw':0,'ints_b1_raw':0,'ints_b2_raw':0,'ints_b3_raw':0}
            if charge != 0:
                dict_all = dict_base.copy()
                dict_all.update(dict_mz_max)
                dict_all.update(dict_isotoplogues)
                result.append(dict_all)
        except:
            pass
    return result

def find_isotopologues(df1, mzml_data, mzml_dict,n_jobs=4):  # n_jobs应该在终端设置为可调，目前还没设置(设置在congfig中了)
    # delete blank rows
    df1 = df1.dropna(subset=['scan'])
    with Pool(n_jobs) as p:
        results = p.starmap(process_row, [(i, df1, mzml_data, mzml_dict) for i in range(len(df1))])
    df = pd.DataFrame([item for sublist in results for item in sublist])
    return df

def correct_isotopic_peaks_base(arr1, arr2, arr3):
    """Perform a re-calibration of isotopic peaks according to the Region of Interest (ROI). If any of the isotopic peaks b_3, b_2, or 
    b_1 within a group inside the ROI has an intensity of 0, indicating no detection, then the intensity and mass-to-charge ratio (mz) 
    at the corresponding position in all groups of isotopic peaks within that ROI will be set to 0. This implies that if any of these 
    specific isotopic peaks are not detected, all corresponding positions in the isotopic groups within that ROI are treated as not detected as well."""

    for i in range(3):
        if np.any(arr1[:, i] == 0):
            arr1[:, i] = 0
            arr2[:, i] = 0
            arr3[:, i] = 0      
    return arr1, arr2, arr3

@timeit
def correct_isotopic_peaks(df):

    '''Re-calibrate the isotopic peaks based on the ROI. If b_3, b_2, or b_1 in a group within the ROI has an intensity of 0, 
    then the intensity and mz of the same position in all groups of isotopic peaks in that ROI will be set to 0.'''

    df_cor = pd.DataFrame()
    
    # 去掉df中的列m0_mz	m1_mz	m2_mz	m3_mz	m0_ints	m1_ints	m2_ints	m3_ints	m2_m1	m2_m0	m2_m0_10	m2_m1_10	b2_b1	b2_b1_10	m1_m0	m1_m0_10
    # df = df.drop(['m0_mz','m1_mz','m2_mz','m3_mz','m0_ints','m1_ints','m2_ints','m3_ints','m2_m1','m2_m0','m2_m0_10','m2_m1_10','b2_b1','b2_b1_10','m1_m0','m1_m0_10'],axis=1)
    #循环self.df_features中的每一行，将target_roi相同的scan，class_pred分别整合到一个list中
    #获取self.df_features中的target_roi列的唯一值
    target_roi_list = df['id_roi'].unique()
    for id in target_roi_list:
        #获取target_roi为id的行
        df_ = df[df['id_roi']==id].copy()
        
        #roi中每个scan中提取的一组同位素峰的mz
        isotope_mz = df_[['mz_b_3','mz_b_2','mz_b_1','mz_b0','mz_b1','mz_b2','mz_b3']].values
        #roi中每个scan中提取的一组同位素峰的intensity
        isotope_ints = df_[['ints_b_3','ints_b_2','ints_b_1','ints_b0','ints_b1','ints_b2','ints_b3']].values
        isotope_ints_raw = df_[['ints_b_3_raw','ints_b_2_raw','ints_b_1_raw','ints_b0_raw','ints_b1_raw','ints_b2_raw','ints_b3_raw']].values
        # isotope_ints_relative = isotope_ints_raw/isotope_ints_raw[3]

        #校正
        isotope_ints,isotope_ints_raw,isotope_mz,= correct_isotopic_peaks_base(isotope_ints,isotope_ints_raw,isotope_mz)
         #roi中每个scan中提取的一组同位素峰对应的峰强度的总和

        #还原
        df_.loc[:, ['mz_b_3','mz_b_2','mz_b_1','mz_b0','mz_b1','mz_b2','mz_b3']] = isotope_mz
        df_.loc[:, ['ints_b_3','ints_b_2','ints_b_1','ints_b0','ints_b1','ints_b2','ints_b3']] = isotope_ints
        df_.loc[:, ['ints_b_3_raw','ints_b_2_raw','ints_b_1_raw','ints_b0_raw','ints_b1_raw','ints_b2_raw','ints_b3_raw']] = isotope_ints_raw
        df_cor = pd.concat([df_cor,df_],axis=0)
    
    df_cor = df_cor.reset_index(drop=True)
    #利用mass_spectrum_calc对df_cor中的数据进行校正

    df_new = df_cor.apply(lambda row: mass_spectrum_calc(row, row['charge']), axis=1)

    df_new = pd.DataFrame(df_new.tolist())
    df_cor = pd.concat([df_cor,df_new],axis=1)
    return df_cor
    
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

def add_is_isotopes(df):
    is_isotopes_list = []
    #提取每行的ints_b2,ints_b1,ints_b1,ints_b2,ints_b3
    for i in range(len(df)):
        ints_b_3 = df['ints_b_3'].iloc[i]
        ints_b_2 = df['ints_b_2'].iloc[i]
        ints_b_1 = df['ints_b_1'].iloc[i]
        ints_b0 = df['ints_b0'].iloc[i]
        ints_b1 = df['ints_b1'].iloc[i]
        ints_b2 = df['ints_b2'].iloc[i]
        ints_b3 = df['ints_b3'].iloc[i]
        is_isotopes_list.append(is_isotopes(ints_b_3,ints_b_2,ints_b_1,ints_b0,ints_b1,ints_b2,ints_b3))
    df['is_isotopes'] = is_isotopes_list
    
    return df

def halo_evaluation(df):
    """
    Evaluate the halo score of the given DataFrame. The halo score is calculated based on the intensity of the isotopic peaks. 
    """

    #以此将self.df_features中的数据按照roi进行分组
    df_evaluation = pd.DataFrame()
    df_roi_mean_for_prediction = pd.DataFrame()
    df_roi_total_for_prediction = pd.DataFrame()
    #循环self.df_features中的每一行，将target_roi相同的scan，class_pred分别整合到一个list中
    #获取self.df_features中的target_roi列的唯一值
    target_roi_list = df['id_roi'].unique()
    for id in target_roi_list:
        #获取target_roi为id的行
        df_ = df[df['id_roi']==id]
        charge = df_['charge'].tolist()
        #统计charge中出现次数最多的元素
        charge = max(charge,key=charge.count)
        #获取该行的scan列
        counter_list = df_['scan'].tolist()    
        #获取该行的class_pred列
        scan_based_halo_class_list = df_['class_pred'].tolist()
        #每个scan中的目标precursor,即同位素峰中最强的一个
        MS1_precursor = df_['mz_b0'].tolist()
        #roi中每个scan中提取的一组同位素峰的mz
        isotope_mz = df_[['mz_b_3','mz_b_2','mz_b_1','mz_b0','mz_b1','mz_b2','mz_b3']].values
        #roi中每个scan中提取的一组同位素峰的intensity
        isotope_ints = df_[['ints_b_3','ints_b_2','ints_b_1','ints_b0','ints_b1','ints_b2','ints_b3']].values
        isotope_ints_raw = df_[['ints_b_3_raw','ints_b_2_raw','ints_b_1_raw','ints_b0_raw','ints_b1_raw','ints_b2_raw','ints_b3_raw']].values
        #roi中每个scan中提取的一组同位素峰对应的mz,如果非零求其平均值
       #求isotope_mz = df_[['mz_b_3','mz_b_2','mz_b_1','mz_b0','mz_b1','mz_b2','mz_b3']].values每一列非零值的平均数，同一列中如果全为0，则平均值为0
        isotope_mz_mean = np.where(isotope_mz!=0,isotope_mz, np.nan).mean(axis=0)
        # 将nan替换为0
        isotope_mz_mean = np.where(np.isnan(isotope_mz_mean.astype(float)), 0, isotope_mz_mean)

        
        #roi中每个scan中提取的一组同位素峰对应的峰强度的平均值
        isotope_ints_mean = isotope_ints_raw.mean(axis=0)
        #roi中每个scan中提取的一组同位素峰对应的相对峰强度的平均值
        isotope_ints_mean_relative = isotope_ints_mean/isotope_ints_mean[3]

         #roi中每个scan中提取的一组同位素峰对应的峰强度的总和
        isotope_ints_total = isotope_ints_raw.sum(axis=0)
        isotope_ints_total_relative = isotope_ints_total/isotope_ints_total[3]

        #将isotope_mz_mean和isotope_ints_mean合并为一个字典
        
        isotope_ints_mean_dict = dict(zip(['ints_b_3','ints_b_2','ints_b_1','ints_b0','ints_b1','ints_b2','ints_b3'],isotope_ints_mean_relative))
        isotope_ints_total_dict = dict(zip(['ints_b_3','ints_b_2','ints_b_1','ints_b0','ints_b1','ints_b2','ints_b3'],isotope_ints_total_relative))
        isotope_mz_mean_dict = dict(zip(['mz_b_3','mz_b_2','mz_b_1','mz_b0','mz_b1','mz_b2','mz_b3'],isotope_mz_mean))
        #将isotope_mz_mean_dict和isotope_ints_mean_dict合并为一个字典
        isotope_ints_mean_dict.update(isotope_mz_mean_dict)
        roi_mean_new_features = mass_spectrum_calc(isotope_ints_mean_dict,charge)
        isotope_ints_mean_dict.update(roi_mean_new_features)
        isotope_ints_mean_dict = pd.DataFrame(isotope_ints_mean_dict,index=[0])
        isotope_ints_mean_dict['id_roi'] = id
        df_roi_mean_for_prediction = pd.concat([df_roi_mean_for_prediction,isotope_ints_mean_dict],axis=0)

        isotope_ints_total_dict.update(isotope_mz_mean_dict)
        roi_total_new_features = mass_spectrum_calc(isotope_ints_total_dict,charge)
        isotope_ints_total_dict.update(roi_total_new_features)
        isotope_ints_total_dict = pd.DataFrame(isotope_ints_total_dict,index=[0])
        isotope_ints_total_dict['id_roi'] = id
        df_roi_total_for_prediction = pd.concat([df_roi_total_for_prediction,isotope_ints_total_dict],axis=0)

        scan_based_halo_class,scan_based_halo_score,scan_based_halo_sub_class,scan_based_halo_sub_score,scan_based_halo_ratio = roi_scan_based_halo_evaluation(scan_based_halo_class_list)

        #获取该行的RT
        rt = df_['RT'].tolist()
        rt_left = min(rt)
        rt_right = max(rt)
        #获取改行的所有intensity_max1之和
        precursor_ints_sum  = sum(df_['intensity_max1'].tolist())

        #添加到df_evaluation中的新行
        df_evaluation = pd.concat([df_evaluation, pd.Series({'id_roi':id,'RTmean':(rt_left+rt_right)/2,'precursor_ints_sum':precursor_ints_sum,'RT_left':rt_left,'RT_right':rt_right,'counter_list':counter_list,'scan_based_halo_class_list':scan_based_halo_class_list,'scan_based_halo_class':scan_based_halo_class,'scan_based_halo_score':scan_based_halo_score,'scan_based_halo_sub_class':scan_based_halo_sub_class,'scan_based_halo_sub_score':scan_based_halo_sub_score,
                                                             'MS1_precursor':MS1_precursor,'charge':charge,'isotope_mz':isotope_mz,'isotope_ints':isotope_ints,'isotope_ints_raw':isotope_ints_raw,'isotope_mz_mean':isotope_mz_mean,'isotope_ints_mean':isotope_ints_mean,'isotope_ints_total':isotope_ints_total,'isotope_ints_total_relative':isotope_ints_total_relative,
                                                                'isotope_ints_mean_relative':isotope_ints_mean_relative,'roi_mean_new_features':roi_mean_new_features,'roi_total_new_features':roi_total_new_features,'scan_based_halo_ratio':scan_based_halo_ratio,'precursor_ints':df_['intensity_max1'].tolist()})], axis=1)
    df_evaluation = df_evaluation.T
    # df_evaluation = df_evaluation.reset_index(drop=True)
    return df_evaluation,df_roi_mean_for_prediction,df_roi_total_for_prediction

def merge_close_values(df, mz_tolerance=100, rt_tolerance=0.5):
    """
    Merge close values in a DataFrame based on mz and RTmean columns.

    Args:
        df (pandas.DataFrame): The input DataFrame containing mz, RTmean, and other columns.
        mz_tolerance (float, optional): The tolerance value for mz difference. Defaults to 100.
        rt_tolerance (float, optional): The tolerance value for RTmean difference. Defaults to 0.5.

    Returns:
        pandas.DataFrame: The merged DataFrame with averaged values for each group.

    """
    df = df.sort_values(by=['mz', 'RTmean'])
    df['mz_diff'] = df['mz'].diff().abs() * 1e6 / df['mz']
    df['RTmean_diff'] = df['RTmean'].diff().abs()
    group = (df['mz_diff'] > mz_tolerance) | (df['RTmean_diff'] > rt_tolerance)
    group = group.cumsum()
    df = df.groupby(['roi_mean_pred', 'roi_total_pred', group]).agg({'RTmean':'mean', 'mz':'mean', 'roi_mean_pred':'mean', 'roi_total_pred':'mean', 'scan_based_halo_score':'mean', 'scan_based_halo_ratio':'mean', 'H-score':'mean'})
    df = df.sort_values(by=['RTmean','mz'])
    return df

def blank_subtract(df, blank_df, mz_tolerance=100, rt_tolerance=2):
    """
    Subtract blank values from the given dataframe.

    Args:
        df (pandas.DataFrame): The dataframe containing the data to be subtracted.
        blank_df (pandas.DataFrame): The dataframe containing the blank values.
        mz_tolerance (float, optional): The tolerance for the difference in m/z values. Defaults to 100 ppm.
        rt_tolerance (float, optional): The tolerance for the difference in retention time values. Defaults to 2 min.

    Returns:
        pandas.DataFrame: The dataframe with blank values subtracted.
    """

    for i, row in df.iterrows():
        mz_diff = (blank_df['mz'] - row['mz']).abs() * 1e6 / row['mz']
        rt_diff = (blank_df['RTmean'] - row['RTmean']).abs()
        if any((mz_diff < mz_tolerance) & (rt_diff < rt_tolerance)):
            df = df.drop(i)
    return df

def extract_ms2_of_rois(spectra_all,df_halo_evaluation,ms2ms1_linked_df,out_path):
    mgf_spectra = []
    # data = mzml.read(mzml_path,use_index=True,read_schema=True)
    # df_halo_evaluation = pd.read_csv(halo_evaluation_path)
    for i in range(len(df_halo_evaluation) ):
   
        df = df_halo_evaluation.iloc[[i]]
        
        charge = df['charge'].iloc[0]
        #获取该行的roi_ms2_index列
        counter_list = df['roi_ms2_index'].tolist()[0]
        scans = spectra_all[spectra_all['scan'].isin(counter_list)]

        for i in range(len(scans)):
            scan = scans.iloc[i]
            index = scan['scan']
            
            #获取spectra中除了'm/z array', 'intensity array'以外的所有信息存入params
            params = {k: v for k, v in scan.items() if k not in ['tic','precursor','m/z array', 'intensity array']}
            #获取precursor
            precursor_mz = ms2ms1_linked_df[ms2ms1_linked_df['MS2']==index]['precursor'].values[0]
            precursor_intensity = ms2ms1_linked_df[ms2ms1_linked_df['MS2']==index]['precursor_ints'].values[0]
            if precursor_intensity > 1e5: # 只添加强度大于1e5的MS2谱图
                #params中添加'PEPMASS'键值对
                params['PEPMASS'] = f"{precursor_mz*charge} {precursor_intensity}"
                params['CHARGE'] = f"{charge}"
                params['RTINSECONDS'] = scan['rt']*60
                
                #获取mz和intensity
                #只保留小于等于precursor_mz的mz和intensity
                mz_mask = scan['m/z array'] <= precursor_mz
                mz = scan['m/z array'][mz_mask]
                ints = scan['intensity array'][mz_mask]
                
                #组成mgf格式的字典
                mgf_dict = { 'params':params,'m/z array':mz,'intensity array':ints}
                
                mgf_spectra.append(mgf_dict)

    mgf.write(mgf_spectra, out_path)