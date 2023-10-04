import numpy as np
from asari.peaks import *
from asari.default_parameters import PARAMETERS
import pymzml
import pandas as pd
from ..my_dataset.dataset_methods import mass_spectrum_calc_2
def judge_charge(a):
    b = []
    for i in a:
        for j in a:
            b.append(i-j)
    b = np.array(b)
    
    b = b.reshape(len(a),len(a))
    # print(b)a
    for i in range(0,len(b)):
        for j in range(0,len(b)):
            if abs(b[i][j] - 1) < 0.02:
                b[i][j] = 1
            elif abs(b[i][j] - 0.5) < 0.02:
                b[i][j] = 0.5
            elif abs(b[i][j] - 0.33) < 0.02:
                b[i][j] = 0.33
            else:
                b[i][j] = 0
    # print(b)
    c = {}
    for i in range(0,len(b)):
        for j in range(0,len(b)):
            if b[i][j] in c:
                c[b[i][j]] += 1
            else:
                c[b[i][j]] = 1
    # print(c)
    d = 0
    for i in c:
        if i != 0:
            if c[i] > d:
                d = c[i]
                e = i
    if d == 0:
        return 0
    if e == 1:
        return 1
    elif e == 0.5:
        return 2
    elif e == 0.33:
        return 3
    else:
        return 0


def feature_extractor(file_name: str,para) -> pd.DataFrame:
    """
    Extract features from an mzML file and return a DataFrame containing the feature information.

    Args:
        file_name: The path to the mzML file.

    Returns:
        A DataFrame containing feature information, including the quality-time coordinates of the features, peak area, peak height, and other information.
    """
    ms_expt = pymzml.run.Reader(file_name)

    # set parameters
    parameters = PARAMETERS
    parameters['min_prominence_threshold'] = int(para.min_prominence_threshold * parameters['min_peak_height'])

    # extract mass tracks

    list_mass_tracks1 = extract_massTracks_(ms_expt, mz_tolerance_ppm=para.mz_tolerance_ppm, min_intensity=para.asari_min_intensity, min_timepoints=para.min_timepoints, min_peak_height=para.min_peak_height)

    # reformat mass tracks to fit the input format of batch_deep_detect_elution_peaks
    list_mass_tracks = [{'id_number': i, 'mz': x[0], 'intensity': x[1]} for i, x in enumerate(list_mass_tracks1["tracks"])]

    # extract features
    features = batch_deep_detect_elution_peaks(list_mass_tracks, number_of_scans=len(list_mass_tracks1['rt_numbers']), parameters=parameters)

    df_features = pd.DataFrame(features)

    return df_features

def correct_df_charge(df):

    def most_common(lst):
        return max(set(lst), key=lst.count)

    #遍历df
    #如果Charge与对应roi的charge_corrected不同，则修改为charge_corrected
    def correct_charge(row):
        if row['Charge'] != df2.loc[row['roi']]['charge_corrected']:
            row['a2-a1'] = row['a2-a1']/row['Charge']*df2.loc[row['roi']]['charge_corrected']
            row['a1-a0'] = row['a1-a0']/row['Charge']*df2.loc[row['roi']]['charge_corrected']
            row['a0-b1'] = row['a0-b1']/row['Charge']*df2.loc[row['roi']]['charge_corrected']
            row['b1-b2'] = row['b1-b2']/row['Charge']*df2.loc[row['roi']]['charge_corrected']
            row['a2-a0'] = row['a2-a0']/row['Charge']*df2.loc[row['roi']]['charge_corrected']

            row['Charge'] = df2.loc[row['roi']]['charge_corrected']
        return row
    
    #统计相同roi的行charge都有哪些
    df2 = df.groupby(['roi']).agg({'Charge':lambda x: list(x)})

    #统计charge中多数是什么，保存到矫正的charrge中
    df2['charge_corrected'] = df2['Charge'].apply(most_common)
    df = df.apply(correct_charge,axis=1)
    return df


def process_spectrum(ms1_spectra,df,df2,min_intensity):
    for scan in range(len(ms1_spectra)):
        rt = ms1_spectra[scan]['scanList']['scan'][0]['scan start time']*60
        mz = ms1_spectra[scan]['m/z array']
        intensity = ms1_spectra[scan]['intensity array']
        #只保留intensity大于min_intensity的峰
        mz = mz[intensity>min_intensity]
        intensity = intensity[intensity>min_intensity]

        for i in range(len(df)):
            if df.loc[i,'left_base'] <=scan<=df.loc[i,'right_base']:
                mz_flited = pd.Series(mz).between(df.loc[i,'mz']-0.02,df.loc[i,'mz']+0.02)
                mz_flited = mz_flited[mz_flited].index.tolist()
                #将mz_flited中的强度最大的mz及其intensity提取出来
                try:
                    mz_max = mz[mz_flited[intensity[mz_flited].argmax()]]
                    intensity_max = intensity[mz_flited].max()

                    mz_flited2 = pd.Series(mz).between(df.loc[i,'mz']-2.1,df.loc[i,'mz']+3.1)
                    mz_flited2 = mz_flited2[mz_flited2].index.tolist()
                    mz_max2 = mz[mz_flited2[intensity[mz_flited2].argmax()]]
                    intensity_max2 = intensity[mz_flited2].max()
                    if mz_max2 == mz_max:
                        is_iso = 'y'
                        mz_charge_list = mz[mz_flited2]
                        ints_charge_list = intensity[mz_flited2]/intensity_max2
                        #选取mz_charge_list中强度最大的前5个峰
                        #若不足五个峰，则选取全部
                        if len(mz_charge_list) >= 5:
                            mz_charge_list_calc = mz_charge_list[ints_charge_list.argsort()[-5:][::-1]]
                        else:
                            #按照强度顺序排列
                            mz_charge_list_calc = mz_charge_list[ints_charge_list.argsort()[::-1]]

                        Charge = judge_charge(mz_charge_list_calc)
                        #获取M-2峰的mz值，ints值
                        #M-2峰的mz值为mz_max-2/Charge,误差为0.01，若有多个结果取ints最大的
                        #若没有M-2峰，则M-2_mz为0，M-2_ints为0
                        
                        mz_m2 = pd.Series(mz_charge_list).between(mz_max-2/Charge-0.01,mz_max-2/Charge+0.01)
                        mz_m2 = mz_m2[mz_m2].index.tolist()
                        if len(mz_m2) != 0:
                            ints_m2 = ints_charge_list[mz_m2]
                            mz_m2 = mz_charge_list[mz_m2[ints_charge_list[mz_m2].argmax()]]
                            ints_m2 = ints_m2.max()
                        else:
                            mz_m2 = 0
                            ints_m2 = 0

                        #获取M-1峰的mz值，ints值

                        mz_m1 = pd.Series(mz_charge_list).between(mz_max-1/Charge-0.01,mz_max-1/Charge+0.01)
                        mz_m1 = mz_m1[mz_m1].index.tolist()
                        if len(mz_m1) != 0:
                            ints_m1 = ints_charge_list[mz_m1]
                            mz_m1 = mz_charge_list[mz_m1[ints_charge_list[mz_m1].argmax()]]
                            ints_m1 = ints_m1.max()
                        else:
                            mz_m1 = 0
                            ints_m1 = 0
                            
                        #获取M+1峰的mz值，ints值
                        mz_p1 = pd.Series(mz_charge_list).between(mz_max+1/Charge-0.02,mz_max+1/Charge+0.02)
                        mz_p1 = mz_p1[mz_p1].index.tolist()
                        if len(mz_p1) != 0:
                            ints_p1 = ints_charge_list[mz_p1]
                            mz_p1 = mz_charge_list[mz_p1[ints_charge_list[mz_p1].argmax()]]
                            ints_p1 = ints_p1.max()
                        else:
                            mz_p1 = 0
                            ints_p1 = 0

                        # #获取M+2峰的mz值，ints值

                        mz_p2 = pd.Series(mz_charge_list).between(mz_max+2/Charge-0.02,mz_max+2/Charge+0.02)
                        mz_p2 = mz_p2[mz_p2].index.tolist()
                        if len(mz_p2) != 0:
                            ints_p2 = ints_charge_list[mz_p2]
                            mz_p2 = mz_charge_list[mz_p2[intensity[mz_p2].argmax()]]
                            ints_p2 = ints_p2.max()
                        else:
                            mz_p2 = 0
                            ints_p2 = 0

                        #获取M+3峰的mz值，ints值
                        mz_p3 = pd.Series(mz_charge_list).between(mz_max+3/Charge-0.02,mz_max+3/Charge+0.02)
                        mz_p3 = mz_p3[mz_p3].index.tolist()
                        if len(mz_p3) != 0:
                            ints_p3 = ints_charge_list[mz_p3]
                            mz_p3 = mz_charge_list[mz_p3[intensity[mz_p3].argmax()]]
                            ints_p3 = ints_p3.max()
                        else:
                            mz_p3 = 0
                            ints_p3 = 0

                    else:
                        is_iso = 'n'
                        mz_charge_list = ''
                        ints_charge_list = ''
                        Charge = ''
                        mz_m2 = 0
                        ints_m2 = 0
                        mz_m1 = 0   
                        ints_m1 = 0
                        mz_p1 = 0
                        ints_p1 = 0
                        mz_p2 = 0
                        ints_p2 = 0
                        mz_p3 = 0
                        ints_p3 = 0



                    #将质谱信息存入df2
                    if is_iso == 'y' and  mz_p1 != 0 and mz_p2 != 0:
                        mz_a2_a1 = mz_p2 - mz_p1
                        mz_a1_a0 = mz_p1 - mz_max
                        mz_a2_a0 = mz_p2 - mz_max
                        if ints_p1 == 0:
                            a2a1 = 0
                        else:
                            a2a1 = ints_p2/ints_p1
                        mass_d = mz_a2_a1*Charge-1.008665
                        if mz_m1 == 0:
                            mz_a0_b1 = 0
                            mz_b1_b2 = 0
                        else:
                            if mz_max - mz_m1 > 1.2:
                                mz_a0_b1 = 0
                            else:
                                mz_a0_b1 = mz_max - mz_m1

                            if (mz_m1 - mz_m2) > 1.2:
                                mz_b1_b2 = 0
                            else:
                                mz_b1_b2 = mz_m1 - mz_m2
                        
                        a0_norm = mz_max/2000
                        #mass_spectrum_calc2
                        new_a0_mz,new_a1_mz,new_a2_mz,new_a3_mz,new_a0_ints,new_a1_ints,new_a2_ints,new_a3_ints,new_a2_a1,new_a2_a0 = mass_spectrum_calc_2(mz_m2,mz_m1,mz_max,mz_p1,mz_p2,mz_p3,ints_m2,ints_m1,intensity_max,ints_p1,ints_p2,ints_p3)

                        #将new_item存入df2
                        df2 = pd.concat([df2,pd.DataFrame([[i,scan,rt,mz_max,intensity_max,
                                                            is_iso,mz_charge_list,ints_charge_list,
                                                            Charge,mz_m2,ints_m2,mz_m1,ints_m1,
                                                            mz_p1,ints_p1,mz_p2,ints_p2,mz_p3,ints_p3,
                                                            mz_a2_a1*Charge,mz_a1_a0*Charge,a2a1,mass_d, 
                                                            mz_a0_b1*Charge,mz_b1_b2*Charge,a0_norm,mz_a2_a0*Charge,mz_charge_list_calc,
                                                            new_a0_mz,new_a1_mz,new_a2_mz,new_a3_mz,
                                                            new_a0_ints,new_a1_ints,new_a2_ints,new_a3_ints,
                                                            new_a2_a1*Charge,new_a2_a0*Charge]],
                                                            columns=['roi','scan','rt','mz','intensity',
                                                                        'is_iso','mz_charge_list','ints_charge_list',
                                                                        'Charge','mz_m2','b_2','mz_m1','b_1',
                                                                        'mz_p1','a1','mz_p2','a2','mz_p3','a3',
                                                                        'a2-a1','a1-a0','a2a1','mass_d', 
                                                                        'a0-b1','b1-b2','a0_norm','a2-a0','charge_calc',
                                                                        'new_a0_mz','new_a1_mz','new_a2_mz','new_a3_mz',
                                                                        'new_a0_ints','new_a1_ints','new_a2_ints','new_a3_ints',
                                                                        'new_a2_a1','new_a2_a0'
                                                                        ])],ignore_index=True)
                        
                except:
                    # print(scan,i,'error')
                    pass
    return df2
if __name__ == '__main__':

    a = [724.70568848 ,725.70318604, 725.20709229, 726.20495605 ,726.70135498]
    print(judge_charge(a))
