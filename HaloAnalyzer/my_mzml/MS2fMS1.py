from pyteomics import mzxml, mzml
import numpy as np
import pandas as pd
import time
import numpy as np

import pandas as pd
# from ..my_dataset.dataset_methods import mass_spectrum_calc_2

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

def MS1_MS2_connected(filename,vendor=''):

    """
    将MS1和MS2连接起来，返回一个dataframe，包含MS2的index以及与之对应的MS1的index

    质谱DDA采集模式下，信号采集顺序为MS1,而后是与此MS1相对应的一个或多个MS2(根据采集设置而定)

    在waters采集中MS1的function=1

    """
 
    MS1_MS2_connected = {}
    MS1_MS2_connected['MS1'] = []
    MS1_MS2_connected['MS2'] = []
    MS1_MS2_connected['precursor'] = []
    MS1_scan_list = []
    level1_spectra = []

    spectra=mzml.read(filename)

    if vendor == 'waters':
        for s in spectra:
            try:
                if s['ms level'] == 1 and s['id'].split(' ')[0] == "function=1":
                    MS1_scan_list.append(s['index'])
                    level1_spectra.append(s)
                elif s['ms level'] == 2:
                    precursor_mz = s['precursorList']['precursor'][0]['selectedIonList']['selectedIon'][0]['selected ion m/z']
                    MS1_MS2_connected['MS1'].append(MS1_scan_list[-1])
                    MS1_MS2_connected['MS2'].append(s['index'])
                    MS1_MS2_connected['precursor'].append(precursor_mz)
                else:
                    continue
            except:
                continue

    else:
        
        for s in spectra:
            try:
                if s['ms level'] == 1:
                    MS1_scan_list.append(s['index'])
                    level1_spectra.append(s)
                elif s['ms level'] == 2:
                    precursor_mz = s['precursorList']['precursor'][0]['selectedIonList']['selectedIon'][0]['selected ion m/z']
                    MS1_MS2_connected['MS1'].append(MS1_scan_list[-1])
                    MS1_MS2_connected['MS2'].append(s['index'])
                    MS1_MS2_connected['precursor'].append(precursor_mz)
                else:
                    continue
            except:
                continue
    # transfer MS1_MS2_connected to a dataframe
    MS1_MS2_connected = pd.DataFrame(MS1_MS2_connected)

    return MS1_MS2_connected,level1_spectra

def precursor_isotopes(mz,intensity,precursor,precursor_error=0.3):

    """
    校正并提取precursor的同位素峰,
    返回同位素峰的m/z列表: mz_list
        同位素峰的相对强度列表: int_rel_list
        同位素峰的绝对强度列表: int_list
        precursor的电荷: Charge
    """
    mz_flited = pd.Series(mz).between(precursor-precursor_error,precursor+precursor_error)

    mz_flited = mz_flited[mz_flited].index.tolist()
    
    #将mz_flited中的强度最大的mz及其intensity提取出来
    mz_list = []
    int_list = []
    int_rel_list = []
    Charge = 0
    try:
        mz_max = mz[mz_flited[intensity[mz_flited].argmax()]]
        intensity_max = intensity[mz_flited].max()

        #精确提取[mz_max-2.1,mz_max+3.1]范围内的特定的mz及其intensity
        # 提取与mz_max+1附近的mz及其intensity
        mz_f1 = pd.Series(mz).between(mz_max+1-0.02,mz_max+1+0.02)
        mz_f1 = mz_f1[mz_f1].index.tolist()
        if mz_f1 == []:
            mz_1 = 0
            int_1 = 0
        else:
            mz_1 = mz[mz_f1[intensity[mz_f1].argmax()]]
            int_1 = intensity[mz_f1].max()
            
        # 提取与mz_max+2附近的mz及其intensity
        mz_f2 = pd.Series(mz).between(mz_max+2-0.02,mz_max+2+0.02)
        mz_f2 = mz_f2[mz_f2].index.tolist()
        if mz_f2 == []:
            mz_2 = 0
            int_2 = 0
        else:
            mz_2 = mz[mz_f2[intensity[mz_f2].argmax()]]
            int_2 = intensity[mz_f2].max()


        # 提取与mz_max+3附近的mz及其intensity
        mz_f3 = pd.Series(mz).between(mz_max+3.004-0.02,mz_max+3.004+0.02)
        mz_f3 = mz_f3[mz_f3].index.tolist()
        if mz_f3 == []:
            mz_3 = 0
            int_3 = 0
        else:
            mz_3 = mz[mz_f3[intensity[mz_f3].argmax()]]
            int_3 = intensity[mz_f3].max()    


        # 提取与mz_max-2附近的mz及其intensity
        mz_f5 = pd.Series(mz).between(mz_max-2-0.02,mz_max-2+0.02)
        mz_f5 = mz_f5[mz_f5].index.tolist()
        if mz_f5 == []:
            mz_b2 = 0
            int_b2 = 0
        else:
            mz_b2 = mz[mz_f5[intensity[mz_f5].argmax()]]
            int_b2 = intensity[mz_f5].max()

                # 提取与mz_max-1附近的mz及其intensity
        mz_f4 = pd.Series(mz).between(mz_max-1-0.02,mz_max-1+0.02)
        mz_f4 = mz_f4[mz_f4].index.tolist()
        if mz_f4 == []:
            mz_b1 = 0
            int_b1 = 0
        else:
            mz_b1 = mz[mz_f4[intensity[mz_f4].argmax()]]
            int_b1 = intensity[mz_f4].max()

        # 写入list
        mz_list = [mz_b2,mz_b1,mz_max,mz_1,mz_2,mz_3]
        int_list = [int_b2,int_b1,intensity_max,int_1,int_2,int_3]

        # 取int_list中最大的值
        mz_list = np.array(mz_list)
        int_list = np.array(int_list)
        intensity_max2 = int_list.max()
        mz_max2 = mz_list[int_list.argmax()]
 
        int_rel_list = int_list/intensity_max2

        # 过滤掉ints_rel_list中小于0.02的值(去除杂信号)
        # int_rel_list 中大于0.02的保留，小于0.02的置为0
        mz_list = [mz_list[i] if int_rel_list[i] > 0.005 else 0 for i in range(len(int_rel_list))]
        int_rel_list = [int_rel_list[i] if int_rel_list[i] > 0.005 else 0 for i in range(len(int_rel_list))]
        int_list = [int_list[i] if int_rel_list[i] > 0.005 else 0 for i in range(len(int_rel_list))]
        
        # 如果mz_b1为0，则mz_b2也置为0(同位素峰中若没有M-1峰，则M-2峰也没有)
        if mz_list[1] == 0:
            mz_list[0] = 0
            int_rel_list[0] = 0
            
        #计算charge
        Charge = judge_charge(mz_list)

        # 获取电荷校正过的mz_list,ints_list
    
        mz_list = (np.array(mz_list)*Charge).tolist()
    
        #获取电荷校正过的M-2峰的mz值，ints值
        
    except:
        pass
    
    return mz_list,int_rel_list,int_list,Charge
    

def isotopes(mz,intensity,precursor,precursor_error=0.3):

    """以最高峰为基准，获取同位素峰"""

    mz_list,int_rel_list,int_list,Charge = precursor_isotopes(mz,intensity,precursor,precursor_error)
    if mz_list != []:
        mz_max = mz_list[2]

        # 获取强度最高的峰的index
        index_max = int_rel_list.index(max(int_rel_list))
        mz_max2 = mz_list[index_max]

        if mz_max2 == mz_max:
            mz_list,int_rel_list,int_list,Charge = precursor_isotopes(mz,intensity,precursor,precursor_error)
        else:
            mz_list,int_rel_list,int_list,Charge = precursor_isotopes(mz,intensity,mz_max2,precursor_error)

    return mz_list,int_rel_list,int_list,Charge
        
# 写入dataframe
def precursor_isotope_df(mz,intensity,precursor,precursor_error=0.3):
    """将precursor_isotopes的结果写入dataframe"""
    mz_list,int_rel_list,int_list,Charge = isotopes(mz,intensity,precursor,precursor_error)
    if mz_list != []:
        mz_m2,mz_m1,mz_max,mz_p1,mz_p2,mz_p3 = mz_list
        int_m2,int_m1,int_max,int_p1,int_p2,int_p3 = int_rel_list
        # 获取强度最高的峰绝对强度
        intensity_max2 = int_list[2]
    else:
        mz_m2,mz_m1,mz_max,mz_p1,mz_p2,mz_p3 = 0,0,0,0,0,0
        int_m2,int_m1,int_max,int_p1,int_p2,int_p3 = 0,0,0,0,0,0
        intensity_max2 = 0

    #将质谱信息存入precursor_iso
    precursor_iso = pd.DataFrame()
    is_iso_2 = is_halo_isotopes(int_m2,int_m1,int_max,int_p1,int_p2,int_p3)
    if is_iso_2:    
        mz_a2_a1 = mz_p2 - mz_p1
        mz_a1_a0 = mz_p1 - mz_max
        mz_a2_a0 = mz_p2 - mz_max

        a2a1 = int_p2/int_p1

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

        new_a0_mz,new_a1_mz,new_a2_mz,new_a3_mz,new_a0_ints,new_a1_ints,\
            new_a2_ints,new_a3_ints,new_a2_a1,new_a2_a0 = \
                mass_spectrum_calc_2(mz_m2,mz_m1,mz_max,mz_p1,mz_p2,mz_p3,\
                                        int_m2,int_m1,int_max,int_p1,int_p2,int_p3)
        new_a2_a1_10 = (new_a2_a1*Charge)**10
        new_a2_a0_10 = (new_a2_a0*Charge-1)**10

        new_a2_a1_20 = (new_a2_a1*Charge)**20
        new_a2_a0_20 = (new_a2_a0*Charge-1)**20

        new_a2_a1_15 = (new_a2_a1*Charge)**15
        new_a2_a0_15 = (new_a2_a0*Charge-1)**15

        new_a2_a1_5 = (new_a2_a1*Charge)**5
        new_a2_a0_5 = (new_a2_a0*Charge-1)**5

        new_a2_a1_4 = (new_a2_a1*Charge)**4
        new_a2_a0_4 = (new_a2_a0*Charge-1)**4

        new_a2_a1_8 = (new_a2_a1*Charge)**8
        new_a2_a0_8 = (new_a2_a0*Charge-1)**8

        new_a2_a1_ints = new_a2_ints/new_a1_ints
        
        #将new_item存入precursor_iso
        precursor_iso = pd.DataFrame([[mz_max,intensity_max2,
                                    mz_list,int_rel_list,
                                    Charge,mz_m2,int_m2,mz_m1,int_m1,
                                        mz_p1,int_p1,mz_p2,int_p2,mz_p3,int_p3,
                                        mz_a2_a1,mz_a1_a0,a2a1, 
                                        mz_a0_b1,mz_b1_b2,a0_norm,mz_a2_a0,
                                        new_a0_mz,new_a1_mz,new_a2_mz,new_a3_mz,
                                        new_a0_ints,new_a1_ints,new_a2_ints,new_a3_ints,
                                        new_a2_a1*Charge,new_a2_a0*Charge,new_a2_a1_10,new_a2_a0_10,new_a2_a1_5,new_a2_a0_5,new_a2_a1_8,new_a2_a0_8,
                                        new_a2_a1_4,new_a2_a0_4,new_a2_a0_15,new_a2_a1_15,new_a2_a1_20,new_a2_a0_20,new_a2_a1_ints]],
                                        columns=['mz','intensity',
                                                    'mz_charge_list','ints_charge_list',
                                                    'Charge','mz_m2','b_2','mz_m1','b_1',
                                                    'mz_p1','a1','mz_p2','a2','mz_p3','a3',
                                                    'a2-a1','a1-a0','a2a1', 
                                                    'a0-b1','b1-b2','a0_norm','a2-a0',
                                                    'new_a0_mz','new_a1_mz','new_a2_mz','new_a3_mz',
                                                    'new_a0_ints','new_a1_ints','new_a2_ints','new_a3_ints',
                                                    'new_a2_a1','new_a2_a0','new_a2_a1_10','new_a2_a0_10','new_a2_a1_5','new_a2_a0_5','new_a2_a1_8','new_a2_a0_8',
                                                    'new_a2_a1_4','new_a2_a0_4','new_a2_a0_15','new_a2_a1_15','new_a2_a1_20','new_a2_a0_20','new_a2_a1_ints'
                                                    ])
    else:
        pass

    return precursor_iso

def is_halo_isotopes(b_2,b_1,a0,a1,a2,a3):
    """
    根据六个数据库8亿多个分子的同位素峰强度的统计结果,判断是否为同位素峰
    判断主要依据卤化物的同位素峰强度的统计结果

    """

    if a1>0.06:
        if b_2 == 0:
            if b_1==0:
                is_isotope = 1
            else:
                if  b_1 > 0.5:
                    is_isotope = 1
                else:
                    is_isotope = 0
        else:
            if b_2>0.3:
                if b_1>0.02:
                   is_isotope = 1
                else:
                    is_isotope = 0
            else:
                is_isotope = 0
    else:
        is_isotope =0

    return is_isotope

def process_spectrum(file,min_intensity,precursor_error = 0.3, roi_precusor_error = 10, gap_scans = 3,min_points = 2,precursor_min_intensity = 10):
    MS1_MS2_con,ms1_spectra = MS1_MS2_connected(file)

    ROI = [] 
    for scan in range(len(ms1_spectra)):
        rt = ms1_spectra[scan]['scanList']['scan'][0]['scan start time']*60
        mz = ms1_spectra[scan]['m/z array']
        intensity = ms1_spectra[scan]['intensity array']
        int_min = intensity.min()
        intensity = intensity - int_min
        #只保留intensity大于min_intensity的峰
        mz = mz[intensity>min_intensity]
        intensity = intensity[intensity>min_intensity]
        index = ms1_spectra[scan]['index']

        # MS1_MS2_connected中MS1列中与index值相等的行取出
        # 取出的行中precursor列的值

        precursors = (MS1_MS2_con[MS1_MS2_con['MS1']==index]['precursor']).unique()
        

        if precursors.size == 0:
            continue
      
        for precursor in precursors:
            df_p = precursor_isotope_df(mz, intensity, precursor, precursor_error).assign(rt=rt, scan=scan, index=index)
            
            if df_p.empty:
                continue
            precursor_mz = df_p['new_a0_mz'].values[0]

            df_p['precursor'] = precursor
            
            df_p['precursor_a0_ints'] = df_p['new_a0_ints'].values[0]*df_p['intensity'].values[0]
            df_p['precursor_a1_ints'] = df_p['new_a1_ints'].values[0]*df_p['intensity'].values[0]
            df_p['precursor_a2_ints'] = df_p['new_a2_ints'].values[0]*df_p['intensity'].values[0]
            df_p['precursor_a3_ints'] = df_p['new_a3_ints'].values[0]*df_p['intensity'].values[0]

            # Check if there exists a region with mean m/z value within ppm of mz
            found = False

            if len(ROI) == 0:
                df_p['mzmean'] = precursor_mz
                ROI.append(df_p)
                continue
            for j in range(len(ROI)):
                
                if (abs(ROI[j]['mzmean'].values[0] - precursor_mz) / precursor_mz) * 1e6 <= roi_precusor_error and scan - ROI[j]['scan'].values[-1] <= gap_scans:
                    # Append mz to the region and update the mean m/z value
                    ROI[j] = pd.concat([ROI[j], df_p], ignore_index=True)
                    ROI[j]['mzmean'] = ROI[j]['new_a0_mz'].mean()
    
                    found = True
                    break

            # If no region was found, initialise a new region
            if not found:
                df_p['mzmean'] = precursor_mz
                ROI.append(df_p)

    
    # 过滤掉ROI中少于min_points个峰的region

    ROI = [i for i in ROI if len(i) >= min_points]

    # 将ROI转换为dataframe，并增加一列ROI_index
    df_roi = pd.DataFrame()
    for i in range(len(ROI)):
        ROI[i]['roi'] = i
        # 过滤掉precuror_ints小于precursor_min_intensity的峰
        ROI[i] = ROI[i][ROI[i]['precursor_a0_ints'] > precursor_min_intensity]
        df_roi = pd.concat([df_roi,ROI[i]],ignore_index=True)

    return df_roi

def mass_spectrum_calc_2(b_2_mz,b_1_mz,a0_mz,a1_mz,a2_mz,a3_mz,b_2,b_1,a0,a1,a2,a3):
    #将b_2_mz,b_1_mz,a0_mz,a1_mz,a2_mz,a3_mz中第一个不为0的值赋给new_a0_mz，其后的值赋给new_a1_mz,new_a2_mz,new_a3_mz

    if b_2_mz != 0:
        new_a0_mz = b_2_mz
        new_a1_mz = b_1_mz
        new_a2_mz = a0_mz
        new_a3_mz = a1_mz
        new_a0_ints = b_2
        new_a1_ints = b_1
        new_a2_ints = 1
        new_a3_ints = a1
    elif b_1_mz != 0:
        new_a0_mz = b_1_mz
        new_a1_mz = a0_mz
        new_a2_mz = a1_mz
        new_a3_mz = a2_mz
        new_a0_ints = b_1
        new_a1_ints = 1
        new_a2_ints = a1
        new_a3_ints = a2

    elif a0_mz != 0:
        new_a0_mz = a0_mz
        new_a1_mz = a1_mz
        new_a2_mz = a2_mz
        new_a3_mz = a3_mz
        new_a0_ints = 1
        new_a1_ints = a1
        new_a2_ints = a2
        new_a3_ints = a3
        
    new_a2_a1 = new_a2_mz - new_a1_mz
    new_a2_a0 = new_a2_mz - new_a0_mz

    return new_a0_mz,new_a1_mz,new_a2_mz,new_a3_mz,new_a0_ints,new_a1_ints,new_a2_ints,new_a3_ints,new_a2_a1,new_a2_a0



if __name__=="__main__":
    
    start_time = time.time()
    # file=r"J:\chenmingxu\Repeated_fermentation\cmx_11-23\raw.pro\mzML\test\cmx_11_23_M3_p9_bottle_2_nr1.mzML"
    # file=r'J:\wangmengyuan\dataset\mzmls\MSV000087166_Callophycus_compounds\MSV000087166_VHW11_80D_negMSMS_01.mzML'
    # file=r'J:\wangmengyuan\dataset\mzmls\MSV000087166_Callophycus_compounds\MSV000087166_VHW11_84C_negMSMS_06.mzML'
    # file = r"J:\wangmengyuan\dataset\mzmls\Strepomyces_x_80_M3_cmx_p18_D3_nr1.mzML"
    file= r'J:\wangmengyuan\dataset\mzmls\MSV000085027_B499_MSMS_1-A_2_01_17362_Bruker_multi_cl_Br.mzML'


    MS1_MS2_con,ms1_spectra = MS1_MS2_connected(file)
    # # # b.to_csv(r"C:\Users\xyy\Desktop\cmx_11_23_M3_p9_bottle_2_nr2.csv",index=False)
    # print(MS1_MS2_con)

    df = process_spectrum(file,1,precursor_error = 0.3, gap_scans =10,roi_precusor_error=30,min_points = 1,precursor_min_intensity = 10)
    # # ROI = ROI(file,100)
    print(df)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")


