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

def MS1_MS2_connected(filename):

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
    # transfer MS1_MS2_connected to a dataframe
    MS1_MS2_connected = pd.DataFrame(MS1_MS2_connected)


    return MS1_MS2_connected,level1_spectra

def precursor_isotopes(mz,intensity,precursor,error=0.3):

    """
    校正并提取precursor的同位素峰,返回一个dataframe

    """
    precursor_iso =pd.DataFrame()
    mz_flited = pd.Series(mz).between(precursor-error,precursor+error)

    mz_flited = mz_flited[mz_flited].index.tolist()

    #将mz_flited中的强度最大的mz及其intensity提取出来
    try:
        mz_max = mz[mz_flited[intensity[mz_flited].argmax()]]
        intensity_max = intensity[mz_flited].max()

        mz_flited2 = pd.Series(mz).between(mz_max-2.1,mz_max+3.1)
        mz_flited2 = mz_flited2[mz_flited2].index.tolist()
        mz_max2 = mz[mz_flited2[intensity[mz_flited2].argmax()]]
        intensity_max2 = intensity[mz_flited2].max()  
             
        if mz_max2 == mz_max:
            is_iso = 'y'
            mz_charge_list = mz[mz_flited2]
            ints_charge_list = intensity[mz_flited2]/intensity_max2

            #过滤掉ints_charge_list中小于0.02的值(去除杂信号)
            mz_charge_list = mz_charge_list[ints_charge_list>0.02]
            ints_charge_list = ints_charge_list[ints_charge_list>0.02]

            #选取mz_charge_list中强度最大的前5个峰
            #若不足五个峰，则选取全部
            if len(mz_charge_list) <=1:
                pass
            elif len(mz_charge_list) >= 5:
                mz_charge_list_calc = mz_charge_list[ints_charge_list.argsort()[-5:][::-1]]
            else:
                #按照强度顺序排列
                mz_charge_list_calc = mz_charge_list[ints_charge_list.argsort()[::-1]]
            
            #计算charge
            Charge = judge_charge(mz_charge_list_calc)
     
            #获取M-2峰的mz值，ints值
            #M-2峰的mz值为mz_max-2/Charge,误差为0.02，若有多个结果取ints最大的
            #若没有M-2峰，则M-2_mz为0，M-2_ints为0
            
            mz_m2 = pd.Series(mz_charge_list).between(mz_max-2/Charge-0.02,mz_max-2/Charge+0.02)
            mz_m2 = mz_m2[mz_m2].index.tolist()
            if len(mz_m2) != 0:
                ints_m2 = ints_charge_list[mz_m2]
                mz_m2 = mz_charge_list[mz_m2[ints_charge_list[mz_m2].argmax()]]
                ints_m2 = ints_m2.max()
            else:
                mz_m2 = 0
                ints_m2 = 0
   
            #获取M-1峰的mz值，ints值

            mz_m1 = pd.Series(mz_charge_list).between(mz_max-1/Charge-0.02,mz_max-1/Charge+0.02)
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



        #将质谱信息存入precursor_iso    
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
            is_iso = is_isotopes(ints_m2,ints_m1,intensity_max,ints_p1,ints_p2,ints_p3)

            if is_iso:

                new_a0_mz,new_a1_mz,new_a2_mz,new_a3_mz,new_a0_ints,new_a1_ints,new_a2_ints,new_a3_ints,new_a2_a1,new_a2_a0 = mass_spectrum_calc_2(mz_m2,mz_m1,mz_max,mz_p1,mz_p2,mz_p3,ints_m2,ints_m1,intensity_max,ints_p1,ints_p2,ints_p3)
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

                precursor_iso = pd.DataFrame([[mz_max,intensity_max,
                                                    is_iso,mz_charge_list,ints_charge_list,
                                                    Charge,mz_m2,ints_m2,mz_m1,ints_m1,
                                                    mz_p1,ints_p1,mz_p2,ints_p2,mz_p3,ints_p3,
                                                    mz_a2_a1*Charge,mz_a1_a0*Charge,a2a1,mass_d, 
                                                    mz_a0_b1*Charge,mz_b1_b2*Charge,a0_norm,mz_a2_a0*Charge,mz_charge_list_calc,
                                                    new_a0_mz,new_a1_mz,new_a2_mz,new_a3_mz,
                                                    new_a0_ints,new_a1_ints,new_a2_ints,new_a3_ints,
                                                    new_a2_a1*Charge,new_a2_a0*Charge,new_a2_a1_10,new_a2_a0_10,new_a2_a1_5,new_a2_a0_5,new_a2_a1_8,new_a2_a0_8,
                                                    new_a2_a1_4,new_a2_a0_4,new_a2_a0_15,new_a2_a1_15,new_a2_a1_20,new_a2_a0_20,new_a2_a1_ints]],
                                                    columns=['mz','intensity',
                                                                'is_iso','mz_charge_list','ints_charge_list',
                                                                'Charge','mz_m2','b_2','mz_m1','b_1',
                                                                'mz_p1','a1','mz_p2','a2','mz_p3','a3',
                                                                'a2-a1','a1-a0','a2a1','mass_d', 
                                                                'a0-b1','b1-b2','a0_norm','a2-a0','charge_calc',
                                                                'new_a0_mz','new_a1_mz','new_a2_mz','new_a3_mz',
                                                                'new_a0_ints','new_a1_ints','new_a2_ints','new_a3_ints',
                                                                'new_a2_a1','new_a2_a0','new_a2_a1_10','new_a2_a0_10','new_a2_a1_5','new_a2_a0_5','new_a2_a1_8','new_a2_a0_8',
                                                                'new_a2_a1_4','new_a2_a0_4','new_a2_a0_15','new_a2_a1_15','new_a2_a1_20','new_a2_a0_20','new_a2_a1_ints'
                                                                ])
            
    except:
        pass
    return precursor_iso

def is_isotopes(b_2,b_1,a0,a1,a2,a3):
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

def process_spectrum(file,min_intensity,precursor_error = 0.3, gap_scans = 3,min_points = 5,precursor_min_intensity = 5000):
    MS1_MS2_con,ms1_spectra = MS1_MS2_connected(file)
    df11 = pd.DataFrame()

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
            df_p = precursor_isotopes(mz, intensity, precursor, precursor_error).assign(rt=rt, scan=scan, index=index)
    
            if df_p.empty:
                continue
            precursor_mz = df_p['new_a0_mz'].values[0]
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
                
                if (abs(ROI[j]['mzmean'].values[0] - precursor_mz) / precursor_mz) * 1e6 <= 10 and scan - ROI[j]['scan'].values[-1] <= gap_scans:
                    # Append mz to the region and update the mean m/z value
                    ROI[j] = pd.concat([ROI[j], df_p], ignore_index=True)
                    ROI[j]['mzmean'] = ROI[j]['mz'].mean()
    
                    found = True
                    break

            # If no region was found, initialise a new region
            if not found:
                df_p['mzmean'] = precursor_mz
                ROI.append(pd.DataFrame(df_p))
    
    # 过滤掉ROI中少于5个峰的region
    ROI = [i for i in ROI if len(i) >= min_points]

    # 将ROI转换为dataframe，并增加一列ROI_index
    df_roi = pd.DataFrame()
    for i in range(len(ROI)):
        ROI[i]['roi'] = i
        # 过滤掉precuror_ints小于5000的峰
        ROI[i] = ROI[i][ROI[i]['precursor_a0_ints'] > 5000]
        df_roi = pd.concat([df_roi,ROI[i]],ignore_index=True)
    # print(df_roi)

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
    file=r'D:\python\my_pick_halo_4\source_data\mzmls\Meropenem.mzML'

    # MS1_MS2_con,ms1_spectra = MS1_MS2_connected(file)
    # # b.to_csv(r"C:\Users\xyy\Desktop\cmx_11_23_M3_p9_bottle_2_nr2.csv",index=False)
    # print(MS1_MS2_con['MS1'])

    df = process_spectrum(file,100)
    # ROI = ROI(file,100)
    print(df)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")


