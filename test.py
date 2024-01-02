from molmass import Formula
import pandas as pd
from pyteomics import mgf,mzml
import numpy as np

import os
from collections import Counter

# path = r"C:\Users\xyy\Desktop\python\HaloAnalyzer_training\020_main_2\dataset"

# files = os.listdir(path)
# # df_ = pd.DataFrame()
# for file in files:
#     if file.endswith('.csv'):
#         file_path = os.path.join(path,file)
#         df = pd.read_csv(file_path)
#         # df_ = pd.concat([df_,df],ignore_index=True)
#         df['m1_m0'] = df['m1_mz'] - df['m0_mz']
#         df.to_csv(file_path,index=False)
def MS1_MS2_connected(spectra,error = 0.3,source='mzml'):
    # vendor = mzml_dict['vendor']
    precursor_error = error
    """
    将MS1和MS2连接起来，返回一个dataframe，包含MS2的index以及与之对应的MS1的index，以及经过MS1图谱校正的MS2的precursor_mz

    质谱DDA采集模式下，信号采集顺序为MS1,而后是与此MS1相对应的一个或多个MS2(根据采集设置而定)
    根据MS2的precursor_mz，在设定误差内找到与之对应的MS1的mz，从而对MS2的precursor进行校正

    在waters采集中MS1的function=1

    """
 
    MS1_MS2_connected = {}
    MS1_MS2_connected['MS1'] = []
    MS1_MS2_connected['MS2'] = []
    MS1_MS2_connected['precursor'] = []
    MS1_MS2_connected['precursor_ints'] = []
    # MS1_MS2_connected['rt'] = []
    # MS1_MS2_connected['MS1_counter'] = []
    MS1_MS1_index = []
    MS1_rt = []
    MS1_counter_list = []
    MS1_counter=-1
    
    
    for s in spectra:
        try:
            if s['ms level'] == 1:
                mz_list = s['m/z array']
                ints_list = s['intensity array']
                MS1_counter += 1
                MS1_MS1_index.append(s['index'])
                # MS1_rt.append(s['rt'])
                MS1_counter_list.append(MS1_counter)
                
            elif s['ms level'] == 2:
                if source in ['mzML', 'mzml']:
                    precursor_mz_source = s['precursorList']['precursor'][0]['selectedIonList']['selectedIon'][0]['selected ion m/z']
                elif source in ['mzXML','mzxml']:
                    precursor_mz_source = s['precursor'][0]['precursorMz']
                #找到mz中与precursor_mz相差在0.3的所有mz_list中最高的峰
                mz_list1 = mz_list[np.abs(mz_list-precursor_mz_source)<precursor_error]
                ints_list1 = ints_list[np.abs(mz_list-precursor_mz_source)<precursor_error]
                precursor_mz = mz_list1[np.argmax(ints_list1)]
                precusor_ints = ints_list1[np.argmax(ints_list1)]

                # MS1_MS2_connected['MS1_counter'].append(MS1_counter_list[-1])
                # MS1_MS2_connected['rt'].append(MS1_rt[-1])
                MS1_MS2_connected['MS1'].append(MS1_MS1_index[-1])

                MS1_MS2_connected['MS2'].append(s['index'])
                MS1_MS2_connected['precursor'].append(precursor_mz)
                MS1_MS2_connected['precursor_ints'].append(precusor_ints)
            else:
                continue
        except:
            continue

    # transfer MS1_MS2_connected to a dataframe
    print(len(MS1_MS2_connected['MS1']))
    print(len(MS1_MS2_connected['MS2']))
    print(len(MS1_MS2_connected['precursor']))
    print(len(MS1_MS2_connected['precursor_ints']))
    MS1_MS2_connected = pd.DataFrame(MS1_MS2_connected)
    
    MS1_MS2_connected = MS1_MS2_connected[MS1_MS2_connected['precursor_ints']>=1000]

    return MS1_MS2_connected


if __name__ == '__main__':
    file = r'H:\opendatabase\CASMI_\2022\3_Data_20231206T034508Z_001\3_Data\mzML_Data\1_Priority_Challenges_1_250\neg\E_M35_negPFP_04.mzml'
    file = r'H:\opendatabase\CASMI_\2022\3_Data_20231206T034508Z_001\3_Data\mzML_Data\1_Priority_Challenges_1_250\pos\A_M4_posPFP_01.mzml'
    spectra = mzml.read(file)
    s = MS1_MS2_connected(spectra,error =0.01,source='mzml')
    # for s in spectra:
    #     if s['index'] == 111: 
    #         print(s)
    #         break
    # print(s)
    #         #