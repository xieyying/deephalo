from molmass import Formula
import pandas as pd
from pyteomics import mgf,mzml

import os

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




# path = r"C:\Users\xyy\Desktop\python\HaloAnalyzer_training\020_main_2\dataset\base.csv"

# df = pd.read_csv(path)
# #group 0-8
# # 将group9改为group8
# df['group'] = df['group'].replace(9,6)
# df['group'] = df['group'].replace(8,6)
# df = df[df['group']<=7]
# df.to_csv(r"C:\Users\xyy\Desktop\python\HaloAnalyzer_training\020_main_2\dataset\base1.csv",index=False)




# file = r'D:\python\wangmengyuan\dataset\mzmls\\Meropenem.mzML'

# data = mzml.read(file,use_index=True,read_schema=True)
# data = pd.DataFrame(data)
# data = data[data['ms level'].isin([1, 2])].copy()
# data['rt'] = data['scanList']['scan'][0]['scan start time'] * 60
# data['precursor'] = data.get('precursorList')
# data = data[['index', 'ms level', 'm/z array', 'intensity array', 'total ion current', 'rt', 'precursor']]
# data.columns = ['scan', 'ms level', 'm/z array', 'intensity array', 'tic', 'rt', 'precursor']


import numpy as np

def set_zeros_where_column_has_zero(arr_list):
    for i in range(arr_list[0].shape[1]):
        if 0 in arr_list[0][:, i]:
            for arr in arr_list:
                arr[:, i] = 0
    return arr_list

# 测试
import numpy as np

def set_zeros_where_column_has_zero(arr1, arr2, arr3):
    for i in range(3):
        if 0 in arr1[:, i]:
            arr1[:, i] = 0
            arr2[:, i] = 0
            arr3[:, i] = 0
    return arr1, arr2, arr3

# 测试
arr1 = np.array([
    [1, 2, 3, 0],
    [4, 5, 6, 7],
    [8, 9, 0, 11],
    [12, 13, 14, 15]
])

arr2 = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
])

arr3 = np.array([
    [17, 18, 19, 20],
    [21, 22, 23, 24],
    [25, 26, 27, 28],
    [29, 30, 31, 32]
])

arr1, arr2, arr3 = set_zeros_where_column_has_zero(arr1, arr2, arr3)
print(arr1)
print(arr2)
print(arr3)