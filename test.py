from molmass import Formula
import pandas as pd
from pyteomics import mgf,mzml

import os

# path = r"C:\Users\xyy\Desktop\python\HaloAnalyzer_training\0_2_0_test\dataset"

# files = os.listdir(path)

# for file in files:
#     if file.endswith('.csv'):
#         file_path = os.path.join(path,file)
#         df = pd.read_csv(file_path)
#         # df['a2_a1'] = df['mz_a2']-df['mz_a1']
#         # df['a1_a0'] = df['mz_a1']-df['mz_a0']
#         df['new_a1_a0'] = df['new_a1_mz']-df['new_a0_mz']
#         df.to_csv(file_path,index=False)
path = r"C:\Users\xyy\Desktop\python\HaloAnalyzer_training\020_main_1\dataset\base.csv"

df = pd.read_csv(path)
#group 0-8
# 将group9改为group8
# df['group'] = df['group'].replace(9,8)
# df = df[df['group']<=8]
# df.to_csv(r"C:\Users\xyy\Desktop\python\HaloAnalyzer_training\020_main_1\dataset\base1.csv",index=False)

df = df[df['group']==4]

file = r'D:\python\wangmengyuan\dataset\mzmls\cmx_11_23_M3_p9_bottle_1.mzML'

data = mzml.read(file,use_index=True,read_schema=True)
data = pd.DataFrame(data)
data = data[data['ms level'].isin([1, 2])].copy()
data['rt'] = data['scanList']['scan'][0]['scan start time'] * 60
data['precursor'] = data.get('precursorList')
data = data[['index', 'ms level', 'm/z array', 'intensity array', 'total ion current', 'rt', 'precursor']]
data.columns = ['scan', 'ms level', 'm/z array', 'intensity array', 'tic', 'rt', 'precursor']
