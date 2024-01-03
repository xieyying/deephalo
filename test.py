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


if __name__ == '__main__':
    path = r'D:\python\wangmengyuan\dataset\mzmls\mgf_from_public_database\pattern_mgf\test'
    files = os.listdir(path)
    mgf_merged = []
    for file in files:
        print(file)
        if file.endswith('.mgf'):
            file_path = os.path.join(path,file)
            print(file_path)
            mgf_file = mgf.MGF(file_path)
            for spectrum in mgf_file:
                # print(spectrum)
                mgf_merged.append(spectrum)
    print(len(mgf_merged))
            
    mgf.write(mgf_merged,r'D:\python\wangmengyuan\dataset\mzmls\mgf_from_public_database\pattern_mgf\test\merged.mgf')
                
   
    