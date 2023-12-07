from molmass import Formula
import pandas as pd
from pyteomics import mgf,mzml

file = r'D:\python\wangmengyuan\dataset\datasets\NPAtlas_download.csv'
# file = r'D:\python\wangmengyuan\dataset\datasets\uniqueNaturalProduct_formula_from_COCONUT.csv'
df = pd.read_csv(file,low_memory=False)
formula = df ["mol_formula"].tolist()
# formula = df ['Formula'].tolist()
# 
br = []
Cl = []
for formula in formula:
    formula_dict = Formula(formula).composition().dataframe().to_dict()['Count']
    if formula_dict.get('Fe'):
        if formula_dict.get('Cl') or formula_dict.get('Br') :
            print(formula)
        
#         continue
#     br.append(formula_dict.get('Br'))
#     Cl.append(formula_dict.get('Cl'))
# # 统计不同数量的Br和Cl的数量
# from collections import Counter
# br = Counter(br)
# Cl = Counter(Cl)
# print(br)
# print(Cl)

# file = r'C:\Users\xyy\Desktop\python\HaloAnalyzer_training\0_2_0_test\dataset\base.csv'
# df = pd.read_csv(file)

# # 获取group为0-8的数据
# df = df[df['group']<9]
# df.to_csv(r'C:\Users\xyy\Desktop\python\HaloAnalyzer_training\0_2_0_test\dataset\base1.csv',index=False)

# # 将group改为7
# df['group'] = 5
# df.to_csv(r'C:\Users\xyy\Desktop\python\HaloAnalyzer_training\0_2_0_test\dataset\Fe.csv',index=False)

# file = r'H:\opendatabase\CASMI_\2022\3_Data-20231206T034508Z-003\3_Data\mzML Data\2_Bonus - Challenges 251-500\pos\A_M1_posPFP_01.mzml'

# spectra = mzml.read(file)
# s1 = []
# s2 = []
# for s in spectra:
#     if s['ms level'] == 2:
#         s2.append(s)
#     elif s['ms level'] == 1:
#         s1.append(s)
#     else:
#         pass
# print(len(s1))
# print(len(s2))
    
# a = Formula('C70H8OCl3').isotope.mass
# b = Formula('C70H8OCl3').spectrum()
# print(a)
# print(b)

