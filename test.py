from molmass import Formula
import pandas as pd

# # file = r'D:\python\wangmengyuan\dataset\datasets\NPAtlas_download.csv'
# file = r'D:\python\wangmengyuan\dataset\datasets\uniqueNaturalProduct_formula_from_COCONUT.csv'
# df = pd.read_csv(file)
# # formula = df ["mol_formula"].tolist()
# formula = df ['Formula'].tolist()

# br = []
# Cl = []
# for formula in formula:
#     formula_dict = Formula(formula).composition().dataframe().to_dict()['Count']
#     if formula_dict.get('H') == None or formula_dict.get('C') == None:
#         continue
#     br.append(formula_dict.get('Br'))
#     Cl.append(formula_dict.get('Cl'))
# # 统计不同数量的Br和Cl的数量
# from collections import Counter
# br = Counter(br)
# Cl = Counter(Cl)
# print(br)
# print(Cl)

file = r'C:\Users\xyy\Desktop\python\HaloAnalyzer_training\0_2_0_test\dataset\base.csv'
df = pd.read_csv(file)

# 获取group为0-8的数据
df = df[df['group']<9]
df.to_csv(r'C:\Users\xyy\Desktop\python\HaloAnalyzer_training\0_2_0_test\dataset\base1.csv',index=False)

# # 将group改为7
# df['group'] = 5
# df.to_csv(r'C:\Users\xyy\Desktop\python\HaloAnalyzer_training\0_2_0_test\dataset\Fe.csv',index=False)