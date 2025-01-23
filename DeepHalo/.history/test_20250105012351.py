import pandas as pd


fold = r'D:\workissues\manuscript\halo_mining\HaloAnalyzer\SIRIUS_comparison\real_data\DCA_SIRIUS_DeepHalo_results\deephalo'
file = 'CASMI_2016_myxo_delete_Se_test.csv'
file = fold + '\\' + file

df = pd.read_csv(file)

df_ = df[df['halo']!=df['ChloroDBPFinder']]
df_.to_csv(fold + '\\' + 'diff.csv', index=False)
print(df)

# 计算TP, FP, FN, TN
TP = df[(df['halo']==1) & (df['ChloroDBPFinder']==1)].shape[0]
FP = df[(df['halo']==0) & (df['ChloroDBPFinder']==1)].shape[0]
FN = df[(df['halo']==1) & (df['ChloroDBPFinder']==0)].shape[0]
TN = df[(df['halo']==0) & (df['ChloroDBPFinder']==0)].shape[0]
print('TP: ', TP)
print('FP: ', FP)
print('FN: ', FN)
print('TN: ', TN)

# 计算precision, recall
precision = TP / (TP + FP)
recall = TP / (TP + FN)
F1_score = 2 * precision * recall / (precision + recall)
print('precision: ', precision)
print('recall: ', recall)
print('F1_score: ', F1_score)