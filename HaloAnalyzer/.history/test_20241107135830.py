import pandas as pd
import os
from molmass import Formula

file = r'D:\workissues\manuscript\halo_mining\HaloAnalyzer\datasets\training_validation_dataset\proprocessed\dataset\base.csv'
df = pd.read_csv(file)
#从base.csv中取出group=3的数据
df_3 = df[df['group'] == 3]
df = df[df['group'] != 3]

print(len(df_3))
df_3['formula_dic'] = df_3['formula'].apply(lambda x: Formula(x).composition().dataframe().to_dict()['Count'])
# 只保留Se个数等于1的
df_3 = df_3[df_3['formula_dic'].apply(lambda x: x.get('Se') == 1)]
#remove ‘formula_dic’列
df_3 = df_3.drop(columns=['formula_dic'])
df_3.to_csv(r'D:\workissues\manuscript\halo_mining\HaloAnalyzer\datasets\training_validation_dataset\proprocessed\dataset\base_3_.csv',index=False)
print(len(df_3))
print(len(df))
df = pd.concat([df,df_3],ignore_index=True)

# file2= r'D:\workissues\manuscript\halo_mining\HaloAnalyzer\datasets\training_validation_dataset\origin\grouped_NPatlas_coconuts'
# files = os.listdir(file2)
# files = [file for file in files if file.endswith('.csv')]

# for file in files:
#     df = pd.read_csv(os.path.join(file2,file))
#     df_3 = df[df['group'] == 4]
#     print(file)
#     print(df_3)