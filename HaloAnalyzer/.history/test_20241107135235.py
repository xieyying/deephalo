import pandas as pd
import os

file = r'D:\workissues\manuscript\halo_mining\HaloAnalyzer\datasets\training_validation_dataset\proprocessed\dataset\base.csv'
df = pd.read_csv(file)
df_3 = df[df['group'] == 3]
#从base.csv中取出group=3的数据
df = df[df['group'] != 3]
df_3.to_csv(r'D:\workissues\manuscript\halo_mining\HaloAnalyzer\datasets\training_validation_dataset\proprocessed\dataset\base_3.csv',index=False)
print(len(df_3))
print(len(df))

# file2= r'D:\workissues\manuscript\halo_mining\HaloAnalyzer\datasets\training_validation_dataset\origin\grouped_NPatlas_coconuts'
# files = os.listdir(file2)
# files = [file for file in files if file.endswith('.csv')]

# for file in files:
#     df = pd.read_csv(os.path.join(file2,file))
#     df_3 = df[df['group'] == 4]
#     print(file)
#     print(df_3)