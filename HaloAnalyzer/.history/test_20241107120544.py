import pandas as pd

file = r'D:\workissues\manuscript\halo_mining\HaloAnalyzer\datasets\training_validation_dataset\proprocessed\dataset\base.csv'
df = pd.read_csv(file)
df_3 = df[df['group'] == 3]
df_3.to_csv(r'D:\workissues\manuscript\halo_mining\HaloAnalyzer\datasets\training_validation_dataset\proprocessed\dataset\base_3.csv',index=False)
print(len(df_3))
print(len(df))

