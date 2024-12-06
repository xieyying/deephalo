import pandas as pd

file = r'D:\workissues\manuscript\halo_mining\HaloAnalyzer\datasets\training_validation_dataset\proprocessed\dataset\base.csv'
df = pd.read_csv(file)
df_3 = df[df['group'] == 3]
print(len(df_3))
print(len(df))