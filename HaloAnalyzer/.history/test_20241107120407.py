import pandas as pd

file = r'D:\workissues\manuscript\halo_mining\HaloAnalyzer\datasets\training_validation_dataset\proprocessed\dataset\base.csv'
df = pd.read_csv(file)
df_3 = df[df['group'] == 3]
print(len(df_3))
print(len(df))

file2 = r'C:\Users\xyy\Desktop\python\HaloAnalyzer_training\022_six_dataset_openms_noClFe\2M_fake_molecules\dataset\base_show\base_original.csv'
df2 = pd.read_csv(file2)
df2_3 = df2[df2['group'] == 3]
print(len(df2_3))
print(len(df2))