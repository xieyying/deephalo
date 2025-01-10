import pandas as pd


fold = r'D:\workissues\manuscript\halo_mining\HaloAnalyzer\SIRIUS_comparison\real_data\DCA_SIRIUS_DeepHalo_results\deephalo'
file = 'CASMI_2016_myxo_delete_Se_test.csv'
file = fold + '\\' + file

df = pd.read_csv(file)

df = df[df['halo']!=df['ChloroDBPFinder']]
df.to_csv(fold + '\\' + 'diff.csv', index=False)
print(df)