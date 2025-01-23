import pandas as pd
import os

file= r'C:\Users\xyy\Desktop\test\tem.csv'
file2 = r'D:\workissues\manuscript\halo_mining\HaloAnalyzer\CASM_data\deephalo_results_for_CASM_data\DeepHalo_V2_results\target_compounds.csv'

df_1 = pd.read_csv(file)
df_2 = pd.read_csv(file2)
df_new = pd.DataFrame(columns = df_2.columns)

#按照df_1的第一列的值和第二列的值序号，将df_2的第一列的值和第二列的值对应的行提取出来
for idx,row in df_1.iterrows():
    file_match = df_2['File'] == row['File']
    mass_match = df_2['Precursor m/z (Da)'] == row['Precursor m/z (Da)']
    rt_match = df_2['RT [min]'] == row['RT [min]']
    mathch = file_match & mass_match & rt_match
    df_new = pd.concat([df_new,df_2[mathch]])
print(df_new)
