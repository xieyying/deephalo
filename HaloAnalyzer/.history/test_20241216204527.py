import os
import pandas as pd


deephalo_result = r'D:\workissues\manuscript\halo_mining\mining\result\54_strep_1_micromono\55_strep_deephalo.tsv'
df = pd.read_csv(deephalo_result, sep='\t')
strains = df['ATTRIBUTE_strains'].unique()
print(len(strains))