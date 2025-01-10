import pandas as pd
import os
import ast

file= r'C:\Users\xyy\Desktop\test\tem.csv'
df = pd.read_csv(file)
#设第一行为列名
df = pd.read_csv(file, header=0)
len = df[df['Halo'] != 1].shape[0]
print(len)

