import pandas as pd
import os
import ast

file= r'C:\Users\xyy\Desktop\test\tem.csv'
df = pd.read_csv(file)
#设第一行为列名
df = pd.read_csv(file, header=0)
df['intensity'] = df['intensity'].apply(lambda x: ast.literal_eval(str(x)))

