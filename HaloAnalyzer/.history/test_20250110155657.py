import pandas as pd
import os
import ast

file= r'C:\Users\xyy\Desktop\test\tem.csv'
df = pd.read_csv(file)
#设第一行为列名
df = pd.read_csv(file, header=0)
# df['HS_inty'] = df['HS_inty'].apply(lambda x: ast.literal_eval(x) if x!= -1 else -1)
# df['HS_inty'] = df['HS_inty'].apply(lambda x: x[0] if x != -1 else -1)
# df.to_csv(file, index=False)
print(len(df))

