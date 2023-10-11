
# file1=r"C:\Users\Administrator\Desktop\training_result\NPatlas_200_1600\config.toml"
# file2=r"C:\Users\Administrator\Desktop\training_result\NPatlas_200_1600\config_edited.toml"

# with open(file1,'r') as f1:
#     #将f1中的0 = 1 行替换为0 = 2，将f1中的1 = 1 行替换为1 = 2，将f1中的2 = 1 行替换为2 = 2，将f1中的3 = 1 行替换为3 = 1
#     lines = f1.readlines()
# line_nw = []
# for i in range(len(lines)):
#     if lines[i].startswith("0 = 1"):
#         lines[i] = "0 = 2\n"
#         line_nw.append(lines[i])
#     elif lines[i].startswith("1 = 1"):
#         lines[i] = "1 = 2\n"
#         line_nw.append(lines[i])
#     elif lines[i].startswith("2 = 1"):
#         lines[i] = "2 = 2\n"
#         line_nw.append(lines[i])
#     elif lines[i].startswith("3 = 1"):
#         lines[i] = "3 = 1\n"
#         line_nw.append(lines[i])
#     else:
#         line_nw.append(lines[i])
# with open(file2,'w') as f2:
#         for l in line_nw:
#             f2.write(l)

import pandas as pd
import os
path=r"C:\Users\xyy\Desktop\python\HaloAnalyzer_training\test1\train_dataset"
      
# # 获取以csv结尾的文件
# files = os.listdir(path)
# csv_files = []
# for file in files:
#     if file.endswith('.csv'):
#         csv_files.append(file)
# for f in csv_files:
#     df=pd.read_csv(path+'/'+f)
#     new_a2_a1=df['new_a2_a1']
#     # new_a2_a1_10为new_a2_a1的10次方
#     new_a2_a1_10=new_a2_a1**10
#     df['new_a2_a1_10']=new_a2_a1_10
#     df.to_csv(path+'/'+f,index=False)

# f=path+"\\selected_data.csv"
# df=pd.read_csv(f)
# new_a2_a1=df['new_a2_a1']
# # new_a2_a1_10为new_a2_a1的10次方
# new_a2_a1_10=new_a2_a1**10
# df['new_a2_a1_10']=new_a2_a1_10
# df.to_csv(f,index=False)
import numpy as np
a=np.array([1,2,3,4,5])
b = np.subtract.outer(a, a)
b = b.reshape(len(a),len(a))
print(b)


