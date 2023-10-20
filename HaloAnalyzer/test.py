
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
      
# 获取以csv结尾的文件
files = os.listdir(path)
csv_files = []
for file in files:
    if file.endswith('.csv'):
        csv_files.append(file)
for f in csv_files:
    df=pd.read_csv(path+'/'+f)
    # new_a2_a1=df['new_a2_a1']
    # new_a2_a0=df['new_a2_a0']
    new_a2_ints=df['new_a2_ints']
    new_a1_ints=df['new_a1_ints']
    new_a2_a1_ints=new_a2_ints/new_a1_ints
    df['new_a2_a1_ints']=new_a2_a1_ints
    # new_a2_a1_10为new_a2_a1的10次方
    # new_a2_a1_20=new_a2_a1**20
    # new_a2_a0_20=(new_a2_a0-1)**20
    # df['new_a2_a1_20']=new_a2_a1_20
    # df['new_a2_a0_20']=new_a2_a0_20

    # new_a2_a1_15=new_a2_a1**15
    # new_a2_a0_15=(new_a2_a0-1)**15
    # df['new_a2_a1_15']=new_a2_a1_15
    # df['new_a2_a0_15']=new_a2_a0_15

    # new_a2_a1_5=new_a2_a1**5
    # new_a2_a0_5=(new_a2_a0-1)**5
    # df['new_a2_a1_5']=new_a2_a1_5
    # df['new_a2_a0_5']=new_a2_a0_5

    # new_a2_a1_10=new_a2_a1**10
    # new_a2_a0_10=(new_a2_a0-1)**10
    # df['new_a2_a1_10']=new_a2_a1_10
    # df['new_a2_a0_10']=new_a2_a0_10

    # new_a2_a1_8=new_a2_a1**8
    # new_a2_a0_8=(new_a2_a0-1)**8
    # df['new_a2_a1_8']=new_a2_a1_8
    # df['new_a2_a0_8']=new_a2_a0_8

    # new_a2_a1_3=new_a2_a1**3
    # new_a2_a0_3=(new_a2_a0-1)**3
    # df['new_a2_a1_3']=new_a2_a1_3
    # df['new_a2_a0_3']=new_a2_a0_3  

    df.to_csv(path+'/'+f,index=False)

# f=path+"\\selected_data.csv"
# df=pd.read_csv(f)
# new_a2_a1=df['new_a2_a1']
# new_a2_a0 = df['new_a2_a0']
# # new_a2_a1_10为new_a2_a1的10次方
# new_a2_a1_10 = new_a2_a1**10
# new_a2_a0_10 = (new_a2_a0-1)**10
# df['new_a2_a1_10']=new_a2_a1_10
# df.to_csv(f,index=False)

import numpy as np
# a=np.array([1,2,3,4,5])
# b = np.subtract.outer(a, a)
# b = b.reshape(len(a),len(a))
# print(b)



def adding_noise_to_intensity (Intensity,sigma_IR=0.05, sigma_IA=0.005):

    """
    The parameters and function come from the reference:

    'Meusel, M.;  Hufsky, F.;  Panter, F.;  Krug, D.;  Müller, R.; Böcker, S., 
    Predicting the Presence of Uncommon Elements in Unknown Biomolecules from Isotope Patterns.
    Anal Chem 2016, 88 (15), 7556-66.'

    parameters:

    Intesntiy: simulated intensity for a compound by molmass

    sigma_IR = 0.05 for training set, 0.04 and 0.07 for evaluation set
    sigma_IA =0.005 for training set, 0.0015 and 0.006 for evaluation set

    """
    # Generate the relative noise for intensity
    relative_noise = np.random.uniform(1,1+sigma_IR)

    # Generate the absolute noise for intensity
    absolute_noise = np.random.uniform(0, sigma_IA)

    # Calculate the simulated intensity
    I_simulated = Intensity * relative_noise + absolute_noise

    return I_simulated

def adding_noise_to_mass (mass, M=0.005):

    """
    The parameters and function come from the reference:

    'Meusel, M.;  Hufsky, F.;  Panter, F.;  Krug, D.;  Müller, R.; Böcker, S., 
    Predicting the Presence of Uncommon Elements in Unknown Biomolecules from Isotope Patterns.
    Anal Chem 2016, 88 (15), 7556-66.'

    parameters:

    mass     : simulated mass for a compound by molmass

    M=0.0015 for training set, 0.0013 and 0.0018 for evaluation set

    """
    # Generate the relative noise for mass
    mass_noise = np.random.uniform(-M, M)

    # Calculate the simulated mass
    m_simulated = mass + mass_noise
    
    return m_simulated


# import concurrent.futures

# new_a0_ints=df['new_a0_ints']
# new_a1_ints=df['new_a1_ints']
# new_a2_ints=df['new_a2_ints']
# new_a3_ints=df['new_a3_ints']
# new_a2_a0=df['new_a2_a0']
# new_a2_a1=df['new_a2_a1']

# # adding noise to intensity
# new_a0_ints_simulated = []
# new_a1_ints_simulated = []
# new_a2_ints_simulated = []
# new_a3_ints_simulated = []

# with concurrent.futures.ThreadPoolExecutor() as executor:
#     new_a0_ints_simulated = list(executor.map(adding_noise_to_intensity, new_a0_ints))
#     new_a1_ints_simulated = list(executor.map(adding_noise_to_intensity, new_a1_ints))
#     new_a2_ints_simulated = list(executor.map(adding_noise_to_intensity, new_a2_ints))
#     new_a3_ints_simulated = list(executor.map(adding_noise_to_intensity, new_a3_ints))

# # adding noise to mass
# new_a2_a1_simulated = []
# new_a2_a0_simulated = []

# with concurrent.futures.ThreadPoolExecutor() as executor:
#     new_a2_a1_simulated = list(executor.map(adding_noise_to_mass, new_a2_a1))
#     new_a2_a0_simulated = list(executor.map(adding_noise_to_mass, new_a2_a0))

# new_a2_a1_10=np.array(new_a2_a1_simulated)**10
# df['new_a2_a1_10']=new_a2_a1_10
# new_a2_a0_10=(np.array(new_a2_a0_simulated)-1)**10
# df['new_a2_a0_10']=new_a2_a0_10

# df['new_a0_ints']=new_a0_ints_simulated
# df['new_a1_ints']=new_a1_ints_simulated
# df['new_a2_ints']=new_a2_ints_simulated
# df['new_a3_ints']=new_a3_ints_simulated
# df['new_a2_a1']=new_a2_a1_simulated
# df['new_a2_a0']=new_a2_a0_simulated
# df.to_csv(path+"\\selected_data_with_noise.csv",index=False)


