import numpy as np
from molmass import Formula,Spectrum


def adding_noise_to_intensity (Intensity,sigma_IR=0.05, sigma_IA=0.00005):

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

def adding_noise_to_mass (mass, M=0.0015):

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

def get_iron_additive_isotopes(formula):
    """get the isotopic distribution of the formula with iron additive"""
    f=Formula(formula+"Fe")-Formula('H2')
    return f.spectrum()
def get_boron_additive_isotopes(formula):
    """get the isotopic distribution of the formula with boron additive"""
    f=Formula(formula+"B")-Formula('H3')
    return f.spectrum()
def get_selenium_additive_isotopes(formula):
    """get the isotopic distribution of the formula with selenium additive"""
    f=Formula(formula+"Se")
    return f.spectrum()

def other_requirements_trainable_clf(formula):
    """判断分子式是否满足其他要求，满足返回1，否则返回0
    要求：分子式中的元素只能是C,H,O,N,S,P,F,I的子集"""
    #将formula列中的公式转为字典
    f_dict = Formula(formula).composition().dataframe().to_dict()['Count']
    #如果f_dict中的keys是[C,H,O,N,S]的子集，则返回1，否则返回0
    if set(f_dict.keys()).issubset(set(['C','H','O','N','S','P','F','I'])):
        return 1
    else:
        return 0

def mass_spectrum_calc(dict_isos):
    # 目前没用到
    b_2_mz = dict_isos['mz_b_2']
    b_1_mz = dict_isos['mz_b_1']
    b0_mz = dict_isos['mz_b0']
    b1_mz = dict_isos['mz_b1']
    b2_mz = dict_isos['mz_b2']
    b3_mz = dict_isos['mz_b3']
    b_2 = dict_isos['ints_b_2']
    b_1 = dict_isos['ints_b_1']
    b0 = dict_isos['ints_b0']
    b1 = dict_isos['ints_b1']
    b2 = dict_isos['ints_b2']
    b3 = dict_isos['ints_b3']

    #校正质谱数据   
    if b1_mz>0.1:
        b1_b0 = b1_mz - b0_mz
    else:
        b1_b0 = 0
    if b2_mz>0.1:
        b2_b0 = b2_mz - b0_mz
        b2_b1 = b2_mz - b1_mz
    else:
        b2_b0 = 0
        b2_b1 = 0
    
    if b3_mz>0.1:
        b3_b0 = b3_mz - b0_mz
        b3_b1 = b3_mz - b1_mz
        b3_b2 = b3_mz - b2_mz
    else:
        b3_b0 = 0
        b3_b1 = 0
        b3_b2 = 0
    
    if b_1_mz<0.1:
        b0_b_1 = 0
    else:
        b0_b_1 = b0_mz-b_1_mz

    if b_2_mz<0.1:
        b_1_b_2 = 0
    else:
        b_1_b_2 = b_1_mz-b_2_mz
    
    b0_norm = b0/2000
    return {'mz_b_2':b_2_mz,'mz_b_1':b_1_mz,'mz_b0':b0_mz,'mz_b1':b1_mz,'mz_b2':b2_mz,'mz_b3':b3_mz,
            'ints_b_2':b_2,'ints_b_1':b_1,'ints_b0':b0,'ints_b1':b1,'ints_b2':b2,'ints_b3':b3,
            'b1_b0':b1_b0,'b2_b0':b2_b0,'b2_b1':b2_b1,'b0_b_1':b0_b_1,'b_1_b_2':b_1_b_2,
            'b0_norm':b0_norm,'b3_b0':b3_b0,'b3_b1':b3_b1,'b3_b2':b3_b2}

def mass_spectrum_calc_2(dict_features) -> dict:
    """校正质谱数据"""
    # 将以最高峰为a0的质谱数据转化为以mz最小的峰为m0的质谱数据
    mz_list = [dict_features['mz_b_3'],dict_features['mz_b_2'],dict_features['mz_b_1'],dict_features['mz_b0'],dict_features['mz_b1'],dict_features['mz_b2'],dict_features['mz_b3']]
    ints_list = [dict_features['ints_b_3'],dict_features['ints_b_2'],dict_features['ints_b_1'],1,dict_features['ints_b1'],dict_features['ints_b2'],dict_features['ints_b3']]
    for i in range(len(ints_list)):
        if ints_list[i] != 0:
            index = i
            break
    m0_mz,m1_mz,m2_mz,m3_mz = mz_list[index],mz_list[index+1],mz_list[index+2],mz_list[index+3]
    m0_ints,m1_ints,m2_ints,m3_ints = ints_list[index],ints_list[index+1],ints_list[index+2],ints_list[index+3]
    

    if m2_mz !=0:
        m2_m1 = m2_mz - m1_mz
        m2_m0 = m2_mz - m0_mz
    else:
        m2_m1 = 1.002
        m2_m0 = 2.002
    m2_m0_10 = (m2_m0-1)**10
    m2_m1_10 = m2_m1**10 

    if m1_mz !=0:
        m1_m0 = m1_mz - m0_mz
    else:    
        m1_m0 = 1.002
    m1_m0_10 = m1_m0**10

    b2= dict_features['mz_b2']
    b1= dict_features['mz_b1']
    if b2 !=0:
        b2_b1 = b2 - b1
    else:
        b2_b1 = 1.002
    b2_b1_10 = b2_b1**10

    #以字典的形式返回
    return {'m0_mz':m0_mz,'m1_mz':m1_mz,'m2_mz':m2_mz,'m3_mz':m3_mz,
            'm0_ints':m0_ints,'m1_ints':m1_ints,'m2_ints':m2_ints,'m3_ints':m3_ints,
            'm2_m1':m2_m1,'m2_m0':m2_m0,
            'm2_m0_10':m2_m0_10,'m2_m1_10':m2_m1_10,'b2_b1':b2_b1,'b2_b1_10':b2_b1_10,'m1_m0':m1_m0,'m1_m0_10':m1_m0_10}

def get_hydroisomer_isotopes(formula,ratio,min_intensity=0.0001) -> Spectrum:
    """get the isotopic distribution of the formula with hydrogen isotope"""
    spectrum1=Formula(formula).spectrum()
    spectrum2=Formula(formula+"H2").spectrum()

    min_fraction: float = 1e-16
    # min_intensity: float =1e-16

    #新建一个spectrum的字典
    spectrum={}
    for key1,items in sorted(spectrum1.items()):
        # print(key1)
        f=items.fraction
        m=items.mass
        k=items.massnumber
        if f<min_fraction:
            continue
        if key1 in spectrum2:
            f2=spectrum2[key1].fraction
            m2=spectrum2[key1].mass
            if f2<min_fraction:
                continue
            s_0 = (f * m + f2 * m2*ratio) / (f + f2*ratio)
            s_1 =f + f2*ratio
            spectrum[k]=[s_0, s_1,1.0]
        else:
            spectrum[k] = [m, f,1.0]

    # filter low intensities
    if min_intensity is not None:
        norm = 100 / max(v[1] for v in spectrum.values())
        for massnumber, value in spectrum.copy().items():
            if value[1] * norm < min_intensity:
                del spectrum[massnumber]
            else:
                spectrum[massnumber][2] = value[1]*norm
    # print(spectrum)
    return Spectrum(spectrum)

if __name__ =='__main__':

    import os
    import pandas as pd
    
    # #添加噪音
    # df_new =pd.DataFrame()
    
    # key_mz=['m2_m1','m1_m0','b2_b1']
    # key_ints = ['ints_b_3', 'ints_b_2','ints_b_1','ints_b0',"ints_b1",'ints_b2','ints_b3']
    # path = r'C:\Users\xyy\Desktop\python\HaloAnalyzer_training\020_main_2\dataset'
    # path2 = r'C:\Users\xyy\Desktop\python\HaloAnalyzer_training\020_main_2\dataset\noise'
    # files = os.listdir(path)
    # for f in files:
    #     if f.endswith('.csv'):
    #         f_path = os.path.join(path,f)
    #         df = pd.read_csv(f_path)
            
    #         for i in range(3):
    #             df_ =pd.DataFrame()
    #             m2_m1 = df['m2_m1']
    #             m1_m0 = df ['m1_m0']
    #             b2_b1 = df['b2_b1']

    #             ints_b_3 = df['ints_b_3']
    #             ints_b_2 = df['ints_b_2']
    #             ints_b_1 = df['ints_b_1']
    #             ints_b0 = df['ints_b0']
    #             ints_b1 = df['ints_b1']
    #             ints_b2 = df['ints_b2']
    #             ints_b3 = df['ints_b3']
                

    #             df_['m2_m1'] = adding_noise_to_mass(m2_m1, 0.003)
    #             df_['m1_m0'] = adding_noise_to_mass(m1_m0,0.003)
    #             df_['b2_b1'] = adding_noise_to_mass(b2_b1,0.003)

    #             df_['m2_m1_10'] = df['m2_m1']**10
    #             df_['m1_m0_10'] = df['m1_m0']**10
    #             df_['b2_b1'] = df['b2_b1']**10

    #             df_["ints_b_3"] = adding_noise_to_intensity(ints_b_3, 0.005,0.005)
    #             df_["ints_b_2"] = adding_noise_to_intensity(ints_b_2, 0.005,0.005)
    #             df_["ints_b_1"] = adding_noise_to_intensity(ints_b_1, 0.005,0.005)
    #             df_["ints_b0"] = adding_noise_to_intensity(ints_b0, 0.005,0.005)
    #             df_["ints_b1"] = adding_noise_to_intensity(ints_b1, 0.005,0.005)
    #             df_["ints_b2"] = adding_noise_to_intensity(ints_b2, 0.005,0.005)
    #             df_["ints_b3"] = adding_noise_to_intensity(ints_b3, 0.005,0.005)
    #             df_['group'] = df['group']


    #             df_new  = pd.concat([df_new,df_],ignore_index = True)
    # df_new.to_csv(r'C:\Users\xyy\Desktop\python\HaloAnalyzer_training\020_main_2\dataset\noise.csv')


                
    path = r'C:\Users\xyy\Desktop\python\HaloAnalyzer_training\020_main_2\dataset\noise\test'
    files = os.listdir(path)
    df_ = pd.DataFrame()
    for f in files:
        
        if f.endswith('.csv'):
            f_path = os.path.join(path,f)
            print(f_path)
            df = pd.read_csv(f_path)
            df_ = pd.concat([df_,df],ignore_index=True)
    df_.to_csv(r'C:\Users\xyy\Desktop\python\HaloAnalyzer_training\020_main_2\dataset\noise.csv',index=False)
        










    # (Intensity,sigma_IR=0.05, sigma_IA=0.00005):