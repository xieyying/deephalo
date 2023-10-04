from molmass import Formula
import pandas as pd
import pickle
import numpy as np
import molmass
def Isotope_simulation(f,type=None,rate=None):
    fm = Formula(f)
    if type == 'hydro':
        fm_isos = get_hydroisomer_isotopes(f,rate,0.1).dataframe()
    elif type == 'dehydro':
        fm_isos = get_dehydroisomer_isotopes(f,rate,0.1).dataframe()
    elif type == 'Fe':
        fm_isos = get_iron_additive_isotopes(f).dataframe()
    else:
        fm_isos =fm.spectrum(min_intensity=1).dataframe()

    i = 0 
    b_1_int = 0
    b_2_int = 0
    b_2_mz = 0
    b_1_mz = 0

    while round(fm_isos.iloc[i]['Intensity %'],2) != 100.0:
        b_2_mz = b_1_mz
        b_2_int = b_1_int
        b_1_mz = fm_isos.iloc[i]['Relative mass']
        b_1_int = fm_isos.iloc[i]['Intensity %']
        i+=1
    
    a_0_mz = fm_isos.iloc[i]['Relative mass']
    a_0_int = fm_isos.iloc[i]['Intensity %']

    try:
        a_1_mz = fm_isos.iloc[i+1]['Relative mass']
        a_1_int = fm_isos.iloc[i+1]['Intensity %']
    except:
        a_1_mz = 0
        a_1_int = 0

    try:
        a_2_mz = fm_isos.iloc[i+2]['Relative mass']
        a_2_int = fm_isos.iloc[i+2]['Intensity %']
    except:
        a_2_mz = 0
        a_2_int = 0

    try:
        a_3_mz = fm_isos.iloc[i+3]['m/z']
        a_3_int = fm_isos.iloc[i+3]['Intensity %']
    except:
        a_3_mz = 0
        a_3_int = 0

    return b_2_mz,b_1_mz,a_0_mz,a_1_mz,a_2_mz,a_3_mz,b_2_int/100,b_1_int/100,a_0_int/100,a_1_int/100,a_2_int/100,a_3_int/100

def formula_trainable_clf(formula_dict):
    
    if formula_dict.get('H') == None or formula_dict.get('C') == None:
        return 0
    elif formula_dict.get('H') < 2 or formula_dict.get('C') < 6:
        return 0
    elif formula_dict.get('S') != None and formula_dict.get('S') > 4:
        return 0
    else:
        return 1

def formula_base_clf(formula_dict):
    
    if ('Br' in formula_dict.keys()) or ('Cl' in formula_dict.keys()):
        group_type = 0
    elif 'Fe' in formula_dict.keys():
        group_type = 1
    else:
        group_type = 2
    return group_type

def formula_sub_clf(formula_dict,optional_param=None):
    if optional_param == 'dehydro':
        group_type = 3
    elif optional_param == 'hydro':
        group_type = 3
    else:
        if ('Br' in formula_dict.keys()) and ('Cl' in formula_dict.keys()):
            group_type = 0
        elif ('Br' in formula_dict.keys()) or ('Cl' in formula_dict.keys()):
            if ('Br' in formula_dict.keys()) and formula_dict['Br']>1:
                group_type = 0
            elif ('Cl' in formula_dict.keys()) and formula_dict['Cl']>3:
                group_type = 0            
            elif ('Br' in formula_dict.keys()) and formula_dict['Br']==1 :
                group_type = 1
            elif ('Cl' in formula_dict.keys()) and formula_dict['Cl']==3:
                group_type = 1
            elif ('Cl' in formula_dict.keys()) and formula_dict['Cl']>=1:
                group_type = 2
        else:
            group_type = 4
    return group_type

def formula_element_clf(formula_dict,element):
    if  (element in formula_dict.keys()):
        return 1
    else:
        return 0

# def formula_hydro_clf(optional_param=None):
#     if optional_param == 'dehydro':
#         group_type = 0
#     elif optional_param == 'hydro':
#         group_type = 0
#     else:
#         group_type = 1
#     return group_type

def formula_groups_clf(formula,optional_param=None):
    #将formula列中的公式转为字典
    formula_dict = Formula(formula).composition().dataframe().to_dict()['Count']
    #判断是否可训练
    is_train = formula_trainable_clf(formula_dict)
    #判断clf1的group
    if optional_param == 'dehydro':
        group = 0
    elif optional_param == 'hydro':
        group = 0
    else:
        group = formula_base_clf(formula_dict)
    #判断clf2的sub_group
    sub_group = formula_sub_clf(formula_dict,optional_param)

    #判断clf3的group
    # hydro_group = formula_hydro_clf(optional_param)
    return is_train,group,sub_group#,hydro_group

def mass_spectrum_calc(b_2_mz,b_1_mz,a0_mz,a1_mz,a2_mz,a3_mz,b_2,b_1,a0,a1,a2,a3):
    #校正质谱数据   
    if a1_mz>0.1:
        a1_a0 = a1_mz - a0_mz
    else:
        a1_a0 = 0
    if a2_mz>0.1:
        a2_a0 = a2_mz - a0_mz
        a2_a1 = a2_mz - a1_mz
    else:
        a2_a0 = 0
        a2_a1 = 0
    
    if a3_mz>0.1:
        a3_a0 = a3_mz - a0_mz
        a3_a1 = a3_mz - a1_mz
        a3_a2 = a3_mz - a2_mz
    else:
        a3_a0 = 0
        a3_a1 = 0
        a3_a2 = 0
    
    if b_1_mz<0.1:
        a0_b1 = 0
    else:
        a0_b1 = a0_mz-b_1_mz

    if b_2_mz<0.1:
        b1_b2 = 0
    else:
        b1_b2 = b_1_mz-b_2_mz
    
    a0_norm = a0/2000
    return a0_norm,a1_a0,a2_a0,a2_a1,a3_a0,a3_a1,a3_a2,a0_b1,b1_b2

def mass_spectrum_calc_2(b_2_mz,b_1_mz,a0_mz,a1_mz,a2_mz,a3_mz,b_2,b_1,a0,a1,a2,a3):
    #将b_2_mz,b_1_mz,a0_mz,a1_mz,a2_mz,a3_mz中第一个不为0的值赋给new_a0_mz，其后的值赋给new_a1_mz,new_a2_mz,new_a3_mz

    if b_2_mz != 0:
        new_a0_mz = b_2_mz
        new_a1_mz = b_1_mz
        new_a2_mz = a0_mz
        new_a3_mz = a1_mz
        new_a0_ints = b_2
        new_a1_ints = b_1
        new_a2_ints = a0
        new_a3_ints = a1
    elif b_1_mz != 0:
        new_a0_mz = b_1_mz
        new_a1_mz = a0_mz
        new_a2_mz = a1_mz
        new_a3_mz = a2_mz
        new_a0_ints = b_1
        new_a1_ints = a0
        new_a2_ints = a1
        new_a3_ints = a2

    elif a0_mz != 0:
        new_a0_mz = a0_mz
        new_a1_mz = a1_mz
        new_a2_mz = a2_mz
        new_a3_mz = a3_mz
        new_a0_ints = a0
        new_a1_ints = a1
        new_a2_ints = a2
        new_a3_ints = a3
        
    new_a2_a1 = new_a2_mz - new_a1_mz
    new_a2_a0 = new_a2_mz - new_a0_mz

    return new_a0_mz,new_a1_mz,new_a2_mz,new_a3_mz,new_a0_ints,new_a1_ints,new_a2_ints,new_a3_ints,new_a2_a1,new_a2_a0


 

def isos_calc(f):
    """
    计算分子式的同位素分布(目前没用？)
    
    """
    fm = Formula(f)
    fm_isos =fm.spectrum(min_intensity=0.1).dataframe()

    m0_mz = fm_isos.iloc[0]['Relative mass']
    m0_int = fm_isos.iloc[0]['Intensity %']
    try:
        m1_int = fm_isos.iloc[1]['Intensity %']
    except:
        m1_int = 0

    try:
        m2_int = fm_isos.iloc[2]['Intensity %']
    except:
        m2_int = 0
    p1 = m0_mz
    p2 = m1_int/m0_int
    if m1_int == 0:
        p3 = 0
    else:
        p3 = m2_int/m1_int
    return p1,p2,p3

def dataset_statistics_save(data):
    #将data中formula_dict_keys列的值转换成列表
    #然后合并所有列表，然后去重
    formula_dict_keys = data['formula_dict_keys'].tolist()
    msg1 = list(set([j for i in formula_dict_keys for j in i]))
    #如果列表中有'e-'，则删除
    if 'e-' in msg1:
        msg1.remove('e-')

    
    #统计self.data中formula各个可训练的数量
    msg2 = data.groupby('is_train').count()['formula']
    #转为dataframe
    msg2 = pd.DataFrame(msg2)
    msg2 = msg2.rename(columns={'formula':'count'})
    msg2 = msg2.reset_index()
    #将is_train列的值转换成字符串
    #0:不可训练，1:可训练
    msg2['is_train'] = msg2['is_train'].apply(lambda x: '不可训练' if x==0 else '可训练')
    #用altair画图，title为可训练的数量，x轴为is_train，y轴为count
    

    #统计self.data中formula各个group的数量
    msg3 = data.groupby('group').count()['formula']
    #转为dataframe
    msg3 = pd.DataFrame(msg3)
    msg3 = msg3.rename(columns={'formula':'count'})
    msg3 = msg3.reset_index()
    #将group列的值转换成字符串
    #0:BrCl,1:Br,2:Cl,3:None
    msg3['group'] = msg3['group'].apply(lambda x: 'BrCl' if x==0 else ('Br' if x==1 else ('Cl' if x==2 else 'None')))



    msg4 = data['formula_mass']
    msg4 = pd.DataFrame(msg4)




    #用pickle将保存msg1，msg2，msg3，msg4保存到一个文件中
    with open(r'train_dataset/dataset_statistics_customized.pkl','wb') as f:
        pickle.dump([msg1,msg2,msg3,msg4],f)  

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

    f=Formula(formula+"Fe")-Formula('H3')
    return f.spectrum()

def other_requirements_trainable_clf(formula):
    #将formula列中的公式转为字典
    f_dict = Formula(formula).composition().dataframe().to_dict()['Count']
    #如果f_dict中的keys是[C,H,O,N,S]的子集，则返回1，否则返回0
    if set(f_dict.keys()).issubset(set(['C','H','O','N','S'])):
        return 1
    else:
        return 0


def get_hydroisomer_isotopes(formula,ratio,min_intensity=0.1):

    spectrum1=molmass.Formula(formula).spectrum()
    spectrum2=molmass.Formula(formula+"H2").spectrum()
    # print(spectrum1,spectrum2)

    min_fraction: float = 1e-16
    # min_intensity: float =1e-16
    #新建一个spectrum类

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
    # print(spectrum)
    # return spectrum
    # filter low intensities
    if min_intensity is not None:
        norm = 100 / max(v[1] for v in spectrum.values())
        for massnumber, value in spectrum.copy().items():
            if value[1] * norm < min_intensity:
                del spectrum[massnumber]
            else:
                spectrum[massnumber][2] = value[1]*norm
    # print(spectrum)
    return molmass.Spectrum(spectrum)

def get_dehydroisomer_isotopes(formula,ratio,min_intensity=0.1):

    spectrum2=molmass.Formula(formula).spectrum()
    dehydroisomer=Formula(formula)-Formula("H2")
    spectrum1=molmass.Formula(str(dehydroisomer)).spectrum()
    # print(spectrum1,spectrum2)

    min_fraction: float = 1e-16
    # min_intensity: float =1e-16

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
    for key2 in spectrum2:
        if key2 not in spectrum1:
            spectrum[key2]=[spectrum2[key2].mass,spectrum2[key2].fraction,1.0]


    # print(spectrum)
    # return spectrum
    # filter low intensities
    if min_intensity is not None:
        norm = 100 / max(v[1] for v in spectrum.values())
        for massnumber, value in spectrum.copy().items():
            if value[1] * norm < min_intensity:
                del spectrum[massnumber]
            else:
                spectrum[massnumber][2] = value[1]*norm
    return molmass.Spectrum(spectrum)
if __name__ == '__main__':
    
    # a = 50
    # b = adding_noise_to_mass(a)
    # print(b)
    f='C6H12O6'
    #b_2_mz,b_1_mz,a_0_mz,a_1_mz,a_2_mz,a_3_mz,b_2_int/100,b_1_int/100,a_0_int/100,a_1_int/100,a_2_int/100,a_3_int/100
    # fm_isos =Formula(f).spectrum(min_intensity=1).dataframe()
    # print(fm_isos)
    print(Formula(f).spectrum(min_intensity=1).dataframe())
    # print(get_hydroisomer_isotopes(f,0.2))
    print(get_dehydroisomer_isotopes(f,0.2))


