from molmass import Formula
import pandas as pd
from .methods_sub import get_iron_additive_isotopes,get_hydroisomer_isotopes,get_dehydroisomer_isotopes,adding_noise_to_mass,adding_noise_to_intensity,mass_spectrum_calc,mass_spectrum_calc_2

def formula_clf(formula_dict,type=None):
    """
    Returns a classifier based on the formula given.
    """

    #根据分子式，判断是否可训练
    if formula_dict.get('H') == None or formula_dict.get('C') == None:
        trainable = 'no'
    elif formula_dict.get('H') < 2 or formula_dict.get('C') < 6:
        trainable = 'no'
    elif formula_dict.get('S') != None and formula_dict.get('S') > 4:
        trainable = 'no'
    else:
        trainable = 'yes'

    #根据分子式，判断其基本类别
    if ('Br' in formula_dict.keys()) or ('Cl' in formula_dict.keys()):
        base_group = 0
    elif 'Fe' in formula_dict.keys():
        base_group = 1
    else:
        base_group = 2

    #根据分子式，判断其具体类别
    if type == 'dehydro':
        sub_group = 0
    elif type == 'hydro':
        sub_group = 2
    elif type == 'hydro2':
        sub_group = 1
    elif type == 'hydro3':
        sub_group = 0       
    elif ('Br' in formula_dict.keys()) and ('Cl' in formula_dict.keys()):
        sub_group = 0
    elif ('Br' in formula_dict.keys()) or ('Cl' in formula_dict.keys()):
        if ('Br' in formula_dict.keys()) and formula_dict['Br']>1:
            sub_group = 0
        elif ('Cl' in formula_dict.keys()) and formula_dict['Cl']>3:
            sub_group = 0            
        elif ('Br' in formula_dict.keys()) and formula_dict['Br']==1 :
            sub_group = 1
        elif ('Cl' in formula_dict.keys()) and formula_dict['Cl']==3:
            sub_group = 1
        elif ('Cl' in formula_dict.keys()) and formula_dict['Cl']>=1:
            sub_group = 2
    elif 'Fe' in formula_dict.keys():
        sub_group = 4 
    else:
        sub_group = 3

    #根据分子式，判断其hydro_group
    if type == 'dehydro':
        hydro_group = 0
    elif type == 'hydro':
        hydro_group = 4
    elif type == 'hydro2':
        hydro_group = 5
    elif type == 'hydro3':
        hydro_group = 6

    elif ('Br' in formula_dict.keys()) and ('Cl' in formula_dict.keys()):
        hydro_group = 0
    elif ('Br' in formula_dict.keys()) or ('Cl' in formula_dict.keys()):
        if ('Br' in formula_dict.keys()) and formula_dict['Br']>1:
            hydro_group = 0
        elif ('Cl' in formula_dict.keys()) and formula_dict['Cl']>3:
            hydro_group = 0            
        elif ('Br' in formula_dict.keys()) and formula_dict['Br']==1 :
            hydro_group = 1
        elif ('Cl' in formula_dict.keys()) and formula_dict['Cl']==3:
            hydro_group = 1
        elif ('Cl' in formula_dict.keys()) and formula_dict['Cl']>=1:
            hydro_group = 2
    elif 'Fe' in formula_dict.keys():
        hydro_group = 7
    else:
        hydro_group = 3

    return trainable,base_group,sub_group,hydro_group

def Isotope_simulation(formula,type=None,rate=None):
    
    fm = Formula(formula)
    if type in ['hydro','hydro2','hydro3']:
        fm_isos = get_hydroisomer_isotopes(formula,rate,0.0001).dataframe()
    elif type == 'dehydro':
        fm_isos = get_dehydroisomer_isotopes(formula,rate,0.0001).dataframe()
    elif type == 'Fe':
        fm_isos = get_iron_additive_isotopes(formula).dataframe()
    else:
        fm_isos =fm.spectrum(min_intensity=0.0001).dataframe()

    i = 0 
    b_1_int = 0
    b_2_int = 0
    b_2_mz = 0
    b_1_mz = 0

    while round(fm_isos.iloc[i]['Intensity %'],7) != 100.0:
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

def create_data(formula,datalist,type='base',rate=None):
    if not isinstance(formula, str):
        raise ValueError(formula,'formula must be a string')
    #将formula转化为formula_dict
    formula_dict = Formula(formula).composition().dataframe().to_dict()['Count']
    #根据formula判断训练标签
    trainable,base_group,sub_group,hydro_group =formula_clf(formula_dict,type=type)

    if trainable == 'no':
        return  pd.DataFrame(columns=datalist)
    elif type in ['Fe','hydro','hydro2','hydro3','dehydro']:

        if not set(formula_dict.keys()).issubset(set(['C','H','O','N','S'])):
            return pd.DataFrame(columns=datalist)

    #模拟质谱数据
    b_2_mz,b_1_mz,a0_mz,a1_mz,a2_mz,a3_mz,b_2,b_1,a0,a1,a2,a3 = Isotope_simulation(formula,type,rate)

    if type == 'noise':
        #为质谱数据添加噪音
        #mz
        b_2_mz, b_1_mz, a0_mz, a1_mz, a2_mz, a3_mz = map(adding_noise_to_mass, [b_2_mz, b_1_mz, a0_mz, a1_mz, a2_mz, a3_mz])
        #intensity
        b_2, b_1, a0, a1, a2, a3 = map(adding_noise_to_intensity, [b_2, b_1, a0, a1, a2, a3])  
        
    elif type == 'Fe':
        base_group = 1
        sub_group = 4
        hydro_group = 7
    
    
    a0_norm,a1_a0,a2_a0,a2_a1,a3_a0,a3_a1,a3_a2,a0_b1,b1_b2=mass_spectrum_calc(b_2_mz,b_1_mz,a0_mz,a1_mz,a2_mz,a3_mz,b_2,b_1,a0,a1,a2,a3)
    new_a0_mz,new_a1_mz,new_a2_mz,new_a3_mz,new_a0_ints,new_a1_ints,new_a2_ints,new_a3_ints,new_a2_a1,new_a2_a0 = mass_spectrum_calc_2(b_2_mz,b_1_mz,a0_mz,a1_mz,a2_mz,a3_mz,b_2,b_1,a0,a1,a2,a3)

    df = pd.DataFrame(
        [[formula, base_group, sub_group, hydro_group, b_2_mz, b_1_mz, a0_mz, a1_mz, a2_mz, a3_mz,
        b_2, b_1, a0, a1, a2, a3, a1_a0, a2_a0, a2_a1, a0_b1, b1_b2, a0_norm, a3_a0, a3_a1, a3_a2,
        new_a0_mz, new_a1_mz, new_a2_mz, new_a3_mz, new_a0_ints, new_a1_ints, new_a2_ints, new_a3_ints,
        new_a2_a1, new_a2_a0]],
        columns=datalist)
    return df
    
if __name__ == '__main__':
    formula = 'C19H37NO5'
    datalist = ['formula','base_group','sub_group','hydro_group','b_2_mz','b_1_mz','a0_mz','a1_mz','a2_mz','a3_mz',
    'b_2','b_1','a0','a1','a2','a3','a1_a0','a2_a0','a2_a1','a0_b1','b1_b2','a0_norm','a3_a0','a3_a1','a3_a2',
    'new_a0_mz','new_a1_mz','new_a2_mz','new_a3_mz','new_a0_ints','new_a1_ints','new_a2_ints','new_a3_ints',
    'new_a2_a1','new_a2_a0']
    df = create_data((formula,datalist),type='base',rate=0.1)

    print(df)

