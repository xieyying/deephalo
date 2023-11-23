from molmass import Formula
import pandas as pd
from .methods_sub import get_iron_additive_isotopes,get_boron_additive_isotopes,\
    get_selenium_additive_isotopes,get_hydroisomer_isotopes,\
    adding_noise_to_mass,adding_noise_to_intensity,mass_spectrum_calc,mass_spectrum_calc_2
    

def formula_clf(formula_dict,type=None) :
    """
    Returns a classifier based on the formula given.
    """

    #根据分子式，判断是否可训练
    if formula_dict.get('H') == None or formula_dict.get('C') == None:
        trainable = 'no'
    elif formula_dict.get('H') < 3 or formula_dict.get('C') < 2:
        trainable = 'no'
    elif formula_dict.get('S') != None and formula_dict.get('S') > 4:
        trainable = 'no'
    else:
        trainable = 'yes'

    #根据分子式，判断类别
    if type == 'hydro':
        group = 4

    elif ('Br' in formula_dict.keys()) and ('Cl' in formula_dict.keys()):
        group = 0
    elif ('Br' in formula_dict.keys()) or ('Cl' in formula_dict.keys()):
        if ('Br' in formula_dict.keys()) and formula_dict['Br']>1:
            group = 0
        elif ('Cl' in formula_dict.keys()) and formula_dict['Cl']>3:
            group = 0            
        elif ('Br' in formula_dict.keys()) and formula_dict['Br']==1 :
            group = 1
        elif ('Cl' in formula_dict.keys()) and formula_dict['Cl']==3:
            group = 1
        elif ('Cl' in formula_dict.keys()) and formula_dict['Cl']>=1:
            group = 2
    elif ('Se' in formula_dict.keys() ):
        if formula_dict['Se']<=2:
            group = 7
        else:
            group = 11
    elif ('B' in formula_dict.keys()):
        if  formula_dict['B']<=2:
            group = 6
        elif formula_dict['B']<=4:
            group = 12
        else:
            group = 13
    elif ('Fe' in formula_dict.keys()):
        if formula_dict['Fe']<=3:
            group = 5
        else:
            group = 14
    elif 'S' in formula_dict.keys():
        if formula_dict['S']<=4:
            group = 8
        elif formula_dict['S']==5:
            group = 9
        else:
            group = 10

    else:
        group = 3

    return trainable,group

def Isotope_simulation(formula,type=None,rate=None) -> dict:
    """基于分子式，模拟质谱同位素分布，返回模拟质谱数"""
    fm = Formula(formula)
    if type =='hydro':
        fm_isos = get_hydroisomer_isotopes(formula,rate,0.0001).dataframe()
    elif type == 'Fe':
        fm_isos = get_iron_additive_isotopes(formula).dataframe()
    elif type == 'B':
        fm_isos = get_boron_additive_isotopes(formula).dataframe()
    elif type == 'Se':
        fm_isos = get_selenium_additive_isotopes(formula).dataframe()
    else:
        fm_isos =fm.spectrum(min_intensity=0.0001).dataframe()

    i = 0 
    b_1_int = 0
    b_2_int = 0
    b_3_int = 0
    b_3_mz = 0
    b_2_mz = 0
    b_1_mz = 0
    

    while round(fm_isos.iloc[i]['Intensity %'],7) != 100.0:
        b_3_mz = b_2_mz
        b_2_mz = b_1_mz
        b_3_int = b_2_int
        b_2_int = b_1_int
        b_1_mz = fm_isos.iloc[i]['Relative mass']
        b_1_int = fm_isos.iloc[i]['Intensity %']
        # if b_1_int >=1:
        #     b_1_mz = fm_isos.iloc[i]['Relative mass']
        #     b_1_int = fm_isos.iloc[i]['Intensity %']
        # else:
        #     b_1_int = 0
        #     b_1_mz = 0
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
        
        a_3_mz = fm_isos.iloc[i+3]['Relative mass']
        a_3_int = fm_isos.iloc[i+3]['Intensity %']

    except:
        a_3_mz = 0
        a_3_int = 0
    try:
        a_4_mz = fm_isos.iloc[i+4]['Relative mass']
        a_4_int = fm_isos.iloc[i+4]['Intensity %']
    except:
        a_4_mz = 0
        a_4_int = 0

    return {'mz_b3':b_3_mz,'mz_b2':b_2_mz,'mz_b1':b_1_mz,'mz_a0':a_0_mz,\
            'mz_a1':a_1_mz,'mz_a2':a_2_mz,'mz_a3':a_3_mz,'mz_a4':a_4_mz,\
            'ints_b3':b_3_int/100,'ints_b2':b_2_int/100,'ints_b1':b_1_int/100,\
            'ints_a0':a_0_int/100,'ints_a1':a_1_int/100,'ints_a2':a_2_int/100,\
            'ints_a3':a_3_int/100,'ints_a4':a_4_int/100} 

def create_data(formula,type='base',rate=None) -> pd.DataFrame:
    """基于分子式及其类型，模拟质谱数据，返回模拟质谱数"""
    if not isinstance(formula, str):
        raise ValueError(formula,'formula must be a string')
    #将formula转化为formula_dict
    formula_dict = Formula(formula).composition().dataframe().to_dict()['Count']
    #根据formula判断训练标签
    trainable,group =formula_clf(formula_dict,type=type)

    if trainable == 'no':
        return  pd.DataFrame()
    elif type in ['Fe','B','Se','hydro']:

        if not set(formula_dict.keys()).issubset(set(['C','H','O','N','S','P','F','I'])):
            return pd.DataFrame()

    
    #模拟质谱数据
    dict_isos = Isotope_simulation(formula,type,rate)

    if type == 'noise':
        #为质谱数据添加噪音
        #更新dict_isos中的数据，逐个增加噪音
        for key in dict_isos.keys():
            if key in ['mz_b3','mz_b2','mz_b1','mz_a0','mz_a1','mz_a2','mz_a3','mz_a4']:
                dict_isos[key] = adding_noise_to_mass(dict_isos[key])
            else:
                dict_isos[key] = adding_noise_to_intensity(dict_isos[key])
      
    elif type == 'Fe':
        group = 5
    elif type == 'B':
        group = 6
    elif type == 'Se':
        group = 7
    
    dict_base = {'formula':formula,'group':group}
    # dict_isos_calc=mass_spectrum_calc(dict_isos)
    dict_isos_calc_new = mass_spectrum_calc_2(dict_isos)

    #合并dict_base,dict_isos_calc和dict_isos_calc_new
    dict_all = dict_base.copy()
    dict_all.update(dict_isos)
    dict_all.update(dict_isos_calc_new)
    df = pd.DataFrame([dict_all])

    return df
    
if __name__ == '__main__':
    formula = 'C19H37NO5'

    df = create_data((formula),type='base',rate=0.1)

    print(df)

