from molmass import Formula
import pandas as pd
from .methods_sub import *
import numpy as np

def formula_clf(formula_dict,type=None) :
    """
    Returns a class based on the formula given.

    Args:
    formula_dict: dict, formula_dict
    type: str, type of formula

    Returns:
    trainable: str, whether the formula can be trained
    group: int, group of the formula
    
    """
    #根据分子式，判断是否可训练
    if formula_dict.get('R') != None:
        trainable = 'no'
    elif formula_dict.get('H') == None or formula_dict.get('C') == None:
        trainable = 'no'
    elif formula_dict.get('H') < 3 or formula_dict.get('C') < 1:
        trainable = 'no'
    elif 'S' in formula_dict.keys() and formula_dict.get('S') >4 :
        trainable = 'no'
    else:
        trainable = 'yes'

    #根据分子式，判断类别
    if type == 'hydro':
        group = 7

    elif ('Br' in formula_dict.keys()) and ('Cl' in formula_dict.keys()):
        if 'B' in formula_dict.keys() or 'Se' in formula_dict.keys() or 'Fe' in formula_dict.keys():
            group = 10
        else:
            group = 0
    elif ('Br' in formula_dict.keys()) or ('Cl' in formula_dict.keys()):
        if 'B' in formula_dict.keys() or 'Se' in formula_dict.keys() or 'Fe' in formula_dict.keys():
            group = 10
        elif ('Br' in formula_dict.keys()) and formula_dict['Br']>1:
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
        group = 3

    elif ('B' in formula_dict.keys()):
        group = 4

    elif ('Fe' in formula_dict.keys()):
        group = 5

    else:
        group = 6

    return trainable,group

def isotope_simulation(formula,type=None,rate=None) -> dict:
    """基于分子式，模拟质谱同位素分布，返回模拟质谱数"""
    fm = Formula(formula)
    if type =='hydro':
        fm_isos = get_dehydroisomer_isotopes(formula,rate,0.0001).dataframe()
    elif type == 'Fe':
        fm_isos = get_iron_additive_isotopes(formula).dataframe()
    elif type == 'B':
        fm_isos = get_boron_additive_isotopes(formula).dataframe()
    elif type == 'Se':
        fm_isos = get_selenium_additive_isotopes(formula).dataframe()
    else:
        fm_isos =fm.spectrum(min_intensity=0.000001).dataframe()

    # print(fm_isos)
    #获取relative_mass
    relative_mass = fm_isos['Relative mass'].tolist()[:7]
    #获取intensity
    intensity = fm_isos['Intensity %'].tolist()[:7]
    
    if max(intensity) < 99.8:
        print(max(intensity),type,formula)
    #如果len(relative_mass) < 7,则补全relative_mass和intensity
    while len(relative_mass) < 7:
        relative_mass.append(0)
        intensity.append(0)

    intensity = np.array(intensity)/max(intensity)
    #转为字典
    dict_isos = {'mz_0':relative_mass[0],'mz_1':relative_mass[1],'mz_2':relative_mass[2],'mz_3':relative_mass[3],
                'mz_4':relative_mass[4],'mz_5':relative_mass[5],'mz_6':relative_mass[6],
                'p0_int':intensity[0],'p1_int':intensity[1],'p2_int':intensity[2],'p3_int':intensity[3],
                'p4_int':intensity[4],'p5_int':intensity[5],'p6_int':intensity[6]}
    return dict_isos    

def create_data(formula,type='base',rate=None,):#return_from_max_ints=True) -> pd.DataFrame:
    """基于分子式及其类型，模拟质谱数据，返回模拟质谱数"""
    if not isinstance(formula, str):
        raise ValueError(formula,'formula must be a string')
    #将formula转化为formula_dict，用于判断训练标签
    formula_dict = Formula(formula).composition().dataframe().to_dict()['Count']
    #根据formula判断训练标签
    trainable,group =formula_clf(formula_dict,type=type)

    if trainable == 'no':
        return  pd.DataFrame()
    elif type in ['Fe','B','Se','hydro']:

        if not set(formula_dict.keys()).issubset(set(['C','H','O','N'])):
            return pd.DataFrame()
    elif type == '2M':
        formula = formula + formula
        # 如果formula_dict_keys中有Cl或Br
    elif type == '2M-Cl-Br':
        if 'Cl' in formula_dict.keys():
            formula = formula -Formula('Cl')
        elif 'Br' in formula_dict.keys():
            formula = formula -Formula('Br')
        
    #模拟质谱数据
    dict_isos = isotope_simulation(formula,type,rate,)#return_from_max_ints)

    if type == 'Se':
        group = 3
    elif type == 'B':
        group = 4
    elif type == 'Fe':
        group = 5
    
    dict_base = {'formula':formula,'group':group}
    # dict_isos_calc=mass_spectrum_calc(dict_isos)
    dict_isos_calc = mass_spectrum_calc(dict_isos,1)

    #合并dict_base,dict_isos_calc和dict_isos_calc_new
    dict_all = dict_base.copy()
    dict_all.update(dict_isos)
    dict_all.update(dict_isos_calc)
    df = pd.DataFrame([dict_all])

    return df

    
if __name__ == '__main__':
    df = create_data('C30H15Br7O9S','base')
    print(df)

