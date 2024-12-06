from molmass import Formula
# your code here
def formula_clf(formula_dict,type=None) :
    """
    Returns a classifier based on the formula given.
    """

    #根据分子式，判断是否可训练
    if formula_dict.get('H') == None or formula_dict.get('C') == None:
        trainable = 'no'
    elif formula_dict.get('H') < 1 or formula_dict.get('C') < 1:
        trainable = 'no'
    elif formula_dict.get('S') != None and formula_dict.get('S') > 4:
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
a = 'C10H10N2O2S'
b =(Formula(a).composition().dataframe().to_dict()['Count'])
print(b)

print(formula_clf(b))