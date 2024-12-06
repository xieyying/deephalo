from molmass import Formula

a = 'C10H10N2O2S'
b =(Formula(a).composition().dataframe().to_dict()['Count'])
print(b)