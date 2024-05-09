import pandas as pd
import ast
import numpy as np
from pyteomics import mgf

def read_halo_evalution_mgf(path):
    """read isotopes in halo evaluation csv file  to mgf file"""
    df = pd.read_csv(path)
    mgf_list = []
    for i in range(len(df)):
        rt = df.loc[i, 'RTmean']
        precursor = df.loc[i, 'mz']
        charge = df.loc[i, 'charge']
        
        isotope_mz_str = df.loc[i, 'isotope_mz']
        isotope_mz_list_str = isotope_mz_str[2:-2].split(']\n [')
        # Convert the list of strings to a list of lists of floats
        isotope_mz = [list(map(float, x.split())) for x in isotope_mz_list_str]
        
        isotope_int_str = df.loc[i, 'isotope_ints']
        isotope_int_list_str = isotope_int_str[2:-2].split(']\n [')
        # Convert the list of strings to a list of lists of floats
        isotope_int = [list(map(float, x.split())) for x in isotope_int_list_str]
        
        for t  in range(len(isotope_mz)):
            params = {'rt': rt, 'precursor': precursor, 'charge': charge, 'formula': 'CHON', 'compound_name': 'XXXX'}
            mgf_dict = { 'params':params,'m/z array':np.array(isotope_mz[t]),'intensity array':np.array(isotope_int[t])}
            mgf_list.append(mgf_dict)
            
        
    mgf.write(mgf_list, path.split('.csv')[0] + '_isotopes.mgf')
        
if __name__ == '__main__':
    path = r'C:\Users\xyy\Desktop\python\HaloAnalyzer_training\021_main_six_dataset\test_mzml_prediction\Lopadostoma_americanum_LF_19_485_2_P12_LiSha_CAZ_E12_halo_evaluation.csv'
    read_halo_evalution_mgf(path)