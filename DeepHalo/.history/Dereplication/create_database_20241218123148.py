import pandas as pd
from molmass import Formula
from multiprocessing import Pool
from HaloAnalyzer.Dataset.methods_main import isotope_simulation
from functools import partial
import os

class Dataset:
    def __init__(self, path, key) -> None:
        """
        This class is used to create a dataset from a given file.
        Compared to my_dataset.py in Dataset, this file does not remove duplicates,
        retains all information from the original file, and does not reset the index,
        making it easier to merge later.
        """
        self.key = key
        self.path = path
        # Read the database file
        if path.endswith('.json'):
            self.data = pd.read_json(path)
        elif path.endswith('.csv'):
            self.data = pd.read_csv(path, low_memory=False)
        
        # Drop rows with missing values in the specified key column
        self.data = self.data.dropna(subset=[key])
        # Rename the specified key column to 'formula'
        self.data = self.data.rename(columns={key: 'formula'})

    def create_dataset(self, type):
        """
        Create a specified dataset based on the formulas in self.data.
        """
        #isotope_simulation
        self.data['dict_isos'] = self.data['formula'].apply(isotope_simulation)
        self.data['mz_0'] = self.df_data['dict_isos'].apply(lambda x: x['mz_0'])
        self.data['p0_int'] = self.df_data['dict_isos'].apply(lambda x: x['p0_int'])
        self.data['p1_int'] = self.df_data['dict_isos'].apply(lambda x: x['p1_int'])
        self.data['p2_int'] = self.df_data['dict_isos'].apply(lambda x: x['p2_int'])
        self.data['p3_int'] = self.df_data['dict_isos'].apply(lambda x: x['p3_int'])
        self.data['p4_int'] = self.df_data['dict_isos'].apply(lambda x: x['p4_int'])
        #del column 'dict_isos'
        self.df_data = self.data.drop(columns=['dict_isos'])
        
        # Add columns for M+H, M+Na, and M+NH4
        df_data_pos_3 = self.df_data[self.df_data['formula_x'].str.contains(r'\+3')]
        df_data_pos_3.loc[:, 'M+H'] = df_data_pos_3['mz_0'] - 1.007825 * 2
        
        df_data_pos_2 = self.df_data[self.df_data['formula_x'].str.contains(r'\+2')]
        df_data_pos_2.loc[:, 'M+H'] = df_data_pos_2['mz_0'] - 1.007825

        df_data_pos_1 = self.df_data[~self.df_data['formula_x'].str.contains(r'\+2') & ~self.df_data['formula_x'].str.contains(r'\+3')]
        df_data_pos_1 = df_data_pos_1[df_data_pos_1['formula_x'].str.contains(r'\+')]
        df_data_pos_1.loc[:, 'M+H'] = df_data_pos_1['mz_0']

        df_data_neg = self.df_data[self.df_data['formula_x'].str.contains(r'\-')]
        df_data_neg.loc[:, 'M+H'] = 0 / df_data_neg['mz_0']
        
        df_data_neut = self.df_data[~self.df_data['formula_x'].str.contains(r'\+') & ~self.df_data['formula_x'].str.contains(r'\-')]
        df_data_neut.loc[:, 'M+H'] = df_data_neut['mz_0'] + 1.007825

        self.df_data = pd.concat([df_data_pos_3, df_data_pos_2, df_data_pos_1, df_data_neg, df_data_neut], ignore_index=True)
        self.df_data['M+Na'] = self.df_data['M+H'] + 21.981943
        self.df_data['M+NH4'] = self.df_data['M+H'] + 17.026549

    def work_flow(self):
        """
        Execute the workflow to create and save the dataset.
        """
        # Create the base dataset
        self.create_dataset('base')
        
        # Save the dataset to the specified path
        self.df_data.to_csv(self.path.replace('.csv', '_for_dereplication.csv'))

if __name__ == '__main__':
    # Create a Dataset object and execute the workflow
    database_file = r'D:\workissues\manuscript\halo_mining\mining\dereplication_database\FunGBE\FunGBE_V202412.csv'
    test = Dataset(database_file, 'formula')
    test.work_flow()

