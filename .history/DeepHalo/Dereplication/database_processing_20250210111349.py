import pandas as pd
from multiprocessing import Pool
from ..Dataset.methods_main import isotope_simulation
from molmass.molmass import FormulaError
import logging

class DereplicationDataset:
    def __init__(self, path, key) -> None:
        """
        This class is used to create a dataset from a given file.
        Compared to my_dataset.py in Dataset, this file does not remove duplicates,
        retains all information from the original file, and does not reset the index,
        making it easier to merge later.
        """  
        # Read the database file
        if path.endswith('.json'):
            self.data = pd.read_json(path)
        elif path.endswith('.csv'):
            self.data = pd.read_csv(path, low_memory=False)
        
        # Drop rows with missing values in the specified key column
        self.data = self.data.dropna(subset=[key])
        # Rename the specified key column to 'formula'
        self.data = self.data.rename(columns={key: 'formula'})
        self.path = path
    def safe_isotope_simulation(formula):
        """
        Safely apply isotope simulation with error handling
        """
        try:
            return isotope_simulation(formula)
        except FormulaError as e:
            logging.warning(f"Invalid formula: {formula}. Error: {str(e)}")
            return {
                'mz_0': 0,
                'p0_int': 0,
                'p1_int': 0,
                'p2_int': 0,
                'p3_int': 0,
                'p4_int': 0
            }

    def create_dataset(self, type):
        """
        Create a specified dataset based on the formulas in self.data.
        """
        #isotope_simulation
        self.data['dict_isos'] = self.data['formula'].apply(isotope_simulation)
        self.data['mz_0'] = self.data['dict_isos'].apply(lambda x: x['mz_0'])
        self.data['p0_int'] = self.data['dict_isos'].apply(lambda x: x['p0_int'])
        self.data['p1_int'] = self.data['dict_isos'].apply(lambda x: x['p1_int'])
        self.data['p2_int'] = self.data['dict_isos'].apply(lambda x: x['p2_int'])
        self.data['p3_int'] = self.data['dict_isos'].apply(lambda x: x['p3_int'])
        self.data['p4_int'] = self.data['dict_isos'].apply(lambda x: x['p4_int'])
        #del column 'dict_isos'
        self.df_data = self.data.drop(columns=['dict_isos'])
        
        # Add columns for M+H, M+Na, and M+NH4
        df_data_pos_3 = self.df_data[self.df_data['formula'].str.contains(r'\+3')]
        df_data_pos_3 = df_data_pos_3.copy()
        df_data_pos_3.loc[:, 'M+H'] = df_data_pos_3['mz_0'] - 1.007825 * 2
        
        df_data_pos_2 = self.df_data[self.df_data['formula'].str.contains(r'\+2')]
        df_data_pos_2 = df_data_pos_2.copy()
        df_data_pos_2.loc[:, 'M+H'] = df_data_pos_2['mz_0'] - 1.007825

        df_data_pos_1 = self.df_data[~self.df_data['formula'].str.contains(r'\+2') & ~self.df_data['formula'].str.contains(r'\+3')]
        df_data_pos_1 = df_data_pos_1[df_data_pos_1['formula'].str.contains(r'\+')]
        df_data_pos_1 = df_data_pos_1.copy()
        df_data_pos_1.loc[:, 'M+H'] = df_data_pos_1['mz_0']

        df_data_neg = self.df_data[self.df_data['formula'].str.contains(r'\-')]
        df_data_neg = df_data_neg.copy()
        df_data_neg.loc[:, 'M+H'] = 0 / df_data_neg['mz_0']
        
        df_data_neut = self.df_data[~self.df_data['formula'].str.contains(r'\+') & ~self.df_data['formula'].str.contains(r'\-')]
        df_data_neut = df_data_neut.copy()
        df_data_neut.loc[:, 'M+H'] = df_data_neut['mz_0'] + 1.007825

        self.df_data = pd.concat([df_data_pos_3, df_data_pos_2, df_data_pos_1, df_data_neg, df_data_neut], ignore_index=True)
        self.df_data = self.df_data.copy()
        self.df_data['M+Na'] = self.df_data['M+H'] + 21.981943
        self.df_data['M+NH4'] = self.df_data['M+H'] + 17.026549

    def work_flow(self):
        """
        Execute the workflow to create and save the dataset.
        """
        # Create the base dataset
        self.create_dataset('base')
        
        # Save the dataset to the specified path
        return self.df_data

if __name__ == '__main__':
    pass

