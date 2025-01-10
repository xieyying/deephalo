import pandas as pd
from molmass import Formula
from multiprocessing import Pool
from HaloAnalyzer.Dataset.methods_main import create_data
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
        # Create a pool of workers
        pool = Pool(4)
        # Create a partial function with the specified type
        func = partial(create_data, type=type)
        # Map the function to the formulas in the data
        dfs = pool.map(func, [formula for formula in self.data['formula']])
        # Close the pool
        pool.close()
        # Concatenate the results into a single DataFrame
        df = pd.concat(dfs, ignore_index=True)
        self.df_data = df
        
        # Merge self.data with self.df_data
        self.df_data = self.data.merge(self.df_data, left_index=True, right_index=True)
        
        # Add columns for compound_m_plus_h, compound_m_plus_na, and compound_m_plus_nh4 if they do not exist
        if 'compound_m_plus_h' not in self.df_data.columns:
            self.df_data['compound_m_plus_h'] = self.df_data['mz_0'] + 1.007825
        if 'compound_m_plus_na' not in self.df_data.columns:
            self.df_data['compound_m_plus_na'] = self.df_data['mz_0'] + 22.989218
        if 'compound_m_plus_nh4' not in self.df_data.columns:
            self.df_data['compound_m_plus_nh4'] = self.df_data['mz_0'] + 18.033823

    def save(self, path):
        """
        Save the dataset to a CSV file.

        Parameters:
        path: str, the path to save the file
        """
        # Save the DataFrame to a CSV file
        self.df_data.to_csv(path, index=False)

    def work_flow(self):
        """
        Execute the workflow to create and save the dataset.
        """
        # Create the base dataset
        self.create_dataset('base')
        
        # Save the dataset to the specified path
        
        self.save(self.path.replace('.csv', '_for_dereplication.csv')) 

if __name__ == '__main__':
    # Create a Dataset object and execute the workflow
    database_file =r'D:\workissues\manuscript\halo_mining\mining\dereplication_database\FunGBE\FunGBE_V202412.csv'
    test = Dataset(database_file, 'formula')
    test.work_flow()