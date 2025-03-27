import pandas as pd
import os
import numpy as np
from .method_main import cosine_similarity, combine_columns

class Dereplication:
    def __init__(self, databases, Deephalo_output,error_ppm=10,Inty_cosine_score=0.96) -> None:
        """
        Perform dereplication based on databases and the output of Deephalo.
        To dereplicate compounds based on multiple databases.

        Parameters:
        databases (dict): {'dataname': Paths to the database file}.
        Deephalo_output (pd.DataFrame): Deephalo output results.

        Attributes:
        data (dict): {'dataname': DataFrame of the database}.
        Deephalo_output (pd.DataFrame): Deephalo output results.
        """
        self.data = databases
        self.Deephalo_output = Deephalo_output
        self.error_ppm = error_ppm
        self.Inty_cosine_score = Inty_cosine_score


    def dereplication(self):
        """
        Perform dereplication using the database.
        """
        for dataname, data in self.data.items():
            # Ensure relevant columns are numeric
            data['M+H'] = pd.to_numeric(data['M+H'], errors='coerce')
            data['M+Na'] = pd.to_numeric(data['M+Na'], errors='coerce')
            # data['M+NH4'] = pd.to_numeric(data['M+NH4'], errors='coerce')
            
            # Rename any column containing 'name' (case insensitive) to 'compound_names'
            name_columns = [col for col in data.columns if ('compound_name' or 'compound_names' or 'compound name' or 'compound names') in col.lower()]
            if name_columns:
                # If multiple columns match, use the first one
                data = data.rename(columns={name_columns[0]: 'compound_names'})
            else:
                raise ValueError('The database does not contain a column with compound names.')
            
            for idx, row in self.Deephalo_output.iterrows():
                mz = (row['mz']*row['charge'])-((row['charge']-1)*1.007825)
                # Find compounds in the database that are close to the mz value in Deephalo_output
                data_h = data[(abs(data['M+H'] - mz) <= mz * self.error_ppm*1e-6) & (data['formula'].str.contains('Br|Cl'))]
                data_na = data[(abs(data['M+Na'] - mz) <= mz * self.error_ppm*1e-6) & (data['formula'].str.contains('Br|Cl'))]
                # data_nh4 = data[(abs(data['M+NH4'] - mz) <= mz * 1e-5) & (data['formula'].str.contains('Br|Cl'))]
                data_ = pd.concat([data_h, data_na], ignore_index=True)
                dereplications = {'compound_names': [], 'Inty_cosine_score': [], 'error_ppm': [], 'Smiles': [], 'adducts': []}

                if data_.empty:
                    self.Deephalo_output.loc[idx, f'compound_names_{dataname}'] = 'None'
                    self.Deephalo_output.loc[idx, f'Inty_cosine_score_{dataname}'] = 0
                    self.Deephalo_output.loc[idx, f'error_ppm_{dataname}'] = 1e6
                    self.Deephalo_output.loc[idx, f'dereplication_{dataname}'] = 'None'
                    self.Deephalo_output.loc[idx, f'Smiles_{dataname}'] = 'None'
                    self.Deephalo_output.loc[idx, f'adducts_{dataname}'] = 'None'
                else:
                    for idx_, row_ in data_.iterrows():
                        # Calculate Inty_cosine_score
                        row['inty_list'] = row[['p0_int', 'p1_int', 'p2_int', 'p3_int', 'p4_int']].tolist()
                        row_['inty_list'] = row_[['p0_int', 'p1_int', 'p2_int', 'p3_int', 'p4_int']].tolist()
                        row_['Inty_cosine_score'] = cosine_similarity(row['inty_list'], row_['inty_list'])
                        if row_['Inty_cosine_score']>self.Inty_cosine_score:
                            dereplications['compound_names'].append(row_['compound_names'])
                            dereplications['Inty_cosine_score'].append(row_['Inty_cosine_score'])
                            dereplications['Smiles'].append(row_.get('Smiles', 'None'))
                            
                            error_h = abs(mz - row_['M+H']) / mz * 1e6
                            error_na = abs(mz - row_['M+Na']) / mz * 1e6
                            # error_nh4 = abs(mz - row_['M+NH4']) / mz * 1e6
                            min_error = min(error_h, error_na)
                            dereplications['error_ppm'].append(min_error)
                            
                            if min_error == error_h:
                                dereplications['adducts'].append('M+H')
                            elif min_error == error_na:
                                dereplications['adducts'].append('M+Na')
                            # else:
                            #     dereplications['adducts'].append('M+NH4')
                            else:
                                pass
                        else:
                            pass
                                
          
                    if dereplications['compound_names']:
                        self.Deephalo_output.loc[idx, f'dereplication_{dataname}'] = str(dereplications)
                        dereplications_df = pd.DataFrame(dereplications)
                        # max_intensity_idx = dereplications_df['Inty_cosine_score'].idxmax()
                        #if multiple compounds in dereplications_df, keep all compounds
                        self.Deephalo_output.loc[idx, f'compound_names_{dataname}'] = str(dereplications_df['compound_names'])
                        self.Deephalo_output.loc[idx, f'Inty_cosine_score_{dataname}'] = str(dereplications_df['Inty_cosine_score'])
                        self.Deephalo_output.loc[idx, f'error_ppm_{dataname}'] = str(dereplications_df['error_ppm'])
                        self.Deephalo_output.loc[idx, f'Smiles_{dataname}'] = str(dereplications_df['Smiles'])
                        self.Deephalo_output.loc[idx, f'adducts_{dataname}'] = str(dereplications_df['adducts'])
                        
                        
                        # self.Deephalo_output.loc[idx, f'compound_names_{dataname}'] = dereplications_df.loc[max_intensity_idx, 'compound_names']
                        # self.Deephalo_output.loc[idx, f'Inty_cosine_score_{dataname}'] = dereplications_df.loc[max_intensity_idx, 'Inty_cosine_score']
                        # self.Deephalo_output.loc[idx, f'error_ppm_{dataname}'] = dereplications_df.loc[max_intensity_idx, 'error_ppm']
                        # self.Deephalo_output.loc[idx, f'Smiles_{dataname}'] = dereplications_df.loc[max_intensity_idx, 'Smiles']
                        # self.Deephalo_output.loc[idx, f'adducts_{dataname}'] = dereplications_df.loc[max_intensity_idx, 'adducts']
                    else:
                        self.Deephalo_output.loc[idx, f'compound_names_{dataname}'] = 'None'
                        self.Deephalo_output.loc[idx, f'Inty_cosine_score_{dataname}'] = 0
                        self.Deephalo_output.loc[idx, f'error_ppm_{dataname}'] = 1e6
                        self.Deephalo_output.loc[idx, f'dereplication_{dataname}'] = 'None'
                        self.Deephalo_output.loc[idx, f'Smiles_{dataname}'] = 'None'
                        self.Deephalo_output.loc[idx, f'adducts_{dataname}'] = 'None'
                            
                    
        return self.Deephalo_output

    def merge_columns(self, df, datanames):
        for col in ['compound_names', 'Inty_cosine_score', 'error_ppm', 'Smiles', 'adducts']:
            columns = [f'{col}_{dataname}' for dataname in datanames]
            df[col] = df.apply(lambda row: combine_columns(row, columns), axis=1)
        return df
    
    def workflow(self):
        """
        Execute the workflow to perform dereplication.
        """
        df = self.dereplication()
        datanames = list(self.data.keys())
        df = self.merge_columns(df, datanames)
        return df

