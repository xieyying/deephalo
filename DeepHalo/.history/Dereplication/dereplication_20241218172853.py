import pandas as pd
import os
import numpy as np

class Dereplication:
    def __init__(self, databases, Deephalo_output) -> None:
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
        self.data = {}
        for dataname, path in databases.items():
            data = pd.read_csv(path, low_memory=False, encoding='utf-8-sig')
            self.data[dataname] = data
        self.Deephalo_output = Deephalo_output

    def dereplication(self):
        """
        Perform dereplication using the database.
        """
        for dataname, data in self.data.items():
            # Ensure relevant columns are numeric
            data['M+H'] = pd.to_numeric(data['M+H'], errors='coerce')
            data['M+Na'] = pd.to_numeric(data['M+Na'], errors='coerce')
            # data['M+NH4'] = pd.to_numeric(data['M+NH4'], errors='coerce')
            
            # Rename columns to 'compound_names' if necessary
            if 'compound_name' in data.columns:
                data = data.rename(columns={'compound_name': 'compound_names'})
            elif 'names' in data.columns:
                data = data.rename(columns={'names': 'compound_names'})
            elif 'compound' in data.columns:
                data = data.rename(columns={'compound': 'compound_names'})
            elif 'compounds' in data.columns:
                data = data.rename(columns={'compounds': 'compound_names'})
            elif 'compound_names' in data.columns:
                pass
            else:
                raise ValueError('The database does not contain a column with compound names.')
            
            for idx, row in self.Deephalo_output.iterrows():
                mz = (row['mz']*row['charge'])-((row['charge']-1)*1.007825)
                # Find compounds in the database that are close to the mz value in Deephalo_output
                data_h = data[(abs(data['M+H'] - mz) <= mz * 1e-5) & (data['formula'].str.contains('Br|Cl'))]
                data_na = data[(abs(data['M+Na'] - mz) <= mz * 1e-5) & (data['formula'].str.contains('Br|Cl'))]
                # data_nh4 = data[(abs(data['M+NH4'] - mz) <= mz * 1e-5) & (data['formula'].str.contains('Br|Cl'))]
                data_ = pd.concat([data_h, data_na], ignore_index=True)
                dereplications = {'compound_names': [], 'intensity_score': [], 'error_ppm': [], 'Smiles': [], 'adducts': []}

                if data_.empty:
                    self.Deephalo_output.loc[idx, f'compound_names_{dataname}'] = 'None'
                    self.Deephalo_output.loc[idx, f'intensity_score_{dataname}'] = 0
                    self.Deephalo_output.loc[idx, f'error_ppm_{dataname}'] = 1e6
                    self.Deephalo_output.loc[idx, f'dereplication_{dataname}'] = 'None'
                    self.Deephalo_output.loc[idx, f'Smiles_{dataname}'] = 'None'
                    self.Deephalo_output.loc[idx, f'adducts_{dataname}'] = 'None'
                else:
                    for idx_, row_ in data_.iterrows():
                        # Calculate intensity_score
                        row['inty_list'] = row[['p0_int', 'p1_int', 'p2_int', 'p3_int', 'p4_int']].tolist()
                        row_['inty_list'] = row_[['p0_int', 'p1_int', 'p2_int', 'p3_int', 'p4_int']].tolist()
                        row_['intensity_score'] = cosine_similarity(row['inty_list'], row_['inty_list'])
                        if row_['intensity_score']>0.96:
                            dereplications['compound_names'].append(row_['compound_names'])
                            dereplications['intensity_score'].append(row_['intensity_score'])
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
                                
          
                    if dereplications:
                        self.Deephalo_output.loc[idx, f'dereplication_{dataname}'] = str(dereplications)
                        dereplications_df = pd.DataFrame(dereplications)
                        max_intensity_idx = dereplications_df['intensity_score'].idxmax()
                        self.Deephalo_output.loc[idx, f'compound_names_{dataname}'] = dereplications_df.loc[max_intensity_idx, 'compound_names']
                        self.Deephalo_output.loc[idx, f'intensity_score_{dataname}'] = dereplications_df.loc[max_intensity_idx, 'intensity_score']
                        self.Deephalo_output.loc[idx, f'error_ppm_{dataname}'] = dereplications_df.loc[max_intensity_idx, 'error_ppm']
                        self.Deephalo_output.loc[idx, f'Smiles_{dataname}'] = dereplications_df.loc[max_intensity_idx, 'Smiles']
                        self.Deephalo_output.loc[idx, f'adducts_{dataname}'] = dereplications_df.loc[max_intensity_idx, 'adducts']
                    else:
                        self.Deephalo_output.loc[idx, f'compound_names_{dataname}'] = 'None'
                        self.Deephalo_output.loc[idx, f'intensity_score_{dataname}'] = 0
                        self.Deephalo_output.loc[idx, f'error_ppm_{dataname}'] = 1e6
                        self.Deephalo_output.loc[idx, f'dereplication_{dataname}'] = 'None'
                        self.Deephalo_output.loc[idx, f'Smiles_{dataname}'] = 'None'
                        self.Deephalo_output.loc[idx, f'adducts_{dataname}'] = 'None'
                            
                    
        return self.Deephalo_output

    def merge_columns(self, df, datanames):
        for col in ['compound_names', 'intensity_score', 'error_ppm', 'Smiles', 'adducts']:
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
          
def cosine_similarity(inty_list1, inty_list2):
    """
    Calculate the cosine similarity between inty_list1 and inty_list2.

    Parameters:
    inty_list1 (list or array): First list of intensities.
    inty_list2 (list or array): Second list of intensities.

    Returns:
    float: Cosine similarity between inty_list1 and inty_list2.
    """
    # Convert lists to numpy arrays of float type
    inty_list1 = np.array(inty_list1, dtype=float)
    inty_list2 = np.array(inty_list2, dtype=float)

    # Calculate the dot product
    dot_product = np.dot(inty_list1, inty_list2)
    
    # Calculate the norms (magnitudes) of the vectors
    norm1 = np.linalg.norm(inty_list1)
    norm2 = np.linalg.norm(inty_list2)
    
    # Calculate the cosine similarity
    if norm1 == 0 or norm2 == 0:
        return 0.0  # Avoid division by zero
    cosine_similarity = dot_product / (norm1 * norm2)
    
    return cosine_similarity

def combine_columns( row, columns):
    for col in columns:
        if row[col] != 'None' and row[col] != 1e6:
            return row[col]
    return 'None' if 'error_ppm' not in columns else 1e6


if __name__ == '__main__':
    """
    Example usage of the Dereplication class.
    
    """
    databases = {'NPAtlas':r'D:\workissues\manuscript\halo_mining\mining\dereplication_database\NPAtlas\NPAtlas_download_2024_03_for_dereplication_bacteria.csv',
                 'FunGBE':r'D:\workissues\manuscript\halo_mining\mining\dereplication_database\FunGBE\FunGBE_V202412_for_dereplication.csv'}
    
    Deephalo_output_result = r'C:\Users\xyy\Desktop\python\HaloAnalyzer_training\022_six_dataset_openms_noClFe\2M_fake_molecules\result\20241218_OSMAC_V3\halo'
    
    files = os.listdir(Deephalo_output_result)
    Deephalo_outputs = [file for file in files if file.endswith('feature.csv')]
    dereplication_folder = Deephalo_output_result.replace('halo', 'dereplication')
    os.makedirs(dereplication_folder, exist_ok=True)
    
    for Deephalo_output in Deephalo_outputs:
        Deephalo_output_df = pd.read_csv(os.path.join(Deephalo_output_result, Deephalo_output))
        df = Dereplication(databases, Deephalo_output_df).workflow()
        df.to_csv(os.path.join(dereplication_folder, Deephalo_output), index=False)