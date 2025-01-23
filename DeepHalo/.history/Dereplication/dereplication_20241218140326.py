import pandas as pd
import numpy as np
import os

class Dereplication:
    def __init__(self, databases, Deephalo_output) -> None:
        """
        Perform dereplication based on database and the output of Deephalo.

        Parameters:
        databases (list): [Paths to the database file].
        Deephalo_output (pd.DataFrame): Deephalo output results.

        Attributes:
        data (pd.DataFrame): The database.
        Deephalo_output (pd.DataFrame): Deephalo output results.
        """
        self.data = []
        for database in databases:
            data = pd.read_csv(database, low_memory=False, encoding='utf-8-sig')
            self.data.append(data)
            
        self.Deephalo_output = Deephalo_output
        
    def dereplication(self):
        """
        Perform dereplication using the database.
        """
        # Ensure relevant columns are numeric
        self.data['compound_m_plus_h'] = pd.to_numeric(self.data['compound_m_plus_h'], errors='coerce')
        self.data['compound_m_plus_na'] = pd.to_numeric(self.data['compound_m_plus_na'], errors='coerce')
        self.data['compound_m_plus_nh4'] = pd.to_numeric(self.data['compound_m_plus_nh4'], errors='coerce')
        
        # Rename columns to 'compound_names' if necessary
        if 'compound_name' in self.data.columns:
            self.data = self.data.rename(columns={'compound_name': 'compound_names'})
        elif 'names' in self.data.columns:
            self.data = self.data.rename(columns={'names': 'compound_names'})
        elif 'compound' in self.data.columns:
            self.data = self.data.rename(columns={'compound': 'compound_names'})
        elif 'compounds' in self.data.columns:
            self.data = self.data.rename(columns={'compounds': 'compound_names'})
        elif 'compound_names' in self.data.columns:
            pass
        else:
            raise ValueError('The database does not contain a column with compound names.')
        
        for idx, row in self.Deephalo_output.iterrows():
            # Find compounds in the database that are close to the mz value in Deephalo_output
            self.data_h = self.data[(abs(self.data['compound_m_plus_h'] - row['mz']) <= row['mz'] * 1e-5) & (self.data['formula'].str.contains('Br|Cl'))]
            self.data_na = self.data[(abs(self.data['compound_m_plus_na'] - row['mz']) <= row['mz'] * 1e-5) & (self.data['formula'].str.contains('Br|Cl'))]
            self.data_nh4 = self.data[(abs(self.data['compound_m_plus_nh4'] - row['mz']) <= row['mz'] * 1e-5) & (self.data['formula'].str.contains('Br|Cl'))]
            self.data_ = pd.concat([self.data_h, self.data_na, self.data_nh4], ignore_index=True)
            dereplications = {'compound_names': [], 'intensity_score': [], 'error_ppm': [], 'Smiles': [], 'adducts': []}

            if self.data_.empty:
                self.Deephalo_output.loc[idx, 'compound_names'] = 'None'
                self.Deephalo_output.loc[idx, 'intensity_score'] = 0
                self.Deephalo_output.loc[idx, 'error_ppm'] = 1e6
                self.Deephalo_output.loc[idx, 'dereplication'] = 'None'
                self.Deephalo_output.loc[idx, 'Smiles'] = 'None'
                self.Deephalo_output.loc[idx, 'adducts'] = 'None'
            else:
                for idx_, row_ in self.data_.iterrows():
                    dereplications['compound_names'].append(row_['compound_names'])
                    row['inty_list'] = row[['p0_int', 'p1_int', 'p2_int', 'p3_int', 'p4_int']].tolist()
                    row_['inty_list'] = row_[['p0_int', 'p1_int', 'p2_int', 'p3_int', 'p4_int']].tolist()
                    # Calculate intensity_score
                    row_['intensity_score'] = cosine_similarity(row['inty_list'], row_['inty_list'])
                    dereplications['intensity_score'].append(row_['intensity_score'])
                    dereplications['Smiles'].append(row_.get('Smiles', 'None'))
                    
                    error_h = abs(row['mz'] - row_['compound_m_plus_h']) / row['mz'] * 1e6
                    error_na = abs(row['mz'] - row_['compound_m_plus_na']) / row['mz'] * 1e6
                    error_nh4 = abs(row['mz'] - row_['compound_m_plus_nh4']) / row['mz'] * 1e6
                    min_error = min(error_h, error_na, error_nh4)
                    dereplications['error_ppm'].append(min_error)
                    
                    if min_error == error_h:
                        dereplications['adducts'].append('M+H')
                    elif min_error == error_na:
                        dereplications['adducts'].append('M+Na')
                    else:
                        dereplications['adducts'].append('M+NH4')

                self.Deephalo_output.loc[idx, 'dereplication'] = str(dereplications)
                dereplications_df = pd.DataFrame(dereplications)
                max_intensity_idx = dereplications_df['intensity_score'].idxmax()
                self.Deephalo_output.loc[idx, 'compound_names'] = dereplications_df.loc[max_intensity_idx, 'compound_names']
                self.Deephalo_output.loc[idx, 'intensity_score'] = dereplications_df.loc[max_intensity_idx, 'intensity_score']
                self.Deephalo_output.loc[idx, 'error_ppm'] = dereplications_df.loc[max_intensity_idx, 'error_ppm']
                self.Deephalo_output.loc[idx, 'Smiles'] = dereplications_df.loc[max_intensity_idx, 'Smiles']
                self.Deephalo_output.loc[idx, 'adducts'] = dereplications_df.loc[max_intensity_idx, 'adducts']
                
        return self.Deephalo_output
              
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

if __name__ == '__main__':
    database = r'K:\open-database\NPAtlas\V2024_03\dereplicate_database_base_bacteria.csv'
    Deephalo_output_result = r'C:\Users\xyy\Desktop\python\HaloAnalyzer_training\022_six_dataset_openms_noClFe\2M_fake_molecules\result\20241218_OSMAC_V3\halo'
    
    files = os.listdir(Deephalo_output_result)
    Deephalo_outputs = [file for file in files if file.endswith('feature.csv')]
    dereplication_folder = Deephalo_output_result.replace('halo', 'dereplication')
    os.makedirs(dereplication_folder, exist_ok=True)
    
    for Deephalo_output in Deephalo_outputs:
        Deephalo_output_df = pd.read_csv(os.path.join(Deephalo_output_result, Deephalo_output))
        dereplication_ = Dereplication(database, Deephalo_output_df)
        df = dereplication_.dereplication()
        df.to_csv(os.path.join(dereplication_folder, Deephalo_output), index=False)