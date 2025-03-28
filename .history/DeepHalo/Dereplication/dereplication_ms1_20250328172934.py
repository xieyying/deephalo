import pandas as pd
import numpy as np
import concurrent.futures
import os
import shutil

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
        self.Deephalo_output = self.Deephalo_output.copy()
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
                        if len(dereplications_df)>1:
                            self.Deephalo_output.loc[idx, f'compound_names_{dataname}'] = str(dereplications_df['compound_names'].tolist())
                            self.Deephalo_output.loc[idx, f'Inty_cosine_score_{dataname}'] = str(dereplications_df['Inty_cosine_score'].tolist())
                            self.Deephalo_output.loc[idx, f'error_ppm_{dataname}'] = str(dereplications_df['error_ppm'].tolist())
                            self.Deephalo_output.loc[idx, f'Smiles_{dataname}'] = str(dereplications_df['Smiles'].tolist())
                            self.Deephalo_output.loc[idx, f'adducts_{dataname}'] = str(dereplications_df['adducts'].tolist())
                        else:
                            self.Deephalo_output.loc[idx, f'compound_names_{dataname}'] = dereplications_df.loc[0, 'compound_names']
                            self.Deephalo_output.loc[idx, f'Inty_cosine_score_{dataname}'] = dereplications_df.loc[0, 'Inty_cosine_score']
                            self.Deephalo_output.loc[idx, f'error_ppm_{dataname}'] = dereplications_df.loc[0, 'error_ppm']
                            self.Deephalo_output.loc[idx, f'Smiles_{dataname}'] = dereplications_df.loc[0, 'Smiles']
                            self.Deephalo_output.loc[idx, f'adducts_{dataname}'] = dereplications_df.loc[0, 'adducts']
                        
                    else:
                        self.Deephalo_output.loc[idx, f'compound_names_{dataname}'] = 'None'
                        self.Deephalo_output.loc[idx, f'Inty_cosine_score_{dataname}'] = 0
                        self.Deephalo_output.loc[idx, f'error_ppm_{dataname}'] = 1e6
                        self.Deephalo_output.loc[idx, f'dereplication_{dataname}'] = 'None'
                        self.Deephalo_output.loc[idx, f'Smiles_{dataname}'] = 'None'
                        self.Deephalo_output.loc[idx, f'adducts_{dataname}'] = 'None'
                            
                    
        return self.Deephalo_output
    def merge_columns(self, df, datanames):
        df = df.copy()  # Ensure working on a copy to avoid SettingWithCopyWarning
        for col in ['compound_names', 'Inty_cosine_score', 'error_ppm', 'Smiles', 'adducts']:
            columns = [f'{col}_{dataname}' for dataname in datanames]
            df.loc[:, col] = df.apply(lambda row: combine_columns(row, columns), axis=1)
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
    inty_list1 = np.array(inty_list1, dtype=float)
    inty_list2 = np.array(inty_list2, dtype=float)
    dot_product = np.dot(inty_list1, inty_list2)
    norm1 = np.linalg.norm(inty_list1)
    norm2 = np.linalg.norm(inty_list2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)

def combine_columns(row, columns):
    for col in columns:
        if row[col] not in ['None', 1e6]:
            return row[col]
    return 'None' if 'error_ppm' not in columns else 1e6

def process_dereplication_file(file, Deephalo_output_result, dereplication_database, para, dereplication_folder):
    """
    Process a single dereplication file.
    """
    # Read the feature file
    file_path = os.path.join(Deephalo_output_result, file)
    Deephalo_output_df = pd.read_csv(file_path)
    if para.FeatureFilter_H_score_threshold < 0.4:
        # Split data based on H_score threshold
        df_halo = Deephalo_output_df[Deephalo_output_df['H_score'] >= 0.4]
        df_non_halo = Deephalo_output_df[Deephalo_output_df['H_score'] < 0.4]
        
        # Run the dereplication process on the high-confidence (halo) subset
        df_derep = Dereplication(dereplication_database, df_halo, para.dereplication_error, para.dereplication_Inty_cosine_score).workflow()
        # df_non_halo中没有的列，用None填充
        df_non_halo = df_non_halo.copy()
        for col in df_derep.columns:
            if col not in df_non_halo.columns:
                df_non_halo[col] = None
        # Combine results with non-halo features
        df_final = pd.concat([df_derep, df_non_halo])
    else:
        df_final = Dereplication(dereplication_database, Deephalo_output_df, para.dereplication_error, para.dereplication_Inty_cosine_score).workflow()
    
    # Save results to dereplication folder
    output_path = os.path.join(dereplication_folder, file)
    df_final.to_csv(output_path, index=False)
    
    print(f"Dereplication completed for {file}")
    return file

def dereplicationms1(para, dereplication_database):
    # Confirm that the feature files exist in the Deephalo output folder
    Deephalo_output_result = os.path.join(para.args_project_path, 'result/halo')
    if not os.path.exists(Deephalo_output_result):
        raise FileNotFoundError(f"{Deephalo_output_result} does not exist, please check the path")

    # Read the feature files from the Deephalo output folder
    files = os.listdir(Deephalo_output_result)
    Deephalo_outputs = [file for file in files if file.endswith('feature.csv')]
    # Create the output folder for dereplication results
    dereplication_folder = os.path.join(para.args_project_path, 'dereplication')
    os.makedirs(dereplication_folder, exist_ok=True)
    
    if dereplication_database is not None:
        # Process each file in parallel using ProcessPoolExecutor
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(
                    process_dereplication_file,
                    file,
                    Deephalo_output_result,
                    dereplication_database,
                    para,
                    dereplication_folder
                )
                for file in Deephalo_outputs
            ]
            
            # Wait for all tasks to complete and retrieve results
            for future in concurrent.futures.as_completed(futures):
                try:
                    processed_file = future.result()
                    print(f"Successfully processed: {processed_file}")
                except Exception as e:
                    print(f"Error processing file: {e}") 
    else:
        #将Deephalo_outputs中的文件直接复制到dereplication_folder中
        for file in Deephalo_outputs:
            src_path = os.path.join(Deephalo_output_result, file)
            dst_path = os.path.join(dereplication_folder, file)
            shutil.copy(src_path, dst_path)         
    return dereplication_folder

