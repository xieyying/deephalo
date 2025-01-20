import pandas as pd
import os
from HaloAnalyzer.Dereplication.dereplication import Dereplication


if __name__ == '__main__':
    """
    Example usage of the Dereplication class.
    
    """
    # databases = {'NPAtlas':r'D:\workissues\manuscript\halo_mining\mining\dereplication_database\NPAtlas\NPAtlas_download_2024_03_for_dereplication_bacteria.csv',
    #              'FunGBE':r'D:\workissues\manuscript\halo_mining\mining\dereplication_database\FunGBE\FunGBE_V202412_for_dereplication.csv'}
    databases = {'testbase':r'E:\github\renew\HaloAnalyzer\HaloAnalyzer\Test\Resoure\Demo_dereplication_database_for_dereplication.csv'}
    Deephalo_output_result = r'C:\Users\xq75\Desktop\p_test\result\test'
    
    files = os.listdir(Deephalo_output_result)
    Deephalo_outputs = [file for file in files if file.endswith('feature.csv')]
    dereplication_folder = Deephalo_output_result.replace('test', 'dereplication')
    print(dereplication_folder)
    os.makedirs(dereplication_folder, exist_ok=True)
    
    for Deephalo_output in Deephalo_outputs:
        Deephalo_output_df = pd.read_csv(os.path.join(Deephalo_output_result, Deephalo_output))
        df = Dereplication(databases, Deephalo_output_df,50).workflow()
        df.to_csv(os.path.join(dereplication_folder, Deephalo_output), index=False)
    
    dereplication_folder_2 = Deephalo_output_result.replace('test', 'dereplication_2') 
    os.makedirs(dereplication_folder_2, exist_ok=True)
    files = os.listdir(dereplication_folder)
    for  f in files:
        df = pd.read_csv(os.path.join(dereplication_folder, f))
        df = df[df['Inty_cosine_score']>=0.96]
        if df.empty:
            continue
        df.to_csv(os.path.join(dereplication_folder_2, f), index=False)