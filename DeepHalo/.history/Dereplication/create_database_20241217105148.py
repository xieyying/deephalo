import pandas as pd
from molmass import Formula
from multiprocessing import Pool
from HaloAnalyzer.Dataset.methods_main import create_data
from functools import partial
import os

class Dataset():
    def __init__(self,path,key) -> None:
        """
        与Dataset中的my_dataset.py文件相比，这个文件的主要区别在于：
        不去重,并保留原始文件中的所有信息
        """

        #读取数据库中的formula
        if path.endswith('.json'):
            self.data = pd.read_json(path)
   
        elif path.endswith('.csv'):
            self.data = pd.read_csv(path,low_memory=False)
  
        self.data = self.data.dropna(subset=[key])
        self.data = self.data.rename(columns={key:'formula'})

    def create_dataset(self,type):#return_from_max_ints):
        """
        依据self.data中的formula，构建指定数据集

        """
        #基础数据集
        pool = Pool(4)
        func = partial(create_data, type=type,)#return_from_max_ints=return_from_max_ints)
        dfs = pool.map(func, [formula for formula in self.data['formula']])
        pool.close()
        df = pd.concat(dfs,ignore_index=True)
        self.df_data = df
             
    def save(self,path):
        """
        将数据集保存为csv文件

        参数:
        path: str，保存路径
        """
        #将self.data与self.df_data合并
        self.df_data = self.data.merge(self.df_data,left_index=True,right_index=True)
        
        #如果没有compound_m_plus_h,compound_m_plus_na,compound_m_plus_nh4列，则添加
        if 'compound_m_plus_h' not in self.df_data.columns:
            self.df_data['compound_m_plus_h'] = self.df_data['mz_0']+1.007825
        if 'compound_m_plus_na' not in self.df_data.columns:
            self.df_data['compound_m_plus_na'] = self.df_data['mz_0']+22.989218
        if 'compound_m_plus_nh4' not in self.df_data.columns:
            self.df_data['compound_m_plus_nh4'] = self.df_data['mz_0']+18.033823
        
        self.df_data.to_csv(path,index=False)

    def work_flow(self):

        self.create_dataset('base')#para.return_from_max_ints)
        
        #如果不存在dataset文件夹，则创建

        # if not os.path.exists('./dereplicate_database'):
        #     os.mkdir('./dereplicate_database')
        # self.save('./dereplicate_database/'+'base.csv')
        self.save(r'K:\open-database\NPAtlas\V2024_03\dereplicate_database_base.csv')
        
if __name__ == '__main__':
    test = Dataset(r'K:\open-database\NPAtlas\V2024_03\NPAtlas_download_2024_03.csv','compound_molecular_formula')
    test.work_flow()