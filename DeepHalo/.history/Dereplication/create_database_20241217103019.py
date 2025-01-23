import pandas as pd
from molmass import Formula
from multiprocessing import Pool
from Dataset.methods_main import create_data
from functools import partial
import os

class Dataset():
    def __init__(self,path,key) -> None:
        """
        与Dataset中的my_dataset.py文件相比，这个文件的主要区别在于：
        不去重
        """

        #读取数据库中的formula
        if path.endswith('.json'):
            self.data = pd.read_json(path)[[key]].dropna()
            #将key列名改为formula
            self.data = self.data.rename(columns={key:'formula'})
        elif path.endswith('.csv'):
            self.data = pd.read_csv(path,low_memory=False)[[key]].dropna()
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
        #利用merge将self.data和self.df_data合并，根据index
        self.df_data = self.data.merge(self.df_data,left_index=True,right_index=True)
        
        #添加compound_m_plus_h,compound_m_plus_na,compound_m_plus_nh4列
        
        
        self.df_data.to_csv(path,index=False)

    def work_flow(self,type):

        self.create_dataset('base')#para.return_from_max_ints)
        
        #如果不存在dataset文件夹，则创建

        if not os.path.exists('./dereplicate_database'):
            os.mkdir('./dereplicate_database')
        self.save('./dereplicate_database/'+type+'.csv')
        
if __name__ == '__main__':
    test = Dataset('K:\open-database\NPAtlas\V2024_03\NPAtlas_download_2024_03.csv','compound_molecular_formula')
    test.work_flow(100,1000,['C','H','O','N','S'],'hydro')