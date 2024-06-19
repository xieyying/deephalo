import pandas as pd
from molmass import Formula
from multiprocessing import Pool
from .methods_main import create_data
from functools import partial
import os

class dataset():
    def __init__(self,path,key) -> None:
        """
        此类可依据数据库中真实化合物的formula，构建基础数据集。
        并可依据基础数据集，构建加噪音数据集、模拟螯合铁数据集、模拟脱氢数据集、模拟加氢数据集。

        参数：
        path:str，真实数据库文件路径
        key:str，数据库中真实化合物的formula列名

        属性：
        data:pd.DataFrame，数据库中真实化合物的formula

        方法：
        filter:过滤数据
        filt:多进程的方式map filter，默认使用全部cpu
        create_dataset:依据self.data中的formula，构建指定数据集
        save:将数据集保存为csv文件
        work_flow:创建指定数据集的工作流程

        使用示例：
        test = dataset('test.json','mol_formula')
        test.work_flow(100,1000,['C','H','O','N','S'],'hydro')
        """
        #读取数据库中真实化合物的formula
        if path.endswith('.json'):
            self.data = pd.read_json(path)[[key]].dropna()
            #将key列名改为formula
            self.data = self.data.rename(columns={key:'formula'})
        elif path.endswith('.csv'):
            self.data = pd.read_csv(path,low_memory=False)[[key]].dropna()
            self.data = self.data.rename(columns={key:'formula'})

        #将self.data中的数唯一化
        self.data = self.data.drop_duplicates(subset=['formula'],keep='first')

        #重置index
        self.data = self.data.reset_index(drop=True)
        
    def filter(self,Multi_core_run_parameters):

        
        df = pd.DataFrame(columns=['formula'])
        try:
            i,mz_start,mz_end,elements_list = Multi_core_run_parameters
            formula = self.data.iloc[i]['formula']
            formula_mass = Formula(formula).isotope.mass
            formula_dict = Formula(formula).composition().dataframe().to_dict()['Count']
            formula_dict_keys = list(formula_dict.keys())
            if formula_mass >= mz_start and formula_mass <= mz_end:
            #如果formula_dict_keys中元素都在target_elements中的元素中，则将formula添加到df中
                if set(formula_dict_keys).issubset(set(elements_list)):
                    #pending
                    df = pd.concat([df,pd.DataFrame([[formula]],columns=['formula'])],ignore_index=True)
            return df
        except:
            return df
        
    def filt(self,mz_start,mz_end,elements_list):
        """
        多进程的方式map filter，默认使用全部cpu

        参数:
        min_mass: int，最小质量
        max_mass: int，最大质量
        target_elements: list，目标元素

        返回值:
        结果会直接修改self.data，无返回值
        """
        pool = Pool(4)
        dfs = pool.map(self.filter, [(i,mz_start,mz_end,elements_list) for i in range(len(self.data))])
        pool.close()
        df = pd.concat(dfs,ignore_index=True)

        self.data = df


    def create_dataset(self,type,rates,):#return_from_max_ints):
        """
        依据self.data中的formula，构建特定的数据集，不同的type对应不同的数据集，区别如下：
        基础数据集：基于数据库中真实化合物的formula。
        加噪音数据集：基于基础数据集，加入噪音。
        模拟螯合铁数据集：基于formula，模拟螯合铁后的数据。
        模拟脱氢数据集：基于formula，模拟脱氢后的数据。
        模拟加氢数据集：基于formula，模拟加氢后数据。

        参数:
        type: str，数据集类型，可选值有'base','Fe','B','Se','hydro'。
        rates: list，加氢数据集的加氢率。
        repeats: int，重复次数。

        """
        if type in ['base','Fe','B','Se','S']:
            #基础数据集
            pool = Pool(4)
            func = partial(create_data, type=type,)#return_from_max_ints=return_from_max_ints)
            dfs = pool.map(func, [formula for formula in self.data['formula']])
            pool.close()
            df = pd.concat(dfs,ignore_index=True)
            self.df_data = df
            
        elif type =='hydro':
            df = pd.DataFrame()
            #模拟加氢数据集
            pool = Pool(4)
            for rate in rates:
                func = partial(create_data, type=type,rate=rate,)#return_from_max_ints=return_from_max_ints)
                dfs = pool.map(func, [formula for formula in self.data['formula']])
                df0 = pd.concat(dfs, ignore_index=True)
                df = pd.concat([df,df0],ignore_index=True)
            pool.close()
            
            self.df_data = df
        else:
            raise ValueError('type must be in [base,Fe,B,Se,hydro]')
        
    def save(self,path):
        """
        将数据集保存为csv文件

        参数:
        path: str，保存路径
        """
        #过滤，只保留group<=7的数据
        self.df_data = self.df_data[self.df_data['group']<=7]
        self.df_data.to_csv(path,index=False)

    def work_flow(self,para,type):
        """
        创建指定数据集的工作流程.
        
        参数:
        min_mass: int，最小质量
        max_mass: int，最大质量
        target_elements: list，目标元素
        type: str，数据集类型，可选值有'base','Fe','B','Se','hydro'。
        rates: list，加氢数据集的加氢率。

        返回值:
        此过程耗时较长，故以文件的形式保存数据集。
        """
        self.filt(para.mz_start,para.mz_end,para.elements_list)
        self.create_dataset(type,para.rate_for_hydro,)#para.return_from_max_ints)
        
        #如果不存在dataset文件夹，则创建

        if not os.path.exists('./dataset'):
            os.mkdir('./dataset')
        self.save('./dataset/'+type+'.csv')
        # if para.return_from_max_ints:
        #     self.save('./dataset/'+type+'_from_max_ints.csv')
        # else:
        #     self.save('./dataset/'+type+'.csv')   

class datasets(dataset):
    """
    dataset的子类，用于合并多个dataset

    参数:
    datas: list，dataset的列表

    """
    def __init__(self,datas) -> None:
        self.data = pd.concat(datas,axis=0)
        print('原始数据',len(self.data))
        #将self.data中的数唯一化
        self.data = self.data.drop_duplicates(subset=['formula'],keep='first')
        print('去重后数据',len(self.data))
        #重置index
        self.data = self.data.reset_index(drop=True)
        
if __name__ == '__main__':
    test = dataset('F:/XinBackup/source_data/datasets/NPAtlas_download.json','mol_formula')
    test.work_flow(100,1000,['C','H','O','N','S'],'hydro')