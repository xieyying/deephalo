import pandas as pd
from molmass import Formula
from multiprocessing import Pool
from .methods_main import create_data
from functools import partial
import os

class dataset():
    def __init__(self,path,key) -> None:
        """
        依据数据库中真实化合物的formula，构建数据集。
        基础数据集：基于数据库中真实化合物的formula，直接构建数据集。
        加噪音数据集：基于基础数据集，加入噪音，构建数据集。
        模拟螯合铁数据集：基于formula，模拟螯合铁后的分子式，构建数据集。
        模拟脱氢数据集：基于formula，模拟脱氢后的分子式，构建数据集。
        模拟加氢数据集：基于formula，模拟加氢后的分子式，构建数据集。
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

        #定义数据集的列名
        self.datalist = [
            'formula',
            'group',
            'sub_group_type',
            'hydro_group',
            'b_2_mz',
            'b_1_mz',
            'a0_mz',
            'a1_mz',
            'a2_mz',
            'a3_mz',
            'b_2',
            'b_1',
            'a0',
            'a1',
            'a2',
            'a3',
            'a1-a0',
            'a2-a0',
            'a2-a1',
            'a0-b1',
            'b1-b2',
            'a0_norm',
            'a3-a0',
            'a3-a1',
            'a3-a2',
            'new_a0',
            'new_a1',
            'new_a2',
            'new_a3',
            'new_a0_ints',
            'new_a1_ints',
            'new_a2_ints',
            'new_a3_ints',
            'new_a2_a1',
            'new_a2_a0'
        ]
    
    #用于过滤数据，返回符合要求的数据
    def filter(self,lp):
        df = pd.DataFrame(columns=['formula'])
        try:
            i,min_mass,max_mass,target_elements = lp[0],lp[1],lp[2],lp[3]
            formula = self.data.iloc[i]['formula']
            formula_mass = Formula(formula).isotope.mass
            formula_dict = Formula(formula).composition().dataframe().to_dict()['Count']
            formula_dict_keys = list(formula_dict.keys())
            if formula_mass >= min_mass and formula_mass <= max_mass:
            #如果formula_dict_keys中元素都在target_elements中的元素中，则将formula添加到df中
                if set(formula_dict_keys).issubset(set(target_elements)):
                    #pending
                    df = pd.concat([df,pd.DataFrame([[formula]],columns=['formula'])],ignore_index=True)
            return df
        except:
            return df
        
    #多进程的方式map filter，默认使用全部cpu
    def filt(self,min_mass,max_mass,target_elements):
        
        pool = Pool()
        dfs = pool.map(self.filter, [   (i,min_mass,max_mass,target_elements) for i in self.data.index] )
        pool.close()
        df = pd.concat(dfs,ignore_index=True)

        self.data = df


    def create_dataset(self,type,rates,repeats):
        """
        依据self.data中的formula，构建数据集。
        基础数据集：基于数据库中真实化合物的formula，直接构建数据集。
        加噪音数据集：基于基础数据集，加入噪音，构建数据集。
        模拟螯合铁数据集：基于formula，模拟螯合铁后的分子式，构建数据集。
        模拟脱氢数据集：基于formula，模拟脱氢后的分子式，构建数据集。
        模拟加氢数据集：基于formula，模拟加氢后的分子式，构建数据集。
        """
        if type in ['base','Fe']:
            #基础数据集
            pool = Pool()
            func = partial(create_data, datalist=self.datalist, type=type)
            dfs = pool.map(func, [formula for formula in self.data['formula']])
            pool.close()
            df = pd.concat(dfs,ignore_index=True)
            self.df_data = df
        elif type == 'noise':
            
            #重复repeats次
            self.data = pd.concat([self.data]*repeats,ignore_index=True)
        
            #加噪音数据集
            pool = Pool()
            func = partial(create_data, datalist=self.datalist, type=type)
            dfs = pool.map(func, [formula for formula in self.data['formula']])
            pool.close()
            df = pd.concat(dfs, ignore_index=True)
            self.df_data = df

        elif type in ['hydro','hydro2','hydro3','dehydro']:
            df = pd.DataFrame()
            #模拟加氢数据集
            pool = Pool()
            for rate in rates:
                func = partial(create_data, datalist=self.datalist, type=type,rate=rate)
                dfs = pool.map(func, [formula for formula in self.data['formula']])
                df0 = pd.concat(dfs, ignore_index=True)
                df = pd.concat([df,df0],ignore_index=True)
            pool.close()
            
            self.df_data = df
        else:
            raise ValueError('type must be in [base,noise,Fe,hydro,dehydro]')
        
    def save(self,path):
        self.df_data.to_csv(path,index=False)

    def work_flow(self,min_mass,max_mass,target_elements,type,rates=[0.5,1]):
        self.filt(min_mass,max_mass,target_elements)
        self.create_dataset(type,rates,repeats=2)
        #如果不存在dataset文件夹，则创建

        if not os.path.exists('./dataset'):
            os.mkdir('./dataset')
        self.save('./dataset/'+type+'.csv')   

class datasets(dataset):
    def __init__(self,datas) -> None:
        self.data = pd.concat(datas,axis=0)
        print('原始数据',len(self.data))
        #将self.data中的数唯一化
        self.data = self.data.drop_duplicates(subset=['formula'],keep='first')
        print('去重后数据',len(self.data))
        #重置index
        self.data = self.data.reset_index(drop=True)
        self.datalist = [
            'formula',
            'group',
            'sub_group_type',
            'hydro_group',
            'b_2_mz',
            'b_1_mz',
            'a0_mz',
            'a1_mz',
            'a2_mz',
            'a3_mz',
            'b_2',
            'b_1',
            'a0',
            'a1',
            'a2',
            'a3',
            'a1-a0',
            'a2-a0',
            'a2-a1',
            'a0-b1',
            'b1-b2',
            'a0_norm',
            'a3-a0',
            'a3-a1',
            'a3-a2',
            'new_a0',
            'new_a1',
            'new_a2',
            'new_a3',
            'new_a0_ints',
            'new_a1_ints',
            'new_a2_ints',
            'new_a3_ints',
            'new_a2_a1',
            'new_a2_a0'
        ]

if __name__ == '__main__':
    test = dataset('F:/XinBackup/source_data/datasets/NPAtlas_download.json','mol_formula')
    
    test.work_flow(100,1000,['C','H','O','N','S'],'hydro')