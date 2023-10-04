import pandas as pd
from .dataset_methods import Isotope_simulation,formula_groups_clf,isos_calc,dataset_statistics_save,adding_noise_to_intensity,adding_noise_to_mass
from .dataset_methods import mass_spectrum_calc,formula_trainable_clf,formula_base_clf,formula_element_clf,other_requirements_trainable_clf
from .dataset_methods import mass_spectrum_calc_2
from molmass import Formula
from multiprocessing import Pool
import time,os
class dataset():
    def __init__(self,path,key) -> None:
        if path.endswith('.json'):
            self.data = pd.read_json(path)[[key]].dropna()
            #将key列名改为formula
            self.data = self.data.rename(columns={key:'formula'})
        elif path.endswith('.csv'):
            self.data = pd.read_csv(path,low_memory=False)[[key]].dropna()
            self.data = self.data.rename(columns={key:'formula'})

        # print('原始数据',len(self.data))
        #将self.data中的数唯一化
        self.data = self.data.drop_duplicates(subset=['formula'],keep='first')
        # print('去重后数据',len(self.data))
        #重置index
        self.data = self.data.reset_index(drop=True)
        self.datalist = ['formula','group','sub_group_type',
                         'b_2_mz','b_1_mz','a0_mz','a1_mz','a2_mz','a3_mz',
                         'b_2','b_1','a0','a1','a2','a3',
                         'a1-a0','a2-a0','a2-a1','a0-b1','b1-b2',
                         'a0_norm','a3-a0','a3-a1','a3-a2',
                         'new_a0','new_a1','new_a2','new_a3',
                         'new_a0_ints','new_a1_ints','new_a2_ints','new_a3_ints',
                         'new_a2_a1','new_a2_a0'
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

    #通过分子式模拟质谱数据
    def simulation_data(self,para):
        i = para[0]
        formula = self.data.iloc[i]['formula']
        #根据formula判断训练标签
        is_train,group,sub_group =formula_groups_clf(formula)
        if is_train == 0:
            return  pd.DataFrame(columns=self.datalist)
        #模拟质谱数据
        b_2_mz,b_1_mz,a0_mz,a1_mz,a2_mz,a3_mz,b_2,b_1,a0,a1,a2,a3 = Isotope_simulation(formula)

        #计算质谱数据中的差值   
        a0_norm,a1_a0,a2_a0,a2_a1,a3_a0,a3_a1,a3_a2,a0_b1,b1_b2=mass_spectrum_calc(b_2_mz,b_1_mz,a0_mz,a1_mz,a2_mz,a3_mz,b_2,b_1,a0,a1,a2,a3)
        new_a0_mz,new_a1_mz,new_a2_mz,new_a3_mz,new_a0_ints,new_a1_ints,new_a2_ints,new_a3_ints,new_a2_a1,new_a2_a0 = mass_spectrum_calc_2(b_2_mz,b_1_mz,a0_mz,a1_mz,a2_mz,a3_mz,b_2,b_1,a0,a1,a2,a3)
        #pandas.concat instead of append
        df = pd.DataFrame(
                            [[formula,group,sub_group,
                            b_2_mz,b_1_mz,a0_mz,a1_mz,a2_mz,a3_mz,
                            b_2,b_1,a0,a1,a2,a3,
                            a1_a0,a2_a0,a2_a1,a0_b1,b1_b2,
                            a0_norm,a3_a0,a3_a1,a3_a2,
                            new_a0_mz,new_a1_mz,new_a2_mz,new_a3_mz,
                            new_a0_ints,new_a1_ints,new_a2_ints,new_a3_ints,
                            new_a2_a1,new_a2_a0]],
                            columns=self.datalist
                            )
    
        return df
    
    #创建基础训练数据集，使用多进程
    def creat_classify_data(self,repeat=10):
        #开始计时
        time_start = time.time()
        #创建进程池
        pool = Pool()
        #使用多进程的方式map simulation_data
        result = pool.map(self.simulation_data,[(i,repeat) for i in self.data.index])
        #关闭进程池
        pool.close()
        #结束计时
        time_end = time.time()

        print('creat_classify_data totally cost',time_end-time_start)
        #将result中所有df转为一个DataFrame
        df = pd.concat(result,ignore_index=True)
        #过滤掉group为none的数据
        df = df[df['group'].notnull()]
        #重置index
        df = df.reset_index(drop=True)


        #保存到csv文件
        if not os.path.exists(r'./train_dataset'):
            os.makedirs(r'./train_dataset')

        df.to_csv(r'./train_dataset/selected_data.csv',index=False)
        print('训练数据(不加噪音)', len(df))

    #为模拟的质谱添加噪音信号
    def add_noise_to_data(self,para):
        #para[0]为index，para[1]为重复次数
        i = para[0]
        repeat_times = para[1]
        formula = self.data.iloc[i]['formula']
        #根据formula判断训练标签
        is_train,group,sub_group =formula_groups_clf(formula)
        if is_train == 0:
            return  pd.DataFrame(columns=self.datalist)
        
        #用一行代码创建一个空的DataFrame
        df = pd.DataFrame(columns=self.datalist)
        for j in range(repeat_times):
            #模拟质谱数据
            b_2_mz,b_1_mz,a0_mz,a1_mz,a2_mz,a_3_mz,b_2,b_1,a0,a1,a2,a3 = Isotope_simulation(formula)
            #为质谱数据添加噪音
            #mz
            b_2_mz, b_1_mz, a0_mz, a1_mz, a2_mz, a3_mz = map(adding_noise_to_mass, [b_2_mz, b_1_mz, a0_mz, a1_mz, a2_mz, a_3_mz])
            #intensity
            b_2, b_1, a0, a1, a2, a3 = map(adding_noise_to_intensity, [b_2, b_1, a0, a1, a2, a3])  
            #计算质谱数据中的差值              
            a0_norm,a1_a0,a2_a0,a2_a1,a3_a0,a3_a1,a3_a2,a0_b1,b1_b2=mass_spectrum_calc(b_2_mz,b_1_mz,a0_mz,a1_mz,a2_mz,a3_mz,b_2,b_1,a0,a1,a2,a3)
            new_a0_mz,new_a1_mz,new_a2_mz,new_a3_mz,new_a0_ints,new_a1_ints,new_a2_ints,new_a3_ints,new_a2_a1,new_a2_a0 = mass_spectrum_calc_2(b_2_mz,b_1_mz,a0_mz,a1_mz,a2_mz,a_3_mz,b_2,b_1,a0,a1,a2,a3)

            #将结果合并为单个dataframe
            df = pd.concat([df,pd.DataFrame(
                            [[formula,group,sub_group,
                            b_2_mz,b_1_mz,a0_mz,a1_mz,a2_mz,a3_mz,
                            b_2,b_1,a0,a1,a2,a3,
                            a1_a0,a2_a0,a2_a1,a0_b1,b1_b2,
                            a0_norm,a3_a0,a3_a1,a3_a2,
                            new_a0_mz,new_a1_mz,new_a2_mz,new_a3_mz,
                            new_a0_ints,new_a1_ints,new_a2_ints,new_a3_ints,
                            new_a2_a1,new_a2_a0]],
                            columns=self.datalist
                            )],ignore_index=True)
        return df
    
    #创建基础训练数据集，并用add_noise_to_data为数据添加噪音，repeat为重复次数，使用多进程
    def creat_classify_data_with_nose(self,repeat=10):
        time_start = time.time()
        pool = Pool()
        result = pool.map(self.add_noise_to_data,[(i,repeat) for i in self.data.index])
        pool.close()
        time_end = time.time()
        print('creat_classify_data_with_nose totally cost',time_end-time_start)
        #将result中所有df转为一个DataFrame
        df = pd.concat(result,ignore_index=True)
        #过滤掉group为none的数据
        df = df[df['group'].notnull()]
        #重置index
        df = df.reset_index(drop=True)


        #保存到csv文件
        if not os.path.exists(r'./train_dataset'):
            os.makedirs(r'./train_dataset')

        df.to_csv(r'./train_dataset/selected_data_with_noise.csv',index=False)
        print('训练数据（加噪音）', len(df))

    #模拟+Fe的质谱数据
    def simulation_add_Fe_data(self,para):
        i = para[0]
        formula = self.data.iloc[i]['formula']
        #根据formula判断训练标签
        is_train,group,sub_group =formula_groups_clf(formula)
        if is_train == 0:
            return  pd.DataFrame(columns=self.datalist)
        is_iron_additive_trainable = other_requirements_trainable_clf(formula)
        if is_iron_additive_trainable == 0:
            return  pd.DataFrame(columns=self.datalist)
        group = 1
        #模拟质谱数据
        b_2_mz,b_1_mz,a0_mz,a1_mz,a2_mz,a3_mz,b_2,b_1,a0,a1,a2,a3 = Isotope_simulation(formula,type='Fe')
        #计算质谱数据中的差值
        a0_norm,a1_a0,a2_a0,a2_a1,a3_a0,a3_a1,a3_a2,a0_b1,b1_b2=mass_spectrum_calc(b_2_mz,b_1_mz,a0_mz,a1_mz,a2_mz,a3_mz,b_2,b_1,a0,a1,a2,a3)
        new_a0_mz,new_a1_mz,new_a2_mz,new_a3_mz,new_a0_ints,new_a1_ints,new_a2_ints,new_a3_ints,new_a2_a1,new_a2_a0 = mass_spectrum_calc_2(b_2_mz,b_1_mz,a0_mz,a1_mz,a2_mz,a3_mz,b_2,b_1,a0,a1,a2,a3)
        #pandas.concat instead of append
        df = pd.DataFrame(
                            [[formula,group,sub_group,
                            b_2_mz,b_1_mz,a0_mz,a1_mz,a2_mz,a3_mz,
                            b_2,b_1,a0,a1,a2,a3,
                            a1_a0,a2_a0,a2_a1,a0_b1,b1_b2,
                            a0_norm,a3_a0,a3_a1,a3_a2,
                            new_a0_mz,new_a1_mz,new_a2_mz,new_a3_mz,
                            new_a0_ints,new_a1_ints,new_a2_ints,new_a3_ints,
                            new_a2_a1,new_a2_a0]],
                            columns=self.datalist
                            )

        return df
    
    #创建+Fe训练数据集，使用多进程
    def creat_add_Fe_data(self,repeats=10):
        #开始计时
        time_start = time.time()
        #创建进程池
        pool = Pool()
        #使用多进程的方式map simulation_data
        result = pool.map(self.simulation_add_Fe_data,[(i,repeats) for i in self.data.index])
        #关闭进程池
        pool.close()
        #结束计时
        time_end = time.time()

        print('creat_add_Fe_data totally cost',time_end-time_start)
        #将result中所有df转为一个DataFrame
        df = pd.concat(result,ignore_index=True)
        #过滤掉group为none的数据
        df = df[df['group'].notnull()]
        #重置index
        df = df.reset_index(drop=True)


        #保存到csv文件
        if not os.path.exists(r'./train_dataset'):
            os.makedirs(r'./train_dataset')

        df.to_csv(r'./train_dataset/selected_add_Fe_data.csv',index=False)
        print('训练数据(+Fe)', len(df))

    #通过分子式模拟hydroisomer质谱数据
    def simulation_hydroisomer_data(self,para):
        i = para[0]
        formula = self.data.iloc[i]['formula']
        rates = para[1]
        #根据formula判断训练标签
        #模拟dehydroisomer质谱数据
        is_train,group,sub_group =formula_groups_clf(formula,optional_param='hydro')
        if is_train == 0:
            return  pd.DataFrame(columns=self.datalist)
        other_requirements_trainable = other_requirements_trainable_clf(formula)
        if other_requirements_trainable == 0:
            return  pd.DataFrame(columns=self.datalist)
        #用一行代码创建一个空的DataFrame
        df = pd.DataFrame(columns=self.datalist)
        for rate in rates:
            b_2_mz,b_1_mz,a0_mz,a1_mz,a2_mz,a3_mz,b_2,b_1,a0,a1,a2,a3 = Isotope_simulation(formula,type='hydro',rate=rate)
            #计算质谱数据中的差值
            a0_norm,a1_a0,a2_a0,a2_a1,a3_a0,a3_a1,a3_a2,a0_b1,b1_b2=mass_spectrum_calc(b_2_mz,b_1_mz,a0_mz,a1_mz,a2_mz,a3_mz,b_2,b_1,a0,a1,a2,a3)
            new_a0_mz,new_a1_mz,new_a2_mz,new_a3_mz,new_a0_ints,new_a1_ints,new_a2_ints,new_a3_ints,new_a2_a1,new_a2_a0 = mass_spectrum_calc_2(b_2_mz,b_1_mz,a0_mz,a1_mz,a2_mz,a3_mz,b_2,b_1,a0,a1,a2,a3)
            #pandas.concat instead of append
            df0 = pd.DataFrame(
                                [[formula,group,sub_group,
                                b_2_mz,b_1_mz,a0_mz,a1_mz,a2_mz,a3_mz,
                                b_2,b_1,a0,a1,a2,a3,
                                a1_a0,a2_a0,a2_a1,a0_b1,b1_b2,
                                a0_norm,a3_a0,a3_a1,a3_a2,
                                new_a0_mz,new_a1_mz,new_a2_mz,new_a3_mz,
                                new_a0_ints,new_a1_ints,new_a2_ints,new_a3_ints,
                                new_a2_a1,new_a2_a0]],
                                columns=self.datalist
                                )
            df = pd.concat([df,df0],ignore_index=True)
        return df
    #通过分子式模拟dehydroisomer质谱数据
    def simulation_dehydroisomer_data(self,para):
        i = para[0]
        formula = self.data.iloc[i]['formula']
        rates = para[1]
        df = pd.DataFrame(columns=self.datalist)
        #根据formula判断训练标签
        #模拟dehydroisomer质谱数据
        is_train,group,sub_group =formula_groups_clf(formula,optional_param='dehydro')
        if is_train == 0:
            return  pd.DataFrame(columns=self.datalist)
        other_requirements_trainable = other_requirements_trainable_clf(formula)
        if other_requirements_trainable == 0:
            return  pd.DataFrame(columns=self.datalist)
        for rate in rates:
            b_2_mz,b_1_mz,a0_mz,a1_mz,a2_mz,a3_mz,b_2,b_1,a0,a1,a2,a3 = Isotope_simulation(formula,rate=rate,type='dehydro')
            #计算质谱数据中的差值
            a0_norm,a1_a0,a2_a0,a2_a1,a3_a0,a3_a1,a3_a2,a0_b1,b1_b2=mass_spectrum_calc(b_2_mz,b_1_mz,a0_mz,a1_mz,a2_mz,a3_mz,b_2,b_1,a0,a1,a2,a3)
            new_a0_mz,new_a1_mz,new_a2_mz,new_a3_mz,new_a0_ints,new_a1_ints,new_a2_ints,new_a3_ints,new_a2_a1,new_a2_a0 = mass_spectrum_calc_2(b_2_mz,b_1_mz,a0_mz,a1_mz,a2_mz,a3_mz,b_2,b_1,a0,a1,a2,a3)
            #pandas.concat instead of append
            df0 = pd.DataFrame(
                                [[formula,group,sub_group,
                                b_2_mz,b_1_mz,a0_mz,a1_mz,a2_mz,a3_mz,
                                b_2,b_1,a0,a1,a2,a3,
                                a1_a0,a2_a0,a2_a1,a0_b1,b1_b2,
                                a0_norm,a3_a0,a3_a1,a3_a2,
                                new_a0_mz,new_a1_mz,new_a2_mz,new_a3_mz,
                                new_a0_ints,new_a1_ints,new_a2_ints,new_a3_ints,
                                new_a2_a1,new_a2_a0]],
                                columns=self.datalist
                                )
            df = pd.concat([df,df0],ignore_index=True)
        return df
    
    #创建hydroisomer训练数据集，使用多进程
    def creat_hydroisomer_data(self,rates=[0.2, 0.4, 0.6, 1.0, 1/0.6, 1/0.4, 1/0.2]):
        #开始计时
        time_start = time.time()
        #创建进程池
        pool = Pool()
        #使用多进程的方式map simulation_data
        result = pool.map(self.simulation_hydroisomer_data,[(i,rates) for i in self.data.index])
        #关闭进程池
        pool.close()
        #结束计时
        time_end = time.time()

        print('creat_hydroisomer_data totally cost',time_end-time_start)
        #将result中所有df转为一个DataFrame
        df = pd.concat(result,ignore_index=True)
        #过滤掉group为none的数据
        df = df[df['group'].notnull()]
        #重置index
        df = df.reset_index(drop=True)


        #保存到csv文件
        if not os.path.exists(r'./train_dataset'):
            os.makedirs(r'./train_dataset')

        df.to_csv(r'./train_dataset/selected_hydroisomer_data.csv',index=False)
        print('训练数据(hydroisomer)', len(df))
    
    #创建dehydroisomer训练数据集，使用多进程
    def creat_dehydroisomer_data(self,rates=[0.2, 0.4, 0.6, 1.0, 1/0.6, 1/0.4, 1/0.2]):
        #开始计时
        time_start = time.time()
        #创建进程池
        pool = Pool()
        #使用多进程的方式map simulation_data
        result = pool.map(self.simulation_dehydroisomer_data,[(i,rates) for i in self.data.index])
        #关闭进程池
        pool.close()
        #结束计时
        time_end = time.time()

        print('creat_dehydroisomer_data totally cost',time_end-time_start)
        #将result中所有df转为一个DataFrame
        df = pd.concat(result,ignore_index=True)
        #过滤掉group为none的数据
        df = df[df['group'].notnull()]
        #重置index
        df = df.reset_index(drop=True)


        #保存到csv文件
        if not os.path.exists(r'./train_dataset'):
            os.makedirs(r'./train_dataset')

        df.to_csv(r'./train_dataset/selected_dehydroisomer_data.csv',index=False)
        print('训练数据(dehydroisomer)', len(df))

    #统计分子式的基本信息
    def data_statistics(self,i):
        formula = self.data.iloc[i]['formula']
        #计算formula的单一质量
        formula_mass = Formula(formula).isotope.mass

        #将formula列中的公式转为字典
        formula_dict = Formula(formula).composition().dataframe().to_dict()['Count']
        formula_dict_keys = list(formula_dict.keys())
        #判断数据是否可以用来训练
        is_train = formula_trainable_clf(formula_dict)
        #依据字典中的元素种类和个数，判断group的类型
        group = formula_base_clf(formula_dict)
        #依据字典中的元素是否含有Fe金属元素，将结果存入self.data中
        sub_group = formula_element_clf(formula_dict,'Fe')
        #依据f的元素是否含有电荷，将结果存入self.data中
        charge = Formula(formula).charge
        #
        m0_mz,m1_m0,m2_m1 = isos_calc(formula)
        return {'formula':formula,'formula_mass':formula_mass,'formula_dict': formula_dict,'formula_dict_keys':formula_dict_keys,'is_train':is_train,'group':group,'sub_group':sub_group,'charge':charge,'m0_mz':m0_mz,'m1_m0':m1_m0,'m2_m1':m2_m1}

    #统计数据库中所有数据的基本信息，用多进程加速，并保存到pickle文件中
    def data_statistics_customized(self):
        self.keys = []
        data_len = len(self.data)
        pool = Pool()
        result = pool.map(self.data_statistics,range(data_len))
        pool.close()
        #将result转为DataFrame
        data = pd.DataFrame(result)

        dataset_statistics_save(data)

class datasets(dataset):
    def __init__(self,datas) -> None:
        self.data = pd.concat(datas,axis=0)
        print('原始数据',len(self.data))
        #将self.data中的数唯一化
        self.data = self.data.drop_duplicates(subset=['formula'],keep='first')
        print('去重后数据',len(self.data))
        #重置index
        self.data = self.data.reset_index(drop=True)
        self.datalist = ['formula','group','sub_group_type',
                         'b_2_mz','b_1_mz','a0_mz','a1_mz','a2_mz','a3_mz',
                         'b_2','b_1','a0','a1','a2','a3',
                         'a1-a0','a2-a0','a2-a1','a0-b1','b1-b2',
                         'a0_norm','a3-a0','a3-a1','a3-a2',
                         'new_a0','new_a1','new_a2','new_a3',
                         'new_a0_ints','new_a1_ints','new_a2_ints','new_a3_ints',
                         'new_a2_a1','new_a2_a0'
                         ]
if __name__ == '__main__':
    a = dataset(r'source_data/datasets/NPAtlas_download.json','mol_formula')
    # a.data_statistics_customized()
    a.creat_classify_data()
    a.creat_classify_data_with_nose(repeat=1)



