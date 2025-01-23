import pandas as pd
import 

class dereplication():
    def __init__(self,database,Deephalo_output) -> None:
        """
        依据Deephalo的输出结果，对数据库进行去重

        参数：
        database:str，数据库文件路径
        Deephalo_output:df，Deephalo输出结果

        属性：
        data:pd.DataFrame，数据库
        Deephalo_output:pd.DataFrame，Deephalo输出结果
        """
        self.data = pd.read_csv(database)
        self.Deephalo_output = Deephalo_output
        
    def dereplication(self):
        """
        利用数据库进行去重
        """
        #获得与deephalo结果中的mz相近的数据库中的化合物
        self.data['mz'] = self.data['compound_m_plus_h']
        self.Deephalo_output['mz'] = self.Deephalo_output['mz']
        self.data_ = self.data[abs(self.data['mz']-self.Deephalo_output['mz'])<=self.Deephalo_output['mz']*1e-5]
        
        for idx,row in self.Deephalo_output.iterrows():
            #查找与Deephalo_output中的化合物相近的数据库中的化合物
            self.data_ = self.data[(abs(self.data['mz'] - row['mz']) <= row['mz'] * 1e-5) & (self.data['formula'].str.contains('Br|Cl'))]
            if self.data_.empty:
                self.Deephalo_output['dereplication'] = 'None'
            else:
                dereplications = {'compound_name':[],'intensity_score':[],'error':[]} #数据库中必须包含字段compound_name
                #将self.data_中的数据写入dereplications
                for idx_,row_ in self.data_.iterrows():
                    dereplications['compound_name'].append(row_['compound_name'])
                    dereplications['intensity_score'].append(row_['intensity_score'])
                    dereplications['error_ppm'].append(abs(row_['mz']-row_['compound_m_plus_h'])/row_['mz']*1e6)
                self.Deephalo_output['dereplication'] = self.data_['formula'].values[0]
 
        
        