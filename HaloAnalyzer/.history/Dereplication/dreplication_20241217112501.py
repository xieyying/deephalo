import pandas as pd
import numpy as np

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
                #计算intensity_score
                
                dereplications = {'compound_name':[],'intensity_score':[],'error':[]} #数据库中必须包含字段compound_name
                #将self.data_中的数据写入dereplications
                for idx_,row_ in self.data_.iterrows():
                    dereplications['compound_name'].append(row_['compound_name'])
                    dereplications['intensity_score'].append(row_['intensity_score'])
                    dereplications['error_ppm'].append(abs(row_['mz']-row_['compound_m_plus_h'])/row_['mz']*1e6)
                self.Deephalo_output['dereplication'] = dereplications
    
    def intensity_score(self, inty_list1, inty_list2):
        """
        利用余弦定理计算inty_list1和inty_list2的相似度
        """
        inty_list1 = np.array(inty_list1)
        inty_list2 = np.array(inty_list2)
        
def cosine_similarity(inty_list1, inty_list2):
    """
    Calculate the cosine similarity between inty_list1 and inty_list2.

    Parameters:
    inty_list1 (list or array): First list of intensities.
    inty_list2 (list or array): Second list of intensities.

    Returns:
    float: Cosine similarity between inty_list1 and inty_list2.
    """
    # Convert lists to numpy arrays
    inty_list1 = np.array(inty_list1)
    inty_list2 = np.array(inty_list2)
    
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

# Example usage
inty_list1 = [1, 2, 3]
inty_list2 = [4, 5, 6]
similarity = cosine_similarity(inty_list1, inty_list2)
print(f"Cosine similarity: {similarity}")
        
        