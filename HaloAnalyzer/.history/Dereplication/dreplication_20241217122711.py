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
        self.data = pd.read_csv(database,low_memory=False,encoding='utf-8-sig')
        self.Deephalo_output = Deephalo_output
        
    def dereplication(self):
        """
        利用数据库进行去重
        """
        #获得与deephalo结果中的mz相近的数据库中的化合物
        self.data['mz'] = self.data['compound_m_plus_h']
        
        for idx,row in self.Deephalo_output.iterrows():
            #查找与Deephalo_output中的化合物相近的数据库中的化合物
            self.data_ = self.data[(abs(self.data['mz'] - row['mz']) <= row['mz'] * 1e-5) & (self.data['formula'].str.contains('Br|Cl'))]
            dereplications = {'compound_names':[],'intensity_score':[],'error_ppm':[]}

            if self.data_.empty:
                self.Deephalo_output.loc[idx,'compound_names'] = 'None'
                self.Deephalo_output.loc[idx,'intensity_score'] = 0
                self.Deephalo_output.loc[idx,'error_ppm'] = 1e6
                self.Deephalo_output.loc[idx,'dereplications'] = str(dereplications)
            else:
                #计算intensity_score

                #数据库中必须包含字段compound_names
                #将self.data_中的数据写入dereplications
                for idx_,row_ in self.data_.iterrows():
                    dereplications['compound_names'].append(row_['compound_names'])
                    row['inty_list'] = row[['p0_int','p1_int','p2_int','p3_int','p4_int']].tolist()
                    row_['inty_list'] = row_[['p0_int','p1_int','p2_int','p3_int','p4_int']].tolist()
                    row_['intensity_score'] = cosine_similarity(row['inty_list'],row_['inty_list'])   
                    dereplications['intensity_score'].append(row_['intensity_score'])
                    dereplications['error_ppm'].append(abs(row_['mz']-row_['compound_m_plus_h'])/row_['mz']*1e6)
                self.Deephalo_output['dereplication'] = str(dereplications)
                dereplications = pd.DataFrame(dereplications)
                self.Deephalo_output.loc[idx,'compound_names'] = dereplications.loc[dereplications['intensity_score'].idxmax(),'compound_names']
                self.Deephalo_output.loc[idx,'intensity_score'] = dereplications.loc[dereplications['intensity_score'].idxmax(),'intensity_score']
                self.Deephalo_output.loc[idx,'error_ppm'] = dereplications.loc[dereplications['intensity_score'].idxmax(),'error_ppm']
                
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

    # inty_list2 = [1,0.215312253,0.3,0.003639219,0.000333976]
    # inty_list1 = [1,0.215312253,0,0,0]
    # similarity = cosine_similarity(inty_list1, inty_list2)
    # print(f"Cosine similarity: {similarity}")
    
    database = r'K:\open-database\NPAtlas\V2024_03\dereplicate_database_base.csv'
    Deephalo_output = pd.read_csv(r'C:\Users\xyy\Desktop\python\HaloAnalyzer_training\022_six_dataset_openms_noClFe\2M_fake_molecules\result\halo\Strepomyces_A30_M5_cmx_p16_D5_nr1_feature.csv')
    dereplication = dereplication(database,Deephalo_output)
    df = dereplication.dereplication()
    print(df)
    df.to_csv(r'C:\Users\xyy\Desktop\python\HaloAnalyzer_training\022_six_dataset_openms_noClFe\2M_fake_molecules\result\halo\Strepomyces_A30_M5_cmx_p16_D5_nr1_feature_dereplication.csv',index=False)
        

        