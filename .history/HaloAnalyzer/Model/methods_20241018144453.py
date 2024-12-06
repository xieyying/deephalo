import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np

def create_dataset(features,paths,batch_size,model='base'):
    """
    创建模型训练所需的数据集，tf.data.Dataset格式
    
    Args:
    features: list, 特征名称
    paths: list, 数据集路径
    batch_size: int, batch大小
    
    Returns:
    train_dataset: tf.data.Dataset, 训练数据集
    val_dataset: tf.data.Dataset, 验证数据集
    X_test: np.array, 验证集特征
    Y_test: np.array, 验证集标签
    val_: pd.DataFrame, 验证集
    """
    features+=['group']
    df = pd.DataFrame()
    for path in paths:
        df_ = pd.read_csv(path)
        df = pd.concat([df,df_],axis=0)
    
    df = df[df['mz_0'] <= 2000]
    # 计算每个类别的样本数量,保留此部分
    print('Total_data: ',len(df))
    class_counts = np.bincount(df['group'].values) 
    print('Whole_data_class_counts: ',class_counts)

    train_,val_ = train_test_split(df,test_size=0.2,random_state=6)
    #save data
    # train_.to_csv('./train_data.csv',index=False)
    # val_.to_csv('./validation_data.csv',index=False)
    
    train = train_[features]
    val = val_[features]
    train_target = train.pop('group')
    val_target = val.pop('group')

    X_train = train.values
    X_test = val.values

    Y_train = train_target.values
    Y_test = val_target.values
    

    # # 计算类别权重
    # class_weights = 1 / class_counts 
    # print('class_weights: ',class_weights)

    # 计算root CSW权重
    root_csw_weights = 1 / np.sqrt(class_counts)
    print('root_csw_weights: ',root_csw_weights)

    # 计算square CSW权重
    # square_csw_weights = 1 / (class_counts ** 2)
    # print('square_csw_weight',square_csw_weights)
        
    
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    val_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
    train_dataset = train_dataset.shuffle(len(X_train)).batch(batch_size)
    val_dataset = val_dataset.shuffle(len(X_test)).batch(batch_size)
    
    if model == 'base':
        return train_dataset,val_dataset,X_test, Y_test, val_
    elif model == 'search':
        return X_train,Y_train,X_test, Y_test
    
    
    

