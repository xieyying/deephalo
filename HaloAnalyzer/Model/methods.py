import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

def create_dataset(features,paths,batch_size):
    """创建模型训练所需的数据集，tf.data.Dataset格式"""
    features+=['group']
    df = pd.DataFrame()
    for path in paths:
        df_ = pd.read_csv(path)
        df = pd.concat([df,df_],axis=0)

    train_,val_ = train_test_split(df,test_size=0.2,random_state=6)
    train = train_[features]
    val = val_[features]

    train_target = train.pop('group')
    val_target = val.pop('group')

    X_train = train.values
    X_test = val.values

    Y_train = train_target.values
    Y_test = val_target.values
    
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    val_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
    train_dataset = train_dataset.shuffle(len(X_train)).batch(batch_size)
    val_dataset = val_dataset.shuffle(len(X_test)).batch(batch_size)
    return train_dataset,val_dataset,X_test, Y_test, val_