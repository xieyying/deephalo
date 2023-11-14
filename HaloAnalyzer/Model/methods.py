import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

def create_dataset(features,paths,batch_size):
    features+=['base_group','sub_group','hydro_group']
    df = pd.DataFrame()
    for path in paths:
        df_ = pd.read_csv(path)
        df = pd.concat([df,df_],axis=0)

    data_v= df[features]

    train,val = train_test_split(data_v,test_size=0.2,random_state=6)

    train_target = train.pop('base_group')
    val_target = val.pop('base_group')
    train_sub_group = train.pop('sub_group')
    val_sub_group = val.pop('sub_group')
    train_hydro_group = train.pop('hydro_group')
    val_hydro_group = val.pop('hydro_group')

    X_train = train.values
    X_test = val.values
    Y_train = train_target.values
    Y_test = val_target.values
    sub_group_train = train_sub_group.values
    sub_group_test = val_sub_group.values
    hydro_group_train = train_hydro_group.values
    hydro_group_test = val_hydro_group.values
    
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, (Y_train,sub_group_train,hydro_group_train)))
    val_dataset = tf.data.Dataset.from_tensor_slices((X_test, (Y_test,sub_group_test,hydro_group_test)))
    train_dataset = train_dataset.shuffle(len(X_train)).batch(batch_size)
    val_dataset = val_dataset.shuffle(len(X_test)).batch(batch_size)
    return train_dataset,val_dataset,X_test, Y_test,sub_group_test,hydro_group_test