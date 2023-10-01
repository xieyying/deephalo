import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

class base:
    def __init__(self,features,train_batch,val_batch,parameters,dataset,
                 data_weight = 2,
                 noise_data_weight = 1,
                 add_fe_weight = 1,
                 hydroisomer_weight = 1,
                 save = True,use_noise_data='True',use_add_fe_data='True',use_hydroisomer_data='True'):
        self.use_noise_data = use_noise_data
        self.use_add_fe_data = use_add_fe_data
        self.use_hydroisomer_data = use_hydroisomer_data
        features_data = pd.read_csv(dataset[0])
        features_data['weight'] = data_weight
        features_data_with_noise = pd.read_csv(dataset[1])
        features_data_with_noise['weight'] = noise_data_weight
        features_data_add_Fe = pd.read_csv(dataset[2])
        features_data_add_Fe['weight'] = add_fe_weight
        features_data_hydroisomer = pd.read_csv(dataset[3])
        features_data_hydroisomer['weight'] = hydroisomer_weight
        self.features_data = features_data
        if use_noise_data == 'True':
            self.features_data = pd.concat([self.features_data,features_data_with_noise],axis=0)
        if use_add_fe_data == 'True':
            self.features_data = pd.concat([self.features_data,features_data_add_Fe],axis=0)
        if use_hydroisomer_data == 'True':
            self.features_data = pd.concat([self.features_data,features_data_hydroisomer],axis=0)


        self.train_batch = train_batch
        self.val_batch = val_batch
        self.parameters = parameters
        self.save = save
        self.features = features.copy()

        features.append('group')
        features.append('weight')
        features.append('formula')
        features.append('sub_group_type')
        features.append('hydro_group')
        
        data_v= self.features_data[features]
        
        train,val = train_test_split(data_v,test_size=0.2,random_state=26)
        self.val = val
        train_target = train.pop('group')
        val_target = val.pop('group')
        train_sub_group = train.pop('sub_group_type')
        val_sub_group = val.pop('sub_group_type')


        train_hydro_group = train.pop('hydro_group')
        val_hydro_group = val.pop('hydro_group')
        train_weight = train.pop('weight')
        val_weight = val.pop('weight')
        train_formula = train.pop('formula')
        val_formula = val.pop('formula')
        self.X_train = train.values
        self.X_test = val.values
        self.X_train_sub_group = train_sub_group.values
        self.X_test_sub_group = val_sub_group.values
        self.X_train_hydro_group = train_hydro_group.values
        self.X_test_hydro_group = val_hydro_group.values
        self.X_train_weight = train_weight.values
        self.X_test_weight = val_weight.values
        self.Y_train = train_target.values
        self.Y_test = val_target.values
        self.formula_train = train_formula.values
        self.formula_test = val_formula.values
        self.clf = None