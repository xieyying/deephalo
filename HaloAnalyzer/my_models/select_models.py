from .ann import pick_halo_ann

class my_model:
    """
    ann_para: {dense1,dense1_drop,dense2,dense2_drop,classes}
    svm_para: {C,kernel,degree,coef0,class_weight,decision_function_shape}
    svm_para: {n_estimators,max_depth,class_weight}
    """
    def __init__(self,method:str,
                 features:list,
                 train_batch:int,
                 val_batch:int,
                 parameters:dict,
                 dataset:str=[r'./train_dataset/selected_data.csv',r'./train_dataset/selected_data_with_noise.csv',r'.\train_dataset\selected_add_Fe_data.csv',r'.\train_dataset\selected_hydroisomer_data.csv'],
                 save_to_file = True,
                 data_weight = 2,
                 noise_data_weight = 1,
                 use_noise_data='True',
                 use_add_fe_data='True',
                 use_hydroisomer_data='True') -> None:
        self.modle_type = method
        self.dataset_features = features
        self.train_batch = train_batch
        self.val_batch = val_batch        
        self.modle_para = parameters
        self.dataset = dataset
        self.save = save_to_file
        if method == 'ann':
            self.model = pick_halo_ann(features = self.dataset_features,train_batch=self.train_batch,val_batch=self.val_batch,parameters=self.modle_para,dataset=self.dataset,data_weight=data_weight,noise_data_weight=noise_data_weight,save=self.save,use_noise_data= use_noise_data,use_add_fe_data=use_add_fe_data,use_hydroisomer_data=use_hydroisomer_data)

