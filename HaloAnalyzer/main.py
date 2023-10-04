#import packages

from .my_dataset.dataset_base import dataset,datasets
from .my_models.select_models import my_model
from .my_mzml.analysis import analysis_by_asari
from .parameters import run_parameters

def load_config():
    return run_parameters()
     

def pipeline_dataset():
    para = load_config()
    datas = []
    for data in para.datasets:
        datas.append(dataset(data[0],data[1]).data)
    
    data = datasets(datas)
    data.filt(para.mz_start,para.mz_end,para.elements_list)
    data.creat_classify_data(para.repeat)
    data.creat_add_Fe_data()
    data.creat_hydroisomer_data(rates=para.rates)
    data.creat_dehydroisomer_data(rates=para.rates)
    data.creat_classify_data_with_nose(para.repeat)
    data.data_statistics_customized()

#Model Pipeline
def pipeline_model():
    para = load_config()
    a = my_model('ann',
            train_batch=para.train_batch,
            val_batch=para.val_batch,
            parameters={'dense1':para.dense1,
                        'dense1_drop':para.dense1_drop,
                        'dense2':para.dense2,
                        'dense2_drop':para.dense2_drop,
                        'dense3':para.dense3,
                        'classes':para.classes,
                        'epochs':para.epochs,
                        'base_weight':para.base_weight,
                        'sub_weight':para.sub_weight,
                        'hydroisomer_weight':para.hydroisomer_weight},
            features=para.features_list,
            data_weight=para.data_weight,
            noise_data_weight=para.noise_data_weight,
            add_fe_weight=para.add_fe_weight,
            hydroisomer_weight=para.hydroisomer_weight,
            use_noise_data=para.use_noise_data,
            use_add_fe_data=para.use_add_fe_data,
            use_hydroisomer_data=para.use_hydroisomer_data,
            )

    a.model.show_CM(pre_trained=False)

#Find Halo Pipeline
def pipeline_find_halo(mzml_path):
    para = load_config()
    a = analysis_by_asari(mzml_path,para)
    a.asari_workflow()


