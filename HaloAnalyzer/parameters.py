import tomli
import importlib_resources

class RunParameters:
    """
    读取config.toml文件中的参数
    """
    def __init__(self):
        file_path = importlib_resources.files('HaloAnalyzer') / 'config.toml'
        with open(file_path,'rb') as f:
            config = tomli.load(f)

        self.config = config
        #dataset参数
        self.type_list = config['datasets']['type_list']
        self.elements_list = config['datasets']['element_list']
        self.datasets = config['datasets']['paths']
        self.mz_start = config['datasets']['mz_start']
        self.mz_end = config['datasets']['mz_end']
        self.rate_for_hydro = config['datasets']['rate_for_hydro']
        # self.return_from_max_ints = config['datasets']['return_from_max_ints']

        #model_data参数
        self.features_list = config['model_data']['feature_list']
        self.use_fe_data = config['model_data']['use_add_fe_data']
        self.use_b_data = config['model_data']['use_add_b_data']
        self.use_se_data = config['model_data']['use_add_se_data']
        self.use_hydroisomer_data = config['model_data']['use_hydroisomer_data']
        
        #model_construct参数
        self.train_batch = config['model_construct']['train_batch']
        self.val_batch = config['model_construct']['val_batch']


        self.classes = config['model_construct']['classes']
        self.epochs = config['model_construct']['epochs']
        self.learning_rate = config['model_construct']['learning_rate']
        self.classes_weight = {int(k): v for k, v in config['model_construct_class_weight']['classes_weight'].items()}
        
        #mzml参数
        self.mass_trace_detection = config['FeatureFinding']['mass_trace_detection']
        self.elution_peak_detection = config['FeatureFinding']['elution_peak_detection']
        self.feature_detection = config['FeatureFinding']['feature_detection']

        # self.FeatureMapProcessor_mz_error = config['FeatureMapProcessor']['mz_error']
        # self.FeatureMapProcessor_rt_error = config['FeatureMapProcessor']['rt_error']
        self.FeatureMapProcessor_min_num_of_masstraces = config['FeatureMapProcessor']['min_num_of_masstraces']
        self.FeatureMapProcessor_min_feature_int = config['FeatureMapProcessor']['min_feature_int']


        #vis参数
        self.vis_path = config['visualization']['path']


if __name__ == '__main__':
    parameters = RunParameters()
    print((parameters.datasets))
