import tomli,tomli_w
import importlib_resources

class run_parameters:
    def __init__(self):
        file_path = importlib_resources.files('HaloAnalyzer') / 'config.toml'
        with open(file_path,'rb') as f:
            config = tomli.load(f)

        self.config = config
        #dataset参数
        self.elements_list = config['datasets']['element_list']
        self.features_list = config['datasets']['feature_list']
        self.repeat = config['datasets']['repeat']
        self.datasets = config['datasets']['paths']
        self.mz_start = config['datasets']['mz_start']
        self.mz_end = config['datasets']['mz_end']
        #model_data参数
        self.data_weight = config['model_data']['data_weight']

        self.noise_data_weight = config['model_data']['noise_data_weight']
        self.use_noise_data = config['model_data']['use_noise_data']
        self.add_fe_data_weight = config['model_data']['add_fe_data_weight']
        self.hydroisomer_data_weight = config['model_data']['hydroisomer_data_weight']
        self.use_fe_data = config['model_data']['use_add_fe_data']
        self.use_hydroisomer_data = config['model_data']['use_hydroisomer_data']
        #model_construct参数
        self.train_batch = config['model_construct']['train_batch']
        self.val_batch = config['model_construct']['val_batch']
        self.dense1 = config['model_construct']['dense1']
        self.dense1_drop = config['model_construct']['dense1_drop']
        self.dense2 = config['model_construct']['dense2']
        self.dense2_drop = config['model_construct']['dense2_drop']
        self.dense3 = config['model_construct']['dense3']
        self.base_classes = config['model_construct']['base_classes']
        self.sub_classes = config['model_construct']['sub_classes']
        self.hydro_classes = config['model_construct']['hydro_classes']
        self.epochs = config['model_construct']['epochs']
        # self.class_weight = {int(k):v for k,v in config['model_construct']['class_weight'].items()}
        self.base_weight = {int(k): v for k, v in config['model_construct_class_weight']['base_weight'].items()}
        self.sub_weight = {int(k): v for k, v in config['model_construct_class_weight']['sub_weight'].items()}
        self.hydroisomer_weight = {int(k): v for k, v in config['model_construct_class_weight']['hydroisomer_weight'].items()}
        

        #asari参数
        self.min_prominence_threshold = config['asari']['min_prominence_threshold']
        self.mz_tolerance_ppm = config['asari']['mz_tolerance_ppm']
        self.asari_min_intensity = config['asari']['min_intensity']
        self.min_timepoints = config['asari']['min_timepoints']
        self.min_peak_height = config['asari']['min_peak_height']

        #mzml参数
        self.mzml_min_intensity = config['mzml']['min_intensity']

        #vis参数
        self.vis_path = config['visualization']['path']

    def update(self,new_parameters):
        #对参数进行更新
        file_path = importlib_resources.files('HaloAnalyzer') / 'config.toml'
        with open(file_path,'wb') as f:
            tomli_w.dump(new_parameters,f)
        self.__init__()


if __name__ == '__main__':
    parameters = run_parameters()
    print((parameters.class_weight))
