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
        self.repeat_for_noise = config['datasets']['repeat_for_noise']
        self.datasets = config['datasets']['paths']
        self.mz_start = config['datasets']['mz_start']
        self.mz_end = config['datasets']['mz_end']
        self.rate_for_hydro = config['datasets']['rate_for_hydro']
        self.rate_for_hydro2 = config['datasets']['rate_for_hydro2']
        self.rate_for_hydro3 = config['datasets']['rate_for_hydro3']
        #model_data参数
        self.features_list = config['model_data']['feature_list']
        self.use_noise_data = config['model_data']['use_noise_data']
        self.use_fe_data = config['model_data']['use_add_fe_data']
        self.use_hydroisomer_data = config['model_data']['use_hydroisomer_data']
        # self.data_weight = config['model_data']['data_weight']
        # self.noise_data_weight = config['model_data']['noise_data_weight']
        # self.add_fe_data_weight = config['model_data']['add_fe_data_weight']
        # self.hydroisomer_data_weight = config['model_data']['hydroisomer_data_weight']

        #model_construct参数
        self.train_batch = config['model_construct']['train_batch']
        self.val_batch = config['model_construct']['val_batch']

        self.base_classes = config['model_construct']['base_classes']
        self.sub_classes = config['model_construct']['sub_classes']
        self.hydro_classes = config['model_construct']['hydro_classes']
        self.epochs = config['model_construct']['epochs']
        self.learning_rate = config['model_construct']['learning_rate']
        self.base_weight = {int(k): v for k, v in config['model_construct_class_weight']['base_weight'].items()}
        self.sub_weight = {int(k): v for k, v in config['model_construct_class_weight']['sub_weight'].items()}
        self.hydroisomer_weight = {int(k): v for k, v in config['model_construct_class_weight']['hydroisomer_weight'].items()}
        

        #asari参数
        self.asari_dict = {'min_prominence_threshold':config['asari']['min_prominence_threshold'],
                           'mz_tolerance_ppm':config['asari']['mz_tolerance_ppm'],
                           'min_intensity':config['asari']['min_intensity'],
                           'min_timepoints':config['asari']['min_timepoints'],
                           'min_peak_height':config['asari']['min_peak_height'],}
        #mzml参数
        self.mzml_dict = {'min_intensity':config['mzml']['min_intensity'],
                          'vendor':config['mzml']['vendor'],
                          'precursor_error':config['mzml']['precursor_error'],
                          'ROI_identify_method':config['mzml']['ROI_identify_method'],}


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
    print(type((parameters.use_fe_data)))
