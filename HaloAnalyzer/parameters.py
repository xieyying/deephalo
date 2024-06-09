import tomli,tomli_w
import importlib_resources

class run_parameters:
    """
    读取config.toml文件中的参数
    """
    def __init__(self):
        file_path = importlib_resources.files('HaloAnalyzer') / 'config.toml'
        with open(file_path,'rb') as f:
            config = tomli.load(f)

        self.config = config
        #dataset参数
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
        self.mzml_dict = {'min_intensity':config['mzml']['min_intensity'],
                          'vendor':config['mzml']['vendor'],
                          'precursor_error':config['mzml']['precursor_error'],
                          'ROI_identify_method':config['mzml']['ROI_identify_method'],
                          'min_element_roi':config['mzml']['min_element_roi'],
                          'min_element_sum':config['mzml']['min_element_sum'],
                          }


        #vis参数
        self.vis_path = config['visualization']['path']


if __name__ == '__main__':
    parameters = run_parameters()
    print((parameters.datasets))
