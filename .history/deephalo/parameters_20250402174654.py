import tomli
import importlib_resources

class RunParameters:
    """
    读取config.toml文件中的参数
    """
    def __init__(self, user_config=None):
        if user_config is not None:
            file_path = user_config
        else:
            file_path = importlib_resources.files('deephalo') / 'config.toml'
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
        self.feature_grouping = config['FeatureFinding']['feature_grouping']

        self.FeatureMapProcessor_min_num_of_masstraces = config['FeatureMapProcessor']['min_num_of_masstraces']
        self.FeatureMapProcessor_min_feature_int = config['FeatureMapProcessor']['min_feature_int']
        self.FeatureMapProcessor_min_scan_number = config['FeatureMapProcessor']['min_scan_number']
        self.FeatureMapProcessor_use_mass_difference = config['FeatureMapProcessor']['use_mass_difference']
        
        self.FeatureFilter_H_score_threshold = config['FeatureFilter']['H_score_threshold']
        self.FeatureFilter_Anomaly_detection_threshold = config['FeatureFilter']['Anomaly_detection_threshold']

        # dereplication parameters
        self.dereplication_error = config['Dereplication']['error_ppm']
        self.dereplication_Inty_cosine_score = config['Dereplication']['Inty_cosine_score']

def save_config(config_dict: dict, output_path: str) -> None:
    """Save config dictionary to TOML file with preserved formatting"""
    
    def format_value(v):
        if isinstance(v, list):
            lines = []
            for item in v:
                if isinstance(item, list) and len(item) == 2:
                    # Handle comments if present in original
                    comment = item[2] if len(item) > 2 else ''
                    lines.append(f'    [{repr(item[0])}, {repr(item[1])}],{f" # {comment}" if comment else ""}')
            return '[\n' + '\n'.join(lines) + '\n]'
        return repr(v)

    with open(output_path, 'w') as f:
        # Write header comments
        f.write('# This file is a configuration file used to set parameters for DeepHalo.\n\n')
        f.write('# Parameters for the analyze-mzml subcommand.\n\n')
        
        for section, params in config_dict.items():
            f.write(f'[{section}]\n')
            for key, value in params.items():
                formatted_value = format_value(value)
                f.write(f'{key} = {formatted_value}\n')
            f.write('\n')
            
if __name__ == '__main__':
    parameters = RunParameters()
    print((parameters.datasets))
