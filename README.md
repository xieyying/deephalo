# DeepHalo: A Deep Learning-Implanted Workflow for High-Throughput Mining of New Halogenates from Complicated Samples

> DeepHalo is a software toolkit for detecting and dereplicating compounds containing chloride or bromide atoms from microbial cultures based on high-resolution mass data.
It is designed to work with .mzML raw mass data.
A approach of judging electronic number was proposed based on matrix.
A artificial neural network was obtained by training 150 million unique real molecules
Each scan of mass spectra was evaluated for the presence of halo element to eliminate the accuracy decrease of combined mass data
A apporach of evaluting the presence of halo element from the prediction results of each scan was developed.
A dereplication process based on microbial secondary metabolites database -NPAtlas, accurate molecular weights, halo elements was integrated 

## Main Features
- pending

## Where to get it？
The source code is hosted on GitHub at: https://github.com/xieyying/DeepHalo

Binary installers of DeepHalo are available at the Python Package Index (PyPI).

## Dependencies
- pandas ==  2.0.3
- numpy ==  1.22.0     
- molmass ==  2023.8.30
- scikit-learn ==  1.3.1    
- tensorflow ==  2.10.1
- keras ==  2.10.0
- keras_tuner ==  1.4.6
- matplotlib ==  3.8.0 
- pyopenms ==  3.1.0
- scipy ==  1.11.4  
- tomli ==  2.0.1
- tomli-w ==  1.0.0
- importlib_resources == 6.4.0
- mzml2gnps == 1.0.3
- networkx == 3.4.2
- typer == 0.15.1

## Installation

> [!NOTE]
> Currently, only Python 3.10 is supported.
##### Installation from PyPi
- DeepHalo can be installed from PyPI:
```sh
pip install DeepHalo
``` 
##### Installation from sources
- In the DeepHalo directory, installing in development mode:
```sh
pip install -e .
```

## Quickstart
Print the console script usage:
```sh
halo --help
```
If you are not sure about the meaning of each parameter, you can get detailed information using the following command:

```sh
halo sub_command --help
```
###  Main Functions
- Create train dataset for training: 
```sh
halo create-dataset [project_path]
```
- Train model: 
```sh
halo create-model [project_path]
```
- Analyze mzml file:
```sh
halo analyze-mzml [project_path] [mzml_file_path]
```
- Dereplication: 
```sh
halo dereplication [project_path]  -g [gnps_result_path] -ud [user_dataset_path] -udk [user_dataset_key_colunm]
```
If you want to change the config parameters, you can first modify the [config file](DeepHalo/config.toml)([download it here](https://github.com/xieyying/DeepHalo/blob/xin-back/DeepHalo/config.toml))
you can then add the following parameters to replace the default config file：
```sh
-c [user_config_file]
```

## License
This code repository is licensed under the [MIT License](LICENSE).
