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
- pending

## Dependencies
- pending

## Installation from sourece

## Quickstart

- Create train dataset for training：halo create-dataset [project_path]

- train model：halo create-model [project_path]

- analyze mzml: halo analyze-mzml [project_path] [mzml_file_path]

- dereplication: halo dereplication [project_path]  -g [gnps_result_path] -ud [user_dataset_path] -udk [user_dataset_key_colunm]

## Getting Help

## License

## Reference




