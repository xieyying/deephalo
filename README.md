# HaloAnalyzer

> HaloAnalyzer is a software toolkit for detecting and dereplicating compounds containing chloride or bromide atoms from microbial cultures based on high-resolution mass data.
It is designed to work with .mzML raw mass data.
A approach of judging electronic number was proposed based on matrix.
A artificial neural network was obtained by training 150 million unique real molecules
Each scan of mass spectra was evaluated for the presence of halo element to eliminate the accuracy decrease of combined mass data
A apporach of evaluting the presence of halo element from the prediction results of each scan was developed.
A dereplication process based on microbial secondary metabolites database -NPAtlas, accurate molecular weights, halo elements was integrated 

## Introduction
- HaloAnalyzer is a tool for analyzing MZML files from mass spectrometry experiments.

## Installation
- cd to the root directory of the project
- install from source with pip:
`pip install -e .`

## Usage
If installed from source, one can run `halo` as a command in a terminal, followed by a subcommand for specific tasks.

For help information:
`halo -h`

To create a new training dataset:
`halo create_dataset -o <output_dir>`

To train a new model:
`halo train_model -o <output_dir>`

To predict halo states of a new MZML file:
`halo analyze_mzml -i <input_file> -o <output_dir>`

To visualize the predicted halo states:
`halo vis_result -j <project_dir>`

## Output
- A typical run on disk may generatae a directory like this:
```bash

project
├── train_dataset
│   ├── selected_data_with_noise.csv
│   ├── selected_data.csv
│   └── dataset_statistics_customized.pkl
├── trained_models
│   ├── pick_halo_ann_wrong_data.csv
│   ├── pick_halo_ann_sum.pkl
│   ├── pick_halo_ann.png
│   └── pick_halo_ann.h5
├── test_mzml_prediction
│   ├── tic.csv
│   ├── target.csv
│   ├── features.csv
└── └── log.txt
```

## License


