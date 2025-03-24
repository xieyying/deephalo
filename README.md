# DeepHalo

**Open-Source Pipeline for High-Confidence and High-throughput Detection of Halogenated Compounds in Complex HRMS Data**

---

## Core Features

### 1. Halogen Prediction
- **Element Prediction Model (EPM)**
  - Dual-branch Isotope Neural Network (IsoNN) architecture
  - High accuracy Cl/Br detection (>98.6% precision)
  - Wide mass range coverage (50-2000 Da)
  - Robust interference resistance to B/Se/Fe/dehydro isomers

### 2. Quality Control System
- **Dual Validation Strategy**
  - Statistical rule-based mass correction
  - Autoencoder-based Anomaly Detection Model (ADM) for intensity pattern validation
- **Multi-Level Scoring**
  - Feature-level centroid analysis
  - Scan-level validation
  - H-score integration for peak overlap/saturation error elimination

### 3. Enhanced Dereplication
- **Dual-Strategy Approach**
  - Custom Database Matching
    - Exact mass analysis
    - Halogen presence verification
    - Isotope intensity similarity scoring
  - GNPS Integration
    - MS2 molecular networking
    - Element dimension annotation
    - GraphML file enhancement
---

##  Technical Advantages

- **High Throughput**
  - Batch analysis of unlimited LC-MS/MS datasets
  - Rapid processing (<30 sec/sample) on standard hardware (Core i9, 16GB RAM)

- **High Accuracy**
  - >98.6% precision in halogen detection
  - Validated across both simulated and experimental LC-MS datasets

- **Comprehensive Integration**
  - Input: Supports `.mzML` format
  - Output: Cytoscape-compatible network files
  - Seamless integration with GNPS molecular networking

- **Enhanced Dereplication**
  - Embeds element detection results into GNPS output GraphML files
  - Significantly higher efficiency compared to molecular networking alone
  - Enables molecular network annotation in the element dimension
---

## Target Applications
- Natural product discovery  
- Halogenated metabolite annotation  

---

## Key Differentiators
1. First integrated multi-level halogen detection system  
2. Novel dual-branch IsoNN architecture
3. Comprehensive dereplication workflow
4. Enhanced GNPS molecular networking

---

*For methodology details and validation datasets, see [Methods and Benchmarks](#).*  

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


**Note**  
Python 3.10 is required. Verify your Python version with:  
```bash
 python --version
```

### Install from PyPI
```bash
pip install DeepHalo
``` 
### Install from Local Wheel
```bash
pip install path/to/DeepHalo-xxx.whl
```

### Install from Source
```bash
git clone https://github.com/xieyying/DeepHalo.git
cd DeepHalo
pip install -e .
```

## Quickstart
### High-throughput Detection of Halogenated Compounds
```bash
halo analyze-mzml -i /path/to/mzml_files -o /path/to/output_directory -ms2
```
### Dereplication
```bash
halo dereplication -o /path/to/output_directory -g /path/to/GNPS_results -ud /path/to/custom_database.csv -udk Formula
```
## Full Usage Guide
### Get help
```bash
halo --help                 # Show all commands
halo analyze-mzml --help    # Detailed parameters for a subcommand
```
###  Main Functions
- Analyze mzml file:
```bash
halo analyze-mzml -i <input_path -o <project_path> [-c <config_file>] [-b <blank_samples_dir>] [-ms2]
```
- Dereplication: 
```bash
halo dereplication -o <project_path> -g <GNPS_folder> -ud <user_database.csv> -udk <formula_column_name>
```
- Create training dataset: 
```sh
halo create-dataset [project_path]
```
- Train model: 
```sh
halo create-model [project_path]
```
If you need to modify configuration parameters, edit the config file ([download it here](https://github.com/xieyying/DeepHalo/blob/xin-back/DeepHalo/config.toml)) and then override the default settings:
```bash
-c [user_config_file]
```

## License
This code repository is licensed under the [MIT License](LICENSE).
