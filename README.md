![DeepHalo Logo](./logo.png)

**A deep learning-integrated workflow for high-throughput discovery of halogenated metabolites from HRMS data.**
---

## Core Features

### 1. Halogen Prediction
- **Element Prediction Model (EPM)**
  - Dual-branch Isotope Neural Network (IsoNN) architecture
  - High accuracy Cl/Br detection (>99.9% precision based on benchmark results)
  - Wide mass range coverage (50-2000 Da)
  - Robust interference resistance to B/Se/Fe/dehydro isomers

### 2. Isotope Pattern Validation
- **Dual Validation System**
  - Mass Dimension: Statistical rule-based correction.
  - Intensity Dimension: Autoencoder-based Anomaly Detection Model (ADM).

### 3. Multi-Level Halogen Confidence Scoring (H-score)
- **Dual levels**
  - Prediction based on centroid-level isotope patterns
  - Prediction based on Scan-level isotope patterns
  - H-score integration for comprehensive assessment on the above both levels

### 3. Enhanced Dereplication
- **Dual-Strategy Approach**
  - MS1-Based Dereplication Using Custom Database Matching
    - Exact mass analysis
    - Halogen presence verification
    - Isotope intensity similarity scoring
  - MS2-Based Dereplication by Integrating GNPS
    - MS2 molecular networking
    - Halogenated compound annotation
    - GraphML file enhancement
---

##  Technical Advantages

- **High Throughput**
  - end-to-end automated analysis
  - Batch processing of unlimited LC-MS/MS datasets
  - Rapid processing (several to dozens of seconds per sample) on standard hardware (Core i9, 16GB RAM)

- **High Accuracy**
  - More than 98.3% precision in halogen detection across simulated and experimental LC-MS datasets.
  - Comprehensively validation across both simulated and experimental LC-MS datasets

- **Comprehensive Integration**
  - Input: Supports `.mzML` format
  - Output: Cytoscape-compatible network files
  - Seamless integration with GNPS molecular networking

- **Enhanced Dereplication**
  - Embeds halogen prediction results into GNPS output GraphML files
  - Significantly higher dereplicaton rate compared to molecular networking alone
---

## Target Applications
- Natural product discovery  
- Halogenated metabolite annotation
- Pharmacological Research
- Environmental Monitoring
- Water Quality Analysis

---

## Key Differentiators
1. Deep leaning-based halogen prediction resistance to Fe/dehydro isomers
2. First Isotope Pattern Validation strategies specific for halogenated molecules
3. hierarchical halogen scoring system (H-score) 
4. Comprehensive dereplication workflow
5. Enhanced GNPS molecular networking
6. Automatic and Rapid Processing

---

*For methodology details and validation datasets, see [Methods](bioRxiv 2025, 2025.08.11.669588).*  

## Where to get it？
The source code is hosted on GitHub at: https://github.com/xieyying/deephalo

Binary installers of deephalo are available at the Python Package Index (PyPI)[deephalo](https://pypi.org/project/deephalo/).

Standalone Executable (Recommended for Windows Users) are available at: [百度网盘](https://pan.baidu.com/s/1RCSnKfOwcrvMKIL7ZH4XQw?pwd=wuti)

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

### For EXE Users:
- Double click the exe to launch DeepHalo.

### For WHL Users:
**Note**  
Python 3.10 is required. Verify your Python version with:  
```bash
 python --version
```

#### Install from PyPI
```bash
pip install deephalo
``` 
#### Install from Local Wheel
```bash
pip install path/to/deephalo-xxx.whl
```

#### Install from Source for developer
```bash
git clone https://github.com/xieyying/deephalo
cd deephalo
pip install -e .
```

## Quickstart

### For EXE Users:
- Double click the exe to launch DeepHalo.

### For WHL Users:

#### High-throughput Detection of Halogenated Compounds
```bash
halo detect -i /path/to/mzml_files -o /path/to/output_directory -ms2
```
#### Dereplication
```bash
halo dereplicate -o /path/to/output_directory -g /path/to/GNPS_results -ud /path/to/custom_database.csv
```
## Full Usage Guide for WHL Users:
### Get help
```bash
halo --help                 # Show all commands
halo detect --help          # Detailed parameters for the subcommand 'detect'
halo dereplicate --help     # Detailed parameters for the subcommand 'dereplicate'
```
### Main Functions for WHL Users:

- **Analyze mzML file:**
    ```bash
    halo detect -i <input_path> -o <project_path> [-c <config_file>] [-b <blank_samples_dir>] [-ob] [-ms2]
    ```
- **Dereplication:** 
    ```bash
    halo dereplicate -o <project_path> [-g <GNPS_folder>] [-ud <user_database.csv>]
    ```
- **Create training dataset:** 
    ```bash
    halo create-ds <project_path> [-c <config_file>]
    ```
- **Train model:** 
    ```bash
    halo train <project_path> [-c <config_file>] [-m search]
    ```

If you need to modify configuration parameters, edit the config file ([download it here](https://github.com/xieyying/DeepHalo/tree/main/DeepHalo/config.toml)) and override the default settings by specifying:
```bash
-c [user_config_file]
 ```
*See documentation for more applications.*

## License
This code repository is licensed under the [MIT License](LICENSE).
