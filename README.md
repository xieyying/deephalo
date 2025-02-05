# DeepHalo: High-Confidence Detection of Halogenated Compounds in Untargeted Metabolomics  
**An open-source pipeline for mining chlorine/bromine-containing natural products from complex HRMS data**

---

##  Core Innovations

### Halogen Prediction
- **Element Prediction Model (EPM)**  
  - Bimodal DNN Architecture  
  - Mass range: 50-2000 Da (wider coverage than existing tools)  
  - Detects Cl/Br with interference resistance to B/Se/Fe/dehydro isomers  

### Isotope Validation
- **Dual detection system**:  
  - Statistical rule-based filtering  
  - Autoencoder Deep Model (ADM) for anomaly detection  

### Multi-Level Scoring
- **H-score integration**:  
  - Feature centroid analysis  
  - Scan-level validation  
  - Eliminates oversaturation/peak overlap errors  

### Dereplication
- **Dual-strategy approach**:  
  1. Custom database matching:  
     - Exact mass  
     - Halogen pattern  
     - Isotope intensity similarity  
  2. MS2 networking via GNPS  
  - Improved identification efficiency  

---

##  Technical Advantages
- **Throughput**: batch analysis of LC-MS/MS datasets with rapid processing times ( <30 sec/sample) on standard laptop hardware (Core i9, 16GB RAM)   
- **Accuracy**: more than 98.6% precision in halogenated compound detection across simulated and experimental LC-MS datasets (n=4)
- **Interoperability**:  
  - Input: `.mzML` 
  - Output: Cytoscape-compatible network files  

---

## Target Applications
- Natural product discovery  
- Halogenated metabolite annotation  

---

## Key Differentiators
1. First tool integrating scan-level halogen validation  
2. Bimodal DNN Architecture for element prediction  
3. Dual dereplication combining:  
   - Custom database matching  
   - MS2 networking  

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

# DeepHalo: High-Confidence Detection of Halogenated Compounds in Untargeted Metabolomics

**An open-source pipeline for mining chlorine/bromine-containing natural products from complex HRMS data**

---

## Core Innovations

### Halogen Prediction
- **Element Prediction Model (EPM)**  
  - Bimodal DNN Architecture  
  - Mass range: 50-2000 Da (wider coverage than existing tools)  
  - Detects Cl/Br with interference resistance to B/Se/Fe/dehydro isomers  

### Isotope Validation
- **Dual detection system**:  
  - Statistical rule-based filtering  
  - Autoencoder Deep Model (ADM) for anomaly detection  

### Multi-Level Scoring
- **H-score integration**:  
  - Feature centroid analysis  
  - Scan-level validation  
  - Eliminates oversaturation/peak overlap errors  

### Dereplication
- **Dual-strategy approach**:  
  1. Custom database matching:  
     - Exact mass  
     - Halogen pattern  
     - Isotope intensity similarity  
  2. MS2 networking via GNPS  
  - Improved identification efficiency  

---

## Technical Advantages
- **Throughput**: Batch analysis of LC-MS/MS datasets with rapid processing times (<30 sec/sample) on standard laptop hardware (Core i9, 16GB RAM)  
- **Accuracy**: >98.6% precision in halogenated compound detection across simulated and experimental LC-MS datasets (n=4)  
- **Interoperability**:  
  - Input: `.mzML`  
  - Output: Cytoscape-compatible network files  

---

## Target Applications
- Natural product discovery  
- Halogenated metabolite annotation  

---

## Key Differentiators
1. First tool integrating scan-level halogen validation  
2. Bimodal DNN Architecture for element prediction  
3. Dual dereplication combining:  
   - Custom database matching  
   - MS2 networking  

---

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
- 
```bash
pip install path/to/DeepHalo-xxx.whl
```

### Install from Source
- In the DeepHalo directory, installing in development mode:
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
