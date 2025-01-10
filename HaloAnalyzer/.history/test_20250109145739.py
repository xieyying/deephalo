import pandas as pd
import os

folder = r'D:\workissues\manuscript\halo_mining\HaloAnalyzer\Simulated_LC_MS\LC_MS_information_from_papers\result_comparison\DeepHalo_analysis_results_using\result_20241213\halo_MD\comparison_result_MD_RE'

files = os.listdir(folder)
files = [file for file in files if file.endswith('deephalo.csv') or file.endswith('in_openms.csv')]
print(files)
names = [file.split('_feature_')[0] for file in files].unique()
