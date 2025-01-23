import os
import pandas as pd

fold = r'C:\Users\xyy\Desktop\python\HaloAnalyzer_training\022_six_dataset_openms_noClFe\2M_fake_molecules\result\halo'

files = os.listdir(fold)
print('Total files:', len(files))

file_names = [file.split('_M')[0] for file in files]
strains = set(file_names)
print('Total file names:', len(strains))