import os
import pandas as pd

fold = r'C:\Users\xyy\Desktop\python\HaloAnalyzer_training\022_six_dataset_openms_noClFe\2M_fake_molecules\result\20241218_OSMAC_V3\dereplication'
output_fold= fold.replace('dereplication', 'dereplication_with_known')
os.makedirs(output_fold, exist_ok=True)
files = os.listdir(fold)
print('Total files:', len(files))
for file in files:
    if file.endswith('feature.csv'):
        df = pd.read_csv(os.path.join(fold, file))
        df = df[df['Smiles'].notnull()]
        if len(df) == 0:
            continue
        df.to_csv(os.path.join(output_fold, file), index=False)
