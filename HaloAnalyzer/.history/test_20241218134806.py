import os
import pandas as pd

fold = r'D:\workissues\manuscript\halo_mining\mining\dereplication_database\NPAtlas\NPAtlas_download_2024_03_for_dereplication.csv'
df = pd.read_csv(fold)
df =  df[df['origin_type']=='bacterium']
df.to_csv(fold.replace('.csv','_bacteria.csv'),index=False)