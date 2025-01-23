import os
import pandas as pd


df = pd.read_csv(r'K:\open-database\NPAtlas\V2024_03\dereplicate_database_base.csv')
df = df[df['origin_type'] == 'Bacterium']