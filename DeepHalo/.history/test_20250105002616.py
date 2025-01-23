import pandas as pd


fold = r'D:\workissues\manuscript\halo_mining\HaloAnalyzer\SIRIUS_comparison\real_data\DCA_SIRIUS_DeepHalo_results\deephalo'
file = 'CASMI_2016_myxo_delete_Se.csv'
file = fold + '\\' + file

df = pd.read_csv(file)
mz_0 = df['mz_0']
df['mz_0'] = mz_0
p1_p0_int = df['p1_int'] / df['p0_int']
df['p1_p0_int'] = p1_p0_int
p2_p0_int = df['p2_int'] / df['p0_int']
df['p2_p0_int'] = p2_p0_int
p1_p0_p2_p0_int = df['p1_int'] / df['p0_int'] - df['p2_int'] / df['p0_int']
df['p1_p0_p2_p0_int'] = p1_p0_p2_p0_int
m1_m0 = df['mz_1'] - df['mz_0']
df['m1_m0'] = m1_m0
m2_m0 = df['mz_2'] - df['mz_0']
df['m2_m0'] = m2_m0
df.to_csv(file.replace('.csv','_test.csv'), index=False)