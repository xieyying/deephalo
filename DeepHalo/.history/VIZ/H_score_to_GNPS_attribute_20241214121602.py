import networkx as nx
import os
import pandas as pd

# 读取 GraphML 文件
GNPS_folder = r'D:\workissues\manuscript\halo_mining\mining\result\OSMAC_50_6_GNPS'
#以.graphml结尾的文件   
graphml_file = [file for file in os.listdir(GNPS_folder) if file.endswith('.graphml')][0]

graph = nx.read_graphml(os.path.join(GNPS_folder, graphml_file))
# 读取DeepHalo results
DeepHalo_result = r'D:\workissues\manuscript\halo_mining\mining\result\OSMAC_500_6'
halo_folder = os.path.join(DeepHalo_result, 'halo')
halo_files = [file for file in os.listdir(halo_folder) if file.endswith('feature.csv')]

# 设置默认值或条件
default_h_score_mean = None  # 或者使用其他表示缺失值的标记

# 为每个节点添加新属性 'H_scoreMean'
for node, data in graph.nodes(data=True):
    UniqueFileSources = data['UniqueFileSources'].tolist()
    precursor_mass = data['precursor mass']
    RTmean = data['RTMean']
    
    file_names = [name.split('.mzML')[0].split('.mzml')[0] for name in UniqueFileSources]
    #取file_names与halo_files的交集
    halo_files_names = [file for file in halo_files if file.split('_feature.csv')[0] in file_names]
    halo_data_ = pd.DataFrame()
    for f in halo_files_names:
        halo_file = os.path.join(halo_folder, f)
        halo_data = pd.read_csv(halo_file)
        #取出halo_data中mz与precursor_mass相近的数据        
        halo_data = halo_data[(halo_data['mz'] > (precursor_mass - 0.03))& (halo_data['mz'] < (precursor_mass + 0.03))]
        halo_data = halo_data[(halo_data['RTMean'] > (RTmean - 60))& (halo_data['RTMean'] < (RTmean + 60))]
        halo_data_ = pd.concat([halo_data_, halo_data])
    if len(halo_data_) > 0:
        H_scoreMean = halo_data_['H_score'].mean()
    else:
        H_scoreMean = default_h_score_mean
    graph.nodes[node]['H_scoreMean'] = H_scoreMean
    
# 将修改后的图写回 GraphML 文件
nx.write_graphml(graph, os.path.join(GNPS_folder, graphml_file.replace('.graphml', '_H_score.graphml')))