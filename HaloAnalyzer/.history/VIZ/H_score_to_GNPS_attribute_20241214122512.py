import networkx as nx
import os
import pandas as pd

def add_h_score_mean_to_graphml(gnps_folder, deephalo_result, default_h_score_mean=0.0):
    """
    读取 GraphML 文件并为每个节点添加 'H_scoreMean' 属性。

    参数:
    gnps_folder (str): GNPS 文件夹路径。
    deephalo_result (str): DeepHalo 结果文件夹路径。
    default_h_score_mean (float, optional): 默认的 H_scoreMean 值。如果没有找到匹配的数据，将使用该值。

    返回:
    None
    """
    # 读取 GraphML 文件
    graphml_file = [file for file in os.listdir(gnps_folder) if file.endswith('.graphml')][0]
    graph = nx.read_graphml(os.path.join(gnps_folder, graphml_file))

    # 读取 DeepHalo 结果
    halo_folder = os.path.join(deephalo_result, 'halo')
    halo_files = [file for file in os.listdir(halo_folder) if file.endswith('feature.csv')]

    # 为每个节点添加新属性 'H_scoreMean'
    for node, data in graph.nodes(data=True):
        unique_file_sources = data.get('UniqueFileSources', '').split(';')
        precursor_mass = data.get('precursor mass', None)
        rt_mean = data.get('RTMean', None)

        if precursor_mass is None or rt_mean is None:
            graph.nodes[node]['H_scoreMean'] = default_h_score_mean
            continue

        file_names = [name.split('.mzML')[0].split('.mzml')[0] for name in unique_file_sources]
        # 取 file_names 与 halo_files 的交集
        halo_files_names = [file for file in halo_files if file.split('_feature.csv')[0] in file_names]
        halo_data_ = pd.DataFrame()

        for f in halo_files_names:
            halo_file = os.path.join(halo_folder, f)
            halo_data = pd.read_csv(halo_file)
            # 取出 halo_data 中 mz 与 precursor_mass 相近的数据        
            halo_data = halo_data[(halo_data['mz'] > (precursor_mass - 0.03)) & (halo_data['mz'] < (precursor_mass + 0.03))]
            # halo_data = halo_data[(halo_data['RT'] > (rt_mean - 60)) & (halo_data['RT'] < (rt_mean + 60))]
            halo_data_ = pd.concat([halo_data_, halo_data])

        if not halo_data_.empty:
            h_score_mean = halo_data_['H_score'].mean()
        else:
            h_score_mean = default_h_score_mean

        graph.nodes[node]['H_scoreMean'] = h_score_mean

    # 将修改后的图写回 GraphML 文件
    output_file = os.path.join(gnps_folder, graphml_file.replace('.graphml', '_H_score.graphml'))
    nx.write_graphml(graph, output_file)
    print(f"Modified GraphML file saved to {output_file}")

# 调用函数
gnps_folder = r'D:\workissues\manuscript\halo_mining\mining\result\OSMAC_50_6_GNPS'
deephalo_result = r'D:\workissues\manuscript\halo_mining\mining\result\OSMAC_500_6'
add_h_score_mean_to_graphml(gnps_folder, deephalo_result)