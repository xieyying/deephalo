import numpy as np
import networkx as nx
import os
import pandas as pd
import ast

def cosine_similarity(inty_list1, inty_list2):
    """
    Calculate the cosine similarity between inty_list1 and inty_list2.

    Parameters:
    inty_list1 (list or array): First list of intensities.
    inty_list2 (list or array): Second list of intensities.

    Returns:
    float: Cosine similarity between inty_list1 and inty_list2.
    """
    # Convert lists to numpy arrays of float type
    inty_list1 = np.array(inty_list1, dtype=float)
    inty_list2 = np.array(inty_list2, dtype=float)

    # Calculate the dot product
    dot_product = np.dot(inty_list1, inty_list2)
    
    # Calculate the norms (magnitudes) of the vectors
    norm1 = np.linalg.norm(inty_list1)
    norm2 = np.linalg.norm(inty_list2)
    
    # Calculate the cosine similarity
    if norm1 == 0 or norm2 == 0:
        return 0.0  # Avoid division by zero
    cosine_similarity = dot_product / (norm1 * norm2)
    
    return cosine_similarity

def combine_columns( row, columns):
    for col in columns:
        if row[col] != 'None' and row[col] != 1e6:
            return row[col]
    return 'None' if 'error_ppm' not in columns else 1e6


def add_deephalo_results_to_graphml(gnps_folder, deephalo_result_dereplication_folder):
    """
    Read the GraphML file and add the 'H_scoreMean' attribute to each node.

    Parameters:
    gnps_folder (str): Path to the GNPS folder.
    deephalo_result (str): Path to the DeepHalo results folder.
    default_h_score_mean (float, optional): Default H_scoreMean value. This value will be used if no matching data is found.

    Returns:
    None
    """
    default_h_score_mean=0.0
    default_compound_names = ''
    default_Inty_cosine_score = 0.0
    default_error_ppm = ''
    default_Smiles = ''
    default_adducts = ''
    default_Feature_based_prediction = ''
    
    # Read the GraphML file
    graphml_file = next(file for file in os.listdir(gnps_folder) if file.endswith('.graphml'))
    graph = nx.read_graphml(os.path.join(gnps_folder, graphml_file))

    # Read DeepHalo results
    halo_files = [file for file in os.listdir(deephalo_result_dereplication_folder) if file.endswith('feature.csv')]

    # Add new attributes to each node
    for node, data in graph.nodes(data=True):
        unique_file_sources = data.get('UniqueFileSources', '').split('|')
        # print(f"Processing node {node},unique_file_sources:{unique_file_sources}")
        precursor_mass = data.get('precursor mass', None)
        rt_mean = data.get('RTMean', None)

        if precursor_mass is None or rt_mean is None:
            graph.nodes[node]['H_scoreMean'] = default_h_score_mean
            graph.nodes[node]['compound_names'] = default_compound_names
            graph.nodes[node]['Inty_cosine_score'] = default_Inty_cosine_score
            graph.nodes[node]['error_ppm'] = default_error_ppm
            graph.nodes[node]['Smiles'] = default_Smiles
            graph.nodes[node]['Adducts'] = default_adducts
            graph.nodes[node]['Feature_based_prediction'] = default_Feature_based_prediction
            
            continue

        file_names = [name.split('.mzML')[0].split('.mzml')[0] for name in unique_file_sources]
        # Get the intersection of file_names and halo_files
        halo_files_names = [file for file in halo_files if file.split('_feature.csv')[0] in file_names]
        halo_data_ = pd.DataFrame()

        for f in halo_files_names:
            # print(f"Processing file {f}")
            halo_file = os.path.join(deephalo_result_dereplication_folder, f)
            halo_data = pd.read_csv(halo_file)
            # Extract data from halo_data where mz is close to precursor_mass        
            halo_data = halo_data[halo_data['mz'].between(precursor_mass - 0.05, precursor_mass + 0.05)]
            if len(halo_data) > 1:
                # Extract data from halo_data where mz is close to precursor_mass
                mz_diff = round(abs(halo_data['mz'] - precursor_mass),5)
                halo_data = halo_data[mz_diff == mz_diff.min()]
                
            # halo_data = halo_data[(halo_data['RT'] > (rt_mean - 60)) & (halo_data['RT'] < (rt_mean + 60))]
            halo_data_ = pd.concat([halo_data_, halo_data])
        
        #reset index
        halo_data_ = halo_data_.reset_index(drop=True)

        if not halo_data_.empty:
            h_score_mean = round(float(halo_data_['H_score'].mean()), 2)
            # try:
            #     Feature_based_prediction = str(halo_data_.loc[halo_data_['Inty_cosine_score'].idxmax(), 'Feature_based_prediction'])
            # except:
            Feature_based_prediction = str(halo_data_['Feature_based_prediction'].tolist())
            
            # if halo_data_['Inty_cosine_score']中的元素有不为None的
            if halo_data_['compound_names'].notnull().any():
                idx = halo_data_[halo_data_['compound_names'].notnull()].index
                if len(idx)/len(halo_data_) > 0.5:
                    
                    #获取不为None的元素的index            
                    # print(f"Processing node {node},error_ppm:{halo_data_.loc[idx, 'error_ppm'][0]}")
                    # print(f"Processing node {node},Inty_cosine_score:{halo_data_['Inty_cosine_score'][idx].values}")
                    # print(f"Processing node {node},Smiles:{halo_data_.loc[idx, 'Smiles'].tolist()}")
                    
                    #mean of Inty_cosine_score
                    #如果是数字，取平均值
                    if len(halo_data_['compound_names'][idx].unique().tolist()) == 1:
                        # 如果是数字，取平均值
                        
                        Inty_cosine_score = halo_data_['Inty_cosine_score'].apply(lambda x: ast.literal_eval(x) if type(x) is str else float(x))
                        Inty_cosine_score = np.array(Inty_cosine_score.tolist())
                        Inty_cosine_score_mean = np.mean(Inty_cosine_score, axis=0)
                        Inty_cosine_score = Inty_cosine_score_mean.tolist()
                        print(f"Processing node {node},Inty_cosine_score:{Inty_cosine_score}")
                        
                        
                        print(f"Processing node {node},Inty_cosine_score:{Inty_cosine_score}")
                        
                        
                    else:
                        compounds= halo_data_['compound_names'][idx].unique
                        print(f"Processing node {node},compounds:{compounds}")
                    
                        
                    
                    
                    # Inty_cosine_score = round(float(halo_data_['Inty_cosine_score'].mean()), 4)
                    # print(f"Processing node {node},Inty_cosine_score:{Inty_cosine_score}")
                    # error_ppm = str(halo_data_.loc[idx, 'error_ppm'].tolist())
                    # Smiles = str(halo_data_.loc[idx, 'Smiles'].tolist())
                    # Adducts = str(halo_data_.loc[idx, 'adducts'].tolist())
                else:
                    compound_names = ''
                    Inty_cosine_score = 0
                    error_ppm = 1e6
                    Smiles = ''
                    Adducts = ''               
            else:
                compound_names = ''
                Inty_cosine_score = 0
                error_ppm = 1e6
                Smiles = ''
                Adducts = ''
        else:

            h_score_mean = default_h_score_mean
            compound_names = default_compound_names
            Inty_cosine_score = default_Inty_cosine_score
            error_ppm = default_error_ppm
            Smiles = default_Smiles
            Adducts = default_adducts
            Feature_based_prediction = default_Feature_based_prediction

        graph.nodes[node]['H_scoreMean'] = h_score_mean
        graph.nodes[node]['compound_names'] = compound_names
        graph.nodes[node]['Inty_cosine_score'] = Inty_cosine_score
        graph.nodes[node]['error_ppm'] = error_ppm
        graph.nodes[node]['Smiles'] = Smiles
        graph.nodes[node]['Adducts'] = Adducts
        graph.nodes[node]['Feature_based_prediction'] = Feature_based_prediction
    
    for node in graph.nodes:
        for attr, value in graph.nodes[node].items():
            # print(f"node:{node},attr:{attr},value:{type(value)}")
            if isinstance(value, (list, dict, set)):
                graph.nodes[node][attr] = str(value)  # Convert to string
            elif value is None:
                graph.nodes[node][attr] = ''  # Convert None to empty string

        

    # Write the modified graph back to a GraphML file
    output_file = os.path.join(gnps_folder, graphml_file.replace('.graphml', '_adding_DeepHalo_results.graphml'))
    nx.write_graphml(graph, output_file)
    print(f"Modified GraphML file saved to {output_file}")
