import os
import numpy as np
import pandas as pd
import networkx as nx
import ast

def cosine_similarity(inty_list1, inty_list2):
    inty_list1 = np.array(inty_list1, dtype=float)
    inty_list2 = np.array(inty_list2, dtype=float)
    dot_product = np.dot(inty_list1, inty_list2)
    norm1 = np.linalg.norm(inty_list1)
    norm2 = np.linalg.norm(inty_list2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)

def combine_columns(row, columns):
    for col in columns:
        if row[col] not in ['None', 1e6]:
            return row[col]
    return 'None' if 'error_ppm' not in columns else 1e6

def parse_inty(x):
    return ast.literal_eval(x) if isinstance(x, str) else float(x)

def compute_attributes(halo_data_, default_Feature_based_prediction):
    # Compute basic attributes from non-empty halo_data_
    h_score_mean = round(float(halo_data_['H_score'].mean()), 2)
    Feature_based_prediction = str(halo_data_['Feature_based_prediction'].tolist())
    
    compound_valid = halo_data_['compound_names'].notnull()
    if compound_valid.any():
        indices = halo_data_[compound_valid].index
        # If all non-null compound_names are the same and they dominate the cluster
        if len(halo_data_['compound_names'].dropna().unique()) == 1 and (len(indices) / len(halo_data_)) > 0.5:
            # Process Inty_cosine_score
            scores_series = halo_data_['Inty_cosine_score'].loc[indices].apply(parse_inty)
            scores_arr = np.array(scores_series.tolist())
            mean_score = np.mean(scores_arr, axis=0)
            Inty_cosine_score = str(np.round(mean_score, 4).tolist())
            compound_names = str(halo_data_['compound_names'].loc[indices].unique()[0])
            
            # Process error_ppm
            error_series = halo_data_['error_ppm'].loc[indices].apply(parse_inty)
            error_arr = np.array(error_series.tolist())
            mean_error = np.mean(error_arr, axis=0)
            error_ppm = str(np.round(mean_error, 4).tolist())
            
            # Process Smiles and Adducts
            Smiles = str(halo_data_['Smiles'].loc[indices].unique()[0])
            Adducts = str(halo_data_['adducts'].loc[indices].unique()[0])
            return h_score_mean, compound_names, Inty_cosine_score, error_ppm, Smiles, Adducts, Feature_based_prediction
    # Default blank attributes if compounds are not valid
    return h_score_mean, '', '', '', '', '', Feature_based_prediction

def add_deephalo_results_to_graphml(gnps_folder, deephalo_result_dereplication_folder):
    # Define defaults
    default_h_score_mean = 0.0
    default_compound_names = ''
    default_Inty_cosine_score = ''
    default_error_ppm = ''
    default_Smiles = ''
    default_adducts = ''
    default_Feature_based_prediction = ''
    
    # Read GraphML file from the provided folder
    graphml_file = next(file for file in os.listdir(gnps_folder) if file.endswith('.graphml'))
    graph = nx.read_graphml(os.path.join(gnps_folder, graphml_file))
    
    # List DeepHalo result files ending with "feature.csv"
    halo_files = [file for file in os.listdir(deephalo_result_dereplication_folder) if file.endswith('feature.csv')]
    
    for node, data in graph.nodes(data=True):
        unique_file_sources = data.get('UniqueFileSources', '').split('|')
        precursor_mass = data.get('precursor mass', None)
        rt_mean = data.get('RTMean', None)
        
        # If necessary values are missing, set defaults for the node
        if precursor_mass is None or rt_mean is None:
            graph.nodes[node].update({
                'H_scoreMean': default_h_score_mean,
                'compound_names': default_compound_names,
                'Inty_cosine_score': default_Inty_cosine_score,
                'error_ppm': default_error_ppm,
                'Smiles': default_Smiles,
                'Adducts': default_adducts,
                'Feature_based_prediction': default_Feature_based_prediction,
            })
            continue
        
        # Retrieve file base names from the UniqueFileSources field
        file_names = [name.split('.mzML')[0].split('.mzml')[0] for name in unique_file_sources]
        # Get halo files that match the base names
        halo_files_names = [file for file in halo_files if file.split('_feature.csv')[0] in file_names]
        
        halo_data_ = pd.DataFrame()
        for f in halo_files_names:
            halo_file = os.path.join(deephalo_result_dereplication_folder, f)
            halo_data = pd.read_csv(halo_file)
            # Filter by m/z proximity to precursor_mass
            halo_data = halo_data[halo_data['mz'].between(precursor_mass - 0.05, precursor_mass + 0.05)]
            # If multiple rows are present, pick the row(s) with the minimal m/z difference
            if len(halo_data) > 1:
                mz_diff = abs(halo_data['mz'] - precursor_mass).round(5)
                halo_data = halo_data[mz_diff == mz_diff.min()]
            halo_data_ = pd.concat([halo_data_, halo_data])
        halo_data_ = halo_data_.reset_index(drop=True)
        
        # If valid halo data is found, compute the attributes; otherwise use defaults.
        if not halo_data_.empty:
            h_score_mean = round(float(halo_data_['H_score'].mean()), 2)
            comp_attrs = compute_attributes(halo_data_, default_Feature_based_prediction)
            # Unpack attributes
            compound_names, Inty_cosine_score, error_ppm, Smiles, Adducts = comp_attrs[1:6]
        else:
            h_score_mean = default_h_score_mean
            compound_names = default_compound_names
            Inty_cosine_score = default_Inty_cosine_score
            error_ppm = default_error_ppm
            Smiles = default_Smiles
            Adducts = default_adducts
            default_Feature_based_prediction = default_Feature_based_prediction
        
        # Update graph node attributes
        graph.nodes[node].update({
            'H_scoreMean': h_score_mean,
            'compound_names': compound_names,
            'Inty_cosine_score': Inty_cosine_score,
            'error_ppm': error_ppm,
            'Smiles': Smiles,
            'Adducts': Adducts,
            'Feature_based_prediction': default_Feature_based_prediction,
        })
    
    # Ensure that nodes’ attributes are stored as plain strings
    for node in graph.nodes:
        for attr, value in graph.nodes[node].items():
            if isinstance(value, (list, dict, set)):
                graph.nodes[node][attr] = str(value)
            elif value is None:
                graph.nodes[node][attr] = ''
    
    # Write the modified graph to a new GraphML file
    output_file = os.path.join(gnps_folder, graphml_file.replace('.graphml', '_adding_DeepHalo_results.graphml'))
    nx.write_graphml(graph, output_file)
    print(f"Modified GraphML file saved to {output_file}")