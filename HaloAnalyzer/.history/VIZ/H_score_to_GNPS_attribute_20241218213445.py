import networkx as nx
import os
import pandas as pd

def add_deephalo_results_to_graphml(gnps_folder, deephalo_result, default_h_score_mean=0.0):
    """
    Read the GraphML file and add the 'H_scoreMean' attribute to each node.

    Parameters:
    gnps_folder (str): Path to the GNPS folder.
    deephalo_result (str): Path to the DeepHalo results folder.
    default_h_score_mean (float, optional): Default H_scoreMean value. This value will be used if no matching data is found.

    Returns:
    None
    """
    default_compound_names = 'None'
    default_intensity_score = 0
    default_error_ppm = 1e6
    default_Smiles = 'None'
    default_adducts = 'None'
    
    # Read the GraphML file
    graphml_file = next(file for file in os.listdir(gnps_folder) if file.endswith('.graphml'))
    graph = nx.read_graphml(os.path.join(gnps_folder, graphml_file))

    # Read DeepHalo results
    halo_folder = os.path.join(deephalo_result, 'halo')
    halo_files = [file for file in os.listdir(halo_folder) if file.endswith('feature.csv')]

    # Add new attributes to each node
    for node, data in graph.nodes(data=True):
        unique_file_sources = data.get('UniqueFileSources', '').split(';')
        precursor_mass = data.get('precursor mass', None)
        rt_mean = data.get('RTMean', None)
        compound_name = data.get('compound_name', None)

        if precursor_mass is None or rt_mean is None or compound_name:
            graph.nodes[node]['H_scoreMean'] = default_h_score_mean
            graph.nodes[node]['compound_names'] = default_compound_names
            graph.nodes[node]['intensity_score'] = default_intensity_score
            graph.nodes[node]['error_ppm'] = default_error_ppm
            graph.nodes[node]['Smiles'] = default_Smiles
            graph.nodes[node]['Adducts'] = default_adducts
            
            continue

        file_names = [name.split('.mzML')[0].split('.mzml')[0] for name in unique_file_sources]
        # Get the intersection of file_names and halo_files
        halo_files_names = [file for file in halo_files if file.split('_feature.csv')[0] in file_names]
        halo_data_ = pd.DataFrame()

        for f in halo_files_names:
            halo_file = os.path.join(halo_folder, f)
            halo_data = pd.read_csv(halo_file)
            # Extract data from halo_data where mz is close to precursor_mass        
            halo_data = halo_data[(halo_data['mz'] > (precursor_mass - 0.03)) & (halo_data['mz'] < (precursor_mass + 0.03))]
            if len(halo_data) > 1:
                # Extract data from halo_data where mz is close to precursor_mass
                mz_diff = abs(halo_data['mz'] - precursor_mass)
                halo_data = halo_data[mz_diff == mz_diff.min()]
                
            # halo_data = halo_data[(halo_data['RT'] > (rt_mean - 60)) & (halo_data['RT'] < (rt_mean + 60))]
            halo_data_ = pd.concat([halo_data_, halo_data])

        if not halo_data_.empty:
            h_score_mean = halo_data_['H_score'].mean()
            compound_names = str(halo_data_.loc[halo_data_['intensity_score'].idxmax(), 'compound_names'])
            intensity_score = halo_data_.loc[halo_data_['intensity_score'].idxmax(), 'intensity_score']
            error_ppm = halo_data_.loc[halo_data_['intensity_score'].idxmax(), 'error_ppm']
            Smiles = str(halo_data_.loc[halo_data_['intensity_score'].idxmax(), 'Smiles'])
            Adducts = str(halo_data_.loc[halo_data_['intensity_score'].idxmax(), 'adducts'])
        else:
            h_score_mean = default_h_score_mean
            compound_names = default_compound_names
            intensity_score = default_intensity_score
            error_ppm = default_error_ppm
            Smiles = default_Smiles
            Adducts = default_adducts

        graph.nodes[node]['H_scoreMean'] = h_score_mean
        graph.nodes[node]['compound_names'] = compound_names
        graph.nodes[node]['intensity_score'] = intensity_score
        graph.nodes[node]['error_ppm'] = error_ppm
        graph.nodes[node]['Smiles'] = Smiles
        graph.nodes[node]['Adducts'] = Adducts
        

    # Write the modified graph back to a GraphML file
    output_file = os.path.join(gnps_folder, graphml_file.replace('.graphml', '_H_score.graphml'))
    nx.write_graphml(graph, output_file)
    print(f"Modified GraphML file saved to {output_file}")

if __name__ == '__main__':
    gnps_folder = r'D:\workissues\manuscript\halo_mining\mining\result\54_strep_1_micromono_blank_GNPS'
    deephalo_result = r'D:\workissues\manuscript\halo_mining\mining\result\54_strep_1_micromono_blank'
    add_deephalo_results_to_graphml(gnps_folder, deephalo_result)