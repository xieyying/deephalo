import networkx as nx
import os

# Define the folder containing the GraphML files
gnps_folder = r'D:\workissues\manuscript\halo_mining\mining\result\53_strep_1_micomono_blank_0.6_5'

# Get the GraphML files
graphml_files = [file for file in os.listdir(gnps_folder) if file.endswith('.graphml')]
print(graphml_files)  

# Read the GraphML files into NetworkX graphs
graphs = [nx.read_graphml(os.path.join(gnps_folder, file)) for file in graphml_files]

# Create a new graph to merge the nodes and edges
merged_graph = nx.Graph()

# Merge the nodes and edges from the graphs
for graph in graphs:
    merged_graph.add_nodes_from(graph.nodes(data=True))
    merged_graph.add_edges_from(graph.edges(data=True))

# Save the merged graph to a new GraphML file
output_file = os.path.join(gnps_folder, 'merged_graph.graphml')
# nx.write_graphml(merged_graph, output_file)

print(f"Merged graph saved to {output_file}")