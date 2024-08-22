
# import numpy as np
# import h5py
# import os
# import sys
# sys.path.append('..')
# import copy
# import torch
# import matplotlib.ticker as mtick
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, confusion_matrix 
# from tractography.label_encoder import convert_labels_list
# from tractography.plot_connectome import plot_connectome
# import networkx as nx
# import numpy as np
# from sklearn.metrics import mean_squared_error, precision_recall_fscore_support
# from scipy.stats import pearsonr

# def network_metrics(connectome_matrix):
#     import networkx as nx
#     import pdb
#     pdb.set_trace()
#     # Assuming G is your connectome graph (could be loaded from an adjacency matrix)
#     G = nx.from_numpy_array(connectome_matrix)
#     # Global properties
#     clustering_coefficient = nx.average_clustering(G)
#     path_length = nx.average_shortest_path_length(G)
#     small_worldness = clustering_coefficient / path_length

#     # Local properties
#     node_degrees = dict(G.degree())
#     clustering_coefficients = nx.clustering(G)
#     node_strengths = dict(G.degree(weight='weight'))

# def global_reaching_centrality(G):
#     reachability = {}
#     for node in G.nodes():
#         reachability[node] = nx.single_source_shortest_path_length(G, node)
#     CmaxR = max(sum(reachability[node].values()) / (len(G.nodes()) - 1) for node in G.nodes())
#     GRC = sum(CmaxR - (sum(reachability[node].values()) / (len(G.nodes()) - 1)) for node in G.nodes()) / (len(G.nodes()) - 1)
#     return GRC
    
# def small_worldness(G):
#     L = nx.average_shortest_path_length(G, weight='weight')
#     C = nx.average_clustering(G, weight='weight')
#     rand_graph = nx.random_reference(G, niter=5)
#     L_rand = nx.average_shortest_path_length(rand_graph, weight='weight')
#     C_rand = nx.average_clustering(rand_graph, weight='weight')
#     return (C / C_rand) / (L / L_rand)

# def compute_connectome_metrics(ground_truth_matrix, predicted_matrix):
#     # Create graphs from the adjacency matrices
#     G_true = nx.from_numpy_matrix(ground_truth_matrix)
#     G_pred = nx.from_numpy_matrix(predicted_matrix)

#     # Node-level centrality metrics
#     degree_centrality_true = dict(G_true.degree(weight='weight'))
#     eigenvector_centrality_true = nx.eigenvector_centrality_numpy(G_true, weight='weight')
#     betweenness_centrality_true = nx.betweenness_centrality(G_true, weight='weight', normalized=True)

#     degree_centrality_pred = dict(G_pred.degree(weight='weight'))
#     eigenvector_centrality_pred = nx.eigenvector_centrality_numpy(G_pred, weight='weight')
#     betweenness_centrality_pred = nx.betweenness_centrality(G_pred, weight='weight', normalized=True)

#     # Global network properties
#     global_efficiency_true = nx.global_efficiency(G_true)
#     global_efficiency_pred = nx.global_efficiency(G_pred)

#     modularity_true = nx.algorithms.community.modularity(G_true, nx.algorithms.community.greedy_modularity_communities(G_true, weight='weight'))
#     modularity_pred = nx.algorithms.community.modularity(G_pred, nx.algorithms.community.greedy_modularity_communities(G_pred, weight='weight'))

#     # Global Reaching Centrality (GRC)
#     GRC_true = global_reaching_centrality(G_true)
#     GRC_pred = global_reaching_centrality(G_pred)

#     # Additional metrics
#     clustering_coefficient_true = nx.average_clustering(G_true, weight='weight')
#     clustering_coefficient_pred = nx.average_clustering(G_pred, weight='weight')

#     path_length_true = nx.average_shortest_path_length(G_true, weight='weight')
#     path_length_pred = nx.average_shortest_path_length(G_pred, weight='weight')

#     # Small-worldness: Requires calculating both clustering coefficient and path length
#     small_worldness_true = small_worldness(G_true)
#     small_worldness_pred = small_worldness(G_pred)

#     # Edge overlap (binary comparison)
#     y_true = nx.to_numpy_array(G_true).flatten()
#     y_pred = nx.to_numpy_array(G_pred).flatten()
#     precision, recall, f1, _ = precision_recall_fscore_support(y_true > 0, y_pred > 0, average='binary')

#     # Correlation of edge weights
#     correlation, _ = pearsonr(ground_truth_matrix.flatten(), predicted_matrix.flatten())

#     # Mean Squared Error
#     mse = mean_squared_error(ground_truth_matrix, predicted_matrix)

#     # Prepare results
#     results = {
#         'Precision': precision,  # Edge overlap precision: Accuracy of predicted edges.
#         'Recall': recall,  # Edge overlap recall: Sensitivity to detecting actual edges.
#         'F1-Score': f1,  # Edge overlap F1-Score: Balance between precision and recall.
#         'Correlation': correlation,  # Correlation of edge weights: Similarity of edge strengths.
#         'MSE': mse,  # Mean Squared Error: Overall difference between matrices.

#         'Global Efficiency (True)': global_efficiency_true,  # Network integration efficiency: Information transfer efficiency.
#         'Global Efficiency (Predicted)': global_efficiency_pred,

#         'Modularity (True)': modularity_true,  # Community structure strength: Functional segregation.
#         'Modularity (Predicted)': modularity_pred,

#         'Global Reaching Centrality (True)': GRC_true,  # Hierarchical network structure: Reflects the reach of central nodes.
#         'Global Reaching Centrality (Predicted)': GRC_pred,

#         'Clustering Coefficient (True)': clustering_coefficient_true,  # Local network density: Functional specialization.
#         'Clustering Coefficient (Predicted)': clustering_coefficient_pred,

#         'Path Length (True)': path_length_true,  # Shortest paths in the network: Reflects global integration and processing speed.
#         'Path Length (Predicted)': path_length_pred,

#         'Small-worldness (True)': small_worldness_true,  # Balance between local and global efficiency: Brain network's optimal information processing.
#         'Small-worldness (Predicted)': small_worldness_pred,

#         'Degree Centrality (True)': degree_centrality_true,  # Node connectivity: Basic influence measure in the network.
#         'Degree Centrality (Predicted)': degree_centrality_pred,

#         'Eigenvector Centrality (True)': eigenvector_centrality_true,  # Influence of nodes considering neighbors' importance.
#         'Eigenvector Centrality (Predicted)': eigenvector_centrality_pred,

#         'Betweenness Centrality (True)': betweenness_centrality_true,  # Control over information flow: Key communication nodes.
#         'Betweenness Centrality (Predicted)': betweenness_centrality_pred
#     }

#     return results
# def plot_connectomes(original_labels, predicted_labels, encoding_type, num_f_brain=None, num_labels=85, out_path='output'):
#     """Decode labels, create connectomes, and plot them."""
#     if num_f_brain==None:
#         decoded_original = convert_labels_list(original_labels, encoding_type=encoding_type, mode='decode', num_labels=num_labels)
#         decoded_predicted = convert_labels_list(predicted_labels, encoding_type=encoding_type, mode='decode', num_labels=num_labels)
#         # Create connectome matrices
#         original_connectome = create_connectome(decoded_original, num_labels=num_labels)
#         predicted_connectome = create_connectome(decoded_predicted, num_labels=num_labels)
#         difference_connectome = original_connectome - predicted_connectome

#         # Save connectome matrices directly as CSV files
#         os.makedirs(f"{out_path}/total/", exist_ok=True)
#         original_csv_path = f"{out_path}/total/original_connectome.csv"
#         predicted_csv_path = f"{out_path}/total/predicted_connectome.csv"
#         difference_csv_path = f"{out_path}/total/difference_connectome.csv"

#         np.savetxt(original_csv_path, original_connectome, delimiter=',')
#         np.savetxt(predicted_csv_path, predicted_connectome, delimiter=',')
#         np.savetxt(difference_csv_path, difference_connectome, delimiter=',')

#         # Plot connectomes
#         plot_connectome(original_csv_path, f"{out_path}/total/original_connectome.png", f"Original Connectome")
#         plot_connectome(predicted_csv_path, f"{out_path}/total/predicted_connectome.png", f"Predicted Connectome")
#         plot_connectome(original_csv_path, f"{out_path}/total/original_connectome_logscaled.png", f"Original Connectome", log_scale=True)
#         plot_connectome(predicted_csv_path, f"{out_path}/total/predicted_connectome_logscaled.png", f"Predicted Connectome", log_scale=True)
#         plot_connectome(difference_csv_path, f"{out_path}/total/difference_connectome.png", f"Difference Connectome (Original - Predicted)", difference=True)
#         # with diagonal
#         plot_connectome(original_csv_path, f"{out_path}/total/original_connectome_withdiagonal.png", f"Original Connectome with Diagonal", zero_diagonal=False)
#         plot_connectome(predicted_csv_path, f"{out_path}/total/predicted_connectome_withdiagonal.png", f"Predicted Connectome with Diagonal", zero_diagonal=False)
#         plot_connectome(original_csv_path, f"{out_path}/total/original_connectome_logscaled_withdiagonal.png", f"Original Connectome with Diagonal", log_scale=True, zero_diagonal=False)
#         plot_connectome(predicted_csv_path, f"{out_path}/total/predicted_connectome_logscaled_withdiagonal.png", f"Predicted Connectome with Diagonal", log_scale=True, zero_diagonal=False)
#         plot_connectome(difference_csv_path, f"{out_path}/total/difference_connectome_diagonal_withdiagonal.png", f"Difference Connectome with Diagonal (Original - Predicted)", difference=True, zero_diagonal=False)

#         # print(f"Connectomes plotted and saved in {out_path}")
        
#         metrics = compute_connectome_metrics(ground_truth_matrix, predicted_matrix)
#     # Per subject
#     else:
#         n_subjects=len(original_labels)/num_f_brain
#         for subject in range(int(n_subjects)):
#             print(f"Building connectomes for subject {subject}")
#             # Decode the labels
#             decoded_original = convert_labels_list(original_labels[subject*num_f_brain:(subject+1)*num_f_brain], encoding_type=encoding_type, mode='decode', num_labels=num_labels)
#             decoded_predicted = convert_labels_list(predicted_labels[subject*num_f_brain:(subject+1)*num_f_brain], encoding_type=encoding_type, mode='decode', num_labels=num_labels)
#             # Create connectome matrices
#             original_connectome = create_connectome(decoded_original, num_labels=num_labels)
#             predicted_connectome = create_connectome(decoded_predicted, num_labels=num_labels)
#             difference_connectome = original_connectome - predicted_connectome
            
#             # Save connectome matrices directly as CSV files
#             os.makedirs(f"{out_path}/subject_{subject}/", exist_ok=True)
#             original_csv_path = f"{out_path}/subject_{subject}/original_connectome_subject.csv"
#             predicted_csv_path = f"{out_path}/subject_{subject}/predicted_connectome.csv"
#             difference_csv_path = f"{out_path}/subject_{subject}/difference_connectome.csv"

#             np.savetxt(original_csv_path, original_connectome, delimiter=',')
#             np.savetxt(predicted_csv_path, predicted_connectome, delimiter=',')
#             np.savetxt(difference_csv_path, difference_connectome, delimiter=',')
                    
#             # Plot connectomes
#             plot_connectome(original_csv_path, f"{out_path}/subject_{subject}/original_connectome.png", f"Original Connectome for Subject {subject}")
#             plot_connectome(predicted_csv_path, f"{out_path}/subject_{subject}/predicted_connectome.png", f"Predicted Connectome for Subject {subject}")
#             plot_connectome(original_csv_path, f"{out_path}/subject_{subject}/original_connectome_logscaled.png", f"Original Connectome for Subject {subject}", log_scale=True)
#             plot_connectome(predicted_csv_path, f"{out_path}/subject_{subject}/predicted_connectome_logscaled.png", f"Predicted Connectome for Subject {subject}", log_scale=True)
#             plot_connectome(difference_csv_path, f"{out_path}/subject_{subject}/difference_connectome.png", f"Difference Connectome (Original - Predicted) for Subject {subject}", difference=True)

#             # print(f"Connectomes plotted and saved in {out_path}")
            

