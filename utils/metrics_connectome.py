import sys
sys.path.append('..')
import numpy as np
import os
import networkx as nx
from sklearn.metrics import mean_squared_error, precision_recall_fscore_support
from scipy.stats import pearsonr
from tractography.label_encoder import convert_labels_list
from tractography.plot_connectome import plot_connectome

class ConnectomeMetrics:
    def __init__(self, org_labels_lst, org_predicted_lst, encoding_type='default', num_labels=85, out_path='output'):
        self.org_labels_lst = org_labels_lst
        self.org_predicted_lst = org_predicted_lst
        self.encoding_type = encoding_type
        self.num_labels = num_labels
        self.out_path = out_path
        self.results = {}
        
        # Decode labels from 1D list to 2D node pairs
        self.true_labels = self.decode_labels(self.org_labels_lst)
        self.pred_labels = self.decode_labels(self.org_predicted_lst)
        
        # Compute connectomes
        self.true_connectome_matrix = self.create_connectome(self.true_labels)
        self.pred_connectome_matrix = self.create_connectome(self.pred_labels)
        
        # Compute metrics
        self.plot_connectomes(zero_diagonal=True)
        self.plot_connectomes(zero_diagonal=False)
        
        self.compute_metrics()

    def decode_labels(self, labels_lst):
        decoded_labels = convert_labels_list(labels_lst, encoding_type=self.encoding_type, mode='decode', num_labels=self.num_labels)
        return decoded_labels

    def create_connectome(self, labels):
        connectome_matrix = np.zeros((self.num_labels, self.num_labels))
        for i in range(len(labels) - 1):
            connectome_matrix[labels[i], labels[i+1]] += 1
        return connectome_matrix

    def global_reaching_centrality(self, G):
        reachability = {}
        for node in G.nodes():
            reachability[node] = nx.single_source_shortest_path_length(G, node)
        CmaxR = max(sum(reachability[node].values()) / (len(G.nodes()) - 1) for node in G.nodes())
        GRC = sum(CmaxR - (sum(reachability[node].values()) / (len(G.nodes()) - 1)) for node in G.nodes()) / (len(G.nodes()) - 1)
        return GRC

    def small_worldness(self, G):
        L = nx.average_shortest_path_length(G, weight='weight')
        C = nx.average_clustering(G, weight='weight')
        rand_graph = nx.random_reference(G, niter=5)
        L_rand = nx.average_shortest_path_length(rand_graph, weight='weight')
        C_rand = nx.average_clustering(rand_graph, weight='weight')
        return (C / C_rand) / (L / L_rand)

    def compute_metrics(self):
        G_true = nx.from_numpy_array(self.true_connectome_matrix)
        G_pred = nx.from_numpy_array(self.pred_connectome_matrix)

        # Node-level centrality metrics
        metrics = {
            'Degree Centrality': (dict(G_true.degree(weight='weight')), dict(G_pred.degree(weight='weight'))),
            'Eigenvector Centrality': (nx.eigenvector_centrality_numpy(G_true, weight='weight'), nx.eigenvector_centrality_numpy(G_pred, weight='weight')),
            'Betweenness Centrality': (nx.betweenness_centrality(G_true, weight='weight', normalized=True), nx.betweenness_centrality(G_pred, weight='weight', normalized=True))
        }

        # Global network properties
        metrics.update({
            'Global Efficiency': (nx.global_efficiency(G_true), nx.global_efficiency(G_pred)),
            'Modularity': (
                nx.algorithms.community.modularity(G_true, nx.algorithms.community.greedy_modularity_communities(G_true, weight='weight')),
                nx.algorithms.community.modularity(G_pred, nx.algorithms.community.greedy_modularity_communities(G_pred, weight='weight'))
            ),
            'Global Reaching Centrality': (self.global_reaching_centrality(G_true), self.global_reaching_centrality(G_pred)),
            'Clustering Coefficient': (nx.average_clustering(G_true, weight='weight'), nx.average_clustering(G_pred, weight='weight')),
            'Path Length': (nx.average_shortest_path_length(G_true, weight='weight'), nx.average_shortest_path_length(G_pred, weight='weight')),
            'Small-worldness': (self.small_worldness(G_true), self.small_worldness(G_pred))
        })

        # Edge overlap and other comparison metrics
        y_true = nx.to_numpy_array(G_true).flatten()
        y_pred = nx.to_numpy_array(G_pred).flatten()
        precision, recall, f1, _ = precision_recall_fscore_support(y_true > 0, y_pred > 0, average='binary')
        correlation, _ = pearsonr(self.true_connectome_matrix.flatten(), self.pred_connectome_matrix.flatten())
        mse = mean_squared_error(self.true_connectome_matrix, self.pred_connectome_matrix)

        # Storing the results
        self.results = {
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Correlation': correlation,
            'MSE': mse
        }
        self.results.update(metrics)

    def plot_connectomes(self, zero_diagonal=True, log_scale=True):
        plot_specs=''
        if log_scale==True:
            plot_specs+='_logscaled'
        if zero_diagonal==False:
            plot_specs+='_withdiagonal'

        os.makedirs(self.out_path, exist_ok=True)
        original_csv_path = os.path.join(self.out_path, "original_connectome.csv")
        predicted_csv_path = os.path.join(self.out_path, "predicted_connectome.csv")
        difference_csv_path = os.path.join(self.out_path, "difference_connectome.csv")

        difference_connectome = self.true_connectome_matrix - self.pred_connectome_matrix 

        # Save connectomes
        np.savetxt(original_csv_path, self.true_connectome_matrix, delimiter=',')
        np.savetxt(predicted_csv_path, self.pred_connectome_matrix, delimiter=',')
        np.savetxt(difference_csv_path, difference_connectome, delimiter=',')

        # Plot connectomes
        plot_connectome(original_csv_path, f"{self.out_path}/original_connectome{plot_specs}.png", 
                        f"Original Connectome {plot_specs.replace('_', ' ')}", zero_diagonal=zero_diagonal, log_scale=log_scale)
        plot_connectome(predicted_csv_path, f"{self.out_path}/predicted_connectome{plot_specs}.png", 
                        f"Predicted Connectome {plot_specs.replace('_', ' ')}", zero_diagonal=zero_diagonal, log_scale=log_scale)
        plot_connectome(difference_csv_path, f"{self.out_path}/difference_connectome{plot_specs}.png", 
                        f"Difference Connectome {plot_specs.replace('_', ' ')} (Original - Predicted)", difference=True, zero_diagonal=zero_diagonal)
