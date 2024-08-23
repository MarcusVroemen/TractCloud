import sys
sys.path.append('..')
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
from sklearn.metrics import mean_squared_error, precision_recall_fscore_support
from scipy.stats import pearsonr
from tractography.label_encoder import convert_labels_list

def create_connectome(labels, num_labels):
    connectome_matrix = np.zeros((num_labels, num_labels))
    for i in range(len(labels) - 1):
        x=labels[i][0]
        y=labels[i][1]
        connectome_matrix[x, y] += 1
        if x!=y:
            connectome_matrix[y, x] += 1
    return connectome_matrix

def save_connectome(connectome_matrix, out_path, title='true'):
    os.makedirs(out_path, exist_ok=True)
    csv_path = os.path.join(self.out_path, f"connectome_{title}.csv")
    np.savetxt(csv_path, connectome_matrix, delimiter=',')

def plot_connectome(connectome_matrix, output_file, title, zero_diagonal=True, log_scale=False, difference=False):
    # Set the diagonal to zero
    if zero_diagonal:
        np.fill_diagonal(connectome_matrix, 0)
    
    if difference==True:
        cmap = plt.get_cmap('RdBu_r')  # Red-white-blue colormap
        norm = mcolors.TwoSlopeNorm(vmin=connectome_matrix.min(), vcenter=0, vmax=connectome_matrix.max())
    elif difference=='percent':
        cmap = plt.get_cmap('RdBu_r')  # Red-white-blue colormap
        norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    else:
        cmap = plt.get_cmap('viridis')

        if log_scale:
            norm = mcolors.LogNorm(vmin=max(connectome_matrix.min(), 1), vmax=connectome_matrix.max())
            # Handle zero values by replacing them with a small positive value for log scale
            connectome_matrix = np.where(connectome_matrix == 0, 1e-6, connectome_matrix)
        else:
            norm = None

    # Plot the connectome matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(connectome_matrix, cmap=cmap, norm=norm)
    plt.colorbar(label='Connection Strength')
    plt.title(title)
    plt.xlabel('node')
    plt.ylabel('node')
    
    # Save the plot as an image file
    plt.savefig(output_file, bbox_inches='tight', dpi=500)
    plt.close()


class ConnectomeMetrics:
    def __init__(self, true_labels=None, pred_labels=None, true_conenctome=None, pred_connectome=None,
                 encoding='default', num_labels=85, out_path='output', state_labels_encoded=True):
        self.true_labels = true_labels
        self.pred_labels = pred_labels
        self.true_connectome = true_connectome
        self.pred_connectome = pred_connectome
        
        self.encoding = encoding
        self.num_labels = num_labels
        self.out_path = out_path
        self.results = {}
        
        # Decode labels from 1D list to 2D node pairs if necessary
        if state_labels_encoded==True and self.true_labels!=None and self.pred_labels!=None:
            self.true_labels = convert_labels_list(self.true_labels, encoding_type=self.encoding, 
                                                mode='decode', num_labels=self.num_labels)
            self.pred_labels = convert_labels_list(self.pred_labels, encoding_type=self.encoding, 
                                                mode='decode', num_labels=self.num_labels)
        elif state_labels_encoded==True and self.true_labels==None and self.pred_labels==None:
            print('Input labels to decode them')    
        
        # Compute connectomes if necessary
        if self.true_connectome==None and self.true_labels!=None:
            self.true_connectome = create_connectome(self.true_labels, self.num_labels)
        elif self.true_connectome==None and self.true_labels==None:
            print('Input valid labels or connectome')    
            
        if self.pred_connectome==None and self.pred_labels!=None:
            self.pred_connectome = create_connectome(self.pred_labels, self.num_labels)
        elif self.pred_connectome==None and self.pred_labels==None:
            print('Input valid labels or connectome')    
        
        # Compute difference and percentile change connectomes
        self.difference_connectome = difference_conenctome()
        self.percentchange_connectome = percentile_change_connectome()
        
        # Save connectomes
        save_connectome(self.true_connectome, self.out_path, title='true')
        save_connectome(self.pred_connectome, self.out_path, title='pred')
        save_connectome(self.difference_connectome, self.out_path, title='diff')
        save_connectome(self.percentchange_connectome, self.out_path, title='perc')
        
        # Save different connectome plots
        self.plot_connectomes(zero_diagonal=True)
        self.plot_connectomes(zero_diagonal=False)
        self.plot_connectomes(log_scale=False, zero_diagonal=True)
        self.plot_connectomes(log_scale=False, zero_diagonal=False)
        
        # Compute metrics
        # try:
        #     self.compute_metrics()
        # except:
        #     pass

    def difference_conenctome(self):
        return self.true_connectome - self.pred_connectome_matrix
    
    def percentile_change_connectome(self):
        percentchange_connectome = (self.true_connectome - self.pred_connectome_matrix) / self.true_connectome
        return np.nan_to_num(percentchange_connectome, nan=0.0, posinf=0.0, neginf=0.0)

    def plot_connectome(connectome_matrix, title='', zero_diagonal=True, log_scale=True):
        plot_specs=''
        if log_scale==True:
            plot_specs+='_logscaled'
        if zero_diagonal==False:
            plot_specs+='_withdiagonal'

        plot_connectome(self.true_connectome, f"{self.out_path}/connectome_true{plot_specs}.png", 
                        f"True connectome {plot_specs.replace('_', ' ')}", zero_diagonal=zero_diagonal, log_scale=log_scale)
        plot_connectome(self.pred_connectome, f"{self.out_path}/connectome_pred{plot_specs}.png", 
                        f"Predicted connectome {plot_specs.replace('_', ' ')}", zero_diagonal=zero_diagonal, log_scale=log_scale)
        
        if not log_scale:
            plot_connectome(self.difference_connectome, f"{self.out_path}/connectome_diff{plot_specs}.png", 
                            f"Difference connectome {plot_specs.replace('_', ' ')} (True-Predicted)", difference=True, zero_diagonal=zero_diagonal)
            plot_connectome(self.percentchange_connectome, f"{self.out_path}/connectome_perc{plot_specs}.png", 
                            f"Percent change connectome {plot_specs.replace('_', ' ')} ((True-Predicted)/True)", difference='percent', zero_diagonal=zero_diagonal)

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
        G_true = nx.from_numpy_array(self.true_connectome)
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
        correlation, _ = pearsonr(self.true_connectome.flatten(), self.pred_connectome_matrix.flatten())
        mse = mean_squared_error(self.true_connectome, self.pred_connectome_matrix)

        # Storing the results
        self.results = {
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Correlation': correlation,
            'MSE': mse
        }
        self.results.update(metrics)