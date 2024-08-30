import sys
sys.path.append('..')
import numpy as np
import os
import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.metrics import mean_squared_error, precision_recall_fscore_support, classification_report, accuracy_score, confusion_matrix 
from scipy.stats import pearsonr, spearmanr, wasserstein_distance
from utils.metrics_plots import classify_report, process_curves, calculate_acc_prec_recall_f1, best_swap, save_best_weights
from numpy.linalg import norm
import networkx as nx
import pprint
from tractography.label_encoder import convert_labels_list

def create_connectome(labels, num_labels):
    connectome_matrix = np.zeros((num_labels, num_labels))
    if type(labels)==dict: # dict with tuples (nodes) as keys and scores as values
        for key, value in labels.items():
            x = key[0] 
            y = key[1]
            connectome_matrix[x, y]=value
            if x!=y:
                connectome_matrix[y, x]=value
    else:
        for i in range(len(labels) - 1):
            x=labels[i][0]
            y=labels[i][1]
            connectome_matrix[x, y] += 1
            if x!=y:
                connectome_matrix[y, x] += 1
    return connectome_matrix

def save_connectome(connectome_matrix, out_path, title='true'):
    os.makedirs(out_path, exist_ok=True)
    csv_path = os.path.join(out_path, f"connectome_{title}.csv")
    np.savetxt(csv_path, connectome_matrix, delimiter=',')

def plot_connectome(connectome_matrix, output_file, title, zero_diagonal, log_scale, difference=False):
    # Set the diagonal to zero
    if zero_diagonal:
        np.fill_diagonal(connectome_matrix, 0)
    
    if difference==True:
        cmap = plt.get_cmap('RdBu_r')  # Red-white-blue colormap
        norm = mcolors.TwoSlopeNorm(vmin=connectome_matrix.min(), vcenter=0, vmax=connectome_matrix.max())
    elif difference=='percent':
        cmap = plt.get_cmap('RdBu_r')  # Red-white-blue colormap
        norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    elif difference=='accuracy':
        cmap = plt.get_cmap('BuGn')
        norm = mcolors.TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1)
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

def label_wise_accuracy(true_labels, pred_labels):
    # Get the set of unique labels
    unique_labels = set(true_labels) | set(pred_labels)
    
    # Initialize dictionaries to store correct predictions and total counts
    correct_predictions = {label: 0 for label in unique_labels}
    total_counts = {label: 0 for label in unique_labels}
    
    # Iterate over true and predicted labels
    for true, pred in zip(true_labels, pred_labels):
        total_counts[true] += 1
        if true == pred:
            correct_predictions[true] += 1

    # Compute accuracy for each label
    label_accuracy = {}
    for label in unique_labels:
        if total_counts[label] > 0:
            label_accuracy[label] = correct_predictions[label] / total_counts[label]
        else:
            label_accuracy[label] = 1.0
    
    return label_accuracy

class ConnectomeMetrics:
    def __init__(self, true_labels=None, pred_labels=None, encoding='default', num_labels=85, out_path='output'): # , state_labels_encoded=True
        self.true_labels = true_labels
        self.pred_labels = pred_labels
        
        self.encoding = encoding
        self.num_labels = num_labels
        self.out_path = out_path
        self.results = {}
        
        # Decode labels from 1D list to 2D node pairs if necessary
        # if state_labels_encoded==True and self.true_labels!=None and self.pred_labels!=None:
        self.true_labels_decoded = convert_labels_list(self.true_labels, encoding_type=self.encoding, 
                                            mode='decode', num_labels=self.num_labels)
        self.pred_labels_decoded = convert_labels_list(self.pred_labels, encoding_type=self.encoding, 
                                            mode='decode', num_labels=self.num_labels)
        # elif state_labels_encoded==True and self.true_labels==None and self.pred_labels==None:
        #     print('Input labels to decode them')    
        
        self.true_connectome = create_connectome(self.true_labels_decoded, self.num_labels)
        self.pred_connectome = create_connectome(self.pred_labels_decoded, self.num_labels)

        # Save connectomes
        save_connectome(self.true_connectome, self.out_path, title='true')
        save_connectome(self.pred_connectome, self.out_path, title='pred')
        
        # Save different connectome plots
        self.plot_connectomes(zero_diagonal=True, log_scale=True)
        self.plot_connectomes(zero_diagonal=False, log_scale=True)
        self.plot_connectomes(zero_diagonal=True, log_scale=False)
        self.plot_connectomes(zero_diagonal=False, log_scale=False)
        
        # Compute, save and plot alternate "connectomes"
        self.difference_conenctome()
        self.percentile_change_connectome()
        self.accuracy_connectome()
        
        self.compute_metrics()
        # pprint.pprint(self.results)
        self.compute_network_metrics()
        # pprint.pprint(self.results)
        # import pdb
        # pdb.set_trace()
        
        
    def plot_connectomes(self, zero_diagonal, log_scale):
        plot_specs=''
        if log_scale==True:
            plot_specs+='_logscaled'
        if zero_diagonal==False:
            plot_specs+='_withdiagonal'

        plot_connectome(self.true_connectome, f"{self.out_path}/connectome_true{plot_specs}.png", 
                        f"True connectome{plot_specs.replace('_', ' ')}", zero_diagonal=zero_diagonal, log_scale=log_scale)
        plot_connectome(self.pred_connectome, f"{self.out_path}/connectome_pred{plot_specs}.png", 
                        f"Predicted connectome{plot_specs.replace('_', ' ')}", zero_diagonal=zero_diagonal, log_scale=log_scale)

    def difference_conenctome(self):
        self.difference_connectome = self.true_connectome - self.pred_connectome
        
        save_connectome(self.difference_connectome, self.out_path, title='diff')

        # Plot with and without diagonal
        plot_connectome(self.difference_connectome, f"{self.out_path}/connectome_diff.png", 
                        f"Difference connectome (True-Predicted)", difference=True, zero_diagonal=True, log_scale=False)
        plot_connectome(self.difference_connectome, f"{self.out_path}/connectome_diff_withdiagonal.png", 
                        f"Difference connectome with diagonal (True-Predicted)", difference=True, zero_diagonal=False, log_scale=False)
        
    def percentile_change_connectome(self):
        np.seterr(divide='ignore', invalid='ignore')
        percentchange_connectome = (self.true_connectome - self.pred_connectome) / self.true_connectome
        self.percentchange_connectome = np.nan_to_num(percentchange_connectome, nan=0.0, posinf=0.0, neginf=0.0)

        save_connectome(self.percentchange_connectome, self.out_path, title='perc')

        # Plot with and without diagonal
        plot_connectome(self.percentchange_connectome, f"{self.out_path}/connectome_perc.png", 
                        f"Percent change connectome ((True-Predicted)/True)", difference='percent', zero_diagonal=True, log_scale=False)
        plot_connectome(self.percentchange_connectome, f"{self.out_path}/connectome_perc_withdiagonal.png", 
                        f"Percent change connectome with diagonal ((True-Predicted)/True)", difference='percent', zero_diagonal=False, log_scale=False)

    def accuracy_connectome(self):
        # Compute accuracy per label
        accuracy_per_label_decoded = label_wise_accuracy(self.true_labels_decoded, self.pred_labels_decoded)
        self.acc_connectome=create_connectome(accuracy_per_label_decoded, self.num_labels)
        save_connectome(self.acc_connectome, self.out_path, title='acc')
        plot_connectome(self.acc_connectome, f"{self.out_path}/connectome_acc.png", 
                            f"Accuracy connectome", difference='accuracy', zero_diagonal=False, log_scale=False)

    def compute_metrics(self):
        # Edge overlap and other comparison metrics
        acc = accuracy_score(self.true_labels, self.pred_labels)
        mac_precision, mac_recall, mac_f1, support = precision_recall_fscore_support(self.true_labels, self.pred_labels, beta=1.0, average='macro', zero_division=np.nan) # ignore empty labels
        weighted_precision, weighted_recall, weighted_f1, support = precision_recall_fscore_support(self.true_labels, self.pred_labels, beta=1.0, average='weighted', zero_division=np.nan) # ignore empty labels
        
        # Correlation and similarity metrics
        pearson_corr, _ = pearsonr(self.true_connectome.flatten(), self.pred_connectome.flatten())
        spearman_corr, _ = spearmanr(self.true_connectome.flatten(), self.pred_connectome.flatten())
        mse = mean_squared_error(self.true_connectome, self.pred_connectome)
        # Cosine similarity
        # cos_sim = cosine_similarity(self.true_connectome.flatten().reshape(1, -1), self.pred_connectome.flatten().reshape(1, -1))[0][0]

        # Frobenius Norm
        frobenius_norm = norm(self.true_connectome - self.pred_connectome, 'fro')

        # Earth Mover's Distance (Wasserstein distance)
        emd = wasserstein_distance(self.true_connectome.flatten(), self.pred_connectome.flatten())

        # Storing the results
        metrics = {
            'Accuracy' : acc,
            'F1-Score (macro)': mac_f1,
            'Precision (macro)': mac_precision,
            'Recall (macro)': mac_recall,
            'F1-Score (weighted)': weighted_f1,
            'Precision (weighted)': weighted_precision,
            'Recall (weighted)': weighted_recall,
            'MSE': mse,
            'Pearson Correlation': pearson_corr,
            'Spearman Correlation': spearman_corr,
            # 'Cosine Similarity': cos_sim,
            'Frobenius Norm': frobenius_norm,
            'Earth Mover\'s Distance': emd
        }
        self.results.update(metrics)

    def global_reaching_centrality(self, G):
        """Compute Global Reaching Centrality (GRC) for a graph G."""
        reachability = {node: sum(nx.single_source_shortest_path_length(G, node).values()) / (len(G) - 1) for node in G}
        CmaxR = max(reachability.values())
        GRC = sum(CmaxR - reach for reach in reachability.values()) / (len(G) - 1)
        return GRC
    
    # def random_edge_sample(self, G, sample_size):
    #     """Returns a subgraph with a random sample of edges."""
    #     edges = list(G.edges(data=True))
    #     if len(edges)>sample_size:
    #         sampled_edges = random.sample(edges, sample_size)
    #         H = nx.Graph()
    #         H.add_edges_from(sampled_edges)
    #         return H
    #     else:
    #         return G

    def small_worldness(self, G, niter=5):
        """Compute the small-worldness of a graph G using omega and sigma."""
        L = nx.average_shortest_path_length(G, weight='weight')
        C = nx.average_clustering(G, weight='weight')
        rand_graph = nx.random_reference(G, niter=niter)
        L_rand = nx.average_shortest_path_length(rand_graph, weight='weight')
        C_rand = nx.average_clustering(rand_graph, weight='weight')
        omega = L_rand / L - C / C_rand
        sigma = (C / C_rand) / (L / L_rand)
        return omega, sigma        

    def compute_network_metrics(self):
        """Compute a variety of network metrics for true and predicted connectomes."""
        G_true = nx.from_numpy_array(self.true_connectome)
        G_pred = nx.from_numpy_array(self.pred_connectome)

        omega_true, sigma_true = self.small_worldness(G_true, 1)
        omega_pred, sigma_pred = self.small_worldness(G_pred, 1)

        metrics = {
            'Degree Centrality': (
                dict(G_true.degree(weight='weight')), 
                dict(G_pred.degree(weight='weight'))
            ),
            'Eigenvector Centrality': (
                nx.eigenvector_centrality_numpy(G_true, weight='weight'), 
                nx.eigenvector_centrality_numpy(G_pred, weight='weight')
            ),
            'Betweenness Centrality': (
                nx.betweenness_centrality(G_true, weight='weight', normalized=True), 
                nx.betweenness_centrality(G_pred, weight='weight', normalized=True)
            ),
            'Global Efficiency': (
                nx.global_efficiency(G_true), 
                nx.global_efficiency(G_pred)
            ),
            'Local Efficiency': (
                nx.local_efficiency(G_true), 
                nx.local_efficiency(G_pred)
            ),
            'Modularity': (
                nx.algorithms.community.modularity(G_true, nx.algorithms.community.greedy_modularity_communities(G_true, weight='weight')), 
                nx.algorithms.community.modularity(G_pred, nx.algorithms.community.greedy_modularity_communities(G_pred, weight='weight'))
            ),
            'Global Reaching Centrality': (
                self.global_reaching_centrality(G_true), 
                self.global_reaching_centrality(G_pred)
            ),
            'Clustering Coefficient': (
                nx.average_clustering(G_true, weight='weight'), 
                nx.average_clustering(G_pred, weight='weight')
            ),
            'Path Length': (
                nx.average_shortest_path_length(G_true, weight='weight'), 
                nx.average_shortest_path_length(G_pred, weight='weight')
            ),
            'Small-worldness Omega': (omega_true, omega_pred),
            'Small-worldness Sigma': (sigma_true, sigma_pred),
            'Network Density': (nx.density(G_true), nx.density(G_pred))
        }

        self.results.update(metrics)
        
    def format_metrics(self):
        return """
            Metrics Summary:
            ----------------
            Accuracy: {Accuracy:.4f}
            F1-Score (Macro): {F1-Score (macro):.4f}
            Precision (Macro): {Precision (macro):.4f}
            Recall (Macro): {Recall (macro):.4f}
            F1-Score (Weighted): {F1-Score (weighted):.4f}
            Precision (Weighted): {Precision (weighted):.4f}
            Recall (Weighted): {Recall (weighted):.4f}
            MSE: {MSE:.4f}
            Pearson Correlation: {Pearson Correlation:.4f}
            Spearman Correlation: {Spearman Correlation:.4f}
            Frobenius Norm: {Frobenius Norm:.4f}
            Earth Mover's Distance: {Earth Mover's Distance:.4f}

            Network Metrics:
            ----------------
            Degree Centrality (True, Pred): {Degree Centrality[0]}, {Degree Centrality[1]}
            Eigenvector Centrality (True, Pred): {Eigenvector Centrality[0]}, {Eigenvector Centrality[1]}
            Betweenness Centrality (True, Pred): {Betweenness Centrality[0]}, {Betweenness Centrality[1]}
            Global Efficiency (True, Pred): {Global Efficiency[0]:.4f}, {Global Efficiency[1]:.4f}
            Local Efficiency (True, Pred): {Local Efficiency[0]:.4f}, {Local Efficiency[1]:.4f}
            Modularity (True, Pred): {Modularity[0]:.4f}, {Modularity[1]:.4f}
            Global Reaching Centrality (True, Pred): {Global Reaching Centrality[0]:.4f}, {Global Reaching Centrality[1]:.4f}
            Clustering Coefficient (True, Pred): {Clustering Coefficient[0]:.4f}, {Clustering Coefficient[1]:.4f}
            Path Length (True, Pred): {Path Length[0]:.4f}, {Path Length[1]:.4f}
            Small-worldness Omega (True, Pred): {Small-worldness Omega[0]:.4f}, {Small-worldness Omega[1]:.4f}
            Small-worldness Sigma (True, Pred): {Small-worldness Sigma[0]:.4f}, {Small-worldness Sigma[1]:.4f}
            Network Density (True, Pred): {Network Density[0]:.4f}, {Network Density[1]:.4f}
            """.format(**self.results)