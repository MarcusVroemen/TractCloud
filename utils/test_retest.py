"""
Test retest
- Is our method more stable?

Connectomic consistency is calculated as a single score for a dataset reflecting the variation of connectivity across samples, 
where the dataset might consist of connectomes of the same person acquired across time or of different people. In order to calculate 
consistency of a set of connectomes, we first calculated the average similarity of each connectome relative to the rest of the dataset. 
We then considered the mean of these similarity scores to quantify connectomic consistency, 
where higher values indicate that the generated connectomes are consistent across the dataset.

"""
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import cosine
from scipy.linalg import norm
from scipy.optimize import linear_sum_assignment


# Function to compute Pearson correlation between connectomes
def compute_pearson_correlation(connectome1, connectome2):
    # Use the upper triangle of the connectomes
    triu_indices = np.triu_indices_from(connectome1, k=0) # Set to 1 to exclude diagonal
    vec1 = connectome1[triu_indices]
    vec2 = connectome2[triu_indices]
    # Compute Pearson correlation
    return pearsonr(vec1, vec2)

# Function to compute matching accuracy using the Hungarian algorithm
def compute_matching_accuracy(connectome1, connectome2):
    # Get feature vectors for each node (row) in the connectomes
    n_nodes = connectome1.shape[0]
    cost_matrix = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            # Use Euclidean distance as the cost function
            cost_matrix[i, j] = np.linalg.norm(connectome1[i, :] - connectome2[j, :])
    
    # Solve the linear assignment problem using the Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Calculate matching accuracy as the percentage of nodes correctly mapped
    matching_accuracy = np.sum(row_ind == col_ind) / n_nodes * 100
    return matching_accuracy

# Function to compute consistency metrics within one dataset or between two datasets
def compute_connectomic_consistency(connectomes1, names1, connectomes2=None, names2=None, metric='pearson'):
    n_connectomes = len(connectomes1)
    consistency_scores = []
    scores_table = []

    for i in range(n_connectomes):
        similarity_scores = []
        connectome1 = connectomes1[i]

        # Similarity of connectome with connectomes2 or within connectomes1
        if connectomes2 is None:
            # Consistency within a single dataset
            for j in range(n_connectomes):
                if i == j:
                    continue
                connectome2 = connectomes1[j]
                if metric == 'pearson':
                    similarity, p = compute_pearson_correlation(connectome1, connectome2)
                elif metric == 'matching_accuracy':
                    similarity = compute_matching_accuracy(connectome1, connectome2)
                    p=None
                else:
                    raise ValueError(f"Unknown metric: {metric}")
                similarity_scores.append(similarity)
                
                scores_table.append({
                    'Connectome1': names1[i],
                    'Connectome2': names1[j],
                    'Metric': metric,
                    'Similarity': similarity,
                    'p': p
                })
                
        else:
            # Consistency between two datasets (test vs retest)
            if names1[i]==names2[i]:
                connectome2 = connectomes2[i]
                try:
                    if metric == 'pearson':
                        similarity = compute_pearson_correlation(connectome1, connectome2)
                    elif metric == 'matching_accuracy':
                        similarity = compute_matching_accuracy(connectome1, connectome2)
                        p=None
                    else:
                        raise ValueError(f"Unknown metric: {metric}")
                    similarity_scores.append(similarity)

                    scores_table.append({
                        'Connectome1': names1[i],
                        'Connectome2': names2[i],
                        'Metric': metric,
                        'Similarity': similarity,
                        'p': p
                    })
                except:
                    print(names1[i], names2[i])
            
        # Average similarity of the current connectome relative to the rest or the paired connectome
        consistency_scores.append(np.mean(similarity_scores))
        

    # Mean and standard deviation of all connectome consistency scores
    return np.mean(consistency_scores), np.std(consistency_scores), pd.DataFrame(scores_table)

# Load connectomes (example loading mechanism from your existing code)
def load_connectomes(base_path, subject_ids, dataset, atlas):
    connectomes = []
    connectomes_labels = []
    for subject_id in subject_ids:
        try:
            # Load the connectome from the CSV file
            path = os.path.join(base_path, dataset, subject_id, f'output/connectome_matrix_{atlas}.csv')
            connectome = np.loadtxt(path, delimiter=',')
            connectomes.append(connectome)
            connectomes_labels.append(subject_id+dataset)
        except:
            continue
    return connectomes, connectomes_labels

# Example usage
base_path = '/media/volume/HCP_diffusion_MV'
subject_ids_txt = "/media/volume/HCP_diffusion_MV/test_retest_subjects.txt"
with open(subject_ids_txt, 'r') as file:
    subject_ids = [line.strip() for line in file]

atlases = ["aparc+aseg", "aparc.a2009s+aseg"]
datasets = ['test', 'retest']

# Load the connectomes for test and retest datasets
connectomes_test, names_test = load_connectomes(base_path, subject_ids, 'test', atlas=atlases[0])
connectomes_retest, names_retest = load_connectomes(base_path, subject_ids, 'retest', atlas=atlases[0])

# Consistency within the test dataset
test_pearson_mean, test_pearson_std, test_pearson_table = compute_connectomic_consistency(connectomes_test, names_test, metric='pearson')
print(f'Test Pearson consistency: {test_pearson_mean:.3f} ± {test_pearson_std:.3f}')

# Consistency within the retest dataset
retest_pearson_mean, retest_pearson_std, retest_pearson_table = compute_connectomic_consistency(connectomes_retest, names_retest, metric='pearson')
print(f'Retest Pearson consistency: {retest_pearson_mean:.3f} ± {retest_pearson_std:.3f}')

# Consistency between test and retest datasets
test_retest_pearson_mean, test_retest_pearson_std, test_retest_pearson_table = compute_connectomic_consistency(connectomes_test, names_test, connectomes_retest, names_retest, metric='pearson')
print(f'Test-Retest Pearson consistency: {test_retest_pearson_mean:.3f} ± {test_retest_pearson_std:.3f}')

if False:
    # Consistency within the test dataset (matching accuracy)
    test_ma_mean, test_ma_std, test_ma_table = compute_connectomic_consistency(connectomes_test, names_test, metric='matching_accuracy')
    print(f'Test Matching Accuracy consistency: {test_ma_mean:.3f} ± {test_ma_std:.3f}')

    # Consistency within the retest dataset (matching accuracy)
    retest_ma_mean, retest_ma_std, retest_ma_table = compute_connectomic_consistency(connectomes_retest, names_retest, metric='matching_accuracy')
    print(f'Retest Matching Accuracy consistency: {retest_ma_mean:.3f} ± {retest_ma_std:.3f}')

    # Consistency between test and retest datasets (matching accuracy)
    test_retest_ma_mean, test_retest_ma_std, test_retest_ma_table = compute_connectomic_consistency(connectomes_test, names_test, connectomes_retest, names_retest, metric='matching_accuracy')
    print(f'Test-Retest Matching Accuracy consistency: {test_retest_ma_mean:.3f} ± {test_retest_ma_std:.3f}')

# print(test_pearson_table)
# print(retest_pearson_table)
print(test_retest_pearson_table)
# Consistency within one dataset: variation of connectivity across samples
# Consistency within the test and retest sets
# Consisency between test and retest samples
# report mean+std

