import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import cosine
from scipy.linalg import norm
from scipy.stats import pearsonr
out_path = '/media/volume/HCP_diffusion_MV/connectomes_eval'
# Paths and subjects
base_path = '/media/volume/HCP_diffusion_MV/data/'

# Path to the subject IDs text file
# subject_ids_txt = "/media/volume/connectomes_MV/TractCloud/tractography/subjects_tractography_output_50test.txt"
# with open(subject_ids_txt, 'r') as file:
#     subject_ids = [line.strip() for line in file]

# print(subject_ids)
# subject_ids = ['993675', '793465']

# Dictionaries to store connectomes
connectomes_true = {}
connectomes_pred = {}

# Load connectomes
# for subject_id in subject_ids:
#     true_path = os.path.join(base_path, subject_id, 'TractCloud/connectome/connectome_true.csv')
#     pred_path = os.path.join(base_path, subject_id, 'TractCloud/connectome/connectome_pred.csv')
    
#     connectome_true = np.loadtxt(true_path, delimiter=',')
#     connectome_pred = np.loadtxt(pred_path, delimiter=',')
    
#     connectomes_true[subject_id] = connectome_true
#     connectomes_pred[subject_id] = connectome_pred

# Function for pairwise comparison
def compare_connectomes(connectomes_true, connectomes_pred):
    results = {}
    subject_ids = list(connectomes_true.keys())
    
    for i, subj_id1 in enumerate(subject_ids):
        for subj_id2 in subject_ids:
            # Compute metrics for same-subject comparison
            corr_same, _ = pearsonr(connectomes_true[subj_id1].flatten(), connectomes_pred[subj_id1].flatten())
            cosine_same = 1 - cosine(connectomes_true[subj_id1].flatten(), connectomes_pred[subj_id1].flatten())
            frobenius_same = norm(connectomes_true[subj_id1] - connectomes_pred[subj_id1], 'fro')
            
            # Cross-subject comparison
            # if subj_id1!=subj_id2:
            corr_cross, _ = pearsonr(connectomes_true[subj_id1].flatten(), connectomes_pred[subj_id2].flatten())
            cosine_cross = 1 - cosine(connectomes_true[subj_id1].flatten(), connectomes_pred[subj_id2].flatten())
            frobenius_cross = norm(connectomes_true[subj_id1] - connectomes_pred[subj_id2], 'fro')
            # else:
            #     corr_cross = np.nan
            #     cosine_cross = np.nan
            #     frobenius_cross = np.nan
                
            results[(subj_id1, subj_id2)] = {
                'Correlation (Same)': corr_same,
                'Cosine Similarity (Same)': cosine_same,
                'Frobenius Norm (Same)': frobenius_same,
                'Correlation (Cross)': corr_cross,
                'Cosine Similarity (Cross)': cosine_cross,
                'Frobenius Norm (Cross)': frobenius_cross
            }
    
    return results

# Perform the comparisons
# comparison_results = compare_connectomes(connectomes_true, connectomes_pred)

# # Pretty-print the results
# import pprint
# pprint.pprint(comparison_results)


# Function to plot heatmap
def plot_heatmap(matrix, title):
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, cmap='coolwarm', square=True, cbar_kws={"shrink": 0.8})
    plt.title(title)
    plt.savefig(os.path.join(out_path, f'{title}.png'), bbox_inches='tight', dpi=500)
    plt.close()

# Create and plot heatmaps for Pearson Correlation, Cosine Similarity, and Frobenius Norm
# for metric in ['Correlation (Same)', 'Cosine Similarity (Same)', 'Frobenius Norm (Same)']:
#     matrix = pd.DataFrame(
#         [[comparison_results[(s1, s2)][metric] for s2 in subject_ids] for s1 in subject_ids],
#         index=subject_ids, columns=subject_ids
#     )
#     plot_heatmap(matrix, f'{metric} Heatmap')

# Scatter plot for correlation vs cosine similarity
def plot_scatter(subject_ids, comparison_results, metric1, metric2):
    plt.figure(figsize=(8, 6))
    for subj_id1 in subject_ids:
        for subj_id2 in subject_ids:
            plt.scatter(
                comparison_results[(subj_id1, subj_id2)][metric1],
                comparison_results[(subj_id1, subj_id2)][metric2],
                label=f'{subj_id1} vs {subj_id2}'
            )
    plt.xlabel(metric1)
    plt.ylabel(metric2)
    plt.title(f'{metric1} vs {metric2} (Same Subjects)')
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.savefig(s.path.join(out_path, f'{metric1}_vs_{metric2}.png'), bbox_inches='tight', dpi=500)
    plt.close()

# Plot scatter plots for each metric pair
# plot_scatter(subject_ids, comparison_results, 'Correlation (Same)', 'Cosine Similarity (Same)')
# plot_scatter(subject_ids, comparison_results, 'Correlation (Same)', 'Frobenius Norm (Same)')
# plot_scatter(subject_ids, comparison_results, 'Cosine Similarity (Same)', 'Frobenius Norm (Same)')


def compare_connectomes(connectomes_test, connectomes_retest):
    results = {}
    subject_ids = list(connectomes_test.keys())
    
    for i, subj_id1 in enumerate(subject_ids):
        for subj_id2 in subject_ids:
            # Compute metrics for same-subject comparison
            corr_in, _ = pearsonr(connectomes_test[subj_id1].flatten(), connectomes_retest[subj_id1].flatten())
            corr_same, _ = pearsonr(connectomes_test[subj_id1].flatten(), connectomes_test[subj_id1].flatten())
            cosine_same = 1 - cosine(connectomes_test[subj_id1].flatten(), connectomes_retest[subj_id1].flatten())
            frobenius_same = norm(connectomes_test[subj_id1] - connectomes_retest[subj_id1], 'fro')
            
            # Cross-subject comparison
            if subj_id1!=subj_id2:
                corr_cross, _ = pearsonr(connectomes_test[subj_id1].flatten(), connectomes_retest[subj_id2].flatten())
                cosine_cross = 1 - cosine(connectomes_test[subj_id1].flatten(), connectomes_retest[subj_id2].flatten())
                frobenius_cross = norm(connectomes_test[subj_id1] - connectomes_retest[subj_id2], 'fro')
            else:
                corr_cross = np.nan
                cosine_cross = np.nan
                frobenius_cross = np.nan
                
            results[(subj_id1, subj_id2)] = {
                'Correlation test-test': corr_in,
                'Correlation test-retest': corr_same,
                'Cosine Similarity (Same)': cosine_same,
                'Frobenius Norm (Same)': frobenius_same,
                'Correlation (Cross)': corr_cross,
                'Cosine Similarity (Cross)': cosine_cross,
                'Frobenius Norm (Cross)': frobenius_cross
            }
    
    return results

for subject_id in ['917255', '877168']: #'861456'
    true_path = os.path.join('/media/volume/HCP_diffusion_MV/test/', subject_id, 'output/connectome_matrix_aparc+aseg.csv')
    pred_path = os.path.join('/media/volume/HCP_diffusion_MV/retest/', subject_id, 'output/connectome_matrix_aparc+aseg.csv')
    
    connectome_true = np.loadtxt(true_path, delimiter=',')
    connectome_pred = np.loadtxt(pred_path, delimiter=',')
    
    connectomes_true[subject_id] = connectome_true
    connectomes_pred[subject_id] = connectome_pred

comparison_results = compare_connectomes(connectomes_true, connectomes_pred)

# Pretty-print the results
import pprint
pprint.pprint(comparison_results)


