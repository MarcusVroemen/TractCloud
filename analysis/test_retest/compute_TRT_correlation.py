import os
import pandas as pd
import numpy as np
from itertools import combinations

# Load subject IDs
subject_file = '/media/volume/MV_HCP/subjects_tractography_output_1000_test.txt'
with open(subject_file, 'r') as f:
    subject_ids = [line.strip() for line in f.readlines()]

# Atlases to process
atlases = ["aparc+aseg", "aparc.a2009s+aseg"]

# Function to extract upper triangular values (exclude diagonal)
def upper_triangle(matrix):
    return matrix[np.triu_indices_from(matrix, k=1)]  # Excludes diagonal with k=1

# Function to process intrasubject correlations
def compute_intrasubject_correlation(subject_id, atlas):
    pred_path = f'/media/volume/MV_HCP/HCP_MRtrix/pred/{subject_id}/TractCloud/connectome_{atlas}_pred.csv'
    true_path = f'/media/volume/MV_HCP/HCP_MRtrix/true/{subject_id}/TractCloud/connectome_{atlas}_true.csv'

    if os.path.exists(pred_path) and os.path.exists(true_path):
        pred_matrix = pd.read_csv(pred_path, header=None).values
        true_matrix = pd.read_csv(true_path, header=None).values

        pred_upper = upper_triangle(pred_matrix)
        true_upper = upper_triangle(true_matrix)

        correlation = np.corrcoef(pred_upper, true_upper)[0, 1]
        return {"subject_id": subject_id, "atlas": atlas, "intrasubject_correlation": correlation}
    else:
        print(f"Warning: Files for subject {subject_id} and atlas {atlas} not found.")
        return None

# Function to compute intersubject correlations
def compute_intersubject_correlations(atlas, true_matrices):
    intersubject_correlations = []
    for (sub1, mat1), (sub2, mat2) in combinations(true_matrices.items(), 2):
        mat1_upper = upper_triangle(mat1)
        mat2_upper = upper_triangle(mat2)
        correlation = np.corrcoef(mat1_upper, mat2_upper)[0, 1]
        intersubject_correlations.append({"subject_1": sub1, "subject_2": sub2, "atlas": atlas, "correlation": correlation})
    return intersubject_correlations

# DataFrames to store results
intrasubject_df = pd.DataFrame()
intersubject_df = pd.DataFrame()

# Dictionary to store true connectomes for intersubject correlation
true_connectomes = {atlas: {} for atlas in atlases}

# Process all subjects sequentially for intrasubject correlation
for atlas in atlases:
    for subject_id in subject_ids:
        result = compute_intrasubject_correlation(subject_id, atlas)
        if result is not None:
            intrasubject_df = pd.concat([intrasubject_df, pd.DataFrame([result])], ignore_index=True)
            # Store true connectomes for intersubject computation
            true_path = f'/media/volume/MV_HCP/HCP_MRtrix/true/{subject_id}/TractCloud/connectome_{atlas}_true.csv'
            true_connectomes[atlas][subject_id] = pd.read_csv(true_path, header=None).values

# Compute intersubject correlations for each atlas
for atlas in atlases:
    correlations = compute_intersubject_correlations(atlas, true_connectomes[atlas])
    intersubject_df = pd.concat([intersubject_df, pd.DataFrame(correlations)], ignore_index=True)

# Save results
output_dir = '/media/volume/HCP_diffusion_MV/TractCloud/analysis/data'
os.makedirs(output_dir, exist_ok=True)

intrasubject_df.to_csv(os.path.join(output_dir, 'intrasubject_correlation.csv'), index=False)
intersubject_df.to_csv(os.path.join(output_dir, 'intersubject_correlation.csv'), index=False)

print("Correlation computation completed.")
print(f"Intrasubject correlations saved to {os.path.join(output_dir, 'intrasubject_correlation.csv')}")
print(f"Intersubject correlations saved to {os.path.join(output_dir, 'intersubject_correlation.csv')}")

# Compute mean and std for intrasubject correlations
intrasubject_stats = intrasubject_df.groupby('atlas')['intrasubject_correlation'].agg(['mean', 'std']).reset_index()

# Compute mean and std for intersubject correlations
intersubject_stats = intersubject_df.groupby('atlas')['correlation'].agg(['mean', 'std']).reset_index()

# Print formatted results
print("\n--- Results ---")
for _, row in intrasubject_stats.iterrows():
    print(f"{row['atlas']} - Intrasubject: Mean={row['mean']:.3f}, Std={row['std']:.3f}")

for _, row in intersubject_stats.iterrows():
    print(f"{row['atlas']} - Intersubject: Mean={row['mean']:.3f}, Std={row['std']:.3f}")
