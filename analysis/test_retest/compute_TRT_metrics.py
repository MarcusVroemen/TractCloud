import pandas as pd
import numpy as np
from scipy.stats import pearsonr, variation
from pingouin import intraclass_corr
import os

# Load the dataset
cwd = os.getcwd()
csv_path = os.path.join(cwd, "data/TRT_combined_aggregated_metrics.csv")
data = pd.read_csv(csv_path)

# Map atlas names
atlas_mapping = {"aparc+aseg": "84 ROIs", "aparc.a2009s+aseg": "164 ROIs"}
data['atlas'] = data['atlas'].replace(atlas_mapping)

# Define graph measures
measures = ['Modularity', 'Clustering Coefficient', 'Path Length', 'Global Efficiency', 'Local Efficiency',
            'Assortativity', 'Global Reaching Centrality', 'Network Density']

# Initialize results
correlation_results = []
icc_cv_results = []

# Create output paths
output_dir = os.path.join(cwd, "results")
os.makedirs(output_dir, exist_ok=True)

# Process each atlas
for atlas in data['atlas'].unique():
    atlas_data = data[data['atlas'] == atlas]
    
    # Test-retest correlation for each subject
    for subject_id in atlas_data['subject_id'].unique():
        subject_data = atlas_data[atlas_data['subject_id'] == subject_id]
        if len(subject_data) < 2:
            print(f"Skipping subject {subject_id} in {atlas}: only one session available.")
            continue
        
        session_1 = subject_data.iloc[0]
        session_2 = subject_data.iloc[1]
        
        # Calculate inter- and intra-subject correlations for each measure
        for measure in measures:
            # Extract ground truth and predicted values
            gt_session_1 = session_1[f"{measure} true"]
            gt_session_2 = session_2[f"{measure} true"]
            pred_session_1 = session_1[f"{measure} pred"]
            pred_session_2 = session_2[f"{measure} pred"]
            
            # Compute intrasubject correlations
            intra_gt_corr, _ = pearsonr([gt_session_1], [gt_session_2])
            intra_pred_corr, _ = pearsonr([pred_session_1], [pred_session_2])
            
            # Add to results
            correlation_results.append({
                'Atlas': atlas,
                'Subject ID': subject_id,
                'Measure': measure,
                'Intra-Subject GT Correlation': intra_gt_corr,
                'Intra-Subject Pred Correlation': intra_pred_corr
            })
    
    # ICC and CV% across all subjects for each measure
    for measure in measures:
        measure_data = atlas_data[[f"{measure} true", f"{measure} pred"]]
        measure_data_melted = measure_data.melt(var_name='Type', value_name='Value')
        
        # Compute ICC
        icc = intraclass_corr(data=measure_data_melted, targets='Type', raters='Value', ratings='Value')
        icc_value = icc['ICC'][0]
        
        # Compute CV%
        cv_true = variation(measure_data[f"{measure} true"].dropna()) * 100
        cv_pred = variation(measure_data[f"{measure} pred"].dropna()) * 100
        
        # Add to results
        icc_cv_results.append({
            'Atlas': atlas,
            'Measure': measure,
            'ICC': icc_value,
            'CV% True': cv_true,
            'CV% Pred': cv_pred
        })

# Save results to CSV
correlation_results_df = pd.DataFrame(correlation_results)
correlation_results_df.to_csv(os.path.join(output_dir, "correlation_results.csv"), index=False)

icc_cv_results_df = pd.DataFrame(icc_cv_results)
icc_cv_results_df.to_csv(os.path.join(output_dir, "icc_cv_results.csv"), index=False)

# Print completion message
print(f"Analysis completed. Results saved in {output_dir}.")
