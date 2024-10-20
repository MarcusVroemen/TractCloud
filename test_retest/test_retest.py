import os
import numpy as np
from network_measures import compute_graph_measures
import pandas as pd
from scipy.stats import f
from sklearn.utils import resample
from scipy.stats import pearsonr, spearmanr
from scipy.optimize import linear_sum_assignment  # For Hungarian algorithm

# Define the paths to the directories
base_dirs = ['/media/volume/HCP_diffusion_MV/retest', '/media/volume/HCP_diffusion_MV/test']

#TODO could add consistency in the form of correlation and matching accuracy
# Connectomic consistency: a systematic stability analysis of structural and functional connectivity Yusuf Osmanlıoğlu, Jacob A. Alappatt, Drew Parker, Ragini Verma
# test_retest_old.py

def search_and_load_csv(base_dirs):
    """Function to search for the CSV files (connectomes), load and store them"""
    connectome_data = []
    for base_dir in base_dirs:
        dataset_type = 'retest' if 'retest' in base_dir else 'test'
        
        for root, dirs, files in os.walk(base_dir):
            # Check for the connectome_pred.csv and connectome_true.csv files
            if 'connectome_pred.csv' in files and 'connectome_true.csv' in files:
                subject_id = root.split('/')[-3]

                # Load connectome_pred.csv
                pred_path = os.path.join(root, 'connectome_pred.csv')
                pred_data = np.loadtxt(pred_path, delimiter=',')
                
                # Load connectome_true.csv
                true_path = os.path.join(root, 'connectome_true.csv')
                true_data = np.loadtxt(true_path, delimiter=',')
                
                # Store the information in the list
                connectome_data.append({
                    'subject_id': subject_id,
                    'dataset_type': dataset_type,
                    'pred_data': pred_data,
                    'true_data': true_data
                })
    if False:
        for entry in connectome_data:
            print(f"Subject ID: {entry['subject_id']}, Dataset: {entry['dataset_type']}")
            print(f"True Shape: {entry['true_data'].shape}, Pred Shape: {entry['pred_data'].shape}\n")
    
    # Only keep subjects with both the test and retest data    
    return filter_subjects_with_both(connectome_data)

def filter_subjects_with_both(connectome_data):
    # Create a dictionary to track the presence of subjects in both datasets
    subject_tracker = {}
    
    # Populate the tracker with test and retest occurrences
    for entry in connectome_data:
        subject_id = entry['subject_id']
        dataset_type = entry['dataset_type']
        
        if subject_id not in subject_tracker:
            subject_tracker[subject_id] = {'test': False, 'retest': False}
        
        # Update the tracker based on dataset type
        subject_tracker[subject_id][dataset_type] = True

    # Collect subject IDs that are missing either test or retest data
    deleted_subjects = [
        subject_id for subject_id, datasets in subject_tracker.items()
        if not (datasets['test'] and datasets['retest'])
    ]

    print(f"Deleted Subjects: {deleted_subjects}")

    # Keep only subjects that have both test and retest entries
    filtered_data = [
        entry for entry in connectome_data
        if subject_tracker[entry['subject_id']]['test'] and subject_tracker[entry['subject_id']]['retest']
    ]
    print(f"Number of Subjects Remaining: {len(filtered_data)//2}")  # Divided by 2 as each subject has test and retest entries

    return filtered_data

# Function to process all connectomes and compute graph metrics
def process_connectomes(connectome_data):
    metrics_list = []

    for entry in connectome_data:

        subject_id = entry['subject_id']
        dataset_type = entry['dataset_type']
        
        # Compute metrics for both true and predicted connectomes
        for connectome_type, connectome in zip(['true', 'pred'], [entry['true_data'], entry['pred_data']]):
            # metrics = compute_graph_measures(connectome)
            metrics = {}
            metrics.update({
                'subject_id': subject_id,
                'dataset_type': dataset_type,
                'connectome_type': connectome_type,
                'connectome': connectome
            })
            metrics_list.append(metrics)
            print(subject_id, dataset_type, connectome_type)
    
    # Create a DataFrame from the list of metrics
    metrics_df = pd.DataFrame(metrics_list)
    
    return metrics_df

# Load the data and compute the metrics
# connectome_data = search_and_load_csv(base_dirs)
# metrics_df = process_connectomes(connectome_data)

# metrics_df.to_csv('metrics.csv')
metrics_df = pd.read_csv('metrics.csv')

########################################################


def compute_icc(test_data, retest_data):
    """
    Compute ICC (Intraclass Correlation Coefficient) for a given graph metric and data type.

    Returns:
    - float: ICC value for the given metric.
    - float: p-value for the ICC test.
    """
    # Compute mean scores across test and retest for each subject
    subject_means = np.mean([test_data, retest_data], axis=0)
    
    # Compute grand mean across all subjects and sessions
    grand_mean = np.mean(subject_means)
    
    # Mean Square Between Subjects (MSB) & Mean Square Within Subjects (MSW)
    MSB = np.sum((subject_means - grand_mean)**2) / (len(subject_means) - 1)
    MSW = (np.sum((test_data - subject_means)**2) + np.sum((retest_data - subject_means)**2)) / (2 * (len(subject_means) - 1))
    
    # ICC calculation
    k = 2 # Connectomes per subjects
    ICC = (MSB - MSW) / (MSB + (k - 1) * MSW)
    
    # Compute F-statistic for significance testing
    F_stat = MSB / MSW
    
    # Degrees of freedom for between and within subjects
    df_between = len(subject_means) - 1
    df_within = df_between
    
    # Compute p-value from F-distribution
    p_value = 1 - f.cdf(F_stat, df_between, df_within)
    
    return ICC, p_value

def compute_within_between_subject_differences(test_data, retest_data):
    """
    Compute within-subject and between-subject differences for a metric.

    Returns:
    - delta_ws (float): Mean of within-subject differences.
    - delta_bs (float): Mean of between-subject differences.
    """
    N = len(test_data)
    
    # Compute the mean value across both test and retest datasets
    mean_value = np.mean(np.concatenate([test_data, retest_data]))
    
    # Compute within-subject differences (delta_ws) as absolute difference
    delta_ws = np.mean(np.abs(test_data - retest_data))
    
    # Compute between-subject differences (delta_bs)
    delta_bs = np.mean([
        np.mean(np.abs(test_data[i] - retest_data[np.arange(N) != i]))
        for i in range(N)
    ])
    
    # Convert the differences to percentages relative to the mean value
    delta_ws_percentage = (delta_ws / mean_value) * 100
    delta_bs_percentage = (delta_bs / mean_value) * 100
    
    return delta_ws_percentage, delta_bs_percentage

def percentile_bootstrap(test_data, retest_data, n_iterations=5000):
    """
    Perform percentile bootstrap to compare within- and between-subject differences.
    
    Parameters:
    - n_iterations (int): Number of bootstrap iterations to perform.
    
    Returns:
    - p_value (float): p-value computed from the bootstrap.
    """
    N = len(test_data)
    original_diff = np.mean([
        np.mean(np.abs(test_data[i] - retest_data[np.arange(N) != i])) - np.abs(test_data[i] - retest_data[i])
        for i in range(N)
    ])
    
    boot_diffs = []
    for _ in range(n_iterations):
        resampled_indices = resample(np.arange(N), replace=True)
        resampled_test = test_data[resampled_indices]
        resampled_retest = retest_data[resampled_indices]
        
        delta_ws_resampled, delta_bs_resampled = compute_within_between_subject_differences(resampled_test, resampled_retest)
        boot_diffs.append(delta_bs_resampled - delta_ws_resampled)
    
    boot_diffs = np.array(boot_diffs)
    p_value = np.mean(boot_diffs > original_diff)
    
    return p_value

def compute_cv_percentage(test_data, retest_data):
    """
    Compute the percentage coefficient of variation (CV%) for a metric.

    Returns:
    - cv_percentage (float): CV% value for the given metric.
    """
    # Compute the intrasubject standard deviation (SD)
    intrasubject_sd = np.std([test_data, retest_data], axis=0)
    
    # Compute the overall measurement mean
    overall_mean = np.mean(np.concatenate([test_data, retest_data]))
    
    # Calculate CV% as (intrasubject SD / overall mean) * 100
    cv_percentage = (np.mean(intrasubject_sd) / overall_mean) * 100
    
    return cv_percentage

def test_retest_analysis(metrics_df, metrics):
    """
    Compute ICC and p-value for all graph metrics for both 'true' and 'pred' data types.
    
    Parameters:
    - metrics_df (pd.DataFrame): DataFrame containing all graph metrics for test and retest.
    - metrics (list): List of metric names to compute ICC for.
    - connectome_types (list): List of data types to compute ICC for (e.g., ['true', 'pred']).
    
    Returns:
    - pd.DataFrame: DataFrame containing ICC values and p-values for each metric and data type.
    """
    results = []

    for metric in metrics:
        temp_data = {'Metric': metric}

        for data_type in ['true', 'pred']:
            # Filter data for true and pred, and separate test and retest
            df_filtered = metrics_df[metrics_df['connectome_type'] == data_type]
            test_data = df_filtered[df_filtered['dataset_type'] == 'test'][metric].values
            retest_data = df_filtered[df_filtered['dataset_type'] == 'retest'][metric].values
            
            print(f'Metric {metric} {data_type} has {np.sum(np.isnan(test_data))} nans in test and {np.sum(np.isnan(retest_data))} nans in retest')            
            test_data[np.isnan(test_data)] = np.nanmean(test_data)
            retest_data[np.isnan(retest_data)] = np.nanmean(retest_data)

            # Mean and standard deviation
            temp_data[f'{data_type} test mean±std'] = f"{np.nanmean(test_data):1.3f}±{np.nanstd(test_data):1.4f}"
            temp_data[f'{data_type} retest mean±std'] = f"{np.nanmean(retest_data):1.3f}±{np.nanstd(retest_data):1.4f}"
            # temp_data[f'{data_type} test mean±std'] = f"{np.format_float_scientific(np.mean(test_data), precision=2)}±{np.format_float_scientific(np.std(test_data), precision=2)}"
            # temp_data[f'{data_type} retest mean±std'] = f"{np.format_float_scientific(np.mean(retest_data), precision=2)}±{np.format_float_scientific(np.std(retest_data), precision=2)}"
            
            # Intraclass correlation coeffient (ICC) and p-value
            icc, p_value = compute_icc(test_data, retest_data)
            temp_data[f'{data_type} ICC'] = f"{icc:1.3f}"
            temp_data[f'{data_type} ICC p value'] = f"{p_value:1.4f}"
            
            # Coefficient of variation (CV%) percentage
            cv_percentage = compute_cv_percentage(test_data, retest_data)
            temp_data[f'{data_type} CV%'] = f"{cv_percentage:1.4f}"
        
            # Within-subject (WS) and between-subject (BS) percentage differences
            delta_ws, delta_bs = compute_within_between_subject_differences(test_data, retest_data)
            temp_data[f'{data_type} delta_ws %'] = f"{delta_ws:1.2f}"
            temp_data[f'{data_type} delta_bs %'] = f"{delta_bs:1.2f}"
            
            # Bootstrap p-value for between-subject differences
            p_value_bootstrap = percentile_bootstrap(test_data, retest_data)
            temp_data[f'{data_type} BS p value %'] = f"{p_value_bootstrap:1.4f}"

        results.append(temp_data)
    
    df = pd.DataFrame(results)
    
    return df

# Assuming 'metrics_df' is the DataFrame from the previous step, which includes all graph metrics for test and retest
metrics = [
    'Characteristic Path Length (L)', 'Mean Clustering Coefficient (C)', 
    # 'Small-worldness (Omega)', 'Small-worldness (Sigma)', 
    'Mean Normalized Betweenness Centrality (B)', 
    'Mean Global Efficiency (E)', 'Mean Local Efficiency', 'Mean Degree', 'Mean Strength', 'Modularity (MOD)'
]

# Compute ICC and p-values for all metrics
results_test_retest = test_retest_analysis(metrics_df, metrics)
print(results_test_retest)
results_test_retest.to_csv('results_test_retest.csv')


# Correlation computation
def compute_pearson_correlation(connectome1, connectome2):
    triu_indices = np.triu_indices_from(connectome1, k=0)  # Use the upper triangle of the connectomes
    vec1 = connectome1[triu_indices]
    vec2 = connectome2[triu_indices]
    return pearsonr(vec1, vec2)

# Mean Squared Error computation
def compute_mse(connectome1, connectome2):
    return np.mean((connectome1 - connectome2) ** 2)

# Matching Accuracy computation
def compute_matching_accuracy(connectome1, connectome2):
    cost_matrix = np.zeros((connectome1.shape[0], connectome2.shape[0]))
    
    # Fill the cost matrix based on Euclidean distances between connectivity patterns
    for i in range(connectome1.shape[0]):
        for j in range(connectome2.shape[0]):
            cost_matrix[i, j] = np.linalg.norm(connectome1[i] - connectome2[j])
    
    # Solve the assignment problem using the Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Matching accuracy: percentage of nodes correctly matched
    correct_matches = sum(row_ind == col_ind)
    total_nodes = connectome1.shape[0]
    
    return (correct_matches / total_nodes) * 100

def correlation(metrics_df):
    """
    """
    results = []

    for data_type in ['true', 'pred']:
        temp_data = {'data_type': data_type}
        
        # Filter data for true and pred, and separate test and retest
        df_filtered = metrics_df[metrics_df['connectome_type'] == data_type]
        test_data = df_filtered[df_filtered['dataset_type'] == 'test']['connectome'].values
        retest_data = df_filtered[df_filtered['dataset_type'] == 'retest']['connectome'].values
        
        BS_R_test, BS_R_retest, WS_R = [], [], []
        BS_MSE_test, BS_MSE_retest, WS_MSE = [], [], []
        BS_MA_test, BS_MA_retest, WS_MA = [], [], []
        
        for i in range(len(test_data)):
            for j in range(len(test_data)):
                
                if i != j:
                    r_test, _ = compute_pearson_correlation(test_data[i], test_data[j])
                    BS_R_test.append(r_test)
                    r_retest, _ = compute_pearson_correlation(retest_data[i], retest_data[j])
                    BS_R_retest.append(r_retest)
                    
                    mse_test = compute_mse(test_data[i], test_data[j])
                    BS_MSE_test.append(mse_test)
                    mse_retest = compute_mse(retest_data[i], retest_data[j])
                    BS_MSE_retest.append(mse_retest)
                    
                    ma_test = compute_matching_accuracy(test_data[i], test_data[j])
                    BS_MA_test.append(ma_test)
                    ma_retest = compute_matching_accuracy(retest_data[i], retest_data[j])
                    BS_MA_retest.append(ma_retest)
                else:
                    r, _ = compute_pearson_correlation(test_data[i], retest_data[j])
                    WS_R.append(r)
                    
                    mse = compute_mse(test_data[i], retest_data[j])
                    WS_MSE.append(mse)
                    
                    ma = compute_matching_accuracy(test_data[i], retest_data[j])
                    WS_MA.append(ma)
        
        # Store results for this data type
        temp_data[f'BS r value (test)'] = f"{np.mean(BS_R_test):1.3f}±{np.std(BS_R_test):1.4f}"
        temp_data[f'BS r value (retest)'] = f"{np.mean(BS_R_retest):1.3f}±{np.std(BS_R_retest):1.4f}"
        temp_data[f'WS r value'] = f"{np.mean(WS_R):1.3f}±{np.std(WS_R):1.4f}"
        
        temp_data[f'BS mse (test) *1e6'] = f"{np.mean(BS_MSE_test)/1e6:1.3f}±{np.std(BS_MSE_test)/1e6:1.4f}"
        temp_data[f'BS mse (retest) *1e6'] = f"{np.mean(BS_MSE_retest)/1e6:1.3f}±{np.std(BS_MSE_retest)/1e6:1.4f}"
        temp_data[f'WS mse *1e6'] = f"{np.mean(WS_MSE)/1e6:1.3f}±{np.std(WS_MSE)/1e6:1.4f}"

        temp_data[f'BS ma (test) %'] = f"{np.mean(BS_MA_test):1.3f}±{np.std(BS_MA_test):1.4f}"
        temp_data[f'BS ma (retest) %'] = f"{np.mean(BS_MA_retest):1.3f}±{np.std(BS_MA_retest):1.4f}"
        temp_data[f'WS ma %'] = f"{np.mean(WS_MA):1.3f}±{np.std(WS_MA):1.4f}"

        results.append(temp_data)
    
    df = pd.DataFrame(results)
    return df


# correlation_test_retest = correlation(metrics_df)
# print(correlation_test_retest)
# correlation_test_retest.to_csv('correlation_test_retest.csv')