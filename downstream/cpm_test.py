from CPM.python.cpm import *
import numpy as np
import pandas as pd
import os
import logging

# Define atlas and directories
atlas = 'aparc+aseg' #'aparc.a2009s+aseg'
data_dir = '/media/volume/HCP_diffusion_MV/data'
corr_methods = ['pearson', 'spearman']  # Add all correlation methods to test
cvtypes=['splithalf', '5k']
significances=[0.001, 0.1]

# Set up logging
def setup_logging(logfile='results/train_cpm.log'):
    logging.basicConfig(filename=logfile, 
                        filemode='w',  # Overwrite log file every run
                        format='%(message)s',  # %(asctime)s - 
                        level=logging.INFO)    
setup_logging(f'results/gridsearch_{atlas}.log')

# Read subject IDs from file
def read_subject_ids(input_file):
    with open(input_file, 'r') as file:
        dirnames = [int(line.strip()) for line in file.readlines()]
    return dirnames
# subject_ids_list = '/media/volume/HCP_diffusion_MV/subjects_tractography_output_150.txt'
subject_ids_list = '/media/volume/connectomes_MV/TractCloud/downstream/test_subjects.txt'
subject_ids = read_subject_ids(subject_ids_list)

def normalize_array(arr):
    norm_arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
    return norm_arr

# Load matrices
X={}
matrices = []
for subject_id in subject_ids:
    csv_file = f'/media/volume/HCP_diffusion_MV/test/{subject_id}/TractCloud/connectome/connectome_true.csv'
    # csv_file = f'/media/volume/HCP_diffusion_MV/data/{subject_id}/output/connectome_matrix_aparc+aseg.csv'
    matrix = np.loadtxt(csv_file, delimiter=',')
    # matrices.append(normalize_array(matrix))
    matrices.append(matrix)
X['true'] = np.stack(matrices, axis=2)

matrices = []
for subject_id in subject_ids:
    csv_file = f'/media/volume/HCP_diffusion_MV/test/{subject_id}/TractCloud/connectome/connectome_pred.csv'
    matrix = np.loadtxt(csv_file, delimiter=',')
    # matrices.append(normalize_array(matrix))
    matrices.append(matrix)
X['pred'] = np.stack(matrices, axis=2)

# Load behavioral data and features
base_path = '/media/volume/connectomes_MV/TractCloud/downstream/'
Y_raw = pd.read_csv(os.path.join(base_path, 'HCP_behavioral_data.csv'))
Y = Y_raw[Y_raw['Subject'].isin(subject_ids)]

with open('features.txt', 'r') as file:
    features = [line.strip() for line in file.readlines()]

cvtype=cvtypes[0]
corr_method=corr_methods[0]
# significance=significances[1]


# import pdb
# pdb.set_trace()
results_list=[]
# for feature in features:  # Loop over selected features
for feature in ['ReadEng_Unadj', 'PicVocab_Unadj']:  # Loop over selected features
    for data in ['true', 'pred']:
        # for cvtype in cvtypes:
            for corr_method in corr_methods:
                for significance in significances:
                    # Prepare the target variable (phenotype)

                    try:
                        # if feature=='Age':

                        y = np.asarray(Y[feature].fillna(Y[feature].mean()))
                        
                        Rpos_mean, ppos_mean, Rneg_mean, pneg_mean = run_validate(X[data], y, cvtype=cvtype, corr_method=corr_method, significance=significance)

                        # Log the results for each feature
                        logging.info(f'-----{feature} ({corr_method})-----')
                        logging.info(f'POS R: {Rpos_mean:.2f} p: {ppos_mean:.2f}')
                        logging.info(f'NEG R: {Rneg_mean:.2f} p: {pneg_mean:.2f}')
                        if ppos_mean <= 0.05 or pneg_mean <= 0.05:
                            logging.info(f'{feature} is significant!')
                        logging.info(f'----------------')

                        # Check significance
                        significant_pos = ppos_mean <= 0.05
                        significant_neg = pneg_mean <= 0.05

                        # Collect results in a list
                        results_list.append({
                            'True/Pred': data,
                            'Feature': feature,
                            'Corr_Method': corr_method,
                            'Significance': significance,
                            'cvtype' : cvtype,
                            'Rpos_mean': Rpos_mean,
                            'ppos_mean': ppos_mean,
                            'Rneg_mean': Rneg_mean,
                            'pneg_mean': pneg_mean,
                            'Significant_Pos': significant_pos,
                            'Significant_Neg': significant_neg
                        })
                    except:
                        print(f'error for {feature} {data} {cvtype} {corr_method} {significance}')

# Convert the list to a DataFrame
results_df = pd.DataFrame(results_list)

# Save results to CSV
results_df.to_csv(f'results/cpm_comparison.csv', index=False)
print("Results saved to CSV.")

