import os
import pandas as pd 
import numpy as np

def save_dirnames_to_file(parent_dir, output_file):
    # Get list of all entries in the parent directory
    entries = os.listdir(parent_dir)
    
    # Filter the list to keep only directories
    dirnames = [entry for entry in entries if os.path.isdir(os.path.join(parent_dir, entry))]
    
    # Write directory names to the output file
    with open(output_file, 'w') as file:
        for dirname in dirnames:
            file.write(f"{dirname}\n")

def read_subject_ids(input_file):
    # Read the file and return the list of directory names
    with open(input_file, 'r') as file:
        dirnames = [line.strip() for line in file.readlines()]
    return dirnames


dir_retest = "/media/volume/HCP_diffusion_MV/retest"
# dir_test = "/media/volume/HCP_diffusion_MV/test"
subject_ids_file = "retest_subjects.txt"  
# save_dirnames_to_file(parent_dir, subject_ids_file)
subject_ids = read_subject_ids(subject_ids_file)



def combine_flattened_matrices(subject_ids, atlas):
    data = []
    # Read connectomes from subjects and flatten and append them
    for subject_id in subject_ids:
        csv_file = f'/media/volume/HCP_diffusion_MV/data/{subject_id}/output/connectome_matrix_{atlas}.csv'
        df = pd.read_csv(csv_file, header=None)
        matrix = np.array(df)
        flattened = matrix.flatten()
        data.append([subject_id] + flattened.tolist())
    combined_df = pd.DataFrame(data)
    
    # Set headers
    combined_df.columns = ['subject_id'] + [f'feature_{i}' for i in range(combined_df.shape[1] - 1)]
    combined_df.set_index('subject_id', inplace=True)

    return combined_df


subject_ids = read_subject_ids('/media/volume/connectomes_MV/TractCloud/tractography/txt_files/subjects_tractography_output.txt')
atlas='aparc+aseg'
combined_df = combine_flattened_matrices(subject_ids, atlas)

output_csv = 'combined_connectomes.csv'
combined_df.to_csv(output_csv, index_label='subject_id')


def filter_csv_by_subject_ids(csv_file, subject_ids, output_file):
    # Read the secondary CSV file
    df = pd.read_csv(csv_file)
    # import pdb
    # pdb.set_trace()
    # Filter the DataFrame to include only rows where the 'subject_id' is in subject_ids
    df['Subject']=df['Subject'].astype(str)
    filtered_df = df[df['Subject'].isin(subject_ids)]
    
    # Write the filtered DataFrame to a new CSV file
    filtered_df.to_csv(output_file, index=False)

# Usage
csv_file = 'HCP_behavioral_data.csv'  # Replace with the path to your secondary CSV file
output_file = 'HCP_behavioral_data_subj.csv'  # Replace with the path to your secondary CSV file

filter_csv_by_subject_ids(csv_file, subject_ids, output_file)

