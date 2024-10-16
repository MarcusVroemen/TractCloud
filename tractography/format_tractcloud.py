import os
import numpy as np
import whitematteranalysis as wma
import pickle
from sklearn.model_selection import train_test_split
from concurrent.futures import ProcessPoolExecutor
from label_encoder import encode_labels_txt
from format_plot import *
import sys
sys.path.append('..')
from utils.tract_feat import *

def read_tractography(tractography_path):
    """Read tractography data and convert it to a feature array."""
    pd_tractography = wma.io.read_polydata(tractography_path)
    fiber_array = CustomFiberArray()
    fiber_array.convert_from_polydata(pd_tractography, points_per_fiber=15, distribution='exponential', decay_factor=decay_factor)
    feat = np.dstack((fiber_array.fiber_array_r, fiber_array.fiber_array_a, fiber_array.fiber_array_s))
    return feat, fiber_array

def read_labels(labels_path):
    """Read and parse labels from a text file."""
    labels = []
    with open(labels_path, 'r') as file:
        lines = file.readlines()
        for line in lines[1:]:  # Skip the first line
            label_pair = line.strip().split()
            labels.append((int(label_pair[0]), int(label_pair[1])))
    return labels

def encode_labels(subject_id, data_dir, encoding_type='default'):
    """Encode labels using a custom encoding method."""
    output_dir = os.path.join(data_dir, subject_id, "output")
    labels_path = os.path.join(output_dir, "labels_100K_aparc+aseg.txt")
    encoded_labels_path = os.path.join(output_dir, f"labels_encoded_{encoding_type}.txt")

    # Encode labels using your label_encoder.py
    encode_labels_txt(labels_path, encoded_labels_path, encoding_type)
    return encoded_labels_path

def threshold(labels, streamlines, min_fibers=10):
    """
    Filter streamlines and labels, removing any fibers assigned to labels 
    with less than `min_fibers` fibers.
    """
    # Count occurrences of each label
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    # Identify labels with at least `min_fibers` fibers
    valid_labels = unique_labels[counts >= min_fibers]
    
    # Print the labels that are being deleted
    deleted_labels = unique_labels[counts < min_fibers]    
    if deleted_labels.size > 0:
        print(f"Labels deleted due to too few streamlines: {deleted_labels}")
    
    # Create a mask to filter out fibers with invalid labels
    mask = np.isin(labels, valid_labels)
    
    # Filter the streamlines and labels
    filtered_streamlines = streamlines[mask]
    filtered_labels = labels[mask]
    
    return filtered_labels, filtered_streamlines

def process_subject(subject_index, subject_id, data_dir, encoding_type):
    """Process tractography and labels for a specific subject."""
    print(f"Reading subject {subject_id}")
    output_dir = os.path.join(data_dir, subject_id, "output")
    encoded_labels_path = encode_labels(subject_id, data_dir, encoding_type)
    tractography_path = os.path.join(output_dir, STREAMLINE_FILE)
    streamlines, _ = read_tractography(tractography_path)
    
    labels = np.loadtxt(encoded_labels_path, dtype=int)
    # subject_ids = np.full(labels.shape, subject_id, dtype=int) # use this if you want to work with indexes instead of the HCP ids

    # Apply the filter to remove fibers with less than `min_fibers` labels
    # labels, streamlines = threshold(labels, streamlines, MIN_FIBERS)

    subject_ids = np.full(labels.shape, subject_index, dtype=int)
    
    return streamlines, labels, subject_ids

def preprocess_data(data_dir, subjects_with_output_file, output_dir, encoding_type, chunk_size=64):
    """Preprocess data by reading and processing subjects in chunks."""
    # Read the subject IDs from the output file
    with open(subjects_with_output_file, 'r') as f:
        subjects_with_output = f.read().splitlines()
    
    total_subjects = len(subjects_with_output)
    print(f"Total number of subjects: {total_subjects}")
    
    # Initialize the subject index counter
    global_subject_index = 0
    
    for start in range(0, total_subjects, chunk_size):
        end = min(start + chunk_size, total_subjects)
        features_list = []
        labels_list = []
        subject_ids_list = []
        # streamlines, labels, subject_ids = process_subject(0, subjects_with_output[0], data_dir, encoding_type)
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_subject, global_subject_index + i, subject_id, data_dir, encoding_type) 
                       for i, subject_id in enumerate(subjects_with_output[start:end])]
            
            for future in futures:
                streamlines, labels, subject_ids = future.result()
                if streamlines.shape[0] == labels.shape[0]:
                    features_list.append(streamlines)
                    labels_list.append(labels)
                    subject_ids_list.append(subject_ids)
                else:
                    print(f"Mismatch in streamlines and labels for subject id {subject_ids[0]}")
        
        # Update the global subject index counter
        global_subject_index += (end - start)
        
        features = np.vstack(features_list)
        labels = np.hstack(labels_list)
        subject_ids = np.hstack(subject_ids_list)
        
        # Determine the number of labels based on encoding type
        if encoding_type == 'default':
            label_names = [f"{i}_{j}" for i in range(85) for j in range(85)]
            total_labels = 7225
        elif encoding_type == 'symmetric':
            label_names = [f"{i}_{j}" for i in range(85) for j in range(i, 85)]
            total_labels = (86 * 85) / 2
        else:
            raise ValueError(f"Unsupported encoding type: {encoding_type}")
        
        # Append to the output files
        save_data(features, labels, label_names, subject_ids, start, output_dir, encoding_type, total_labels)
        
def extract_data_by_subjects(subject_ids, selected_subjects, features, labels):
    """Extract data corresponding to specific subject IDs."""
    mask = np.isin(subject_ids, selected_subjects)
    return features[mask], labels[mask], subject_ids[mask]

def save_data(features, labels, label_names, subject_ids, start_idx, output_dir, encoding_type, total_labels, split_ratios=(0.7, 0.1, 0.2)):
    """Save data into train, validation, and test splits."""
    unique_subject_ids = np.unique(subject_ids)

    # Split the subject IDs into train, temp (val + test)
    train_subjects, temp_subjects = train_test_split(
        unique_subject_ids, test_size=(1 - split_ratios[0]), random_state=42)
    val_subjects, test_subjects = train_test_split(
        temp_subjects, test_size=split_ratios[2] / (split_ratios[1] + split_ratios[2]), random_state=42)

    # Extract train, validation, and test data based on the subject splits
    X_train, y_train, subj_train = extract_data_by_subjects(subject_ids, train_subjects, features, labels)
    X_val, y_val, subj_val = extract_data_by_subjects(subject_ids, val_subjects, features, labels)
    X_test, y_test, subj_test = extract_data_by_subjects(subject_ids, test_subjects, features, labels)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    train_path = os.path.join(output_dir, 'train.pickle')
    val_path = os.path.join(output_dir, 'val.pickle')
    test_path = os.path.join(output_dir, 'test.pickle')
    
    # Initialize dictionaries for storing data
    print("Saving data to pickle files")
    if os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path):
        with open(train_path, 'rb') as file:
            train_dict = pickle.load(file)
        with open(val_path, 'rb') as file:
            val_dict = pickle.load(file)
        with open(test_path, 'rb') as file:
            test_dict = pickle.load(file)
        
        train_feat = np.concatenate([train_dict['feat'], X_train], axis=0)
        train_label = np.concatenate([train_dict['label'], y_train], axis=0)
        train_subj = np.concatenate([train_dict['subject_id'], subj_train], axis=0)
        
        val_feat = np.concatenate([val_dict['feat'], X_val], axis=0)
        val_label = np.concatenate([val_dict['label'], y_val], axis=0)
        val_subj = np.concatenate([val_dict['subject_id'], subj_val], axis=0)
        
        test_feat = np.concatenate([test_dict['feat'], X_test], axis=0)
        test_label = np.concatenate([test_dict['label'], y_test], axis=0)
        test_subj = np.concatenate([test_dict['subject_id'], subj_test], axis=0)
        
    else:
        train_feat = X_train
        train_label = y_train
        train_subj = subj_train
        val_feat = X_val
        val_label = y_val
        val_subj = subj_val
        test_feat = X_test
        test_label = y_test
        test_subj = subj_test
    
    with open(train_path, 'wb') as file:
        pickle.dump({'feat': train_feat, 'label': train_label, 'label_name': label_names, 'subject_id': train_subj}, file)
    with open(val_path, 'wb') as file:
        pickle.dump({'feat': val_feat, 'label': val_label, 'label_name': label_names, 'subject_id': val_subj}, file)
    with open(test_path, 'wb') as file:
        pickle.dump({'feat': test_feat, 'label': test_label, 'label_name': label_names, 'subject_id': test_subj}, file)
    
    subject_ids_split = {
        "train": np.unique(subj_train),
        "val": np.unique(subj_val),
        "test": np.unique(subj_test)
    }
    
    metadata_path = os.path.join(output_dir, 'metadata.pickle')
    # update_metadata(features, labels, subject_ids, metadata_path, subject_ids_split, total_labels)
# 
    print(f"Chunk {start_idx+1} successfully saved to {output_dir}")

def update_metadata(features, labels, subject_ids, metadata_path, subject_ids_split, total_labels):
    """Update and save metadata based on processed data."""
    if os.path.exists(metadata_path):
        with open(metadata_path, 'rb') as file:
            metadata = pickle.load(file)
    else:
        metadata = {
            'fibers_per_subject': {},
            'fibers_per_label': {},
            'labels_per_fiber_count': {},
            'split_subjects': {'train': [], 'val': [], 'test': []},
            'fibers_per_label_per_subject': {}
        }

    # Process and aggregate metadata
    unique_subjects = np.unique(subject_ids)
    for subj in unique_subjects:
        # Collect the number of fibers per subject
        fibers_per_subject = np.sum(subject_ids == subj)
        metadata['fibers_per_subject'][subj] = fibers_per_subject
        
        # Collect the number of fibers per label for this subject
        labels_for_subject = labels[subject_ids == subj]
        unique_labels, fibers_per_label = np.unique(labels_for_subject, return_counts=True)
        metadata['fibers_per_label_per_subject'][subj] = dict(zip(unique_labels, fibers_per_label))
        
        for label, count in zip(unique_labels, fibers_per_label):
            if label in metadata['fibers_per_label']:
                metadata['fibers_per_label'][label] += count
            else:
                metadata['fibers_per_label'][label] = count

        # Collect the number of labels per fiber count
        for count in fibers_per_label:
            if count in metadata['labels_per_fiber_count']:
                metadata['labels_per_fiber_count'][count] += 1
            else:
                metadata['labels_per_fiber_count'][count] = 1

        # Track which subjects are in each split
        if subj in subject_ids_split['train']:
            split = 'train'
        elif subj in subject_ids_split['val']:
            split = 'val'
        elif subj in subject_ids_split['test']:
            split = 'test'
        else:
            split = 'unknown'

        metadata['split_subjects'][split].append(subj)
        
    detected_labels = len(metadata['fibers_per_label'])
    zero_fiber_labels = total_labels - detected_labels
    metadata['labels_per_fiber_count'][0] = zero_fiber_labels

    # Save the metadata
    with open(metadata_path, 'wb') as file:
        pickle.dump(metadata, file)

    print(f"Metadata successfully updated and saved to {metadata_path}")




    return average_fibers_per_label, std_fibers_per_label

if __name__ == "__main__":
    STREAMLINE_FILE="streamlines_100K_MNI.vtk"
    
    encoding = 'symmetric'
    decay_factor = 0
    data_size=1000
    data_dir = "/media/volume/sdc/HCP_MRtrix"
    output_dir = f"/media/volume/sdc/TrainData_MRtrix_{data_size}_{encoding}_100K_MNI/" #_D{round(decay_factor)}
    subjects_with_output_file = os.path.join(data_dir, f"../subjects_tractography_output_{data_size}.txt")
    
    preprocess_data(data_dir, subjects_with_output_file, output_dir, encoding, chunk_size=125)

    # Make plots
    metadata = load_metadata(os.path.join(output_dir, 'metadata.pickle'))
    plot_fibers_per_subject(metadata, output_dir)
    plot_fibers_per_label(metadata, output_dir, sort=False)
    plot_labels_per_fiber_count(metadata, output_dir, encoding)
    plot_labels_per_fiber_count(metadata, output_dir, encoding, 50)
    plot_labels_per_fiber_count(metadata, output_dir, encoding, 100)
    compute_fiber_stats(metadata)
    
