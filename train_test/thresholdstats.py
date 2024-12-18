


import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

num_labels = {"aparc+aseg": 85, "aparc.a2009s+aseg": 165}

# Load data
split = 'val'
root = '/media/volume/MV_HCP/TrainData_MRtrix_1000_MNI_100K'
with open(os.path.join(root, '{}.pickle'.format(split)), 'rb') as file:
    data_dict = pickle.load(file)

data_dict['label_name_aparc+aseg'] = [f"{i}_{j}" for i in range(85) for j in range(i, 85)]
features = data_dict['feat']
subject_ids = data_dict['subject_id']

labels = []
label_names = []
for atlas_name in ['aparc.a2009s+aseg']:
    atlas_label_names = data_dict[f'label_name_{atlas_name}']
    label_names.append(atlas_label_names)
    
    atlas_labels = data_dict[f'label_{atlas_name}']
    atlas_labels = np.where(np.isin(atlas_labels, list(range(num_labels[atlas_name]))), 0, atlas_labels)
    

    # Calculate fibers per class
    unique_labels, counts = np.unique(atlas_labels, return_counts=True)

    thresholds = list(range(0, 101, 1))
    excluded_classes = []
    excluded_fibers = []
    # Histogram: Fibers per class with log-scaled x-axis
    plt.figure(figsize=(10, 6))
    bins = np.logspace(np.log10(counts.min()), np.log10(counts.max()), 100)
    plt.hist(counts, bins=bins, log=True, color='skyblue', edgecolor='black')
    plt.xscale('log')
    plt.xlabel("Number of Fibers (log scale)")
    plt.ylabel("Classes (log scale)")
    plt.title("Distribution of Number of Fibers per Class")
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.savefig('/media/volume/HCP_diffusion_MV/TractCloud/plots/FibersPerClass_{}'.format(atlas_name.replace('.','_')))


    # Analyze for each threshold level
    for threshold in thresholds:
        subject_count = len(np.unique(subject_ids))
        min_subjects_required = (threshold / 100) * subject_count
        
        # Count number of subjects per label
        label_subject_counts = {label: 0 for label in unique_labels}
        for subject_id in np.unique(subject_ids):
            subject_indices = np.where(subject_ids == subject_id)[0]
            subject_labels = np.unique(atlas_labels[subject_indices])
            for label in subject_labels:
                label_subject_counts[label] += 1
        
        # Identify rare labels for the current threshold
        rare_labels = [label for label, count in label_subject_counts.items() if count < min_subjects_required]
        num_excluded_classes = len(rare_labels)
        num_excluded_fibers = np.sum([counts[i] for i, label in enumerate(unique_labels) if label in rare_labels])

        excluded_classes.append(num_excluded_classes)
        excluded_fibers.append(num_excluded_fibers)

    # Plot the results across thresholds
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel("Threshold Level (%)")
    ax1.set_ylabel("Excluded Classes", color="tab:blue")
    ax1.plot(thresholds, excluded_classes, color="tab:blue", label="Excluded Classes") #marker="o", 
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Excluded Fibers", color="tab:red")
    ax2.plot(thresholds, excluded_fibers, color="tab:red", label="Excluded Fibers") # marker="x", linestyle="--"
    ax2.tick_params(axis="y", labelcolor="tab:red")

    plt.title("Impact of Thresholding on Class and Fiber Exclusion")
    fig.tight_layout()
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)
    plt.savefig('/media/volume/HCP_diffusion_MV/TractCloud/plots/ClassesPerThreshold_{}'.format(atlas_name.replace('.','_')))



