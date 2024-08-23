#!/bin/bash
# Training params
model_name="dgcnn"              # model
epoch=20                        # epoch
batch_size=1024                 # batch size
lr=1e-3                         # learning rate
class_weighting=0               # 0 no, 1 default, >1 increase impact
encoding='symmetric'

# Data
input_data="MRtrix_100"         # training data, 800 clusters + 800 outliers
num_f_brain=20000                # the number of streamlines in a brain
num_p_fiber=15                  # the number of points on a streamline
rot_ang_lst="45_10_10"          # data rotating
scale_ratio_range="0.45_0.05"   # data scaling
trans_dis=50                    # data translation
atlas="DK(85x85)"

aug_times=0        # determine how many augmented data you want in training
test_aug_times=0   # you may train on data with heavier augmentation and test on data with lighter or no augmentation.

# Local-global representation
k="20"   # local, neighbor streamlines
k_global="500"   # global, randomly selected streamlines in the whole-brain
k_ds_rate=1  # downsample the tractography when calculating neighbor streamlines
k_point_level="5"  # point-level neighbors on one streamline

# Paths
local_global_rep_folder=k${k}_kg${k_global}_ds${k_ds_rate}_kp${k_point_level}_bs${batch_size}_nf${num_f_brain}_np${num_p_fiber}_epoch${epoch}_lr${lr}_classweight${class_weighting}
weight_path_base=../ModelWeights/Data${input_data}_Rot${rot_ang_lst}Scale-${scale_ratio_range}Trans${trans_dis}AugTimes${aug_times}_Unrelated100HCP_${model_name}_${encoding}encoding/${local_global_rep_folder}
# input_path=/media/volume/HCP_diffusion_MV/TrainData_${input_data}_${encoding}



# test params
test_realdata_batch_size=1024   # the batch size for testing on real data. larger batch size may further accelerate the parcellation.
k_ds_rate_for_testing=0.1       # downsample the tractography when calculating neighbor streamlines.
# for test_dataset in ${test_dataset_lst}; do 

# subject and paths
subject_idx='993675'

tractography_path=/media/volume/HCP_diffusion_MV/data/${subject_idx}/output/streamlines.vtk
out_path=/media/volume/HCP_diffusion_MV/data/${subject_idx}/TractCloud

python test_realdata.py --num_classes 3655 --encoding ${encoding} --class_weighting ${class_weighting} --connectome --atlas ${atlas} \
    --weight_path_base ${weight_path_base} --tractography_path ${tractography_path} --out_path ${out_path} \
    --test_realdata_batch_size ${test_realdata_batch_size} --k_ds_rate ${k_ds_rate_for_testing}