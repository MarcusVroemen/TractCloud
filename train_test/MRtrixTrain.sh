#!/bin/bash
# Training params
model_name="pointnet"              # model dgcnn or pointnet
epoch=150                        # epoch
batch_size=1024                 # batch size
lr=1e-3                         # learning rate

# Data
input_data="MRtrix_1000_MNI_100K"         # training data, 800 clusters + 800 outliers
num_f_brain=10000              # the number of streamlines in a brain
num_p_fiber=15                  # the number of points on a streamline
threshold=0
atlas="aparc+aseg,aparc.a2009s+aseg"             # aparc+aseg or aparc.a2009s+aseg or aparc+aseg,aparc.a2009s+aseg

# Local-global representation
k="0"   # local, neighbor streamlines
k_global="0"   # global, randomly selected streamlines in the whole-brain
k_ds_rate=0.1  # downsample the tractography when calculating neighbor streamlines
k_point_level="5"  # point-level neighbors on one streamline

# Augmentation
rot_ang_lst="45_10_10"          # data rotating
scale_ratio_range="0.45_0.05"   # data scaling
trans_dis=50                    # data translation
aug_times=0        # determine how many augmented data you want in training
test_aug_times=0   # you may train on data with heavier augmentation and test on data with lighter or no augmentation.
# Parameters not used
class_weighting="0"             # 0 no, 1 default, >1 increase impact
weight_decay="0"                # weight decay for Adam
decay_factor="1"    	        # Multiplicative factor of learning rate decay

# Paths
input_path=/media/volume/MV_HCP/TrainData_${input_data}
local_global_rep_folder=k${k}_kg${k_global}_bs${batch_size}_nf${num_f_brain}_epoch${epoch}_lr${lr}_THR${threshold}_${atlas}
out_path=../ModelWeights/Data${input_data}_${model_name}/${local_global_rep_folder}

######### Train/Validation/Test #########
python train.py --threshold ${threshold} --weight_decay ${weight_decay} --class_weighting ${class_weighting} --connectome --atlas ${atlas} --k_ds_rate ${k_ds_rate} --rot_ang_lst ${rot_ang_lst} --scale_ratio_range ${scale_ratio_range} --trans_dis ${trans_dis} --aug_times ${aug_times} --k ${k} --k_point_level ${k_point_level} --k_global ${k_global} --num_fiber_per_brain ${num_f_brain} --num_point_per_fiber ${num_p_fiber} --input_path ${input_path} --epoch ${epoch} --out_path_base ${out_path} --model_name $model_name --train_batch_size ${batch_size} --val_batch_size ${batch_size} --test_batch_size ${batch_size}  --lr ${lr}
python test.py --threshold ${threshold} --connectome --atlas ${atlas} --out_path_base ${out_path} --aug_times ${test_aug_times} --input_path ${input_path}

