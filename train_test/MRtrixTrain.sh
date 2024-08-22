#!/bin/bash
# Training params
model_name="dgcnn"              # model
epoch=20                        # epoch
batch_size=1024                 # batch size
lr=1e-3                         # learning rate
class_weighting=0               # 0 no, 1 default, >1 increase impact

# Data
input_data="MRtrix_100"         # training data, 800 clusters + 800 outliers
num_f_brain=1000                # the number of streamlines in a brain
num_p_fiber=15                  # the number of points on a streamline
rot_ang_lst="45_10_10"          # data rotating
scale_ratio_range="0.45_0.05"   # data scaling
trans_dis=50                    # data translation
atlas="DK(85x85)"

aug_times=10        # determine how many augmented data you want in training
test_aug_times=10   # you may train on data with heavier augmentation and test on data with lighter or no augmentation.

# Local-global representation
k="20"   # local, neighbor streamlines
k_global="500"   # global, randomly selected streamlines in the whole-brain
k_ds_rate=1  # downsample the tractography when calculating neighbor streamlines
k_point_level="5"  # point-level neighbors on one streamline

# Paths
local_global_rep_folder=k${k}_kg${k_global}_ds${k_ds_rate}_kp${k_point_level}_bs${batch_size}_nf${num_f_brain}_np${num_p_fiber}_epoch${epoch}_lr${lr}_classweight${class_weighting}
out_path=../ModelWeights/Data${input_data}_Rot${rot_ang_lst}Scale-${scale_ratio_range}Trans${trans_dis}AugTimes${aug_times}_Unrelated100HCP_${model_name}/${local_global_rep_folder}
# input_path=../TrainData_${input_data}
input_path=/media/volume/HCP_diffusion_MV/TrainData_${input_data}

# Train/Validation/Test --recenter --include_org_data --use_tracts_testing
# python train.py --class_weighting ${class_weighting} --connectome --atlas ${atlas} --k_ds_rate ${k_ds_rate} --rot_ang_lst ${rot_ang_lst} --scale_ratio_range ${scale_ratio_range} --trans_dis ${trans_dis} --aug_times ${aug_times} --k ${k} --k_point_level ${k_point_level} --k_global ${k_global} --num_fiber_per_brain ${num_f_brain} --num_point_per_fiber ${num_p_fiber} --input_path ${input_path} --epoch ${epoch} --out_path_base ${out_path} --model_name $model_name --train_batch_size ${batch_size} --val_batch_size ${batch_size} --test_batch_size ${batch_size}  --lr ${lr}
python test.py --connectome --atlas ${atlas} --out_path_base ${out_path} --aug_times ${test_aug_times} # --input_path ${input_path}
