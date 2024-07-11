#!/bin/bash

# test params
test_realdata_batch_size=1024   # the batch size for testing on real data. larger batch size may further accelerate the parcellation.
k_ds_rate_for_testing=0.1       # downsample the tractography when calculating neighbor streamlines.
# subject_idx_lst='100206 100307 100408'  # Testing datasets in our paper (except the private tumor data). We put one example data for each testing dataset.
subject_idx_lst='100307'  # Testing datasets in our paper (except the private tumor data). We put one example data for each testing dataset.
n_tracts=10M
test_dataset=HCP_MRTRIX
for subject_idx in ${subject_idx_lst}; do 
    # model weight path
    weight_path_base=../TrainedModel
    
    # paths for example subjects of dHCP, ABCD, HCP, PPMI 
    tractography_path=../TestData/${test_dataset}/${subject_idx}_fibers_${n_tracts}.vtk

    # run test
    out_SS_cluster_path=../parcellation_results/${test_dataset}/${subject_idx}/SS   # SS: subject space
    python test_realdata.py --weight_path_base ${weight_path_base} --tractography_path ${tractography_path} --out_path ${out_SS_cluster_path} \
    --test_realdata_batch_size ${test_realdata_batch_size} --k_ds_rate ${k_ds_rate_for_testing} \
    --model_name dgcnn

done