import numpy as np
csv_path = "/media/volume/HCP_diffusion_MV/TractCloud/ModelWeights/MNI_DataMRtrix_100_Rot45_10_10Scale-0.45_0.05Trans50AugTimes0_Unrelated100HCP_dgcnn/k0_kg0_ds0.1_kp5_bs1024_nf500000_np15_epoch20_lr1e-3/log_NoAug/connectome_diff.csv"
data = np.loadtxt(csv_path, delimiter=',')


# Find indices where the absolute value is greater than 10000
indices = np.argwhere(np.abs(data) > 10000) + 1

# Print the indices
print(indices)