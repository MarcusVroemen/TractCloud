#!/bin/bash

# Define the path for the conda environment
ENV_PATH="/home/exouser/miniconda3/envs/TractCloud"

# Create the conda environment at the specified path
# conda create --prefix $ENV_PATH python=3.8 -y

# Activate the environment
# source activate $ENV_PATH

# Install the required packages
conda install --prefix $ENV_PATH pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -y

conda install --prefix $ENV_PATH -c fvcore -c iopath -c conda-forge fvcore iopath -y
conda install --prefix $ENV_PATH -c bottler nvidiacub -y

# Install additional packages using pip
$ENV_PATH/bin/pip install pytorch3d
$ENV_PATH/bin/pip install git+https://github.com/SlicerDMRI/whitematteranalysis.git
$ENV_PATH/bin/pip install h5py
$ENV_PATH/bin/pip install seaborn
$ENV_PATH/bin/pip install scikit-learn
$ENV_PATH/bin/pip install openpyxl

conda install --prefix $ENV_PATH pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -y

# export PYTHONPATH=/home/exouser/anaconda3/envs/TractCloud/lib/python3.8/site-packages:$PYTHONPATH
# python -c "import sys; print(sys.path)"
# -> should contain the previous path

# chmod +x setup_tractcloud.sh
# ./setup_tractcloud.sh
# conda remove -n TractCloud --all


conda create -n tractcloud python=3.9
conda activate tractcloud

conda install pytorch=1.13.0 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia -y
conda install -c iopath iopath -y
conda install -c bottler nvidiacub -y

pip install h5py seaborn scikit-learn openpyxl

conda install pytorch3d -c pytorch3d

pip install git+https://github.com/SlicerDMRI/whitematteranalysis.git

pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1110/download.html
