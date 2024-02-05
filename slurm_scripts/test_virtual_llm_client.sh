#!/bin/bash


#! Moving to the project folder
cd /nfs-share/ls985/projects/pollen_worker
#! Activate Poetry environment
poetry shell
poetry run pip install flash-attn==2.3.2 --no-build-isolation
#! Add the appropriate CUDA version to the path
export PATH=/usr/local/cuda-12.1/bin${PATH:+:${PATH}}
#! Check the output of `nvcc -V`
nvcc -V
#! Set data paths
DATA_ROOT=/home/ls985/c4
DATA_ROOT_MDS=/home/ls985/mds-c4
DATA_ROOT_SMALL=/home/ls985/my-copy-c4
DATA_ROOT_SMALL_MDS=/home/ls985/my-mds-copy-c4
#! Set config paths
CONFIG_MPT_125M=/nfs-share/ls985/projects/pollen_worker/llm-foundry-scripts/train/yamls/pretrain/mpt-125m.yaml

#! Saving path
DATETIME=$(date '+%Y%m%d_%H%M%S')
SAVE_PATH="/nfs-share/ls985/projects/pollen_worker/checkpoints/$DATETIME"

#! Test client
poetry run composer pollen_worker/clients/virtual_llm_client.py $CONFIG_MPT_125M train_loader.dataset.split=train_small eval_loader.dataset.split=val_small data_local=$DATA_ROOT_SMALL
poetry run python -m pollen_worker.clients.virtual_llm_client $CONFIG_MPT_125M train_loader.dataset.split=train_small eval_loader.dataset.split=val_small data_local=$DATA_ROOT_SMALL
