#!/bin/bash
#! Example SLURM job script for Mauao (A40 GPUs, Intel(R) Xeon(R) Gold 6152 CPU @ 2.10GHz).

#!#############################################################
#!#### Modify the options in this section as appropriate ######
#!#############################################################

#! sbatch directives begin here ###############################
#! Name of the job:
#SBATCH -J llm
#! Which node the job will be allocated
#SBATCH -w mauao
#! How many (MPI) tasks will there be in total?
#! Note probably this should not exceed the total number of GPUs in use.
#SBATCH --ntasks=1
#! Specify the number of GPUs per node.
#! Note that the job submission script will enforce no more than 11 cpus per GPU.
#SBATCH --gres=gpu:1
#! (Optional) Specify the number of CPU cores.
#SBATCH -c 11
#!SBATCH --dependency=afterany:79061

#! sbatch directives end here (put any additional directives above this line)

#! Activate the Poetry environment:
echo -e "\nActivating Poetry Environment:\n============\n"
eval "poetry shell"

#! Number of nodes and tasks per node allocated by SLURM (do not change):
export numnodes=$SLURM_JOB_NUM_NODES
export numtasks=$SLURM_NTASKS
export mpi_tasks_per_node=$(echo "$SLURM_TASKS_PER_NODE" | sed -e  's/^\([0-9][0-9]*\).*$/\1/')
# Get number of available GPUs
export NUM_GPUS_PER_NODE=`echo $CUDA_VISIBLE_DEVICES | awk 'BEGIN{FS=","};{print NF}'`
export NUM_GPUS=$(($NUM_GPUS_PER_NODE * $numnodes))
echo -e "\nAddig the path to the latest CUDA:\n==================\n"
export PATH=/usr/local/cuda-12.1/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
echo -e "\nChecking the output of nvcc -V\n`nvcc -V`:\n==================\n"
#! Set data paths
DATA_ROOT=/home/ls985/c4
DATA_ROOT_MDS=/home/ls985/mds-c4
DATA_ROOT_SMALL=/home/ls985/my-copy-c4
DATA_ROOT_SMALL_MDS=/home/ls985/my-mds-copy-c4
#! Set config paths
CONFIG_MPT_125M=scripts/train/yamls/pretrain/mpt-125m.yaml
#! Saving path
DATETIME=$(date '+%Y%m%d_%H%M%S')
SAVE_PATH="/nfs-share/ls985/projects/llm-foundry/checkpoints/$DATETIME"

#! ############################################################
#! Modify the settings below to specify the application's environment, location 
#! and launch method:

#! Full path to application executable:
# script="poetry run composer --world_size $NUM_GPUS --node_rank 0 --master_addr 127.0.0.1 --master_port 6379 scripts/train/train.py $CONFIG_MPT_125M"
script="poetry run python scripts/train/train.py $CONFIG_MPT_125M"

#! Run options for the application:
# options=""
options="train_loader.dataset.split=train_small eval_loader.dataset.split=val_small data_local='$DATA_ROOT_SMALL' device_train_microbatch_size=20 save_interval=10ba save_num_checkpoints_to_keep=1 save_folder='$SAVE_PATH' loggers.wandb='{project: 'llm', name: 'test-mpt-125m',}' train_loader.num_workers=12 eval_loader.num_workers=12 max_duration=9600ba autoresume=True"

#! Work directory (i.e. where the job will run):
workdir="$SLURM_SUBMIT_DIR"  # The value of SLURM_SUBMIT_DIR sets workdir to the directory
                             # in which sbatch is run.

#! Choose this for a pure shared-memory OpenMP parallel program on a single node:
#! (OMP_NUM_THREADS threads will be created):
SCRIPT="$script $options"


###############################################################
### You should not have to change anything below this line ####
###############################################################

cd $workdir
echo -e "Changed directory to `pwd`.\n"

JOBID=$SLURM_JOB_ID

echo -e "JobID: $JOBID\n======"
echo "Time: `date`"
echo "Running on master node: `hostname`"
echo "Current directory: `pwd`"
echo -e "\nnumtasks=$numtasks, numnodes=$numnodes, mpi_tasks_per_node=$mpi_tasks_per_node (OMP_NUM_THREADS=$OMP_NUM_THREADS)"

echo -e "\nRunning SCRIPT:\n==================\n$SCRIPT\n"

eval $SCRIPT
