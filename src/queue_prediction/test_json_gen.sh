#!/bin/sh
#SBATCH --account=m4410
#SBATCH --qos=regular
#SBATCH --time=06:00:00
#SBATCH --constraint=gpu 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4 
#SBATCH --output=off_training_generation.%j.out
#SBATCH --job-name=off_training_generation
#SBATCH --mail-user=sf850@scarletmail.rutgers.edu
#SBATCH --mail-type=ALL

conda activate carbon
# Define the list of nodes
node_list=(4 8 16 32)

# Run the Python script for each node configuration
# For training data: from 2022-12-30 to 2024-07-30, 578 days in total.
# Number of snapshots: 578 * 24 hours = 13872 hours. However, it runs 48 hours for warmup, thus the last snapshot begins at 13872 - 48 = 13824 hours.
# A total of 3,456 samples.

python3 offline_data_gen.py \
    -parallel \
    -num_samples 2 \
    -interval 4 \
    -od /pscratch/sd/s/syfan/carbon-minimum-scheduling/src/queue_prediction/json_data/test \
    -workload filtered_iw_log.log \
    -start_time 2022-12-30T00:00:00 \
    -warmup_len 2 \
    -workload_len 5 \
    -cd /pscratch/sd/s/syfan/carbon-minimum-scheduling/src/slurm_config \
    -node "4" 



