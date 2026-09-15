#!/bin/bash
#SBATCH -J my_offline_data_gen     # Job name
#SBATCH -p gpu-a100-small             # Queue (partition) name
#SBATCH -N 1                    # Total number of nodes requested
#SBATCH -n 1                    # Total number of mpi tasks requested
#SBATCH -t 8:00:00             # Run time (hh:mm:ss)
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sf850@scarletmail.rutgers.edu

PYTHON_ENV="/work/09796/shuyuanfan4814/ls6/miniconda3/envs/carbon/bin/python"
echo "Using Python from: $PYTHON_ENV"

# Define the list of nodes
node_list=(4 8 16 32)

# Run the Python script for each node configuration
# For training data: from 2022-12-30 to 2024-07-30, 578 days in total.
# Number of snapshots: 578 * 24 hours = 13872 hours. However, it runs 48 hours for warmup, thus the last snapshot begins at 13872 - 48 = 13824 hours.
# A total of 3,456 samples.

for i in "${node_list[@]}"
do
   echo "Running iteration for $i nodes"
   $PYTHON_ENV offline_data_gen.py \
       -parallel \
       -num_samples 3456 \
       -interval 4 \
       -od /work/09796/shuyuanfan4814/ls6/carbon/src/queue_prediction/json_data/training_data_new_${i} \
       -workload filtered_iw_log.log \
       -start_time 2022-12-30T00:00:00 \
       -warmup_len 2 \
       -workload_len 5 \
       -cd /work/09796/shuyuanfan4814/ls6/carbon/src/slurm_config \
       -node "$i" 
done

echo "All iterations completed"



