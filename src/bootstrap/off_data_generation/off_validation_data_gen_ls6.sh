#!/bin/bash
#PBS -A Local-LLM
#PBS -l walltime=6:00:00
#PBS -l filesystems=home:eagle
#PBS -q by-gpu
#PBS -l select=1
#PBS -N f-data-prepare
#PBS -o off-validation.log
#PBS -j oe
#PBS -M sf850@scarletmail.rutgers.edu                                        
#PBS -m bae 

PYTHON_ENV="/work/09796/shuyuanfan4814/ls6/miniconda3/envs/carbon/bin/python"
echo "Using Python from: $PYTHON_ENV"

# Define the list of nodes
node_list=(4 8 16 32)

# Run the Python script for each node configuration
# For validation data: from 2024/08/01 to 2025/01/03, 156 days in total.
# Number of snapshots: 3744 hours. However, it runs 48 hours for warmup, thus the last snapshot begins at 3744 - 48 = 3696 hours.
# A total of 924 samples.

for i in "${node_list[@]}"
do
   echo "Running iteration for $i nodes"
   $PYTHON_ENV offline_data_gen.py \
       -parallel \
       -num_samples 925 \
       -interval 4 \
       -od /work/09796/shuyuanfan4814/ls6/carbon/src/queue_prediction/json_data/ls6_validation_data_new_${i} \
       -workload filtered_iw_validate.log \
       -start_time 2024-08-01T00:00:00 \
       -warmup_len 2 \
       -workload_len 5 \
       -cd /work/09796/shuyuanfan4814/ls6/carbon/src/slurm_config/ls6 \
       -node "$i" 
done

echo "All iterations completed"