#!/bin/bash
#PBS -A Local-LLM
#PBS -l walltime=6:00:00
#PBS -l filesystems=home:eagle
#PBS -q by-gpu
#PBS -l select=1
#PBS -N f-data-prepare
#PBS -o off-train.log
#PBS -j oe
#PBS -M sf850@scarletmail.rutgers.edu                                        
#PBS -m bae 

PYTHON_ENV="/lus/eagle/projects/Local-LLM/shuyuanfan/conda_env/carbon/bin/python"
echo "Using Python from: $PYTHON_ENV"
cd /lus/eagle/projects/Local-LLM/shuyuanfan/carbon-adaptive/src/queue_prediction
# Define the list of nodes
node_list=(4 8 16 32)

# Run the Python script for each node configuration
# For training data: from 2019-12-04 to 2021-04-01, 484 days in total.
# Number of snapshots: 484 * 24 hours = 11616 hours. However, it runs 48 hours for warmup, thus the last snapshot begins at 11616 - 48 = 11568 hours.
# interval 4
# A total of 2892 samples.

for i in "${node_list[@]}"
do
   ray stop --force
   echo "Running iteration for $i nodes"
   $PYTHON_ENV offline_data_gen.py \
       -parallel \
       -num_samples 2892 \
       -interval 4 \
       -od /lus/eagle/projects/Local-LLM/shuyuanfan/carbon-adaptive/src/queue_prediction/json_data/frontera_training_data_new_${i} \
       -workload filtered-frontera-log.log \
       -start_time 2019-12-04T00:00:00 \
       -warmup_len 2 \
       -workload_len 5 \
       -cd /lus/eagle/projects/Local-LLM/shuyuanfan/carbon-adaptive/src/slurm_config/frontera \
       -node "$i" 
done


