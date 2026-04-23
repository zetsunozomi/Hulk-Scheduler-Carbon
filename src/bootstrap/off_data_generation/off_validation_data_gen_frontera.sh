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

PYTHON_ENV="/lus/eagle/projects/Local-LLM/shuyuanfan/conda_env/carbon/bin/python"
echo "Using Python from: $PYTHON_ENV"
cd /lus/eagle/projects/Local-LLM/shuyuanfan/carbon-adaptive/src/queue_prediction
# Define the list of nodes
node_list=(4 8 16 32)

# Run the Python script for each node configuration
# For validation data: from 2021/04/01 to 2021/08/19, 141 days in total.
# Number of snapshots: 3384 hours. However, it runs 48 hours for warmup, thus the last snapshot begins at 3384 - 48 = 3336 hours.
# A total of 834 samples.

for i in "${node_list[@]}"
do
   ray stop --force
   echo "Running iteration for $i nodes"
   $PYTHON_ENV offline_data_gen.py \
       -num_samples 834 \
       -interval 4 \
       -od /lus/eagle/projects/Local-LLM/shuyuanfan/carbon-adaptive/src/queue_prediction/json_data/frontera_validation_data_new_${i} \
       -workload filtered-frontera-validate.log \
       -start_time 2021-04-01T00:00:00 \
       -warmup_len 2 \
       -workload_len 5 \
       -cd /lus/eagle/projects/Local-LLM/shuyuanfan/carbon-adaptive/src/slurm_config/frontera \
       -node "$i" 
done

echo "All iterations completed"