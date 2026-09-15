#!/bin/bash
#SBATCH -J my_offline_data_gen     # Job name
#SBATCH -p gpu-a100-small             # Queue (partition) name
#SBATCH -N 1                    # Total number of nodes requested
#SBATCH -n 1                    # Total number of mpi tasks requested
#SBATCH -t 2:00:00             # Run time (hh:mm:ss)
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sf850@scarletmail.rutgers.edu

PYTHON_ENV="/work/09796/shuyuanfan4814/ls6/miniconda3/envs/carbon/bin/python"
echo "Using Python from: $PYTHON_ENV"

$PYTHON_ENV -u train.py \
    -wd /work/09796/shuyuanfan4814/ls6/carbon/src/queue_prediction/json_data/ \
    -n queue_time_predictor_transformer_16node \
    --training_data training_data_new_16/all_data.json \
    --validation_data validation_data_new_16/all_data.json\
    -epoch 50 \
    -model transformer 