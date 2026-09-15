#!/bin/bash
#SBATCH -J train_base_model
#SBATCH -p gpu-a100-small 
#SBATCH -N 1                    # Total number of nodes requested
#SBATCH -n 1                    # Total number of mpi tasks requested
#SBATCH -t 4:00:00             # Run time (hh:mm:ss)
#SBATCH --output=output_4_8_16_32.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sf850@scarletmail.rutgers.edu

source /work/09796/shuyuanfan4814/ls6/mirage-env/bin/activate

python3 -u validate_base_model.py \
    -wd /work/09796/shuyuanfan4814/ls6/interrupt-free-provisioning-data-generation/data/pickle_data/ \
    -n transformer_4_1 \
    --training_data data_of_the_train_data_new_4.pkl \
    --validation_data data_of_the_validation_data_new_4.pkl\
    -epoch 500 \
    -model transformer