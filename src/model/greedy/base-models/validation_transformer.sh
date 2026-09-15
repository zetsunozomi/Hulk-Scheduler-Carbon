#!/bin/bash
#SBATCH -J base_model_training   
#SBATCH -p gpu-a100-small           
#SBATCH -N 1                   
#SBATCH -n 1                 
#SBATCH -t 12:00:00             
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sf850@scarletmail.rutgers.edu


source /work/09796/shuyuanfan4814/ls6/mirage-env/bin/activate

python3 -u transformer_validation.py \
    --test_data /work/09796/shuyuanfan4814/ls6/interrupt-free-provisioning-data-generation/data/pickle_data/data_of_the_validation_data_new_ \
    --model_file /work/09796/shuyuanfan4814/ls6/interrupt-free-provisioning/src/model/validation/models/transformer_model/model_transformer_4.pt\
    -nnode 4
python3 -u transformer_validation.py \
    --test_data /work/09796/shuyuanfan4814/ls6/interrupt-free-provisioning-data-generation/data/pickle_data/data_of_the_validation_data_new_ \
    --model_file /work/09796/shuyuanfan4814/ls6/interrupt-free-provisioning/src/model/validation/models/transformer_model/model_transformer_8.pt\
    -nnode 8
python3 -u transformer_validation.py \
    --test_data /work/09796/shuyuanfan4814/ls6/interrupt-free-provisioning-data-generation/data/pickle_data/data_of_the_validation_data_new_ \
    --model_file /work/09796/shuyuanfan4814/ls6/interrupt-free-provisioning/src/model/validation/models/transformer_model/model_transformer_16.pt\
    -nnode 16
python3 -u transformer_validation.py \
    --test_data /work/09796/shuyuanfan4814/ls6/interrupt-free-provisioning-data-generation/data/pickle_data/data_of_the_validation_data_new_ \
    --model_file /work/09796/shuyuanfan4814/ls6/interrupt-free-provisioning/src/model/validation/models/transformer_model/model_transformer_32.pt\
    -nnode 32