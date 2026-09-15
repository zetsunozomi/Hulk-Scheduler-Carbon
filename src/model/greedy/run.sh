#!/bin/bash
#SBATCH -J mirage    
#SBATCH -p gpu-a100-small           
#SBATCH -N 1                   
#SBATCH -n 1                 
#SBATCH -t 35:00:00             
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sf850@scarletmail.rutgers.edu

source /work/09796/shuyuanfan4814/ls6/mirage-env/bin/activate

python3 -u test_in_environment.py \
    --base_model_path_4 /work/09796/shuyuanfan4814/ls6/interrupt-free-provisioning/src/model/validation/models/transformer_model/model_transformer_4.pt \
    --base_model_path_8 /work/09796/shuyuanfan4814/ls6/interrupt-free-provisioning/src/model/validation/models/transformer_model/model_transformer_8.pt\
    --base_model_path_16 /work/09796/shuyuanfan4814/ls6/interrupt-free-provisioning/src/model/validation/models/transformer_model/model_transformer_16.pt \
    --base_model_path_32 /work/09796/shuyuanfan4814/ls6/interrupt-free-provisioning/src/model/validation/models/transformer_model/model_transformer_32.pt \
    -sim_config sim_base_validation.json > transformer_validation_7days.out

python3 -u test_in_environment.py \
    --base_model_path_4 /work/09796/shuyuanfan4814/ls6/interrupt-free-provisioning/src/model/validation/models/rf_model/rf_model_4 \
    --base_model_path_8 /work/09796/shuyuanfan4814/ls6/interrupt-free-provisioning/src/model/validation/models/rf_model/rf_model_8\
    --base_model_path_16 /work/09796/shuyuanfan4814/ls6/interrupt-free-provisioning/src/model/validation/models/rf_model/rf_model_16 \
    --base_model_path_32 /work/09796/shuyuanfan4814/ls6/interrupt-free-provisioning/src/model/validation/models/rf_model/rf_model_32 \
    -sim_config sim_base_validation.json > rf_validation_7days.out

python3 -u test_in_environment.py \
    --base_model_path_4 /work/09796/shuyuanfan4814/ls6/interrupt-free-provisioning/src/model/validation/models/xgb_model/xgb_model_4 \
    --base_model_path_8 /work/09796/shuyuanfan4814/ls6/interrupt-free-provisioning/src/model/validation/models/xgb_model/xgb_model_8\
    --base_model_path_16 /work/09796/shuyuanfan4814/ls6/interrupt-free-provisioning/src/model/validation/models/xgb_model/xgb_model_16 \
    --base_model_path_32 /work/09796/shuyuanfan4814/ls6/interrupt-free-provisioning/src/model/validation/models/xgb_model/xgb_model_32 \
    -sim_config sim_base_validation.json > xgb_validation_7days.out
