#!/bin/bash
#PBS -A Local-LLM
#PBS -l walltime=12:00:00
#PBS -l filesystems=home:eagle
#PBS -q by-gpu
#PBS -l select=1
#PBS -N queeu-time-train
#PBS -o queeu-time-train-4node.log
#PBS -j oe
#PBS -M sf850@scarletmail.rutgers.edu                                        
#PBS -m bae 

cd /lus/eagle/projects/Local-LLM/shuyuanfan/carbon-adaptive/src/queue_prediction/experts
PYTHON_ENV="/lus/eagle/projects/Local-LLM/shuyuanfan/conda_env/carbon/bin/python"
echo "Using Python from: $PYTHON_ENV"

$PYTHON_ENV -u train.py \
    -wd /lus/eagle/projects/Local-LLM/shuyuanfan/carbon-adaptive/src/queue_prediction/json_data/ \
    -n frontera_queue_time_predictor_transformer_4node \
    --training_data frontera_training_data_new_4/all_data.json \
    --validation_data frontera_validation_data_new_4/all_data.json\
    -epoch 50 \
    -model transformer 
$PYTHON_ENV -u train.py \
    -wd /lus/eagle/projects/Local-LLM/shuyuanfan/carbon-adaptive/src/queue_prediction/json_data/ \
    -n frontera_queue_time_predictor_transformer_8node \
    --training_data frontera_training_data_new_8/all_data.json \
    --validation_data frontera_validation_data_new_8/all_data.json\
    -epoch 50 \
    -model transformer 
$PYTHON_ENV -u train.py \
    -wd /lus/eagle/projects/Local-LLM/shuyuanfan/carbon-adaptive/src/queue_prediction/json_data/ \
    -n frontera_queue_time_predictor_transformer_16node \
    --training_data frontera_training_data_new_16/all_data.json \
    --validation_data frontera_validation_data_new_16/all_data.json\
    -epoch 50 \
    -model transformer 
$PYTHON_ENV -u train.py \
    -wd /lus/eagle/projects/Local-LLM/shuyuanfan/carbon-adaptive/src/queue_prediction/json_data/ \
    -n frontera_queue_time_predictor_transformer_32node \
    --training_data frontera_training_data_new_32/all_data.json \
    --validation_data frontera_validation_data_new_32/all_data.json\
    -epoch 50 \
    -model transformer 