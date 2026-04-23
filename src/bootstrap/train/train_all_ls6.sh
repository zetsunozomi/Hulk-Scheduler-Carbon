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


cd /work/09796/shuyuanfan4814/ls6/carbon/src/queue_prediction/experts
PYTHON_ENV="/work/09796/shuyuanfan4814/ls6/miniconda3/envs/carbon/bin/python"
echo "Using Python from: $PYTHON_ENV"

$PYTHON_ENV -u train.py \
    -wd /work/09796/shuyuanfan4814/ls6/carbon/src/queue_prediction/json_data/ \
    -n ls6_queue_time_predictor_transformer_4node \
    --training_data ls6_training_data_new_4/all_data.json \
    --validation_data ls6_validation_data_new_4/all_data.json \
    -epoch 50 \
    -model transformer
$PYTHON_ENV -u train.py \
    -wd /work/09796/shuyuanfan4814/ls6/carbon/src/queue_prediction/json_data/ \
    -n ls6_queue_time_predictor_transformer_8node \
    --training_data ls6_training_data_new_8/all_data.json \
    --validation_data ls6_validation_data_new_8/all_data.json \
    -epoch 50 \
    -model transformer
$PYTHON_ENV -u train.py \
    -wd /work/09796/shuyuanfan4814/ls6/carbon/src/queue_prediction/json_data/ \
    -n ls6_queue_time_predictor_transformer_16node \
    --training_data ls6_training_data_new_16/all_data.json \
    --validation_data ls6_validation_data_new_16/all_data.json \
    -epoch 50 \
    -model transformer
$PYTHON_ENV -u train.py \
    -wd /work/09796/shuyuanfan4814/ls6/carbon/src/queue_prediction/json_data/ \
    -n ls6_queue_time_predictor_transformer_32node \
    --training_data ls6_training_data_new_32/all_data.json \
    --validation_data ls6_validation_data_new_32/all_data.json \
    -epoch 50 \
    -model transformer
