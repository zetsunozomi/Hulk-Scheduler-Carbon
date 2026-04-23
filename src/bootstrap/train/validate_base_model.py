import os
import json
import sys
import argparse
import numpy as np
import torch
import ray
import pickle
from predictor import Predictor
from predictor import raw_features_to_np_ndarray
import torch.nn as nn 
ray.init(local_mode=True, ignore_reinit_error=True)

MODEL_CONFIG = {
    'in_dim': 70,      
    'seq_len':288,    
    'embed_size': 512,
    'encoder_nlayers': 8,
    'encoder_nheads': 8,
    'dim_feedforward': 2048,
    'n_hidden_units':[2048,1024,512,256,128],
    'activation': nn.ReLU
}
MODEL_NAME = 'transformer'
PARALLEL_PREPROCESSING = True 

def check_data_for_nan_inf(tensor, name=""):
    if torch.isnan(tensor).any():
        print(f"[WARNING] Found NaN in {name}")
    if torch.isinf(tensor).any():
        print(f"[WARNING] Found Inf in {name}")

def convert_json_to_samples(raw_data):
    
    samples = []
    for idx, subarr in enumerate(raw_data):
        if not subarr:
            continue

        job_info_dict = subarr[-1]
        time_series_list = subarr[:-1]  

        time_limit = job_info_dict.get("time_limit", 0)
        nodes = job_info_dict.get("nodes", 0)
        queue_wait_sec = job_info_dict.get("queue_wait_sec", 0)

        sample_dict = {
            "job_info": {
                "time_limit": time_limit,
                "nodes": nodes
            },
            "queue_wait_time": queue_wait_sec,  
            "time_series": time_series_list    
        }

        
        print(f"[convert_json_to_samples] Simulation #{idx} => # of snapshots: {len(time_series_list)}")

        samples.append(sample_dict)

    return samples

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-wd", "--work_dir", default="./", help="work directory")
    parser.add_argument("-n", "--file_name", required=True, help="model name")
    parser.add_argument("--training_data", required=True, help="training data")
    parser.add_argument("--validation_data", required=True, help="validation data")
    parser.add_argument("-cpu_cores", default=None, help="cpu")
    parser.add_argument("-parallel", action="store_true", default=False, help="parallel running")
    parser.add_argument("-model", default="transformer", help="running model name")
    parser.add_argument("-epoch", type=int, default=300, help="iteration")
    parser.add_argument("-debug", type=bool, default=0, help="debug mode")
    args = parser.parse_args()

    MODEL_NAME = args.model
    PARALLEL_PREPROCESSING = args.parallel

    fname_model = f'model_{args.file_name}.pt'
    train_path = f"{args.work_dir}{args.training_data}"
    validation_path = f"{args.work_dir}{args.validation_data}"
    with open(train_path, "rb") as f:
        train_tensor_input, train_tensor_target = pickle.load(f)
    with open(validation_path, "rb") as f:
        test_tensor_input, test_tensor_target = pickle.load(f)
    print(f"train_tensor_input.shape {train_tensor_input.shape}")
    print(f"train_tensor_target.shape {train_tensor_target.shape}")
    print(f"test_tensor_input.shape {test_tensor_input.shape}")
    print(f"test_tensor_target.shape {test_tensor_target.shape}")
    print(MODEL_NAME)

    def load_pred(ckpt_path):
        c = torch.load(ckpt_path, map_location=torch.device('cpu'))
        print(f"Loading model: {c['model_hparams']}")
        predictor =  Predictor(
            model_name= c['model_name'],
            hparams= c['model_hparams'],
            optimizer_name= c['optimizer_name'],
            base_learning_rate= 1e-3,
            model_state= c['model_state_dict'],
            optimizer_state= c['optimizer_state_dict'],
            device=('cpu'),
            seed=0,
            debug=False
        )
        predictor.model.eval()
        return predictor
    model4_path = f"/work/09796/shuyuanfan4814/ls6/interrupt-free-provisioning-data-generation/prediction_model/current-choice/model_transformer_4_1.pt"
    model8_path = f"/work/09796/shuyuanfan4814/ls6/interrupt-free-provisioning-data-generation/prediction_model/current-choice/model_transformer_8_1.pt"
    model16_path = f"/work/09796/shuyuanfan4814/ls6/interrupt-free-provisioning-data-generation/prediction_model/current-choice/model_transformer_16_1.pt"
    model32_path = f"/work/09796/shuyuanfan4814/ls6/interrupt-free-provisioning-data-generation/prediction_model/current-choice/model_transformer_32_1.pt"
    p4  = load_pred(model4_path)
    p8  = load_pred(model8_path)
    p16 = load_pred(model16_path)
    p32 = load_pred(model32_path)
    '''
    validation_loss = p4.validate(test_tensor_input, test_tensor_target, min_batch_size=128)
    print(f'p4 Mean L1 validation error: {validation_loss:.2f} hours.')
    validation_loss = p8.validate(test_tensor_input, test_tensor_target, min_batch_size=128)
    print(f'p8 Mean L1 validation error: {validation_loss:.2f} hours.')
    validation_loss = p16.validate(test_tensor_input, test_tensor_target, min_batch_size=128)
    print(f'p16 Mean L1 validation error: {validation_loss:.2f} hours.')
    validation_loss = p32.validate(test_tensor_input, test_tensor_target, min_batch_size=128)
    print(f'p32 Mean L1 validation error: {validation_loss:.2f} hours.')
    '''
    # todo: do some inference and calculate it's average
    # 从 test_tensor_input 中取一个 batch（前 64 个样本）
    input_batch = test_tensor_input[:64]

    # 模型推理
    output = p32.inference(input_batch)

    # 打印输出的类型（应为 float）
    print("Output dtype:", output.dtype)

    # 打印输出的平均值
    print("Output mean:", output.mean().item())
