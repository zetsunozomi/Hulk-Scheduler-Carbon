import os
import json
import sys
import argparse
import time
import datetime
import numpy as np
import torch
import ray
import pickle
from tqdm import tqdm

from predictor import Predictor
from predictor import raw_features_to_np_ndarray

import torch.nn as nn 
# ray.init(local_mode=True, ignore_reinit_error=True)

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

def load_data_file(file_path, parallel=False):
    if file_path.endswith('.pkl'):
        print(f"Loading pickle data from {file_path}")
        with open(file_path, "rb") as f:
            return pickle.load(f)
    elif file_path.endswith('.json'):
        # Check for cached pickle file
        cache_path = file_path.replace('.json', '.pkl')
        if os.path.exists(cache_path):
            print(f"Loading cached data from {cache_path}")
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        print(f"Loading json data from {file_path}")
        with open(file_path, 'r') as f:
            raw_data = json.load(f)
        
        samples = convert_json_to_samples(raw_data)
        
        # Prepare inputs for raw_features_to_np_ndarray
        raw_samples_list = []
        targets = []
        for s in samples:
            # Reconstruct the raw sample structure expected by predictor
            # 'job_info' and 'time_series' keys
            raw_samples_list.append({
                'time_series': s['time_series'],
                'job_info': s['job_info']
            })
            targets.append(s['queue_wait_time'])
            
        print(f"Converting features for {len(samples)} samples from {file_path} (Parallel={parallel})...")
        features_arr, _ = raw_features_to_np_ndarray(raw_samples_list, parallel=parallel)
        
        # Re-create numpy array to avoid "expected np.ndarray (got numpy.ndarray)" and "Could not infer dtype" errors.
        # This often happens when data comes from another process (Ray) or due to numpy version mismatches.
        features_arr = np.array(features_arr, dtype=np.float32)
        tensor_input = torch.tensor(features_arr, dtype=torch.float32)
        
        # Targets: reshape and normalize (divide by 3600.0)
        np_targets = np.array(targets).reshape(-1, 1) / 3600.0
        np_targets = np.array(np_targets, dtype=np.float32) # Ensure clean numpy array
        tensor_target = torch.tensor(np_targets, dtype=torch.float32)
        
        check_data_for_nan_inf(tensor_input, name=f"tensor_input_{os.path.basename(file_path)}")
        check_data_for_nan_inf(tensor_target, name=f"tensor_target_{os.path.basename(file_path)}")
        
        # Save to cache
        print(f"Saving processed data to cache: {cache_path}")
        with open(cache_path, "wb") as f:
            pickle.dump((tensor_input, tensor_target), f)

        return tensor_input, tensor_target
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

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
    args = parser.parse_args()

    TRAIN_NEPOCHS = args.epoch
    MODEL_NAME = args.model
    

    fname_model = f'model_{args.file_name}.pt'

    train_path = os.path.join(args.work_dir, args.training_data)
    validation_path = os.path.join(args.work_dir, args.validation_data)

    train_tensor_input, train_tensor_target = load_data_file(train_path, parallel=args.parallel)
    test_tensor_input, test_tensor_target = load_data_file(validation_path, parallel=args.parallel)

    print(f"train_tensor_input.shape {train_tensor_input.shape}")
    print(f"train_tensor_target.shape {train_tensor_target.shape}")
    print(f"test_tensor_input.shape {test_tensor_input.shape}")
    print(f"test_tensor_target.shape {test_tensor_target.shape}")

    print(MODEL_NAME)
    predictor = Predictor(
        model_name=MODEL_NAME,       
        hparams=MODEL_CONFIG,
        optimizer_name='AdamW',
        base_learning_rate=5e-5,
        device='gpu',  
    )
    predictor.set_seed(157)

    model_checkpoint_path = os.path.join(args.work_dir, 'model', fname_model)
    os.makedirs(os.path.dirname(model_checkpoint_path), exist_ok=True)
    train_start_time = time.time()
    print(f"[Training Start] {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    for iepoch in range(TRAIN_NEPOCHS):
        iterator = predictor.batch_iterator(train_tensor_input, train_tensor_target, min_batch_size=128, shuffle=True)
        if args.parallel:
            iterator = tqdm(iterator, desc=f"Epoch {iepoch}", leave=False)

        epoch_start_time = time.time()
        n_steps = 0
        for iter_idx, (data_input, data_target) in enumerate(iterator):
            train_loss = predictor.train_batch(data_input, data_target)
            n_steps += 1
            if args.parallel:
                iterator.set_postfix({'loss': f'{train_loss:.4f}'})
            else:
                print(f'Iteration {iter_idx}, train loss = {train_loss} | Epoch {iepoch}')

        epoch_elapsed = time.time() - epoch_start_time
        avg_step_ms = epoch_elapsed / n_steps * 1000 if n_steps > 0 else 0
        now_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f'[Epoch {iepoch} done] Time: {now_str} | Epoch time: {epoch_elapsed:.1f}s | Avg step time: {avg_step_ms:.1f} ms ({n_steps} steps)')

        validation_loss = predictor.validate(test_tensor_input, test_tensor_target, min_batch_size=128)
        print(f'[Epoch {iepoch}] Mean L1 validation error: {validation_loss:.2f} hours.')
        if iepoch % 100 == 0:
            predictor.save_checkpoint(model_checkpoint_path)
    predictor.save_checkpoint(model_checkpoint_path)
    total_elapsed = time.time() - train_start_time
    print(f"[Training End] {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Total time: {total_elapsed:.1f}s")