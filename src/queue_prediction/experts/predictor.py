import numpy as np
import torch
import random
import sys
import torch.nn as nn
from model import NeuralLinearRegressor, NerualConvRegressor, TransformerRegressor, LSTM
import torch.optim.lr_scheduler as lr_scheduler
import ray
from tqdm import tqdm

class Predictor:
    def __init__(self,
                 model_name,
                 hparams,
                 optimizer_name,
                 base_learning_rate,
                 device,

                 model_state=None,
                 optimizer_state=None,

                 seed=0,
                 debug=True):

        self.set_seed(seed)

        self.model_name = model_name
        self.model_hparams = hparams

        if self.model_name == 'linear':
            self.model = NeuralLinearRegressor(
                in_dim=self.model_hparams['in_dim'] * self.model_hparams['seq_len'],
                n_hidden_units=self.model_hparams['n_hidden_units'],
                activation=self.model_hparams['activation']
            )
        elif self.model_name == 'convolution':
            self.model = NerualConvRegressor(
                in_dim=self.model_hparams['in_dim'],
                seq_len=self.model_hparams['seq_len']
            )
        elif self.model_name == 'transformer':
            self.model = TransformerRegressor(
                in_size=self.model_hparams['in_dim'],
                seq_len=self.model_hparams['seq_len'],
                embed_size=self.model_hparams['embed_size'],
                encoder_nlayers=self.model_hparams['encoder_nlayers'],
                encoder_nheads=self.model_hparams['encoder_nheads'],
                dim_feedforward=self.model_hparams['dim_feedforward']
            )
        elif self.model_name == 'lstm':
            self.model = LSTM(
                input_size=self.model_hparams['in_dim'],
                hidden_size=self.model_hparams["hidden_size"],
                num_layers=self.model_hparams["num_layers"]
            )
        else:
            print(f'Error: Model {self.model_name} is not supported!')
            sys.exit()

        
        if model_state is not None:
            self.model.load_state_dict(model_state)

        self.device = torch.device('cuda' if torch.cuda.is_available() and device == 'gpu' else 'cpu')
        self.model.to(self.device)

        self.base_lr = base_learning_rate
        self.optimizer_name = optimizer_name
        if self.optimizer_name == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.base_lr)
            self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)
        elif self.optimizer_name == 'AdamW':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.base_lr, weight_decay=1e-1)
            self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)
        else:
            print(f'Error: Optimizer {optimizer_name} is not supported!')
            sys.exit()

        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)

        self.l1_loss_mean = nn.L1Loss(reduction='mean')
        self.l1_loss_sum = nn.L1Loss(reduction='sum')
        self.l2_loss_mean = nn.MSELoss(reduction='mean')
        self.l2_loss_sum = nn.MSELoss(reduction='sum')
        self.sm_l1 = nn.SmoothL1Loss(reduction='mean', beta=1.0)

        if debug:
            print(f'Model architecture: {self.model}')
            print(f'Model is placed on device: {next(self.model.parameters()).device}')

    def set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        random.seed(seed)
        np.random.seed(seed)

    def train_batch(self, tensor_input, tensor_target):
        
        if self.model_name == 'convolution':
            tensor_input = torch.unsqueeze(tensor_input, 1)

        tensor_input = tensor_input.to(self.device).float()
        tensor_target = tensor_target.to(self.device).float()

        self.optimizer.zero_grad()
        tensor_output = self.model(tensor_input)
        loss = self.l1_loss_mean(tensor_output, tensor_target)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        return loss.item()

    def inference(self, tensor_input):
        if self.model_name == 'convolution':
            tensor_input = torch.unsqueeze(tensor_input, 1)
        tensor_input = tensor_input.to(self.device).float()
        return self.model(tensor_input)

    def detach_tensor_to_numpy(self, t):
        return t.cpu().detach().numpy()

    def validate(self, tensor_input, tensor_target, min_batch_size=128):
    
        if self.model_name == 'convolution':
            tensor_input = torch.unsqueeze(tensor_input, 1)

        iterator = self.batch_iterator(tensor_input, tensor_target, min_batch_size, shuffle=False)
        output_list = []
        target_list = []

        for iter_idx, data in enumerate(iterator):
            data_input, data_target = data
            data_input = data_input.to(self.device).float()
            with torch.no_grad():
                output = self.model(data_input).cpu().detach()
                output_list.append(output)
                target_list.append(data_target)

        validation_tensor_output = torch.concat(output_list, dim=0)
        validation_tensor_target = torch.concat(target_list, dim=0)
        loss = self.l1_loss_mean(validation_tensor_output, validation_tensor_target)
        current_lr = self.optimizer.param_groups[0]['lr']
        print(f"current lr:{current_lr}")
        self.scheduler.step()
        return loss.numpy().tolist()

    def batch_iterator(self, tensor_input, tensor_output, min_batch_size=128, shuffle=True):
      
        class Samples(torch.utils.data.Dataset):
            def __init__(self, inp, tgt):
                self.inp = inp
                self.tgt = tgt

            def __len__(self):
                return len(self.inp)

            def __getitem__(self, idx):
                return self.inp[idx], self.tgt[idx]

        dataset = Samples(tensor_input, tensor_output)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=min_batch_size,
            shuffle=shuffle,  
            num_workers=0
        )
        return loader

    def save_checkpoint(self, checkpoint_path):
        try:
            torch.save({
                'model_name': self.model_name,
                'model_hparams': self.model_hparams,
                'model_state_dict': self.model.state_dict(),
                'optimizer_name': self.optimizer_name,
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, checkpoint_path)
            print(f"Model {self.model_name} is successfully checkpointed to {checkpoint_path}")
        except:
            print(f'Error: Failed checkpointing to path: {checkpoint_path}!')


@ray.remote
def raw_feature_to_np_ndarray_call_rpc(raw_sample):
    return raw_feature_to_np_ndarray_call(*raw_sample)

def raw_feature_to_np_ndarray_call(time_series_data, job_features):
    sample_features = []

    job_time_limit_minutes = job_features["time_limit"]
    job_nodes = job_features["nodes"]
    job_time_limit = job_time_limit_minutes / 60.0

    for idx, snapshot in enumerate(time_series_data):
        pending_jobs = snapshot["pending_list"]
        running_jobs = snapshot["running_list"]

        def get_stats(jobs, key):
            if key == 'time_left':
                values = []
                for job in jobs:
                    run_time = job.get('run_sec',0)
                    run_time = run_time / 3600.0
                    time_limit = job.get('time_limit')
                    time_limit = time_limit / 60.0
                    time_left = time_limit - run_time
                    values.append(time_left)
                if not values:
                    values = [0.0]
                return [np.percentile(values, p) for p in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]]
    
            values = []
            for job in jobs:
                v = job.get(key, 0)
                if key == 'time_limit':
                    v = v / 60.0  # 分钟 => 小时
                elif key in ('queue_wait_sec', 'run_sec', 'wait_sec'):
                    v = v / 3600.0  # 秒 => 小时
                values.append(v)
            if not values:
                values = [0.0]
            return [np.percentile(values, p) for p in [0, 5, 10, 15, 20, 50, 80, 85, 90, 95, 100]]
        # pending
        pending_count = len(pending_jobs)
        pending_nodes = sum(job["nodes"] for job in pending_jobs)
        # pending_nodes_stats = get_stats(pending_jobs, "nodes")
        # 取统计量的时候相当于重新排序，time_limit和queue wait time的排序明显不同
        # 即：1/20, 19/200 和 19/20, 1/200是完全等价的输入
        # 同时很显然，time left = time_limit - run_time才是更关键的决定因素。
        # 但是取了统计量，相当于排序，而且采样后job根本没有对应关系，这个线性关系实际上是学习不到的
        pending_time_limit_stats = get_stats(pending_jobs, "time_limit")
        pending_queue_stats = get_stats(pending_jobs, "queue_wait_sec")
    
        # pending runtime is meaningless
        # pending_runtime_stats = get_stats(pending_jobs, "run_sec")

        # running
        running_count = len(running_jobs)
        running_nodes = sum(job["nodes"] for job in running_jobs)
        #running_nodes_stats = get_stats(running_jobs, "nodes")
        running_time_limit_stats = get_stats(running_jobs, "time_limit")
        running_queue_stats = get_stats(running_jobs, "queue_wait_sec")
        running_runtime_stats = get_stats(running_jobs, "run_sec")
        running_time_left_stats = get_stats(running_jobs, "time_left")

        step_features = [
            pending_count,
            pending_nodes,
            *pending_time_limit_stats,
            *pending_queue_stats,
            running_count,
            running_nodes,
            *running_time_limit_stats,
            *running_queue_stats,
            *running_runtime_stats,
            *running_time_left_stats
        ]
        #step_features.append(job_time_limit)
        #step_features.append(job_nodes)
        sample_features.append(step_features)

    sample_features_arr = np.array(sample_features, dtype=np.float32)
    job_feature_vector = np.array([job_time_limit, job_nodes], dtype=np.float32)

    return sample_features_arr, job_feature_vector

def raw_features_to_np_ndarray(raw_samples, parallel=False):
    n_samples = len(raw_samples)
    ndarray_features, job_features_list = [], []

    if parallel:
        ray.init(ignore_reinit_error=True)
        rpc_handles = []
        for sample in raw_samples:
            time_series_data = sample['time_series']
            job_info = sample['job_info']
            rpc_handles.append(raw_feature_to_np_ndarray_call_rpc.remote((time_series_data, job_info)))

        # Parallel processing with progress bar
        for handle in tqdm(rpc_handles, desc="Parallel Processing"):
            time_series_features, job_features = ray.get(handle)
            ndarray_features.append(time_series_features)
            job_features_list.append(job_features)
    else:
        # Serial processing with progress bar
        for sample in tqdm(raw_samples, desc="Serial Processing"):
            time_series_data = sample['time_series']
            job_info = sample['job_info']
            time_series_features, job_features = raw_feature_to_np_ndarray_call(time_series_data, job_info)
            ndarray_features.append(time_series_features)
            job_features_list.append(job_features)

    return np.array(ndarray_features), np.array(job_features_list)