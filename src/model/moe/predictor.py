import numpy as np
import torch
import random
import sys
import torch.nn as nn
import sys
sys.path.append("../")
from moe.model import NeuralLinearRegressor, NerualConvRegressor, TransformerRegressor, MixtureOfExperts, \
    StochasticMixtureOfExperts, StochasticTransformer
import ray


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

        if (self.model_name == 'linear'):
            self.model = NeuralLinearRegressor(
                in_dim=self.model_hparams['in_dim'] * self.model_hparams['seq_len'],
                n_hidden_units=self.model_hparams['n_hidden_units'],
                activation=self.model_hparams['activation']
            )
        elif (self.model_name == 'convolution'):
            self.model = NerualConvRegressor(in_dim=self.model_hparams['in_dim'],
                                             seq_len=self.model_hparams['seq_len'], )

        elif (self.model_name == 'transformer'):
            self.model = TransformerRegressor(in_size=self.model_hparams['in_dim'],
                                              seq_len=self.model_hparams['seq_len'],
                                              embed_size=self.model_hparams['embed_size'],
                                              encoder_nlayers=self.model_hparams['encoder_nlayers'],
                                              encoder_nheads=self.model_hparams['encoder_nheads'],
                                              dim_feedforward=self.model_hparams['dim_feedforward'])
        elif (self.model_name == "moe"):
            self.model = MixtureOfExperts(in_dim=self.model_hparams['in_dim'],
                                          expert_models=self.model_hparams['experts'])
        elif self.model_name == "moe_policy":
            self.model = StochasticMixtureOfExperts(num_experts=len(self.model_hparams['experts']),
                                                    expert_models=self.model_hparams['experts'])
        elif self.model_name == "transformer_policy":
            self.model = StochasticTransformer(in_dim=self.model_hparams['in_dim'],
                                               base_model=self.model_hparams['base'])
        else:
            print(f'Error: Model {self.model_name} is not supported!')
            sys.exit()

        if (model_state is not None):
            self.model.load_state_dict(model_state)

        self.device = torch.device('cuda' if torch.cuda.is_available() and device == 'gpu' else 'cpu')
        self.model.to(self.device)

        self.base_lr = base_learning_rate
        self.optimizer_name = optimizer_name
        if (self.optimizer_name == 'Adam'):
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.base_lr)
        elif (self.optimizer_name == 'AdamW'):
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.base_lr) 
        else:
            print(f'Error: Optimizer {optimizer_name} is not supported!')
            sys.exit()

        if (optimizer_state is not None):
            self.optimizer.load_state_dict(optimizer_state)

        self.l1_loss = nn.L1Loss(reduction='none')
        self.mse_loss = nn.MSELoss()

        if (debug):
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
        if (self.model_name == 'convolution'):
            tensor_input = torch.unsqueeze(tensor_input, 1)

        tensor_input = tensor_input.to(self.device)
        tensor_target = tensor_target.to(self.device)
        tensor_input, tensor_target = tensor_input.float(), tensor_target.float()
        self.optimizer.zero_grad()
        tensor_output = self.model(tensor_input)
        loss = self.mse_loss(tensor_output, tensor_target)
        loss.backward()
        self.optimizer.step()
        return loss.cpu().detach().numpy().tolist()

    def inference(self, tensor_input):
        if (self.model_name == 'convolution'):
            tensor_input = torch.unsqueeze(tensor_input, 1)
        tensor_input = tensor_input.to(self.device)
        tensor_input = tensor_input.float()
        return self.model(tensor_input)

    def detach_tensor_to_numpy(self, t):
        return t.cpu().detach().numpy()

    def validate(self, tensor_input, tensor_target, batch_size):
        iterator = self.batch_iterator(tensor_input, tensor_target, batch_size)
        output_list = []
        target_list = []
        for iter, data in enumerate(iterator):
            data_input, data_target = data
            data_input = data_input.to(self.device).float()
            with torch.no_grad():
                output = self.model(data_input).cpu().detach()
                output_list.append(output)
                target = data_target.cpu().detach()
                target_list.append(target)
        validation_tensor_output = torch.concat(output_list, dim=0)
        validation_tensor_target = torch.concat(target_list, dim=0)
        l1_loss = self.l1_loss(validation_tensor_output, validation_tensor_target)
        return l1_loss.numpy().tolist()

    def batch_iterator(self, tensor_input, tensor_output, batch_size):
        class Samples(torch.utils.data.Dataset):
            def __init__(self, input, target):
                self.input = input
                self.target = target

            def __len__(self):
                return len(self.input)

            def __getitem__(self, idx):
                return self.input[idx], self.target[idx]

        iterator = torch.utils.data.DataLoader(Samples(tensor_input, tensor_output),
                                               batch_size=batch_size, shuffle=True, num_workers=0, )
        return iterator

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
    return raw_feature_to_np_ndarray_call(raw_sample)

def raw_feature_to_np_ndarray_call(raw_sample):
    time_series_data, job_info = raw_sample
    sample_features = []
    
    job_time_limit = job_info["time_limit"] / 60.0  # hour
    job_nodes = job_info["nodes"]

    for snapshot in time_series_data:
        pending_jobs = snapshot.get("pending_list", [])
        running_jobs = snapshot.get("running_list", [])

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
                # Match init_version: 11 percentiles for time_left
                return [np.percentile(values, p) for p in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]]
    
            values = []
            for job in jobs:
                v = job.get(key, 0)
                if key == 'time_limit':
                    v = v / 60.0  # min to hour
                elif key in ('queue_wait_sec', 'run_sec', 'wait_sec'):
                    v = v / 3600.0  # s to hour
                values.append(v)
            if not values:
                values = [0.0]
            # Match init_version: 11 percentiles for others
            return [np.percentile(values, p) for p in [0, 5, 10, 15, 20, 50, 80, 85, 90, 95, 100]]

        pending_count = len(pending_jobs)
        pending_nodes = sum(job["nodes"] for job in pending_jobs)
        pending_time_stats = get_stats(pending_jobs, "time_limit")
        pending_queue_stats = get_stats(pending_jobs, "queue_wait_sec")
        # pending_runtime_stats = get_stats(pending_jobs, "run_sec") # Not used in init_version

        running_count = len(running_jobs)
        running_nodes = sum(job["nodes"] for job in running_jobs)
        running_time_stats = get_stats(running_jobs, "time_limit")
        running_queue_stats = get_stats(running_jobs, "queue_wait_sec")
        running_runtime_stats = get_stats(running_jobs, "run_sec")
        running_time_left_stats = get_stats(running_jobs, "time_left") # Added to match init_version

        # Reconstruct exactly the 70-dimension feature vector from init_version
        step_features = [
            pending_count,          
            pending_nodes,          
            *pending_time_stats,    
            *pending_queue_stats,   
            # *pending_runtime_stats, # Missing in init_version

            running_count,          
            running_nodes,          
            *running_time_stats,    
            *running_queue_stats,   
            *running_runtime_stats, 
            *running_time_left_stats
        ]

        sample_features.append(step_features)
        
    job_feature_vector = np.array([job_info["time_limit"], job_info["nodes"]], dtype=np.float32)
    return np.array(sample_features, dtype=np.float32), job_feature_vector

def raw_features_to_np_ndarray(raw_samples, parallel=False):
    ndarray_features = []
    
    for sample in raw_samples:
        time_series_data = sample['time_series']
        job_info = sample['job_info']
        time_series_features, job_features = raw_feature_to_np_ndarray_call((time_series_data, job_info))
    
        # The checkpoints were trained WITHOUT appending job features, so we just use time_series_features (70 dim)
        # job_features_2d = np.repeat(job_features[np.newaxis, :], time_series_features.shape[0], axis=0)
        # full_features = np.concatenate([time_series_features, job_features_2d], axis=1)
        
        ndarray_features.append(time_series_features)
    
    return np.array(ndarray_features)

