import json
import os
import sys
import gc
import argparse
import random
import numpy as np

sys.path.append("../")
sys.path.append("../../")

import torch
import torch.optim as optim
import torch.nn as nn
from torch.distributions import Categorical
from torch.nn.functional import one_hot, log_softmax, softmax
from collections import deque

from env import Env
from sim.application import Application
from moe.model import StochasticMixtureOfExperts
from moe.predictor import raw_features_to_np_ndarray, Predictor

import logging
logging.basicConfig(level=logging.INFO)

class NormalizedTransformer(nn.Module):
    def __init__(self, transformer_model, output_dim):
        super(NormalizedTransformer, self).__init__()
        self.transformer_model = transformer_model
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        out = self.transformer_model(x)
        out_norm = self.norm(out)
        return out_norm

def infer_wrapper(ep_rewards, ep_scaled_rewards, device, actor):
    def callback_infer(sim_info, data_input, reward=None):
        if reward is not None:
            if isinstance(reward, tuple):
                train_r, scaled_r = reward
            else:
                train_r, scaled_r = reward, reward
            ep_rewards[0] = np.concatenate((ep_rewards[0], [train_r]), axis=0)
            ep_scaled_rewards[0] = np.concatenate((ep_scaled_rewards[0], [scaled_r]), axis=0)
            return None

        np_input = raw_features_to_np_ndarray([data_input], parallel=False)
        tensor_in = torch.from_numpy(np_input).float().to(device)
        
        with torch.no_grad():
            logits = actor(tensor_in)
        
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
            
        dist = Categorical(logits=logits)
        action = dist.sample()
        
        print(f"[Callback] Action probabilities: {probs.cpu().numpy()}")
        print(f"[Callback] Action chosen: {action.item()}")
        
        return action.item()
    return callback_infer

class Validator:
    def __init__(self, model_ckpt, model4_path, model8_path, model16_path, model32_path, sim_config, use_cuda=False):
        self.DEVICE = torch.device('cuda' if (torch.cuda.is_available() and use_cuda) else 'cpu')
        self.sim_config = sim_config
        self.env = Env(**sim_config)
        self.app = Application()

        def load_pred(ckpt_path):
            c = torch.load(ckpt_path, map_location=self.DEVICE)
            predictor = Predictor(
                model_name=c['model_name'], hparams=c['model_hparams'], optimizer_name=c['optimizer_name'],
                base_learning_rate=1e-5, model_state=c['model_state_dict'], optimizer_state=c['optimizer_state_dict'],
                device=("cuda" if self.DEVICE.type=='cuda' else 'cpu'), seed=0, debug=False
            )
            predictor.model.eval()
            predictor.model = NormalizedTransformer(predictor.model, output_dim=1)
            return predictor

        p4  = load_pred(model4_path)
        p8  = load_pred(model8_path)
        p16 = load_pred(model16_path)
        p32 = load_pred(model32_path)
        
        self.actor = StochasticMixtureOfExperts(
            num_experts=4,
            expert_models=[p4.model, p8.model, p16.model, p32.model]
        ).to(self.DEVICE)

        ckpt = torch.load(model_ckpt, map_location=self.DEVICE)
        self.actor.load_state_dict(ckpt['actor_state_dict'])
        self.actor.eval()

    def run_episode(self):
        self.env.reset()
        self.app.reset()

        episode_rewards = [np.empty(shape=(0,), dtype=float)]
        episode_scaled_rewards = [np.empty(shape=(0,), dtype=float)]

        callback_infer = infer_wrapper(episode_rewards, episode_scaled_rewards, self.DEVICE, self.actor)

        done = False
        step_idx = 0
        while not done:
            data_input = self.env.get_current_data_input()
            sim_info = [self.env.sim._scheduler.job_logs, self.env.sim._scheduler.avail_nodes]
            next_job_id = 100000 + step_idx
            reward_val, scaled_reward_val, done = self.app.run_job_and_measure_reward(
                env=self.env, new_job_id=next_job_id, infer_func=callback_infer,
                sim_info=sim_info, data_input=data_input
            )
            step_idx += 1

        total_episode_reward = np.sum(episode_rewards[0])
        total_episode_scaled_reward = np.sum(episode_scaled_rewards[0])
        return total_episode_reward, total_episode_scaled_reward

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', required=True, help="环境配置文件")
    parser.add_argument('--model_ckpt', required=True, help="训练好的模型检查点文件")
    parser.add_argument('--model4', required=True)
    parser.add_argument('--model8', required=True)
    parser.add_argument('--model16', required=True)
    parser.add_argument('--model32', required=True)
    parser.add_argument('--episodes', type=int, default=10, help="验证时运行的 episode 数量")
    parser.add_argument('--use_cuda', action='store_true', help="是否使用 GPU")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        sim_json = json.load(f)

    validator = Validator(
        model_ckpt=args.model_ckpt,
        model4_path=args.model4,
        model8_path=args.model8,
        model16_path=args.model16,
        model32_path=args.model32,
        sim_config=sim_json,
        use_cuda=args.use_cuda
    )

    rewards = []
    scaled_rewards = []
    for i in range(args.episodes):
        total_reward, total_scaled_reward = validator.run_episode()
        rewards.append(total_reward)
        scaled_rewards.append(total_scaled_reward)
        print(f"Episode {i+1}/{args.episodes} | Total Reward: {total_reward:.3f}, Scaled Reward: {total_scaled_reward:.3f}")

    print("-" * 30)
    print(f"Average reward over {args.episodes} episodes: {np.mean(rewards):.3f}")
    print(f"Average scaled reward over {args.episodes} episodes: {np.mean(scaled_rewards):.3f}")

if __name__ == "__main__":
    main()