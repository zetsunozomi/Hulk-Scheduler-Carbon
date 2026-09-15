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
import torch.nn.functional as F
from torch.nn.functional import one_hot, log_softmax, softmax
from torch.utils.tensorboard import SummaryWriter
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

class Params:
    GAMMA = 0.99
    BETA = 0.3  
    ACTION_SPACE = 4

    def __init__(self, lr, epoch, batch_size):
        self.ALPHA = lr
        self.NUM_EPOCHS = epoch
        self.BATCH_SIZE = batch_size

class Critic(nn.Module):
    def __init__(self, model4_path, model8_path, model16_path, model32_path, device):
        super(Critic, self).__init__()

        def load_pred(ckpt_path):
            c = torch.load(ckpt_path, map_location=device)
            predictor = Predictor(
                model_name=c['model_name'], hparams=c['model_hparams'], optimizer_name=c['optimizer_name'],
                base_learning_rate=1e-5, model_state=c['model_state_dict'], optimizer_state=c['optimizer_state_dict'],
                device=("cuda" if device.type=='cuda' else 'cpu'), seed=0, debug=False
            )
            predictor.model.eval()
            predictor.model = NormalizedTransformer(predictor.model, output_dim=1)
            return predictor

        p4  = load_pred(model4_path)
        p8  = load_pred(model8_path)
        p16 = load_pred(model16_path)
        p32 = load_pred(model32_path)
        
        self.base = StochasticMixtureOfExperts(
            num_experts=4,
            expert_models=[p4.model, p8.model, p16.model, p32.model]
        )
        # outpus V(s)
        self.value_head = nn.Linear(Params.ACTION_SPACE, 1)

    def forward(self, x):
        features = self.base(x)
        value = self.value_head(features)
        return value.squeeze(-1) 

def infer_wrapper(ep_log_probs, ep_values, ep_logits, ep_rewards, ep_scaled_rewards, device, actor, critic):
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
        
        logits = actor(tensor_in)
        value = critic(tensor_in)
        
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        ep_log_probs[0] = torch.cat((ep_log_probs[0], log_prob), dim=0)
        ep_values[0] = torch.cat((ep_values[0], value), dim=0)
        ep_logits[0] = torch.cat((ep_logits[0], logits), dim=0)
        
        return action.item()
    return callback_infer

class ActorCriticTrainer:
    def __init__(self,
                 train_cfg: Params,
                 use_cuda: bool,
                 model4_path, model8_path, model16_path, model32_path,
                 base_lr: float,
                 sim_config: dict,
                 model_output: str = "gpt_2B/exp_tat_2/train_tat_1/tranformer_ac"):

        self.NUM_EPOCHS = train_cfg.NUM_EPOCHS
        self.ALPHA = train_cfg.ALPHA
        self.BATCH_SIZE = train_cfg.BATCH_SIZE
        self.GAMMA = Params.GAMMA
        self.BETA = Params.BETA
        self.ACTION_SPACE = Params.ACTION_SPACE

        self.DEVICE = torch.device('cuda' if (torch.cuda.is_available() and use_cuda) else 'cpu')
        self.writer = SummaryWriter(comment=f"_AC_App_lr={self.ALPHA}_BS={self.BATCH_SIZE}")
        self.model_output = model_output
        
        self.env = Env(**sim_config)
        self.app = Application()

        self.total_rewards = deque([], maxlen=100)
        self.total_scaled_rewards = deque([], maxlen=100)

        def load_pred(ckpt_path):
            c = torch.load(ckpt_path, map_location=self.DEVICE)
            predictor = Predictor(
                model_name=c['model_name'], hparams=c['model_hparams'], optimizer_name=c['optimizer_name'],
                base_learning_rate=base_lr, model_state=c['model_state_dict'], optimizer_state=c['optimizer_state_dict'],
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

        self.critic = Critic(model4_path, model8_path, model16_path, model32_path, self.DEVICE).to(self.DEVICE)

        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=self.ALPHA
        )

    def solve_environment(self, init_epoch=0, save_checkpoint=-1):
        episode = 0
        epoch = init_epoch

        epoch_log_probs = torch.empty(size=(0,), device=self.DEVICE)
        epoch_values = torch.empty(size=(0,), device=self.DEVICE)
        epoch_discounted_rewards = torch.empty(size=(0,), device=self.DEVICE)
        epoch_logits = torch.empty(size=(0, self.ACTION_SPACE), device=self.DEVICE)
        
        current_epoch_rewards = []
        current_epoch_scaled_rewards = []

        while epoch < self.NUM_EPOCHS:
            log_probs, values, discounted_rewards, logits, total_reward, scaled_reward, episode = self.play_one_episode(episode)
            
            current_epoch_rewards.append(total_reward)
            current_epoch_scaled_rewards.append(scaled_reward)
        
            epoch_log_probs = torch.cat((epoch_log_probs, log_probs), dim=0)
            epoch_values = torch.cat((epoch_values, values), dim=0)
            epoch_discounted_rewards = torch.cat((epoch_discounted_rewards, discounted_rewards), dim=0)
            epoch_logits = torch.cat((epoch_logits, logits), dim=0)

            if episode >= self.BATCH_SIZE:
                epoch += 1

                loss, entropy = self.calculate_loss(epoch_log_probs, epoch_values, epoch_discounted_rewards, epoch_logits)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                avg_total_reward = np.mean(current_epoch_rewards)
                avg_scaled_reward = np.mean(current_epoch_scaled_rewards)
                print(f"[Epoch {epoch}] Loss: {loss.item():.3f}, Avg Reward: {avg_total_reward:.3f}, Avg Scaled Reward: {avg_scaled_reward:.3f}")

                epoch_log_probs = torch.empty(size=(0,), device=self.DEVICE)
                epoch_values = torch.empty(size=(0,), device=self.DEVICE)
                epoch_discounted_rewards = torch.empty(size=(0,), device=self.DEVICE)
                epoch_logits = torch.empty(size=(0, self.ACTION_SPACE), device=self.DEVICE)
                current_epoch_rewards = []
                current_epoch_scaled_rewards = []
                episode = 0

                del loss
                gc.collect()
                torch.cuda.empty_cache()

            episode += 1

        print("=== Training end ===")
        checkpoint_path = f"{self.model_output}_final.pt"
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch,
        }, checkpoint_path)
        print(f"Final model saved to {checkpoint_path}")

    def play_one_episode(self, episode):
        self.env.reset()
        self.app.reset()

        ep_log_probs = [torch.empty(size=(0,), device=self.DEVICE)]
        ep_values = [torch.empty(size=(0,), device=self.DEVICE)]
        ep_logits = [torch.empty(size=(0, self.ACTION_SPACE), device=self.DEVICE)]
        ep_rewards = [np.empty(shape=(0,), dtype=float)]
        ep_scaled_rewards = [np.empty(shape=(0,), dtype=float)]

        callback_infer = infer_wrapper(ep_log_probs, ep_values, ep_logits, ep_rewards, ep_scaled_rewards, self.DEVICE, self.actor, self.critic)

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
        
        log_probs = ep_log_probs[0]
        values = ep_values[0]
        rewards = ep_rewards[0]
        logits = ep_logits[0]

        total_episode_reward = np.sum(rewards)
        total_episode_scaled_reward = np.sum(ep_scaled_rewards[0])

        discounted_rewards = self.get_discounted_rewards(rewards, self.GAMMA)
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32, device=self.DEVICE)
        
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

        return log_probs, values, discounted_rewards, logits, total_episode_reward, total_episode_scaled_reward, episode

    def calculate_loss(self, log_probs, values, discounted_rewards, logits):
       
        advantages = discounted_rewards - values
        
        actor_loss = -(log_probs * advantages.detach()).mean()
        
        critic_loss = F.mse_loss(values, discounted_rewards)
        
        p = softmax(logits, dim=1)
        logp = log_softmax(logits, dim=1)
        ent = -torch.mean(torch.sum(p * logp, dim=1), dim=0)
        entropy_bonus = -self.BETA * ent
        
        total_loss = actor_loss + 0.5 * critic_loss + entropy_bonus
        
        return total_loss, ent

    @staticmethod
    def get_discounted_rewards(rews, gamma):
        out = np.zeros_like(rews, dtype=float)
        running_add = 0.0
        for i in reversed(range(len(rews))):
            running_add = running_add * gamma + rews[i]
            out[i] = running_add
        return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cuda', action='store_true')
    parser.add_argument('-config', required=True)
    parser.add_argument('--model4', required=True)
    parser.add_argument('--model8', required=True)
    parser.add_argument('--model16', required=True)
    parser.add_argument('--model32', required=True)
    parser.add_argument('-train_cfg', required=True)
    parser.add_argument('-save', type=int, default=-1)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        sim_json = json.load(f)

    with open(args.train_cfg, 'r') as f:
        c = json.load(f)
    train_cfg = Params(**c)

    AC_Trainer = ActorCriticTrainer(
        train_cfg=train_cfg,
        use_cuda=args.use_cuda,
        model4_path=args.model4,
        model8_path=args.model8,
        model16_path=args.model16,
        model32_path=args.model32,
        base_lr=1e-5,
        sim_config=sim_json
    )
    AC_Trainer.solve_environment(0, args.save)

if __name__ == "__main__":
    main()