import json
import os
import sys
import gc
import argparse
import random
import numpy as np
from collections import deque

sys.path.append("../")
sys.path.append("../../")

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from env import Env
from sim.application import Application
from moe.model import StochasticMixtureOfExperts
from moe.predictor import raw_features_to_np_ndarray, Predictor

import logging
logging.basicConfig(level=logging.INFO)

class NormalizedQTransformer(nn.Module):
    def __init__(self, transformer_model, action_dim=4):
        super(NormalizedQTransformer, self).__init__()
        self.transformer_model = transformer_model
        self.norm = nn.LayerNorm(action_dim)
        self.fc = nn.Linear(action_dim, action_dim)  

    def forward(self, x):
        out = self.transformer_model(x)
        out = self.norm(out)
        q_values = self.fc(out)
        return q_values

class Params:
    GAMMA = 0.99
    ACTION_SPACE = 4

    def __init__(self, lr, epoch, batch_size):
        self.ALPHA = lr
        self.NUM_EPOCHS = epoch
        self.BATCH_SIZE = batch_size

class DQN:
    def __init__(self,
                 train_cfg: Params,
                 use_cuda: bool,
                 model4_path, model8_path, model16_path, model32_path,
                 base_lr: float,
                 hidden_dim: int,
                 sim_config: dict,
                 model_output: str = "gpt_2B/exp_tat_01/train_tat_1/transformer_dqn"):

        self.NUM_EPOCHS = train_cfg.NUM_EPOCHS
        self.ALPHA = train_cfg.ALPHA
        self.BATCH_SIZE = train_cfg.BATCH_SIZE
        self.GAMMA = Params.GAMMA
        self.ACTION_SPACE = Params.ACTION_SPACE

        self.DEVICE = torch.device('cuda' if (torch.cuda.is_available() and use_cuda) else 'cpu')
        self.writer = SummaryWriter(comment=f"_DQN_App_lr={self.ALPHA}_BS={self.BATCH_SIZE}")
        self.model_output = model_output

        self.env = Env(**sim_config)
        self.app = Application()

        self.replay_buffer = deque(maxlen=5000)
        self.batch_size = train_cfg.BATCH_SIZE

        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995

        def load_pred(ckpt_path):
            c = torch.load(ckpt_path, map_location=self.DEVICE)
            predictor = Predictor(
                model_name=c['model_name'],
                hparams=c['model_hparams'],
                optimizer_name=c['optimizer_name'],
                base_learning_rate=base_lr,
                model_state=c['model_state_dict'],
                optimizer_state=c['optimizer_state_dict'],
                device=("cuda" if self.DEVICE.type == 'cuda' else 'cpu'),
                seed=0,
                debug=False
            )
            predictor.model.eval()
            return predictor.model

        p4 = load_pred(model4_path)
        p8 = load_pred(model8_path)
        p16 = load_pred(model16_path)
        p32 = load_pred(model32_path)

        transformer_moe = StochasticMixtureOfExperts(
            num_experts=4,
            expert_models=[p4, p8, p16, p32]
        )

        self.q_net = NormalizedQTransformer(transformer_moe, action_dim=self.ACTION_SPACE).to(self.DEVICE)
        self.target_net = NormalizedQTransformer(transformer_moe, action_dim=self.ACTION_SPACE).to(self.DEVICE)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.ALPHA)
        self.target_update_freq = 100
        self.step_count = 0

    def select_action(self, state_tensor):
        if random.random() < self.epsilon:
            action = random.randint(0, self.ACTION_SPACE - 1)
        else:
            with torch.no_grad():
                q_values = self.q_net(state_tensor)
                action = torch.argmax(q_values, dim=1).item()
        return action

    def play_one_episode(self, episode):
        self.env.reset()
        self.app.reset()

        total_reward = 0
        done = False
        step_idx = 0

        np_state = raw_features_to_np_ndarray([self.env.get_current_data_input()], parallel=False)
        state = torch.from_numpy(np_state).float().to(self.DEVICE)

        while not done:
            action = self.select_action(state)
            sim_info = [self.env.sim._scheduler.job_logs, self.env.sim._scheduler.avail_nodes]
            next_job_id = 100000 + step_idx

            reward_val, scaled_reward_val, done = self.app.run_job_and_measure_reward(
                env=self.env,
                new_job_id=next_job_id,
                infer_func=None,
                sim_info=sim_info,
                data_input=self.env.get_current_data_input()
            )

            np_next = raw_features_to_np_ndarray([self.env.get_current_data_input()], parallel=False)
            next_state = torch.from_numpy(np_next).float().to(self.DEVICE)
            self.replay_buffer.append((state.cpu().numpy(), action, reward_val, next_state.cpu().numpy(), done))
            total_reward += reward_val
            state = next_state
            step_idx += 1

            self.optimize_model()

        logging.info(f"[Episode {episode}] Reward={total_reward:.3f}, ε={self.epsilon:.3f}")
        return total_reward

    def optimize_model(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.DEVICE)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.DEVICE)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.DEVICE)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.DEVICE)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.DEVICE)

        q_values = self.q_net(states).gather(1, actions)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (1 - dones) * self.GAMMA * next_q

        loss = F.mse_loss(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
            logging.info("[Target Network] Updated")

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def solve_environment(self, init_epoch=0, save_checkpoint=-1):
        epoch = init_epoch
        episode = 0
        total_rewards = []

        while epoch < self.NUM_EPOCHS:
            total_r = self.play_one_episode(episode)
            total_rewards.append(total_r)
            episode += 1

            if episode >= self.BATCH_SIZE:
                epoch += 1
                avg_reward = np.mean(total_rewards[-self.BATCH_SIZE:])
                logging.info(f"[Epoch {epoch}] Avg reward: {avg_reward:.3f}")
                self.writer.add_scalar("Reward/Avg", avg_reward, epoch)
                gc.collect()
                torch.cuda.empty_cache()

            if save_checkpoint > 0 and epoch % save_checkpoint == 0:
                ckpt_path = f"{self.model_output}_epoch{epoch}.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.q_net.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()
                }, ckpt_path)
                logging.info(f"[Checkpoint] Saved to {ckpt_path}")

        torch.save({
            'model_state_dict': self.q_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch,
        }, f"{self.model_output}_final.pt")
        logging.info(f"Final model saved to {self.model_output}_final.pt")

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
    parser.add_argument('-hidden_dim', type=int, default=32)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        sim_json = json.load(f)
    with open(args.train_cfg, 'r') as f:
        c = json.load(f)
    train_cfg = Params(**c)
    DQNTrainer = DQN(
        train_cfg=train_cfg,
        use_cuda=args.use_cuda,
        model4_path=args.model4,
        model8_path=args.model8,
        model16_path=args.model16,
        model32_path=args.model32,
        base_lr=1e-5,
        hidden_dim=args.hidden_dim,
        sim_config=sim_json
    )
    DQNTrainer.solve_environment(0, args.save)

if __name__ == "__main__":
    main()
