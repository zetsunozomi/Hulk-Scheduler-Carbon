import json
import os
import sys
import argparse
import numpy as np

sys.path.append("../")
sys.path.append("../../")

import torch
import torch.nn as nn
from collections import deque
from env import Env
from sim.application import Application
from moe.model import StochasticMixtureOfExperts
from moe.predictor import raw_features_to_np_ndarray, Predictor

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

class DQNValidator:
    def __init__(self,
                 epoch: int,
                 use_cuda: bool,
                 model4_path, model8_path, model16_path, model32_path,
                 validation_checkpoint_path,
                 sim_config: dict,
                 model_output: str = "transformer_dqn_mlp",
                 nhc_weight: float = 0):

        self.NUM_EPOCHS = epoch
        self.DEVICE = torch.device('cuda' if (torch.cuda.is_available() and use_cuda) else 'cpu')
        self.model_output = model_output
        self.env = Env(**sim_config)
        self.app = Application()

        self.episode = 0
        self.nhc_weight = nhc_weight
        self.all_nhc = []
        self.all_tat = []

        def load_pred(ckpt_path):
            c = torch.load(ckpt_path, map_location=self.DEVICE)
            print(f"Loading model: {c['model_hparams']}")
            predictor = Predictor(
                model_name=c['model_name'],
                hparams=c['model_hparams'],
                optimizer_name=c['optimizer_name'],
                base_learning_rate=1e-3,
                model_state=c['model_state_dict'],
                optimizer_state=c['optimizer_state_dict'],
                device=("cuda" if self.DEVICE.type == 'cuda' else 'cpu'),
                seed=0,
                debug=False
            )
            predictor.model.eval()
            return predictor

        p4 = load_pred(model4_path)
        p8 = load_pred(model8_path)
        p16 = load_pred(model16_path)
        p32 = load_pred(model32_path)

        print(f"Loading the DQN checkpoint: {validation_checkpoint_path}")
        transformer_moe = StochasticMixtureOfExperts(
            num_experts=4,
            expert_models=[p4.model, p8.model, p16.model, p32.model]
        ).to(self.DEVICE)

        self.agent = NormalizedQTransformer(transformer_moe, action_dim=4).to(self.DEVICE)
        checkpoint = torch.load(validation_checkpoint_path, map_location=self.DEVICE)
        self.agent.load_state_dict(checkpoint['model_state_dict'])
        self.agent.eval()

    def validate_in_environment(self):
        while self.episode < self.NUM_EPOCHS:
            print("-------------------------------------")
            print(f"Starting episode {self.episode}")
            sum_rewards = self.play_one_episode()
            print(f"[Result of this episode] total reward: {sum_rewards}")
            print(f"Ending episode {self.episode}")
            print("-------------------------------------")
            self.episode += 1
        print("=== DQN Validation end ===")

    def play_one_episode(self):
        self.env.reset()
        self.app.reset()

        ep_tat = []
        ep_nhc = []
        done = False
        step_idx = 0

        while not done:
            data_input = self.env.get_current_data_input()
            action_idx, q_values = self.sample_action(data_input)
            node_map = [4, 8, 16, 32]
            chosen_node = node_map[action_idx]

            next_job_id = 100000 + step_idx
            new_job = self.app.create_new_job(action_idx, self.env, next_job_id)
            negative_turnaround_time, exec_time, done = self.app.run_job_and_measure_reward(self.env, new_job)

            positive_turnaround_time = -negative_turnaround_time
            nh_consumption = chosen_node * exec_time

            ep_tat.append(positive_turnaround_time)
            ep_nhc.append(nh_consumption)

            print(f"[Step={step_idx}] Action={action_idx}, Q={q_values}, TAT={positive_turnaround_time:.3f}, NHC={nh_consumption:.3f}, Done={done}")
            step_idx += 1

        sum_of_tat = np.sum(ep_tat)
        sum_of_nhc = np.sum(ep_nhc)
        print(f"Episode summary → Total TAT={sum_of_tat:.3f}, Total NHC={sum_of_nhc:.3f}")

        self.all_tat.append(sum_of_tat)
        self.all_nhc.append(sum_of_nhc)

        mean_tat = np.mean(self.all_tat)
        mean_nhc = np.mean(self.all_nhc)
        ratio = mean_nhc / mean_tat if mean_tat > 0 else 1
        linear_scale_sum_nhc = sum_of_nhc / ratio
        linear_scale_sum_tat = sum_of_tat

        sum_of_rewards = linear_scale_sum_tat * (1 - self.nhc_weight) + linear_scale_sum_nhc * self.nhc_weight
        print(f"[Episode {self.episode}] Linear scaled reward={sum_of_rewards:.3f} (ratio={ratio:.4f})")

        return sum_of_rewards

    def sample_action(self, data_input):
        np_input = raw_features_to_np_ndarray([data_input], parallel=False)
        tensor_in = torch.from_numpy(np_input).float().to(self.DEVICE)
        q_values = self.agent(tensor_in)
        if q_values.dim() == 1:
            q_values = q_values.unsqueeze(0)
        action_idx = torch.argmax(q_values, dim=1).item()
        print(f"[Q-values] {q_values.detach().cpu().numpy().round(3)} → Action={action_idx}")
        return action_idx, q_values.detach().cpu().numpy().round(3)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cuda', action='store_true')
    parser.add_argument('-config', required=True)
    parser.add_argument('--model4', required=True)
    parser.add_argument('--model8', required=True)
    parser.add_argument('--model16', required=True)
    parser.add_argument('--model32', required=True)
    parser.add_argument('-epoch', required=True)
    parser.add_argument('--nhc_weight', type=float, default=0)
    parser.add_argument('--checkpoint_path', required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        sim_json = json.load(f)

    DQN = DQNValidator(
        epoch=int(args.epoch),
        use_cuda=args.use_cuda,
        model4_path=args.model4,
        model8_path=args.model8,
        model16_path=args.model16,
        model32_path=args.model32,
        validation_checkpoint_path=args.checkpoint_path,
        sim_config=sim_json,
        nhc_weight=args.nhc_weight
    )

    DQN.validate_in_environment()


if __name__ == "__main__":
    main()
