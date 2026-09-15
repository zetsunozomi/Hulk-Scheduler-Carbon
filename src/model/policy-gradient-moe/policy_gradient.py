import json
import os
import sys
import gc
import argparse
import numpy as np

sys.path.append("../")
sys.path.append("../../")

import torch
import torch.optim as optim
import torch.nn as nn
from torch.distributions import Categorical
from torch.nn.functional import one_hot, log_softmax, softmax
from torch.utils.tensorboard import SummaryWriter
from collections import deque

from env import Env
from sim.application import Application
from moe.model import StochasticMixtureOfExperts, HierarchicalStochasticMixtureOfExperts
from moe.predictor import raw_features_to_np_ndarray, Predictor
from carbon_calculator import get_carbon_emission
from carbon_intensity_lookup import get_unit_carbon_intensity
from datetime import timedelta

from colorama import Fore, Style, init
init(autoreset=True)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # random.seed(seed) # If you use python random module
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class naive_agent(nn.Module):
    def __init__(self):
        super().__init__()
        self.gating_module = nn.Linear(4,4)
        nn.init.xavier_uniform_(self.gating_module.weight)
        self.gating_module.bias.data.fill_(1e-5)
    def forward(self,x):
        gating_logits = self.gating_module(x)
        return gating_logits

class TwoStageMoE(nn.Module):
    def __init__(self, num_experts, expert_models, num_actions=None, extra_input_dim=0):
        super().__init__()
        self.experts = nn.ModuleList(expert_models)
        # Freeze experts
        for i in range(len(self.experts)):
            for p in self.experts[i].parameters():
                p.requires_grad = False
            self.experts[i].eval()
            
        input_dim = num_experts + extra_input_dim
        self.num_actions = num_actions if num_actions is not None else num_experts
        
        # 1. Wait/Submit Model - Decides WHEN to act
        # Modified: Wait Net ONLY sees Time/Carbon (extra_input) to prevent infinite waiting due to queue status
        # Dropout
        self.wait_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2) # [Submit, Wait]
        )
        
        # 2. Node Selection Model - Decides HOW to act
        self.node_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_actions) # [4, 8, 16, 32]
        )
        
        self.expert_norm = nn.LayerNorm(num_experts)
        # self.extra_norm = nn.LayerNorm(extra_input_dim) # Removed LayerNorm as requested
        
        # Adaptive Feature Selection (Gating)
        self.wait_gate = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()
        )
        self.node_gate = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()
        )
        
        # Init
        for net in [self.wait_net, self.node_net]:
            for m in net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.constant_(m.bias, 0.01)
            
            nn.init.uniform_(net[-1].weight, -1e-3, 1e-3)
            nn.init.constant_(net[-1].bias, 0)

        for gate in [self.wait_gate, self.node_gate]:
            nn.init.xavier_uniform_(gate[0].weight)
            nn.init.constant_(gate[0].bias, 0)
            
        # 3. Critic Model - Estimates Value of state V(s)
        self.critic_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Init critic
        for m in self.critic_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0.01)


        # === 2. Manual Gating Bias Initialization (Masking Strategy) ===
        # Purpose: Balance Expert Preds (Queue) vs Carbon/Time inputs
        # Structure: First 4 dims are Expert Preds, Last 6 dims are Extra Inputs
        with torch.no_grad():
            # Expert Preds (Indices 0-3): Moderate suppression for queue awareness
            # -1.5 gives sigmoid ≈ 0.18 (not completely off, but reduced)
            self.wait_gate[0].bias[:num_experts].fill_(-1.5) 
            
            # Extra Inputs (Indices 4-9): Favor carbon/time features
            # +1.5 gives sigmoid ≈ 0.82 (strongly on)
            self.wait_gate[0].bias[num_experts:].fill_(1.5)
            
            print("Gating Bias Adjusted: Queue info available, Carbon/Time strongly favored.")

    def forward(self, x, extra_input=None):
        # Get expert preds
        outs = []
        for expert in self.experts:
            with torch.no_grad():
                outs.append(expert(x))
        expert_preds = torch.cat(outs, dim=1) # [B, 4]
        
        # Separate Norms
        expert_preds = self.expert_norm(expert_preds)
        
        if extra_input is not None:
             if extra_input.dim() == 1: extra_input = extra_input.unsqueeze(1)
             # extra_input = self.extra_norm(extra_input) # Removed LayerNorm as requested
             full_input = torch.cat([expert_preds, extra_input], dim=1)
        else:
             full_input = expert_preds
        
        print(f"expert_preds: {expert_preds}")
        print(f"extra_input(sin, cos, current carbon, future carbon): {extra_input}")
        print(f"full input after separate norm: {full_input}")

        # Branch 1: Submit vs Wait
        # Apply Gating for adaptive feature selection
        wait_input = full_input * self.wait_gate(full_input)
        wait_logits = self.wait_net(wait_input)
        
        wait_probs = softmax(wait_logits, dim=1)
        prob_submit = wait_probs[:, 0:1] # Probability of Submitting Now
        prob_wait = wait_probs[:, 1:2]   # Probability of Waiting
        
        # Branch 2: Which Node Config?
        # Node selection still uses the full fused/normalized context with its own gate
        node_input = full_input * self.node_gate(full_input)
        node_logits = self.node_net(node_input)
        node_probs = softmax(node_logits, dim=1)
        
        # Combine:
        # P(Action i) = P(Submit) * P(Node i | Submit) for i in {0..3}
        # P(Wait) = P(Wait)
        final_probs_submit = prob_submit * node_probs
        final_probs = torch.cat([final_probs_submit, prob_wait], dim=1)
        
        # Critic Value
        value = self.critic_net(full_input)
        
        return torch.log(final_probs + 1e-10), value

class PolicyGradient:
    def __init__(self,
                 ckpt_folder: str,
                 batch_size:int,
                 epoch:int,
                 use_cuda: bool,
                 model4_path, model8_path, model16_path, model32_path,
                 base_lr: float,
                 sim_config: dict,
                 model_output: str = "gpt_transformer_policy_mlp",
                 carbon_weight: float=0,
                 app_type: str = "gpt-345m",
                 base_run_h: int = 6,
                 cluster_name: str = "perlmutter",
                 policy_model_type: str = "flat",
                 load_checkpoint: str = None
                 ):
        self.cluster_name = cluster_name
        self.NUM_EPOCHS = epoch
        self.DEVICE = torch.device('cuda' if (torch.cuda.is_available() and use_cuda) else 'cpu')
        self.model_output = model_output
        self.base_lr = base_lr
        self.env = Env(**sim_config)
        self.app = Application(app_type=app_type,base_run_h=base_run_h)
        self.baseline = 0
        self.alpha = 0.1
        self.episode = 0
        self.carbon_weight = carbon_weight
        self.all_nhc = []
        self.all_tat = []
        self.all_carbon = []
        self.ckpt_folder = ckpt_folder
        self.policy_model_type = policy_model_type
        self.batch_size = batch_size
        self.load_checkpoint = load_checkpoint
        
        def load_pred(ckpt_path):
            c= torch.load(ckpt_path, map_location=self.DEVICE)
            print(f"Loading model: {c['model_hparams']}")
            predictor =  Predictor(
                model_name= c['model_name'],
                hparams= c['model_hparams'],
                optimizer_name= c['optimizer_name'],
                base_learning_rate= 1e-3,
                model_state= c['model_state_dict'],
                optimizer_state= c['optimizer_state_dict'],
                device=("cuda" if self.DEVICE.type=='cuda' else 'cpu'),
                seed=0,
                debug=False
            )
            predictor.model.eval()
            for param in predictor.model.parameters():
                param.requires_grad = False
            return predictor
        p4  = load_pred(model4_path)
        p8  = load_pred(model8_path)
        p16 = load_pred(model16_path)
        p32 = load_pred(model32_path)
        
        expert_models_list = [
            p4.model,
            p8.model,
            p16.model,
            p32.model
        ]
        
        # Select Model Class
        # Enhanced Features for Wait Net:
        # - Sin/Cos ToD: Cyclic time encoding
        # - Current Carbon: Immediate carbon intensity
        # - Future Carbon: 6-hour forecast
        # - Carbon Delta: future - current (positive = getting worse, should wait)
        # - Carbon Trend: normalized delta to [-1, 1]
        extra_dim = 6  # sin, cos, current_carbon, future_carbon, carbon_delta, carbon_trend
        
        if self.policy_model_type == "hierarchical":
            print("Using Hierarchical MOE Policy")
            self.agent = HierarchicalStochasticMixtureOfExperts(
                num_experts=4,
                expert_models=expert_models_list,
                num_actions=5,
                extra_input_dim=extra_dim
            ).to(self.DEVICE)
            self.normalize_tod = True
        elif self.policy_model_type == "separate":
            print("Using Separate (Two-Stage) MOE Policy")
            self.agent = TwoStageMoE(
                num_experts=4,
                expert_models=expert_models_list,
                num_actions=4, # Internal node actions
                extra_input_dim=extra_dim
            ).to(self.DEVICE)
            self.normalize_tod = True
        else:
            print("Using Standard (Flat) MOE Policy")
            self.agent = StochasticMixtureOfExperts(
                num_experts=4,
                expert_models=expert_models_list,
                num_actions=5,
                extra_input_dim=extra_dim
            ).to(self.DEVICE)
            self.normalize_tod = False

        #self.agent = naive_agent()
        
        # === Separate Optimizers for Policy and Critic ===
        # Critic needs higher learning rate (3x) to learn value function faster
        # This prevents critic loss from staying high and destabilizing policy
        policy_params = []
        critic_params = []
        
        for name, param in self.agent.named_parameters():
            if 'critic' in name:
                critic_params.append(param)
            else:
                policy_params.append(param)
        
        self.policy_optimizer = optim.Adam(policy_params, lr=self.base_lr)
        self.critic_optimizer = optim.Adam(critic_params, lr=self.base_lr * 3.0)  # 3x learning rate
        
        # GAE Hyperparameters
        self.gamma = 0.95  # Discount factor (reduced from 0.99 for shorter horizon)
        self.gae_lambda = 0.95  # GAE lambda for bias-variance tradeoff
        
        if self.load_checkpoint and os.path.exists(self.load_checkpoint):
            print(f"Loading checkpoint from {self.load_checkpoint}")
            checkpoint = torch.load(self.load_checkpoint, map_location=self.DEVICE)
            self.agent.load_state_dict(checkpoint['model_state_dict'])
            if 'policy_optimizer_state_dict' in checkpoint:
                self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
            if 'critic_optimizer_state_dict' in checkpoint:
                self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            if 'epoch' in checkpoint:
                self.episode = checkpoint['epoch'] + 1
            print(f"Resumed training from episode {self.episode}")


    def solve_environment(self):
        # entropy_coef = 0.2 # Removed fixed value
        
        start_entropy = 0.5
        end_entropy = 0.1  # Increased from 0.01 to maintain exploration longer
        decay_episodes = self.NUM_EPOCHS # Decay over the full course of training
        
        # Zero gradients for both optimizers
        self.policy_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        
        while self.episode < self.NUM_EPOCHS:
            # Calculate dynamic entropy coefficient
            # Linear decay
            if self.episode < decay_episodes:
                progress = self.episode / decay_episodes
                entropy_coef = start_entropy - (start_entropy - end_entropy) * progress
            else:
                entropy_coef = end_entropy
                
            checkpoint_path = f"{self.ckpt_folder}/{self.model_output}_carbonweight{self.carbon_weight}_{self.episode}.pt"
            print("-------------------------------------")
            print(f"Starting episode {self.episode}, Entropy Coef: {entropy_coef:.4f}")
            
            # play_one_episode returns Actor Loss, Entropy, and Critic Loss
            policy_loss_val, entropy_mean, critic_loss_val = self.play_one_episode()
            
            # Entropy Loss: We want to MAXIMIZE entropy, so minimize (-entropy)
            entropy_loss = - entropy_coef * entropy_mean
            
            # === Separate Training for Critic and Policy ===
            
            # 1. Train Critic (multiple updates for better value learning)
            critic_total_loss = (0.5 * critic_loss_val) / self.batch_size
            critic_total_loss.backward(retain_graph=True)
            
            # 2. Train Policy (single update)
            policy_total_loss = (policy_loss_val + entropy_loss) / self.batch_size
            policy_total_loss.backward()

            if (self.episode + 1) % self.batch_size == 0:
                print(f"Update gradients at episode {self.episode}")
                
                # Update Critic (with gradient clipping)
                torch.nn.utils.clip_grad_norm_(self.agent.critic_net.parameters(), max_norm=0.5)
                self.critic_optimizer.step()
                self.critic_optimizer.zero_grad()
                
                # Update Policy (with gradient clipping)
                torch.nn.utils.clip_grad_norm_(
                    [p for name, p in self.agent.named_parameters() if 'critic' not in name], 
                    max_norm=0.5
                )
                self.policy_optimizer.step()
                self.policy_optimizer.zero_grad()

            print(f"[Result of this episode] Policy Loss: {policy_loss_val.item()}, Critic Loss: {critic_loss_val.item()}, Entropy: {entropy_mean.item()}")
            print(f"Ending episode {self.episode}")
            print("-------------------------------------")
            # empty
            self.episode+=1
            if self.episode % 100 == 0 and self.episode !=0:
                checkpoint_path = f"{self.ckpt_folder}/{self.model_output}_carbonweight{self.carbon_weight}_{self.episode}.pt"
                torch.save({
                    'model_state_dict': self.agent.state_dict(),
                    'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
                    'epoch': self.episode,
                }, checkpoint_path)
        print("=== Training end ===")
        checkpoint_path = f"{self.ckpt_folder}/{self.model_output}_carbonweight{self.carbon_weight}_final.pt"
        torch.save({
            'model_state_dict': self.agent.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'epoch': self.episode,
        }, checkpoint_path)
        print(f"Final model saved to {checkpoint_path}")
        
    def play_one_episode(self):
        self.env.reset()
        self.app.reset()

        ep_logp = []
        ep_entropy = []
        ep_values = [] # Collect State Values from Critic
        ep_tat = []
        ep_nhc = []
        ep_carbon = []
        done = False
        step_idx = 0

        while not done:
            #get state (history snapshot + job_info)
            data_input = self.env.get_current_data_input()
            
            # Calculate Future Carbon Trend (Lookahead)
            curr_time = self.env.get_current_time()
            future_carbon_sum = 0
            for h in range(1, 7): # Next 6 hours
                ft = curr_time + timedelta(hours=h)
                future_carbon_sum += get_unit_carbon_intensity(ft, ft.hour)
            avg_future_carbon = future_carbon_sum / 6.0
            data_input['future_carbon_intensity'] = avg_future_carbon

            action_idx, log_prob, entropy, val = self.sample_action(data_input)
            if action_idx == 4: # doesn't choose node, just skip 1 hour.
                self.env.step_time(timedelta(hours=1))
                ep_tat.append(1)
                ep_nhc.append(0)
                ep_carbon.append(0)
                ep_logp.append(log_prob)
                ep_entropy.append(entropy)
                ep_values.append(val)
                step_idx += 1
            elif action_idx <4:
                node_map = [4, 8, 16, 32]
                chosen_node = node_map[action_idx]

                # generate and submit a new job => run => reward => check episode end
                next_job_id = 100000 + step_idx
                new_job = self.app.create_new_job(action_idx, self.env, next_job_id)
                
                negative_turnaround_time, exec_time, done = self.app.run_job_and_measure_reward(self.env, new_job)
                carbon_emission = get_carbon_emission(exec_time, new_job.log.start, chosen_node, self.cluster_name)
                print(Fore.RED + f"carbon emission: {carbon_emission}")
                # now reward is positive!
                positive_turnaround_time = -negative_turnaround_time
                nh_consumption = chosen_node * exec_time
                #reward_val = (1-self.nh_consumption_weight)*positive_turnaround_time + self.nh_consumption_weight*nh_consumption
                ep_tat.append(positive_turnaround_time)
                ep_nhc.append(nh_consumption)
                ep_carbon.append(carbon_emission)
                ep_logp.append(log_prob)
                ep_entropy.append(entropy)
                ep_values.append(val)
                print(f"[result of step={step_idx}], ActionIdx={action_idx}, tat={positive_turnaround_time}, nhc = {nh_consumption}, Done={done}")
                step_idx += 1
                print(f"*******End of this job*******")
        # complete one episode, total return
        print(f"list of ep_tat {ep_tat}")
        print(f"list of ep_nhc {ep_nhc}")
        print(f"list of ep_carbon {ep_carbon}")
        sum_of_tat = np.sum(ep_tat)
        sum_of_nhc = np.sum(ep_nhc)
        sum_of_carbon = np.sum(ep_carbon)
        print(f"tat of this episode: {sum_of_tat}")
        print(f"nhc of this episode: {sum_of_nhc}")
        print(f"carbon of this episode: {sum_of_carbon}")
        self.all_nhc.append(sum_of_nhc)
        self.all_tat.append(sum_of_tat)
        self.all_carbon.append(sum_of_carbon)

        # Fixed SCALING FACTORS
        fixed_mean_tat = 82.0
        fixed_mean_carbon = 130000.0  # Updated to match actual data (was 90000.0)
        

        # --- START: GAE (Generalized Advantage Estimation) ---
        
        # 1. Reconstruct per-step costs from the recorded episodes
        step_costs = []
        for t, c in zip(ep_tat, ep_carbon):
            # Calculate weighted cost for this step
            sc_tat = t / fixed_mean_tat
            sc_carbon = c / fixed_mean_carbon
            
            step_cost = (1 - self.carbon_weight) * sc_tat + self.carbon_weight * sc_carbon
            step_costs.append(step_cost)
        print(f"Step costs: {step_costs}")
        
        # 2. Convert to tensors
        values = torch.cat(ep_values).view(-1).detach()  # [T]
        rewards = torch.tensor(step_costs, dtype=torch.float32).to(self.DEVICE)  # [T]
        
        # 3. Compute GAE (Generalized Advantage Estimation)
        # GAE formula: A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
        # where δ_t = r_t + γV(s_{t+1}) - V(s_t)
        
        advantages = []
        gae = 0
        
        # Work backwards from the last timestep
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                # Terminal state: no next value
                next_value = 0
            else:
                next_value = values[t + 1]
            
            # TD error: δ_t = r_t + γV(s_{t+1}) - V(s_t)
            delta = rewards[t] + self.gamma * next_value - values[t]
            
            # GAE: A_t = δ_t + (γλ)A_{t+1}
            gae = delta + self.gamma * self.gae_lambda * gae
            
            advantages.insert(0, gae.item())
        
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.DEVICE)
        
        # 4. Compute Returns for Critic Training
        # Returns = Advantages + Values
        returns = advantages + values
        
        # 5. Normalize Advantages (CRITICAL for stable training)
        if len(advantages) > 1:
            adv_mean = advantages.mean()
            adv_std = advantages.std()
            advantages_normalized = (advantages - adv_mean) / (adv_std + 1e-8)
        else:
            advantages_normalized = advantages
        
        # 6. Critic Loss: Huber Loss (more robust than MSE)
        values_for_loss = torch.cat(ep_values).view(-1)  # Keep gradient
        returns_target = returns.detach()  # No gradient through returns
        
        critic_loss = torch.nn.functional.smooth_l1_loss(values_for_loss, returns_target)
        
        # 7. Policy Loss with Normalized Advantages
        log_probs = torch.stack(ep_logp).view(-1)
        policy_loss = (log_probs * advantages_normalized.detach()).sum()
        
        entropy_mean = torch.stack(ep_entropy).mean()
        
        # 8. Logging Stats
        sum_of_rewards = rewards.sum().item()
        
        if self.episode == 1:
            self.baseline = sum_of_rewards
        else:
            self.baseline = self.baseline * (1-self.alpha) + sum_of_rewards * self.alpha
        
        print(f"reward (cost) of this episode: {sum_of_rewards}")
        print(f"self.baseline: {self.baseline}")
        
        # === Enhanced Logging for Debugging ===
        print(f"[GAE Stats]")
        print(f"  Value Mean: {values.mean().item():.4f}, Std: {values.std().item():.4f}")
        print(f"  Return Mean: {returns.mean().item():.4f}, Std: {returns.std().item():.4f}")
        print(f"  Advantage Mean: {advantages.mean().item():.4f}, Std: {advantages.std().item():.4f}")
        print(f"  Normalized Adv Mean: {advantages_normalized.mean().item():.4f}, Std: {advantages_normalized.std().item():.4f}")
        
        # Return calculated losses directly
        return policy_loss, entropy_mean, critic_loss
            
    def sample_action(self, data_input):
        # convert tensor
        np_input = raw_features_to_np_ndarray([data_input], parallel=False)
        print(f"[Sample action] Processed features shape: {np_input.shape}")
        tensor_in = torch.tensor(np_input, dtype=torch.float32).to(self.DEVICE)
        
        # Get time of day
        tod = data_input.get('time_of_day', 0.0)
        
        # Use Sin/Cos Cyclic Encoding
        # Normalize 0-24h to 0-2pi
        norm_tod = (tod / 24.0) * 2 * np.pi
        sin_tod = np.sin(norm_tod)
        cos_tod = np.cos(norm_tod)
        
        # Get Carbon Intensity (improved normalization)
        # Typical range: 200-600 g/kWh, normalize to [0, 1]
        carbon_val = data_input.get('carbon_intensity', 400.0)
        norm_carbon = (carbon_val - 200.0) / 400.0  # Maps [200, 600] -> [0, 1]
        norm_carbon = np.clip(norm_carbon, 0.0, 1.0)  # Clamp to [0, 1]
        
        # Get Future Carbon Forecast
        future_carbon_val = data_input.get('future_carbon_intensity', 400.0)
        norm_future_carbon = (future_carbon_val - 200.0) / 400.0
        norm_future_carbon = np.clip(norm_future_carbon, 0.0, 1.0)
        
        # === NEW: Carbon Delta Features ===
        # Positive delta = carbon getting worse = should wait
        # Negative delta = carbon getting better = should submit now
        carbon_delta = future_carbon_val - carbon_val
        
        # Normalize delta to [-1, 1] range
        # Typical delta range: -200 to +200
        norm_carbon_delta = carbon_delta / 200.0
        norm_carbon_delta = np.clip(norm_carbon_delta, -1.0, 1.0)
        
        # Carbon trend indicator: positive if getting worse
        carbon_trend = np.tanh(carbon_delta / 100.0)  # Smooth sigmoid-like mapping
        
        # Enhanced feature vector: [sin, cos, current, future, delta, trend]
        extra_tensor = torch.tensor(
            [[sin_tod, cos_tod, norm_carbon, norm_future_carbon, norm_carbon_delta, carbon_trend]], 
            dtype=torch.float32
        ).to(self.DEVICE)
        
        # gating_logits
        out = self.agent.forward(tensor_in, extra_input=extra_tensor)
        
        # Handle tuple return (Logits, Value) for Actor-Critic
        val = torch.zeros(1, device=self.DEVICE)
        if isinstance(out, tuple):
            logits, val = out
        else:
            logits = out
            
        # softmax
        dist = Categorical(logits=logits)
        print(f"[Sample action] probablity distribution of this job {dist.probs}")
        # sample from softmax
        action_idx = dist.sample()   
        log_prob = dist.log_prob(action_idx)
        entropy = dist.entropy()
        if action_idx.item() == 4:
            print(Fore.YELLOW + f"ActionIdx={action_idx.item()}, Skip 1 hour at ToD {tod}, Carbon: {carbon_val:.1f} -> {future_carbon_val:.1f} (Δ={carbon_delta:.1f})")
        
        return action_idx.item(), log_prob, entropy, val

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cuda', action='store_true')
    parser.add_argument('-config', required=True)
    parser.add_argument('--model4', required=True)
    parser.add_argument('--model8', required=True)
    parser.add_argument('--model16', required=True)
    parser.add_argument('--model32', required=True)
    parser.add_argument('-base_lr', required=True)
    parser.add_argument('-batch_size', required=True)
    parser.add_argument('-epoch', required=True)
    parser.add_argument('--carbon_weight', type=float, default=0)
    parser.add_argument('--app_type', type=str, default="gpt-345m", choices=["gpt-345m", "gpt-1.5b", "gpt-2.0b"])
    parser.add_argument('--ckpt_folder', type=str, default="")
    parser.add_argument('--model_output', type=str, default="model_default")
    parser.add_argument('--base_run_h', type=int, default=6)
    parser.add_argument('--cluster_name', type=str, default="nersc")
    parser.add_argument('--policy_model_type', type=str, default="flat", choices=["flat", "hierarchical", "separate"])
    parser.add_argument('--load_checkpoint', type=str, default=None, help="Path to a checkpoint file to resume training")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        sim_json = json.load(f)

    set_seed(114514) # Set seed for reproducibility


    PG = PolicyGradient(
        batch_size = int(args.batch_size),
        epoch = int(args.epoch),
        use_cuda=args.use_cuda,
        model4_path=args.model4,
        model8_path=args.model8,
        model16_path=args.model16,
        model32_path=args.model32,
        base_lr=float(args.base_lr),
        sim_config=sim_json,
        carbon_weight = args.carbon_weight,
        app_type = args.app_type,
        ckpt_folder = args.ckpt_folder,
        model_output = args.model_output,
        base_run_h = args.base_run_h,
        cluster_name = args.cluster_name,
        policy_model_type = args.policy_model_type,
        load_checkpoint = args.load_checkpoint
    )
    PG.solve_environment()


if __name__ == "__main__":
    main()      
