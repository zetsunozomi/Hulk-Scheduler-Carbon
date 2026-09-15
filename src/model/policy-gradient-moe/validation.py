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
from moe.model import StochasticMixtureOfExperts
from moe.predictor import raw_features_to_np_ndarray, Predictor
from carbon_calculator import get_carbon_emission

from colorama import Fore, Style, init
init(autoreset=True)
from policy_gradient import PolicyGradient as TrainingPolicyGradient

class ValidationPolicyGradient(TrainingPolicyGradient):
    def __init__(self,
                 epoch:int,
                 use_cuda: bool,
                 model4_path, model8_path, model16_path, model32_path,
                 validation_checkpoint_path,
                 sim_config: dict,
                 model_output: str = "gpt345M",
                 app_type: str = "gpt-345m",
                 carbon_weight: float = 0.0,
                 cluster_name: str = "lonestar6-a100",
                 policy_model_type: str = "flat"
                 ):
        
        super().__init__(
            ckpt_folder=".", # Dummy
            batch_size=1,    # Dummy
            epoch=epoch,
            use_cuda=use_cuda,
            model4_path=model4_path,
            model8_path=model8_path,
            model16_path=model16_path,
            model32_path=model32_path,
            base_lr=0.0,     # Dummy
            sim_config=sim_config,
            model_output=model_output,
            carbon_weight=carbon_weight,
            app_type=app_type,
            base_run_h=6,     # Default
            cluster_name=cluster_name,
            policy_model_type=policy_model_type
        )

        self.checkpoint_path = validation_checkpoint_path
        if self.checkpoint_path:
            print(f"Loading the validation checkpoint: {self.checkpoint_path}")
            checkpoint = torch.load(self.checkpoint_path, map_location=self.DEVICE)
            # Try to load state dict if present, else assume checkpoint is state dict or model
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.agent.load_state_dict(checkpoint['model_state_dict'])
            elif isinstance(checkpoint, dict):
                 # Fallback, maybe it is the state dict
                 try:
                     self.agent.load_state_dict(checkpoint)
                 except:
                     print("Could not load state dict from checkpoint dict keys")
            else:
                 pass # Unknown format
            self.agent.eval()
        
    def validate_in_environment(self):
        with torch.no_grad():
            while self.episode < self.NUM_EPOCHS:
                print("-------------------------------------")
                print(f"Starting episode {self.episode}")
                # New return signature: policy_loss, entropy_mean, critic_loss
                policy_loss, entropy_mean, critic_loss = self.play_one_episode()
                print(f"[Result of this episode] Policy Loss: {policy_loss.item():.4f}, Critic Loss: {critic_loss.item():.4f}, Entropy: {entropy_mean.item():.4f}")
                print(f"Ending episode {self.episode}")
                print("-------------------------------------")
                # empty
                self.episode+=1
            print("=== Validation end ===")        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cuda', action='store_true')
    parser.add_argument('-config', required=True)
    parser.add_argument('--model4', required=True)
    parser.add_argument('--model8', required=True)
    parser.add_argument('--model16', required=True)
    parser.add_argument('--model32', required=True)
    parser.add_argument('-epoch', required=True)
    parser.add_argument('--carbon_weight', type=float, default=0.0)
    parser.add_argument('--validation_checkpoint_path', default=None)
    parser.add_argument('--ckpt_folder', default=None)
    parser.add_argument('--model_output', default="gpt345M")
    parser.add_argument('--app_type', type=str, default="gpt-345m", choices=["gpt-345m", "gpt-1.5b", "gpt-2.0b"])
    parser.add_argument('--cluster_name', type=str, default="lonestar6-a100")
    parser.add_argument('--policy_model_type', type=str, default="flat", choices=["flat", "hierarchical", "separate"])
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        sim_json = json.load(f)

    PG = ValidationPolicyGradient(
        epoch = int(args.epoch),
        use_cuda=args.use_cuda,
        model4_path=args.model4,
        model8_path=args.model8,
        model16_path=args.model16,
        model32_path=args.model32,
        validation_checkpoint_path = args.validation_checkpoint_path,
        sim_config=sim_json,
        carbon_weight = args.carbon_weight,
        app_type = args.app_type,
        model_output = args.model_output,
        cluster_name = args.cluster_name,
        policy_model_type = args.policy_model_type
    )
    PG.validate_in_environment()


if __name__ == "__main__":
    main()      
