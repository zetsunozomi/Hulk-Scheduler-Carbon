import sys
sys.path.append("../")
sys.path.append("../../")
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
import argparse
import json
from env import Env
from sim.application import Application
from collections import deque
import numpy as np
from moe.predictor import raw_features_to_np_ndarray, Predictor

class greedy_mode_transformer:
    # output: 4 predicted queue wait time.
    def __init__(self, num_experts: int, expert_models):
        assert num_experts == len(expert_models)
        self.num_experts = num_experts
        self.experts = expert_models

    def forward(self, x):
        outs = []
        torch_tensor = torch.from_numpy(x.reshape(1, 288, 70))
        for expert in self.experts:
            with torch.no_grad():
                out_ex = expert.inference(torch_tensor)  # [batch,1]
            outs.append(out_ex)
        expert_preds = np.stack(outs, axis=1)  # => [batch,4]
        return expert_preds

class greedy_mode_rf_xgboost:
    # output: 4 predicted queue wait time.
    def __init__(self, num_experts: int, expert_models):
        assert num_experts == len(expert_models)
        self.num_experts = num_experts
        self.experts = expert_models

    def forward(self, x):
        outs = []
        for expert in self.experts:
            with torch.no_grad():
                out_ex = expert.predict(x)  # [batch,1]
            outs.append(out_ex)
        expert_preds = np.stack(outs, axis=1)  # => [batch,4]
        return expert_preds

# load in model
class RandomForest:
    def __init__(self,
                num_validation_epochs:int,
                use_cuda: bool,
                model4_path, model8_path, model16_path, model32_path,
                sim_config: dict,
                ):
        self.NUM_EPOCHS = num_validation_epochs
        self.DEVICE = torch.device('cuda' if (torch.cuda.is_available() and use_cuda) else 'cpu')
        self.env = Env(**sim_config)
        self.app = Application()
        #save 100 total return of 100 episode
        self.total_rewards = deque([], maxlen=100)
        def load_pred(ckpt_path):
            if "transformer" not in ckpt_path:
                with open(ckpt_path, "rb") as f: 
                    model = pickle.load(f)
                print(f"succuessfully loaded model from {ckpt_path}")
            else:
                print("transformer detected!")
                c= torch.load(ckpt_path, map_location=self.DEVICE)
                print(f"Loading model: {c['model_hparams']}")
                model =  Predictor(
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
                model.model.eval()
                print(f"succuessfully loaded model from {ckpt_path}")
            return model
        p4  = load_pred(model4_path)
        p8  = load_pred(model8_path)
        p16 = load_pred(model16_path)
        p32 = load_pred(model32_path)
        # as policy
        if "transformer" not in model4_path:
            self.agent = greedy_mode_rf_xgboost(
                num_experts=4,
                expert_models=[p4,p8,p16,p32]
            ) 
        else:
            self.agent = greedy_mode_transformer(
                num_experts=4,
                expert_models=[p4,p8,p16,p32]
            )

    def test_in_environment(self):
        """
        main
        collect batchsize of episode
        calculate loss 
        """
        epoch = 0
        while epoch < self.NUM_EPOCHS:
            epoch+=1
            sum_of_ep_rewards = self.play_one_episode()
            self.total_rewards.append(sum_of_ep_rewards)
            print(f"Episode {epoch}: total reward for application = {sum_of_ep_rewards}")
        print("Validation end")
    def play_one_episode(self):
        """
        episode：
          reset evn
          sample action
          submit and start job 
          more than time pool,then done
        """
        self.env.reset()
        self.app.reset()
        reward_in_episode = []
        #save single action, logits, reward in one episode
        done = False
        step_idx = 0

        while not done:
            #get state (history snapshot + job_info)
            data_input = self.env.get_current_data_input()
            remain = self.app.get_nodehour_left()
            print(remain)
            chosen_node, action_idx = self.sample_action(data_input,remain)
            #check if there is enough time to submit job
            # generat and submit a new job => run => reward => check episode end
            next_job_id = 100000 + step_idx
            new_job = self.app.create_new_job(action_idx, self.env, next_job_id)
            reward_val, effective_run_time, done = self.app.run_job_and_measure_reward(self.env, new_job)
            reward_in_episode.append(reward_val)
            print(f"[play_one_episode] Step={step_idx}, Chosen node ={chosen_node}, Reward={reward_val:.3f}, Done={done}")
            step_idx += 1
        # complete one episode, total return
        sum_of_rewards = sum(reward_in_episode)
        return sum_of_rewards

    def sample_action(self, data_input,remain):
        """
        greedy mode to sample an action:
        with the data input, do the prediction
        """
        # convert tensor
        np_input = raw_features_to_np_ndarray([data_input], parallel=False)
        print(f"Processed features shape: {np_input.shape}")
        tensor_in = torch.from_numpy(np_input).float().to(self.DEVICE)
        # gating_logits => [batch=1, 4]
        tensor_in = tensor_in.numpy().reshape(len(tensor_in), -1)
        logits = self.agent.forward(tensor_in)
        # calculate the score:
        # node_hour_consumption/(wait_time + exec_time)
        # and pick the smallest one.
        wait_4 = logits[0][0]
        wait_8 = logits[0][1]
        wait_16 = logits[0][2]
        wait_32 = logits[0][3]

        factor_4 = 1.0
        factor_8 = 95.0/100.0
        factor_16 = 87.0/100.0
        factor_32 = 67.0/100.0

        if remain - 4 * factor_4 * 48 >= 0:
            exec_time_4 = 48
        else:
            exec_time_4 = remain/(4*factor_4)
        if remain - 8 * factor_8 * 48 >= 0:
            exec_time_8 = 48
        else:
            exec_time_8 = remain/(8*factor_8)
        if remain - 16 * factor_16 * 48 >= 0:
            exec_time_16 = 48
        else:
            exec_time_16 = remain/(16*factor_16)
        if remain - 32 * factor_32 * 48 >= 0:
            exec_time_32 = 48
        else:
            exec_time_32 = remain/(32*factor_32)
        consumption_4 = exec_time_4*4*factor_4
        consumption_8 = exec_time_8*8*factor_8
        consumption_16 = exec_time_16*16*factor_16
        consumption_32 = exec_time_32*32*factor_32
        scores = {
            4: consumption_4 / (wait_4 + exec_time_4),
            8: consumption_8 / (wait_8 + exec_time_8),
            16: consumption_16 / (wait_16 + exec_time_16),
            32: consumption_32 / (wait_32 + exec_time_32),
        }
        action = max(scores, key=scores.get)
        action_idx = [4, 8, 16, 32].index(action)
        print(f"action in this run: {action}, remaining {remain}")
        return action, action_idx

    def calculate_loss(self, epoch_logits, epoch_weighted_log_probs):
        pol_loss = - torch.mean(epoch_weighted_log_probs)
        p = softmax(epoch_logits, dim=1)       # [N,4]
        logp = log_softmax(epoch_logits, dim=1)# [N,4]
        ent = - torch.mean(torch.sum(p * logp, dim=1), dim=0)
        ent_bonus = - self.BETA * ent

        return (pol_loss + ent_bonus), ent

    @staticmethod
    def get_discounted_rewards(rews, gamma):
        out = np.zeros_like(rews, dtype=float)
        for i in range(len(rews)):
            G = 0.0
            power = 0
            for j in range(i, len(rews)):
                G += (gamma ** power) * rews[j]
                power += 1
            out[i] = G
        return out

import sys
parser = argparse.ArgumentParser()
parser.add_argument('-sim_config', required=True)
parser.add_argument('--use_cuda', action='store_true')
parser.add_argument('--base_model_path_4',required=True)
parser.add_argument('--base_model_path_8',required=True)
parser.add_argument('--base_model_path_16',required=True)
parser.add_argument('--base_model_path_32',required=True)
args = parser.parse_args()
with open(args.sim_config, 'r') as f:
    sim_config = json.load(f)

RF = RandomForest(
    num_validation_epochs = 1000,
    use_cuda=args.use_cuda,
    model4_path=args.base_model_path_4,
    model8_path=args.base_model_path_8,
    model16_path=args.base_model_path_16,
    model32_path=args.base_model_path_32,
    sim_config=sim_config
)              
RF.test_in_environment()