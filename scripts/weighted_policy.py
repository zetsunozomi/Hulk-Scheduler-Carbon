"""Two-cost, complete-episode PPO with independent actors for alpha in [0, 1]."""

from copy import deepcopy
import hashlib
import math

from carbon.common import json_text, require
from carbon.learning import Categorical, input_tensors, mlp, nn, ppo_losses, torch
from carbon.policy_inputs import PolicyInputs


class WeightedInputs(PolicyInputs):
    """Keep physical feature scales fixed; reward calibration is separate."""

    def __init__(self, bundle, references):
        super().__init__(bundle, None, references, 'window', 'none')
        self.global_names[1:3] = ['elapsed_hours/feature_Tref', 'alpha']

    def schema(self):
        return {**super().schema(), 'version': 'weighted_policy_inputs_v1',
                'reward_normalization': 'separate pooled observation-only episode means'}

    def encode(self, observation, elapsed_hours, alpha):
        encoded, metadata = super().encode(observation)
        encoded['global'][1:3] = [elapsed_hours/self.tref, alpha]
        metadata['policy_input_sha256'] = hashlib.sha256(json_text(encoded).encode()).hexdigest()
        return encoded, metadata


class WeightedActorCritic(nn.Module):
    def __init__(self, global_size, action_size, alphas, width=128):
        super().__init__()
        require(alphas and len(set(alphas)) == len(alphas) and
                all(math.isfinite(a) and 0 <= a <= 1 for a in alphas), 'Invalid alpha grid')
        self.alphas = tuple(alphas)
        branch = nn.ModuleDict({'actions': mlp(action_size, width),
                                'context': mlp(global_size+width, width),
                                'score': nn.Sequential(nn.Linear(3*width, width), nn.ReLU(), nn.Linear(width, 1))})
        self.actors = nn.ModuleList([branch, *(deepcopy(branch) for _ in alphas[1:])])
        self.critic_actions = mlp(action_size, width)
        self.critic_context = mlp(global_size+width, width)
        self.critic_heads = nn.Linear(width, 2)  # normalized remaining TAT and carbon costs

    def forward(self, global_state, action_state, mask):
        require(bool(mask.any(dim=1).all()), 'All actions masked')
        matches = global_state[:, 2:3] == global_state.new_tensor(self.alphas).unsqueeze(0)
        require(bool((matches.sum(dim=1) == 1).all()), 'Alpha outside trained grid')
        logits = action_state.new_zeros(mask.shape)
        for i, branch in enumerate(self.actors):
            index = matches[:, i].nonzero(as_tuple=True)[0]
            if index.numel() == 0:
                continue
            weights = mask[index].unsqueeze(-1).to(action_state.dtype)
            actions = branch['actions'](action_state[index])
            pooled = (actions*weights).sum(1)/weights.sum(1)
            context = branch['context'](torch.cat((global_state[index], pooled), -1))
            repeated = context.unsqueeze(1).expand_as(actions)
            score = branch['score'](torch.cat((actions, repeated, actions*repeated), -1)).squeeze(-1)
            logits = logits.index_copy(0, index, score.masked_fill(~mask[index], -torch.inf))
        weights = mask.unsqueeze(-1).to(action_state.dtype)
        actions = self.critic_actions(action_state)
        pooled = (actions*weights).sum(1)/weights.sum(1)
        values = self.critic_heads(self.critic_context(torch.cat((global_state, pooled), -1)))
        return Categorical(logits=logits), values


def observe(warmup, rows, rho):
    require(not warmup.get('frozen', False), 'Observation means already frozen')
    for row in rows:
        require(row['split'] == 'train' and row['final_status'] == 'completed', 'Need complete training episodes')
        t, c = row['tat_hours'], row['carbon_g_per_kappa'][rho]
        require(all(math.isfinite(v) and v > 0 for v in (t, c)), 'Invalid observed costs')
        warmup['count'] += 1
        warmup['tat_sum_hours'] += t
        warmup['carbon_sum_g_per_kappa'] += c


def freeze(warmup, rho, rounds):
    require(warmup['count'] > 0 and not warmup['frozen'], 'Invalid calibration state')
    warmup['frozen'] = True
    return {'kind': 'weighted_reward_normalizers_v1', 'split': 'train', 'rho': rho,
            'observation_iterations': rounds, 'episodes': warmup['count'],
            'tat_hours': warmup['tat_sum_hours']/warmup['count'],
            'carbon_g_per_kappa': warmup['carbon_sum_g_per_kappa']/warmup['count'],
            'rule': 'common arithmetic means over complete episodes, then frozen'}


def returns_to_go(increments, normalizers):
    future = [0., 0.]
    returns = []
    for t, c in reversed(increments):
        require(all(math.isfinite(v) and v >= 0 for v in (t, c)), 'Invalid cost increment')
        future = [future[0]+t/normalizers['tat_hours'], future[1]+c/normalizers['carbon_g_per_kappa']]
        returns.append(future[:])
    return returns[::-1]


def cost(tat, carbon, alpha, normalizers):
    return alpha*tat/normalizers['tat_hours'] + (1-alpha)*carbon/normalizers['carbon_g_per_kappa']


def optimize(model, optimizer, episodes, normalizers, settings, rng):
    for episode in episodes:
        alpha = episode['alpha']
        for step, target in zip(episode['steps'], returns_to_go(episode['increments'], normalizers)):
            step['return'] = target
            step['advantage'] = -(alpha*(target[0]-step['values'][0]) +
                                   (1-alpha)*(target[1]-step['values'][1]))
    metrics = []
    for _ in range(settings['ppo_epochs']):
        order = list(range(len(episodes))); rng.shuffle(order)
        for begin in range(0, len(order), settings['minibatch_episodes']):
            batch = [episodes[i] for i in order[begin:begin+settings['minibatch_episodes']]]
            steps = [s for episode in batch for s in episode['steps']]
            distribution, values = model(*input_tensors([s['input'] for s in steps]))
            actions = torch.tensor([s['action'] for s in steps], dtype=torch.long)
            old = torch.tensor([s['log_probability'] for s in steps])
            advantage = torch.tensor([s['advantage'] for s in steps])
            targets = torch.tensor([s['return'] for s in steps])
            losses = ppo_losses(distribution.log_prob(actions), old, advantage, values, targets,
                                distribution.entropy(), len(batch), settings['clip'],
                                settings['value_coefficient'], settings['entropy'])
            require(all(bool(torch.isfinite(x)) for x in losses), 'Nonfinite weighted PPO loss')
            optimizer.zero_grad(set_to_none=True)
            losses[0].backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), settings['gradient_norm'], error_if_nonfinite=True)
            optimizer.step()
            require(all(bool(torch.isfinite(p).all()) for p in model.parameters()), 'Nonfinite PPO parameter')
            metrics.append({'loss': float(losses[0].detach()), 'actor_loss': float(losses[1].detach()),
                            'critic_loss': float(losses[2].detach()), 'entropy': float(losses[3].detach()),
                            'gradient_norm': float(norm)})
    return metrics
