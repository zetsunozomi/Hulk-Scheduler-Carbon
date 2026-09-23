"""Undiscounted complete-episode constrained PPO and budget-specific dual updates."""

from copy import deepcopy
import math

from .common import ContractError, require

try:
    import torch
    from torch import nn
    from torch.distributions import Categorical
except ImportError:
    raise ContractError('PPO needs PyTorch; reuse the existing environment or install requirements-p3.txt') from None


def mlp(in_features, width):
    return nn.Sequential(nn.Linear(in_features,width),nn.ReLU(),nn.Linear(width,width),nn.ReLU())


class ActorCritic(nn.Module):
    def __init__(self, global_size, action_size, width=128, interaction='concat', actor_budgets=None):
        super().__init__()
        require(interaction in {'concat', 'product'}, 'Unknown actor interaction')
        self.interaction = interaction
        self.actor_actions = mlp(action_size,width)
        self.actor_context = mlp(global_size+width,width)
        score_width = (3 if interaction == 'product' else 2)*width
        self.actor_score = nn.Sequential(nn.Linear(score_width,width),nn.ReLU(),nn.Linear(width,1))
        self.critic_actions = mlp(action_size,width)
        self.critic_context = mlp(global_size+width,width)
        self.critic_heads = nn.Linear(width,3)
        # Clone after initializing the ordinary actor AND critic: no extra RNG
        # consumption, identical starting functions, and no shared actor tensors.
        self.actor_budget_values = None if actor_budgets is None else tuple(actor_budgets)
        self.actor_branches = nn.ModuleList()
        if self.actor_budget_values is not None:
            grid = torch.tensor(self.actor_budget_values, dtype=torch.float32)
            require(global_size >= 3 and grid.ndim == 1 and grid.numel() > 0 and
                    bool(torch.isfinite(grid).all()) and bool((grid > 0).all()) and
                    grid.unique().numel() == grid.numel(), 'Invalid independent actor budget grid')
            for _ in self.actor_budget_values[1:]:
                self.actor_branches.append(nn.ModuleDict({'actions': deepcopy(self.actor_actions),
                                                         'context': deepcopy(self.actor_context),
                                                         'score': deepcopy(self.actor_score)}))

    def _actor_logits(self, global_state, action_state, mask, actions, context_net, score):
        weights = mask.unsqueeze(-1).to(action_state.dtype)
        actor = actions(action_state)
        pooled = (actor*weights).sum(dim=1)/weights.sum(dim=1)
        context = context_net(torch.cat((global_state,pooled),dim=-1))
        repeated = context.unsqueeze(1).expand(-1,actor.shape[1],-1)
        # A shared additive context can cancel in softmax when the score head's
        # ReLU gates coincide across actions. The optional product supplies an
        # explicit state/action interaction; it does not guarantee budget use.
        joint = (actor, repeated, actor*repeated) if self.interaction == 'product' else (actor, repeated)
        return score(torch.cat(joint,dim=-1)).squeeze(-1).masked_fill(~mask,-torch.inf)

    def forward(self, global_state, action_state, mask):
        require(bool(mask.any(dim=1).all()), 'All policy actions are masked')
        if self.actor_budget_values is None:
            logits = self._actor_logits(global_state, action_state, mask,
                                        self.actor_actions, self.actor_context, self.actor_score)
        else:
            # policy_inputs_v2 puts the TOTAL budget/Tref at index 2. Never
            # route on remaining budget, and never interpolate unknown ticks.
            grid = global_state.new_tensor(self.actor_budget_values)
            matches = global_state[:, 2:3] == grid.unsqueeze(0)
            require(bool((matches.sum(dim=1) == 1).all()), 'Budget is outside independent actor grid')
            logits = action_state.new_zeros(mask.shape)
            for i in range(len(self.actor_budget_values)):
                index = matches[:, i].nonzero(as_tuple=True)[0]
                if index.numel() == 0:
                    continue
                modules = ((self.actor_actions, self.actor_context, self.actor_score) if i == 0 else
                           tuple(self.actor_branches[i-1][k] for k in ('actions', 'context', 'score')))
                values = self._actor_logits(global_state[index], action_state[index], mask[index], *modules)
                logits = logits.index_copy(0, index, values)
        weights = mask.unsqueeze(-1).to(action_state.dtype)
        critic = self.critic_actions(action_state)
        pooled_value = (critic*weights).sum(dim=1)/weights.sum(dim=1)
        heads = self.critic_heads(self.critic_context(torch.cat((global_state,pooled_value),dim=-1)))
        values = torch.cat((heads[:,:2],torch.sigmoid(heads[:,2:])),dim=-1)
        return Categorical(logits=logits), values


def input_tensors(encoded):
    return (torch.tensor([x['global'] for x in encoded],dtype=torch.float32),
            torch.tensor([x['actions'] for x in encoded],dtype=torch.float32),
            torch.tensor([x['mask'] for x in encoded],dtype=torch.bool))


def sample_action(model, encoded, generator):
    with torch.no_grad():
        distribution, values = model(*input_tensors([encoded]))
        action = torch.multinomial(distribution.probs,1,generator=generator).squeeze(-1)
        return int(action.item()), float(distribution.log_prob(action).item()), values[0].tolist(), distribution.probs[0].tolist()


def monte_carlo_costs(chunk_costs, missed):
    """Gamma=1: carbon increments; a single terminal violation cost."""
    require(chunk_costs and missed in (True,False), 'Complete episode and known miss required')
    future = [0.0,0.0,float(missed)]
    returns = []
    for cost in reversed(chunk_costs):
        require(len(cost)==2 and all(math.isfinite(v) and v>=0 for v in cost), 'Invalid carbon increment')
        future = [future[0]+cost[0],future[1]+cost[1],future[2]]
        returns.append(future[:])
    return list(reversed(returns))


def initial_dual(objective):
    require(objective in {'robust','lower','upper'}, 'Unknown power objective')
    return {'weights':[.5,.5] if objective=='robust' else [1.0,0.0] if objective=='lower' else [0.0,1.0],
            'logits':[0.0,0.0],'lambda':1.0}


def update_dual(dual, episode_means, epsilon, rate_p=.05, rate_lambda=.05, objective='robust'):
    require(len(episode_means)==3 and all(math.isfinite(v) for v in episode_means), 'Nonfinite dual statistics')
    weights, logits = dual['weights'][:], dual['logits'][:]
    if objective=='robust':
        logits = [logits[i]+rate_p*episode_means[i] for i in range(2)]
        peak = max(logits); logits = [v-peak for v in logits]
        exponentials = [math.exp(v) for v in logits]
        weights = [v/sum(exponentials) for v in exponentials]
    multiplier = max(0.0,dual['lambda']+rate_lambda*(episode_means[2]-epsilon))
    require(math.isfinite(multiplier) and all(math.isfinite(v) for v in logits), 'Dual update diverged; no silent cap applied')
    return {'weights':weights,'logits':logits,'lambda':multiplier}


def ppo_losses(new_log_prob, old_log_prob, advantage, values, returns, entropy, episode_count,
               clip=.2, value_coefficient=.5, entropy_coefficient=.01):
    """Sum over each episode's decisions, then average over episodes, never chunks."""
    ratios = torch.exp(new_log_prob-old_log_prob)
    surrogate = torch.minimum(ratios*advantage,torch.clamp(ratios,1-clip,1+clip)*advantage)
    actor = -surrogate.sum()/episode_count
    critic = ((values-returns)**2).mean(dim=-1).sum()/episode_count
    bonus = entropy.sum()/episode_count
    total = actor+value_coefficient*critic-entropy_coefficient*bonus
    return total, actor, critic, bonus


def optimize_ppo(model, optimizer, episodes, duals, settings, shuffle_rng):
    require(episodes, 'Empty PPO rollout')
    # Returns and pre-update critic baselines stay frozen for every PPO epoch.
    for episode in episodes:
        dual = duals[str(episode['beta'])]
        weights = [*dual['weights'],dual['lambda']]
        for step, target in zip(episode['steps'],episode['returns']):
            step['return'] = target
            step['advantage'] = -sum(w*(r-v) for w,r,v in zip(weights,target,step['values']))
    metrics = []
    for _ in range(settings['epochs']):
        order = list(range(len(episodes))); shuffle_rng.shuffle(order)
        for begin in range(0,len(order),settings['minibatch_episodes']):
            batch = [episodes[i] for i in order[begin:begin+settings['minibatch_episodes']]]
            steps = [s for episode in batch for s in episode['steps']]
            distribution, values = model(*input_tensors([s['input'] for s in steps]))
            actions = torch.tensor([s['action'] for s in steps],dtype=torch.long)
            old = torch.tensor([s['log_probability'] for s in steps])
            advantage = torch.tensor([s['advantage'] for s in steps])
            returns = torch.tensor([s['return'] for s in steps])
            losses = ppo_losses(distribution.log_prob(actions),old,advantage,values,returns,distribution.entropy(),len(batch),
                                settings['clip'],settings['value_coefficient'],settings['entropy'])
            require(all(bool(torch.isfinite(x)) for x in losses), 'Nonfinite PPO loss')
            optimizer.zero_grad(set_to_none=True)
            losses[0].backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(),settings['gradient_norm'],error_if_nonfinite=True)
            optimizer.step()
            require(all(bool(torch.isfinite(p).all()) for p in model.parameters()), 'Nonfinite PPO parameter')
            metrics.append({'total_loss':float(losses[0].detach()),'actor_loss':float(losses[1].detach()),
                            'value_loss':float(losses[2].detach()),'entropy_per_episode':float(losses[3].detach()),
                            'gradient_norm_before_clip':float(norm),'episodes':len(batch),'chunks':len(steps)})
    return metrics
