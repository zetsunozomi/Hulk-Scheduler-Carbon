"""A learned complete scale sequence, generated before any target admission."""
from copy import deepcopy
import hashlib
import time

from .common import json_text, require, seconds
from .learning import sample_action


def build_precommitted_plan(bundle, initial, encoder, model, generator):
    require(encoder.wait_features == 'none', 'Precommitted-RL does not use wait advice')
    remaining, allocated = initial['total_updates'], 0.
    require(initial['remaining_updates'] == remaining, 'Precommit requires the initial observation')
    plan = []
    while remaining:
        obs = deepcopy(initial)
        obs['remaining_updates'] = remaining
        # A known function of the chosen sequence, not measured elapsed time.
        obs['remaining_budget_hours'] = initial['budget_hours'] - allocated
        descriptors = []
        for n, profile in bundle.workload.profiles.items():
            work = bundle.workload.plan(n,remaining,not plan)
            descriptors.append({'nodes':n, 'feasible':work is not None,
                                'updates_per_hour':float(profile.updates_per_hour),
                                'eta':float(bundle.workload.eta(n)),
                                'planned_updates':work.updates if work else 0,
                                'actual_seconds':float(seconds(work.actual)) if work else None,
                                'requested_seconds':float(seconds(work.requested)) if work else None})
        obs['actions'] = descriptors
        begun = time.monotonic()
        encoded, metadata = encoder.encode(obs)
        action, logp, values, probabilities = sample_action(model,encoded,generator)
        inference = time.monotonic()-begun
        chosen = descriptors[action]
        require(chosen['feasible'] and chosen['planned_updates']>0, 'Invalid precommitted work progress')
        plan.append({'input':encoded,'action':action,'log_probability':logp,'values':values,
                     'probabilities':probabilities,'metadata':metadata,'inference_seconds':inference,
                     'nodes':chosen['nodes'],'planned_updates':chosen['planned_updates']})
        remaining -= chosen['planned_updates']
        allocated += chosen['actual_seconds']/3600
    binding = [{'nodes':p['nodes'],'planned_updates':p['planned_updates'],
                'policy_input_sha256':p['metadata']['policy_input_sha256']} for p in plan]
    plan_hash = hashlib.sha256(json_text(binding).encode()).hexdigest()
    for p in plan:
        p['metadata'].update(precommitted_plan_sha256=plan_hash, decision_mode='precommitted')
    return plan, {'kind':'precommitted_scale_plan_v1','plan_sha256':plan_hash,'steps':binding,
                  'forecast_issue_utc':plan[0]['metadata']['forecast_issue_utc'],
                  'created_before_first_target_submission':True,
                  'input_contract':'initial queue and CI only; planned work and allocated duration; no observed later elapsed time'}
