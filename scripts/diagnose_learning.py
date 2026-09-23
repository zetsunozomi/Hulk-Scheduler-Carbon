"""Replay four recorded training rounds and inspect frozen actor gradients.

No optimizer steps, new action sampling, validation/test evaluation or source edits.
The diagnostics describe local full-batch gradients, not the actual Adam trajectory.
"""

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from statistics import fmean
import time

from carbon.common import digest, load_json, require
from carbon.config import Bundle
from carbon.environment import Environment
from carbon.learning import ActorCritic, input_tensors, monte_carlo_costs, torch
from carbon.policy_inputs import PolicyInputs
from carbon.policy_runner import ReplayCache, check_policy_inputs, configure_torch, read_checkpoint
from carbon.probe_resume import probe_lock
from carbon.runner import provenance, write_manifest
from diagnose_main import binding, checked_stage, records


ITERATIONS = (1, 4, 8, 16)


def row_key(row):
    return row['iteration'], row['budget_multiplier'], row['rollout_sample'], row['episode_id']


def load_trial(bundle, trial):
    plan = load_json(trial/'run-plan.json')
    summary = load_json(trial/'interaction-summary.json')
    require(summary['status'] == 'complete' and summary['run_plan_sha256'] == digest(trial/'run-plan.json'),
            'Completed interaction trial required')
    require(plan['config_sha256'] == bundle.manifest['config_sha256'] and
            plan['asset_sha256'] == bundle.manifest['asset_sha256'], 'Trial inputs changed')
    require(plan['source_sha256'] == provenance(bundle.root)['source_sha256'] and
            plan['torch'] == str(torch.__version__), 'Trial source/runtime changed')
    for name, sha in plan['script_sha256'].items():
        require(digest(Path(__file__).with_name(name)) == sha, 'Original trial script changed: '+name)
    settings = plan['settings']
    require(summary['settings'] == settings and settings['actor_interaction'] == 'product' and
            settings['wait_features'] == 'none' and settings['decision_mode'] == 'feedback' and
            settings['objective'] == 'robust', 'Expected the product robust feedback trial')
    attempts = sorted(trial.glob('ppo-attempt-[0-9][0-9][0-9]'))
    require(len(attempts) == 1, 'This bounded diagnosis expects one fresh, complete training attempt')
    attempt = attempts[0]; meta = load_json(attempt/'manifest.json')
    require(meta['status'] == 'complete' and meta['resume_checkpoint_sha256'] is None and
            meta['first_iteration'] == 1 and meta['settings'] == settings and
            meta['software']['source_sha256'] == plan['source_sha256'] and
            all(meta.get(k) == v for k, v in bundle.manifest.items()), 'Training manifest differs')
    history = records(attempt/'training.jsonl')
    require([r['iteration'] for r in history] == list(range(1, settings['iterations']+1)), 'Training history incomplete')
    episodes, chunks = records(attempt/'episodes.jsonl'), records(attempt/'chunks.jsonl')
    count = settings['iterations']*len(settings['budgets'])*settings['episodes_per_budget']
    require(len(episodes) == meta['total_episodes'] == summary['ppo_episodes'] == count and
            len(chunks) == meta['total_chunks'] == summary['ppo_chunks'], 'Training counts differ')
    cohort = {e.episode_id for e in bundle.episodes if e.split == 'train'}
    require(all(e['split'] == 'train' and e['episode_id'] in cohort and
                e['final_status'] == 'completed' and not e['censor_flag'] for e in episodes),
            'Only complete training outcomes are allowed')
    require(len({row_key(e) for e in episodes}) == count, 'Duplicate training outcome')
    groups = defaultdict(list)
    for chunk in chunks:
        require(chunk['split'] == 'train', 'Non-training chunk')
        groups[row_key(chunk)].append(chunk)
    require(set(groups) == {row_key(e) for e in episodes}, 'Unexpected/missing training chunks')
    for e in episodes:
        require([c['chunk_id'] for c in groups[row_key(e)]] == list(range(e['chunk_count'])),
                'Missing/duplicate chunk')
    for iteration in ITERATIONS:
        selected = [e for e in episodes if e['iteration'] == iteration]
        expected = {(beta, i) for beta in settings['budgets'] for i in range(settings['episodes_per_budget'])}
        require({(e['budget_multiplier'], e['rollout_sample']) for e in selected} == expected and
                len(selected) == len(expected), 'Incomplete selected training round')
        for sample in range(settings['episodes_per_budget']):
            require(len({e['episode_id'] for e in selected if e['rollout_sample'] == sample}) == 1,
                    'Training arrivals are not paired across budgets')
    files = binding(trial, ('run-plan.json', 'interaction-summary.json'))
    files.update({attempt.name+'/'+k: v for k, v in binding(attempt,
                  ('manifest.json', 'training.jsonl', 'episodes.jsonl', 'chunks.jsonl')).items()})
    for iteration in ITERATIONS:
        if iteration == 1:
            continue
        r = history[iteration-2]; p = attempt/r['checkpoint']; checkpoint = load_json(p)
        require(p.name == f'checkpoint-{iteration-1:06d}.json' and p.parent == attempt and
                digest(p) == r['checkpoint_sha256'], 'Wrong pre-update checkpoint')
        require(Path(checkpoint['weights_file']).name == checkpoint['weights_file'], 'Invalid weights path')
        for name in (p.name, checkpoint['weights_file']):
            files[attempt.name+'/'+name] = digest(attempt/name)
        require(files[attempt.name+'/'+checkpoint['weights_file']] == checkpoint['weights_sha256'], 'Weights changed')
    return settings, meta['references'], attempt, history, episodes, groups, files


def pre_update_model(bundle, encoder, settings, attempt, iteration):
    configure_torch(settings['seed'], settings['threads'])
    model = ActorCritic(len(encoder.global_names), len(encoder.action_names), interaction=settings['actor_interaction'])
    if iteration > 1:
        meta, state = read_checkpoint(attempt/f'checkpoint-{iteration-1:06d}.json')
        check_policy_inputs(meta, encoder, None, bundle)
        require(meta['iteration'] == iteration-1 and meta['settings'] == settings and
                meta['software']['source_sha256'] == provenance(bundle.root)['source_sha256'] and
                meta['software']['torch'] == str(torch.__version__), 'Pre-update checkpoint differs')
        model.load_state_dict(state['model'])
    model.eval()
    return model


def reconstruct(bundle, encoder, model, settings, episodes, groups):
    """Force recorded actions; verify every public input, probability and outcome."""
    cohort = {e.episode_id: e for e in bundle.episodes if e.split == 'train'}
    cache = ReplayCache(bundle, size=1)
    rollout = []
    endpoints = [str(r) for r in bundle.raw['power']['rho_interval']]
    refs = encoder.references['carbon_reference_g_per_kappa']
    physical = ('chunk_id', 'selected_nodes', 'remaining_updates_before', 'remaining_updates_after',
                'completed_updates', 'submit_utc', 'start_utc', 'end_utc', 'requested_walltime_hours',
                'queue_wait_hours', 'phases', 'carbon_g_per_kappa')
    ordered = sorted(episodes, key=lambda e: (cohort[e['episode_id']].arrival, e['episode_id'],
                                            e['budget_multiplier'], e['rollout_sample']))
    for row in ordered:
        original = cohort[row['episode_id']]
        beta = row['budget_multiplier']
        require(row['budget_hours'] == beta*encoder.references['time_reference_hours'], 'Training budget changed')
        episode = replace(original, budget_hours=row['budget_hours'])
        env = Environment(bundle, episode, cache.get(original), row['method'], settings['seed'])
        steps, costs = [], []
        for old in groups[row_key(row)]:
            encoded, info = encoder.encode(env.observe())
            require(info['policy_input_sha256'] == old['policy_input_sha256'], 'Reconstructed training input differs')
            action = encoder.nodes.index(old['selected_nodes'])
            with torch.no_grad():
                distribution, values = model(*input_tensors([encoded]))
                probs = distribution.probs[0].tolist()
                log_prob = float(distribution.log_prob(torch.tensor([action]))[0])
            require(len(probs) == len(old['policy_probabilities']) and
                    max(abs(a-b) for a, b in zip(probs, old['policy_probabilities'])) <= 1e-6 and
                    abs(log_prob-old['selected_log_probability']) <= 1e-5, 'Reconstructed training probabilities differ')
            require(max(abs(a-b) for a, b in zip(values[0].tolist(), old['predicted_cost_to_go'])) <= 2e-6,
                    'Reconstructed critic values differ')
            actual = env.step(old['selected_nodes'])
            require(all(actual[k] == old[k] for k in physical), 'Forced-action replay outcome differs')
            steps.append({'input': encoded, 'action': action, 'log_probability': old['selected_log_probability'],
                          'values': old['predicted_cost_to_go']})
            costs.append([actual['carbon_g_per_kappa'][rho]/refs[rho] for rho in endpoints])
        final = env.summary()
        require(all(final[k] == row[k] for k in ('final_status', 'deadline_miss', 'tat_hours',
                    'completed_updates', 'remaining_updates', 'nodehours', 'carbon_g_per_kappa')),
                'Reconstructed episode differs')
        rollout.append({'beta': beta, 'steps': steps, 'returns': monte_carlo_costs(costs, final['deadline_miss'])})
    return rollout


def gradient_vector(loss, parameters):
    return torch.cat([g.detach().reshape(-1).to(torch.float64)
                      for g in torch.autograd.grad(loss, parameters, retain_graph=True)])


def cosine(a, b):
    scale = float(a.norm()*b.norm())
    return float(torch.dot(a, b))/scale if scale > 0 else None


def gradient_diagnosis(model, rollout, duals, settings):
    """Actor-only local gradients. Keep all decisions, divide by episode count."""
    parameters = [p for n, p in model.named_parameters() if n.startswith('actor_')]
    gradients, probability_gradients, points = {}, {}, []
    for beta in settings['budgets']:
        episodes = [e for e in rollout if e['beta'] == beta]
        require(episodes, 'Missing budget rollout')
        steps = [s for e in episodes for s in e['steps']]
        returns = [r for e in episodes for r in e['returns']]
        require(len(returns) == len(steps), 'Incomplete return sequence')
        distribution, _ = model(*input_tensors([s['input'] for s in steps]))
        selected = torch.tensor([s['action'] for s in steps])
        old = torch.tensor([s['log_probability'] for s in steps])
        ratios = torch.exp(distribution.log_prob(selected)-old)
        require(float(torch.max(torch.abs(ratios.detach()-1))) < 1e-4, 'Expected frozen behavior policy')
        dual = duals[str(beta)]
        carbon_adv = torch.tensor([-sum(w*(r-v) for w, r, v in zip(dual['weights'], ret[:2], s['values'][:2]))
                                   for s, ret in zip(steps, returns)])
        miss_adv = torch.tensor([-dual['lambda']*(ret[2]-s['values'][2]) for s, ret in zip(steps, returns)])
        components = {'carbon': -(ratios*carbon_adv).sum()/len(episodes),
                      'miss': -(ratios*miss_adv).sum()/len(episodes),
                      'entropy': -settings['entropy']*distribution.entropy().sum()/len(episodes)}
        component_grads = {name: gradient_vector(value, parameters) for name, value in components.items()}
        grad = sum(component_grads.values())
        require(bool(torch.isfinite(grad).all()), 'Nonfinite diagnostic gradient')
        gradients[beta] = grad
        first, offset = [], 0
        for e in episodes:
            first.append(offset); offset += len(e['steps'])
        mean_probs = distribution.probs[first].mean(dim=0)
        probability_gradients[beta] = [gradient_vector(p, parameters) for p in mean_probs]
        points.append({'budget_multiplier': beta, 'episodes': len(episodes), 'chunks': len(steps),
                       'initial_action_counts_by_index': dict(Counter(e['steps'][0]['action'] for e in episodes)),
                       'initial_mean_probabilities': mean_probs.detach().tolist(),
                       'mean_episode_returns': [fmean(e['returns'][0][i] for e in episodes) for i in range(3)],
                       'dual': dual, 'gradient_norm': float(grad.norm()),
                       'component_gradient_norms': {k: float(v.norm()) for k, v in component_grads.items()},
                       'carbon_miss_gradient_cosine': cosine(component_grads['carbon'], component_grads['miss'])})
    # The current training design has equal episode counts across budgets.
    require(len({p['episodes'] for p in points}) == 1, 'Unequal budget weights')
    shared = sum(gradients.values())/len(gradients)
    for point in points:
        beta = point['budget_multiplier']; own = gradients[beta]
        point['own_shared_gradient_cosine'] = cosine(own, shared)
        point['own_loss_directional_derivative_along_shared_descent'] = -float(torch.dot(own, shared))
        point['mean_action_probability_derivative_per_unit_descent'] = {
            name: [-float(torch.dot(g, direction))/float(direction.norm()) if float(direction.norm()) > 0 else None
                   for g in probability_gradients[beta]]
            for name, direction in (('own_budget', own), ('shared_budgets', shared))}
    return {'operating_points': points, 'shared_actor_gradient_norm': float(shared.norm()),
            'pairwise_budget_gradient_cosines': [{'left': a, 'right': b, 'cosine': cosine(gradients[a], gradients[b])}
                                                for i, a in enumerate(settings['budgets']) for b in settings['budgets'][i+1:]],
            'interpretation': 'Frozen pre-update full-batch actor gradients including entropy; critic excluded. '
                              'Negative own/shared cosine indicates local conflict. This is not an Adam update, '
                              'the later PPO epochs, or a causal proof of collapse.'}


def write_readout(output, report):
    write_manifest(output/'learning-summary.json', report)
    lines = ['# Frozen learning-signal diagnosis', '',
             '仅重放已记录的 train 动作并计算冻结梯度；没有 optimizer.step、新采样或 test。',
             'cosine < 0 表示该预算局部梯度与共享更新方向冲突。',
             '这是更新开始处的完整 rollout 诊断，不等同于实际 Adam/minibatch/多 epoch 轨迹，也不证明因果。', '',
             '| 轮次 | beta | 旧动作数（按 nodes 顺序） | P(各动作) | own/shared cosine | 碳/违约/熵梯度范数 |',
             '|---:|---:|---|---|---:|---|']
    for result in report['rounds']:
        for p in result['gradients']['operating_points']:
            counts = [p['initial_action_counts_by_index'].get(str(i), p['initial_action_counts_by_index'].get(i, 0))
                      for i in range(len(report['nodes']))]
            cos = p['own_shared_gradient_cosine']
            cos_text = '—' if cos is None else f'{cos:.4f}'
            lines.append(f"| {result['iteration']} | {p['budget_multiplier']:.4f} | {counts} | "
                         + ', '.join(f'{v:.5f}' for v in p['initial_mean_probabilities'])
                         + ' | ' + cos_text)
            lines[-1] += ' | ' + ', '.join(f'{p["component_gradient_norms"][k]:.5f}' for k in ('carbon', 'miss', 'entropy')) + ' |'
    lines += ['', f"Nodes: {report['nodes']}; status: {report['status']}; 完成轮次: {[r['iteration'] for r in report['rounds']]}", '',
              'JSON 另含每轮检查数量、输入绑定、预算梯度夹角、动作概率的局部变化方向及耗时。', '']
    tmp = output/'learning-summary.md.tmp'; tmp.write_text('\n'.join(lines)); tmp.replace(output/'learning-summary.md')


def run(bundle, trial, output, resume=False):
    require(bundle.raw['purpose'] in {'development', 'synthetic'}, 'Development diagnosis only')
    trial, output = Path(trial), Path(output)
    settings, refs, attempt, history, episodes, groups, files = load_trial(bundle, trial)
    encoder = PolicyInputs(bundle, None, refs, settings['forecast_mode'], settings['wait_features'])
    scripts = (Path(__file__), Path(__file__).with_name('diagnose_main.py'), Path(__file__).with_name('main_train.py'))
    plan = {'kind': 'frozen_learning_diagnosis_v1', 'config_sha256': bundle.manifest['config_sha256'],
            'source_sha256': provenance(bundle.root)['source_sha256'], 'torch': str(torch.__version__),
            'script_sha256': {p.name: digest(p) for p in scripts}, 'trial_files': files,
            'iterations': list(ITERATIONS), 'settings': settings, 'split': 'train', 'optimizer_steps': 0}
    with probe_lock(output, resume, name='.learning-diagnosis.lock'):
        path = output/'run-plan.json'
        if path.exists():
            require(load_json(path) == plan, 'Diagnosis inputs/software/settings changed')
        else:
            require(not any(p.name != '.learning-diagnosis.lock' for p in output.iterdir()), 'Unknown existing output')
            write_manifest(path, plan)
        report = {'status': 'partial', 'run_plan_sha256': digest(path), 'nodes': encoder.nodes, 'rounds': []}
        write_readout(output, report)
        for iteration in ITERATIONS:
            expected = {'kind': 'frozen_learning_round_v1', 'iteration': iteration, 'run_plan_sha256': digest(path)}
            def execute(directory):
                directory.mkdir(); write_manifest(directory/'manifest.json', {**expected, 'status': 'running'})
                begun = time.monotonic()
                model = pre_update_model(bundle, encoder, settings, attempt, iteration)
                selected = [e for e in episodes if e['iteration'] == iteration]
                rollout = reconstruct(bundle, encoder, model, settings, selected, groups)
                gradients = gradient_diagnosis(model, rollout, history[iteration-1]['duals_before'], settings)
                for point in gradients['operating_points']:
                    recorded = history[iteration-1]['pre_update_episode_means'][str(point['budget_multiplier'])]
                    require(max(abs(a-b) for a, b in zip(recorded, point['mean_episode_returns'])) < 1e-10,
                            'Reconstructed training means differ')
                result = {'iteration': iteration, 'verified_episodes': len(rollout),
                          'verified_chunks': sum(len(e['steps']) for e in rollout),
                          'gradients': gradients, 'elapsed_seconds': time.monotonic()-begun}
                write_manifest(directory/'gradients.json', result)
                write_manifest(directory/'manifest.json', {**expected, 'status': 'complete'})
            def validate(directory):
                result = load_json(directory/'gradients.json')
                require(result['iteration'] == iteration and result['verified_episodes'] == len(settings['budgets'])*settings['episodes_per_budget'],
                        'Incomplete diagnostic round')
                return result
            result, seal = checked_stage(output/f'round-{iteration:06d}', expected,
                                         ('manifest.json', 'gradients.json'), execute, validate)
            report['rounds'].append({**result, 'stage_files': seal})
            report['status'] = 'complete' if iteration == ITERATIONS[-1] else 'partial'
            write_readout(output, report)
            print(f"Learning diagnosis round {iteration}: {result['verified_episodes']} episodes, {result['verified_chunks']} chunks verified.", flush=True)
        print(f'Learning diagnosis complete: {output}/learning-summary.md', flush=True)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'trial', 'output'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run(Bundle(Path(args.config)), args.trial, args.output, args.resume)


if __name__ == '__main__':
    main()
