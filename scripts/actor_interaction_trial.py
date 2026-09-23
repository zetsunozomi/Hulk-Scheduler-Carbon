"""One predeclared development intervention: explicit actor state/action product.

Reuse sealed fixed/planner outcomes only when all non-policy source files match.
Train a fresh seed11 policy, keeping the prior 64-round budget and environment.
"""

import argparse
import math
from pathlib import Path

from carbon.baselines import check_references, fixed_records
from carbon.common import digest, load_json, require
from carbon.config import Bundle
from carbon.main_pilot import make_budget_grid, ppo_stage
from carbon.policy_runner import read_checkpoint, settings_for, torch
from carbon.probe_resume import probe_lock
from carbon.results import ResultRun
from carbon.runner import provenance, write_manifest
from diagnose_main import binding, comparison, response_summary
from main_train import checkpoint_at, validation_stage


ITERATIONS = 64
VALIDATION_ITERATIONS = (16, 32, 48, 64)
EPISODES_PER_BUDGET = 16
POLICY_CHANGES = {'learning.py', 'policy_runner.py', 'stress.py', '__main__.py'}


def policy_only_changes(before, after):
    """Explicit reuse contract; no ordinary checkpoint/source guard is relaxed."""
    require(set(before) == set(after), 'Source file inventory changed')
    changed = {name for name in before if before[name] != after[name]}
    require(changed <= POLICY_CHANGES, 'Non-policy implementation changed: '+', '.join(sorted(changed-POLICY_CHANGES)))
    return {name: {'before': before[name], 'after': after[name]} for name in sorted(changed)}


def reuse_inputs(bundle, pilot, main, diagnosis, software):
    pilot_plan = load_json(pilot/'pipeline-contract.json')
    require(pilot_plan['config_sha256'] == bundle.manifest['config_sha256'] and
            pilot_plan['asset_sha256'] == bundle.manifest['asset_sha256'], 'Pilot inputs differ')
    changes = policy_only_changes(pilot_plan['source_sha256'], software['source_sha256'])
    references = load_json(pilot/'references.json')
    check_references(references, bundle.manifest)
    require(load_json(pilot/'references-seal.json') == {'references_sha256': digest(pilot/'references.json')},
            'Reference seal differs')
    groups, seals = {}, {}
    for split in ('train', 'validation'):
        directory = pilot/f'fixed-{split}'
        meta = load_json(directory/'manifest.json')
        require(meta['status'] == 'complete' and all(meta.get(k) == v for k, v in bundle.manifest.items()) and
                meta['software']['source_sha256'] == pilot_plan['source_sha256'], 'Fixed inputs/recorded implementation differ')
        expected = {e.episode_id for e in bundle.episodes if e.split == split}
        require(meta['selected_splits'] == [split] and set(meta['selected_episode_ids']) == expected and
                meta['shard_count'] == 1 and meta['shard_index'] == 0 and
                meta['fixed_nodes'] == bundle.raw['cluster']['allowed_nodes'], 'Expected the complete fixed cohort')
        seals[split] = binding(directory, ('manifest.json', 'episodes.jsonl', 'chunks.jsonl'))
        require(load_json(directory/'stage-seal.json') == seals[split], 'Fixed stage seal differs')
        ResultRun(directory, references, allowed_splits={split})
        groups[split] = fixed_records(directory/'episodes.jsonl', split, meta['fixed_nodes'])
        require(all(set(g) == expected for g in groups[split].values()), 'Fixed cohort coverage differs')
    require(references['source_episodes_sha256'] == seals['train']['episodes.jsonl'], 'Reference source differs')
    grid = make_budget_grid(references, groups['train'])
    require(load_json(pilot/'budget-grid.json') == grid, 'Training-derived budget grid changed')
    seals.update(references=digest(pilot/'references.json'), grid=digest(pilot/'budget-grid.json'))
    old_plan = load_json(main/'run-plan.json')
    diag_plan = load_json(diagnosis/'run-plan.json')
    baseline = load_json(diagnosis/'diagnosis-summary.json')
    require(old_plan['pilot_artifacts'] == seals and old_plan['source_sha256'] == pilot_plan['source_sha256'] and
            old_plan['config_sha256'] == bundle.manifest['config_sha256'] and
            old_plan['asset_sha256'] == bundle.manifest['asset_sha256'], 'Original training inputs differ')
    require(diag_plan['pilot'] == seals and diag_plan['source_sha256'] == old_plan['source_sha256'] and
            diag_plan['budget_grid'] == grid and diag_plan['config_sha256'] == old_plan['config_sha256'] and
            diag_plan['main_plan_sha256'] == digest(main/'run-plan.json') and
            diag_plan['main_summary_sha256'] == digest(main/'validation-summary.json'), 'Diagnosis inputs differ')
    require(baseline['status'] == 'complete' and baseline['run_plan_sha256'] == digest(diagnosis/'run-plan.json') and
            baseline['completed_planner_units'] == baseline['expected_planner_units'], 'Diagnosis is incomplete')
    for name, seal in baseline['stage_files'].items():
        require(binding(diagnosis/name, seal) == seal, 'Diagnosis stage seal differs: '+name)
    require(old_plan['torch'] == str(torch.__version__), 'Use the original PyTorch runtime for this controlled trial')
    evidence = {'pilot_artifacts': seals, 'policy_source_changes': changes,
                'original_main_plan_sha256': digest(main/'run-plan.json'),
                'diagnosis_plan_sha256': digest(diagnosis/'run-plan.json'),
                'diagnosis_summary_sha256': digest(diagnosis/'diagnosis-summary.json')}
    return references, grid, baseline, old_plan['settings'], evidence


def write_readout(output, report):
    write_manifest(output/'interaction-summary.json', report)
    lines = ['# Actor interaction development trial', '',
             '新 actor 加入 action_embedding × context_embedding；其余训练设置、输入、预算与环境不变。',
             '只报告 validation，所有预定 checkpoint 保留；不选择最好结果或声称正式 E2/E3 已完成。',
             'C 是最坏归一化端点成本，区间内优势还需分别比较两个端点；Fixed-Mix 是原 validation 拟合参考。', '',
             '| 方法 | D(h) | 完成/总数 | mean TAT(h) | C | miss 下界–上界 | 换规模 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    def fmt(value): return '—' if value is None else f'{value:.3f}'
    points = report['baselines'] + [p for c in report['checkpoints'] for p in c['operating_points']]
    for point in sorted(points, key=lambda p: p['budget_hours']):
        s = point['summary']; lo, hi = s['miss_rate_bounds']
        lines.append(f"| {point['method']} | {point['budget_hours']:.2f} | {s['completed']}/{s['outcomes']} | {fmt(s['mean_tat_hours'])} | {fmt(s['worst_normalized_carbon'])} | {lo:.1%}–{hi:.1%} | {s['observed_scale_change_fraction']:.1%} |")
    lines += ['', '## 原 Fixed-Mix validation 参考', '']
    for mix in report['fixed_mix']:
        lines.append(f"- D={mix['budget_hours']:.2f}h：经验可行={mix['empirical_feasible']}，C={fmt(mix['estimated_objective'])}")
    lines += ['', '## 新 actor 初始预算响应（同一到达配对）', '']
    for checkpoint in report['checkpoints']:
        for point in checkpoint['actor']['budget_response']:
            lines.append(f"- 第 {point['iteration']} 轮：平均最大 TV={point['mean_max_pairwise_budget_TV']:.8f}，最大 TV={point['max_pairwise_budget_TV']:.8f}")
    lines += ['', f"状态：{report['status']}；新策略已评估轮次：{[c['iteration'] for c in report['checkpoints']]}", '',
              'JSON 另含 p95 TAT、node-hours、端点成本、实际序列、动作概率和全部输入绑定。', '']
    tmp = output/'interaction-summary.md.tmp'
    tmp.write_text('\n'.join(lines), encoding='utf-8'); tmp.replace(output/'interaction-summary.md')


def run(bundle, pilot, main, diagnosis, output, resume=False):
    require(bundle.raw['purpose'] in {'development', 'synthetic'}, 'This trial is development only')
    pilot, main, diagnosis, output = map(Path, (pilot, main, diagnosis, output))
    software = provenance(bundle.root)
    references, grid, baseline, previous_settings, reuse = reuse_inputs(bundle, pilot, main, diagnosis, software)
    settings = settings_for(ITERATIONS, episodes_per_budget=EPISODES_PER_BUDGET, minibatch_episodes=16,
                            seed=11, wait_features='none', decision_mode='feedback',
                            budgets=grid['budget_multipliers'], actor_interaction='product')
    without_interaction = lambda value: {k: v for k, v in value.items() if k != 'actor_interaction'}
    require(without_interaction(settings) == without_interaction(previous_settings) and
            previous_settings.get('actor_interaction', 'concat') == 'concat',
            'The intervention may only change actor interaction, not the original training settings')
    scripts = [Path(__file__), Path(__file__).with_name('main_train.py'), Path(__file__).with_name('diagnose_main.py')]
    plan = {'kind': 'actor_interaction_trial_v1', 'config_sha256': bundle.manifest['config_sha256'],
            'asset_sha256': bundle.manifest['asset_sha256'], 'source_sha256': software['source_sha256'],
            'script_sha256': {p.name: digest(p) for p in scripts}, 'torch': str(torch.__version__),
            'reuse': reuse, 'settings': settings, 'budget_grid': grid,
            'validation_iterations': list(VALIDATION_ITERATIONS), 'split': 'validation',
            'initialization': 'fresh seed11; no warm start, no additional data or waiting predictor'}
    with probe_lock(output, resume, name='.interaction-trial.lock'):
        path = output/'run-plan.json'
        if path.exists():
            require(load_json(path) == plan, 'Trial inputs/settings/software changed; preserve this run and use a new output')
        else:
            require(not any(p.name != '.interaction-trial.lock' for p in output.iterdir()), 'Unknown existing output')
            write_manifest(path, plan)
        report = {'kind': 'actor_interaction_development_summary_v1', 'status': 'training', 'panel': bundle.raw['panel'],
                  'run_plan_sha256': digest(path), 'settings': settings, 'budget_grid': grid,
                  'baselines': baseline['comparisons'], 'fixed_mix': baseline['fixed_mix'], 'checkpoints': [],
                  'scope': 'one architecture intervention; paired development outcomes; no test or feedback-value claim'}
        write_readout(output, report)
        print(f"Fresh product-interaction PPO: {settings['iterations']} rounds × {len(settings['budgets'])} budgets × {settings['episodes_per_budget']} episodes; same seed11 and environment.", flush=True)
        checkpoint, previous, history = ppo_stage(bundle, output, pilot/'references.json', settings)
        report.update(status='partial', final_checkpoint=str(checkpoint.relative_to(output)),
                      ppo_episodes=previous['total_episodes'], ppo_chunks=previous['total_chunks'],
                      iteration_timing=[{k: r[k] for k in ('iteration', 'rollout_seconds', 'update_seconds', 'episodes', 'chunks')} for r in history])
        by_iteration = {r['iteration']: r for r in history}
        for iteration in VALIDATION_ITERATIONS:
            candidate = checkpoint_at(output, iteration, by_iteration[iteration])
            meta, _ = read_checkpoint(candidate)
            require(meta['settings'] == settings and meta['architecture']['actor_interaction'] == 'product', 'Wrong actor checkpoint')
            stage = output/f'validation-{iteration:06d}'
            rows, groups, seal = validation_stage(bundle, stage, candidate, settings)
            ResultRun(stage, references, allowed_splits={'validation'})
            points, responses = [], []
            for tick in grid['slider']['ticks']:
                beta = tick['budget_multiplier']
                selected = [r for r in rows if r['budget_multiplier'] == beta]
                chunks = [groups[(r['episode_id'], beta)] for r in selected]
                points.append(comparison(f'Product-PPO-{iteration}', selected, chunks, tick['budget_hours'], references))
                for row, episode_chunks in zip(selected, chunks):
                    probabilities = episode_chunks[0]['policy_probabilities']
                    responses.append({'iteration': iteration, 'episode_id': row['episode_id'], 'budget_multiplier': beta,
                                      'probabilities': probabilities,
                                      'entropy_nats': -sum(p*math.log(p) for p in probabilities if p > 0)})
            report['checkpoints'].append({'iteration': iteration, 'checkpoint_sha256': digest(candidate),
                                          'validation_files': seal, 'operating_points': points,
                                          'actor': response_summary(responses)})
            report['status'] = 'complete' if iteration == VALIDATION_ITERATIONS[-1] else 'partial'
            write_readout(output, report)
            print(f'Product-PPO validation iteration {iteration} complete.', flush=True)
        print(f'Interaction trial complete: {output}/interaction-summary.md', flush=True)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'pilot', 'main', 'diagnosis', 'output'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run(Bundle(Path(args.config)), args.pilot, args.main, args.diagnosis, args.output, args.resume)


if __name__ == '__main__':
    main()
