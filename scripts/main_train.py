"""Bounded development training and paired validation; no test-cohort access.

Kept outside src/carbon so the completed pilot's package hashes remain valid.
"""

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import fmean
import uuid

from carbon.baselines import check_references, fit_fixed_mix
from carbon.common import digest, load_json, require
from carbon.config import Bundle
from carbon.main_pilot import make_budget_grid, ppo_stage, read_jsonl_prefix, verify_fixed
from carbon.policy_runner import evaluate_policy, read_checkpoint, settings_for, torch
from carbon.probe_resume import probe_lock
from carbon.runner import provenance, write_manifest

ITERATIONS = 64
EPISODES_PER_BUDGET = 16
VALIDATION_ITERATIONS = (16, 32, 48, 64)


def sealed_files(directory, names, seal_name, create=False):
    binding = {name: digest(directory/name) for name in names}
    seal = directory/seal_name
    if create and not seal.exists():
        write_manifest(seal, binding)
    require(seal.exists() and load_json(seal) == binding, f'Artifact seal differs: {directory}')
    return binding


def reuse_pilot(bundle, pilot):
    summary = load_json(pilot/'pilot-summary.json')
    require(summary['status'] == 'complete', 'A complete main pilot is required')
    contract = load_json(pilot/'pipeline-contract.json')
    require(contract['config_sha256'] == bundle.manifest['config_sha256'] and
            contract['asset_sha256'] == bundle.manifest['asset_sha256'] and
            contract['source_sha256'] == provenance(bundle.root)['source_sha256'],
            'Pilot configuration, data or package implementation changed')
    groups, bindings = {}, {}
    for split in ('train', 'validation'):
        directory = pilot/f'fixed-{split}'
        bindings[split] = sealed_files(directory, ('manifest.json', 'episodes.jsonl', 'chunks.jsonl'), 'stage-seal.json')
        _, groups[split] = verify_fixed(bundle, directory, split)
    references = load_json(pilot/'references.json')
    check_references(references, bundle.manifest)
    require(load_json(pilot/'references-seal.json') == {'references_sha256': digest(pilot/'references.json')},
            'Pilot references changed')
    require(references['source_episodes_sha256'] == digest(pilot/'fixed-train/episodes.jsonl'),
            'References use different fixed outcomes')
    grid = make_budget_grid(references, groups['train'])
    require(load_json(pilot/'budget-grid.json') == grid, 'Pilot budget grid differs from training data')
    bindings.update(references=digest(pilot/'references.json'), grid=digest(pilot/'budget-grid.json'))
    return references, grid, groups['validation'], bindings


def checkpoint_at(output, iteration, record):
    matches = [p for p in sorted(output.glob(f'ppo-attempt-*/checkpoint-{iteration:06d}.json'))
               if digest(p) == record['checkpoint_sha256']]
    require(matches, f'Missing committed checkpoint at iteration {iteration}')
    return matches[-1]


def validation_stage(bundle, directory, checkpoint, settings):
    if directory.exists():
        meta = load_json(directory/'manifest.json') if (directory/'manifest.json').exists() else {}
        if meta.get('status') != 'complete':
            backup = directory.with_name(directory.name+'.interrupted-'+uuid.uuid4().hex)
            directory.rename(backup)
            print(f'Preserved incomplete validation: {backup}; restarting this checkpoint evaluation.', flush=True)
    if not directory.exists():
        evaluate_policy(bundle, directory, None, checkpoint, split='validation')
    meta = load_json(directory/'manifest.json')
    require(meta['status'] == 'complete' and meta['split'] == 'validation' and
            meta['checkpoint_sha256'] == digest(checkpoint) and
            meta['training_seed'] == settings['seed'] and meta['sampling_seed'] == settings['seed'] and
            meta['budget_multipliers'] == settings['budgets'] and meta['predictor_sha256'] is None and
            meta['decision_mode'] == 'feedback' and meta['policy_rule'] == 'categorical_sampling',
            'Validation checkpoint/settings differ')
    require(all(meta.get(k) == v for k, v in bundle.manifest.items()) and
            meta['software']['source_sha256'] == provenance(bundle.root)['source_sha256'] and
            meta['software']['torch'] == str(torch.__version__), 'Validation inputs/software differ')
    binding = sealed_files(directory, ('manifest.json', 'episodes.jsonl', 'chunks.jsonl', 'plans.jsonl'), 'stage-seal.json', create=True)
    rows = read_jsonl_prefix(directory/'episodes.jsonl')
    chunks = read_jsonl_prefix(directory/'chunks.jsonl')
    expected = {(e.episode_id, beta) for e in bundle.episodes if e.split == 'validation' for beta in settings['budgets']}
    keys = [(r['episode_id'], r['budget_multiplier']) for r in rows]
    require(set(keys) == expected and len(keys) == len(expected) == meta['completed_episode_methods'],
            'Incomplete or duplicate validation cohort')
    require(all(r['split'] == 'validation' and r['policy_checkpoint'] == digest(checkpoint) for r in rows),
            'Validation records belong to another checkpoint/split')
    chunk_groups = defaultdict(list)
    for chunk in chunks:
        key = (chunk['episode_id'], chunk['budget_multiplier'])
        require(key in expected, 'Unexpected validation chunk')
        chunk_groups[key].append(chunk)
    for row in rows:
        group = chunk_groups[(row['episode_id'], row['budget_multiplier'])]
        require([c['chunk_id'] for c in group] == list(range(row['chunk_count'])), 'Missing/duplicate/out-of-order chunk')
    return rows, chunk_groups, binding


def summarize(rows, budget, references, sequences):
    """Censored outcomes stay in denominators; never treat partial cost as total."""
    require(rows, 'Empty comparison cohort')
    completed = [r['final_status'] == 'completed' and not r['censor_flag'] for r in rows]
    misses = [bool(r['tat_hours'] > budget) if done else
              (True if r['observed_elapsed_hours'] >= budget else None)
              for r, done in zip(rows, completed)]
    count = len(rows)
    known_misses, unknown = sum(v is True for v in misses), sum(v is None for v in misses)
    # Repeated allocations at one scale do not constitute dynamic scaling.
    changed = sum(any(a != b for a, b in zip(seq, seq[1:])) for seq in sequences)
    down = sum(any(a > b for a, b in zip(seq, seq[1:])) for seq in sequences)
    up = sum(any(a < b for a, b in zip(seq, seq[1:])) for seq in sequences)
    carbon = {rho: fmean(r['carbon_g_per_kappa'][rho] for r in rows) for rho in references['carbon_reference_g_per_kappa']} if all(completed) else None
    return {'outcomes': count, 'completed': sum(completed), 'incomplete': count-sum(completed),
            'mean_tat_hours': fmean(r['tat_hours'] for r in rows) if all(completed) else None,
            'mean_nodehours': fmean(r['nodehours'] for r in rows) if all(completed) else None,
            'mean_modeled_carbon_g_per_kappa': carbon,
            'worst_normalized_carbon': max(carbon[rho]/ref for rho, ref in references['carbon_reference_g_per_kappa'].items()) if carbon else None,
            'deadline_misses': known_misses, 'deadline_unknown': unknown,
            'miss_rate_bounds': [known_misses/count, (known_misses+unknown)/count],
            'empirical_target_met': all(completed) and known_misses/count <= .05,
            'single_chunk_fraction': sum(r['chunk_count'] == 1 for r in rows)/count,
            'observed_scale_change_fraction': changed/count,
            'observed_scale_down_fraction': down/count, 'observed_scale_up_fraction': up/count}


def write_readout(output, report):
    write_manifest(output/'validation-summary.json', report)
    lines = ['# 主策略 development validation', '',
             '同一批 validation 到达时间；全部预定 checkpoint 均报告，不自动选优。',
             'C 为两功率端点中较大的归一化平均估算碳成本；未知功耗系数未测量。',
             '5% 是经验 miss 门槛；这里没有人口可行性保证、置信区间或 test 结论。',
             '存在未完成任务时整体平均 TAT/C 留空，miss 上下界保留全部任务。', '',
             '| 方法/轮次 | D (h) | 完成/总数 | TAT (h) | C | miss 区间 | 单 chunk | 换规模 | 降规模 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    def number(value):
        return '—' if value is None else f'{value:.3f}'
    entries = report['fixed'] + [item for stage in report['checkpoints'] for item in stage['operating_points']]
    for item in entries:
        s = item['summary']; lo, hi = s['miss_rate_bounds']
        lines.append(f"| {item['method']} | {item['budget_hours']:.2f} | {s['completed']}/{s['outcomes']} | {number(s['mean_tat_hours'])} | {number(s['worst_normalized_carbon'])} | {lo:.1%}–{hi:.1%} | {s['single_chunk_fraction']:.1%} | {s['observed_scale_change_fraction']:.1%} | {s['observed_scale_down_fraction']:.1%} |")
    lines += ['', '## Fixed-Mix 参考', '',
              '直接对已有 fixed validation 结果解 LP，无新增回放。下面是拟合该 validation 集的期望值，',
              '用于检查仅随机选择固定规模是否已足够；不冒充独立测试表现。', '',
              '| D (h) | LP 经验可行 | 期望 C | 固定规模概率 |', '|---|---|---:|---|']
    for mix in report['fixed_mix']:
        lines.append(f"| {mix['budget_hours']:.2f} | {mix['empirical_feasible']} | {number(mix['estimated_objective'])} | {mix['weights']} |")
    lines += ['', f"状态：{report['status']}；已评估轮次：{[s['iteration'] for s in report['checkpoints']]}", '']
    path = output/'validation-summary.md'
    temporary = path.with_suffix('.tmp'); temporary.write_text('\n'.join(lines), encoding='utf-8'); temporary.replace(path)


def run(bundle, pilot, output, seed=11, resume=False):
    require(bundle.raw['purpose'] in {'development', 'synthetic'}, 'This runner is development only')
    pilot, output = Path(pilot), Path(output)
    references, grid, fixed, pilot_binding = reuse_pilot(bundle, pilot)
    settings = settings_for(ITERATIONS, episodes_per_budget=EPISODES_PER_BUDGET,
                            minibatch_episodes=16, seed=seed, wait_features='none',
                            decision_mode='feedback', budgets=grid['budget_multipliers'])
    contract = {'kind': 'main_development_train_v1', 'config_sha256': bundle.manifest['config_sha256'],
                'asset_sha256': bundle.manifest['asset_sha256'], 'source_sha256': provenance(bundle.root)['source_sha256'],
                'orchestrator_sha256': digest(Path(__file__)), 'torch': str(torch.__version__),
                'pilot_artifacts': pilot_binding, 'settings': settings,
                'validation_iterations': list(VALIDATION_ITERATIONS), 'split': 'validation'}
    # Validate LP availability before beginning the potentially long training stage.
    fixed_mix = [fit_fixed_mix(pilot/'fixed-validation', references, tick['budget_hours']) for tick in grid['slider']['ticks']]
    with probe_lock(output, resume, name='.main-train.lock'):
        path = output/'run-plan.json'
        if path.exists():
            require(load_json(path) == contract, 'Run inputs/settings/software changed; preserve this run and use a new directory')
        else:
            require(not any(p.name != '.main-train.lock' for p in output.iterdir()), 'Unknown existing output')
            write_manifest(path, contract)
        print(f"Reusing complete fixed cohorts and exact budget-grid from {pilot}", flush=True)
        print(f"Main PPO: {ITERATIONS} iterations x {len(settings['budgets'])} budgets x {EPISODES_PER_BUDGET} arrivals = {ITERATIONS*len(settings['budgets'])*EPISODES_PER_BUDGET} episodes; seed={seed}; CPU; no wait predictor.", flush=True)
        checkpoint, previous, history = ppo_stage(bundle, output, pilot/'references.json', settings)
        report = {'kind': 'main_development_validation_v1', 'status': 'partial',
                  'scope': 'paired development comparisons; all declared checkpoints; no selection, confidence interval or test claim',
                  'panel': bundle.raw['panel'], 'seed': seed, 'budget_grid': grid, 'settings': settings,
                  'final_checkpoint': str(checkpoint.relative_to(output)), 'ppo_episodes': previous['total_episodes'],
                  'ppo_chunks': previous['total_chunks'], 'fixed': [], 'fixed_mix': fixed_mix, 'checkpoints': [],
                  'iteration_timing': [{k: r[k] for k in ('iteration', 'rollout_seconds', 'update_seconds', 'episodes', 'chunks')} for r in history]}
        for tick in grid['slider']['ticks']:
            for method, group in fixed.items():
                rows = list(group.values()); node = int(method.split('-')[1])
                report['fixed'].append({'method': method, 'budget_hours': tick['budget_hours'],
                                        'summary': summarize(rows, tick['budget_hours'], references, [[node]*r['chunk_count'] for r in rows])})
        by_iteration = {r['iteration']: r for r in history}
        for iteration in VALIDATION_ITERATIONS:
            candidate = checkpoint_at(output, iteration, by_iteration[iteration])
            metadata, _ = read_checkpoint(candidate)
            require(metadata['settings'] == settings and metadata['iteration'] == iteration, 'Candidate settings differ')
            rows, chunk_groups, binding = validation_stage(bundle, output/f'validation-{iteration:06d}', candidate, settings)
            points = []
            for tick, beta in zip(grid['slider']['ticks'], settings['budgets']):
                selected = [r for r in rows if r['budget_multiplier'] == beta]
                require(all(r['budget_hours'] == tick['budget_hours'] for r in selected), 'Validation budget hours differ')
                sequences = [[c['selected_nodes'] for c in chunk_groups[(r['episode_id'], beta)]] for r in selected]
                points.append({'method': f'PPO-{iteration}', 'budget_hours': tick['budget_hours'],
                               'summary': summarize(selected, tick['budget_hours'], references, sequences)})
            report['checkpoints'].append({'iteration': iteration, 'checkpoint_sha256': digest(candidate),
                                          'validation_files': binding, 'operating_points': points})
            report['status'] = 'complete' if iteration == VALIDATION_ITERATIONS[-1] else 'partial'
            write_readout(output, report)
            print(f'Validation iteration {iteration} complete; readout: {output}/validation-summary.md', flush=True)
        print(f'Main development run complete: {output}; no test cohort evaluated.', flush=True)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--pilot', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, choices=(11, 23, 37), default=11)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run(Bundle(Path(args.config)), Path(args.pilot), Path(args.output), args.seed, args.resume)


if __name__ == '__main__':
    main()
