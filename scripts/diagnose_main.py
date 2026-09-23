"""Development diagnosis on a compute node: frozen actors and paired planners.

No training, test evaluation, budget changes or edits to existing run artifacts.
Kept outside carbon so the existing checkpoint/source bindings stay valid.
"""

import argparse
from collections import Counter, defaultdict
from copy import copy
from dataclasses import replace
import math
from pathlib import Path
from statistics import fmean
import time
import uuid

from carbon.common import digest, load_json, require
from carbon.config import Bundle
from carbon.dependence import audit_dependence
from carbon.environment import Environment, initial_replay
from carbon.learning import ActorCritic, input_tensors, torch
from carbon.policy_inputs import PolicyInputs, shared_inputs
from carbon.policy_runner import check_policy_inputs, configure_torch, read_checkpoint
from carbon.probe_resume import probe_lock
from carbon.results import ResultRun
from carbon.runner import provenance, run_planners, write_manifest, write_record
from carbon.waits import queue_contract
from main_train import reuse_pilot, summarize


METHODS = ('Plan-once', 'Rollout-MPC')
ITERATIONS = (16, 32, 48, 64)
PATHS = 256
SEED = 11
EPSILON = .05


def records(path):
    # Completed stages must have entirely valid JSONL, including their last row.
    import json
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]


def binding(directory, names):
    return {name: digest(Path(directory)/name) for name in names}


def checked_stage(directory, expected, names, execute, validate):
    """Resume whole small units; preserve interrupted units, reject changed seals."""
    directory = Path(directory)
    manifest = directory/'manifest.json'
    seal = directory/'stage-seal.json'
    if directory.exists():
        if seal.exists():
            require(load_json(seal) == binding(directory, names), f'Stage seal differs: {directory}')
        if manifest.exists():
            old = load_json(manifest)
            require(all(old.get(k) == v for k, v in expected.items()), f'Stage contract differs: {directory}')
        else:
            old = {}
        if old.get('status') != 'complete':
            require(not seal.exists(), f'Sealed stage is incomplete: {directory}')
            backup = directory.with_name(directory.name+'.interrupted-'+uuid.uuid4().hex)
            directory.rename(backup)
            print(f'Preserved incomplete stage: {backup}', flush=True)
    if not directory.exists():
        directory.parent.mkdir(parents=True, exist_ok=True)
        execute(directory)
    meta = load_json(manifest)
    require(meta.get('status') == 'complete' and all(meta.get(k) == v for k, v in expected.items()),
            f'Incomplete or incompatible stage: {directory}')
    result = validate(directory)
    hashes = binding(directory, names)
    if seal.exists():
        require(load_json(seal) == hashes, f'Stage seal differs: {directory}')
    else:
        write_manifest(seal, hashes)
    return result, hashes


def episode_groups(directory, expected_keys, split, budget=None):
    rows = records(directory/'episodes.jsonl')
    chunks = records(directory/'chunks.jsonl')
    keys = [(r['episode_id'], r['method']) for r in rows]
    require(len(keys) == len(expected_keys) and set(keys) == set(expected_keys), 'Unpaired or duplicate planner outcomes')
    groups = defaultdict(list)
    for c in chunks:
        key = c['episode_id'], c['method']
        require(key in expected_keys and c['split'] == split, 'Unexpected planner chunk')
        require(budget is None or c['budget_hours'] == budget, 'Planner chunk budget changed')
        groups[key].append(c)
    for row in rows:
        require(row['split'] == split and (budget is None or row['budget_hours'] == budget), 'Planner split/budget changed')
        group = groups[(row['episode_id'], row['method'])]
        require([c['chunk_id'] for c in group] == list(range(row['chunk_count'])), 'Missing or duplicate planner chunk')
        require(row['censor_flag'] or row['remaining_updates'] == 0, 'Completed planner left unfinished work')
    return rows, groups


def frozen_main(bundle, main, references, grid, pilot_binding, software):
    plan = load_json(main/'run-plan.json')
    report = load_json(main/'validation-summary.json')
    require(plan['config_sha256'] == bundle.manifest['config_sha256'] and
            plan['asset_sha256'] == bundle.manifest['asset_sha256'] and
            plan['source_sha256'] == software['source_sha256'] and
            plan['orchestrator_sha256'] == digest(Path(__file__).with_name('main_train.py')) and
            plan['torch'] == str(torch.__version__) and plan['pilot_artifacts'] == pilot_binding,
            'Main run inputs/software differ')
    settings = plan['settings']
    require(settings['seed'] == SEED and settings['budgets'] == grid['budget_multipliers'] and
            settings['wait_features'] == 'none' and settings['decision_mode'] == 'feedback' and
            settings['objective'] == 'robust' and settings['epsilon'] == EPSILON,
            'Expected the frozen seed11 robust feedback run')
    require(report['status'] == 'complete' and report['settings'] == settings and
            report['budget_grid'] == grid and plan['validation_iterations'] == list(ITERATIONS) and
            [c['iteration'] for c in report['checkpoints']] == list(ITERATIONS),
            'All four predeclared validation checkpoints are required')
    encoder = PolicyInputs(bundle, None, references, 'window', 'none')
    candidates = []
    cohort = {e.episode_id for e in bundle.episodes if e.split == 'validation'}
    expected = {(eid, beta) for eid in cohort for beta in settings['budgets']}
    for item in report['checkpoints']:
        iteration = item['iteration']
        matches = [p for p in main.glob(f'ppo-attempt-*/checkpoint-{iteration:06d}.json')
                   if digest(p) == item['checkpoint_sha256']]
        require(len(matches) == 1, f'Missing/ambiguous checkpoint {iteration}')
        path = matches[0]
        meta, state = read_checkpoint(path)
        check_policy_inputs(meta, encoder, None, bundle)
        require(meta['iteration'] == iteration and meta['settings'] == settings and
                meta['software']['source_sha256'] == software['source_sha256'] and
                meta['software']['torch'] == str(torch.__version__), 'Frozen checkpoint changed')
        stage = main/f'validation-{iteration:06d}'
        require(binding(stage, item['validation_files']) == item['validation_files'], 'Frozen validation hash mismatch')
        manifest = load_json(stage/'manifest.json')
        require(manifest['status'] == 'complete' and manifest['split'] == 'validation' and
                manifest['checkpoint_sha256'] == digest(path) and manifest['budget_multipliers'] == settings['budgets'],
                'Frozen validation contract differs')
        rows = ResultRun(stage, references, allowed_splits={'validation'}).rows
        chunks = records(stage/'chunks.jsonl')
        keys = [(r['episode_id'], r['budget_multiplier']) for r in rows]
        require(len(keys) == len(expected) and set(keys) == expected, 'Frozen validation cohort differs')
        groups = defaultdict(list)
        for chunk in chunks:
            key = chunk['episode_id'], chunk['budget_multiplier']
            require(key in expected and chunk['split'] == 'validation', 'Unexpected frozen validation chunk')
            groups[key].append(chunk)
        for row in rows:
            group = groups[(row['episode_id'], row['budget_multiplier'])]
            require(row['split'] == 'validation' and row['policy_checkpoint'] == digest(path) and
                    row['budget_hours'] == row['budget_multiplier']*references['time_reference_hours'] and
                    [c['chunk_id'] for c in group] == list(range(row['chunk_count'])), 'Invalid frozen validation row')
        model = ActorCritic(len(encoder.global_names), len(encoder.action_names))
        model.load_state_dict(state['model']); model.eval()
        candidates.append({'iteration': iteration, 'path': path, 'model': model, 'rows': rows,
                           'groups': groups, 'validation_files': item['validation_files']})
    return encoder, candidates


def response_summary(rows):
    points, effects = [], []
    for iteration in sorted({r['iteration'] for r in rows}):
        selected = [r for r in rows if r['iteration'] == iteration]
        budgets = sorted({r['budget_multiplier'] for r in selected})
        for beta in budgets:
            group = [r for r in selected if r['budget_multiplier'] == beta]
            points.append({'iteration': iteration, 'budget_multiplier': beta, 'arrivals': len(group),
                           'mean_probabilities': [fmean(r['probabilities'][i] for r in group) for i in range(len(group[0]['probabilities']))],
                           'mean_entropy_nats': fmean(r['entropy_nats'] for r in group)})
        pairs = defaultdict(dict)
        for row in selected:
            require(row['budget_multiplier'] not in pairs[row['episode_id']], 'Duplicate actor response')
            pairs[row['episode_id']][row['budget_multiplier']] = row['probabilities']
        distances = []
        for probabilities in pairs.values():
            require(set(probabilities) == set(budgets), 'Unpaired actor budgets')
            distances.append(max(sum(abs(a-b) for a, b in zip(probabilities[x], probabilities[y]))/2
                                 for x in budgets for y in budgets))
        effects.append({'iteration': iteration, 'mean_max_pairwise_budget_TV': fmean(distances),
                        'max_pairwise_budget_TV': max(distances)})
    return {'operating_points': points, 'budget_response': effects}


def actor_stage(bundle, output, encoder, candidates, grid):
    expected = {'kind': 'frozen_initial_actor_diagnosis_v1', 'split': 'validation'}
    episodes = sorted((e for e in bundle.episodes if e.split == 'validation'), key=lambda e: e.arrival)
    expected_keys = {(c['iteration'], e.episode_id, b) for c in candidates for e in episodes for b in grid['budget_multipliers']}

    def execute(directory):
        directory.mkdir()
        write_manifest(directory/'manifest.json', {**expected, 'status': 'running'})
        with (directory/'responses.jsonl').open('x', encoding='utf-8') as stream:
            for original in episodes:
                base = initial_replay(bundle, original)
                for tick in grid['slider']['ticks']:
                    episode = replace(original, budget_hours=tick['budget_hours'])
                    encoded, info = encoder.encode(Environment(bundle, episode, base, 'actor-diagnosis').observe())
                    for candidate in candidates:
                        old = candidate['groups'][(episode.episode_id, tick['budget_multiplier'])][0]
                        require(info['policy_input_sha256'] == old['policy_input_sha256'], 'Reconstructed actor input differs from evaluated input')
                        with torch.no_grad():
                            distribution, _ = candidate['model'](*input_tensors([encoded]))
                            probabilities = distribution.probs[0].tolist()
                        require(len(probabilities) == len(old['policy_probabilities']) and
                                all(math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-7)
                                    for a, b in zip(probabilities, old['policy_probabilities'])), 'Frozen actor probabilities changed')
                        write_record(stream, {'iteration': candidate['iteration'], 'episode_id': episode.episode_id,
                                              'budget_multiplier': tick['budget_multiplier'], 'budget_hours': tick['budget_hours'],
                                              'nodes': encoder.nodes, 'probabilities': probabilities,
                                              'entropy_nats': -sum(p*math.log(p) for p in probabilities if p > 0),
                                              'policy_input_sha256': info['policy_input_sha256']})
                print(f'Actor input/probability check: {original.episode_id}', flush=True)
        write_manifest(directory/'manifest.json', {**expected, 'status': 'complete', 'rows': len(expected_keys)})

    def validate(directory):
        rows = records(directory/'responses.jsonl')
        keys = [(r['iteration'], r['episode_id'], r['budget_multiplier']) for r in rows]
        require(len(keys) == len(expected_keys) and set(keys) == expected_keys, 'Incomplete actor diagnosis')
        return response_summary(rows)

    return checked_stage(output/'actor', expected, ('manifest.json', 'responses.jsonl'), execute, validate)


def planner_stage(bundle, directory, episode, tick, predictor, references, software):
    require(episode.split == 'validation', 'Planner diagnosis only accepts validation arrivals')
    expected = {**bundle.manifest, 'methods': list(METHODS), 'seed': SEED, 'planning_paths': PATHS,
                'planner_internal_miss_tolerance': EPSILON, 'budget_multiplier': tick['budget_multiplier'],
                'selected_episode_ids': [episode.episode_id], 'selected_splits': ['validation'],
                'predictor_sha256': digest(predictor), 'references_sha256': digest(references)}

    def execute(path):
        # The original manifest still declares the full input cohort; this stage
        # explicitly selects just one arrival. Propagate only the background cache.
        subset = copy(bundle)
        subset.episodes = [episode]
        run_planners(subset, path, predictor, references, split='validation', paths=PATHS,
                     miss_tolerance=EPSILON, seed=SEED, methods=METHODS, budget_multiplier=tick['budget_multiplier'])
        if hasattr(subset, '_background_prefix'):
            bundle._background_prefix = subset._background_prefix

    def validate(path):
        meta = load_json(path/'manifest.json')
        require(meta['software']['source_sha256'] == software['source_sha256'] and
                meta['completed_episode_methods'] == len(METHODS), 'Planner implementation/count differs')
        ResultRun(path, load_json(references), allowed_splits={'validation'})
        return episode_groups(path, {(episode.episode_id, m) for m in METHODS}, 'validation', tick['budget_hours'])

    return checked_stage(directory, expected, ('manifest.json', 'episodes.jsonl', 'chunks.jsonl'), execute, validate)


def comparison(method, rows, groups, budget, references):
    sequences = [[c['selected_nodes'] for c in group] for group in groups]
    result = summarize(rows, budget, references, sequences)
    result['sequence_counts'] = dict(Counter('->'.join(map(str, s)) for s in sequences))
    result['mean_chunks'] = fmean(r['chunk_count'] for r in rows)
    result['p95_tat_hours'] = (sorted(r['tat_hours'] for r in rows)[math.ceil(.95*len(rows))-1]
                               if result['incomplete'] == 0 else None)
    if method in METHODS:
        flat = [c for group in groups for c in group]
        result['planning_seconds'] = sum(c['decision_planning_seconds'] for c in flat)
        result['replanning_decisions'] = sum(c['replanned'] for c in flat)
        result['fallback_episode_fraction'] = fmean(any(c['planner_fallback_reason'] for c in group) for group in groups)
        result['mean_initial_predicted_miss'] = fmean(group[0]['estimated_miss'] for group in groups)
    return {'method': method, 'budget_hours': budget, 'summary': result}


def write_report(output, report):
    write_manifest(output/'diagnosis-summary.json', report)
    lines = ['# Frontera–7B development diagnosis', '',
             '全部四档预算；固定原策略、256 planning paths、5% internal miss threshold、seed11。',
             '复用旧 fixed/PPO 结果；无训练、test、选优或总体可行性认证。',
             'C 为两端点中较大的归一化平均 modeled carbon；删失保留分母，整体成本留空。', '',
             '| 方法 | D(h) | 完成/总数 | TAT(h) | C | miss 下界–上界 | 换规模 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    def fmt(x):
        return '—' if x is None else f'{x:.3f}'
    for point in report['comparisons']:
        s = point['summary']; lo, hi = s['miss_rate_bounds']
        lines.append(f"| {point['method']} | {point['budget_hours']:.2f} | {s['completed']}/{s['outcomes']} | {fmt(s['mean_tat_hours'])} | {fmt(s['worst_normalized_carbon'])} | {lo:.1%}–{hi:.1%} | {s['observed_scale_change_fraction']:.1%} |")
    lines += ['', '## Fixed-Mix（原 validation LP 拟合期望，非独立 test）', '']
    for m in report['fixed_mix']:
        lines.append(f"- D={m['budget_hours']:.2f}h：经验可行={m['empirical_feasible']}，C={fmt(m['estimated_objective'])}，weights={m['weights']}")
    lines += ['', '## 同一初始状态下的预算响应', '',
              'TV 是两档预算动作分布的总变差距离；0 表示完全相同，1 表示不重叠。',
              '每个到达取四档中最大 TV 后汇总；这是输入敏感性诊断，不证明反馈收益。', '']
    for row in report.get('actor', {}).get('budget_response', []):
        lines.append(f"- checkpoint {row['iteration']}：平均最大 TV={row['mean_max_pairwise_budget_TV']:.8f}，最大={row['max_pairwise_budget_TV']:.8f}")
    lines += ['', f"状态：{report['status']}；已完成 planner 到达×预算单元：{report['completed_planner_units']}/{report['expected_planner_units']}。", '',
              '完整预测 miss、fallback、序列计数、耗时和输入哈希见 JSON；运行未完成时不输出部分 planner cohort 的均值。',
              'interpretation：只赢 fixed 不证明反馈；MPC 与 Plan-once 相近也可能来自等待模型误差。', '']
    tmp = output/'diagnosis-summary.md.tmp'
    tmp.write_text('\n'.join(lines), encoding='utf-8'); tmp.replace(output/'diagnosis-summary.md')


def run(bundle, pilot, main, e1, output, resume=False):
    require(bundle.raw['purpose'] in {'development', 'synthetic'}, 'Diagnosis is development only')
    pilot, main, e1, output = map(Path, (pilot, main, e1, output))
    configure_torch(SEED, 1)
    references, grid, fixed, pilot_binding = reuse_pilot(bundle, pilot)
    software = provenance(bundle.root)
    encoder, candidates = frozen_main(bundle, main, references, grid, pilot_binding, software)
    predictor = e1/'model/model.json'
    shared_inputs(bundle, predictor, references, True)
    model_meta = load_json(e1/'model/manifest.json')
    require(model_meta['status'] == 'complete' and model_meta['model_sha256'] == digest(predictor), 'E1 model seal differs')
    probe_binding = {}
    for split in ('train', 'validation'):
        directory = e1/f'{split}-probes'
        meta = load_json(directory/'manifest.json')
        require(meta['status'] == 'complete' and meta['probe_split'] == split and
                meta['probes_sha256'] == digest(directory/'probes.jsonl') and
                queue_contract(meta) == queue_contract(bundle.manifest), 'E1 probes incomplete or changed')
        probe_binding[split] = binding(directory, ('manifest.json', 'probes.jsonl'))
    episodes = sorted((e for e in bundle.episodes if e.split == 'validation'), key=lambda e: e.arrival)
    require(episodes, 'No validation arrivals')
    contract = {'kind': 'main_diagnosis_v1', 'config_sha256': bundle.manifest['config_sha256'],
                'asset_sha256': bundle.manifest['asset_sha256'], 'source_sha256': software['source_sha256'],
                'orchestrator_sha256': digest(Path(__file__)), 'main_train_sha256': digest(Path(__file__).with_name('main_train.py')),
                'torch': str(torch.__version__), 'pilot': pilot_binding, 'main_summary_sha256': digest(main/'validation-summary.json'),
                'main_plan_sha256': digest(main/'run-plan.json'), 'predictor_sha256': digest(predictor), 'probes': probe_binding,
                'checkpoint_sha256': {str(c['iteration']): digest(c['path']) for c in candidates},
                'budget_grid': grid, 'methods': list(METHODS), 'planning_paths': PATHS, 'internal_miss_tolerance': EPSILON,
                'seed': SEED, 'split': 'validation', 'episode_ids': [e.episode_id for e in episodes]}
    with probe_lock(output, resume, name='.diagnosis.lock'):
        path = output/'run-plan.json'
        if path.exists():
            require(load_json(path) == contract, 'Diagnosis inputs/settings/software changed; use a new output directory')
        else:
            require(not any(p.name != '.diagnosis.lock' for p in output.iterdir()), 'Unknown existing output')
            write_manifest(path, contract)
        report = {'kind': 'main_diagnosis_summary_v1', 'status': 'partial', 'panel': bundle.raw['panel'],
                  'scope': 'development validation only; no selection, retraining or test outcomes',
                  'run_plan_sha256': digest(path), 'comparisons': [], 'fixed_mix': load_json(main/'validation-summary.json')['fixed_mix'],
                  'completed_planner_units': 0, 'expected_planner_units': len(episodes)*len(grid['budget_multipliers']),
                  'stage_files': {}}
        for tick in grid['slider']['ticks']:
            budget, beta = tick['budget_hours'], tick['budget_multiplier']
            for method, group in fixed.items():
                rows = list(group.values()); node = int(method.split('-')[1])
                groups = [[{'selected_nodes': node}]*r['chunk_count'] for r in rows]
                report['comparisons'].append(comparison(method, rows, groups, budget, references))
            for c in candidates:
                rows = [r for r in c['rows'] if r['budget_multiplier'] == beta]
                groups = [c['groups'][(r['episode_id'], beta)] for r in rows]
                report['comparisons'].append(comparison(f"PPO-{c['iteration']}", rows, groups, budget, references))
        write_report(output, report)
        report['actor'], report['stage_files']['actor'] = actor_stage(bundle, output, encoder, candidates, grid)
        write_report(output, report)
        planner_directories = []
        for tick_index, tick in enumerate(grid['slider']['ticks']):
            by_method = {method: ([], []) for method in METHODS}
            for arrival_index, episode in enumerate(episodes):
                stage = output/f'planners/budget-{tick_index:02d}/arrival-{arrival_index:04d}'
                begun = time.monotonic()
                (rows, groups), hashes = planner_stage(bundle, stage, episode, tick, predictor, pilot/'references.json', software)
                for row in rows:
                    by_method[row['method']][0].append(row)
                    by_method[row['method']][1].append(groups[(row['episode_id'], row['method'])])
                planner_directories.append(stage)
                report['stage_files'][str(stage.relative_to(output))] = hashes
                report['completed_planner_units'] += 1
                write_report(output, report)
                print(f"Planner unit {report['completed_planner_units']}/{report['expected_planner_units']} checked; {time.monotonic()-begun:.1f}s", flush=True)
            for method, (rows, groups) in by_method.items():
                report['comparisons'].append(comparison(method, rows, groups, tick['budget_hours'], references))
            write_report(output, report)
        # Fixed runs plus the final frozen policy cover episode lengths without
        # treating four correlated checkpoints as extra independent evidence.
        inputs = [pilot/'fixed-train', pilot/'fixed-validation', main/'validation-000064', *planner_directories]
        expected = {'kind': 'development_dependence_diagnostics', 'purpose': bundle.raw['purpose']}
        def execute_dependence(directory):
            audit_dependence([e1/'train-probes', e1/'validation-probes'], directory, inputs)
        def validate_dependence(directory):
            meta = load_json(directory/'manifest.json')
            require(meta['audit_sha256'] == digest(directory/'audit.json') and
                    meta['series_sha256'] == digest(directory/'series.jsonl'), 'Dependence hashes differ')
            return load_json(directory/'audit.json')['episode_spans']
        report['episode_spans'], report['stage_files']['dependence'] = checked_stage(
            output/'dependence', expected, ('manifest.json', 'audit.json', 'series.jsonl', 'report.md'),
            execute_dependence, validate_dependence)
        report['status'] = 'complete'
        write_report(output, report)
        print(f'Diagnosis complete: {output}/diagnosis-summary.md', flush=True)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--pilot', required=True)
    parser.add_argument('--main', required=True)
    parser.add_argument('--e1', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run(Bundle(Path(args.config)), args.pilot, args.main, args.e1, args.output, args.resume)


if __name__ == '__main__':
    main()
