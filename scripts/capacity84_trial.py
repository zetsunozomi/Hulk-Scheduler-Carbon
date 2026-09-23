"""Fresh Frontera C84 development run; unchanged learning and chunk rules.

Stages rebuild fixed references, E1 and both planners before matched PPO and
four validation checkpoints. C128 artifacts are never used as C84 outcomes.
"""

import argparse
from collections import Counter
from copy import deepcopy
import math
from pathlib import Path
from statistics import fmean
import uuid

from carbon.baselines import check_references, fit_fixed_mix, make_references, weighted_quantile
from carbon.common import digest, load_json, require
from carbon.config import Bundle
from carbon.main_pilot import fixed_stage, make_budget_grid, ppo_stage
from carbon.policy_runner import read_checkpoint, settings_for, torch
from carbon.probe_resume import probe_lock
from carbon.results import ResultRun
from carbon.runner import provenance, write_manifest
from carbon.wait_pipeline import run_wait_pipeline
from diagnose_main import comparison, planner_stage, records, response_summary
from main_train import checkpoint_at, sealed_files
from main_train import validation_stage
from capacity84_report import write_readout


ITERATIONS = 64
EPISODES_PER_BUDGET = 16
VALIDATION_ITERATIONS = (16, 32, 48, 64)
PROBE_INTERVAL_SECONDS = 21600
NODES = [4, 16, 64]
SCRIPTS = ('capacity84_trial.py', 'capacity84_report.py', 'main_train.py', 'diagnose_main.py')


def capacity84_config(original):
    config = deepcopy(original)
    config['panel'] = 'AMSP-7B-frontera-C84'
    config['cluster'].update(
        name='Frontera-RTX-capacity-84-AMSP-target-scenario',
        partition='frontera-rtx-demand-homogeneous-nodes', nodes=84, allowed_nodes=NODES,
        provenance='84-node Frontera RTX resource pool per data/README.md and author instruction; historical demand retains its node counts. Target throughput uses published AMSP A800 profiles, not measured RTX performance.')
    config['workload']['profiles'] = {k: v for k, v in config['workload']['profiles'].items() if int(k) <= 84}
    return config


def validate_scope(bundle):
    c = bundle.raw
    require(c['purpose'] in {'development', 'synthetic'}, 'C84 trial is development only')
    require(c['cluster']['nodes'] == 84 and c['cluster']['allowed_nodes'] == NODES and
            set(c['workload']['profiles']) == set(map(str, NODES)), 'Expected C84 and exactly 4/16/64 actions')
    if c['purpose'] != 'synthetic':
        original = load_json(bundle.root/'configs/amsp-frontera-7b.development.json')
        require(c == capacity84_config(original), 'Only capacity, available actions and scenario labels may change')


def settings(budgets=None):
    options = dict(episodes_per_budget=EPISODES_PER_BUDGET, minibatch_episodes=16,
                   seed=11, wait_features='none', decision_mode='feedback',
                   forecast_mode='window', objective='robust',
                   actor_interaction='product', actor_budget_mode='independent')
    if budgets is not None:
        options['budgets'] = budgets
    return settings_for(ITERATIONS, **options)


def queue_stage(bundle, output):
    path = output/'e1'
    run_wait_pipeline(bundle, path, interval_seconds=PROBE_INTERVAL_SECONDS,
                      resume=path.exists(), e1=True)
    names = ('train-probes/manifest.json', 'train-probes/probes.jsonl',
             'validation-probes/manifest.json', 'validation-probes/probes.jsonl',
             'model/manifest.json', 'model/model.json',
             'validation-diagnostics/manifest.json', 'validation-diagnostics/metrics.json',
             'validation-diagnostics/predictions.jsonl', 'dependence-queue/manifest.json',
             'dependence-queue/audit.json', 'dependence-queue/series.jsonl')
    binding = sealed_files(path, names, 'stage-seal.json', create=True)
    summary = {}
    for split in ('train', 'validation'):
        meta = load_json(path/f'{split}-probes/manifest.json')
        require(meta['status'] == 'complete' and meta['config_sha256'] == bundle.manifest['config_sha256'],
                'Queue stage config/completion differs')
        summary[split] = {k: meta[k] for k in ('rows', 'labeled_rows', 'censored_rows',
                                               'queue_summary', 'wait_summary_by_request')}
    return binding, summary


def fixed_inputs(bundle, output):
    train = fixed_stage(bundle, output/'fixed-train', 'train')
    refs_path = output/'references.json'
    if not refs_path.exists():
        pending = output/f'.references-{uuid.uuid4().hex}.json'
        make_references(output/'fixed-train', pending)
        pending.replace(refs_path)
    refs = load_json(refs_path)
    check_references(refs, bundle.manifest)
    require(refs['source_episodes_sha256'] == digest(output/'fixed-train/episodes.jsonl'),
            'References do not bind current C84 training outcomes')
    sealed_files(output, ('references.json',), 'references-seal.json', create=True)
    grid = make_budget_grid(refs, train)
    grid_path = output/'budget-grid.json'
    if grid_path.exists():
        require(load_json(grid_path) == grid, 'Budget grid differs from training outcomes')
    else:
        write_manifest(grid_path, grid)
    validation = fixed_stage(bundle, output/'fixed-validation', 'validation')
    fixed_chunks = records(output/'fixed-validation/chunks.jsonl')
    fixed_points, mixes = [], []
    for tick in grid['slider']['ticks']:
        budget = tick['budget_hours']
        for method, group in validation.items():
            rows = list(group.values())
            chunks = [[c for c in fixed_chunks if c['episode_id'] == r['episode_id'] and c['method'] == method]
                      for r in rows]
            fixed_points.append(comparison(method, rows, chunks, budget, refs))
        mixes.append(fit_fixed_mix(output/'fixed-validation', refs, budget))
    print('C84 budget hours: '+', '.join(f"{t['budget_hours']:.4f}" for t in grid['slider']['ticks']), flush=True)
    return refs, grid, fixed_points, mixes, validation


def fixed_reference_points(fixed, mixes, groups, refs):
    """Mixture expected costs/misses and p95 use the full outcome distribution."""
    result = []
    for mix in mixes:
        budget = mix['budget_hours']
        points = [p for p in fixed if p['budget_hours'] == budget]
        eligible = [p for p in points if p['summary']['empirical_target_met']]
        if eligible:
            best = min(eligible, key=lambda p: p['summary']['worst_normalized_carbon'])
            result.append({'method': 'Best-Fixed', 'budget_hours': budget,
                           'selected_method': best['method'], 'summary': deepcopy(best['summary'])})
        else:
            result.append({'method': 'Best-Fixed', 'budget_hours': budget, 'summary': None,
                           'reason': 'no empirical feasible fixed scale'})
        if not mix['empirical_feasible']:
            result.append({'method': 'Fixed-Mix', 'budget_hours': budget, 'summary': None,
                           'reason': 'LP empirically infeasible'})
            continue
        weighted = [(r, weight/len(groups['Fixed-'+n])) for n, weight in mix['weights'].items()
                    for r in groups['Fixed-'+n].values() if weight > 0]
        costs = {rho: sum(weight*r['carbon_g_per_kappa'][rho] for r, weight in weighted)
                 for rho in refs['carbon_reference_g_per_kappa']}
        miss = sum(weight*(r['tat_hours'] > budget) for r, weight in weighted)
        summary = {
            'outcomes': len(next(iter(groups.values()))), 'completed': len(next(iter(groups.values()))),
            'incomplete': 0, 'mean_tat_hours': sum(weight*r['tat_hours'] for r, weight in weighted),
            'p95_tat_hours': weighted_quantile([(r['tat_hours'], w) for r, w in weighted], .95),
            'mean_nodehours': sum(weight*r['nodehours'] for r, weight in weighted),
            'mean_chunks': sum(weight*r['chunk_count'] for r, weight in weighted),
            'mean_modeled_carbon_g_per_kappa': costs,
            'worst_normalized_carbon': max(costs[rho]/ref for rho, ref in refs['carbon_reference_g_per_kappa'].items()),
            'miss_rate_bounds': [miss, miss], 'empirical_target_met': miss <= mix['epsilon']+1e-10,
            'observed_scale_change_fraction': 0., 'observed_scale_down_fraction': 0.,
            'observed_scale_up_fraction': 0.,
            'single_chunk_fraction': sum(w*(r['chunk_count'] == 1) for r, w in weighted),
            'interpretation': 'validation-fitted mixture expectations, not independent test observations'}
        require(math.isclose(summary['worst_normalized_carbon'], mix['estimated_objective'], abs_tol=1e-9),
                'Mixture curve cost disagrees with fitted objective')
        result.append({'method': 'Fixed-Mix', 'budget_hours': budget, 'weights': mix['weights'], 'summary': summary})
    return result


def planners(bundle, output, refs, grid, software):
    selected = sorted((e for e in bundle.episodes if e.split == 'validation'), key=lambda e: e.arrival)
    by_budget = {tick['budget_multiplier']: [] for tick in grid['slider']['ticks']}
    seals = {}
    total = len(selected)*len(by_budget)
    for index, episode in enumerate(selected):
        for i, tick in enumerate(grid['slider']['ticks']):
            path = output/'planners'/f'budget-{i:02d}'/episode.episode_id
            (rows, groups), seal = planner_stage(bundle, path, episode, tick, output/'e1/model/model.json',
                                                 output/'references.json', software)
            by_budget[tick['budget_multiplier']].append((rows, groups))
            seals[str(path.relative_to(output))] = seal
            print(f'C84 planner unit {index*len(by_budget)+i+1}/{total} complete', flush=True)
    points = []
    for tick in grid['slider']['ticks']:
        for method in ('Plan-once', 'Rollout-MPC'):
            rows, chunks = [], []
            for rs, groups in by_budget[tick['budget_multiplier']]:
                for row in rs:
                    if row['method'] == method:
                        rows.append(row); chunks.append(groups[(row['episode_id'], method)])
            points.append(comparison(method, rows, chunks, tick['budget_hours'], refs))
    return points, seals


def actor_readout(rows, groups, iteration, nodes):
    responses, behavior = [], []
    for beta in sorted({r['budget_multiplier'] for r in rows}):
        first = [groups[(r['episode_id'], beta)][0] for r in rows if r['budget_multiplier'] == beta]
        require(all(len(c['policy_probabilities']) == len(nodes) for c in first), 'Action probability dimension differs')
        behavior.append({'budget_multiplier': beta, 'arrivals': len(first),
                         'sampled_first_nodes': dict(Counter(c['selected_nodes'] for c in first)),
                         'first_argmax_nodes': dict(Counter(nodes[max(range(len(nodes)), key=lambda i:c['policy_probabilities'][i])] for c in first)),
                         'first_probability_ranges': [[min(c['policy_probabilities'][i] for c in first),
                                                       max(c['policy_probabilities'][i] for c in first)] for i in range(len(nodes))]})
    for row in rows:
        p = groups[(row['episode_id'], row['budget_multiplier'])][0]['policy_probabilities']
        responses.append({'iteration': iteration, 'episode_id': row['episode_id'],
                          'budget_multiplier': row['budget_multiplier'], 'probabilities': p,
                          'entropy_nats': -sum(x*math.log(x) for x in p if x > 0)})
    return {**response_summary(responses), 'within_budget_behavior': behavior, 'nodes': nodes}


def run(bundle, output, stage='all', resume=False):
    validate_scope(bundle)
    output = Path(output)
    require(stage in {'all', 'prepare', 'train'}, 'Unknown stage')
    software = provenance(bundle.root)
    template = settings(); template.pop('budgets')
    contract = {'kind': 'frontera_capacity84_development_v1', 'config_sha256': bundle.manifest['config_sha256'],
                'asset_sha256': bundle.manifest['asset_sha256'], 'source_sha256': software['source_sha256'],
                'script_sha256': {name: digest(Path(__file__).with_name(name)) for name in SCRIPTS},
                'torch': str(torch.__version__), 'capacity_nodes': 84, 'allowed_nodes': NODES,
                'training_settings_without_derived_budgets': template,
                'validation_iterations': list(VALIDATION_ITERATIONS), 'probe_interval_seconds': PROBE_INTERVAL_SECONDS,
                'planner': {'methods': ['Plan-once', 'Rollout-MPC'], 'paths': 256, 'epsilon': .05, 'seed': 11},
                'initialization': 'fresh seed11; no C128 checkpoint or outcomes reused',
                'scope': 'one panel; unchanged chunk/learning rules; train and development validation only'}
    # Reject an old/foreign output before even opening a lock file there.
    path = output/'pipeline-contract.json'
    if path.exists():
        require(load_json(path) == contract, 'C84 inputs/settings/software changed; use a new run directory')
    elif output.exists():
        require(not any(p.name != '.capacity84.lock' for p in output.iterdir()), 'Unknown existing C84 output')
    with probe_lock(output, resume, name='.capacity84.lock'):
        path = output/'pipeline-contract.json'
        if path.exists():
            require(load_json(path) == contract, 'C84 inputs/settings/software changed; use a new run directory')
        else:
            require(not any(p.name != '.capacity84.lock' for p in output.iterdir()), 'Unknown existing C84 output')
            write_manifest(path, contract)
        prepared = output/'prepared-seal.json'
        if stage == 'train':
            require(prepared.exists(), 'Run --stage prepare first, or use --stage all')
        # Check frozen preparation before any stage can modify output artifacts.
        if prepared.exists():
            for name, sha in load_json(prepared).items():
                require(digest(output/name) == sha, 'C84 prepared artifact changed: '+name)
        refs, grid, fixed, mixes, fixed_groups = fixed_inputs(bundle, output)
        configured = settings(grid['budget_multipliers'])
        plan = {**contract, 'settings': configured, 'budget_grid': grid,
                'references_sha256': digest(output/'references.json'),
                'fixed_train_seal': load_json(output/'fixed-train/stage-seal.json'),
                'fixed_validation_seal': load_json(output/'fixed-validation/stage-seal.json')}
        if (output/'run-plan.json').exists():
            require(load_json(output/'run-plan.json') == plan, 'C84 derived run plan changed')
        else:
            write_manifest(output/'run-plan.json', plan)
        queue_binding, queue = queue_stage(bundle, output)
        planner_points, planner_seals = planners(bundle, output, refs, grid, software)
        preparation_files = ['run-plan.json', 'references.json', 'budget-grid.json',
                             'fixed-train/stage-seal.json', 'fixed-validation/stage-seal.json']
        preparation_files += ['e1/'+name for name in queue_binding]
        preparation_files += [directory+'/'+name for directory, seal in planner_seals.items() for name in seal]
        preparation = {name: digest(output/name) for name in preparation_files}
        if prepared.exists():
            require(load_json(prepared) == preparation, 'C84 preparation binding differs')
        else:
            write_manifest(prepared, preparation)
        baseline = fixed+fixed_reference_points(fixed, mixes, fixed_groups, refs)+planner_points
        report = {'kind': 'capacity84_development_summary_v1', 'status': 'prepared',
                  'run_plan_sha256': digest(output/'run-plan.json'), 'panel': bundle.raw['panel'],
                  'settings': configured, 'nodes': NODES, 'budget_grid': grid, 'queue': queue,
                  'baselines': baseline, 'fixed_mix': mixes, 'checkpoints': [],
                  'scope': 'development validation; all checkpoints reported; no test or confidence interval claims'}
        # Re-running preparation must not erase a completed/partial training report.
        previous_report = output/'capacity84-summary.json'
        if previous_report.exists():
            saved = load_json(previous_report)
            require(saved['run_plan_sha256'] == report['run_plan_sha256'] and
                    saved['baselines'] == baseline and saved['queue'] == queue,
                    'Existing report binds another run or baseline outcomes')
            if saved['checkpoints']:
                report = saved
        if stage == 'prepare':
            write_readout(output, report)
            print(f'C84 preparation complete: {output}/capacity84-summary.md', flush=True)
            return report
        report['status'] = 'training'
        write_readout(output, report)
        print(f'C84 PPO: {ITERATIONS} rounds x 4 budgets x {EPISODES_PER_BUDGET} arrivals; fresh independent product actors, shared critic.', flush=True)
        checkpoint, metadata, history = ppo_stage(bundle, output, output/'references.json', configured)
        report.update(status='partial', final_checkpoint=str(checkpoint.relative_to(output)),
                      ppo_episodes=metadata['total_episodes'], ppo_chunks=metadata['total_chunks'], checkpoints=[],
                      iteration_timing=[{k:r[k] for k in ('iteration', 'episodes', 'chunks', 'rollout_seconds', 'update_seconds')} for r in history])
        by_iteration = {r['iteration']:r for r in history}
        for iteration in VALIDATION_ITERATIONS:
            candidate = checkpoint_at(output, iteration, by_iteration[iteration])
            meta, _ = read_checkpoint(candidate)
            require(meta['settings'] == configured and meta['feature_schema']['nodes'] == NODES,
                    'C84 policy checkpoint settings/actions differ')
            rows, groups, seal = validation_stage(bundle, output/f'validation-{iteration:06d}', candidate, configured)
            ResultRun(output/f'validation-{iteration:06d}', refs, allowed_splits={'validation'})
            points = []
            for tick in grid['slider']['ticks']:
                selected = [r for r in rows if r['budget_multiplier'] == tick['budget_multiplier']]
                points.append(comparison(f'PPO-{iteration}', selected,
                                         [groups[(r['episode_id'], r['budget_multiplier'])] for r in selected],
                                         tick['budget_hours'], refs))
            report['checkpoints'].append({'iteration': iteration, 'checkpoint_sha256': digest(candidate),
                                          'validation_files': seal, 'operating_points': points,
                                          'actor': actor_readout(rows, groups, iteration, NODES)})
            report['status'] = 'complete' if iteration == VALIDATION_ITERATIONS[-1] else 'partial'
            write_readout(output, report)
            print(f'C84 validation {iteration} and curves complete.', flush=True)
        print(f'C84 trial complete: {output}/capacity84-summary.md', flush=True)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--stage', choices=('all', 'prepare', 'train'), default='all')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run(Bundle(Path(args.config)), Path(args.output), args.stage, args.resume)


if __name__ == '__main__':
    main()
