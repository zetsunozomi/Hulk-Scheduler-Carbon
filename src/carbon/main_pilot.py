"""Full fixed references and a bounded, restartable main-policy development run."""

import math
from pathlib import Path
import uuid

from .baselines import check_references, fixed_records, make_references
from .common import digest, load_json, require
from .policy_runner import read_checkpoint, settings_for, train_policy, torch
from .probe_resume import probe_lock
from .runner import provenance, run_fixed, write_manifest
from .slider import slider_contract

PILOT_SETTINGS = settings_for(5, episodes_per_budget=4, minibatch_episodes=4,
                              seed=11, wait_features='none', decision_mode='feedback')


def verify_fixed(bundle, directory, split):
    meta = load_json(directory/'manifest.json')
    require(meta['status'] == 'complete', 'Fixed stage is incomplete')
    require(all(meta.get(k) == v for k, v in bundle.manifest.items()), 'Fixed stage configuration/input differs')
    require(meta['software']['source_sha256'] == provenance(bundle.root)['source_sha256'], 'Fixed stage software differs')
    expected = {e.episode_id for e in bundle.episodes if e.split == split}
    require(meta['selected_splits'] == [split] and set(meta['selected_episode_ids']) == expected and
            meta['shard_count'] == 1 and meta['shard_index'] == 0,
            'References/pilot require the entire declared fixed cohort, not a timing shard')
    nodes = bundle.raw['cluster']['allowed_nodes']
    require(meta['fixed_nodes'] == nodes, 'Fixed stage is missing a scale')
    rows = fixed_records(directory/'episodes.jsonl', split, nodes)
    require(all(set(group) == expected for group in rows.values()), 'Fixed episode coverage differs')
    for group in rows.values():
        for row in group.values():
            require(row['completed_updates'] == bundle.workload.total_updates and row['remaining_updates'] == 0,
                    'Fixed result did not complete the declared work')
    return meta, rows


def fixed_stage(bundle, directory, split):
    seal = directory/'stage-seal.json'
    if directory.exists():
        meta = load_json(directory/'manifest.json') if (directory/'manifest.json').is_file() else {}
        if meta.get('status') != 'complete':
            backup = directory.with_name(directory.name+'.interrupted-'+uuid.uuid4().hex)
            directory.rename(backup)
            print(f'Preserved incomplete fixed stage: {backup}; restarting this stage.', flush=True)
    if not directory.exists():
        run_fixed(bundle, directory, bundle.raw['cluster']['allowed_nodes'], split)
    meta, rows = verify_fixed(bundle, directory, split)
    binding = {name: digest(directory/name) for name in ('manifest.json','episodes.jsonl','chunks.jsonl')}
    if seal.exists():
        require(load_json(seal) == binding, 'Completed fixed stage was modified')
    else:
        write_manifest(seal, binding)
    print(f"Fixed {split}: {len(meta['selected_episode_ids'])} arrivals, {meta['completed_episode_methods']} outcomes, {meta['elapsed_seconds']:.1f}s", flush=True)
    return rows


def make_budget_grid(references, train_rows):
    """Use only training Fixed-4 to span the time scale of multi-chunk actions."""
    values = sorted(row['tat_hours'] for row in train_rows['Fixed-4'].values())
    require(values and all(math.isfinite(v) and v > 0 for v in values), 'Invalid Fixed-4 training TAT')
    lower = references['time_reference_hours']
    upper = values[math.ceil(.95*len(values))-1]
    require(upper > lower, 'Training Fixed-4 p95 must exceed the fastest fixed mean to define a slider span')
    positions = [0., .25, .5, 1.]
    budgets = [1.+s*(upper/lower-1.) for s in positions]
    return {'kind':'training_fixed_span_budget_grid_v1',
            'source_split':'train', 'source_episodes_sha256':references['source_episodes_sha256'],
            'positions':positions, 'budget_multipliers':budgets,
            'time_reference_hours':lower, 'upper_budget_hours':upper,
            'upper_rule':'inverse empirical CDF at 0.95 of complete training Fixed-4 TAT',
            'scope':'four budget targets; training percentile is not a validation/test deadline guarantee',
            'slider':slider_contract(lower, budgets)}


def read_jsonl_prefix(path):
    """Completed lines only; an interrupted trailing write is not an observation."""
    import json
    rows = []
    with path.open('rb') as handle:
        for line in handle:
            if not line.endswith(b'\n'):
                break
            rows.append(json.loads(line))
    return rows


def ppo_stage(bundle, output, references_path, settings):
    attempts = sorted(output.glob('ppo-attempt-[0-9][0-9][0-9]'))
    checkpoint, best_iteration = None, 0
    for attempt in attempts:
        path = attempt/'manifest.json'
        if not path.exists():
            continue
        meta = load_json(path)
        require(meta['settings'] == settings, 'Pilot attempt settings differ')
        if meta.get('last_checkpoint'):
            candidate = attempt/meta['last_checkpoint']
            require(candidate.parent == attempt and candidate.name == meta['last_checkpoint'], 'Invalid checkpoint path')
            previous, _ = read_checkpoint(candidate)
            require(previous['settings'] == settings and previous['software']['source_sha256'] == provenance(bundle.root)['source_sha256'],
                    'Pilot checkpoint settings/software differ')
            require(previous['software']['torch'] == str(torch.__version__), 'Pilot checkpoint PyTorch version differs')
            check_references(previous['references'], bundle.manifest)
            require(previous['references'] == load_json(references_path), 'Pilot checkpoint references differ')
            require(previous['iteration'] == meta['completed_iteration'], 'Pilot checkpoint progress differs')
            if previous['iteration'] > best_iteration:
                checkpoint, best_iteration = candidate, previous['iteration']
    if best_iteration < settings['iterations']:
        next_attempt = max((int(p.name.rsplit('-',1)[1]) for p in attempts), default=-1)+1
        require(next_attempt < 1000, 'Too many pilot attempts')
        attempt = output/f'ppo-attempt-{next_attempt:03d}'
        result = train_policy(bundle, attempt, None, references_path, settings, checkpoint)
        checkpoint = attempt/result['last_checkpoint']
    require(checkpoint is not None, 'No completed pilot checkpoint')
    previous, _ = read_checkpoint(checkpoint)
    require(previous['iteration'] == settings['iterations'], 'Pilot has not reached its declared final iteration')
    # Resumed attempts may contain incomplete rollouts; take only recorded updates
    # on the chosen checkpoint chain. Earlier completed iterations never repeat.
    iterations = {}
    for attempt in sorted(output.glob('ppo-attempt-[0-9][0-9][0-9]')):
        path = attempt/'training.jsonl'
        if path.exists():
            for row in read_jsonl_prefix(path):
                candidate = attempt/row['checkpoint']
                if row['iteration'] <= previous['iteration'] and candidate.exists() and digest(candidate) == row['checkpoint_sha256']:
                    iterations[row['iteration']] = row
    require(set(iterations) == set(range(1,settings['iterations']+1)), 'Missing completed pilot iteration records')
    return checkpoint, previous, [iterations[k] for k in sorted(iterations)]


def run_main_pilot(bundle, output, resume=False):
    require(bundle.raw['purpose'] in {'development','synthetic'}, 'This short pilot is not a research training run')
    output = Path(output)
    with probe_lock(output, resume, name='.main-pilot.lock'):
        contract = {'kind':'main_development_pilot_v1','config_sha256':bundle.manifest['config_sha256'],
                    'asset_sha256':bundle.manifest['asset_sha256'],
                    'source_sha256':provenance(bundle.root)['source_sha256'],
                    'pilot_settings_before_train_budget_grid':PILOT_SETTINGS}
        contract_path = output/'pipeline-contract.json'
        if contract_path.exists():
            require(load_json(contract_path) == contract, 'Pipeline inputs/settings/software changed; preserve this run and use a new directory')
        else:
            require(not any(p.name != '.main-pilot.lock' for p in output.iterdir()), 'Unknown existing pilot output')
            write_manifest(contract_path, contract)
        train = fixed_stage(bundle, output/'fixed-train', 'train')
        references_path = output/'references.json'
        if not references_path.exists():
            make_references(output/'fixed-train', references_path)
        references = load_json(references_path)
        check_references(references, bundle.manifest)
        require(references['source_episodes_sha256'] == digest(output/'fixed-train/episodes.jsonl'), 'References use different fixed outcomes')
        ref_seal = output/'references-seal.json'
        ref_binding = {'references_sha256':digest(references_path)}
        if ref_seal.exists():
            require(load_json(ref_seal) == ref_binding, 'Training references were modified')
        else:
            write_manifest(ref_seal, ref_binding)
        grid = make_budget_grid(references, train)
        grid_path = output/'budget-grid.json'
        if grid_path.exists():
            require(load_json(grid_path) == grid, 'Training budget grid changed')
        else:
            write_manifest(grid_path, grid)
        print('Slider budgets (hours): '+', '.join(f"{t['budget_hours']:.2f}" for t in grid['slider']['ticks']), flush=True)
        validation = fixed_stage(bundle, output/'fixed-validation', 'validation')
        settings = {**PILOT_SETTINGS, 'budgets':grid['budget_multipliers']}
        print('Main PPO pilot: 5 iterations x 4 budgets x 4 arrivals = 80 complete episodes, seed=11, wait predictor=none.', flush=True)
        checkpoint, previous, history = ppo_stage(bundle, output, references_path, settings)
        fixed_summary = {}
        from statistics import fmean
        for split, groups in (('train',train),('validation',validation)):
            fixed_summary[split] = {name:{
                'outcomes':len(rows), 'mean_tat_hours':fmean(r['tat_hours'] for r in rows.values()),
                'mean_nodehours':fmean(r['nodehours'] for r in rows.values()),
                'mean_modeled_carbon_g_per_kappa':{str(rho):fmean(r['carbon_g_per_kappa'][str(rho)] for r in rows.values()) for rho in bundle.raw['power']['rho_interval']},
                'chunk_counts':sorted({r['chunk_count'] for r in rows.values()})}
                for name, rows in groups.items()}
        summary = {'kind':'main_development_pilot_summary','status':'complete','panel':bundle.raw['panel'],
                   'scope':'full fixed train/validation and short training pilot; no selected or tested RL-performance claim',
                   'fixed':fixed_summary,'budget_grid':grid, 'settings':settings,
                   'checkpoint':str(checkpoint.relative_to(output)), 'checkpoint_sha256':digest(checkpoint),
                   'ppo_iterations':previous['iteration'],'ppo_episodes':previous['total_episodes'],
                   'ppo_chunks':previous['total_chunks'], 'wait_predictor_sha256':previous['predictor_sha256'],
                   'iteration_timing':[{'iteration':r['iteration'],'rollout_seconds':r['rollout_seconds'],'update_seconds':r['update_seconds'],'episodes':r['episodes'],'chunks':r['chunks']} for r in history]}
        write_manifest(output/'pilot-summary.json', summary)
        print(f"Main pilot complete: {output}; checkpoint={summary['checkpoint']}; iterations={summary['ppo_iterations']}; episodes={summary['ppo_episodes']}", flush=True)
        print('Iteration rollout seconds: '+', '.join(f"{r['rollout_seconds']:.2f}" for r in history), flush=True)
        print('Readout: '+str(output/'pilot-summary.json'), flush=True)
        return summary
