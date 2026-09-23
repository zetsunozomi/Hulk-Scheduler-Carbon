"""Frozen-policy environment sensitivity; source contracts remain strict.

This separate runner changes only one declared environment assumption. It does
not certify that its checkpoint/budget was chosen before seeing stress results.
"""
from copy import deepcopy
from dataclasses import replace
import hashlib
from pathlib import Path
import time

from .common import digest, integer, number, require
from .config import Bundle
from .environment import Environment, initial_replay
from .forecast import CausalForecast
from .planning import PlanningPolicy, RolloutPlanner
from .policy_inputs import PolicyInputs, shared_inputs
from .runner import provenance, write_manifest, write_record

VARIANTS = ('width2', 'fcfs', 'overhead60', 'overhead1800')


def derive_config(source, variant):
    require(variant in VARIANTS, 'Unknown environment sensitivity')
    require(source.raw['trace'].get('role') == 'workload_template' and
            source.raw['execution'].get('initial_state_mode') == 'empty_warmup',
            'Environment sensitivity requires a rebuilt workload-template source')
    raw = deepcopy(source.raw)
    # Absolute input paths make this archived derived config self-contained on
    # the running host. The source config stays portable and unchanged.
    raw['root'] = str(source.root)
    for name, path in source.assets.items():
        raw[name]['path'] = str(path)
    raw['panel'] += '-' + variant
    if variant == 'width2':
        require(raw['trace'].get('node_multiplier', 1) == 1, 'Width sensitivity requires unscaled source demands')
        raw['trace'].update(node_multiplier=2, oversize_policy='cap')
    elif variant == 'fcfs':
        require(raw['cluster']['scheduler']['model'] == 'conservative_backfill_v1', 'FCFS sensitivity requires the main backfill source')
        raw['cluster']['scheduler']['model'] = 'fcfs_v1'
    else:
        half = int(variant.removeprefix('overhead')) // 2
        for profile in raw['workload']['profiles'].values():
            profile.update(initialization_seconds=half, restart_seconds=half, checkpoint_seconds=half)
    return raw


def run_stress(source, output, predictor_path, checkpoint_path, variant, budget_multiplier,
               split='test', paths=256, miss_tolerance=.05):
    from .policy_runner import (ActorCritic, check_policy_inputs, collect_episode, configure_torch,
                                read_checkpoint, torch)
    require(split in {'validation', 'test'}, 'Stress evaluation uses validation or test only')
    paths = integer(paths, 'planning paths')
    beta = float(number(budget_multiplier, 'budget multiplier', strict=True))
    require(0 <= miss_tolerance <= 1, 'Invalid MPC risk threshold')
    metadata, state = read_checkpoint(checkpoint_path)
    settings = metadata['settings']
    require(settings['objective'] == 'robust' and settings['forecast_mode'] == 'window' and settings.get('wait_features') == 'none' and settings.get('decision_mode','feedback') == 'feedback',
            'Environment sensitivity evaluates the main robust policy without wait advice')
    require(beta in settings['budgets'], 'Stress budget must belong to the trained grid')
    references, predictor = shared_inputs(source, predictor_path, metadata['references'])
    encoder = PolicyInputs(source, predictor, references)
    check_policy_inputs(metadata, encoder, predictor, source)
    software = {**provenance(source.root), 'torch': str(torch.__version__)}
    require(metadata['software']['source_sha256'] == software['source_sha256'], 'Checkpoint implementation changed')
    require(metadata['software']['torch'] == software['torch'], 'Checkpoint PyTorch version differs')
    raw = derive_config(source, variant)
    output = Path(output)
    require(not output.exists(), f'Output already exists: {output}')
    output.mkdir(parents=True)
    write_manifest(output/'scenario-config.json', raw)
    target = Bundle(output/'scenario-config.json')
    episodes = [replace(e, budget_hours=beta*references['time_reference_hours'])
                for e in target.episodes if e.split == split]
    require(episodes, 'Empty stress cohort')
    seed = settings['seed']
    configure_torch(seed, settings['threads'])
    model = ActorCritic(len(encoder.global_names), len(encoder.action_names),
                       interaction=settings.get('actor_interaction', 'concat'),
                       actor_budgets=settings['budgets'] if settings.get('actor_budget_mode','shared')=='independent' else None)
    model.load_state_dict(state['model']); model.eval()
    forecast = CausalForecast(source.ci, *source.splits['train'], calendar_timezone=source.raw['trace']['timezone'])
    checkpoint_hash = digest(checkpoint_path)
    manifest = {**target.manifest, 'kind': 'frozen_environment_sensitivity_v1', 'status': 'running',
                'source_config_sha256': source.manifest['config_sha256'], 'source_config': source.raw,
                'checkpoint_sha256': checkpoint_hash, 'predictor_sha256': predictor.version,
                'policy_predictor_sha256': None, 'wait_predictor_role': 'Rollout-MPC only',
                'references': references, 'feature_schema': encoder.schema(), 'variant': variant,
                'split': split, 'selected_episode_ids': [e.episode_id for e in episodes],
                'methods': ['ScaleDown', 'Rollout-MPC'], 'training_seed': seed, 'budget_multiplier': beta,
                'planning_paths': paths, 'planner_internal_miss_tolerance': miss_tolerance,
                'software': software, 'completed_episode_methods': 0,
                'selection_status': 'Frozen source artifacts; caller must archive pre-stress validation selection of checkpoint, budget and MPC threshold. No stress-based selection is performed.',
                'frozen': ['weights', 'wait_predictor', 'normalizers', 'references', 'CI', 'cohort', 'work', 'rates', 'power']}
    write_manifest(output/'manifest.json', manifest)
    begun, current = time.monotonic(), None
    try:
        with (output/'chunks.jsonl').open('x') as chunks, (output/'episodes.jsonl').open('x') as records:
            for episode in episodes:
                current = {'episode_id': episode.episode_id, 'variant': variant, 'budget_multiplier': beta}
                base = initial_replay(target, episode)
                paired = int.from_bytes(hashlib.sha256(f'{seed}:{episode.episode_id}:{beta}'.encode()).digest()[:8], 'big')
                generator = torch.Generator().manual_seed(paired)
                identity = {**current, 'epsilon': settings['epsilon'], 'policy_checkpoint': checkpoint_hash,
                            'power_objective': 'robust', 'policy_forecast_mode': 'window'}
                _, _, summary = collect_episode(target, episode, model, encoder, generator, base,
                                                'ScaleDown', seed, chunks, identity)
                write_record(records, summary); manifest['completed_episode_methods'] += 1
                paired_mpc = seed + int.from_bytes(hashlib.sha256(episode.episode_id.encode()).digest()[:4], 'big')
                planner = RolloutPlanner(target.workload, target.raw['power'], predictor, references, paths, miss_tolerance)
                policy = PlanningPolicy(planner, forecast, 'mpc', paired_mpc)
                env = Environment(target, episode, base, 'Rollout-MPC', seed)
                while env.status == 'running':
                    action, details = policy.choose(env.observe())
                    details['planner_fallback_reason'] = details.pop('fallback_reason')
                    chunk = env.step(action); policy.record_execution(chunk)
                    write_record(chunks, {**chunk, **details, **current})
                write_record(records, {**env.summary(), **current})
                manifest['completed_episode_methods'] += 1
                print(f'{episode.episode_id} {variant}: completed both methods', flush=True)
        manifest['status'] = 'complete'
    except Exception as exc:
        manifest.update(status='failed', failure={**(current or {}), 'error_type': type(exc).__name__, 'message': str(exc)})
        write_manifest(output/'failure.json', manifest['failure'])
        raise
    finally:
        manifest['elapsed_seconds'] = time.monotonic()-begun
        write_manifest(output/'manifest.json', manifest)
    return manifest
