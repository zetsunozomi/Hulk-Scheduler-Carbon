"""C84 weighted PPO, reusing sealed fixed runs and calibrating reward on-policy."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import uuid

from carbon.baselines import check_references, fixed_records
from carbon.common import digest, load_json, require
from carbon.config import Bundle
from carbon.environment import Environment
from carbon.learning import sample_action, torch
from carbon.policy_runner import ReplayCache, configure_torch, nested_tuple
from carbon.probe_resume import probe_lock
from carbon.runner import write_manifest, write_record
from weighted_policy import WeightedActorCritic, WeightedInputs, cost, freeze, observe, optimize


DEFAULTS = dict(alphas=[0., .2, .5, .8, 1.], iterations=64, observation_iterations=5,
                episodes_per_alpha=16, validation_iterations=[16, 32, 48, 64], rho='1.0',
                seed=11, threads=1, width=128, learning_rate=3e-4, ppo_epochs=4,
                minibatch_episodes=16, clip=.2, value_coefficient=.5, entropy=.01, gradient_norm=.5)
SCRIPTS = ('weighted_policy.py', 'weighted84_trial.py', 'weighted84_report.py')
NODES = [4, 16, 64]


def records(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def seal(directory, names=None):
    path = directory/'stage-seal.json'
    if names is not None:
        require(not path.exists(), 'Stage already sealed')
        write_manifest(path, {name: digest(directory/name) for name in names})
    values = load_json(path)
    require(values and all(Path(name).name == name for name in values), 'Invalid stage seal')
    require(all((directory/name).is_file() and digest(directory/name) == sha for name, sha in values.items()),
            f'Sealed artifact changed: {directory}')
    return values


def source_inputs(bundle, source):
    """Read existing C84 fixed artifacts; no preparation or predictor execution."""
    require(bundle.raw['cluster']['nodes'] == 84 and bundle.raw['cluster']['allowed_nodes'] == NODES,
            'Expected C84 actions 4/16/64')
    refs = load_json(source/'references.json')
    check_references(refs, bundle.manifest)
    require(refs['source_episodes_sha256'] == digest(source/'fixed-train/episodes.jsonl'),
            'Reference source changed')
    binding = {'references.json': digest(source/'references.json')}
    current = {p.name: digest(p) for p in (bundle.root/'src/carbon').glob('*.py')}
    for split in ('train', 'validation'):
        directory = source/f'fixed-{split}'
        sealed = seal(directory)
        require(set(sealed) == {'manifest.json', 'chunks.jsonl', 'episodes.jsonl'}, 'Unexpected fixed seal')
        meta = load_json(directory/'manifest.json')
        require(meta['status'] == 'complete' and meta['config_sha256'] == bundle.manifest['config_sha256'],
                'Fixed configuration/completion differs')
        require(meta['asset_sha256'] == bundle.manifest['asset_sha256'], 'Fixed assets differ')
        require(meta['software']['source_sha256'] == current, 'Fixed replay implementation changed')
        groups = fixed_records(directory/'episodes.jsonl', split, NODES)
        ids = {e.episode_id for e in bundle.episodes if e.split == split}
        require(all(set(group) == ids for group in groups.values()), 'Fixed cohort differs')
        binding.update({f'fixed-{split}/{name}': sha for name, sha in sealed.items()})
        binding[f'fixed-{split}/stage-seal.json'] = digest(directory/'stage-seal.json')
    return refs, binding


def collect(bundle, episode, model, encoder, generator, base, alpha, settings, stream, identity, normalizers):
    env = Environment(bundle, episode, base, 'Weighted-PPO', settings['seed'])
    steps, increments, nodes = [], [], []
    initial_probabilities = None
    while env.status == 'running':
        before = env.replay.time
        elapsed = (before-episode.arrival).total_seconds()/3600
        encoded, metadata = encoder.encode(env.observe(), elapsed, alpha)
        action, log_prob, values, probabilities = sample_action(model, encoded, generator)
        if initial_probabilities is None:
            initial_probabilities = probabilities
        chunk = env.step(encoder.nodes[action])
        chunk.pop('budget_hours', None)
        dt = (env.replay.time-before).total_seconds()/3600
        carbon = chunk['carbon_g_per_kappa'][settings['rho']]
        write_record(stream, {**chunk, **metadata, **identity, 'alpha': alpha,
                             'policy_probabilities': probabilities, 'selected_log_probability': log_prob,
                             'predicted_cost_to_go': values, 'tat_increment_hours': dt,
                             'reward': -cost(dt, carbon, alpha, normalizers) if normalizers else None})
        steps.append({'input': encoded, 'action': action, 'log_probability': log_prob, 'values': values})
        increments.append([dt, carbon]); nodes.append(encoder.nodes[action])
    summary = env.summary()
    summary.pop('budget_hours', None); summary.pop('deadline_miss', None)
    summary.update(identity, alpha=alpha, selected_nodes=nodes,
                   initial_probabilities=initial_probabilities, policy_rule='categorical_sampling')
    if summary['final_status'] == 'completed' and normalizers:
        summary['weighted_cost'] = cost(summary['tat_hours'], summary['carbon_g_per_kappa'][settings['rho']], alpha, normalizers)
    return {'alpha': alpha, 'steps': steps, 'increments': increments}, summary


def read_checkpoint(directory):
    sealed = seal(directory)
    require({'checkpoint.json', 'checkpoint.pt', 'episodes.jsonl', 'chunks.jsonl', 'training.json'} <= set(sealed),
            'Incomplete iteration seal')
    meta = load_json(directory/'checkpoint.json')
    require(meta['kind'] == 'weighted_ppo_checkpoint_v1', 'Unknown weighted checkpoint')
    return meta, torch.load(directory/'checkpoint.pt', map_location='cpu', weights_only=True)


def sync_normalizers(output, normalizers):
    path = output/'reward-normalizers.json'
    if path.exists():
        require(normalizers is not None and load_json(path) == normalizers, 'Reward normalizers changed')
    elif normalizers:
        write_manifest(path, normalizers)


def validate(bundle, output, iteration, encoder, settings, normalizers):
    source = output/f'iteration-{iteration:06d}'
    checkpoint, state = read_checkpoint(source)
    require(checkpoint['normalizers'] == normalizers, 'Checkpoint normalizers differ')
    directory = output/f'validation-{iteration:06d}'
    binding = {'iteration': iteration, 'checkpoint_sha256': digest(source/'checkpoint.pt'),
               'normalizers': normalizers, 'split': 'validation'}
    if directory.exists():
        seal(directory)
        require(load_json(directory/'manifest.json') == binding, 'Validation binding differs')
        return
    pending = output/f'.validation-{iteration:06d}-{uuid.uuid4().hex}'
    pending.mkdir()
    model = WeightedActorCritic(len(encoder.global_names), len(encoder.action_names), settings['alphas'], settings['width'])
    model.load_state_dict(state['model']); model.eval()
    cache = ReplayCache(bundle)
    with (pending/'chunks.jsonl').open('x') as chunks, (pending/'episodes.jsonl').open('x') as episodes:
        for episode in (e for e in bundle.episodes if e.split == 'validation'):
            base = cache.get(episode)
            for alpha in settings['alphas']:
                paired_seed = int.from_bytes(hashlib.sha256(f'{settings["seed"]}:{episode.episode_id}:{alpha}'.encode()).digest()[:8], 'big')
                generator = torch.Generator().manual_seed(paired_seed)
                _, row = collect(bundle, episode, model, encoder, generator, base, alpha, settings, chunks,
                                 {'iteration': iteration, 'phase': 'validation'}, normalizers)
                write_record(episodes, row)
                require(row['final_status'] == 'completed', 'Incomplete validation episode; fix coverage, do not drop')
                print(f'weighted validation {iteration}: {episode.episode_id} alpha={alpha}: completed', flush=True)
    write_manifest(pending/'manifest.json', binding)
    seal(pending, ['chunks.jsonl', 'episodes.jsonl', 'manifest.json'])
    pending.rename(directory)


def run(bundle, output, source, resume=False, stage='all', settings=None):
    settings = {**DEFAULTS, **(settings or {})}
    require(stage in {'all', 'train', 'validate'}, 'Unknown stage')
    require(0 < settings['observation_iterations'] < settings['iterations'] and settings['episodes_per_alpha'] > 0,
            'Invalid observation/training length')
    require(settings['validation_iterations'] and all(settings['observation_iterations'] < i <= settings['iterations']
            for i in settings['validation_iterations']), 'Invalid validation iterations')
    require(settings['rho'] in {str(r) for r in bundle.raw['power']['rho_interval']}, 'Unknown carbon endpoint')
    output, source = Path(output).resolve(), Path(source).resolve()
    require(output != source and source not in output.parents and output not in source.parents, 'Use separate weighted output')
    if output.exists():
        require(resume, 'Output exists; use --resume')
        require((output/'run-plan.json').is_file() and load_json(output/'run-plan.json').get('kind') == 'weighted84_v1',
                'Unknown existing weighted output')
    refs, source_binding = source_inputs(bundle, source)
    encoder = WeightedInputs(bundle, refs)
    scripts = Path(__file__).parent
    contract = {'kind': 'weighted84_v1', 'config_sha256': bundle.manifest['config_sha256'],
                'asset_sha256': bundle.manifest['asset_sha256'], 'settings': settings,
                'source': str(source), 'source_binding': source_binding, 'feature_schema': encoder.schema(),
                'software': {'torch': str(torch.__version__),
                             'core': {p.name: digest(p) for p in (bundle.root/'src/carbon').glob('*.py')},
                             'scripts': {name: digest(scripts/name) for name in SCRIPTS}}}
    if output.exists():
        require(load_json(output/'run-plan.json') == contract, 'Weighted inputs/settings/software changed')
    configure_torch(settings['seed'], settings['threads'])
    with probe_lock(output, resume, '.weighted.lock'):
        if not (output/'run-plan.json').exists():
            write_manifest(output/'run-plan.json', contract)
        return run_locked(bundle, output, source, encoder, settings, stage)


def run_locked(bundle, output, source, encoder, settings, stage):
    model = WeightedActorCritic(len(encoder.global_names), len(encoder.action_names), settings['alphas'], settings['width'])
    optimizer = torch.optim.Adam(model.parameters(), lr=settings['learning_rate'])
    generator = torch.Generator().manual_seed(settings['seed']); rng = random.Random(settings['seed'])
    warmup = dict(count=0, tat_sum_hours=0., carbon_sum_g_per_kappa=0., frozen=False)
    normalizers = None
    completed, total_episodes, total_chunks = 0, 0, 0
    directories = sorted(output.glob('iteration-[0-9]*'))
    for expected, directory in enumerate(directories, 1):
        require(directory.name == f'iteration-{expected:06d}', 'Committed iterations are not consecutive')
        seal(directory)
    if directories:
        meta, state = read_checkpoint(directories[-1])
        completed, total_episodes, total_chunks = (meta[k] for k in ('iteration', 'total_episodes', 'total_chunks'))
        require(completed == len(directories) and completed <= settings['iterations'], 'Invalid checkpoint iteration')
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        generator.set_state(state['sampling_rng']); rng.setstate(nested_tuple(meta['shuffle_rng']))
        warmup, normalizers = meta['warmup'], meta['normalizers']
    sync_normalizers(output, normalizers)
    require(stage != 'validate' or completed == settings['iterations'], 'Training is not complete')
    status = {'kind': 'weighted84_v1', 'status': 'running', 'completed_iteration': completed,
              'total_episodes': total_episodes, 'total_chunks': total_chunks}
    write_manifest(output/'manifest.json', status)
    cache = ReplayCache(bundle)
    cohort = [e for e in bundle.episodes if e.split == 'train']
    require(cohort, 'Empty training cohort')
    try:
        if stage != 'validate':
            for iteration in range(completed+1, settings['iterations']+1):
                observing = iteration <= settings['observation_iterations']
                phase = 'observe' if observing else 'train'
                pending = output/f'.iteration-{iteration:06d}-{uuid.uuid4().hex}'; pending.mkdir()
                rollout, rows = [], []
                # Each alpha sees the same sampled arrival list in each round.
                arrivals = [cohort[rng.randrange(len(cohort))] for _ in range(settings['episodes_per_alpha'])]
                model.eval()
                with (pending/'chunks.jsonl').open('x') as chunks, (pending/'episodes.jsonl').open('x') as episodes:
                    for sample, episode in enumerate(arrivals):
                        base = cache.get(episode)
                        for alpha in settings['alphas']:
                            trajectory, row = collect(bundle, episode, model, encoder, generator, base, alpha, settings, chunks,
                                                      {'iteration': iteration, 'sample': sample, 'phase': phase}, normalizers)
                            write_record(episodes, row)
                            require(row['final_status'] == 'completed', 'Incomplete training episode; fix coverage, do not drop')
                            rollout.append(trajectory); rows.append(row)
                metrics = []
                if observing:
                    observe(warmup, rows, settings['rho'])
                    if iteration == settings['observation_iterations']:
                        normalizers = freeze(warmup, settings['rho'], settings['observation_iterations'])
                else:
                    model.train()
                    metrics = optimize(model, optimizer, rollout, normalizers, settings, rng)
                total_episodes += len(rows); total_chunks += sum(len(e['steps']) for e in rollout)
                torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                            'sampling_rng': generator.get_state()}, pending/'checkpoint.pt')
                write_manifest(pending/'checkpoint.json', {'kind': 'weighted_ppo_checkpoint_v1', 'iteration': iteration,
                               'total_episodes': total_episodes, 'total_chunks': total_chunks, 'warmup': warmup,
                               'normalizers': normalizers, 'shuffle_rng': rng.getstate()})
                write_manifest(pending/'training.json', {'iteration': iteration, 'phase': phase, 'episodes': len(rows),
                               'optimizer_steps': len(metrics), 'optimizer_minibatches': metrics})
                seal(pending, ['checkpoint.pt', 'checkpoint.json', 'training.json', 'episodes.jsonl', 'chunks.jsonl'])
                pending.rename(output/f'iteration-{iteration:06d}')
                sync_normalizers(output, normalizers)
                status.update(completed_iteration=iteration, total_episodes=total_episodes, total_chunks=total_chunks)
                write_manifest(output/'manifest.json', status)
                print(f'weighted {phase} {iteration}/{settings["iterations"]}: episodes={total_episodes}, '
                      f'chunks={total_chunks}, optimizer_steps={len(metrics)}', flush=True)
                if iteration == settings['observation_iterations']:
                    print(f'Frozen reward means: TAT={normalizers["tat_hours"]:.6f}h, '
                          f'carbon={normalizers["carbon_g_per_kappa"]:.6f}, n={normalizers["episodes"]}', flush=True)
        status['status'] = 'trained'
        if stage in {'all', 'validate'}:
            for iteration in settings['validation_iterations']:
                validate(bundle, output, iteration, encoder, settings, normalizers)
            from weighted84_report import write_readout
            write_readout(output, source, settings, normalizers)
            status['status'] = 'complete'
    except BaseException as exc:
        status.update(status='partial', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        write_manifest(output/'manifest.json', status)
    print(f'Weighted C84 {status["status"]}: {output}', flush=True)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/amsp-frontera-7b-c84.development.json')
    parser.add_argument('--source', default='results/amsp-frontera-7b-c84-seed11')
    parser.add_argument('--output', default='results/amsp-frontera-7b-c84-weighted-seed11')
    parser.add_argument('--stage', choices=['all', 'train', 'validate'], default='all')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    require(os.environ.get('PBS_JOBID') or load_json(args.config)['purpose'] == 'synthetic',
            'Real replay requires a PBS allocation; use scripts/sophia_weighted84.sh')
    run(Bundle(args.config), args.output, args.source, args.resume, args.stage)


if __name__ == '__main__':
    main()
