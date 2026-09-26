"""Run complete training-free curves on declared synthetic job demand streams."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

from carbon.carbon import CarbonSeries
from carbon.common import digest, iso, load_json, require, timestamp
from carbon.environment import Environment
from carbon.probe_resume import probe_lock
from carbon.replay import Replay
from carbon.workload import Workload
from fixed_max_walltime import FixedMaxWalltimeWorkload
from opportunity_policy import OpportunityPolicy
from opportunity_traces import generate_cohort, generate_trace, trace_summary

ROOT = Path(__file__).resolve().parents[1]


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.pending')
    with temp.open('w') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def methods(design):
    return [(f'Fixed-{n}', n, None) for n in (4, 8, 16, 32)] + [
        (f'Dynamic-{a:.2f}', None, a) for a in design['alphas']]


def load_inputs(root, design_path):
    root, design_path = Path(root), Path(design_path)
    design = load_json(design_path)
    require(design['version'] == 'opportunity-v1', 'Expected opportunity-v1 design')
    require(design['purpose'] == 'constructed_mechanism_development', 'Synthetic development label required')
    require(design['target_request_rule'] == 'all_methods_request_48h_including_final', 'Unexpected request protocol')
    require(design['seeds'] and len(set(design['seeds'])) == len(design['seeds']), 'Distinct seeds required')
    require(design['alphas'] == sorted(set(design['alphas'])) and design['alphas'][0] == 0 and design['alphas'][-1] == 1,
            'Ordered slider grid must include 0 and 1')
    require(len({key for key, _, _ in methods(design)}) == len(methods(design)), 'Alpha labels collide')
    require(type(design['episodes_per_seed']) is int and design['episodes_per_seed'] > 0, 'Positive cohort size required')
    require(0 < design['arrival_start_hours'] < design['arrival_stop_hours'] and
            design['arrival_stop_hours'] + design['episode_timeout_hours'] < design['trace_hours'],
            'Need warmup and full common follow-up before trace end')
    require(design['pressure_cap_hours'] > 0 and design['constant_ci_g_per_kwh'] > 0, 'Positive policy/CI settings required')
    raw = load_json(root / design['profile_config'])
    # Validate the unchanged old profile before constructing a new scenario.
    # Do not bypass or relax the previous experiment's whole-config binding.
    from scaling_profiles import validate_bundle
    binding = validate_bundle(SimpleNamespace(raw=raw, root=root))
    require(binding['model'] == 'xl' and binding['scenario'] == 'e050', 'v1 fixes XL/e050')
    require(raw['cluster']['nodes'] == 84 and raw['cluster']['allowed_nodes'] == [4, 8, 16, 32], 'Expected C84')
    require(raw['cluster']['max_request_seconds'] == 48 * 3600, 'Expected 48h target cap')
    ci_path = root / raw['ci']['path']
    require(digest(ci_path) == raw['ci']['sha256'], 'ERCOT CI asset hash mismatch')
    ci = CarbonSeries.load(ci_path, raw['ci'])
    origin = timestamp(design['origin_utc'])
    ci.integral(origin, origin + timedelta(hours=design['trace_hours']))  # Reject gaps before expensive work.
    files = ['src/carbon/' + name for name in ('common.py', 'trace.py', 'replay.py', 'workload.py', 'environment.py', 'carbon.py', 'probe_resume.py')]
    files += ['scripts/' + name for name in ('fixed_max_walltime.py', 'opportunity_policy.py', 'opportunity_traces.py', 'opportunity_trial.py', 'opportunity_report.py')]
    source_hashes = {}
    for name in files:
        staged = Path(__file__).parent / Path(name).name
        path = staged if name.startswith('scripts/opportunity_') else root / name
        source_hashes[name] = digest(path)
    work = FixedMaxWalltimeWorkload(Workload(raw['workload'], raw['cluster']['allowed_nodes'],
                                          raw['cluster']['max_request_seconds'], raw['cluster']['walltime_resolution_seconds']))
    policy = OpportunityPolicy(work, design['pressure_cap_hours'])
    contract = {'kind': 'opportunity-v1', 'design': design, 'profile_binding': binding,
                'profile_config_sha256': digest(root / design['profile_config']), 'ci_sha256': digest(ci_path),
                'source_sha256': source_hashes, 'request_rule': design['target_request_rule'],
                'cost_references': {'time_hours': policy.time_reference, 'nodehours': policy.nodehour_reference},
                'policy': 'public-request-calendar + remaining-work weighted cost; pressure clipped at 48h; no training',
                'method_count': len(methods(design)), 'power_rho': 1.0,
                'ci_usage': 'Policy uses node-hours. Score identical allocations with constant400 and ERCOT archive; no future CI input.'}
    return design, raw, ci, work, contract


def make_bundle(raw, work, ci, origin, end, timeout_hours):
    scenario = {'purpose': 'synthetic', 'panel': 'opportunity-v1-constructed-C84',
                'cluster': deepcopy(raw['cluster']), 'workload': deepcopy(raw['workload']),
                'power': deepcopy(raw['power']), 'execution': {'max_episode_seconds': timeout_hours * 3600}}
    # Keep both power coefficients in raw accounting, but comparisons use rho=1.
    return SimpleNamespace(raw=scenario, workload=work, ci=ci, trace_end=end,
                           splits={'validation': (origin, end)})


def run_method(bundle, initial, episode, method, nodes, alpha, policy, design):
    env = Environment(bundle, episode, initial, method)
    decisions = []
    started = time.monotonic()
    while env.status == 'running':
        selected = nodes
        if alpha is not None:
            selected, decision = policy.select(env.replay.visible(), env.remaining, not env.chunks, alpha)
            decisions.append(decision)
        env.step(selected)
    summary = env.summary()
    require(all(c['requested_walltime_hours'] == 48 for c in env.chunks), 'Target request protocol diverged')
    if env.status == 'completed':
        require(summary['completed_updates'] == bundle.workload.total_updates and not summary['remaining_updates'],
                'Completed outcome lost work')
    summary.update(alpha=alpha, constant_carbon_kg_per_kappa=(summary['nodehours'] * design['constant_ci_g_per_kwh'] / 1000
                                                            if summary['nodehours'] is not None else None),
                   ercot_carbon_kg_per_kappa=(summary['carbon_g_per_kappa']['1.0'] / 1000
                                              if summary['carbon_g_per_kappa'] is not None else None),
                   selected_nodes=[c['selected_nodes'] for c in env.chunks],
                   replay_wall_seconds=time.monotonic() - started)
    return {'summary': summary, 'chunks': env.chunks, 'decisions': decisions}


def validate_saved(record, contract_hash, episode, allowed_methods):
    require(record['contract_sha256'] == contract_hash and record['episode'] == episode, 'Saved episode input changed')
    require(set(record['methods']) <= set(allowed_methods), 'Unknown saved method')
    for method, result in record['methods'].items():
        require(canonical_hash(result['payload']) == result['sha256'], 'Saved method payload hash mismatch')
        data = result['payload']
        require(data['summary']['method'] == method and data['summary']['episode_id'] == episode['episode_id'] and
                data['summary']['initial_arrival_utc'] == episode['arrival_utc'], 'Saved result identity mismatch')
        require(data['summary']['final_status'] in {'completed', 'censored', 'timeout'}, 'Incomplete saved method')
        require(all(c['requested_walltime_hours'] == 48 for c in data['chunks']), 'Saved request protocol mismatch')


def run_seed(root, design_path, output, scenario, seed, resume):
    design, raw, ci, work, contract = load_inputs(root, design_path)
    folder = Path(output) / scenario / f'seed-{seed}'
    contract_hash = canonical_hash(contract)
    origin = timestamp(design['origin_utc'])
    end = origin + timedelta(hours=design['trace_hours'])
    jobs, rows = generate_trace(design, scenario, seed)
    cohort = generate_cohort(design, seed)
    require(jobs and all(0 < j.nodes <= 84 for j in jobs), 'Invalid synthetic job widths')
    seed_plan = {'contract_sha256': contract_hash, 'scenario': scenario, 'seed': seed,
                 'trace_sha256': canonical_hash(rows), 'cohort_sha256': canonical_hash(cohort),
                 'trace_summary': trace_summary(design, rows, 84), 'status': 'declared_inputs'}
    with probe_lock(folder, resume, '.opportunity.lock'):
        marker = folder / 'inputs.json'
        if marker.exists():
            require(load_json(marker) == seed_plan, 'Seed inputs/code changed; use a fresh output directory')
            require(load_json(folder / 'trace.json') == rows and load_json(folder / 'cohort.json') == cohort, 'Saved generated inputs changed')
        else:
            save_json(folder / 'trace.json', rows)
            save_json(folder / 'cohort.json', cohort)
            save_json(marker, seed_plan)
        initial = Replay(jobs, origin, end, raw['cluster'], history_seconds=0,
                         sample_seconds=3600, initialize_from_observed=False)
        bundle = make_bundle(raw, work, ci, origin, end, design['episode_timeout_hours'])
        policy = OpportunityPolicy(work, design['pressure_cap_hours'])
        method_grid = methods(design)
        expected_methods = [m[0] for m in method_grid]
        paths = []
        for i, item in enumerate(cohort):
            path = folder / 'episodes' / (item['episode_id'] + '.json')
            paths.append(path)
            record = load_json(path) if path.exists() else {'contract_sha256': contract_hash, 'episode': item, 'methods': {}}
            validate_saved(record, contract_hash, item, expected_methods)
            if set(record['methods']) == set(expected_methods):
                print(f'{scenario}/seed{seed}/{i+1:02d}: restored all {len(method_grid)} outcomes', flush=True)
                continue
            arrival = timestamp(item['arrival_utc'])
            initial.advance_to(arrival, before_dispatch=True)
            episode = SimpleNamespace(arrival=arrival, episode_id=item['episode_id'], split='validation',
                                      budget_hours=design['episode_timeout_hours'])
            for method, n, alpha in method_grid:
                if method in record['methods']:
                    continue
                payload = run_method(bundle, initial, episode, method, n, alpha, policy, design)
                record['methods'][method] = {'payload': payload, 'sha256': canonical_hash(payload)}
                save_json(path, record)  # Resume at completed method; no checkpoint binaries.
                s = payload['summary']
                print(f'{scenario}/seed{seed}/{i+1:02d} {method}: {s["final_status"]}, '
                      f'nodes={s["selected_nodes"]}, TAT={s["tat_hours"]}, '
                      f'nodeh={s["nodehours"]}, replay={s["replay_wall_seconds"]:.2f}s', flush=True)
        save_json(folder / 'complete.json', {'contract_sha256': contract_hash, 'outcomes': len(cohort) * len(method_grid),
                                             'episode_sha256': {str(p.relative_to(folder)): digest(p) for p in paths}})
    return f'{scenario}/seed-{seed}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', default='all', help='all or a scenario name from the selected design')
    parser.add_argument('--design', default='configs/opportunity-v1-seed11.json')
    parser.add_argument('--repo-root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--workers', type=int, default=min(4, int(os.environ.get('SLURM_CPUS_PER_TASK', '1'))))
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--check', action='store_true', help='Read-only input check; no simulation or training')
    parser.add_argument('--report-only', action='store_true', help='Rebuild plots from local JSON text; no replay')
    args = parser.parse_args()
    require(args.workers > 0, 'workers must be positive')
    if args.report_only:
        from opportunity_report import report_all
        report_all((args.output or args.repo_root / 'results/opportunity-v1-xl-e050-seed11').resolve())
        return
    from frontera_old_gpt import require_allocation
    require_allocation(args.check)
    root = args.repo_root.resolve()
    design_path = (root / args.design).resolve()
    design, raw, ci, work, contract = load_inputs(root, design_path)
    require(args.trace == 'all' or args.trace in design['scenarios'], 'Unknown trace scenario')
    names = list(design['scenarios']) if args.trace == 'all' else [args.trace]
    output = (args.output or root / 'results/opportunity-v1-xl-e050-seed11').resolve()
    require(output != root and output not in root.parents, 'Output cannot be the repository root or its ancestor')
    # Matplotlib is the sole optional dependency; fail early on compute if absent.
    import importlib.util
    require(importlib.util.find_spec('matplotlib') is not None, 'Install requirements-plots.txt in the carbon environment')
    print(f'Python: {sys.executable}\nSynthetic mechanism experiment, NO training/checkpoints/wait probes.', flush=True)
    print(f'Traces={names}; seeds={design["seeds"]}; arrivals/seed={design["episodes_per_seed"]}; methods={len(methods(design))}', flush=True)
    print(f'XL/e050, C84, actions4/8/16/32, all target requests48h; output={output}', flush=True)
    if (output / 'run-plan.json').exists():
        require(load_json(output / 'run-plan.json') == contract, 'Output contract differs; use a new output directory')
    for name in names:
        for seed in design['seeds']:
            jobs, rows = generate_trace(design, name, seed)
            require(jobs and all(0 < j.nodes <= 84 and j.runtime <= j.requested for j in jobs), 'Invalid generated trace')
            print(f'{name}/seed{seed}: {trace_summary(design, rows, 84)}', flush=True)
            if (output / name / f'seed-{seed}').exists():
                require(args.resume or args.check, 'Output exists; append --resume')
    if args.check:
        print('Input/dependency checks passed. No replay or training executed.', flush=True)
        return
    # Only metadata creation is shared between concurrently submitted scenarios.
    from opportunity_report import metadata_lock
    with metadata_lock(output / '.plan.lock'):
        plan_path = output / 'run-plan.json'
        if plan_path.exists():
            require(load_json(plan_path) == contract, 'Concurrent output contract differs')
        else:
            require(not any(p for p in output.iterdir() if p.name != '.plan.lock'), 'Unrecognized output directory')
            save_json(plan_path, contract)
    tasks = [(str(root), str(design_path), str(output), name, seed, args.resume)
             for name in names for seed in design['seeds']]
    from opportunity_report import report_scenario, report_all
    with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as pool:
        futures = {pool.submit(run_seed, *task): task[3] for task in tasks}
        completed = {name: 0 for name in names}
        for future in as_completed(futures):
            print('Completed ' + future.result(), flush=True)
            name = futures[future]
            completed[name] += 1
            if completed[name] == len(design['seeds']):
                report_scenario(output, name)
    # Global summary uses a short lock, allowing separate --trace submissions.
    report_all(output)
    print(f'Complete. Inspect {output}/summary.md and each scenario/curves-constant.png or curves-ercot.png', flush=True)


if __name__ == '__main__':
    main()
