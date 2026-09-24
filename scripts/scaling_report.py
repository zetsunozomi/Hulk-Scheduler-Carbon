"""Full time--carbon curves for declared analytic profiles; no best-fixed selection."""

from collections import Counter
import hashlib
import json
from statistics import fmean

from carbon.common import load_json, require
from carbon.runner import write_manifest


def records(directory):
    path = directory/'episodes.jsonl'
    raw = path.read_bytes()
    seal = load_json(directory/'stage-seal.json')
    require(hashlib.sha256(raw).hexdigest() == seal['episodes.jsonl'], 'Episode seal differs')
    rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    require(rows and all(r['split'] == 'validation' and r['final_status'] == 'completed'
                        and not r['censor_flag'] and r['exposure_is_complete'] for r in rows),
            'Require all completed validation outcomes, including every arrival')
    return rows


def cohort(rows):
    result = {(r['episode_id'], r['initial_arrival_utc'], r['completed_updates']) for r in rows}
    require(len(result) == len(rows), 'Duplicate outcome in configuration')
    return result


def point(rows, **labels):
    def actions(row):
        # Canonical fixed episode records omit selected_nodes; their method and
        # chunk count completely specify the allocation sequence.
        if row['method'].startswith('Fixed-'):
            return [int(row['method'].removeprefix('Fixed-'))] * row['chunk_count']
        return row['selected_nodes']
    return dict(labels, outcomes=len(rows), mean_tat_hours=fmean(r['tat_hours'] for r in rows),
                mean_carbon_g_per_kappa={rho: fmean(r['carbon_g_per_kappa'][rho] for r in rows)
                                        for rho in ('0.25', '1.0')},
                mean_nodehours=fmean(r['nodehours'] for r in rows),
                mean_chunks=fmean(r['chunk_count'] for r in rows),
                switched_episodes=sum(len(set(actions(r))) > 1 for r in rows),
                action_counts=dict(sorted(Counter(str(n) for r in rows for n in actions(r)).items())))


def write_readout(output, source, settings, normalizers):
    binding = load_json(output/'run-plan.json')['source_binding']['profile_inputs']
    require(binding['kind'] == 'analytic_scaling_profile_v2', 'Wrong analytic profile binding')
    policy = load_json(source/'references.json')['fixed_request_policy']
    fixed_rows = records(source/'fixed-validation')
    require({r['method'] for r in fixed_rows} == {f'Fixed-{n}' for n in (4,8,16,32)}, 'Fixed scales differ')
    fixed, expected = [], None
    for n in (4, 8, 16, 32):
        group = [r for r in fixed_rows if r['method'] == f'Fixed-{n}']
        current = cohort(group)
        expected = current if expected is None else expected
        require(current == expected, 'Fixed cohorts differ')
        for alpha in settings['alphas']:
            fixed.append(point(group, method=f'Fixed-{n}', alpha=alpha, iteration=None))
    checkpoints = []
    for iteration in settings['validation_iterations']:
        rows = records(output/f'validation-{iteration:06d}')
        require({r['alpha'] for r in rows} == set(settings['alphas']), 'Validation alpha grid differs')
        require(all(r['iteration'] == iteration for r in rows), 'Validation iteration differs')
        for alpha in settings['alphas']:
            group = [r for r in rows if r['alpha'] == alpha]
            require(cohort(group) == expected, 'Dynamic/fixed cohorts or work differ')
            checkpoints.append(point(group, method='Weighted-PPO', alpha=alpha, iteration=iteration))
    summary = {'kind': 'analytic_scaling_summary_v2', 'status': 'complete', 'settings': settings,
               'profile_inputs': binding, 'fixed_request_policy': policy,
               'reward_normalizers': normalizers, 'fixed': fixed, 'checkpoints': checkpoints,
               'comparison': 'Full fixed-scale and alpha-ordered dynamic curves; inspect lower-left shift. No per-alpha best-fixed ranking.',
               'measurement_status': 'Analytically assumed throughput, modeled carbon, historical-demand replay.'}
    write_manifest(output/'scaling-summary.json', summary)
    lines = [f'# Scaling sensitivity: {binding["model"]} / {binding["scenario"]}', '',
             f'T16={binding["anchor_training_hours"]:g}h; efficiency32/4={binding["efficiency32_vs4"]:.0%}; '
             f'seed={settings["seed"]}; rho={settings["rho"]}.', '',
             'These throughput profiles are assumed inputs, not measured training speeds.',
             'Compare the full dynamic time-carbon curve with all four fixed points; lower-left is better.',
             f'Fixed requests {policy["requested_seconds"]/3600:g}h on every chunk; '
             'dynamic requests rounded planned duration. Gains include both node choice and request sizing.', '',
             '| Iteration | Configuration | Carbon kg CO2/kappa | TAT h | Node-hours | Mean chunks | Switched episodes |',
             '|---|---|---:|---:|---:|---:|---:|']
    for p in [*[r for r in fixed if r['alpha'] == settings['alphas'][0]], *checkpoints]:
        label = p['method'] if p['iteration'] is None else f'alpha={p["alpha"]:g}'
        iteration = '-' if p['iteration'] is None else p['iteration']
        lines.append(f'| {iteration} | {label} | {p["mean_carbon_g_per_kappa"][settings["rho"]]/1000:.6f} | '
                     f'{p["mean_tat_hours"]:.6f} | {p["mean_nodehours"]:.6f} | '
                     f'{p["mean_chunks"]:.3f} | {p["switched_episodes"]}/{p["outcomes"]} |')
    lines += ['', 'All declared checkpoints and alpha points are retained, including overlaps and dominated points.',
              'Do not interpret a fitted connecting segment as a measured intermediate scale.',
              'The 48h cap is fixed; model length and efficiency can change the number of chunks.',
              'Development validation only. No test-set or statistical-significance claim.', '']
    (output/'scaling-summary.md').write_text('\n'.join(lines))
