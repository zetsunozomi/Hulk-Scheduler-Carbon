"""Re-score paired fixed outcomes and all weighted PPO checkpoints."""

from collections import Counter
import csv
from statistics import fmean

from carbon.common import require
from carbon.runner import write_manifest
from weighted_policy import cost


def summarize(rows, alpha, rho, normalizers):
    require(rows and all(r['final_status'] == 'completed' for r in rows), 'Need complete paired outcomes')
    return {'outcomes': len(rows), 'mean_tat_hours': fmean(r['tat_hours'] for r in rows),
            'mean_carbon_g_per_kappa': {key: fmean(r['carbon_g_per_kappa'][key] for r in rows)
                                      for key in rows[0]['carbon_g_per_kappa']},
            'mean_nodehours': fmean(r['nodehours'] for r in rows),
            'mean_weighted_cost': fmean(cost(r['tat_hours'], r['carbon_g_per_kappa'][rho], alpha, normalizers) for r in rows)}


def write_readout(output, source, settings, normalizers):
    from old_gpt_trial import records
    fixed = records(source/'fixed-validation/episodes.jsonl')
    points = []
    for alpha in settings['alphas']:
        for nodes in (4, 8, 16, 32):
            group = [r for r in fixed if r['method'] == f'Fixed-{nodes}']
            points.append({'alpha': alpha, 'method': f'Fixed-{nodes}', 'iteration': None,
                           **summarize(group, alpha, settings['rho'], normalizers)})
    checkpoints = []
    for iteration in settings['validation_iterations']:
        rows = records(output/f'validation-{iteration:06d}/episodes.jsonl')
        for alpha in settings['alphas']:
            group = [r for r in rows if r['alpha'] == alpha]
            require(len(group) == len(fixed)//4 and len({r['episode_id'] for r in group}) == len(group),
                    'Weighted/fixed validation cohort size differs')
            best = min((p for p in points if p['alpha'] == alpha), key=lambda p: p['mean_weighted_cost'])
            baseline = {r['episode_id']: r for r in fixed if r['method'] == best['method']}
            require(set(baseline) == {r['episode_id'] for r in group}, 'Weighted/fixed validation arrivals differ')
            summary = summarize(group, alpha, settings['rho'], normalizers)
            differences = [cost(r['tat_hours'], r['carbon_g_per_kappa'][settings['rho']], alpha, normalizers) -
                           cost(baseline[r['episode_id']]['tat_hours'], baseline[r['episode_id']]['carbon_g_per_kappa'][settings['rho']], alpha, normalizers)
                           for r in group]
            frequencies = Counter(n for r in group for n in r['selected_nodes'])
            checkpoints.append({'alpha': alpha, 'method': 'Weighted-PPO', 'iteration': iteration, **summary,
                                'best_fixed': best['method'], 'best_fixed_cost': best['mean_weighted_cost'],
                                'improvement_percent': 100*(1-summary['mean_weighted_cost']/best['mean_weighted_cost']),
                                'paired_wins': sum(d < -1e-10 for d in differences),
                                'paired_ties': sum(abs(d) <= 1e-10 for d in differences),
                                'switched_episodes': sum(len(set(r['selected_nodes'])) > 1 for r in group),
                                'action_counts': dict(sorted(frequencies.items())),
                                'initial_mean_probabilities': [fmean(r['initial_probabilities'][i] for r in group) for i in range(4)],
                                'initial_argmax_counts': dict(Counter((4,8,16,32)[max(range(4), key=lambda i: r['initial_probabilities'][i])] for r in group))})
    summary = {'kind': 'old_gpt_summary_v1', 'status': 'complete', 'settings': settings,
               'reward_normalizers': normalizers, 'fixed': points, 'checkpoints': checkpoints,
               'comparison': 'same validation arrivals; best fixed chosen on validation, no significance claim',
               'fixed_mix': 'For a fixed linear scalar cost, a mixture expectation cannot beat the cheapest fixed constituent.'}
    write_manifest(output/'old-gpt-summary.json', summary)
    fields = ['iteration', 'alpha', 'method', 'mean_tat_hours', 'mean_carbon_g_per_kappa', 'mean_weighted_cost', 'mean_nodehours']
    with (output/'weighted-curves.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore'); writer.writeheader()
        writer.writerows({**p, 'mean_carbon_g_per_kappa': p['mean_carbon_g_per_kappa'][settings['rho']]} for p in [*points, *checkpoints])
    lines = ['# Old GPT-2 C84 weighted PPO', '',
             'Fixed and dynamic use the same configuration walltime cap (48h in development).', '',
             f'Observation-only rounds: {settings["observation_iterations"]}; update rounds: '
             f'{settings["iterations"]-settings["observation_iterations"]}; alpha: {settings["alphas"]}.', '',
             f'Frozen training means ({normalizers["episodes"]} complete episodes): '
             f'TAT={normalizers["tat_hours"]:.6f}h; carbon={normalizers["carbon_g_per_kappa"]:.6f} g/kappa at rho={settings["rho"]}.', '',
             'Minimize J = alpha*TAT/Tref + (1-alpha)*carbon/Cref; PPO reward = -J.', '',
             '| Round | alpha | PPO cost | Best fixed | Cost gain % | TAT h | Switches | Paired wins |',
             '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for p in checkpoints:
        lines.append(f'| {p["iteration"]} | {p["alpha"]:g} | {p["mean_weighted_cost"]:.6f} | {p["best_fixed"]} | '
                     f'{p["improvement_percent"]:+.3f} | {p["mean_tat_hours"]:.3f} | '
                     f'{p["switched_episodes"]}/{p["outcomes"]} | {p["paired_wins"]}/{p["outcomes"]} |')
    lines += ['', 'Positive gain means lower weighted cost. Best fixed is selected on this validation cohort.',
              'Switching and sampled action diversity alone do not establish useful state feedback.',
              'Carbon is modeled job-attributed operational carbon per unknown common kappa, not measured emissions.',
              'No deadline constraint, violation penalty, dual update, or wait-predictor input is used.',
              'All declared checkpoints are reported; no test-set claim or statistical significance claim.',
              'For this linear objective, optimal episode-level Fixed-Mix has the same expected cost as Best-Fixed.', '']
    (output/'old-gpt-summary.md').write_text('\n'.join(lines))
    plot(output, points, checkpoints, settings, normalizers)


def plot(output, fixed, checkpoints, settings, normalizers):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    iterations = settings['validation_iterations']
    fig, axes = plt.subplots((len(iterations)+1)//2, 2, figsize=(12, 5*((len(iterations)+1)//2)),
                             squeeze=False, constrained_layout=True)
    baseline = [p for p in fixed if p['alpha'] == settings['alphas'][0]]
    for ax, iteration in zip(axes.flat, iterations):
        dynamic = [p for p in checkpoints if p['iteration'] == iteration]
        def xy(points):
            return ([p['mean_carbon_g_per_kappa'][settings['rho']]/1000 for p in points],
                    [p['mean_tat_hours'] for p in points])
        ax.plot(*xy(baseline), 'o-', color='black', label='Fixed 4/8/16/32 nodes')
        ax.plot(*xy(dynamic), 'o-', color='#e3ad00', label='Dynamic, alpha order')
        labels = {}
        for p in [*baseline, *dynamic]:
            key = (p['mean_carbon_g_per_kappa'][settings['rho']]/1000, p['mean_tat_hours'])
            labels.setdefault(key, []).append(p['method'] if p['method'].startswith('Fixed') else f'a={p["alpha"]:g}')
        for pos, names in labels.items():
            ax.annotate(', '.join(names), pos, xytext=(4, 5), textcoords='offset points', fontsize=8)
        ax.set(title=f'Validation iteration {iteration}', xlabel=f'Modeled carbon (kg / kappa), rho={settings["rho"]}',
               ylabel='Mean turnaround time (hours)')
        ax.grid(alpha=.2); ax.legend(fontsize=8); ax.margins(.15)
    for ax in list(axes.flat)[len(iterations):]:
        ax.set_visible(False)
    for suffix in ('png', 'pdf'):
        fig.savefig(output/f'weighted-curves.{suffix}', dpi=160)
    plt.close(fig)
