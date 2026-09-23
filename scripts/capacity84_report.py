"""Descriptive development curves from completed C84 outcomes only."""

import csv
import math
import os
from pathlib import Path
import tempfile

from carbon.runner import write_manifest


COLORS = {'Fixed-4': '#777777', 'Fixed-16': '#8c6bb1', 'Fixed-64': '#a6761d',
          'Best-Fixed': '#222222', 'Fixed-Mix': '#009e73',
          'Plan-once': '#56b4e9', 'Rollout-MPC': '#d55e00'}


def curve_rows(report):
    rows = []
    points = report['baselines']+[p for c in report['checkpoints'] for p in c['operating_points']]
    for point in points:
        s = point['summary']
        row = {'method': point['method'], 'budget_hours': point['budget_hours'],
               'selected_fixed': point.get('selected_method'), 'reason': point.get('reason'),
               'statistic_type': 'validation_fitted_expectation' if point['method'] == 'Fixed-Mix' else 'validation_outcome_mean',
               'mean_tat_hours': None, 'p95_tat_hours': None, 'C': None,
               'miss_lower': None, 'miss_upper': None, 'empirical_target_met': False,
               'mean_nodehours': None, 'mean_chunks': None, 'scale_change_fraction': None,
               'completed': None, 'outcomes': None,
               'mean_carbon_lower_endpoint': None, 'mean_carbon_upper_endpoint': None}
        if s is not None:
            row.update({key: s.get(key) for key in ('mean_tat_hours', 'p95_tat_hours', 'mean_nodehours',
                                                   'mean_chunks', 'completed', 'outcomes', 'empirical_target_met')})
            row.update(C=s['worst_normalized_carbon'], miss_lower=s['miss_rate_bounds'][0],
                       miss_upper=s['miss_rate_bounds'][1],
                       scale_change_fraction=s['observed_scale_change_fraction'])
            carbon = s.get('mean_modeled_carbon_g_per_kappa')
            if carbon:
                ends = sorted(carbon, key=float)
                row.update(mean_carbon_lower_endpoint=carbon[ends[0]], mean_carbon_upper_endpoint=carbon[ends[-1]])
        rows.append(row)
    return rows


def csv_file(path, rows, fields):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def render_curves(output, report, rows):
    # All figures are standalone artifacts; no home-directory cache writes.
    with tempfile.TemporaryDirectory(prefix='carbon-c84-plots-') as cache:
        previous = {k: os.environ.get(k) for k in ('MPLCONFIGDIR', 'XDG_CACHE_HOME')}
        for k, value in previous.items():
            if value is None:
                os.environ[k] = cache
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            def save(fig, name):
                for suffix in ('png', 'pdf'):
                    temporary = output/f'.{name}.tmp.{suffix}'
                    fig.savefig(temporary, dpi=170, bbox_inches='tight')
                    temporary.replace(output/f'{name}.{suffix}')
                plt.close(fig)

            def draw(ax, points, field, color, label, style='-'):
                points = sorted(points, key=lambda p:p['budget_hours'])
                x = [p['budget_hours'] for p in points]
                y = [p[field] if p[field] is not None else math.nan for p in points]
                ax.plot(x, y, color=color, linestyle=style, label=label, linewidth=1.5, alpha=.85)
                for a, b, p in zip(x, y, points):
                    if math.isfinite(b):
                        ax.scatter([a], [b], color=color, edgecolors=color,
                                   facecolors=color if p['empirical_target_met'] else 'white', s=32, zorder=4)

            latest = max((c['iteration'] for c in report['checkpoints']), default=None)
            methods = list(COLORS)+([f'PPO-{latest}'] if latest is not None else [])
            fig, axes = plt.subplots(1, 3, figsize=(15, 5.6))
            colors = {**COLORS, f'PPO-{latest}': '#0072b2'}
            for method in methods:
                selected = [r for r in rows if r['method'] == method]
                for ax, field in zip(axes, ('C', 'miss_upper', 'mean_tat_hours')):
                    draw(ax, selected, field, colors[method], method,
                         '--' if method.startswith('Fixed-') else '-')
            budgets = [t['budget_hours'] for t in report['budget_grid']['slider']['ticks']]
            for ax, label in zip(axes, ('Worst endpoint normalized carbon C', 'Deadline-miss rate / censor upper bound', 'Mean turnaround time (hours)')):
                ax.set_xlabel('Completion budget D (hours)'); ax.set_ylabel(label)
                ax.set_xticks(budgets, [f'{b:.3g}' for b in budgets]); ax.grid(alpha=.2)
                ax.spines[['top', 'right']].set_visible(False)
            axes[1].axhline(.05, color='#222222', linestyle=':', linewidth=1)
            axes[1].set_ylim(-.03, 1.03)
            fig.suptitle(report['panel']+' | '+('baselines prepared' if latest is None else f'PPO checkpoint {latest}'))
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(.5, .01), frameon=False, fontsize=9)
            fig.text(.5, .16, 'Development validation; no uncertainty intervals. Open markers: observed target not met.\n'
                     'Fixed-Mix is fitted on this validation cohort. Lines connect evaluated ticks only.', ha='center', va='center', fontsize=8)
            fig.tight_layout(rect=(0, .24, 1, .95)); save(fig, 'budget-curves')

            if not report['checkpoints']:
                return
            fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
            for index, c in enumerate(report['checkpoints']):
                method = f"PPO-{c['iteration']}"; color = plt.get_cmap('viridis')(.15+.2*index)
                selected = [r for r in rows if r['method'] == method]
                draw(axes[0], selected, 'C', color, method)
                draw(axes[1], selected, 'scale_change_fraction', color, method)
            for method in ('Best-Fixed', 'Fixed-Mix'):
                draw(axes[0], [r for r in rows if r['method'] == method], 'C', COLORS[method], method, '--')
            for ax, label in zip(axes, ('Worst endpoint normalized carbon C', 'Episodes changing scale / all episodes')):
                ax.set_xlabel('Completion budget D (hours)'); ax.set_ylabel(label); ax.grid(alpha=.2)
                ax.set_xticks(budgets, [f'{b:.3g}' for b in budgets]); ax.legend(fontsize=8, frameon=False)
            axes[1].set_ylim(-.03, 1.03)
            fig.suptitle('All scheduled checkpoints | descriptive development validation')
            fig.tight_layout(); save(fig, 'checkpoint-curves')

            fig, axes = plt.subplots(1, len(report['checkpoints']), figsize=(4*len(report['checkpoints']), 4), squeeze=False)
            for ax, c in zip(axes[0], report['checkpoints']):
                points = c['actor']['operating_points']; bottom = [0.]*len(points)
                for i, n in enumerate(report['nodes']):
                    values = [p['mean_probabilities'][i] for p in points]
                    ax.bar(range(len(points)), values, bottom=bottom, label=f'{n} nodes')
                    bottom = [a+b for a, b in zip(bottom, values)]
                ax.set_title(f"Checkpoint {c['iteration']}")
                ax.set_xticks(range(len(points)), [f'{b:.3g}' for b in budgets], rotation=25)
                ax.set_xlabel('Budget hours'); ax.set_ylim(0, 1); ax.set_ylabel('Mean first-action probability')
            axes[0][-1].legend(fontsize=8)
            fig.suptitle('Behavior probabilities; diversity does not establish useful feedback')
            fig.tight_layout(); save(fig, 'policy-behavior')
        finally:
            for k, value in previous.items():
                if value is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = value


def write_readout(output, report):
    output = Path(output)
    rows = curve_rows(report)
    csv_file(output/'budget-curves.csv', rows, list(rows[0]))
    behavior = []
    for c in report['checkpoints']:
        for point, detail in zip(c['actor']['operating_points'], c['actor']['within_budget_behavior']):
            for i, n in enumerate(report['nodes']):
                behavior.append({'iteration': c['iteration'], 'budget_multiplier': point['budget_multiplier'],
                                 'nodes': n, 'arrivals': point['arrivals'],
                                 'mean_probability': point['mean_probabilities'][i],
                                 'min_probability': detail['first_probability_ranges'][i][0],
                                 'max_probability': detail['first_probability_ranges'][i][1],
                                 'sampled_first_count': detail['sampled_first_nodes'].get(n, detail['sampled_first_nodes'].get(str(n), 0)),
                                 'first_argmax_count': detail['first_argmax_nodes'].get(n, detail['first_argmax_nodes'].get(str(n), 0))})
    if behavior:
        csv_file(output/'policy-behavior.csv', behavior, list(behavior[0]))
    lines = ['# Frontera–7B C84 development results', '',
             '84 nodes; actions 4/16/64. Fresh independent product actors with shared critic.',
             'Same work, chunk rules, overhead, trace/CI/calendar and learning settings as C128; new capacity-specific references/budgets.',
             'All comparisons use paired development validation arrivals. No test access, confidence intervals or formal method adoption.',
             'C is the maximum of the two normalized mean modeled carbon endpoint costs. Fixed-Mix is a validation-fitted expectation.', '',
             '| Method | D(h) | C | Mean TAT(h) | Miss lower–upper | Changed scale |',
             '|---|---:|---:|---:|---:|---:|']
    fmt = lambda x: '—' if x is None else f'{x:.6f}'
    for r in rows:
        miss = '—' if r['miss_upper'] is None else f"{r['miss_lower']:.1%}–{r['miss_upper']:.1%}"
        changed = '—' if r['scale_change_fraction'] is None else f"{r['scale_change_fraction']:.1%}"
        lines.append(f"| {r['method']} | {r['budget_hours']:.3f} | {fmt(r['C'])} | {fmt(r['mean_tat_hours'])} | {miss} | {changed} |")
    lines += ['', 'Unavailable Best-Fixed/Fixed-Mix points are infeasible at the empirical target; all raw fixed points remain visible.', '',
              '## Queue evidence', '']
    for split, q in report['queue'].items():
        lines.append(f"- {split}: {q['rows']} probes, {q['censored_rows']} censored; queue summary: {q['queue_summary']}")
    lines += ['', '## Behavior', '', 'Mean first-action probabilities in action order '+str(report['nodes'])+':', '']
    for c in report['checkpoints']:
        for p in c['actor']['operating_points']:
            lines.append(f"- checkpoint {c['iteration']}, beta={p['budget_multiplier']:.6f}: "+', '.join(f'{x:.4%}' for x in p['mean_probabilities']))
    lines += ['', '## Artifacts', '',
              '- `budget-curves.png/pdf/csv`: all baselines and the latest evaluated checkpoint; carbon, misses and actual turnaround versus budget.',
              '- `checkpoint-curves.png/pdf`: all scheduled checkpoint cost and scale-change curves.',
              '- `policy-behavior.png/pdf/csv`: first-action probability means/ranges, sampled counts and argmax counts.',
              '- `capacity84-summary.json`: exact values, sequences, stage bindings and settings.', '',
              f"Status: {report['status']}; validation checkpoints: {[c['iteration'] for c in report['checkpoints']]}", '']
    render_curves(output, report, rows)
    temporary = output/'capacity84-summary.md.tmp'
    temporary.write_text('\n'.join(lines), encoding='utf-8'); temporary.replace(output/'capacity84-summary.md')
    # Completion is published only after tables and figures are successfully written.
    write_manifest(output/'capacity84-summary.json', report)
