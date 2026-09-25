"""Text-only result ingestion, full fixed/dynamic curves and frontier diagnostics."""
import argparse
from contextlib import contextmanager
import csv
import fcntl
import hashlib
import json
from pathlib import Path
from statistics import mean

from carbon.common import digest, load_json, require


def value_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    pending = path.with_name(path.name + '.pending')
    pending.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    pending.replace(path)


@contextmanager
def metadata_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def lower_frontier(points):
    """Lower convex envelope, allowing mixtures of the fixed policies."""
    candidates = sorted(set(points))
    nondominated = []
    for x, y in candidates:
        if nondominated and (x == nondominated[-1][0] or y >= nondominated[-1][1]):
            continue
        nondominated.append((x, y))
    hull = []
    for p in nondominated:
        while len(hull) >= 2:
            a, b = hull[-2:]
            cross = (b[0]-a[0])*(p[1]-b[1]) - (b[1]-a[1])*(p[0]-b[0])
            if cross > 0:
                break
            hull.pop()
        hull.append(p)
    return hull


def frontier_time(hull, carbon):
    for x, y in hull:
        if abs(carbon-x) <= 1e-9:
            return y
    for a, b in zip(hull, hull[1:]):
        if a[0] <= carbon <= b[0]:
            return a[1] + (carbon-a[0])/(b[0]-a[0])*(b[1]-a[1])
    return None  # Never extrapolate beyond the measured fixed range.


def summarize(records, contract):
    design = contract['design']
    method_names = [f'Fixed-{n}' for n in (4, 8, 16, 32)] + [f'Dynamic-{a:.2f}' for a in design['alphas']]
    rows = []
    for method in method_names:
        data = [r['summary'] for r in records if r['summary']['method'] == method]
        require(data, f'Missing method {method}')
        complete = all(r['final_status'] == 'completed' for r in data)
        row = {'method': method, 'alpha': data[0]['alpha'], 'outcomes': len(data),
               'completed': sum(r['final_status'] == 'completed' for r in data),
               'mean_tat_hours': mean(r['tat_hours'] for r in data) if complete else None,
               'mean_nodehours': mean(r['nodehours'] for r in data) if complete else None,
               'mean_constant_carbon_kg_per_kappa': mean(r['constant_carbon_kg_per_kappa'] for r in data) if complete else None,
               'mean_ercot_carbon_kg_per_kappa': mean(r['ercot_carbon_kg_per_kappa'] for r in data) if complete else None,
               'mean_chunks': mean(len(r['selected_nodes']) for r in data),
               'switched_fraction': mean(len(set(r['selected_nodes'])) > 1 for r in data),
               'downscaled_fraction': mean(any(b < a for a, b in zip(r['selected_nodes'], r['selected_nodes'][1:])) for r in data)}
        rows.append(row)
    require(len({r['outcomes'] for r in rows}) == 1, 'Unpaired method cohorts')
    valid = all(r['completed'] == r['outcomes'] for r in rows)
    diagnostics = {'all_methods_complete': valid, 'arrival_count': rows[0]['outcomes']}
    for ci in ('constant', 'ercot'):
        field = f'mean_{ci}_carbon_kg_per_kappa'
        hull = lower_frontier([(r[field], r['mean_tat_hours']) for r in rows[:4]]) if valid else []
        deltas = []
        for r in rows:
            comparison = frontier_time(hull, r[field]) if valid else None
            delta = r['mean_tat_hours'] - comparison if comparison is not None else None
            r[f'{ci}_gap_vs_fixed_hull_hours'] = delta
            if r['alpha'] is not None and delta is not None:
                deltas.append(delta)
        diagnostics[ci] = {'fixed_lower_convex_frontier': hull,
                           'dynamic_points_below_fixed_hull': sum(d < -1e-7 for d in deltas),
                           'dynamic_points_above_fixed_hull': sum(d > 1e-7 for d in deltas),
                           'dynamic_points_on_fixed_hull': sum(abs(d) <= 1e-7 for d in deltas),
                           'dynamic_points_outside_fixed_carbon_range': sum(r['alpha'] is not None and r[f'{ci}_gap_vs_fixed_hull_hours'] is None for r in rows) if valid else None,
                           'distinct_dynamic_points': len({(round(r[field], 7), round(r['mean_tat_hours'], 7)) for r in rows[4:]}) if valid else None,
                           'best_gap_hours': min(deltas) if deltas else None,
                           'note': 'Negative gap means below the fixed lower convex frontier at equal carbon; outside-range points are retained without extrapolation.'}
    return rows, diagnostics


def plot_curves(folder, rows, scenario, valid):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for ci in ('constant', 'ercot'):
        field = f'mean_{ci}_carbon_kg_per_kappa'
        fixed = [r for r in rows[:4] if r[field] is not None]
        dynamic = [r for r in rows[4:] if r[field] is not None]
        fig, ax = plt.subplots(figsize=(7.5, 5.1))
        ax.plot([r[field] for r in fixed], [r['mean_tat_hours'] for r in fixed], 'o-', color='black', lw=1.9, label='Fixed 4/8/16/32')
        ax.plot([r[field] for r in dynamic], [r['mean_tat_hours'] for r in dynamic], 'o-', color='#E5B700', markeredgecolor='#7D6200', lw=2.2, label='Dynamic (no training)')
        for r in fixed:
            ax.annotate(r['method'].replace('Fixed-', ''), (r[field], r['mean_tat_hours']), xytext=(5, 7), textcoords='offset points', fontsize=9)
        groups = {}
        for r in dynamic:
            key = (round(r[field], 7), round(r['mean_tat_hours'], 7))
            groups.setdefault(key, []).append(f'{r["alpha"]:g}')
        for (x, y), alphas in groups.items():
            text = 'a=' + (','.join(alphas) if len(alphas) <= 3 else alphas[0] + '…' + alphas[-1])
            ax.annotate(text, (x, y), xytext=(5, -13), textcoords='offset points', fontsize=8, color='#806300')
        ax.set_xlabel('Modeled carbon (kg / kappa)')
        ax.set_ylabel('Mean turnaround time (hours)')
        label = 'constant CI = 400 g/kWh' if ci == 'constant' else 'ERCOT archival CI overlay'
        ax.set_title(f'{scenario} · {label}\nSynthetic arrivals · XL/e050 · all methods request 48h', fontsize=11)
        ax.margins(x=.16, y=.12)
        ax.grid(alpha=.2)
        ax.legend(fontsize=9)
        footer = 'All slider points shown, including duplicates and dominated points. Lines join evaluated settings.'
        if not valid:
            footer = 'INCOMPLETE COHORT: no frontier claim. Missing methods have unfinished episodes.'
        fig.text(.5, .012, footer, ha='center', fontsize=7)
        fig.tight_layout(rect=(0, .035, 1, 1))
        for ext in ('png', 'pdf'):
            fig.savefig(folder / f'curves-{ci}.{ext}', dpi=170)
        plt.close(fig)


def report_scenario(output, scenario):
    output = Path(output)
    contract = load_json(output / 'run-plan.json')
    design = contract['design']
    contract_hash = value_hash(contract)
    folder = output / scenario
    records, seed_reports, sources = [], {}, {}
    method_names = {f'Fixed-{n}' for n in (4, 8, 16, 32)} | {f'Dynamic-{a:.2f}' for a in design['alphas']}
    for seed in design['seeds']:
        seed_dir = folder / f'seed-{seed}'
        if not (seed_dir / 'complete.json').is_file():
            return None
        seal = load_json(seed_dir / 'complete.json')
        require(seal['contract_sha256'] == contract_hash, 'Seed completion contract mismatch')
        inputs = load_json(seed_dir / 'inputs.json')
        require(inputs['contract_sha256'] == contract_hash and inputs['scenario'] == scenario and inputs['seed'] == seed, 'Seed input identity mismatch')
        cohort = load_json(seed_dir / 'cohort.json')
        require(value_hash(cohort) == inputs['cohort_sha256'] and len(cohort) == design['episodes_per_seed'], 'Cohort was modified')
        require(value_hash(load_json(seed_dir / 'trace.json')) == inputs['trace_sha256'], 'Generated trace was modified')
        expected = {f'episodes/{e["episode_id"]}.json' for e in cohort}
        require(set(seal['episode_sha256']) == expected, 'Episode completion seal differs from cohort')
        seed_records = []
        for item in cohort:
            relative = f'episodes/{item["episode_id"]}.json'
            path = seed_dir / relative
            require(digest(path) == seal['episode_sha256'][relative], 'Completed episode file hash mismatch')
            record = load_json(path)
            require(record['contract_sha256'] == contract_hash and record['episode'] == item and set(record['methods']) == method_names,
                    'Episode/method identity mismatch')
            for method, data in record['methods'].items():
                require(value_hash(data['payload']) == data['sha256'], 'Outcome payload hash mismatch')
                require(data['payload']['summary']['method'] == method and
                        data['payload']['summary']['episode_id'] == item['episode_id'], 'Method payload identity mismatch')
                seed_records.append(data['payload'])
            sources[str(path.relative_to(output))] = digest(path)
        seed_rows, seed_diagnostics = summarize(seed_records, contract)
        seed_reports[str(seed)] = {'rows': seed_rows, 'diagnostics': seed_diagnostics, 'trace_summary': inputs['trace_summary']}
        records.extend(seed_records)
    rows, diagnostics = summarize(records, contract)
    with metadata_lock(folder / '.report.lock'):
        report = {'kind': 'constructed_mechanism_development', 'scenario': scenario, 'contract_sha256': contract_hash,
                  'rows': rows, 'diagnostics': diagnostics, 'per_seed': seed_reports, 'source_sha256': sources}
        write_json(folder / 'summary.json', report)
        with (folder / 'curves.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        lines = [f'# {scenario}: training-free synthetic curve', '',
                 'Constructed development workload; no empirical trace or measured energy claim.',
                 'Same 48h target request rule and same exogenous arrivals for all methods. No training or checkpoints.', '',
                 f'All paired outcomes complete: **{diagnostics["all_methods_complete"]}**; arrivals: {diagnostics["arrival_count"]}.', '',
                 '| Method | TAT h | Constant-CI kg/κ | ERCOT kg/κ | Constant frontier gap h | ERCOT frontier gap h | Switched |',
                 '|---|---:|---:|---:|---:|---:|---:|']
        def fmt(v):
            return 'incomplete/outside' if v is None else f'{v:.3f}'
        for r in rows:
            lines.append('| ' + r['method'] + ' | ' + ' | '.join(fmt(r[k]) for k in
                ('mean_tat_hours', 'mean_constant_carbon_kg_per_kappa', 'mean_ercot_carbon_kg_per_kappa',
                 'constant_gap_vs_fixed_hull_hours', 'ercot_gap_vs_fixed_hull_hours', 'switched_fraction')) + ' |')
        lines += ['', 'Negative gap = below the full fixed lower convex frontier at the same carbon. No extrapolation.',
                  'All slider values remain in the plot, including overlaps and losses. No outcome-based policy selection.',
                  'Per-seed curves, trace load, node-hours and path-switch rates are in summary.json; complete paths and visible decisions are in seed-*/episodes/.',
                  'The constant-CI panel isolates queue/scaling effects. The ERCOT panel rescores identical allocations; the policy never reads future CI.', '']
        (folder / 'summary.md').write_text('\n'.join(lines))
        plot_curves(folder, rows, scenario, diagnostics['all_methods_complete'])
    print(f'Curve ready: {folder}; constant={diagnostics["constant"]}; ERCOT={diagnostics["ercot"]}', flush=True)
    return report


def report_all(output):
    output = Path(output)
    contract = load_json(output / 'run-plan.json')
    reports = {}
    for name in contract['design']['scenarios']:
        if (output / name / 'summary.json').is_file():
            existing = load_json(output / name / 'summary.json')
            require(existing['contract_sha256'] == value_hash(contract), 'Scenario report input changed')
        result = report_scenario(output, name)
        if result is not None:
            reports[name] = result
    with metadata_lock(output / '.summary.lock'):
        # Re-read completed reports under the lock so concurrent scenario jobs
        # cannot replace a newer overview with a stale subset.
        complete = {}
        for name in contract['design']['scenarios']:
            path = output / name / 'summary.json'
            if path.exists():
                row = load_json(path)
                require(row['contract_sha256'] == value_hash(contract), 'Scenario report contract mismatch')
                complete[name] = row['diagnostics']
        write_json(output / 'summary.json', {'kind': contract['kind'], 'completed_scenarios': complete,
                                           'pending_scenarios': [n for n in contract['design']['scenarios'] if n not in complete]})
        lines = ['# RL-free opportunity curves', '', 'Synthetic mechanism development, no training. Negative gaps are favorable.', '',
                 '| Trace | Complete cohort | Constant CI: below / above | ERCOT: below / above | Best constant gap h |',
                 '|---|---|---:|---:|---:|']
        for name in contract['design']['scenarios']:
            if name not in complete:
                lines.append(f'| {name} | pending | — | — | — |')
                continue
            d = complete[name]
            lines.append(f'| {name} | {d["all_methods_complete"]} | '
                         f'{d["constant"]["dynamic_points_below_fixed_hull"]} / {d["constant"]["dynamic_points_above_fixed_hull"]} | '
                         f'{d["ercot"]["dynamic_points_below_fixed_hull"]} / {d["ercot"]["dynamic_points_above_fixed_hull"]} | '
                         f'{d["constant"]["best_gap_hours"]} |')
        lines += ['', 'Each trace directory has summary.md/json, curves.csv and curves-{constant,ercot}.png/pdf.',
                  'Binary plots stay out of Git. Recreate them locally from the returned text with --report-only.', '']
        (output / 'summary.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    report_all(parser.parse_args().output)
