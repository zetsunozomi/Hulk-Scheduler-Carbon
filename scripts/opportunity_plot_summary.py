"""Render returned summary JSON only; no raw episodes, traces or checkpoints."""
import argparse
from pathlib import Path

from carbon.common import load_json, require
from opportunity_report import plot_curves, value_hash


def load_summaries(output):
    output = Path(output)
    contract = load_json(output / 'run-plan.json')
    design = contract['design']
    expected_methods = [f'Fixed-{n}' for n in (4, 8, 16, 32)] + [f'Dynamic-{a:.2f}' for a in design['alphas']]
    expected_arrivals = len(design['seeds']) * design['episodes_per_seed']
    reports = []
    for name in design['scenarios']:
        path = output / name / 'summary.json'
        if not path.is_file():
            print(f'No saved summary yet: {name}', flush=True)
            continue
        report = load_json(path)
        require(report['contract_sha256'] == value_hash(contract), f'{name}: run-plan/summary mismatch')
        require(report['scenario'] == name, f'{name}: wrong scenario identity')
        rows = report['rows']
        require([r['method'] for r in rows] == expected_methods, f'{name}: missing/reordered curve points')
        require(all(r['outcomes'] == expected_arrivals for r in rows), f'{name}: cohort count mismatch')
        require(set(report['per_seed']) == {str(s) for s in design['seeds']}, f'{name}: seed set mismatch')
        complete = all(r['completed'] == r['outcomes'] for r in rows)
        require(report['diagnostics']['all_methods_complete'] == complete, f'{name}: completion flag mismatch')
        reports.append((name, report))
    require(reports, 'No scenario summary.json files found in this result directory')
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path, help='Result directory containing run-plan.json and scenario summaries')
    parser.add_argument('--figures', type=Path, help='Optional separate figure directory; otherwise write beside summaries')
    args = parser.parse_args()
    for name, report in load_summaries(args.output):
        folder = (args.figures or args.output) / name
        folder.mkdir(parents=True, exist_ok=True)
        plot_curves(folder, report['rows'], name, report['diagnostics']['all_methods_complete'])
        print(f'Rendered saved summaries: {folder}/curves-{{constant,ercot}}.png/pdf', flush=True)
    print('Summary configuration bindings checked; original episode hashes were not re-audited. No replay run.', flush=True)


if __name__ == '__main__':
    main()
