"""Deterministic assumed throughput profiles; this module never runs a replay."""

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DESIGN = 'data/scaling_sensitivity/design.json'
PREFIX = 'analytic-scaling-'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_design(root=ROOT):
    design = json.loads((root/DESIGN).read_text())
    if (design['kind'] != 'analytic_scaling_design_v2' or design['nodes'] != [4, 8, 16, 32]
            or design['anchor_nodes'] != 16 or set(design['models']) != {'medium', 'xl'}
            or design['work_units'] != 100000 or design['efficiency_reference_nodes'] != 4):
        raise ValueError('Unexpected scaling design; version and review changed protocols explicitly')
    if digest(root/design['template']) != design['template_sha256']:
        raise ValueError('Scaling template changed; review before updating its design hash')
    for model in design['models'].values():
        if not math.isfinite(model['anchor_training_hours']) or model['anchor_training_hours'] <= 0:
            raise ValueError('Anchor runtime must be finite and positive')
    if not design['scenarios']:
        raise ValueError('Missing declared efficiency vectors')
    for scenario in design['scenarios'].values():
        eta = scenario['efficiency_vs4']
        if (set(eta) != {'4','8','16','32'} or eta['4'] != 1 or
                any(not math.isfinite(e) or e <= 0 for e in eta.values())):
            raise ValueError('Supply four finite positive efficiencies normalized at four nodes')
    for name in ('code', 'table'):
        if digest(root/design['anchor_source'][name]) != design['anchor_source'][name+'_sha256']:
            raise ValueError('Legacy anchor source changed; inspect the original timing evidence')
    return design


def scenario_points(design, model, scenario):
    """One runtime anchor plus per-scale efficiencies fully defines fixed-work rates."""
    eta = design['scenarios'][scenario]['efficiency_vs4']
    anchor = design['anchor_nodes']
    hours = design['models'][model]['anchor_training_hours']
    return [dict(nodes=n, training_hours=hours*anchor*eta[str(anchor)]/(n*eta[str(n)]),
                 updates_per_hour=(design['work_units']/hours)*n*eta[str(n)]/(anchor*eta[str(anchor)]),
                 efficiency_vs4=eta[str(n)]) for n in design['nodes']]


def config_for(design, model, scenario, root=ROOT):
    raw = deepcopy(json.loads((root/design['template']).read_text()))
    points = scenario_points(design, model, scenario)
    hours = design['models'][model]['anchor_training_hours']
    raw['purpose'] = 'development'
    raw['panel'] = f'analytic-scaling-v2-{model}-{scenario}-frontera-C84'
    raw['cluster']['name'] = 'Frontera-demand-C84-analytic-scaling-scenario'
    raw['cluster']['provenance'] = 'Declared 84-node demand replay with analytically assumed target throughput; no hardware performance measurement.'
    raw['workload'] = {
        'name': PREFIX + f'v2-{model}-{scenario}',
        'optimizer_updates': design['work_units'],
        'global_batch': design['anchor_measurement']['global_batch'],
        'gpus_per_node': design['anchor_measurement']['gpus_per_node'],
        'sequence_length': design['anchor_measurement']['sequence_length'],
        'profile_status': 'assumed', 'precision': design['anchor_measurement']['precision'],
        'software': 'analytical throughput input, not an executed neural-network training stack',
        'correctness_artifact': 'tests/test_scaling_sensitivity.py; work-accounting checks only',
        'provenance': (
            f'Explicit efficiency-vector sensitivity, {DESIGN}; workload={model}. '
            f'T(n)={hours:g}*16*eta16/(n*eta_n) hours for 100000 abstract work units; '
            f'eta={design["scenarios"][scenario]["efficiency_vs4"]}. '
            '16-node timing comes from legacy source; other-scale efficiencies are assumed. '
            'Batch, sequence length, precision and GPU count are author-reported anchor metadata. '
            'Microbatch=1 and accumulation values only satisfy simulated batch accounting; '
            'the measured microbatch/parallelism configuration is not recovered. '
            'Initialization/restart/checkpoint costs are separate assumptions. '
            'No task outcomes were used to fit these throughput values.'),
        'profiles': {str(p['nodes']): {
            'updates_per_hour': p['updates_per_hour'],
            'initialization_seconds': design['setup_seconds'],
            'restart_seconds': design['setup_seconds'],
            'checkpoint_seconds': design['checkpoint_seconds'],
            'microbatch': 1, 'accumulation': design['anchor_measurement']['global_batch']//
                (design['anchor_measurement']['gpus_per_node']*p['nodes']),
        } for p in points},
    }
    return raw


def validate_bundle(bundle):
    from carbon.common import require
    design = read_design(bundle.root)
    identity = bundle.raw['workload']['name'].removeprefix(PREFIX)
    parts = identity.split('-', 2)
    require(len(parts) == 3 and parts[0] == 'v2', 'Expected explicit-vector v2 profile')
    _, model, scenario = parts
    require(model in design['models'] and scenario in design['scenarios'], 'Unknown analytic model/scenario')
    expected = config_for(design, model, scenario, bundle.root)
    # root may be resolved to an absolute path in a temporary read-only check.
    actual = {k: v for k, v in bundle.raw.items() if k != 'root'}
    require(actual == {k: v for k, v in expected.items() if k != 'root'},
            'Analytic scaling config differs from its declared design')
    return {'kind': 'analytic_scaling_profile_v2', 'model': model, 'scenario': scenario,
            'profile_status': 'assumed', 'anchor_nodes': 16,
            'anchor_training_hours': design['models'][model]['anchor_training_hours'],
            'efficiency_vs4': design['scenarios'][scenario]['efficiency_vs4'],
            'efficiency32_vs4': design['scenarios'][scenario]['efficiency_vs4']['32'],
            'design_sha256': digest(bundle.root/DESIGN),
            'implementation_sha256': {name: digest(bundle.root/'scripts'/name) for name in
                                      ('scaling_profiles.py', 'frontera_scaling_sensitivity.py', 'scaling_report.py')}}


def generated_files(root=ROOT):
    design = read_design(root)
    rows = ['# Assumed scaling inputs (not experiment results)', '',
            'Medium T16=22.6h; XL T16=110h. Fixed work: 100000 optimizer steps.',
            'T(n)=T16*16*eta16/(n*eta_n). No queue wait or checkpoint overhead is included below.', '',
            '| Model | Scenario | eta4 / eta8 / eta16 / eta32 | T4 h | T8 h | T16 h | T32 h |',
            '|---|---|---|---:|---:|---:|---:|']
    files = {}
    for model in design['models']:
        for scenario in design['scenarios']:
            points = scenario_points(design, model, scenario)
            path = root/f'configs/scaling-v2-{model}-{scenario}-frontera-c84.development.json'
            files[path] = json.dumps(config_for(design, model, scenario, root), indent=2) + '\n'
            efficiencies = ' / '.join(f'{p["efficiency_vs4"]:.1%}' for p in points)
            rows.append(f'| {model} | {scenario} | {efficiencies} | ' +
                        ' | '.join(f'{p["training_hours"]:.3f}' for p in points) + ' |')
    rows += ['', 'Regenerate: `python scripts/scaling_profiles.py --write`.',
             'Check determinism: `python scripts/scaling_profiles.py --check`.', '']
    files[root/'data/scaling_sensitivity/PROFILES.md'] = '\n'.join(rows)
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--write', action='store_true')
    mode.add_argument('--check', action='store_true')
    args = parser.parse_args()
    for path, content in generated_files().items():
        if args.write:
            path.write_text(content)
        elif not path.is_file() or path.read_text() != content:
            raise SystemExit(f'Generated input missing or changed: {path}; review and regenerate')
        print(('Wrote: ' if args.write else 'Verified: ') + str(path.relative_to(ROOT)))


if __name__ == '__main__':
    main()
