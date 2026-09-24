"""Run one declared analytic scaling scenario using the existing weighted PPO."""

import argparse
import os
from pathlib import Path

from scaling_profiles import ROOT, read_design, validate_bundle


def main():
    design = read_design()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, choices=tuple(design['models']))
    parser.add_argument('--scenario', required=True, choices=tuple(design['scenarios']))
    parser.add_argument('--seed', type=int, choices=design['seeds'], default=11)
    parser.add_argument('--stage', choices=('all', 'fixed', 'train', 'validate'), default='all')
    parser.add_argument('--source', default=os.environ.get('CARBON_SOURCE'))
    parser.add_argument('--output', default=os.environ.get('CARBON_OUTPUT'))
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--check', action='store_true', help='Read config/assets only; no torch, simulation or training')
    args = parser.parse_args()
    from frontera_old_gpt import require_allocation
    require_allocation(args.check)
    from carbon.common import load_json, require
    from carbon.config import Bundle
    bundle = Bundle(ROOT/f'configs/scaling-v2-{args.model}-{args.scenario}-frontera-c84.development.json')
    binding = validate_bundle(bundle)
    prefix = f'scaling-v2-{args.model}-{args.scenario}-frontera-c84'
    source = Path(args.source or ROOT/f'results/{prefix}-fixed-max48').resolve()
    output = Path(args.output or ROOT/f'results/{prefix}-max48-weighted-seed{args.seed}').resolve()
    require(output != source and source not in output.parents and output not in source.parents,
            'Fixed and PPO outputs must be separate')
    if source.exists():
        marker = source/'old-gpt-fixed-plan.json'
        require(marker.is_file() and load_json(marker).get('profile_inputs') == binding,
                'Fixed source belongs to another profile/design; use a new source')
    if output.exists():
        marker = output/'run-plan.json'
        require(marker.is_file() and load_json(marker)['source_binding']['profile_inputs'] == binding,
                'PPO output belongs to another profile/design; use a new output')
        require(args.check or args.stage == 'fixed' or args.resume, 'Output exists; use --resume')
    print(f'ASSUMED scaling profile: {args.model}/{args.scenario}; efficiency32/4={binding["efficiency32_vs4"]:.0%}; '
          f'T16={binding["anchor_training_hours"]:g}h; seed={args.seed}; rho=1.0', flush=True)
    print(f'Fixed source: {source}\nPPO output: {output}', flush=True)
    print('Fixed requests 48h; dynamic requests rounded planned duration. All returned curves must disclose both rules.', flush=True)
    if args.check:
        print('Config, design and asset checks passed. No replay/training; runtime dependencies checked on compute.', flush=True)
        return
    import torch
    from packaging.version import Version
    require(Version('2.6') <= Version(torch.__version__) < Version('3'), 'PyTorch >=2.6,<3 required')
    from frontera_old_gpt import prepare_fixed
    from old_gpt_trial import run
    if args.stage == 'validate':
        require(args.resume and (output/'run-plan.json').is_file(), 'Validation requires existing PPO and --resume')
    prepare_fixed(bundle, source)
    if args.stage != 'fixed':
        run(bundle, output, source, resume=args.resume and output.exists(), stage=args.stage,
            settings={'seed': args.seed, 'rho': design['primary_power_rho']})


if __name__ == '__main__':
    main()
