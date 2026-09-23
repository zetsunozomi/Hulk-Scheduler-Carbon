"""Old GPT-2: build independent fixed baselines, then four-scale weighted PPO."""
import argparse
import os
from pathlib import Path
import socket
import sys


def require_allocation(check=False):
    if not check and (not os.environ.get('SLURM_JOB_ID') or
                      socket.gethostname().split('.')[0].startswith('login')):
        raise RuntimeError('Use sbatch from login, or bash on an allocated compute node. --check is read-only.')


def prepare_fixed(bundle, source, check=False):
    from carbon.baselines import make_references
    from carbon.common import digest, load_json, require
    from carbon.main_pilot import fixed_stage
    from carbon.probe_resume import probe_lock
    from carbon.runner import write_manifest
    from old_gpt_trial import source_inputs

    source = Path(source).resolve()
    require(bundle.raw['cluster']['nodes'] == 84 and bundle.raw['cluster']['allowed_nodes'] == [4, 8, 16, 32],
            'Expected C84 old-GPT actions 4/8/16/32')
    from old_gpt_profiles import validate_bundle
    profile_binding = validate_bundle(bundle)
    marker = source/'old-gpt-fixed-plan.json'
    contract = {'kind': 'old_gpt_fixed84_v1', 'profile_inputs': profile_binding, 'config_sha256': bundle.manifest['config_sha256'],
                'asset_sha256': bundle.manifest['asset_sha256'],
                'core': {p.name: digest(p) for p in (bundle.root/'src/carbon').glob('*.py')}}
    required = ['references.json'] + [f'fixed-{split}/{name}' for split in ('train', 'validation')
                                    for name in ('manifest.json', 'episodes.jsonl', 'chunks.jsonl', 'stage-seal.json')]
    if all((source/name).is_file() for name in required):
        source_inputs(bundle, source)
        print(f'Existing old-GPT fixed inputs verified: {source}', flush=True)
        return
    if source.exists():
        require(marker.is_file() and load_json(marker) == contract,
                f'Incomplete/foreign fixed source: {source}; use a new CARBON_SOURCE, or copy all nine fixed files.')
    if check:
        print(f'Fixed inputs will be built/resumed on compute: {source}', flush=True)
        return
    with probe_lock(source, source.exists(), '.frontera-fixed.lock'):
        if marker.exists():
            require(load_json(marker) == contract, 'Fixed preparation inputs/code changed')
        else:
            write_manifest(marker, contract)
        for split in ('train', 'validation'):
            fixed_stage(bundle, source/f'fixed-{split}', split)
        refs = source/'references.json'
        if not refs.exists():
            temporary = source/'references.pending.json'
            # The previous interruption may have left only an uncommitted reference file.
            if temporary.exists():
                temporary.rename(source/f'references.interrupted-{os.getpid()}.json')
            make_references(source/'fixed-train', temporary)
            temporary.replace(refs)
        source_inputs(bundle, source)
    print('Old-GPT fixed preparation complete. No wait probes or planner fitting.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('medium', 'large', 'xl'), default='medium')
    parser.add_argument('--source', default=os.environ.get('CARBON_SOURCE'))
    parser.add_argument('--output', default=os.environ.get('CARBON_OUTPUT'))
    parser.add_argument('--stage', choices=('all', 'fixed', 'train', 'validate'), default='all')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--check', action='store_true', help='Read inputs/dependencies only; no replay or training')
    args = parser.parse_args()
    require_allocation(args.check)
    import torch, sklearn, matplotlib, scipy
    from packaging.version import Version
    from carbon.common import require
    from carbon.config import Bundle
    from old_gpt_trial import run
    require(sys.version_info >= (3, 10), 'Python >= 3.10 required')
    require(Version('2.6') <= Version(torch.__version__) < Version('3'), 'PyTorch >=2.6,<3 required')
    print(f'Python: {sys.executable}\nPyTorch: {torch.__version__}\n'
          f'sklearn: {sklearn.__version__}; scipy: {scipy.__version__}; matplotlib: {matplotlib.__version__}', flush=True)
    prefix = f'old-gpt-frontera-{args.model}-c84'
    source = Path(args.source or f'results/{prefix}-fixed').resolve()
    output = Path(args.output or f'results/{prefix}-weighted-seed11').resolve()
    require(output != source and source not in output.parents and output not in source.parents,
            'Fixed source and weighted output must be separate directories')
    require(args.stage == 'fixed' or args.check or not output.exists() or args.resume,
            'Weighted output already exists; use --resume')
    if args.stage == 'validate':
        require(args.resume and (output/'run-plan.json').is_file(), 'Validation requires existing PPO and --resume')
    bundle = Bundle(f'configs/old-gpt-frontera-{args.model}-c84.development.json')
    print(f'Profile: {args.model}; actions: 4/8/16/32 nodes; fixed and dynamic request cap: 48h', flush=True)
    print(f'Fixed source: {source}\nPPO output: {output}', flush=True)
    require(bundle.raw['purpose'] in {'synthetic', 'development'}, 'This entry is development only')
    prepare_fixed(bundle, source, check=args.check)
    if args.check:
        print('Read-only check complete. No simulation, training or test evaluation was run.', flush=True)
        return
    if args.stage != 'fixed':
        # A timeout during fixed preparation can precede creation of a PPO output.
        # Existing PPO contracts still verify exact source path, hashes and torch version.
        run(bundle, output, source, resume=args.resume and output.exists(), stage=args.stage)


if __name__ == '__main__':
    main()
