"""Import node-hour measurements as legacy fixed-work replay profiles.

The schema's batch/GPU integers below are bookkeeping units, not recovered
training metadata. Rates alone determine progress; see data/old_gpt/README.md.
"""
from decimal import Decimal
from pathlib import Path

from carbon.common import digest, load_json, require

NODES = (4, 8, 16, 32)
ASSET = 'data/old_gpt/profiles.json'


def workload_config(asset, model):
    require(asset['unit'] == 'nodes' and asset['work_iterations'] == 100000,
            'Old GPT input must describe 100000 iterations at node counts')
    profile = asset['profiles'][model]
    require(set(profile['node_hours']) == set(map(str, NODES)), 'Old GPT scale table differs')
    return {
        'name': f'old-GPT2-{model}-100000-legacy-iterations',
        'optimizer_updates': asset['work_iterations'],
        'global_batch': 32, 'gpus_per_node': 1, 'sequence_length': 1,
        'profile_status': 'assumed',
        'precision': 'unknown; not used by replay',
        'software': 'unknown measurement stack; legacy iteration is one replay work unit',
        'correctness_artifact': 'data/old_gpt/README.md; arithmetic tests only, no elastic training correctness claim',
        'provenance': (
            f"Author-confirmed measured speeds: {profile['label']}, {ASSET}. "
            'Source node-hours / nodes gives time for 100000 legacy iterations. '
            'Physical GPUs/node, batch, sequence length and optimizer equivalence are unknown. '
            'Schema bridge ONLY: gpus_per_node=1 normalized worker/node, global_batch=32 abstract units, '
            'microbatch=1, accumulation=32/nodes, sequence_length=1 sentinel. '
            'These are NOT measurement metadata. profile_status=assumed covers this replay mapping; '
            'the speed table itself is author-attested measured data. '
            'Initialization/restart/checkpoint are each assumed 300 seconds, not recovered measurements.'),
        'profiles': {str(n): {
            'updates_per_hour': float(Decimal(asset['work_iterations'])*n/Decimal(profile['node_hours'][str(n)])),
            'initialization_seconds': 300, 'restart_seconds': 300, 'checkpoint_seconds': 300,
            'microbatch': 1, 'accumulation': 32//n,
        } for n in NODES},
    }


def validate_bundle(bundle):
    """Bind the table and source snapshot as well as the resolved simulator config."""
    if bundle.raw['purpose'] == 'synthetic':
        return {'kind': 'synthetic_fixture'}
    require(bundle.raw['purpose'] == 'development', 'Legacy profiles are development inputs')
    root = Path(bundle.root)
    asset = load_json(root/ASSET)
    snapshot = asset['source']['snapshot']
    require(snapshot == 'data/old_gpt/source-table.tex', 'Unexpected source snapshot')
    require(digest(root/snapshot) == asset['source']['snapshot_sha256'], 'Old GPT source snapshot changed')
    matches = [key for key in asset['profiles'] if bundle.raw['workload'] == workload_config(asset, key)]
    require(len(matches) == 1, 'Configuration differs from canonical old GPT profile; do not reuse another baseline')
    require(bundle.raw['cluster']['allowed_nodes'] == list(NODES) and bundle.raw['cluster']['nodes'] == 84,
            'Expected old GPT C84 action scales')
    return {'kind': 'old_gpt_profile_v1', 'model': matches[0],
            'profiles_sha256': digest(root/ASSET), 'source_snapshot_sha256': digest(root/snapshot)}
