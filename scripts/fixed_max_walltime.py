"""Fixed baselines reserve the full walltime cap and release on actual completion.

This adapter leaves the archived core/AMSP replay implementation untouched.
The separate manifest contract prevents reuse of precise-final-request results.
"""
from copy import copy
from dataclasses import replace
from pathlib import Path

from carbon.common import digest, duration, load_json, require


def request_contract(bundle):
    return {
        'kind': 'fixed_max_walltime_v1',
        'fixed_rule': 'request_max_walltime_every_chunk_including_final',
        'requested_seconds': bundle.raw['cluster']['max_request_seconds'],
        'actual_runtime_rule': 'unchanged_work_plan_release_on_completion',
        'dynamic_rule': 'ceil_planned_allocation_to_walltime_resolution',
        'implementation_sha256': digest(Path(__file__)),
    }


class FixedMaxWalltimeWorkload:
    def __init__(self, workload):
        self._workload = workload

    def __getattr__(self, name):
        return getattr(self._workload, name)

    def plan(self, nodes, remaining, first):
        plan = self._workload.plan(nodes, remaining, first)
        if plan is None:
            return None
        return replace(plan, requested=duration(self._workload.max_request))


def fixed_bundle(bundle):
    """A separate execution view; no mutation of dynamic plans or raw config."""
    view = copy(bundle)
    view.workload = FixedMaxWalltimeWorkload(bundle.workload)
    view.manifest = {**bundle.manifest, 'fixed_request_policy': request_contract(bundle)}
    return view


def make_references(fixed_directory, output):
    from carbon.baselines import make_references as core_references
    from carbon.runner import write_manifest
    policy = load_json(fixed_directory/'manifest.json').get('fixed_request_policy')
    require(policy and policy['kind'] == 'fixed_max_walltime_v1', 'Missing fixed max-walltime provenance')
    references = core_references(fixed_directory, output)
    references['fixed_request_policy'] = policy
    write_manifest(output, references)
    return references


def check_request_contract(artifact, bundle):
    require(artifact.get('fixed_request_policy') == request_contract(bundle),
            'Fixed request policy differs or is missing; rebuild max48 baselines in a new directory')
