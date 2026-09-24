"""Full-walltime requests affect reservations, never useful work or idle charging."""
from contextlib import redirect_stdout
from datetime import timedelta
from decimal import Decimal
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys

from carbon.baselines import make_references
from carbon.common import ContractError, digest, json_text, load_json
from carbon.config import Bundle
from carbon.environment import Environment
from carbon.main_pilot import fixed_stage
from carbon.replay import Replay, Request
from tests.helpers import ROOT, T0, cluster

with patch.object(sys, 'path', [str(ROOT/'scripts'), *sys.path]):
    from fixed_max_walltime import fixed_bundle, request_contract
    import frontera_old_gpt as entry
    import old_gpt_trial as trial


class MaxWalltimeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        capture = redirect_stdout(io.StringIO()); capture.__enter__()
        self.addCleanup(capture.__exit__, None, None, None)
        raw = load_json(ROOT/'configs/synthetic-p2.json')
        raw['root'] = str(ROOT); raw['cluster']['nodes'] = 84
        raw['trace']['role'] = 'workload_template'
        raw['execution']['initial_state_mode'] = 'empty_warmup'
        self.config = self.root/'config.json'; self.config.write_text(json_text(raw))
        self.bundle = Bundle(self.config)
        self.source = self.root/'fixed'

    def test_only_request_changes_and_dynamic_workload_is_unchanged(self):
        original = json_text(self.bundle.manifest)
        view = fixed_bundle(self.bundle)
        self.assertNotIn('fixed_request_policy', self.bundle.manifest)
        self.assertEqual(view.manifest['fixed_request_policy'], request_contract(self.bundle))
        for n in (4,8,16,32):
            for remaining in (1,17,100):
                for first in (True,False):
                    before = self.bundle.workload.plan(n,remaining,first)
                    after = view.workload.plan(n,remaining,first)
                    self.assertEqual(after.requested.total_seconds(), 300)
                    for field in ('nodes','updates','remaining_before','setup','training','checkpoint','first','actual'):
                        self.assertEqual(getattr(before,field), getattr(after,field))
                    self.assertEqual(self.bundle.workload.plan(n,remaining,first), before)
        self.assertEqual(original, json_text(self.bundle.manifest))

    def test_no_idle_charge_or_added_work_in_empty_queue(self):
        episode = next(e for e in self.bundle.episodes if e.split == 'train')
        for n in (4,8,16,32):
            summaries = []
            for bundle in (self.bundle, fixed_bundle(self.bundle)):
                base = Replay([], episode.arrival, bundle.trace_end, bundle.raw['cluster'])
                env = Environment(bundle, episode, base, f'Fixed-{n}')
                while env.status == 'running':
                    env.step(n)
                summaries.append(env.summary())
                self.assertEqual(env.remaining, 0)
                if bundle is not self.bundle:
                    self.assertTrue(all(abs(c['requested_walltime_hours']-300/3600)<1e-12 for c in env.chunks))
                    self.assertLess(env.chunks[-1]['planned_allocation_hours'], 300/3600)
            for key in ('tat_hours','nodehours','carbon_g_per_kappa','completed_updates','chunk_count'):
                self.assertEqual(summaries[0][key], summaries[1][key])

    def test_backfill_uses_long_request_but_releases_at_actual_finish(self):
        short = self.bundle.workload.plan(4,1,True)
        long = fixed_bundle(self.bundle).workload.plan(4,1,True)
        starts = []
        for plan in (short,long):
            replay = Replay([],T0,T0+timedelta(hours=1),cluster(8),sample_seconds=60)
            replay.inject(Request('running',4,T0,timedelta(seconds=180),timedelta(seconds=180),True))
            replay.until('running','start',replay.coverage_end)
            replay.inject(Request('head',8,T0,timedelta(seconds=300),timedelta(seconds=300),True))
            replay.inject(Request('target',4,T0,plan.requested,plan.actual,True))
            allocation = replay.until('target','complete',replay.coverage_end)
            self.assertEqual(allocation.end-allocation.start, short.actual)
            starts.append(allocation.start)
        self.assertLess(starts[0],starts[1])

    def test_all_three_development_profiles_really_request_48h(self):
        from carbon.workload import Workload
        for model in ('medium','large','xl'):
            raw = load_json(ROOT/f'configs/old-gpt-frontera-{model}-c84.development.json')
            from fixed_max_walltime import FixedMaxWalltimeWorkload
            work = Workload(raw['workload'],[4,8,16,32],172800,60)
            fixed = FixedMaxWalltimeWorkload(work)
            for n in (4,8,16,32):
                remaining, first = 100000, True
                while remaining:
                    before = work.plan(n,remaining,first)
                    plan = fixed.plan(n,remaining,first)
                    self.assertEqual(plan.requested.total_seconds(),172800)
                    self.assertEqual(plan.actual,before.actual)
                    remaining -= plan.updates; first = False
                self.assertLess(plan.actual.total_seconds(),172800)

    def test_reject_old_precise_baselines_without_overwriting(self):
        self.source.mkdir()
        for split in ('train','validation'):
            fixed_stage(self.bundle,self.source/f'fixed-{split}',split)
        make_references(self.source/'fixed-train',self.source/'references.json')
        before = {str(p):digest(p) for p in self.source.rglob('*') if p.is_file()}
        with self.assertRaisesRegex(ContractError,'Fixed request policy'):
            entry.prepare_fixed(self.bundle,self.source)
        self.assertEqual(before,{str(p):digest(p) for p in self.source.rglob('*') if p.is_file()})

    def test_resume_fixed_stage_and_bind_references(self):
        def interrupted(bundle,directory,split):
            if split == 'validation':
                raise RuntimeError('synthetic stop')
            return fixed_stage(bundle,directory,split)
        with patch('carbon.main_pilot.fixed_stage',side_effect=interrupted):
            with self.assertRaisesRegex(RuntimeError,'synthetic stop'):
                entry.prepare_fixed(self.bundle,self.source)
        original = (self.source/'fixed-train/episodes.jsonl').read_bytes()
        entry.prepare_fixed(self.bundle,self.source)
        self.assertEqual(original,(self.source/'fixed-train/episodes.jsonl').read_bytes())
        references,_ = trial.source_inputs(self.bundle,self.source)
        self.assertEqual(references['fixed_request_policy'], request_contract(self.bundle))
        for split in ('train','validation'):
            meta = load_json(self.source/f'fixed-{split}/manifest.json')
            self.assertEqual(meta['fixed_request_policy'], references['fixed_request_policy'])
            rows = trial.records(self.source/f'fixed-{split}/chunks.jsonl')
            self.assertTrue(all(abs(r['requested_walltime_hours']-300/3600)<1e-12 for r in rows))


if __name__ == '__main__':
    unittest.main()
