from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
import json
from pathlib import Path
import tempfile
import unittest

from carbon.common import ContractError
from carbon.config import Bundle
from carbon.environment import initial_replay
from carbon.fidelity import evaluate_replay
from carbon.replay import Replay, Request
from carbon.trace import TraceJob, load_trace
from carbon.workload import Workload
from tests.helpers import ROOT, T0, bundle, cluster


class PublishedProfileTests(unittest.TestCase):
    def test_units_batch_and_published_scale_support(self):
        data = json.loads((ROOT/'data/amsp/profiles.json').read_text())
        self.assertEqual(data['selected_gpus'],[32,128,512,1024])
        for model,config in data['workloads'].items():
            work = Workload(config,data['allowed_nodes'],172800,60)
            for row in data['rows']:
                if row['model'] == model and row['selected']:
                    n = row['nodes'];p = work.profiles[n]
                    self.assertAlmostEqual(float(p.updates_per_hour),row['gpus']*row['tgs_digitized']*3600/4194304,places=7)
                    self.assertEqual(n*8*p.microbatch*p.accumulation,1024)
                    self.assertIsNotNone(work.plan(n,work.total_updates,True))
            self.assertTrue(192 <= config['optimizer_updates']/float(work.profiles[4].updates_per_hour) < 193)

    def test_extended_scale_config_and_work_conservation(self):
        b = bundle(); raw = deepcopy(b.raw)
        raw['root'] = str(ROOT)
        raw['cluster']['nodes'] = 128
        raw['cluster']['allowed_nodes'] = [4,16,64,128]
        data = json.loads((ROOT/'data/amsp/profiles.json').read_text())
        raw['workload'] = deepcopy(data['workloads']['7B'])
        raw['workload']['optimizer_updates'] = 2
        for profile in raw['workload']['profiles'].values():
            profile.update(initialization_seconds=0,restart_seconds=0,checkpoint_seconds=0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'config.json';path.write_text(json.dumps(raw))
            loaded = Bundle(path)
        self.assertEqual(list(loaded.workload.profiles),[4,16,64,128])
        from carbon.environment import Environment
        env = Environment(loaded,loaded.episodes[0],initial_replay(loaded,loaded.episodes[0]),'Fixed-128')
        # The synthetic split is short; use a zero-overhead profile for this
        # accounting check while retaining the published rate and batch.
        env.work.profiles[128] = replace(env.work.profiles[128],initialization_seconds=0,restart_seconds=0,checkpoint_seconds=0)
        env.step(128)
        self.assertEqual(env.summary()['completed_updates'],2)
        self.assertEqual(env.chunks[0]['global_next_sample_index'],2048)


class ScenarioTests(unittest.TestCase):
    def test_fcfs_head_blocking_differs_from_backfill(self):
        starts=[]
        for model in ('conservative_backfill_v1','fcfs_v1'):
            c=cluster(8);c['scheduler'].update(model=model,dispatch_interval_seconds=1,size_weight=0)
            replay=Replay([],T0,T0+timedelta(hours=1),c)
            replay.inject(Request('running',4,T0,timedelta(seconds=100),timedelta(seconds=100)))
            replay.advance_to(T0+timedelta(seconds=1),before_dispatch=True)
            at=replay.time
            replay.inject(Request('a-large',8,at,timedelta(seconds=20),timedelta(seconds=20)))
            replay.inject(Request('b-small',4,at,timedelta(seconds=10),timedelta(seconds=10),True))
            starts.append(replay.until('b-small','start',T0+timedelta(hours=1)).start)
        self.assertEqual(starts,[T0+timedelta(seconds=1),T0+timedelta(seconds=120)])

    def test_integer_resource_rescaling_preserves_admissions(self):
        for model in ('conservative_backfill_v1','fcfs_v1'):
            schedules=[]
            for factor in (1,3):
                c=cluster(8*factor);c['scheduler'].update(model=model,dispatch_interval_seconds=1)
                replay=Replay([],T0,T0+timedelta(hours=1),c)
                for ident,n,d in [('a',5,40),('b',4,10),('c',3,20)]:
                    replay.inject(Request(ident,n*factor,T0,timedelta(seconds=d+10),timedelta(seconds=d),True))
                schedules.append([replay.until(ident,'complete',T0+timedelta(hours=1)).start for ident in ('a','b','c')])
            self.assertEqual(schedules[0],schedules[1])

    def test_constructed_state_ignores_logged_admissions(self):
        c=cluster(8)
        job=TraceJob('old',8,T0-timedelta(hours=2),T0-timedelta(hours=1),T0+timedelta(hours=1),timedelta(hours=3))
        self.assertIn('old',Replay([job],T0,T0+timedelta(hours=2),c).running)
        self.assertFalse(Replay([job],T0,T0+timedelta(hours=2),c,initialize_from_observed=False).running)
        b=bundle();b.raw['execution']['initial_state_mode']='empty_warmup';b.raw['trace']['role']='workload_template'
        first=initial_replay(b,b.episodes[0]);first_state=first.visible()
        later=initial_replay(b,b.episodes[1])
        self.assertEqual(first.time,b.episodes[0].arrival)
        self.assertEqual(first.visible(),first_state)
        fresh=Replay(b.jobs,b.trace_start,b.trace_end,b.raw['cluster'],initialize_from_observed=False)
        fresh.advance_to(b.episodes[1].arrival,before_dispatch=True)
        self.assertEqual(later.visible(),fresh.visible())
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ContractError,'undefined'):
                evaluate_replay(b,Path(tmp)/'wrong-ground-truth')

    def test_retrospective_research_preserves_prior_use_disclosure(self):
        raw = deepcopy(bundle().raw)
        raw.update(root=str(ROOT), purpose='research')
        raw['workload']['profile_status'] = 'published'
        raw['holdout_audit'].update(test_is_untouched=False, notes='Prior train/validation use; retrospective benchmark.')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'config.json'; path.write_text(json.dumps(raw))
            with self.assertRaisesRegex(ContractError, 'untouched'):
                Bundle(path)
            raw['holdout_audit']['evaluation_design'] = 'retrospective_temporal'
            path.write_text(json.dumps(raw)); loaded = Bundle(path)
            self.assertFalse(loaded.manifest['resolved_config']['holdout_audit']['test_is_untouched'])

    def test_width_transform_is_explicit_and_audited(self):
        b=bundle();t=deepcopy(b.raw['trace']);t.update(role='workload_template',node_multiplier=2,oversize_policy='cap')
        original,_=load_trace(b.assets['trace'],b.raw['trace'],32)
        changed,report=load_trace(b.assets['trace'],t,32)
        for old,new in zip(original,changed):
            self.assertEqual(new.nodes,min(2*old.nodes,32));self.assertEqual(new.runtime,old.runtime)
            self.assertEqual(new.submit,old.submit);self.assertEqual(new.requested,old.requested)
        self.assertGreater(report['counts']['scenario_node_seconds'],report['counts']['original_node_seconds'])
        t['role']='historical'
        with self.assertRaisesRegex(ContractError,'requires workload_template'):
            load_trace(b.assets['trace'],t,32)
