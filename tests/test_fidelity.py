from contextlib import redirect_stdout
from datetime import timedelta
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from carbon.__main__ import main
from carbon.common import ContractError,iso,json_text,load_json
from carbon.fidelity import AdmissionRecorder,evaluate_replay
from carbon.replay import Replay
from carbon.trace import TraceJob
from tests.helpers import ROOT,T0,cluster


def job(name,submit,start,end,requested=60):
    return TraceJob(name,4,T0+timedelta(seconds=submit),T0+timedelta(seconds=start),
                    T0+timedelta(seconds=end),timedelta(seconds=requested))


class FidelityTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)

    def bundle(self,jobs,end=1200):
        raw={'cluster':cluster(4),'execution':{'warmup_seconds':600}}
        return SimpleNamespace(jobs=tuple(jobs),raw=raw,root=ROOT,trace_start=T0-timedelta(hours=1),
                               trace_end=T0+timedelta(hours=1),splits={'validation':(T0,T0+timedelta(seconds=end))},
                               manifest={'purpose':'synthetic','panel':'fidelity-fixture'})

    def test_continuous_replay_boundary_jobs_and_analytical_discrepancy(self):
        jobs=[job('running',-1200,-900,120,1200),job('pending',-660,120,180),
              job('A',0,180,240),job('B',30,240,300),job('hidden_delay',660,900,960)]
        meta=evaluate_replay(self.bundle(jobs),self.root/'output')
        rows={r['job_id']:r for r in map(json.loads,(self.root/'output/jobs.jsonl').read_text().splitlines())}
        self.assertEqual(set(rows),{'A','B','hidden_delay'})
        self.assertEqual(meta['initial_running_jobs'],1);self.assertEqual(meta['initial_pending_jobs'],1)
        self.assertEqual(rows['A']['replay_start_utc'],iso(T0+timedelta(minutes=3)))
        self.assertEqual(rows['B']['wait_error_hours'],0.)
        self.assertAlmostEqual(rows['hidden_delay']['wait_error_hours'],-4/60)
        scores=load_json(self.root/'output/metrics.json')['overall']
        self.assertEqual(scores['jobs'],3);self.assertEqual(scores['paired_admissions'],3)
        self.assertEqual(scores['metrics']['median_absolute_error_hours'],0.)
        self.assertAlmostEqual(scores['metrics']['p90_absolute_error_hours'],4/60)
        self.assertEqual(meta['status'],'complete')

    def test_observed_and_replayed_censoring_are_separate_and_not_filtered(self):
        # A hidden historical priority let C start before B, unlike this replay.
        jobs=[job('B',0,120,720,1800),job('C',60,60,90),job('D',240,720,780)]
        evaluate_replay(self.bundle(jobs,end=300),self.root/'output')
        rows={r['job_id']:r for r in map(json.loads,(self.root/'output/jobs.jsonl').read_text().splitlines())}
        self.assertFalse(rows['C']['observed_wait_censored']);self.assertTrue(rows['C']['replay_wait_censored'])
        self.assertTrue(rows['D']['observed_wait_censored']);self.assertTrue(rows['D']['replay_wait_censored'])
        self.assertIsNone(rows['D']['observed_start_utc']);self.assertIsNone(rows['C']['wait_error_hours'])
        scores=load_json(self.root/'output/metrics.json')['overall']
        self.assertEqual((scores['jobs'],scores['paired_admissions']),(3,1))
        self.assertEqual((scores['observed_wait_censored'],scores['replay_wait_censored']),(1,2))

    def test_recorder_does_not_change_dispatch_or_keep_completed_backgrounds(self):
        jobs=[job('A',0,0,30),job('B',20,60,90)]
        arguments=(jobs,T0,T0+timedelta(hours=1),cluster(4))
        ordinary=Replay(*arguments,sample_seconds=60)
        recorder=AdmissionRecorder(*arguments,sample_seconds=60,scored_ids=['A','B'])
        for at in (T0,T0+timedelta(seconds=30),T0+timedelta(seconds=60),T0+timedelta(minutes=5)):
            ordinary.advance_to(at);recorder.advance_to(at)
            self.assertEqual(ordinary.visible(),recorder.visible())
        self.assertEqual(recorder.admissions,{'A':T0,'B':T0+timedelta(seconds=60)})
        self.assertEqual(recorder.finished,{})

    def test_initial_running_jobs_cannot_be_scored_and_boundary_admissions_are_censored(self):
        jobs=[job('B',0,0,300,300),job('C',1,300,360)]
        evaluate_replay(self.bundle(jobs,end=300),self.root/'output')
        rows={r['job_id']:r for r in map(json.loads,(self.root/'output/jobs.jsonl').read_text().splitlines())}
        self.assertTrue(rows['C']['observed_wait_censored']);self.assertTrue(rows['C']['replay_wait_censored'])
        with self.assertRaisesRegex(ContractError,'No historical submissions'):
            evaluate_replay(self.bundle([]),self.root/'empty')

    def test_cli_runs_with_qwen_ci_and_power_intentionally_unfilled(self):
        config=load_json(ROOT/'configs/synthetic-p2.json')
        config.update(root=str(ROOT),workload=None,ci=None,power=None,cohort=None)
        path=self.root/'queue.json';path.write_text(json_text(config))
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(['evaluate-replay','--config',str(path),'--output',str(self.root/'cli'),'--split','train']),0)
        meta=load_json(self.root/'cli/manifest.json')
        self.assertEqual(meta['validated_scope'],'queue_only');self.assertEqual(meta['cohort_jobs'],2)
