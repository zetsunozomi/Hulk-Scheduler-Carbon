from contextlib import redirect_stdout
from datetime import timedelta
import io
from pathlib import Path
import tempfile
import unittest
from carbon.common import iso, load_json
from carbon.probes import collect_probes
from carbon.trace import TraceJob
from tests.helpers import T0, bundle

class ProbeReadinessTests(unittest.TestCase):
    def collect(self, jobs):
        b=bundle();b.jobs=tuple(jobs)
        b.raw['execution'].update(warmup_seconds=0,initial_state_mode='empty_warmup')
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'probes';printed=io.StringIO()
            with redirect_stdout(printed):
                meta=collect_probes(b,out,start=iso(T0+timedelta(seconds=60)),
                                    stop=iso(T0+timedelta(seconds=120)),request_seconds=[60])
            self.assertEqual(load_json(out/'manifest.json')['queue_summary'],meta['queue_summary'])
            return meta,printed.getvalue()

    def test_empty_window_is_flagged_but_preserved(self):
        meta,printed=self.collect([])
        self.assertEqual(meta['rows'],4)
        self.assertEqual(meta['queue_summary']['snapshots'],1)
        self.assertEqual(meta['queue_summary']['empty_background_snapshots'],1)
        self.assertEqual(meta['queue_summary']['sampled_mean_running_fraction'],0.)
        self.assertIn('does not check busy-queue behavior',printed)
        self.assertTrue(all(s['mean_wait_hours']==0. and s['positive_wait_rows']==0
                            for s in meta['wait_summary_by_request'].values()))

    def test_no_new_submissions_does_not_mean_no_background_work(self):
        jobs=[TraceJob('carry',32,T0,T0,T0+timedelta(seconds=180),timedelta(seconds=180))]
        meta,printed=self.collect(jobs)
        summary=meta['queue_summary']
        self.assertEqual(summary['background_submissions_in_probe_window'],0)
        self.assertEqual(summary['empty_background_snapshots'],0)
        self.assertEqual(summary['sampled_mean_running_fraction'],1.)
        self.assertNotIn('does not check busy-queue behavior',printed)
        for row in meta['wait_summary_by_request'].values():
            self.assertEqual(row['positive_wait_rows'],1)
            self.assertAlmostEqual(row['mean_wait_hours'],120/3600)
            self.assertEqual(row['censored_rows'],0)
