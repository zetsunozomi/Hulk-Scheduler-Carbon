from contextlib import redirect_stdout
from datetime import timedelta
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from carbon.common import ContractError, iso, load_json
from carbon.probe_resume import LEGACY_COLLECTOR, probe_lock
from carbon.probes import collect_probes
from carbon.replay import Replay
from carbon.runner import write_record
from carbon.wait_pipeline import run_wait_pipeline
from tests.helpers import T0, bundle


class ProbeResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.b = bundle()
        self.args = dict(interval_seconds=60, start=iso(T0+timedelta(seconds=1200)),
                         stop=iso(T0+timedelta(seconds=1500)))
        self.stdout = redirect_stdout(io.StringIO())
        self.stdout.__enter__()
        self.addCleanup(self.stdout.__exit__, None, None, None)

    def collect(self, path, **kw):
        return collect_probes(self.b, path, **self.args, **kw)

    def interrupt(self, path, count=5):
        writes = 0
        def killed(handle, row):
            nonlocal writes
            write_record(handle, row)
            writes += 1
            if writes == count:
                raise KeyboardInterrupt()
        with patch('carbon.probes.write_record', side_effect=killed):
            with self.assertRaises(KeyboardInterrupt):
                self.collect(path)
        # Hard kills need not execute finally, leaving stale counters/status.
        meta = load_json(path/'manifest.json')
        meta.update(status='running', rows=0, labeled_rows=0, censored_rows=0)
        (path/'manifest.json').write_text(json.dumps(meta))

    def test_mid_snapshot_resume_is_byte_identical_and_skips_saved_admissions(self):
        clean = self.root/'clean'; partial = self.root/'partial'
        baseline = self.collect(clean)
        self.interrupt(partial, 13)
        with patch.object(Replay, 'until', autospec=True, side_effect=Replay.until) as admission:
            result = self.collect(partial, resume=True)
        self.assertEqual(admission.call_count, baseline['rows']-13)
        self.assertEqual((partial/'probes.jsonl').read_bytes(), (clean/'probes.jsonl').read_bytes())
        for key in ('queue_summary', 'wait_summary_by_request', 'rows', 'labeled_rows', 'censored_rows'):
            self.assertEqual(result[key], baseline[key])
        self.assertEqual(result['resume_sessions'][0]['retained_rows'], 13)

    def test_recognized_legacy_manifest_and_torn_tail_are_salvaged(self):
        clean = self.root/'clean'; partial = self.root/'partial'
        self.collect(clean); self.interrupt(partial)
        meta = load_json(partial/'manifest.json')
        meta.pop('probe_engine')
        meta['software']['source_sha256']['probes.py'] = LEGACY_COLLECTOR
        original = json.dumps(meta)
        (partial/'manifest.json').write_text(original)
        with (partial/'probes.jsonl').open('ab') as handle:
            handle.write(b'{"arrival_utc": "2024-')
        result = self.collect(partial, resume=True)
        self.assertEqual((partial/'probes.jsonl').read_bytes(), (clean/'probes.jsonl').read_bytes())
        backup = partial/result['resume_sessions'][0]['backup']
        self.assertEqual((backup/'manifest.json').read_text(), original)
        self.assertEqual((backup/'torn-tail.bin').read_bytes(), b'{"arrival_utc": "2024-')
        self.assertEqual(result['software'], meta['software'])

    def test_complete_row_without_newline_is_preserved(self):
        partial = self.root/'partial'; clean = self.root/'clean'
        self.collect(clean); self.interrupt(partial)
        p = partial/'probes.jsonl'; p.write_bytes(p.read_bytes().removesuffix(b'\n'))
        self.collect(partial, resume=True)
        self.assertEqual(p.read_bytes(), (clean/'probes.jsonl').read_bytes())

    def test_rejects_duplicate_middle_corruption_and_contract_change_before_append(self):
        for case in ('duplicate', 'middle_corruption', 'changed_interval', 'changed_engine'):
            with self.subTest(case=case):
                path = self.root/case
                self.interrupt(path)
                p = path/'probes.jsonl'; data = p.read_bytes()
                if case == 'duplicate':
                    p.write_bytes(data+data.splitlines(keepends=True)[0])
                elif case == 'middle_corruption':
                    p.write_bytes(data+b'{bad}\n')
                else:
                    meta = load_json(path/'manifest.json')
                    if case == 'changed_interval': meta['probe_interval_seconds'] = 120
                    else: meta['probe_engine']['source_sha256']['replay.py'] = 'changed'
                    (path/'manifest.json').write_text(json.dumps(meta))
                before = p.read_bytes(); manifest = (path/'manifest.json').read_bytes()
                with self.assertRaises(ContractError): self.collect(path, resume=True)
                self.assertEqual(p.read_bytes(), before)
                self.assertEqual((path/'manifest.json').read_bytes(), manifest)

    def test_completed_resume_is_noop_but_checks_hash(self):
        path = self.root/'complete'; self.collect(path)
        original = (path/'manifest.json').read_bytes()
        with patch('carbon.probes.Replay', side_effect=AssertionError('must not replay')):
            self.collect(path, resume=True)
        self.assertEqual((path/'manifest.json').read_bytes(), original)
        with (path/'probes.jsonl').open('ab') as handle: handle.write(b'\n')
        with self.assertRaises(ContractError): self.collect(path, resume=True)

    def test_second_writer_is_rejected(self):
        path = self.root/'locked'
        with probe_lock(path, False):
            with self.assertRaisesRegex(ContractError, 'Another writer'):
                self.collect(path, resume=True)

    def test_resume_new_directory_and_no_implicit_overwrite(self):
        path = self.root/'new'
        result = self.collect(path, resume=True)
        self.assertEqual(result['status'], 'complete')
        with self.assertRaises(ContractError): self.collect(path)

    def test_pipeline_recovers_validation_and_skips_complete_model(self):
        path = self.root/'pipeline'; writes = 0
        def killed(handle, row):
            nonlocal writes
            write_record(handle, row)
            if row['split'] == 'validation':
                writes += 1
                if writes == 5: raise KeyboardInterrupt()
        with patch('carbon.probes.write_record', side_effect=killed):
            with self.assertRaises(KeyboardInterrupt):
                run_wait_pipeline(self.b, path, interval_seconds=300)
        model = (path/'model/model.json').read_bytes()
        with patch('carbon.wait_pipeline.fit_wait_model', side_effect=AssertionError('must not refit')):
            run_wait_pipeline(self.b, path, interval_seconds=300, resume=True)
            run_wait_pipeline(self.b, path, interval_seconds=300, resume=True)
        self.assertEqual((path/'model/model.json').read_bytes(), model)
        self.assertEqual(load_json(path/'validation-diagnostics/manifest.json')['status'], 'complete')
        # A stale or modified completed model is not silently reused or overwritten.
        p = path/'model/model.json'; p.write_bytes(model+b' ')
        with self.assertRaisesRegex(ContractError, 'hash mismatch'):
            run_wait_pipeline(self.b, path, interval_seconds=300, resume=True)
        self.assertEqual(p.read_bytes(), model+b' ')

    def test_e1_retries_only_unfinished_diagnostic_and_preserves_it(self):
        self.b.raw['trace']['role'] = 'workload_template'
        path = self.root/'e1'
        run_wait_pipeline(self.b, path, interval_seconds=300, e1=True)
        meta = load_json(path/'validation-diagnostics/manifest.json')
        meta['status'] = 'running'
        (path/'validation-diagnostics/manifest.json').write_text(json.dumps(meta))
        run_wait_pipeline(self.b, path, interval_seconds=300, resume=True, e1=True)
        backups = list(path.glob('validation-diagnostics.interrupted-*'))
        self.assertEqual(len(backups), 1)
        self.assertEqual(load_json(backups[0]/'manifest.json')['status'], 'running')
        self.assertEqual(load_json(path/'dependence-queue/manifest.json')['status'], 'complete')
