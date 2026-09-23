"""Frontera migration contracts, tested only with tiny synthetic inputs/stub commands."""
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from carbon.common import ContractError, digest, json_text, load_json
from carbon.config import Bundle
from carbon.main_pilot import fixed_stage
from tests.helpers import ROOT

with patch.object(sys, 'path', [str(ROOT/'scripts'), *sys.path]):
    import frontera_weighted84 as entry
    import weighted84_trial as weighted


class FronteraEntryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        raw = load_json(ROOT/'configs/synthetic-p2.json')
        raw['root'] = str(ROOT); raw['cluster']['nodes'] = 84
        raw['cluster']['allowed_nodes'] = [4, 16, 64]
        raw['workload']['profiles']['64'] = {**raw['workload']['profiles']['32'], 'accumulation': 2}
        raw['workload']['profiles'] = {str(n): raw['workload']['profiles'][str(n)] for n in (4,16,64)}
        raw['trace']['role'] = 'workload_template'; raw['execution']['initial_state_mode'] = 'empty_warmup'
        self.config = self.root/'config.json'; self.config.write_text(json_text(raw))
        self.bundle = Bundle(self.config)
        self.source = self.root/'fixed'
        self.output = self.root/'weighted'
        self.stack = redirect_stdout(io.StringIO()); self.stack.__enter__()
        self.addCleanup(self.stack.__exit__, None, None, None)

    def test_check_does_not_replay_or_create_results(self):
        with patch('carbon.main_pilot.fixed_stage') as fixed:
            entry.prepare_fixed(self.bundle, self.source, check=True)
            fixed.assert_not_called()
        self.assertFalse(self.source.exists())

    def test_fixed_interrupt_resume_reuse_and_tamper(self):
        def interrupted(bundle, directory, split):
            if split == 'validation':
                raise RuntimeError('simulated job timeout')
            return fixed_stage(bundle, directory, split)
        with patch('carbon.main_pilot.fixed_stage', side_effect=interrupted):
            with self.assertRaisesRegex(RuntimeError, 'timeout'):
                entry.prepare_fixed(self.bundle, self.source)
        original = (self.source/'fixed-train/episodes.jsonl').read_bytes()
        from carbon.runner import run_fixed
        with patch('carbon.main_pilot.run_fixed', wraps=run_fixed) as replay:
            entry.prepare_fixed(self.bundle, self.source)
            self.assertEqual(replay.call_count, 1)
            self.assertEqual(replay.call_args.args[3], 'validation')
        self.assertEqual(original, (self.source/'fixed-train/episodes.jsonl').read_bytes())
        weighted.source_inputs(self.bundle, self.source)
        all_hashes = {str(p): digest(p) for p in self.source.rglob('*') if p.is_file()}
        with patch('carbon.main_pilot.fixed_stage') as fixed:
            entry.prepare_fixed(self.bundle, self.source)
            fixed.assert_not_called()
        self.assertEqual(all_hashes, {str(p): digest(p) for p in self.source.rglob('*') if p.is_file()})
        broken = self.source/'fixed-train/chunks.jsonl'
        broken.write_text(broken.read_text()+'\n')
        with self.assertRaisesRegex(ContractError, 'Sealed artifact changed'):
            entry.prepare_fixed(self.bundle, self.source)

    def test_foreign_partial_source_is_not_overwritten(self):
        self.source.mkdir(); (self.source/'keep').write_text('prior experiment')
        with self.assertRaisesRegex(ContractError, 'foreign'):
            entry.prepare_fixed(self.bundle, self.source)
        self.assertEqual((self.source/'keep').read_text(), 'prior experiment')

    def test_allocation_guard_rejects_login_even_with_salloc_id(self):
        with patch.dict(os.environ, {}, clear=True):
            entry.require_allocation(check=True)
            with self.assertRaisesRegex(RuntimeError, 'sbatch'):
                entry.require_allocation()
        with patch.dict(os.environ, {'SLURM_JOB_ID':'fake'}), patch.object(entry.socket, 'gethostname', return_value='login4.frontera'):
            with self.assertRaisesRegex(RuntimeError, 'allocated compute'):
                entry.require_allocation()
        with patch.dict(os.environ, {'SLURM_JOB_ID':'fake'}), patch.object(entry.socket, 'gethostname', return_value='c001.frontera'):
            entry.require_allocation()

    def test_resume_after_fixed_timeout_starts_new_ppo(self):
        argv = ['frontera', '--config', str(self.config), '--source', str(self.source),
                '--output', str(self.output), '--resume', '--stage', 'train']
        with patch.object(sys, 'argv', argv), patch.object(entry, 'require_allocation'), \
                patch.object(entry, 'prepare_fixed') as prepare, patch.object(weighted, 'run') as run:
            entry.main()
            prepare.assert_called_once()
            self.assertEqual(run.call_args.kwargs, {'resume':False, 'stage':'train'})
        self.output.mkdir()
        with patch.object(sys, 'argv', argv), patch.object(entry, 'require_allocation'), \
                patch.object(entry, 'prepare_fixed'), patch.object(weighted, 'run') as run:
            entry.main()
            self.assertTrue(run.call_args.kwargs['resume'])

    def test_slurm_spooled_launcher_forwards_args_and_captures_stderr(self):
        (self.root/'src/carbon').mkdir(parents=True)
        script = self.root/'spooled.sh'
        script.write_bytes((ROOT/'scripts/frontera_weighted84.sh').read_bytes())
        fake = self.root/'fake-python'
        fake.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\necho captured-stderr >&2\nexit 7\n')
        fake.chmod(0o700)
        env = {k:v for k,v in os.environ.items() if not k.startswith(('SLURM_', 'CARBON_'))}
        env.update(SLURM_JOB_ID='123', SLURM_SUBMIT_DIR=str(self.root), CARBON_PYTHON=str(fake))
        outside = self.root/'outside'; outside.mkdir()
        result = subprocess.run(['bash', str(script), '--resume', '--stage', 'train'],
                                cwd=outside, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 7)
        for value in ('--resume', 'train', 'scripts/frontera_weighted84.py'):
            self.assertIn(value, result.stdout)
        log, = (self.root/'out').glob('*.log')
        self.assertIn('captured-stderr', log.read_text())


if __name__ == '__main__':
    unittest.main()
