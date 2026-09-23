"""Legacy profile import and tiny synthetic replay; no real-cluster training."""
from contextlib import redirect_stdout
from decimal import Decimal
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
from carbon.workload import Workload
from tests.helpers import ROOT

with patch.object(sys, 'path', [str(ROOT/'scripts'), *sys.path]):
    import old_gpt_profiles as profiles
    import old_gpt_trial as trial
    import frontera_old_gpt as entry


class ProfileTests(unittest.TestCase):
    def test_table_conversion_nodes_and_metadata(self):
        asset = load_json(ROOT/profiles.ASSET)
        self.assertEqual(digest(ROOT/asset['source']['snapshot']), asset['source']['snapshot_sha256'])
        expected = {'medium': [139.87,207.09,361.60,640.42],
                    'large': [272.68,410.66,680.,1262.41],
                    'xl': [866.80,1220.,1760.,2894.16]}
        for model, row in expected.items():
            raw = load_json(ROOT/f'configs/old-gpt-frontera-{model}-c84.development.json')
            self.assertEqual(raw['workload'], profiles.workload_config(asset, model))
            self.assertEqual(raw['cluster']['allowed_nodes'], [4,8,16,32])
            self.assertEqual(raw['cluster']['max_request_seconds'], 172800)
            work = Workload(raw['workload'], [4,8,16,32], 172800, 60)
            self.assertEqual(work.total_updates, 100000)
            for n, nodehours in zip(profiles.NODES, row):
                reconstructed = float(Decimal(100000)*n/work.profiles[n].updates_per_hour)
                self.assertAlmostEqual(reconstructed, nodehours, places=9)
                self.assertIn(str(nodehours).rstrip('0').rstrip('.'), (ROOT/'data/old_gpt/source-table.tex').read_text())
            self.assertIn('NOT measurement metadata', raw['workload']['provenance'])
            self.assertIn('physical_gpus_per_node', asset['unknown_measurement_metadata'])

    def test_48h_opportunities_without_changing_work(self):
        asset = load_json(ROOT/profiles.ASSET)
        counts = {}
        for model in ('medium', 'large', 'xl'):
            work = Workload(profiles.workload_config(asset, model), list(profiles.NODES), 172800, 60)
            counts[model] = []
            for n in profiles.NODES:
                remaining, chunks = 100000, 0
                while remaining:
                    plan = work.plan(n, remaining, chunks == 0)
                    self.assertLessEqual(plan.requested.total_seconds(), 172800)
                    remaining -= plan.updates; chunks += 1
                counts[model].append(chunks)
        self.assertEqual(counts, {'medium':[1,1,1,1], 'large':[2,2,1,1], 'xl':[5,4,3,2]})


class PipelineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        capture = redirect_stdout(io.StringIO()); capture.__enter__()
        self.addCleanup(capture.__exit__, None, None, None)
        raw = load_json(ROOT/'configs/synthetic-p2.json')
        raw['root'] = str(ROOT); raw['cluster']['nodes'] = 84
        raw['trace']['role'] = 'workload_template'
        raw['execution']['initial_state_mode'] = 'empty_warmup'
        config = self.root/'config.json'; config.write_text(json_text(raw))
        self.bundle = Bundle(config)
        self.source = self.root/'fixed'
        self.output = self.root/'policy'
        self.settings = dict(iterations=3, observation_iterations=1, episodes_per_alpha=1,
                             validation_iterations=[2,3], ppo_epochs=1, width=8, minibatch_episodes=5)

    def test_read_only_baseline_rebuild_and_foreign_rejection(self):
        entry.prepare_fixed(self.bundle, self.source, check=True)
        self.assertFalse(self.source.exists())
        entry.prepare_fixed(self.bundle, self.source)
        refs, _ = trial.source_inputs(self.bundle, self.source)
        before = {str(p): digest(p) for p in self.source.rglob('*') if p.is_file()}
        entry.prepare_fixed(self.bundle, self.source)
        self.assertEqual(before, {str(p):digest(p) for p in self.source.rglob('*') if p.is_file()})
        self.assertEqual({r['method'] for r in trial.records(self.source/'fixed-train/episodes.jsonl')},
                         {'Fixed-4','Fixed-8','Fixed-16','Fixed-32'})
        raw = load_json(self.root/'config.json'); raw['workload']['name'] = 'another-profile'
        other = self.root/'other.json'; other.write_text(json_text(raw))
        with self.assertRaises(ContractError):
            entry.prepare_fixed(Bundle(other), self.source)
        foreign = self.root/'foreign'; foreign.mkdir(); (foreign/'keep').write_text('AMSP')
        with self.assertRaisesRegex(ContractError, 'foreign'):
            entry.prepare_fixed(self.bundle, foreign)
        self.assertEqual((foreign/'keep').read_text(), 'AMSP')

    def test_four_action_training_report_and_exact_resume(self):
        entry.prepare_fixed(self.bundle, self.source)
        trial.run(self.bundle, self.output, self.source, settings=self.settings)
        summary = load_json(self.output/'old-gpt-summary.json')
        self.assertEqual(len(summary['fixed']), 20)
        self.assertEqual(len(summary['checkpoints']), 10)
        self.assertTrue(all(len(p['initial_mean_probabilities']) == 4 for p in summary['checkpoints']))
        self.assertTrue((self.output/'weighted-curves.pdf').read_bytes().startswith(b'%PDF'))
        self.assertIn('mean_carbon_g_per_kappa', (self.output/'weighted-curves.csv').read_text())
        resumed = self.root/'resumed'
        original = trial.collect
        def interrupt(*args):
            if args[9]['iteration'] == 2:
                raise KeyboardInterrupt('synthetic stop')
            return original(*args)
        with patch.object(trial, 'collect', side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt):
                trial.run(self.bundle, resumed, self.source, settings=self.settings)
        trial.run(self.bundle, resumed, self.source, resume=True, settings=self.settings)
        for name in ('reward-normalizers.json', 'old-gpt-summary.json', 'weighted-curves.csv'):
            self.assertEqual((self.output/name).read_bytes(), (resumed/name).read_bytes())
        left = trial.read_checkpoint(self.output/'iteration-000003')[1]
        right = trial.read_checkpoint(resumed/'iteration-000003')[1]
        self.assertTrue(all(trial.torch.equal(v, right['model'][k]) for k,v in left['model'].items()))
        with self.assertRaisesRegex(ContractError, 'changed'):
            trial.run(self.bundle, resumed, self.source, resume=True, settings={**self.settings, 'seed':12})

    def test_slurm_guard_and_spooled_launcher(self):
        with patch.dict(os.environ, {}, clear=True):
            entry.require_allocation(check=True)
            with self.assertRaisesRegex(RuntimeError, 'sbatch'):
                entry.require_allocation()
        with patch.dict(os.environ, {'SLURM_JOB_ID':'123'}), patch.object(entry.socket, 'gethostname', return_value='login4.frontera'):
            with self.assertRaises(RuntimeError):
                entry.require_allocation()
        (self.root/'src/carbon').mkdir(parents=True)
        script = self.root/'spooled.sh'; script.write_bytes((ROOT/'scripts/frontera_old_gpt.sh').read_bytes())
        fake = self.root/'fake-python'
        fake.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\necho captured-stderr >&2\nexit 7\n'); fake.chmod(0o700)
        env = {k:v for k,v in os.environ.items() if not k.startswith(('SLURM_', 'CARBON_'))}
        env.update(SLURM_JOB_ID='123', SLURM_SUBMIT_DIR=str(self.root), CARBON_PYTHON=str(fake))
        outside = self.root/'outside'; outside.mkdir()
        result = subprocess.run(['bash',str(script),'--model','xl','--resume'], cwd=outside, env=env, capture_output=True,text=True)
        self.assertEqual(result.returncode, 7)
        for value in ('scripts/frontera_old_gpt.py', '--model', 'xl', '--resume', 'captured-stderr'):
            self.assertIn(value, result.stdout)
        log, = (self.root/'out').glob('*.log')
        self.assertIn('captured-stderr', log.read_text())


if __name__ == '__main__':
    unittest.main()
