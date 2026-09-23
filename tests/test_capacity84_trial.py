"""C84 orchestration checks on a six-hour synthetic trace, never real replay."""

from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from carbon.common import ContractError, digest, json_text, load_json
from carbon.config import Bundle
from carbon.policy_runner import optimize_ppo
from tests.helpers import ROOT

with patch.object(sys, 'path', [str(ROOT/'scripts'), *sys.path]):
    import capacity84_trial as trial
    import capacity84_report as reporting


class Capacity84Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        stream = redirect_stdout(io.StringIO()); stream.__enter__()
        self.addCleanup(stream.__exit__, None, None, None)

    def synthetic_bundle(self):
        raw = load_json(ROOT/'configs/synthetic-p2.json')
        raw['root'] = str(ROOT); raw['cluster']['nodes'] = 84
        raw['cluster']['allowed_nodes'] = [4, 16, 64]
        raw['workload']['profiles']['64'] = {**raw['workload']['profiles']['32'], 'accumulation': 2}
        raw['workload']['profiles'] = {str(n): raw['workload']['profiles'][str(n)] for n in [4, 16, 64]}
        raw['trace']['role'] = 'workload_template'; raw['execution']['initial_state_mode'] = 'empty_warmup'
        path = self.root/'synthetic-c84.json'; path.write_text(json_text(raw))
        return Bundle(path)

    def test_production_config_changes_only_capacity_actions_and_labels(self):
        old = load_json(ROOT/'configs/amsp-frontera-7b.development.json')
        new = load_json(ROOT/'configs/amsp-frontera-7b-c84.development.json')
        trial.validate_scope(SimpleNamespace(raw=new, root=ROOT))
        self.assertEqual(new['cluster']['nodes'], 84)
        self.assertEqual(new['cluster']['allowed_nodes'], [4, 16, 64])
        self.assertEqual(set(new['workload']['profiles']), {'4', '16', '64'})
        # Compare all retained physics/data/learning-independent configuration.
        aligned = deepcopy(new)
        aligned['panel'] = old['panel']; aligned['cluster'] = old['cluster']
        aligned['workload']['profiles']['128'] = old['workload']['profiles']['128']
        self.assertEqual(aligned, old)
        for field in ('scheduler', 'max_request_seconds', 'walltime_resolution_seconds'):
            self.assertEqual(new['cluster'][field], old['cluster'][field])
        with self.assertRaisesRegex(ContractError, 'Expected C84'):
            trial.validate_scope(SimpleNamespace(raw=old, root=ROOT))
        new['workload']['optimizer_updates'] += 1
        with self.assertRaisesRegex(ContractError, 'Only capacity'):
            trial.validate_scope(SimpleNamespace(raw=new, root=ROOT))

    def test_full_synthetic_pipeline_resume_curves_and_artifact_guards(self):
        bundle = self.synthetic_bundle(); output = self.root/'run'
        software = {'source_sha256': {p.name: digest(p) for p in (ROOT/'src/carbon').glob('*.py')},
                    'git_commit': 'synthetic-test', 'git_status': 'synthetic-test'}
        with ExitStack() as stack:
            for target in ('capacity84_trial.provenance', 'carbon.runner.provenance', 'carbon.probes.provenance',
                           'carbon.policy_runner.provenance', 'carbon.main_pilot.provenance', 'main_train.provenance'):
                stack.enter_context(patch(target, return_value=software))
            stack.enter_context(patch.object(trial, 'ITERATIONS', 2))
            stack.enter_context(patch.object(trial, 'EPISODES_PER_BUDGET', 1))
            stack.enter_context(patch.object(trial, 'VALIDATION_ITERATIONS', (1, 2)))
            stack.enter_context(patch.object(trial, 'PROBE_INTERVAL_SECONDS', 300))
            prepared = trial.run(bundle, output, stage='prepare')
            self.assertEqual(prepared['status'], 'prepared')
            self.assertFalse(list(output.glob('ppo-attempt-*')))
            self.assertEqual(len(prepared['baselines']), 28)  # 3 fixed + best + mix + 2 planners, four budgets
            self.assertEqual(set(prepared['fixed_mix'][0]['validation_miss_rates']), {'4', '16', '64'})
            self.assertEqual(prepared['nodes'], [4, 16, 64])
            before = load_json(output/'prepared-seal.json')
            calls = 0
            def interrupt_training(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise KeyboardInterrupt()
                return optimize_ppo(*args, **kwargs)
            with patch('carbon.policy_runner.optimize_ppo', side_effect=interrupt_training):
                with self.assertRaises(KeyboardInterrupt):
                    trial.run(bundle, output, stage='train', resume=True)
            self.assertEqual(load_json(output/'ppo-attempt-000/manifest.json')['completed_iteration'], 1)
            self.assertEqual(len(trial.records(output/'ppo-attempt-000/episodes.jsonl')), 8)
            original = trial.validation_stage
            def interrupt(bundle, path, checkpoint, settings):
                if path.name == 'validation-000002':
                    path.mkdir(); (path/'partial.txt').write_text('preserve this')
                    raise KeyboardInterrupt()
                return original(bundle, path, checkpoint, settings)
            with patch.object(trial, 'validation_stage', side_effect=interrupt):
                with self.assertRaises(KeyboardInterrupt):
                    trial.run(bundle, output, stage='train', resume=True)
            self.assertEqual(load_json(output/'capacity84-summary.json')['status'], 'partial')
            with patch('carbon.main_pilot.train_policy', side_effect=AssertionError('must not retrain completed PPO')):
                complete = trial.run(bundle, output, stage='train', resume=True)
            self.assertEqual(complete['status'], 'complete')
            self.assertEqual(complete['ppo_episodes'], 8)
            self.assertEqual(load_json(output/'ppo-attempt-001/manifest.json')['first_iteration'], 2)
            self.assertEqual([c['iteration'] for c in complete['checkpoints']], [1, 2])
            self.assertTrue(list(output.glob('validation-000002.interrupted-*/partial.txt')))
            for c in complete['checkpoints']:
                self.assertEqual(len(c['operating_points']), 4)
                self.assertTrue(all(len(p['mean_probabilities']) == 3 for p in c['actor']['operating_points']))
            for name in ('budget-curves', 'checkpoint-curves', 'policy-behavior'):
                self.assertGreater((output/(name+'.png')).stat().st_size, 1000)
                self.assertTrue((output/(name+'.pdf')).read_bytes().startswith(b'%PDF'))
            shutil.copyfile(output/'budget-curves.png', '/tmp/carbon-c84-synthetic-curve-check.png')
            self.assertEqual(before, load_json(output/'prepared-seal.json'))
            self.assertTrue(all(digest(output/name) == sha for name, sha in before.items()))
            # Preparation rerun preserves the evaluated report and does not reexecute stages.
            with patch('carbon.main_pilot.run_fixed', side_effect=AssertionError('must reuse sealed fixed')), \
                 patch('diagnose_main.run_planners', side_effect=AssertionError('must reuse sealed planners')):
                preserved = trial.run(bundle, output, stage='prepare', resume=True)
            self.assertEqual(preserved['status'], 'complete')
            self.assertEqual(len(preserved['checkpoints']), 2)
            curves = reporting.curve_rows(complete)
            self.assertEqual(len(curves), 36)
            self.assertFalse(any('128' in p['method'] for p in curves))
            with patch.object(trial, 'ITERATIONS', 3):
                with self.assertRaisesRegex(ContractError, 'inputs/settings/software changed'):
                    trial.run(bundle, output, resume=True)
            path = output/'e1/model/model.json'; path.write_bytes(path.read_bytes()+b' ')
            with self.assertRaisesRegex(ContractError, 'prepared artifact changed'):
                trial.run(bundle, output, resume=True)

    def test_foreign_output_is_rejected_without_modifying_it(self):
        bundle = self.synthetic_bundle(); output = self.root/'old-c128'; output.mkdir()
        (output/'run-plan.json').write_text('{"old": true}')
        before = {p.name: digest(p) for p in output.iterdir()}
        with patch.object(trial, 'provenance', return_value={'source_sha256': {}}):
            with self.assertRaisesRegex(ContractError, 'Unknown existing C84 output'):
                trial.run(bundle, output, resume=True)
        self.assertEqual(before, {p.name: digest(p) for p in output.iterdir()})

    def test_mixture_expectations_use_weighted_tails_and_real_misses(self):
        refs = {'carbon_reference_g_per_kappa': {'0.25': 1., '1.0': 1.}}
        def row(t, c):
            return {'tat_hours': t, 'carbon_g_per_kappa': {'0.25': c, '1.0': c},
                    'nodehours': c, 'chunk_count': 1}
        groups = {'Fixed-4': {'a':row(100., 1.)}, 'Fixed-16': {'a':row(10., 2.)}}
        mix = {'budget_hours': 20., 'empirical_feasible': True, 'weights': {'4': .1, '16': .9},
               'epsilon': .1, 'estimated_objective': 1.9}
        points = trial.fixed_reference_points([], [mix], groups, refs)
        s = points[1]['summary']
        self.assertAlmostEqual(s['mean_tat_hours'], 19.)
        self.assertEqual(s['p95_tat_hours'], 100.)
        self.assertAlmostEqual(s['miss_rate_bounds'][0], .1)  # mean TAT < D does not imply zero misses
        self.assertIsNone(points[0]['summary'])
        self.assertEqual(s['observed_scale_change_fraction'], 0.)

    def test_launcher_requires_allocation_logs_and_forwards_resume(self):
        script = ROOT/'scripts/sophia_capacity84.sh'
        env = {k:v for k,v in os.environ.items() if not k.startswith(('PBS_', 'CARBON_'))}
        r = subprocess.run(['bash', str(script)], cwd=self.root, env=env, capture_output=True, text=True)
        self.assertEqual(r.returncode, 2)
        (self.root/'src/carbon').mkdir(parents=True)
        fake = self.root/'fake-python'
        fake.write_text('#!/usr/bin/env bash\nif [[ "$*" == "-B -" ]]; then cat >/dev/null; exit 0; fi\nprintf "%s\\n" "$@"\necho synthetic-stderr >&2\nexit 7\n')
        fake.chmod(0o700)
        env.update(PBS_JOBID='synthetic-test', CARBON_PYTHON=str(fake), CARBON_RESUME='1', CARBON_STAGE='prepare')
        r = subprocess.run(['bash', str(script)], cwd=self.root, env=env, capture_output=True, text=True)
        self.assertEqual(r.returncode, 7)
        self.assertIn('--resume', r.stdout); self.assertIn('prepare', r.stdout)
        log, = (self.root/'out').glob('capacity84.*.log')
        self.assertIn('synthetic-stderr', log.read_text())
        self.assertIn('scripts/capacity84_trial.py', log.read_text())


if __name__ == '__main__':
    unittest.main()
