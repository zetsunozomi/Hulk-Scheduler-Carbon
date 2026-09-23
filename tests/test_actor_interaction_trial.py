"""Tiny synthetic integration; no real trace, production model or long training."""

from contextlib import redirect_stdout
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from carbon.baselines import fixed_records, make_references
from carbon.common import ContractError, digest, load_json
from carbon.config import Bundle
from carbon.main_pilot import make_budget_grid
from carbon.policy_runner import settings_for
from carbon.runner import provenance, run_fixed
from tests.helpers import ROOT

with patch.object(sys, 'path', [str(ROOT/'scripts'), *sys.path]):
    spec = importlib.util.spec_from_file_location('actor_interaction_trial_script', ROOT/'scripts/actor_interaction_trial.py')
    trial = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trial)


class InteractionTrialTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        stream = redirect_stdout(io.StringIO()); stream.__enter__()
        self.addCleanup(stream.__exit__, None, None, None)

    def test_reuse_rejects_replay_feature_and_file_inventory_changes(self):
        before = {'learning.py': 'old', 'policy_runner.py': 'old', 'replay.py': 'same', 'policy_inputs.py': 'same'}
        after = {**before, 'learning.py': 'new'}
        self.assertEqual(trial.policy_only_changes(before, after), {'learning.py': {'before': 'old', 'after': 'new'}})
        for name in ('replay.py', 'policy_inputs.py'):
            with self.assertRaisesRegex(ContractError, 'Non-policy implementation'):
                trial.policy_only_changes(before, {**after, name: 'changed'})
        with self.assertRaisesRegex(ContractError, 'inventory'):
            trial.policy_only_changes(before, {**after, 'new_module.py': 'new'})

    def test_two_round_synthetic_trial_recovers_validation_without_retraining(self):
        bundle = Bundle(ROOT/'configs/synthetic-p2.json')
        pilot = self.root/'pilot'; pilot.mkdir()
        run_fixed(bundle, pilot/'fixed-train', bundle.raw['cluster']['allowed_nodes'], split='train')
        references = make_references(pilot/'fixed-train', pilot/'references.json')
        groups = fixed_records(pilot/'fixed-train/episodes.jsonl', 'train', bundle.raw['cluster']['allowed_nodes'])
        grid = make_budget_grid(references, groups)
        settings = settings_for(2, episodes_per_budget=1, minibatch_episodes=16, seed=11,
                                budgets=grid['budget_multipliers'], actor_interaction='concat')
        reused = references, grid, {'comparisons': [], 'fixed_mix': []}, settings, {'synthetic_binding': True}
        software = provenance(ROOT)
        original = trial.validation_stage
        stopped = False
        def interrupt(bundle, directory, checkpoint, settings):
            nonlocal stopped
            if directory.name == 'validation-000002' and not stopped:
                stopped = True; directory.mkdir(); (directory/'partial.txt').write_text('preserve')
                raise KeyboardInterrupt()
            return original(bundle, directory, checkpoint, settings)
        output = self.root/'trial'
        before = {str(p): digest(p) for p in pilot.rglob('*') if p.is_file()}
        with patch.object(trial, 'ITERATIONS', 2), patch.object(trial, 'EPISODES_PER_BUDGET', 1), \
             patch.object(trial, 'VALIDATION_ITERATIONS', (1, 2)), patch.object(trial, 'reuse_inputs', return_value=reused), \
             patch.object(trial, 'provenance', return_value=software), \
             patch('carbon.policy_runner.provenance', return_value=software), \
             patch('carbon.main_pilot.provenance', return_value=software), \
             patch('main_train.provenance', return_value=software):
            args = (bundle, pilot, self.root/'main', self.root/'diagnosis', output)
            with patch.object(trial, 'validation_stage', side_effect=interrupt):
                with self.assertRaises(KeyboardInterrupt): trial.run(*args)
            partial = load_json(output/'interaction-summary.json')
            self.assertEqual(partial['status'], 'partial')
            self.assertEqual([c['iteration'] for c in partial['checkpoints']], [1])
            with patch('carbon.main_pilot.train_policy', side_effect=AssertionError('do not train again')):
                report = trial.run(*args, resume=True)
            self.assertEqual(report['status'], 'complete')
            self.assertEqual(report['ppo_episodes'], 8)
            self.assertEqual([c['iteration'] for c in report['checkpoints']], [1, 2])
            self.assertEqual(report['settings']['actor_interaction'], 'product')
            self.assertTrue(list(output.glob('validation-000002.interrupted-*/partial.txt')))
            self.assertEqual(before, {str(p): digest(p) for p in pilot.rglob('*') if p.is_file()})
            with patch('carbon.main_pilot.train_policy', side_effect=AssertionError('do not train again')), \
                 patch('main_train.evaluate_policy', side_effect=AssertionError('do not evaluate again')):
                self.assertEqual(trial.run(*args, resume=True), report)

    def test_launcher_blocks_login_and_forwards_interactive_resume(self):
        script = ROOT/'scripts/sophia_actor_interaction.sh'
        env = {k: v for k, v in os.environ.items() if not k.startswith(('PBS_', 'CARBON_'))}
        result = subprocess.run(['bash', str(script)], cwd=self.root, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('PBS interactive allocation', result.stderr)
        capture = self.root/'args'; fake = self.root/'python'
        fake.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$CARBON_CAPTURE"\n'); fake.chmod(0o700)
        (self.root/'src/carbon').mkdir(parents=True)
        env.update(PBS_JOBID='synthetic-not-a-real-job', PBS_O_WORKDIR=str(self.root),
                   CARBON_PYTHON=str(fake), CARBON_CAPTURE=str(capture))
        result = subprocess.run(['bash', str(script), '--resume'], cwd=self.root, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        args = capture.read_text().splitlines()
        self.assertEqual(args[:3], ['-u', '-B', 'scripts/actor_interaction_trial.py'])
        self.assertEqual(args[-1], '--resume')
        self.assertIn('results/amsp-interaction-frontera-7b-seed11', args)


if __name__ == '__main__':
    unittest.main()
