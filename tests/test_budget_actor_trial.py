"""Tiny synthetic control, including paired arrivals and interrupted validation."""

from contextlib import redirect_stdout
import csv
from datetime import timedelta
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
from carbon.common import digest, iso, json_text, load_json, timestamp
from carbon.config import Bundle
from carbon.main_pilot import make_budget_grid
from carbon.policy_runner import settings_for, train_policy
from carbon.runner import provenance, run_fixed
from tests.helpers import ROOT

with patch.object(sys, 'path', [str(ROOT/'scripts'), *sys.path]):
    spec = importlib.util.spec_from_file_location('budget_actor_trial_script', ROOT/'scripts/budget_actor_trial.py')
    trial = importlib.util.module_from_spec(spec); spec.loader.exec_module(trial)


class BudgetActorTrialTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        stream = redirect_stdout(io.StringIO()); stream.__enter__(); self.addCleanup(stream.__exit__, None, None, None)

    def test_paired_arrivals_and_validation_resume_without_retraining(self):
        raw = load_json(ROOT/'configs/synthetic-p2.json'); raw['root'] = str(ROOT)
        with (ROOT/raw['cohort']['path']).open() as f: cohort = list(csv.DictReader(f))
        original = next(r for r in cohort if r['split'] == 'train')
        for i in (1, 2):
            cohort.append({**original, 'episode_id': f'extra-train-{i}',
                           'arrival_utc': iso(timestamp(original['arrival_utc'])+timedelta(minutes=2*i))})
        cohort_path = self.root/'cohort.csv'
        with cohort_path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(original)); writer.writeheader(); writer.writerows(cohort)
        raw['cohort'].update(path=str(cohort_path), sha256=digest(cohort_path))
        config = self.root/'config.json'; config.write_text(json_text(raw)); bundle = Bundle(config)
        pilot = self.root/'pilot'; pilot.mkdir()
        run_fixed(bundle, pilot/'fixed-train', bundle.raw['cluster']['allowed_nodes'], split='train')
        refs = make_references(pilot/'fixed-train', pilot/'references.json')
        fixed = fixed_records(pilot/'fixed-train/episodes.jsonl', 'train', bundle.raw['cluster']['allowed_nodes'])
        grid = make_budget_grid(refs, fixed)
        previous = settings_for(2, episodes_per_budget=2, minibatch_episodes=16,
                                budgets=grid['budget_multipliers'], actor_interaction='product')
        product = self.root/'product'
        software = provenance(ROOT)
        with patch('carbon.policy_runner.provenance', return_value=software):
            train_policy(bundle, product, None, pilot/'references.json', previous)
        old_rows = trial.records(product/'episodes.jsonl')
        self.assertGreater(len({r['episode_id'] for r in old_rows}), 1)
        reuse = refs, grid, {'comparisons': [], 'fixed_mix': []}, previous, {'pilot_artifacts': {}}
        original_validation = trial.validation_stage
        def interrupt(bundle, stage, checkpoint, settings):
            if stage.name == 'validation-000002':
                stage.mkdir(); (stage/'partial.txt').write_text('preserved'); raise KeyboardInterrupt()
            return original_validation(bundle, stage, checkpoint, settings)
        output = self.root/'trial'
        before = {str(p): digest(p) for p in product.rglob('*') if p.is_file()}
        with patch.object(trial, 'ITERATIONS', 2), patch.object(trial, 'EPISODES_PER_BUDGET', 2), \
             patch.object(trial, 'VALIDATION_ITERATIONS', (1, 2)), patch.object(trial, 'reuse_inputs', return_value=reuse), \
             patch.object(trial, 'frozen_product', return_value=(previous, old_rows, [], {})), \
             patch.object(trial, 'provenance', return_value=software), \
             patch('carbon.policy_runner.provenance', return_value=software), \
             patch('carbon.main_pilot.provenance', return_value=software), \
             patch('main_train.provenance', return_value=software):
            args = (bundle, pilot, self.root/'main', self.root/'diagnosis', product, self.root/'learning', output)
            with patch.object(trial, 'validation_stage', side_effect=interrupt):
                with self.assertRaises(KeyboardInterrupt): trial.run(*args)
            self.assertEqual(load_json(output/'budget-actor-summary.json')['status'], 'partial')
            with patch('carbon.main_pilot.train_policy', side_effect=AssertionError('must not retrain')):
                report = trial.run(*args, resume=True)
            self.assertEqual(report['status'], 'complete')
            self.assertEqual(report['matched_training_arrivals'], 16)
            self.assertEqual(report['settings']['actor_budget_mode'], 'independent')
            self.assertTrue(list(output.glob('validation-000002.interrupted-*/partial.txt')))
            self.assertEqual(before, {str(p): digest(p) for p in product.rglob('*') if p.is_file()})
            with patch('carbon.main_pilot.train_policy', side_effect=AssertionError('must not retrain')), \
                 patch('main_train.evaluate_policy', side_effect=AssertionError('must not reevaluate')):
                self.assertEqual(trial.run(*args, resume=True), report)

    def test_launcher_guard_logging_and_resume(self):
        script = ROOT/'scripts/sophia_budget_actor.sh'
        env = {k: v for k, v in os.environ.items() if not k.startswith(('PBS_', 'CARBON_'))}
        result = subprocess.run(['bash', str(script)], cwd=self.root, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        (self.root/'src/carbon').mkdir(parents=True)
        fake = self.root/'fake-python'
        fake.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@"\necho fake-stderr >&2\nexit 7\n'); fake.chmod(0o700)
        env.update(PBS_JOBID='synthetic-only', PBS_O_WORKDIR=str(self.root), CARBON_PYTHON=str(fake), CARBON_RESUME='1')
        result = subprocess.run(['bash', str(script)], cwd=self.root, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 7); self.assertIn('--resume', result.stdout)
        log, = (self.root/'out').glob('budget-actor.*.log')
        self.assertIn('scripts/budget_actor_trial.py', log.read_text()); self.assertIn('fake-stderr', log.read_text())


if __name__ == '__main__': unittest.main()
