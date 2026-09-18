from contextlib import redirect_stdout
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from carbon.common import ContractError, digest, load_json
from carbon.config import Bundle
from carbon.main_pilot import run_main_pilot
from tests.helpers import ROOT

spec = importlib.util.spec_from_file_location('main_train_script', ROOT/'scripts/main_train.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class MainTrainTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); self.bundle = Bundle(ROOT/'configs/synthetic-p2.json')
        stream = redirect_stdout(io.StringIO()); stream.__enter__()
        self.addCleanup(stream.__exit__, None, None, None)

    def test_new_budget_recomputes_fixed_misses_and_repeated_chunks_are_not_switches(self):
        refs = {'carbon_reference_g_per_kappa': {'0.25': 10., '1.0': 20.}}
        row = {'final_status': 'completed', 'censor_flag': False, 'tat_hours': 50.,
               'deadline_miss': False, 'chunk_count': 5, 'nodehours': 200.,
               'carbon_g_per_kappa': {'0.25': 15., '1.0': 25.}}
        result = runner.summarize([row], 40., refs, [[4]*5])
        self.assertEqual(result['deadline_misses'], 1)
        self.assertEqual(result['worst_normalized_carbon'], 1.5)
        self.assertEqual(result['observed_scale_change_fraction'], 0)
        self.assertFalse(result['empirical_target_met'])
        result = runner.summarize([row], 60., refs, [[4, 16, 4, 4, 4]])
        self.assertEqual(result['deadline_misses'], 0)
        self.assertEqual(result['observed_scale_change_fraction'], 1)
        self.assertEqual(result['observed_scale_up_fraction'], 1)
        self.assertEqual(result['observed_scale_down_fraction'], 1)

    def test_censoring_does_not_drop_denominator_or_pass_off_partial_carbon_as_total(self):
        refs = {'carbon_reference_g_per_kappa': {'0.25': 10., '1.0': 20.}}
        row = {'final_status': 'censored', 'censor_flag': True, 'tat_hours': None,
               'observed_elapsed_hours': 20., 'nodehours': None, 'chunk_count': 1,
               'carbon_g_per_kappa': None, 'observed_carbon_g_per_kappa': {'0.25': 1., '1.0': 2.}}
        later = {**row, 'observed_elapsed_hours': 40.}
        result = runner.summarize([row, later], 40., refs, [[4], [16]])
        self.assertEqual(result['incomplete'], 2)
        self.assertEqual(result['miss_rate_bounds'], [.5, 1.])
        self.assertIsNone(result['mean_tat_hours'])
        self.assertIsNone(result['mean_modeled_carbon_g_per_kappa'])
        self.assertIsNone(result['worst_normalized_carbon'])
        self.assertFalse(result['empirical_target_met'])

    def test_reuses_pilot_recovers_partial_validation_and_rejects_tampering(self):
        pilot = self.root/'pilot'; output = self.root/'train'
        run_main_pilot(self.bundle, pilot)
        before = {str(p.relative_to(pilot)): digest(p) for p in pilot.rglob('*') if p.is_file()}
        original = runner.evaluate_policy
        interrupted = False
        def interrupt_second(bundle, path, predictor, checkpoint, **kwargs):
            nonlocal interrupted
            if path.name == 'validation-000004' and not interrupted:
                interrupted = True
                path.mkdir(); (path/'partial.txt').write_text('preserve me')
                raise KeyboardInterrupt()
            return original(bundle, path, predictor, checkpoint, **kwargs)
        with patch.object(runner, 'ITERATIONS', 4), patch.object(runner, 'EPISODES_PER_BUDGET', 2), \
             patch.object(runner, 'VALIDATION_ITERATIONS', (2, 4)), \
             patch('carbon.main_pilot.run_fixed', side_effect=AssertionError('no repeated fixed replay')), \
             patch('carbon.policy_inputs.WaitPredictor.load', side_effect=AssertionError('no predictor')):
            with patch.object(runner, 'evaluate_policy', side_effect=interrupt_second):
                with self.assertRaises(KeyboardInterrupt): runner.run(self.bundle, pilot, output)
            partial = load_json(output/'validation-summary.json')
            self.assertEqual(partial['status'], 'partial')
            self.assertEqual([s['iteration'] for s in partial['checkpoints']], [2])
            with patch('carbon.main_pilot.train_policy', side_effect=AssertionError('no repeated training')), \
                 patch.object(runner, 'evaluate_policy', wraps=original) as evaluate:
                result = runner.run(self.bundle, pilot, output, resume=True)
            self.assertEqual(evaluate.call_count, 1)
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(result['ppo_episodes'], 32)
            self.assertEqual([s['iteration'] for s in result['checkpoints']], [2, 4])
            self.assertEqual(len(result['fixed']), 16)
            self.assertEqual(len(result['fixed_mix']), 4)
            self.assertTrue(list(output.glob('validation-000004.interrupted-*/partial.txt')))
            self.assertIn('Fixed-Mix', (output/'validation-summary.md').read_text())
            for stage in result['checkpoints']:
                self.assertEqual(len(stage['operating_points']), 4)
                self.assertTrue(all(p['summary']['outcomes'] == 1 for p in stage['operating_points']))
            with patch.object(runner, 'evaluate_policy', side_effect=AssertionError('no repeated evaluation')):
                self.assertEqual(runner.run(self.bundle, pilot, output, resume=True), result)
            path = output/'validation-000002/chunks.jsonl'; path.write_bytes(path.read_bytes()+b' ')
            with self.assertRaisesRegex(ContractError, 'seal differs'):
                runner.run(self.bundle, pilot, output, resume=True)
        after = {str(p.relative_to(pilot)): digest(p) for p in pilot.rglob('*') if p.is_file()}
        self.assertEqual(after, before)

    def test_changed_training_plan_and_changed_pilot_are_rejected(self):
        pilot = self.root/'pilot'; output = self.root/'train'
        run_main_pilot(self.bundle, pilot)
        with patch.object(runner, 'ITERATIONS', 2), patch.object(runner, 'EPISODES_PER_BUDGET', 2), \
             patch.object(runner, 'VALIDATION_ITERATIONS', (2,)):
            runner.run(self.bundle, pilot, output)
            with self.assertRaisesRegex(ContractError, 'Run inputs/settings/software changed'):
                runner.run(self.bundle, pilot, output, seed=23, resume=True)
        path = pilot/'fixed-train/episodes.jsonl'; path.write_bytes(path.read_bytes()+b' ')
        with self.assertRaisesRegex(ContractError, 'seal differs'):
            runner.reuse_pilot(self.bundle, pilot)
