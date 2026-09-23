"""Frozen-gradient checks using tiny synthetic fixtures only."""

from contextlib import redirect_stdout
from copy import deepcopy
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from carbon.baselines import make_references
from carbon.common import ContractError, digest, load_json
from carbon.config import Bundle
from carbon.learning import ActorCritic, input_tensors, ppo_losses, torch
from carbon.policy_inputs import PolicyInputs
from carbon.policy_runner import settings_for, train_policy
from carbon.runner import provenance, run_fixed, write_manifest
from tests.helpers import ROOT

with patch.object(sys, 'path', [str(ROOT/'scripts'), *sys.path]):
    spec = importlib.util.spec_from_file_location('diagnose_learning_script', ROOT/'scripts/diagnose_learning.py')
    runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)


class LearningDiagnosisTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        stream = redirect_stdout(io.StringIO()); stream.__enter__(); self.addCleanup(stream.__exit__, None, None, None)
        torch.set_num_threads(1); torch.manual_seed(11)

    def fixture(self):
        bundle = Bundle(ROOT/'configs/synthetic-p2.json')
        run_fixed(bundle, self.root/'fixed', bundle.raw['cluster']['allowed_nodes'], split='train')
        refs_path = self.root/'references.json'; refs = make_references(self.root/'fixed', refs_path)
        settings = settings_for(2, budgets=[1., 2.], episodes_per_budget=2, minibatch_episodes=2,
                                epochs=1, actor_interaction='product')
        trial = self.root/'trial'; trial.mkdir()
        software = provenance(ROOT)
        with patch('carbon.policy_runner.provenance', return_value=software):
            train_policy(bundle, trial/'ppo-attempt-000', None, refs_path, settings)
        plan = {'config_sha256': bundle.manifest['config_sha256'], 'asset_sha256': bundle.manifest['asset_sha256'],
                'source_sha256': software['source_sha256'], 'torch': str(torch.__version__),
                'script_sha256': {}, 'settings': settings}
        write_manifest(trial/'run-plan.json', plan)
        meta = load_json(trial/'ppo-attempt-000/manifest.json')
        write_manifest(trial/'interaction-summary.json', {'status': 'complete', 'settings': settings,
                       'run_plan_sha256': digest(trial/'run-plan.json'),
                       'ppo_episodes': meta['total_episodes'], 'ppo_chunks': meta['total_chunks']})
        return bundle, trial, refs, settings, software

    def test_two_round_replay_checks_frozen_behavior_and_resumes_without_replay(self):
        bundle, trial, refs, settings, software = self.fixture()
        before = {str(p): digest(p) for p in trial.rglob('*') if p.is_file()}
        output = self.root/'diagnosis'
        with patch.object(runner, 'ITERATIONS', (1, 2)), patch.object(runner, 'provenance', return_value=software):
            report = runner.run(bundle, trial, output)
            self.assertEqual(report['status'], 'complete')
            self.assertEqual(sum(r['verified_episodes'] for r in report['rounds']), 8)
            self.assertEqual(before, {str(p): digest(p) for p in trial.rglob('*') if p.is_file()})
            with patch.object(runner, 'reconstruct', side_effect=AssertionError('must reuse sealed stages')):
                self.assertEqual(runner.run(bundle, trial, output, resume=True), report)
            stage = output/'round-000001/gradients.json'; stage.write_text('{}')
            with self.assertRaisesRegex(ContractError, 'seal differs'):
                runner.run(bundle, trial, output, resume=True)

    def test_changed_input_probability_or_replay_outcome_is_rejected(self):
        bundle, trial, refs, settings, software = self.fixture()
        with patch.object(runner, 'ITERATIONS', (1, 2)), patch.object(runner, 'provenance', return_value=software):
            _, _, attempt, _, episodes, groups, _ = runner.load_trial(bundle, trial)
        encoder = PolicyInputs(bundle, None, refs)
        model = runner.pre_update_model(bundle, encoder, settings, attempt, 1)
        row = next(e for e in episodes if e['iteration'] == 1)
        for field, value, message in [('policy_input_sha256', 'wrong', 'input differs'),
                                      ('policy_probabilities', [0., 0., 0., 1.], 'probabilities differ'),
                                      ('queue_wait_hours', -1., 'replay outcome differs')]:
            changed = deepcopy(groups); changed[runner.row_key(row)][0][field] = value
            with self.assertRaisesRegex(ContractError, message):
                runner.reconstruct(bundle, encoder, model, settings, [row], changed)

    def test_decomposed_gradient_matches_actual_ppo_actor_loss(self):
        model = ActorCritic(3, 2, width=4, interaction='product')
        settings = settings_for(1, budgets=[1.], actor_interaction='product')
        dual = {'weights': [.2, .8], 'lambda': 1.7}
        encoded = {'global': [1., 2., 3.], 'actions': [[.5, 1.], [1., .5]], 'mask': [True, True]}
        with torch.no_grad():
            dist, values = model(*input_tensors([encoded]))
        step = {'input': encoded, 'action': 1, 'log_probability': float(dist.log_prob(torch.tensor([1]))[0]),
                'values': values[0].tolist()}
        rollout = [{'beta': 1., 'steps': [step, deepcopy(step)], 'returns': [[1., 2., 1.], [.3, .5, 1.]]},
                   {'beta': 1., 'steps': [deepcopy(step)], 'returns': [[.5, .7, 0.]]}]
        before = {k: v.clone() for k, v in model.state_dict().items()}
        result = runner.gradient_diagnosis(model, rollout, {'1.0': dual}, settings)
        steps = [s for e in rollout for s in e['steps']]; returns = [r for e in rollout for r in e['returns']]
        distribution, predictions = model(*input_tensors([s['input'] for s in steps]))
        adv = torch.tensor([-sum(w*(r-v) for w, r, v in zip(dual['weights']+[dual['lambda']], ret, s['values']))
                            for s, ret in zip(steps, returns)])
        total, _, _, _ = ppo_losses(distribution.log_prob(torch.tensor([s['action'] for s in steps])),
                                   torch.tensor([s['log_probability'] for s in steps]), adv, predictions,
                                   torch.tensor(returns), distribution.entropy(), 2,
                                   value_coefficient=0., entropy_coefficient=settings['entropy'])
        params = [p for n, p in model.named_parameters() if n.startswith('actor_')]
        norm = float(runner.gradient_vector(total, params).norm())
        self.assertAlmostEqual(result['shared_actor_gradient_norm'], norm, places=6)
        self.assertAlmostEqual(result['operating_points'][0]['own_shared_gradient_cosine'], 1., places=7)
        self.assertTrue(all(torch.equal(v, before[k]) for k, v in model.state_dict().items()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_opposing_and_zero_gradients_are_not_reported_as_agreement(self):
        self.assertEqual(runner.cosine(torch.tensor([1., 2.]), torch.tensor([-1., -2.])), -1.)
        self.assertIsNone(runner.cosine(torch.zeros(2), torch.ones(2)))

    def test_launcher_guard_logging_and_resume_forwarding(self):
        script = ROOT/'scripts/sophia_diagnose_learning.sh'
        env = {k: v for k, v in os.environ.items() if not k.startswith(('PBS_', 'CARBON_'))}
        result = subprocess.run(['bash', str(script)], cwd=self.root, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        (self.root/'src/carbon').mkdir(parents=True)
        fake = self.root/'fake-python'
        fake.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@"\necho fake-stderr >&2\nexit 7\n'); fake.chmod(0o700)
        env.update(PBS_JOBID='synthetic-only', PBS_O_WORKDIR=str(self.root), CARBON_PYTHON=str(fake), CARBON_RESUME='1')
        for _ in range(2):
            result = subprocess.run(['bash', str(script)], cwd=self.root, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 7)
            self.assertIn('--resume', result.stdout); self.assertIn('fake-stderr', result.stdout)
        logs = list((self.root/'out').glob('diagnose-learning.*.log'))
        self.assertEqual(len(logs), 2)
        self.assertTrue(all('fake-stderr' in p.read_text() for p in logs))


if __name__ == '__main__':
    unittest.main()
