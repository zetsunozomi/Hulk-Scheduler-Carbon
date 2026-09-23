"""Small synthetic replay checks; never execute a real experiment on login nodes."""

from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from carbon.baselines import make_references
from carbon.common import ContractError, digest, json_text, load_json
from carbon.config import Bundle
from carbon.learning import torch
from carbon.policy_runner import configure_torch
from carbon.runner import run_fixed
from tests.helpers import ROOT

with patch.object(sys, 'path', [str(ROOT/'scripts'), *sys.path]):
    import weighted84_trial as trial
    import weighted_policy as policy


class WeightedTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        stream = redirect_stdout(io.StringIO()); stream.__enter__()
        self.addCleanup(stream.__exit__, None, None, None)
        self.settings = dict(iterations=7, observation_iterations=5, episodes_per_alpha=1,
                             validation_iterations=[6, 7], ppo_epochs=1, width=16, minibatch_episodes=5)

    def fixture(self):
        raw = load_json(ROOT/'configs/synthetic-p2.json')
        raw['root'] = str(ROOT); raw['cluster']['nodes'] = 84
        raw['cluster']['allowed_nodes'] = [4, 16, 64]
        raw['workload']['profiles']['64'] = {**raw['workload']['profiles']['32'], 'accumulation': 2}
        raw['workload']['profiles'] = {str(n): raw['workload']['profiles'][str(n)] for n in (4,16,64)}
        raw['trace']['role'] = 'workload_template'; raw['execution']['initial_state_mode'] = 'empty_warmup'
        path = self.root/'config.json'; path.write_text(json_text(raw))
        bundle = Bundle(path)
        source = self.root/'fixed-source'; source.mkdir()
        software = {'source_sha256': {p.name: digest(p) for p in (ROOT/'src/carbon').glob('*.py')}}
        with patch('carbon.runner.provenance', return_value=software):
            for split in ('train', 'validation'):
                directory = source/f'fixed-{split}'
                run_fixed(bundle, directory, [4, 16, 64], split)
                trial.seal(directory, ['manifest.json', 'episodes.jsonl', 'chunks.jsonl'])
        make_references(source/'fixed-train', source/'references.json')
        return bundle, source

    def test_weighted_reward_endpoints_and_complete_episode_means(self):
        warmup = dict(count=0, tat_sum_hours=0., carbon_sum_g_per_kappa=0., frozen=False)
        rows = [dict(split='train', final_status='completed', tat_hours=t, carbon_g_per_kappa={'1.0': c})
                for t, c in [(2., 10.), (6., 30.)]]
        policy.observe(warmup, rows, '1.0')
        refs = policy.freeze(warmup, '1.0', 5)
        self.assertEqual((refs['tat_hours'], refs['carbon_g_per_kappa']), (4., 20.))
        with self.assertRaisesRegex(ContractError, 'frozen'):
            policy.observe(warmup, rows, '1.0')
        increments = [[1., 3.], [4., 9.], [3., 8.]]
        returns = policy.returns_to_go(increments, refs)
        self.assertEqual(returns[0], [2., 1.])
        self.assertEqual(policy.cost(8.,20.,0.,refs), 1.)
        self.assertEqual(policy.cost(8.,20.,1.,refs), 2.)
        for a in trial.DEFAULTS['alphas']:
            self.assertAlmostEqual(sum(-policy.cost(t,c,a,refs) for t,c in increments), -policy.cost(8.,20.,a,refs))

    def test_actor_independence_alpha_zero_and_one(self):
        configure_torch(11, 1)
        model = policy.WeightedActorCritic(3, 2, [0., 1.], 8)
        for left, right in zip(model.actors[0].parameters(), model.actors[1].parameters()):
            self.assertTrue(torch.equal(left, right)); self.assertNotEqual(left.data_ptr(), right.data_ptr())
        g = torch.tensor([[1.,0.,0.], [1.,0.,1.]])
        actions = torch.tensor([[[1.,0.],[0.,1.]], [[1.,0.],[0.,1.]]])
        mask = torch.tensor([[True, False], [True, True]])
        distribution, values = model(g, actions, mask)
        self.assertEqual(values.shape, (2,2)); self.assertEqual(distribution.probs[0,1].item(), 0.)
        distribution, _ = model(g[:1], actions[:1], torch.ones_like(mask[:1]))
        distribution.logits[0,0].backward()
        self.assertTrue(all(p.grad is None for p in model.actors[1].parameters()))
        with self.assertRaisesRegex(ContractError, 'Alpha'):
            model(torch.tensor([[1.,0.,.5]]), actions[:1], mask[:1])

    def test_pipeline_five_frozen_rounds_rewards_resume_and_report(self):
        bundle, source = self.fixture()
        output = self.root/'full'
        original_source = {str(p.relative_to(source)): digest(p) for p in source.rglob('*') if p.is_file()}
        result = trial.run(bundle, output, source, settings=self.settings)
        self.assertEqual(result['status'], 'complete'); self.assertEqual(result['total_episodes'], 35)
        checkpoints = [trial.read_checkpoint(output/f'iteration-{i:06d}') for i in range(1,8)]
        encoder = policy.WeightedInputs(bundle, load_json(source/'references.json'))
        configure_torch(11,1)
        initial = policy.WeightedActorCritic(len(encoder.global_names), len(encoder.action_names), trial.DEFAULTS['alphas'], 16)
        for meta, state in checkpoints[:5]:
            self.assertFalse(state['optimizer']['state'])
            self.assertTrue(all(torch.equal(v, initial.state_dict()[k]) for k,v in state['model'].items()))
            self.assertEqual(load_json(output/f'iteration-{meta["iteration"]:06d}/training.json')['optimizer_steps'], 0)
        self.assertTrue(any(not torch.equal(v, initial.state_dict()[k]) for k,v in checkpoints[5][1]['model'].items()))
        refs = load_json(output/'reward-normalizers.json')
        self.assertEqual(refs['episodes'], 25)
        warm_rows = [r for i in range(1,6) for r in trial.records(output/f'iteration-{i:06d}/episodes.jsonl')]
        self.assertAlmostEqual(refs['tat_hours'], sum(r['tat_hours'] for r in warm_rows)/25)
        self.assertAlmostEqual(refs['carbon_g_per_kappa'], sum(r['carbon_g_per_kappa']['1.0'] for r in warm_rows)/25)
        self.assertEqual(checkpoints[4][0]['normalizers'], checkpoints[-1][0]['normalizers'])
        chunks = trial.records(output/'iteration-000006/chunks.jsonl')
        episodes = trial.records(output/'iteration-000006/episodes.jsonl')
        for row in episodes:
            selected = [c for c in chunks if c['alpha'] == row['alpha']]
            self.assertAlmostEqual(sum(c['tat_increment_hours'] for c in selected), row['tat_hours'])
            self.assertAlmostEqual(sum(c['reward'] for c in selected), -row['weighted_cost'])
            for chunk in selected:
                self.assertAlmostEqual(chunk['tat_increment_hours'], chunk['queue_wait_hours']+chunk['observed_allocation_hours'])
                self.assertNotIn('budget_hours', chunk)
        summary = load_json(output/'weighted84-summary.json')
        self.assertEqual(len(summary['fixed']), 15); self.assertEqual(len(summary['checkpoints']), 10)
        self.assertTrue((output/'weighted-curves.pdf').read_bytes().startswith(b'%PDF'))
        shutil.copyfile(output/'weighted-curves.png', '/tmp/carbon-weighted84-synthetic.png')
        # Mid-observation crash: only the first two completed rounds count.
        resumed = self.root/'resumed'
        collect = trial.collect
        def interrupt_warmup(*args):
            if args[9]['iteration'] == 3:
                raise KeyboardInterrupt('synthetic interruption')
            return collect(*args)
        with patch.object(trial, 'collect', side_effect=interrupt_warmup):
            with self.assertRaises(KeyboardInterrupt):
                trial.run(bundle, resumed, source, settings=self.settings)
        self.assertEqual(len(list(resumed.glob('iteration-*'))), 2)
        # A second interruption during updates exercises restored Adam state too.
        optimize = trial.optimize
        calls = 0
        def interrupt_update(*args):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise KeyboardInterrupt('synthetic update interruption')
            return optimize(*args)
        with patch.object(trial, 'optimize', side_effect=interrupt_update):
            with self.assertRaises(KeyboardInterrupt):
                trial.run(bundle, resumed, source, resume=True, settings=self.settings)
        self.assertEqual(load_json(resumed/'manifest.json')['completed_iteration'], 6)
        # Interrupted validation must reuse completed training and validation.
        validate = trial.validate
        def interrupt_validation(*args):
            if args[2] == 7:
                raise KeyboardInterrupt('synthetic validation interruption')
            return validate(*args)
        with patch.object(trial, 'validate', side_effect=interrupt_validation):
            with self.assertRaises(KeyboardInterrupt):
                trial.run(bundle, resumed, source, resume=True, settings=self.settings)
        self.assertTrue((resumed/'validation-000006').exists())
        with patch.object(trial, 'optimize', side_effect=AssertionError('no retraining')):
            done = trial.run(bundle, resumed, source, resume=True, settings=self.settings)
        self.assertEqual(done['status'], 'complete')
        self.assertEqual(refs, load_json(resumed/'reward-normalizers.json'))
        a, b = trial.read_checkpoint(output/'iteration-000007'), trial.read_checkpoint(resumed/'iteration-000007')
        self.assertEqual(a[0], b[0])
        self.assertTrue(all(torch.equal(v,b[1]['model'][k]) for k,v in a[1]['model'].items()))
        self.assertEqual(summary, load_json(resumed/'weighted84-summary.json'))
        self.assertEqual(original_source, {str(p.relative_to(source)): digest(p) for p in source.rglob('*') if p.is_file()})
        # Tampered calibration and a changed run plan both fail closed.
        path = resumed/'reward-normalizers.json'
        changed = {**refs, 'tat_hours': 1.}; path.write_text(json_text(changed))
        with self.assertRaisesRegex(ContractError, 'normalizers changed'):
            trial.run(bundle, resumed, source, resume=True, settings=self.settings)
        with self.assertRaisesRegex(ContractError, 'settings/software changed'):
            trial.run(bundle, output, source, resume=True, settings={**self.settings, 'iterations': 8})

    def test_foreign_output_and_tampered_source_rejected(self):
        bundle, source = self.fixture()
        output = self.root/'foreign'; output.mkdir(); (output/'run-plan.json').write_text('{"kind":"old"}')
        before = {p.name: digest(p) for p in output.iterdir()}
        with self.assertRaisesRegex(ContractError, 'Unknown existing'):
            trial.run(bundle, output, source, resume=True, settings=self.settings)
        self.assertEqual(before, {p.name: digest(p) for p in output.iterdir()})
        path = source/'fixed-validation/episodes.jsonl'; path.write_bytes(path.read_bytes()+b' ')
        with self.assertRaisesRegex(ContractError, 'Sealed artifact changed'):
            trial.run(bundle, self.root/'rejected', source, settings=self.settings)
        self.assertFalse((self.root/'rejected').exists())

    def test_launcher_allocation_guard_logging_and_resume(self):
        script = ROOT/'scripts/sophia_weighted84.sh'
        env = {k:v for k,v in os.environ.items() if not k.startswith(('PBS_', 'CARBON_'))}
        result = subprocess.run(['bash', str(script)], cwd=self.root, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        (self.root/'src/carbon').mkdir(parents=True)
        fake = self.root/'fake-python'
        fake.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@"\necho synthetic-stderr >&2\nexit 7\n'); fake.chmod(0o700)
        env.update(PBS_JOBID='synthetic', CARBON_PYTHON=str(fake), CARBON_RESUME='1')
        result = subprocess.run(['bash', str(script)], cwd=self.root, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 7); self.assertIn('--resume', result.stdout)
        log, = (self.root/'out').glob('weighted84.*.log')
        self.assertIn('synthetic-stderr', log.read_text()); self.assertIn('scripts/weighted84_trial.py', log.read_text())


if __name__ == '__main__':
    unittest.main()
