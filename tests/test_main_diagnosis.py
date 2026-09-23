"""Small contract checks: no AMSP replay, training or production checkpoint load."""

from contextlib import redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from carbon.common import ContractError, load_json
from carbon.config import Episode
from carbon.runner import write_manifest
from tests.helpers import ROOT

with patch.object(sys, 'path', [str(ROOT/'scripts'), *sys.path]):
    spec = importlib.util.spec_from_file_location('diagnose_main_script', ROOT/'scripts/diagnose_main.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)


class DiagnosisTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        output = redirect_stdout(io.StringIO()); output.__enter__()
        self.addCleanup(output.__exit__, None, None, None)

    def test_sealed_stage_skips_work_and_rejects_modified_result(self):
        stage = self.root/'stage'; calls = []
        expected = {'kind': 'test-stage', 'budget': 3}
        def execute(path):
            calls.append(path); path.mkdir()
            write_manifest(path/'manifest.json', {**expected, 'status': 'complete'})
            (path/'rows.jsonl').write_text('{"value": 2}\n')
        def validate(path): return runner.records(path/'rows.jsonl')
        args = (stage, expected, ('manifest.json', 'rows.jsonl'), execute, validate)
        first = runner.checked_stage(*args)
        self.assertEqual(runner.checked_stage(*args), first)
        self.assertEqual(len(calls), 1)
        (stage/'rows.jsonl').write_text('{"value": 3}\n')
        with self.assertRaisesRegex(ContractError, 'seal differs'): runner.checked_stage(*args)
        self.assertEqual(len(calls), 1)

    def test_interrupted_stage_is_preserved_but_changed_contract_is_rejected(self):
        stage = self.root/'stage'; stage.mkdir()
        expected = {'kind': 'test-stage', 'budget': 3}
        write_manifest(stage/'manifest.json', {**expected, 'status': 'running'})
        (stage/'partial').write_text('old partial work')
        def execute(path):
            path.mkdir(); write_manifest(path/'manifest.json', {**expected, 'status': 'complete'})
        runner.checked_stage(stage, expected, ('manifest.json',), execute, lambda p: None)
        backups = list(self.root.glob('stage.interrupted-*/partial'))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_text(), 'old partial work')
        with self.assertRaisesRegex(ContractError, 'contract differs'):
            runner.checked_stage(stage, {**expected, 'budget': 4}, ('manifest.json',), execute, lambda p: None)

    def test_complete_unsealed_stage_is_checked_before_sealing(self):
        stage = self.root/'stage'; stage.mkdir()
        write_manifest(stage/'manifest.json', {'kind': 'test', 'status': 'complete'})
        (stage/'rows.jsonl').write_text('{"value":')
        with self.assertRaises(json.JSONDecodeError):
            runner.checked_stage(stage, {'kind': 'test'}, ('manifest.json', 'rows.jsonl'),
                                 lambda p: self.fail('must not rerun completed data'),
                                 lambda p: runner.records(p/'rows.jsonl'))
        self.assertFalse((stage/'stage-seal.json').exists())

    def test_budget_response_pairs_arrivals_instead_of_marginal_means(self):
        rows = []
        for arrival, probabilities in [('a', ([1., 0.], [0., 1.])), ('b', ([0., 1.], [1., 0.]))]:
            for beta, p in zip((1., 2.), probabilities):
                rows.append({'iteration': 16, 'episode_id': arrival, 'budget_multiplier': beta,
                             'probabilities': p, 'entropy_nats': 0.})
        summary = runner.response_summary(rows)
        # Marginal action frequencies are identical but each arrival responds fully.
        self.assertEqual(summary['operating_points'][0]['mean_probabilities'], [.5, .5])
        self.assertEqual(summary['budget_response'][0]['mean_max_pairwise_budget_TV'], 1.)
        with self.assertRaisesRegex(ContractError, 'Unpaired'): runner.response_summary(rows[:-1])
        with self.assertRaisesRegex(ContractError, 'Duplicate'): runner.response_summary(rows+[rows[0]])

    def test_censoring_and_repeated_scale_are_preserved_in_comparison(self):
        refs = {'carbon_reference_g_per_kappa': {'0.25': 10., '1.0': 20.}}
        done = {'final_status': 'completed', 'censor_flag': False, 'tat_hours': 4.,
                'chunk_count': 2, 'nodehours': 8., 'carbon_g_per_kappa': {'0.25': 10., '1.0': 20.}}
        partial = {'final_status': 'censored', 'censor_flag': True, 'tat_hours': None,
                   'observed_elapsed_hours': 6., 'chunk_count': 1, 'nodehours': None, 'carbon_g_per_kappa': None}
        summary = runner.comparison('PPO-64', [done, partial],
                                    [[{'selected_nodes': 16}]*2, [{'selected_nodes': 64}]], 5., refs)['summary']
        self.assertEqual(summary['outcomes'], 2)
        self.assertEqual(summary['miss_rate_bounds'], [.5, .5])
        self.assertEqual(summary['observed_scale_change_fraction'], 0.)
        self.assertIsNone(summary['worst_normalized_carbon'])
        self.assertIsNone(summary['p95_tat_hours'])
        self.assertEqual(summary['sequence_counts'], {'16->16': 1, '64': 1})

    def test_planner_group_rejects_wrong_budget_missing_pairs_and_test(self):
        stage = self.root/'planner'; stage.mkdir()
        rows = [{'episode_id': 'a', 'method': m, 'split': 'validation', 'budget_hours': 5.,
                 'chunk_count': 1, 'censor_flag': False, 'remaining_updates': 0} for m in runner.METHODS]
        chunks = [{**r, 'chunk_id': 0} for r in rows]
        def write(name, values):
            (stage/name).write_text(''.join(json.dumps(r)+'\n' for r in values))
        write('episodes.jsonl', rows); write('chunks.jsonl', chunks)
        keys = {('a', m) for m in runner.METHODS}
        self.assertEqual(len(runner.episode_groups(stage, keys, 'validation', 5.)[0]), 2)
        with self.assertRaisesRegex(ContractError, 'budget'): runner.episode_groups(stage, keys, 'validation', 6.)
        write('episodes.jsonl', rows[:1])
        with self.assertRaisesRegex(ContractError, 'Unpaired'): runner.episode_groups(stage, keys, 'validation', 5.)
        write('episodes.jsonl', [{**r, 'split': 'test'} for r in rows])
        with self.assertRaisesRegex(ContractError, 'split'): runner.episode_groups(stage, keys, 'validation', 5.)

    def test_planner_stage_selects_one_validation_arrival_without_changing_source_cohort(self):
        episode = SimpleNamespace(episode_id='a', split='validation')
        other = SimpleNamespace(episode_id='test-a', split='test')
        bundle = SimpleNamespace(manifest={'panel': 'test'}, episodes=[episode, other])
        software = {'source_sha256': {'core': 'unchanged'}}
        predictor = self.root/'predictor.json'; predictor.write_text('{}')
        references = self.root/'references.json'; references.write_text('{}')
        tick = {'budget_multiplier': 2., 'budget_hours': 5.}
        observed = []
        def fake_run(subset, path, predictor_path, reference_path, **kwargs):
            observed.append((subset.episodes, kwargs))
            subset._background_prefix = 'background-only-cache'
            path.mkdir()
            write_manifest(path/'manifest.json', {**bundle.manifest, 'methods': list(runner.METHODS),
                'seed': 11, 'planning_paths': 256, 'planner_internal_miss_tolerance': .05,
                'budget_multiplier': 2., 'selected_episode_ids': ['a'], 'selected_splits': ['validation'],
                'predictor_sha256': runner.digest(predictor_path), 'references_sha256': runner.digest(reference_path),
                'software': software, 'completed_episode_methods': 2, 'status': 'complete'})
            rows = [{'episode_id': 'a', 'method': m, 'split': 'validation', 'budget_hours': 5.,
                     'chunk_count': 1, 'censor_flag': False, 'remaining_updates': 0} for m in runner.METHODS]
            (path/'episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            (path/'chunks.jsonl').write_text(''.join(json.dumps({**r, 'chunk_id': 0})+'\n' for r in rows))
        with patch.object(runner, 'run_planners', side_effect=fake_run), patch.object(runner, 'ResultRun'):
            args = (bundle, self.root/'planner', episode, tick, predictor, references, software)
            runner.planner_stage(*args); runner.planner_stage(*args)
        self.assertEqual(len(observed), 1)
        self.assertEqual(observed[0][0], [episode])
        self.assertEqual(observed[0][1], {'split': 'validation', 'paths': 256, 'miss_tolerance': .05,
                                         'seed': 11, 'methods': runner.METHODS, 'budget_multiplier': 2.})
        self.assertEqual(bundle.episodes, [episode, other])
        self.assertEqual(bundle._background_prefix, 'background-only-cache')

    def test_research_config_rejected_before_loading_artifacts(self):
        with self.assertRaisesRegex(ContractError, 'development only'):
            runner.run(SimpleNamespace(raw={'purpose': 'research'}), None, None, None, None)

    def test_actor_reconstructs_frozen_probabilities_and_rejects_mismatch(self):
        runner.torch.set_num_threads(1)
        episode = Episode('a', 0, 'validation', 10.)
        bundle = SimpleNamespace(episodes=[episode])
        grid = {'budget_multipliers': [1., 2.], 'slider': {'ticks': [
            {'budget_multiplier': b, 'budget_hours': b} for b in (1., 2.)]}}
        def encode(observation):
            budget = observation['budget']
            return {'global': [1., budget, budget], 'actions': [[0.], [1.]], 'mask': [True, True]}, {'policy_input_sha256': str(budget)}
        encoder = SimpleNamespace(nodes=[4, 16], encode=encode)
        def model(global_state, actions, mask):
            logits = runner.torch.stack((global_state[:, 1], -global_state[:, 1]), dim=-1)
            return runner.torch.distributions.Categorical(logits=logits), None
        groups = {}
        for beta in (1., 2.):
            encoded, info = encode({'budget': beta})
            distribution, _ = model(*runner.input_tensors([encoded]))
            groups[('a', beta)] = [{'policy_input_sha256': info['policy_input_sha256'],
                                   'policy_probabilities': distribution.probs[0].tolist()}]
        candidates = [{'iteration': 16, 'groups': groups, 'model': model}]
        def environment(bundle, episode, base, method):
            return SimpleNamespace(observe=lambda: {'budget': episode.budget_hours})
        with patch.object(runner, 'initial_replay', return_value='initial-state') as replay, \
             patch.object(runner, 'Environment', side_effect=environment):
            before = runner.actor_stage(bundle, self.root/'good', encoder, candidates, grid)
            self.assertEqual(runner.actor_stage(bundle, self.root/'good', encoder, candidates, grid), before)
            self.assertEqual(replay.call_count, 1)
            self.assertGreater(before[0]['budget_response'][0]['mean_max_pairwise_budget_TV'], 0.)
            self.assertEqual(load_json(self.root/'good/actor/manifest.json')['rows'], 2)
            groups[('a', 1.)][0]['policy_probabilities'] = [.5, .5]
            with self.assertRaisesRegex(ContractError, 'probabilities changed'):
                runner.actor_stage(bundle, self.root/'bad', encoder, candidates, grid)
            self.assertFalse((self.root/'bad/actor/stage-seal.json').exists())
            self.assertEqual(load_json(self.root/'bad/actor/manifest.json')['status'], 'running')

    def test_launcher_guard_and_interactive_argument_forwarding(self):
        script = ROOT/'scripts/sophia_diagnose_main.sh'
        env = {k: v for k, v in os.environ.items() if not k.startswith(('PBS_', 'CARBON_'))}
        blocked = subprocess.run(['bash', str(script)], cwd=self.root, env=env, capture_output=True, text=True)
        self.assertEqual(blocked.returncode, 2)
        self.assertIn('active PBS interactive allocation', blocked.stderr)
        help_result = subprocess.run(['bash', str(script), '--help'], cwd=self.root, env=env, capture_output=True, text=True)
        self.assertEqual(help_result.returncode, 0)
        capture = self.root/'args'; fake_python = self.root/'fake-python'
        fake_python.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$CARBON_CAPTURE"\n')
        fake_python.chmod(0o700)
        (self.root/'src/carbon').mkdir(parents=True)
        env.update(PBS_JOBID='unit-test-not-a-real-job', PBS_O_WORKDIR=str(self.root),
                   CARBON_PYTHON=str(fake_python), CARBON_CAPTURE=str(capture), CARBON_RESUME='1')
        result = subprocess.run(['bash', str(script)], cwd=self.root, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        args = capture.read_text().splitlines()
        self.assertEqual(args[:3], ['-u', '-B', 'scripts/diagnose_main.py'])
        self.assertEqual(args[-1], '--resume')
        self.assertIn('results/amsp-diagnose-frontera-7b-seed11', args)


if __name__ == '__main__':
    unittest.main()
