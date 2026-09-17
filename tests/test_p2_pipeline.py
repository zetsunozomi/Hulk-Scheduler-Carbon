from contextlib import redirect_stdout
from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

from carbon.__main__ import main
from carbon.baselines import fit_fixed_mix, fixed_records, make_references, weighted_quantile
from carbon.common import ContractError, json_text, load_json
from carbon.config import Bundle
from carbon.diagnostics import atom_quantile, summarize_scores, wait_scores
from carbon.runner import run_fixed
from tests.helpers import ROOT


class DiagnosticTests(unittest.TestCase):
    def test_discrete_distribution_scores_and_censor_counts(self):
        scores = wait_scores([0,2],1)
        self.assertAlmostEqual(scores['crps_hours'], .5)
        self.assertEqual(scores['mean_error_hours'], 0)
        self.assertEqual(atom_quantile([0,2], .5), 0)
        self.assertEqual(atom_quantile([0,2], .9), 2)
        result = summarize_scores([{'censored':False,'scores':scores}, {'censored':True,'scores':None}])
        self.assertEqual(result['labeled_rows'], 1)
        self.assertEqual(result['censored_rows'], 1)
        self.assertEqual(summarize_scores([{'censored':True,'scores':None}])['metrics'], None)

    def test_mixture_quantile_uses_distribution_not_average_quantiles(self):
        self.assertEqual(weighted_quantile([(1,.94),(100,.06)],.95), 100)
        self.assertEqual(weighted_quantile([(1,.96),(100,.04)],.95), 1)

    def test_queue_config_accepts_unmeasured_qwen_only_for_probes(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_json(ROOT/'configs/synthetic-p2.json')
            config.update(root=str(ROOT), workload=None, ci=None, power=None, cohort=None)
            path = Path(tmp)/'queue.json'; path.write_text(json_text(config))
            queue = Bundle(path, queue_only=True)
            self.assertEqual(queue.manifest['validated_scope'], 'queue_only')
            self.assertEqual(queue.episodes, ())
            with self.assertRaises((ContractError,TypeError)):
                Bundle(path)


@unittest.skipUnless(importlib.util.find_spec('scipy'), 'requires P2 fitting dependencies')
class FixedMixTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)/'fixed'
        with redirect_stdout(io.StringIO()):
            run_fixed(Bundle(ROOT/'configs/synthetic-p2.json'), self.path, [4,8,16,32])
        self.references = make_references(self.path, Path(self.tmp.name)/'references.json')
        original = [json.loads(line) for line in (self.path/'episodes.jsonl').read_text().splitlines()]
        rows = [r for r in original if r['split'] != 'validation']
        for row in original:
            if row['split'] != 'validation':
                continue
            n = int(row['method'].split('-')[1])
            for index in range(2):
                value = deepcopy(row); value['episode_id'] += str(index)
                value['tat_hours'] = [.01,.3][index] if n == 4 else .1
                value['carbon_g_per_kappa'] = {rho: {4:2,8:6,16:8,32:10}[n] for rho in row['carbon_g_per_kappa']}
                rows.append(value)
        (self.path/'episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))

    def test_lp_constrains_miss_probability_not_mean_time(self):
        result = fit_fixed_mix(self.path, self.references, .2, .1)
        self.assertTrue(result['empirical_feasible'])
        self.assertAlmostEqual(result['weights']['4'], .2)
        self.assertAlmostEqual(result['weights']['8'], .8)
        self.assertEqual(result['validation_miss_rates']['4'], .5)

    def test_infeasible_and_changed_inputs_are_rejected(self):
        result = fit_fixed_mix(self.path, self.references, .001)
        self.assertFalse(result['empirical_feasible']); self.assertIsNone(result['weights'])
        changed = deepcopy(self.references)
        changed['resolved_config']['ci']['queue_to_ci_offset_seconds'] = 3600
        with self.assertRaisesRegex(ContractError, 'interpretation differs'):
            fit_fixed_mix(self.path, changed, .2)

    def test_incomplete_pairing_cannot_fit(self):
        path = self.path/'episodes.jsonl'
        rows = path.read_text().splitlines(); path.write_text('\n'.join(rows[:-1])+'\n')
        with self.assertRaisesRegex(ContractError, 'not paired'):
            fixed_records(path, 'validation', [4,8,16,32])


@unittest.skipUnless(importlib.util.find_spec('sklearn'), 'requires P2 fitting dependencies')
class PipelineTests(unittest.TestCase):
    def test_synthetic_fit_validate_and_complete_work_planners(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            config = str(ROOT/'configs/synthetic-p2.json')
            def command(*parts):
                self.assertEqual(main([str(x) for x in parts]), 0)
            command('probe-waits','--config',config,'--output',root/'train','--interval-seconds',300)
            command('probe-waits','--config',config,'--output',root/'validation','--split','validation','--interval-seconds',1200)
            command('fit-waits','--probes',root/'train','--output',root/'model','--trees',16)
            command('evaluate-waits','--probes',root/'validation','--predictor',root/'model/model.json','--output',root/'diagnostics')
            metrics = load_json(root/'diagnostics/metrics.json')
            self.assertEqual(metrics['overall']['rows'],72)
            self.assertEqual(set(metrics['groups']['nodes']), {'4','8','16','32'})
            self.assertEqual(metrics['overall']['censored_rows'],0)
            command('run-fixed','--config',config,'--output',root/'fixed')
            command('make-references','--fixed-run',root/'fixed','--output',root/'refs.json')
            command('run-planners','--config',config,'--output',root/'planning','--predictor',root/'model/model.json',
                    '--references',root/'refs.json','--paths',16)
            episodes = [json.loads(line) for line in (root/'planning/episodes.jsonl').read_text().splitlines()]
            self.assertEqual({r['method'] for r in episodes}, {'Rollout-MPC','Plan-once','Queue-blind-MPC'})
            self.assertTrue(all(r['completed_updates']==100 and r['final_status']=='completed' for r in episodes))
            chunks = [json.loads(line) for line in (root/'planning/chunks.jsonl').read_text().splitlines()]
            once = [r for r in chunks if r['method']=='Plan-once']
            self.assertTrue(once[0]['replanned'])
            self.assertTrue(all(not r['replanned'] for r in once[1:]))
            self.assertTrue(all(r['predictor_version'] and r['forecast_input_id'] for r in chunks))
