"""Analytic input and reporting checks; no PPO or historical trace replay."""

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from carbon.common import ContractError
from carbon.workload import Workload

ROOT = Path(__file__).resolve().parents[1]
with patch.object(sys, 'path', [str(ROOT/'scripts'), *sys.path]):
    import scaling_profiles as profiles
    import scaling_report as report
    import old_gpt_profiles as legacy
    import frontera_old_gpt as entry


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.design = profiles.read_design()

    def test_generated_artifacts_match_declared_inputs(self):
        for path, expected in profiles.generated_files().items():
            self.assertEqual(path.read_text(), expected)

    def test_anchor_efficiency_and_fixed_work_for_every_scenario(self):
        for model, settings in self.design['models'].items():
          for scenario, spec in self.design['scenarios'].items():
            raw = profiles.config_for(self.design, model, scenario)
            work = Workload(raw['workload'], [4,8,16,32], 172800, 60)
            self.assertEqual(work.total_updates, 100000)
            self.assertEqual(work.global_batch, 512)
            self.assertEqual(work.gpus_per_node, 4)
            self.assertAlmostEqual(100000/float(work.profiles[16].updates_per_hour), settings['anchor_training_hours'])
            for n in (4,8,16,32):
                self.assertAlmostEqual(float(work.eta(n)), spec['efficiency_vs4'][str(n)])
                remaining, updates, training_seconds, chunks = 100000, 0, 0., 0
                while remaining:
                    plan = work.plan(n, remaining, chunks == 0)
                    self.assertLessEqual(plan.requested.total_seconds(), 172800)
                    self.assertGreater(plan.updates, 0)
                    training_seconds += plan.training.total_seconds()
                    remaining -= plan.updates; updates += plan.updates; chunks += 1
                self.assertEqual(updates, 100000)
                self.assertAlmostEqual(training_seconds/3600, 100000/float(work.profiles[n].updates_per_hour), places=8)
                max_updates = int(float(work.profiles[n].updates_per_hour)*(48-600/3600))
                self.assertEqual(chunks, math.ceil(100000/max_updates))

    def test_linear_limit_has_constant_pure_training_nodehours(self):
      for model, spec in self.design['models'].items():
        for p in profiles.scenario_points(self.design, model, 'e100'):
            self.assertAlmostEqual(p['nodes']*p['training_hours'], 16*spec['anchor_training_hours'])
            self.assertAlmostEqual(p['efficiency_vs4'], 1.)

    def test_regression_saturation_and_different_shapes_are_preserved(self):
        def hours(scenario):
            return [p['training_hours'] for p in profiles.scenario_points(self.design,'medium',scenario)]
        self.assertGreater(hours('e010')[-1], hours('e010')[0])
        self.assertTrue(all(abs(t-22.6)<1e-10 for t in hours('e0125')))
        self.assertLess(hours('e120')[-1], hours('e120')[0]/8)
        self.assertNotEqual(hours('early_knee'), hours('late_knee'))
        self.assertEqual(self.design['scenarios']['early_knee']['efficiency_vs4']['32'],
                         self.design['scenarios']['late_knee']['efficiency_vs4']['32'])
        self.assertEqual(set(self.design['models']), {'medium','xl'})
        self.assertEqual(len(self.design['scenarios']),14)

    def test_binding_rejects_changed_timing_or_power(self):
        raw = profiles.config_for(self.design, 'medium', 'e050')
        bundle = SimpleNamespace(root=ROOT, raw=raw)
        result = legacy.validate_bundle(bundle)
        self.assertEqual(result['profile_status'], 'assumed')
        self.assertAlmostEqual(result['efficiency32_vs4'], .5)
        for section, key, value in [('power', 'reference_kw', 2.), ('workload', 'profile_status', 'measured')]:
            changed = deepcopy(raw); changed[section][key] = value
            with self.assertRaises(ContractError):
                profiles.validate_bundle(SimpleNamespace(root=ROOT, raw=changed))
        changed = deepcopy(raw); changed['workload']['profiles']['32']['updates_per_hour'] *= 1.1
        with self.assertRaises(ContractError):
            profiles.validate_bundle(SimpleNamespace(root=ROOT, raw=changed))

    def test_existing_legacy_inputs_still_have_canonical_workloads(self):
        asset = json.loads((ROOT/legacy.ASSET).read_text())
        for model in ('medium', 'large', 'xl'):
            raw = json.loads((ROOT/f'configs/old-gpt-frontera-{model}-c84.development.json').read_text())
            self.assertEqual(raw['workload'], legacy.workload_config(asset, model))

    def test_login_guard_even_with_job_id(self):
        with patch.object(entry.socket, 'gethostname', return_value='login1.example'), \
             patch.dict(entry.os.environ, {'SLURM_JOB_ID': '123'}):
            with self.assertRaises(RuntimeError):
                entry.require_allocation()
            entry.require_allocation(check=True)


def row(nodes, arrival, **extra):
    return dict(episode_id=f'arrival-{arrival}', initial_arrival_utc=f'2024-01-0{arrival+1}T00:00:00Z',
                split='validation', final_status='completed', censor_flag=False,
                exposure_is_complete=True, completed_updates=100000, selected_nodes=[nodes],
                chunk_count=1, nodehours=float(nodes), tat_hours=100/nodes+arrival,
                carbon_g_per_kappa={'0.25': float(nodes*200), '1.0': float(nodes*300)},
                method=f'Fixed-{nodes}', **extra)


def seal_rows(directory, rows):
    directory.mkdir(parents=True, exist_ok=True)
    raw = ''.join(json.dumps(r)+'\n' for r in rows).encode()
    (directory/'episodes.jsonl').write_bytes(raw)
    (directory/'stage-seal.json').write_text(json.dumps({'episodes.jsonl': hashlib.sha256(raw).hexdigest()}))


class ReportTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root = Path(temp.name); self.source = self.root/'fixed'; self.output = self.root/'policy'
        self.output.mkdir()
        fixed = [row(n,i) for n in (4,8,16,32) for i in (0,1)]
        for r in fixed:
            del r['selected_nodes']  # Matches canonical fixed-run episode schema.
        seal_rows(self.source/'fixed-validation', fixed)
        (self.source/'references.json').write_text(json.dumps({'fixed_request_policy': {'requested_seconds':172800}}))
        binding = {'kind':'analytic_scaling_profile_v2','model':'medium','scenario':'e050',
                   'anchor_training_hours':22.6,'efficiency32_vs4':.5}
        (self.output/'run-plan.json').write_text(json.dumps({'source_binding':{'profile_inputs':binding}}))
        self.settings = {'alphas':[0.,1.], 'validation_iterations':[16,32], 'rho':'1.0', 'seed':11}
        for it in self.settings['validation_iterations']:
            rows = []
            for alpha, n in ((0.,4),(1.,32)):
                for i in (0,1):
                    r = row(n,i,alpha=alpha,iteration=it); r['method']='Weighted-PPO'; rows.append(r)
            seal_rows(self.output/f'validation-{it:06d}', rows)

    def test_report_keeps_full_curves_and_assumed_provenance(self):
        report.write_readout(self.output, self.source, self.settings, {})
        summary = json.loads((self.output/'scaling-summary.json').read_text())
        self.assertEqual(len(summary['fixed']), 8)
        self.assertEqual(len(summary['checkpoints']), 4)
        self.assertTrue(all(p['outcomes']==2 for p in summary['fixed']+summary['checkpoints']))
        self.assertTrue(all('best_fixed' not in p and 'improvement_percent' not in p for p in summary['checkpoints']))
        self.assertEqual(summary['checkpoints'][0]['mean_tat_hours'],25.5)
        self.assertEqual(summary['fixed'][0]['action_counts'], {'4':2})
        self.assertIn('assumed',summary['measurement_status'])

    def test_no_silent_dropping_of_unmatched_arrival(self):
        path = self.output/'validation-000032'
        rows = report.records(path); rows[0]['episode_id']='foreign-arrival'
        seal_rows(path,rows)
        with self.assertRaises(ContractError):
            report.write_readout(self.output,self.source,self.settings,{})

    def test_tampered_seal_and_duplicate_outcome_are_rejected(self):
        path = self.output/'validation-000016'
        rows = report.records(path)
        with (path/'episodes.jsonl').open('a') as stream:
            stream.write(json.dumps(rows[0])+'\n')
        with self.assertRaises(ContractError):
            report.records(path)
        seal_rows(path,rows+[rows[0]])
        with self.assertRaises(ContractError):
            report.write_readout(self.output,self.source,self.settings,{})


if __name__ == '__main__':
    unittest.main()
