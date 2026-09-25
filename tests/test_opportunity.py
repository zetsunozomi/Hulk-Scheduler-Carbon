"""Bounded synthetic checks only: no historical replay, training or model files."""
from copy import deepcopy
from datetime import timedelta
from decimal import Decimal
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(os.environ.get('CARBON_TEST_REPO', Path(__file__).resolve().parents[1]))
STAGED = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(STAGED / 'scripts'), str(ROOT / 'scripts'), str(ROOT / 'src')]

from carbon.common import ContractError, iso, timestamp
from carbon.replay import Replay
from carbon.trace import TraceJob
from carbon.workload import Workload
from fixed_max_walltime import FixedMaxWalltimeWorkload
from opportunity_policy import OpportunityPolicy, public_pressure
from opportunity_report import frontier_time, lower_frontier, summarize, report_scenario
from opportunity_traces import generate_cohort, generate_trace
from opportunity_trial import canonical_hash, make_bundle, run_method, run_seed, save_json, validate_saved


class ConstantCI:
    unit = 'gCO2/kWh'

    def integral(self, start, end):
        return Decimal(400) * Decimal(str((end - start).total_seconds())) / 3600


class OpportunityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.design = json.loads((STAGED / 'configs/opportunity-v1.json').read_text())
        cls.raw = json.loads((ROOT / cls.design['profile_config']).read_text())
        c = cls.raw['cluster']
        cls.work = FixedMaxWalltimeWorkload(Workload(cls.raw['workload'], c['allowed_nodes'], c['max_request_seconds'], c['walltime_resolution_seconds']))

    def test_calendar_respects_future_reservations_not_just_idle_nodes(self):
        state = {'capacity_nodes': 84, 'available_nodes': 16,
                 'running': [{'nodes': 68, 'requested_remaining_seconds': 7200}],
                 'pending': [{'nodes': 84, 'elapsed_wait_seconds': 3600, 'requested_seconds': 86400}]}
        self.assertEqual(public_pressure(state, [16], 1, 48)[16], 0)
        self.assertEqual(public_pressure(state, [16], 48, 48)[16], 26)

    def test_policy_decides_with_remaining_work_and_pressure(self):
        policy = OpportunityPolicy(self.work)
        state = {'capacity_nodes': 84, 'available_nodes': 84, 'running': [], 'pending': []}
        self.assertEqual(policy.select(state, 100000, True, 1)[0], 32)
        self.assertEqual(policy.select(state, 100000, True, 0)[0], 4)
        state['pending'] = [{'nodes': 64, 'elapsed_wait_seconds': 7200, 'requested_seconds': 30*3600}]
        self.assertEqual(policy.select(state, 30973, False, 1)[0], 16)
        state['pending'][0]['requested_seconds'] = 10*3600
        self.assertEqual(policy.select(state, 30973, False, 1)[0], 32)

    def test_hidden_future_fields_cannot_change_action(self):
        policy = OpportunityPolicy(self.work)
        state = {'capacity_nodes': 84, 'available_nodes': 20,
                 'running': [{'nodes': 64, 'requested_remaining_seconds': 30*3600}], 'pending': []}
        original = policy.select(state, 30973, False, .8)[0]
        poisoned = deepcopy(state)
        poisoned['future'] = [{'nodes': 84, 'actual': 1}]
        poisoned['running'][0].update(actual_seconds=1, end='tomorrow', priority=10**12)
        self.assertEqual(policy.select(poisoned, 30973, False, .8)[0], original)

    def test_pressure_is_capped(self):
        state = {'capacity_nodes': 84, 'available_nodes': 0,
                 'running': [{'nodes': 84, 'requested_remaining_seconds': 1000*3600}], 'pending': []}
        self.assertEqual(public_pressure(state, [4, 32], 48, 48), {4: 48, 32: 48})

    def test_trace_is_deterministic_and_independent_of_target_design(self):
        a, rows = generate_trace(self.design, 'wide-long', 11)
        changed = deepcopy(self.design)
        changed.update(alphas=[0, 1], episodes_per_seed=1, arrival_start_hours=300)
        b, other = generate_trace(changed, 'wide-long', 11)
        self.assertEqual(rows, other)
        self.assertEqual(a, b)
        self.assertNotEqual(rows, generate_trace(self.design, 'wide-long', 23)[1])

    def test_short_control_has_matched_arrivals(self):
        a = generate_trace(self.design, 'wide-long', 11)[1]
        b = generate_trace(self.design, 'short-control', 11)[1]
        self.assertEqual([(r['nodes'], r['submit_seconds']) for r in a], [(r['nodes'], r['submit_seconds']) for r in b])
        self.assertLess(sum(r['actual_seconds'] for r in b), sum(r['actual_seconds'] for r in a))

    def test_all_scenarios_have_valid_service_demands_and_shared_cohort(self):
        for name in self.design['scenarios']:
            jobs, rows = generate_trace(self.design, name, 11)
            self.assertGreater(len(jobs), 0)
            self.assertTrue(all(0 < j.nodes <= 84 and timedelta(0) < j.runtime <= j.requested for j in jobs))
            self.assertEqual(len({j.job_id for j in jobs}), len(jobs))
        arrivals = generate_cohort(self.design, 11)
        self.assertEqual(len(arrivals), 12)
        self.assertEqual(arrivals, generate_cohort(self.design, 11))
        self.assertEqual(arrivals, sorted(arrivals, key=lambda r: r['arrival_utc']))

    def run_constructed(self, hours):
        origin = timestamp(self.design['origin_utc'])
        end = origin + timedelta(days=20)
        submit = origin + timedelta(hours=20)
        job = TraceJob('background:one', 64, submit, submit, submit+timedelta(hours=hours), timedelta(hours=hours))
        initial = Replay([job], origin, end, self.raw['cluster'], history_seconds=0, sample_seconds=3600, initialize_from_observed=False)
        initial.advance_to(origin, before_dispatch=True)
        bundle = make_bundle(self.raw, self.work, ConstantCI(), origin, end, 400)
        episode = SimpleNamespace(arrival=origin, episode_id='tiny-mechanism', split='validation', budget_hours=400)
        policy = OpportunityPolicy(self.work)
        fixed = run_method(bundle, initial, episode, 'Fixed-32', 32, None, policy, self.design)
        dynamic = run_method(bundle, initial, episode, 'Dynamic-1.00', None, 1., policy, self.design)
        self.assertEqual(initial.time, origin)
        self.assertFalse(initial.finished)
        return fixed, dynamic

    def test_constructed_positive_case_uses_same_engine_and_48h_requests(self):
        fixed, dynamic = self.run_constructed(30)
        f, d = fixed['summary'], dynamic['summary']
        self.assertEqual(d['selected_nodes'], [32, 16])
        self.assertLess(d['tat_hours'], f['tat_hours'])
        self.assertLess(d['nodehours'], f['nodehours'])
        for data in (fixed, dynamic):
            self.assertEqual(data['summary']['completed_updates'], 100000)
            self.assertAlmostEqual(data['summary']['constant_carbon_kg_per_kappa'], data['summary']['ercot_carbon_kg_per_kappa'])
            self.assertTrue(all(c['requested_walltime_hours'] == 48 for c in data['chunks']))

    def test_short_block_does_not_trigger_slow_continuation_at_alpha1(self):
        fixed, dynamic = self.run_constructed(10)
        self.assertEqual(dynamic['summary']['selected_nodes'], [32, 32])
        self.assertEqual(dynamic['summary']['tat_hours'], fixed['summary']['tat_hours'])

    def test_frontier_keeps_stronger_fixed_mixture_and_no_extrapolation(self):
        hull = lower_frontier([(1, 10), (2, 9), (3, 4), (4, 5), (3, 6)])
        self.assertEqual(hull, [(1, 10), (3, 4)])
        self.assertEqual(frontier_time(hull, 2), 7)
        self.assertIsNone(frontier_time(hull, .5))

    def test_resume_rejects_tampering_and_changed_input(self):
        episode = {'episode_id': 'a', 'arrival_utc': '2020-10-01T00:00:00Z'}
        payload = {'summary': {'method': 'Fixed-4', 'episode_id': 'a', 'initial_arrival_utc': episode['arrival_utc'], 'final_status': 'completed'},
                   'chunks': [{'requested_walltime_hours': 48}]}
        record = {'contract_sha256': 'one', 'episode': episode, 'methods': {'Fixed-4': {'payload': payload, 'sha256': canonical_hash(payload)}}}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'result.json'
            save_json(path, record)
            restored = json.loads(path.read_text())
            validate_saved(restored, 'one', episode, ['Fixed-4'])
            with self.assertRaises(ContractError):
                validate_saved(restored, 'two', episode, ['Fixed-4'])
            restored['methods']['Fixed-4']['payload']['summary']['method'] = 'Fixed-32'
            with self.assertRaises(ContractError):
                validate_saved(restored, 'one', episode, ['Fixed-4'])

    def test_incomplete_methods_do_not_get_completion_only_means(self):
        names = [f'Fixed-{n}' for n in (4, 8, 16, 32)] + ['Dynamic-0.00', 'Dynamic-1.00']
        records = []
        for i, method in enumerate(names):
            for j in range(2):
                timeout = method == 'Dynamic-1.00' and j == 1
                records.append({'summary': {'method': method, 'alpha': None if i < 4 else float(i-4),
                    'final_status': 'timeout' if timeout else 'completed', 'tat_hours': None if timeout else 100-i,
                    'nodehours': None if timeout else 100+i, 'constant_carbon_kg_per_kappa': None if timeout else 10+i,
                    'ercot_carbon_kg_per_kappa': None if timeout else 11+i, 'selected_nodes': [4, 8]}})
        rows, diagnostics = summarize(records, {'design': {'alphas': [0., 1.]}})
        self.assertIsNone(rows[-1]['mean_tat_hours'])
        self.assertFalse(diagnostics['all_methods_complete'])
        self.assertEqual(diagnostics['constant']['dynamic_points_below_fixed_hull'], 0)

    def test_tiny_seed_pipeline_resume_and_text_only_report(self):
        # A few hours of tiny synthetic service, not a formal experiment.
        design = deepcopy(self.design)
        design.update(seeds=[11], episodes_per_seed=2, alphas=[0., 1.], trace_hours=12,
                      arrival_start_hours=2, arrival_stop_hours=3, episode_timeout_hours=6)
        design['scenarios'] = {'wide-long': {'kind': 'renewal', 'streams': [
            {'nodes': [64], 'interarrival_hours': [1, 2], 'runtime_hours': [.1, .2], 'request_hours': .5}]}}
        raw = deepcopy(self.raw)
        raw['workload']['optimizer_updates'] = 100
        work = FixedMaxWalltimeWorkload(Workload(raw['workload'], [4, 8, 16, 32], 172800, 60))
        contract = {'kind': 'opportunity-v1', 'design': design}
        inputs = (design, raw, ConstantCI(), work, contract)
        with tempfile.TemporaryDirectory() as folder, patch('opportunity_trial.load_inputs', return_value=inputs), redirect_stdout(io.StringIO()):
            save_json(Path(folder)/'run-plan.json', contract)
            run_seed(ROOT, 'unused-unit-fixture', folder, 'wide-long', 11, False)
            with patch('opportunity_trial.run_method', side_effect=AssertionError('Resume reran a completed method')):
                run_seed(ROOT, 'unused-unit-fixture', folder, 'wide-long', 11, True)
            # Renderer is checked separately with this same bounded fixture.
            with patch('opportunity_report.plot_curves'):
                report = report_scenario(folder, 'wide-long')
            self.assertTrue(report['diagnostics']['all_methods_complete'])
            self.assertEqual(report['diagnostics']['arrival_count'], 2)
            self.assertEqual(len(report['rows']), 6)
            self.assertTrue((Path(folder)/'wide-long/curves.csv').is_file())


if __name__ == '__main__':
    unittest.main()
