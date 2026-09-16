from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
import unittest

from carbon.common import seconds
from carbon.environment import Environment, initial_replay
from tests.helpers import bundle


class EnvironmentTests(unittest.TestCase):
    def make(self, n=4, episode=None):
        b = bundle()
        e = episode or b.episodes[0]
        return Environment(b, e, initial_replay(b, e), f"Fixed-{n}")

    def test_complete_all_work_even_after_budget_miss(self):
        env = self.make()
        while env.status == "running":
            env.step(4)
        result = env.summary()
        self.assertEqual(result["completed_updates"], env.work.total_updates)
        self.assertTrue(result["deadline_miss"])
        self.assertGreater(result["tat_hours"], env.episode.budget_hours)
        self.assertEqual(result["remaining_updates"], 0)
        for prev, nxt in zip(env.chunks, env.chunks[1:]):
            self.assertEqual(prev["end_utc"], nxt["submit_utc"])
        self.assertEqual(env.chunks[-1]["global_next_sample_index"], env.work.total_updates * env.work.global_batch)

    def test_request_and_phase_logs_sum_to_accounting(self):
        env = self.make()
        while env.status == "running":
            env.step(4)
        summary = env.summary()
        for c in env.chunks:
            self.assertLessEqual(c["observed_allocation_hours"], c["requested_walltime_hours"])
            phase_total = sum(p["exposure_node_gco2e_per_kwh_hours"] for p in c["phases"])
            self.assertAlmostEqual(phase_total, sum(c["exposure_by_nodes"].values()))
            self.assertEqual(c["phases"][-1]["phase"], "checkpoint")
        self.assertAlmostEqual(summary["A"], sum(c["A"] for c in env.chunks))
        self.assertAlmostEqual(summary["nodehours"], sum(c["observed_allocation_hours"] * c["selected_nodes"] for c in env.chunks))

    def test_censor_during_queue_has_unknown_total_not_zero(self):
        env = self.make(32)
        env.boundary = env.episode.arrival + timedelta(seconds=30)
        env.step(32)
        s = env.summary()
        self.assertEqual(s["final_status"], "censored")
        self.assertIsNone(s["carbon_g_per_kappa"])
        self.assertIsNone(s["tat_hours"])
        self.assertIsNone(s["deadline_miss"])
        self.assertEqual(s["observed_nodehours"], 0)
        self.assertIsNone(env.chunks[0]["start_utc"])

    def test_partial_training_exposure_and_uncommitted_updates(self):
        env = self.make()
        env.boundary = env.episode.arrival + timedelta(seconds=100)
        log = env.step(4)
        self.assertEqual(log["completed_updates"], 0)
        self.assertGreater(log["processed_updates_uncommitted"], 0)
        self.assertGreater(log["A"], 0)
        self.assertEqual(env.remaining, env.work.total_updates)
        self.assertIsNone(log["end_utc"])
        self.assertEqual(log["phases"][-1]["phase"], "training")

    def test_checkpoint_must_finish_to_commit_updates(self):
        env = self.make()
        p = env.work.plan(4, env.remaining, True)
        env.boundary = env.episode.arrival + p.setup + p.training + timedelta(seconds=1)
        log = env.step(4)
        self.assertEqual(log["completed_updates"], 0)
        self.assertEqual(log["processed_updates_uncommitted"], p.updates)
        self.assertEqual(log["phases"][-1]["phase"], "checkpoint")

    def test_pairing_and_action_switch_conserve_state(self):
        b = bundle(); e = b.episodes[0]; base = initial_replay(b, e)
        a = Environment(b, e, base, "switching")
        other = Environment(b, e, base, "switching")
        for n in [4, 8, 16, 32]:
            if a.status != "running":
                break
            self.assertEqual(a.observe(), other.observe())
            self.assertEqual(a.step(n), other.step(n))
        self.assertEqual(a.status, "completed")
        self.assertEqual(a.summary(), other.summary())
        self.assertEqual(base.time, e.arrival)

    def test_terminal_exact_boundary_and_timeout(self):
        b = bundle(); e = b.episodes[1]
        env = self.make(32, e)
        p = env.work.plan(32, env.remaining, True)
        env.boundary = e.arrival + p.actual
        env.step(32)
        self.assertEqual(env.status, "completed")
        env = self.make(32, e)
        env.boundary = e.arrival + timedelta(seconds=10)
        env.boundary_reason = "episode_timeout"
        env.step(32)
        self.assertEqual(env.status, "timeout")
