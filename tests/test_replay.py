from datetime import timedelta
import unittest

from carbon.common import ContractError
from carbon.replay import Replay, Request
from carbon.trace import TraceJob
from tests.helpers import T0, cluster

def request(name, nodes=4, submit=T0, actual=600, requested=None):
    return Request(name, nodes, submit, timedelta(seconds=requested or actual), timedelta(seconds=actual), True)


class ReplayTests(unittest.TestCase):
    def sim(self, jobs=(), capacity=4):
        return Replay(jobs, T0, T0 + timedelta(days=1), cluster(capacity), sample_seconds=60)

    def test_stop_exactly_at_completion_before_later_arrival(self):
        bg = TraceJob("bg", 1, T0 + timedelta(minutes=100), T0 + timedelta(minutes=100), T0 + timedelta(minutes=101), timedelta(minutes=1))
        s = self.sim([bg]); s.inject(request("target"))
        a = s.until("target", "complete", s.coverage_end)
        self.assertEqual(a.end, T0 + timedelta(minutes=10))
        self.assertEqual(s.time, a.end)
        self.assertEqual(s.index, 0)

    def test_start_and_completion_are_different_boundaries(self):
        s = self.sim(); s.inject(request("target", actual=15))
        self.assertEqual(s.until("target", "start", s.coverage_end).start, T0)
        self.assertEqual(s.time, T0)
        s.until("target", "complete", s.coverage_end)
        self.assertEqual(s.time, T0 + timedelta(seconds=15))
        self.assertEqual(s.visible()["available_nodes"], 4)

    def test_cross_boundary_running_and_pending(self):
        running = TraceJob("running", 4, T0 - timedelta(hours=2), T0 - timedelta(hours=1), T0 + timedelta(minutes=10), timedelta(hours=2))
        pending = TraceJob("pending", 4, T0 - timedelta(minutes=5), T0 + timedelta(minutes=20), T0 + timedelta(minutes=21), timedelta(minutes=1))
        s = self.sim([running, pending])
        state = s.visible()
        self.assertEqual(len(state["running"]), 1)
        self.assertEqual(state["pending"][0]["elapsed_wait_seconds"], 300)
        self.assertEqual(state["running"][0]["requested_remaining_seconds"], 3600)

    def test_snapshot_independence_and_target_queue_feedback(self):
        bg = TraceJob("bg", 4, T0 + timedelta(minutes=1), T0 + timedelta(minutes=1), T0 + timedelta(minutes=2), timedelta(minutes=1))
        base = self.sim([bg]); a, b = base.clone(), base.clone()
        a.inject(request("target", actual=600))
        a.advance_to(T0 + timedelta(minutes=2))
        b.advance_to(T0 + timedelta(minutes=2))
        self.assertIn("bg", a.pending)
        self.assertNotIn("bg", b.pending)
        self.assertEqual(base.time, T0)
        self.assertEqual(base.index, 0)
        self.assertEqual(len(base.history), 0)

    def test_observations_do_not_reveal_hidden_future_runtime(self):
        early = TraceJob("bg", 4, T0, T0, T0 + timedelta(minutes=5), timedelta(hours=1))
        later = TraceJob("bg", 4, T0, T0, T0 + timedelta(minutes=50), timedelta(hours=1))
        a, b = self.sim([early]), self.sim([later])
        for s in (a, b):
            s.advance_to(T0 + timedelta(minutes=1))
        self.assertEqual(a.visible(), b.visible())
        self.assertEqual(a.visible_history([0, 60, 7200]), b.visible_history([0, 60, 7200]))
        self.assertTrue(a.visible_history([7200])[0]["missing"])

    def test_backfill_uses_request_not_hidden_runtime(self):
        s = self.sim(capacity=8)
        s.inject(request("running", nodes=4, actual=600, requested=3600))
        s.until("running", "start", s.coverage_end)
        s.inject(request("head", nodes=8, actual=600, requested=600))
        s.inject(request("backfill", nodes=4, actual=60, requested=1800))
        s.advance_to(T0 + timedelta(minutes=1))
        self.assertIn("backfill", s.running)
        self.assertIn("head", s.pending)

    def test_equal_priority_order_deterministic_and_tick_wait_explicit(self):
        a, b = self.sim(), self.sim()
        for s, names in ((a, ["b", "a"]), (b, ["a", "b"])):
            for name in names:
                s.inject(request(name, actual=15))
            s.until("a", "complete", s.coverage_end)
        self.assertEqual(a.time, b.time)
        a.inject(request("next", submit=a.time, actual=10))
        a.until("next", "start", a.coverage_end)
        self.assertEqual(a.time, T0 + timedelta(minutes=2))

    def test_invalid_requests_and_backwards_time_rejected(self):
        s = self.sim()
        with self.assertRaises(ContractError):
            s.inject(request("oversized", nodes=8))
        s.advance_to(T0 + timedelta(minutes=1))
        with self.assertRaises(ContractError):
            s.advance_to(T0)

    def test_reservation_calendar_matches_exhaustive_integer_time_oracle(self):
        # Independent slow oracle, small artificial node/time domain only.
        import random
        rng = random.Random(19)
        for _ in range(20):
            starts, free, reservations = [T0], [8], []
            for j in range(12):
                nodes, length = rng.randint(1, 8), rng.randint(1, 10)
                expected = next(t for t in range(121) if all(
                    nodes + sum(n for s, e, n in reservations if s <= x < e) <= 8
                    for x in range(t, t+length)))
                req = request(str(j), nodes=nodes, actual=length)
                actual = Replay._reserve(starts, free, req)
                self.assertEqual(actual, T0 + timedelta(seconds=expected))
                reservations.append((expected, expected+length, nodes))
