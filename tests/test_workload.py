from copy import deepcopy
from decimal import Decimal
import unittest

from carbon.common import ContractError, seconds
from carbon.workload import Workload
from tests.helpers import bundle


class WorkTests(unittest.TestCase):
    def test_fixed_work_and_final_partial_chunk_all_scales(self):
        w = bundle().workload
        for n in w.profiles:
            remaining, total, chunks = w.total_updates, 0, 0
            while remaining:
                p = w.plan(n, remaining, chunks == 0)
                self.assertGreater(p.updates, 0)
                self.assertLessEqual(p.actual, p.requested)
                self.assertLessEqual(seconds(p.requested), w.max_request)
                remaining -= p.updates
                total += p.updates
                chunks += 1
            self.assertEqual(total, w.total_updates)
            self.assertEqual(remaining, 0)

    def test_partial_request_uses_actual_work_and_preserves_save(self):
        w = bundle().workload
        p = w.plan(4, 1, False)
        self.assertEqual(p.updates, 1)
        self.assertEqual(seconds(p.training), Decimal(10))
        self.assertEqual(seconds(p.actual), Decimal(30))
        self.assertEqual(seconds(p.requested), Decimal(60))
        self.assertEqual(seconds(p.checkpoint), Decimal(15))

    def test_scale_switch_conserves_updates_and_batch(self):
        w = bundle().workload
        remaining = w.total_updates
        for i, n in enumerate([4, 8, 16, 32]):
            if not remaining:
                break
            p = w.plan(n, remaining, i == 0)
            remaining -= p.updates
        self.assertEqual(remaining, 0)
        for n, p in w.profiles.items():
            self.assertEqual(n * w.gpus_per_node * p.microbatch * p.accumulation, w.global_batch)

    def test_invalid_batch_and_zero_progress(self):
        b = bundle()
        c = deepcopy(b.raw["workload"])
        c["global_batch"] += 1
        with self.assertRaisesRegex(ContractError, "global batch"):
            Workload(c, [4, 8, 16, 32], 300, 60)
        c = deepcopy(b.raw["workload"])
        for p in c["profiles"].values():
            p["initialization_seconds"] = 300
        with self.assertRaisesRegex(ContractError, "No feasible initial"):
            Workload(c, [4, 8, 16, 32], 300, 60)

    def test_initialization_and_restart_are_distinct(self):
        w = bundle().workload
        self.assertEqual(seconds(w.plan(4, 100, True).setup), 15)
        self.assertEqual(seconds(w.plan(4, 100, False).setup), 5)
        self.assertEqual(w.eta(4), 1)
