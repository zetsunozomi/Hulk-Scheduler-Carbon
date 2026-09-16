from datetime import timedelta
from decimal import Decimal
import unittest

from carbon.carbon import CarbonSeries, CIRecord, Exposure
from carbon.common import ContractError
from tests.helpers import T0, bundle


class CarbonTests(unittest.TestCase):
    def series(self, second=200):
        return CarbonSeries([
            CIRecord(T0, T0 + timedelta(hours=1), Decimal(400), T0 + timedelta(hours=1)),
            CIRecord(T0 + timedelta(hours=1), T0 + timedelta(hours=2), Decimal(second), T0 + timedelta(hours=3))],
            "synthetic", "synthetic")

    def test_exact_fractional_integration_and_additivity(self):
        c = self.series()
        begin, middle, end = T0 + timedelta(minutes=45), T0 + timedelta(hours=1), T0 + timedelta(hours=1, minutes=30)
        self.assertEqual(c.integral(begin, end), 200)
        self.assertEqual(c.integral(begin, end), c.integral(begin, middle) + c.integral(middle, end))

    def test_gap_and_overrun_fail(self):
        c = self.series()
        with self.assertRaises(ContractError):
            c.integral(T0, T0 + timedelta(hours=3))
        gap = CarbonSeries([c.records[0], CIRecord(T0 + timedelta(minutes=90), T0 + timedelta(hours=2), Decimal(100), T0 + timedelta(hours=2))], "synthetic", "synthetic")
        with self.assertRaisesRegex(ContractError, "gap"):
            gap.integral(T0, T0 + timedelta(hours=2))

    def test_future_revisions_do_not_change_available_history(self):
        a, b = self.series(200), self.series(999)
        self.assertEqual(a.observations_available_at(T0 + timedelta(minutes=90)),
                         b.observations_available_at(T0 + timedelta(minutes=90)))
        self.assertEqual(len(a.observations_available_at(T0 + timedelta(hours=2))), 1)

    def test_exposure_rescore_matches_direct_cost(self):
        b = bundle(); exposure = Exposure(b.workload.profiles)
        exposure.add(4, [("training", T0, T0 + timedelta(minutes=30))], self.series())
        exposure.add(16, [("training", T0 + timedelta(hours=1), T0 + timedelta(hours=2))], self.series())
        summary = exposure.summary(b.workload, b.raw["power"])
        for rho in b.raw["power"]["rho_interval"]:
            r = Decimal(str(rho))
            expected = sum(v * (r + (1-r) * b.workload.eta(n)) for n, v in exposure.L.items()) * Decimal("1.5")
            self.assertAlmostEqual(summary["carbon_g_per_kappa"][str(rho)], float(expected))
        self.assertIsNone(summary["carbon_g"])

    def test_explicit_calendar_offset(self):
        original = self.series()
        shifted = CarbonSeries(original.records, "synthetic", "synthetic", offset_seconds=3600)
        self.assertEqual(shifted.integral(T0, T0 + timedelta(hours=1)), 200)
