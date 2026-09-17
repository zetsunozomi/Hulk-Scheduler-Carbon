from datetime import timedelta
from decimal import Decimal
import unittest

from carbon.carbon import CarbonSeries, CIRecord
from carbon.common import ContractError
from carbon.forecast import CausalForecast, FrozenForecast
from tests.helpers import T0


def series(future=999, late=777):
    return CarbonSeries([
        CIRecord(T0-timedelta(days=7), T0-timedelta(days=7)+timedelta(hours=1), Decimal(100), T0-timedelta(days=7)+timedelta(hours=2)),
        CIRecord(T0-timedelta(hours=2), T0-timedelta(hours=1), Decimal(200), T0),
        CIRecord(T0-timedelta(hours=1), T0, Decimal(late), T0+timedelta(hours=24)),
        CIRecord(T0, T0+timedelta(hours=1), Decimal(future), T0+timedelta(hours=2))], 'synthetic', 'test')


class ForecastTests(unittest.TestCase):
    def test_unreleased_and_future_values_do_not_change_forecast(self):
        a = CausalForecast(series(), T0-timedelta(days=1), T0).issue(T0)
        b = CausalForecast(series(88888, 99999), T0-timedelta(days=1), T0).issue(T0)
        self.assertEqual(a, b)
        self.assertEqual(a.hourly_values[0], 100)  # same hour of prior week
        self.assertEqual(a.hourly_values[2], 200)  # available train fallback

    def test_forecast_is_frozen_even_for_far_future_integrals(self):
        model = CausalForecast(series(), T0-timedelta(days=1), T0)
        initial = model.issue(T0)
        before = initial.integral(T0, T0+timedelta(days=14))
        later = model.issue(T0+timedelta(days=1))
        self.assertNotEqual(initial.hourly_values, later.hourly_values)
        self.assertEqual(initial.integral(T0, T0+timedelta(days=14)), before)

    def test_training_climatology_is_asof_train_end_not_later_releases(self):
        model = CausalForecast(series(), T0-timedelta(days=1), T0, history_weeks=.01)
        late = model.issue(T0+timedelta(days=10))
        self.assertTrue(all(v == 200 for v in late.hourly_values))
        early = model.issue(T0)
        self.assertEqual(early.hourly_values[12], 200)

    def test_no_past_history_fails_instead_of_future_fallback(self):
        c = CarbonSeries([CIRecord(T0,T0+timedelta(hours=1),Decimal(10),T0+timedelta(hours=2))], 'test', 'test')
        with self.assertRaisesRegex(ContractError, 'No published'):
            CausalForecast(c,T0,T0+timedelta(hours=3)).issue(T0)

    def test_fractional_windows_use_exact_hour_boundaries(self):
        values = tuple(range(168))
        forecast = FrozenForecast(T0,values,'UTC',timedelta(0),'test','window',1,T0)
        a,b,c = T0+timedelta(minutes=30),T0+timedelta(hours=1),T0+timedelta(hours=2,minutes=15)
        self.assertEqual(forecast.integral(a,c), 1.5)
        self.assertEqual(forecast.integral(a,c), forecast.integral(a,b)+forecast.integral(b,c))
        with self.assertRaises(ContractError): forecast.integral(T0-timedelta(seconds=1),T0)

    def test_current_ci_ablation_uses_latest_released_value(self):
        forecast = CausalForecast(series(), T0-timedelta(days=1),T0).issue(T0,mode='current')
        self.assertEqual(set(forecast.hourly_values), {200.0})
