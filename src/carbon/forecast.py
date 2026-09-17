"""A frozen, decision-time CI forecast. Planning never queries future observations."""

from bisect import bisect_right
from dataclasses import dataclass
from datetime import timedelta
import hashlib
import json
import math
from zoneinfo import ZoneInfo

from .common import iso, require


def hour_of_week(at, zone):
    local = at.astimezone(zone)
    return local.weekday() * 24 + local.hour


@dataclass(frozen=True)
class FrozenForecast:
    issued_at: object
    hourly_values: tuple
    calendar_timezone: str
    offset: timedelta
    input_id: str
    mode: str
    available_records: int
    last_observation_end: object

    def integral(self, start, end):
        require(self.issued_at <= start <= end, "Forecast integral precedes its issue time")
        zone = ZoneInfo(self.calendar_timezone)
        cursor, end = start + self.offset, end + self.offset
        value = 0.0
        while cursor < end:
            boundary = min(end, cursor.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1))
            value += self.hourly_values[hour_of_week(cursor, zone)] * (boundary - cursor).total_seconds() / 3600
            cursor = boundary
        return value

    def metadata(self):
        return {"forecast_issue_utc": iso(self.issued_at), "forecast_input_id": self.input_id,
                "forecast_mode": self.mode, "forecast_available_records": self.available_records,
                "forecast_latest_observation_end_utc": iso(self.last_observation_end)}


class CausalForecast:
    def __init__(self, ci, train_start, train_end, history_weeks=8, calendar_timezone="UTC"):
        require(train_start < train_end and history_weeks > 0, "Invalid forecast training interval/history")
        self.ci, self.train_start, self.train_end = ci, train_start + ci.offset, train_end + ci.offset
        self.lookback = timedelta(weeks=history_weeks)
        self.zone = ZoneInfo(calendar_timezone)
        self.calendar_timezone = calendar_timezone
        # Each interval is split at UTC-hour boundaries for duration-weighted buckets.
        # Raw sources remain immutable; values are accessed only after availability filtering.
        self.records = ci.records
        self.ends = [r.end for r in ci.records]
        self._frozen_training = None
        self._cache = {}

    def _aggregate(self, records, begin=None, end=None):
        totals, weights = [0.0] * 168, [0.0] * 168
        for record in records:
            cursor = max(record.start, begin) if begin is not None else record.start
            stop = min(record.end, end) if end is not None else record.end
            while cursor < stop:
                boundary = min(stop, cursor.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1))
                hours = (boundary - cursor).total_seconds() / 3600
                slot = hour_of_week(cursor, self.zone)
                totals[slot] += float(record.value) * hours
                weights[slot] += hours
                cursor = boundary
        return totals, weights

    def issue(self, at, mode="window"):
        require(mode in {"window", "current"}, "Unknown CI forecast mode")
        cutoff = at + self.ci.offset
        # Repeated decisions in one hour may see different releases, so cache by exact issue time.
        key = (at, mode)
        if key in self._cache:
            return self._cache[key]
        right = bisect_right(self.ends, cutoff)
        observed = [r for r in self.records[:right] if r.available_at <= cutoff]
        require(observed, f"No published CI history at decision {iso(at)}; provide earlier CI history")
        latest = observed[-1]
        if mode == "current":
            values = [float(latest.value)] * 168
        else:
            recent = [r for r in observed if r.end > cutoff - self.lookback]
            sums, counts = self._aggregate(recent, cutoff - self.lookback, cutoff)
            # Training fallback is frozen by train_end, and is prefix-only for earlier training decisions.
            training_cutoff = min(cutoff, self.train_end)
            if cutoff >= self.train_end and self._frozen_training is not None:
                train_sum, train_count = self._frozen_training
            else:
                training = [r for r in observed if r.end <= training_cutoff and
                            r.available_at <= training_cutoff and r.end > self.train_start]
                train_sum, train_count = self._aggregate(training, self.train_start, training_cutoff)
                if cutoff >= self.train_end:
                    self._frozen_training = (train_sum, train_count)
            if sum(train_count):
                fallback = sum(train_sum) / sum(train_count)
            else:
                # Earliest training decisions can precede the first train observation.
                # Their warmup history supplies an explicitly causal cold-start mean.
                all_sum, all_count = self._aggregate(observed)
                require(sum(all_count) > 0, "No usable CI observation duration")
                fallback = sum(all_sum) / sum(all_count)
            values = [sums[i] / counts[i] if counts[i] else
                      train_sum[i] / train_count[i] if train_count[i] else fallback for i in range(168)]
        require(all(math.isfinite(v) and v >= 0 for v in values), "Invalid CI forecast")
        payload = {"issue": iso(at), "mode": mode, "values": values,
                   "calendar_timezone": self.calendar_timezone, "offset_seconds": self.ci.offset.total_seconds(),
                   "latest_observation_end": iso(latest.end), "count": len(observed)}
        input_id = hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()
        result = FrozenForecast(at, tuple(values), self.calendar_timezone, self.ci.offset, input_id, mode, len(observed), latest.end)
        if len(self._cache) >= 512:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = result
        return result
