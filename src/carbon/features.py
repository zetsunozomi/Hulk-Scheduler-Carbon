"""Fixed-schema features from the ordinary-user observation, never Replay internals."""

import math
from statistics import fmean
from zoneinfo import ZoneInfo

from .common import require, timestamp

FEATURE_VERSION = "public_queue_v1"


def quantile(values, probability):
    require(values and 0 <= probability <= 1, "Quantile requires data and a probability")
    values = sorted(values)
    position = probability * (len(values) - 1)
    lo = int(position)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def calendar_features(at, calendar_timezone="UTC"):
    local = at.astimezone(ZoneInfo(calendar_timezone))
    hour = local.hour + local.minute / 60 + local.second / 3600
    return [math.sin(2 * math.pi * hour / 24), math.cos(2 * math.pi * hour / 24),
            math.sin(2 * math.pi * local.weekday() / 7), math.cos(2 * math.pi * local.weekday() / 7)]


class QueueFeatures:
    """Numeric slots and missing masks are stable across fitting and inference."""

    fields = {
        "pending": ("requested_seconds", "elapsed_wait_seconds"),
        "running": ("requested_seconds", "elapsed_runtime_seconds", "requested_remaining_seconds"),
    }

    def __init__(self, lags, calendar_timezone="UTC"):
        require(lags and lags == sorted(set(lags)) and lags[0] == 0, "Invalid feature lags")
        self.lags = list(lags)
        self.calendar_timezone = calendar_timezone
        ZoneInfo(calendar_timezone)
        names = []
        for lag in lags:
            prefix = f"lag_{lag}."
            base = ["missing", "sample_age_hours", "capacity", "available_fraction"]
            for group, fields in self.fields.items():
                base += [group + ".log_count", group + ".node_fraction"]
                base += [f"{group}.{field}.{stat}" for field in fields for stat in ("log_mean_hours", "log_p90_hours", "log_max_hours")]
            names += [prefix + name for name in base]
        self.history_names = names + ["hour_sin", "hour_cos", "weekday_sin", "weekday_cos"]
        self.names = self.history_names + ["request.node_fraction", "request.log_hours"]

    def history(self, history, at):
        require([h["lag_seconds"] for h in history] == self.lags, "History feature schema changed")
        result = []
        for sample in history:
            if sample["missing"]:
                count = 4 + sum(2 + 3 * len(fields) for fields in self.fields.values())
                result += [1.0] + [0.0] * (count - 1)
                continue
            state = sample["state"]
            observed_at = timestamp(state["timestamp_utc"])
            require(observed_at <= at, "Feature snapshot is from the future")
            capacity = state["capacity_nodes"]
            require(capacity > 0, "Nonpositive feature capacity")
            result += [0.0, sample["sample_age_seconds"] / 3600, float(capacity), state["available_nodes"] / capacity]
            for group, fields in self.fields.items():
                jobs = state[group]
                result += [math.log1p(len(jobs)), sum(j["nodes"] for j in jobs) / capacity]
                for field in fields:
                    values = [j[field] / 3600 for j in jobs]
                    stats = (fmean(values), quantile(values, .9), max(values)) if values else (0, 0, 0)
                    require(all(v >= 0 and math.isfinite(v) for v in stats), "Invalid public queue values")
                    result += [math.log1p(v) for v in stats]
        result += calendar_features(at, self.calendar_timezone)
        require(len(result) == len(self.history_names), "Internal feature length mismatch")
        return result

    def request(self, history, at, nodes, requested_seconds):
        require(not history[0]["missing"], "Current public state is required")
        capacity = history[0]["state"]["capacity_nodes"]
        require(0 < nodes <= capacity and requested_seconds > 0, "Invalid request descriptor")
        return self.history(history, at) + [nodes / capacity, math.log1p(requested_seconds / 3600)]

    def metadata(self):
        return dict(version=FEATURE_VERSION, lags=self.lags, calendar_timezone=self.calendar_timezone, names=self.names)
