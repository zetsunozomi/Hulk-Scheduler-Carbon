"""Stepwise realized CI and reusable per-scale exposure; no forecast information here."""

from bisect import bisect_right
import csv
from dataclasses import dataclass
from decimal import Decimal

from .common import ContractError, duration, iso, number, require, seconds, timestamp


@dataclass(frozen=True)
class CIRecord:
    start: object
    end: object
    value: Decimal
    available_at: object


class CarbonSeries:
    def __init__(self, records, source, region, unit="gCO2e/kWh", offset_seconds=0):
        require(records, "CI series is empty")
        self.records = tuple(sorted(records, key=lambda r: r.start))
        self.starts = tuple(r.start for r in self.records)
        self.source, self.region, self.unit = source, region, unit
        # Offset explicitly maps queue UTC -> CI UTC; never silently aligns calendar years.
        self.offset = duration(offset_seconds) if offset_seconds >= 0 else -duration(-offset_seconds)
        require(unit in {"gCO2e/kWh", "gCO2/kWh"}, "CI unit must be gCO2/kWh or gCO2e/kWh; convert inputs explicitly")
        for i, r in enumerate(self.records):
            require(r.start < r.end, "CI interval must have positive duration")
            require(r.value.is_finite() and r.value >= 0, "CI value must be finite and nonnegative")
            require(r.available_at >= r.start, "CI observation availability precedes observation interval")
            if i:
                require(self.records[i - 1].end <= r.start, "Overlapping or duplicate CI intervals")

    @classmethod
    def load(cls, path, config):
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            value_column = {"gCO2/kWh": "gco2_per_kwh", "gCO2e/kWh": "gco2e_per_kwh"}.get(config["unit"])
            require(value_column is not None, "Unsupported CI unit")
            required = {"start_utc", "end_utc", value_column, "available_at_utc"}
            require(required <= set(reader.fieldnames or ()), f"CI CSV requires {sorted(required)}")
            records = []
            for line, row in enumerate(reader, 2):
                try:
                    records.append(CIRecord(timestamp(row["start_utc"]), timestamp(row["end_utc"]),
                                            number(row[value_column], "CI"), timestamp(row["available_at_utc"])))
                except (ValueError, KeyError) as exc:
                    raise ContractError(f"CI {path}:{line}: {exc}") from exc
        return cls(records, config["source"], config["region"], config["unit"], config["queue_to_ci_offset_seconds"])

    def integral(self, start, end):
        """Declared CI unit times hours. Fail on gaps; never silently extrapolate."""
        require(start <= end, "Negative CI integration interval")
        if start == end:
            return Decimal(0)
        cursor, stop = start + self.offset, end + self.offset
        i = bisect_right(self.starts, cursor) - 1
        require(i >= 0, f"CI begins after requested start {iso(cursor)}")
        total = Decimal(0)
        while cursor < stop:
            require(i < len(self.records), f"CI ends before {iso(stop)}")
            r = self.records[i]
            require(r.start <= cursor < r.end, f"CI gap at {iso(cursor)}")
            boundary = min(stop, r.end)
            total += r.value * seconds(boundary - cursor) / 3600
            cursor, i = boundary, i + 1
        return total

    def observations_available_at(self, decision_time):
        """Only released, completed observations, in the mapped CI calendar (for P2)."""
        cutoff = decision_time + self.offset
        return tuple(r for r in self.records if r.end <= cutoff and r.available_at <= cutoff)


class Exposure:
    def __init__(self, nodes):
        self.L = {n: Decimal(0) for n in nodes}

    def add(self, nodes, phases, ci):
        values = []
        for name, start, end in phases:
            exposure = nodes * ci.integral(start, end)
            self.L[nodes] += exposure
            exposure_key = ("exposure_node_gco2_per_kwh_hours" if ci.unit == "gCO2/kWh"
                            else "exposure_node_gco2e_per_kwh_hours")
            values.append({"phase": name, "start_utc": iso(start), "end_utc": iso(end),
                           "ci_unit": ci.unit, exposure_key: float(exposure)})
        return values

    def summary(self, workload, power):
        a = sum(self.L.values(), Decimal(0))
        b = sum((workload.eta(n) * value for n, value in self.L.items()), Decimal(0))
        spec = number(power["reference_kw"], "reference_kw", strict=True)
        endpoints = {str(rho): float(spec * (b + number(rho, "rho") * (a - b))) for rho in power["rho_interval"]}
        kappa = power["workload_coefficient"]
        return {"exposure_by_nodes": {str(n): float(v) for n, v in self.L.items()},
                "A": float(a), "B": float(b), "carbon_g_per_kappa": endpoints,
                "carbon_g": None if kappa is None else {r: float(number(kappa, "kappa", strict=True)) * v for r, v in endpoints.items()},
                "power_is_measured": False}
