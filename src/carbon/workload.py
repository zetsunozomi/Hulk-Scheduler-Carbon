"""One work definition drives scheduling, progress and accounting."""

from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from datetime import timedelta

from .common import ContractError, duration, integer, number, require, seconds


@dataclass(frozen=True)
class ScaleProfile:
    nodes: int
    updates_per_hour: Decimal
    initialization_seconds: Decimal
    restart_seconds: Decimal
    checkpoint_seconds: Decimal
    microbatch: int
    accumulation: int


@dataclass(frozen=True)
class ChunkPlan:
    nodes: int
    updates: int
    remaining_before: int
    setup: timedelta
    training: timedelta
    checkpoint: timedelta
    requested: timedelta
    first: bool

    @property
    def actual(self):
        return self.setup + self.training + self.checkpoint

    def phases(self, start):
        train_start = start + self.setup
        save_start = train_start + self.training
        return [
            ("initialization" if self.first else "restart", start, train_start),
            ("training", train_start, save_start),
            ("checkpoint", save_start, save_start + self.checkpoint),
        ]


class Workload:
    def __init__(self, config, allowed_nodes, max_request_seconds, walltime_resolution_seconds):
        self.name = config["name"]
        self.total_updates = integer(config["optimizer_updates"], "optimizer_updates")
        self.global_batch = integer(config["global_batch"], "global_batch")
        self.gpus_per_node = integer(config["gpus_per_node"], "gpus_per_node")
        self.max_request = number(max_request_seconds, "max_request_seconds", strict=True)
        self.resolution = number(walltime_resolution_seconds, "walltime_resolution_seconds", strict=True)
        require(self.max_request % self.resolution == 0, "max_request must be a multiple of walltime resolution")
        self.profiles = {}
        require(set(config["profiles"]) == {str(n) for n in allowed_nodes},
                "Workload profiles must match the allowed action scales exactly")
        for n in allowed_nodes:
            p = config["profiles"][str(n)]
            self.profiles[n] = ScaleProfile(
                n, number(p["updates_per_hour"], f"rate[{n}]", strict=True),
                number(p["initialization_seconds"], f"initialization[{n}]"),
                number(p["restart_seconds"], f"restart[{n}]"),
                number(p["checkpoint_seconds"], f"checkpoint[{n}]"),
                integer(p["microbatch"], f"microbatch[{n}]"),
                integer(p["accumulation"], f"accumulation[{n}]"))
            profile = self.profiles[n]
            require(n * self.gpus_per_node * profile.microbatch * profile.accumulation == self.global_batch,
                    f"Scale {n}: nodes * GPUs * microbatch * accumulation != global batch")
            # Validate duration precision before running any episodes.
            for value in (profile.initialization_seconds, profile.restart_seconds, profile.checkpoint_seconds):
                duration(value)
        require(4 in self.profiles, "Fixed-4 reference profile is required for eta normalization")
        require(any(self.plan(n, self.total_updates, True) for n in allowed_nodes), "No feasible initial action")
        require(any(self.plan(n, self.total_updates, False) for n in allowed_nodes), "No feasible continuation action")

    def eta(self, nodes):
        return self.profiles[nodes].updates_per_hour * 4 / (self.profiles[4].updates_per_hour * nodes)

    def plan(self, nodes, remaining, first):
        require(nodes in self.profiles, f"Unsupported scale: {nodes}")
        require(isinstance(remaining, int) and 0 < remaining <= self.total_updates, "Invalid remaining updates")
        p = self.profiles[nodes]
        setup = p.initialization_seconds if first else p.restart_seconds
        useful_seconds = self.max_request - setup - p.checkpoint_seconds
        updates = min(remaining, int((p.updates_per_hour * useful_seconds / 3600).to_integral_value(rounding=ROUND_FLOOR)))
        if updates <= 0:
            return None
        train_us = int((Decimal(updates) * 3600 * 1000000 / p.updates_per_hour).to_integral_value(rounding=ROUND_CEILING))
        actual_seconds = setup + Decimal(train_us) / 1000000 + p.checkpoint_seconds
        request = (actual_seconds / self.resolution).to_integral_value(rounding=ROUND_CEILING) * self.resolution
        require(request <= self.max_request, "Rounded request exceeds walltime cap")
        return ChunkPlan(nodes, updates, remaining, duration(setup), timedelta(microseconds=train_us),
                         duration(p.checkpoint_seconds), duration(request), first)
