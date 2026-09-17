"""Full-work causal rollout planning, shared by MPC, plan-once and queue-blind control."""

from bisect import bisect_right
from dataclasses import dataclass
from datetime import timedelta
import itertools
import math
import random
import time

from .common import require, seconds, timestamp


@dataclass(frozen=True)
class PlanningResult:
    sequence: tuple
    plan_key: tuple
    estimated_miss: float
    endpoint_costs: tuple
    objective: float
    estimated_feasible: bool
    candidate_plans: int
    sample_paths: int
    planning_seconds: float

    def metadata(self):
        return {"planned_sequence": list(self.sequence), "plan_key": list(self.plan_key),
                "estimated_miss": self.estimated_miss, "estimated_endpoint_costs": list(self.endpoint_costs),
                "estimated_worst_normalized_cost": self.objective, "estimated_feasible": self.estimated_feasible,
                "candidate_plans": self.candidate_plans, "planning_paths": self.sample_paths,
                "planning_seconds": self.planning_seconds,
                "planner_approximation": "current queue/calendar features frozen; independent waits; current-issued CI forecast"}


def past_cost_estimate(executions, ci, work, power, decision):
    """Use released CI; fill unreleased past intervals with their submission forecast.

    True exposure logs may be computed with the revised realized CI archive for
    evaluation. They are deliberately not read by the controller.
    """
    costs = [0.0, 0.0]
    cutoff = decision + ci.offset
    for item in executions:
        n, forecast = item["nodes"], item["forecast"]
        begin, end = item["start"], item["end"]
        cursor = begin + ci.offset
        stop = end + ci.offset
        integral = 0.0
        while cursor < stop:
            index = bisect_right(ci.starts, cursor) - 1
            if index >= 0 and ci.records[index].start <= cursor < ci.records[index].end:
                record = ci.records[index]
                boundary = min(stop, record.end)
                if record.end <= cutoff and record.available_at <= cutoff:
                    integral += float(record.value) * (boundary-cursor).total_seconds() / 3600
                else:
                    integral += forecast.integral(cursor-ci.offset, boundary-ci.offset)
            else:
                boundary = min(stop, ci.records[index+1].start) if index+1 < len(ci.records) else stop
                integral += forecast.integral(cursor-ci.offset, boundary-ci.offset)
            cursor = boundary
        for i, rho in enumerate(power["rho_interval"]):
            scale_power = float(power["reference_kw"]) * (float(rho) + (1-float(rho)) * float(work.eta(n)))
            costs[i] += n * scale_power * integral
    return costs


class RolloutPlanner:
    def __init__(self, work, power, predictor, references, paths=256, miss_tolerance=.05, queue_blind=False):
        require(isinstance(paths, int) and paths > 0 and 0 <= miss_tolerance <= 1, "Invalid planner settings")
        self.work, self.power, self.predictor = work, power, predictor
        self.paths, self.miss_tolerance, self.queue_blind = paths, miss_tolerance, queue_blind
        self.normalizers = [references["carbon_reference_g_per_kappa"][str(r)] for r in power["rho_interval"]]
        require(all(math.isfinite(v) and v > 0 for v in self.normalizers), "Planner normalizers must be positive")

    def plans(self, remaining, first):
        unique = {}
        nodes = sorted(self.work.profiles)
        for key in itertools.product(nodes, repeat=3):
            left, steps = remaining, []
            while left:
                n = key[min(len(steps), 2)]
                plan = self.work.plan(n, left, first and not steps)
                if plan is None:
                    break
                steps.append(plan)
                left -= plan.updates
                require(len(steps) <= 10000, "More than 10000 chunks to completion; audit workload/walltime configuration")
            if left == 0:
                identity = tuple(p.nodes for p in steps)
                unique.setdefault(identity, (key, steps))
        require(unique, "No complete positive-progress plan")
        return list(unique.values())

    def decide(self, observation, at, forecast, first, past_cost=(0.0, 0.0), seed=0):
        require(forecast.issued_at == at, "MPC must use a forecast frozen at this decision")
        begun = time.monotonic()
        plans = self.plans(observation["remaining_updates"], first)
        rng = random.Random(seed)
        uniforms = [[rng.random() for _ in range(max(len(s) for _, s in plans))] for _ in range(self.paths)]
        atom_cache = {}
        summaries = []
        for key, steps in plans:
            means, misses = [0.0, 0.0], 0
            for path in range(self.paths):
                cursor = at
                carbon = [0.0, 0.0]
                for j, plan in enumerate(steps):
                    identity = (plan.nodes, float(seconds(plan.requested)))
                    if identity not in atom_cache:
                        atom_cache[identity] = [0.0] if self.queue_blind else self.predictor.atoms(
                            observation["history"], at, *identity)
                    atoms = atom_cache[identity]
                    wait = atoms[min(int(uniforms[path][j]*len(atoms)), len(atoms)-1)]
                    start = cursor + timedelta(hours=wait)
                    end = start + plan.actual
                    exposure = plan.nodes * forecast.integral(start, end)
                    for i, rho in enumerate(self.power["rho_interval"]):
                        carbon[i] += float(self.power["reference_kw"]) * (float(rho)+(1-float(rho))*float(self.work.eta(plan.nodes))) * exposure
                    cursor = end
                misses += (cursor-at).total_seconds()/3600 > observation["remaining_budget_hours"]
                for i in range(2):
                    means[i] += carbon[i]/self.paths
            total = tuple(float(past_cost[i])+means[i] for i in range(2))
            objective = max(total[i]/self.normalizers[i] for i in range(2))
            miss = misses/self.paths
            summaries.append((miss <= self.miss_tolerance, objective, miss, key, steps, total))
        feasible = [r for r in summaries if r[0]]
        chosen = min(feasible, key=lambda r:(r[1],r[3])) if feasible else min(summaries, key=lambda r:(r[2],r[1],r[3]))
        ok, objective, miss, key, steps, total = chosen
        return PlanningResult(tuple(p.nodes for p in steps), key, miss, total, objective, ok,
                              len(plans), self.paths, time.monotonic()-begun)


class PlanningPolicy:
    def __init__(self, planner, forecaster, mode="mpc", seed=11):
        require(mode in {"mpc", "plan-once"}, "Unknown planning feedback mode")
        self.planner, self.forecaster, self.mode, self.seed = planner, forecaster, mode, seed
        self.executions, self.result, self.forecast = [], None, None
        self.chunk_index = 0

    def choose(self, observation):
        at = timestamp(observation["history"][0]["state"]["timestamp_utc"])
        replanned = self.result is None or self.mode == "mpc"
        if replanned:
            self.forecast = self.forecaster.issue(at)
            past = past_cost_estimate(self.executions, self.forecaster.ci, self.planner.work, self.planner.power, at)
            self.result = self.planner.decide(observation, at, self.forecast, self.chunk_index == 0,
                                               past, self.seed + self.chunk_index)
        index = self.chunk_index if self.mode == "plan-once" else 0
        require(index < len(self.result.sequence), "Plan-once ran beyond its complete-work plan")
        metadata = {**self.forecast.metadata(), **self.result.metadata(), "replanned": replanned,
                    "decision_planning_seconds": self.result.planning_seconds if replanned else 0.0,
                    "predictor_version": self.planner.predictor.version,
                    "fallback_reason": None if self.result.estimated_feasible else "no_estimated_feasible_plan"}
        return self.result.sequence[index], metadata

    def record_execution(self, chunk):
        # Keep only observable allocation times; no realized carbon or hidden queue state.
        if chunk["start_utc"] is not None:
            self.executions.append({"nodes": chunk["selected_nodes"], "start": timestamp(chunk["start_utc"]),
                                    "end": timestamp(chunk["observed_until_utc"]), "forecast": self.forecast})
        self.chunk_index += 1
