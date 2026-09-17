"""Canonical P1 environment: checkpoint-boundary actions, fixed work, full accounting."""

from decimal import Decimal, ROUND_FLOOR

from .carbon import Exposure
from .common import ContractError, duration, iso, number, require, seconds
from .replay import Replay, Request


def initial_replay(bundle, episode):
    config = bundle.raw
    execution = config["execution"]
    if execution.get("initial_state_mode", "observed") == "empty_warmup":
        # All episodes in this scenario share one background origin. Reusing
        # the unmodified background prefix is safe; target runs use clones.
        replay = getattr(bundle, "_background_prefix", None)
        if replay is None or replay.time > episode.arrival:
            replay = Replay(bundle.jobs, bundle.trace_start, bundle.trace_end, config["cluster"],
                            max(execution["history_lags_seconds"]), execution["history_sample_seconds"],
                            initialize_from_observed=False)
        replay.advance_to(episode.arrival, before_dispatch=True)
        bundle._background_prefix = replay
        return replay.clone()
    begin = episode.arrival - duration(execution["warmup_seconds"])
    replay = Replay(bundle.jobs, begin, bundle.trace_end, config["cluster"],
                    max(execution["history_lags_seconds"]), execution["history_sample_seconds"],
                    initialize_from_observed=execution.get("initial_state_mode", "observed") == "observed")
    replay.advance_to(episode.arrival, before_dispatch=True)
    return replay


class Environment:
    def __init__(self, bundle, episode, initial, method, seed=0):
        require(initial.time == episode.arrival, "Initial snapshot does not match cohort arrival")
        self.bundle, self.episode, self.method, self.seed = bundle, episode, method, seed
        self.replay = initial.clone()
        self.work = bundle.workload
        self.remaining = self.work.total_updates
        self.chunks = []
        self.exposure = Exposure(self.work.profiles)
        self.nodehours = Decimal(0)
        self.status = "running"
        self.stop_reason = None
        self.boundary = bundle.splits[episode.split][1]
        self.boundary_reason = "split_end" if self.boundary < bundle.trace_end else "trace_end"
        timeout = bundle.raw["execution"]["max_episode_seconds"]
        if timeout is not None and episode.arrival + duration(timeout) < self.boundary:
            self.boundary = episode.arrival + duration(timeout)
            self.boundary_reason = "episode_timeout"

    def observe(self):
        elapsed = seconds(self.replay.time - self.episode.arrival)
        descriptors = []
        for n in self.work.profiles:
            plan = self.work.plan(n, self.remaining, not self.chunks) if self.remaining else None
            descriptors.append({"nodes": n, "feasible": plan is not None,
                                "updates_per_hour": float(self.work.profiles[n].updates_per_hour),
                                "eta": float(self.work.eta(n)),
                                "planned_updates": plan.updates if plan else 0,
                                "actual_seconds": float(seconds(plan.actual)) if plan else None,
                                "requested_seconds": float(seconds(plan.requested)) if plan else None})
        return {"remaining_updates": self.remaining, "total_updates": self.work.total_updates,
                "remaining_budget_hours": self.episode.budget_hours - float(elapsed / 3600),
                "budget_hours": self.episode.budget_hours,
                "history": self.replay.visible_history(self.bundle.raw["execution"]["history_lags_seconds"]),
                "actions": descriptors}

    def _identity(self):
        return {"schema_version": 1, "episode_id": self.episode.episode_id, "panel": self.bundle.raw["panel"],
                "purpose": self.bundle.raw["purpose"], "split": self.episode.split,
                "initial_arrival_utc": iso(self.episode.arrival), "method": self.method,
                "seed": self.seed, "budget_hours": self.episode.budget_hours,
                "ci_unit": self.bundle.ci.unit,
                "carbon_unit": "gCO2" if self.bundle.ci.unit == "gCO2/kWh" else "gCO2e",
                "forecast_issue_utc": None, "forecast_input_id": None,
                "predictor_version": None, "policy_checkpoint": None}

    def step(self, nodes):
        require(self.status == "running", "Cannot step a terminal episode")
        plan = self.work.plan(nodes, self.remaining, not self.chunks)
        require(plan is not None, f"Scale {nodes} is masked: no positive progress fits walltime")
        submitted = self.replay.time
        job_id = f"target:{self.episode.episode_id}:{len(self.chunks)}"
        self.replay.inject(Request(job_id, nodes, submitted, plan.requested, plan.actual, target=True))
        allocation = self.replay.until(job_id, "complete", self.boundary)
        completed = allocation is not None
        if not completed:
            allocation = self.replay.running.get(job_id)
        observed_end = self.replay.time
        phases = []
        if allocation is not None:
            for name, start, end in plan.phases(allocation.start):
                if start < observed_end:
                    phases.append((name, start, min(end, observed_end)))
        chunk_exposure = Exposure(self.work.profiles)
        phase_logs = chunk_exposure.add(nodes, phases, self.bundle.ci)
        for n, value in chunk_exposure.L.items():
            self.exposure.L[n] += value
        allocated = sum((seconds(end - start) for _, start, end in phases), Decimal(0))
        self.nodehours += nodes * allocated / 3600
        updates = plan.updates if completed else 0  # Progress is committed only after checkpoint completes.
        self.remaining -= updates
        processed_uncommitted = 0
        if not completed and allocation is not None:
            training_seconds = sum((seconds(end - start) for name, start, end in phases if name == "training"), Decimal(0))
            processed_uncommitted = min(plan.updates, int((training_seconds * self.work.profiles[nodes].updates_per_hour / 3600).to_integral_value(rounding=ROUND_FLOOR)))
        if self.remaining == 0:
            self.status = "completed"
        elif not completed:
            self.status = "timeout" if self.boundary_reason == "episode_timeout" else "censored"
            self.stop_reason = self.boundary_reason
        elif self.replay.time == self.boundary:
            # A checkpoint can finish exactly at the boundary while work remains.
            self.status = "timeout" if self.boundary_reason == "episode_timeout" else "censored"
            self.stop_reason = self.boundary_reason
        log = {**self._identity(), "chunk_id": len(self.chunks), "selected_nodes": nodes,
               "remaining_updates_before": plan.remaining_before, "planned_updates": plan.updates,
               "completed_updates": updates, "processed_updates_uncommitted": processed_uncommitted,
               "remaining_updates_after": self.remaining,
               "global_next_sample_index": (self.work.total_updates - self.remaining) * self.work.global_batch,
               "requested_walltime_hours": float(seconds(plan.requested) / 3600),
               "planned_allocation_hours": float(seconds(plan.actual) / 3600),
               "submit_utc": iso(submitted), "start_utc": iso(allocation.start) if allocation else None,
               "end_utc": iso(allocation.end) if completed else None, "observed_until_utc": iso(observed_end),
               "queue_wait_hours": float(seconds(allocation.start - submitted) / 3600) if allocation else None,
               "observed_allocation_hours": float(allocated / 3600), "phases": phase_logs,
               "retry_count": 0, "fallback_reason": self.stop_reason,
               "final_status": "completed" if completed else self.status,
               **chunk_exposure.summary(self.work, self.bundle.raw["power"])}
        self.chunks.append(log)
        return log

    def summary(self):
        require(self.status != "running", "Episode is not terminal")
        elapsed = seconds(self.replay.time - self.episode.arrival) / 3600
        deadline = number(self.episode.budget_hours, "budget")
        complete = self.status == "completed"
        # A run still unfinished at the exact deadline must finish after it.
        miss = bool(elapsed > deadline) if complete else (True if elapsed >= deadline else None)
        account = self.exposure.summary(self.work, self.bundle.raw["power"])
        return {**self._identity(), "final_status": self.status, "censor_flag": not complete,
                "stop_reason": self.stop_reason, "completed_updates": self.work.total_updates - self.remaining,
                "remaining_updates": self.remaining, "chunk_count": len(self.chunks),
                "observed_until_utc": iso(self.replay.time), "observed_elapsed_hours": float(elapsed),
                "tat_hours": float(elapsed) if complete else None, "deadline_miss": miss,
                "observed_nodehours": float(self.nodehours), "nodehours": float(self.nodehours) if complete else None,
                "exposure_by_nodes": account["exposure_by_nodes"], "A": account["A"], "B": account["B"],
                "exposure_is_complete": complete,
                "observed_carbon_g_per_kappa": account["carbon_g_per_kappa"],
                "carbon_g_per_kappa": account["carbon_g_per_kappa"] if complete else None,
                "carbon_g": account["carbon_g"] if complete else None,
                "power_is_measured": False}
