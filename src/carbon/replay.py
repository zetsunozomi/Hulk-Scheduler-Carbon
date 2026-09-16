"""Deterministic priority/backfill replay with explicit exact-time stop boundaries.

This is a documented node-level approximation, not an emulation of all Slurm
features. Reservations use requested walltime; hidden actual duration is used
only for completion events. Priority uses age and size, never logged priority.
"""

from bisect import bisect_right
from collections import deque
import copy
from dataclasses import dataclass
from datetime import timedelta

from .common import ContractError, duration, iso, require, seconds


@dataclass(frozen=True)
class Request:
    job_id: str
    nodes: int
    submit: object
    requested: timedelta
    actual: timedelta
    target: bool = False


@dataclass(frozen=True)
class Allocation:
    request: Request
    start: object

    @property
    def end(self):
        return self.start + self.request.actual

    @property
    def reservation_end(self):
        return self.start + self.request.requested


class Replay:
    def __init__(self, jobs, start, coverage_end, cluster, history_seconds=172800, sample_seconds=600):
        require(start < coverage_end, "Replay start must precede coverage end")
        self.time, self.coverage_end, self.origin = start, coverage_end, start
        self.capacity = cluster["nodes"]
        self.config = cluster["scheduler"]
        self.tick = duration(self.config["dispatch_interval_seconds"])
        self.sample_step = duration(sample_seconds)
        self.history_span = duration(history_seconds)
        self.next_tick = start
        self.next_sample = start
        self.pending, self.running, self.finished = {}, {}, {}
        self.history = deque()
        self._submitted = set()
        future = []
        for job in jobs:
            request = Request(job.job_id, job.nodes, job.submit, job.requested, job.runtime)
            self._validate(request)
            if job.submit >= start:
                if job.submit < coverage_end:
                    future.append(request)
            elif job.observed_start >= start:
                self.pending[job.job_id] = request
                self._submitted.add(job.job_id)
            elif job.observed_end > start:
                self.running[job.job_id] = Allocation(request, job.observed_start)
                self._submitted.add(job.job_id)
        self.future = tuple(sorted(future, key=lambda r: (r.submit, r.job_id)))
        self.index = 0
        self._check_capacity()

    def clone(self):
        # Immutable exogenous records may be shared; all evolving state is independent.
        result = copy.copy(self)
        for name in ("pending", "running", "finished", "history", "_submitted"):
            setattr(result, name, copy.deepcopy(getattr(self, name)))
        return result

    def _validate(self, request):
        require(0 < request.nodes <= self.capacity, f"Invalid nodes for {request.job_id}")
        require(timedelta(0) < request.actual <= request.requested, f"Invalid duration for {request.job_id}")

    def inject(self, request):
        self._validate(request)
        require(request.submit == self.time, "Injected request must submit at the current decision time")
        require(request.job_id not in self._submitted, f"Duplicate request: {request.job_id}")
        self.pending[request.job_id] = request
        self._submitted.add(request.job_id)

    def _check_capacity(self):
        require(sum(a.request.nodes for a in self.running.values()) <= self.capacity,
                "Running jobs exceed capacity; check initial state and partition membership")

    def _arrivals_and_completions(self):
        for job_id, allocation in list(self.running.items()):
            if allocation.end <= self.time:
                require(allocation.end == self.time, "Replay skipped a completion event")
                del self.running[job_id]
                # Background completions need not accumulate for years of replay.
                if allocation.request.target:
                    self.finished[job_id] = allocation
        while self.index < len(self.future) and self.future[self.index].submit <= self.time:
            request = self.future[self.index]
            require(request.submit == self.time, "Replay skipped an arrival event")
            require(request.job_id not in self._submitted, "Duplicate background arrival")
            self.pending[request.job_id] = request
            self._submitted.add(request.job_id)
            self.index += 1

    def _priority(self, request):
        age = float(seconds(self.time - request.submit)) / self.config["age_max_seconds"]
        return self.config["age_weight"] * min(age, 1.0) + self.config["size_weight"] * request.nodes / self.capacity

    @staticmethod
    def _reserve(starts, free, request):
        """Reserve the earliest contiguous capacity in a piecewise-constant calendar.

        Each request scans the calendar once. No minute-by-minute expansion or
        repeated pairwise interval overlap scans are needed for long queues.
        """
        candidate = None
        for index, available in enumerate(free):
            if available < request.nodes:
                candidate = None
                continue
            if candidate is None:
                candidate = starts[index]
            stop = candidate + request.requested
            if index + 1 == len(starts) or starts[index + 1] >= stop:
                break
        require(candidate is not None, "Backfill could not reserve a capacity-feasible request")
        stop = candidate + request.requested
        # Insert both boundaries before decrementing [candidate, stop).
        for boundary in (candidate, stop):
            index = bisect_right(starts, boundary) - 1
            if starts[index] != boundary:
                starts.insert(index + 1, boundary)
                free.insert(index + 1, free[index])
        left, right = bisect_right(starts, candidate) - 1, bisect_right(starts, stop) - 1
        for index in range(left, right):
            free[index] -= request.nodes
            require(free[index] >= 0, "Negative reservation capacity")
        return candidate

    def _dispatch(self):
        if not self.pending:
            return
        releases = {}
        for allocation in self.running.values():
            end = allocation.reservation_end
            require(end > self.time, "Active allocation outlived requested walltime")
            releases[end] = releases.get(end, 0) + allocation.request.nodes
        starts = [self.time]
        free = [self.capacity - sum(a.request.nodes for a in self.running.values())]
        for end, nodes in sorted(releases.items()):
            starts.append(end)
            free.append(free[-1] + nodes)
        ordered = sorted(self.pending.values(), key=lambda r: (-self._priority(r), r.submit, r.job_id))
        # Conservative backfill: reserve every considered higher-priority request.
        # Exact interval boundaries prevent actual-runtime oracle reservations.
        for request in ordered[:self.config["max_job_test"]]:
            candidate = self._reserve(starts, free, request)
            if candidate == self.time:
                self.running[request.job_id] = Allocation(request, self.time)
                del self.pending[request.job_id]
        self._check_capacity()

    def visible(self):
        """Ordinary-user fields only; no actual runtime, end time or priority."""
        return {
            "timestamp_utc": iso(self.time), "capacity_nodes": self.capacity,
            "available_nodes": self.capacity - sum(a.request.nodes for a in self.running.values()),
            "pending": [{"nodes": r.nodes, "requested_seconds": float(seconds(r.requested)),
                         "elapsed_wait_seconds": float(seconds(self.time - r.submit))}
                        for r in sorted(self.pending.values(), key=lambda r: r.job_id)],
            "running": [{"nodes": a.request.nodes,
                         "requested_seconds": float(seconds(a.request.requested)),
                         "elapsed_runtime_seconds": float(seconds(self.time - a.start)),
                         "elapsed_wait_seconds": float(seconds(a.start - a.request.submit)),
                         "requested_remaining_seconds": max(0.0, float(seconds(a.reservation_end - self.time)))}
                        for a in sorted(self.running.values(), key=lambda a: a.request.job_id)]}

    def visible_history(self, lag_seconds):
        samples = list(self.history)
        times = [t for t, _ in samples]
        result = []
        for lag in lag_seconds:
            desired = self.time - duration(lag)
            if lag == 0:
                result.append({"lag_seconds": lag, "missing": False, "sample_age_seconds": 0, "state": self.visible()})
                continue
            index = bisect_right(times, desired) - 1
            if index < 0:
                result.append({"lag_seconds": lag, "missing": True, "sample_age_seconds": None, "state": None})
            else:
                sampled_at, state = samples[index]
                result.append({"lag_seconds": lag, "missing": False,
                               "sample_age_seconds": float(seconds(self.time - sampled_at)), "state": copy.deepcopy(state)})
        return result

    def _dispatch_and_sample(self):
        if self.next_tick == self.time:
            self._dispatch()
            self.next_tick += self.tick
        if self.next_sample == self.time:
            self.history.append((self.time, self.visible()))
            # Retain one sample before the longest lag, for as-of lookup.
            while len(self.history) > 1 and self.history[1][0] < self.time - self.history_span:
                self.history.popleft()
            self.next_sample += self.sample_step

    def _next_time(self, limit):
        candidates = [limit, self.next_tick, self.next_sample]
        if self.index < len(self.future):
            candidates.append(self.future[self.index].submit)
        candidates.extend(a.end for a in self.running.values())
        result = min(candidates)
        require(result > self.time, "Non-progressing simulation event")
        return result

    def advance_to(self, target, before_dispatch=False):
        require(self.time <= target <= self.coverage_end, "Advance outside known trace coverage")
        while True:
            self._arrivals_and_completions()
            if self.time == target and before_dispatch:
                return
            self._dispatch_and_sample()
            if self.time == target:
                return
            self.time = self._next_time(target)

    def until(self, job_id, phase, limit):
        """Stop at the exact start/completion or at the declared censor boundary."""
        require(phase in {"start", "complete"}, "Unknown stop phase")
        require(job_id in self._submitted, "Unknown target request")
        require(self.time <= limit <= self.coverage_end, "Invalid stopping boundary")
        while True:
            self._arrivals_and_completions()
            if job_id in self.finished:
                return self.finished[job_id]
            if phase == "start" and job_id in self.running:
                return self.running[job_id]
            if self.time == limit:
                return None
            self._dispatch_and_sample()
            if phase == "start" and job_id in self.running:
                return self.running[job_id]
            self.time = self._next_time(limit)
