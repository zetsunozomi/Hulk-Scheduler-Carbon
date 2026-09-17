"""Versioned, fail-closed configuration and asset/cohort validation."""

import csv
import heapq
from dataclasses import dataclass
from pathlib import Path

from .carbon import CarbonSeries
from .common import ContractError, digest, duration, integer, iso, load_json, number, require, timestamp
from .trace import load_trace
from .workload import Workload


@dataclass(frozen=True)
class Episode:
    episode_id: str
    arrival: object
    split: str
    budget_hours: float


def keys(data, required, optional=(), location="config"):
    require(isinstance(data, dict), f"{location}: expected an object")
    require(set(required) <= data.keys(), f"{location}: missing keys {sorted(set(required) - data.keys())}")
    require(data.keys() <= set(required) | set(optional),
            f"{location}: unknown keys {sorted(data.keys() - set(required) - set(optional))}")


def nonempty(value, name):
    require(isinstance(value, str) and bool(value.strip()), f"{name}: supply a nonempty string")


def unfilled(data, prefix=""):
    allowed_null = {"trace.dst_fold", "power.workload_coefficient", "execution.max_episode_seconds"}
    if isinstance(data, dict):
        return [p for key, value in data.items() for p in unfilled(value, f"{prefix}.{key}".lstrip("."))]
    if isinstance(data, list):
        return [p for i, value in enumerate(data) for p in unfilled(value, f"{prefix}[{i}]")]
    return [prefix] if data is None and prefix not in allowed_null else []


def asset(root, config, label):
    nonempty(config["path"], f"{label}.path")
    path = (root / config["path"]).resolve()
    require(path.is_file(), f"{label}: missing file {path}")
    actual = digest(path)
    require(isinstance(config["sha256"], str) and actual == config["sha256"],
            f"{label}: SHA256 mismatch; actual={actual}; audit source before updating the manifest")
    return path


class Bundle:
    def __init__(self, path, queue_only=False):
        self.path = Path(path).resolve()
        self.raw = c = load_json(self.path)
        missing = [field for field in unfilled(c) if not (queue_only and
                   field.split(".")[0] in {"workload", "ci", "power", "cohort"})]
        require(not missing, "Unfilled config values: " + ", ".join(missing))
        keys(c, {"schema_version", "purpose", "panel", "root", "cluster", "trace", "workload", "ci", "power", "splits", "cohort", "execution", "holdout_audit"})
        require(c["schema_version"] == 1, "Unsupported config schema_version")
        require(c["purpose"] in {"synthetic", "development", "research"}, "Invalid purpose")
        nonempty(c["panel"], "panel")
        nonempty(c["root"], "root")
        root = (self.path.parent / c["root"]).resolve()
        self.root = root

        cl = c["cluster"]
        keys(cl, {"name", "partition", "nodes", "allowed_nodes", "max_request_seconds", "walltime_resolution_seconds", "scheduler", "provenance"}, location="cluster")
        for key in ("name", "partition", "provenance"):
            nonempty(cl[key], f"cluster.{key}")
        cl["nodes"] = integer(cl["nodes"], "cluster.nodes")
        require(isinstance(cl["allowed_nodes"], list) and cl["allowed_nodes"], "allowed_nodes must be a nonempty list")
        require(all(isinstance(n, int) and not isinstance(n, bool) for n in cl["allowed_nodes"]), "allowed_nodes must contain integers")
        require(cl["allowed_nodes"] == sorted(set(cl["allowed_nodes"])), "allowed_nodes must be sorted and unique")
        require(set(cl["allowed_nodes"]) <= {4, 8, 16, 32, 64, 128} and 4 in cl["allowed_nodes"], "Supported scales are subsets of 4/8/16/32/64/128 including 4")
        require(max(cl["allowed_nodes"]) <= cl["nodes"], "Action exceeds partition capacity")
        sched = cl["scheduler"]
        keys(sched, {"model", "dispatch_interval_seconds", "max_job_test", "age_weight", "age_max_seconds", "size_weight"}, location="scheduler")
        require(sched["model"] in {"conservative_backfill_v1", "fcfs_v1"}, "Unsupported scheduler model")
        for key in ("dispatch_interval_seconds", "max_job_test", "age_max_seconds"):
            sched[key] = integer(sched[key], f"scheduler.{key}")
        for key in ("age_weight", "size_weight"):
            sched[key] = float(number(sched[key], f"scheduler.{key}"))

        t = c["trace"]
        keys(t, {"path", "sha256", "timezone", "dst_fold", "coverage_start_utc", "coverage_end_utc", "coverage_attestation", "provenance", "zero_duration_policy", "overrun_policy"}, optional={"role", "node_multiplier", "oversize_policy"}, location="trace")
        require(t.get("role", "historical") in {"historical", "workload_template"}, "Unknown trace role")
        integer(t.get("node_multiplier", 1), "trace.node_multiplier")
        require(t.get("oversize_policy", "error") in {"error", "cap"}, "Unknown oversize policy")
        require(t.get("role", "historical") == "workload_template" or
                (t.get("node_multiplier", 1) == 1 and t.get("oversize_policy", "error") == "error"),
                "Resource transformation is only allowed for declared workload templates")
        nonempty(t["timezone"], "trace.timezone")
        for key in ("coverage_attestation", "provenance"):
            nonempty(t[key], f"trace.{key}")
        require(t["dst_fold"] in (None, 0, 1), "dst_fold must be null, 0 or 1")
        require(t["zero_duration_policy"] in {"error", "drop"}, "Unknown zero-duration policy")
        require(t["overrun_policy"] in {"error", "clip"}, "Unknown overrun policy")
        self.trace_start, self.trace_end = timestamp(t["coverage_start_utc"]), timestamp(t["coverage_end_utc"])
        require(self.trace_start < self.trace_end, "Empty trace coverage")

        if not queue_only:
            w = c["workload"]
            keys(w, {"name", "optimizer_updates", "global_batch", "gpus_per_node", "profiles", "provenance", "profile_status", "sequence_length", "precision", "software", "correctness_artifact"}, location="workload")
            for key in ("name", "provenance", "precision", "software", "correctness_artifact"):
                nonempty(w[key], f"workload.{key}")
            integer(w["sequence_length"], "sequence_length")
            require(w["profile_status"] in {"synthetic", "measured", "published", "assumed"}, "Unknown profile_status")
            for p in w["profiles"].values():
                keys(p, {"updates_per_hour", "initialization_seconds", "restart_seconds", "checkpoint_seconds", "microbatch", "accumulation"}, location="workload profile")
            self.workload = Workload(w, cl["allowed_nodes"], cl["max_request_seconds"], cl["walltime_resolution_seconds"])
            require(not (c["purpose"] == "research" and w["profile_status"] not in {"measured", "published"}),
                    "Research runs require measured/published profiles with provenance")

            ci = c["ci"]
            keys(ci, {"path", "sha256", "source", "region", "unit", "type", "availability_rule", "queue_to_ci_offset_seconds", "alignment_description"}, location="ci")
            for key in ("source", "region", "availability_rule", "alignment_description"):
                nonempty(ci[key], f"ci.{key}")
            require(ci["type"] == "average_operational", "CI type must be average_operational")
            require(isinstance(ci["queue_to_ci_offset_seconds"], int), "CI calendar offset must be integer seconds")

            p = c["power"]
            keys(p, {"reference_kw", "workload_coefficient", "rho_interval", "provenance", "phase_assumption"}, location="power")
            number(p["reference_kw"], "reference_kw", strict=True)
            if p["workload_coefficient"] is not None:
                number(p["workload_coefficient"], "workload_coefficient", strict=True)
            require(isinstance(p["rho_interval"], list) and len(p["rho_interval"]) == 2, "rho_interval needs two endpoints")
            low, high = [number(v, "rho") for v in p["rho_interval"]]
            require(low < high <= 1, "Require 0 <= rho_low < rho_high <= 1")
            nonempty(p["provenance"], "power.provenance")
            require(p["phase_assumption"] == "same_mean_power_at_given_scale", "Unsupported phase-power assumption")

        e = c["execution"]
        keys(e, {"warmup_seconds", "history_sample_seconds", "history_lags_seconds", "max_episode_seconds", "target_failure_model"}, optional={"initial_state_mode"}, location="execution")
        mode = e.get("initial_state_mode", "observed")
        require(mode in {"observed", "empty_warmup"}, "Unknown initial state mode")
        require(t.get("role", "historical") != "workload_template" or mode == "empty_warmup",
                "Workload templates must rebuild state; historical admissions cannot initialize a changed cluster")
        integer(e["warmup_seconds"], "warmup_seconds", 0)
        integer(e["history_sample_seconds"], "history_sample_seconds")
        require(e["target_failure_model"] == "no_failures", "P1 does not model target hardware failures or automatic retries")
        require(e["history_lags_seconds"] and e["history_lags_seconds"][0] == 0, "History lags must include current state first")
        require(e["history_lags_seconds"] == sorted(set(e["history_lags_seconds"])), "History lags must be sorted and unique")
        for lag in e["history_lags_seconds"]:
            integer(lag, "history lag", 0)
        if e["max_episode_seconds"] is not None:
            integer(e["max_episode_seconds"], "max_episode_seconds")

        keys(c["splits"], {"train", "validation", "test"}, location="splits")
        self.splits = {}
        for name, interval in c["splits"].items():
            require(isinstance(interval, list) and len(interval) == 2, f"{name}: expected [start,end)")
            start, end = map(timestamp, interval)
            require(self.trace_start <= start < end <= self.trace_end, f"{name}: split outside trace coverage")
            self.splits[name] = (start, end)
        require(self.splits["train"][1] <= self.splits["validation"][0] and
                self.splits["validation"][1] <= self.splits["test"][0], "Splits must be chronological and disjoint")
        h = c["holdout_audit"]
        keys(h, {"test_is_untouched", "notes"}, optional={"evaluation_design"}, location="holdout_audit")
        require(isinstance(h["test_is_untouched"], bool), "test_is_untouched must be a boolean")
        nonempty(h["notes"], "holdout_audit.notes")
        design = h.get("evaluation_design", "prospective_temporal")
        require(design in {"prospective_temporal", "retrospective_temporal"}, "Unknown evaluation design")
        if c["purpose"] == "research" and design == "prospective_temporal":
            require(h["test_is_untouched"], "Prospective research requires an audited untouched test interval")

        if not queue_only:
            keys(c["cohort"], {"path", "sha256", "provenance"}, location="cohort")
            nonempty(c["cohort"]["provenance"], "cohort.provenance")
        asset_names = ("trace",) if queue_only else ("trace", "ci", "cohort")
        self.assets = {name: asset(root, c[name], name) for name in asset_names}
        self.jobs, self.trace_report = load_trace(self.assets["trace"], t, cl["nodes"])
        self.episodes = ()
        if not queue_only:
            self.ci = CarbonSeries.load(self.assets["ci"], c["ci"])
            self.episodes = self._cohort()
            if mode == "observed":
                self._check_initial_capacity()
        self.manifest = {"schema_version": 1, "purpose": c["purpose"], "panel": c["panel"],
                         "validated_scope": "queue_only" if queue_only else "full_experiment",
                         "config_sha256": digest(self.path), "trace_audit": self.trace_report,
                         "asset_sha256": {k: digest(v) for k, v in self.assets.items()},
                         "resolved_config": c, "cohort_size": len(self.episodes),
                         "scheduler_model": sched["model"],
                         "initial_state_rule": ("observed_running_and_pending_at_replay_start_then_continuous_replay" if mode == "observed" else "empty_at_trace_coverage_start_then_continuous_replay_no_historical_admissions"),
                         "target_failure_model": e["target_failure_model"]}

    def _check_initial_capacity(self):
        jobs = sorted(self.jobs, key=lambda j: j.observed_start)
        index, active_nodes, ends = 0, 0, []
        begins = sorted({e.arrival - duration(self.raw["execution"]["warmup_seconds"]) for e in self.episodes})
        for begin in begins:
            while index < len(jobs) and jobs[index].observed_start < begin:
                job = jobs[index]
                heapq.heappush(ends, (job.observed_end, job.job_id, job.nodes))
                active_nodes += job.nodes
                index += 1
            while ends and ends[0][0] <= begin:
                _, _, nodes = heapq.heappop(ends)
                active_nodes -= nodes
            require(active_nodes <= self.raw["cluster"]["nodes"],
                    f"Initial running jobs at {iso(begin)} require {active_nodes} nodes; audit partition/capacity")

    def _cohort(self):
        episodes, seen = [], set()
        with self.assets["cohort"].open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            require(set(reader.fieldnames or ()) == {"episode_id", "arrival_utc", "split", "budget_hours"}, "Invalid cohort columns")
            for line, row in enumerate(reader, 2):
                try:
                    require(None not in row and None not in row.values(), "Malformed cohort row")
                    name = row["episode_id"]
                    require(name and name not in seen, "Duplicate/empty episode ID")
                    seen.add(name)
                    require(row["split"] in self.splits, "Unknown cohort split")
                    arrival = timestamp(row["arrival_utc"])
                    start, end = self.splits[row["split"]]
                    require(start <= arrival < end, "Arrival outside its declared split")
                    require(arrival - duration(self.raw["execution"]["warmup_seconds"]) >= self.trace_start,
                            "Insufficient trace history for requested warmup")
                    # Preflight all potential observed execution windows, including gaps.
                    self.ci.integral(arrival, end)
                    episodes.append(Episode(name, arrival, row["split"], float(number(row["budget_hours"], "budget_hours", strict=True))))
                except (ValueError, KeyError) as exc:
                    raise ContractError(f"Cohort line {line}: {exc}") from exc
        require(episodes, "Empty cohort")
        return tuple(sorted(episodes, key=lambda e: (e.arrival, e.episode_id)))
