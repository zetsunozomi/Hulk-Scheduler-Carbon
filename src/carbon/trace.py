"""Immutable background arrivals with explicit cleaning and timezone rules."""

from collections import Counter
import csv
from dataclasses import dataclass
from pathlib import Path

from .common import ContractError, digest, duration, integer, iso, require, timestamp


@dataclass(frozen=True)
class TraceJob:
    job_id: str
    nodes: int
    submit: object
    observed_start: object
    observed_end: object
    requested: object

    @property
    def runtime(self):
        return self.observed_end - self.observed_start


def read_rows(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        if Path(path).suffix.lower() == ".csv":
            reader = csv.DictReader(handle)
            for line, row in enumerate(reader, 2):
                require(None not in row and None not in row.values(), f"{path}:{line}: malformed CSV row")
                yield line, row
        else:
            header = handle.readline().split()
            require(header, f"{path}: empty trace")
            for line, raw in enumerate(handle, 2):
                values = raw.split()
                if not values or all(set(v) <= {"-"} for v in values):
                    continue
                require(len(values) == len(header), f"{path}:{line}: expected {len(header)} fields, got {len(values)}")
                yield line, dict(zip(header, values))


def load_trace(path, config, capacity=None):
    jobs, counts, seen = [], Counter(), set()
    required = {"JobID", "NNodes", "Submit", "Start", "End", "TimelimitR"}
    for line, row in read_rows(path):
        try:
            require(required <= row.keys(), f"Required trace columns: {sorted(required)}")
            job_id = row["JobID"]
            require(job_id and job_id not in seen, f"Duplicate/empty job ID {job_id!r}")
            seen.add(job_id)
            counts["input_rows"] += 1
            nodes = integer(row["NNodes"], "NNodes")
            parse = lambda key: timestamp(row[key], config["timezone"], config.get("dst_fold"))
            submit, start, end = parse("Submit"), parse("Start"), parse("End")
            requested = duration(integer(row["TimelimitR"], "TimelimitR") * 60)
            require(submit <= start <= end, "Require Submit <= Start <= End")
            if end == start:
                counts["zero_duration_rows"] += 1
                require(config["zero_duration_policy"] == "drop", "Zero duration requires explicit drop policy")
                continue
            if end - start > requested:
                counts["duration_over_request_rows"] += 1
                require(config["overrun_policy"] == "clip", "Duration exceeds requested walltime")
                end = start + requested
            original_nodes = nodes
            multiplier = integer(config.get("node_multiplier", 1), "node_multiplier")
            require(config.get("role", "historical") == "workload_template" or
                    (multiplier == 1 and config.get("oversize_policy", "error") == "error"),
                    "Resource transformation requires workload_template role")
            nodes *= multiplier
            if capacity is not None and nodes > capacity and config.get("oversize_policy", "error") == "cap":
                counts["width_capped_rows"] += 1
                nodes = capacity
            counts["original_node_seconds"] += original_nodes * int((end-start).total_seconds())
            counts["scenario_node_seconds"] += nodes * int((end-start).total_seconds())
            if capacity is not None:
                require(nodes <= capacity, f"Job requests {nodes} nodes, capacity is {capacity}; audit partition membership")
            jobs.append(TraceJob("background:" + job_id, nodes, submit, start, end, requested))
        except (ValueError, KeyError) as exc:
            raise ContractError(f"{path}:{line}: {exc}") from exc
    require(jobs, "No retained background jobs")
    jobs.sort(key=lambda j: (j.submit, j.job_id))
    counts["retained_rows"] = len(jobs)
    report = {"sha256": digest(path), "counts": dict(counts),
              "submit_start_utc": iso(jobs[0].submit), "submit_end_utc": iso(jobs[-1].submit),
              "observed_end_utc": iso(max(j.observed_end for j in jobs)),
              "max_requested_nodes": max(j.nodes for j in jobs),
              "timezone": config["timezone"], "cleaning": {k: config[k] for k in ("zero_duration_policy", "overrun_policy")},
              "role": config.get("role", "historical"), "node_multiplier": config.get("node_multiplier", 1),
              "oversize_policy": config.get("oversize_policy", "error")}
    return tuple(jobs), report
