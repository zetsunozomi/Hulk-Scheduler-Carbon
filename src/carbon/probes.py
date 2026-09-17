"""Counterfactual wait labels from independent requests at one public snapshot."""

from datetime import timedelta
from pathlib import Path
import time

from .common import digest, integer, iso, json_text, require, timestamp
from .features import QueueFeatures
from .replay import Replay, Request
from .runner import provenance, write_manifest, write_record


def collect_probes(bundle, output, split="train", interval_seconds=21600,
                   request_seconds=None, start=None, stop=None):
    require(split in bundle.splits, "Unknown probe split")
    step = timedelta(seconds=integer(interval_seconds, "probe interval_seconds"))
    c, execution = bundle.raw["cluster"], bundle.raw["execution"]
    maximum = integer(c["max_request_seconds"], "max request")
    resolution = integer(c["walltime_resolution_seconds"], "resolution")
    lengths = sorted(set(request_seconds or [max(resolution, int(maximum * f // resolution) * resolution)
                                           for f in (.25, .5, 1)]))
    require(all(isinstance(v, int) and 0 < v <= maximum and v % resolution == 0 for v in lengths),
            "Probe request lengths must be positive walltime multiples within cap")
    warmup = timedelta(seconds=execution["warmup_seconds"])
    split_start, split_end = bundle.splits[split]
    start = timestamp(start) if start else max(split_start, bundle.trace_start + warmup)
    stop = timestamp(stop) if stop else split_end
    require(split_start <= start < stop <= split_end and start - warmup >= bundle.trace_start,
            "Probe arrival interval/warmup outside declared split/coverage")
    output = Path(output)
    require(not output.exists(), f"Output already exists: {output}")
    output.mkdir(parents=True)
    features = QueueFeatures(execution["history_lags_seconds"], bundle.raw["trace"]["timezone"])
    queue_summary = {
        "scope": "unweighted probe snapshots, not time-integrated utilization or policy performance",
        "background_submissions_in_probe_window": sum(start <= j.submit < stop for j in bundle.jobs),
        "snapshots": 0, "empty_background_snapshots": 0,
        "sampled_mean_running_fraction": None, "max_pending_jobs": 0,
    }
    running_fraction_sum = 0.0
    waits_by_request = {f"{n}:{length}": {"nodes": n, "requested_seconds": length,
                        "rows": 0, "labeled_rows": 0, "positive_wait_rows": 0,
                        "censored_rows": 0, "mean_wait_hours": None, "max_wait_hours": None}
                        for n in c["allowed_nodes"] for length in lengths}
    wait_sums = {key: 0.0 for key in waits_by_request}
    metadata = {**bundle.manifest, "software": provenance(bundle.root), "kind": "wait_probes",
                "probe_split": split, "probe_interval_seconds": interval_seconds,
                "probe_start_utc": iso(start), "probe_stop_utc": iso(stop),
                "label_boundary_utc": iso(split_end), "request_seconds": lengths,
                "feature_schema": features.metadata(), "status": "running", "rows": 0,
                "labeled_rows": 0, "censored_rows": 0,
                "queue_summary": queue_summary, "wait_summary_by_request": waits_by_request,
                "probe_semantics": "clone snapshot per request; stop at admission; no workload-speed input"}
    write_manifest(output / "manifest.json", metadata)
    begun = time.monotonic()
    try:
        origin = bundle.trace_start if execution.get("initial_state_mode", "observed") == "empty_warmup" else start - warmup
        base = Replay(bundle.jobs, origin, bundle.trace_end, c,
                      max(execution["history_lags_seconds"]), execution["history_sample_seconds"],
                      initialize_from_observed=execution.get("initial_state_mode", "observed") == "observed")
        with (output / "probes.jsonl").open("x", encoding="utf-8") as stream:
            arrival = start
            while arrival < stop:
                base.advance_to(arrival, before_dispatch=True)
                history = base.visible_history(execution["history_lags_seconds"])
                state = history[0]["state"]
                queue_summary["snapshots"] += 1
                queue_summary["empty_background_snapshots"] += int(not state["running"] and not state["pending"])
                running_fraction_sum += 1 - state["available_nodes"] / state["capacity_nodes"]
                queue_summary["sampled_mean_running_fraction"] = running_fraction_sum / queue_summary["snapshots"]
                queue_summary["max_pending_jobs"] = max(queue_summary["max_pending_jobs"], len(state["pending"]))
                snapshot_id = f"{split}:{iso(arrival)}"
                for n in c["allowed_nodes"]:
                    for length in lengths:
                        clone = base.clone()
                        identifier = f"probe:{snapshot_id}:{n}:{length}"
                        # The probe's future runtime cannot affect its own admission.
                        clone.inject(Request(identifier, n, arrival, timedelta(seconds=length),
                                             timedelta(seconds=length), target=True))
                        allocation = clone.until(identifier, "start", split_end)
                        known = allocation is not None and allocation.start < split_end
                        row = {"snapshot_id": snapshot_id, "split": split, "arrival_utc": iso(arrival),
                               "nodes": n, "requested_seconds": length,
                               "features": features.request(history, arrival, n, length),
                               "wait_hours": (allocation.start - arrival).total_seconds() / 3600 if known else None,
                               "label_observed_at_utc": iso(allocation.start) if known else None,
                               "label_boundary_utc": iso(split_end), "censored": not known}
                        write_record(stream, row)
                        metadata["rows"] += 1
                        metadata["labeled_rows" if known else "censored_rows"] += 1
                        key = f"{n}:{length}"
                        summary = waits_by_request[key]
                        summary["rows"] += 1
                        summary["labeled_rows" if known else "censored_rows"] += 1
                        if known:
                            wait = row["wait_hours"]
                            summary["positive_wait_rows"] += int(wait > 0)
                            wait_sums[key] += wait
                            summary["mean_wait_hours"] = wait_sums[key] / summary["labeled_rows"]
                            summary["max_wait_hours"] = max(summary["max_wait_hours"] or 0., wait)
                arrival += step
                print(f"wait probes: {snapshot_id}; rows={metadata['rows']}, censored={metadata['censored_rows']}", flush=True)
        metadata["probes_sha256"] = digest(output / "probes.jsonl")
        metadata["status"] = "complete"
        empty = queue_summary["empty_background_snapshots"]
        count = queue_summary["snapshots"]
        print(f"Probe queue summary: background submissions={queue_summary['background_submissions_in_probe_window']}, "
              f"empty snapshots={empty}/{count}, "
              f"sampled running fraction={queue_summary['sampled_mean_running_fraction']:.4f}", flush=True)
        if empty == count:
            print("All sampled background states were empty; this window does not check busy-queue behavior. "
                  "Keep the samples; do not infer full-trace performance from this window.", flush=True)
        for summary in waits_by_request.values():
            print("Probe wait summary: " + json_text(summary).strip().replace("\n", " "), flush=True)
    except Exception as exc:
        metadata["status"] = "failed"
        metadata["failure"] = dict(error_type=type(exc).__name__, message=str(exc))
        (output / "failure.json").write_text(json_text(metadata["failure"]), encoding="utf-8")
        raise
    finally:
        metadata["elapsed_seconds"] = time.monotonic() - begun
        write_manifest(output / "manifest.json", metadata)
    return metadata
