"""Counterfactual wait labels from independent requests at one public snapshot."""

from datetime import timedelta
from pathlib import Path
import time

from .common import digest, integer, iso, json_text, require, timestamp
from .features import QueueFeatures
from .replay import Replay, Request
from .probe_resume import engine_contract, prepare_append, probe_lock, recover_prefix
from .runner import provenance, write_manifest, write_record


def collect_probes(bundle, output, split="train", interval_seconds=21600,
                   request_seconds=None, start=None, stop=None, resume=False):
    with probe_lock(output, resume):
        return _collect_probes(bundle, output, split, interval_seconds, request_seconds, start, stop, resume)


def _collect_probes(bundle, output, split, interval_seconds, request_seconds, start, stop, resume):
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
    metadata['probe_engine'] = engine_contract(metadata['software'])
    saved_rows = []
    previous_elapsed = 0.
    if resume and (output / 'manifest.json').exists():
        old, saved_rows, recovery = recover_prefix(output, metadata)
        if old['status'] == 'complete':
            print(f"Probe dataset already complete and verified: {output}; rows={old['rows']}", flush=True)
            return old
        backup = prepare_append(output, recovery)
        previous_elapsed = old.get('elapsed_seconds', 0.)
        metadata['software'] = old['software']
        metadata['resume_sessions'] = old.get('resume_sessions', []) + [{
            'retained_rows': len(saved_rows), 'backup': backup,
            'software': provenance(bundle.root), 'prefix_sha256': digest(output / 'probes.jsonl') if (output / 'probes.jsonl').exists() else None,
            'discarded_incomplete_tail_bytes': len(recovery['tail'])}]
        metadata['status'] = 'recovering'
        print(f"Resume: validated {len(saved_rows)} saved rows; rebuilding background replay without recomputing their probes.", flush=True)
    elif resume:
        require(not any(p.name != '.probe.lock' for p in output.iterdir()),
                'Cannot resume: nonempty probe directory has no manifest')

    def count_row(row):
        known = not row['censored']
        metadata['rows'] += 1
        metadata['labeled_rows' if known else 'censored_rows'] += 1
        key = f"{row['nodes']}:{row['requested_seconds']}"
        summary = waits_by_request[key]
        summary['rows'] += 1
        summary['labeled_rows' if known else 'censored_rows'] += 1
        if known:
            wait = row['wait_hours']
            summary['positive_wait_rows'] += int(wait > 0)
            wait_sums[key] += wait
            summary['mean_wait_hours'] = wait_sums[key] / summary['labeled_rows']
            summary['max_wait_hours'] = max(summary['max_wait_hours'] or 0., wait)

    for row in saved_rows:
        count_row(row)
    write_manifest(output / 'manifest.json', metadata)
    begun = time.monotonic()
    last_recovery_update = begun
    visited = 0
    try:
        origin = bundle.trace_start if execution.get("initial_state_mode", "observed") == "empty_warmup" else start - warmup
        base = Replay(bundle.jobs, origin, bundle.trace_end, c,
                      max(execution["history_lags_seconds"]), execution["history_sample_seconds"],
                      initialize_from_observed=execution.get("initial_state_mode", "observed") == "observed")
        with (output / "probes.jsonl").open("a" if resume else "x", encoding="utf-8") as stream:
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
                        vector = features.request(history, arrival, n, length)
                        if visited < len(saved_rows):
                            require(saved_rows[visited]['features'] == vector,
                                    f'Restored background features differ at saved row {visited + 1}')
                            visited += 1
                            continue
                        if metadata['status'] == 'recovering':
                            print(f"Resume ready: retained {len(saved_rows)} rows; next={snapshot_id}, nodes={n}, request={length}s", flush=True)
                            metadata['status'] = 'running'
                        clone = base.clone()
                        identifier = f"probe:{snapshot_id}:{n}:{length}"
                        # The probe's future runtime cannot affect its own admission.
                        clone.inject(Request(identifier, n, arrival, timedelta(seconds=length),
                                             timedelta(seconds=length), target=True))
                        allocation = clone.until(identifier, "start", split_end)
                        known = allocation is not None and allocation.start < split_end
                        row = {"snapshot_id": snapshot_id, "split": split, "arrival_utc": iso(arrival),
                               "nodes": n, "requested_seconds": length,
                               "features": vector,
                               "wait_hours": (allocation.start - arrival).total_seconds() / 3600 if known else None,
                               "label_observed_at_utc": iso(allocation.start) if known else None,
                               "label_boundary_utc": iso(split_end), "censored": not known}
                        write_record(stream, row)
                        count_row(row)
                        visited += 1
                arrival += step
                metadata['elapsed_seconds'] = previous_elapsed + time.monotonic() - begun
                metadata['elapsed_scope'] = 'recorded session seconds; a hard kill can lose time since the last snapshot manifest'
                write_manifest(output / 'manifest.json', metadata)
                if visited > len(saved_rows):
                    print(f"wait probes: {snapshot_id}; rows={metadata['rows']}, censored={metadata['censored_rows']}", flush=True)
                elif time.monotonic() - last_recovery_update >= 30:
                    print(f"Restoring background: {snapshot_id}; checked {visited}/{len(saved_rows)} saved rows", flush=True)
                    last_recovery_update = time.monotonic()
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
    except BaseException as exc:
        metadata["status"] = "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed"
        metadata["failure"] = dict(error_type=type(exc).__name__, message=str(exc))
        (output / "failure.json").write_text(json_text(metadata["failure"]), encoding="utf-8")
        raise
    finally:
        metadata["elapsed_seconds"] = previous_elapsed + time.monotonic() - begun
        write_manifest(output / "manifest.json", metadata)
    return metadata
