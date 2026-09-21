"""Inventory existing files only; no replay, UTC assumption, or coverage attestation.

Usage: python audit_trace_metadata.py /path/to/carbon-latest > trace_inventory.json
"""

import hashlib
import json
import math
import sys
from collections import Counter
from datetime import datetime
from itertools import combinations
from pathlib import Path


def inspect(root, relative):
    path = root / relative
    rows, bad_lines = [], []
    with path.open() as handle:
        header = handle.readline().split()
        for line, text in enumerate(handle, 2):
            fields = text.split()
            if not fields or all(set(f) <= {"-"} for f in fields):
                continue
            if len(fields) != len(header):
                bad_lines.append(line)
                continue
            rows.append(dict(zip(header, fields)))
    stamps = {key: [] for key in ("Submit", "Start", "End")}
    counts, months, ids, fingerprints = Counter(), Counter(), Counter(), set()
    node_counts = Counter()
    maximum_nodes = 0
    for row in rows:
        ids[row["JobID"]] += 1
        fingerprints.add(tuple(row.get(k) for k in header))
        try:
            times = {key: datetime.fromisoformat(row[key]) for key in stamps}
            if any(t.tzinfo is not None for t in times.values()):
                raise ValueError("Unexpected timezone-aware source")
            for key, value in times.items():
                stamps[key].append(value)
            months[times["Submit"].strftime("%Y-%m")] += 1
            node_counts[int(row["NNodes"])] += 1
            maximum_nodes = max(maximum_nodes, int(row["NNodes"]))
            counts["nonpositive_node_requests"] += int(int(row["NNodes"]) <= 0)
            counts["nonpositive_time_limits"] += int(int(row["TimelimitR"]) <= 0)
            submit, start, end = (times[k] for k in ("Submit", "Start", "End"))
            counts["invalid_time_order"] += int(not submit <= start <= end)
            counts["zero_duration"] += int(start == end)
            counts["runtime_over_requested_limit"] += int((end-start).total_seconds() > 60*int(row["TimelimitR"]))
        except ValueError:
            counts["unparseable_rows"] += 1
    result = {
        "path": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size, "parsed_rows": len(rows),
        "malformed_line_numbers": bad_lines,
        "duplicate_job_ids": sum(n-1 for n in ids.values()),
        "maximum_single_job_nodes": maximum_nodes,
        "rows_by_requested_nodes": dict(sorted(node_counts.items())),
        "naive_timestamp_ranges": {
            k: [min(v).isoformat(), max(v).isoformat()] if v else None
            for k, v in stamps.items()
        },
        "submitted_rows_by_month": dict(sorted(months.items())),
        "raw_row_flags_not_yet_cleaned": dict(counts),
    }
    return result, set(ids), fingerprints


def main():
    root = Path(sys.argv[1]).resolve()
    files = [
        "data/filtered/filtered-frontera-rtx.log",
        "data/filtered/filtered-longhorn-v100.log",
        "data/filtered/filtered-ls6.log",
        "data/filtered/filtered-ls6-new.log",
        "data/new/filtered_iw_log.log",
        "data/new/filtered_iw_validate.log",
    ]
    records = {path: inspect(root, path) for path in files}
    pairs = []
    for left, right in combinations(files, 2):
        shared_ids = records[left][1] & records[right][1]
        shared_rows = records[left][2] & records[right][2]
        pairs.append({"left": left, "right": right,
                      "shared_job_ids": len(shared_ids),
                      "identical_full_rows": len(shared_rows)})
    config_path = "src/model/policy-gradient-moe/sim_validation.json"
    old = json.loads((root / config_path).read_text())
    result = {
        "schema_version": 1, "kind": "read_only_source_metadata_inventory",
        "experiment_results": False,
        "limitations": [
            "Raw timestamp ranges are naive local strings; timezone is unconfirmed.",
            "First/last job timestamps do not establish complete collection coverage.",
            "Maximum single-job size does not establish queue capacity or membership.",
            "Shared IDs alone do not establish common source; IDs can recur across clusters.",
            "Metadata inspection does not establish an untouched test period.",
            "No scheduler replay, workload simulation, or policy evaluation was performed.",
        ],
        "files": [records[path][0] for path in files], "pairwise_overlap": pairs,
        "existing_validation_config": {
            "path": config_path,
            "sha256": hashlib.sha256((root/config_path).read_bytes()).hexdigest(),
            "job_log": old["job_log"],
            "naive_interval": [old["log_start_time"], old["log_end_time"]],
            "actual_access_history_confirmed": False,
        },
        "planned_bound_design_check": {
            "is_empirical_result": False,
            "formula_at_zero_misses": "1 - (alpha / comparisons) ** (1 / effective_blocks)",
            "alpha": 0.05, "epsilon": 0.05,
            "minimum_equal_independent_blocks": {
                str(n): math.ceil(math.log(0.05 / n) / math.log(0.95))
                for n in (1, 4, 16, 32, 64)
            },
            "limitations": "Design calculation for the implemented conservative bound; does not establish block independence or select a final block length.",
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
