"""Training references and validation-only fixed policy mixture."""

from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import fmean

from .common import ContractError, digest, json_text, load_json, require


def check_references(references, manifest):
    require(references["kind"] == "training_references_v1" and references["split"] == "train", "Invalid training references")
    require(references["panel"] == manifest["panel"], "References belong to another panel")
    for section in ("cluster", "workload", "power", "splits", "execution", "holdout_audit"):
        require(references["resolved_config"][section] == manifest["resolved_config"][section], f"Reference {section} differs from run")
    for section in ("ci", "trace", "cohort"):
        before = {k:v for k,v in references["resolved_config"][section].items() if k not in {"path", "provenance"}}
        after = {k:v for k,v in manifest["resolved_config"][section].items() if k not in {"path", "provenance"}}
        require(before == after, f"Reference {section} interpretation differs from run")
        require(references["asset_sha256"][section] == manifest["asset_sha256"][section], f"Reference {section} differs")


def fixed_records(path, split, nodes):
    by_method = defaultdict(dict)
    with Path(path).open(encoding="utf-8") as stream:
        for row in map(json.loads, stream):
            if row["split"] != split or row["method"] not in {f"Fixed-{n}" for n in nodes}:
                continue
            require(row["final_status"] == "completed" and not row["censor_flag"], "Incomplete fixed cohort; references/mixture require full outcomes")
            require(row["episode_id"] not in by_method[row["method"]], "Duplicate fixed episode outcome")
            by_method[row["method"]][row["episode_id"]] = row
    require(set(by_method) == {f"Fixed-{n}" for n in nodes}, "Missing fixed baseline(s)")
    ids = set(by_method[f"Fixed-{nodes[0]}"])
    require(ids and all(set(group) == ids for group in by_method.values()), "Fixed outcomes are not paired")
    rows = [row for group in by_method.values() for row in group.values()]
    require(len({r["panel"] for r in rows}) == 1 and len({r.get("carbon_unit", "gCO2e") for r in rows}) == 1,
            "Mixed panels or emission units")
    return by_method


def make_references(fixed_directory, output):
    source, output = Path(fixed_directory), Path(output)
    manifest = load_json(source/"manifest.json")
    require(manifest["status"] == "complete", "Incomplete fixed run")
    require(not output.exists(), f"Output already exists: {output}")
    nodes = manifest["resolved_config"]["cluster"]["allowed_nodes"]
    rows = fixed_records(source/"episodes.jsonl", "train", nodes)
    means = {method: fmean(row["tat_hours"] for row in group.values()) for method, group in rows.items()}
    endpoints = manifest["resolved_config"]["power"]["rho_interval"]
    costs = {str(r): fmean(row["carbon_g_per_kappa"][str(r)] for row in rows['Fixed-4'].values()) for r in endpoints}
    require(min(means.values()) > 0 and min(costs.values()) > 0, "Nonpositive training reference")
    result = {"kind": "training_references_v1", "time_reference_hours": min(means.values()),
              "carbon_reference_g_per_kappa": costs, "fixed_mean_tat_hours": means,
              "split": "train", "panel": manifest["panel"], "purpose": manifest["purpose"],
              "source_episodes_sha256": digest(source/"episodes.jsonl"),
              "asset_sha256": manifest["asset_sha256"], "resolved_config": manifest["resolved_config"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json_text(result), encoding="utf-8")
    return result


def fit_fixed_mix(fixed_directory, references, budget_hours, epsilon=.05):
    try:
        from scipy.optimize import linprog
    except ImportError:
        raise ContractError('Fixed-Mix fitting needs scipy: "$CARBON_PYTHON" -m pip install -r requirements-p2.txt') from None
    require(0 <= epsilon <= 1 and math.isfinite(budget_hours) and budget_hours > 0, "Invalid fixed-mixture budget/tolerance")
    source = Path(fixed_directory)
    manifest = load_json(source/"manifest.json")
    require(manifest["status"] == "complete", "Incomplete validation fixed run")
    check_references(references, manifest)
    nodes = manifest["resolved_config"]["cluster"]["allowed_nodes"]
    groups = fixed_records(source/"episodes.jsonl", "validation", nodes)
    endpoints = list(references["carbon_reference_g_per_kappa"])
    require(all(math.isfinite(v) and v > 0 for v in references["carbon_reference_g_per_kappa"].values()), "Invalid carbon references")
    costs = [[fmean(r["carbon_g_per_kappa"][rho] for r in groups[f"Fixed-{n}"].values()) /
              references["carbon_reference_g_per_kappa"][rho] for n in nodes] for rho in endpoints]
    misses = [fmean(r["tat_hours"] > budget_hours for r in groups[f"Fixed-{n}"].values()) for n in nodes]
    result = linprog([0.0]*len(nodes)+[1.0], A_ub=[c+[-1.0] for c in costs]+[misses+[0.0]],
                     b_ub=[0.0]*len(costs)+[epsilon], A_eq=[[1.0]*len(nodes)+[0.0]], b_eq=[1.0],
                     bounds=[(0, 1)]*len(nodes)+[(0, None)], method="highs")
    require(result.success or result.status == 2, f"Fixed-Mix solver failed: {result.message}")
    weights = {str(n):float(result.x[i]) for i,n in enumerate(nodes)} if result.success else None
    return {"kind": "fixed_mix_v1", "fit_split": "validation", "budget_hours": budget_hours, "epsilon": epsilon,
            "panel": manifest["panel"], "purpose": manifest["purpose"],
            "weights": weights, "empirical_feasible": bool(result.success),
            "estimated_objective": float(result.fun) if result.success else None,
            "validation_miss_rates": dict(zip(map(str,nodes),misses)),
            "source_episodes_sha256": digest(source/"episodes.jsonl"),
            "reference_source_episodes_sha256": references["source_episodes_sha256"],
            "asset_sha256": manifest["asset_sha256"],
            "statistical_feasibility": "pending paired calendar-block validation; LP feasibility alone is not sufficient"}


def weighted_quantile(values_weights, probability):
    require(values_weights and 0 < probability <= 1, "Invalid mixture quantile")
    total = sum(w for _,w in values_weights)
    require(total > 0 and all(w >= 0 for _,w in values_weights), "Invalid mixture weights")
    cumulative = 0.0
    for value, weight in sorted(values_weights):
        cumulative += weight
        if cumulative >= probability*total - 1e-12:
            return value
    return max(v for v,_ in values_weights)
