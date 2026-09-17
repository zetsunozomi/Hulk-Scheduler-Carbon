"""Paired fixed-scale runner; all outputs are auditable, append-only JSONL."""

import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from dataclasses import replace

from . import __version__
from .common import ContractError, digest, json_text, require
from .environment import Environment, initial_replay


def provenance(root):
    def git(*args):
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    package = Path(__file__).parent
    return {"version": __version__, "python": sys.version, "platform": platform.platform(),
            "git_commit": git("rev-parse", "HEAD"), "git_status": git("status", "--porcelain"),
            "source_sha256": {p.name: digest(p) for p in sorted(package.glob("*.py"))}}


def write_record(handle, value):
    handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
    handle.flush()


def write_manifest(path, value):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json_text(value), encoding="utf-8")
    temporary.replace(path)


def run_fixed(bundle, output, nodes, split=None, shard_index=0, shard_count=1):
    require(nodes and len(nodes) == len(set(nodes)), "Choose unique fixed scales")
    require(set(nodes) <= bundle.workload.profiles.keys(), "Requested fixed scale is not in allowed_nodes")
    require(shard_count > 0 and 0 <= shard_index < shard_count, "Invalid shard index/count")
    episodes = [e for e in bundle.episodes if split is None or e.split == split]
    episodes = [e for e in episodes if int.from_bytes(hashlib.sha256(e.episode_id.encode()).digest()[:8], "big") % shard_count == shard_index]
    require(episodes, "Selected cohort/shard is empty")
    output = Path(output)
    require(not output.exists(), f"Output already exists: {output}; choose a new run directory")
    output.mkdir(parents=True)
    manifest = {**bundle.manifest, "software": provenance(bundle.root), "command": sys.argv,
                "fixed_nodes": nodes, "selected_episode_ids": [e.episode_id for e in episodes],
                "selected_splits": sorted({e.split for e in episodes}),
                "shard_index": shard_index, "shard_count": shard_count,
                "status": "running", "completed_episode_methods": 0}
    manifest_path = output / "manifest.json"
    write_manifest(manifest_path, manifest)
    started = time.monotonic()
    current = None
    try:
        with (output / "chunks.jsonl").open("x", encoding="utf-8") as chunk_file, \
             (output / "episodes.jsonl").open("x", encoding="utf-8") as episode_file:
            for episode in episodes:
                current = {"episode_id": episode.episode_id, "method": "initial_state"}
                base = initial_replay(bundle, episode)
                for n in nodes:
                    method = f"Fixed-{n}"
                    current = {"episode_id": episode.episode_id, "method": method}
                    env = Environment(bundle, episode, base, method)
                    while env.status == "running":
                        write_record(chunk_file, env.step(n))
                    write_record(episode_file, env.summary())
                    manifest["completed_episode_methods"] += 1
                    print(f"{episode.episode_id} {method}: {env.status}, chunks={len(env.chunks)}", flush=True)
        manifest["status"] = "complete"
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["failure"] = {**(current or {}), "error_type": type(exc).__name__, "message": str(exc)}
        (output / "failure.json").write_text(json_text(manifest["failure"]), encoding="utf-8")
        raise
    finally:
        manifest["elapsed_seconds"] = time.monotonic() - started
        write_manifest(manifest_path, manifest)
    return manifest


def run_planners(bundle, output, predictor_path, references_path, split="validation", paths=256,
                 miss_tolerance=.05, seed=11, methods=("Rollout-MPC", "Plan-once", "Queue-blind-MPC"),
                 budget_multiplier=None):
    from .common import load_json, timestamp
    from .baselines import check_references
    from .forecast import CausalForecast
    from .planning import PlanningPolicy, RolloutPlanner
    from .waits import WaitPredictor
    references = load_json(references_path)
    check_references(references, bundle.manifest)
    predictor = WaitPredictor.load(predictor_path)
    predictor.check_inputs(bundle.manifest)
    require(predictor.artifact["trace_sha256"] == bundle.manifest["asset_sha256"]["trace"], "Wait model trace differs")
    require(predictor.artifact["cluster"] == bundle.raw["cluster"], "Wait model cluster differs")
    require(predictor.features.lags == bundle.raw["execution"]["history_lags_seconds"], "Wait model history lags differ")
    require(timestamp(predictor.artifact["train_label_boundary_utc"]) <= bundle.splits["train"][1], "Wait model used labels beyond training split")
    require(bundle.raw["purpose"] != "research" or predictor.artifact["purpose"] != "synthetic", "Synthetic predictor cannot drive research")
    require(methods and len(methods) == len(set(methods)) and
            set(methods) <= {"Rollout-MPC", "Plan-once", "Queue-blind-MPC"}, "Invalid planning methods")
    episodes = [e for e in bundle.episodes if e.split == split]
    require(episodes, "Selected planner cohort is empty")
    if budget_multiplier is not None:
        require(budget_multiplier > 0, "Budget multiplier must be positive")
        episodes = [replace(e, budget_hours=budget_multiplier*references["time_reference_hours"]) for e in episodes]
    output = Path(output)
    require(not output.exists(), f"Output already exists: {output}")
    output.mkdir(parents=True)
    forecast = CausalForecast(bundle.ci, *bundle.splits["train"], calendar_timezone=bundle.raw["trace"]["timezone"])
    manifest = {**bundle.manifest, "software": provenance(bundle.root), "command": sys.argv,
                "methods": list(methods), "seed": seed, "planning_paths": paths,
                "planner_internal_miss_tolerance": miss_tolerance, "budget_multiplier": budget_multiplier,
                "selected_episode_ids": [e.episode_id for e in episodes],
                "selected_splits": sorted({e.split for e in episodes}),
                "predictor_sha256": digest(predictor_path), "references_sha256": digest(references_path),
                "status": "running", "completed_episode_methods": 0}
    write_manifest(output/"manifest.json", manifest)
    started, current = time.monotonic(), None
    try:
        with (output/"chunks.jsonl").open("x", encoding="utf-8") as chunk_file, \
             (output/"episodes.jsonl").open("x", encoding="utf-8") as episode_file:
            for episode in episodes:
                current = {"episode_id": episode.episode_id, "method": "initial_state"}
                base = initial_replay(bundle, episode)
                paired_seed = seed + int.from_bytes(hashlib.sha256(episode.episode_id.encode()).digest()[:4], "big")
                for method in methods:
                    current["method"] = method
                    planner = RolloutPlanner(bundle.workload, bundle.raw["power"], predictor, references,
                                             paths, miss_tolerance, queue_blind=method == "Queue-blind-MPC")
                    policy = PlanningPolicy(planner, forecast, "plan-once" if method == "Plan-once" else "mpc", paired_seed)
                    env = Environment(bundle, episode, base, method, seed)
                    while env.status == "running":
                        action, metadata = policy.choose(env.observe())
                        metadata["planner_fallback_reason"] = metadata.pop("fallback_reason")
                        chunk = env.step(action)
                        policy.record_execution(chunk)
                        write_record(chunk_file, {**chunk, **metadata})
                    summary = env.summary()
                    write_record(episode_file, summary)
                    manifest["completed_episode_methods"] += 1
                    print(f"{episode.episode_id} {method}: {summary['final_status']}, chunks={summary['chunk_count']}", flush=True)
        manifest["status"] = "complete"
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["failure"] = {**(current or {}), "error_type": type(exc).__name__, "message": str(exc)}
        (output/"failure.json").write_text(json_text(manifest["failure"]), encoding="utf-8")
        raise
    finally:
        manifest["elapsed_seconds"] = time.monotonic()-started
        write_manifest(output/"manifest.json", manifest)
    return manifest
