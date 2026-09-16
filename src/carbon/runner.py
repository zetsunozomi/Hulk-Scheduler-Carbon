"""Paired fixed-scale runner; all outputs are auditable, append-only JSONL."""

import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

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
