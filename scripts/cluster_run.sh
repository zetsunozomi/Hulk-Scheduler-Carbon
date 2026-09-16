#!/usr/bin/env bash
#SBATCH --job-name=carbon-p0p1
#SBATCH --cpus-per-task=1
# Supply account/partition/time/memory via sbatch arguments appropriate to your site.
set -euo pipefail
if [[ $# -lt 2 ]]; then
  echo 'Usage: bash scripts/cluster_run.sh CONFIG OUTPUT [run-fixed arguments...]' >&2
  exit 2
fi
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
python_bin="${CARBON_PYTHON:-python3}"
config_path="$1"
output_path="$2"
shift 2
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  : "${CARBON_SHARDS:?Set CARBON_SHARDS to the count of zero-based array tasks}"
  output_path="$output_path/shard-$SLURM_ARRAY_TASK_ID"
  set -- "$@" --shard-index "$SLURM_ARRAY_TASK_ID" --shard-count "$CARBON_SHARDS"
fi
"$python_bin" -m carbon validate --config "$config_path"
exec "$python_bin" -m carbon run-fixed --config "$config_path" --output "$output_path" "$@"
