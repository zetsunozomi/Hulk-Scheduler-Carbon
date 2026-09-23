#!/usr/bin/env bash
# EDIT HERE when moving clusters. bash ignores these directives; sbatch reads them.
#SBATCH --job-name=carbon-run
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sf850@scarletmail.rutgers.edu
# Uncomment needed site options below (change ##SBATCH to #SBATCH).
##SBATCH --account=YOUR_ACCOUNT
##SBATCH --partition=YOUR_PARTITION
##SBATCH --qos=YOUR_QOS
##SBATCH --constraint=YOUR_CONSTRAINT

set -euo pipefail

# EDIT HERE: Python on the executing cluster; an existing export overrides this.
export CARBON_PYTHON="${CARBON_PYTHON:-/pscratch/sd/s/syfan/conda/envs/carbon/bin/python}"
# Add any required module load commands here, before running Python.
if [[ $# -lt 2 ]]; then
  echo 'Usage: bash scripts/cluster_run.sh CONFIG OUTPUT [run-fixed arguments...]' >&2
  exit 2
fi
# sbatch executes a spool copy, so its script directory is not the repository.
# Submit from the repository root; direct bash also works by absolute script path.
if [[ -d "$PWD/src/carbon" ]]; then
  repo_root="$PWD"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "$SLURM_SUBMIT_DIR/src/carbon" ]]; then
  repo_root="$SLURM_SUBMIT_DIR"
else
  repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
fi
[[ -d "$repo_root/src/carbon" ]] || { echo 'Cannot locate repository; submit from its root.' >&2; exit 2; }
cd "$repo_root"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
python_bin="$CARBON_PYTHON"
if ! command -v "$python_bin" >/dev/null 2>&1; then
  echo "Python is not executable or not found: $python_bin" >&2
  echo 'Set CARBON_PYTHON to the interpreter you intend to use.' >&2
  exit 2
fi
"$python_bin" -B - <<'PYTHON_CHECK'
import sys
if sys.version_info < (3, 10):
    raise SystemExit("P0/P1 requires Python 3.10+; selected: " + sys.version)
print("Using Python: " + sys.executable, flush=True)
print("Python version: " + sys.version.split()[0], flush=True)
PYTHON_CHECK
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
