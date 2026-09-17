#!/usr/bin/env bash
# EDIT HERE when moving clusters. bash ignores these directives; sbatch reads them.
#SBATCH --job-name=carbon-e1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
# Uncomment needed site options below (change ##SBATCH to #SBATCH).
##SBATCH --account=YOUR_ACCOUNT
##SBATCH --partition=YOUR_PARTITION
##SBATCH --qos=YOUR_QOS
##SBATCH --constraint=YOUR_CONSTRAINT

set -euo pipefail

# EDIT HERE: Python on the executing cluster; an existing export overrides this.
export CARBON_PYTHON="${CARBON_PYTHON:-/pscratch/sd/s/syfan/conda/envs/carbon/bin/python}"
# Add any required module load commands here, before running Python.
if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo 'Usage: bash scripts/cluster_e1.sh CONFIG OUTPUT [PROBE_INTERVAL_SECONDS]' >&2
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
config_path="$1"
output_path="$2"
bash scripts/cluster_waits.sh "$config_path" "$output_path" "${3:-21600}"
trace_role="$("$python_bin" -B -c 'import json,sys; print(json.load(open(sys.argv[1]))["trace"].get("role","historical"))' "$config_path")"
if [[ "$trace_role" == historical ]]; then
  "$python_bin" -B -m carbon evaluate-replay --config "$config_path" --output "$output_path/replay-validation" --split validation
else
  echo 'Constructed workload: historical admission fidelity is inapplicable; predictor diagnostics use simulator probes.'
fi
"$python_bin" -B -m carbon audit-dependence --probes "$output_path/train-probes" "$output_path/validation-probes" \
  --output "$output_path/dependence-queue"
echo "E1 queue stage complete: $output_path. Scenario coverage, episode duration and full-policy results require review; no Qwen profiling is required."
