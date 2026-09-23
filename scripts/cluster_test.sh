#!/usr/bin/env bash
# EDIT HERE when moving clusters. bash ignores these directives; sbatch reads them.
#SBATCH --job-name=carbon-test
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
if [[ $# -lt 5 ]]; then
  echo 'Usage: bash scripts/cluster_test.sh CONFIG OUTPUT WAIT_MODEL SELECTION CHECKPOINT_DIR [CHECKPOINT_DIR ...]' >&2
  exit 2
fi
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo 'This wrapper executes the complete frozen test plan; do not run it as an array.' >&2
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
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
python_bin="$CARBON_PYTHON"
config_path="$1"
output_path="$2"
predictor_path="$3"
selection_path="$4"
shift 4
"$python_bin" -B -m carbon run-test --config "$config_path" --output "$output_path" \
  --predictor "$predictor_path" --selection "$selection_path" --checkpoints "$@"
"$python_bin" -B -m carbon report-test --run "$output_path" --output "$output_path/report"
exec "$python_bin" -B -m carbon export-results --run "$output_path" --report "$output_path/report" \
  --output "$output_path/tables" --tables-only
