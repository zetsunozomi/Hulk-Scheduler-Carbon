#!/usr/bin/env bash
# EDIT HERE when moving clusters. bash ignores these directives; sbatch reads them.
#SBATCH --job-name=carbon-stress
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
[[ $# -ge 6 ]] || { echo 'Usage: cluster_stress.sh CONFIG OUTPUT WAIT_MODEL CHECKPOINT VARIANT BETA [options]' >&2; exit 2; }
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
config="$1"; output="$2"; predictor="$3"; checkpoint="$4"; variant="$5"; beta="$6"; shift 6
exec "$python_bin" -B -m carbon run-stress --config "$config" --output "$output" --predictor "$predictor" --checkpoint "$checkpoint" --variant "$variant" --budget-multiplier "$beta" "$@"
