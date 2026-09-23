#!/usr/bin/env bash
# EDIT HERE when moving clusters. bash ignores these directives; sbatch reads them.
#SBATCH --job-name=carbon-ppo
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
  echo 'Usage: bash scripts/cluster_ppo.sh CONFIG OUTPUT WAIT_MODEL_OR_DASH REFERENCES ITERATIONS [train-ppo options...]' >&2
  exit 2
fi
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo 'Launch one declared seed per job; this wrapper does not map array indices to seeds.' >&2
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
"$python_bin" -B - <<'PYTHON_CHECK'
import sys
if sys.version_info < (3,10):
    raise SystemExit('Requires Python 3.10+: '+sys.version)
try:
    import torch
except ImportError:
    raise SystemExit('Install PPO dependencies with "$CARBON_PYTHON" -m pip install -r requirements-p3.txt')
if tuple(int(v) for v in str(torch.__version__).split('.')[:2]) < (2,6):
    raise SystemExit('Requires PyTorch 2.6+; selected: '+str(torch.__version__))
print('Using Python: '+sys.executable,flush=True)
print('PyTorch: '+str(torch.__version__)+'; policy device: cpu',flush=True)
PYTHON_CHECK
config_path="$1"
output_path="$2"
model_path="$3" # Use - for the main policy; a model is needed only with --wait-features advice.
references_path="$4"
iterations="$5"
shift 5
train_args=(train-ppo --config "$config_path" --output "$output_path"
  --references "$references_path" --iterations "$iterations")
if [[ "$model_path" != "-" ]]; then train_args+=(--predictor "$model_path"); fi
exec "$python_bin" -B -m carbon "${train_args[@]}" "$@"
