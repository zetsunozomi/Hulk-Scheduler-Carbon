#!/usr/bin/env bash
# Sophia local launcher; ignored by Git. Based on scripts/cluster_waits.sh.
# Refresh this local copy after changes to its generic template.
# bash ignores these directives; sbatch reads them.
#SBATCH --job-name=carbon-waits
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
export CARBON_PYTHON="${CARBON_PYTHON:-/lus/eagle/projects/Local-LLM/shuyuanfan/conda_env/carbon/bin/python}"
# Add any required module load commands here, before running Python.
if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo 'Usage: bash scripts/sophia_waits.sh CONFIG OUTPUT [PROBE_INTERVAL_SECONDS]' >&2
  exit 2
fi
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo 'This pipeline is one job; Slurm arrays are not supported.' >&2
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
if sys.version_info < (3, 10):
    raise SystemExit("Requires Python 3.10+; selected: " + sys.version)
try:
    import scipy, sklearn
except ImportError:
    raise SystemExit('Install fitting dependencies with "$CARBON_PYTHON" -m pip install -r requirements-p2.txt')
if sklearn.__version__ != '1.6.1':
    raise SystemExit('This pipeline pins scikit-learn 1.6.1; install requirements-p2.txt first')
print("Using Python: " + sys.executable, flush=True)
print("scikit-learn=" + sklearn.__version__ + ", scipy=" + scipy.__version__, flush=True)
PYTHON_CHECK
config_path="$1"
output_path="$2"
interval_seconds="${3:-21600}"
if [[ -e "$output_path" ]]; then
  echo "Output already exists: $output_path; choose a new directory." >&2
  exit 2
fi
mkdir -p -- "$(dirname -- "$output_path")"
mkdir -- "$output_path"
"$python_bin" -B -m carbon probe-waits --config "$config_path" --output "$output_path/train-probes" --split train --interval-seconds "$interval_seconds"
"$python_bin" -B -m carbon fit-waits --probes "$output_path/train-probes" --output "$output_path/model"
"$python_bin" -B -m carbon probe-waits --config "$config_path" --output "$output_path/validation-probes" --split validation --interval-seconds "$interval_seconds"
"$python_bin" -B -m carbon evaluate-waits --probes "$output_path/validation-probes" --predictor "$output_path/model/model.json" --output "$output_path/validation-diagnostics"
echo "Wait pipeline complete: $output_path"
