#!/usr/bin/env bash
#PBS -N carbon-main
#PBS -A Local-LLM
#PBS -q by-gpu
#PBS -l select=1:ngpus=1:ncpus=32:mem=120gb
#PBS -l place=pack:shared
#PBS -l walltime=02:00:00
#PBS -l filesystems=home:eagle
#PBS -S /bin/bash
#PBS -j oe
#PBS -k doe
# EDIT ABOVE when moving sites. Resource shape matches successful Sophia job
# 186384; walltime is increased to 2h (by-gpu currently permits up to 24h).
# qsub this file from the repository root, or bash it in an existing allocation.
# This script never submits another allocation; the model still runs on CPU.

set -euo pipefail

# EDIT HERE: Python on the executing cluster; an existing export overrides this.
export CARBON_PYTHON="${CARBON_PYTHON:-/lus/eagle/projects/Local-LLM/shuyuanfan/conda_env/carbon/bin/python}"
# Add any required module load commands here, before running Python.
# EDIT HERE: one default run, shared by qsub and interactive bash.
config_path="${CARBON_CONFIG:-configs/amsp-frontera-7b.development.json}"
pilot_path="${CARBON_PILOT:-results/amsp-main-frontera-7b-pilot}"
output_path="${CARBON_OUTPUT:-results/amsp-main-frontera-7b-seed11}"
if [[ "${1:-}" == '--help' || "${1:-}" == '-h' ]]; then
  cat <<'USAGE'
Batch:       qsub scripts/sophia_main_train.sh
Batch resume: qsub -v CARBON_RESUME=1 scripts/sophia_main_train.sh
Interactive: bash scripts/sophia_main_train.sh [--resume]
Explicit:    bash scripts/sophia_main_train.sh CONFIG PILOT_OUTPUT OUTPUT [--seed 11|23|37] [--resume]
Defaults may be overridden with CARBON_CONFIG, CARBON_PILOT, CARBON_OUTPUT,
and CARBON_PYTHON. With qsub, export selected overrides using -v.
USAGE
  exit 0
fi
# Keep the existing three-positional-argument interface for other runs.
if [[ $# -gt 0 && "$1" != --* ]]; then
  if [[ $# -lt 3 ]]; then
    echo 'Provide CONFIG PILOT_OUTPUT OUTPUT together, or omit all three. See --help.' >&2
    exit 2
  fi
  config_path="$1"
  pilot_path="$2"
  output_path="$3"
  shift 3
fi
case "${CARBON_RESUME:-0}" in
  0) ;;
  1) set -- --resume "$@" ;;
  *) echo 'CARBON_RESUME must be 0 or 1.' >&2; exit 2 ;;
esac
# PBS may execute a spool copy. Direct bash also works by absolute script path.
if [[ -d "$PWD/src/carbon" ]]; then
  repo_root="$PWD"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "$SLURM_SUBMIT_DIR/src/carbon" ]]; then
  repo_root="$SLURM_SUBMIT_DIR"
elif [[ -n "${PBS_O_WORKDIR:-}" && -d "$PBS_O_WORKDIR/src/carbon" ]]; then
  repo_root="$PBS_O_WORKDIR"
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
# Reuse the existing environment. Install only if its torch is missing/incompatible.
if ! "$python_bin" -B -c 'import torch; v=tuple(int(x) for x in torch.__version__.split(".")[:2]); assert (2,6)<=v<(3,0)' ; then
  "$python_bin" -m pip install -r requirements-p3.txt --index-url https://download.pytorch.org/whl/cpu
fi
printf 'Repository: %s\nConfig: %s\nPilot: %s\nOutput: %s\nPBS job: %s\n' \
  "$repo_root" "$config_path" "$pilot_path" "$output_path" "${PBS_JOBID:-interactive}"
exec "$python_bin" -u -B scripts/main_train.py --config "$config_path" --pilot "$pilot_path" --output "$output_path" "$@"
