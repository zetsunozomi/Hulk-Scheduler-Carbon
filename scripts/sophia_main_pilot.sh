#!/usr/bin/env bash
# Sophia uses PBS/qsub. Run with bash inside an existing compute allocation.
# For batch submission, add the site's confirmed #PBS resource/account/queue
# directives here. This script does not request or submit another allocation.

set -euo pipefail

# EDIT HERE: Python on the executing cluster; an existing export overrides this.
export CARBON_PYTHON="${CARBON_PYTHON:-/lus/eagle/projects/Local-LLM/shuyuanfan/conda_env/carbon/bin/python}"
# Add any required module load commands here, before running Python.
if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo 'Usage: bash scripts/sophia_main_pilot.sh CONFIG OUTPUT [--resume]' >&2
  exit 2
fi
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
config_path="$1"
output_path="$2"
shift 2
exec "$python_bin" -B -m carbon run-main-pilot --config "$config_path" --output "$output_path" "$@"
