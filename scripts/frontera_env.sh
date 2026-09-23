#!/usr/bin/env bash
# Run on the login node. Installs dependencies only; no replay or training.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
mkdir -p out
log_path="$(mktemp "out/frontera-env.$(date -u +%Y%m%dT%H%M%SZ).XXXXXX.log")"
exec > >(tee -a "$log_path") 2>&1
# SITE SETTING: keep this prefix identical to frontera_weighted84.sh.
carbon_env="${CARBON_ENV:-$HOME/.conda/envs/carbon}"
conda_bin="${CONDA_EXE:-/work2/09796/shuyuanfan4814/frontera/miniconda3/bin/conda}"
[[ -x "$conda_bin" ]] || { echo "Conda unavailable: $conda_bin; set CONDA_EXE." >&2; exit 2; }
[[ "$carbon_env" == /* && "$carbon_env" != "$HOME" ]] || { echo 'CARBON_ENV must be an absolute environment prefix, not HOME itself.' >&2; exit 2; }
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# The python3 module/base shell must not inject packages into this environment.
unset PYTHONHOME PYTHONPATH
printf 'Environment: %s\nInstall log: %s/%s\n' "$carbon_env" "$repo_root" "$log_path"
if [[ ! -d "$carbon_env/conda-meta" ]]; then
  [[ ! -e "$carbon_env" ]] || { echo 'Prefix exists but is not a Conda environment; choose a different CARBON_ENV.' >&2; exit 2; }
  "$conda_bin" create --yes --prefix "$carbon_env" --override-channels --channel conda-forge python=3.11 pip
fi
python_bin="$carbon_env/bin/python"
"$python_bin" -c 'import sys; assert sys.version_info >= (3,10), "Use a new Python 3.11 environment prefix"; print(sys.version)'
lock_path="$carbon_env/carbon-pip-freeze.txt"
if [[ ! -f "$lock_path" ]]; then
  # Only binary packages: avoid compiling large packages on a login node.
  "$python_bin" -m pip install --only-binary=:all: --index-url https://download.pytorch.org/whl/cpu -r requirements-p3.txt
  "$python_bin" -m pip install --only-binary=:all: -r requirements-p2.txt -r requirements-plots.txt
else
  echo 'Existing frozen environment: checking it without reinstalling/upgrading packages.'
fi
"$python_bin" -m pip check
"$python_bin" - <<'PY'
import sys, torch, sklearn, matplotlib, scipy
from packaging.version import Version
assert (3,10) <= sys.version_info
assert Version('2.6') <= Version(torch.__version__) < Version('3')
assert torch.version.cuda is None, 'Expected CPU-only PyTorch in this environment'
assert sklearn.__version__ == '1.6.1'
assert Version('3.8') <= Version(matplotlib.__version__) < Version('4')
print('Python:', sys.executable)
print('torch:', torch.__version__, 'CUDA:', torch.version.cuda)
print('sklearn:', sklearn.__version__, 'scipy:', scipy.__version__, 'matplotlib:', matplotlib.__version__)
PY
if [[ ! -f "$lock_path" ]]; then
  "$python_bin" -m pip freeze > "$lock_path.tmp"
  mv -- "$lock_path.tmp" "$lock_path"
  "$conda_bin" list --prefix "$carbon_env" --explicit > "$carbon_env/carbon-conda-explicit.txt"
else
  "$python_bin" -m pip freeze > "$log_path.pip-freeze.txt"
  if ! cmp -s "$lock_path" "$log_path.pip-freeze.txt"; then
    echo "Environment differs from its first freeze; inspect $log_path.pip-freeze.txt before resuming training." >&2
    exit 2
  fi
fi
printf 'Ready. Python: %s\nFrozen packages: %s\n' "$python_bin" "$lock_path"
