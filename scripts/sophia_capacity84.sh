#!/usr/bin/env bash
#PBS -N carbon-c84
#PBS -A Local-LLM
#PBS -q by-gpu
#PBS -l select=1:ngpus=1:ncpus=32:mem=120gb
#PBS -l place=pack:shared
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:eagle
#PBS -S /bin/bash
#PBS -j oe
#PBS -k doe

set -euo pipefail
if [[ "${1:-}" == '--help' || "${1:-}" == '-h' ]]; then
  cat <<'USAGE'
PBS interactive, whole run: bash scripts/sophia_capacity84.sh
Resume:                     bash scripts/sophia_capacity84.sh --resume
Preparation only:           bash scripts/sophia_capacity84.sh --stage prepare
Then training + validation: bash scripts/sophia_capacity84.sh --stage train --resume
Batch:                      qsub scripts/sophia_capacity84.sh
Batch resume:               qsub -v CARBON_RESUME=1 scripts/sophia_capacity84.sh
Fresh C84 fixed/E1/planners/references/grid/PPO. No C128 outcomes/checkpoint reused.
The 1h PBS limit is resumable; it is not a runtime estimate for the whole pipeline.
Logs: out/capacity84.<job>.<UTC>.<unique>.log
Overrides: CARBON_PYTHON, CARBON_OUTPUT (new directory), CARBON_RESUME, CARBON_STAGE.
USAGE
  exit 0
fi
if [[ -z "${PBS_JOBID:-}" ]]; then
  echo 'Use qsub from the login node, or bash inside a PBS interactive allocation.' >&2
  exit 2
fi
case "${CARBON_RESUME:-0}" in
  0) ;;
  1) set -- --resume "$@" ;;
  *) echo 'CARBON_RESUME must be 0 or 1.' >&2; exit 2 ;;
esac
case "${CARBON_STAGE:-all}" in
  all|prepare|train) ;;
  *) echo 'CARBON_STAGE must be all, prepare or train.' >&2; exit 2 ;;
esac
if [[ -d "$PWD/src/carbon" ]]; then
  repo_root="$PWD"
elif [[ -n "${PBS_O_WORKDIR:-}" && -d "$PBS_O_WORKDIR/src/carbon" ]]; then
  repo_root="$PBS_O_WORKDIR"
else
  repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$repo_root"
mkdir -p out
log_path="$(mktemp "out/capacity84.${PBS_JOBID}.$(date -u +%Y%m%dT%H%M%SZ).XXXXXX.log")"
exec > >(tee -a "$log_path") 2>&1
printf 'Log: %s/%s\nPBS job: %s\nRepository: %s\n' "$repo_root" "$log_path" "$PBS_JOBID" "$repo_root"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python_bin="${CARBON_PYTHON:-/lus/eagle/projects/Local-LLM/shuyuanfan/conda_env/carbon/bin/python}"
command -v "$python_bin" >/dev/null 2>&1 || { echo "Python unavailable: $python_bin" >&2; exit 2; }
"$python_bin" -B - <<'PY'
import torch, scipy, sklearn, matplotlib
assert sklearn.__version__ == '1.6.1', 'Expected the existing scikit-learn 1.6.1 environment'
assert (2, 6) <= tuple(int(x) for x in torch.__version__.split('.')[:2]) < (3, 0)
print('Dependencies: torch', torch.__version__, 'scikit-learn', sklearn.__version__, 'matplotlib', matplotlib.__version__, flush=True)
PY
output_path="${CARBON_OUTPUT:-results/amsp-frontera-7b-c84-seed11}"
printf 'Output: %s\n' "$output_path"
exec "$python_bin" -u -B scripts/capacity84_trial.py \
  --config configs/amsp-frontera-7b-c84.development.json \
  --output "$output_path" --stage "${CARBON_STAGE:-all}" "$@"
