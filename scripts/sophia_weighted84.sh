#!/usr/bin/env bash
#PBS -N carbon-weighted84
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
Interactive allocation: bash scripts/sophia_weighted84.sh
Resume:                 bash scripts/sophia_weighted84.sh --resume
Batch:                  qsub scripts/sophia_weighted84.sh
Batch resume:           qsub -v CARBON_RESUME=1 scripts/sophia_weighted84.sh
Uses completed C84 fixed artifacts; starts new PPO, five observation-only rounds.
64 total rounds (5 observe + 59 update), alpha=0/0.2/0.5/0.8/1, carbon rho=1.
Optional: --stage train (training only), --stage validate --resume.
Overrides: CARBON_PYTHON, CARBON_SOURCE, CARBON_OUTPUT, CARBON_RESUME, CARBON_STAGE.
Logs: out/weighted84.<job>.<UTC>.<unique>.log
The 1h allocation is resumable, not a whole-experiment runtime estimate.
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
  all|train|validate) ;;
  *) echo 'CARBON_STAGE must be all, train or validate.' >&2; exit 2 ;;
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
log_path="$(mktemp "out/weighted84.${PBS_JOBID}.$(date -u +%Y%m%dT%H%M%SZ).XXXXXX.log")"
exec > >(tee -a "$log_path") 2>&1
printf 'Log: %s/%s\nPBS job: %s\n' "$repo_root" "$log_path" "$PBS_JOBID"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python_bin="${CARBON_PYTHON:-/lus/eagle/projects/Local-LLM/shuyuanfan/conda_env/carbon/bin/python}"
command -v "$python_bin" >/dev/null 2>&1 || { echo "Python unavailable: $python_bin" >&2; exit 2; }
exec "$python_bin" -u -B scripts/weighted84_trial.py \
  --config configs/amsp-frontera-7b-c84.development.json \
  --source "${CARBON_SOURCE:-results/amsp-frontera-7b-c84-seed11}" \
  --output "${CARBON_OUTPUT:-results/amsp-frontera-7b-c84-weighted-seed11}" \
  --stage "${CARBON_STAGE:-all}" "$@"
