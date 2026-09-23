#!/usr/bin/env bash
#PBS -N carbon-learn-diag
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
Login node:      qsub scripts/sophia_diagnose_learning.sh
Resume:          qsub -v CARBON_RESUME=1 scripts/sophia_diagnose_learning.sh
PBS interactive: bash scripts/sophia_diagnose_learning.sh [--resume]
Replay recorded training rounds 1/4/8/16; inspect frozen actor gradients.
No optimizer steps, new action sampling, or test evaluation.
Console output is also saved to out/diagnose-learning.<job>.<UTC>.<unique>.log.
Overrides: CARBON_CONFIG, CARBON_TRIAL, CARBON_OUTPUT, CARBON_PYTHON.
USAGE
  exit 0
fi
if [[ -z "${PBS_JOBID:-}" ]]; then
  echo 'Use qsub from the login node, or bash inside an active PBS interactive allocation.' >&2
  exit 2
fi
case "${CARBON_RESUME:-0}" in
  0) ;;
  1) set -- --resume "$@" ;;
  *) echo 'CARBON_RESUME must be 0 or 1.' >&2; exit 2 ;;
esac
if [[ -d "$PWD/src/carbon" ]]; then
  repo_root="$PWD"
elif [[ -n "${PBS_O_WORKDIR:-}" && -d "$PBS_O_WORKDIR/src/carbon" ]]; then
  repo_root="$PBS_O_WORKDIR"
else
  repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
fi
[[ -d "$repo_root/src/carbon" ]] || { echo 'Submit from the repository root.' >&2; exit 2; }
cd "$repo_root"
mkdir -p out
log_path="$(mktemp "out/diagnose-learning.${PBS_JOBID}.$(date -u +%Y%m%dT%H%M%SZ).XXXXXX.log")"
exec > >(tee -a "$log_path") 2>&1
printf 'Log: %s/%s\n' "$repo_root" "$log_path"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python_bin="${CARBON_PYTHON:-/lus/eagle/projects/Local-LLM/shuyuanfan/conda_env/carbon/bin/python}"
command -v "$python_bin" >/dev/null 2>&1 || { echo "Python unavailable: $python_bin" >&2; exit 2; }
output_path="${CARBON_OUTPUT:-results/amsp-learning-diagnose-frontera-7b-seed11}"
printf 'PBS job: %s\nRepository: %s\nOutput: %s\n' "$PBS_JOBID" "$repo_root" "$output_path"
exec "$python_bin" -u -B scripts/diagnose_learning.py \
  --config "${CARBON_CONFIG:-configs/amsp-frontera-7b.development.json}" \
  --trial "${CARBON_TRIAL:-results/amsp-interaction-frontera-7b-seed11}" \
  --output "$output_path" "$@"
