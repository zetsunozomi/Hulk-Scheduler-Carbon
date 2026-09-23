#!/usr/bin/env bash
#PBS -N carbon-diagnose
#PBS -A Local-LLM
#PBS -q by-gpu
#PBS -l select=1:ngpus=1:ncpus=32:mem=120gb
#PBS -l place=pack:shared
#PBS -l walltime=04:00:00
#PBS -l filesystems=home:eagle
#PBS -S /bin/bash
#PBS -j oe
#PBS -k doe

set -euo pipefail
if [[ "${1:-}" == '--help' || "${1:-}" == '-h' ]]; then
  cat <<'USAGE'
Login node:       qsub scripts/sophia_diagnose_main.sh
Resume:           qsub -v CARBON_RESUME=1 scripts/sophia_diagnose_main.sh
PBS interactive:  bash scripts/sophia_diagnose_main.sh [--resume]
Same Frontera-7B validation cohort, all four frozen budgets, seed11.
No training or test evaluation. Complete arrival/budget stages are reused.
Console output is also saved to out/diagnose-main.<job>.<UTC>.<unique>.log.
Override CARBON_CONFIG, CARBON_PILOT, CARBON_MAIN, CARBON_E1, CARBON_OUTPUT,
or CARBON_PYTHON via the environment (qsub -v for batch jobs).
USAGE
  exit 0
fi
if [[ -z "${PBS_JOBID:-}" ]]; then
  echo 'Run with qsub from the login node, or bash inside an active PBS interactive allocation.' >&2
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
log_path="$(mktemp "out/diagnose-main.${PBS_JOBID}.$(date -u +%Y%m%dT%H%M%SZ).XXXXXX.log")"
exec > >(tee -a "$log_path") 2>&1
printf 'Log: %s/%s\n' "$repo_root" "$log_path"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python_bin="${CARBON_PYTHON:-/lus/eagle/projects/Local-LLM/shuyuanfan/conda_env/carbon/bin/python}"
config_path="${CARBON_CONFIG:-configs/amsp-frontera-7b.development.json}"
pilot_path="${CARBON_PILOT:-results/amsp-main-frontera-7b-pilot}"
main_path="${CARBON_MAIN:-results/amsp-main-frontera-7b-seed11}"
e1_path="${CARBON_E1:-results/amsp-e1-frontera}"
output_path="${CARBON_OUTPUT:-results/amsp-diagnose-frontera-7b-seed11}"
command -v "$python_bin" >/dev/null 2>&1 || { echo "Python unavailable: $python_bin" >&2; exit 2; }
printf 'PBS job: %s\nRepository: %s\nOutput: %s\n' "$PBS_JOBID" "$repo_root" "$output_path"
exec "$python_bin" -u -B scripts/diagnose_main.py --config "$config_path" \
  --pilot "$pilot_path" --main "$main_path" --e1 "$e1_path" --output "$output_path" "$@"
