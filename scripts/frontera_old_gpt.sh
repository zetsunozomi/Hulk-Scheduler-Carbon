#!/bin/bash
#SBATCH --job-name=carbon-old-gpt
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --output=out/frontera-old-gpt-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sf850@scarletmail.rutgers.edu

set -euo pipefail
if [[ "${1:-}" == '--help' || "${1:-}" == '-h' ]]; then
  cat <<'USAGE'
Login check (no replay): bash scripts/frontera_old_gpt.sh --check
Login submission:       mkdir -p out; sbatch scripts/frontera_old_gpt.sh
Batch resume:           sbatch scripts/frontera_old_gpt.sh --resume
Interactive compute:    bash scripts/frontera_old_gpt.sh [--resume]
Active profiles: --model medium (default) | xl. Large is retired; historical inputs remain archived.
Fixed requests 48h on every chunk, including final; dynamic requests planned duration up to 48h.
Default outputs contain max48; old precise-request baselines cannot be resumed here.
Stages: --stage fixed | train | validate | all (default all).
Each profile has a separate fixed source, rebuilt on the compute node; no wait probes or old PPO.
Site settings: SBATCH header; Python defaults to $HOME/.conda/envs/carbon/bin/python.
Overrides: CARBON_ENV, CARBON_PYTHON, CARBON_SOURCE, CARBON_OUTPUT, CARBON_RESUME.
Four hours is a resumable allocation limit, not an experiment runtime estimate.
USAGE
  exit 0
fi
if [[ -d "$PWD/src/carbon" ]]; then
  repo_root="$PWD"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "$SLURM_SUBMIT_DIR/src/carbon" ]]; then
  repo_root="$SLURM_SUBMIT_DIR"
else
  repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$repo_root"
mkdir -p out
log_path="$(mktemp "out/frontera-old-gpt.${SLURM_JOB_ID:-check}.$(date -u +%Y%m%dT%H%M%SZ).XXXXXX.log")"
exec > >(tee -a "$log_path") 2>&1
printf 'Log: %s/%s\nSlurm job: %s\nHost: %s\n' "$repo_root" "$log_path" "${SLURM_JOB_ID:-none}" "$(hostname)"
# SITE SETTING: absolute Python avoids the loaded python3 module/base environment.
carbon_env="${CARBON_ENV:-$HOME/.conda/envs/carbon}"
python_bin="${CARBON_PYTHON:-$carbon_env/bin/python}"
[[ -x "$python_bin" ]] || { echo "Python unavailable: $python_bin; run bash scripts/frontera_env.sh first." >&2; exit 2; }
export PYTHONPATH="$repo_root/src" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
unset PYTHONHOME
case "${CARBON_RESUME:-0}" in
  0) ;;
  1) set -- --resume "$@" ;;
  *) echo 'CARBON_RESUME must be 0 or 1.' >&2; exit 2 ;;
esac
exec "$python_bin" -u -B scripts/frontera_old_gpt.py "$@"
