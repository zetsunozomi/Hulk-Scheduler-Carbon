#!/bin/bash
#SBATCH --job-name=carbon-scaling
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --output=out/frontera-scaling-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sf850@scarletmail.rutgers.edu

set -euo pipefail
if [[ "${1:-}" == '--help' || "${1:-}" == '-h' ]]; then
  cat <<'USAGE'
Check inputs: bash scripts/frontera_scaling_sensitivity.sh --model xl --scenario e050 --check
Run one case: sbatch scripts/frontera_scaling_sensitivity.sh --model xl --scenario e050 --seed 11
Resume: append --resume. Stages: --stage fixed|train|validate|all.
Models: medium/xl; seeds: 11/23/37. Scenarios: see data/scaling_sensitivity/PROFILES.md.
One fixed source per model/scenario; build it once before submitting multiple seeds.
16-node anchors: Medium=22.6h; XL=110h. Efficiency vectors are assumed.
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
log_path="$(mktemp "out/frontera-scaling.${SLURM_JOB_ID:-check}.$(date -u +%Y%m%dT%H%M%SZ).XXXXXX")"
exec > >(tee -a "$log_path") 2>&1
printf 'Log: %s/%s\nSlurm job: %s\nHost: %s\n' "$repo_root" "$log_path" "${SLURM_JOB_ID:-none}" "$(hostname)"
scaling_env="${CARBON_ENV:-$HOME/.conda/envs/carbon}"
python_bin="${CARBON_PYTHON:-$scaling_env/bin/python}"
[[ -x "$python_bin" ]] || { echo "Python unavailable: $python_bin" >&2; exit 2; }
export PYTHONPATH="$repo_root/src" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
unset PYTHONHOME
exec "$python_bin" -u -B scripts/frontera_scaling_sensitivity.py "$@"
