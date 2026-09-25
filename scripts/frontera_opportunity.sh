#!/bin/bash
#SBATCH --job-name=carbon-opportunity
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=out/frontera-opportunity-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sf850@scarletmail.rutgers.edu

set -euo pipefail
if [[ "${1:-}" == '--help' || "${1:-}" == '-h' ]]; then
  cat <<'USAGE'
One job, all five synthetic traces and all fixed/dynamic curve points:
  mkdir -p out
  sbatch scripts/frontera_opportunity.sh
Optional read-only check:
  bash scripts/frontera_opportunity.sh --check
One trace only:
  sbatch scripts/frontera_opportunity.sh --trace wide-long
Resume completed methods after timeout: append --resume.
Compute-node interactive: bash scripts/frontera_opportunity.sh [same arguments]
Recreate local plots from returned text: append --report-only (no replay).
Trace names: wide-long, layered-long, burst-mix, loose-walltime, short-control.
No RL, no training, no model weights; CPU replay plus matplotlib only.
Environment: CARBON_PYTHON, otherwise CARBON_ENV/bin/python, otherwise
  $HOME/.conda/envs/carbon/bin/python.
Slurm partition/time/CPU/mail settings are all at the top of this same file.
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
log_base="$(mktemp "out/frontera-opportunity.${SLURM_JOB_ID:-local}.$(date -u +%Y%m%dT%H%M%SZ).XXXXXX")"
log_path="${log_base}.log"
mv -- "$log_base" "$log_path"
exec > >(tee -a "$log_path") 2>&1
printf 'Log: %s/%s\nSlurm job: %s\nHost: %s\n' "$repo_root" "$log_path" "${SLURM_JOB_ID:-none}" "$(hostname)"
opportunity_env="${CARBON_ENV:-$HOME/.conda/envs/carbon}"
python_bin="${CARBON_PYTHON:-$opportunity_env/bin/python}"
[[ -x "$python_bin" ]] || { echo "Python unavailable: $python_bin" >&2; exit 2; }
export PYTHONPATH="$repo_root/src" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/carbon-mpl-${USER:-user}}"
unset PYTHONHOME
exec "$python_bin" -u -B scripts/opportunity_trial.py "$@"
