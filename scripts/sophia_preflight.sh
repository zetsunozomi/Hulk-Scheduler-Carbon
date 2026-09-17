#!/usr/bin/env bash
# Sophia local launcher; ignored by Git. Based on scripts/cluster_preflight.sh.
# Refresh this local copy after changes to its generic template.
# bash ignores these directives; sbatch reads them.
#SBATCH --job-name=carbon-preflight
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
# Uncomment needed site options below (change ##SBATCH to #SBATCH).
##SBATCH --account=YOUR_ACCOUNT
##SBATCH --partition=YOUR_PARTITION
##SBATCH --qos=YOUR_QOS
##SBATCH --constraint=YOUR_CONSTRAINT

set -euo pipefail

# EDIT HERE: Python on the executing cluster; an existing export overrides this.
export CARBON_PYTHON="${CARBON_PYTHON:-/lus/eagle/projects/Local-LLM/shuyuanfan/conda_env/carbon/bin/python}"
# Add any required module load commands here, before running Python.
[[ $# == 2 ]] || { echo 'Usage: sophia_preflight.sh CONFIG OUTPUT' >&2; exit 2; }
# sbatch executes a spool copy, so its script directory is not the repository.
# Submit from the repository root; direct bash also works by absolute script path.
if [[ -d "$PWD/src/carbon" ]]; then
  repo_root="$PWD"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "$SLURM_SUBMIT_DIR/src/carbon" ]]; then
  repo_root="$SLURM_SUBMIT_DIR"
else
  repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
fi
[[ -d "$repo_root/src/carbon" ]] || { echo 'Cannot locate repository; submit from its root.' >&2; exit 2; }
cd "$repo_root"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
python_bin="$CARBON_PYTHON"
[[ ! -e "$2" ]] || { echo "Output exists: $2" >&2; exit 2; }
mkdir -p -- "$2"
"$python_bin" -B -m carbon validate --config "$1" > "$2/validated-inputs.json"
"$python_bin" -B - "$1" "$2" <<'PY'
from datetime import timedelta
from pathlib import Path
import sys
from carbon.config import Bundle
from carbon.probes import collect_probes
from carbon.common import iso
b=Bundle(sys.argv[1]); start=b.splits['train'][0]
stop=min(start+timedelta(days=2),b.splits['train'][1])
collect_probes(b,Path(sys.argv[2])/'queue-probes',split='train',
               start=iso(start),stop=iso(stop),interval_seconds=21600,
               request_seconds=[21600,172800])
PY
echo "Preflight complete: $2. This is scenario validation and timing, not a paper performance result."
