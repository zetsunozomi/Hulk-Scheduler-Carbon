#!/usr/bin/env bash
# Sophia uses PBS/qsub. Run with bash inside an existing compute allocation.
# For batch submission, add the site's confirmed #PBS resource/account/queue
# directives here. This script does not request or submit another allocation.

set -euo pipefail

# EDIT HERE: Python on the executing cluster; an existing export overrides this.
export CARBON_PYTHON="${CARBON_PYTHON:-/lus/eagle/projects/Local-LLM/shuyuanfan/conda_env/carbon/bin/python}"
# Add any required module load commands here, before running Python.
if [[ $# -lt 2 ]]; then
  echo 'Usage: bash scripts/sophia_fixed.sh CONFIG OUTPUT [run-fixed arguments...]' >&2
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
config_path="$1"
output_path="$2"
shift 2
echo "Fixed-scale replay: $config_path -> $output_path (default split: train)"
"$python_bin" -B -m carbon run-fixed --config "$config_path" --output "$output_path" --split train "$@"
"$python_bin" -B - "$output_path" <<'SUMMARY'
from collections import defaultdict
import json
from pathlib import Path
from statistics import fmean
import sys

root=Path(sys.argv[1])
manifest=json.loads((root/'manifest.json').read_text())
if manifest['status']!='complete':
    raise SystemExit('Fixed run is incomplete; preserve its output and inspect the failure.')
rows=[json.loads(line) for line in (root/'episodes.jsonl').read_text().splitlines()]
groups=defaultdict(list)
for row in rows:
    groups[row['method']].append(row)
summary={
    'kind':'fixed_run_readout',
    'scope':'declared cohort/shard only; a timing shard is not a paper result or full training reference',
    'panel':manifest['panel'], 'splits':manifest['selected_splits'],
    'selected_episode_ids':manifest['selected_episode_ids'],
    'shard_index':manifest['shard_index'], 'shard_count':manifest['shard_count'],
    'elapsed_seconds':manifest['elapsed_seconds'], 'methods':{}}
print(f"Fixed replay complete: {len(manifest['selected_episode_ids'])} arrivals, {len(rows)} outcomes, replay wall time={manifest['elapsed_seconds']:.1f}s")
for method,values in sorted(groups.items(),key=lambda item:int(item[0].split('-')[1])):
    completed=sum(r['final_status']=='completed' and not r['censor_flag'] for r in values)
    full=completed==len(values)
    item={
        'outcomes':len(values), 'completed':completed, 'incomplete':len(values)-completed,
        'mean_tat_hours':fmean(r['tat_hours'] for r in values) if full else None,
        'mean_nodehours':fmean(r['nodehours'] for r in values) if full else None,
        'mean_modeled_carbon_g_per_kappa':{
            str(rho):fmean(r['carbon_g_per_kappa'][str(rho)] for r in values)
            for rho in manifest['resolved_config']['power']['rho_interval']} if full else None,
        'chunk_counts':[r['chunk_count'] for r in values]}
    summary['methods'][method]=item
    print(method+': '+json.dumps(item,sort_keys=True,allow_nan=False))
(root/'fixed-summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True,allow_nan=False)+'\n')
print('Readout saved: '+str(root/'fixed-summary.json'))
SUMMARY
