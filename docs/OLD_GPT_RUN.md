# Frontera: old GPT-2 scaling experiment

This adds the old draft's Medium / XL speed tables alongside AMSP.
AMSP remains available through `scripts/frontera_weighted84.sh`, unchanged.
Profile source and units: [data/old_gpt/README.md](../data/old_gpt/README.md).

## What will run

| Setting | Old GPT experiment |
| --- | --- |
| Default model | Medium (345M); choose `--model xl` separately |
| Available scales | **4, 8, 16, 32 nodes**, not GPU counts |
| Work | 100,000 legacy training iterations per episode |
| Queue / capacity | Existing Frontera RTX demand-template replay, 84 nodes |
| Carbon | Existing ERCOT series and normalized modeled node power; no energy measurement |
| Requested walltime | **Fixed: always 48h, including final chunk. Dynamic: rounded planned duration, at most 48h.** |
| Assumed overhead | Initialization/restart/checkpoint: 300s each |
| Fixed baseline | Rebuild Fixed-4/8/16/32 on 73 train + 24 validation arrivals: **388 outcomes per model** |
| PPO | Same five observation rounds, then 59 update rounds; 16 arrivals per alpha per round |
| Alpha | 0, 0.2, 0.5, 0.8, 1; alpha weights time, 1-alpha weights carbon |
| Validation | Every declared checkpoint 16/32/48/64 on the same 24 arrivals; no test evaluation |

Fixed references are computed from the new **training** baselines. Reward
normalizers are separately fitted from the new model's first five observation
rounds. No AMSP reference, fixed result or policy checkpoint is reused.
All queue delays, repeated admission and checkpoint/restart overheads count.
A fixed request of 48h does **not** allocate or charge 48h if the task finishes
earlier. It changes the scheduler reservation; actual runtime, completed work,
node-hours and carbon use the unchanged work plan and actual allocation.
The fixed and dynamic request-duration rules now differ, and reports record
that fact: improvements combine node selection and request sizing.
The PPO actions still select the node count for each checkpoint/resubmit chunk;
this adds no Slurm scheduler modification and no wait-time prediction input.

Medium completes at all four scales within one 48h request. It cannot demonstrate
within-job dynamic scale changes under this setting. XL has a longer
workload and can have multiple allocations. Keeping the 48h cap is intentional:
the work-segmentation and PPO recipe stay unchanged; the fixed submission
rule now always reserves the full cap. All four fixed points remain on the
black comparison curve; we do not replace that curve with one train-selected scale. A new profile does not guarantee that alpha points will separate or beat
the fixed curve.

## Site settings (same installed environment)

Edit the `#SBATCH` header of `scripts/frontera_old_gpt.sh` if moving clusters:
partition `small`, 1 node, 1 task, 1 CPU, **4 hours actual job time**, no explicit
account or GPU request. The 48h cap above is **simulated request walltime**, not
the Slurm allocation running the simulator. Four hours is a resumable allocation
limit, not a promised experiment runtime.

Mail: `--mail-type=ALL`, `--mail-user=sf850@scarletmail.rutgers.edu`.
Python defaults to `$HOME/.conda/envs/carbon/bin/python`; override with
`CARBON_PYTHON=/absolute/path/bin/python` or `CARBON_ENV=/absolute/env`.
The existing AMSP environment suffices; no new dependencies or installation.

## Start on Frontera

After committing/pushing the new files locally, on the cluster:

```bash
cd /scratch2/09796/shuyuanfan4814/carbon
git pull --ff-only
mkdir -p out
sbatch scripts/frontera_old_gpt.sh --model medium
```

That one job first builds new fixed baselines, then trains and validates PPO.
Do not use `CARBON_SOURCE` pointing to an AMSP directory. Defaults are isolated:

```text
configs/old-gpt-frontera-medium-c84.development.json
results/old-gpt-frontera-medium-c84-fixed-max48/
results/old-gpt-frontera-medium-c84-max48-weighted-seed11/
```

The `max48` directories are new. Old GPT results from the previous precise-final-
request rule remain intact in their old directories and cannot be reused as the
new baseline. Both fixed splits and training references must be rebuilt. PPO's
feature scaling uses those references, so the new default PPO run starts from
scratch too. Do not point `CARBON_SOURCE` or `CARBON_OUTPUT` at the old directories.
Baseline manifests, references and PPO run plans record `fixed_request_policy`;
missing or different policies fail validation rather than silently mixing runs.

`--model xl` selects separate configs and both output directories.
No command runs all three models automatically.

If you want to inspect the rebuilt baseline first:

```bash
sbatch scripts/frontera_old_gpt.sh --model medium --stage fixed
# After that job finishes, verify/reuse its baseline and train:
sbatch scripts/frontera_old_gpt.sh --model medium
```

An optional login check reads config, data hashes and dependencies, without
simulation/training: `bash scripts/frontera_old_gpt.sh --model medium --check`.
The same script supports `bash` inside a Slurm compute allocation. Real replay
is rejected on login nodes even if `SLURM_JOB_ID` happens to be set.

## Progress and restart

```bash
squeue -u "$USER"
tail -f out/frontera-old-gpt-JOBID.out
# After a timeout or interruption (never run two writers on the same output):
sbatch scripts/frontera_old_gpt.sh --model medium --resume
```

Fixed preparation seals completed train/validation stages. On interruption an
unfinished fixed stage is rebuilt, while a completed stage is reused after
verification. PPO resumes from the last **complete iteration** including optimizer
and RNG state; unfinished iteration output remains diagnostic only. Completed
validation checkpoints are reused after verification. Changed profiles, config,
input hashes, implementation or settings fail validation instead of silently
mixing experiments. `CARBON_SOURCE` / `CARBON_OUTPUT` may select alternative,
separate output paths. A completed PPO can be revalidated with
`--stage validate --resume`.

## Results and Git return (text only)

Each model's PPO directory contains `old-gpt-summary.json`, `old-gpt-summary.md`
and `weighted-curves.csv`. The CSV includes modeled carbon and turnaround time.
On the cluster, `weighted-curves.png/pdf` plots **carbon on x, turnaround on y**,
black connected Fixed-4/8/16/32 points and yellow alpha-ordered dynamic points
for all four validation checkpoints. These binary figures remain ignored.

After completion, the approved result-return workflow is:

```bash
git add -- out/ results/
git status --short
git commit -m "Return old GPT validation output text"
git push
```

Then locally `git pull --ff-only`. Existing ignore rules allow output text and
only training iterations 10/32/64; all validation text is retained. **Checkpoints
and binary files stay on the cluster. Do not use `git add -f`, archives or a
checkpoint download.** Coordinate pushes so local code changes and cluster result
commits do not diverge. Cluster code edits are still not part of this workflow.

## Pending-job scale-down: future deterministic candidate

Not implemented or enabled by this change. A future rule may check only the
target's observed pending time at fixed intervals and reduce the request by one
scale after a fixed threshold, stopping at 4 nodes. Thresholds would be frozen
from training inputs, not selected using validation outcomes. This would not
require an RL action or a wait-time predictor. Any cancel/resubmit version must
retain elapsed waiting in turnaround time and reset priority age as submission
semantics require; no free reservation or priority retention is assumed.

Large is retired from this active entry. Its historical table/config is retained for provenance.
The broader assumed-profile experiment is documented in [SCALING_SENSITIVITY.md](SCALING_SENSITIVITY.md).
