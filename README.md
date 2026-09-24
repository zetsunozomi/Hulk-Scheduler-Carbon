# Carbon-aware provisioning replay

The canonical entry point is **`python -m carbon`** on Python 3.10+.
Replay, accounting, causal forecasts and planning use the standard library.
Wait-model fitting and the Fixed-Mix LP need `requirements-p2.txt`; no GPU or
PyTorch is needed for these components. Budget-conditioned PPO uses PyTorch 2.6+
on CPU (`requirements-p3.txt`) and has synthetic functional coverage.

## Current entry: Frontera Slurm

Edit code and [paper/](paper/README.md) locally; the cluster only pulls and runs. Follow [FRONTERA_RUN.md](docs/FRONTERA_RUN.md) for the HOME Conda environment, one-script sbatch/interactive launch, logs and resume. The current candidate is [C84 weighted PPO](docs/WEIGHTED84_RUN.md). Missing C84 fixed inputs are rebuilt in the same compute job, without wait probes. Git returns selected output text; checkpoints and binary results stay on the cluster. [RESEARCH_HANDOFF.md](docs/RESEARCH_HANDOFF.md) retains the research history and limitations.


### Additional profile: old GPT-2

[OLD_GPT_RUN.md](docs/OLD_GPT_RUN.md) adds Medium / XL at **4/8/16/32 nodes**.
Use `sbatch scripts/frontera_old_gpt.sh --model medium` (or `xl`).
Each model builds its own fixed baselines and trains from scratch; both fixed
and dynamic retain the 48h execution cap. Fixed now requests 48h even for the
final chunk; dynamic requests rounded planned duration. New outputs use `max48`
directories. AMSP's existing entry and historical results remain unchanged.

### Analytic scaling sensitivity

Medium (T16=22.6h) and XL (T16=110h) each use 14 explicitly assumed
4/8/16/32-node efficiency vectors, spanning 10%–120% endpoint efficiency and
several intermediate curve shapes. The full matrix has 28 profiles and 84 PPO
runs across three seeds. Large is retired from the active legacy entry; its
source data remain archived. Design, anchor audit and commands are in
[docs/SCALING_SENSITIVITY.md](docs/SCALING_SENSITIVITY.md). Inputs are prepared;
no results or universal hardware-coverage claim are made.

## Start here

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python3 -B -m unittest discover -s tests -v
python3 -B -m carbon validate --config configs/synthetic.json
```

The checked-in example is **synthetic contract-test data**. Its profiles, CI,
capacity and results are not evidence for the paper. For an end-to-end check:

```bash
python3 -B -m carbon run-fixed --config configs/synthetic.json \
  --output results/synthetic-check --nodes 4 8 16 32
```

Each output directory is new and immutable to this runner. A second invocation
with the same output path fails instead of overwriting results.

## Earlier pipeline reference

The following describes earlier C128/budget experiments. Use the Frontera entry above for the current C84/weighted run.

Read [the short Chinese runbook](docs/CLUSTER_RUNBOOK.md) for the current story,
portable site settings and first CPU submission. Published AMSP March 2024
Figure 12 supplies LLaMA 7B/13B/30B throughput at 4/16/64/128 eight-A800 nodes.
Input extraction is archived in `data/amsp/profiles.json`; no new Qwen or power
measurement is requested. The two historical streams separately define a
128-node resource-demand scenario, with admissions rebuilt from the trace
prefix. They do not predict arbitrary machines' production queues. Texas CI
is an external regional scenario. Six `configs/amsp-*.development.json` files
use disclosed retrospective chronological splits: old train/validation use
precludes an untouched-test claim.

The shared wait predictor, causal CI forecast, planners, and Fixed-Mix are
implemented and verified on synthetic fixtures, alongside complete-episode PPO
training, checkpoint resume and categorical evaluation. Development experiments have run; the current weighted candidate has not yet established an advantage over fixed scales. `configs/synthetic.json` is a software test, and
`configs/cluster.template.json` is an internal configuration reference.

Each `scripts/cluster_*.sh` is both an interactive job script and an sbatch
script. Edit its top `#SBATCH` resource/site settings and `CARBON_PYTHON` path.
From the repository root, run `bash scripts/cluster_preflight.sh CONFIG OUTPUT`
inside an interactive allocation, or `sbatch scripts/cluster_preflight.sh CONFIG
OUTPUT` from the login node. The body executes the work directly; it never
submits another job. Slurm output is `slurm-carbon-preflight-JOBID.out` in the
submission directory. These CPU resources are independent of the simulated
128-node cluster. The scripts locate the repository using the current directory
or `SLURM_SUBMIT_DIR`, accommodating Slurm's copied script location.

Install any missing dependencies into the selected `$CARBON_PYTHON`, reusing
the existing environment. The first preflight needs only the standard library.

`scripts/cluster_e1.sh CONFIG OUTPUT` runs training probes, GBT fitting,
validation probes and wait/dependence diagnostics. Constructed scenarios skip
recorded-admission fidelity: original starts are not ground truth after changing
the scenario. See [E1 semantics](docs/E1_CONTRACT.md).
Use `python -m carbon --help` for `make-references`, `run-planners`, and
`fit-fixed-mix`. `configs/synthetic-p2.json` contains artificial CI warmup history
for their software checks; it is not a research configuration.

`scripts/cluster_ppo.sh CONFIG OUTPUT - REFERENCES ITERATIONS` runs one
declared main-policy training seed without a wait model. The main policy reads
public queue history, actual remaining budget/work, physical action descriptors,
and 28 six-hour causal CI forecast bins. For the E3 learned-sequence comparison, keep `-` and add
`--decision-mode precommitted`; it is named Precommitted-RL and archives all
actions before the first submission. Other options include `--seed`, `--budgets` and `--resume`.
Iterations must be explicitly chosen before a run. Resume accepts checkpoint
JSON, requires identical settings/input/code/runtime, and writes a new directory.
`run-policy --checkpoint ...` evaluates the saved categorical policy on validation
by default. These raw checkpoint results still need validation selection and
the final held-out report. `select-policies --spec ... --output ...` now freezes
validation operating points and the non-RL comparator; missing seeds/baselines,
unpaired cohorts and incomplete accounting cannot silently pass selection.
`run-test` then executes every frozen point on the declared test cohort, matching
checkpoint hashes from supplied training directories. `report-test` produces paired
per-seed summaries, crossed seed/calendar variability, censoring and power analyses.
The cluster wrapper combines these steps and a sealed table/data export; see
`scripts/cluster_test.sh`. `render-results --export ... --output ...` renders that
export without recomputing replay or statistics. Plotting optionally uses
`requirements-plots.txt`; experiment execution and table export do not need it.
E3 uses Precommitted-RL as its single trained mechanism comparison. Match the
main policy's complete budget grid, arrival sampling, seeds and interaction
budget during training; evaluate the mechanism at one prespecified budget per panel. Current-CI and wait advice remain development options, outside the formal design.
Single-endpoint training remains the bounded E4 comparison. The new v2 feature
schema requires new training; v1 checkpoints cannot be relabeled or reused.
E1 probe/model contracts are unchanged. No prediction accuracy target gates
the main method; queue-shift sensitivity still requires the planned replay tests.

## Package layout

| Module | Responsibility |
|---|---|
| `config.py`, `trace.py` | Versioned manifest, hashes, split/cohort validation, trace cleaning and UTC conversion |
| `replay.py` | Persistent queue, requested-walltime backfill, exact event stops, independent snapshots |
| `workload.py` | Integer updates, fixed global batch, full/partial chunks, setup and checkpoint overhead |
| `environment.py` | One environment for all policies; visible observations, phase logs, censoring |
| `carbon.py` | Realized CI integration and reusable exposure/power endpoint accounting |
| `runner.py`, `__main__.py` | Paired configuration-selected fixed-scale execution, provenance, JSONL results and failure records |
| `features.py`, `probes.py`, `waits.py`, `diagnostics.py` | Public request-conditioned features, chronological GBT/residual fitting, held-out probe errors and coverage |
| `fidelity.py`, `dependence.py` | Continuous recorded-job admission discrepancy and development-only queue/outcome lag diagnostics |
| `forecast.py`, `planning.py`, `baselines.py` | Causal CI forecasts, full-work MPC/Plan-once/queue-blind control, training references, validation-only Fixed-Mix LP |
| `policy_inputs.py`, `learning.py`, `policy_runner.py` | Shared observable actor/critic inputs, complete-episode PPO, per-budget duals, resumable checkpoints, categorical evaluation |
| `results.py`, `selection.py`, `statistics.py` | Validated paired outcomes, validation-only selection, explicitly scoped miss bounds and calendar-block ratio intervals |
| `heldout.py`, `reporting.py` | Frozen test execution, sealed result inputs, per-seed and pooled summaries, paired cost/miss evidence |
| `exporting.py`, `plotting.py` | Sealed figure data, E2/E4 PDF and PNG pages, E3 LaTeX/Markdown tables, visible missing results |
| `power_analysis.py` | Same-log endpoint differences, break-even rho, scale-error radius, phase overhead and constant-CI rescore |

See [the execution contract](docs/EXECUTION_CONTRACT.md),
[the implementation checklist](docs/P0_P1_STATUS.md), and
[the example data description](examples/README.md).

## Relationship to the old code

`src/sim`, `src/model` and `src/queue_prediction` are retained as historical
implementations. Their previous training entry points do not implement the new
execution contract and are not used by `carbon`. The new replay retains
node-level allocation, age/size priority and requested-walltime reservations,
with an explicitly versioned conservative-backfill model. It is not a complete
Slurm emulator and is not bit-for-bit equivalent to the old scheduler.

The planning baselines use the same execution/accounting as the fixed policies.
Validation selection, frozen test execution, seed reporting and power
postprocessing are implemented. E1 replay/predictor/dependence diagnostics are
also available. Figure/table export is implemented and checked on synthetic
inputs; formal inputs and statistical settings are not frozen. See the
[statistical contract](docs/STATISTICS_CONTRACT.md) and
[export semantics](docs/EXPORT_CONTRACT.md) for scope and assumptions.
Actual distributed training/checkpoint correctness is outside this simulation
study; simulated update counters do not establish it. Chunk overhead is an
explicit assumption, with 60/600/1800-second scenarios. `run-stress` freezes
source models and normalization while changing one environment assumption;
see the runbook. Its outputs have a separate sensitivity manifest, preserving
ordinary artifact-matching checks.

## Optional packaging

Running from `PYTHONPATH=src` works offline without installation. If packaging is
preferred and setuptools is already available:

```bash
python3 -m pip install --no-build-isolation --no-deps -e .
carbon-replay --help
```

Only `carbon` is packaged; legacy model directories are excluded.

The user slider selects physical completion budget, not an RL reward weight.
`run-policy --slider-position S` selects a supported tick from the checkpoint
(default grid positions 0, .25, .5, 1). Unsupported positions are rejected, not
interpolated. Checkpoint/evaluation/export metadata preserve the tick mapping.
This interface does not guarantee a deadline or monotone learned outcomes.
