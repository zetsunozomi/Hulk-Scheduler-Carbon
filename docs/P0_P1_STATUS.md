# Current status: 2026-09-17 AMSP scenario revision

Wait-dependency update: the main policy now defaults to `--wait-features none`,
with no predictor artifact required. The sole E3 learning comparison is now Precommitted-RL: the full sequence
is archived before the first target submission, using initial public inputs
and known planned work/allocation time. The earlier advice ablation is retired. The 28-bin CI
sequence is relative to submission, not a predicted admission. E1 continues
unchanged for planning-baseline inputs and descriptive errors. All 145 synthetic software
tests passed, including predictor-free training/evaluation, precommit-before-
execution and initial-only information checks, supported slider ticks, plan-file
seal protection, and frozen-environment replay. These are not paper results.

The earlier entries below are a historical implementation log, not current user
prerequisites. Published AMSP 2024 profiles replace the Qwen/Vista plan. No GPU
profiling, power measurement or new untouched log is required. Current launch
instructions and outstanding experiments are in [CLUSTER_RUNBOOK.md](CLUSTER_RUNBOOK.md).

Launcher update: each `scripts/cluster_*.sh` now runs directly with `bash` in an
interactive allocation or submits directly with `sbatch`. Site directives and
Python path live at the top of each script. The outer submission wrapper and
environment template were removed; older entries below describe prior states.

Added constructed-state replay, optional strict FCFS, audited width transforms,
4/16/64/128-node actions, retrospective evaluation disclosure, six prepared
configs, portable site settings, preflight and frozen-environment sensitivity.
All new paper outcomes remain pending. These are software checks and prepared
inputs, not measured scheduling improvements.

---

# P0/P1 implementation and handoff status

Implementation date: 2026-09-16. This records software verification, not paper
experiment results. The draft and its pending E1–E4 results are unchanged.

## Acceptance checklist

| Requirement | Implementation | Verification |
|---|---|---|
| Reproducible runtime and command entry | `pyproject.toml`, `carbon.__main__`, standard-library-only runtime | CLI contract tests; offline wheel build; cluster wrapper synthetic check |
| Explicit inputs, units, paths and versions | `config.py`, SHA256 asset checks, resolved config/code provenance | Missing template fields, wrong hashes and unknown keys rejected |
| Data/cohort/split contract | `trace.py`, `Bundle._cohort`, initial-capacity audit | Duplicate/invalid rows and split overlap rejected; fixed cohort paired across methods |
| Cross-boundary initial state | `Replay.__init__`, `initial_replay` | Running and pending jobs submitted before replay start are retained |
| Persistent independent episodes | `Replay.clone`, `Environment` | Clones reproduce results independently; target occupancy delays background work |
| Correct time stops | `Replay.until(start/complete)` | Ten-minute job stops at minute ten despite later arrival at minute 100 |
| Work and allocation consistency | `Workload.plan`, `Environment.step` | Integer updates, constant batch, partial requests, phase sums and cross-scale switching |
| No zero-progress actions | Feasibility masks and request validation | All-infeasible initialization rejected; positive-progress work plans verified |
| Public causal observations | `visible`, `visible_history`, released CI observation API | Changing hidden future runtimes/CI does not change past observations |
| Deadline and boundary semantics | `Environment.summary` | Work continues after budget miss; queue/training/checkpoint censoring tested |
| Exact accounting and power uncertainty | `CarbonSeries.integral`, `Exposure` | Fractional-hour integration, gaps, phase conservation, direct endpoint rescore equivalence |
| Complete logs and visible failures | `run_fixed`, atomic manifest, JSONL records | Deterministic repeated outputs; overwrite refusal; injected failure leaves failed manifest |
| Scalable deterministic reservations | Piecewise capacity calendar | Compared with exhaustive integer-time oracle on small synthetic queues |
| Cluster handoff without automatic Git/submission | `scripts/cluster_run.sh`, runbook | Shell syntax check and synthetic invocation; no commit/push/sbatch executed |

## Verification performed

- 34 standard-library unit/integration tests passed on Python 3.11.15.
- Two full synthetic paired runs produced byte-identical chunk and episode
  records, covering four scales and three synthetic cohorts.
- Cluster wrapper completed a synthetic four-scale check.
- Wheel built offline with no dependencies; legacy modules excluded.
- New parser read the existing Frontera and current LS6 training files. This
  was format/cleaning compatibility only, using an explicit UTC test assumption;
  it does **not** establish their true source timezone or partition capacity.
- No real trace replay, RL training, physical power measurement, distributed
  training correctness run, or paper experiment was performed locally.

## Input ownership update (2026-09-16)

- Existing traces and `sim_validation.json` are present. Their path migration,
  split/cohort generation, and configuration are implementation work, not a
  blank template assigned to the user. Capacity/timezone inconsistencies still
  need resolution before claiming a calibrated real-partition replay.
- The user will supply new Qwen3 measurements. Keep model sizes, hardware,
  rates, and overheads pending; do not substitute legacy GPT-2 profiles.
- Texas ERCOT consumption-based operational CO2 intensity has been converted
  from EIA's public archive into `data/texas_eia`. See its source manifest for
  units, the explicit two-hour forward-fill limit, and availability scenario.
- The cluster interpreter is known:
  `/pscratch/sd/s/syfan/conda/envs/carbon/bin/python`. Replay/accounting remain stdlib-only;
  the later P2 fitting dependencies are listed in `requirements-p2.txt`.
- Fresh-holdout access history and actual cluster launch remain separate from
  software verification. Formal Qwen3 results are still pending.

The input converter and CO2/CO2e separation bring the local suite to 40 tests.
These are software checks, not an experiment result.

## Scope boundaries

The canonical replay is a newly versioned conservative-backfill approximation;
old Slurm-like code and its empirical results are retained as history. E1 still
needs to establish replay fidelity on the selected partition. The short actual
training restore/switch check is also pending. The held-out report is implemented;
figure export is implemented and actual real-data evaluations remain pending.

## P2 software update (2026-09-16)

- Added public queue/request features, independent wait probes, chronological
  GBT fitting with purged out-of-fold residual atoms, and validation diagnostics
  by scale, request length and visible queue regime. Censored labels are counted.
- Added causal rolling CI forecasts, complete-work MPC, Plan-once, Queue-blind-MPC,
  training-only normalization references and validation-only Fixed-Mix LP.
  Unreleased CI from already executed intervals cannot enter a planner's past cost.
- The fitted GBT is exported as JSON; prediction and planning need no sklearn.
  Fitting pins sklearn 1.6.1. `scripts/cluster_waits.sh` reuses CARBON_PYTHON.
- 62 tests passed with fitting dependencies, including a synthetic end-to-end
  probe/fit/evaluate/reference/three-planner run. These are software checks only.
- Recorded-job replay fidelity, real queue configuration, Qwen3 measurements,
  the paper's statistical results remain pending. LP feasibility and
  descriptive predictor coverage are not population-level guarantees.

## P3 software update (2026-09-16)

- Added a shared budget-conditioned PPO actor and independent three-head critic,
  both using the same public inputs. Normalization uses frozen training references.
- Costs use complete episodes and gamma=1; gradients sum chunks before averaging
  episodes. Budget-specific duals update from pre-update episode averages.
  Censored training tasks are logged and stop training; they are not dropped.
- Checkpoints retain model/optimizer and both random states. An interrupted
  synthetic run resumes to identical model tensors, optimizer and sampling state.
- Added explicit Current-CI and single-endpoint variants, categorical checkpoint
  evaluation, and a CPU cluster wrapper reusing CARBON_PYTHON.
- 74 tests passed, including an analytical low-cost learning check and synthetic
  replay training/evaluation. No real trace training or paper experiment ran.
  Raw checkpoints are not selected policies; see the subsequent P4 update below.

## P4 selection and analysis update (2026-09-16)

- Added validation-only candidate selection with all declared budgets/seeds and
  strong baselines, paired-cohort checks, unknown censored costs, exact mixtures,
  and frozen Best-Fixed/non-RL comparator identities. Unsupported outcomes remain visible.
- Added nondegenerate miss bounds with explicit scope/assumptions and paired
  calendar-block ratio intervals. Research retains the manuscript's calendar
  scope; finite-cohort bounds are development diagnostics, not a weaker substitute.
- Added endpoint/break-even/scale-error analysis and conservation-checked phase
  overhead/constant-CI rescoring. These reuse logs without replaying the scheduler.
- A small synthetic pipeline ran fixed policies, three planners, all three PPO
  seeds, and validation selection. Its one-arrival policy results were correctly
  marked unsupported. Formal tests/results are still pending.
- All 91 software tests passed, including selection, weighted miss bounds, and
  power/phase conservation. Selection also ran without torch or sklearn.
- Final test orchestration, cross-seed report, plots, and E1 recorded-job fidelity
  remain to connect; software checks do not fill manuscript result placeholders.

Follow `docs/CLUSTER_RUNBOOK.md` for the manual Git and cluster sequence.

## P5 held-out execution and report update (2026-09-16)

- Added hash-matched frozen test execution and a cluster wrapper using the existing
  CARBON_PYTHON environment. All declared seeds, fixed scales, frozen mixture
  weights and validation-selected planner settings are retained.
- Test reports verify file/code/cohort bindings, expose censored/unsupported
  entries, and include each seed, pooled-distribution p95, crossed seed/calendar
  variability, paired cost/miss checks, block summaries and phase/power analysis.
- Joint comparison flags include both methods' miss bounds and two cost endpoints;
  small block counts or insufficient bootstrap tail resolution disable support.
  Seed-population guarantees are not inferred from a few seeds.
- 100 software tests passed. A synthetic end-to-end run trained three one-iteration
  policies, selected two budgets, executed the frozen holdout, and produced the
  report. No synthetic values were inserted into the manuscript.
- Figure export, E1 recorded-job fidelity and dependence audit, real queue config,
  Qwen3 measurements and actual cluster experiments remain pending.

## P6 E1 queue diagnostics update (2026-09-16)

- Added one continuous background-only replay against recorded admission times,
  with boundary-state restoration, submission-based cohorts and separate observed/
  modeled wait censoring. This is not counterfactual historical ground truth.
- Added predictor p50/p90 absolute error, same-snapshot/request-length node ranking,
  explicit tie/unknown-pair counts and portable predictor inference timing.
- Added development-only calendar lag diagnostics, with exact gaps and optional
  per-method/seed/budget outcome series and episode spans. No automatic block
  choice or independence certificate is issued; Qwen3 spans remain pending.
- 113 software tests passed. The new cluster E1 wrapper completed a synthetic
  queue-only run with every Qwen3/CI/power/cohort value intentionally null.
- No historical trace experiment ran locally, no statistical block settings were
  frozen, and manuscript result placeholders remain unfilled. Formal E1 checks
  still require cluster execution on the confirmed real queue configuration.

## P7 scientific export update (2026-09-16)

- Added hash-checked data export and independent rendering from that sealed data.
  The cluster test wrapper produces tables/JSON without plotting dependencies.
- E2 pages retain fixed scales, every budget, individual seeds and censoring/miss
  status. E4 keeps the comparator fixed, includes rho=0 stress and labels missing
  endpoint counterparts. E3 generates a booktabs fragment and a readable table.
- Paired calendar/crossed-seed marginal intervals preserve episode weights and
  ratios of means; they do not replace joint cost/miss evidence from the report.
- 122 software tests passed. Synthetic PDF pages and a separate explicit layout
  fixture exercised missing intervals, censoring, endpoint variants and interval
  bars. PDFs were rendered and visually inspected; the LaTeX table compiled.
- No exported synthetic values were inserted into the manuscript. Formal cluster
  results, real-profile panel assembly and final result insertion remain pending.
