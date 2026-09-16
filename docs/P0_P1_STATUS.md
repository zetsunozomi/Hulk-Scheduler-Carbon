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

## Manual inputs still needed before real runs

These cannot be inferred honestly from the current repository:

1. Confirmed cluster/partition capacity, allowed actions, trace timezone and
   continuous arrival coverage (including gaps and cross-boundary jobs).
2. Workload profiles and provenance matching the intended paper workloads,
   including batch/update settings and initialization/restart/save costs.
3. Texas average operational CI with source, unit and availability/alignment rule.
4. Predeclared chronological splits/cohort; access-history audit for any fresh
   research holdout. Development work can use already inspected data explicitly.
5. Cluster Python executable and local Slurm account/partition/resource settings.

`configs/cluster.template.json` is intentionally incomplete. `validate` reports
unfilled fields and blocks missing/mismatched assets. Synthetic inputs never
silently stand in for these real inputs.

## Scope boundaries

The canonical replay is a newly versioned conservative-backfill approximation;
old Slurm-like code and its empirical results are retained as history. E1 still
needs to establish replay fidelity on the selected partition. The short actual
training restore/switch check is also pending. P2 wait prediction/forecast/MPC,
P3 PPO, and P4 inferential statistics/figures are subsequent implementation work.

Follow `docs/CLUSTER_RUNBOOK.md` for the manual Git and cluster sequence.
