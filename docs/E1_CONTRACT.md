# E1 queue and input diagnostics

These are runnable diagnostics, not completed paper evidence. Published AMSP
profiles are archived; actual elastic checkpoint execution is not claimed.

## Current constructed-scenario mode

The two source streams are separate workload templates, not new-machine wait
measurements. `empty_warmup` starts at the fixed trace coverage origin without
historical admissions, advances continuously, and excludes at least 28 days
before scoring. Target episodes clone that background and continue their own
modified state across chunks. Record demand distributions, source cleaning,
queue occupancy, wait tails and prefix sensitivity before formal evaluation.
The retrospective split discloses prior train/validation use. Predictor labels
come from independent simulator probes; they do not validate live admissions.
`evaluate-replay` rejects this mode. The historical comparison below applies
only to a historical-cluster configuration and is not required for AMSP panels.

## Cluster entry

`bash scripts/cluster_e1.sh CONFIG OUTPUT [PROBE_INTERVAL_SECONDS]` runs inside
an interactive allocation; `sbatch` accepts that same script and arguments from
the repository root. Edit its top Slurm directives and Python path when moving
clusters. It uses the existing CARBON_PYTHON environment and P2 fitting dependencies. It runs the wait
pipeline and a queue dependence diagnostic. Historical mode additionally runs recorded-job validation.
Workload, CI, power and cohort fields may remain null in this queue-only config.
The AMSP development configurations explicitly declare scenario capacity, timestamp interpretation and splits; original-platform metadata are not silently inferred.
No real trace experiments are run on the Mac; only synthetic software fixtures.

## Two different sources of truth

- `replay-validation/jobs.jsonl`: observed historical admission versus admission
  in one continuous background-only replay. All submissions in the declared
  window enter the cohort. Initial running/pending jobs restore boundary state;
  initial observed admissions are not scored. Each cleaned historical runtime
  is reused after its simulated start. Missing scheduler state is not inferred.
- `validation-diagnostics/predictions.jsonl`: portable waiting-distribution
  predictions versus counterfactual simulator probes cloned at one public state.
  These labels cannot validate unobserved historical alternative requests.

Recorded-job results report signed bias, MAE and median/p90/p95 absolute wait
error among paired admissions, by node count and submission day. Both observed
and replay censoring are counted separately on the entire submission cohort.
An admission exactly at the label boundary is unknown within that split. A job
may finish after the boundary while its earlier admission remains a valid wait
label. The diagnostic never resets the queue at each scored submission.

Prediction results report median-prediction MAE, p50/p90 absolute error, mean-
prediction RMSE/bias, CRPS and coverage, grouped by nodes/request length/queue
regime. Pairwise ranking compares node counts at the same snapshot and requested
duration, using predicted mean waits. Observed ties and censored pairs have
separate counts; predicted ties get half credit. Missing actions fail the report
rather than improving the score. The report retains both pair-weighted accuracy
and the mean across usable snapshot/request-length groups.

Empirical error quantiles use the inverse empirical CDF. The prediction atom
quantiles also use the inverse CDF. These are descriptive diagnostics, without
IID-job confidence intervals or a probability-calibration guarantee. Predictor
inference wall time is recorded separately from replay and file IO.

## Calendar dependence before test

`audit-dependence` accepts development probes and optional fixed/planner/frozen-
policy evaluation directories via `--episode-runs`. Test-containing manifests
are rejected before outcome parsing. Evolving-policy training rollouts are not
used as a stationary outcome series.

The output retains exact-time lag pairs, gaps, and unknown waits. Constant or
insufficient series have null correlation, not zero. Queue occupancy and mean
waits are separate from per-panel/method/seed/budget TAT, endpoint cost and
allocation-weighted CI series. It reports the public-feature history length and
observed episode spans; censored spans are lower bounds. No automatic lag cutoff
selects a block length or certifies independence. Review the dependence patterns
and complete episode spans before freezing the analysis, and retain the audit
hash. The queue-only diagnostic must be supplemented with full target-episode spans from development runs.

Existing logs suffice for the dependence report and prediction tail/rank scores.
They do not add new policy training, power measurements, or a new experiment grid.


## Probe readiness summary

Probe manifests retain `queue_summary` (window submission count, number of
sampled empty states, unweighted sampled running fraction, maximum pending
count) and `wait_summary_by_request` (complete/censored/positive-wait counts,
mean and maximum known wait by nodes and requested duration). All-empty samples
are explicitly noted without dropping them or changing the window. Zero new
submissions alone do not imply an empty queue: carried-over jobs are counted
in each snapshot. These descriptive checks are not time-integrated utilization,
production validation, or a reason to select only busy periods.
