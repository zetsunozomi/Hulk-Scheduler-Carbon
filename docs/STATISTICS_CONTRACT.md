# Selection and uncertainty contract

This documents software semantics. No real experiment results or final
statistical settings have been frozen yet.

## Validation selection

`carbon select-policies --spec SPEC --output OUTPUT` consumes a references file,
budget/epsilon grid, statistical settings, declared seeds and candidate run paths.
It writes `selection.json`, a short `selection.md`, and a hash manifest. All fixed
scales, Fixed-Mix, Rollout-MPC, Plan-once, and every declared full-policy seed must
be present at every budget. The implementation will prepare the real spec; this
is not another input template assigned to the author.

Research specs must explicitly freeze `selection_rule` before examining
validation outcomes. The revised manuscript uses `empirical_miss`: among
complete candidates whose observed miss rate is at most epsilon, select the
minimum worst normalized mean carbon. Apply the same criterion to every method,
checkpoint and comparator. If none qualifies, retain the least-observed-miss
complete candidate as a labeled fallback. Censored candidates do not qualify.

Candidates must use the same cohort, workload, CI interpretation and replay
core. Complete mean carbon is unknown if a candidate has censored probability
mass. Fixed outcomes can be rescored at different deadlines without replay;
budget-dependent policies/planners must have actually run at that deadline.
Fixed-Mix uses its exact component distribution, including for p95 TAT.

`empirical_target_met` and `selection_criterion_met` record observed attainment
and the frozen selection rule. `supported` separately requires the upper miss
bound to meet epsilon. It is not turned on by empirical selection. Bounds retain
their Bonferroni allocation across every declared candidate-budget view.
Best-Fixed and the strongest eligible non-RL comparator are frozen on validation,
without choosing a winning RL/planning seed.

The alternative `confidence_upper_miss` rule retains the original strict
confidence-filtered selection for an explicitly predeclared certification study.
Legacy non-research specs without the field retain that behavior. Failure of a
bound to pass is inconclusive evidence, not proof of infeasibility. At zero
misses the single-comparison 95% bound needs 59 equal independent blocks to
certify a 5% miss target; this is not a minimum size for empirical selection.

## Two distinct scopes

- `calendar_blocks`: the manuscript's research uncertainty scope. Each block's
  miss fraction is a bounded observation. The bound assumes independent blocks;
  the audited block length is an approximation to that assumption, not proof.
  Too few blocks or an absent audit disables support. Within-block dependence
  is unrestricted. The cohort's fixed block-size weights are retained.
- `fixed_cohort`: synthetic/development diagnostic only. Conditional on the
  entire frozen historical cohort, independent action streams provide sampling
  replication for stochastic policies. Fixed/analytically marginalized mixture
  results are exact on that finite cohort. This does not replace a calendar
  generalization claim. Research selection rejects this easier scope.
- `descriptive`: point outcomes only; no feasibility support.

For independent bounded observations Xi with fixed weights wi summing to one,
use `n_eff = 1/max(wi)` and invert
`n_eff * kl(observed_mean, upper_mean) = log(1/alpha)`.
Zero observed misses therefore give `1 - alpha**(1/n_eff)`, never an automatic
zero upper bound. Unknown censored misses count as one in an upper bound.

The bounded-variable exponential-moment inequality is given in
[Garivier and Cappé, Lemma 9](https://arxiv.org/pdf/1102.2490).
For the weighted extension used here, convexity in wi bounds each log moment
by wi/max(w) times its value at max(w); Jensen's inequality then combines the
possibly different means. Chernoff optimization gives the expression above.
This is a fixed-sample bound, not the paper's random-stopping KL-UCB theorem.
Calendar independence remains an explicit assumption. For validation-fitted
Fixed-Mix under that scope, combine simultaneous fixed-component bounds instead
of pretending that the fitted weights were chosen before seeing validation.

## Cost comparisons and log reuse

Calendar bootstrap draws keep both methods, both power endpoints and all
arrivals within a block together. Ratios divide means; they do not average
per-run ratios. Two-endpoint one-sided bounds use Bonferroni adjustment. The
bootstrap is approximate and needs the dependence audit and enough blocks.
`run-test` consumes the sealed selection and matches the selected checkpoint
hashes in supplied training directories. It freezes a test plan before executing
outcomes, checks the cohort hash and software, reuses the fixed runs for exact
mixtures, and keeps all selected seeds and unsupported descriptive fallbacks.
Failed execution writes a failure manifest; it is not accepted as a complete report.
A fresh output directory is required; automatic partial-run resume is not implemented.

`report-test` verifies the plan, selection, result file hashes and settings. It
reports each seed and the equal-weight distribution across every declared seed;
pooled p95 is not an average of seed p95s. No complete-case intersection is taken.
One missing/unsupported seed remains visible. Censoring prevents full-cohort
carbon claims even when an observed deadline miss is already known.

Per-panel joint evidence allocates alpha across one miss upper bound for each
frozen test view and two endpoint upper bounds for every listed contrast, across
all budgets. Both policies must be validation-supported and test-feasible before
a carbon gain gets a budgeted-improvement flag. Bootstrap tails must contain at
least 10 expected draws at the adjusted probability; otherwise flags remain off.
This is approximate bootstrap inference, not exact finite-sample cost coverage.
The separate `observed_budgeted_carbon_reduction` field requires complete
validation/test cohorts, both methods meeting the observed miss target, and
both endpoint ratios of mean cost below one. It is a descriptive point-estimate
comparison, not statistical significance or a population-feasibility certificate.
Records separately expose validation/test observed target attainment and miss
bound support. All seeds, target violations and inconclusive results remain.
The report also exposes the zero-miss bound floor and required number of equal
independent blocks at selection, so unsupported certification is visible before test.

Cross-seed variability uses an independent seed multiset and calendar-block
multiset per bootstrap draw, crossed over the complete matrix. This follows the
separate row/column resampling idea in [Owen's pigeonhole bootstrap](https://arxiv.org/abs/0712.1111).
Calendar draws and both cost endpoints remain paired with the frozen comparator.
With few seeds these intervals are descriptive, approximate variability; the
paper must not treat them as a calibrated guarantee for future training seeds.
The separate all-declared-seeds flag requires every frozen seed's contrast to
pass its adjusted cost/miss checks. It concerns those checkpoints, not a seed population.

Power postprocessing uses mean per-scale exposure for endpoint differences,
break-even rho and independent scale-error radius; it never averages per-job
radii. Phase logs must conserve each episode's exposure and node-hours.
Constant-CI rescoring keeps the executed intervals fixed and is labeled as a
rescore, not a newly simulated controller. No node-power measurement is inferred.
