# Result figures and tables

`export-results --run RUN --report REPORT --output OUTPUT` checks the sealed
test execution, report, cohort and candidate bindings before deriving figure
data. A new output directory is required. It never edits the paper.

`--tables-only` needs only the standard library. The cluster test wrapper now
produces this export after its report. `render-results --export EXPORT --output
OUTPUT` verifies the exported hashes and renders those same numbers without
replaying the queue or recomputing statistical summaries. Rendering alone needs
`requirements-plots.txt` (matplotlib); this dependency is optional for experiments.

## Outputs

- `E2-main.pdf`: one page per frozen budget, mean TAT versus normalized modeled
  carbon, alongside observed deadline misses and available confidence bounds.
  Fixed scales remain visible. RL shows the equal-weight mean across every
  declared seed and small individual-seed points. Unsupported points are open;
  censored/full-carbon-unavailable groups are explicitly listed.
- `E4-power.pdf`: frozen-policy carbon ratio versus rho, with the same frozen
  comparator throughout. Rho=0 is a separately marked ideal-limit stress point;
  no connecting curve is drawn through the gap to the declared interval.
  Endpoint-trained counterparts appear only where they are in the frozen plan.
  Break-even roots and independent scale-error radii remain in the JSON data.
- `E3-mechanisms.tex` and `.md`: full method/ablation rows by budget, including
  absent ablations as `not in plan`. The LaTeX fragment uses booktabs. Endpoint
  ratios use the same frozen comparator; labels L/H refer to power endpoints,
  not confidence bounds. Missing values are not zero or substituted results.
- `plot-data.json`: all methods, budgets, seeds, original summaries/phase data,
  paired marginal intervals, power curves, and source hashes. No successful-seed
  subset is used. The manifest explicitly records missing E3 and E4 counterparts.

PDF pages have matching PNGs. Each panel is normalized using its own training
reference; the exporter does not pool absolute carbon across workloads.

## Uncertainty and claims

Plot intervals resample whole calendar blocks and, for RL aggregates, an
independent seed multiset crossed with those blocks. Sums retain each block's
episode count; they do not average block means or per-episode carbon ratios.
Both endpoint ratios use the same paired comparator draw. Missing/insufficient
blocks produce no interval, not a zero-width interval. A censored seed prevents
a complete aggregate cost curve.

Displayed intervals are marginal approximate variability intervals. They do not
replace the report's more conservative simultaneous cost/miss checks. With few
seeds they are not calibrated guarantees over future training seeds. Filled E2
markers indicate validation/test deadline support, not automatic carbon gains.
An `observed target met` status means every declared seed meets the empirical
validation/test miss target while confidence support remains inconclusive.
It stays an open marker. `miss above tolerance` and censoring remain distinct;
no observed-target label is a population-feasibility certificate.
The miss panel shows a conservative observed range under censoring and the
largest available per-seed upper bound; no partial outcome is treated as complete.

Synthetic/development exports are conspicuously marked. Layout-only QA fixtures
are temporary test artifacts, never paper evidence. Real multi-panel assembly
and final manuscript insertion wait for the declared AMSP panels and actual
cluster results; manuscript result slots remain empty meanwhile.
