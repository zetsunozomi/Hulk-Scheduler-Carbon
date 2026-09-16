# P0/P1 execution contract

## Scope and units

- All internal timestamps are timezone-aware UTC. Input trace timezone and DST
  fold policy are explicit. Nonexistent local times are rejected; ambiguous
  times require an offset or a declared fold. Source timezone must be verified.
- Requests and observed job durations are distinct. Durations have microsecond
  precision; requested walltime rounds upward to the configured resolution.
- CI is average operational `gCO2e/kWh`, represented as nonoverlapping half-open
  intervals. Missing realized CI is an error. The source's availability rule is
  preserved. P1 exposes only observations whose intervals have ended and whose
  release times are no later than the decision. P2 will implement forecasting.
- A signed fixed offset can explicitly map queue UTC to CI UTC for a declared
  counterfactual calendar. It does not guess years, weekdays, timezones or DST.

## Initial state and persistent replay

For each cohort arrival, replay begins `warmup_seconds` earlier. Jobs submitted
before that boundary are included if they were still running or pending:

1. Observed start < replay start < cleaned observed end: seed the allocation,
   preserving start and elapsed runtime.
2. Submit < replay start <= observed start: seed a pending request.
3. Submit >= replay start: retain the original exogenous arrival schedule.

The declared source coverage must include the warmup. The manifest's coverage
attestation is the data owner's statement that arrivals are complete, including
empty intervals. A maximum observed submission date cannot prove coverage.
No target is injected during warmup. At the target's initial arrival, process
completions and background arrivals, then clone the state before dispatch.
Each method advances its own clone. Target allocations delay background jobs;
these effects persist across all chunks in that episode.

Hidden observed durations drive physical completion events only. Public
observations include requested walltime, elapsed runtime, elapsed wait, counts,
nodes and requested remaining walltime. They exclude true remaining runtime,
future end times, logged priority and future arrivals. History samples follow a
fixed time cadence. Lag lookup uses the latest sample no later than the desired
timestamp and reports its age; unavailable history has an explicit missing mask.

## Scheduler model: `conservative_backfill_v1`

The scheduler is a node-level approximation, not full Slurm. It supports:

- Homogeneous nodes within one confirmed partition; allocation is exclusive.
- Priority = `age_weight * min(elapsed_wait / age_max_seconds, 1)` +
  `size_weight * requested_nodes / capacity`. Ties use submit time and job ID.
- Dispatch at a fixed interval, with phase anchored at the replay start.
- At each dispatch, consider at most `max_job_test` pending jobs. Reserve a
  feasible interval for every considered higher-priority job before considering
  a lower-priority job. Reservations use **requested** walltime for both running
  and pending jobs. Reservation boundaries are exact, not discretized bins.
- Free resources at exact actual completion. A new target chunk is submitted
  immediately after its predecessor completes; the next scheduler dispatch can
  still impose its ordinary tick delay.

Completions and external arrivals at the same instant are processed before
dispatch. Stopping on a target completion returns before dispatch at that
instant, allowing its next chunk to join the same pending queue. Starting and
finishing are separate `Replay.until` conditions. The simulator never advances
to a later background arrival merely to report target completion.

The old implementation's minute stepping, finite backfill window and
intermediate non-backfill scans are not retained. Fair-share, user/QOS policy,
heterogeneous GPUs, node topology, outages and target hardware failures are not
modeled. Freeze this scheduler version/configuration and validate replay queue
behavior in E1 before interpreting paper results. Old experiment results do not
carry over automatically.

## Fixed work and phases

At each decision, for scale n, initial/restart cost r, checkpoint cost h,
throughput q (updates/hour), limit H (seconds), and remaining integer U:

```
v = min(U, floor(q * (H-r-h) / 3600))
training_seconds = ceil_to_microsecond(3600*v/q)
actual_seconds = r + training_seconds + h
requested_seconds = ceil_to_walltime_resolution(actual_seconds)
```

An action with v <= 0 is masked. H must be a multiple of request resolution.
Actual allocation ends at the end of the final checkpoint, including a final
partial chunk. Initial setup and subsequent restart use separate profiles.
Every scale must satisfy `global_batch = nodes * GPUs/node * microbatch *
accumulation`. The same prescribed number of optimizer updates is completed.

Updates are committed only after their checkpoint phase completes. Phase logs
record initialization/restart, training and checkpoint intervals exactly once.
`global_next_sample_index` is derived from committed updates and global batch.
It is a simulator bookkeeping field, not a distributed checkpoint artifact.
The actual training system must separately implement global sample indexing,
optimizer resharding and a short correctness test.

## Deadlines, censoring and failures

- A budget miss never terminates execution. Run until all work completes or a
  declared split/trace boundary or explicit `max_episode_seconds` is reached.
- Episodes and methods remain in the predeclared cohort. Censored summaries
  have `tat_hours`, total `nodehours`, total `carbon_g_per_kappa` and `carbon_g`
  set to null. Observed elapsed time, exposure and allocated node-hours remain
  available under explicitly labeled fields.
- If unfinished at/past the deadline, miss=true; if censored before it,
  miss=null. Completed exactly at the deadline is not a miss.
- An interrupted chunk contributes observed resource exposure but no committed
  updates. Its already executed updates appear as `processed_updates_uncommitted`.
- P1 declares `target_failure_model=no_failures` and `retry_count=0`. It does not
  silently invent successful retries. Runtime/input failures produce a failed
  run manifest and `failure.json`, and terminate with a nonzero exit code.

## Exposure and cost

For each node scale, store L_n = n * integral(CI(t) dt_hours) over every observed
allocated phase. Compute eta_n = (q_n/n) / (q_4/4), A=sum(L), B=sum(eta*L).

```
C(rho) / kappa = P_reference * (B + rho*(A-B))
```

The same mean power at a given scale is assumed for initialization, training and
checkpoint. Neither `P_reference` nor the rho interval is a measured power
model. If kappa is unknown, `carbon_g` is null and endpoint values are explicitly
`carbon_g_per_kappa`. Within-panel ratios can later cancel kappa. Do not combine
absolute costs across workloads by assuming equal unknown kappa.

The four-dimensional exposure supports subsequent power rescoring without queue
reruns. Keep policy fixed while rescoring. Future queue/forecast interventions
require new replay and are outside P0/P1.

## Output files

- `manifest.json`: input hashes, resolved config, trace audit, selected cohort,
  scheduler model, code hashes, Git commit/dirty state, Python version, command,
  completion/failure status and wall-clock execution time. Written atomically.
- `chunks.jsonl`: stable episode/method/chunk keys, work progress, requested and
  observed durations, submit/start/end times, phases, exposure and endpoint cost.
- `episodes.jsonl`: full-cohort completion/miss/censor status, totals or explicitly
  observed partial values. All policies share the same initial cohort.
- `failure.json`: current episode/method and exception for an aborted run.

Fixed policies have null predictor/forecast/policy-checkpoint fields, because
they do not use those components. Output directories cannot be overwritten.
Sharding uses SHA256(episode_id), keeping all selected fixed methods for an
episode together. Split boundaries limit outcomes, not just initial arrivals.
