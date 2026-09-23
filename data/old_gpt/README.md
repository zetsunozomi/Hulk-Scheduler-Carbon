# Old GPT-2 node scaling profiles

Source: the local old draft `old_formats/carbon_SC26/6design.tex`, section
“Training-Workload Profiles”. The relevant source is copied verbatim into
`source-table.tex`; `profiles.json` records its SHA-256 and the original file's
SHA-256. The author confirms that the speeds were measured and **4/8/16/32
are nodes**. The old figure's “GPU Count” label is wrong.

The source table is **node-hours for 100,000 training iterations**, not hours:

| Profile | 4 nodes | 8 nodes | 16 nodes | 32 nodes |
| --- | ---: | ---: | ---: | ---: |
| Medium (345M) | 139.87 | 207.09 | 361.60 | 640.42 |
| Large | 272.68 | 410.66 | 680.00 | 1262.41 |
| XL (1.5B) | 866.80 | 1220.00 | 1760.00 | 2894.16 |

Conversion: `training_hours = node_hours / nodes`,
`updates_per_hour = 100000 * nodes / node_hours`. One legacy training iteration
is represented as one replay work unit. We do not round the intermediate times.
Medium's 16-node time is 22.6h, Large's 42.5h, XL's 110h.
The draft table is authoritative; the older `baseline.csv` rounds XL/8 to
1219.99 instead of 1220.00. Neither that CSV's outcome columns nor
`src/sim/application.py`'s different 345M/1.5B/2.0B profiles are imported.

## Metadata and accounting

The available source does not recover physical GPUs per node, machine per
profile, global batch, microbatch/accumulation, sequence length, precision,
software, repetitions, measurement uncertainty, or Large's parameter count.
We do not infer those from node counts or use this table to claim elastic
optimizer correctness, equivalent convergence, or cross-machine generality.

The current simulator schema requires those batch/GPU integers. Its compatibility
mapping uses **one abstract worker per node**, 32 abstract batch units,
microbatch 1 and accumulation `32 / nodes`; sequence length 1 is an unused
sentinel. They only satisfy the simulator's integer work-accounting invariant.
They are **not measured training metadata or a claim of one physical GPU per
node**. `global_next_sample_index` in output is therefore an abstract work
counter, not a recovered real sample index. `profile_status=assumed` describes
this replay mapping; the underlying speed table remains author-attested measured
data. These restrictions are repeated in each config's workload provenance.

Initialization, restart and checkpoint each cost **300 seconds**, a declared
scenario assumption shared with the AMSP experiment. The imported table defines
pure training time; those overheads are added by replay. Carbon still uses the
existing normalized node-power model, not measured node energy. The historical
Frontera demand stream and ERCOT series define a constructed scenario; they do
not establish that the old GPT measurements came from Frontera hardware.

## Experiment boundary

AMSP configs, core code, entry scripts and results remain unchanged. Each old
GPT profile has independent config, fixed train/validation baselines, reference
normalizers and PPO results. The arrival cohort file under `data/amsp/cohorts/`
is reused deliberately for the same arrival dates; it contains no speed or
baseline outcomes. No wait-time predictor or queue probes are needed.

Both fixed and dynamic requests retain the **48-hour cap**, as requested.
All Medium scales finish in one allocation, including assumed overheads;
Medium therefore compares initial node choice, not within-job switching.
Large can span allocations at 4/8 nodes; XL spans allocations at every scale.
This is a scaling-profile comparison, not a new PPO recipe or promised Pareto gain.

Launch instructions: [docs/OLD_GPT_RUN.md](../../docs/OLD_GPT_RUN.md).
