# Cluster development findings — updated 2026-09-22

Scope: Frontera × AMSP LLaMA-7B, seed11, development validation. These are real cluster artifacts, not formal six-panel E2–E4 results or an untouched-test evaluation.

Capacity note (2026-09-22): the completed results below all used the declared C128 scenario. The author requires trace-specific resource-pool capacity (Frontera RTX84; LS6 handled separately later). A new [C84 run](../../docs/CAPACITY84_RUN.md) is prepared: actions4/16/64, unchanged learning/chunk rules, capacity-specific fixed/E1/planner/reference/grid preparation, fresh current independent-actor training, and budget curves. No real C84 outcome is available yet. Old outcomes remain valid only as C128 development evidence and are not relabelled or used as C84 baselines.

## Completed evidence

- Main PPO: 64 rounds, 4,096 committed episodes, 4,270 chunks. All four scheduled checkpoints evaluated on 24 arrivals × 4 budgets, with no censoring. The interrupted attempt's 16 uncommitted episodes are excluded.
- Diagnosis job reported by the user: `187563.sophia-pbs-01.lab.alcf.anl.gov` on `sophia-gpu-05`.
- Diagnosis: all 96 arrival/budget units, 192 planner episodes, and 384 frozen-actor input/probability reconstructions complete. The summary's run-plan hash and all 98 actor/planner/dependence stage bindings were checked directly on cluster.
- Sources: `results/amsp-diagnose-frontera-7b-seed11/{diagnosis-summary.json,run-plan.json}`, per-unit planner logs, and `dependence/audit.json`.

## Main outcomes

C below is the larger endpoint-normalized mean carbon, not measured emissions. Deadline misses use all 24 arrivals. Values are descriptive; no confidence intervals or test comparisons have been computed for this development run.

| Budget (h) | PPO-64 C / misses | Plan-once C / misses | MPC C / misses | Validation-fitted Fixed-Mix C |
|---:|---:|---:|---:|---:|
| 15.1077 | 1.1668 / 14 | 1.1356 / 14 | 1.1277 / 14 | infeasible |
| 59.7915 | 1.1668 / 0 | 1.1061 / 1 | 1.1061 / 1 | 1.1220 |
| 104.4754 | 1.1668 / 0 | 1.1008 / 1 | 1.0973 / 1 | 1.0908 |
| 193.8431 | 1.1750 / 0 | 1.0888 / 0 | 1.0889 / 0 | 1.0888 |

The tightest budget cannot reach a 5% empirical miss target on these arrivals with the current action/chunk definition: first actions at 4/16 nodes already allocate about 48h; first actions at 64/128 finish in one chunk, and their fixed outcomes miss on at least 14/24 arrivals even with hindsight choice. Multi-chunk fallback plans at this infeasible budget can have much longer TAT without increasing their already-certain terminal miss indicator. This is consistent with the specified carbon/miss objective, not evidence of deadline-aware speedup.

At 59.79h, both planners execute exactly the same outcomes and sequences: 19 × `[16,4]`, 1 × `[16,16]`, 4 × `[64]`. Mean TAT is 50.143h, p95 58.813h, and one arrival misses. The worst-normalized objective is 1.42% below the fitted Fixed-Mix, but the endpoint ratios relative to that mix are **1.000370 at rho=.25 and 0.985763 at rho=1**. Hence this does **not** establish an interval-wide carbon improvement over Fixed-Mix. The mix was fitted on this same validation cohort, which further limits inference.

At 104.48h, only one of 24 action sequences changes between Plan-once and MPC (`[4,16]` to `[4,64]`). MPC improves the objective by about 0.31% relative to Plan-once, but both remain above Fixed-Mix. At 193.84h, seven sequences change; MPC has almost identical carbon, while mean TAT rises from 86.683h to 98.040h and p95 from 129.009h to 174.933h. Neither outcome supports a substantial feedback contribution.

The planners demonstrate executable mixed-scale opportunities and improve the PPO objective at loose budgets. They do not yet establish that feedback, RL, or interval-wide power robustness beats the strongest fixed alternative.

## Frozen actor diagnosis

Reconstructed public inputs and probabilities match all old evaluation records. Mean per-arrival maximum TV between the four budget-conditioned action distributions is:

| Checkpoint | Mean maximum budget TV |
|---:|---:|
| 16 | 0.00007321 |
| 32 | 0.00001574 |
| 48 | 0.00000263 |
| 64 | 0.00000666 |

These confirm near-invariance of the evaluated actor, without identifying a unique optimization cause. All 384 original PPO evaluations lack scale switching.

## Dependence and cost

The extended dependence audit includes 676 episode records, no censoring, maximum observed duration **211.1097h**, and p95 duration **192.8431h**. Its span floor is 211.1097h (about 8.80 days), so the formerly suggested seven-day blocks cannot simply be adopted under the current audit rule. No block length has been selected and independence is not certified. The 676 rows contain paired methods/budgets; they are not 676 independent arrivals.

The sum of planner stage internal elapsed times is 305.42s; summed planner-decision time is 169.84s. These exclude top-level loading, actor checks, validation/sealing overhead and dependence postprocessing, and are not the complete job wall time.

## Completed actor interaction trial: negative result

Job `187569.sophia-pbs-01.lab.alcf.anl.gov` completed the [declared product-interaction trial](../../docs/ACTOR_INTERACTION_RUN.md). The log was read directly from `out/actor-interaction.187569.sophia-pbs-01.lab.alcf.anl.gov.20260921T045031Z.IfdRpt.log`; outputs are in `results/amsp-interaction-frontera-7b-seed11/`. It trained from fresh seed11 for 64 rounds, 4,096 episodes and 4,252 chunks, with all four scheduled validation checkpoints complete. Source/script bindings, all 64 checkpoint JSON/weight hashes, training-record checkpoint hashes and four validation stage seals were checked. None of the 384 validation episodes was censored. Raw episode means and all budget-specific lambda updates match the training log.

The extra 16,384 actor score weights did not recover useful budget conditioning. Across the four evaluations, sequences were 381 × `[64]`, 2 × `[128]`, and 1 × `[16,16]`; none changed scale. The final checkpoint executed `[64]` for **all 96** arrival/budget pairs. Its mean TAT was **22.12465h** and worst-normalized carbon **1.16678070** at every budget. Misses were **14/24, 0/24, 0/24, 0/24**. Relative to validation-fitted Fixed-Mix, its objective was **3.9884%, 6.9680%, and 7.1632% higher** at the three looser budgets. These are descriptive validation comparisons, not test confidence intervals.

| Checkpoint | Product actor mean maximum budget TV |
|---:|---:|
| 16 | 0.0010888624 |
| 32 | 0.0004434137 |
| 48 | 0.0009275539 |
| 64 | 0.0001243455 |

The final TV is larger than the original actor's but still small in absolute terms. Final mean P(64) was 99.7595%–99.7719% across budgets. The candidate has not been adopted as the formal main architecture; the failure shows that adding this interaction alone was insufficient, without disproving other budget-conditioned architectures.

### Early training evidence

By round 10, initial P(64) already averaged 92.15%–92.94% across budgets. In rounds **10–64**, the three looser budgets supplied **2,640 initial decisions and zero sampled 4/16-node first actions**. At the tightest budget only two of 880 first actions used 4/16 nodes in that window. From the saved logs we can identify early loss of small-scale exploration, but not its unique cause: shared-budget gradients, critic estimation, early sampling variance and objective dynamics remain possible contributors.

Full training fixed outcomes show this is not evidence that 64 nodes minimizes cost everywhere: Fixed-16 has C=1.039144 versus Fixed-64 C=1.143187, with misses 1/73 at 59.79h and 0/73 at the two wider budgets. Fixed-4 meets the widest training target empirically with 3/73 misses and C=1.0. Fixed returns are not counterfactual Q-values for the evolving stochastic continuation policy; they establish a useful comparison, not the exact missing actor gradient.

The training manifest reports **8,189.12s**, including **8,178.78s** rollout and **8.64s** optimizer time. The four evaluation manifests sum to **173.53s**. These do not include all top-level loading/reporting overhead and are not a measured end-to-end PBS wall time.

## Completed frozen learning-signal diagnosis

The [learning diagnosis](../../docs/LEARNING_DIAGNOSIS_RUN.md) completed inside PBS job `187569`, with log `out/diagnose-learning.187569.sophia-pbs-01.lab.alcf.anl.gov.20260921T073722Z.GjFi5X.log`. The four prescribed rounds **1/4/8/16** verified all **256 episodes and 332 chunks**: public input hashes, behavior probabilities, critic values and forced-action replay outcomes match the original training records. The diagnosis run-plan, all 12 original-trial file bindings and four stage seals were checked. No optimizer steps, new action sampling, validation selection or test access occurred. Internal stage times sum to **103.2441s**, excluding top-level loading/reporting overhead.

There is direct evidence of *local actor-gradient conflict* for wider budgets early in this run:

| Round | Budget (h) | Own/shared gradient cosine | Small-action probability derivative: own descent | Same derivative: shared descent |
|---:|---:|---:|---:|---:|
| 1 | 193.8431 | -0.418123 | +0.213556 | -0.302821 |
| 4 | 104.4754 | -0.028229 | +0.050478 | -0.839949 |
| 4 | 193.8431 | -0.686710 | +0.755268 | -0.831532 |

Small-action probability is P(4)+P(16), averaged over that round's initial states. Derivatives are per unit parameter-space descent direction, not realized probability changes or the historical Adam steps. In these cases the wider budget's own local update favors small actions, while the shared direction suppresses them. At round 4 the two tighter budgets' gradients align strongly (cosine **0.966607**). Their violation-gradient norms (**0.73488**, **1.26590**) exceed their carbon-gradient norms (**0.02941**, **0.11907**), consistent with early constraint-related signals having substantial influence on the shared actor.

By round 8 all three loose-budget samples start only at 64/128 nodes. At round 16 all 64 sampled episodes start at 64 nodes, although the local full-batch direction then slightly favors recovering small-action probability. Thus these snapshots do not by themselves explain the entire later trajectory or prove the unique cause of collapse. Adam preconditioning, global clipping, shuffled minibatches, subsequent PPO epochs, shared critic estimation and sampling variance still matter.

## Completed control: independent budget actors

The [independent budget actor control](../../docs/BUDGET_ACTOR_RUN.md) completed through PBS job `187577`: 64 rounds, 4,096 committed training episodes, 6,534 chunks and all 384 validation episodes without censoring. Each of the same four budget ticks uses a separate product actor branch; the critic remains shared. Branches were cloned after the ordinary actor and critic were initialized, preserving initial functions and random-number consumption. Original settings, four budgets, arrival-sampling workflow and four validation checkpoints were retained. All 4,096 committed arrival instances match the completed shared-product trial.

This removes direct cross-budget sharing of actor parameters, but adds **365,955 actor parameters** and does not remove all interaction through shared critic estimation, global gradient clipping or optimizer scheduling. It is not a parameter-matched causal ablation, a guarantee of improvement, or adoption as the formal main method. The same cost, miss, fixed/planner and endpoint comparisons remain necessary.

Before changing core source files, the completed shared-product implementation and its diagnostic scripts were archived and verified at `results/source-archives/product-shared-5e6c7ddcbb91/` (41 files, hashes in `archive.json`). Ordinary checkpoint/source mismatch guards remain strict; the new runner only reuses sealed old outcomes under the explicit policy-file-change contract.

The [2026-09-22 status](../../docs/PROJECT_STATUS_2026-09-22.md) contains the complete outcome interpretation and remaining E1–E4 work. A read-only audit verified 37 source and four script bindings, all 64 checkpoint metadata bindings, the four evaluated weight files and validation seals, and recomputed training means and validation costs/misses/sequences. Interrupted attempts contain 96 uncommitted episode rows and 135 chunk rows, excluded from committed counts. Audit artifact: `out/budget-actor-completed-audit.20260922T024912Z.json`. No new inference, replay, optimizer or test access was performed for this audit.

| Budget (h) | Final C | Misses | Changed scale | Final C vs empirical Best-Fixed | Final C vs fitted Fixed-Mix |
|---:|---:|---:|---:|---:|---:|
| 15.1077 | 1.166781 | 14/24 | 0/24 | infeasible | infeasible |
| 59.7915 | 1.211431 | 0/24 | 1/24 | +3.827% | +7.968% |
| 104.4754 | 1.092272 | 0/24 | 10/24 | +0.007% | +0.137% |
| 193.8431 | 1.131543 | 0/24 | 19/24 | +3.603% | +3.927% |

The previous all-budget collapse to 64 nodes is absent in this candidate's evaluated states through round64. At the two loose budgets, scale-change counts at checkpoints16/32/48/64 are respectively 12/11/12/10 and 18/16/19/19 out of24. However, each budget still has the same first-action argmax across all24 validation arrivals: 64/64/16/16. At104.48h initial P(16)=99.61% on average. At193.84h initial mean P(4)/P(16)=46.79%/48.08%; their respective ranges across arrivals are only45.05%–47.19% and46.37%–48.54%. Stochastic diversity and within-run switching are real; useful queue/CI feedback is not established by those observations or by the large cross-budget TV (0.999801734 at round64).

There is a small positive intermediate result worth retaining: round32 at104.48h has C=1.090670745, misses0/24, about0.1396% below Fixed-16 and0.0096% below fitted Fixed-Mix. Endpoint ratios to the mix are0.999284912 and0.999903636. Round48 is slightly below Fixed-16 but above the mix; round64 is approximately tied with Fixed-16 and above the mix. These are single-panel/seed descriptive validation estimates without uncertainty intervals, not stable or formal gains.

Formal E2–E4 and matching Precommitted-RL remain incomplete. The candidate is not adopted as the formal main method. Remaining development should separate persistent non-single-action behavior from useful state response and improved constrained carbon cost before expanding the panel/seed grid.
