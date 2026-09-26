# burst-mix: training-free synthetic curve

Constructed development workload; no empirical trace or measured energy claim.
Same 48h target request rule and same exogenous arrivals for all methods. No training or checkpoints.

All paired outcomes complete: **True**; arrivals: 36.

| Method | TAT h | Constant-CI kg/κ | ERCOT kg/κ | Constant frontier gap h | ERCOT frontier gap h | Switched |
|---|---:|---:|---:|---:|---:|---:|
| Fixed-4 | 278.205 | 445.092 | 432.658 | 0.000 | 0.000 | 0.000 |
| Fixed-8 | 176.892 | 560.899 | 543.687 | 0.000 | 0.000 | 0.000 |
| Fixed-16 | 111.767 | 707.200 | 685.225 | 0.000 | 0.000 | 0.000 |
| Fixed-32 | 75.101 | 891.251 | 856.905 | 0.000 | 0.000 | 0.000 |
| Dynamic-0.00 | 278.205 | 445.092 | 432.658 | 0.000 | 0.000 | 0.000 |
| Dynamic-0.10 | 278.205 | 445.092 | 432.658 | 0.000 | 0.000 | 0.000 |
| Dynamic-0.20 | 178.334 | 557.613 | 540.248 | -1.432 | -1.696 | 0.139 |
| Dynamic-0.30 | 113.081 | 702.879 | 681.533 | -0.610 | -0.385 | 0.111 |
| Dynamic-0.40 | 112.487 | 705.269 | 683.331 | -0.140 | -0.151 | 0.056 |
| Dynamic-0.50 | 75.494 | 868.310 | 835.923 | -4.178 | -4.089 | 0.333 |
| Dynamic-0.60 | 75.413 | 870.428 | 837.990 | -3.837 | -3.728 | 0.306 |
| Dynamic-0.70 | 74.957 | 874.908 | 842.451 | -3.400 | -3.231 | 0.250 |
| Dynamic-0.80 | 74.465 | 879.387 | 845.390 | -3.000 | -3.096 | 0.194 |
| Dynamic-0.90 | 74.465 | 879.387 | 845.390 | -3.000 | -3.096 | 0.194 |
| Dynamic-1.00 | 74.465 | 883.231 | 849.434 | -2.234 | -2.231 | 0.139 |

Negative gap = below the full fixed lower convex frontier at the same carbon. No extrapolation.
All slider values remain in the plot, including overlaps and losses. No outcome-based policy selection.
Per-seed curves, trace load, node-hours and path-switch rates are in summary.json; complete paths and visible decisions are in seed-*/episodes/.
The constant-CI panel isolates queue/scaling effects. The ERCOT panel rescores identical allocations; the policy never reads future CI.
