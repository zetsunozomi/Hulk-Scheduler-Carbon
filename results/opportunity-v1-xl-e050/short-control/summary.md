# short-control: training-free synthetic curve

Constructed development workload; no empirical trace or measured energy claim.
Same 48h target request rule and same exogenous arrivals for all methods. No training or checkpoints.

All paired outcomes complete: **True**; arrivals: 36.

| Method | TAT h | Constant-CI kg/κ | ERCOT kg/κ | Constant frontier gap h | ERCOT frontier gap h | Switched |
|---|---:|---:|---:|---:|---:|---:|
| Fixed-4 | 278.205 | 445.092 | 432.658 | 0.000 | 0.000 | 0.000 |
| Fixed-8 | 175.293 | 560.899 | 543.007 | 0.000 | 0.000 | 0.000 |
| Fixed-16 | 110.511 | 707.200 | 684.337 | 0.000 | 0.000 | 0.000 |
| Fixed-32 | 72.701 | 891.251 | 857.370 | 0.000 | 0.000 | 0.000 |
| Dynamic-0.00 | 278.205 | 445.092 | 432.658 | 0.000 | 0.000 | 0.000 |
| Dynamic-0.10 | 278.205 | 445.092 | 432.658 | 0.000 | 0.000 | 0.000 |
| Dynamic-0.20 | 175.293 | 560.899 | 543.007 | 0.000 | 0.000 | 0.000 |
| Dynamic-0.30 | 110.511 | 707.200 | 684.337 | 0.000 | 0.000 | 0.000 |
| Dynamic-0.40 | 110.511 | 707.200 | 684.337 | 0.000 | 0.000 | 0.000 |
| Dynamic-0.50 | 76.485 | 860.776 | 828.333 | -2.476 | -2.561 | 0.528 |
| Dynamic-0.60 | 76.485 | 860.776 | 828.333 | -2.476 | -2.561 | 0.528 |
| Dynamic-0.70 | 76.485 | 860.776 | 828.333 | -2.476 | -2.561 | 0.528 |
| Dynamic-0.80 | 76.485 | 860.776 | 828.333 | -2.476 | -2.561 | 0.528 |
| Dynamic-0.90 | 76.485 | 860.776 | 828.333 | -2.476 | -2.561 | 0.528 |
| Dynamic-1.00 | 72.701 | 891.251 | 857.370 | 0.000 | 0.000 | 0.000 |

Negative gap = below the full fixed lower convex frontier at the same carbon. No extrapolation.
All slider values remain in the plot, including overlaps and losses. No outcome-based policy selection.
Per-seed curves, trace load, node-hours and path-switch rates are in summary.json; complete paths and visible decisions are in seed-*/episodes/.
The constant-CI panel isolates queue/scaling effects. The ERCOT panel rescores identical allocations; the policy never reads future CI.
