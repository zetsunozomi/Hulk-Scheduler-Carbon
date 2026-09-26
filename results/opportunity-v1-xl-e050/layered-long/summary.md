# layered-long: training-free synthetic curve

Constructed development workload; no empirical trace or measured energy claim.
Same 48h target request rule and same exogenous arrivals for all methods. No training or checkpoints.

All paired outcomes complete: **True**; arrivals: 36.

| Method | TAT h | Constant-CI kg/κ | ERCOT kg/κ | Constant frontier gap h | ERCOT frontier gap h | Switched |
|---|---:|---:|---:|---:|---:|---:|
| Fixed-4 | 278.205 | 445.092 | 432.658 | 0.000 | 0.000 | 0.000 |
| Fixed-8 | 175.293 | 560.899 | 543.007 | 0.000 | 0.000 | 0.000 |
| Fixed-16 | 110.511 | 707.200 | 684.337 | 0.000 | 0.000 | 0.000 |
| Fixed-32 | 80.403 | 891.251 | 857.788 | 0.000 | 0.000 | 0.000 |
| Dynamic-0.00 | 278.205 | 445.092 | 432.658 | 0.000 | 0.000 | 0.000 |
| Dynamic-0.10 | 278.205 | 445.092 | 432.658 | 0.000 | 0.000 | 0.000 |
| Dynamic-0.20 | 175.293 | 560.899 | 543.007 | 0.000 | 0.000 | 0.000 |
| Dynamic-0.30 | 110.511 | 707.200 | 684.337 | 0.000 | 0.000 | 0.000 |
| Dynamic-0.40 | 110.511 | 707.200 | 684.337 | 0.000 | 0.000 | 0.000 |
| Dynamic-0.50 | 78.045 | 853.449 | 823.329 | -8.542 | -8.339 | 0.556 |
| Dynamic-0.60 | 77.546 | 858.564 | 825.960 | -8.204 | -8.382 | 0.500 |
| Dynamic-0.70 | 77.261 | 862.408 | 829.091 | -7.860 | -8.123 | 0.444 |
| Dynamic-0.80 | 77.075 | 864.012 | 831.216 | -7.784 | -7.940 | 0.417 |
| Dynamic-0.90 | 78.028 | 865.284 | 832.193 | -6.622 | -6.817 | 0.417 |
| Dynamic-1.00 | 78.314 | 867.524 | 833.948 | -5.971 | -6.227 | 0.389 |

Negative gap = below the full fixed lower convex frontier at the same carbon. No extrapolation.
All slider values remain in the plot, including overlaps and losses. No outcome-based policy selection.
Per-seed curves, trace load, node-hours and path-switch rates are in summary.json; complete paths and visible decisions are in seed-*/episodes/.
The constant-CI panel isolates queue/scaling effects. The ERCOT panel rescores identical allocations; the policy never reads future CI.
