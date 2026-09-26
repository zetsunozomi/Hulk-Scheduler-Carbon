# loose-walltime: training-free synthetic curve

Constructed development workload; no empirical trace or measured energy claim.
Same 48h target request rule and same exogenous arrivals for all methods. No training or checkpoints.

All paired outcomes complete: **True**; arrivals: 36.

| Method | TAT h | Constant-CI kg/κ | ERCOT kg/κ | Constant frontier gap h | ERCOT frontier gap h | Switched |
|---|---:|---:|---:|---:|---:|---:|
| Fixed-4 | 278.205 | 445.092 | 432.658 | 0.000 | 0.000 | 0.000 |
| Fixed-8 | 175.293 | 560.899 | 543.007 | 0.000 | 0.000 | 0.000 |
| Fixed-16 | 110.930 | 707.200 | 683.909 | 0.000 | 0.000 | 0.000 |
| Fixed-32 | 87.767 | 891.251 | 860.026 | 0.000 | 0.000 | 0.000 |
| Dynamic-0.00 | 278.205 | 445.092 | 432.658 | 0.000 | 0.000 | 0.000 |
| Dynamic-0.10 | 278.205 | 445.092 | 432.658 | 0.000 | 0.000 | 0.000 |
| Dynamic-0.20 | 175.293 | 560.899 | 543.007 | 0.000 | 0.000 | 0.000 |
| Dynamic-0.30 | 112.139 | 704.034 | 680.566 | -0.184 | -0.318 | 0.111 |
| Dynamic-0.40 | 111.774 | 705.154 | 681.866 | -0.057 | -0.090 | 0.083 |
| Dynamic-0.50 | 81.757 | 836.137 | 809.107 | -12.947 | -12.707 | 0.833 |
| Dynamic-0.60 | 81.425 | 838.377 | 811.573 | -12.996 | -12.714 | 0.806 |
| Dynamic-0.70 | 81.425 | 838.377 | 811.573 | -12.996 | -12.714 | 0.806 |
| Dynamic-0.80 | 80.019 | 849.577 | 821.713 | -12.993 | -12.787 | 0.667 |
| Dynamic-0.90 | 80.019 | 849.577 | 821.713 | -12.993 | -12.787 | 0.667 |
| Dynamic-1.00 | 79.792 | 854.056 | 825.379 | -12.656 | -12.532 | 0.611 |

Negative gap = below the full fixed lower convex frontier at the same carbon. No extrapolation.
All slider values remain in the plot, including overlaps and losses. No outcome-based policy selection.
Per-seed curves, trace load, node-hours and path-switch rates are in summary.json; complete paths and visible decisions are in seed-*/episodes/.
The constant-CI panel isolates queue/scaling effects. The ERCOT panel rescores identical allocations; the policy never reads future CI.
