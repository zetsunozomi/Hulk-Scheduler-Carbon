# Scaling sensitivity: medium / e050

T16=22.6h; efficiency32/4=50%; seed=11; rho=1.0.

These throughput profiles are assumed inputs, not measured training speeds.
Compare the full dynamic time-carbon curve with all four fixed points; lower-left is better.
Fixed requests 48h on every chunk; dynamic requests rounded planned duration. Gains include both node choice and request sizing.

| Iteration | Configuration | Carbon kg CO2/kappa | TAT h | Node-hours | Mean chunks | Switched episodes |
|---|---|---:|---:|---:|---:|---:|
| - | Fixed-4 | 87.126152 | 80.579055 | 229.127059 | 2.000 | 0/24 |
| - | Fixed-8 | 108.627054 | 49.072486 | 288.335444 | 1.000 | 0/24 |
| - | Fixed-16 | 138.172804 | 39.009028 | 364.266667 | 1.000 | 0/24 |
| - | Fixed-32 | 171.611385 | 36.170441 | 460.920785 | 1.000 | 0/24 |
| 16 | alpha=0 | 87.458957 | 77.547017 | 230.395004 | 2.000 | 3/24 |
| 16 | alpha=0.2 | 95.107253 | 71.143594 | 249.577042 | 1.750 | 9/24 |
| 16 | alpha=0.5 | 152.186315 | 39.760264 | 401.375415 | 1.000 | 0/24 |
| 16 | alpha=0.8 | 173.570441 | 36.652386 | 460.920785 | 1.000 | 0/24 |
| 16 | alpha=1 | 173.570441 | 36.652386 | 460.920785 | 1.000 | 0/24 |
| 32 | alpha=0 | 88.373164 | 78.020966 | 232.085598 | 2.000 | 7/24 |
| 32 | alpha=0.2 | 92.674755 | 78.195951 | 241.422617 | 1.917 | 10/24 |
| 32 | alpha=0.5 | 138.172804 | 39.009028 | 364.266667 | 1.000 | 0/24 |
| 32 | alpha=0.8 | 173.570441 | 36.652386 | 460.920785 | 1.000 | 0/24 |
| 32 | alpha=1 | 173.570441 | 36.652386 | 460.920785 | 1.000 | 0/24 |
| 48 | alpha=0 | 95.556585 | 69.070122 | 256.215037 | 1.750 | 9/24 |
| 48 | alpha=0.2 | 91.736754 | 78.085229 | 239.286330 | 1.917 | 8/24 |
| 48 | alpha=0.5 | 138.172804 | 39.009028 | 364.266667 | 1.000 | 0/24 |
| 48 | alpha=0.8 | 173.570441 | 36.652386 | 460.920785 | 1.000 | 0/24 |
| 48 | alpha=1 | 173.570441 | 36.652386 | 460.920785 | 1.000 | 0/24 |
| 64 | alpha=0 | 89.295683 | 77.140306 | 235.397910 | 1.958 | 9/24 |
| 64 | alpha=0.2 | 91.455924 | 78.221603 | 238.863681 | 1.917 | 7/24 |
| 64 | alpha=0.5 | 138.172804 | 39.009028 | 364.266667 | 1.000 | 0/24 |
| 64 | alpha=0.8 | 173.570441 | 36.652386 | 460.920785 | 1.000 | 0/24 |
| 64 | alpha=1 | 173.570441 | 36.652386 | 460.920785 | 1.000 | 0/24 |

All declared checkpoints and alpha points are retained, including overlaps and dominated points.
Do not interpret a fitted connecting segment as a measured intermediate scale.
The 48h cap is fixed; model length and efficiency can change the number of chunks.
Development validation only. No test-set or statistical-significance claim.
