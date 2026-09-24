# Scaling sensitivity: xl / e050

T16=110h; efficiency32/4=50%; seed=11; rho=1.0.

These throughput profiles are assumed inputs, not measured training speeds.
Compare the full dynamic time-carbon curve with all four fixed points; lower-left is better.
Fixed requests 48h on every chunk; dynamic requests rounded planned duration. Gains include both node choice and request sizing.

| Iteration | Configuration | Carbon kg CO2/kappa | TAT h | Node-hours | Mean chunks | Switched episodes |
|---|---|---:|---:|---:|---:|---:|
| - | Fixed-4 | 424.573937 | 346.224596 | 1112.730524 | 6.000 | 0/24 |
| - | Fixed-8 | 536.533708 | 225.542287 | 1402.246259 | 4.000 | 0/24 |
| - | Fixed-16 | 675.534164 | 164.094922 | 1768.000000 | 3.000 | 0/24 |
| - | Fixed-32 | 851.153587 | 121.679611 | 2228.127714 | 2.000 | 0/24 |
| 16 | alpha=0 | 463.136141 | 307.650446 | 1215.884388 | 5.625 | 19/24 |
| 16 | alpha=0.2 | 558.225166 | 240.332278 | 1455.117988 | 4.417 | 23/24 |
| 16 | alpha=0.5 | 782.432838 | 141.674266 | 2014.513616 | 2.333 | 18/24 |
| 16 | alpha=0.8 | 839.698129 | 122.322989 | 2205.313792 | 2.000 | 3/24 |
| 16 | alpha=1 | 851.153587 | 121.679611 | 2228.127714 | 2.000 | 0/24 |
| 32 | alpha=0 | 424.573937 | 346.224596 | 1112.730524 | 6.000 | 0/24 |
| 32 | alpha=0.2 | 516.276387 | 255.323638 | 1350.275067 | 4.792 | 19/24 |
| 32 | alpha=0.5 | 849.105839 | 122.050056 | 2222.112878 | 2.000 | 1/24 |
| 32 | alpha=0.8 | 851.153587 | 121.679611 | 2228.127714 | 2.000 | 0/24 |
| 32 | alpha=1 | 851.153587 | 121.679611 | 2228.127714 | 2.000 | 0/24 |
| 48 | alpha=0 | 433.727568 | 337.278618 | 1135.626337 | 5.958 | 7/24 |
| 48 | alpha=0.2 | 515.114504 | 256.800643 | 1348.135193 | 4.833 | 20/24 |
| 48 | alpha=0.5 | 844.633245 | 122.428447 | 2210.083205 | 2.000 | 3/24 |
| 48 | alpha=0.8 | 851.153587 | 121.679611 | 2228.127714 | 2.000 | 0/24 |
| 48 | alpha=1 | 779.521704 | 143.804864 | 2048.425837 | 2.458 | 15/24 |
| 64 | alpha=0 | 464.273249 | 306.366971 | 1221.966286 | 5.583 | 19/24 |
| 64 | alpha=0.2 | 516.892787 | 258.302893 | 1355.884237 | 4.792 | 20/24 |
| 64 | alpha=0.5 | 817.770688 | 134.987485 | 2109.409451 | 2.208 | 13/24 |
| 64 | alpha=0.8 | 839.698129 | 122.322989 | 2205.313792 | 2.000 | 3/24 |
| 64 | alpha=1 | 848.809543 | 121.981702 | 2219.728172 | 2.000 | 1/24 |

All declared checkpoints and alpha points are retained, including overlaps and dominated points.
Do not interpret a fitted connecting segment as a measured intermediate scale.
The 48h cap is fixed; model length and efficiency can change the number of chunks.
Development validation only. No test-set or statistical-significance claim.
