# RL-free opportunity curves

Synthetic mechanism development, no training. Negative gaps are favorable.

| Trace | Complete cohort | Constant CI: below / above | ERCOT: below / above | Best constant gap h |
|---|---|---:|---:|---:|
| burst-mix | True | 9 / 0 | 9 / 0 | -4.17805774933646 |
| layered-long | True | 6 / 0 | 6 / 0 | -8.541780448208812 |
| loose-walltime | True | 8 / 0 | 8 / 0 | -12.99613338291276 |
| short-control | True | 5 / 0 | 5 / 0 | -2.4764864094410655 |
| wide-long | True | 6 / 0 | 6 / 0 | -21.559420094171372 |

Each trace directory has summary.md/json, curves.csv and curves-{constant,ercot}.png/pdf.
Binary plots stay out of Git. Recreate them locally from the returned text with --report-only.
