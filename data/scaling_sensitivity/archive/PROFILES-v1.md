# Assumed scaling inputs (not experiment results)

Pure training hours; every scenario fixes T(16)=22.6h and 100000 abstract work units.

| Scenario | Efficiency 32 vs 4 | gamma | T4 h | T8 h | T16 h | T32 h |
|---|---:|---:|---:|---:|---:|---:|
| e025 | 25% | 0.333333 | 35.875 | 28.474 | 22.600 | 17.938 |
| e050 | 50% | 0.666667 | 56.948 | 35.875 | 22.600 | 14.237 |
| e075 | 75% | 0.861654 | 74.624 | 41.067 | 22.600 | 12.437 |
| e100 | 100% | 1.000000 | 90.400 | 45.200 | 22.600 | 11.300 |

Regenerate: `python scripts/scaling_profiles.py --write`.
Check determinism: `python scripts/scaling_profiles.py --check`.
