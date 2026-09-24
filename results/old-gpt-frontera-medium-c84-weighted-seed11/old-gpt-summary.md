# Old GPT-2 C84 weighted PPO

Fixed and dynamic use the same configuration walltime cap (48h in development).

Observation-only rounds: 5; update rounds: 59; alpha: [0.0, 0.2, 0.5, 0.8, 1.0].

Frozen training means (400 complete episodes): TAT=30.374218h; carbon=119580.107673 g/kappa at rho=1.0.

Minimize J = alpha*TAT/Tref + (1-alpha)*carbon/Cref; PPO reward = -J.

| Round | alpha | PPO cost | Best fixed | Cost gain % | TAT h | Switches | Paired wins |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 16 | 0 | 0.440589 | Fixed-4 | +0.000 | 47.091 | 0/24 | 0/24 |
| 16 | 0.2 | 0.662544 | Fixed-4 | +0.000 | 47.091 | 0/24 | 0/24 |
| 16 | 0.5 | 0.995477 | Fixed-8 | -2.494 | 47.091 | 0/24 | 3/24 |
| 16 | 0.8 | 1.246863 | Fixed-8 | -7.438 | 39.611 | 0/24 | 0/24 |
| 16 | 1 | 1.364362 | Fixed-16 | -6.235 | 41.441 | 0/24 | 9/24 |
| 32 | 0 | 0.440589 | Fixed-4 | +0.000 | 47.091 | 0/24 | 0/24 |
| 32 | 0.2 | 0.662544 | Fixed-4 | +0.000 | 47.091 | 0/24 | 0/24 |
| 32 | 0.5 | 0.995477 | Fixed-8 | -2.494 | 47.091 | 0/24 | 3/24 |
| 32 | 0.8 | 1.160540 | Fixed-8 | +0.000 | 39.083 | 0/24 | 0/24 |
| 32 | 1 | 1.380989 | Fixed-16 | -7.530 | 41.946 | 0/24 | 12/24 |
| 48 | 0 | 0.447815 | Fixed-4 | -1.640 | 46.713 | 0/24 | 0/24 |
| 48 | 0.2 | 0.674526 | Fixed-4 | -1.808 | 46.334 | 0/24 | 0/24 |
| 48 | 0.5 | 0.995477 | Fixed-8 | -2.494 | 47.091 | 0/24 | 3/24 |
| 48 | 0.8 | 1.160540 | Fixed-8 | +0.000 | 39.083 | 0/24 | 0/24 |
| 48 | 1 | 1.380989 | Fixed-16 | -7.530 | 41.946 | 0/24 | 12/24 |
| 64 | 0 | 0.447815 | Fixed-4 | -1.640 | 46.713 | 0/24 | 0/24 |
| 64 | 0.2 | 0.674526 | Fixed-4 | -1.808 | 46.334 | 0/24 | 0/24 |
| 64 | 0.5 | 0.995477 | Fixed-8 | -2.494 | 47.091 | 0/24 | 3/24 |
| 64 | 0.8 | 1.160540 | Fixed-8 | +0.000 | 39.083 | 0/24 | 0/24 |
| 64 | 1 | 1.380989 | Fixed-16 | -7.530 | 41.946 | 0/24 | 12/24 |

Positive gain means lower weighted cost. Best fixed is selected on this validation cohort.
Switching and sampled action diversity alone do not establish useful state feedback.
Carbon is modeled job-attributed operational carbon per unknown common kappa, not measured emissions.
No deadline constraint, violation penalty, dual update, or wait-predictor input is used.
All declared checkpoints are reported; no test-set claim or statistical significance claim.
For this linear objective, optimal episode-level Fixed-Mix has the same expected cost as Best-Fixed.
