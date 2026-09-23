# C84 weighted PPO

Observation-only rounds: 5; update rounds: 59; alpha: [0.0, 0.2, 0.5, 0.8, 1.0].

Frozen training means (400 complete episodes): TAT=60.717609h; carbon=301303.786890 g/kappa at rho=1.0.

Minimize J = alpha*TAT/Tref + (1-alpha)*carbon/Cref; PPO reward = -J.

| Round | alpha | PPO cost | Best fixed | Cost gain % | TAT h | Switches | Paired wins |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 16 | 0 | 0.976055 | Fixed-4 | -0.045 | 176.796 | 12/24 | 7/24 |
| 16 | 0.2 | 1.056963 | Fixed-64 | -1.222 | 55.675 | 4/24 | 2/24 |
| 16 | 0.5 | 0.942350 | Fixed-64 | +0.000 | 46.910 | 0/24 | 0/24 |
| 16 | 0.8 | 0.840496 | Fixed-64 | +0.000 | 46.910 | 0/24 | 0/24 |
| 16 | 1 | 0.772594 | Fixed-64 | +0.000 | 46.910 | 0/24 | 0/24 |
| 32 | 0 | 0.969180 | Fixed-4 | +0.660 | 158.883 | 15/24 | 11/24 |
| 32 | 0.2 | 1.041742 | Fixed-64 | +0.236 | 48.177 | 1/24 | 1/24 |
| 32 | 0.5 | 0.942350 | Fixed-64 | +0.000 | 46.910 | 0/24 | 0/24 |
| 32 | 0.8 | 0.840496 | Fixed-64 | +0.000 | 46.910 | 0/24 | 0/24 |
| 32 | 1 | 0.772594 | Fixed-64 | +0.000 | 46.910 | 0/24 | 0/24 |
| 48 | 0 | 0.966047 | Fixed-4 | +0.981 | 181.359 | 16/24 | 10/24 |
| 48 | 0.2 | 1.044203 | Fixed-64 | +0.000 | 46.910 | 0/24 | 0/24 |
| 48 | 0.5 | 0.942350 | Fixed-64 | +0.000 | 46.910 | 0/24 | 0/24 |
| 48 | 0.8 | 0.840496 | Fixed-64 | +0.000 | 46.910 | 0/24 | 0/24 |
| 48 | 1 | 0.772594 | Fixed-64 | +0.000 | 46.910 | 0/24 | 0/24 |
| 64 | 0 | 0.981506 | Fixed-4 | -0.604 | 201.635 | 15/24 | 4/24 |
| 64 | 0.2 | 1.044203 | Fixed-64 | +0.000 | 46.910 | 0/24 | 0/24 |
| 64 | 0.5 | 0.942350 | Fixed-64 | +0.000 | 46.910 | 0/24 | 0/24 |
| 64 | 0.8 | 0.840496 | Fixed-64 | +0.000 | 46.910 | 0/24 | 0/24 |
| 64 | 1 | 0.886721 | Fixed-64 | -14.772 | 53.840 | 2/24 | 0/24 |

Positive gain means lower weighted cost. Best fixed is selected on this validation cohort.
Switching and sampled action diversity alone do not establish useful state feedback.
Carbon is modeled job-attributed operational carbon per unknown common kappa, not measured emissions.
No deadline constraint, violation penalty, dual update, or wait-predictor input is used.
All declared checkpoints are reported; no test-set claim or statistical significance claim.
For this linear objective, optimal episode-level Fixed-Mix has the same expected cost as Best-Fixed.
