# Synthetic contract fixture

All three CSV files are hand-authored functional-test inputs, not measurements.
`trace.csv` deliberately includes a running job and a pending job submitted
before replay start, later arrivals, and a small queue. `ci.csv` is a synthetic
hourly step function with explicit end-of-interval availability. `cohort.csv`
contains one artificial arrival per synthetic split.

The workload has 100 artificial updates and a five-minute request cap to force
multiple chunks at some scales and a partial final chunk. Its global batch is
128 at every scale. These values are checked in `configs/synthetic.json` and
must not be presented as measured throughput or paper results.
