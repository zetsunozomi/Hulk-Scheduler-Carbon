# Carbon-aware provisioning replay

P0/P1 provides a reproducible, fixed-work simulation and accounting foundation.
The canonical entry point is **`python -m carbon`**. It uses Python 3.10+ and the
standard library; it needs no GPU, PyTorch, Ray, pandas or scikit-learn.

## Start here

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python3 -B -m unittest discover -s tests -v
python3 -B -m carbon validate --config configs/synthetic.json
```

The checked-in example is **synthetic contract-test data**. Its profiles, CI,
capacity and results are not evidence for the paper. For an end-to-end check:

```bash
python3 -B -m carbon run-fixed --config configs/synthetic.json \
  --output results/synthetic-check --nodes 4 8 16 32
```

Each output directory is new and immutable to this runner. A second invocation
with the same output path fails instead of overwriting results.

## Cluster handoff

Read [the Chinese cluster runbook](docs/CLUSTER_RUNBOOK.md) for manual Git steps,
required real inputs, validation, Slurm submission and expected outputs. Start
real configuration from `configs/cluster.template.json`; unresolved fields fail
validation. Cluster experiments must use confirmed profiles, partitions, CI and
predeclared cohorts. The template intentionally does not infer these from old
hardcoded constants.

## Package layout

| Module | Responsibility |
|---|---|
| `config.py`, `trace.py` | Versioned manifest, hashes, split/cohort validation, trace cleaning and UTC conversion |
| `replay.py` | Persistent queue, requested-walltime backfill, exact event stops, independent snapshots |
| `workload.py` | Integer updates, fixed global batch, full/partial chunks, setup and checkpoint overhead |
| `environment.py` | One environment for all policies; visible observations, phase logs, censoring |
| `carbon.py` | Realized CI integration and reusable exposure/power endpoint accounting |
| `runner.py`, `__main__.py` | Paired Fixed-4/8/16/32 execution, provenance, JSONL results and failure records |

See [the execution contract](docs/EXECUTION_CONTRACT.md),
[the implementation checklist](docs/P0_P1_STATUS.md), and
[the example data description](examples/README.md).

## Relationship to the old code

`src/sim`, `src/model` and `src/queue_prediction` are retained as historical
implementations. Their previous training entry points do not implement the new
execution contract and are not used by `carbon`. The new replay retains
node-level allocation, age/size priority and requested-walltime reservations,
with an explicitly versioned conservative-backfill model. It is not a complete
Slurm emulator and is not bit-for-bit equivalent to the old scheduler.

P2's wait regressor, CI forecasting and planning baselines, and P3's constrained
PPO are not part of P0/P1. P1 records the inputs and exposure they will share.
Actual distributed training/checkpoint correctness remains a separate short
cluster check; simulated update counters do not establish it.

## Optional packaging

Running from `PYTHONPATH=src` works offline without installation. If packaging is
preferred and setuptools is already available:

```bash
python3 -m pip install --no-build-isolation --no-deps -e .
carbon-replay --help
```

Only `carbon` is packaged; legacy model directories are excluded.
