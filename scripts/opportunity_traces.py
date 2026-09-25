"""Exogenous, parameterized ordinary job arrivals, independent of all targets."""
from datetime import timedelta
import hashlib
import random

from carbon.common import iso, timestamp
from carbon.trace import TraceJob


def rng_for(seed, domain):
    value = hashlib.sha256(f'opportunity-v1:{seed}:{domain}'.encode()).digest()
    return random.Random(int.from_bytes(value[:16], 'big'))


def generate_trace(design, scenario, seed):
    spec = design['scenarios'][scenario]
    origin = timestamp(design['origin_utc'])
    horizon = design['trace_hours']
    rows = []

    def add(submit_hours, nodes, runtime_hours, requested_hours):
        if submit_hours >= horizon:
            return
        actual = max(1, round(runtime_hours * 3600))
        requested = round(requested_hours * 3600)
        if not 0 < actual <= requested:
            raise ValueError('Generated runtime exceeds requested walltime')
        rows.append({'submit_seconds': round(submit_hours * 3600), 'nodes': nodes,
                     'actual_seconds': actual, 'requested_seconds': requested})

    if spec['kind'] == 'renewal':
        for i, stream in enumerate(spec['streams']):
            # Separate RNGs keep arrival times identical for wide-long and the
            # short control despite their different runtime distributions.
            arrivals = rng_for(seed, f'renewal-{i}-arrivals')
            values = rng_for(seed, f'renewal-{i}-values')
            low, high = stream['interarrival_hours']
            t = arrivals.uniform(0, high)
            while t < horizon:
                add(t, values.choice(stream['nodes']), values.uniform(*stream['runtime_hours']), stream['request_hours'])
                t += arrivals.uniform(low, high)
    elif spec['kind'] == 'waves':
        arrivals = rng_for(seed, 'wave-arrivals')
        values = rng_for(seed, 'wave-values')
        t = arrivals.uniform(0, spec['interarrival_hours'][1])
        while t < horizon:
            for nodes in values.choice(spec['bundles']):
                add(t + values.uniform(0, spec['submit_spread_hours']), nodes,
                    values.uniform(*spec['runtime_hours']), spec['request_hours'])
            t += arrivals.uniform(*spec['interarrival_hours'])
    else:
        raise ValueError('Unknown background generator')
    rows.sort(key=lambda r: (r['submit_seconds'], r['nodes']))
    jobs = []
    for i, row in enumerate(rows):
        submit = origin + timedelta(seconds=row['submit_seconds'])
        row.update(job_id=f'background:{i:06d}', submit_utc=iso(submit))
        # These fields encode service demand only. Replay must always start
        # empty and rebuild admissions, never initialize from these endpoints.
        jobs.append(TraceJob(row['job_id'], row['nodes'], submit, submit,
                             submit + timedelta(seconds=row['actual_seconds']),
                             timedelta(seconds=row['requested_seconds'])))
    return tuple(jobs), rows


def generate_cohort(design, seed):
    origin = timestamp(design['origin_utc'])
    start, stop = design['arrival_start_hours'], design['arrival_stop_hours']
    count = design['episodes_per_seed']
    rng = rng_for(seed, 'target-arrivals')
    width = (stop - start) / count
    return [{'episode_id': f'seed{seed}-arrival{i:03d}',
             'arrival_utc': iso(origin + timedelta(seconds=round((start + (i + rng.random()) * width) * 3600)))}
            for i in range(count)]


def trace_summary(design, rows, capacity):
    actual = sum(r['nodes'] * r['actual_seconds'] / 3600 for r in rows)
    requested = sum(r['nodes'] * r['requested_seconds'] / 3600 for r in rows)
    return {'jobs': len(rows), 'offered_nodehours': actual, 'offered_requested_nodehours': requested,
            'offered_load_over_full_horizon': actual / (design['trace_hours'] * capacity),
            'requested_to_actual_nodehour_ratio': requested / actual if actual else None,
            'note': 'Offered service demand, not achieved utilization. Arrivals are generated through the full horizon; their end may exceed it.'}
