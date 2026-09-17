"""Validate an append-only probe prefix; never deserialize executable checkpoints."""

from contextlib import contextmanager
from datetime import timedelta
import fcntl
import json
import math
from pathlib import Path
import uuid

from .common import ContractError, digest, iso, load_json, require, timestamp

# The deployed 644e841 collector has identical sampling/admission semantics.
LEGACY_COLLECTOR = '7cd4c45e1f099977368ad2cf0961993f189ec6e906e08978138aa3074c54b2c9'
CORE = ('common.py', 'config.py', 'trace.py', 'replay.py', 'features.py',
        'carbon.py', 'workload.py', 'runner.py')


def engine_contract(software):
    sources = software['source_sha256']
    return {'version': 1, 'source_sha256': {
        name: sources[name] for name in (*CORE, 'probes.py', 'probe_resume.py')}}


@contextmanager
def probe_lock(output, resume, name='.probe.lock'):
    output = Path(output)
    require(resume or not output.exists(), f'Output already exists: {output}; use --resume to validate and continue')
    output.mkdir(parents=True, exist_ok=resume)
    with (output / name).open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ContractError(f'Another writer is active: {output}') from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def recover_prefix(output, expected):
    """Return validated records and recovery metadata, without writing anything."""
    output = Path(output)
    path = output / 'manifest.json'
    require(path.is_file(), 'Cannot resume probes without their original manifest')
    old = load_json(path)
    for key in expected:
        if key in {'software', 'status', 'rows', 'labeled_rows', 'censored_rows',
                   'queue_summary', 'wait_summary_by_request', 'probe_engine'}:
            continue
        require(old.get(key) == expected[key], f'Cannot resume: probe contract differs at {key}')
    current = expected['probe_engine']
    if 'probe_engine' in old:
        require(old['probe_engine'] == current, 'Cannot resume: probe engine changed')
    else:
        source = old.get('software', {}).get('source_sha256', {})
        require(source.get('probes.py') == LEGACY_COLLECTOR,
                'Cannot resume: unrecognized legacy probe collector')
        require(all(source.get(k) == current['source_sha256'][k] for k in CORE),
                'Cannot resume: legacy replay/feature dependencies changed')
    require(old.get('status') in {'running', 'recovering', 'interrupted', 'failed', 'complete'},
            'Cannot resume: unknown probe status')
    data = output / 'probes.jsonl'
    if old['status'] == 'complete':
        require(data.is_file() and digest(data) == old.get('probes_sha256'),
                'Completed probe dataset hash mismatch')
    start = timestamp(expected['probe_start_utc'])
    stop = timestamp(expected['probe_stop_utc'])
    boundary = timestamp(expected['label_boundary_utc'])
    step = timedelta(seconds=expected['probe_interval_seconds'])
    pairs = [(n, length) for n in expected['resolved_config']['cluster']['allowed_nodes']
             for length in expected['request_seconds']]
    rows, valid_end, tail, needs_newline = [], 0, b'', False
    if data.exists():
        with data.open('rb') as handle:
            for raw in handle:
                try:
                    row = json.loads(raw)
                except (ValueError, UnicodeDecodeError):
                    require(not raw.endswith(b'\n'), 'Malformed complete probe row; refusing automatic repair')
                    tail = raw
                    break
                index = len(rows)
                arrival = start + step * (index // len(pairs))
                n, length = pairs[index % len(pairs)]
                require(arrival < stop, 'Probe prefix has extra rows')
                identity = {'snapshot_id': f"{expected['probe_split']}:{iso(arrival)}",
                            'split': expected['probe_split'], 'arrival_utc': iso(arrival),
                            'nodes': n, 'requested_seconds': length,
                            'label_boundary_utc': iso(boundary)}
                require(isinstance(row, dict) and all(row.get(k) == v for k, v in identity.items()),
                        f'Probe row {index + 1} is not the expected contiguous prefix')
                vector = row.get('features')
                require(isinstance(vector, list) and len(vector) == len(expected['feature_schema']['names']) and
                        all(type(v) in (int, float) and math.isfinite(v) for v in vector),
                        f'Invalid features in probe row {index + 1}')
                require(type(row.get('censored')) is bool, 'Invalid probe censoring flag')
                if row['censored']:
                    require(row.get('wait_hours') is None and row.get('label_observed_at_utc') is None,
                            'Censored probe has a completed label')
                else:
                    wait = row.get('wait_hours')
                    observed = timestamp(row['label_observed_at_utc'])
                    require(type(wait) in (int, float) and math.isfinite(wait) and arrival <= observed < boundary and
                            math.isclose(wait, (observed-arrival).total_seconds()/3600, abs_tol=1e-9),
                            'Probe label and timestamp disagree')
                rows.append(row)
                valid_end += len(raw)
                needs_newline = not raw.endswith(b'\n')
    if old['status'] == 'complete':
        require(not tail and len(rows) == old['rows'] and
                start + step * (len(rows) // len(pairs)) >= stop and len(rows) % len(pairs) == 0,
                'Completed probe dataset has an incomplete grid')
    return old, rows, {'valid_bytes': valid_end, 'tail': tail, 'needs_newline': needs_newline}


def prepare_append(output, recovery):
    """Back up the manifest and any torn tail before explicitly repairing EOF."""
    output = Path(output)
    backup = output / 'resume-history' / uuid.uuid4().hex
    backup.mkdir(parents=True)
    (backup / 'manifest.json').write_bytes((output / 'manifest.json').read_bytes())
    data = output / 'probes.jsonl'
    if recovery['tail']:
        (backup / 'torn-tail.bin').write_bytes(recovery['tail'])
        with data.open('r+b') as handle:
            handle.truncate(recovery['valid_bytes'])
    if recovery['needs_newline']:
        with data.open('ab') as handle:
            handle.write(b'\n')
    return str(backup.relative_to(output))
