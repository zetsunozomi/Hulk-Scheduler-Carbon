"""Continuous background-only replay against recorded job admissions.

Historical runtimes are reused at simulated admissions. Initial running jobs
restore the boundary state, not validation labels. This does not validate
unobserved alternative requests or hidden Slurm features.
"""

from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from statistics import fmean
import time

from .common import digest, iso, require, timestamp
from .diagnostics import atom_quantile
from .replay import Replay
from .runner import provenance, write_manifest, write_record


class AdmissionRecorder(Replay):
    """Observe the unchanged dispatch implementation without retaining all jobs."""
    def __init__(self,*args,scored_ids,**kwargs):
        super().__init__(*args,**kwargs)
        self.scored_ids = set(scored_ids)
        self.admissions = {}

    def _dispatch(self):
        before = set(self.running)
        super()._dispatch()
        for identity in set(self.running)-before:
            if identity in self.scored_ids:
                require(identity not in self.admissions,'A background job was admitted twice')
                self.admissions[identity] = self.running[identity].start


def discrepancy_summary(rows):
    paired = [r for r in rows if r['wait_error_hours'] is not None]
    result = {'jobs':len(rows),'paired_admissions':len(paired),
              'observed_wait_censored':sum(r['observed_wait_censored'] for r in rows),
              'replay_wait_censored':sum(r['replay_wait_censored'] for r in rows),
              'either_wait_censored':sum(r['observed_wait_censored'] or r['replay_wait_censored'] for r in rows),
              'metrics':None}
    if paired:
        errors = [r['wait_error_hours'] for r in paired]
        absolute = list(map(abs,errors))
        result['metrics'] = {'mean_error_hours':fmean(errors),'mae_hours':fmean(absolute),
                             'median_absolute_error_hours':atom_quantile(absolute,.5),
                             'p90_absolute_error_hours':atom_quantile(absolute,.9),
                             'p95_absolute_error_hours':atom_quantile(absolute,.95),
                             'observed_mean_wait_hours':fmean(r['observed_wait_hours'] for r in paired),
                             'replay_mean_wait_hours':fmean(r['replay_wait_hours'] for r in paired)}
    return result


def evaluate_replay(bundle,output,split='validation',start=None,stop=None):
    require(bundle.raw.get('trace', {}).get('role', 'historical') == 'historical' and
            bundle.raw['execution'].get('initial_state_mode', 'observed') == 'observed',
            'Historical admission fidelity is undefined for a constructed workload scenario')
    require(split in {'train','validation','test'},'Unknown fidelity split')
    split_start,boundary = bundle.splits[split]
    warmup = timedelta(seconds=bundle.raw['execution']['warmup_seconds'])
    start = timestamp(start) if start else max(split_start,bundle.trace_start+warmup)
    stop = timestamp(stop) if stop else boundary
    require(split_start<=start<stop<=boundary and start-warmup>=bundle.trace_start,
            'Fidelity arrival window/warmup is outside declared coverage')
    # Select solely by submission, not by how well either outcome turns out.
    jobs = [j for j in bundle.jobs if start<=j.submit<stop]
    require(jobs,'No historical submissions in the declared fidelity window')
    output = Path(output)
    require(not output.exists(),f'Output already exists: {output}')
    output.mkdir(parents=True)
    c = bundle.raw['cluster']
    meta = {**bundle.manifest,'kind':'recorded_job_fidelity_v1','status':'running','software':provenance(bundle.root),
            'score_split':split,'replay_origin_utc':iso(start-warmup),'submission_start_utc':iso(start),
            'submission_stop_utc':iso(stop),'label_boundary_utc':iso(boundary),'cohort_jobs':len(jobs),
            'runtime_rule':'reuses each cleaned logged runtime after its simulated admission',
            'cohort_rule':'all retained submissions in the declared window; no completion-based filtering',
            'metric_scope':'paired observed and replay admissions before the split boundary; all censoring remains counted',
            'uncertainty':'descriptive discrepancies; inverse empirical-CDF quantiles; no IID job-level confidence intervals',
            'claim_limit':'one continuous background-only replay; not a hidden-state Slurm reconstruction or counterfactual-action ground truth'}
    write_manifest(output/'manifest.json',meta)
    begun = time.monotonic()
    try:
        replay = AdmissionRecorder(bundle.jobs,start-warmup,boundary,c,history_seconds=0,
                                   sample_seconds=c['scheduler']['dispatch_interval_seconds'],scored_ids=[j.job_id for j in jobs])
        meta['initial_running_jobs'] = len(replay.running)
        meta['initial_pending_jobs'] = len(replay.pending)
        require(set(replay.running).isdisjoint(replay.scored_ids),'Initial observed starts must not enter the scored cohort')
        # Do not dispatch at the split endpoint or consult labels beyond it.
        replay.advance_to(boundary,before_dispatch=True)
        rows,groups = [],defaultdict(list)
        with (output/'jobs.jsonl').open('x',encoding='utf-8') as stream:
            for job in jobs:
                observed = job.observed_start if job.observed_start<boundary else None
                simulated = replay.admissions.get(job.job_id)
                observed_wait = (observed-job.submit).total_seconds()/3600 if observed is not None else None
                replay_wait = (simulated-job.submit).total_seconds()/3600 if simulated is not None else None
                row = {'job_id':job.job_id,'split':split,'nodes':job.nodes,'submit_utc':iso(job.submit),
                       'requested_seconds':job.requested.total_seconds(),'observed_start_utc':iso(observed),
                       'replay_start_utc':iso(simulated),'label_boundary_utc':iso(boundary),
                       'observed_wait_hours':observed_wait,'replay_wait_hours':replay_wait,
                       'observed_wait_censored':observed is None,'replay_wait_censored':simulated is None,
                       'censored_wait_lower_bound_hours':(boundary-job.submit).total_seconds()/3600,
                       'wait_error_hours':replay_wait-observed_wait if observed is not None and simulated is not None else None}
                write_record(stream,row);rows.append(row)
                groups[('nodes',str(job.nodes))].append(row)
                groups[('submission_utc_day',iso(job.submit)[:10])].append(row)
        metrics = {'overall':discrepancy_summary(rows),'groups':{}}
        for (category,key),values in sorted(groups.items()):
            metrics['groups'].setdefault(category,{})[key] = discrepancy_summary(values)
        write_manifest(output/'metrics.json',metrics)
        lines = ['# Recorded-job replay discrepancy','',f"Split: {split}; purpose: {meta['purpose']}.",'',
                 f"Submissions: {len(rows)}; paired admissions: {metrics['overall']['paired_admissions']}.",
                 f"Observed/replay censored waits: {metrics['overall']['observed_wait_censored']}/{metrics['overall']['replay_wait_censored']}.",'',
                 'A continuous background replay uses recorded arrivals and durations; initial running jobs only restore boundary state.',
                 'Errors describe paired completed waits. All censored waits remain in jobs.jsonl and the counts above.',
                 'This is distinct from predictor error on simulator-generated probes. No live-node power measurement is involved.']
        if metrics['overall']['metrics']:
            scores = metrics['overall']['metrics']
            lines += ['',f"Paired wait discrepancy (hours): MAE={scores['mae_hours']:.4f}, median absolute error={scores['median_absolute_error_hours']:.4f}, p90 absolute error={scores['p90_absolute_error_hours']:.4f}, signed bias={scores['mean_error_hours']:.4f}."]
        (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        meta.update(status='complete',jobs_sha256=digest(output/'jobs.jsonl'),metrics_sha256=digest(output/'metrics.json'))
    except Exception as exc:
        meta.update(status='failed',failure={'error_type':type(exc).__name__,'message':str(exc)})
        write_manifest(output/'failure.json',meta['failure'])
        raise
    finally:
        meta['elapsed_seconds'] = time.monotonic()-begun
        write_manifest(output/'manifest.json',meta)
    return meta
