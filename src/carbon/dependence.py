"""Development-only calendar dependence diagnostics; never certify independence."""

from collections import defaultdict
from datetime import timedelta
import json
import math
from pathlib import Path
from statistics import fmean

from .common import digest, iso, number, require, timestamp
from .diagnostics import atom_quantile
from .runner import provenance, write_manifest, write_record
from .waits import queue_contract


DEFAULT_LAGS_HOURS = [6.,12.,24.,48.,72.,168.,336.]


def lag_correlations(series,lag_hours):
    """Exact-time lag pairs, preserving gaps and unknown censored observations."""
    result = []
    for hours in lag_hours:
        delta = timedelta(hours=float(number(hours,'lag hours',strict=True)))
        pairs = [(value,series[at+delta]) for at,value in sorted(series.items())
                 if value is not None and at+delta in series and series[at+delta] is not None]
        correlation = None
        reason = 'fewer than three observed lag pairs'
        if len(pairs)>=3:
            left,right = fmean(x for x,_ in pairs),fmean(y for _,y in pairs)
            vx = math.fsum((x-left)**2 for x,_ in pairs)
            vy = math.fsum((y-right)**2 for _,y in pairs)
            if vx>0 and vy>0:
                correlation = max(-1.,min(1.,math.fsum((x-left)*(y-right) for x,y in pairs)/math.sqrt(vx*vy)))
                reason = None
            else:
                reason = 'constant values in at least one side of the lag pairs'
        result.append({'lag_hours':hours,'observed_pairs':len(pairs),'correlation':correlation,'unavailable_reason':reason})
    return result


def audit_dependence(probe_directories,output,episode_directories=(),lag_hours=None):
    output = Path(output)
    require(not output.exists(),f'Output already exists: {output}')
    require(probe_directories,'Provide development probes')
    lags = [float(number(v,'lag hours',strict=True)) for v in (DEFAULT_LAGS_HOURS if lag_hours is None else lag_hours)]
    require(lags and len(set(lags))==len(lags),'Declare nonempty unique dependence lags')
    lags.sort()
    inputs,metadata = [],[]
    # Reject test artifacts from their manifests before reading outcome rows.
    for directory in map(Path,probe_directories):
        meta = json.loads((directory/'manifest.json').read_text())
        require(meta['kind']=='wait_probes' and meta['status']=='complete','Incomplete dependence probe input')
        require(meta['probe_split'] in {'train','validation'},'Dependence design must not inspect test probe outcomes')
        require(digest(directory/'probes.jsonl')==meta['probes_sha256'],'Dependence probe hash mismatch')
        if metadata:
            require(queue_contract(meta)==queue_contract(metadata[0]) and meta['purpose']==metadata[0]['purpose'] and
                    meta['feature_schema']==metadata[0]['feature_schema'] and meta['request_seconds']==metadata[0]['request_seconds'],
                    'Dependence inputs use different queue/probe contracts')
        metadata.append(meta)
        inputs.append({'path':str(directory),'manifest_sha256':digest(directory/'manifest.json'),'probes_sha256':meta['probes_sha256']})
    first = metadata[0]
    names = first['feature_schema']['names']
    indices = {key:names.index(key) for key in ('lag_0.available_fraction','lag_0.pending.node_fraction','lag_0.running.node_fraction')}
    nodes = first['resolved_config']['cluster']['allowed_nodes']
    expected = {(n,length) for n in nodes for length in first['request_seconds']}
    snapshots = {}
    for directory,meta in zip(map(Path,probe_directories),metadata):
        local = defaultdict(dict)
        with (directory/'probes.jsonl').open(encoding='utf-8') as stream:
            for row in map(json.loads,stream):
                require(row['split']==meta['probe_split'],'Dependence probe split differs from metadata')
                at = timestamp(row['arrival_utc'])
                require(timestamp(meta['probe_start_utc'])<=at<timestamp(meta['probe_stop_utc']),'Probe outside declared sampling window')
                key = row['nodes'],row['requested_seconds']
                require(key not in local[at],'Duplicate dependence probe action')
                require(len(row['features'])==len(names),'Dependence feature dimensions differ')
                if row['censored']:
                    require(row['wait_hours'] is None,'Censored dependence probe has a fabricated wait')
                else:
                    observed = timestamp(row['label_observed_at_utc'])
                    require(at<=observed<timestamp(meta['label_boundary_utc']),'Dependence label crosses split boundary')
                    require(math.isfinite(row['wait_hours']) and row['wait_hours']>=0 and
                            math.isclose((observed-at).total_seconds()/3600,row['wait_hours'],rel_tol=1e-9,abs_tol=1e-9),
                            'Dependence wait label disagrees with timestamps')
                local[at][key] = row
        for at,actions in local.items():
            require(at not in snapshots,'Overlapping development probe snapshots')
            require(set(actions)==expected,'Dependence snapshots require all declared probe actions')
            rows = list(actions.values())
            values = {key:rows[0]['features'][index] for key,index in indices.items()}
            require(all(all(math.isfinite(row['features'][index]) and row['features'][index]==values[key]
                            for key,index in indices.items()) for row in rows),'Inconsistent public state at a probe snapshot')
            values['probe_censor_fraction'] = fmean(float(row['censored']) for row in rows)
            for n in nodes:
                group = [row for row in rows if row['nodes']==n]
                values[f'wait_mean_hours.nodes_{n}'] = fmean(r['wait_hours'] for r in group) if all(not r['censored'] for r in group) else None
            snapshots[at] = values
    require(snapshots,'No development snapshots')
    series = {key:{at:values[key] for at,values in snapshots.items()} for key in next(iter(snapshots.values()))}
    durations,episode_inputs,episode_censored = [],[],0
    episode_series = defaultdict(lambda:defaultdict(list))
    for directory in map(Path,episode_directories):
        meta = json.loads((directory/'manifest.json').read_text())
        declared = meta.get('selected_splits') or ([meta['split']] if 'split' in meta else [])
        require(meta['status']=='complete' and declared and set(declared)<={'train','validation'},
                'Episode-span audit requires complete declared development runs, never test results')
        require('fixed_nodes' in meta or 'methods' in meta or meta.get('kind')=='policy_evaluation',
                'Use fixed/planner/frozen-policy evaluation logs, not evolving-policy training rollouts')
        require(queue_contract(meta)==queue_contract(first) and meta['purpose']==first['purpose'],'Episode-span queue differs from probes')
        episode_inputs.append({'path':str(directory),'manifest_sha256':digest(directory/'manifest.json'),
                               'episodes_sha256':digest(directory/'episodes.jsonl')})
        with (directory/'episodes.jsonl').open(encoding='utf-8') as stream:
            for row in map(json.loads,stream):
                require(row['split'] in declared,'Episode-span row has an undeclared split')
                split_start,split_end = map(timestamp,meta['resolved_config']['splits'][row['split']])
                require(split_start<=timestamp(row['initial_arrival_utc'])<=timestamp(row['observed_until_utc'])<=split_end,
                        'Episode span crosses its development split')
                elapsed = float(number(row['observed_elapsed_hours'],'observed episode span'))
                require(math.isclose((timestamp(row['observed_until_utc'])-timestamp(row['initial_arrival_utc'])).total_seconds()/3600,
                                     elapsed,rel_tol=1e-9,abs_tol=1e-9),'Episode span disagrees with timestamps')
                durations.append(elapsed);episode_censored += int(row['censor_flag'])
                at = timestamp(row['initial_arrival_utc'])
                group = json.dumps([row['panel'],row['method'],row['seed'],row['budget_hours']],separators=(',',':'))
                values = {'tat_hours':row['tat_hours'],'censor_fraction':float(row['censor_flag'])}
                for endpoint in map(str,meta['resolved_config']['power']['rho_interval']):
                    values['carbon_g_per_kappa.rho_'+endpoint] = None if row['censor_flag'] else row['carbon_g_per_kappa'][endpoint]
                values['allocation_weighted_ci'] = (row['A']/row['nodehours']) if not row['censor_flag'] and row['nodehours']>0 else None
                for metric,value in values.items():
                    require(value is None or (math.isfinite(value) and value>=0),'Invalid development outcome series')
                    episode_series[(group,metric)][at].append(value)
    history = max(first['resolved_config']['execution']['history_lags_seconds'])/3600
    correlations = {key:lag_correlations(values,lags) for key,values in series.items()}
    result = {'kind':'development_dependence_diagnostics_v1','purpose':first['purpose'],
              'software':provenance(Path(__file__).resolve().parents[2]),
              'queue_contract':queue_contract(first),'probe_inputs':inputs,'episode_inputs':episode_inputs,
              'snapshot_count':len(snapshots),'start_utc':iso(min(snapshots)),'end_utc':iso(max(snapshots)),
              'feature_history_hours':history,'lag_hours':lags,'lag_correlations':correlations,
              'episode_lag_correlations':[
                  {'panel_method_seed_budget':json.loads(group),'metric':metric,'arrival_count':len(values),
                   'repeated_arrivals':sum(len(items)>1 for items in values.values()),
                   'lag_correlations':lag_correlations({at:fmean(items) if all(v is not None for v in items) else None
                                                        for at,items in values.items()},lags)}
                  for (group,metric),values in sorted(episode_series.items())],
              'series_missing_counts':{key:sum(v is None for v in values.values()) for key,values in series.items()},
              'episode_spans':{'rows':len(durations),'censored_rows':episode_censored,
                               'p95_observed_hours':atom_quantile(durations,.95) if durations else None,
                               'max_observed_hours':max(durations) if durations else None},
              'block_span_floor_hours':max(history,max(durations,default=0.)),
              'block_span_floor_is_incomplete':not durations or episode_censored>0,
              'selected_block_hours':None,'independence_certified':False,
              'scope':'exact-time pairwise Pearson lag correlations; gaps/unknown waits preserved; episode outcomes stay separate by panel/method/seed/budget, repeated arrivals averaged unless any is censored; no p-values or automatic independence cutoff',
              'next_action':'review dependence and complete episode spans before freezing a block length; this diagnostic alone does not establish independent blocks'}
    output.mkdir(parents=True)
    with (output/'series.jsonl').open('x',encoding='utf-8') as stream:
        for at,values in sorted(snapshots.items()):write_record(stream,{'arrival_utc':iso(at),**values})
    write_manifest(output/'audit.json',result)
    lines = ['# Development dependence diagnostics','',
             f"Purpose: {result['purpose']}; snapshots: {len(snapshots)}; observed episode spans: {len(durations)}.",
             f"Input history: {history:g} h; observed span floor: {result['block_span_floor_hours']:g} h.",'',
             'Lag correlations retain exact calendar gaps and unknown censored waits. A missing/constant series has no estimated correlation.',
             'The span floor is incomplete until workload episodes are available and uncensored. It is not a recommended block length.',
             'No block length was selected and independence was not certified. Review audit.json before freezing analysis settings.']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write_manifest(output/'manifest.json',{'kind':'development_dependence_diagnostics','status':'complete',
                                         'purpose':first['purpose'],'audit_sha256':digest(output/'audit.json'),
                                         'series_sha256':digest(output/'series.jsonl')})
    return result
