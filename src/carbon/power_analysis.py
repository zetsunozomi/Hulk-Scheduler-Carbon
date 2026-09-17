"""Same-log power sensitivity and phase accounting; no scheduler or power measurements."""

from collections import defaultdict
import json
import math

from .common import require, timestamp


def mean_exposure(view):
    require(view.complete(),'Complete exposure is unknown for a censored candidate')
    totals = defaultdict(list)
    for parts in view.parts.values():
        for weight,row in parts:
            for n,value in row['exposure_by_nodes'].items():
                totals[n].append(weight*value)
    return {n:math.fsum(values)/len(view.parts) for n,values in sorted(totals.items())}


def scale_factors(work,rho):
    require(math.isfinite(rho) and 0<=rho<=1,'Power rho must be in [0,1]')
    return {str(n):rho+(1-rho)*float(work.eta(n)) for n in work.profiles}


def exposure_comparison(candidate,comparator,work,power,nominal_rho):
    require(candidate.arrivals==comparator.arrivals,'Power sensitivity requires paired cohorts')
    left,right = mean_exposure(candidate),mean_exposure(comparator)
    require(set(left)==set(right),'Scale exposure dimensions differ')
    delta = {n:left[n]-right[n] for n in left}
    da = math.fsum(delta.values())
    db = math.fsum(float(work.eta(int(n)))*value for n,value in delta.items())
    slope = da-db
    root = -db/slope if slope!=0 else None
    low,high = power['rho_interval']
    endpoint_delta = [power['reference_kw']*(db+rho*slope) for rho in (low,high)]
    factors = scale_factors(work,nominal_rho)
    nominal = math.fsum(factors[n]*v for n,v in delta.items())
    sensitivity = math.fsum(factors[n]*abs(v) for n,v in delta.items() if n!='4')
    radius = min(1.,-nominal/sensitivity) if nominal<0 and sensitivity>0 else 1. if nominal<0 else 0.
    return {'mean_exposure_candidate':left,'mean_exposure_comparator':right,'mean_exposure_difference':delta,
            'delta_A':da,'delta_B':db,'endpoint_carbon_difference_per_kappa':endpoint_delta,
            'break_even_rho':root,'break_even_in_declared_interval':root is not None and low<=root<=high,
            'nominal_rho':nominal_rho,'nominal_scale_factors':factors,
            'nominal_difference_without_common_power':nominal,'independent_scale_error_slope':sensitivity,
            'strict_improvement_error_radius_supremum':radius,'radius_endpoint_is_excluded':True,
            'fixed_reference_scale':4,'power_is_measured':False,
            'scope':'conditional algebra for frozen policies and mean exposures; uncertainty is separate, not a per-job-radius average'}


def chunk_identity(row):
    return row['episode_id'],row['method'],row['seed'],row['budget_hours']


def phase_accounting(run,view):
    require(view.complete(),'Complete phase accounting is unknown for a censored candidate')
    expected = {chunk_identity(row):row for parts in view.parts.values() for _,row in parts}
    chunks = defaultdict(list)
    with (run.path/'chunks.jsonl').open(encoding='utf-8') as stream:
        for record in map(json.loads,stream):
            identity = chunk_identity(record)
            if identity in expected:
                chunks[identity].append(record)
    require(set(chunks)==set(expected),'Missing chunk logs for selected outcomes')
    per_episode = {}
    phase_names = ('initialization','restart','training','checkpoint')
    for identity,rows in chunks.items():
        summary = expected[identity]
        rows.sort(key=lambda r:r['chunk_id'])
        require([r['chunk_id'] for r in rows]==list(range(summary['chunk_count'])),'Missing/duplicate chunk identifiers')
        nodehours = {name:defaultdict(float) for name in phase_names}
        exposure = {name:defaultdict(float) for name in phase_names}
        actions = defaultdict(float)
        for chunk in rows:
            n = str(chunk['selected_nodes']); actions[n] += 1
            key = 'exposure_node_gco2_per_kwh_hours' if chunk['ci_unit']=='gCO2/kWh' else 'exposure_node_gco2e_per_kwh_hours'
            allocation_hours = 0.
            for phase in chunk['phases']:
                name = phase['phase']
                require(name in phase_names,'Unknown allocation phase')
                hours = (timestamp(phase['end_utc'])-timestamp(phase['start_utc'])).total_seconds()/3600
                require(hours>=0 and math.isfinite(phase[key]) and phase[key]>=0,'Invalid phase interval/exposure')
                nodehours[name][n] += int(n)*hours
                exposure[name][n] += phase[key]
                allocation_hours += hours
            require(math.isclose(allocation_hours,chunk['observed_allocation_hours'],rel_tol=1e-9,abs_tol=1e-9),'Phase duration does not conserve allocation')
        by_scale = {n:sum(exposure[name].get(n,0) for name in phase_names) for n in summary['exposure_by_nodes']}
        require(all(math.isclose(value,summary['exposure_by_nodes'][n],rel_tol=1e-9,abs_tol=1e-9) for n,value in by_scale.items()),
                'Phase exposure does not conserve episode exposure')
        require(math.isclose(sum(sum(v.values()) for v in nodehours.values()),summary['nodehours'],rel_tol=1e-9,abs_tol=1e-9),
                'Phase node-hours do not conserve the episode')
        per_episode[identity] = {'nodehours':nodehours,'exposure':exposure,'actions':actions,'last_action':str(rows[-1]['selected_nodes'])}
    result = {'mean_phase_nodehours':{p:defaultdict(float) for p in phase_names},
              'mean_phase_exposure':{p:defaultdict(float) for p in phase_names},
              'mean_action_counts':defaultdict(float),'final_action_probabilities':defaultdict(float)}
    count = len(view.parts)
    for components in view.parts.values():
        for weight,row in components:
            values = per_episode[chunk_identity(row)]
            for phase in phase_names:
                for n,value in values['nodehours'][phase].items():
                    result['mean_phase_nodehours'][phase][n] += weight*value/count
                for n,value in values['exposure'][phase].items():
                    result['mean_phase_exposure'][phase][n] += weight*value/count
            for n,value in values['actions'].items():
                result['mean_action_counts'][n] += weight*value/count
            result['final_action_probabilities'][values['last_action']] += weight/count
    total = sum(sum(v.values()) for v in result['mean_phase_nodehours'].values())
    overhead = total-sum(result['mean_phase_nodehours']['training'].values())
    result.update(overhead_nodehour_fraction=overhead/total if total>0 else None,
                  scope='weighted means of already executed allocation phases; final actions describe behavior, not causal effects')
    return result


def constant_ci_rescore(phases,work,power,ci_value,rhos):
    require(math.isfinite(ci_value) and ci_value>=0,'Invalid constant CI value')
    by_scale = defaultdict(float)
    for values in phases['mean_phase_nodehours'].values():
        for n,value in values.items():
            by_scale[n] += value
    return {'ci_value':ci_value,'carbon_g_per_kappa':{str(rho):power['reference_kw']*ci_value*sum(scale_factors(work,rho)[n]*hours for n,hours in by_scale.items()) for rho in rhos},
            'scope':'constant-CI rescore of the same execution logs, not a rerun of the controller'}
