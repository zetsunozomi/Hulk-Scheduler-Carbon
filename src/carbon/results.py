"""Validated result views, including exact per-run fixed-policy mixtures."""

from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import fmean

from .baselines import check_references, weighted_quantile
from .common import digest, integer, load_json, require, timestamp
from .workload import Workload


class ResultRun:
    def __init__(self,path,references,allowed_splits=None):
        self.path = Path(path)
        self.manifest = load_json(self.path/'manifest.json')
        require(self.manifest['status']=='complete','Cannot analyze an incomplete/failed run')
        check_references(references,self.manifest)
        require(self.manifest['purpose']==references['purpose'],'Mixed experiment purposes')
        if allowed_splits is not None and self.manifest['purpose']=='research':
            declared = self.manifest.get('selected_splits')
            if declared is None and self.manifest.get('split'):
                declared = [self.manifest['split']]
            require(declared is not None,'Research selection requires declared result split metadata')
            require(set(declared)<=set(allowed_splits),'Result manifest contains a disallowed split (validation input already contains test results, or test input contains development results)')
        self.references = references
        require(all(math.isfinite(v) and v>0 for v in references['carbon_reference_g_per_kappa'].values()),'Invalid carbon references')
        self.provenance = {'manifest_sha256':digest(self.path/'manifest.json'),
                           'episodes_sha256':digest(self.path/'episodes.jsonl')}
        config = self.manifest['resolved_config']
        cluster = config['cluster']
        self.work = Workload(config['workload'],cluster['allowed_nodes'],cluster['max_request_seconds'],cluster['walltime_resolution_seconds'])
        self.rows = []
        with (self.path/'episodes.jsonl').open(encoding='utf-8') as stream:
            for row in map(json.loads,stream):
                if allowed_splits is not None and row['split'] not in allowed_splits:
                    require(self.manifest['purpose']!='research','Result split differs from its manifest')
                    continue
                self._validate(row)
                self.rows.append(row)
        require(self.rows,'Empty result run')

    def _validate(self,row):
        require(row['panel']==self.manifest['panel'] and row['purpose']==self.manifest['purpose'],'Result identity differs from manifest')
        config = self.manifest['resolved_config']
        start,end = map(timestamp,config['splits'][row['split']])
        arrival = timestamp(row['initial_arrival_utc'])
        require(start<=arrival<end,'Result arrival is outside its declared split')
        require(row['ci_unit']==config['ci']['unit'],'Mixed CI units')
        expected_unit = 'gCO2' if row['ci_unit']=='gCO2/kWh' else 'gCO2e'
        require(row['carbon_unit']==expected_unit,'Emission unit mismatch')
        lvalues = row['exposure_by_nodes']
        require(set(lvalues)=={str(n) for n in self.work.profiles},'Missing scale exposure')
        require(all(math.isfinite(v) and v>=0 for v in lvalues.values()),'Invalid scale exposure')
        a = sum(lvalues.values())
        b = sum(float(self.work.eta(int(n)))*v for n,v in lvalues.items())
        require(math.isclose(a,row['A'],rel_tol=1e-9,abs_tol=1e-9) and
                math.isclose(b,row['B'],rel_tol=1e-9,abs_tol=1e-9),'Exposure conservation failed')
        power = config['power']
        for rho in power['rho_interval']:
            value = power['reference_kw']*(b+rho*(a-b))
            require(math.isclose(value,row['observed_carbon_g_per_kappa'][str(rho)],rel_tol=1e-9,abs_tol=1e-9),
                    'Recorded cost differs from exposure and power model')
        complete = not row['censor_flag']
        require(complete==(row['final_status']=='completed')==bool(row['exposure_is_complete']),'Inconsistent completion flags')
        require(math.isfinite(row['observed_elapsed_hours']) and row['observed_elapsed_hours']>=0,'Invalid elapsed time')
        require(math.isfinite(row['observed_nodehours']) and row['observed_nodehours']>=0,'Invalid node-hours')
        elapsed = (timestamp(row['observed_until_utc'])-arrival).total_seconds()/3600
        require(arrival<=timestamp(row['observed_until_utc'])<=end,'Outcome crosses its split boundary')
        require(math.isclose(elapsed,row['observed_elapsed_hours'],rel_tol=1e-9,abs_tol=1e-9),'Elapsed time disagrees with timestamps')
        if complete:
            require(row['completed_updates']==self.work.total_updates and row['remaining_updates']==0,'Complete run has unfinished work')
            require(row['carbon_g_per_kappa']==row['observed_carbon_g_per_kappa'],'Complete carbon differs from observed accounting')
            require(row['tat_hours']==row['observed_elapsed_hours'] and row['nodehours']==row['observed_nodehours'],
                    'Complete time/allocation differs from observed accounting')
        else:
            require(row['carbon_g_per_kappa'] is None and row['tat_hours'] is None and row['nodehours'] is None,
                    'Censored run has a fabricated complete outcome')

    def candidate(self,method,beta,split,seed=None,kind='planner'):
        require(kind in {'fixed','planner','policy'},'Unknown candidate kind')
        if kind=='fixed':
            require(method.startswith('Fixed-') and int(method.split('-')[1]) in self.manifest.get('fixed_nodes',[]),'Not a fixed-policy run')
        elif kind=='policy':
            require(self.manifest.get('kind')=='policy_evaluation','Not a policy checkpoint evaluation')
            require(self.manifest.get('policy_rule')=='categorical_sampling','Unexpected policy evaluation rule')
        else:
            require(method in self.manifest.get('methods',[]),'Not a planning run')
        budget = beta*self.references['time_reference_hours']
        chosen = {}
        for row in self.rows:
            if row['split']!=split or row['method']!=method or (seed is not None and row['seed']!=seed):
                continue
            if kind!='fixed' and not math.isclose(row['budget_hours'],budget,rel_tol=1e-10,abs_tol=1e-10):
                continue
            require(row['episode_id'] not in chosen,'Duplicate candidate episode; do not merge repeated evaluations or seeds')
            chosen[row['episode_id']] = [(1.0,row)]
        require(chosen,f'No paired outcomes for {method}, beta={beta}, split={split}')
        require(len({parts[0][1]['seed'] for parts in chosen.values()})==1,'Mixed seeds in one candidate')
        return CandidateView(method,kind,beta,budget,chosen,randomized=kind!='fixed')

    def mixture(self,model_path,beta,split):
        model = model_path if isinstance(model_path,dict) else load_json(model_path)
        require(model['kind']=='fixed_mix_v1' and model['fit_split']=='validation','Invalid Fixed-Mix model')
        require(model['panel']==self.manifest['panel'] and model['purpose']==self.manifest['purpose'],'Fixed-Mix panel/purpose differs')
        require(model['reference_source_episodes_sha256']==self.references['source_episodes_sha256'],'Fixed-Mix references differ')
        for asset in ('trace','ci'):
            require(model['asset_sha256'][asset]==self.manifest['asset_sha256'][asset],'Fixed-Mix inputs differ')
        require(math.isclose(model['budget_hours'],beta*self.references['time_reference_hours'],rel_tol=1e-10),'Fixed-Mix budget differs')
        if split=='validation':
            require(model['source_episodes_sha256']==self.provenance['episodes_sha256'],'Fixed-Mix validation source differs')
        require(bool(model['empirical_feasible'])==(model['weights'] is not None),'Inconsistent Fixed-Mix feasibility')
        if model['weights'] is None:
            return None
        weights = model['weights']
        require(set(weights)=={str(n) for n in self.work.profiles},'Fixed-Mix scales differ')
        require(all(math.isfinite(v) and v>=0 for v in weights.values()) and math.isclose(sum(weights.values()),1,abs_tol=1e-8),'Invalid mixture simplex')
        total_weight = math.fsum(weights.values())
        weights = {n:w/total_weight for n,w in weights.items()}
        components = {n:self.candidate('Fixed-'+n,beta,split,kind='fixed') for n in weights}
        ids = set(next(iter(components.values())).parts)
        require(all(set(c.parts)==ids for c in components.values()),'Unpaired fixed-mixture outcomes')
        parts = {identity:[(weight,components[n].parts[identity][0][1]) for n,weight in weights.items() if weight>0] for identity in sorted(ids)}
        return CandidateView('Fixed-Mix','mixture',beta,model['budget_hours'],parts,randomized=False)


class CandidateView:
    def __init__(self,method,kind,beta,budget_hours,parts,randomized):
        self.method,self.kind,self.beta,self.budget_hours = method,kind,beta,budget_hours
        self.parts,self.randomized = parts,randomized
        self.arrivals = {}
        for identity,components in parts.items():
            dates = {row['initial_arrival_utc'] for _,row in components}
            require(len(dates)==1,'Mixture components have different arrivals')
            self.arrivals[identity] = dates.pop()

    def misses(self):
        result = {}
        for identity,components in self.parts.items():
            lower,upper = 0.,0.
            for weight,row in components:
                known = (row['tat_hours']>self.budget_hours) if not row['censor_flag'] else (True if row['observed_elapsed_hours']>=self.budget_hours else None)
                lower += weight*float(known is True)
                upper += weight*float(known is not False)
            result[identity] = min(1.,lower),min(1.,upper)
        return result

    def complete(self):
        return all(not row['censor_flag'] for parts in self.parts.values() for weight,row in parts if weight>0)

    def endpoint_costs(self,endpoints):
        require(self.complete(),'Full carbon is unknown for censored candidates')
        return {identity:[sum(w*r['carbon_g_per_kappa'][str(e)] for w,r in rows) for e in endpoints] for identity,rows in self.parts.items()}

    def summary(self,references):
        rows = [(w,row) for components in self.parts.values() for w,row in components]
        count = len(self.parts)
        miss = self.misses()
        result = {'method':self.method,'kind':self.kind,'budget_multiplier':self.beta,'budget_hours':self.budget_hours,
                  'episodes':count,'complete':self.complete(),'censored_probability_mass':sum(w*r['censor_flag'] for w,r in rows)/count,
                  'miss_lower':fmean(v[0] for v in miss.values()),'miss_upper':fmean(v[1] for v in miss.values()),
                  'mean_carbon_g_per_kappa':None,'mean_tat_hours':None,'p95_tat_hours':None,'mean_nodehours':None,
                  'worst_normalized_carbon':None}
        if not self.complete():
            return result
        costs = {rho:sum(w*row['carbon_g_per_kappa'][rho] for w,row in rows)/count for rho in references['carbon_reference_g_per_kappa']}
        result.update(mean_carbon_g_per_kappa=costs,mean_tat_hours=sum(w*r['tat_hours'] for w,r in rows)/count,
                      p95_tat_hours=weighted_quantile([(r['tat_hours'],w) for w,r in rows],.95),
                      mean_nodehours=sum(w*r['nodehours'] for w,r in rows)/count,
                      worst_normalized_carbon=max(v/references['carbon_reference_g_per_kappa'][rho] for rho,v in costs.items()),
                      chunk_count_fractions={str(k):sum(w for w,r in rows if (r['chunk_count']==k if k<3 else r['chunk_count']>=3))/count for k in (1,2,3)})
        return result
