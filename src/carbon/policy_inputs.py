"""Causal policy inputs and frozen, training-reference scaling; no torch needed."""

from datetime import timedelta
import hashlib
import math
from statistics import fmean

from .baselines import check_references
from .common import json_text, load_json, require, timestamp
from .diagnostics import atom_quantile
from .forecast import CausalForecast
from .waits import WaitPredictor


def shared_inputs(bundle, predictor_path, references):
    if not isinstance(references, dict):
        references = load_json(references)
    check_references(references, bundle.manifest)
    predictor = WaitPredictor.load(predictor_path)
    predictor.check_inputs(bundle.manifest)
    require(predictor.features.lags == bundle.raw['execution']['history_lags_seconds'], 'Predictor history differs')
    require(timestamp(predictor.artifact['train_label_boundary_utc']) <= bundle.splits['train'][1],
            'Wait model used labels beyond training split')
    require(bundle.raw['purpose'] != 'research' or predictor.artifact['purpose'] != 'synthetic',
            'Synthetic predictor cannot drive research')
    return references, predictor


class PolicyInputs:
    action_names = ['mean_wait/Tref','p90_wait/Tref','rate*Tref/Utotal','planned_updates/Utotal',
                    'duration/Tref','eta','forecast_exposure/Lref']

    def __init__(self, bundle, predictor, references, forecast_mode='window'):
        require(forecast_mode in {'window','current'}, 'Unknown policy forecast mode')
        self.predictor, self.references, self.mode = predictor, references, forecast_mode
        self.nodes = list(bundle.workload.profiles)
        self.forecaster = CausalForecast(bundle.ci, *bundle.splits['train'],
                                        calendar_timezone=bundle.raw['trace']['timezone'])
        self.tref = references['time_reference_hours']
        self.lref = references['carbon_reference_g_per_kappa'][str(bundle.raw['power']['rho_interval'][1])] / bundle.raw['power']['reference_kw']
        require(all(math.isfinite(v) and v > 0 for v in (self.tref,self.lref)), 'Invalid policy reference scales')
        capacity = bundle.raw['cluster']['nodes']
        history_hours = max(1.0, max(predictor.features.lags)/3600)
        scales = []
        for name in predictor.features.history_names:
            if name.endswith('.sample_age_hours'):
                scales.append(history_hours)
            elif name.endswith('.capacity'):
                scales.append(capacity)
            elif name.endswith('.log_count'):
                scales.append(math.log1p(capacity))
            elif name.endswith('_hours'):
                scales.append(math.log1p(self.tref))
            else:
                scales.append(1.0)
        self.queue_scales = scales
        self.global_names = ['remaining_updates/Utotal','remaining_budget/Tref','budget/Tref'] + predictor.features.history_names

    def schema(self):
        return {'version':'policy_inputs_v1','global_names':self.global_names,'action_names':self.action_names,
                'nodes':self.nodes,'queue_divisors':self.queue_scales,'time_reference_hours':self.tref,
                'exposure_reference':self.lref,'forecast_mode':self.mode,
                'normalization':'fixed positive divisors from training references and declared capacity/history; no online fitting or clipping'}

    def encode(self, observation):
        at = timestamp(observation['history'][0]['state']['timestamp_utc'])
        forecast = self.forecaster.issue(at, self.mode)
        queue = self.predictor.features.history(observation['history'], at)
        total = observation['total_updates']
        global_vector = [observation['remaining_updates']/total, observation['remaining_budget_hours']/self.tref,
                         observation['budget_hours']/self.tref] + [v/s for v,s in zip(queue,self.queue_scales)]
        require([a['nodes'] for a in observation['actions']] == self.nodes, 'Policy action order changed')
        actions, mask, predictions = [], [], []
        for item in observation['actions']:
            n, feasible = item['nodes'], item['feasible']
            mask.append(feasible)
            if not feasible:
                actions.append([0.0]*len(self.action_names))
                predictions.append({'nodes':n,'feasible':False})
                continue
            atoms = self.predictor.atoms(observation['history'], at, n, item['requested_seconds'])
            exposure = n * fmean(forecast.integral(at+timedelta(hours=w),
                                                  at+timedelta(hours=w, seconds=item['actual_seconds'])) for w in atoms)
            mean, p90 = fmean(atoms), atom_quantile(atoms,.9)
            actions.append([mean/self.tref,p90/self.tref,item['updates_per_hour']*self.tref/total,
                            item['planned_updates']/total,item['actual_seconds']/3600/self.tref,
                            item['eta'],exposure/self.lref])
            predictions.append({'nodes':n,'feasible':True,'mean_wait_hours':mean,'p90_wait_hours':p90,
                                'forecast_exposure':exposure,'planned_updates':item['planned_updates'],
                                'actual_seconds':item['actual_seconds'],'requested_seconds':item['requested_seconds']})
        require(any(mask), 'No feasible policy action')
        require(all(math.isfinite(v) for v in global_vector + [v for row in actions for v in row]), 'Nonfinite policy input')
        encoded = {'global':global_vector,'actions':actions,'mask':mask}
        metadata = {**forecast.metadata(),'predictor_version':self.predictor.version,
                    'policy_input_sha256':hashlib.sha256(json_text(encoded).encode()).hexdigest(),
                    'action_predictions':predictions}
        return encoded, metadata
