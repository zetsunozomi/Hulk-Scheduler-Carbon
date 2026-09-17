"""Public decision inputs; wait predictions are optional advice, never a gate."""

from datetime import timedelta
import hashlib
import math
from statistics import fmean

from .baselines import check_references
from .common import json_text, load_json, require, timestamp
from .diagnostics import atom_quantile
from .features import QueueFeatures
from .forecast import CausalForecast
from .waits import WaitPredictor


def shared_inputs(bundle, predictor_path, references, use_wait_predictor=True):
    if not isinstance(references, dict):
        references = load_json(references)
    check_references(references, bundle.manifest)
    if not use_wait_predictor:
        # Do not open, validate, or require a predictor artifact for the main policy.
        return references, None
    require(predictor_path is not None, 'Wait advice/planning requires --predictor')
    predictor = WaitPredictor.load(predictor_path)
    predictor.check_inputs(bundle.manifest)
    require(predictor.features.lags == bundle.raw['execution']['history_lags_seconds'], 'Predictor history differs')
    require(timestamp(predictor.artifact['train_label_boundary_utc']) <= bundle.splits['train'][1],
            'Wait model used labels beyond training split')
    require(bundle.raw['purpose'] != 'research' or predictor.artifact['purpose'] != 'synthetic',
            'Synthetic predictor cannot drive research')
    return references, predictor


class PolicyInputs:
    core_action_names = ['nodes/capacity','rate*Tref/Utotal','planned_updates/Utotal',
                         'duration/Tref','requested_limit/Tref','eta']
    advice_names = ['mean_wait/Tref','p90_wait/Tref','forecast_exposure/Lref']
    forecast_bin_hours = 6
    forecast_bins = 28

    def __init__(self, bundle, predictor, references, forecast_mode='window', wait_features='none'):
        require(forecast_mode in {'window','current'}, 'Unknown policy forecast mode')
        require(wait_features in {'none','advice'}, 'Unknown wait-feature mode')
        require(wait_features != 'advice' or predictor is not None, 'Wait advice requires a predictor')
        self.predictor = predictor if wait_features == 'advice' else None
        self.references, self.mode, self.wait_features = references, forecast_mode, wait_features
        self.features = QueueFeatures(bundle.raw['execution']['history_lags_seconds'], bundle.raw['trace']['timezone'])
        self.nodes = list(bundle.workload.profiles)
        self.capacity = bundle.raw['cluster']['nodes']
        self.forecaster = CausalForecast(bundle.ci, *bundle.splits['train'],
                                        calendar_timezone=bundle.raw['trace']['timezone'])
        self.tref = references['time_reference_hours']
        self.lref = references['carbon_reference_g_per_kappa'][str(bundle.raw['power']['rho_interval'][1])] / bundle.raw['power']['reference_kw']
        # Fixed-4 exposure / (4*Tref): a training-derived scale, not measured CI.
        self.ciref = self.lref / (4*self.tref)
        require(all(math.isfinite(v) and v > 0 for v in (self.tref,self.lref,self.ciref)), 'Invalid policy reference scales')
        history_hours = max(1.0, max(self.features.lags)/3600)
        scales = []
        for name in self.features.history_names:
            if name.endswith('.sample_age_hours'):
                scales.append(history_hours)
            elif name.endswith('.capacity'):
                scales.append(self.capacity)
            elif name.endswith('.log_count'):
                scales.append(math.log1p(self.capacity))
            elif name.endswith('_hours'):
                scales.append(math.log1p(self.tref))
            else:
                scales.append(1.0)
        self.queue_scales = scales
        self.forecast_names = [f'CI_forecast_{i*6}_{(i+1)*6}h/CIref' for i in range(self.forecast_bins)]
        self.global_names = ['remaining_updates/Utotal','remaining_budget/Tref','budget/Tref'] + self.features.history_names + self.forecast_names
        self.action_names = self.core_action_names + (self.advice_names if self.predictor is not None else [])

    @property
    def predictor_version(self):
        return self.predictor.version if self.predictor is not None else None

    def schema(self):
        return {'version':'policy_inputs_v2','global_names':self.global_names,'action_names':self.action_names,
                'nodes':self.nodes,'queue_divisors':self.queue_scales,'time_reference_hours':self.tref,
                'exposure_reference':self.lref,'ci_reference':self.ciref,'forecast_mode':self.mode,
                'wait_features':self.wait_features,'queue_feature_schema':self.features.metadata(),
                'forecast_bin_hours':self.forecast_bin_hours,'forecast_bins':self.forecast_bins,
                'normalization':'fixed positive divisors from training references and declared capacity/history; no online fitting or clipping'}

    def encode(self, observation):
        at = timestamp(observation['history'][0]['state']['timestamp_utc'])
        forecast = self.forecaster.issue(at, self.mode)
        queue = self.features.history(observation['history'], at)
        total = observation['total_updates']
        ci = [forecast.integral(at+timedelta(hours=i*self.forecast_bin_hours),
                                at+timedelta(hours=(i+1)*self.forecast_bin_hours)) /
              (self.forecast_bin_hours*self.ciref) for i in range(self.forecast_bins)]
        global_vector = [observation['remaining_updates']/total, observation['remaining_budget_hours']/self.tref,
                         observation['budget_hours']/self.tref] + [v/s for v,s in zip(queue,self.queue_scales)] + ci
        require([a['nodes'] for a in observation['actions']] == self.nodes, 'Policy action order changed')
        actions, mask, descriptors = [], [], []
        for item in observation['actions']:
            n, feasible = item['nodes'], item['feasible']
            mask.append(feasible)
            if not feasible:
                actions.append([0.0]*len(self.action_names))
                descriptors.append({'nodes':n,'feasible':False})
                continue
            values = [n/self.capacity,item['updates_per_hour']*self.tref/total,
                      item['planned_updates']/total,item['actual_seconds']/3600/self.tref,
                      item['requested_seconds']/3600/self.tref,item['eta']]
            details = {'nodes':n,'feasible':True,'planned_updates':item['planned_updates'],
                       'actual_seconds':item['actual_seconds'],'requested_seconds':item['requested_seconds']}
            if self.predictor is not None:
                atoms = self.predictor.atoms(observation['history'], at, n, item['requested_seconds'])
                exposure = n * fmean(forecast.integral(at+timedelta(hours=w),
                                                      at+timedelta(hours=w, seconds=item['actual_seconds'])) for w in atoms)
                mean, p90 = fmean(atoms), atom_quantile(atoms,.9)
                values += [mean/self.tref,p90/self.tref,exposure/self.lref]
                details.update(mean_wait_hours=mean,p90_wait_hours=p90,forecast_exposure=exposure)
            actions.append(values); descriptors.append(details)
        require(any(mask), 'No feasible policy action')
        require(all(math.isfinite(v) for v in global_vector + [v for row in actions for v in row]), 'Nonfinite policy input')
        encoded = {'global':global_vector,'actions':actions,'mask':mask}
        metadata = {**forecast.metadata(),'predictor_version':self.predictor_version,'policy_wait_features':self.wait_features,
                    'policy_input_sha256':hashlib.sha256(json_text(encoded).encode()).hexdigest(),
                    'action_descriptors':descriptors}
        return encoded, metadata
