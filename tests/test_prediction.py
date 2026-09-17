from copy import deepcopy
from datetime import timedelta
import importlib.util
import unittest

from carbon.common import ContractError, iso
from carbon.features import QueueFeatures
from carbon.waits import chronological_folds, export_regressor, predict_regressor
from carbon.environment import initial_replay
from tests.helpers import T0, bundle


class FeatureTests(unittest.TestCase):
    def test_hidden_fields_are_ignored_and_request_length_is_explicit(self):
        b = bundle();base=initial_replay(b,b.episodes[0]);lags=b.raw['execution']['history_lags_seconds']
        history=base.visible_history(lags);features=QueueFeatures(lags)
        baseline=features.request(history,base.time,4,60)
        edited=deepcopy(history)
        for sample in edited:
            if sample['state']:
                sample['state']['future_realized_ci']=99999
                for job in sample['state']['pending']+sample['state']['running']:
                    job['actual_duration']=99999;job['end_time']='2099-01-01T00:00:00Z'
        self.assertEqual(baseline,features.request(edited,base.time,4,60))
        other=features.request(history,base.time,4,120)
        self.assertEqual(baseline[:-1],other[:-1]);self.assertNotEqual(baseline[-1],other[-1])
        self.assertEqual(len(baseline),len(features.names))

    def test_chronological_folds_purge_late_labels_and_group_ties(self):
        rows=[]
        for hour in range(8):
            for nodes in (4,8):
                arrival=T0+timedelta(hours=hour)
                observed=arrival+timedelta(hours=3 if hour==1 else 0)
                rows.append(dict(arrival_utc=iso(arrival), label_observed_at_utc=iso(observed),nodes=nodes))
        folds=list(chronological_folds(rows,3))
        first_train,first_test,begin=folds[0]
        self.assertEqual(begin,T0+timedelta(hours=2))
        self.assertEqual(first_train,[0,1])
        for training,validation,begin in folds:
            train_times={rows[i]['arrival_utc'] for i in training}
            val_times={rows[i]['arrival_utc'] for i in validation}
            self.assertTrue(train_times.isdisjoint(val_times))
            self.assertTrue(all(rows[i]['label_observed_at_utc']<iso(begin) for i in training))

    @unittest.skipUnless(importlib.util.find_spec('sklearn'), 'requires P2 fitting dependencies')
    def test_export_matches_sklearn_including_float32_thresholds(self):
        import numpy as np
        from sklearn.ensemble import GradientBoostingRegressor
        rng=np.random.default_rng(13);x=rng.normal(size=(100,6));y=x[:,0]**2+3*x[:,1]
        model=GradientBoostingRegressor(n_estimators=12,max_depth=3,random_state=1).fit(x,y)
        exported=export_regressor(model)
        self.assertTrue(np.allclose([predict_regressor(exported,v) for v in x],model.predict(x),atol=1e-12))
        threshold=model.estimators_[0,0].tree_.threshold[0];feature=model.estimators_[0,0].tree_.feature[0]
        for delta in (-1e-10,0,1e-10):
            row=x[0].copy();row[feature]=threshold+delta
            self.assertAlmostEqual(predict_regressor(exported,row),model.predict(row[None,:])[0],places=10)
