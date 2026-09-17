from copy import deepcopy
from dataclasses import replace
from decimal import Decimal
import importlib.util
import io
import json
import unittest
from unittest.mock import patch

from carbon.common import ContractError
from carbon.config import Bundle
from carbon.environment import Environment,initial_replay
from carbon.policy_inputs import PolicyInputs
from carbon.slider import slider_contract,budget_at_position
from tests.helpers import ROOT


class SliderTests(unittest.TestCase):
    def test_slider_changes_physical_budget_and_never_interpolates_performance(self):
        mapping=slider_contract(24,[1,1.25,1.5,2])
        self.assertEqual([t['position'] for t in mapping['ticks']],[0,.25,.5,1])
        self.assertEqual([t['budget_hours'] for t in mapping['ticks']],[24,30,36,48])
        self.assertEqual(budget_at_position(.25,[1,1.25,1.5,2]),1.25)
        for bad in (.4,-1,float('nan')):
            with self.assertRaises(ContractError):budget_at_position(bad,[1,1.25,1.5,2])
        self.assertEqual(slider_contract(24,[1.5])['ticks'][0]['position'],0)
        with self.assertRaises(ContractError):slider_contract(24,[1,1])


@unittest.skipUnless(importlib.util.find_spec('torch'),'requires torch')
class PrecommitTests(unittest.TestCase):
    def setUp(self):
        self.bundle=Bundle(ROOT/'configs/synthetic-p2.json');self.episode=self.bundle.episodes[0]
        self.base=initial_replay(self.bundle,self.episode)
        self.refs={'time_reference_hours':1,'carbon_reference_g_per_kappa':{'0.25':100,'1.0':100}}
        self.encoder=PolicyInputs(self.bundle,None,self.refs)
        self.fixed4=lambda *args:(0,0.,[0.,0.,0.],[1.,0.,0.,0.])

    def test_plan_is_independent_of_future_ci_and_uses_initial_history_only(self):
        from carbon.precommit import build_precommitted_plan
        obs=Environment(self.bundle,self.episode,self.base,'test').observe()
        with patch('carbon.precommit.sample_action',side_effect=self.fixed4):
            before,record=build_precommitted_plan(self.bundle,obs,self.encoder,None,None)
            changed=deepcopy(self.bundle);at=self.episode.arrival
            changed.ci.records=[replace(r,value=Decimal(99999)) if r.available_at>at else r for r in changed.ci.records]
            other_encoder=PolicyInputs(changed,None,self.refs)
            after,other=build_precommitted_plan(changed,obs,other_encoder,None,None)
        self.assertEqual(record,other);self.assertGreater(len(before),1)
        self.assertEqual([x['input'] for x in before],[x['input'] for x in after])
        self.assertEqual(sum(x['planned_updates'] for x in before),self.bundle.workload.total_updates)
        self.assertTrue(all(x['input']['global'][2:]==before[0]['input']['global'][2:] for x in before))
        self.assertGreater(before[0]['input']['global'][1],before[-1]['input']['global'][1])

    def test_complete_plan_is_archived_before_execution_and_no_later_observation_is_read(self):
        from carbon.policy_runner import collect_episode
        chunks,plans=io.StringIO(),io.StringIO();observe=Environment.observe;step=Environment.step
        observations=[]
        def once(env):
            observations.append(len(env.chunks))
            self.assertEqual(len(observations),1,'Precommitted policy observed a later admission')
            return observe(env)
        def execute(env,n):
            record=json.loads(plans.getvalue())
            self.assertEqual(len(record['steps']),4)
            self.assertEqual(record['steps'][len(env.chunks)]['nodes'],n)
            return step(env,n)
        with patch('carbon.precommit.sample_action',side_effect=self.fixed4), \
             patch.object(Environment,'observe',new=once),patch.object(Environment,'step',new=execute):
            steps,returns,summary=collect_episode(self.bundle,self.episode,None,self.encoder,None,self.base,
                                                 'Precommitted-RL',11,chunks,{},'precommitted',plans)
        self.assertEqual(observations,[0]);self.assertEqual(summary['final_status'],'completed')
        self.assertIsNotNone(returns);self.assertEqual(len(steps),len(summary['precommitted_nodes']))
        self.assertEqual(summary['precommitted_nodes'],[4]*4)
        self.assertTrue(all(json.loads(line)['precommitted_plan_sha256']==summary['precommitted_plan_sha256']
                            for line in chunks.getvalue().splitlines()))
