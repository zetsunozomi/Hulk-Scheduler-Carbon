from copy import deepcopy
from datetime import timedelta
from decimal import Decimal
import unittest

from carbon.carbon import CarbonSeries, CIRecord
from carbon.environment import Environment, initial_replay
from carbon.forecast import FrozenForecast
from carbon.planning import PlanningPolicy, RolloutPlanner, past_cost_estimate
from tests.helpers import T0,bundle


class KnownWaits:
    version='synthetic-analytical-test'
    def __init__(self,delays=None):self.delays=delays or {};self.calls=0
    def atoms(self,history,at,nodes,requested):
        self.calls+=1
        return self.delays.get(nodes,[0.0])


class KnownForecast:
    def __init__(self,ci):self.calls=0;self.ci=ci
    def issue(self,at):
        self.calls+=1
        return FrozenForecast(at,(100.0,)*168,'UTC',timedelta(0),'synthetic-constant','window',1,at)


class PlanningTests(unittest.TestCase):
    def setup_env(self):
        b=bundle();e=b.episodes[0]
        return b,Environment(b,e,initial_replay(b,e),'test')

    def planner(self,b,wait=None,blind=False):
        refs={'carbon_reference_g_per_kappa':{'0.25':100.,'1.0':100.}}
        return RolloutPlanner(b.workload,b.raw['power'],wait or KnownWaits(),refs,paths=8,miss_tolerance=0,queue_blind=blind)

    def test_complete_work_no_queue_loose_budget_prefers_fixed4(self):
        b,env=self.setup_env();obs=env.observe();obs['remaining_budget_hours']=100
        forecast=KnownForecast(b.ci).issue(env.replay.time)
        result=self.planner(b).decide(obs,env.replay.time,forecast,True)
        self.assertEqual(set(result.sequence),{4})
        self.assertTrue(result.estimated_feasible)
        self.assertGreater(result.candidate_plans,4)
        self.assertGreater(len(result.sequence),2)  # tail must run all the way to completion
        total_time=0;left=b.workload.total_updates
        for i,n in enumerate(result.sequence):
            chunk=b.workload.plan(n,left,i==0);left-=chunk.updates;total_time+=chunk.actual.total_seconds()/3600
        self.assertEqual(left,0)
        self.assertAlmostEqual(result.endpoint_costs[1],4*1.5*100*total_time)

    def test_tight_budget_selects_fast_scale_and_does_not_terminate_at_deadline(self):
        b,env=self.setup_env();obs=env.observe();obs['remaining_budget_hours']=.1
        forecast=KnownForecast(b.ci).issue(env.replay.time)
        result=self.planner(b).decide(obs,env.replay.time,forecast,True)
        self.assertEqual(result.sequence,(32,));self.assertEqual(result.estimated_miss,0)
        obs['remaining_budget_hours']=-1
        result=self.planner(b).decide(obs,env.replay.time,forecast,True)
        self.assertEqual(result.estimated_miss,1)
        self.assertFalse(result.estimated_feasible);self.assertGreater(len(result.sequence),0)

    def test_queue_blind_changes_planning_without_changing_environment(self):
        b,env=self.setup_env();before=env.observe();obs=deepcopy(before);obs['remaining_budget_hours']=1
        waits=KnownWaits({4:[10.],8:[10.],16:[10.]});forecast=KnownForecast(b.ci).issue(env.replay.time)
        informed=self.planner(b,waits).decide(obs,env.replay.time,forecast,True)
        blind=self.planner(b,waits,True).decide(obs,env.replay.time,forecast,True)
        self.assertEqual(informed.sequence,(32,));self.assertEqual(set(blind.sequence),{4})
        self.assertEqual(env.observe(),before)

    def test_plan_once_does_not_refresh_forecast_or_wait_model(self):
        b,env=self.setup_env();forecast=KnownForecast(b.ci);waits=KnownWaits()
        policy=PlanningPolicy(self.planner(b,waits),forecast,'plan-once')
        first,meta=policy.choose(env.observe());self.assertTrue(meta['replanned'])
        policy.record_execution(env.step(first));calls=waits.calls
        self.assertEqual(env.status,'running')
        second,meta=policy.choose(env.observe())
        self.assertFalse(meta['replanned']);self.assertEqual(forecast.calls,1);self.assertEqual(waits.calls,calls)
        self.assertEqual(meta['decision_planning_seconds'],0)

    def test_past_cost_never_reads_unreleased_realized_ci(self):
        b,_=self.setup_env();forecast=KnownForecast(b.ci).issue(T0)
        executions=[dict(nodes=4,start=T0,end=T0+timedelta(hours=1),forecast=forecast)]
        def ci(value):return CarbonSeries([CIRecord(T0,T0+timedelta(hours=1),Decimal(value),T0+timedelta(days=1))], 'test','test')
        a=past_cost_estimate(executions,ci(1),b.workload,b.raw['power'],T0+timedelta(hours=2))
        z=past_cost_estimate(executions,ci(999999),b.workload,b.raw['power'],T0+timedelta(hours=2))
        self.assertEqual(a,z);self.assertEqual(a,[600.,600.])
        released=past_cost_estimate(executions,ci(1),b.workload,b.raw['power'],T0+timedelta(days=1))
        self.assertEqual(released,[6.,6.])

    def test_single_final_chunk_deduplicates_unused_plan_positions(self):
        b,env=self.setup_env();obs=env.observe();obs['remaining_updates']=1
        result=self.planner(b).decide(obs,env.replay.time,KnownForecast(b.ci).issue(env.replay.time),False)
        self.assertEqual(result.candidate_plans,4);self.assertEqual(len(result.sequence),1)
