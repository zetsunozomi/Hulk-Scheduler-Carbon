from copy import deepcopy
from dataclasses import replace
from decimal import Decimal
import importlib.util
import random
import unittest

from carbon.common import ContractError
from carbon.config import Bundle
from carbon.environment import Environment, initial_replay
from carbon.features import QueueFeatures
from carbon.policy_inputs import PolicyInputs
from tests.helpers import ROOT

HAS_TORCH = importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    from carbon.learning import ActorCritic, initial_dual, input_tensors, monte_carlo_costs, optimize_ppo, ppo_losses, sample_action, torch, update_dual


class ConstantWaits:
    version = 'synthetic-test-predictor'
    def __init__(self,lags): self.features=QueueFeatures(lags)
    def atoms(self,history,at,nodes,requested_seconds): return [0,.25,.5,.75]


class PolicyInputTests(unittest.TestCase):
    def test_future_ci_and_hidden_fields_cannot_change_actor_or_critic_inputs(self):
        bundle = Bundle(ROOT/'configs/synthetic-p2.json')
        episode=bundle.episodes[0];env=Environment(bundle,episode,initial_replay(bundle,episode),'test')
        observation=env.observe();at=env.replay.time
        predictor=ConstantWaits(bundle.raw['execution']['history_lags_seconds'])
        refs={'time_reference_hours':1,'carbon_reference_g_per_kappa':{'0.25':100,'1.0':100}}
        before=PolicyInputs(bundle,predictor,refs).encode(observation)
        changed=deepcopy(bundle)
        changed.ci.records=[replace(r,value=Decimal(99999)) if r.available_at>at or r.end>at else r for r in changed.ci.records]
        altered=deepcopy(observation);altered['hidden_future_cost']=999
        for h in altered['history']:
            if h['state']:
                for job in h['state']['pending']+h['state']['running']:
                    job['actual_duration']=999999;job['future_end']='2099-01-01'
        after=PolicyInputs(changed,predictor,refs).encode(altered)
        self.assertEqual(before,after)

    def test_current_ci_changes_only_exposure_descriptor_and_keeps_masks(self):
        bundle=Bundle(ROOT/'configs/synthetic-p2.json');episode=bundle.episodes[0]
        env=Environment(bundle,episode,initial_replay(bundle,episode),'test');obs=env.observe()
        obs['actions'][0]['feasible']=False
        predictor=ConstantWaits(bundle.raw['execution']['history_lags_seconds'])
        refs={'time_reference_hours':1,'carbon_reference_g_per_kappa':{'0.25':100,'1.0':100}}
        full,_=PolicyInputs(bundle,predictor,refs).encode(obs)
        current,_=PolicyInputs(bundle,predictor,refs,'current').encode(obs)
        self.assertEqual(full['global'],current['global']);self.assertEqual(full['mask'],[False,True,True,True])
        self.assertEqual(full['actions'][0],[0.0]*7)
        for a,b in zip(full['actions'][1:],current['actions'][1:]):
            self.assertEqual(a[:-1],b[:-1]);self.assertNotEqual(a[-1],b[-1])


@unittest.skipUnless(HAS_TORCH,'requires P3 PyTorch dependency')
class LearningTests(unittest.TestCase):
    def setUp(self): torch.set_num_threads(1);torch.manual_seed(11)

    def test_returns_are_undiscounted_and_terminal_miss_occurs_once(self):
        returns=monte_carlo_costs([[1,2],[3,4],[5,6]],True)
        self.assertEqual(returns,[[9,12,1.0],[8,10,1.0],[5,6,1.0]])
        self.assertEqual(monte_carlo_costs([[1,2]],False),[[1,2,0.0]])
        with self.assertRaises(ContractError):monte_carlo_costs([[1,2]],None)

    def test_dual_weights_follow_expected_episode_cost_and_miss_constraint(self):
        initial=initial_dual('robust')
        changed=update_dual(initial,[1,3,.2],.05)
        self.assertGreater(changed['weights'][1],changed['weights'][0])
        self.assertAlmostEqual(changed['weights'][1]/changed['weights'][0],__import__('math').exp(.1))
        self.assertAlmostEqual(changed['lambda'],1+.05*.15)
        self.assertEqual(initial['weights'],[.5,.5])
        large=deepcopy(initial);large['lambda']=1e6
        self.assertGreater(update_dual(large,[1,1,1],0)['lambda'],1e6)
        zero=deepcopy(initial);zero['lambda']=0
        self.assertEqual(update_dual(zero,[1,1,0],1)['lambda'],0)
        single=initial_dual('lower')
        self.assertEqual(update_dual(single,[1,100,.2],.05,objective='lower')['weights'],[1,0])

    def test_surrogate_gradient_averages_episodes_not_chunks(self):
        new=torch.zeros(3,requires_grad=True)
        loss,actor,_,_=ppo_losses(new,torch.zeros(3),torch.ones(3),torch.zeros(3,3),torch.zeros(3,3),torch.zeros(3),2)
        self.assertAlmostEqual(actor.item(),-1.5)
        loss.backward();self.assertTrue(torch.allclose(new.grad,torch.tensor([-.5,-.5,-.5])))
        # A higher cost has negative reward advantage and lowers its action probability.
        logits=torch.tensor([0.,0.],requires_grad=True)
        selected=torch.log_softmax(logits,0)[0:1]
        loss,*_=ppo_losses(selected,selected.detach(),torch.tensor([-2.]),torch.zeros(1,3),torch.zeros(1,3),torch.zeros(1),1)
        loss.backward();self.assertGreater(logits.grad[0].item(),0)

    def test_clipping_stops_gradients_outside_the_improving_region(self):
        new=torch.tensor([__import__('math').log(2),__import__('math').log(.5)],requires_grad=True)
        loss,*_=ppo_losses(new,torch.zeros(2),torch.tensor([1.,-1.]),torch.zeros(2,3),torch.zeros(2,3),torch.zeros(2),2)
        loss.backward();self.assertTrue(torch.equal(new.grad,torch.zeros(2)))

    def test_categorical_sampling_masks_and_independent_critic(self):
        model=ActorCritic(2,3)
        for p in model.parameters():torch.nn.init.zeros_(p)
        encoded={'global':[0.,1.],'actions':[[1.,2.,3.]]*4,'mask':[True,False,True,True]}
        generator=torch.Generator().manual_seed(5)
        selected=[sample_action(model,encoded,generator)[0] for _ in range(40)]
        self.assertEqual(set(selected),{0,2,3})
        distribution,values=model(*input_tensors([encoded]))
        self.assertEqual(distribution.probs[0,1].item(),0)
        self.assertEqual(values.shape,(1,3));self.assertEqual(values[0,2].item(),.5)
        distribution.log_prob(torch.tensor([0])).sum().backward()
        self.assertTrue(all(p.grad is None for name,p in model.named_parameters() if name.startswith('critic_')))
        encoded['mask']=[False]*4
        with self.assertRaises(ContractError):model(*input_tensors([encoded]))

    def test_optimizer_learns_lower_cost_on_a_known_one_decision_problem(self):
        from carbon.policy_runner import settings_for
        model=ActorCritic(2,3,width=16)
        encoded={'global':[0.,1.],'actions':[[float(n),0.,0.] for n in range(4)],'mask':[True]*4}
        def expected_cost():
            with torch.no_grad():
                d,_=model(*input_tensors([encoded]))
                return float((d.probs[0]*torch.tensor([1.,2.,3.,4.])).sum())
        before=expected_cost();optimizer=torch.optim.Adam(model.parameters(),lr=.003)
        generator=torch.Generator().manual_seed(5);rng=random.Random(5)
        settings=settings_for(10,budgets=[1.],episodes_per_budget=64,epochs=2,minibatch_episodes=32,entropy=0.)
        duals={'1.0':initial_dual('robust')}
        for _ in range(10):
            episodes=[]
            for _ in range(64):
                action,logp,values,_=sample_action(model,encoded,generator)
                step={'input':encoded,'action':action,'log_probability':logp,'values':values}
                episodes.append({'beta':1.,'steps':[step],'returns':[[action+1.,action+1.,0.]]})
            optimize_ppo(model,optimizer,episodes,duals,settings,rng)
        self.assertLess(expected_cost(),before-.25)
        self.assertEqual(duals,{'1.0':initial_dual('robust')})
