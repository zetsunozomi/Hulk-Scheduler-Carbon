from contextlib import redirect_stdout
from copy import deepcopy
from datetime import timedelta
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from carbon.__main__ import main
from carbon.baselines import make_references
from carbon.common import ContractError,json_text,load_json
from carbon.config import Bundle
from carbon.environment import Environment
from carbon.features import QueueFeatures
from carbon.runner import run_fixed
from carbon.waits import queue_contract
from tests.helpers import ROOT

HAS_TORCH=importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    from carbon.policy_runner import evaluate_policy,read_checkpoint,save_checkpoint,settings_for,train_policy,torch


@unittest.skipUnless(HAS_TORCH,'requires P3 PyTorch dependency')
class PolicyRunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.bundle=Bundle(ROOT/'configs/synthetic-p2.json')
        self.predictor=self.root/'waits.json';features=QueueFeatures(self.bundle.raw['execution']['history_lags_seconds'])
        artifact={'kind':'wait_distribution_v1','feature_schema':features.metadata(),
                  'regressor':{'initial':0.,'learning_rate':.05,'trees':[],'features':len(features.names),'input_dtype':'float32'},
                  'residual_atoms':{str(n):[0.]*32 for n in self.bundle.workload.profiles},
                  'purpose':'synthetic','trace_sha256':self.bundle.manifest['asset_sha256']['trace'],
                  'train_label_boundary_utc':self.bundle.raw['splits']['train'][1],
                  'queue_contract':queue_contract(self.bundle.manifest)}
        self.predictor.write_text(json_text(artifact));self.references=self.root/'refs.json'
        with redirect_stdout(io.StringIO()):
            run_fixed(self.bundle,self.root/'fixed',[4,8,16,32])
        make_references(self.root/'fixed',self.references)
        self.settings=settings_for(2,budgets=[1,2],episodes_per_budget=2,epochs=2,minibatch_episodes=2)

    def run_train(self,name,settings=None,resume=None):
        with redirect_stdout(io.StringIO()):
            return train_policy(self.bundle,self.root/name,self.predictor,self.references,settings or self.settings,resume)

    def test_frozen_environment_run_preserves_source_and_artifact_guards(self):
        from carbon.stress import derive_config, run_stress
        from carbon.waits import WaitPredictor
        raw = deepcopy(self.bundle.raw); raw['root'] = str(ROOT)
        raw['trace']['role'] = 'workload_template'
        raw['execution']['initial_state_mode'] = 'empty_warmup'
        config = self.root/'source.json'; config.write_text(json_text(raw))
        self.bundle = Bundle(config)
        artifact = load_json(self.predictor)
        artifact['queue_contract'] = queue_contract(self.bundle.manifest)
        self.predictor.write_text(json_text(artifact))
        with redirect_stdout(io.StringIO()):
            run_fixed(self.bundle,self.root/'source-fixed',[4,8,16,32])
        self.references = self.root/'source-refs.json'
        make_references(self.root/'source-fixed',self.references)
        self.run_train('source-training')
        checkpoint=self.root/'source-training/checkpoint-000002.json'
        before=config.read_bytes()
        with redirect_stdout(io.StringIO()):
            result=run_stress(self.bundle,self.root/'stress',self.predictor,checkpoint,'fcfs',1.,split='validation',paths=2)
        self.assertEqual(result['status'],'complete')
        self.assertEqual(result['completed_episode_methods'],2)
        rows=[json.loads(s) for s in (self.root/'stress/episodes.jsonl').read_text().splitlines()]
        self.assertEqual({r['method'] for r in rows},{'ScaleDown','Rollout-MPC'})
        self.assertTrue(all(r['completed_updates']==100 for r in rows))
        self.assertEqual(config.read_bytes(),before)
        changed=Bundle(self.root/'stress/scenario-config.json')
        with self.assertRaises(ContractError):
            WaitPredictor.load(self.predictor).check_inputs(changed.manifest)
        self.assertEqual(result['feature_schema'],load_json(checkpoint)['feature_schema'])
        for variant,total in [('overhead60',60),('overhead1800',1800)]:
            derived=derive_config(self.bundle,variant)
            for profile in derived['workload']['profiles'].values():
                self.assertEqual(profile['restart_seconds']+profile['checkpoint_seconds'],total)
                self.assertEqual(profile['initialization_seconds'],profile['restart_seconds'])
        width=derive_config(self.bundle,'width2')
        self.assertEqual(width['trace']['node_multiplier'],2)
        self.assertEqual(width['trace']['oversize_policy'],'cap')
        self.assertEqual(width['workload'],self.bundle.raw['workload'])

    def test_end_to_end_and_reproducible_categorical_evaluation(self):
        result=self.run_train('training')
        self.assertEqual(result['status'],'complete');self.assertEqual(result['total_episodes'],8)
        history=[json.loads(line) for line in (self.root/'training/training.jsonl').read_text().splitlines()]
        self.assertEqual(history[1]['duals_before'],history[0]['duals_after'])
        rows=[json.loads(line) for line in (self.root/'training/episodes.jsonl').read_text().splitlines()]
        self.assertTrue(all(r['split']=='train' and r['completed_updates']==100 for r in rows))
        checkpoint=self.root/'training/checkpoint-000002.json'
        for name in ('a','b'):
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(['run-policy','--config',str(ROOT/'configs/synthetic-p2.json'),
                                      '--predictor',str(self.predictor),'--checkpoint',str(checkpoint),'--output',str(self.root/name)]),0)
        self.assertEqual((self.root/'a/episodes.jsonl').read_bytes(),(self.root/'b/episodes.jsonl').read_bytes())
        evaluation=[json.loads(line) for line in (self.root/'a/episodes.jsonl').read_text().splitlines()]
        self.assertEqual(len(evaluation),2)
        self.assertTrue(all(r['completed_updates']==100 and r['policy_rule']=='categorical_sampling' for r in evaluation))
        with self.assertRaisesRegex(ContractError,'trained grid'):
            evaluate_policy(self.bundle,self.root/'bad-budget',self.predictor,checkpoint,budgets=[10])

    def test_resume_matches_uninterrupted_optimizer_model_and_sampling(self):
        settings=settings_for(3,budgets=[1,2],episodes_per_budget=2,epochs=1,minibatch_episodes=2)
        self.run_train('full',settings)
        def interrupted(*args,**kwargs):
            path=save_checkpoint(*args,**kwargs)
            raise RuntimeError('synthetic interruption after complete checkpoint')
        with patch('carbon.policy_runner.save_checkpoint',side_effect=interrupted):
            with self.assertRaisesRegex(RuntimeError,'synthetic interruption'):
                self.run_train('stopped',settings)
        self.assertEqual(load_json(self.root/'stopped/manifest.json')['status'],'failed')
        resume=self.root/'stopped/checkpoint-000001.json'
        self.run_train('resumed',settings,resume)
        a,state_a=read_checkpoint(self.root/'full/checkpoint-000003.json')
        b,state_b=read_checkpoint(self.root/'resumed/checkpoint-000003.json')
        self.assertEqual(a['duals'],b['duals']);self.assertEqual(a['arrival_shuffle_rng'],b['arrival_shuffle_rng'])
        for key in state_a['model']:
            self.assertTrue(torch.equal(state_a['model'][key],state_b['model'][key]),key)
        for key,row in state_a['optimizer']['state'].items():
            for name,value in row.items():
                other=state_b['optimizer']['state'][key][name]
                self.assertTrue(torch.equal(value,other) if isinstance(value,torch.Tensor) else value==other)
        self.assertTrue(torch.equal(state_a['sampling_rng'],state_b['sampling_rng']))
        bad=deepcopy(settings);bad['budgets']=[1.]
        with self.assertRaisesRegex(ContractError,'settings changed'):
            self.run_train('changed',bad,resume)

    def test_censored_episode_is_logged_and_aborts_before_optimization(self):
        step=Environment.step
        def truncated(env,nodes):
            env.boundary=env.replay.time+timedelta(seconds=1)
            return step(env,nodes)
        with patch('carbon.policy_runner.Environment.step',new=truncated),patch('carbon.policy_runner.optimize_ppo') as optimize:
            with self.assertRaisesRegex(ContractError,'Incomplete training episode'):
                self.run_train('censored')
            optimize.assert_not_called()
        record=json.loads((self.root/'censored/episodes.jsonl').read_text().splitlines()[0])
        self.assertTrue(record['censor_flag']);self.assertIsNone(record['carbon_g_per_kappa'])
        self.assertEqual(load_json(self.root/'censored/manifest.json')['status'],'failed')

    def test_single_endpoint_current_ci_and_checkpoint_tamper_detection(self):
        settings=settings_for(1,budgets=[1.5],episodes_per_budget=2,epochs=1,objective='lower',forecast_mode='current')
        self.run_train('ablation',settings)
        checkpoint=self.root/'ablation/checkpoint-000001.json'
        metadata,_=read_checkpoint(checkpoint)
        self.assertEqual(metadata['duals']['1.5']['weights'],[1,0])
        chunks=[json.loads(line) for line in (self.root/'ablation/chunks.jsonl').read_text().splitlines()]
        self.assertTrue(all(c['forecast_mode']=='current' for c in chunks))
        weights=checkpoint.parent/metadata['weights_file'];weights.write_bytes(weights.read_bytes()+b'corruption')
        with self.assertRaisesRegex(ContractError,'hash mismatch'):read_checkpoint(checkpoint)
