from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from carbon.common import ContractError, load_json
from carbon.config import Bundle
from carbon.main_pilot import make_budget_grid, run_main_pilot, verify_fixed
from carbon.policy_runner import read_checkpoint
from carbon.runner import run_fixed, write_manifest
from tests.helpers import ROOT


class MainPilotTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.bundle=Bundle(ROOT/'configs/synthetic-p2.json')
        self.output=redirect_stdout(io.StringIO());self.output.__enter__()
        self.addCleanup(self.output.__exit__,None,None,None)

    def test_grid_spans_training_fixed4_p95_with_four_ticks(self):
        rows={'Fixed-4':{str(i):{'tat_hours':100+i} for i in range(20)}}
        refs={'time_reference_hours':10.,'source_episodes_sha256':'train-hash'}
        grid=make_budget_grid(refs,rows)
        self.assertEqual(grid['upper_budget_hours'],118)
        self.assertEqual(grid['positions'],[0,.25,.5,1])
        self.assertEqual(len(grid['budget_multipliers']),4)
        self.assertAlmostEqual(grid['slider']['ticks'][-1]['budget_hours'],118)
        self.assertAlmostEqual(grid['slider']['ticks'][1]['budget_hours'],37)
        self.assertEqual(grid['source_split'],'train')
        self.assertEqual(grid['source_episodes_sha256'],'train-hash')

    def test_full_pilot_without_wait_model_then_idempotent_resume(self):
        path=self.root/'pilot'
        with patch('carbon.policy_inputs.WaitPredictor.load',side_effect=AssertionError('main must not load a wait model')):
            result=run_main_pilot(self.bundle,path)
        self.assertEqual(result['ppo_iterations'],5)
        self.assertEqual(result['ppo_episodes'],80)
        self.assertIsNone(result['wait_predictor_sha256'])
        self.assertEqual(len(result['iteration_timing']),5)
        for split in ('train','validation'):
            self.assertEqual(len(result['fixed'][split]),4)
        with patch('carbon.main_pilot.run_fixed',side_effect=AssertionError('must reuse fixed results')), \
             patch('carbon.main_pilot.train_policy',side_effect=AssertionError('must reuse completed pilot')):
            again=run_main_pilot(self.bundle,path,resume=True)
        self.assertEqual(again,result)
        # Do not silently accept a changed complete result, even if still valid JSON.
        p=path/'fixed-train/episodes.jsonl';p.write_bytes(p.read_bytes()+b' ')
        with self.assertRaises((ContractError,json.JSONDecodeError)):
            run_main_pilot(self.bundle,path,resume=True)

    def test_interrupted_ppo_resumes_from_completed_iteration_with_same_final_tensors(self):
        import torch
        interrupted=self.root/'interrupted'; clean=self.root/'clean';killed=False
        def kill_after_first(path,value):
            nonlocal killed
            write_manifest(path,value)
            if path.parent.name.startswith('ppo-attempt-') and path.name=='manifest.json' and value.get('completed_iteration')==1 and not killed:
                killed=True
                raise KeyboardInterrupt()
        with patch('carbon.policy_runner.write_manifest',side_effect=kill_after_first):
            with self.assertRaises(KeyboardInterrupt):run_main_pilot(self.bundle,interrupted)
        self.assertTrue((interrupted/'ppo-attempt-000/checkpoint-000001.json').exists())
        resumed=run_main_pilot(self.bundle,interrupted,resume=True)
        baseline=run_main_pilot(self.bundle,clean)
        self.assertEqual(resumed['ppo_episodes'],80)
        self.assertEqual(load_json(interrupted/'ppo-attempt-001/manifest.json')['first_iteration'],2)
        _,actual=read_checkpoint(interrupted/resumed['checkpoint'])
        _,expected=read_checkpoint(clean/baseline['checkpoint'])
        for key in actual['model']:
            self.assertTrue(torch.equal(actual['model'][key],expected['model'][key]),key)
        self.assertTrue(torch.equal(actual['sampling_rng'],expected['sampling_rng']))

    def test_timing_shard_and_changed_pipeline_contract_are_rejected(self):
        path=self.root/'fixed'
        run_fixed(self.bundle,path,self.bundle.raw['cluster']['allowed_nodes'],'train')
        meta=load_json(path/'manifest.json');meta['shard_count']=16
        write_manifest(path/'manifest.json',meta)
        with self.assertRaisesRegex(ContractError,'entire declared fixed cohort'):
            verify_fixed(self.bundle,path,'train')
        path=self.root/'pilot';run_main_pilot(self.bundle,path)
        contract=load_json(path/'pipeline-contract.json');contract['config_sha256']='different'
        write_manifest(path/'pipeline-contract.json',contract)
        with self.assertRaisesRegex(ContractError,'Pipeline inputs'):
            run_main_pilot(self.bundle,path,resume=True)
