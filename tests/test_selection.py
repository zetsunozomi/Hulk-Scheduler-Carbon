from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from carbon.baselines import make_references
from carbon.common import ContractError,digest,json_text,load_json
from carbon.config import Bundle
from carbon.results import ResultRun
from carbon.runner import run_fixed
from carbon.selection import select_policies,select_record
from tests.helpers import ROOT
from tests.test_statistics import settings


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);bundle=Bundle(ROOT/'configs/synthetic-p2.json')
        with redirect_stdout(io.StringIO()):run_fixed(bundle,self.root/'fixed',[4,8,16,32])
        self.references=make_references(self.root/'fixed',self.root/'refs.json')
        self.fixed_manifest=load_json(self.root/'fixed/manifest.json')
        self.fixed_rows=[json.loads(line) for line in (self.root/'fixed/episodes.jsonl').read_text().splitlines()]
        self.beta=1.;self.budget=self.references['time_reference_hours']
        self.spec={'schema_version':1,'references':'refs.json','budgets':[self.beta],'epsilon':1.,
                   'statistics':settings('fixed_cohort'),'required_policy_seeds':[11,23],'candidates':[]}
        self.spec['statistics']['dependence_audit']=None
        for n in (4,8,16,32):
            self.spec['candidates'].append({'id':f'fixed{n}','kind':'fixed','method':f'Fixed-{n}','run':'fixed','budgets':[self.beta]})
        mixture={'kind':'fixed_mix_v1','fit_split':'validation','panel':bundle.raw['panel'],'purpose':'synthetic',
                 'weights':{'4':.5,'8':.5,'16':0.,'32':0.},'empirical_feasible':True,'epsilon':1.,'budget_hours':self.budget,
                 'source_episodes_sha256':digest(self.root/'fixed/episodes.jsonl'),
                 'reference_source_episodes_sha256':self.references['source_episodes_sha256'],
                 'asset_sha256':self.fixed_manifest['asset_sha256']}
        (self.root/'mixture.json').write_text(json_text(mixture))
        self.spec['candidates'].append({'id':'mix','kind':'mixture','method':'Fixed-Mix','run':'fixed','model':'mixture.json','budgets':[self.beta]})
        for method,name in [('Rollout-MPC','mpc'),('Plan-once','once')]:
            self.write_candidate(name,method,8,11,'planner')
        for seed in (11,23):
            self.write_candidate(f'policy{seed}-cheap','ScaleDown',4,seed,'policy')
            self.write_candidate(f'policy{seed}-dear','ScaleDown',16,seed,'policy')

    def write_candidate(self,name,method,n,seed,kind):
        # Synthetic result fixtures reuse actual fixed replay accounting; this tests
        # selection/aggregation only, not a claimed planner or policy performance.
        path=self.root/name;path.mkdir()
        manifest=deepcopy(self.fixed_manifest)
        manifest.pop('fixed_nodes');manifest['selected_splits']=['validation']
        rows=[deepcopy(r) for r in self.fixed_rows if r['split']=='validation' and r['method']==f'Fixed-{n}']
        for row in rows:
            row.update(method=method,seed=seed,budget_hours=self.budget,deadline_miss=row['tat_hours']>self.budget)
        if kind=='policy':
            manifest.update(kind='policy_evaluation',training_seed=seed,sampling_seed=seed,policy_rule='categorical_sampling',
                            checkpoint_sha256=hashlib.sha256(name.encode()).hexdigest(),predictor_sha256='a'*64)
            for row in rows:row['policy_rule']='categorical_sampling'
        else:
            manifest.update(methods=[method],seed=seed,planning_paths=256,planner_internal_miss_tolerance=.05,predictor_sha256='a'*64)
        (path/'manifest.json').write_text(json_text(manifest))
        (path/'episodes.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
        entry={'id':name,'kind':kind,'method':method,'run':name,'budgets':[self.beta]}
        if kind=='policy':entry['seed']=seed
        self.spec['candidates'].append(entry)

    def run_selection(self,name='selected'):
        (self.root/'spec.json').write_text(json_text(self.spec))
        return select_policies(self.root/'spec.json',self.root/name)

    def test_selects_per_seed_checkpoint_and_freezes_strongest_non_rl(self):
        result=self.run_selection()
        points=result['operating_points']['1.0']
        for seed in ('11','23'):
            self.assertEqual(points['ScaleDown'][seed]['selected_id'],f'policy{seed}-cheap')
            self.assertTrue(points['ScaleDown'][seed]['supported'])
        self.assertEqual(result['comparators']['1.0']['strongest_non_rl']['selected_id'],'fixed4')
        self.assertEqual(result['scope'],'validation_only')
        self.assertEqual(len(result['records']),len(self.spec['candidates']))
        self.assertAlmostEqual(result['alpha_per_candidate'],.05/len(result['records']))
        self.assertTrue((self.root/'selected/selection.md').exists())

    def test_zero_misses_do_not_make_tiny_validation_cohort_supported(self):
        self.spec['epsilon']=.05
        model=load_json(self.root/'mixture.json');model['epsilon']=.05
        model.update(weights=None,empirical_feasible=False)
        (self.root/'mixture.json').write_text(json_text(model))
        result=self.run_selection()
        for seed,point in result['operating_points']['1.0']['ScaleDown'].items():
            self.assertFalse(point['supported']);self.assertIsNotNone(point['selected_id'])
        self.assertFalse(result['operating_points']['1.0']['Fixed-Mix']['baseline']['supported'])
        mix=next(r for r in result['records'] if r['id']=='mix')
        self.assertEqual(mix['summary']['reason'],'validation LP infeasible')

    def test_empirical_rule_selects_carbon_under_target_without_claiming_confidence(self):
        # A 4% miss candidate can be the lower-carbon admissible choice at 5%.
        # Inconclusive confidence bounds must not silently turn that rule into
        # minimum observed misses, nor become a population-feasibility claim.
        candidates=[{'id':'lower-carbon','summary':{'complete':True,'worst_normalized_carbon':.6},
                     'deadline':{'observed_upper':.04,'confidence_upper':.20},
                     'empirical_eligible':True,'confidence_eligible':False,'eligible':True},
                    {'id':'zero-observed-misses','summary':{'complete':True,'worst_normalized_carbon':1.},
                     'deadline':{'observed_upper':0.,'confidence_upper':.16},
                     'empirical_eligible':True,'confidence_eligible':False,'eligible':True}]
        result=select_record(candidates,'empirical_miss')
        self.assertEqual(result['selected_id'],'lower-carbon')
        self.assertTrue(result['empirical_target_met']);self.assertTrue(result['selection_criterion_met'])
        self.assertFalse(result['supported'])
        for row in candidates:row['eligible']=False
        strict=select_record(candidates,'confidence_upper_miss')
        self.assertEqual(strict['selected_id'],'zero-observed-misses')
        self.assertFalse(strict['selection_criterion_met']);self.assertFalse(strict['supported'])

    def test_explicit_empirical_rule_keeps_calendar_bound_and_all_seeds(self):
        self.spec['selection_rule']='empirical_miss'
        self.spec['statistics']=settings('calendar_blocks')
        self.spec['statistics']['dependence_audit']=None
        result=self.run_selection()
        self.assertEqual(result['selection_rule'],'empirical_miss')
        for seed in ('11','23'):
            point=result['operating_points']['1.0']['ScaleDown'][seed]
            self.assertEqual(point['selected_id'],f'policy{seed}-cheap')
            self.assertTrue(point['selection_criterion_met']);self.assertTrue(point['empirical_target_met'])
            self.assertFalse(point['supported'])
        self.assertTrue(all(r['empirical_eligible'] for r in result['records']))
        self.assertFalse(any(r['confidence_eligible'] for r in result['records']))
        self.assertGreater(result['zero_miss_calendar_bound_floor'],.05)

    def test_unknown_selection_rule_is_rejected(self):
        self.spec['selection_rule']='choose_after_test'
        with self.assertRaisesRegex(ContractError,'Unknown selection_rule'):self.run_selection()

    def test_missing_seed_or_strong_baseline_cannot_silently_shrink_comparison(self):
        original=deepcopy(self.spec)
        self.spec['candidates']=[r for r in self.spec['candidates'] if r.get('seed')!=23]
        with self.assertRaisesRegex(ContractError,'Missing policy seeds|every budget and seed'):self.run_selection()
        self.spec=original
        self.spec['candidates']=[r for r in self.spec['candidates'] if r['method']!='Plan-once']
        with self.assertRaisesRegex(ContractError,'requires Fixed-Mix'):self.run_selection()

    def test_changed_cohort_and_accounting_are_rejected(self):
        path=self.root/'policy11-cheap/episodes.jsonl';row=json.loads(path.read_text())
        row['episode_id']='unpaired';path.write_text(json.dumps(row)+'\n')
        with self.assertRaisesRegex(ContractError,'identical predeclared'):self.run_selection()
        row['episode_id']='synthetic-validation';row['A']+=1;path.write_text(json.dumps(row)+'\n')
        with self.assertRaisesRegex(ContractError,'Exposure conservation'):self.run_selection()

    def test_test_only_run_is_not_accepted_as_validation(self):
        path=self.root/'policy11-cheap/episodes.jsonl';row=json.loads(path.read_text())
        source=next(r for r in self.fixed_rows if r['split']=='test' and r['method']=='Fixed-4')
        changed=deepcopy(source);changed.update(method=row['method'],seed=row['seed'],budget_hours=self.budget,policy_rule='categorical_sampling')
        path.write_text(json.dumps(changed)+'\n')
        with self.assertRaisesRegex(ContractError,'Empty result|No paired outcomes'):self.run_selection()

    def test_research_cannot_substitute_the_easier_fixed_cohort_scope(self):
        references=deepcopy(self.references);references['purpose']='research'
        (self.root/'refs.json').write_text(json_text(references))
        with self.assertRaisesRegex(ContractError,'manuscript calendar-block scope'):self.run_selection()

    def test_research_holdout_is_rejected_before_episode_values_are_read(self):
        references=deepcopy(self.references);references['purpose']='research'
        manifest=deepcopy(self.fixed_manifest);manifest['purpose']='research'
        (self.root/'fixed/manifest.json').write_text(json_text(manifest))
        (self.root/'fixed/episodes.jsonl').write_text('Not parsed: the manifest already declares a test split.\n')
        with self.assertRaisesRegex(ContractError,'already contains test results'):
            ResultRun(self.root/'fixed',references,allowed_splits={'train','validation'})

    def test_mixture_quantile_and_unknown_carbon_preserve_component_distribution(self):
        run=ResultRun(self.root/'fixed',self.references)
        mixture=run.mixture(self.root/'mixture.json',self.beta,'validation')
        summary=mixture.summary(self.references)
        times=[r['tat_hours'] for r in self.fixed_rows if r['split']=='validation' and r['method'] in {'Fixed-4','Fixed-8'}]
        self.assertEqual(summary['p95_tat_hours'],max(times))
        self.assertAlmostEqual(summary['mean_tat_hours'],sum(times)/2)
        identity=next(iter(mixture.parts));weight,row=mixture.parts[identity][0]
        row.update(censor_flag=True,tat_hours=None,nodehours=None,carbon_g_per_kappa=None)
        self.assertFalse(mixture.summary(self.references)['complete'])
        self.assertIsNone(mixture.summary(self.references)['mean_carbon_g_per_kappa'])
