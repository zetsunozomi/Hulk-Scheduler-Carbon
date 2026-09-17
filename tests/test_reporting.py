from copy import deepcopy
from datetime import timedelta
import json
from pathlib import Path
import shutil
import unittest

from carbon.common import ContractError,digest,iso,json_text,load_json
from carbon.heldout import frozen_jobs,load_selection,locate_checkpoints
from carbon.reporting import mean_seed_view,report_test
from carbon.results import ResultRun
from carbon.statistics import CalendarBlocks,crossed_seed_ratio_intervals
from tests.helpers import T0
from tests import test_selection
from tests.test_statistics import settings


class ReportTests(unittest.TestCase):
    def setUp(self):
        fixture=test_selection.SelectionTests('test_selects_per_seed_checkpoint_and_freezes_strongest_non_rl')
        fixture.setUp();self.addCleanup(fixture.doCleanups)
        self.fixture=fixture;self.root=fixture.root
        self.selected=fixture.run_selection()
        self.test=self.root/'test';self.test.mkdir()
        shutil.copytree(self.root/'selected',self.test/'selection')
        self.jobs=frozen_jobs(self.selected)
        self.raw_chunks=[json.loads(line) for line in (self.root/'fixed/chunks.jsonl').read_text().splitlines()]
        self.write_test_run()

    def write_test_run(self):
        fixture=self.fixture
        manifest={'kind':'heldout_execution_v1','status':'complete','runs':{},'run_files_sha256':{}}
        for identity,betas in self.jobs.items():
            entry=self.selected['candidates'][identity]
            fixed=entry['kind'] in {'fixed','mixture'}
            relative='fixed' if fixed else identity
            manifest['runs'][identity]={str(beta):relative for beta in betas}
            path=self.test/relative
            if path.exists():continue
            path.mkdir()
            n=4 if entry['kind']=='policy' else 8
            rows=[deepcopy(r) for r in fixture.fixed_rows if r['split']=='test' and (fixed or r['method']==f'Fixed-{n}')]
            chunks=[deepcopy(r) for r in self.raw_chunks if r['split']=='test' and (fixed or r['method']==f'Fixed-{n}')]
            run_manifest=deepcopy(fixture.fixed_manifest if fixed else load_json(self.root/identity/'manifest.json'))
            run_manifest['selected_splits']=['test']
            if not fixed:
                for row in rows+chunks:
                    row.update(method=entry['method'],seed=entry.get('seed',11),budget_hours=fixture.budget)
            (path/'episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            (path/'chunks.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in chunks))
            (path/'manifest.json').write_text(json_text(run_manifest))
        cohort={r['episode_id']:r['initial_arrival_utc'] for r in fixture.fixed_rows if r['split']=='test'}
        plan={'kind':'heldout_plan_v1','selection_sha256':digest(self.test/'selection/selection.json'),
              'asset_sha256':fixture.references['asset_sha256'],'purpose':'synthetic','panel':self.selected['panel'],
              'test_cohort':cohort,'jobs':self.jobs,'power_analysis_nominal_rho':.625,'constant_ci_rescore_value':1.}
        (self.test/'plan.json').write_text(json_text(plan));manifest['plan_sha256']=digest(self.test/'plan.json')
        (self.test/'manifest.json').write_text(json_text(manifest));self.reseal_runs()

    def reseal_runs(self):
        manifest=load_json(self.test/'manifest.json')
        manifest['run_files_sha256']={name:{f:digest(self.test/name/f) for f in ('manifest.json','episodes.jsonl','chunks.jsonl')}
                                      for name in {p for paths in manifest['runs'].values() for p in paths.values()}}
        (self.test/'manifest.json').write_text(json_text(manifest))

    def test_report_preserves_seed_points_and_comparator_and_phase_conservation(self):
        result=report_test(self.test,self.root/'report')
        policies=[r for r in result['records'] if r['method']=='ScaleDown']
        self.assertEqual({r['seed'] for r in policies},{'11','23'})
        self.assertTrue(all(r['selected_id'].endswith('cheap') for r in policies))
        for row in policies:
            self.assertGreater(row['phases']['overhead_nodehour_fraction'],0)
            self.assertEqual(len(row['by_calendar_block']),1)
        self.assertTrue(all(r['comparator_id']=='fixed4' for r in result['comparisons']))
        self.assertEqual(len(result['seed_aggregates']),1)
        self.assertFalse(result['seed_aggregates'][0]['all_declared_seed_contrasts_supported'])
        self.assertFalse(result['research_evidence'])
        self.assertIsNone(result['seed_aggregates'][0]['carbon_variability']['two_sided_interval'])
        self.assertTrue((self.root/'report/report.md').exists())

    def test_observed_target_and_uncertain_miss_bound_remain_separate_through_export(self):
        from carbon.exporting import export_results
        self.fixture.spec['selection_rule']='empirical_miss'
        self.fixture.spec['statistics']=settings('calendar_blocks')
        self.fixture.spec['statistics']['dependence_audit']=None
        self.selected=self.fixture.run_selection('empirical-selected')
        self.jobs=frozen_jobs(self.selected)
        for name in ('manifest.json','selection.json'):
            shutil.copy2(self.root/'empirical-selected'/name,self.test/'selection'/name)
        self.write_test_run()
        result=report_test(self.test,self.root/'empirical-report')
        self.assertEqual(result['selection_rule'],'empirical_miss')
        self.assertTrue(all(r['test_empirical_target_met'] and r['validation_empirical_target_met'] for r in result['records']))
        self.assertFalse(any(r['test_feasible'] or r['validation_supported'] for r in result['records']))
        self.assertFalse(any(r['budgeted_interval_improvement_supported'] for r in result['comparisons']))
        export_results(self.test,self.root/'empirical-report',self.root/'empirical-export',figures=False)
        data=load_json(self.root/'empirical-export/plot-data.json')
        self.assertTrue(all(g['status']=='observed target met' for g in data['groups']))
        self.assertEqual({r['seed'] for r in result['records'] if r['method']=='ScaleDown'},{'11','23'})

    def test_censoring_keeps_the_seed_and_disables_full_carbon_aggregate(self):
        path=self.test/'policy23-cheap/episodes.jsonl';row=json.loads(path.read_text())
        row.update(censor_flag=True,final_status='censored',exposure_is_complete=False,
                   carbon_g_per_kappa=None,tat_hours=None,nodehours=None,remaining_updates=1,completed_updates=99)
        path.write_text(json.dumps(row)+'\n');self.reseal_runs()
        result=report_test(self.test,self.root/'report')
        row=next(r for r in result['records'] if r['method']=='ScaleDown' and r['seed']=='23')
        self.assertIsNone(row['summary']['mean_carbon_g_per_kappa'])
        self.assertFalse(row['test_empirical_target_met'])
        self.assertFalse(result['seed_aggregates'][0]['summary']['complete'])
        self.assertIsNone(result['seed_aggregates'][0]['carbon_variability'])
        self.assertEqual(len([r for r in result['records'] if r['method']=='ScaleDown']),2)

    def test_changed_checkpoint_missing_seed_and_changed_cohort_fail_closed(self):
        path=self.test/'policy11-cheap/manifest.json';original=path.read_text();manifest=json.loads(original)
        manifest['checkpoint_sha256']='b'*64;path.write_text(json_text(manifest));self.reseal_runs()
        with self.assertRaisesRegex(ContractError,'checkpoint_sha256 changed'):report_test(self.test,self.root/'report')
        path.write_text(original);self.reseal_runs()
        path=self.test/'manifest.json';original=path.read_text();manifest=json.loads(original)
        manifest['runs'].pop('policy23-cheap');path.write_text(json_text(manifest))
        with self.assertRaisesRegex(ContractError,'Missing/extra selected'):report_test(self.test,self.root/'report')
        path.write_text(original)
        path=self.test/'policy11-cheap/episodes.jsonl';row=json.loads(path.read_text());row['episode_id']='replacement'
        path.write_text(json.dumps(row)+'\n');self.reseal_runs()
        with self.assertRaisesRegex(ContractError,'entire frozen cohort'):report_test(self.test,self.root/'report')

    def test_selection_and_result_tampering_are_detected(self):
        path=self.test/'policy11-cheap/episodes.jsonl';path.write_text(path.read_text()+'\n')
        with self.assertRaisesRegex(ContractError,'file hash mismatch'):report_test(self.test,self.root/'report')
        path=self.test/'selection/selection.json';path.write_text(path.read_text()+'\n')
        with self.assertRaisesRegex(ContractError,'Selection hash mismatch'):load_selection(self.test/'selection')

    def test_seed_average_quantile_is_pooled_not_average_of_quantiles(self):
        refs=self.fixture.references
        run=ResultRun(self.root/'fixed',refs)
        low=run.candidate('Fixed-4',1.,'test',kind='fixed')
        high=run.candidate('Fixed-32',1.,'test',kind='fixed');high.method=low.method
        pooled=mean_seed_view([low,high]).summary(refs)
        self.assertEqual(pooled['p95_tat_hours'],max(low.summary(refs)['p95_tat_hours'],high.summary(refs)['p95_tat_hours']))
        self.assertNotEqual(pooled['p95_tat_hours'],sum(v.summary(refs)['p95_tat_hours'] for v in (low,high))/2)

    def test_frozen_cohort_asset_change_rejected(self):
        path=self.test/'fixed/manifest.json';manifest=load_json(path)
        manifest['asset_sha256']['cohort']='c'*64;path.write_text(json_text(manifest));self.reseal_runs()
        with self.assertRaisesRegex(ContractError,'Reference cohort differs'):report_test(self.test,self.root/'report')

    def test_checkpoint_lookup_requires_exact_metadata_and_weights(self):
        path=self.root/'training';path.mkdir();(path/'checkpoint-000001.pt').write_bytes(b'synthetic weights')
        metadata={'kind':'ppo_checkpoint_v1','weights_file':'checkpoint-000001.pt','weights_sha256':digest(path/'checkpoint-000001.pt')}
        (path/'checkpoint-000001.json').write_text(json_text(metadata));value=digest(path/'checkpoint-000001.json')
        self.assertEqual(locate_checkpoints([path],{value})[value],(path/'checkpoint-000001.json').resolve())
        with self.assertRaisesRegex(ContractError,'Missing frozen checkpoint'):locate_checkpoints([path],{'0'*64})
        (path/'checkpoint-000001.pt').write_bytes(b'changed')
        with self.assertRaisesRegex(ContractError,'weights hash mismatch'):locate_checkpoints([path],{value})


class CrossedBootstrapTests(unittest.TestCase):
    def test_identical_seed_ratio_and_seed_variability_are_not_iid_episode_noise(self):
        arrivals={str(i):iso(T0+timedelta(days=i)) for i in range(30)}
        blocks=CalendarBlocks(arrivals,settings())
        baseline={i:[float(int(i)+1),float(2*(int(i)+1))] for i in arrivals}
        same={str(s):{i:[.8*v for v in row] for i,row in baseline.items()} for s in (11,23,37)}
        result=crossed_seed_ratio_intervals(same,baseline,blocks)
        for bounds in result['two_sided_interval']:
            for v in bounds:self.assertAlmostEqual(v,.8)
        varied={str(s):{i:[factor*v for v in row] for i,row in baseline.items()} for s,factor in ((11,.5),(23,1.),(37,1.5))}
        result=crossed_seed_ratio_intervals(varied,baseline,blocks)
        self.assertAlmostEqual(result['endpoint_ratios'][0],1.)
        self.assertLess(result['two_sided_interval'][0][0],.9)
        self.assertGreater(result['two_sided_interval'][0][1],1.1)
        self.assertFalse(result['confirmatory_claim'])

    def test_missing_calendar_pair_is_not_dropped(self):
        blocks=CalendarBlocks({'a':iso(T0),'b':iso(T0+timedelta(days=1))},settings())
        with self.assertRaisesRegex(ContractError,'identical calendar pairs'):
            crossed_seed_ratio_intervals({'11':{'a':[1.,1.]}},{'a':[2.,2.],'b':[2.,2.]},blocks)
