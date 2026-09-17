from copy import deepcopy
from datetime import timedelta
import json
from pathlib import Path
import unittest

from carbon.common import ContractError,iso,json_text,load_json,digest
from carbon.exporting import carbon_at,export_results,group_status,load_export_data,paired_plot_statistics,render_export,tex_escape
from carbon.reporting import report_test
from carbon.statistics import CalendarBlocks
from tests.helpers import T0
from tests import test_reporting
from tests.test_statistics import settings


class PlotStatisticsTests(unittest.TestCase):
    def test_observed_target_status_cannot_hide_a_miss_or_missing_seed(self):
        record={'summary':{'complete':True,'miss_lower':0.},'validation_supported':False,'test_feasible':False,
                'validation_empirical_target_met':True,'test_empirical_target_met':True}
        self.assertEqual(group_status([record],.05),'observed target met')
        violation=deepcopy(record);violation['summary']['miss_lower']=.06
        self.assertEqual(group_status([record,violation],.05),'miss above tolerance')
        missing=deepcopy(record);missing['summary']=None
        self.assertEqual(group_status([record,missing],.05),'unavailable')

    def test_ratios_are_of_paired_means_and_seed_variation_is_retained(self):
        arrivals={str(i):iso(T0+timedelta(days=i)) for i in range(20)}
        blocks=CalendarBlocks(arrivals,settings())
        power={'reference_kw':1.,'rho_interval':[.25,1.]}
        base={i:[2.,2.,float(int(i)+1),float(int(i)+1)] for i in arrivals}
        by_seed={str(seed):{i:[factor,1.,factor*v[2],factor*v[3]] for i,v in base.items()}
                 for seed,factor in [(11,.5),(23,1.),(37,1.5)]}
        result=paired_plot_statistics(by_seed,base,blocks,power,.625,2.)
        self.assertAlmostEqual(result['point']['endpoint_ratio_0'],1.)
        self.assertAlmostEqual(result['point']['normalized_carbon'],10.5/2)
        self.assertLess(result['intervals']['mean_tat_hours'][0],1.)
        self.assertGreater(result['intervals']['mean_tat_hours'][1],1.)
        flat={'11':{i:[1.,1.,1.,1.] for i in arrivals}}
        result=paired_plot_statistics(flat,base,blocks,power,.625,1.)
        self.assertAlmostEqual(result['point']['endpoint_ratio_0'],1/10.5)
        self.assertNotAlmostEqual(result['point']['endpoint_ratio_0'],sum(1/(i+1) for i in range(20))/20)

    def test_missing_blocks_and_unpaired_rows_cannot_create_zero_error_bars(self):
        conf=settings();conf['dependence_audit']=None
        blocks=CalendarBlocks({'a':iso(T0)},conf);moments={'a':[1.,1.,2.,1.]}
        result=paired_plot_statistics({'11':moments},moments,blocks,{'reference_kw':1.,'rho_interval':[.25,1.]},.5,1.)
        self.assertIsNone(result['intervals'])
        with self.assertRaisesRegex(ContractError,'identical complete paired'):
            paired_plot_statistics({'11':{}},moments,blocks,{'reference_kw':1.,'rho_interval':[.25,1.]},.5,1.)

    def test_sweep_evaluates_linear_exposures_including_ideal_stress(self):
        power={'reference_kw':2.}
        self.assertEqual(carbon_at(10,4,0,power),8)
        self.assertEqual(carbon_at(10,4,.25,power),11)
        self.assertEqual(carbon_at(10,4,1,power),20)

    def test_unequal_blocks_keep_episode_weights_in_each_draw(self):
        conf=settings();conf.update(minimum_blocks=2,replicates=300)
        blocks=CalendarBlocks({'a':iso(T0),'b':iso(T0+timedelta(minutes=1)),
                               'c':iso(T0+timedelta(days=1))},conf)
        base={i:[1.,1.,v,v] for i,v in [('a',1.),('b',1.),('c',100.)]}
        candidate={'11':{i:[1.,1.,1.,1.] for i in base}}
        result=paired_plot_statistics(candidate,base,blocks,{'reference_kw':1.,'rho_interval':[.25,1.]},.5,1.)
        self.assertAlmostEqual(result['point']['endpoint_ratio_0'],3/102)
        self.assertEqual(result['intervals']['endpoint_ratio_0'],[.01,1.])


class ExportTests(unittest.TestCase):
    def setUp(self):
        case=test_reporting.ReportTests('test_report_preserves_seed_points_and_comparator_and_phase_conservation')
        case.setUp();self.addCleanup(case.doCleanups)
        self.fixture=case;self.root=case.root;self.test=case.test
        report_test(self.test,self.root/'report')

    def test_tables_keep_all_seeds_missing_ablations_and_fixed_comparator(self):
        result=export_results(self.test,self.root/'report',self.root/'export',figures=False)
        data=load_json(self.root/'export/plot-data.json')
        self.assertEqual(result['status'],'complete');self.assertFalse(result['figures_exported'])
        self.assertIn('Current-CI',data['missing_e3_methods']['1.0'])
        self.assertEqual(data['endpoint_ablation_budgets'],[])
        group=next(g for g in data['groups'] if g['method']=='ScaleDown')
        self.assertEqual(set(group['seed_labels']),{'11','23'})
        self.assertEqual(group['comparator_id'],'fixed4')
        self.assertEqual(len(group['seed_points']),2)
        self.assertEqual(group['power_curve'][0]['rho'],0.)
        self.assertEqual(group['power_curve'][0]['scenario'],'ideal-limit stress')
        self.assertTrue(all(p['ratio_of_means']==1 for p in group['power_curve']))
        self.assertIn('SYNTHETIC - not paper evidence',(self.root/'export/E3-mechanisms.tex').read_text())
        self.assertIn('not in plan',(self.root/'export/E3-mechanisms.md').read_text())
        with self.assertRaisesRegex(ContractError,'already exists'):
            export_results(self.test,self.root/'report',self.root/'export',figures=False)

    def test_censored_seed_prevents_a_complete_aggregate_curve(self):
        path=self.test/'policy23-cheap/episodes.jsonl';row=json.loads(path.read_text())
        row.update(censor_flag=True,final_status='censored',exposure_is_complete=False,carbon_g_per_kappa=None,
                   tat_hours=None,nodehours=None,remaining_updates=1,completed_updates=99)
        path.write_text(json.dumps(row)+'\n');self.fixture.reseal_runs()
        report_test(self.test,self.root/'report-censored')
        data=load_export_data(self.test,self.root/'report-censored')
        group=next(g for g in data['groups'] if g['method']=='ScaleDown')
        self.assertEqual(group['status'],'censored')
        self.assertEqual(len(group['seed_records']),2)
        self.assertIsNone(group['statistics']);self.assertNotIn('power_curve',group)
        self.assertFalse(group['summary']['complete'])

    def test_report_and_raw_file_changes_are_rejected(self):
        path=self.root/'report/report.json';original=path.read_text();path.write_text(original+'\n')
        with self.assertRaisesRegex(ContractError,'Report hash mismatch'):
            load_export_data(self.test,self.root/'report')
        path.write_text(original)
        report=load_json(path);report['records'][0]['summary']['mean_tat_hours']+=1
        path.write_text(json_text(report));seal=load_json(self.root/'report/manifest.json');seal['report_sha256']=digest(path)
        (self.root/'report/manifest.json').write_text(json_text(seal))
        with self.assertRaisesRegex(ContractError,'summary differs from sealed raw'):
            load_export_data(self.test,self.root/'report')

    def test_latex_escape_keeps_labels_literal(self):
        self.assertEqual(tex_escape('A&B_1%'),r'A\&B\_1\%')
        self.assertEqual(tex_escape(r'\input{x}'),r'\textbackslash{}input\{x\}')

    def test_standalone_renderer_checks_data_hash_before_rendering(self):
        export_results(self.test,self.root/'report',self.root/'export',figures=False)
        path=self.root/'export/plot-data.json';path.write_text(path.read_text()+'\n')
        with self.assertRaisesRegex(ContractError,'Export data hash mismatch'):
            render_export(self.root/'export',self.root/'figures')
        self.assertFalse((self.root/'figures').exists())
