from contextlib import redirect_stdout
from copy import deepcopy
from datetime import timedelta
import io
import json
from pathlib import Path
import tempfile
import unittest

from carbon.common import ContractError,digest,json_text,load_json
from carbon.config import Bundle
from carbon.dependence import audit_dependence,lag_correlations
from carbon.diagnostics import pairwise_rank_scores,summarize_scores,wait_scores
from carbon.probes import collect_probes
from carbon.runner import run_fixed
from tests.helpers import ROOT,T0


class RankingTests(unittest.TestCase):
    def rows(self):
        return [dict(snapshot_id='snapshot',requested_seconds=60,nodes=n,arrival_utc='2024-01-01T00:00:00Z',
                     censored=False,wait_hours=wait,predicted_atoms_hours=[predicted])
                for n,wait,predicted in ((4,1,1),(8,2,1),(16,3,3),(32,4,2))]

    def test_ranking_counts_ties_censoring_and_misordering(self):
        rows=self.rows();result=pairwise_rank_scores(rows,[4,8,16,32],[60])
        self.assertEqual(result['ordered_pairs'],6);self.assertEqual(result['predicted_ties'],1)
        self.assertAlmostEqual(result['pair_weighted_accuracy'],4.5/6)
        rows[-1].update(censored=True,wait_hours=None)
        rows[1]['wait_hours']=1
        result=pairwise_rank_scores(rows,[4,8,16,32],[60])
        self.assertEqual((result['censored_pairs'],result['observed_ties'],result['ordered_pairs']),(3,1,2))
        self.assertEqual(result['pair_weighted_accuracy'],1.)

    def test_missing_or_duplicate_action_is_not_a_better_ranking_score(self):
        rows=self.rows()
        with self.assertRaisesRegex(ContractError,'every declared node'):
            pairwise_rank_scores(rows[:-1],[4,8,16,32],[60])
        with self.assertRaisesRegex(ContractError,'Duplicate'):
            pairwise_rank_scores(rows+[rows[0]],[4,8,16,32],[60])
        with self.assertRaisesRegex(ContractError,'Missing probe request length'):
            pairwise_rank_scores(rows,[4,8,16,32],[60,120])

    def test_tail_absolute_error_is_not_quantile_of_predicted_wait(self):
        rows=[{'censored':False,'scores':wait_scores([value],0)} for value in (0,1,2,100)]
        metrics=summarize_scores(rows)['metrics']
        self.assertEqual(metrics['mae_of_median_hours'],25.75)
        self.assertEqual(metrics['p50_absolute_error_of_median_hours'],1)
        self.assertEqual(metrics['p90_absolute_error_of_median_hours'],100)


class DependenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.bundle=Bundle(ROOT/'configs/synthetic-p2.json')
        with redirect_stdout(io.StringIO()):
            collect_probes(self.bundle,self.root/'probes','train',interval_seconds=600)

    def test_exact_calendar_gaps_and_constant_series_are_not_zero_correlation(self):
        series={T0+timedelta(hours=i):float(i) for i in (0,1,2,4,5,6)}
        result=lag_correlations(series,[1,2,100])
        self.assertEqual(result[0]['observed_pairs'],4)
        self.assertAlmostEqual(result[0]['correlation'],1.)
        self.assertEqual(result[2]['observed_pairs'],0);self.assertIsNone(result[2]['correlation'])
        constant=lag_correlations({at:1. for at in series},[1])[0]
        self.assertIsNone(constant['correlation']);self.assertIn('constant',constant['unavailable_reason'])
        series[T0+timedelta(hours=1)]=None
        self.assertEqual(lag_correlations(series,[1])[0]['observed_pairs'],2)

    def test_queue_only_audit_does_not_choose_blocks_or_claim_complete_span_review(self):
        result=audit_dependence([self.root/'probes'],self.root/'audit',lag_hours=[1/6,1/3,24])
        self.assertEqual(result['snapshot_count'],10)
        self.assertTrue(result['block_span_floor_is_incomplete'])
        self.assertEqual(result['feature_history_hours'],48)
        self.assertIsNone(result['selected_block_hours']);self.assertFalse(result['independence_certified'])
        self.assertTrue((self.root/'audit/report.md').exists())

    def test_complete_development_episodes_supply_span_floor(self):
        with redirect_stdout(io.StringIO()):run_fixed(self.bundle,self.root/'fixed',[4,8,16,32],split='validation')
        result=audit_dependence([self.root/'probes'],self.root/'audit',[self.root/'fixed'],lag_hours=[1/6])
        self.assertEqual(result['episode_spans']['rows'],4)
        self.assertFalse(result['block_span_floor_is_incomplete'])
        self.assertGreater(result['episode_spans']['max_observed_hours'],0)
        self.assertEqual(len(result['episode_lag_correlations']),20)
        self.assertEqual({r['panel_method_seed_budget'][1] for r in result['episode_lag_correlations']},
                         {'Fixed-4','Fixed-8','Fixed-16','Fixed-32'})

    def test_test_outcomes_are_rejected_from_manifest_before_parsing(self):
        p=self.root/'probes/manifest.json';meta=load_json(p);meta['probe_split']='test';p.write_text(json_text(meta))
        (self.root/'probes/probes.jsonl').write_text('must not be parsed\n')
        with self.assertRaisesRegex(ContractError,'must not inspect test'):
            audit_dependence([self.root/'probes'],self.root/'audit')

    def test_missing_actions_and_censored_labels_are_visible(self):
        p=self.root/'probes/probes.jsonl';rows=[json.loads(line) for line in p.read_text().splitlines()]
        original=deepcopy(rows);rows=rows[1:]
        def save(values):
            p.write_text(''.join(json.dumps(r)+'\n' for r in values))
            meta=load_json(self.root/'probes/manifest.json');meta['probes_sha256']=digest(p)
            (self.root/'probes/manifest.json').write_text(json_text(meta))
        save(rows)
        with self.assertRaisesRegex(ContractError,'all declared probe actions'):
            audit_dependence([self.root/'probes'],self.root/'audit')
        original[0].update(censored=True,wait_hours=None,label_observed_at_utc=None);save(original)
        result=audit_dependence([self.root/'probes'],self.root/'audit',lag_hours=[1/6])
        self.assertEqual(result['series_missing_counts']['wait_mean_hours.nodes_4'],1)
