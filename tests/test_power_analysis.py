from contextlib import redirect_stdout
from copy import deepcopy
from datetime import timedelta
import io
import itertools
from pathlib import Path
import tempfile
import unittest

from carbon.baselines import make_references
from carbon.common import ContractError,iso
from carbon.config import Bundle
from carbon.power_analysis import constant_ci_rescore,exposure_comparison,phase_accounting
from carbon.results import CandidateView,ResultRun
from carbon.runner import run_fixed
from tests.helpers import ROOT,T0,bundle


def exposure_view(name,values):
    parts={}
    for i,value in enumerate(values):
        row={'initial_arrival_utc':iso(T0+timedelta(days=i)),'censor_flag':False,
             'exposure_by_nodes':{str(n):float(value.get(n,0)) for n in (4,8,16,32)}}
        parts[str(i)]=[(1.,row)]
    return CandidateView(name,'policy',1.,1.,parts,True)


class PowerAnalysisTests(unittest.TestCase):
    def test_break_even_detects_a_reversal_inside_the_declared_interval(self):
        b=bundle();baseline=exposure_view('baseline',[{4:20,32:20}])
        candidate=exposure_view('candidate',[{4:19,32:21.5}])
        result=exposure_comparison(candidate,baseline,b.workload,b.raw['power'],.5)
        self.assertAlmostEqual(result['break_even_rho'],1/3)
        self.assertTrue(result['break_even_in_declared_interval'])
        self.assertLess(result['endpoint_carbon_difference_per_kappa'][0],0)
        self.assertGreater(result['endpoint_carbon_difference_per_kappa'][1],0)

    def test_error_radius_uses_mean_exposure_and_matches_all_box_corners(self):
        b=bundle();baseline=exposure_view('baseline',[{4:20,8:20},{4:20,8:20}])
        candidate=exposure_view('candidate',[{4:10,8:24},{4:26,8:10}])
        result=exposure_comparison(candidate,baseline,b.workload,b.raw['power'],1.)
        self.assertEqual(result['strict_improvement_error_radius_supremum'],1.)
        per_job=[]
        for i in range(2):
            a=deepcopy(candidate);a.parts={str(i):a.parts[str(i)]};a.arrivals={str(i):a.arrivals[str(i)]}
            z=deepcopy(baseline);z.parts={str(i):z.parts[str(i)]};z.arrivals={str(i):z.arrivals[str(i)]}
            per_job.append(exposure_comparison(a,z,b.workload,b.raw['power'],1.)['strict_improvement_error_radius_supremum'])
        self.assertAlmostEqual(sum(per_job)/2,.7)
        delta=result['mean_exposure_difference'];factors=result['nominal_scale_factors'];d=.3
        corners=[]
        for errors in itertools.product((-d,d),repeat=3):
            values={'4':1.,**{str(n):factors[str(n)]*(1+error) for n,error in zip((8,16,32),errors)}}
            corners.append(sum(values[n]*delta[n] for n in delta))
        self.assertAlmostEqual(max(corners),result['nominal_difference_without_common_power']+d*result['independent_scale_error_slope'])

    def test_real_chunk_phases_conserve_allocation_and_unit_ci_is_a_rescore(self):
        b=Bundle(ROOT/'configs/synthetic-p2.json')
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            root=Path(tmp);run_fixed(b,root/'fixed',[4,8,16,32])
            refs=make_references(root/'fixed',root/'refs.json');run=ResultRun(root/'fixed',refs)
            fixed=run.candidate('Fixed-4',1.,'validation',kind='fixed')
            phases=phase_accounting(run,fixed)
            self.assertAlmostEqual(phases['overhead_nodehour_fraction'],90/1090)
            scored=constant_ci_rescore(phases,b.workload,b.raw['power'],100.,[0.,.25,1.])
            for value in scored['carbon_g_per_kappa'].values():
                self.assertAlmostEqual(value,1.5*100*4*(100/360+90/3600))
            self.assertEqual(phases['final_action_probabilities'],{'4':1.})
            self.assertIn('not a rerun',scored['scope'])
            path=root/'fixed/chunks.jsonl';lines=path.read_text().splitlines()
            # Removing a required chunk must not manufacture a smaller cost/overhead.
            import json
            lines=[line for line in lines if not (json.loads(line)['episode_id']=='synthetic-validation' and
                                                   json.loads(line)['method']=='Fixed-4' and json.loads(line)['chunk_id']==0)]
            path.write_text('\n'.join(lines)+'\n')
            with self.assertRaisesRegex(ContractError,'chunk identifiers'):phase_accounting(run,fixed)
