from datetime import timedelta
import math
import itertools
import unittest

from carbon.common import ContractError,iso
from carbon.statistics import CalendarBlocks,bounded_mean_upper,deadline_bound,paired_ratio_intervals
from tests.helpers import T0


def settings(scope='calendar_blocks'):
    return {'block_hours':24,'anchor_utc':iso(T0),'alpha':.05,'replicates':300,'seed':11,
            'minimum_blocks':10,'miss_scope':scope,'dependence_audit':'synthetic independent-block fixture'}


class StatisticalTests(unittest.TestCase):
    def test_zero_miss_bound_is_positive_and_scope_changes_replication_unit(self):
        arrivals={str(i):iso(T0+timedelta(hours=i)) for i in range(48)}
        observed={i:(0.,0.) for i in arrivals}
        cohort=deadline_bound(observed,CalendarBlocks(arrivals,settings('fixed_cohort')),.05)
        calendar=deadline_bound(observed,CalendarBlocks(arrivals,settings()),.05)
        self.assertAlmostEqual(cohort['confidence_upper'],1-.05**(1/48))
        self.assertAlmostEqual(calendar['confidence_upper'],1-.05**(1/2))
        self.assertGreater(calendar['confidence_upper'],cohort['confidence_upper'])
        self.assertFalse(calendar['support_enabled'])
        exact=deadline_bound(observed,CalendarBlocks(arrivals,settings('fixed_cohort')),.05,False)
        self.assertEqual(exact['confidence_upper'],0)

    def test_bound_has_nominal_coverage_on_enumerated_binomial_samples(self):
        # Exhaustively sum sampling probabilities, not a noisy Monte Carlo check.
        for n in (1,3,10,20):
            bounds=[bounded_mean_upper(k/n,n,.05) for k in range(n+1)]
            self.assertEqual(bounds,sorted(bounds))
            for p in (.001,.03,.1,.3,.7,.95):
                failed=sum(math.comb(n,k)*p**k*(1-p)**(n-k) for k,u in enumerate(bounds) if u<p)
                self.assertLessEqual(failed,.05+1e-12)

    def test_unequal_blocks_are_not_counted_as_equal_sized_independent_jobs(self):
        arrivals={str(i):iso(T0+timedelta(minutes=i)) for i in range(9)}
        arrivals['later']=iso(T0+timedelta(days=1))
        blocks=CalendarBlocks(arrivals,settings())
        self.assertAlmostEqual(blocks.effective_count,10/9)
        bounded=deadline_bound({i:(0.,1. if i=='later' else 0.) for i in arrivals},blocks,.05)
        self.assertEqual(bounded['unknown_probability_mass'],.1)
        self.assertGreater(bounded['confidence_upper'],.1)

    def test_weighted_bound_covers_unequal_independent_bernoulli_means(self):
        weights=[(i+1)/36 for i in range(8)]
        count=1/max(weights)
        for probabilities in ([.5]*8,[(i+1)/10 for i in range(8)],[.95]*8):
            true_mean=sum(w*p for w,p in zip(weights,probabilities))
            failed=0.
            for sample in itertools.product((0,1),repeat=8):
                observed=sum(w*x for w,x in zip(weights,sample))
                upper=bounded_mean_upper(min(1.,observed),count,.05)
                if upper<true_mean:
                    failed+=math.prod(p if x else 1-p for p,x in zip(probabilities,sample))
            self.assertLessEqual(failed,.05+1e-12)

    def test_bootstrap_keeps_calendar_pairs_and_uses_ratio_of_means(self):
        arrivals={str(i):iso(T0+timedelta(days=i//2,minutes=i%2)) for i in range(60)}
        blocks=CalendarBlocks(arrivals,settings())
        baseline={str(i):[float(i+1),float(2*(i+1))] for i in range(60)}
        improved={key:[.8*v for v in row] for key,row in baseline.items()}
        result=paired_ratio_intervals(improved,baseline,blocks)
        for value in result['endpoint_ratios']+result['joint_one_sided_upper']:
            self.assertAlmostEqual(value,.8)
        self.assertTrue(result['interval_wide_carbon_improvement_supported'])
        for draw in blocks.draws():
            for i in range(0,60,2):
                self.assertEqual(draw.count(str(i)),draw.count(str(i+1)))
        numerator={str(i):[1.,1.] for i in range(60)}
        result=paired_ratio_intervals(numerator,baseline,blocks)
        self.assertAlmostEqual(result['endpoint_ratios'][0],1/30.5)
        self.assertNotAlmostEqual(result['endpoint_ratios'][0],sum(1/(i+1) for i in range(60))/60)

    def test_missing_pairs_zero_denominator_and_unreviewed_dependence_fail_safely(self):
        arrivals={'a':iso(T0),'b':iso(T0+timedelta(days=2))}
        spec=settings();spec['dependence_audit']=None
        blocks=CalendarBlocks(arrivals,spec)
        result=paired_ratio_intervals({'a':[1,1],'b':[1,1]},{'a':[2,2],'b':[2,2]},blocks)
        self.assertIsNone(result['joint_one_sided_upper'])
        self.assertFalse(result['interval_wide_carbon_improvement_supported'])
        with self.assertRaisesRegex(ContractError,'Unpaired'):
            paired_ratio_intervals({'a':[1,1]},{'a':[2,2],'b':[2,2]},blocks)
        with self.assertRaisesRegex(ContractError,'Nonpositive'):
            paired_ratio_intervals({'a':[1,1],'b':[1,1]},{'a':[0,0],'b':[0,0]},blocks)
