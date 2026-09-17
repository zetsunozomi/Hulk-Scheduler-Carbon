"""Paired calendar resampling and explicitly scoped deadline uncertainty."""

from collections import defaultdict
import math
import random
from statistics import fmean

from .common import integer, iso, number, require, timestamp
from .features import quantile


def analysis_settings(config):
    required = {'block_hours','anchor_utc','alpha','replicates','seed','minimum_blocks',
                'miss_scope','dependence_audit'}
    require(set(config)==required, f'Statistical settings require exactly {sorted(required)}')
    result = dict(config)
    result['block_hours'] = float(number(result['block_hours'],'block_hours',strict=True))
    result['anchor_utc'] = iso(timestamp(result['anchor_utc']))
    result['alpha'] = float(number(result['alpha'],'alpha',strict=True))
    require(result['alpha']<1,'alpha must be below one')
    result['replicates'] = integer(result['replicates'],'replicates',100)
    result['seed'] = integer(result['seed'],'seed',0)
    result['minimum_blocks'] = integer(result['minimum_blocks'],'minimum_blocks',2)
    require(result['miss_scope'] in {'fixed_cohort','calendar_blocks','descriptive'},'Unknown miss uncertainty scope')
    require(result['dependence_audit'] is None or isinstance(result['dependence_audit'],str),'Invalid dependence audit')
    return result


def bernoulli_kl(p,q):
    require(0<=p<=1 and 0<=q<=1,'Invalid Bernoulli KL arguments')
    if p==q:
        return 0.0
    if q in (0,1):
        return math.inf
    return (p*math.log(p/q) if p else 0.0) + ((1-p)*math.log((1-p)/(1-q)) if p<1 else 0.0)


def bounded_mean_upper(mean,effective_count,alpha):
    """Invert the bounded-variable Chernoff bound; independence is an assumption.

    For independent Xi in [0,1] with fixed weights wi, effective_count=1/max(wi)
    is conservative. Equal weights give the ordinary sample count. At zero
    observed events the result is positive: 1-alpha**(1/effective_count).
    """
    require(0<=mean<=1 and effective_count>0 and 0<alpha<1,'Invalid upper-bound inputs')
    if mean==1:
        return 1.0
    if mean==0:
        return -math.expm1(math.log(alpha)/effective_count)
    threshold = -math.log(alpha)/effective_count
    low,high = mean,1.0
    for _ in range(80):
        middle = (low+high)/2
        if bernoulli_kl(mean,middle)>threshold:
            high = middle
        else:
            low = middle
    return high


class CalendarBlocks:
    def __init__(self,arrivals,settings):
        require(arrivals,'Empty episode cohort')
        self.settings = analysis_settings(settings)
        anchor = timestamp(self.settings['anchor_utc'])
        blocks = defaultdict(list)
        for identity,at in sorted(arrivals.items()):
            index = math.floor((timestamp(at)-anchor).total_seconds()/(3600*self.settings['block_hours']))
            blocks[index].append(identity)
        self.groups = dict(sorted(blocks.items()))
        self.ids = tuple(sorted(arrivals))
        self.weights = [len(ids)/len(self.ids) for ids in self.groups.values()]
        self.effective_count = 1/max(self.weights)
        self.interval_ready = len(self.groups)>=self.settings['minimum_blocks'] and bool(self.settings['dependence_audit'])

    def metadata(self):
        return {'blocks':len(self.groups),'episodes':len(self.ids),'block_sizes':[len(v) for v in self.groups.values()],
                'effective_blocks_for_bound':self.effective_count,'intervals_enabled':self.interval_ready,
                'dependence_assumption':'calendar blocks are approximately independent/exchangeable; audit is not a proof',
                'settings':self.settings}

    def draws(self,seed=None):
        rng = random.Random(self.settings['seed'] if seed is None else seed)
        groups = list(self.groups.values())
        for _ in range(self.settings['replicates']):
            yield [identity for _ in groups for identity in rng.choice(groups)]


def deadline_bound(observations,blocks,alpha,randomized=True):
    """Input id -> (known lower miss probability, conservative upper probability)."""
    require(set(observations)==set(blocks.ids),'Deadline cohort differs from resampling cohort')
    require(all(0<=lo<=hi<=1 for lo,hi in observations.values()),'Invalid miss probability interval')
    lower = fmean(v[0] for v in observations.values())
    upper = fmean(v[1] for v in observations.values())
    scope = blocks.settings['miss_scope']
    ready = False
    if scope=='fixed_cohort':
        # Deterministic fixed trajectories and exactly marginalized Fixed-Mix have
        # no action-sampling uncertainty on this finite, predeclared cohort.
        bound = bounded_mean_upper(upper,len(observations),alpha) if randomized else upper
        ready = True
        note = 'conditional on this frozen finite arrival cohort and trace; randomized episodes use independent action streams'
    elif scope=='calendar_blocks':
        bound = bounded_mean_upper(upper,blocks.effective_count,alpha)
        ready = blocks.interval_ready
        note = 'conditional on independent calendar blocks; arbitrary within-block dependence is allowed'
    else:
        bound = None
        note = 'descriptive outcomes only; no feasibility support asserted'
    return {'observed_lower':lower,'observed_upper':upper,'confidence_upper':bound,
            'alpha':alpha,'scope':scope,'assumptions':note,'support_enabled':ready,
            'unknown_probability_mass':upper-lower}


def paired_ratio_intervals(numerator,denominator,blocks,alpha=None):
    """Ratio of paired means with simultaneous one-sided endpoint bounds.

    Two endpoint values per id. The same calendar draw is used for both policies
    and both endpoints. No winner, budget or policy is selected inside a draw.
    """
    require(set(numerator)==set(denominator)==set(blocks.ids),'Unpaired carbon comparison')
    require(all(len(row)==2 and all(math.isfinite(v) and v>=0 for v in row)
                for rows in (numerator,denominator) for row in rows.values()),'Invalid carbon observations')
    def ratio(ids):
        bottom = [fmean(denominator[i][e] for i in ids) for e in range(2)]
        require(min(bottom)>0,'Nonpositive comparator mean carbon')
        return [fmean(numerator[i][e] for i in ids)/bottom[e] for e in range(2)]
    point = ratio(blocks.ids)
    result = {'endpoint_ratios':point,'ratio_definition':'ratio_of_means','two_sided_interval':None,
              'joint_one_sided_upper':None,'interval_wide_carbon_improvement_supported':False}
    if not blocks.interval_ready:
        return result
    draws = [ratio(ids) for ids in blocks.draws()]
    alpha = blocks.settings['alpha'] if alpha is None else alpha
    require(0<alpha<1,'Invalid comparison alpha')
    result['two_sided_interval'] = [[quantile([r[e] for r in draws],alpha/2),quantile([r[e] for r in draws],1-alpha/2)] for e in range(2)]
    upper = [quantile([r[e] for r in draws],1-alpha/2) for e in range(2)]
    result.update(joint_one_sided_upper=upper,interval_wide_carbon_improvement_supported=max(upper)<1,
                  nominal_level=1-alpha,method='paired calendar-block percentile bootstrap; one-sided Bonferroni across two endpoints')
    return result


def crossed_seed_ratio_intervals(by_seed,denominator,blocks):
    """Descriptive crossed seed/calendar bootstrap; never pool S*N as IID jobs.

    Seeds and blocks are crossed, not nested. Resample one seed multiset and one
    calendar-block multiset per replicate, using their Cartesian product. With
    few training seeds this is an approximate variability diagnostic, not a
    calibrated guarantee over arbitrary future training seeds.
    """
    require(by_seed and all(set(rows)==set(denominator)==set(blocks.ids) for rows in by_seed.values()),
            'Cross-seed comparison requires all seeds and identical calendar pairs')
    require(all(len(row)==2 and all(math.isfinite(v) and v>=0 for v in row)
                for rows in [denominator,*by_seed.values()] for row in rows.values()),'Invalid carbon observations')
    seeds = tuple(sorted(by_seed))
    def ratio(ids,chosen):
        bottom = [fmean(denominator[i][e] for i in ids) for e in range(2)]
        require(min(bottom)>0,'Nonpositive comparator mean carbon')
        return [fmean(by_seed[s][i][e] for s in chosen for i in ids)/bottom[e] for e in range(2)]
    result = {'endpoint_ratios':ratio(blocks.ids,seeds),'two_sided_interval':None,
              'seed_count':len(seeds),'seed_resampling':'one seed multiset crossed with the same calendar draw for both endpoints',
              'scope':'approximate variability across declared seeds and calendar blocks; few seeds limit calibration',
              'confirmatory_claim':False}
    if not blocks.interval_ready or len(seeds)<2:
        return result
    rng = random.Random(blocks.settings['seed']+17041)
    draws = [ratio(ids,[rng.choice(seeds) for _ in seeds]) for ids in blocks.draws()]
    alpha = blocks.settings['alpha']
    result['two_sided_interval'] = [[quantile([r[e] for r in draws],alpha/2),
                                      quantile([r[e] for r in draws],1-alpha/2)] for e in range(2)]
    return result
