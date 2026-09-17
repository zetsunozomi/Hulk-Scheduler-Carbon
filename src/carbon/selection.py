"""Validation-only operating-point selection, frozen before any test comparison."""

from collections import defaultdict
import math
from pathlib import Path

from .common import digest, integer, json_text, load_json, number, require
from .results import ResultRun
from .runner import write_manifest
from .statistics import CalendarBlocks, analysis_settings, bounded_mean_upper, deadline_bound


def selection_rank(record,rule):
    cost = record['summary']['worst_normalized_carbon']
    bound = record['deadline']['confidence_upper']
    return (bound if rule=='confidence_upper_miss' and bound is not None else record['deadline']['observed_upper'],
            cost if cost is not None else math.inf,record['id'])


def select_record(records,rule):
    eligible = [r for r in records if r['eligible']]
    if eligible:
        best = min(eligible,key=lambda r:(r['summary']['worst_normalized_carbon'],r['id']))
        return {'selected_id':best['id'],'supported':best['confidence_eligible'],
                'empirical_target_met':best['empirical_eligible'],'selection_criterion_met':True,
                'selection_reason':'minimum worst normalized mean carbon under the frozen '+rule+' rule'}
    complete = [r for r in records if r['summary']['complete']]
    best = min(complete,key=lambda r:selection_rank(r,rule)) if complete else None
    return {'selected_id':best['id'] if best else None,'supported':False,
            'empirical_target_met':best['empirical_eligible'] if best else False,'selection_criterion_met':False,
            'selection_reason':'no candidate meets '+rule+'; retain least-miss complete candidate for descriptive test reporting only' if best
                               else 'no candidate has a complete cohort'}


def select_policies(spec_path,output):
    spec_path,output = Path(spec_path),Path(output)
    require(not output.exists(),f'Output already exists: {output}')
    spec = load_json(spec_path)
    required = {'schema_version','references','budgets','epsilon','statistics','required_policy_seeds','candidates'}
    require(required<=set(spec)<=required|{'selection_rule'},
            'Selection spec has missing/unknown fields')
    require(spec['schema_version']==1,'Unsupported selection spec')
    root = spec_path.parent
    references = load_json(root/spec['references'])
    budgets = [float(number(b,'budget',strict=True)) for b in spec['budgets']]
    require(budgets and len(set(budgets))==len(budgets),'Budgets must be nonempty and unique')
    epsilon = float(number(spec['epsilon'],'epsilon'))
    require(epsilon<=1,'epsilon must be at most one')
    seeds = [integer(s,'policy seed',0) for s in spec['required_policy_seeds']]
    require(seeds and len(seeds)==len(set(seeds)),'Declare unique policy seeds')
    settings = analysis_settings(spec['statistics'])
    require(references['purpose']!='research' or settings['miss_scope']=='calendar_blocks',
            'Research selection follows the manuscript calendar-block scope; fixed-cohort bounds are diagnostic only')
    require(references['purpose']!='research' or 'selection_rule' in spec,
            'Research must explicitly freeze selection_rule before validation')
    rule = spec.get('selection_rule','confidence_upper_miss')
    require(rule in {'empirical_miss','confidence_upper_miss'},'Unknown selection_rule')
    audit_hash = None
    if settings['dependence_audit']:
        audit = root/settings['dependence_audit']
        require(audit.is_file() and audit.stat().st_size>0,'Dependence audit must name a nonempty file')
        audit_hash = digest(audit)
    entries = spec['candidates']
    require(entries and len({r['id'] for r in entries})==len(entries),'Candidate IDs must be unique')
    loaded,views,bindings = {},{},{}
    cohort = None
    replay_hashes = None
    baseline_seeds = defaultdict(set)
    groups = defaultdict(list)
    for entry in entries:
        required = {'id','kind','run','method','budgets'}
        require(required<=set(entry)<=required|{'seed','model'},'Invalid candidate specification')
        require(isinstance(entry['id'],str) and entry['id'],'Empty candidate ID')
        kind = entry['kind']
        require(kind in {'fixed','planner','policy','mixture'},'Invalid candidate kind')
        candidate_budgets = [float(number(b,'candidate budget',strict=True)) for b in entry['budgets']]
        require(candidate_budgets and len(set(candidate_budgets))==len(candidate_budgets) and set(candidate_budgets)<=set(budgets),
                'Candidate budgets must be a unique subset of the frozen grid')
        path = (root/entry['run']).resolve()
        if path not in loaded:
            loaded[path] = ResultRun(path,references,allowed_splits={'train','validation'})
        run = loaded[path]
        # Research validation must not consume a file that already contains test outcomes.
        if run.manifest['purpose']=='research':
            require(not any(r['split']=='test' for r in run.rows),'Validation input already contains test results')
        core_names = ['config.py','trace.py','replay.py','workload.py','carbon.py','environment.py']
        core = {name:run.manifest['software']['source_sha256'][name] for name in core_names}
        if replay_hashes is None:
            replay_hashes = core
        require(core==replay_hashes,'Candidates used different replay/accounting implementations')
        seed = entry.get('seed')
        if kind=='policy':
            require(seed in seeds and seed==run.manifest['training_seed'],'Missing/unexpected policy seed')
        elif kind=='planner':
            baseline_seeds[entry['method']].add(run.manifest['seed'])
            require(len(baseline_seeds[entry['method']])==1,'Do not select the best planning seed; freeze its randomness before comparison')
        binding = {**entry,'run_provenance':run.provenance,'panel':run.manifest['panel'],'purpose':run.manifest['purpose']}
        if kind=='policy':
            binding['checkpoint_sha256'] = run.manifest['checkpoint_sha256']
            binding['sampling_seed'] = run.manifest['sampling_seed']
            binding['source_sha256'] = run.manifest['software']['source_sha256']
            binding['predictor_sha256'] = run.manifest['predictor_sha256']
        if kind=='planner':
            binding['planner_settings'] = {key:run.manifest[key] for key in
                                           ('seed','planning_paths','planner_internal_miss_tolerance','predictor_sha256')}
        if kind=='mixture':
            require(entry['method']=='Fixed-Mix' and 'model' in entry,'Mixture model is required')
            binding['mixture_sha256'] = digest(root/entry['model'])
            model = load_json(root/entry['model'])
            binding['mixture_model'] = model
            require(math.isclose(model['epsilon'],epsilon),'Fixed-Mix tolerance differs from selection')
        bindings[entry['id']] = binding
        for beta in candidate_budgets:
            view = run.mixture(root/entry['model'],beta,'validation') if kind=='mixture' else run.candidate(entry['method'],beta,'validation',seed,kind)
            if view is not None:
                if cohort is None:
                    cohort = view.arrivals
                require(view.arrivals==cohort,'Every candidate must cover the identical predeclared validation cohort')
            key = (entry['id'],str(beta))
            views[key] = view
            groups[(str(beta),entry['method'],str(seed) if kind=='policy' else 'baseline')].append(key)
    require(cohort,'No validation cohort')
    # Each policy method is evaluated for all declared seeds wherever it is used.
    for beta,method in {(b,m) for b,m,s in groups if s!='baseline'}:
        require({s for b,m,s in groups if b==beta and m==method}=={str(s) for s in seeds},
                f'Missing policy seeds for {method}, beta={beta}; do not keep only successful seeds')
    for beta in budgets:
        require(all((str(beta),f'Fixed-{n}','baseline') in groups for n in references['resolved_config']['cluster']['allowed_nodes']),
                'All fixed scales must remain in the comparison')
        require(all((str(beta),'ScaleDown',str(seed)) in groups for seed in seeds),'Full policy requires every budget and seed')
        require(all((str(beta),method,'baseline') in groups for method in ('Fixed-Mix','Rollout-MPC','Plan-once')),
                'Primary comparison requires Fixed-Mix, Rollout-MPC and Plan-once for every budget, including infeasible entries')
    blocks = CalendarBlocks(cohort,settings)
    # Uniform bounds across all candidates make the validation selection explicit.
    # Fixed-Mix is data-dependent: its block-scope bound uses simultaneous fixed-component bounds below.
    alpha = settings['alpha']/len(views)
    records = {}
    for key,view in views.items():
        if view is None:
            records[key] = {'id':key[0],'beta':float(key[1]),'summary':{'method':'Fixed-Mix','kind':'mixture','complete':False,
                           'worst_normalized_carbon':None,'reason':'validation LP infeasible'},
                           'deadline':{'confidence_upper':None,'observed_upper':1.,'support_enabled':False},
                           'empirical_eligible':False,'confidence_eligible':False,'eligible':False}
            continue
        summary = view.summary(references)
        deadline = deadline_bound(view.misses(),blocks,alpha,view.randomized)
        if view.kind=='mixture' and settings['miss_scope']=='calendar_blocks':
            entry = bindings[key[0]]
            run = loaded[(root/entry['run']).resolve()]
            model = load_json(root/entry['model'])
            component_bounds = {}
            for n in model['weights']:
                fixed = run.candidate('Fixed-'+n,view.beta,'validation',kind='fixed')
                component_bounds[n] = deadline_bound(fixed.misses(),blocks,alpha,False)['confidence_upper']
            deadline['confidence_upper'] = sum(w*component_bounds[n] for n,w in model['weights'].items())
            deadline['adaptive_mixture_rule'] = 'weighted simultaneous fixed-component bounds; weights fitted on this validation cohort'
        empirical = bool(summary['complete'] and deadline['observed_upper']<=epsilon)
        confidence = bool(summary['complete'] and deadline['support_enabled'] and deadline['confidence_upper']<=epsilon)
        records[key] = {'id':key[0],'beta':view.beta,'summary':summary,'deadline':deadline,
                        'empirical_eligible':empirical,'confidence_eligible':confidence,
                        'eligible':empirical if rule=='empirical_miss' else confidence}
    selections = {}
    for (beta,method,seed),keys in sorted(groups.items()):
        selected = select_record([records[key] for key in keys],rule)
        selections.setdefault(beta,{}).setdefault(method,{})[seed] = selected
    comparators = {}
    for beta in map(str,budgets):
        fixed = [records[(point['baseline']['selected_id'],beta)] for method,point in selections[beta].items()
                 if method.startswith('Fixed-') and method!='Fixed-Mix' and point['baseline']['selected_id'] is not None]
        best_fixed = select_record(fixed,rule)
        contenders = [records[(best_fixed['selected_id'],beta)]] if best_fixed['selected_id'] else []
        for method in ('Fixed-Mix','Rollout-MPC','Plan-once'):
            selected = selections[beta].get(method,{}).get('baseline',{}).get('selected_id')
            if selected:
                contenders.append(records[(selected,beta)])
        comparators[beta] = {'best_fixed':best_fixed,'strongest_non_rl':select_record(contenders,rule)}
    output.mkdir(parents=True,exist_ok=False)
    result = {'kind':'validation_selection_v1','status':'complete','scope':'validation_only','purpose':references['purpose'],
              'panel':references['panel'],'references':references,'references_sha256':digest(root/spec['references']),
              'selection_spec_sha256':digest(spec_path),'budgets':budgets,'epsilon':epsilon,'policy_seeds':seeds,
              'statistics':blocks.metadata(),'dependence_audit_sha256':audit_hash,'replay_source_sha256':replay_hashes,
              'zero_miss_calendar_bound_floor':bounded_mean_upper(0.,blocks.effective_count,alpha),
              'minimum_equal_independent_blocks_for_zero_misses':(math.ceil(math.log(alpha)/math.log1p(-epsilon)) if 0<epsilon<1 else 1 if epsilon==1 else None),
              'selection_rule':rule,'alpha_per_candidate':alpha,
              'selection_adjustment':'confidence bounds use Bonferroni over all declared candidate-budget views; empirical eligibility uses observed miss only',
              'validation_cohort':cohort,'candidates':bindings,'operating_points':selections,'comparators':comparators,
              'records':list(records.values()),
              'claim_limit':'empirical target attainment describes the evaluated cohort; supported requires the separate miss confidence bound and its assumptions'}
    write_manifest(output/'selection.json',result)
    lines = ['# Validation selection','',f"Panel: {result['panel']}. Selection rule: {rule}. Uncertainty scope: {settings['miss_scope']}. No test results were used.",'',
             '| Budget | Method | Seed | Selected candidate | Observed target met | Miss bound supported |','|---|---|---|---|---|---|']
    for beta,methods in selections.items():
        for method,points in methods.items():
            for seed,point in points.items():
                lines.append(f"| {beta} | {method} | {seed} | {point['selected_id'] or 'none'} | {point['empirical_target_met']} | {point['supported']} |")
    lines += ['','Observed target attainment is not a population-feasibility certificate. Failure to support a miss bound does not prove infeasibility.']
    (output/'selection.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write_manifest(output/'manifest.json',{'status':'complete','kind':'validation_selection','selection_sha256':digest(output/'selection.json'),
                                         'spec_sha256':digest(spec_path),'purpose':references['purpose']})
    return result
