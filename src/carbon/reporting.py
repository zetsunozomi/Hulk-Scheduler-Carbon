"""Paired held-out reports, preserving censored/unsupported outcomes and every seed."""

from pathlib import Path

from .common import digest, load_json, require
from .heldout import frozen_jobs, load_selection
from .power_analysis import constant_ci_rescore, exposure_comparison, phase_accounting
from .results import CandidateView, ResultRun
from .runner import write_manifest
from .statistics import CalendarBlocks, crossed_seed_ratio_intervals, deadline_bound, paired_ratio_intervals


def mean_seed_view(views):
    require(views and all(v.arrivals==views[0].arrivals for v in views),'Unpaired seed cohorts')
    require(all(v.method==views[0].method and v.beta==views[0].beta for v in views),'Mixed seed operating points')
    return CandidateView(views[0].method,'policy_seed_average',views[0].beta,views[0].budget_hours,
                         {i:[(w/len(views),r) for view in views for w,r in view.parts[i]] for i in views[0].parts},True)


def load_test_inputs(directory):
    root = Path(directory)
    manifest = load_json(root/'manifest.json')
    require(manifest['kind']=='heldout_execution_v1' and manifest['status']=='complete','Held-out execution is incomplete/failed')
    require(digest(root/'plan.json')==manifest['plan_sha256'],'Held-out plan hash mismatch')
    plan = load_json(root/'plan.json')
    selected = load_selection(root/'selection')
    require(digest(root/'selection/selection.json')==plan['selection_sha256'],'Test execution used a different selection')
    require(plan['jobs']==frozen_jobs(selected),'Held-out job list differs from frozen selections')
    require(set(manifest['runs'])==set(plan['jobs']),'Missing/extra selected candidate runs')
    require(plan['asset_sha256']==selected['references']['asset_sha256'],'Test cohort or input assets changed after selection')
    require(plan['purpose']==selected['purpose'] and plan['panel']==selected['panel'],'Test identity differs from selection')
    require(set(plan['test_cohort']).isdisjoint(selected['validation_cohort']),'Test identities overlap validation')
    loaded,views = {},{}
    for identity,betas in plan['jobs'].items():
        entry = selected['candidates'][identity]
        require(set(manifest['runs'][identity])==set(map(str,betas)),'Missing/extra selected budget results')
        for beta in betas:
            relative = manifest['runs'][identity][str(beta)]
            require(Path(relative).name==relative,'Invalid held-out result path')
            if relative not in loaded:
                for name in ('manifest.json','episodes.jsonl','chunks.jsonl'):
                    require(digest(root/relative/name)==manifest['run_files_sha256'][relative][name],f'Held-out file hash mismatch: {relative}/{name}')
                if 'plans.jsonl' in manifest['run_files_sha256'][relative]:
                    require(digest(root/relative/'plans.jsonl')==manifest['run_files_sha256'][relative]['plans.jsonl'],
                            'Held-out precommitted plan hash mismatch')
                loaded[relative] = ResultRun(root/relative,selected['references'],allowed_splits={'test'})
            run = loaded[relative]
            require(all(run.manifest['software']['source_sha256'][n]==h for n,h in selected['replay_source_sha256'].items()),
                    'Test replay/accounting implementation differs')
            if entry['kind']=='planner':
                require(all(run.manifest[k]==v for k,v in entry['planner_settings'].items()),'Planner settings changed after selection')
            if entry['kind']=='policy':
                for key in ('checkpoint_sha256','predictor_sha256','sampling_seed'):
                    require(run.manifest[key]==entry[key],f'Policy {key} changed after selection')
                require(run.manifest['training_seed']==entry['seed'],'Test training seed differs')
                require(run.manifest['software']['source_sha256']==entry['source_sha256'],'Test policy code differs')
            view = run.mixture(entry['mixture_model'],beta,'test') if entry['kind']=='mixture' else \
                   run.candidate(entry['method'],beta,'test',entry.get('seed'),entry['kind'])
            require(view is not None and view.arrivals==plan['test_cohort'],'Test results do not cover the entire frozen cohort')
            views[(identity,str(beta))] = (view,run)
    return selected,plan,views,manifest


def report_test(directory,output):
    output = Path(output)
    require(not output.exists(),f'Output already exists: {output}')
    selected,plan,views,execution = load_test_inputs(directory)
    references = selected['references']
    settings = selected['statistics']['settings']
    blocks = CalendarBlocks(plan['test_cohort'],settings)
    require(not settings['dependence_audit'] or selected['dependence_audit_sha256'], 'Unsealed development dependence audit')
    endpoints = references['resolved_config']['power']['rho_interval']
    power = references['resolved_config']['power']
    # One miss bound per frozen view and two carbon endpoint bounds per contrast.
    # Count every declared comparison, including those that later prove censored.
    contrasts = sum(1 for identity,beta in views if identity!=selected['comparators'][beta]['strongest_non_rl']['selected_id'])
    family = len(views)+2*contrasts
    alpha = settings['alpha']/max(1,family)
    validation_records = {(r['id'],str(r['beta'])):r for r in selected['records']}
    records,record_map = [],{}
    for beta,methods in selected['operating_points'].items():
        for method,points in methods.items():
            for seed,point in points.items():
                identity = point['selected_id']
                validation = validation_records.get((identity,beta))
                observed_validation = bool(validation and validation['summary']['complete'] and
                                           validation['deadline']['observed_upper']<=selected['epsilon'])
                record = {'beta':float(beta),'method':method,'seed':seed,'selected_id':identity,
                          'validation_supported':point['supported'],'validation_empirical_target_met':observed_validation,
                          'summary':None,'deadline':None,'test_feasible':False,'test_empirical_target_met':False}
                if identity is None:
                    record['unavailable_reason'] = point['selection_reason']
                else:
                    view,run = views[(identity,beta)]
                    summary = view.summary(references)
                    bound = deadline_bound(view.misses(),blocks,alpha,view.randomized)
                    record.update(summary=summary,deadline=bound,
                                  test_empirical_target_met=bool(summary['complete'] and bound['observed_upper']<=selected['epsilon']),
                                  test_feasible=bool(summary['complete'] and bound['support_enabled'] and bound['confidence_upper']<=selected['epsilon']))
                    if summary['complete']:
                        phases = phase_accounting(run,view)
                        record['phases'] = phases
                        record['constant_ci_rescore'] = constant_ci_rescore(phases,run.work,power,plan['constant_ci_rescore_value'],endpoints)
                        record['by_calendar_block'] = []
                        for index,ids in blocks.groups.items():
                            block_view = CandidateView(view.method,view.kind,view.beta,view.budget_hours,{i:view.parts[i] for i in ids},view.randomized)
                            record['by_calendar_block'].append({'index':index,**block_view.summary(references)})
                    record_map[(identity,beta)] = record
                records.append(record)
    comparisons = []
    for beta,methods in selected['operating_points'].items():
        comparator = selected['comparators'][beta]['strongest_non_rl']
        other_id = comparator['selected_id']
        for method,points in methods.items():
            for seed,point in points.items():
                identity = point['selected_id']
                if identity==other_id and identity is not None:
                    continue
                result = {'beta':float(beta),'method':method,'seed':seed,'candidate_id':identity,'comparator_id':other_id,
                          'carbon':None,'power':None,'observed_budgeted_carbon_reduction':False,
                          'budgeted_interval_improvement_supported':False}
                if identity is None or other_id is None:
                    result['unavailable_reason'] = 'validation provided no complete candidate or comparator'
                else:
                    left,run = views[(identity,beta)];right,_ = views[(other_id,beta)]
                    if left.complete() and right.complete():
                        carbon = paired_ratio_intervals(left.endpoint_costs(endpoints),right.endpoint_costs(endpoints),blocks,2*alpha)
                        result['carbon'] = carbon
                        result['power'] = exposure_comparison(left,right,run.work,power,plan['power_analysis_nominal_rho'])
                        result['observed_budgeted_carbon_reduction'] = bool(
                            all(record_map[(i,beta)]['validation_empirical_target_met'] and
                                record_map[(i,beta)]['test_empirical_target_met'] for i in (identity,other_id)) and
                            max(carbon['endpoint_ratios'])<1)
                        # Simultaneous, approximate bootstrap evidence for these frozen
                        # policies only. No claim over an arbitrary future training seed.
                        result['budgeted_interval_improvement_supported'] = bool(
                            point['supported'] and comparator['supported'] and
                            record_map[(identity,beta)]['test_feasible'] and record_map[(other_id,beta)]['test_feasible'] and
                            carbon['interval_wide_carbon_improvement_supported'] and
                            settings['replicates']*alpha>=10)
                    else:
                        result['unavailable_reason'] = 'censored probability mass: full-cohort carbon is unknown'
                comparisons.append(result)
    aggregates = []
    for beta,methods in selected['operating_points'].items():
        for method,points in methods.items():
            if 'baseline' in points:
                continue
            require(set(points)==set(map(str,selected['policy_seeds'])),'Missing declared seed in frozen policy group')
            ids = {seed:p['selected_id'] for seed,p in points.items()}
            aggregate = {'beta':float(beta),'method':method,'seeds':selected['policy_seeds'],'selected_ids':ids,
                         'summary':None,'carbon_variability':None,'power':None,'all_declared_seed_contrasts_supported':False}
            if all(identity is not None for identity in ids.values()):
                seed_views = {seed:views[(identity,beta)][0] for seed,identity in ids.items()}
                pooled = mean_seed_view(list(seed_views.values()))
                aggregate['summary'] = pooled.summary(references)
                aggregate['summary_scope'] = 'equal weights across all frozen checkpoints; p95 of pooled distribution, not mean of seed p95s'
                other_id = selected['comparators'][beta]['strongest_non_rl']['selected_id']
                if other_id is not None and pooled.complete() and views[(other_id,beta)][0].complete():
                    aggregate['carbon_variability'] = crossed_seed_ratio_intervals(
                        {seed:v.endpoint_costs(endpoints) for seed,v in seed_views.items()},
                        views[(other_id,beta)][0].endpoint_costs(endpoints),blocks)
                    work = views[(next(iter(ids.values())),beta)][1].work
                    aggregate['power'] = exposure_comparison(pooled,views[(other_id,beta)][0],work,power,
                                                             plan['power_analysis_nominal_rho'])
                relevant = [r for r in comparisons if r['method']==method and r['beta']==float(beta)]
                aggregate['all_declared_seed_contrasts_supported'] = (len(relevant)==len(points) and all(r['budgeted_interval_improvement_supported'] for r in relevant))
            else:
                aggregate['unavailable_reason'] = 'at least one declared seed has no complete validation candidate; seeds are not dropped'
            aggregates.append(aggregate)
    result = {'kind':'heldout_report_v1','purpose':selected['purpose'],'panel':selected['panel'],
              'selection_sha256':plan['selection_sha256'],'test_plan_sha256':execution['plan_sha256'],
              'references':references,'statistics':blocks.metadata(),'epsilon':selected['epsilon'],
              'selection_rule':selected.get('selection_rule','confidence_upper_miss'),
              'observed_comparison_scope':'complete frozen validation/test cohorts; point estimates do not certify population feasibility or statistical significance',
              'simultaneous_family_size':family,'alpha_per_one_sided_bound':alpha,
              'bootstrap_expected_draws_in_tail':settings['replicates']*alpha,
              'bootstrap_tail_precision_sufficient':settings['replicates']*alpha>=10,
              'uncertainty_scope':'miss bounds condition on frozen policies and independent calendar blocks; carbon intervals are approximate; joint evidence is per panel across listed budgets and contrasts',
              'seed_claim_scope':'all-declared-seed evidence concerns these checkpoints; crossed resampling is descriptive variability, not a guarantee for future training seeds',
              'research_evidence':selected['purpose']=='research','records':records,'comparisons':comparisons,'seed_aggregates':aggregates,
              'comparators':selected['comparators'],'input_manifest_sha256':digest(Path(directory)/'manifest.json')}
    output.mkdir(parents=True)
    write_manifest(output/'report.json',result)
    lines = ['# Held-out results','',f"Panel: {result['panel']}. Purpose: {result['purpose']}.",'',
             'All operating points, seeds and the comparator were frozen on validation. Observed target attainment and miss-bound support are separate.','',
             '| Budget | Method | Seed | Validation observed target | Test observed target | Validation miss bound | Test miss bound | Miss lower/upper | Mean TAT (h) |',
             '|---|---|---|---|---|---|---|---|---|']
    for row in records:
        summary = row['summary']
        miss = f"{summary['miss_lower']:.4f}/{summary['miss_upper']:.4f}" if summary else 'unavailable'
        tat = f"{summary['mean_tat_hours']:.4f}" if summary and summary['mean_tat_hours'] is not None else 'unknown'
        lines.append(f"| {row['beta']} | {row['method']} | {row['seed']} | {row['validation_empirical_target_met']} | {row['test_empirical_target_met']} | {row['validation_supported']} | {row['test_feasible']} | {miss} | {tat} |")
    lines += ['',f"Calendar blocks: {len(blocks.groups)}; interval readiness: {blocks.interval_ready}.",
              f"Expected bootstrap draws in each adjusted tail: {settings['replicates']*alpha:.2f}; at least 10 are required for a support flag.",
              'Observed carbon reductions describe these complete cohorts. Miss bounds and cost intervals must both pass for the stronger budgeted-improvement support flag; an inconclusive bound is not proven infeasibility.',
              'No full-carbon claim is made for censored cohorts.',
              'Detailed endpoint costs/ratios, each seed, crossed seed/calendar variability, block summaries, phase costs and power sensitivity are in report.json.',
              'Synthetic/development outputs are software/development checks, not paper results.']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write_manifest(output/'manifest.json',{'status':'complete','kind':'heldout_report','report_sha256':digest(output/'report.json'),
                                         'purpose':selected['purpose'],'input_manifest_sha256':result['input_manifest_sha256']})
    return result
