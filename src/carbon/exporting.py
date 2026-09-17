"""Publication data derived from sealed held-out logs; no policy reselection."""

from collections import Counter
import math
from pathlib import Path
import random
from statistics import fmean
import shutil

from .common import digest, load_json, require
from .features import quantile
from .power_analysis import exposure_comparison
from .reporting import load_test_inputs, mean_seed_view
from .runner import provenance, write_manifest
from .statistics import CalendarBlocks


E3_METHODS = ('ScaleDown','Rollout-MPC','Queue-blind-MPC','Current-CI','Plan-once')


def carbon_at(a,b,rho,power):
    require(0<=rho<=1,'Power scenario is outside [0,1]')
    return power['reference_kw']*(b+rho*(a-b))


def reference_at(references,rho):
    lo,hi = references['resolved_config']['power']['rho_interval']
    values = references['carbon_reference_g_per_kappa']
    fraction = (rho-lo)/(hi-lo)
    value = values[str(lo)]*(1-fraction)+values[str(hi)]*fraction
    require(value>0,'Nonpositive interpolated training carbon reference')
    return value


def episode_moments(view):
    require(view.complete(),'Full-cohort moments are unknown under censoring')
    return {identity:[sum(w*row[field] for w,row in parts) for field in ('tat_hours','nodehours','A','B')]
            for identity,parts in view.parts.items()}


def paired_plot_statistics(by_seed,denominator,blocks,power,nominal_rho,reference):
    """Crossed seed/block means, paired ratio of means, and marginal plot bars.

    Summing by block avoids expanding every arrival in every bootstrap replicate.
    Denominator is a fixed validation-selected non-RL policy, possibly an exact mix.
    These intervals visualize variability; confirmatory flags come from report.json.
    """
    require(by_seed,'No seed outcomes')
    ids = set(blocks.ids)
    require(all(set(rows)==ids for rows in by_seed.values()) and (denominator is None or set(denominator)==ids),
            'Plot statistics require identical complete paired cohorts')
    require(all(len(row)==4 and all(math.isfinite(v) and v>=0 for v in row)
                for rows in [*by_seed.values(),*([denominator] if denominator else [])] for row in rows.values()),'Invalid plot moments')
    seeds = tuple(sorted(by_seed))
    group_ids = list(blocks.groups.values())
    sizes = [len(group) for group in group_ids]
    sums = {seed:[[math.fsum(rows[i][k] for i in group) for k in range(4)] for group in group_ids]
            for seed,rows in by_seed.items()}
    other = [[math.fsum(denominator[i][k] for i in group) for k in range(4)] for group in group_ids] if denominator else None
    endpoints = power['rho_interval']
    def evaluate(block_counts,seed_counts):
        count = sum(sizes[b]*n for b,n in block_counts.items())
        seed_count = sum(seed_counts.values())
        means = [math.fsum(sums[s][b][k]*nb*ns for s,ns in seed_counts.items() for b,nb in block_counts.items())/(count*seed_count)
                 for k in range(4)]
        result = {'mean_tat_hours':means[0],'mean_nodehours':means[1],
                  'normalized_carbon':carbon_at(means[2],means[3],nominal_rho,power)/reference}
        if other is not None:
            base = [math.fsum(other[b][k]*n for b,n in block_counts.items())/count for k in range(4)]
            for endpoint,rho in enumerate(endpoints):
                bottom = carbon_at(base[2],base[3],rho,power)
                require(bottom>0,'Nonpositive paired comparator carbon')
                result[f'endpoint_ratio_{endpoint}'] = carbon_at(means[2],means[3],rho,power)/bottom
        return result
    point = evaluate(Counter(range(len(sizes))),Counter(seeds))
    result = {'point':point,'intervals':None,'seed_count':len(seeds),'calendar_blocks':len(sizes),
              'interval_scope':'marginal percentile block intervals, with crossed seed resampling for policies; approximate variability, not a simultaneous claim'}
    if not blocks.interval_ready:
        result['intervals_unavailable_reason'] = 'too few calendar blocks or no development dependence audit'
        return result
    rng = random.Random(blocks.settings['seed']);seed_rng = random.Random(blocks.settings['seed']+17041)
    samples = {key:[] for key in point}
    for _ in range(blocks.settings['replicates']):
        counts = Counter(rng.randrange(len(sizes)) for _ in sizes)
        chosen = Counter(seed_rng.choice(seeds) for _ in seeds) if len(seeds)>1 else Counter(seeds)
        for key,value in evaluate(counts,chosen).items():samples[key].append(value)
    alpha = blocks.settings['alpha']
    result['intervals'] = {key:[quantile(values,alpha/2),quantile(values,1-alpha/2)] for key,values in samples.items()}
    return result


def group_status(records,epsilon):
    if any(row['summary'] is None for row in records):return 'unavailable'
    if any(not row['summary']['complete'] for row in records):return 'censored'
    if all(row['validation_supported'] and row['test_feasible'] for row in records):return 'supported'
    if any(row['summary']['miss_lower']>epsilon for row in records):return 'miss above tolerance'
    if all(row.get('validation_empirical_target_met',False) and row.get('test_empirical_target_met',False) for row in records):
        return 'observed target met'
    return 'unsupported'


def load_export_data(run_directory,report_directory):
    report_directory = Path(report_directory)
    seal = load_json(report_directory/'manifest.json')
    require(seal['kind']=='heldout_report' and seal['status']=='complete','Incomplete held-out report')
    require(digest(report_directory/'report.json')==seal['report_sha256'],'Report hash mismatch')
    report = load_json(report_directory/'report.json')
    selected,plan,views,execution = load_test_inputs(run_directory)
    require(report['input_manifest_sha256']==digest(Path(run_directory)/'manifest.json') and
            report['test_plan_sha256']==execution['plan_sha256'] and report['selection_sha256']==plan['selection_sha256'],
            'Report and raw held-out execution differ')
    require(report['references']==selected['references'] and report['purpose']==selected['purpose'],'Report identity/references differ')
    references = selected['references'];power = references['resolved_config']['power']
    nominal = plan['power_analysis_nominal_rho'];reference = reference_at(references,nominal)
    blocks = CalendarBlocks(plan['test_cohort'],selected['statistics']['settings'])
    expected = {(float(beta),method,seed) for beta,methods in selected['operating_points'].items() for method,points in methods.items() for seed in points}
    actual = {(r['beta'],r['method'],r['seed']) for r in report['records']}
    require(expected==actual and len(actual)==len(report['records']),'Report omitted/duplicated operating points or seeds')
    records = {(str(r['beta']),r['method'],r['seed']):r for r in report['records']}
    comparisons = {(str(r['beta']),r['method'],r['seed']):r for r in report['comparisons']}
    groups = []
    for beta,methods in selected['operating_points'].items():
        comparator_id = selected['comparators'][beta]['strongest_non_rl']['selected_id']
        comparator = views[(comparator_id,beta)][0] if comparator_id is not None else None
        for method,points in methods.items():
            rows = [records[(beta,method,seed)] for seed in points]
            group = {'beta':float(beta),'method':method,'seed_labels':list(points),
                     'selected_ids':{seed:point['selected_id'] for seed,point in points.items()},
                     'status':group_status(rows,selected['epsilon']),'summary':None,'statistics':None,'power':None,
                     'seed_points':[],'seed_records':rows,'comparator_id':comparator_id,
                     'comparator_method':selected['candidates'][comparator_id]['method'] if comparator_id else None,
                     'joint_budgeted_improvement_supported_for_all_seeds':all(
                         comparisons.get((beta,method,seed),{}).get('budgeted_interval_improvement_supported',False) for seed in points)}
            chosen = {}
            for seed,point in points.items():
                require(records[(beta,method,seed)]['selected_id']==point['selected_id'],'Report changed validation selection')
                if point['selected_id'] is not None:
                    view,run = views[(point['selected_id'],beta)]
                    require(records[(beta,method,seed)]['summary']==view.summary(references),'Report summary differs from sealed raw outcomes')
                    chosen[seed] = view
                    if view.complete():
                        moments = episode_moments(view)
                        a,b = (fmean(row[k] for row in moments.values()) for k in (2,3))
                        group['seed_points'].append({'seed':seed,'mean_tat_hours':fmean(row[0] for row in moments.values()),
                                                     'normalized_carbon':carbon_at(a,b,nominal,power)/reference})
            if len(chosen)==len(points):
                pooled = next(iter(chosen.values())) if 'baseline' in points else mean_seed_view(list(chosen.values()))
                group['summary'] = pooled.summary(references)
                if pooled.complete():
                    other = episode_moments(comparator) if comparator is not None and comparator.complete() else None
                    group['statistics'] = paired_plot_statistics({seed:episode_moments(v) for seed,v in chosen.items()},
                                                                 other,blocks,power,nominal,reference)
                    if other is not None:
                        group['power'] = exposure_comparison(pooled,comparator,run.work,power,nominal)
                        exposures = group['power']
                        eta = {str(n):float(run.work.eta(n)) for n in run.work.profiles}
                        left = exposures['mean_exposure_candidate'];right = exposures['mean_exposure_comparator']
                        alo,blo = sum(left.values()),sum(eta[n]*v for n,v in left.items())
                        ahi,bhi = sum(right.values()),sum(eta[n]*v for n,v in right.items())
                        lo,hi = power['rho_interval']
                        grid = sorted(set([0.,lo,hi,*[lo+(hi-lo)*j/40 for j in range(41)]]))
                        group['power_curve'] = []
                        for rho in grid:
                            bottom = carbon_at(ahi,bhi,rho,power)
                            require(bottom>0,'Nonpositive power-sweep comparator')
                            group['power_curve'].append({'rho':rho,'ratio_of_means':carbon_at(alo,blo,rho,power)/bottom,
                                                         'scenario':'ideal-limit stress' if rho<lo else 'declared interval'})
            groups.append(group)
    return {'kind':'paper_export_data_v1','panel':selected['panel'],'purpose':selected['purpose'],
            'selection_rule':selected.get('selection_rule','confidence_upper_miss'),
            'nominal_rho':nominal,'rho_interval':power['rho_interval'],'epsilon':selected['epsilon'],
            'budgets':selected['budgets'],'policy_seeds':selected['policy_seeds'],'groups':groups,
            'report_sha256':seal['report_sha256'],'test_manifest_sha256':report['input_manifest_sha256'],
            'statistics':blocks.metadata(),'ci_unit':references['resolved_config']['ci']['unit'],
            'carbon_unit':'gCO2' if references['resolved_config']['ci']['unit']=='gCO2/kWh' else 'gCO2e',
            'missing_e3_methods':{str(beta):[m for m in E3_METHODS if not any(g['beta']==beta and g['method']==m for g in groups)] for beta in selected['budgets']},
            'endpoint_ablation_budgets':[beta for beta in selected['budgets'] if any(g['beta']==beta and g['method'].startswith('ScaleDown-') and g['method'].endswith('-endpoint') for g in groups)],
            'figure_scope':'all frozen budgets and declared seeds; means of complete paired cohorts; no new selection or experiment',
            'software':provenance(Path(__file__).resolve().parents[2])}


def tex_escape(value):
    replacements = {'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_',
                    '{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}'}
    return ''.join(replacements.get(c,c) for c in str(value))


def format_value(value):
    return '--' if value is None else f'{value:.3f}'


def write_tables(data,output):
    header = ['Budget','Method','Ratio at rho L','Ratio at rho H','Miss','Node-h','Status']
    lines = ['# E3 mechanism comparison','',f"Panel: {data['panel']}; purpose: {data['purpose']}.",'',
             '| '+' | '.join(header)+' |','|'+'---|'*len(header)]
    tex = [r'\begingroup',r'\small',r'\setlength{\tabcolsep}{4pt}',r'\begin{tabular}{@{}rlrrrrl@{}}',r'\toprule']
    if data['purpose']!='research':tex.append(r'\multicolumn{7}{l}{\textbf{'+tex_escape(data['purpose'].upper()+' - not paper evidence')+r'}} \\')
    tex += [r'$\beta$ & Method & $C_{-}/C_{b,-}$ & $C_{+}/C_{b,+}$ & Miss & Node-h & Status \\',r'\midrule']
    for beta in data['budgets']:
        for method in E3_METHODS:
            group = next((g for g in data['groups'] if g['beta']==beta and g['method']==method),None)
            summary = group['summary'] if group else None
            point = group['statistics']['point'] if group and group['statistics'] else {}
            row = [f'{beta:g}',method,format_value(point.get('endpoint_ratio_0')),format_value(point.get('endpoint_ratio_1')),
                   format_value(summary['miss_upper'] if summary else None),format_value(summary['mean_nodehours'] if summary else None),
                   group['status'] if group else 'not in plan']
            lines.append('| '+' | '.join(row)+' |');tex.append(' & '.join(map(tex_escape,row))+r' \\')
    tex += [r'\bottomrule',r'\end{tabular}',r'\endgroup']
    lines += ['',f"Power endpoints: {data['rho_interval']}. Each budget uses its validation-selected comparator.",
              'Policy rows pool all declared seeds equally; each seed and marginal paired intervals remain in plot-data.json.',
              'Miss is a conservative observed upper probability under censoring; it is not the confidence upper bound.',
              'Missing entries are not zero. Missing ablations remain listed; this table does not establish a mechanism claim by itself.']
    (output/'E3-mechanisms.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (output/'E3-mechanisms.tex').write_text('\n'.join(tex)+'\n',encoding='utf-8')
    # Preserve per-seed summaries/intervals/phase data and all budget rows in JSON,
    # including methods that are supplementary to the main E2 figure.
    write_manifest(output/'plot-data.json',data)


def export_results(run_directory,report_directory,output,figures=True):
    output = Path(output)
    require(not output.exists(),f'Output already exists: {output}')
    data = load_export_data(run_directory,report_directory)
    if figures:
        from .plotting import render_figures
        render = render_figures
    output.mkdir(parents=True)
    meta = {'kind':'paper_result_export','status':'running','purpose':data['purpose'],'panel':data['panel'],
            'report_sha256':data['report_sha256'],'test_manifest_sha256':data['test_manifest_sha256'],
            'copies_to_manuscript':False,'files_sha256':{}}
    write_manifest(output/'manifest.json',meta)
    try:
        write_tables(data,output)
        if figures:render(data,output)
        (output/'README.md').write_text(
            '# Result export\n\n'+f"Panel: {data['panel']}; purpose: {data['purpose']}.\n\n"+
            'E2-main and E4-power contain one page per frozen budget. The PNGs mirror those pages.\n'+
            'E3-mechanisms.tex is a booktabs fragment; its Markdown companion and plot-data.json retain missing/unsupported entries.\n'+
            'Plot bars show marginal paired calendar variability, crossed with seeds for RL. Confirmatory flags remain those of the sealed report.\n'+
            'Filled markers have miss-bound support; open markers lack that support. Observed target attainment is reported separately and is not a certificate. Missing intervals are not zero uncertainty.\n'+
            'Synthetic/development exports are marked and must not fill paper result slots. No manuscript file is modified.\n',encoding='utf-8')
        meta['files_sha256'] = {str(p.relative_to(output)):digest(p) for p in sorted(output.iterdir()) if p.is_file() and p.name!='manifest.json'}
        meta.update(status='complete',figures_exported=figures,
                    missing_e3_methods=data['missing_e3_methods'],endpoint_ablation_budgets=data['endpoint_ablation_budgets'])
    except Exception as exc:
        meta.update(status='failed',failure={'error_type':type(exc).__name__,'message':str(exc)})
        write_manifest(output/'failure.json',meta['failure']);raise
    finally:
        write_manifest(output/'manifest.json',meta)
    return meta


def render_export(export_directory,output):
    """Render a sealed table/data export without rerunning any statistics/replay."""
    source,output = Path(export_directory),Path(output)
    require(not output.exists(),f'Output already exists: {output}')
    seal = load_json(source/'manifest.json')
    require(seal['kind']=='paper_result_export' and seal['status']=='complete','Incomplete result data export')
    for name in ('plot-data.json','E3-mechanisms.tex','E3-mechanisms.md'):
        require(digest(source/name)==seal['files_sha256'][name],f'Export data hash mismatch: {name}')
    data = load_json(source/'plot-data.json')
    require(data['kind']=='paper_export_data_v1' and data['purpose']==seal['purpose'] and data['panel']==seal['panel'],
            'Export data identity differs from manifest')
    from .plotting import render_figures
    output.mkdir(parents=True)
    meta = {**seal,'status':'running','kind':'paper_figure_render','source_export_manifest_sha256':digest(source/'manifest.json'),
            'render_software':provenance(Path(__file__).resolve().parents[2]),'files_sha256':{}}
    write_manifest(output/'manifest.json',meta)
    try:
        for name in ('plot-data.json','E3-mechanisms.tex','E3-mechanisms.md'):
            shutil.copy2(source/name,output/name)
        render_figures(data,output)
        meta.update(status='complete',figures_exported=True,
                    files_sha256={p.name:digest(p) for p in sorted(output.iterdir()) if p.is_file() and p.name!='manifest.json'})
    except Exception as exc:
        meta.update(status='failed',failure={'error_type':type(exc).__name__,'message':str(exc)})
        write_manifest(output/'failure.json',meta['failure']);raise
    finally:
        write_manifest(output/'manifest.json',meta)
    return meta
