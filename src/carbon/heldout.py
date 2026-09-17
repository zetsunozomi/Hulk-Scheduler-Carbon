"""Execute only validation-frozen operating points on the predeclared holdout."""

from collections import defaultdict
from pathlib import Path
import shutil

from .baselines import check_references
from .common import digest, load_json, require
from .runner import provenance, run_fixed, run_planners, write_manifest


def load_selection(directory):
    directory = Path(directory)
    seal = load_json(directory/'manifest.json')
    require(seal['status']=='complete' and seal['kind']=='validation_selection','Incomplete validation selection')
    require(digest(directory/'selection.json')==seal['selection_sha256'],'Selection hash mismatch')
    selected = load_json(directory/'selection.json')
    require(selected['kind']=='validation_selection_v1' and selected['scope']=='validation_only','Invalid selection scope')
    require(selected.get('selection_rule','confidence_upper_miss') in {'empirical_miss','confidence_upper_miss'},
            'Unknown frozen selection rule')
    return selected


def frozen_jobs(selected):
    jobs = defaultdict(set)
    for beta,methods in selected['operating_points'].items():
        for points in methods.values():
            for point in points.values():
                if point['selected_id'] is not None:
                    jobs[point['selected_id']].add(float(beta))
    for beta,comparators in selected['comparators'].items():
        for point in comparators.values():
            require(point['selected_id'] is None or float(beta) in jobs[point['selected_id']],
                    'Comparator is not a frozen operating point')
    return {identity:sorted(budgets) for identity,budgets in sorted(jobs.items())}


def locate_checkpoints(paths,needed):
    found = {}
    for path in map(Path,paths):
        require(path.exists(),f'Missing checkpoint search path: {path}')
        files = sorted(path.rglob('checkpoint-*.json')) if path.is_dir() else [path]
        for file in files:
            value = digest(file)
            if value in needed and value not in found:
                metadata = load_json(file)
                require(metadata['kind']=='ppo_checkpoint_v1','Invalid selected checkpoint')
                require(Path(metadata['weights_file']).name==metadata['weights_file'],'Invalid weights filename')
                require(digest(file.parent/metadata['weights_file'])==metadata['weights_sha256'],'Selected weights hash mismatch')
                found[value] = file.resolve()
    require(set(found)==set(needed),'Missing frozen checkpoint(s); supply their training directories, not replacement checkpoints')
    return found


def run_test(bundle,selection_directory,output,predictor_path,checkpoint_paths):
    selected = load_selection(selection_directory)
    output = Path(output)
    require(not output.exists(),f'Output already exists: {output}')
    check_references(selected['references'],bundle.manifest)
    require(selected['purpose']==bundle.raw['purpose'],'Test purpose differs from selection')
    software = provenance(bundle.root)
    require(all(software['source_sha256'][name]==value for name,value in selected['replay_source_sha256'].items()),
            'Replay/accounting code changed after validation')
    jobs = frozen_jobs(selected)
    needed = {selected['candidates'][identity]['checkpoint_sha256'] for identity in jobs
              if selected['candidates'][identity]['kind']=='policy'}
    checkpoints = locate_checkpoints(checkpoint_paths,needed)
    predictor_path = Path(predictor_path).resolve()
    predictor_hash = digest(predictor_path)
    for identity in jobs:
        entry = selected['candidates'][identity]
        expected = entry.get('predictor_sha256',entry.get('planner_settings',{}).get('predictor_sha256'))
        require(expected is None or expected==predictor_hash,'Wait predictor changed after validation')
        if entry['kind']=='policy':
            require(entry['source_sha256']==software['source_sha256'],'Policy implementation changed after validation')
            metadata = load_json(checkpoints[entry['checkpoint_sha256']])
            require(metadata['settings']['seed']==entry['seed'],'Selected checkpoint seed differs')
            # Fail before any holdout job if the installed tensor runtime differs.
            import torch
            require(metadata['software']['torch']==str(torch.__version__),'Checkpoint PyTorch version differs')
    cohort = {e.episode_id:e.arrival.isoformat().replace('+00:00','Z') for e in bundle.episodes if e.split=='test'}
    require(cohort,'Empty frozen test cohort')
    plan = {'kind':'heldout_plan_v1','selection_sha256':digest(Path(selection_directory)/'selection.json'),
            'config_sha256':bundle.manifest['config_sha256'],'asset_sha256':bundle.manifest['asset_sha256'],
            'purpose':selected['purpose'],'panel':selected['panel'],'test_cohort':cohort,'jobs':jobs,
            'power_analysis_nominal_rho':sum(bundle.raw['power']['rho_interval'])/2,
            'constant_ci_rescore_value':1.,'software':software,
            'selection_rule':'all frozen points including unsupported descriptive fallbacks; no test reselection'}
    output.mkdir(parents=True)
    (output/'selection').mkdir()
    for name in ('manifest.json','selection.json'):
        shutil.copy2(Path(selection_directory)/name,output/'selection'/name)
    write_manifest(output/'plan.json',plan)
    write_manifest(output/'references.json',selected['references'])
    manifest = {'kind':'heldout_execution_v1','status':'running','purpose':selected['purpose'],
                'plan_sha256':digest(output/'plan.json'),'runs':{},'run_files_sha256':{}}
    write_manifest(output/'manifest.json',manifest)
    current = 'fixed'
    try:
        run_fixed(bundle,output/'fixed',bundle.raw['cluster']['allowed_nodes'],split='test')
        for index,(identity,betas) in enumerate(jobs.items()):
            current = identity
            entry = selected['candidates'][identity]
            if entry['kind'] in {'fixed','mixture'}:
                manifest['runs'][identity] = {str(beta):'fixed' for beta in betas}
            elif entry['kind']=='planner':
                settings = entry['planner_settings']
                manifest['runs'][identity] = {}
                for bindex,beta in enumerate(betas):
                    name = f'planner-{index:04d}-{bindex:03d}'
                    run_planners(bundle,output/name,predictor_path,output/'references.json','test',
                                 settings['planning_paths'],settings['planner_internal_miss_tolerance'],
                                 settings['seed'],[entry['method']],beta)
                    manifest['runs'][identity][str(beta)] = name
            else:
                from .policy_runner import evaluate_policy
                name = f'policy-{index:04d}'
                evaluate_policy(bundle,output/name,predictor_path,checkpoints[entry['checkpoint_sha256']],
                                'test',betas,entry['sampling_seed'])
                manifest['runs'][identity] = {str(beta):name for beta in betas}
            write_manifest(output/'manifest.json',manifest)
        for name in {'fixed'}|{p for paths in manifest['runs'].values() for p in paths.values()}:
            manifest['run_files_sha256'][name] = {f:digest(output/name/f) for f in ('manifest.json','episodes.jsonl','chunks.jsonl')}
        manifest['status'] = 'complete'
    except Exception as exc:
        manifest.update(status='failed',failure={'candidate':current,'error_type':type(exc).__name__,'message':str(exc)})
        write_manifest(output/'failure.json',manifest['failure'])
        raise
    finally:
        write_manifest(output/'manifest.json',manifest)
    return manifest
