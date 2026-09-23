"""One development control: independent actor branches, shared critic.

Same four budgets, fresh seed11, initial functions, PPO settings and arrivals as
the completed shared-product trial. Preserve every scheduled validation result.
"""

import argparse
import math
from pathlib import Path

from carbon.common import digest, load_json, require
from carbon.config import Bundle
from carbon.main_pilot import ppo_stage, read_jsonl_prefix
from carbon.policy_runner import read_checkpoint, settings_for, torch
from carbon.probe_resume import probe_lock
from carbon.results import ResultRun
from carbon.runner import provenance, write_manifest
from actor_interaction_trial import policy_only_changes, reuse_inputs
from diagnose_main import binding, comparison, records, response_summary
from main_train import checkpoint_at, validation_stage


ITERATIONS = 64
VALIDATION_ITERATIONS = (16, 32, 48, 64)
EPISODES_PER_BUDGET = 16


def frozen_product(bundle, product, learning, grid, baseline, old_settings, pilot_binding, software):
    plan = load_json(product/'run-plan.json'); report = load_json(product/'interaction-summary.json')
    require(plan['config_sha256'] == bundle.manifest['config_sha256'] and
            plan['asset_sha256'] == bundle.manifest['asset_sha256'] and
            plan['reuse']['pilot_artifacts'] == pilot_binding and plan['budget_grid'] == grid and
            plan['torch'] == str(torch.__version__), 'Product trial inputs/runtime differ')
    changes = policy_only_changes(plan['source_sha256'], software['source_sha256'])
    for name, sha in plan['script_sha256'].items():
        require(digest(Path(__file__).with_name(name)) == sha, 'Product trial script changed')
    previous = plan['settings']
    clean = lambda value: {k: v for k, v in value.items() if k not in {'actor_interaction', 'actor_budget_mode'}}
    require(clean(previous) == clean(old_settings) and previous['actor_interaction'] == 'product' and
            previous.get('actor_budget_mode', 'shared') == 'shared', 'Expected the matched shared-product trial')
    require(report['status'] == 'complete' and report['run_plan_sha256'] == digest(product/'run-plan.json') and
            report['settings'] == previous and report['budget_grid'] == grid and
            report['baselines'] == baseline['comparisons'] and report['fixed_mix'] == baseline['fixed_mix'] and
            [c['iteration'] for c in report['checkpoints']] == list(VALIDATION_ITERATIONS), 'Product summary differs')
    final_meta = load_json(product/report['final_checkpoint'])
    for c in report['checkpoints']:
        stage = product/f"validation-{c['iteration']:06d}"
        require(binding(stage, c['validation_files']) == c['validation_files'] and
                load_json(stage/'stage-seal.json') == c['validation_files'], 'Product validation seal differs')
        ResultRun(stage, final_meta['references'], allowed_splits={'validation'})
        candidates = [p for p in product.glob(f"ppo-attempt-*/checkpoint-{c['iteration']:06d}.json") if digest(p) == c['checkpoint_sha256']]
        require(len(candidates) == 1, 'Missing/ambiguous product checkpoint')
        meta = load_json(candidates[0]); weight = meta['weights_file']
        require(Path(weight).name == weight and digest(candidates[0].parent/weight) == meta['weights_sha256'] and
                meta['settings'] == previous and meta['software']['source_sha256'] == plan['source_sha256'],
                'Product checkpoint binding differs')
    diagnosis = load_json(learning/'learning-summary.json'); diag_plan = load_json(learning/'run-plan.json')
    require(diagnosis['status'] == 'complete' and diagnosis['run_plan_sha256'] == digest(learning/'run-plan.json') and
            diag_plan['source_sha256'] == plan['source_sha256'] and diag_plan['settings'] == previous and
            diag_plan['iterations'] == [1, 4, 8, 16] and diag_plan['optimizer_steps'] == 0 and
            [r['iteration'] for r in diagnosis['rounds']] == [1, 4, 8, 16], 'Completed frozen learning diagnosis required')
    require(binding(product, diag_plan['trial_files']) == diag_plan['trial_files'], 'Diagnosed training artifacts changed')
    for result in diagnosis['rounds']:
        stage = learning/f"round-{result['iteration']:06d}"
        require(binding(stage, result['stage_files']) == result['stage_files'] and
                load_json(stage/'stage-seal.json') == result['stage_files'], 'Learning diagnosis seal differs')
    attempts = sorted(product.glob('ppo-attempt-[0-9][0-9][0-9]'))
    require(len(attempts) == 1, 'Expected the completed fresh product attempt')
    rows = records(attempts[0]/'episodes.jsonl')
    require(len(rows) == report['ppo_episodes'] and all(r['split'] == 'train' and not r['censor_flag'] for r in rows),
            'Product training cohort differs')
    width = final_meta['architecture']['width']; schema = final_meta['feature_schema']
    actor_parameters = width*(len(schema['global_names'])+len(schema['action_names']))+6*width*width+6*width+1
    evidence = {'product_plan_sha256': digest(product/'run-plan.json'),
                'product_summary_sha256': digest(product/'interaction-summary.json'),
                'product_training_episodes_sha256': digest(attempts[0]/'episodes.jsonl'),
                'learning_plan_sha256': digest(learning/'run-plan.json'),
                'learning_summary_sha256': digest(learning/'learning-summary.json'),
                'product_policy_source_changes': changes,
                'actor_parameters_per_branch': actor_parameters,
                'additional_actor_parameters': (len(previous['budgets'])-1)*actor_parameters}
    points = report['baselines']+[p for c in report['checkpoints'] for p in c['operating_points']]
    return previous, rows, points, evidence


def committed_arrivals(output, history):
    by_attempt = {}; keys = []
    for r in history:
        path = checkpoint_at(output, r['iteration'], r)
        if path.parent not in by_attempt:
            by_attempt[path.parent] = read_jsonl_prefix(path.parent/'episodes.jsonl')
        rows = [e for e in by_attempt[path.parent] if e['iteration'] == r['iteration']]
        require(len(rows) == r['episodes'] and all(e['split'] == 'train' and not e['censor_flag'] for e in rows),
                'Committed training arrivals incomplete')
        keys.extend((e['iteration'], e['budget_multiplier'], e['rollout_sample'], e['episode_id']) for e in rows)
    require(len(keys) == len(set(keys)), 'Duplicate committed training episode')
    return sorted(keys)


def write_readout(output, report):
    write_manifest(output/'budget-actor-summary.json', report)
    lines = ['# Independent budget actor development control', '',
             '四档预算各用独立 product actor，critic 保持共享；初始 actor/critic 函数与 shared-product 相同。',
             '总训练 episode 数、采样流程、PPO/dual/entropy 设置和环境不变；新增参数量已记入 run-plan。',
             '全部预定 validation checkpoint 都报告。这不是等参数因果消融，也未采用为正式方法。', '',
             '| 方法 | D(h) | 完成/总数 | mean TAT(h) | C | miss 下界–上界 | 换规模 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    fmt = lambda v: '—' if v is None else f'{v:.3f}'
    for p in sorted(report['baselines']+[p for c in report['checkpoints'] for p in c['operating_points']], key=lambda p:p['budget_hours']):
        s = p['summary']; lo, hi = s['miss_rate_bounds']
        lines.append(f"| {p['method']} | {p['budget_hours']:.2f} | {s['completed']}/{s['outcomes']} | {fmt(s['mean_tat_hours'])} | {fmt(s['worst_normalized_carbon'])} | {lo:.1%}–{hi:.1%} | {s['observed_scale_change_fraction']:.1%} |")
    lines += ['', 'C 为最坏归一化端点成本；跨区间优势必须另查两个端点。Fixed-Mix 是原 validation 拟合参考。', '']
    for mix in report['fixed_mix']:
        lines.append(f"- Fixed-Mix D={mix['budget_hours']:.2f}h：经验可行={mix['empirical_feasible']}，C={fmt(mix['estimated_objective'])}")
    for c in report['checkpoints']:
        p = c['actor']['budget_response'][0]
        lines.append(f"- 第 {c['iteration']} 轮：平均最大预算 TV={p['mean_max_pairwise_budget_TV']:.8f}")
    lines += ['', f"状态：{report['status']}；与原 product trial 配对训练到达数：{report.get('matched_training_arrivals', 0)}", '']
    tmp = output/'budget-actor-summary.md.tmp'; tmp.write_text('\n'.join(lines)); tmp.replace(output/'budget-actor-summary.md')


def run(bundle, pilot, main, diagnosis, product, learning, output, resume=False):
    require(bundle.raw['purpose'] in {'development', 'synthetic'}, 'This control is development only')
    pilot, main, diagnosis, product, learning, output = map(Path, (pilot, main, diagnosis, product, learning, output))
    software = provenance(bundle.root)
    refs, grid, baseline, old, reuse = reuse_inputs(bundle, pilot, main, diagnosis, software)
    previous, old_rows, points, product_binding = frozen_product(bundle, product, learning, grid, baseline, old,
                                                               reuse['pilot_artifacts'], software)
    settings = settings_for(**{**previous, 'actor_budget_mode': 'independent'})
    without_mode = lambda v: {k: x for k, x in v.items() if k != 'actor_budget_mode'}
    require(without_mode(settings) == without_mode(previous) and settings['iterations'] == ITERATIONS and
            settings['episodes_per_budget'] == EPISODES_PER_BUDGET and settings['seed'] == 11,
            'Only actor budget parameter sharing may change')
    scripts = [Path(__file__).with_name(name) for name in ('budget_actor_trial.py', 'actor_interaction_trial.py', 'main_train.py', 'diagnose_main.py')]
    plan = {'kind': 'independent_budget_actor_trial_v1', 'config_sha256': bundle.manifest['config_sha256'],
            'asset_sha256': bundle.manifest['asset_sha256'], 'source_sha256': software['source_sha256'],
            'script_sha256': {p.name: digest(p) for p in scripts}, 'torch': str(torch.__version__),
            'reuse': reuse, 'product_comparison': product_binding, 'settings': settings, 'budget_grid': grid,
            'validation_iterations': list(VALIDATION_ITERATIONS), 'split': 'validation',
            'initialization': 'fresh seed11; actor branches cloned after critic initialization; no warm start',
            'interpretation': 'Actor parameter isolation control; extra capacity and shared critic/optimizer remain caveats'}
    with probe_lock(output, resume, name='.budget-actor-trial.lock'):
        path = output/'run-plan.json'
        if path.exists(): require(load_json(path) == plan, 'Trial inputs/settings/software changed')
        else:
            require(not any(p.name != '.budget-actor-trial.lock' for p in output.iterdir()), 'Unknown existing output')
            write_manifest(path, plan)
        report = {'status': 'training', 'run_plan_sha256': digest(path), 'settings': settings, 'budget_grid': grid,
                  'baselines': points, 'fixed_mix': baseline['fixed_mix'], 'checkpoints': []}
        write_readout(output, report)
        print(f"Fresh independent-budget product actors: {settings['iterations']} rounds, {len(settings['budgets'])} budgets, {settings['episodes_per_budget']} episodes/budget; critic shared.", flush=True)
        checkpoint, meta, history = ppo_stage(bundle, output, pilot/'references.json', settings)
        expected_arrivals = sorted((e['iteration'], e['budget_multiplier'], e['rollout_sample'], e['episode_id']) for e in old_rows)
        require(committed_arrivals(output, history) == expected_arrivals, 'Training arrivals differ from the shared-product control')
        report.update(status='partial', ppo_episodes=meta['total_episodes'], ppo_chunks=meta['total_chunks'],
                      final_checkpoint=str(checkpoint.relative_to(output)), matched_training_arrivals=len(expected_arrivals),
                      iteration_timing=[{k: r[k] for k in ('iteration', 'rollout_seconds', 'update_seconds', 'episodes', 'chunks')} for r in history])
        by_iteration = {r['iteration']: r for r in history}
        for iteration in VALIDATION_ITERATIONS:
            candidate = checkpoint_at(output, iteration, by_iteration[iteration]); checkpoint_meta, _ = read_checkpoint(candidate)
            require(checkpoint_meta['settings'] == settings and checkpoint_meta['architecture']['actor_budget_mode'] == 'independent', 'Wrong actor checkpoint')
            stage = output/f'validation-{iteration:06d}'
            rows, groups, seal = validation_stage(bundle, stage, candidate, settings)
            ResultRun(stage, refs, allowed_splits={'validation'})
            operating_points, responses = [], []
            for tick in grid['slider']['ticks']:
                beta = tick['budget_multiplier']; selected = [r for r in rows if r['budget_multiplier'] == beta]
                chunks = [groups[(r['episode_id'], beta)] for r in selected]
                operating_points.append(comparison(f'BudgetActor-PPO-{iteration}', selected, chunks, tick['budget_hours'], refs))
                for r, cs in zip(selected, chunks):
                    probabilities = cs[0]['policy_probabilities']
                    responses.append({'iteration': iteration, 'episode_id': r['episode_id'], 'budget_multiplier': beta,
                                      'probabilities': probabilities, 'entropy_nats': -sum(p*math.log(p) for p in probabilities if p>0)})
            report['checkpoints'].append({'iteration': iteration, 'checkpoint_sha256': digest(candidate),
                                          'validation_files': seal, 'operating_points': operating_points,
                                          'actor': response_summary(responses)})
            report['status'] = 'complete' if iteration == VALIDATION_ITERATIONS[-1] else 'partial'
            write_readout(output, report)
            print(f'Independent budget actor validation iteration {iteration} complete.', flush=True)
        print(f'Budget actor trial complete: {output}/budget-actor-summary.md', flush=True)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'pilot', 'main', 'diagnosis', 'product', 'learning', 'output'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run(Bundle(Path(args.config)), args.pilot, args.main, args.diagnosis, args.product, args.learning, args.output, args.resume)


if __name__ == '__main__': main()
