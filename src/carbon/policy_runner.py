"""Train and evaluate a shared budget-conditioned policy on complete replay episodes."""

from collections import OrderedDict
from copy import deepcopy
from dataclasses import replace
import hashlib
import math
from pathlib import Path
import random
from statistics import fmean
import time

from .common import digest, integer, load_json, number, require
from .environment import Environment, initial_replay
from .learning import ActorCritic, initial_dual, monte_carlo_costs, optimize_ppo, sample_action, torch, update_dual
from .policy_inputs import PolicyInputs, shared_inputs
from .runner import provenance, write_manifest, write_record
from .slider import slider_contract, budget_at_position


DEFAULTS = {'episodes_per_budget':32,'epochs':4,'minibatch_episodes':32,'learning_rate':3e-4,
            'clip':.2,'entropy':.01,'gradient_norm':.5,'value_coefficient':.5,
            'epsilon':.05,'dual_rate_p':.05,'dual_rate_lambda':.05,'seed':11,'threads':1,
            'budgets':[1.,1.25,1.5,2.],'forecast_mode':'window','wait_features':'none','decision_mode':'feedback','objective':'robust'}


def settings_for(iterations, **options):
    require(set(options) <= set(DEFAULTS), 'Unknown PPO setting')
    settings = {**deepcopy(DEFAULTS),**options,'iterations':integer(iterations,'iterations')}
    for key in ('episodes_per_budget','epochs','minibatch_episodes','threads'):
        settings[key] = integer(settings[key],key)
    settings['seed'] = integer(settings['seed'],'seed',0)
    for key in ('learning_rate','gradient_norm','dual_rate_p','dual_rate_lambda'):
        settings[key] = float(number(settings[key],key,strict=True))
    for key in ('entropy','value_coefficient','epsilon','clip'):
        settings[key] = float(number(settings[key],key))
    require(0 < settings['clip'] < 1 and settings['epsilon'] <= 1, 'Invalid PPO clip or epsilon')
    settings['budgets'] = [float(number(b,'budget multiplier',strict=True)) for b in settings['budgets']]
    require(settings['budgets'] and len(settings['budgets'])==len(set(settings['budgets'])), 'Budget multipliers must be unique')
    require(settings['objective'] in {'robust','lower','upper'}, 'Unknown power objective')
    require(settings['forecast_mode'] in {'window','current'}, 'Unknown forecast mode')
    require(settings['wait_features'] in {'none','advice'}, 'Unknown wait-feature mode')
    require(settings['decision_mode'] in {'feedback','precommitted'}, 'Unknown decision mode')
    require(settings['decision_mode'] != 'precommitted' or settings['wait_features'] == 'none',
            'Precommitted-RL does not use wait advice')
    return settings


def configure_torch(seed, threads):
    require(tuple(int(v) for v in str(torch.__version__).split('.')[:2]) >= (2,6), 'PPO requires PyTorch 2.6+')
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)


def method_name(settings):
    name = 'ScaleDown' if settings.get('decision_mode','feedback')=='feedback' else 'Precommitted-RL'
    if settings['forecast_mode'] != 'window':
        name = 'Current-CI' if name == 'ScaleDown' else name + '-Current-CI'
    if settings.get('wait_features', 'advice') == 'advice':
        name = 'Predictor-advised' if name == 'ScaleDown' else name + '-advised'
    return name if settings['objective']=='robust' else f"{name}-{settings['objective']}-endpoint"


def nested_tuple(value):
    return tuple(nested_tuple(v) for v in value) if isinstance(value,list) else value


def save_checkpoint(output, iteration, model, optimizer, generator, rng, metadata):
    name = f'checkpoint-{iteration:06d}'
    weights = output/(name+'.pt')
    temporary = output/(name+'.pt.tmp')
    torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'sampling_rng':generator.get_state()},temporary)
    temporary.replace(weights)
    manifest = {**metadata,'kind':'ppo_checkpoint_v1','iteration':iteration,
                'weights_file':weights.name,'weights_sha256':digest(weights),'arrival_shuffle_rng':rng.getstate()}
    write_manifest(output/(name+'.json'),manifest)
    return output/(name+'.json')


def read_checkpoint(path):
    path = Path(path); metadata = load_json(path)
    require(metadata['kind']=='ppo_checkpoint_v1', 'Unsupported PPO checkpoint')
    require(Path(metadata['weights_file']).name==metadata['weights_file'], 'Invalid checkpoint weights path')
    weights = path.parent/metadata['weights_file']
    require(digest(weights)==metadata['weights_sha256'], 'Checkpoint weights hash mismatch')
    # Only tensor/state dictionaries are loaded; arbitrary Python objects are not needed.
    state = torch.load(weights,map_location='cpu',weights_only=True)
    return metadata,state


def check_policy_inputs(metadata, encoder, predictor, bundle):
    require(metadata['feature_schema']==encoder.schema(), 'Checkpoint input schema/normalizers differ')
    require(metadata['predictor_sha256']==encoder.predictor_version, 'Checkpoint wait model differs')
    require(metadata['purpose']==bundle.raw['purpose'], 'Checkpoint experiment purpose differs')


class ReplayCache:
    def __init__(self,bundle,size=8):
        self.bundle,self.size,self.cache = bundle,size,OrderedDict()

    def get(self,episode):
        if episode.episode_id not in self.cache:
            self.cache[episode.episode_id] = initial_replay(self.bundle,episode)
            if len(self.cache)>self.size:
                self.cache.popitem(last=False)
        self.cache.move_to_end(episode.episode_id)
        return self.cache[episode.episode_id]


def collect_episode(bundle, episode, model, encoder, generator, base, method, seed, chunk_stream, identity, decision_mode='feedback', plan_stream=None):
    env = Environment(bundle,episode,base,method,seed)
    steps, costs = [], []
    endpoints = [str(r) for r in bundle.raw['power']['rho_interval']]
    normalizers = [encoder.references['carbon_reference_g_per_kappa'][r] for r in endpoints]
    require(all(math.isfinite(v) and v>0 for v in normalizers), 'Invalid cost normalizers')
    committed = None
    if decision_mode == 'precommitted':
        from .precommit import build_precommitted_plan
        require(plan_stream is not None, 'Precommitted plans must be archived before execution')
        committed, plan_record = build_precommitted_plan(bundle,env.observe(),encoder,model,generator)
        write_record(plan_stream,{**plan_record,**identity,'episode_id':episode.episode_id,'seed':seed})
    while env.status=='running':
        if committed is None:
            begun = time.monotonic()
            encoded, metadata = encoder.encode(env.observe())
            action, log_prob, values, probabilities = sample_action(model,encoded,generator)
            inference_seconds = time.monotonic()-begun
        else:
            item = committed[len(steps)]
            encoded,metadata = item['input'],item['metadata']
            action,log_prob,values,probabilities = item['action'],item['log_probability'],item['values'],item['probabilities']
            inference_seconds = item['inference_seconds']
        metadata['decision_mode'] = decision_mode
        chunk = env.step(encoder.nodes[action])
        write_record(chunk_stream,{**chunk,**metadata,**identity,'policy_probabilities':probabilities,
                                  'selected_log_probability':log_prob,'predicted_cost_to_go':values,
                                  'decision_inference_seconds':inference_seconds,'policy_rule':'categorical_sampling'})
        steps.append({'input':encoded,'action':action,'log_probability':log_prob,'values':values})
        costs.append([chunk['carbon_g_per_kappa'][r]/normalizers[i] for i,r in enumerate(endpoints)])
    summary = {**env.summary(),**identity,'policy_rule':'categorical_sampling','decision_mode':decision_mode}
    if committed is not None:
        summary.update(precommitted_plan_sha256=plan_record['plan_sha256'],
                       precommitted_nodes=[p['nodes'] for p in committed])
    # The caller saves even a censored result before refusing to train on it.
    returns = monte_carlo_costs(costs,summary['deadline_miss']) if summary['final_status']=='completed' else None
    return steps,returns,summary


def train_policy(bundle, output, predictor_path, references_path, settings, resume=None):
    settings = settings_for(**settings)
    references,predictor = shared_inputs(bundle,predictor_path,references_path,settings['wait_features']=='advice')
    encoder = PolicyInputs(bundle,predictor,references,settings['forecast_mode'],settings.get('wait_features','advice'))
    train_episodes = [e for e in bundle.episodes if e.split=='train']
    require(train_episodes,'No training cohort')
    configure_torch(settings['seed'],settings['threads'])
    model = ActorCritic(len(encoder.global_names),len(encoder.action_names))
    optimizer = torch.optim.Adam(model.parameters(),lr=settings['learning_rate'])
    generator = torch.Generator().manual_seed(settings['seed'])
    rng = random.Random(settings['seed'])
    duals = {str(b):initial_dual(settings['objective']) for b in settings['budgets']}
    source = {**provenance(bundle.root),'torch':str(torch.__version__),'device':'cpu','deterministic_algorithms':True}
    first_iteration, total_episodes, total_chunks = 1,0,0
    if resume:
        previous,state = read_checkpoint(resume)
        check_policy_inputs(previous,encoder,predictor,bundle)
        require(previous['settings']==settings, 'Resume settings changed; use the same predeclared total iterations and options')
        require(previous['software']['source_sha256']==source['source_sha256'], 'Resume implementation changed')
        require(previous['software']['torch']==source['torch'], 'Resume PyTorch version differs')
        require(previous['training_cohort_sha256']==bundle.manifest['asset_sha256']['cohort'], 'Resume cohort changed')
        require(previous['iteration']<settings['iterations'], 'Checkpoint already reached the requested iteration count')
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        generator.set_state(state['sampling_rng']); rng.setstate(nested_tuple(previous['arrival_shuffle_rng']))
        duals = previous['duals']; first_iteration = previous['iteration']+1
        total_episodes,total_chunks = previous['total_episodes'],previous['total_chunks']
    output = Path(output)
    require(not output.exists(), f'Output already exists: {output}')
    output.mkdir(parents=True)
    metadata = {'purpose':bundle.raw['purpose'],'settings':settings,'feature_schema':encoder.schema(),
                'predictor_sha256':encoder.predictor_version,'references':references,'software':source,
                'slider':slider_contract(references['time_reference_hours'],settings['budgets']),
                'training_cohort_sha256':bundle.manifest['asset_sha256']['cohort'],
                'architecture':{'width':128,'critic_heads':['lower_endpoint','upper_endpoint','violation'],
                                'shared_actor_critic_parameters':False,'pooling':'mean_of_feasible_actions'}}
    manifest = {**bundle.manifest,**metadata,'kind':'ppo_training','status':'running',
                'resume_checkpoint_sha256':digest(resume) if resume else None,
                'declared_episode_interactions':settings['iterations']*len(settings['budgets'])*settings['episodes_per_budget'],
                'first_iteration':first_iteration,'completed_iteration':first_iteration-1,
                'total_episodes':total_episodes,'total_chunks':total_chunks,
                'selection_status':'training checkpoints only; no validation selection or feasibility claim'}
    write_manifest(output/'manifest.json',manifest)
    begun,current = time.monotonic(),None
    cache = ReplayCache(bundle)
    try:
        with (output/'chunks.jsonl').open('x',encoding='utf-8') as chunks, \
             (output/'episodes.jsonl').open('x',encoding='utf-8') as records, \
             (output/'training.jsonl').open('x',encoding='utf-8') as training, \
             (output/'plans.jsonl').open('x',encoding='utf-8') as plans:
            for iteration in range(first_iteration,settings['iterations']+1):
                rollout_start = time.monotonic()
                before = deepcopy(duals)
                arrivals = [rng.choice(train_episodes) for _ in range(settings['episodes_per_budget'])]
                rollout = []
                model.eval()
                for beta in settings['budgets']:
                    for index, original in enumerate(arrivals):
                        episode = replace(original,budget_hours=beta*references['time_reference_hours'])
                        current = {'iteration':iteration,'budget_multiplier':beta,'rollout_sample':index,
                                   'episode_id':episode.episode_id}
                        identity = {**current,'epsilon':settings['epsilon'],'policy_checkpoint':f'training-pre-update-{iteration}',
                                    'power_objective':settings['objective'],'policy_forecast_mode':settings['forecast_mode'],'policy_wait_features':encoder.wait_features}
                        steps,returns,summary = collect_episode(bundle,episode,model,encoder,generator,cache.get(episode),
                                                                method_name(settings)+'-train',settings['seed'],chunks,identity,settings['decision_mode'],plans)
                        write_record(records,summary)
                        total_episodes += 1; total_chunks += len(steps)
                        manifest.update(total_episodes=total_episodes,total_chunks=total_chunks)
                        require(returns is not None, f"Incomplete training episode {episode.episode_id}; fix predeclared coverage, do not drop or bootstrap it")
                        rollout.append({'beta':beta,'steps':steps,'returns':returns})
                rollout_seconds = time.monotonic()-rollout_start
                update_start = time.monotonic(); model.train()
                optimization = optimize_ppo(model,optimizer,rollout,before,settings,rng)
                update_seconds = time.monotonic()-update_start
                means = {}
                for beta in settings['budgets']:
                    totals = [episode['returns'][0] for episode in rollout if episode['beta']==beta]
                    means[str(beta)] = [fmean(row[i] for row in totals) for i in range(3)]
                    duals[str(beta)] = update_dual(before[str(beta)],means[str(beta)],settings['epsilon'],
                                                   settings['dual_rate_p'],settings['dual_rate_lambda'],settings['objective'])
                checkpoint = save_checkpoint(output,iteration,model,optimizer,generator,rng,
                                             {**metadata,'duals':duals,'total_episodes':total_episodes,'total_chunks':total_chunks})
                write_record(training,{'iteration':iteration,'pre_update_episode_means':means,'duals_before':before,
                                       'duals_after':duals,'optimizer_minibatches':optimization,'rollout_seconds':rollout_seconds,
                                       'update_seconds':update_seconds,'checkpoint':checkpoint.name,
                                       'checkpoint_sha256':digest(checkpoint),'episodes':len(rollout),
                                       'chunks':sum(len(e['steps']) for e in rollout)})
                manifest.update(completed_iteration=iteration,last_checkpoint=checkpoint.name,total_episodes=total_episodes,total_chunks=total_chunks)
                write_manifest(output/'manifest.json',manifest)
                print(f'PPO iteration {iteration}/{settings["iterations"]}: episodes={total_episodes}, chunks={total_chunks}, checkpoint={checkpoint.name}',flush=True)
        manifest['status'] = 'complete'
    except Exception as exc:
        manifest.update(status='failed',failure={**(current or {}),'error_type':type(exc).__name__,'message':str(exc)})
        write_manifest(output/'failure.json',manifest['failure'])
        raise
    finally:
        if (output/'plans.jsonl').exists():
            manifest['plans_sha256'] = digest(output/'plans.jsonl')
        manifest['elapsed_seconds'] = time.monotonic()-begun
        write_manifest(output/'manifest.json',manifest)
    return manifest


def evaluate_policy(bundle, output, predictor_path, checkpoint_path, split='validation', budgets=None, sampling_seed=None, slider_position=None):
    require(split in {'train','validation','test'}, 'Unknown policy evaluation split')
    metadata,state = read_checkpoint(checkpoint_path)
    settings = metadata['settings']
    references,predictor = shared_inputs(bundle,predictor_path,metadata['references'],settings.get('wait_features','advice')=='advice')
    encoder = PolicyInputs(bundle,predictor,references,settings['forecast_mode'],settings.get('wait_features','advice'))
    check_policy_inputs(metadata,encoder,predictor,bundle)
    software = {**provenance(bundle.root),'torch':str(torch.__version__)}
    require(metadata['software']['source_sha256']==software['source_sha256'], 'Checkpoint implementation changed')
    require(metadata['software']['torch']==software['torch'], 'Checkpoint PyTorch version differs')
    if slider_position is not None:
        require(budgets is None, 'Choose budget multipliers or a slider position, not both')
        budgets = [budget_at_position(slider_position,settings['budgets'])]
    selected = settings['budgets'] if budgets is None else [float(b) for b in budgets]
    require(selected and len(selected)==len(set(selected)) and set(selected)<=set(settings['budgets']),
            'Evaluation budgets must be unique members of the trained grid')
    seed = settings['seed'] if sampling_seed is None else integer(sampling_seed,'sampling seed',0)
    configure_torch(seed,settings['threads'])
    model = ActorCritic(len(encoder.global_names),len(encoder.action_names))
    model.load_state_dict(state['model']); model.eval()
    episodes = [e for e in bundle.episodes if e.split==split]
    require(episodes,'Selected policy cohort is empty')
    output = Path(output)
    require(not output.exists(), f'Output already exists: {output}')
    output.mkdir(parents=True)
    checkpoint_hash = digest(checkpoint_path)
    manifest = {**bundle.manifest,'kind':'policy_evaluation','status':'running','split':split,
                'method':method_name(settings),'training_seed':settings['seed'],'sampling_seed':seed,
                'budget_multipliers':selected,'checkpoint_sha256':checkpoint_hash,
                'slider':slider_contract(references['time_reference_hours'],settings['budgets']),
                'decision_mode':settings.get('decision_mode','feedback'),
                'predictor_sha256':encoder.predictor_version,'software':software,
                'policy_rule':'categorical_sampling','completed_episode_methods':0,
                'selection_status':'raw checkpoint evaluation; validation selection/statistical feasibility not asserted'}
    write_manifest(output/'manifest.json',manifest)
    begun,current = time.monotonic(),None
    try:
        with (output/'chunks.jsonl').open('x',encoding='utf-8') as chunks, (output/'episodes.jsonl').open('x',encoding='utf-8') as records, \
             (output/'plans.jsonl').open('x',encoding='utf-8') as plans:
            for original in episodes:
                base = initial_replay(bundle,original)
                for beta in selected:
                    episode = replace(original,budget_hours=beta*references['time_reference_hours'])
                    current = {'episode_id':episode.episode_id,'budget_multiplier':beta}
                    paired = int.from_bytes(hashlib.sha256(f'{seed}:{episode.episode_id}:{beta}'.encode()).digest()[:8],'big')
                    generator = torch.Generator().manual_seed(paired)
                    identity = {**current,'epsilon':settings['epsilon'],'policy_checkpoint':checkpoint_hash,
                                'power_objective':settings['objective'],'policy_forecast_mode':settings['forecast_mode'],'policy_wait_features':encoder.wait_features}
                    _,_,summary = collect_episode(bundle,episode,model,encoder,generator,base,method_name(settings),
                                                   settings['seed'],chunks,identity,settings.get('decision_mode','feedback'),plans)
                    write_record(records,summary)
                    manifest['completed_episode_methods'] += 1
                    print(f'{episode.episode_id} {method_name(settings)} beta={beta}: {summary["final_status"]}',flush=True)
        manifest['status'] = 'complete'
    except Exception as exc:
        manifest.update(status='failed',failure={**(current or {}),'error_type':type(exc).__name__,'message':str(exc)})
        write_manifest(output/'failure.json',manifest['failure']); raise
    finally:
        if (output/'plans.jsonl').exists():
            manifest['plans_sha256'] = digest(output/'plans.jsonl')
        manifest['elapsed_seconds'] = time.monotonic()-begun
        write_manifest(output/'manifest.json',manifest)
    return manifest
