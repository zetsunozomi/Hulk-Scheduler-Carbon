"""Restartable E1 orchestration; probe rows resume, interrupted fits restart."""

from pathlib import Path
import uuid

from .common import digest, load_json, require
from .dependence import audit_dependence
from .diagnostics import evaluate_waits
from .fidelity import evaluate_replay
from .probe_resume import probe_lock
from .probes import collect_probes
from .waits import fit_wait_model


def checked_stage(path, resume, kind, artifacts, validate, run):
    """Skip only verified complete stages; preserve incomplete stages for retry."""
    path = Path(path)
    if path.exists():
        require(resume, f'Output already exists: {path}')
        meta = load_json(path / 'manifest.json') if (path / 'manifest.json').exists() else {}
        if meta.get('status') == 'complete':
            require(meta.get('kind') == kind, f'Unexpected completed stage: {path}')
            for name, key in artifacts.items():
                require((path / name).is_file() and digest(path / name) == meta.get(key),
                        f'Completed stage artifact hash mismatch: {path / name}')
            validate(meta)
            print(f'Already complete and verified: {path}', flush=True)
            return
        backup = path.with_name(path.name + '.interrupted-' + uuid.uuid4().hex)
        path.rename(backup)
        print(f'Preserved unfinished stage at {backup}; restarting this stage.', flush=True)
    run()


def run_wait_pipeline(bundle, output, interval_seconds=21600, resume=False, e1=False):
    output = Path(output)
    with probe_lock(output, resume, name='.pipeline.lock'):
        train, validation = output / 'train-probes', output / 'validation-probes'
        collect_probes(bundle, train, 'train', interval_seconds, resume=resume)
        train_meta = load_json(train / 'manifest.json')
        model = output / 'model'

        def check_model(meta):
            artifact = load_json(model / 'model.json')
            require(artifact['probes_sha256'] == train_meta['probes_sha256'], 'Existing model uses different training probes')
            options = {'n_estimators': 256, 'max_depth': 4, 'learning_rate': .05,
                       'loss': 'squared_error', 'random_state': 11, 'n_iter_no_change': None}
            require(artifact['fit_options'] == options and len(artifact['folds']) == 4 and
                    all(len(v) == 32 for v in artifact['residual_atoms'].values()),
                    'Existing model uses different fitting settings')

        checked_stage(model, resume, 'wait_model', {'model.json': 'model_sha256'},
                      check_model, lambda: fit_wait_model(train, model))
        collect_probes(bundle, validation, 'validation', interval_seconds, resume=resume)
        validation_meta = load_json(validation / 'manifest.json')
        predictor_hash = digest(model / 'model.json')
        diagnostics = output / 'validation-diagnostics'

        def check_diagnostics(meta):
            require(meta['probes_sha256'] == validation_meta['probes_sha256'] and
                    meta['predictor_sha256'] == predictor_hash,
                    'Existing diagnostics use different probes/model')

        checked_stage(diagnostics, resume, 'wait_evaluation',
                      {'metrics.json': 'metrics_sha256', 'predictions.jsonl': 'predictions_sha256'},
                      check_diagnostics, lambda: evaluate_waits(validation, model / 'model.json', diagnostics))
        if e1:
            if bundle.raw['trace'].get('role', 'historical') == 'historical':
                fidelity = output / 'replay-validation'

                def check_fidelity(meta):
                    require(all(meta.get(k) == v for k, v in bundle.manifest.items()) and
                            meta['score_split'] == 'validation' and
                            meta['submission_start_utc'] == validation_meta['probe_start_utc'] and
                            meta['submission_stop_utc'] == validation_meta['probe_stop_utc'],
                            'Existing fidelity run uses different config/split/window')

                checked_stage(fidelity, resume, 'recorded_job_fidelity_v1',
                              {'jobs.jsonl': 'jobs_sha256', 'metrics.json': 'metrics_sha256'},
                              check_fidelity, lambda: evaluate_replay(bundle, fidelity))
            else:
                print('Constructed workload: historical admission fidelity is inapplicable.', flush=True)
            audit = output / 'dependence-queue'

            def check_audit(meta):
                report = load_json(audit / 'audit.json')
                expected = [(digest(p / 'manifest.json'), digest(p / 'probes.jsonl')) for p in (train, validation)]
                actual = [(v['manifest_sha256'], v['probes_sha256']) for v in report['probe_inputs']]
                require(actual == expected and not report['episode_inputs'], 'Existing dependence audit uses different inputs')

            checked_stage(audit, resume, 'development_dependence_diagnostics',
                          {'audit.json': 'audit_sha256', 'series.jsonl': 'series_sha256'},
                          check_audit, lambda: audit_dependence([train, validation], audit))
        print(f"{'E1 queue stage' if e1 else 'Wait pipeline'} complete: {output}. This is not RL training.", flush=True)
