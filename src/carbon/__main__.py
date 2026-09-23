import argparse
import json
from pathlib import Path
import sys

from .common import ContractError, digest, json_text
from .config import Bundle
from .runner import run_fixed, run_planners
from .trace import load_trace


def main(argv=None):
    parser = argparse.ArgumentParser(description="Fixed-work replay, causal planning and budget-conditioned PPO")
    sub = parser.add_subparsers(dest="command", required=True)
    hashes = sub.add_parser("hash", help="Print SHA256 hashes for manifest assets")
    hashes.add_argument("paths", nargs="+")
    audit = sub.add_parser("audit-trace", help="Read-only trace audit; print its hash and cleaning counts")
    audit.add_argument("--trace", required=True)
    audit.add_argument("--timezone", required=True, help="Source timezone; confirm this from trace provenance")
    audit.add_argument("--dst-fold", type=int, choices=[0, 1])
    audit.add_argument("--nodes", type=int)
    audit.add_argument("--zero-duration", choices=["error", "drop"], default="error")
    audit.add_argument("--overrun", choices=["error", "clip"], default="error")
    ci = sub.add_parser("prepare-ci", help="Convert the official EIA ERCOT XLSX to hourly CI; no replay")
    ci.add_argument("--workbook", required=True)
    ci.add_argument("--output", required=True)
    ci.add_argument("--start", default="2019-01-01T00:00:00Z")
    ci.add_argument("--end", default="2025-01-04T00:00:00Z")
    ci.add_argument("--release-lag-hours", type=float, default=24)
    ci.add_argument("--max-forward-fill-hours", type=int, choices=[0, 1, 2], default=0)
    validate = sub.add_parser("validate", help="Check config, hashes, trace, CI coverage and cohort without replay")
    validate.add_argument("--config", required=True)
    pilot = sub.add_parser("run-main-pilot", help="Full fixed references plus a bounded predictor-free PPO development pilot")
    pilot.add_argument("--config", required=True)
    pilot.add_argument("--output", required=True)
    pilot.add_argument("--resume", action="store_true")
    run = sub.add_parser("run-fixed", help="Run the paired fixed policies on a declared cohort")
    run.add_argument("--config", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--nodes", type=int, nargs="+", help="Defaults to every scale declared in the config")
    run.add_argument("--split", choices=["train", "validation", "test"])
    run.add_argument("--shard-index", type=int, default=0)
    run.add_argument("--shard-count", type=int, default=1)
    probe = sub.add_parser("probe-waits", help="Create request-conditioned wait labels independently of target profiles")
    probe.add_argument("--config", required=True)
    probe.add_argument("--output", required=True)
    probe.add_argument("--split", choices=["train", "validation", "test"], default="train")
    probe.add_argument("--interval-seconds", type=int, default=21600)
    probe.add_argument("--request-seconds", type=int, nargs="+")
    probe.add_argument("--start")
    probe.add_argument("--stop")
    probe.add_argument("--resume", action="store_true", help="Validate and append to an interrupted probe dataset")
    pipeline = sub.add_parser("run-wait-pipeline", help="Run E1 queue stages with optional interrupted-run recovery")
    pipeline.add_argument("--config", required=True)
    pipeline.add_argument("--output", required=True)
    pipeline.add_argument("--interval-seconds", type=int, default=21600)
    pipeline.add_argument("--resume", action="store_true")
    pipeline.add_argument("--e1", action="store_true", help="Include replay/dependence diagnostics")
    fit = sub.add_parser("fit-waits", help="Fit chronological GBT and out-of-fold residual atoms")
    fit.add_argument("--probes", required=True)
    fit.add_argument("--output", required=True)
    fit.add_argument("--trees", type=int, default=256)
    fit.add_argument("--depth", type=int, default=4)
    fit.add_argument("--learning-rate", type=float, default=.05)
    fit.add_argument("--folds", type=int, default=4)
    fit.add_argument("--atoms", type=int, default=32)
    fit.add_argument("--seed", type=int, default=11)
    evaluate = sub.add_parser("evaluate-waits", help="Score validation/test probes by scale, request length and queue regime")
    evaluate.add_argument("--probes", required=True)
    evaluate.add_argument("--predictor", required=True)
    evaluate.add_argument("--output", required=True)
    refs = sub.add_parser("make-references", help="Freeze T_ref and endpoint C_ref from training fixed results")
    refs.add_argument("--fixed-run", required=True)
    refs.add_argument("--output", required=True)
    mix = sub.add_parser("fit-fixed-mix", help="Fit validation-only fixed mixture; statistical feasibility remains separate")
    mix.add_argument("--fixed-run", required=True)
    mix.add_argument("--references", required=True)
    mix.add_argument("--budget-hours", required=True, type=float)
    mix.add_argument("--epsilon", type=float, default=.05)
    mix.add_argument("--output", required=True)
    planning = sub.add_parser("run-planners", help="Paired complete-work MPC, plan-once, and queue-blind evaluation")
    planning.add_argument("--config", required=True)
    planning.add_argument("--output", required=True)
    planning.add_argument("--predictor", required=True)
    planning.add_argument("--references", required=True)
    planning.add_argument("--split", choices=["train", "validation", "test"], default="validation")
    planning.add_argument("--paths", type=int, default=256)
    planning.add_argument("--miss-tolerance", type=float, default=.05)
    planning.add_argument("--seed", type=int, default=11)
    planning.add_argument("--methods", nargs="+", default=["Rollout-MPC", "Plan-once", "Queue-blind-MPC"])
    planning.add_argument("--budget-multiplier", type=float)
    training = sub.add_parser("train-ppo", help="Train complete-episode constrained PPO; checkpoints are not selected results")
    training.add_argument("--config", required=True)
    training.add_argument("--output", required=True)
    training.add_argument("--predictor", help="Required only with --wait-features advice; unused by the main policy")
    training.add_argument("--references", required=True)
    training.add_argument("--iterations", type=int, required=True, help="Predeclared total iterations, including completed iterations when resuming")
    training.add_argument("--resume", help="Checkpoint JSON from a previous run; writes a new output directory")
    training.add_argument("--budgets", type=float, nargs="+")
    for option in ("episodes-per-budget", "epochs", "minibatch-episodes", "seed", "threads"):
        training.add_argument("--"+option, type=int)
    for option in ("learning-rate", "clip", "entropy", "gradient-norm", "value-coefficient", "epsilon", "dual-rate-p", "dual-rate-lambda"):
        training.add_argument("--"+option, type=float)
    training.add_argument("--forecast-mode", choices=["window", "current"])
    training.add_argument("--decision-mode", choices=["feedback", "precommitted"], help="Default feedback; precommitted samples the whole scale sequence before submission")
    training.add_argument("--wait-features", choices=["none", "advice"], help="Default none: main policy has no wait-predictor dependency")
    training.add_argument("--objective", choices=["robust", "lower", "upper"])
    training.add_argument("--actor-interaction", choices=["concat", "product"], help="Default concat; product adds explicit state/action interactions")
    training.add_argument("--actor-budget-mode", choices=["shared", "independent"], help="Default shared; independent uses one actor branch per trained budget tick")
    policy = sub.add_parser("run-policy", help="Evaluate a checkpoint with its unchanged categorical sampling rule")
    policy.add_argument("--config", required=True)
    policy.add_argument("--output", required=True)
    policy.add_argument("--predictor", help="Required only by predictor-advised checkpoints")
    policy.add_argument("--checkpoint", required=True)
    policy.add_argument("--split", choices=["train", "validation", "test"], default="validation")
    policy.add_argument("--budgets", type=float, nargs="+")
    policy.add_argument("--slider-position", type=float, help="Select a supported completion-budget slider tick")
    policy.add_argument("--sampling-seed", type=int)
    stress = sub.add_parser("run-stress", help="Freeze full policy/MPC inputs and change one environment assumption")
    for option in ("config", "output", "predictor", "checkpoint"):
        stress.add_argument("--"+option, required=True)
    stress.add_argument("--variant", choices=["width2", "fcfs", "overhead60", "overhead1800"], required=True)
    stress.add_argument("--budget-multiplier", type=float, required=True)
    stress.add_argument("--split", choices=["validation", "test"], default="test")
    stress.add_argument("--paths", type=int, default=256)
    stress.add_argument("--miss-tolerance", type=float, default=.05)
    select = sub.add_parser("select-policies", help="Freeze validation operating points and comparator; no test data selection")
    select.add_argument("--spec", required=True)
    select.add_argument("--output", required=True)
    heldout = sub.add_parser("run-test", help="Execute the complete validation-frozen holdout plan; no test selection")
    heldout.add_argument("--config", required=True)
    heldout.add_argument("--selection", required=True, help="Directory produced by select-policies")
    heldout.add_argument("--output", required=True)
    heldout.add_argument("--predictor", required=True)
    heldout.add_argument("--checkpoints", nargs="+", required=True, help="Training directories or exact checkpoint JSON files; matches frozen hashes")
    report = sub.add_parser("report-test", help="Analyze a completed frozen test run, retaining every seed and unsupported point")
    report.add_argument("--run", required=True)
    report.add_argument("--output", required=True)
    fidelity = sub.add_parser("evaluate-replay", help="Compare a continuous background replay with recorded admissions; historical-cluster mode only")
    fidelity.add_argument("--config", required=True)
    fidelity.add_argument("--output", required=True)
    fidelity.add_argument("--split", choices=["train", "validation", "test"], default="validation")
    fidelity.add_argument("--start")
    fidelity.add_argument("--stop")
    dependence = sub.add_parser("audit-dependence", help="Inspect development calendar dependence and episode spans before choosing blocks")
    dependence.add_argument("--probes", nargs="+", required=True)
    dependence.add_argument("--episode-runs", nargs="*", default=[])
    dependence.add_argument("--lags-hours", nargs="+", type=float)
    dependence.add_argument("--output", required=True)
    export = sub.add_parser("export-results", help="Export sealed held-out results to scientific figures and a LaTeX mechanism table")
    export.add_argument("--run", required=True)
    export.add_argument("--report", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--tables-only", action="store_true", help="Export JSON/Markdown/LaTeX without matplotlib")
    render = sub.add_parser("render-results", help="Render a sealed data/table export without recomputing statistics or replay")
    render.add_argument("--export", dest="data_export", required=True)
    render.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "hash":
            print(json_text({path: digest(path) for path in args.paths}), end="")
        elif args.command == "audit-trace":
            _, report = load_trace(args.trace, {"timezone": args.timezone, "dst_fold": args.dst_fold,
                                                "zero_duration_policy": args.zero_duration, "overrun_policy": args.overrun}, args.nodes)
            print(json_text(report), end="")
        elif args.command == "prepare-ci":
            from .inputs import prepare_eia
            report = prepare_eia(args.workbook, args.output, args.start, args.end, args.release_lag_hours,
                                 args.max_forward_fill_hours)
            print(json_text(report), end="")
        elif args.command == "run-main-pilot":
            from .main_pilot import run_main_pilot
            run_main_pilot(Bundle(args.config), args.output, args.resume)
        elif args.command == "run-stress":
            from .stress import run_stress
            run_stress(Bundle(args.config), args.output, args.predictor, args.checkpoint,
                       args.variant, args.budget_multiplier, args.split, args.paths, args.miss_tolerance)
        elif args.command == "evaluate-replay":
            from .fidelity import evaluate_replay
            evaluate_replay(Bundle(args.config,queue_only=True),args.output,args.split,args.start,args.stop)
        elif args.command == "audit-dependence":
            from .dependence import audit_dependence
            audit_dependence(args.probes,args.output,args.episode_runs,args.lags_hours)
        elif args.command == "probe-waits":
            from .probes import collect_probes
            collect_probes(Bundle(args.config, queue_only=True), args.output, args.split, args.interval_seconds,
                           args.request_seconds, args.start, args.stop, args.resume)
        elif args.command == "run-wait-pipeline":
            from .wait_pipeline import run_wait_pipeline
            run_wait_pipeline(Bundle(args.config, queue_only=True), args.output,
                              args.interval_seconds, args.resume, args.e1)
        elif args.command == "fit-waits":
            from .waits import fit_wait_model
            fit_wait_model(args.probes, args.output, args.trees, args.depth, args.learning_rate, args.folds, args.atoms, args.seed)
        elif args.command == "make-references":
            from .baselines import make_references
            make_references(args.fixed_run, args.output)
        elif args.command == "evaluate-waits":
            from .diagnostics import evaluate_waits
            evaluate_waits(args.probes, args.predictor, args.output)
        elif args.command == "fit-fixed-mix":
            from .baselines import fit_fixed_mix
            from .common import load_json, require
            require(not Path(args.output).exists(), f"Output already exists: {args.output}")
            result = fit_fixed_mix(args.fixed_run, load_json(args.references), args.budget_hours, args.epsilon)
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json_text(result), encoding="utf-8")
        elif args.command == "render-results":
            from .exporting import render_export
            render_export(args.data_export,args.output)
        elif args.command == "export-results":
            from .exporting import export_results
            export_results(args.run,args.report,args.output,not args.tables_only)
        elif args.command == "report-test":
            from .reporting import report_test
            report_test(args.run,args.output)
        elif args.command == "select-policies":
            from .selection import select_policies
            select_policies(args.spec,args.output)
        else:
            bundle = Bundle(args.config)
            if args.command == "validate":
                print(json_text(bundle.manifest), end="")
            elif args.command == "run-test":
                from .heldout import run_test
                run_test(bundle,args.selection,args.output,args.predictor,args.checkpoints)
            elif args.command == "run-fixed":
                run_fixed(bundle, args.output, args.nodes or bundle.raw['cluster']['allowed_nodes'], args.split, args.shard_index, args.shard_count)
            elif args.command == "run-planners":
                run_planners(bundle, args.output, args.predictor, args.references, args.split, args.paths,
                             args.miss_tolerance, args.seed, args.methods, args.budget_multiplier)
            elif args.command == "train-ppo":
                from .policy_runner import DEFAULTS, settings_for, train_policy
                options = {key:getattr(args,key) for key in DEFAULTS if getattr(args,key) is not None}
                train_policy(bundle,args.output,args.predictor,args.references,settings_for(args.iterations,**options),args.resume)
            elif args.command == "run-policy":
                from .policy_runner import evaluate_policy
                evaluate_policy(bundle,args.output,args.predictor,args.checkpoint,args.split,args.budgets,args.sampling_seed,args.slider_position)
    except (ContractError, KeyError, TypeError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
