import argparse
import json
from pathlib import Path
import sys

from .common import ContractError, digest, json_text
from .config import Bundle
from .runner import run_fixed
from .trace import load_trace


def main(argv=None):
    parser = argparse.ArgumentParser(description="P0/P1 provisioning replay and accounting (no RL training)")
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
    validate = sub.add_parser("validate", help="Check config, hashes, trace, CI coverage and cohort without replay")
    validate.add_argument("--config", required=True)
    run = sub.add_parser("run-fixed", help="Run the paired fixed policies on a declared cohort")
    run.add_argument("--config", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--nodes", type=int, nargs="+", default=[4, 8, 16, 32])
    run.add_argument("--split", choices=["train", "validation", "test"])
    run.add_argument("--shard-index", type=int, default=0)
    run.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args(argv)
    try:
        if args.command == "hash":
            print(json_text({path: digest(path) for path in args.paths}), end="")
        elif args.command == "audit-trace":
            _, report = load_trace(args.trace, {"timezone": args.timezone, "dst_fold": args.dst_fold,
                                                "zero_duration_policy": args.zero_duration, "overrun_policy": args.overrun}, args.nodes)
            print(json_text(report), end="")
        else:
            bundle = Bundle(args.config)
            if args.command == "validate":
                print(json_text(bundle.manifest), end="")
            else:
                run_fixed(bundle, args.output, args.nodes, args.split, args.shard_index, args.shard_count)
    except (ContractError, KeyError, TypeError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
