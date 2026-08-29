#!/usr/bin/env python3
"""Train or verify the sealed EUR/USD zero-authority shadow router."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quant_rabbit.eurusd_outcome_learning import (  # noqa: E402
    freeze_training_artifacts,
    observe_prospective,
    parse_utc,
    resolve_current_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train")
    train.add_argument("--signal-ledger", type=Path, required=True)
    train.add_argument("--outcome-ledger", type=Path, required=True)
    train.add_argument("--corrective-ledger", type=Path, required=True)
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--output-root", type=Path, required=True)
    train.add_argument("--activation-at-utc", required=True)
    train.add_argument("--now-at-utc")

    observe = subparsers.add_parser("observe")
    manifest_group = observe.add_mutually_exclusive_group(required=True)
    manifest_group.add_argument("--manifest", type=Path)
    manifest_group.add_argument("--current-pointer", type=Path)
    observe.add_argument("--config", type=Path, required=True)
    observe.add_argument("--shock-signal-ledger", type=Path, required=True)
    observe.add_argument("--shock-outcome-ledger", type=Path, required=True)
    observe.add_argument("--decision-ledger", type=Path, required=True)
    observe.add_argument("--prospective-outcome-ledger", type=Path, required=True)
    observe.add_argument("--prospective-scorecard", type=Path, required=True)
    observe.add_argument("--now-at-utc")

    args = parser.parse_args()
    now = parse_utc(args.now_at_utc) if args.now_at_utc else datetime.now(timezone.utc)
    if args.command == "train":
        result = freeze_training_artifacts(
            signal_ledger_path=args.signal_ledger,
            outcome_ledger_path=args.outcome_ledger,
            corrective_ledger_path=args.corrective_ledger,
            config_path=args.config,
            output_root=args.output_root,
            activation_at_utc=parse_utc(args.activation_at_utc),
            now_utc=now,
        )
    else:
        manifest_path = args.manifest or resolve_current_manifest(args.current_pointer)
        result = observe_prospective(
            manifest_path=manifest_path,
            config_path=args.config,
            shock_signal_ledger_path=args.shock_signal_ledger,
            shock_outcome_ledger_path=args.shock_outcome_ledger,
            decision_ledger_path=args.decision_ledger,
            prospective_outcome_ledger_path=args.prospective_outcome_ledger,
            prospective_scorecard_path=args.prospective_scorecard,
            now_utc=now,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
