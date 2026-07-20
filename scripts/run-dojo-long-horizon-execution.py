#!/usr/bin/env python3
"""Operate the broker-free DOJO long-horizon execution state machine.

The command only validates JSON and appends control-plane claims, coordinate
receipts, terminal manifests, carry slots, and reducer handoffs.  It never
executes a strategy, opens market data, imports broker/live code, calls a
model, computes portfolio economics, or grants promotion/live authority.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from quant_rabbit.dojo_long_horizon_execution import (  # noqa: E402
    DojoLongHorizonExecutionError,
    claim_long_horizon_job,
    claim_next_long_horizon_job,
    initialize_long_horizon_execution_state,
    long_horizon_execution_status,
    record_long_horizon_coordinate_result,
    record_long_horizon_coordinate_results,
    resume_long_horizon_claim,
    seal_long_horizon_attempt,
    validate_long_horizon_terminal_manifest,
)
from quant_rabbit.dojo_long_horizon_plan import (  # noqa: E402
    DojoLongHorizonPlanError,
)
from quant_rabbit.dojo_long_horizon_schedule import (  # noqa: E402
    DojoLongHorizonScheduleError,
)


# The sealed 32,112-coordinate schedule is intentionally larger than one
# control-plane receipt.  128 MiB bounds its parser surface while leaving room
# for the exact fixed denominator; this is a storage safety bound, not a market
# parameter.
MAX_INPUT_BYTES = 128 * 1024 * 1024


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    init = commands.add_parser("init", help="seal one execution manifest/state root")
    _common(init)
    init.add_argument("--runner-binding", type=Path, required=True)
    init.add_argument("--resource-policy", type=Path, required=True)

    status = commands.add_parser("status", help="verify state and report exact counts")
    _common(status)

    claim_next = commands.add_parser(
        "claim-next", help="claim the first ready job in sealed schedule order"
    )
    _common(claim_next)
    claim_next.add_argument("--runner-id", required=True)

    claim = commands.add_parser("claim", help="claim one exact content-addressed job")
    _common(claim)
    claim.add_argument("--job-sha256", required=True)
    claim.add_argument("--runner-id", required=True)

    resume = commands.add_parser("resume", help="resume one exact durable claim")
    _common(resume)
    resume.add_argument("--claim-sha256", required=True)

    record = commands.add_parser(
        "record-cell", help="append one COMPLETE or FAILED coordinate result"
    )
    _common(record)
    record.add_argument("--claim-sha256", required=True)
    record.add_argument("--result", type=Path, required=True)

    record_batch = commands.add_parser(
        "record-cells", help="append an object containing a results array"
    )
    _common(record_batch)
    record_batch.add_argument("--claim-sha256", required=True)
    record_batch.add_argument("--results", type=Path, required=True)

    seal = commands.add_parser("seal", help="seal a full attempt and reducer handoff")
    _common(seal)
    seal.add_argument("--claim-sha256", required=True)

    verify = commands.add_parser(
        "validate-terminal", help="purely validate a scorer-facing terminal manifest"
    )
    _common(verify)
    verify.add_argument("--terminal", type=Path, required=True)
    return parser


def _load(path: Path, *, field: str) -> dict[str, Any]:
    try:
        state = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise DojoLongHorizonExecutionError(f"cannot inspect {field}") from exc
    if (
        not path.is_file()
        or path.is_symlink()
        or not 0 < state.st_size <= MAX_INPUT_BYTES
    ):
        raise DojoLongHorizonExecutionError(
            f"{field} must be a bounded nonempty regular file"
        )

    def reject_constant(token: str) -> None:
        raise DojoLongHorizonExecutionError(f"non-finite JSON is forbidden: {token}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise DojoLongHorizonExecutionError(
                    f"duplicate JSON key is forbidden: {key}"
                )
            result[key] = value
        return result

    try:
        raw = path.read_bytes()
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except DojoLongHorizonExecutionError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DojoLongHorizonExecutionError(f"cannot parse {field}") from exc
    if not isinstance(value, dict):
        raise DojoLongHorizonExecutionError(f"{field} must be one JSON object")
    return value


def _inputs(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    return _load(args.plan, field="plan"), _load(args.schedule, field="schedule")


def _dispatch(args: argparse.Namespace) -> dict[str, Any]:
    plan, schedule = _inputs(args)
    common = {"schedule": schedule, "plan": plan}
    if args.command == "init":
        return initialize_long_horizon_execution_state(
            args.state_dir,
            **common,
            runner_binding=_load(args.runner_binding, field="runner binding"),
            resource_policy=_load(args.resource_policy, field="resource policy"),
        )
    if args.command == "status":
        return long_horizon_execution_status(args.state_dir, **common)
    if args.command == "claim-next":
        return claim_next_long_horizon_job(
            args.state_dir, **common, runner_id=args.runner_id
        )
    if args.command == "claim":
        return claim_long_horizon_job(
            args.state_dir,
            **common,
            job_sha256=args.job_sha256,
            runner_id=args.runner_id,
        )
    if args.command == "resume":
        return resume_long_horizon_claim(
            args.state_dir, **common, claim_sha256=args.claim_sha256
        )
    if args.command == "record-cell":
        return record_long_horizon_coordinate_result(
            args.state_dir,
            **common,
            claim_sha256=args.claim_sha256,
            result=_load(args.result, field="coordinate result"),
        )
    if args.command == "record-cells":
        envelope = _load(args.results, field="coordinate result batch")
        if set(envelope) != {"results"}:
            raise DojoLongHorizonExecutionError(
                "coordinate result batch must contain exactly results"
            )
        raw_results = envelope["results"]
        if not isinstance(raw_results, list):
            raise DojoLongHorizonExecutionError("results must be an array")
        cells = record_long_horizon_coordinate_results(
            args.state_dir,
            **common,
            claim_sha256=args.claim_sha256,
            results=raw_results,
        )
        return {
            "recorded_coordinate_count": len(cells),
            "cell_sha256_values": [cell["cell_sha256"] for cell in cells],
            "broker_mutation_allowed": False,
            "live_permission": False,
            "portfolio_economics_computed": False,
        }
    if args.command == "seal":
        return seal_long_horizon_attempt(
            args.state_dir, **common, claim_sha256=args.claim_sha256
        )
    if args.command == "validate-terminal":
        manifest = _load(
            args.state_dir / "execution-manifest.json", field="execution manifest"
        )
        return validate_long_horizon_terminal_manifest(
            _load(args.terminal, field="terminal manifest"),
            **common,
            execution_manifest=manifest,
        )
    raise DojoLongHorizonExecutionError("unsupported execution-state command")


def _canonical_line(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = _dispatch(args)
    except (
        DojoLongHorizonExecutionError,
        DojoLongHorizonPlanError,
        DojoLongHorizonScheduleError,
        OSError,
    ) as exc:
        print(
            _canonical_line(
                {
                    "status": "REJECTED",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "automatic_deployment_allowed": False,
                    "broker_mutation_allowed": False,
                    "live_permission": False,
                    "order_authority": "NONE",
                    "portfolio_economics_computed": False,
                    "promotion_eligible": False,
                }
            ),
            file=sys.stderr,
        )
        return 2
    print(_canonical_line(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
