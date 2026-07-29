from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from .bitbank import (
    BitbankAPIError,
    BitbankPrivateReadOnlyClient,
    BitbankPublicClient,
)
from .config import CryptoSafetyContract
from .fast import FastPaperConfig, FastPaperRunner, fast_report_markdown
from .improvement import CryptoImprovementEvaluator
from .ledger import CryptoLedger
from .paper import PaperEngine
from .profitability_study import BitbankProfitabilityStudy
from .report import atomic_write_json, atomic_write_text, scan_markdown
from .reporting import IroriSlackSummarySink, PaperShadowReportingWriter
from .shadow import (
    PaperShadowAlreadyRunning,
    PaperShadowService,
    PaperShadowServiceConfig,
)
from .scanner import CryptoMarketScanner
from .strategy_audit import StrategyLabAudit
from .strategies import load_strategy_profiles, strategy_router
from .stream import BitbankPublicStream, BitbankStreamError


def _json_print(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def _keychain_registry_status() -> dict[str, Any]:
    account = os.environ.get(
        "QR_BITBANK_KEYCHAIN_ACCOUNT",
        os.environ.get("USER", "quant_rabbit"),
    )
    prefix = os.environ.get(
        "QR_BITBANK_KEYCHAIN_PREFIX", "QuantRabbit.Bitbank"
    )
    security = os.environ.get("QR_SECURITY_BIN", "/usr/bin/security")
    entries: list[dict[str, Any]] = []
    for suffix in ("readonly_api_key", "readonly_api_secret"):
        service = f"{prefix}.{suffix}"
        try:
            completed = subprocess.run(
                [
                    security,
                    "find-generic-password",
                    "-a",
                    account,
                    "-s",
                    service,
                ],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            present = completed.returncode == 0
        except OSError:
            present = False
        entries.append({"service": service, "present": present})
    return {"account": account, "entries": entries}


def _record_scan(ledger: CryptoLedger, scan: dict[str, Any]) -> None:
    scan_id = str(scan["observed_at_utc"])
    ledger.append(
        "MARKET_SCAN",
        scan_id,
        {
            "observed_at_utc": scan_id,
            "guardian": scan["guardian"],
            "counts": scan["counts"],
            "request_stats": scan["request_stats"],
        },
        dedupe_key=f"market-scan:{scan_id}",
    )
    for pair in scan["pairs"]:
        ledger.append(
            "MARKET_DECISION",
            str(pair["pair"]),
            {
                "observed_at_utc": scan_id,
                "pair": pair["pair"],
                "candidate": pair["candidate"],
                "eligible": pair["eligible"],
                "net_edge_bps": pair["net_edge_bps"],
                "regime": pair["regime"],
                "reasons": pair["reasons"],
                "guardian_state": scan["guardian"]["state"],
                "authority": "NONE",
            },
            dedupe_key=f"market-decision:{scan_id}:{pair['pair']}",
        )


def _paper_cycle(
    client: BitbankPublicClient,
    ledger: CryptoLedger,
    paper: PaperEngine,
    scan: dict[str, Any],
) -> dict[str, Any]:
    pair_by_name = {item["pair"]: item for item in scan["pairs"]}
    fills: list[dict[str, Any]] = []
    if scan["guardian"]["kill_switch"]:
        return {"fills": fills, "skipped": "GUARDIAN_KILL_SWITCH"}
    for intent in scan["virtual_intents"]:
        pair = str(intent["pair"])
        assessment = pair_by_name[pair]
        depth = client.fetch_depth(pair)
        fills.append(
            paper.process_intent(
                intent,
                depth=depth,
                maker_fee_rate=Decimal(
                    str(assessment["maker_fee_rate_quote"])
                ),
                taker_fee_rate=Decimal(
                    str(assessment["taker_fee_rate_quote"])
                ),
            )
        )
    return {"fills": fills, "skipped": None}


def run_scan(args: argparse.Namespace) -> int:
    safety = CryptoSafetyContract.from_env()
    client = BitbankPublicClient()
    scan = CryptoMarketScanner(client, safety=safety).scan()
    if args.output_json:
        atomic_write_json(Path(args.output_json), scan)
    if args.output_markdown:
        atomic_write_text(Path(args.output_markdown), scan_markdown(scan))
    _json_print(scan)
    return 0


def run_canary(args: argparse.Namespace) -> int:
    safety = CryptoSafetyContract.from_env()
    data_dir = Path(args.data_dir)
    ledger = CryptoLedger(data_dir / "ledger.db")
    client = BitbankPublicClient()
    scanner = CryptoMarketScanner(client, safety=safety)
    paper = PaperEngine(
        ledger, initial_cash_jpy=Decimal(str(args.initial_cash_jpy))
    )
    cycles: list[dict[str, Any]] = []
    latest_scan: dict[str, Any] | None = None
    latest_metrics: dict[str, Any] | None = None
    for index in range(args.cycles):
        latest_scan = scanner.scan()
        _record_scan(ledger, latest_scan)
        paper_result = _paper_cycle(client, ledger, paper, latest_scan)
        bids = {
            str(item["pair"]): Decimal(str(item["bid"]))
            for item in latest_scan["pairs"]
        }
        latest_metrics = paper.mark_to_market(bids)
        cycles.append(
            {
                "cycle": index + 1,
                "observed_at_utc": latest_scan["observed_at_utc"],
                "guardian": latest_scan["guardian"],
                "counts": latest_scan["counts"],
                "paper": paper_result,
                "metrics": latest_metrics,
            }
        )
        atomic_write_json(data_dir / "latest_scan.json", latest_scan)
        if index + 1 < args.cycles:
            time.sleep(args.interval_sec)
    assert latest_scan is not None and latest_metrics is not None
    integrity = ledger.verify()
    keychain = _keychain_registry_status()
    try:
        messages = asyncio.run(
            BitbankPublicStream().collect(
                [f"ticker_{args.stream_pair.lower()}"],
                max_messages=1,
                timeout_sec=args.stream_timeout_sec,
            )
        )
        stream_result = {
            "ok": bool(messages),
            "message_count": len(messages),
            "room": f"ticker_{args.stream_pair.lower()}",
            "error": None,
        }
    except (BitbankStreamError, TimeoutError, OSError) as exc:
        stream_result = {
            "ok": False,
            "message_count": 0,
            "room": f"ticker_{args.stream_pair.lower()}",
            "error": type(exc).__name__,
        }
    result = {
        "schema": "QR_CRYPTO_CANARY_V1",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "venue": "bitbank",
        "mode": "READ_ONLY_SHADOW_PAPER",
        "safety": safety.as_dict(),
        "cycles": cycles,
        "final_metrics": latest_metrics,
        "ledger_integrity": integrity,
        "public_stream": stream_result,
        "private_rest": {
            "attempted": False,
            "blocked": True,
            "reason": "ROTATED_KEYCHAIN_CREDENTIAL_ABSENT",
            "keychain": keychain,
        },
    }
    atomic_write_json(data_dir / "canary.json", result)
    atomic_write_text(
        Path(args.report),
        scan_markdown(latest_scan, latest_metrics, result),
    )
    _json_print(result)
    return 0


def run_stream_canary(args: argparse.Namespace) -> int:
    rooms = [f"ticker_{pair.lower()}" for pair in args.pairs]
    try:
        messages = asyncio.run(
            BitbankPublicStream().collect(
                rooms,
                max_messages=args.messages,
                timeout_sec=args.timeout_sec,
            )
        )
    except (BitbankStreamError, TimeoutError, OSError) as exc:
        _json_print(
            {
                "ok": False,
                "mode": "PUBLIC_STREAM_READ_ONLY",
                "error": type(exc).__name__,
            }
        )
        return 2
    summary = {
        "ok": bool(messages),
        "mode": "PUBLIC_STREAM_READ_ONLY",
        "rooms": rooms,
        "message_count": len(messages),
        "room_names": sorted(
            {
                str(item.get("room_name", ""))
                for item in messages
                if item.get("room_name")
            }
        ),
    }
    _json_print(summary)
    return 0 if messages else 2


def _select_fast_pairs(
    client: BitbankPublicClient,
    requested: list[str],
    pair_limit: int,
    *,
    margin_paper: bool,
) -> tuple[
    list[str],
    dict[str, tuple[Decimal, Decimal]],
    dict[str, tuple[Decimal, Decimal]],
]:
    settings = {
        str(row.get("name", "")).lower(): row
        for row in client.fetch_pair_settings()
    }
    tickers = {
        str(row.get("pair", "")).lower(): row
        for row in client.fetch_tickers_jpy()
    }
    eligible = [
        pair
        for pair, spec in settings.items()
        if str(spec.get("quote_asset", "")).lower() == "jpy"
        and bool(spec.get("is_enabled"))
        and not bool(spec.get("stop_order"))
        and not bool(spec.get("stop_order_and_cancel"))
        and (
            not margin_paper
            or (
                spec.get("margin_current_individual_ratio") is not None
                and not bool(spec.get("stop_margin_long_order"))
                and not bool(spec.get("stop_margin_short_order"))
            )
        )
    ]
    ranked = sorted(
        eligible,
        key=lambda pair: Decimal(str(tickers.get(pair, {}).get("last", "0")))
        * Decimal(str(tickers.get(pair, {}).get("vol", "0"))),
        reverse=True,
    )
    selected = (
        [pair.lower() for pair in requested]
        if requested
        else ranked[:pair_limit]
    )
    if not selected:
        raise ValueError("no enabled JPY pair is available for fast Paper")
    unknown = [pair for pair in selected if pair not in eligible]
    if unknown:
        raise ValueError("requested fast Paper pair is unavailable")
    if margin_paper:
        fees = {
            pair: (
                max(
                    Decimal(
                        str(
                            settings[pair].get(
                                "margin_open_maker_fee_rate_quote", "0"
                            )
                        )
                    ),
                    Decimal(
                        str(
                            settings[pair].get(
                                "margin_close_maker_fee_rate_quote", "0"
                            )
                        )
                    ),
                ),
                max(
                    Decimal(
                        str(
                            settings[pair].get(
                                "margin_open_taker_fee_rate_quote", "0"
                            )
                        )
                    ),
                    Decimal(
                        str(
                            settings[pair].get(
                                "margin_close_taker_fee_rate_quote", "0"
                            )
                        )
                    ),
                ),
            )
            for pair in selected
        }
    else:
        fees = {
            pair: (
                Decimal(
                    str(settings[pair].get("maker_fee_rate_quote", "0"))
                ),
                Decimal(
                    str(settings[pair].get("taker_fee_rate_quote", "0"))
                ),
            )
            for pair in selected
        }
    interest = {
        pair: (
            Decimal(str(settings[pair].get("margin_long_interest") or "0")),
            Decimal(str(settings[pair].get("margin_short_interest") or "0")),
        )
        for pair in selected
    }
    return selected, fees, interest


def run_fast_paper(args: argparse.Namespace) -> int:
    safety = CryptoSafetyContract.from_env()
    safety.assert_safe()
    if args.duration_sec <= 0 or args.max_events <= 0:
        raise ValueError("duration and max-events must be positive")
    client = BitbankPublicClient()
    pairs, pair_fees, daily_interest_rates = _select_fast_pairs(
        client,
        list(args.pairs),
        args.pair_limit,
        margin_paper=args.margin_paper,
    )
    data_dir = Path(
        args.data_dir
        or (
            "data/crypto/fast-margin"
            if args.margin_paper
            else "data/crypto/fast-spot"
        )
    )
    ledger = CryptoLedger(data_dir / "ledger.db")
    paper = PaperEngine(
        ledger,
        initial_cash_jpy=Decimal(str(args.initial_cash_jpy)),
        allow_short=args.margin_paper,
        max_leverage=(
            Decimal(str(args.max_leverage))
            if args.margin_paper
            else Decimal("1")
        ),
    )
    fast_config = FastPaperConfig.from_env()
    router = (
        None
        if args.strategy == "FAST_MICROSTRUCTURE"
        else strategy_router(
            args.strategy,
            config_path=(
                Path(args.strategy_config)
                if args.strategy_config
                else None
            ),
            warmup_events=fast_config.warmup_events,
            book_levels=fast_config.book_levels,
            max_data_age_ms=fast_config.max_data_age_ms,
        )
    )
    runner = FastPaperRunner(
        ledger,
        paper,
        config=fast_config,
        router=router,
        strategy_name=args.strategy,
    )
    try:
        result = asyncio.run(
            runner.run(
                pairs,
                pair_fees,
                duration_sec=args.duration_sec,
                max_events=args.max_events,
                daily_interest_rates=(
                    daily_interest_rates if args.margin_paper else {}
                ),
            )
        )
    except (BitbankStreamError, OSError) as exc:
        _json_print(
            {
                "ok": False,
                "mode": "PUBLIC_STREAM_EVENT_DRIVEN_PAPER",
                "error": type(exc).__name__,
                "safety": safety.as_dict(),
            }
        )
        return 2
    atomic_write_json(data_dir / "fast_paper_canary.json", result)
    report = Path(
        args.report
        or (
            "docs/crypto_bitbank_fast_margin_paper_report.md"
            if args.margin_paper
            else "docs/crypto_bitbank_fast_spot_paper_report.md"
        )
    )
    atomic_write_text(report, fast_report_markdown(result))
    if args.summary_only:
        _json_print(
            {
                "schema": result["schema"],
                "mode": result["mode"],
                "strategy": result["strategy"],
                "run_id": result["run_id"],
                "pairs": result["pairs"],
                "safety": result["safety"],
                "guardian": result["guardian"],
                "runtime": result["runtime"],
                "latency": result["latency"],
                "decisions": result["decisions"],
                "metrics": result["metrics"],
                "ledger_integrity": result["ledger_integrity"],
            }
        )
    else:
        _json_print(result)
    return 0 if result["runtime"]["events_processed"] > 0 else 2


def run_shadow_service(args: argparse.Namespace) -> int:
    safety = CryptoSafetyContract.from_env()
    safety.assert_safe()
    margin_paper = args.mode == "margin"
    client = BitbankPublicClient()
    pairs, pair_fees, daily_interest_rates = _select_fast_pairs(
        client,
        list(args.pairs),
        args.pair_limit,
        margin_paper=margin_paper,
    )
    runtime_dir = Path(args.runtime_root) / args.mode
    service = PaperShadowService(
        PaperShadowServiceConfig(
            mode=args.mode,
            runtime_dir=runtime_dir,
            strategy=args.strategy,
            strategy_config=(
                Path(args.strategy_config)
                if args.strategy_config
                else None
            ),
            initial_cash_jpy=Decimal(str(args.initial_cash_jpy)),
            max_leverage=Decimal(str(args.max_leverage)),
            epoch_sec=args.epoch_sec,
            max_events=args.max_events,
            progress_interval_sec=args.progress_interval_sec,
            retry_delay_sec=args.retry_delay_sec,
        ),
        pairs=pairs,
        pair_fees=pair_fees,
        daily_interest_rates=daily_interest_rates,
    )
    try:
        return service.run()
    except PaperShadowAlreadyRunning:
        _json_print(
            {
                "ok": False,
                "blocked": True,
                "reason": "PAPER_SHADOW_ALREADY_RUNNING",
                "mode": args.mode,
            }
        )
        return 4


def run_shadow_report(args: argparse.Namespace) -> int:
    CryptoSafetyContract.from_env().assert_safe()
    runtime_root = Path(args.runtime_root)
    improvement = CryptoImprovementEvaluator(runtime_root).run_once()
    slack = None
    slack_blocker = "NOTION_ROUTE_GATE_UNVERIFIED"
    if args.notion_route_gate_verified:
        if args.irori_route_ref and args.irori_parent_ts:
            try:
                slack = IroriSlackSummarySink(
                    helper_path=Path(args.irori_helper),
                    route_ref=args.irori_route_ref,
                    parent_ts=args.irori_parent_ts,
                )
                slack_blocker = ""
            except ValueError:
                slack_blocker = "IRORI_HELPER_CONFIG_INVALID"
        else:
            slack_blocker = "IRORI_ROUTE_CONFIG_MISSING"
    result = PaperShadowReportingWriter(
        runtime_root,
        slack=slack,
        slack_blocker=slack_blocker,
    ).run_once()
    result["continuous_improvement"] = {
        "evaluation_added": improvement["evaluation_added"],
        "experiment_added": improvement["experiment_added"],
        "evaluations_path": improvement["evaluations_path"],
        "experiments_path": improvement["experiments_path"],
        "live_mutation": False,
    }
    atomic_write_json(runtime_root / "reporting_state.json", result)
    _json_print(result)
    return 0


def run_shadow_evaluate(args: argparse.Namespace) -> int:
    CryptoSafetyContract.from_env().assert_safe()
    result = CryptoImprovementEvaluator(
        Path(args.runtime_root)
    ).run_once(trailing_minutes=args.trailing_minutes)
    _json_print(result)
    return 0


def run_strategy_lab_evaluate(args: argparse.Namespace) -> int:
    CryptoSafetyContract.from_env().assert_safe()
    runtime_root = Path(args.runtime_root)
    results: list[dict[str, Any]] = []
    reporting: list[dict[str, Any]] = []
    for name in load_strategy_profiles(
        Path(args.strategy_config) if args.strategy_config else None
    ):
        slug = name.lower().replace("_", "-")
        strategy_root = runtime_root / slug
        results.append(
            CryptoImprovementEvaluator(
                strategy_root,
                baseline_strategy=name,
            ).run_once(trailing_minutes=args.trailing_minutes)
        )
        reporting.append(
            {
                "strategy": name,
                **PaperShadowReportingWriter(
                    strategy_root,
                    slack_blocker="NOTION_ROUTE_GATE_UNVERIFIED",
                ).run_once(),
            }
        )
    audit = StrategyLabAudit(
        runtime_root,
        baseline_root=Path(args.baseline_root),
        strategy_config=(
            Path(args.strategy_config) if args.strategy_config else None
        ),
    ).run_once()
    payload = {
        "schema": "QR_CRYPTO_STRATEGY_LAB_EVALUATION_RUN_V1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "strategies": results,
        "audit": audit,
        "reporting": reporting,
        "current_report_delivery": {
            "operation_id": audit["operation_id"],
            "durable_record": str(
                runtime_root / "audit" / "audits.jsonl"
            ),
            "sheets_status": "BLOCKED_NO_CONNECTOR",
            "slack_status": "BLOCKED_NOTION_ROUTE_GATE_UNVERIFIED",
            "blind_retry_allowed": False,
        },
        "authority": "NONE",
        "live_mutation": False,
    }
    atomic_write_json(runtime_root / "evaluation_state.json", payload)
    _json_print(payload)
    return 0


def run_profitability_study(args: argparse.Namespace) -> int:
    CryptoSafetyContract.from_env().assert_safe()
    payload = BitbankProfitabilityStudy(
        Path(args.runtime_root),
        research_config=Path(args.research_config),
        output_root=Path(args.output_root),
    ).run_once()
    _json_print(
        {
            "schema": payload["schema"],
            "operation_id": payload["operation_id"],
            "baseline": payload["baseline_contract"]["metrics"],
            "comparisons": payload["isolated_comparisons"],
            "output_root": str(Path(args.output_root)),
            "safety": payload["safety"],
        }
    )
    return 0


def run_private_check(_: argparse.Namespace) -> int:
    CryptoSafetyContract.from_env().assert_safe()
    api_key = os.environ.get("QR_BITBANK_API_KEY", "")
    api_secret = os.environ.get("QR_BITBANK_API_SECRET", "")
    if not api_key or not api_secret:
        _json_print(
            {
                "ok": False,
                "blocked": True,
                "reason": "KEYCHAIN_CREDENTIALS_NOT_PRESENT",
                "operation": "GET_ASSETS_ONLY",
            }
        )
        return 3
    try:
        assets = BitbankPrivateReadOnlyClient(api_key, api_secret).fetch_assets()
    except BitbankAPIError as exc:
        _json_print(
            {
                "ok": False,
                "blocked": True,
                "reason": type(exc).__name__,
                "operation": "GET_ASSETS_ONLY",
            }
        )
        return 3
    _json_print(
        {
            "ok": True,
            "blocked": False,
            "authenticated": True,
            "operation": "GET_ASSETS_ONLY",
            "asset_record_count": len(assets),
            "secret_values_reported": False,
        }
    )
    return 0


def run_margin_status(_: argparse.Namespace) -> int:
    CryptoSafetyContract.from_env().assert_safe()
    api_key = os.environ.get("QR_BITBANK_API_KEY", "")
    api_secret = os.environ.get("QR_BITBANK_API_SECRET", "")
    if not api_key or not api_secret:
        _json_print(
            {
                "ok": False,
                "blocked": True,
                "reason": "KEYCHAIN_CREDENTIALS_NOT_PRESENT",
                "operation": "GET_MARGIN_STATUS_ONLY",
            }
        )
        return 3
    try:
        status = BitbankPrivateReadOnlyClient(
            api_key, api_secret
        ).fetch_margin_status()
    except BitbankAPIError as exc:
        _json_print(
            {
                "ok": False,
                "blocked": True,
                "reason": type(exc).__name__,
                "operation": "GET_MARGIN_STATUS_ONLY",
            }
        )
        return 3
    _json_print(
        {
            "ok": True,
            "blocked": False,
            "operation": "GET_MARGIN_STATUS_ONLY",
            "account_status": status.get("status"),
            "total_margin_balance": status.get("total_margin_balance"),
            "margin_balance_percentage": status.get(
                "total_margin_balance_percentage"
            ),
            "margin_call_percentage": status.get("margin_call_percentage"),
            "losscut_percentage": status.get("losscut_percentage"),
            "buy_credit": status.get("buy_credit"),
            "sell_credit": status.get("sell_credit"),
            "available_balances": status.get("available_balances", []),
            "secret_values_reported": False,
            "mutation_attempted": False,
        }
    )
    return 0


def run_ledger_verify(args: argparse.Namespace) -> int:
    _json_print(CryptoLedger(Path(args.ledger)).verify())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qr-crypto",
        description="bitbank public-data Shadow/Paper tooling (no live authority).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan = subparsers.add_parser("scan", help="Run a public REST market scan.")
    scan.add_argument("--output-json")
    scan.add_argument("--output-markdown")
    scan.set_defaults(func=run_scan)

    canary = subparsers.add_parser(
        "canary", help="Run a bounded public REST + paper canary."
    )
    canary.add_argument("--cycles", type=int, default=1)
    canary.add_argument("--interval-sec", type=float, default=2.0)
    canary.add_argument("--initial-cash-jpy", default="10000")
    canary.add_argument("--stream-pair", default="btc_jpy")
    canary.add_argument("--stream-timeout-sec", type=float, default=15.0)
    canary.add_argument("--data-dir", default="data/crypto")
    canary.add_argument(
        "--report", default="docs/crypto_bitbank_canary_report.md"
    )
    canary.set_defaults(func=run_canary)

    stream = subparsers.add_parser(
        "stream-canary", help="Read bounded public Socket.IO ticker messages."
    )
    stream.add_argument("pairs", nargs="+")
    stream.add_argument("--messages", type=int, default=1)
    stream.add_argument("--timeout-sec", type=float, default=15.0)
    stream.set_defaults(func=run_stream_canary)

    fast = subparsers.add_parser(
        "fast-paper",
        help="Run bounded event-driven Public Stream Paper trading.",
    )
    fast.add_argument("pairs", nargs="*")
    fast.add_argument("--pair-limit", type=int, default=2)
    fast.add_argument("--duration-sec", type=float, default=30.0)
    fast.add_argument("--max-events", type=int, default=20_000)
    fast.add_argument("--initial-cash-jpy", default="10000")
    fast.add_argument("--margin-paper", action="store_true")
    fast.add_argument("--max-leverage", default="2")
    fast.add_argument("--data-dir")
    fast.add_argument("--report")
    fast.add_argument("--summary-only", action="store_true")
    fast.add_argument(
        "--strategy",
        default="FAST_MICROSTRUCTURE",
        help="Paper strategy name; configured siblings remain authority NONE.",
    )
    fast.add_argument("--strategy-config")
    fast.set_defaults(func=run_fast_paper)

    shadow = subparsers.add_parser(
        "shadow-service",
        help="Run one continuous Public Stream Paper Shadow service.",
    )
    shadow.add_argument("--mode", choices=("spot", "margin"), required=True)
    shadow.add_argument("pairs", nargs="*")
    shadow.add_argument("--pair-limit", type=int, default=2)
    shadow.add_argument(
        "--runtime-root",
        default="data/crypto/paper-shadow",
    )
    shadow.add_argument("--initial-cash-jpy", default="10000")
    shadow.add_argument("--max-leverage", default="2")
    shadow.add_argument("--epoch-sec", type=float, default=60.0)
    shadow.add_argument("--max-events", type=int, default=10_000_000)
    shadow.add_argument("--progress-interval-sec", type=float, default=5.0)
    shadow.add_argument("--retry-delay-sec", type=float, default=5.0)
    shadow.add_argument(
        "--strategy",
        default="FAST_MICROSTRUCTURE",
        help="Paper strategy name; use a distinct runtime root per strategy.",
    )
    shadow.add_argument("--strategy-config")
    shadow.set_defaults(func=run_shadow_service)

    reporting = subparsers.add_parser(
        "shadow-report",
        help="Drain Paper trade outboxes into isolated reporting targets.",
    )
    reporting.add_argument(
        "--runtime-root",
        default="data/crypto/paper-shadow",
    )
    reporting.add_argument(
        "--notion-route-gate-verified",
        action="store_true",
        help="Caller directly fetched and verified the current Notion route gate.",
    )
    reporting.add_argument("--irori-route-ref")
    reporting.add_argument("--irori-parent-ts")
    reporting.add_argument(
        "--irori-helper",
        default=(
            "/Users/tossaki/.claude/scheduled-tasks/"
            "_shared/post_slack.sh"
        ),
    )
    reporting.set_defaults(func=run_shadow_report)

    evaluation = subparsers.add_parser(
        "shadow-evaluate",
        help="Persist a crypto Paper profitability and cost RCA window.",
    )
    evaluation.add_argument(
        "--runtime-root",
        default="data/crypto/paper-shadow",
    )
    evaluation.add_argument("--trailing-minutes", type=int, default=60)
    evaluation.set_defaults(func=run_shadow_evaluate)

    lab_evaluation = subparsers.add_parser(
        "strategy-lab-evaluate",
        help="Persist local profitability RCA for every Paper strategy lane.",
    )
    lab_evaluation.add_argument(
        "--runtime-root",
        default="data/crypto/strategy-lab",
    )
    lab_evaluation.add_argument("--strategy-config")
    lab_evaluation.add_argument(
        "--baseline-root",
        default="data/crypto/paper-shadow",
    )
    lab_evaluation.add_argument("--trailing-minutes", type=int)
    lab_evaluation.set_defaults(func=run_strategy_lab_evaluate)

    profitability = subparsers.add_parser(
        "strategy-profitability-study",
        help="Run read-only bitbank Paper one-category comparisons.",
    )
    profitability.add_argument(
        "--runtime-root",
        default="data/crypto/strategy-lab",
    )
    profitability.add_argument(
        "--research-config",
        default="config/crypto_bitbank_research_candidates_v1.json",
    )
    profitability.add_argument(
        "--output-root",
        default="data/crypto/profitability-study",
    )
    profitability.set_defaults(func=run_profitability_study)

    private = subparsers.add_parser(
        "private-check", help="Authenticate and GET assets only."
    )
    private.set_defaults(func=run_private_check)

    margin = subparsers.add_parser(
        "margin-status",
        help="Authenticate and GET margin availability only.",
    )
    margin.set_defaults(func=run_margin_status)

    ledger = subparsers.add_parser(
        "ledger-verify", help="Verify the append-only hash chain."
    )
    ledger.add_argument("--ledger", default="data/crypto/ledger.db")
    ledger.set_defaults(func=run_ledger_verify)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (BitbankAPIError, ValueError) as exc:
        print(
            json.dumps(
                {"ok": False, "error": type(exc).__name__},
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
