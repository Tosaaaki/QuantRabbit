from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from quant_rabbit.automation import AutoTradeCycle, DEFAULT_AUTOTRADE_REPORT
from quant_rabbit.ai_test_bot import (
    AITestBotBacktester,
    DEFAULT_MAX_ACTIVE_BUCKETS,
    DEFAULT_MIN_TRAIN_WIN_RATE_PCT,
    DEFAULT_MIN_TRAIN_TRADES,
    DEFAULT_RUNTIME_SOURCE_TABLES,
    DEFAULT_TRAINING_DAYS,
)
from quant_rabbit.attack_advisor import AttackAdvisor
from quant_rabbit.analysis.chart_reader import DEFAULT_TIMEFRAMES as DEFAULT_PAIR_CHART_TIMEFRAMES
from quant_rabbit.analysis.market_status import (
    compute_market_status,
    write_report as write_market_status_report,
    write_snapshot as write_market_status_snapshot,
)
from quant_rabbit.broker.execution import LiveOrderGateway
from quant_rabbit.broker.oanda import OandaReadOnlyClient
from quant_rabbit.broker.oanda import OandaExecutionClient
from quant_rabbit.certification import DryRunCertifier
from quant_rabbit.completion import CompletionAuditor
from quant_rabbit.coverage import CoverageOptimizer
from quant_rabbit.execution_ledger import ExecutionLedger
from quant_rabbit.execution_replay import ExecutionReplayer
from quant_rabbit.legacy.importer import LegacyImporter
from quant_rabbit.learning import PostTradeLearner
from quant_rabbit.models import BrokerSnapshot, MarketContext, OrderIntent, OrderType, Owner, Quote, Side, TradeMethod
from quant_rabbit.outcome_mart import OutcomeMartBuilder
from quant_rabbit.paths import (
    ROOT,
    DEFAULT_ADVERSE_PARTIAL_CLOSE,
    DEFAULT_ADVERSE_PARTIAL_CLOSE_REPORT,
    DEFAULT_AI_TEST_BOT_BACKTEST,
    DEFAULT_AI_TEST_BOT_BACKTEST_REPORT,
    DEFAULT_AI_ATTACK_ADVICE,
    DEFAULT_AI_ATTACK_ADVICE_REPORT,
    DEFAULT_BROKER_SNAPSHOT,
    DEFAULT_CAMPAIGN_PLAN,
    DEFAULT_CAMPAIGN_REPORT,
    DEFAULT_COMPLETION_STATUS,
    DEFAULT_COMPLETION_STATUS_REPORT,
    DEFAULT_BROKER_INSTRUMENTS,
    DEFAULT_BROKER_INSTRUMENTS_REPORT,
    DEFAULT_PAIR_CHARTS,
    DEFAULT_PAIR_CHARTS_REPORT,
    DEFAULT_CONTEXT_ASSET_CHARTS,
    DEFAULT_CONTEXT_ASSET_CHARTS_REPORT,
    DEFAULT_COVERAGE_OPTIMIZATION,
    DEFAULT_COVERAGE_OPTIMIZATION_REPORT,
    DEFAULT_DAILY_TARGET_REPORT,
    DEFAULT_DAILY_TARGET_STATE,
    DEFAULT_DRY_RUN_CERTIFICATION,
    DEFAULT_DRY_RUN_CERTIFICATION_REPORT,
    DEFAULT_EXECUTION_LEDGER_DB,
    DEFAULT_EXECUTION_LEDGER_REPORT,
    DEFAULT_VERIFICATION_LEDGER,
    DEFAULT_VERIFICATION_LEDGER_REPORT,
    DEFAULT_EXECUTION_REPLAY,
    DEFAULT_EXECUTION_REPLAY_REPORT,
    DEFAULT_GPT_TRADER_DECISION,
    DEFAULT_GPT_TRADER_DECISION_REPORT,
    DEFAULT_HISTORY_DB,
    DEFAULT_IMPORT_REPORT,
    DEFAULT_MARKET_STATUS,
    DEFAULT_MARKET_STATUS_REPORT,
    DEFAULT_LIVE_ORDER_REQUEST,
    DEFAULT_LIVE_ORDER_STAGE_REPORT,
    DEFAULT_LEGACY_ARCHIVE,
    DEFAULT_MARKET_STORY_PROFILE,
    DEFAULT_MARKET_STORY_REPORT,
    DEFAULT_ORDER_INTENT_REPORT,
    DEFAULT_ORDER_INTENTS,
    DEFAULT_POSITION_EXECUTION,
    DEFAULT_POSITION_MANAGEMENT,
    DEFAULT_POSITION_MANAGEMENT_REPORT,
    DEFAULT_PROFIT_PARTIAL_CLOSE,
    DEFAULT_PROFIT_PARTIAL_CLOSE_REPORT,
    DEFAULT_PROFIT_PARTIAL_CLOSE_STATE,
    DEFAULT_POST_TRADE_LEARNING,
    DEFAULT_POST_TRADE_LEARNING_REPORT,
    DEFAULT_RECEIPT_PROMOTION_REPORT,
    DEFAULT_REPLAY_BACKTEST,
    DEFAULT_REPLAY_BACKTEST_REPORT,
    DEFAULT_STRATEGY_PROFILE,
    DEFAULT_STRATEGY_REPORT,
    DEFAULT_TRADER_SETTINGS,
    DEFAULT_TRADER_OVERRIDES,
    DEFAULT_TRADER_DECISION,
    DEFAULT_CROSS_ASSET_SNAPSHOT,
    DEFAULT_CROSS_ASSET_REPORT,
    DEFAULT_FLOW_SNAPSHOT,
    DEFAULT_FLOW_REPORT,
    DEFAULT_CURRENCY_STRENGTH,
    DEFAULT_CURRENCY_STRENGTH_REPORT,
    DEFAULT_LEVELS_SNAPSHOT,
    DEFAULT_LEVELS_REPORT,
    DEFAULT_MARKET_CONTEXT_MATRIX,
    DEFAULT_MARKET_CONTEXT_MATRIX_REPORT,
    DEFAULT_LEARNING_AUDIT,
    DEFAULT_LEARNING_AUDIT_REPORT,
    DEFAULT_FORECAST_HISTORY,
    DEFAULT_PROJECTION_LEDGER,
    DEFAULT_ENTRY_THESIS_LEDGER,
    DEFAULT_MEMORY_HEALTH,
    DEFAULT_MEMORY_HEALTH_REPORT,
    DEFAULT_SELF_IMPROVEMENT_AUDIT,
    DEFAULT_SELF_IMPROVEMENT_AUDIT_REPORT,
    DEFAULT_CALENDAR_SNAPSHOT,
    DEFAULT_CALENDAR_REPORT,
    DEFAULT_COT_SNAPSHOT,
    DEFAULT_COT_REPORT,
    DEFAULT_CODEX_TRADER_DECISION_RESPONSE,
    DEFAULT_OPTION_SKEW,
    DEFAULT_OPTION_SKEW_REPORT,
    DEFAULT_NEWS_SNAPSHOT,
    DEFAULT_NEWS_DIGEST,
    DEFAULT_NEWS_FLOW_LOG,
    DEFAULT_NEWS_HEALTH,
    DEFAULT_NEWS_HEALTH_REPORT,
    DEFAULT_OUTCOME_MART,
    DEFAULT_OUTCOME_MART_REPORT,
)
from quant_rabbit.gpt_trader import DEFAULT_GPT_MAX_LANES, GPTTraderBrain, StaticTraderProvider
from quant_rabbit.instruments import DEFAULT_CONTEXT_ASSETS, DEFAULT_TRADER_PAIRS_ARG
from quant_rabbit.replay import ReplayBacktester
from quant_rabbit.risk import RiskEngine, RiskPolicy, resolve_max_loss_jpy
from quant_rabbit.snapshot_json import snapshot_order_raw, snapshot_payload_order_raw
from quant_rabbit.strategy.ensemble import CampaignPlanner
from quant_rabbit.strategy.intent_generator import IntentGenerator
from quant_rabbit.strategy.market_story import MarketStoryMiner
from quant_rabbit.strategy.miner import StrategyMiner
from quant_rabbit.strategy.profile import StrategyProfile
from quant_rabbit.strategy.receipt_promotion import ReceiptPromoter
from quant_rabbit.target import DailyTargetLedger
from quant_rabbit.trader_prompts import route_trader_prompts

DEFAULT_RUNTIME_MARKET_STORY_REPORT = DEFAULT_MARKET_STORY_PROFILE.with_name("market_story_report.md")


# SL-free production defaults. `scripts/run-autotrade-live.sh` exports the same
# values, but a direct `python3 -m quant_rabbit.cli autotrade-cycle --send`
# invocation (e.g. an ad-hoc smoke run, a manual reproducer, an alternate
# scheduler) used to skip the wrapper and reattach tight broker SLs from
# `intent.sl`. On 2026-05-07 23:24 UTC that path placed Trade 470395 with a
# 5.3-pip SL — exactly the noise-range stop the operator directive 「SLい
# らない」 forbids. The bootstrap only fires when QR_LIVE_ENABLED=1 so unit
# tests (which exercise `cli.main(["autotrade-cycle", ...])` for argument
# validation) do not pollute the rest of the pytest process. Within a live
# session, only an explicit env override can revert any single knob.
_SL_FREE_RUNTIME_DEFAULTS: dict[str, str] = {
    "QR_TRADER_DISABLE_SL_REPAIR": "1",
    "QR_GEOMETRY_ATR_MULT": "5.0",
    "QR_GEOMETRY_SPREAD_FLOOR_MULT": "12.0",
    "QR_MAX_PORTFOLIO_POSITIONS": "10",
    # NAV-pct sizing replaces the legacy fixed unit count so position size
    # auto-scales with equity (user 2026-05-08「BaseUnitを決めると、資産が
    # 増えたときに追従できないよ。％で決めないといけなくない？」). 30% per
    # position lands ≈10000u for EUR_USD at NAV ~227k — three concurrent
    # positions reach ~90% margin utilization, just inside the 92% cap.
    "QR_TRADER_POSITION_NAV_PCT": "30",
    "QR_TRADER_BASE_UNITS": "3000",
    # A losing REVIEW_EXIT must not become an automatic broker close in the
    # SL-free runtime. Loss-side CLOSE requires the explicit gpt_trader Gate A/B
    # path; PositionManager keeps legacy/no-ledger reviews advisory. The only
    # REVIEW_EXIT exception is a next-generation trader position with an entry
    # thesis ledger record and hard structural loss-cut evidence. Profit-only
    # TAKE_PROFIT_MARKET is a separate harvest action, not a loss close.
    "QR_DISABLE_AUTO_CLOSE": "1",
    # SL-free invariant restored 2026-05-13 (feedback_broker_sl_noise_hunt.md).
    # f382dc6 F shipped `QR_NEW_ENTRY_INITIAL_SL=1` to attach a broker SL
    # to every new entry "for noise resistance". In practice the
    # ATR×1.5 floor + session multiplier left the SL inside the M15
    # noise band on thin sessions; combined with the f382dc6 H trailing
    # path (which itself tightened the broker SL toward the next BOS
    # event), the live cycle harvested its own broker SL three times in
    # 16 minutes on 2026-05-13 (AUD_JPY 470989 -52 JPY at 4 pip; 470997
    # -546 JPY at 42 pip; total -1,053 JPY of pure noise loss). The
    # operator directive 「SLいらない」 (`feedback_no_tight_sl_thin_market.md`)
    # is absolute — broker-side SL on new entries is OFF by default,
    # and the trailing path is disabled. Both knobs can still be
    # explicitly turned ON via env if a future setup proves it needs
    # broker-side SL (e.g. tomorrow's NFP), but the live default must
    # not silently re-introduce noise-band stops.
    "QR_NEW_ENTRY_INITIAL_SL": "0",
    "QR_DISABLE_TRAILING_SL": "1",
    # Live fresh entries must carry a current executable pair forecast. This
    # prevents campaign/range coverage lanes from becoming broker-fillable when
    # the prediction layer is stale, missing, or too weak to justify a side.
    "QR_REQUIRE_FORECAST_FOR_LIVE": "1",
    # A prediction that is not logged, projection-tracked, and reconciled with
    # broker transaction truth is not auditable enough for live order flow.
    "QR_REQUIRE_TELEMETRY_FOR_LIVE": "1",
}


def _bootstrap_sl_free_defaults() -> None:
    applied: list[str] = []
    for key, value in _SL_FREE_RUNTIME_DEFAULTS.items():
        if key not in os.environ:
            os.environ[key] = value
            applied.append(key)
    if applied:
        sys.stderr.write(
            "[qr-vnext] applied SL-free runtime defaults for "
            + ",".join(applied)
            + " (override with explicit env to revert)\n"
        )


def _forecast_emission_time_from_snapshot(snapshot_payload: dict[str, Any]) -> datetime | None:
    raw = snapshot_payload.get("fetched_at_utc")
    if raw is None and isinstance(snapshot_payload.get("account"), dict):
        raw = snapshot_payload["account"].get("fetched_at_utc")
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _refresh_current_forecast_history(
    *,
    snapshot_payload: dict[str, Any],
    pair_charts_path: Path,
    pairs: Iterable[str],
    data_root: Path,
    cycle_source: str,
) -> dict[str, Any]:
    """Persist current pair forecasts for position-management-only cycles.

    TraderBrain records forecasts while scoring fresh-entry lanes. When an
    account is already occupied and the cycle routes straight into position
    management, no fresh lane may be scored; persistence then goes stale and
    cannot detect that a held position lost its directional edge.
    """
    unique_pairs = sorted({str(pair) for pair in pairs if str(pair or "").strip()})
    if not unique_pairs:
        return {"recorded": 0, "skipped": {}, "cycle_id": None}
    try:
        pair_charts_payload = json.loads(pair_charts_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {
            "recorded": 0,
            "skipped": {pair: "pair_charts_unavailable" for pair in unique_pairs},
            "cycle_id": None,
        }
    try:
        from quant_rabbit.strategy.forecast_persistence_tracker import record_forecast
        from quant_rabbit.strategy.intent_generator import (
            _forecast_seed_for_pair,
            _forecast_seed_regime_label,
            _load_pair_charts,
            _quote_fresh_for_forecast_seed_telemetry,
            _snapshot_from_json,
        )
        from quant_rabbit.strategy.projection_ledger import (
            projection_telemetry_market_open,
            record_directional_forecast,
        )
    except Exception as exc:
        return {
            "recorded": 0,
            "skipped": {pair: f"forecast_import_failed:{exc.__class__.__name__}" for pair in unique_pairs},
            "cycle_id": None,
        }
    charts = _load_pair_charts(pair_charts_path)
    if not charts:
        return {
            "recorded": 0,
            "skipped": {pair: "pair_charts_empty" for pair in unique_pairs},
            "cycle_id": None,
        }
    try:
        snapshot = _snapshot_from_json(snapshot_payload)
    except Exception as exc:
        return {
            "recorded": 0,
            "skipped": {pair: f"snapshot_parse_failed:{exc.__class__.__name__}" for pair in unique_pairs},
            "cycle_id": None,
        }
    emission_time = _forecast_emission_time_from_snapshot(snapshot_payload)
    projection_market_open = projection_telemetry_market_open(emission_time)
    cycle_id = _forecast_refresh_cycle_id(
        snapshot_payload=snapshot_payload,
        pair_charts_payload=pair_charts_payload,
        cycle_source=cycle_source,
    )
    recorded: dict[str, dict[str, Any]] = {}
    skipped: dict[str, str] = {}
    projection_recorded = 0
    projection_skipped: dict[str, str] = {}
    for pair in unique_pairs:
        if _forecast_history_has_cycle_pair(data_root, pair, cycle_id):
            skipped[pair] = "already_recorded_for_cycle"
            continue
        if pair not in charts:
            skipped[pair] = "pair_chart_missing"
            continue
        quote = snapshot.quotes.get(pair)
        if quote is None:
            skipped[pair] = "quote_missing"
            continue
        forecast = _forecast_seed_for_pair(pair, charts, snapshot)
        if forecast is None:
            skipped[pair] = "forecast_unavailable"
            continue
        raw_chart = charts[pair].get("__raw_chart") if isinstance(charts[pair], dict) else None
        regime = _forecast_seed_regime_label(raw_chart) if isinstance(raw_chart, dict) else None
        if not projection_market_open:
            skipped[pair] = "market_closed_at_forecast_emission"
            continue
        if not _quote_fresh_for_forecast_seed_telemetry(
            getattr(quote, "timestamp_utc", None),
            validation_time_utc=getattr(snapshot, "fetched_at_utc", None),
        ):
            skipped[pair] = "stale_quote_for_forecast_telemetry"
            continue
        record_forecast(forecast, data_root=data_root, cycle_id=cycle_id, now=emission_time)
        try:
            projection_recorded += record_directional_forecast(
                forecast,
                pair=pair,
                current_price=float(quote.mid),
                data_root=data_root,
                regime_at_emission=regime,
                cycle_id=cycle_id,
                now=emission_time,
            )
        except Exception as exc:
            projection_skipped[pair] = f"projection_record_failed:{exc.__class__.__name__}"
        recorded[pair] = {
            "direction": getattr(forecast, "direction", "UNCLEAR"),
            "confidence": float(getattr(forecast, "confidence", 0.0) or 0.0),
        }
    return {
        "recorded": len(recorded),
        "projection_recorded": projection_recorded,
        "forecasts": recorded,
        "skipped": skipped,
        "projection_skipped": projection_skipped,
        "cycle_id": cycle_id,
    }


def _forecast_refresh_cycle_id(
    *,
    snapshot_payload: dict[str, Any],
    pair_charts_payload: dict[str, Any],
    cycle_source: str,
) -> str:
    fetched = snapshot_payload.get("fetched_at_utc") or (
        snapshot_payload.get("account") or {}
    ).get("fetched_at_utc") or "snapshot-unknown"
    charts_generated = pair_charts_payload.get("generated_at_utc") or "charts-unknown"
    return f"{cycle_source}:{fetched}:{charts_generated}"


def _pre_entry_execution_ledger_sync_if_required(
    *,
    telemetry_required: bool,
    snapshot_path: Path | None,
) -> dict[str, Any] | None:
    """Keep direct generate-intents invocations audit-current.

    `autotrade-cycle` already syncs the execution ledger before intent
    pricing. The standalone CLI used by operators and diagnostics can refresh
    a broker snapshot and then price intents without that audit pass, which
    makes every otherwise-valid live lane fail on stale transaction truth.
    This is read-only against OANDA and only writes the append-only local
    ledger/report; missing credentials preserve offline dry-run behavior.
    """
    if not telemetry_required:
        return None
    if _running_under_test_harness() and os.environ.get("QR_LIVE_ENABLED") != "1":
        return None
    if snapshot_path is None or not snapshot_path.exists():
        return None
    try:
        summary = ExecutionLedger(
            db_path=DEFAULT_EXECUTION_LEDGER_DB,
            report_path=DEFAULT_EXECUTION_LEDGER_REPORT,
        ).sync_oanda_transactions(OandaReadOnlyClient())
    except (RuntimeError, OSError, sqlite3.Error, json.JSONDecodeError, ValueError) as exc:
        return {
            "status": "SYNC_FAILED",
            "error": str(exc),
        }
    return {
        "status": summary.status,
        "transactions_seen": summary.transactions_seen,
        "transactions_inserted": summary.transactions_inserted,
        "events_inserted": summary.events_inserted,
        "last_transaction_id": summary.last_transaction_id,
        "baseline_transaction_id": summary.baseline_transaction_id,
    }


def _partial_close_receipt_actions(
    results: list[dict[str, Any]],
    *,
    management_action: str,
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for result in results:
        trade_id = str(result.get("trade_id") or "")
        close_units = result.get("close_units")
        request = None
        if trade_id and close_units:
            request = {
                "type": "CLOSE",
                "trade_id": trade_id,
                "units": str(close_units),
            }
        issues = []
        error = result.get("error")
        if error:
            code = str(error).split(":", 1)[0].strip() or "PARTIAL_CLOSE_ERROR"
            issues.append(
                {
                    "severity": "BLOCK",
                    "code": code,
                    "message": str(error),
                }
            )
        actions.append(
            {
                "trade_id": trade_id,
                "pair": str(result.get("pair") or ""),
                "owner": "trader",
                "management_action": management_action,
                "request": request,
                "issues": issues,
                "sent": bool(result.get("sent")),
                "response": result.get("response"),
                "reasons": [str(result.get("rationale") or management_action)],
            }
        )
    return actions


def _record_position_execution_receipt(
    *,
    output_path: Path,
    ledger_db_path: Path,
    ledger_report_path: Path,
) -> dict[str, Any]:
    try:
        summary = ExecutionLedger(
            db_path=ledger_db_path,
            report_path=ledger_report_path,
        ).record_gateway_receipt(kind="position_execution", receipt_path=output_path)
    except (OSError, sqlite3.Error, json.JSONDecodeError, ValueError) as exc:
        return {"status": "RECORD_FAILED", "error": str(exc)}
    return {
        "status": summary.status,
        "gateway_receipts_inserted": summary.gateway_receipts_inserted,
        "events_inserted": summary.events_inserted,
    }


def _pre_entry_projection_verification_if_required(
    *,
    telemetry_required: bool,
    snapshot_path: Path | None,
    pair_charts_path: Path = DEFAULT_PAIR_CHARTS,
) -> dict[str, Any] | None:
    """Resolve expired projection telemetry before pricing live intents.

    `generate-intents` is the first place that can decide whether a lane is
    LIVE_READY. If the scheduled model skips the playbook's explicit
    `verify-projections` step, every otherwise-valid live lane receives the
    expired-PENDING telemetry blocker. This read-only broker preflight mirrors
    the playbook so model omissions do not turn into systematic no-trade cycles.
    """
    if not telemetry_required:
        return None
    if _running_under_test_harness() and os.environ.get("QR_LIVE_ENABLED") != "1":
        return None
    if snapshot_path is None or not snapshot_path.exists():
        return None
    try:
        from quant_rabbit.analysis.candles import fetch_candles_via_client
        from quant_rabbit.strategy.projection_ledger import (
            load_ledger,
            retryable_truth_timeout_pairs,
            verify_pending,
        )

        now = datetime.now(timezone.utc)
        entries = load_ledger(ROOT / "data")
        expired_pairs: set[str] = set()
        pending_pairs: set[str] = set()
        for entry in entries:
            if getattr(entry, "resolution_status", None) != "PENDING":
                continue
            pair = str(getattr(entry, "pair", "") or "")
            if pair:
                pending_pairs.add(pair)
            try:
                emitted_at = datetime.fromisoformat(
                    str(getattr(entry, "timestamp_emitted_utc", "")).replace("Z", "+00:00")
                )
                window_min = float(getattr(entry, "resolution_window_min", 0) or 0)
            except (TypeError, ValueError):
                continue
            if (now - emitted_at).total_seconds() / 60.0 >= window_min and pair:
                expired_pairs.add(pair)
        retry_timeout_pairs = retryable_truth_timeout_pairs(entries)
        verification_pairs = pending_pairs | retry_timeout_pairs
        if not expired_pairs and not retry_timeout_pairs:
            return {
                "status": "NO_EXPIRED_PENDING",
                "expired_pending_pairs": 0,
                "retryable_timeout_pairs": 0,
            }

        snapshot_payload = json.loads(snapshot_path.read_text())
        quotes_by_pair: dict[str, dict[str, float]] = {}
        for pair_key, quote_data in (snapshot_payload.get("quotes") or {}).items():
            if isinstance(quote_data, dict):
                try:
                    quotes_by_pair[str(pair_key)] = {
                        "bid": float(quote_data.get("bid", 0)),
                        "ask": float(quote_data.get("ask", 0)),
                    }
                except (TypeError, ValueError):
                    continue

        atr_pips_by_pair: dict[str, float] = {}
        if pair_charts_path.exists():
            pair_charts_payload = json.loads(pair_charts_path.read_text())
            for chart in pair_charts_payload.get("charts", []) or []:
                if not isinstance(chart, dict):
                    continue
                pair = str(chart.get("pair") or "")
                if not pair:
                    continue
                for view in chart.get("views", []) or []:
                    if not isinstance(view, dict) or view.get("granularity") != "H1":
                        continue
                    try:
                        atr = float((view.get("indicators") or {}).get("atr_pips"))
                    except (TypeError, ValueError):
                        continue
                    if atr > 0:
                        atr_pips_by_pair[pair] = atr
                        break

        candles_by_pair: dict[str, dict[str, list[Any]]] | None = None
        candle_counts: dict[str, int] = {}
        candle_granularity_counts: dict[str, dict[str, int]] = {}
        candle_errors: dict[str, str] = {}
        m1_count = int(os.environ.get("QR_PROJECTION_VERIFY_M1_COUNT", "1500"))
        m5_count = int(os.environ.get("QR_PROJECTION_VERIFY_M5_COUNT", "1500"))
        if verification_pairs and (m1_count > 0 or m5_count > 0):
            candles_by_pair = {}
            try:
                client = OandaReadOnlyClient()
            except Exception as exc:
                candle_errors["_client"] = f"{type(exc).__name__}: {str(exc)[:160]}"
                candles_by_pair = None
            else:
                for pair in sorted(verification_pairs):
                    pair_candles: dict[str, list[Any]] = {}
                    per_granularity: dict[str, int] = {}
                    for granularity, count in (("M1", m1_count), ("M5", m5_count)):
                        if count <= 0:
                            continue
                        try:
                            candles = list(fetch_candles_via_client(client, pair, granularity, count=count))
                        except Exception as exc:
                            candle_errors[f"{pair}:{granularity}"] = f"{type(exc).__name__}: {str(exc)[:160]}"
                            continue
                        pair_candles[granularity] = candles
                        per_granularity[granularity] = len(candles)
                    if not pair_candles:
                        continue
                    candles_by_pair[pair] = pair_candles
                    candle_counts[pair] = sum(per_granularity.values())
                    candle_granularity_counts[pair] = per_granularity

        counts = verify_pending(
            ROOT / "data",
            quotes_by_pair=quotes_by_pair,
            atr_pips_by_pair=atr_pips_by_pair,
            candles_by_pair=candles_by_pair,
        )
    except (OSError, sqlite3.Error, json.JSONDecodeError, ValueError) as exc:
        return {"status": "VERIFY_FAILED", "error": str(exc)}
    return {
        "status": "OK",
        "expired_pending_pairs": len(expired_pairs),
        "pending_pairs": len(pending_pairs),
        "retryable_timeout_pairs": len(retry_timeout_pairs),
        "resolution_counts": counts,
        "candle_counts": candle_counts,
        "candle_granularity_counts": candle_granularity_counts,
        "candle_errors": candle_errors,
    }


def _forecast_history_has_cycle_pair(data_root: Path, pair: str, cycle_id: str) -> bool:
    path = data_root / "forecast_history.jsonl"
    if not path.exists():
        return False
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if item.get("pair") == pair and item.get("cycle_id") == cycle_id:
                    return True
    except OSError:
        return False
    return False


def _trader_position_pairs(positions: Iterable[Any]) -> list[str]:
    pairs: list[str] = []
    for position in positions:
        owner = getattr(position, "owner", None)
        owner_value = owner.value if hasattr(owner, "value") else str(owner or "")
        if owner_value.lower() != Owner.TRADER.value:
            continue
        pair = str(getattr(position, "pair", "") or "")
        if pair:
            pairs.append(pair)
    return pairs


# Subcommands that read or write against real broker truth (or feed the
# risk-geometry path that does). These always bootstrap the SL-free defaults
# even when QR_LIVE_ENABLED is unset, because the routine-driven SKILL flow
# invokes them outside `scripts/run-autotrade-live.sh`. Anything outside this
# set (status reports, certify-dry-run, import-legacy, mining tools, etc.)
# stays opt-in via QR_LIVE_ENABLED=1 so unit tests can exercise it cleanly.
_LIVE_RUNTIME_COMMANDS: frozenset[str] = frozenset(
    {
        "autotrade-cycle",
        "gpt-trader-decision",
        "stage-live-order",
        "generate-intents",
        # The router is the first branch point in the scheduled SKILL flow.
        # It must see the same SL-free defaults as generate-intents and
        # gpt-trader-decision, otherwise a TP-less runner sends the operator
        # back to position repair before the refreshed entry packet is used.
        "trader-prompt-route",
        # daily-target-state reads each open trader-owned position and
        # decides whether it is `protected`. Under SL-free mode the
        # absence of a stopLossOrder is intentional (per `target.py`
        # `_position_risk` lines 416-423) but that branch only fires
        # when `QR_TRADER_DISABLE_SL_REPAIR=1` is set in the
        # subprocess env. Without the SL-free bootstrap the command
        # produces `missing=['SL']` and `protected=False` for every
        # SL-free trader-owned position, which then bleeds into
        # daily_target_state.json + position_management.json and
        # confuses the operator into proposing redundant PROTECT
        # repairs. Adding the command here ensures every routine
        # invocation sees the same SL-free env that autotrade-cycle
        # sees, so the protected flag is computed consistently
        # across both paths.
        "daily-target-state",
        # Position management is a dry-run sidecar here, but it is part of
        # existing-exposure live state. It must share the SL-free defaults so
        # routine refreshes do not turn intentional TP-only positions into
        # stale repair requirements before GPT verification.
        "position-management",
        "profit-partial-close",
        # completion-status is an audit command, but it classifies open
        # trader exposure. Under the SL-free runtime, TP-only trader
        # positions are intentional; without this bootstrap a standalone
        # audit misreports them as unprotected broker exposure.
        "completion-status",
        # memory-health audits the same live routing artifacts before the
        # prompt branch can move into entry/verify. It does not send orders,
        # but it must classify SL-free live state under the same defaults.
        "memory-health",
    }
)


_AUTOTRADE_EXIT_ZERO_STATUSES: frozenset[str] = frozenset(
    {
        "SENT",
        "STAGED",
        "MONITOR_ONLY_EXPOSURE_OPEN",
        "CANCELED_CONTAMINATED_PENDING",
        "POSITION_ACTION_SENT",
        "POSITION_ACTION_STAGED",
        "POSITION_ACTION_BLOCKED",
        "NO_LIVE_READY_INTENT",
        "NO_TRADE",
        "GPT_WAIT",
        "GPT_REQUEST_EVIDENCE",
        "GPT_REJECTED",
        "GPT_DECISION_NOT_PREFILTERED",
        "GPT_PROTECT",
        "GPT_TIGHTEN_SL",
        "TARGET_REACHED_PROTECT",
    }
)


def _running_under_test_harness() -> bool:
    """True when the cli is invoked from pytest / unittest discovery.

    The SL-free env bootstrap is process-global; firing it from
    `_LIVE_RUNTIME_COMMANDS` during a unit-test run pollutes every
    subsequent test in the same pytest/unittest process (intent_generator
    BLOCK vs WARN, risk engine cap derivation, position manager protection
    flow, etc. all branch on these env vars). The wrapper-driven live
    path still sets QR_LIVE_ENABLED=1 explicitly, so production routines
    bootstrap as before; tests can also force-enable via QR_LIVE_ENABLED=1
    when they want the SL-free path. Detection is conservative: it must
    catch both pytest (sets PYTEST_CURRENT_TEST per active test) and the
    `python -m unittest` entry point (loads `unittest.__main__`).
    """
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return True
    if "unittest" in sys.modules and any(
        name.startswith("unittest.") and name.endswith(("__main__", "main"))
        for name in sys.modules
    ):
        return True
    argv0 = (sys.argv[0] or "").lower()
    return "unittest" in argv0 or "pytest" in argv0


def _refresh_market_story_if_news_is_newer(
    *,
    profile_path: Path,
    report_path: Path,
    news_dir: Path,
) -> bool:
    """Keep the runtime story profile aligned with the out-of-band news digest.

    The hourly news task writes curated headlines into ``logs/``; the trader
    reads ``data/market_story_profile.json``.  A symlinked digest alone is not
    enough if this derived profile is older than the digest.  Refresh before
    intent generation so every decision packet is built from current news.
    """
    news_paths = [news_dir / DEFAULT_NEWS_DIGEST.name, news_dir / DEFAULT_NEWS_FLOW_LOG.name]
    existing_news = [path for path in news_paths if path.exists()]
    profile_missing = not profile_path.exists()
    if not profile_missing and not existing_news:
        return False
    if not profile_missing and existing_news:
        latest_news_mtime = max(path.stat().st_mtime for path in existing_news)
        if latest_news_mtime <= profile_path.stat().st_mtime:
            return False

    MarketStoryMiner(
        report_path=report_path,
        profile_path=profile_path,
        news_root=news_dir,
    ).run()
    sys.stderr.write(
        "[qr-vnext] refreshed market_story_profile from news artifacts "
        f"(profile={profile_path}, news_dir={news_dir})\n"
    )
    return True


def _auto_refresh_market_evidence_if_required(
    *,
    label: str,
    reuse_market_artifacts: bool = False,
    market_context_matrix_path: Path = DEFAULT_MARKET_CONTEXT_MATRIX,
    context_asset_charts_path: Path = DEFAULT_CONTEXT_ASSET_CHARTS,
    broker_instruments_path: Path = DEFAULT_BROKER_INSTRUMENTS,
) -> dict[str, Any]:
    """Refresh the full technical/macro packet before runtime lane pricing.

    The playbook has explicit refresh commands, but the live wrapper ultimately
    invokes `autotrade-cycle`. A direct cycle must not quietly price intents
    from stale pair charts or a missing market-context matrix.
    """

    if reuse_market_artifacts:
        _validate_reusable_market_evidence(
            label=label,
            market_context_matrix_path=market_context_matrix_path,
            context_asset_charts_path=context_asset_charts_path,
            broker_instruments_path=broker_instruments_path,
        )
        return {"status": "SKIPPED", "reason": "reuse_market_artifacts"}
    if os.environ.get("QR_DISABLE_MARKET_EVIDENCE_AUTO_REFRESH", "").strip() in {
        "1",
        "true",
        "TRUE",
        "yes",
        "YES",
    }:
        return {"status": "SKIPPED", "reason": "QR_DISABLE_MARKET_EVIDENCE_AUTO_REFRESH"}
    if _running_under_test_harness():
        return {"status": "SKIPPED", "reason": "test_harness"}

    pairs = tuple(part.strip().upper() for part in DEFAULT_TRADER_PAIRS_ARG.split(",") if part.strip())
    candle_count = int(os.environ.get("QR_MARKET_EVIDENCE_CANDLE_COUNT", "200") or "200")
    try:
        client = OandaReadOnlyClient()

        from quant_rabbit.analysis.calendar import build_calendar_snapshot
        from quant_rabbit.analysis.chart_reader import build_pair_chart
        from quant_rabbit.analysis.context_assets import (
            build_context_asset_charts,
            write_context_asset_charts_report,
        )
        from quant_rabbit.analysis.cot import build_cot_snapshot
        from quant_rabbit.analysis.cross_asset import (
            DEFAULT_CROSS_ASSET_INSTRUMENTS,
            build_cross_asset_snapshot,
        )
        from quant_rabbit.analysis.flow import build_flow_snapshot
        from quant_rabbit.analysis.levels import build_levels_snapshot
        from quant_rabbit.analysis.market_context_matrix import (
            build_market_context_matrix,
            write_market_context_matrix_report,
        )
        from quant_rabbit.analysis.options import build_option_skew_snapshot
        from quant_rabbit.analysis.score_momentum import attach_score_momentum
        from quant_rabbit.analysis.strength import build_strength_snapshot

        broker_instruments = _broker_instruments_snapshot(
            client=client,
            trader_pairs=pairs,
            context_assets=DEFAULT_CONTEXT_ASSETS,
        )
        _write_json(DEFAULT_BROKER_INSTRUMENTS, broker_instruments)
        _write_broker_instruments_report(broker_instruments, DEFAULT_BROKER_INSTRUMENTS_REPORT)

        generated_at = datetime.now(timezone.utc).isoformat()
        charts = [
            build_pair_chart(pair, client=client, timeframes=DEFAULT_PAIR_CHART_TIMEFRAMES, count=candle_count).to_dict()
            for pair in pairs
        ]
        previous_pair_charts = None
        if DEFAULT_PAIR_CHARTS.exists():
            try:
                previous_pair_charts = json.loads(DEFAULT_PAIR_CHARTS.read_text())
            except (OSError, json.JSONDecodeError, ValueError):
                previous_pair_charts = None
        attach_score_momentum(charts, previous_pair_charts, generated_at)
        charts.sort(key=lambda c: max(c["long_score"], c["short_score"]), reverse=True)
        pair_payload = {
            "generated_at_utc": generated_at,
            "timeframes": list(DEFAULT_PAIR_CHART_TIMEFRAMES),
            "candle_count": candle_count,
            "charts": charts,
        }
        _write_json(DEFAULT_PAIR_CHARTS, pair_payload)

        context_asset_charts = build_context_asset_charts(
            client=client,
            instruments=DEFAULT_CONTEXT_ASSETS,
            timeframes=DEFAULT_PAIR_CHART_TIMEFRAMES,
            count=candle_count,
        )
        _write_json(DEFAULT_CONTEXT_ASSET_CHARTS, context_asset_charts)
        write_context_asset_charts_report(context_asset_charts, DEFAULT_CONTEXT_ASSET_CHARTS_REPORT)

        cross_asset = build_cross_asset_snapshot(
            client=client,
            instruments=DEFAULT_CROSS_ASSET_INSTRUMENTS,
            correlation_pairs=pairs,
            count=candle_count,
        )
        _write_json(DEFAULT_CROSS_ASSET_SNAPSHOT, cross_asset.to_dict())

        flow = build_flow_snapshot(
            client=client,
            pairs=pairs,
            include_books=os.environ.get("QR_FLOW_INCLUDE_BOOKS", "").strip().lower() in {"1", "true", "yes"},
        )
        _write_json(DEFAULT_FLOW_SNAPSHOT, flow.to_dict())

        strength = build_strength_snapshot(client=client)
        _write_json(DEFAULT_CURRENCY_STRENGTH, strength.to_dict())

        levels = build_levels_snapshot(client=client, pairs=pairs)
        _write_json(DEFAULT_LEVELS_SNAPSHOT, levels.to_dict())

        calendar = build_calendar_snapshot(pairs=pairs)
        _write_json(DEFAULT_CALENDAR_SNAPSHOT, calendar.to_dict())

        cot = build_cot_snapshot(fetch=True)
        _write_json(DEFAULT_COT_SNAPSHOT, cot.to_dict())

        option_skew = build_option_skew_snapshot(pairs=pairs)
        _write_json(DEFAULT_OPTION_SKEW, option_skew.to_dict())

        matrix = build_market_context_matrix(
            pair_charts_path=DEFAULT_PAIR_CHARTS,
            cross_asset_path=DEFAULT_CROSS_ASSET_SNAPSHOT,
            flow_path=DEFAULT_FLOW_SNAPSHOT,
            currency_strength_path=DEFAULT_CURRENCY_STRENGTH,
            levels_path=DEFAULT_LEVELS_SNAPSHOT,
            calendar_path=DEFAULT_CALENDAR_SNAPSHOT,
            cot_path=DEFAULT_COT_SNAPSHOT,
            option_skew_path=DEFAULT_OPTION_SKEW,
            context_asset_charts_path=DEFAULT_CONTEXT_ASSET_CHARTS,
        )
        _write_json(market_context_matrix_path, matrix)
        write_market_context_matrix_report(matrix, DEFAULT_MARKET_CONTEXT_MATRIX_REPORT)
    except Exception as exc:
        raise RuntimeError(f"{label} market evidence refresh failed: {exc}") from exc

    side_rows = [
        reading
        for side_map in (matrix.get("pairs") or {}).values()
        if isinstance(side_map, dict)
        for reading in side_map.values()
        if isinstance(reading, dict)
    ]
    issues: list[str] = []
    for payload in (
        cross_asset.to_dict(),
        context_asset_charts,
        broker_instruments,
        flow.to_dict(),
        strength.to_dict(),
        levels.to_dict(),
        calendar.to_dict(),
        cot.to_dict(),
        option_skew.to_dict(),
        matrix,
    ):
        issues.extend(str(item) for item in payload.get("issues", []) or [] if str(item).strip())
    summary = {
        "status": "REFRESHED",
        "pairs": len(pairs),
        "pair_charts_generated_at_utc": generated_at,
        "market_context_pairs": len(matrix.get("pairs") or {}),
        "market_context_matrix_path": str(market_context_matrix_path),
        "context_assets": len(context_asset_charts.get("charts") or []),
        "context_assets_tradeable": len(broker_instruments.get("context_assets_tradeable") or []),
        "context_assets_not_tradeable": len(broker_instruments.get("context_assets_not_tradeable") or []),
        "supports": sum(len(row.get("supports") or []) for row in side_rows),
        "rejects": sum(len(row.get("rejects") or []) for row in side_rows),
        "warnings": sum(len(row.get("warnings") or []) for row in side_rows),
        "missing": sum(len(row.get("missing") or []) for row in side_rows),
        "issues": issues[:12],
    }
    sys.stderr.write(
        "[qr-vnext] auto-refreshed market evidence "
        + json.dumps(summary, ensure_ascii=False, sort_keys=True)
        + "\n"
    )
    return summary


def _validate_reusable_market_evidence(
    *,
    label: str,
    market_context_matrix_path: Path,
    context_asset_charts_path: Path,
    broker_instruments_path: Path,
) -> None:
    """Refuse reused live artifacts when candidate discovery lacks matrix context."""

    if not market_context_matrix_path.exists():
        raise RuntimeError(
            f"{label} cannot reuse market artifacts: missing required market evidence artifact "
            f"market_context_matrix={market_context_matrix_path}; run without --reuse-market-artifacts "
            "or refresh market context before live candidate discovery"
        )
    try:
        payload = json.loads(market_context_matrix_path.read_text())
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(
            f"{label} cannot reuse market artifacts: unreadable market_context_matrix="
            f"{market_context_matrix_path}: {exc}; refresh market context before live candidate discovery"
        ) from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("pairs"), dict) or not payload["pairs"]:
        raise RuntimeError(
            f"{label} cannot reuse market artifacts: market_context_matrix="
            f"{market_context_matrix_path} has no pair/side matrix rows; refresh market context before "
            "live candidate discovery"
        )
    required_optional_context = (
        ("context_asset_charts", context_asset_charts_path, "charts"),
        ("broker_instruments", broker_instruments_path, "tradeable_instruments"),
    )
    for name, path, required_key in required_optional_context:
        if not path.exists():
            raise RuntimeError(
                f"{label} cannot reuse market artifacts: missing required market evidence artifact "
                f"{name}={path}; run without --reuse-market-artifacts or refresh market context before "
                "live candidate discovery"
            )
        try:
            context_payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise RuntimeError(
                f"{label} cannot reuse market artifacts: unreadable {name}={path}: {exc}; "
                "refresh market context before live candidate discovery"
            ) from exc
        if not isinstance(context_payload, dict) or not isinstance(context_payload.get(required_key), list):
            raise RuntimeError(
                f"{label} cannot reuse market artifacts: {name}={path} has no {required_key} rows; "
                "refresh market context before live candidate discovery"
            )


def _broker_instruments_snapshot(
    *,
    client: OandaReadOnlyClient,
    trader_pairs: tuple[str, ...],
    context_assets: tuple[str, ...],
) -> dict[str, Any]:
    generated_at = datetime.now(timezone.utc).isoformat()
    try:
        instruments = client.account_instruments()
    except Exception as exc:
        return {
            "generated_at_utc": generated_at,
            "schema_version": 1,
            "status": "FETCH_FAILED",
            "tradeability_policy": "BROKER_ACCOUNT_INSTRUMENTS_REQUIRED_FOR_LIVE_TRADE_UNIVERSE",
            "tradeable_instruments": [],
            "context_assets_tradeable": [],
            "context_assets_not_tradeable": list(context_assets),
            "trader_pairs_missing": [],
            "specs": {},
            "issues": [f"BROKER_INSTRUMENTS_FETCH_FAILED:{exc.__class__.__name__}:{exc}"],
        }
    specs: dict[str, dict[str, Any]] = {}
    for item in instruments:
        name = str(item.get("name") or "").upper()
        if not name:
            continue
        specs[name] = {
            "type": item.get("type"),
            "displayPrecision": item.get("displayPrecision"),
            "pipLocation": item.get("pipLocation"),
            "tradeUnitsPrecision": item.get("tradeUnitsPrecision"),
            "minimumTradeSize": item.get("minimumTradeSize"),
            "marginRate": item.get("marginRate"),
        }
    tradeable = sorted(specs)
    tradeable_set = set(tradeable)
    return {
        "generated_at_utc": generated_at,
        "schema_version": 1,
        "status": "OK",
        "tradeability_policy": "BROKER_ACCOUNT_INSTRUMENTS_REQUIRED_FOR_LIVE_TRADE_UNIVERSE",
        "tradeable_instruments": tradeable,
        "context_assets_tradeable": [asset for asset in context_assets if asset in tradeable_set],
        "context_assets_not_tradeable": [asset for asset in context_assets if asset not in tradeable_set],
        "trader_pairs_missing": [pair for pair in trader_pairs if pair not in tradeable_set],
        "specs": specs,
        "issues": [],
    }


def _write_broker_instruments_report(payload: dict[str, Any], report_path: Path) -> None:
    lines = [
        "# Broker Instruments",
        "",
        f"- Generated at UTC: `{payload.get('generated_at_utc')}`",
        f"- Status: `{payload.get('status')}`",
        f"- Tradeability policy: `{payload.get('tradeability_policy')}`",
        f"- Tradeable instruments: `{len(payload.get('tradeable_instruments') or [])}`",
        f"- Context assets tradeable: `{len(payload.get('context_assets_tradeable') or [])}`",
        f"- Context assets not tradeable: `{len(payload.get('context_assets_not_tradeable') or [])}`",
        "",
        "## Context Assets",
        "",
        "| Instrument | Broker tradeable |",
        "|---|---|",
    ]
    tradeable = set(payload.get("context_assets_tradeable") or [])
    for asset in [*payload.get("context_assets_tradeable", []), *payload.get("context_assets_not_tradeable", [])]:
        lines.append(f"| `{asset}` | `{asset in tradeable}` |")
    missing = [str(item) for item in payload.get("trader_pairs_missing", []) or [] if str(item).strip()]
    if missing:
        lines.extend(["", "## Missing Trader Pairs", ""])
        lines.extend(f"- `{pair}`" for pair in missing)
    issues = [str(item) for item in payload.get("issues", []) or [] if str(item).strip()]
    if issues:
        lines.extend(["", "## Issues", ""])
        lines.extend(f"- {issue}" for issue in issues[:40])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="qr-vnext")
    sub = parser.add_subparsers(dest="command", required=True)

    p_import = sub.add_parser("import-legacy", help="Import legacy archive into the vNext history DB.")
    p_import.add_argument("--archive", type=Path, default=DEFAULT_LEGACY_ARCHIVE)
    p_import.add_argument("--db", type=Path, default=DEFAULT_HISTORY_DB)
    p_import.add_argument("--report", type=Path, default=DEFAULT_IMPORT_REPORT)

    p_snapshot = sub.add_parser("broker-snapshot", help="Read current broker truth without placing orders.")
    p_snapshot.add_argument("--pairs", default=DEFAULT_TRADER_PAIRS_ARG)
    p_snapshot.add_argument("--output", type=Path, default=None)

    p_charts = sub.add_parser("pair-charts", help="Compute multi-timeframe indicator scores per pair.")
    p_charts.add_argument("--pairs", default=DEFAULT_TRADER_PAIRS_ARG)
    p_charts.add_argument("--timeframes", default=",".join(DEFAULT_PAIR_CHART_TIMEFRAMES))
    p_charts.add_argument("--count", type=int, default=200)
    p_charts.add_argument("--output", type=Path, default=DEFAULT_PAIR_CHARTS)
    p_charts.add_argument("--report", type=Path, default=DEFAULT_PAIR_CHARTS_REPORT)

    p_cross = sub.add_parser("cross-asset-snapshot", help="Cross-asset/inter-market snapshot (DXY, US bonds, SPX, Gold, Oil, BTC).")
    p_cross.add_argument("--instruments", default="")  # empty → use defaults
    p_cross.add_argument("--correlation-pairs", default=DEFAULT_TRADER_PAIRS_ARG)
    p_cross.add_argument("--granularity", default="H1")
    p_cross.add_argument("--count", type=int, default=200)
    p_cross.add_argument("--output", type=Path, default=DEFAULT_CROSS_ASSET_SNAPSHOT)
    p_cross.add_argument("--report", type=Path, default=DEFAULT_CROSS_ASSET_REPORT)

    p_context_assets = sub.add_parser(
        "context-asset-charts",
        help="Multi-timeframe technical charts for non-FX context assets (Gold, Oil, SPX, US yields, BTC).",
    )
    p_context_assets.add_argument("--instruments", default="")  # empty -> use defaults
    p_context_assets.add_argument("--timeframes", default=",".join(DEFAULT_PAIR_CHART_TIMEFRAMES))
    p_context_assets.add_argument("--count", type=int, default=200)
    p_context_assets.add_argument("--output", type=Path, default=DEFAULT_CONTEXT_ASSET_CHARTS)
    p_context_assets.add_argument("--report", type=Path, default=DEFAULT_CONTEXT_ASSET_CHARTS_REPORT)

    p_flow = sub.add_parser("flow-snapshot", help="Spread time-series per pair; OANDA books are opt-in.")
    p_flow.add_argument("--pairs", default=DEFAULT_TRADER_PAIRS_ARG)
    p_flow.add_argument("--top-n", type=int, default=5)
    p_flow.add_argument("--spread-lookback-minutes", type=int, default=60)
    p_flow.add_argument("--include-books", action="store_true", help="Also call OANDA orderBook/positionBook endpoints when the account has book entitlement.")
    p_flow.add_argument("--output", type=Path, default=DEFAULT_FLOW_SNAPSHOT)
    p_flow.add_argument("--report", type=Path, default=DEFAULT_FLOW_REPORT)

    p_strength = sub.add_parser("currency-strength", help="G8 currency strength meter from a 28-pair matrix.")
    p_strength.add_argument("--granularity", default="H1")
    p_strength.add_argument("--lookback-bars", type=int, default=24)
    p_strength.add_argument("--fetch-count", type=int, default=50)
    p_strength.add_argument("--output", type=Path, default=DEFAULT_CURRENCY_STRENGTH)
    p_strength.add_argument("--report", type=Path, default=DEFAULT_CURRENCY_STRENGTH_REPORT)

    p_levels = sub.add_parser("levels-snapshot", help="Pivots, PDH/PDL/PDC, session ranges, round-numbers per pair.")
    p_levels.add_argument("--pairs", default=DEFAULT_TRADER_PAIRS_ARG)
    p_levels.add_argument("--output", type=Path, default=DEFAULT_LEVELS_SNAPSHOT)
    p_levels.add_argument("--report", type=Path, default=DEFAULT_LEVELS_REPORT)

    p_matrix = sub.add_parser("market-context-matrix", help="Pair/side evidence matrix from chart, macro, flow, levels, calendar, COT, and options artifacts.")
    p_matrix.add_argument("--pair-charts", type=Path, default=DEFAULT_PAIR_CHARTS)
    p_matrix.add_argument("--context-asset-charts", type=Path, default=DEFAULT_CONTEXT_ASSET_CHARTS)
    p_matrix.add_argument("--cross-asset", type=Path, default=DEFAULT_CROSS_ASSET_SNAPSHOT)
    p_matrix.add_argument("--flow", type=Path, default=DEFAULT_FLOW_SNAPSHOT)
    p_matrix.add_argument("--currency-strength", type=Path, default=DEFAULT_CURRENCY_STRENGTH)
    p_matrix.add_argument("--levels", type=Path, default=DEFAULT_LEVELS_SNAPSHOT)
    p_matrix.add_argument("--calendar", type=Path, default=DEFAULT_CALENDAR_SNAPSHOT)
    p_matrix.add_argument("--cot", type=Path, default=DEFAULT_COT_SNAPSHOT)
    p_matrix.add_argument("--option-skew", type=Path, default=DEFAULT_OPTION_SKEW)
    p_matrix.add_argument("--output", type=Path, default=DEFAULT_MARKET_CONTEXT_MATRIX)
    p_matrix.add_argument("--report", type=Path, default=DEFAULT_MARKET_CONTEXT_MATRIX_REPORT)

    p_cal = sub.add_parser("economic-calendar", help="ForexFactory weekly XML feed + per-pair event-window flags.")
    p_cal.add_argument("--pairs", default=DEFAULT_TRADER_PAIRS_ARG)
    p_cal.add_argument("--pre-minutes", type=int, default=30)
    p_cal.add_argument("--post-minutes", type=int, default=30)
    p_cal.add_argument("--impact", default="High,Medium")
    p_cal.add_argument("--no-fetch", action="store_true", help="Skip fetching (for offline tests).")
    p_cal.add_argument("--output", type=Path, default=DEFAULT_CALENDAR_SNAPSHOT)
    p_cal.add_argument("--report", type=Path, default=DEFAULT_CALENDAR_REPORT)

    p_cot = sub.add_parser("cot-snapshot", help="CFTC Commitments of Traders weekly disaggregated report.")
    p_cot.add_argument("--no-fetch", action="store_true")
    p_cot.add_argument("--output", type=Path, default=DEFAULT_COT_SNAPSHOT)
    p_cot.add_argument("--report", type=Path, default=DEFAULT_COT_REPORT)

    p_opt = sub.add_parser("option-skew", help="Option-skew (RR/IV) adapter — placeholder until vendor feed is wired.")
    p_opt.add_argument("--pairs", default=DEFAULT_TRADER_PAIRS_ARG)
    p_opt.add_argument("--tenors", default="1W,1M,3M")
    p_opt.add_argument("--output", type=Path, default=DEFAULT_OPTION_SKEW)
    p_opt.add_argument("--report", type=Path, default=DEFAULT_OPTION_SKEW_REPORT)

    p_dreview = sub.add_parser(
        "daily-review",
        help="Build data/trader_overrides.json from recent realized P&L (Module C producer).",
    )
    p_dreview.add_argument(
        "--ledger-db",
        type=Path,
        default=Path("data/execution_ledger.db"),
        help="Path to execution_ledger.db (default: data/execution_ledger.db).",
    )
    p_dreview.add_argument(
        "--output",
        type=Path,
        default=Path("data/trader_overrides.json"),
        help="Where to write trader_overrides.json (default: data/trader_overrides.json).",
    )
    p_dreview.add_argument(
        "--lookback-hours",
        type=float,
        default=None,
        help="Lookback window in hours (default: env QR_DAILY_REVIEW_LOOKBACK_HOURS or 24).",
    )
    p_dreview.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print report but do not write the output file.",
    )

    p_news = sub.add_parser("news-snapshot", help="Fetch public news feeds into ignored data/log artifacts.")
    p_news.add_argument("--no-fetch", action="store_true", help="Skip network fetch and emit a missing-feed issue.")
    p_news.add_argument("--lookback-hours", type=int, default=None)
    p_news.add_argument("--max-items", type=int, default=None)
    p_news.add_argument("--digest-items", type=int, default=None)
    p_news.add_argument("--flow-entries", type=int, default=None)
    p_news.add_argument("--output", type=Path, default=DEFAULT_NEWS_SNAPSHOT)
    p_news.add_argument("--digest", type=Path, default=DEFAULT_NEWS_DIGEST)
    p_news.add_argument("--flow-log", type=Path, default=DEFAULT_NEWS_FLOW_LOG)

    p_news_health = sub.add_parser("news-health", help="Audit freshness, automation state, and market-story sync for news artifacts.")
    p_news_health.add_argument("--news-items", type=Path, default=DEFAULT_NEWS_SNAPSHOT)
    p_news_health.add_argument("--digest", type=Path, default=DEFAULT_NEWS_DIGEST)
    p_news_health.add_argument("--flow-log", type=Path, default=DEFAULT_NEWS_FLOW_LOG)
    p_news_health.add_argument("--market-story-profile", type=Path, default=DEFAULT_MARKET_STORY_PROFILE)
    p_news_health.add_argument("--calendar", type=Path, default=DEFAULT_CALENDAR_SNAPSHOT)
    p_news_health.add_argument(
        "--automation",
        type=Path,
        default=Path.home() / ".codex" / "automations" / "qr-news-digest" / "automation.toml",
    )
    p_news_health.add_argument(
        "--weekend-state",
        type=Path,
        default=Path.home() / ".codex" / "quant_rabbit_weekend_task_state.json",
    )
    p_news_health.add_argument("--output", type=Path, default=DEFAULT_NEWS_HEALTH)
    p_news_health.add_argument("--report", type=Path, default=DEFAULT_NEWS_HEALTH_REPORT)
    p_news_health.add_argument("--verify-fetch", action="store_true", help="Run a read-only live fetch probe.")
    p_news_health.add_argument("--strict", action="store_true", help="Return non-zero when status is BLOCK.")
    p_news_health.add_argument("--now-utc", default=None, help="Override current UTC timestamp for tests/audits.")

    p_prompt = sub.add_parser("trader-prompt-route", help="Route the trader to the minimal prompt branch for current state.")
    p_prompt.add_argument("--snapshot", type=Path, default=DEFAULT_BROKER_SNAPSHOT)
    p_prompt.add_argument("--target-state", type=Path, default=DEFAULT_DAILY_TARGET_STATE)
    p_prompt.add_argument("--intents", type=Path, default=DEFAULT_ORDER_INTENTS)
    p_prompt.add_argument("--pair-charts", type=Path, default=DEFAULT_PAIR_CHARTS)
    p_prompt.add_argument("--cross-asset", type=Path, default=DEFAULT_CROSS_ASSET_SNAPSHOT)
    p_prompt.add_argument("--flow", type=Path, default=DEFAULT_FLOW_SNAPSHOT)
    p_prompt.add_argument("--currency-strength", type=Path, default=DEFAULT_CURRENCY_STRENGTH)
    p_prompt.add_argument("--levels", type=Path, default=DEFAULT_LEVELS_SNAPSHOT)
    p_prompt.add_argument("--market-context-matrix", type=Path, default=DEFAULT_MARKET_CONTEXT_MATRIX)
    p_prompt.add_argument("--calendar", type=Path, default=DEFAULT_CALENDAR_SNAPSHOT)
    p_prompt.add_argument("--cot", type=Path, default=DEFAULT_COT_SNAPSHOT)
    p_prompt.add_argument("--option-skew", type=Path, default=DEFAULT_OPTION_SKEW)
    p_prompt.add_argument("--attack-advice", type=Path, default=DEFAULT_AI_ATTACK_ADVICE)
    p_prompt.add_argument("--learning-audit", type=Path, default=DEFAULT_LEARNING_AUDIT)
    p_prompt.add_argument("--memory-health", type=Path, default=DEFAULT_MEMORY_HEALTH)
    p_prompt.add_argument("--strategy-profile", type=Path, default=DEFAULT_STRATEGY_PROFILE)
    p_prompt.add_argument("--trader-overrides", type=Path, default=DEFAULT_TRADER_OVERRIDES)
    p_prompt.add_argument("--decision-response", type=Path, default=DEFAULT_CODEX_TRADER_DECISION_RESPONSE)
    p_prompt.add_argument("--gpt-decision", type=Path, default=DEFAULT_GPT_TRADER_DECISION)
    p_prompt.add_argument("--live-order", type=Path, default=DEFAULT_LIVE_ORDER_REQUEST)
    p_prompt.add_argument("--position-execution", type=Path, default=DEFAULT_POSITION_EXECUTION)
    p_prompt.add_argument("--autotrade-report", type=Path, default=DEFAULT_AUTOTRADE_REPORT)
    p_prompt.add_argument("--include-content", action="store_true")

    p_mine = sub.add_parser("mine-strategy", help="Mine legacy evidence into a strategy profile.")
    p_mine.add_argument("--db", type=Path, default=DEFAULT_HISTORY_DB)
    p_mine.add_argument("--report", type=Path, default=DEFAULT_STRATEGY_REPORT)
    p_mine.add_argument("--profile", type=Path, default=DEFAULT_STRATEGY_PROFILE)
    p_mine.add_argument("--loss-cap-jpy", type=float, default=None)
    p_mine.add_argument("--target-state", type=Path, default=DEFAULT_DAILY_TARGET_STATE)

    p_story = sub.add_parser("mine-market-stories", help="Mine narrative, market regime, and chart-story evidence.")
    p_story.add_argument("--archive", type=Path, default=DEFAULT_LEGACY_ARCHIVE)
    p_story.add_argument("--report", type=Path, default=DEFAULT_MARKET_STORY_REPORT)
    p_story.add_argument("--profile", type=Path, default=DEFAULT_MARKET_STORY_PROFILE)
    p_story.add_argument("--news-dir", type=Path, default=ROOT / "logs")
    p_story.add_argument("--db", type=Path, default=DEFAULT_HISTORY_DB)

    p_market_status = sub.add_parser("market-status", help="Write deterministic FX market/session status.")
    p_market_status.add_argument("--output", type=Path, default=DEFAULT_MARKET_STATUS)
    p_market_status.add_argument("--report", type=Path, default=DEFAULT_MARKET_STATUS_REPORT)

    p_campaign = sub.add_parser("plan-campaign", help="Build a multi-desk daily campaign plan.")
    p_campaign.add_argument("--start-balance", type=float, required=True)
    p_campaign.add_argument("--target-return-pct", type=float, default=10.0)
    p_campaign.add_argument("--strategy-profile", type=Path, default=DEFAULT_STRATEGY_PROFILE)
    p_campaign.add_argument("--market-story-profile", type=Path, default=DEFAULT_MARKET_STORY_PROFILE)
    p_campaign.add_argument("--report", type=Path, default=DEFAULT_CAMPAIGN_REPORT)
    p_campaign.add_argument("--plan", type=Path, default=DEFAULT_CAMPAIGN_PLAN)

    p_target = sub.add_parser("daily-target-state", help="Record daily 10%% target progress from broker truth.")
    p_target.add_argument("--start-balance", type=float, default=None)
    p_target.add_argument("--target-return-pct", type=float, default=None)
    p_target.add_argument(
        "--target-profit-jpy",
        type=float,
        default=None,
        help="Absolute JPY profit objective for this campaign day; converted to return %% from broker-truth start equity.",
    )
    p_target.add_argument("--realized-pl", type=float, default=None)
    p_target.add_argument(
        "--daily-risk-budget",
        type=float,
        default=None,
        help="Absolute JPY override. Prefer --daily-risk-pct (NAV-anchored, auto-scales with equity).",
    )
    p_target.add_argument(
        "--daily-risk-pct",
        type=float,
        default=None,
        help="Daily risk budget as %% of starting NAV (e.g. 4.5). Overrides --daily-risk-budget and persisted JPY.",
    )
    p_target.add_argument(
        "--target-trades-per-day",
        type=int,
        default=None,
        help="Override expected trade pace (per_trade cap = daily_risk_budget / pace).",
    )
    p_target.add_argument("--snapshot", type=Path, default=None)
    p_target.add_argument("--state", type=Path, default=DEFAULT_DAILY_TARGET_STATE)
    p_target.add_argument("--report", type=Path, default=DEFAULT_DAILY_TARGET_REPORT)
    p_target.add_argument("--execution-ledger-db", type=Path, default=DEFAULT_EXECUTION_LEDGER_DB)

    p_replay = sub.add_parser("replay-backtest", help="Replay imported legacy days against 10%% target coverage.")
    p_replay.add_argument("--db", type=Path, default=DEFAULT_HISTORY_DB)
    p_replay.add_argument("--start-balance", type=float, required=True)
    p_replay.add_argument("--target-return-pct", type=float, default=10.0)
    p_replay.add_argument("--max-loss", type=float, default=None)
    p_replay.add_argument("--daily-risk-pct", type=float, default=None)
    p_replay.add_argument("--target-trades-per-day", type=int, default=None)
    p_replay.add_argument("--target-state", type=Path, default=DEFAULT_DAILY_TARGET_STATE)
    p_replay.add_argument("--max-days", type=int, default=None)
    p_replay.add_argument("--output", type=Path, default=DEFAULT_REPLAY_BACKTEST)
    p_replay.add_argument("--report", type=Path, default=DEFAULT_REPLAY_BACKTEST_REPORT)

    p_ai_test = sub.add_parser(
        "ai-test-bot-backtest",
        help="Walk-forward backtest an AI-managed parameter policy over imported legacy records.",
    )
    p_ai_test.add_argument("--db", type=Path, default=DEFAULT_HISTORY_DB)
    p_ai_test.add_argument("--start-balance", type=float, required=True)
    p_ai_test.add_argument("--target-return-pct", type=float, default=10.0)
    p_ai_test.add_argument("--max-loss", type=float, default=None)
    p_ai_test.add_argument("--daily-risk-pct", type=float, default=None)
    p_ai_test.add_argument("--target-trades-per-day", type=int, default=None)
    p_ai_test.add_argument("--target-state", type=Path, default=DEFAULT_DAILY_TARGET_STATE)
    p_ai_test.add_argument("--training-days", type=int, default=DEFAULT_TRAINING_DAYS)
    p_ai_test.add_argument("--min-train-trades", type=int, default=DEFAULT_MIN_TRAIN_TRADES)
    p_ai_test.add_argument("--min-train-win-rate-pct", type=float, default=DEFAULT_MIN_TRAIN_WIN_RATE_PCT)
    p_ai_test.add_argument("--max-active-buckets", type=int, default=DEFAULT_MAX_ACTIVE_BUCKETS)
    p_ai_test.add_argument("--source-tables", default=",".join(DEFAULT_RUNTIME_SOURCE_TABLES))
    p_ai_test.add_argument("--execution-ledger-db", type=Path, default=DEFAULT_EXECUTION_LEDGER_DB)
    p_ai_test.add_argument("--no-dedupe-opportunities", action="store_true")
    p_ai_test.add_argument("--max-validation-days", type=int, default=None)
    p_ai_test.add_argument("--output", type=Path, default=DEFAULT_AI_TEST_BOT_BACKTEST)
    p_ai_test.add_argument("--report", type=Path, default=DEFAULT_AI_TEST_BOT_BACKTEST_REPORT)

    p_outcome = sub.add_parser("build-outcome-mart", help="Build read-only archive outcome features for lane ranking.")
    p_outcome.add_argument("--db", type=Path, default=DEFAULT_HISTORY_DB)
    p_outcome.add_argument("--execution-ledger-db", type=Path, default=DEFAULT_EXECUTION_LEDGER_DB)
    p_outcome.add_argument("--output", type=Path, default=DEFAULT_OUTCOME_MART)
    p_outcome.add_argument("--report", type=Path, default=DEFAULT_OUTCOME_MART_REPORT)

    p_coverage = sub.add_parser("optimize-coverage", help="Measure live-ready target coverage and emit gap tasks.")
    p_coverage.add_argument("--intents", type=Path, default=DEFAULT_ORDER_INTENTS)
    p_coverage.add_argument("--target-state", type=Path, default=DEFAULT_DAILY_TARGET_STATE)
    p_coverage.add_argument("--replay", type=Path, default=DEFAULT_REPLAY_BACKTEST)
    p_coverage.add_argument("--ai-backtest", type=Path, default=DEFAULT_AI_TEST_BOT_BACKTEST)
    p_coverage.add_argument("--strategy-profile", type=Path, default=DEFAULT_STRATEGY_PROFILE)
    p_coverage.add_argument("--market-context-matrix", type=Path, default=DEFAULT_MARKET_CONTEXT_MATRIX)
    p_coverage.add_argument("--output", type=Path, default=DEFAULT_COVERAGE_OPTIMIZATION)
    p_coverage.add_argument("--report", type=Path, default=DEFAULT_COVERAGE_OPTIMIZATION_REPORT)

    p_attack = sub.add_parser("ai-attack-advice", help="Rank current LIVE_READY lanes using read-only AI parameter advice.")
    p_attack.add_argument("--intents", type=Path, default=DEFAULT_ORDER_INTENTS)
    p_attack.add_argument("--target-state", type=Path, default=DEFAULT_DAILY_TARGET_STATE)
    p_attack.add_argument("--ai-backtest", type=Path, default=DEFAULT_AI_TEST_BOT_BACKTEST)
    p_attack.add_argument("--outcome-mart", type=Path, default=DEFAULT_OUTCOME_MART)
    p_attack.add_argument("--coverage", type=Path, default=DEFAULT_COVERAGE_OPTIMIZATION)
    p_attack.add_argument("--output", type=Path, default=DEFAULT_AI_ATTACK_ADVICE)
    p_attack.add_argument("--report", type=Path, default=DEFAULT_AI_ATTACK_ADVICE_REPORT)

    p_laudit = sub.add_parser(
        "learning-audit",
        help="Audit learning evidence, live ranking influence, and recent effect metrics.",
    )
    p_laudit.add_argument("--db", type=Path, default=DEFAULT_EXECUTION_LEDGER_DB)
    p_laudit.add_argument("--output", type=Path, default=DEFAULT_LEARNING_AUDIT)
    p_laudit.add_argument("--report", type=Path, default=DEFAULT_LEARNING_AUDIT_REPORT)
    p_laudit.add_argument("--ai-backtest", type=Path, default=DEFAULT_AI_TEST_BOT_BACKTEST)
    p_laudit.add_argument("--outcome-mart", type=Path, default=DEFAULT_OUTCOME_MART)
    p_laudit.add_argument("--post-trade-learning", type=Path, default=DEFAULT_POST_TRADE_LEARNING)
    p_laudit.add_argument("--ai-attack-advice", type=Path, default=DEFAULT_AI_ATTACK_ADVICE)
    p_laudit.add_argument("--window-hours", type=float, default=168.0)
    p_laudit.add_argument("--min-effect-sample", type=int, default=30)

    p_mhealth = sub.add_parser(
        "memory-health",
        help="Audit short, medium, long, and position memory artifacts before trader routing.",
    )
    p_mhealth.add_argument("--snapshot", type=Path, default=DEFAULT_BROKER_SNAPSHOT)
    p_mhealth.add_argument("--target-state", type=Path, default=DEFAULT_DAILY_TARGET_STATE)
    p_mhealth.add_argument("--order-intents", type=Path, default=DEFAULT_ORDER_INTENTS)
    p_mhealth.add_argument("--strategy-profile", type=Path, default=DEFAULT_STRATEGY_PROFILE)
    p_mhealth.add_argument("--forecast-history", type=Path, default=DEFAULT_FORECAST_HISTORY)
    p_mhealth.add_argument("--projection-ledger", type=Path, default=DEFAULT_PROJECTION_LEDGER)
    p_mhealth.add_argument("--learning-audit", type=Path, default=DEFAULT_LEARNING_AUDIT)
    p_mhealth.add_argument("--entry-thesis-ledger", type=Path, default=DEFAULT_ENTRY_THESIS_LEDGER)
    p_mhealth.add_argument("--execution-ledger-db", type=Path, default=DEFAULT_EXECUTION_LEDGER_DB)
    p_mhealth.add_argument("--output", type=Path, default=DEFAULT_MEMORY_HEALTH)
    p_mhealth.add_argument("--report", type=Path, default=DEFAULT_MEMORY_HEALTH_REPORT)

    p_self_audit = sub.add_parser(
        "self-improvement-audit",
        help="Aggregate profitability, memory, verification, and decision-history holes into ranked repair tasks.",
    )
    p_self_audit.add_argument("--db", type=Path, default=DEFAULT_EXECUTION_LEDGER_DB)
    p_self_audit.add_argument(
        "--history-db",
        type=Path,
        default=None,
        help="Optional DB for audit run history. Defaults to --db.",
    )
    p_self_audit.add_argument("--output", type=Path, default=DEFAULT_SELF_IMPROVEMENT_AUDIT)
    p_self_audit.add_argument("--report", type=Path, default=DEFAULT_SELF_IMPROVEMENT_AUDIT_REPORT)
    p_self_audit.add_argument("--snapshot", type=Path, default=DEFAULT_BROKER_SNAPSHOT)
    p_self_audit.add_argument("--target-state", type=Path, default=DEFAULT_DAILY_TARGET_STATE)
    p_self_audit.add_argument("--order-intents", type=Path, default=DEFAULT_ORDER_INTENTS)
    p_self_audit.add_argument("--market-context-matrix", type=Path, default=DEFAULT_MARKET_CONTEXT_MATRIX)
    p_self_audit.add_argument("--memory-health", type=Path, default=DEFAULT_MEMORY_HEALTH)
    p_self_audit.add_argument("--learning-audit", type=Path, default=DEFAULT_LEARNING_AUDIT)
    p_self_audit.add_argument("--ai-test-bot-backtest", type=Path, default=DEFAULT_AI_TEST_BOT_BACKTEST)
    p_self_audit.add_argument("--verification-ledger", type=Path, default=DEFAULT_VERIFICATION_LEDGER)
    p_self_audit.add_argument("--forecast-history", type=Path, default=DEFAULT_FORECAST_HISTORY)
    p_self_audit.add_argument("--projection-ledger", type=Path, default=DEFAULT_PROJECTION_LEDGER)
    p_self_audit.add_argument("--entry-thesis-ledger", type=Path, default=DEFAULT_ENTRY_THESIS_LEDGER)
    p_self_audit.add_argument("--gpt-decision", type=Path, default=DEFAULT_GPT_TRADER_DECISION)
    p_self_audit.add_argument("--trader-decision", type=Path, default=DEFAULT_TRADER_DECISION)
    p_self_audit.add_argument("--position-management", type=Path, default=ROOT / "data" / "position_management.json")
    p_self_audit.add_argument("--thesis-evolution", type=Path, default=ROOT / "data" / "thesis_evolution_report.json")
    p_self_audit.add_argument("--position-thesis", type=Path, default=ROOT / "data" / "position_thesis_report.json")
    p_self_audit.add_argument("--forecast-persistence", type=Path, default=ROOT / "data" / "forecast_persistence_report.json")
    p_self_audit.add_argument("--coverage-optimization", type=Path, default=DEFAULT_COVERAGE_OPTIMIZATION)
    p_self_audit.add_argument("--window-hours", type=float, default=168.0)

    p_exec_replay = sub.add_parser("replay-execution", help="Replay live-ready order receipts over a quote path.")
    p_exec_replay.add_argument("--intents", type=Path, default=DEFAULT_ORDER_INTENTS)
    p_exec_replay.add_argument("--prices", type=Path, required=True)
    p_exec_replay.add_argument("--target-jpy", type=float, default=0.0)
    p_exec_replay.add_argument("--lane-id", default=None)
    p_exec_replay.add_argument("--output", type=Path, default=DEFAULT_EXECUTION_REPLAY)
    p_exec_replay.add_argument("--report", type=Path, default=DEFAULT_EXECUTION_REPLAY_REPORT)

    p_ledger = sub.add_parser("execution-ledger-sync", help="Sync OANDA transactions into the append-only execution ledger.")
    p_ledger.add_argument("--db", type=Path, default=DEFAULT_EXECUTION_LEDGER_DB)
    p_ledger.add_argument("--report", type=Path, default=DEFAULT_EXECUTION_LEDGER_REPORT)
    p_ledger.add_argument("--since-transaction-id", default=None)

    p_vledger = sub.add_parser(
        "verification-ledger-audit",
        help="Record cycle verifiability and effect metrics into the execution ledger DB.",
    )
    p_vledger.add_argument("--db", type=Path, default=DEFAULT_EXECUTION_LEDGER_DB)
    p_vledger.add_argument("--output", type=Path, default=DEFAULT_VERIFICATION_LEDGER)
    p_vledger.add_argument("--report", type=Path, default=DEFAULT_VERIFICATION_LEDGER_REPORT)
    p_vledger.add_argument("--snapshot", type=Path, default=DEFAULT_BROKER_SNAPSHOT)
    p_vledger.add_argument("--order-intents", type=Path, default=DEFAULT_ORDER_INTENTS)
    p_vledger.add_argument("--gpt-decision", type=Path, default=DEFAULT_GPT_TRADER_DECISION)
    p_vledger.add_argument("--live-order", type=Path, default=DEFAULT_LIVE_ORDER_REQUEST)
    p_vledger.add_argument("--position-execution", type=Path, default=DEFAULT_POSITION_EXECUTION)
    p_vledger.add_argument("--thesis-evolution", type=Path, default=Path("data/thesis_evolution_report.json"))
    p_vledger.add_argument("--position-thesis", type=Path, default=Path("data/position_thesis_report.json"))
    p_vledger.add_argument("--forecast-persistence", type=Path, default=Path("data/forecast_persistence_report.json"))
    p_vledger.add_argument("--ai-backtest", type=Path, default=DEFAULT_AI_TEST_BOT_BACKTEST)
    p_vledger.add_argument("--outcome-mart", type=Path, default=DEFAULT_OUTCOME_MART)
    p_vledger.add_argument("--post-trade-learning", type=Path, default=DEFAULT_POST_TRADE_LEARNING)
    p_vledger.add_argument("--ai-attack-advice", type=Path, default=DEFAULT_AI_ATTACK_ADVICE)
    p_vledger.add_argument("--learning-audit", type=Path, default=DEFAULT_LEARNING_AUDIT)
    p_vledger.add_argument("--window-hours", type=float, default=168.0)

    p_learn = sub.add_parser("learn-post-trade", help="Create receipt-backed post-trade learning candidates.")
    p_learn.add_argument("--outcome", type=Path, default=None)
    p_learn.add_argument("--live-order", type=Path, default=DEFAULT_LIVE_ORDER_REQUEST)
    p_learn.add_argument("--position-execution", type=Path, default=DEFAULT_POSITION_EXECUTION)
    p_learn.add_argument("--trader-decision", type=Path, default=DEFAULT_TRADER_DECISION)
    p_learn.add_argument("--gpt-decision", type=Path, default=DEFAULT_GPT_TRADER_DECISION)
    p_learn.add_argument("--output", type=Path, default=DEFAULT_POST_TRADE_LEARNING)
    p_learn.add_argument("--report", type=Path, default=DEFAULT_POST_TRADE_LEARNING_REPORT)

    p_trailing = sub.add_parser(
        "trailing-sl-update",
        help="Tighten broker-side SL on trader-owned positions when M15/M30/H1 BOS prints against the position.",
    )
    p_trailing.add_argument("--snapshot", type=Path, default=DEFAULT_BROKER_SNAPSHOT)
    p_trailing.add_argument("--pair-charts", type=Path, default=DEFAULT_PAIR_CHARTS)
    p_trailing.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute updates but do not call broker.replace_trade_dependent_orders.",
    )

    p_pthesis = sub.add_parser(
        "position-thesis-check",
        help="Apply full prediction stack to each open position; emit EXTEND/HOLD/REVIEW_CLOSE verdicts to data/position_thesis_report.json.",
    )
    p_pthesis.add_argument("--snapshot", type=Path, default=DEFAULT_BROKER_SNAPSHOT)
    p_pthesis.add_argument("--pair-charts", type=Path, default=DEFAULT_PAIR_CHARTS)
    p_pthesis.add_argument("--output", type=Path, default=Path("data/position_thesis_report.json"))

    p_tevol = sub.add_parser(
        "thesis-evolution-check",
        aliases=["thesis-evolution"],
        help="Compare each open position's CURRENT forecast against its entry-time thesis; emit STILL_VALID/WEAKENED/BROKEN to data/thesis_evolution_report.json. Fresh BROKEN/RECOMMEND_CLOSE is hard Gate A and carries standing structural loss-cut authorization.",
    )
    p_tevol.add_argument("--snapshot", type=Path, default=DEFAULT_BROKER_SNAPSHOT)
    p_tevol.add_argument("--pair-charts", type=Path, default=DEFAULT_PAIR_CHARTS)
    p_tevol.add_argument("--output", type=Path, default=Path("data/thesis_evolution_report.json"))

    p_fperst = sub.add_parser(
        "forecast-persistence-check",
        help="Refresh/read forecast_history.jsonl and emit per-position persistence verdicts (RECOMMEND_CLOSE on N-cycle flip / EXTEND on N-cycle aligned / HOLD on mixed) to data/forecast_persistence_report.json. Fresh RECOMMEND_CLOSE is soft Gate A and still needs explicit env/token Gate B.",
    )
    p_fperst.add_argument("--snapshot", type=Path, default=DEFAULT_BROKER_SNAPSHOT)
    p_fperst.add_argument("--pair-charts", type=Path, default=DEFAULT_PAIR_CHARTS)
    p_fperst.add_argument("--output", type=Path, default=Path("data/forecast_persistence_report.json"))

    p_pmanage = sub.add_parser(
        "position-management",
        help="Refresh PositionManager's read-only sidecar from the latest broker snapshot.",
    )
    p_pmanage.add_argument("--snapshot", type=Path, default=DEFAULT_BROKER_SNAPSHOT)
    p_pmanage.add_argument("--pair-charts", type=Path, default=DEFAULT_PAIR_CHARTS)
    p_pmanage.add_argument("--trader-decision", type=Path, default=DEFAULT_TRADER_DECISION)
    p_pmanage.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Directory for entry thesis / forecast ledgers. Defaults to --snapshot parent.",
    )
    p_pmanage.add_argument("--output", type=Path, default=DEFAULT_POSITION_MANAGEMENT)
    p_pmanage.add_argument("--report", type=Path, default=DEFAULT_POSITION_MANAGEMENT_REPORT)

    p_plim = sub.add_parser(
        "generate-predictive-limits",
        help="Generate LIMIT orders from Grade-A forward-projection setups (path Step B + liquidity sweep fades).",
    )
    p_plim.add_argument("--snapshot", type=Path, default=DEFAULT_BROKER_SNAPSHOT)
    p_plim.add_argument("--pair-charts", type=Path, default=DEFAULT_PAIR_CHARTS)
    p_plim.add_argument("--send", action="store_true", help="Send LIMITs via OANDA. Default is dry-run JSON only.")
    p_plim.add_argument("--output", type=Path, default=Path("data/predictive_limit_orders.json"))

    p_vproj = sub.add_parser(
        "verify-projections",
        help="Verify pending forward-projection predictions against candle price truth; resolves HIT/MISS/TIMEOUT.",
    )
    p_vproj.add_argument("--snapshot", type=Path, default=DEFAULT_BROKER_SNAPSHOT)
    p_vproj.add_argument("--pair-charts", type=Path, default=DEFAULT_PAIR_CHARTS)
    p_vproj.add_argument(
        "--m1-count",
        type=int,
        default=int(os.environ.get("QR_PROJECTION_VERIFY_M1_COUNT", "1500")),
        help="Recent M1 candles to fetch per pending pair for path-based verification. Use 0 to fall back to snapshot quotes.",
    )
    p_vproj.add_argument(
        "--m5-count",
        type=int,
        default=int(os.environ.get("QR_PROJECTION_VERIFY_M5_COUNT", "1500")),
        help="Recent M5 candles to fetch per pending pair as older-truth fallback. Use 0 to disable.",
    )

    p_tprebal = sub.add_parser(
        "tp-rebalance",
        help="Expand/contract TP on trader-owned and manual/tagless positions as market regime shifts.",
    )
    p_tprebal.add_argument("--snapshot", type=Path, default=DEFAULT_BROKER_SNAPSHOT)
    p_tprebal.add_argument("--pair-charts", type=Path, default=DEFAULT_PAIR_CHARTS)
    p_tprebal.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute adjustments but do not call broker.replace_trade_dependent_orders.",
    )

    p_advpart = sub.add_parser(
        "adverse-partial-close",
        help="Partial-close trader-owned positions that are ≥1.5×ATR underwater with no reversal — frees margin without abandoning thesis.",
    )
    p_advpart.add_argument("--snapshot", type=Path, default=DEFAULT_BROKER_SNAPSHOT)
    p_advpart.add_argument("--pair-charts", type=Path, default=DEFAULT_PAIR_CHARTS)
    p_advpart.add_argument("--output", type=Path, default=DEFAULT_ADVERSE_PARTIAL_CLOSE)
    p_advpart.add_argument("--report", type=Path, default=DEFAULT_ADVERSE_PARTIAL_CLOSE_REPORT)
    p_advpart.add_argument("--execution-ledger-db", type=Path, default=DEFAULT_EXECUTION_LEDGER_DB)
    p_advpart.add_argument("--execution-ledger-report", type=Path, default=DEFAULT_EXECUTION_LEDGER_REPORT)
    p_advpart.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute actions but do not call broker.close_trade(). This is the default unless --send is set.",
    )
    p_advpart.add_argument("--send", action="store_true", help="Send partial-close requests to OANDA.")
    p_advpart.add_argument(
        "--confirm-live",
        action="store_true",
        help="Required with --send; prevents accidental live loss-side partial close.",
    )

    p_profitpart = sub.add_parser(
        "profit-partial-close",
        help="Stage/send profit-side partial closes for trader-owned and manual/tagless positions at ATR milestones.",
    )
    p_profitpart.add_argument("--snapshot", type=Path, default=DEFAULT_BROKER_SNAPSHOT)
    p_profitpart.add_argument("--pair-charts", type=Path, default=DEFAULT_PAIR_CHARTS)
    p_profitpart.add_argument("--state", type=Path, default=DEFAULT_PROFIT_PARTIAL_CLOSE_STATE)
    p_profitpart.add_argument("--output", type=Path, default=DEFAULT_PROFIT_PARTIAL_CLOSE)
    p_profitpart.add_argument("--report", type=Path, default=DEFAULT_PROFIT_PARTIAL_CLOSE_REPORT)
    p_profitpart.add_argument("--execution-ledger-db", type=Path, default=DEFAULT_EXECUTION_LEDGER_DB)
    p_profitpart.add_argument("--execution-ledger-report", type=Path, default=DEFAULT_EXECUTION_LEDGER_REPORT)
    p_profitpart.add_argument("--send", action="store_true", help="Send partial-close requests to OANDA.")
    p_profitpart.add_argument(
        "--confirm-live",
        action="store_true",
        help="Required with --send; prevents accidental live position reduction.",
    )

    p_cert = sub.add_parser("certify-dry-run", help="Certify dry-run receipts before live expansion.")
    p_cert.add_argument("--coverage", type=Path, default=DEFAULT_COVERAGE_OPTIMIZATION)
    p_cert.add_argument("--execution-replay", type=Path, default=DEFAULT_EXECUTION_REPLAY)
    p_cert.add_argument("--post-trade-learning", type=Path, default=DEFAULT_POST_TRADE_LEARNING)
    p_cert.add_argument("--order-intents", type=Path, default=DEFAULT_ORDER_INTENTS)
    p_cert.add_argument("--live-order", type=Path, default=DEFAULT_LIVE_ORDER_REQUEST)
    p_cert.add_argument("--position-execution", type=Path, default=DEFAULT_POSITION_EXECUTION)
    p_cert.add_argument("--gpt-decision", type=Path, default=DEFAULT_GPT_TRADER_DECISION)
    p_cert.add_argument("--output", type=Path, default=DEFAULT_DRY_RUN_CERTIFICATION)
    p_cert.add_argument("--report", type=Path, default=DEFAULT_DRY_RUN_CERTIFICATION_REPORT)

    p_complete = sub.add_parser("completion-status", help="Summarize blockers to full system completion.")
    p_complete.add_argument("--broker-snapshot", type=Path, default=DEFAULT_BROKER_SNAPSHOT)
    p_complete.add_argument("--order-intents", type=Path, default=DEFAULT_ORDER_INTENTS)
    p_complete.add_argument("--target-state", type=Path, default=DEFAULT_DAILY_TARGET_STATE)
    p_complete.add_argument("--coverage", type=Path, default=DEFAULT_COVERAGE_OPTIMIZATION)
    p_complete.add_argument("--replay-backtest", type=Path, default=DEFAULT_REPLAY_BACKTEST)
    p_complete.add_argument("--execution-replay", type=Path, default=DEFAULT_EXECUTION_REPLAY)
    p_complete.add_argument("--dry-run-certification", type=Path, default=DEFAULT_DRY_RUN_CERTIFICATION)
    p_complete.add_argument("--live-order", type=Path, default=DEFAULT_LIVE_ORDER_REQUEST)
    p_complete.add_argument("--output", type=Path, default=DEFAULT_COMPLETION_STATUS)
    p_complete.add_argument("--report", type=Path, default=DEFAULT_COMPLETION_STATUS_REPORT)

    p_gpt = sub.add_parser("gpt-trader-decision", help="Verify a Codex-written trader decision against broker truth.")
    p_gpt.add_argument("--snapshot", type=Path, required=True)
    p_gpt.add_argument("--intents", type=Path, default=DEFAULT_ORDER_INTENTS)
    p_gpt.add_argument("--campaign-plan", type=Path, default=DEFAULT_CAMPAIGN_PLAN)
    p_gpt.add_argument("--strategy-profile", type=Path, default=DEFAULT_STRATEGY_PROFILE)
    p_gpt.add_argument("--market-story-profile", type=Path, default=DEFAULT_MARKET_STORY_PROFILE)
    p_gpt.add_argument("--market-status", type=Path, default=DEFAULT_MARKET_STATUS)
    p_gpt.add_argument("--target-state", type=Path, default=DEFAULT_DAILY_TARGET_STATE)
    p_gpt.add_argument("--attack-advice", type=Path, default=DEFAULT_AI_ATTACK_ADVICE)
    p_gpt.add_argument("--learning-audit", type=Path, default=DEFAULT_LEARNING_AUDIT)
    p_gpt.add_argument("--market-context-matrix", type=Path, default=DEFAULT_MARKET_CONTEXT_MATRIX)
    p_gpt.add_argument("--decision-response", type=Path, default=None)
    p_gpt.add_argument("--max-lanes", type=int, default=DEFAULT_GPT_MAX_LANES)
    p_gpt.add_argument("--output", type=Path, default=DEFAULT_GPT_TRADER_DECISION)
    p_gpt.add_argument("--report", type=Path, default=DEFAULT_GPT_TRADER_DECISION_REPORT)

    p_intents = sub.add_parser("generate-intents", help="Generate dry-run order intents from campaign lanes.")
    p_intents.add_argument("--campaign-plan", type=Path, default=DEFAULT_CAMPAIGN_PLAN)
    p_intents.add_argument("--strategy-profile", type=Path, default=DEFAULT_STRATEGY_PROFILE)
    p_intents.add_argument("--snapshot", type=Path)
    p_intents.add_argument("--output", type=Path, default=DEFAULT_ORDER_INTENTS)
    p_intents.add_argument("--report", type=Path, default=DEFAULT_ORDER_INTENT_REPORT)
    p_intents.add_argument("--max-loss-jpy", type=float, default=None)
    p_intents.add_argument("--max-loss-pct", type=float, default=None)
    p_intents.add_argument("--risk-equity-jpy", type=float, default=None)
    p_intents.add_argument("--max-candidates", type=int, default=56)
    p_intents.add_argument("--market-context-matrix", type=Path, default=DEFAULT_MARKET_CONTEXT_MATRIX)
    p_intents.add_argument("--market-story-profile", type=Path, default=DEFAULT_MARKET_STORY_PROFILE)
    p_intents.add_argument("--market-story-report", type=Path, default=DEFAULT_RUNTIME_MARKET_STORY_REPORT)
    p_intents.add_argument("--market-news-dir", type=Path, default=ROOT / "logs")
    p_intents.add_argument(
        "--no-refresh-market-story",
        action="store_true",
        help="Skip automatic market-story refresh from logs/news_digest.md before intent generation.",
    )

    p_promote = sub.add_parser("promote-receipts", help="Promote strategy profiles from dry-run order receipts.")
    p_promote.add_argument("--profile", type=Path, default=DEFAULT_STRATEGY_PROFILE)
    p_promote.add_argument("--intents", type=Path, default=DEFAULT_ORDER_INTENTS)
    p_promote.add_argument("--output-profile", type=Path, default=None)
    p_promote.add_argument("--report", type=Path, default=DEFAULT_RECEIPT_PROMOTION_REPORT)

    p_stage = sub.add_parser("stage-live-order", help="Stage or explicitly send one live-ready OANDA order.")
    p_stage.add_argument("--intents", type=Path, default=DEFAULT_ORDER_INTENTS)
    p_stage.add_argument("--strategy-profile", type=Path, default=DEFAULT_STRATEGY_PROFILE)
    p_stage.add_argument("--lane-id", default=None)
    p_stage.add_argument("--output", type=Path, default=DEFAULT_LIVE_ORDER_REQUEST)
    p_stage.add_argument("--report", type=Path, default=DEFAULT_LIVE_ORDER_STAGE_REPORT)
    p_stage.add_argument("--max-loss-jpy", type=float, default=None)
    p_stage.add_argument("--max-loss-pct", type=float, default=None)
    p_stage.add_argument("--risk-equity-jpy", type=float, default=None)
    p_stage.add_argument("--send", action="store_true")
    p_stage.add_argument("--confirm-live", action="store_true")
    p_stage.add_argument("--execution-ledger-db", type=Path, default=DEFAULT_EXECUTION_LEDGER_DB)
    p_stage.add_argument("--execution-ledger-report", type=Path, default=DEFAULT_EXECUTION_LEDGER_REPORT)

    p_auto = sub.add_parser("autotrade-cycle", help="Run one safe automated trading cycle.")
    p_auto.add_argument("--send", action="store_true")
    p_auto.add_argument("--report", type=Path, default=DEFAULT_AUTOTRADE_REPORT)
    p_auto.add_argument("--use-gpt-trader", action="store_true")
    p_auto.add_argument("--gpt-decision-response", type=Path, default=None)
    p_auto.add_argument("--gpt-max-lanes", type=int, default=DEFAULT_GPT_MAX_LANES)
    p_auto.add_argument(
        "--reuse-market-artifacts",
        action="store_true",
        help=(
            "Use the existing broker snapshot and order intents for GPT verification/selection; "
            "the live gateway still fetches fresh broker truth before staging or sending."
        ),
    )
    p_auto.add_argument(
        "--refresh-market-story",
        dest="refresh_market_story",
        action="store_true",
        help="Refresh market story from news before scoring lanes (default)",
    )
    p_auto.add_argument(
        "--no-refresh-market-story",
        dest="refresh_market_story",
        action="store_false",
        help="Disable market story refresh for this cycle",
    )
    p_auto.set_defaults(refresh_market_story=True)
    p_auto.add_argument("--market-news-dir", type=Path, default=ROOT / "logs")
    p_auto.add_argument("--max-loss-jpy", type=float, default=None)
    p_auto.add_argument("--max-loss-pct", type=float, default=None)
    p_auto.add_argument("--risk-equity-jpy", type=float, default=None)
    p_auto.add_argument("--trader-settings", type=Path, default=DEFAULT_TRADER_SETTINGS)
    p_auto.add_argument("--execution-ledger-db", type=Path, default=DEFAULT_EXECUTION_LEDGER_DB)
    p_auto.add_argument("--execution-ledger-report", type=Path, default=DEFAULT_EXECUTION_LEDGER_REPORT)
    p_auto.add_argument("--gpt-verification-ledger", type=Path, default=None)
    p_auto.add_argument("--gpt-verification-ledger-report", type=Path, default=None)
    p_auto.add_argument("--gpt-market-status", type=Path, default=None)
    p_auto.add_argument("--gpt-market-status-report", type=Path, default=None)

    p_risk = sub.add_parser("risk-dry-run", help="Validate an order intent against a JSON snapshot.")
    p_risk.add_argument("--intent", type=Path, required=True)
    p_risk.add_argument("--snapshot", type=Path, required=True)
    p_risk.add_argument("--for-live-send", action="store_true")
    p_risk.add_argument("--strategy-profile", type=Path, default=DEFAULT_STRATEGY_PROFILE)
    p_risk.add_argument("--no-strategy-profile", action="store_true")
    p_risk.add_argument("--max-loss-jpy", type=float, default=None)
    p_risk.add_argument("--max-loss-pct", type=float, default=None)
    p_risk.add_argument("--risk-equity-jpy", type=float, default=None)

    args = parser.parse_args(argv)

    # SL-free runtime defaults: every cli command that touches sizing /
    # risk geometry must see the same env knobs as the wrapper exports.
    # Previously the bootstrap fired only for `autotrade-cycle`, so a
    # `generate-intents` or `stage-live-order` invocation that bypassed
    # the wrapper got 3000u BASE_UNITS instead of POSITION_NAV_PCT-derived
    # sizing (≈9,200u at NAV 226k × 30%). 2026-05-08 14:56 incident: a
    # routine cycle opened a 3-pair LONG basket at 3000u/pair instead of
    # the operator-anchored attack-mode size because intents had been
    # generated under stale env. Move the bootstrap to the top of main()
    # so every command sees the same defaults the wrapper would set.
    #
    # Bootstrap fires when either QR_LIVE_ENABLED=1 (wrapper path) or the
    # subcommand is a known live-runtime command. 2026-05-11 incident:
    # `gpt-trader-decision` invoked standalone from the SKILL flow (not
    # through run-autotrade-live.sh) had QR_LIVE_ENABLED unset, so
    # QR_TRADER_DISABLE_SL_REPAIR stayed unset, and the verifier saw
    # trader-owned TP-only EUR_USD LONG positions as "non-layerable",
    # rejecting every basket-expansion TRADE with EXPOSURE_BLOCKS_TRADE.
    # The runtime gate exists only to keep unit tests isolated; commands
    # that read real broker truth must always see SL-free defaults.
    if os.environ.get("QR_LIVE_ENABLED") == "1" or (
        args.command in _LIVE_RUNTIME_COMMANDS and not _running_under_test_harness()
    ):
        _bootstrap_sl_free_defaults()

    if args.command == "import-legacy":
        summary = LegacyImporter(args.archive, args.db, args.report).run()
        print(
            json.dumps(
                {
                    "archive": str(summary.archive),
                    "db_path": str(summary.db_path),
                    "report_path": str(summary.report_path),
                    "source_files": summary.source_files,
                    "legacy_rows": summary.legacy_rows,
                    "live_trade_events": summary.live_trade_events,
                    "journal_events": summary.journal_events,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "generate-intents":
        market_evidence_refresh = _auto_refresh_market_evidence_if_required(
            label="generate-intents",
            reuse_market_artifacts=False,
            market_context_matrix_path=args.market_context_matrix,
        )
        if not args.no_refresh_market_story:
            _refresh_market_story_if_news_is_newer(
                profile_path=args.market_story_profile,
                report_path=args.market_story_report,
                news_dir=args.market_news_dir,
            )
        forecast_refresh: dict[str, Any] | None = None
        telemetry_required = os.environ.get("QR_REQUIRE_TELEMETRY_FOR_LIVE", "").strip() in {
            "1", "true", "TRUE", "yes", "YES"
        }
        execution_ledger_sync = _pre_entry_execution_ledger_sync_if_required(
            telemetry_required=telemetry_required,
            snapshot_path=args.snapshot,
        )
        projection_verification = _pre_entry_projection_verification_if_required(
            telemetry_required=telemetry_required,
            snapshot_path=args.snapshot,
            pair_charts_path=DEFAULT_PAIR_CHARTS,
        )
        if (
            telemetry_required
            and (not _running_under_test_harness() or os.environ.get("QR_LIVE_ENABLED") == "1")
            and args.snapshot is not None
            and args.snapshot.exists()
        ):
            # IntentGenerator synthesizes the forecast used by executable lanes
            # and records that exact forecast under the pre-entry cycle id before
            # live validation. A separate CLI pre-refresh can compute a slightly
            # different calibrated confidence, win the cycle_id+pair idempotency
            # race in forecast_history.jsonl, and then make the generated intent
            # fail its own telemetry audit as a mismatch.
            forecast_refresh = {
                "status": "DELEGATED_TO_INTENT_GENERATOR",
                "reason": "avoid duplicate pre-entry forecast synthesis before live validation",
            }
        summary = IntentGenerator(
            campaign_plan=args.campaign_plan,
            strategy_profile=args.strategy_profile,
            output_path=args.output,
            report_path=args.report,
            market_context_matrix_path=args.market_context_matrix,
            max_loss_jpy=_resolve_max_loss_from_args(
                max_loss_jpy=args.max_loss_jpy,
                max_loss_pct=args.max_loss_pct,
                risk_equity_jpy=args.risk_equity_jpy,
                label="generate-intents",
            ),
        ).run(snapshot_path=args.snapshot, max_candidates=args.max_candidates)
        print(
            json.dumps(
                {
                    "output_path": str(summary.output_path),
                    "report_path": str(summary.report_path),
                    "candidates_seen": summary.candidates_seen,
                    "generated": summary.generated,
                    "needs_snapshot": summary.needs_snapshot,
                    "dry_run_passed": summary.dry_run_passed,
                    "execution_ledger_sync": execution_ledger_sync,
                    "forecast_refresh": forecast_refresh,
                    "live_ready": summary.live_ready,
                    "market_evidence_refresh": market_evidence_refresh,
                    "projection_verification": projection_verification,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "promote-receipts":
        summary = ReceiptPromoter(
            profile_path=args.profile,
            intents_path=args.intents,
            output_profile=args.output_profile,
            report_path=args.report,
        ).run()
        print(
            json.dumps(
                {
                    "profile_path": str(summary.profile_path),
                    "intents_path": str(summary.intents_path),
                    "report_path": str(summary.report_path),
                    "profiles_seen": summary.profiles_seen,
                    "receipts_seen": summary.receipts_seen,
                    "promoted": summary.promoted,
                    "still_blocked": summary.still_blocked,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "stage-live-order":
        try:
            client = OandaExecutionClient()
            ledger = ExecutionLedger(db_path=args.execution_ledger_db, report_path=args.execution_ledger_report)
            ledger.sync_oanda_transactions(client)
            summary = LiveOrderGateway(
                client=client,
                strategy_profile=args.strategy_profile,
                output_path=args.output,
                report_path=args.report,
                live_enabled=os.environ.get("QR_LIVE_ENABLED") == "1",
                max_loss_jpy=_resolve_max_loss_from_args(
                    max_loss_jpy=args.max_loss_jpy,
                    max_loss_pct=args.max_loss_pct,
                    risk_equity_jpy=args.risk_equity_jpy,
                    label="stage-live-order",
                ),
            ).run(intents_path=args.intents, lane_id=args.lane_id, send=args.send, confirm_live=args.confirm_live)
            ledger.record_gateway_receipt(kind="live_order", receipt_path=args.output)
            ledger.sync_oanda_transactions(client)
        except (RuntimeError, OSError, json.JSONDecodeError, sqlite3.Error, ValueError) as exc:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2, sort_keys=True))
            return 2
        print(
            json.dumps(
                {
                    "status": summary.status,
                    "lane_id": summary.lane_id,
                    "output_path": str(summary.output_path),
                    "report_path": str(summary.report_path),
                    "sent": summary.sent,
                    "risk_issues": summary.risk_issues,
                    "strategy_issues": summary.strategy_issues,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if summary.status in {"STAGED", "SENT"} else 2
    if args.command == "autotrade-cycle":
        # Bootstrap moved to top of main() for all commands; this branch
        # used to call it inline before the move.
        try:
            _auto_refresh_market_evidence_if_required(
                label="autotrade-cycle",
                reuse_market_artifacts=args.reuse_market_artifacts,
            )
            use_gpt_trader = args.use_gpt_trader or os.environ.get("QR_GPT_TRADER_ENABLED") == "1"
            gpt_provider = _static_gpt_provider(
                decision_response=args.gpt_decision_response,
                required=use_gpt_trader,
            )
            summary = AutoTradeCycle(
                report_path=args.report,
                target_state_path=DEFAULT_DAILY_TARGET_STATE,
                target_report_path=DEFAULT_DAILY_TARGET_REPORT,
                use_gpt_trader=use_gpt_trader,
                gpt_provider=gpt_provider,
                gpt_max_lanes=args.gpt_max_lanes,
                gpt_verification_ledger_path=args.gpt_verification_ledger,
                gpt_verification_ledger_report_path=args.gpt_verification_ledger_report,
                gpt_market_status_path=args.gpt_market_status,
                gpt_market_status_report_path=args.gpt_market_status_report,
                reuse_market_artifacts=args.reuse_market_artifacts,
                trader_settings_path=args.trader_settings,
                execution_ledger_db_path=args.execution_ledger_db,
                execution_ledger_report_path=args.execution_ledger_report,
                refresh_market_story=args.refresh_market_story,
                market_news_root=args.market_news_dir,
                live_enabled=os.environ.get("QR_LIVE_ENABLED") == "1",
                max_loss_jpy=args.max_loss_jpy,
                max_loss_pct=args.max_loss_pct,
                risk_equity_jpy=args.risk_equity_jpy,
            ).run(send=args.send)
        except (RuntimeError, OSError, ValueError, json.JSONDecodeError) as exc:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2, sort_keys=True))
            return 2
        print(
            json.dumps(
                {
                    "status": summary.status,
                    "report_path": str(summary.report_path),
                    "snapshot_path": str(summary.snapshot_path),
                    "intents_path": str(summary.intents_path),
                    "selected_lane_id": summary.selected_lane_id,
                    "selected_lane_ids": list(summary.selected_lane_ids),
                    "deterministic_lane_id": summary.deterministic_lane_id,
                    "decision_source": summary.decision_source,
                    "sent": summary.sent,
                    "sent_count": summary.sent_count,
                    "positions": summary.positions,
                    "orders": summary.orders,
                    "live_ready": summary.live_ready,
                    "receipt_promotions": summary.receipt_promotions,
                    "canceled_orders": list(summary.canceled_orders),
                    "position_management_action": summary.position_management_action,
                    "position_execution_status": summary.position_execution_status,
                    "position_execution_sent": summary.position_execution_sent,
                    "target_status": summary.target_status,
                    "target_remaining_jpy": summary.target_remaining_jpy,
                    "target_progress_pct": summary.target_progress_pct,
                    "selected_lane_score": summary.selected_lane_score,
                    "selected_lane_size_multiple": summary.selected_lane_size_multiple,
                    "gpt_status": summary.gpt_status,
                    "gpt_action": summary.gpt_action,
                    "gpt_allowed": summary.gpt_allowed,
                    "gpt_issues": summary.gpt_issues,
                    "gpt_error": summary.gpt_error,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if summary.status in _AUTOTRADE_EXIT_ZERO_STATUSES else 2
    if args.command == "plan-campaign":
        summary = CampaignPlanner(
            strategy_profile=args.strategy_profile,
            market_story_profile=args.market_story_profile,
            report_path=args.report,
            plan_path=args.plan,
        ).run(start_balance_jpy=args.start_balance, target_return_pct=args.target_return_pct)
        target_summary = DailyTargetLedger(
            state_path=DEFAULT_DAILY_TARGET_STATE,
            report_path=DEFAULT_DAILY_TARGET_REPORT,
            pace_backtest_path=DEFAULT_AI_TEST_BOT_BACKTEST,
        ).run(start_balance_jpy=args.start_balance, target_return_pct=args.target_return_pct)
        print(
            json.dumps(
                {
                    "report_path": str(summary.report_path),
                    "plan_path": str(summary.plan_path),
                    "target_state_path": str(target_summary.state_path),
                    "target_report_path": str(target_summary.report_path),
                    "target_jpy": summary.target_jpy,
                    "remaining_target_jpy": target_summary.remaining_target_jpy,
                    "lanes": summary.lanes,
                    "actionable_lanes": summary.actionable_lanes,
                    "rejected_lanes": summary.rejected_lanes,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "mine-market-stories":
        summary = MarketStoryMiner(
            archive=args.archive,
            report_path=args.report,
            profile_path=args.profile,
            news_root=args.news_dir,
            db_path=args.db,
        ).run()
        print(
            json.dumps(
                {
                    "archive": str(summary.archive),
                    "report_path": str(summary.report_path),
                    "profile_path": str(summary.profile_path),
                    "news_dir": str(args.news_dir),
                    "artifacts": summary.artifacts,
                    "story_lines": summary.story_lines,
                    "pairs": summary.pairs,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "market-status":
        status = compute_market_status()
        write_market_status_snapshot(status, args.output)
        write_market_status_report(status, args.report)
        print(
            json.dumps(
                {
                    "output_path": str(args.output),
                    "report_path": str(args.report),
                    "is_fx_open": status.is_fx_open,
                    "closed_reason": status.closed_reason,
                    "active_sessions": list(status.active_sessions),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "broker-snapshot":
        pairs = [part.strip().upper() for part in args.pairs.split(",") if part.strip()]
        try:
            snapshot = OandaReadOnlyClient().snapshot(pairs)
        except RuntimeError as exc:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2, sort_keys=True))
            return 2
        text = _snapshot_to_json(snapshot)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text + "\n")
            payload = {
                "output_path": str(args.output),
                "positions": len(snapshot.positions),
                "orders": len(snapshot.orders),
                "quotes": len(snapshot.quotes),
            }
            if snapshot.account is not None:
                payload["account"] = {
                    "nav_jpy": snapshot.account.nav_jpy,
                    "balance_jpy": snapshot.account.balance_jpy,
                    "unrealized_pl_jpy": snapshot.account.unrealized_pl_jpy,
                    "margin_used_jpy": snapshot.account.margin_used_jpy,
                    "margin_available_jpy": snapshot.account.margin_available_jpy,
                    "hedging_enabled": snapshot.account.hedging_enabled,
                }
            print(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            print(text)
        return 0
    if args.command == "pair-charts":
        from quant_rabbit.analysis.chart_reader import build_pair_chart
        from quant_rabbit.analysis.score_momentum import attach_score_momentum

        pairs = [part.strip().upper() for part in args.pairs.split(",") if part.strip()]
        timeframes = tuple(part.strip().upper() for part in args.timeframes.split(",") if part.strip())
        try:
            client = OandaReadOnlyClient()
        except RuntimeError as exc:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2, sort_keys=True))
            return 2

        charts = []
        for pair in pairs:
            chart = build_pair_chart(pair, client=client, timeframes=timeframes, count=args.count)
            charts.append(chart)

        generated_at = datetime.now(timezone.utc).isoformat()
        chart_payloads = [chart.to_dict() for chart in charts]
        previous_pair_charts = None
        if args.output and args.output.exists():
            try:
                previous_pair_charts = json.loads(args.output.read_text())
            except (OSError, json.JSONDecodeError, ValueError):
                previous_pair_charts = None
        attach_score_momentum(chart_payloads, previous_pair_charts, generated_at)
        chart_payloads.sort(key=lambda c: max(c["long_score"], c["short_score"]), reverse=True)
        output_payload = {
            "generated_at_utc": generated_at,
            "timeframes": list(timeframes),
            "candle_count": int(args.count),
            "charts": chart_payloads,
        }
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            lines = [
                "# Pair Charts Report",
                "",
                f"- Generated at UTC: `{generated_at}`",
                f"- Timeframes: `{','.join(timeframes)}`",
                f"- Candles per timeframe: `{args.count}`",
                "",
                "## Pair Score Table",
                "",
                "| Pair | Side | Long | Short | Momentum | Regime | Story |",
                "|---|---|---|---|---|---|---|",
            ]
            for c in chart_payloads:
                side = "LONG" if c["long_score"] >= c["short_score"] else "SHORT"
                story = c["chart_story"].replace("|", "/")
                confluence = c.get("confluence") if isinstance(c.get("confluence"), dict) else {}
                momentum = (confluence or {}).get("score_momentum") if isinstance((confluence or {}).get("score_momentum"), dict) else {}
                momentum_cell = "-"
                if momentum:
                    momentum_cell = (
                        f"{momentum.get('direction', 'FLAT')} "
                        f"gapΔ={float(momentum.get('score_gap_delta', 0.0)):+.3f}/"
                        f"{float(momentum.get('elapsed_min', 0.0)):.0f}m"
                    )
                lines.append(
                    f"| `{c['pair']}` | `{side}` | `{c['long_score']:.3f}` | `{c['short_score']:.3f}` | `{momentum_cell}` | `{c['dominant_regime']}` | {story} |"
                )
            lines.extend([
                "",
                "## How To Read",
                "",
                "- Long/Short scores are 0..1 indicator-agreement values weighted by timeframe (D>H4>H1>M30>M15>M5>M1).",
                "- Momentum is the previous-snapshot slope of long-short score_gap; UP means the chart's evidence is rotating toward long, even if the current score still leans short.",
                "- A high score is a *signal of where the chart leans*, not an order. The trader still chooses.",
                "- Regime is the dominant tag across timeframes (TREND_UP/DOWN, RANGE, IMPULSE_UP/DOWN, FAILURE_RISK, UNCLEAR).",
                "- Pairs are sorted by max(long, short); the top entries are where edges line up.",
            ])
            args.report.write_text("\n".join(lines) + "\n")

        print(json.dumps({
            "output_path": str(args.output) if args.output else None,
            "report_path": str(args.report) if args.report else None,
            "pairs": len(charts),
            "top": [
                {"pair": c["pair"], "side": "LONG" if c["long_score"] >= c["short_score"] else "SHORT",
                 "long": round(c["long_score"], 3), "short": round(c["short_score"], 3), "regime": c["dominant_regime"],
                 "momentum": ((c.get("confluence") or {}).get("score_momentum") or {}).get("direction")}
                for c in chart_payloads[:5]
            ],
        }, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "cross-asset-snapshot":
        from quant_rabbit.analysis.cross_asset import (
            DEFAULT_CROSS_ASSET_INSTRUMENTS,
            DEFAULT_CORRELATION_PAIRS,
            build_cross_asset_snapshot,
        )
        try:
            client = OandaReadOnlyClient()
        except RuntimeError as exc:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2, sort_keys=True))
            return 2
        instruments = (
            tuple(p.strip() for p in args.instruments.split(",") if p.strip())
            if args.instruments else DEFAULT_CROSS_ASSET_INSTRUMENTS
        )
        corr_pairs = tuple(p.strip().upper() for p in args.correlation_pairs.split(",") if p.strip())
        snap = build_cross_asset_snapshot(
            client=client, instruments=instruments, correlation_pairs=corr_pairs,
            granularity=args.granularity, count=int(args.count),
        )
        payload = snap.to_dict()
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        if args.report:
            lines = [
                "# Cross-Asset Snapshot",
                "",
                f"- Generated at UTC: `{snap.generated_at_utc}`",
                f"- Granularity: `{snap.granularity}`",
                f"- Candles per series: `{snap.candle_count}`",
                "",
                "## Synthetic DXY",
            ]
            sd = snap.synthetic_dxy
            if sd:
                lines.append(f"- last={sd.last_value}, Δ24h={sd.change_pct_24h}%, Δ5d={sd.change_pct_5d}%, components={list(sd.components_used)}")
            else:
                lines.append("- (insufficient basket coverage)")
            lines.extend(["", "## Yield Spreads", ""])
            for y in snap.yield_spreads:
                lines.append(f"- **{y.name}**: a={y.a_last}, b={y.b_last}, spread={y.spread_last}, Δ24h={y.spread_change_24h}, issue={y.issue}")
            lines.extend(["", "## Asset Readings", "", "| Instrument | Last | Δ24h% | Δ5d% | Z(60) | RV(60) | Trend |", "|---|---|---|---|---|---|---|"])
            for a in snap.assets:
                lines.append(f"| `{a.instrument}` | {a.last_price} | {a.change_pct_24h} | {a.change_pct_5d} | {a.z_score_60} | {a.realized_vol_60} | {a.trend_label} |")
            if snap.issues:
                lines.extend(["", "## Issues", ""] + [f"- {i}" for i in snap.issues])
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text("\n".join(lines) + "\n")
        print(json.dumps({
            "output_path": str(args.output),
            "report_path": str(args.report),
            "assets_fetched": sum(1 for a in snap.assets if a.fetched),
            "issues": list(snap.issues),
            "synthetic_dxy_last": snap.synthetic_dxy.last_value if snap.synthetic_dxy else None,
        }, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "context-asset-charts":
        from quant_rabbit.analysis.context_assets import (
            build_context_asset_charts,
            write_context_asset_charts_report,
        )
        try:
            client = OandaReadOnlyClient()
        except RuntimeError as exc:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2, sort_keys=True))
            return 2
        instruments = (
            tuple(p.strip().upper() for p in args.instruments.split(",") if p.strip())
            if args.instruments
            else DEFAULT_CONTEXT_ASSETS
        )
        timeframes = tuple(p.strip().upper() for p in args.timeframes.split(",") if p.strip())
        payload = build_context_asset_charts(
            client=client,
            instruments=instruments,
            timeframes=timeframes or DEFAULT_PAIR_CHART_TIMEFRAMES,
            count=int(args.count),
        )
        if args.output:
            _write_json(args.output, payload)
        if args.report:
            write_context_asset_charts_report(payload, args.report)
        print(json.dumps({
            "output_path": str(args.output),
            "report_path": str(args.report),
            "assets": len(payload.get("charts") or []),
            "issues": list(payload.get("issues") or [])[:10],
            "role": payload.get("role"),
        }, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "flow-snapshot":
        from quant_rabbit.analysis.flow import build_flow_snapshot
        try:
            client = OandaReadOnlyClient()
        except RuntimeError as exc:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2, sort_keys=True))
            return 2
        pairs = tuple(p.strip().upper() for p in args.pairs.split(",") if p.strip())
        snap = build_flow_snapshot(
            client=client, pairs=pairs,
            top_n=int(args.top_n), spread_lookback_minutes=int(args.spread_lookback_minutes),
            include_books=bool(args.include_books),
        )
        payload = snap.to_dict()
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        if args.report:
            lines = [
                "# Flow Snapshot (Spread + Optional Books)",
                "",
                f"- Generated at UTC: `{snap.generated_at_utc}`",
                f"- Book fetch enabled: `{snap.book_fetch_enabled}`",
            ]
            if snap.book_fetch_reason:
                lines.append(f"- Book fetch reason: `{snap.book_fetch_reason}`")
            lines.extend([
                "",
                "## Spread State",
                "",
                "| Pair | Current(p) | Median(p) | P90(p) | Max(p) | Samples | Stress |",
                "|---|---|---|---|---|---|---|",
            ])
            for s in snap.spreads:
                lines.append(f"| `{s.instrument}` | {s.current_pips} | {s.median_pips} | {s.p90_pips} | {s.max_pips} | {s.sample_size} | `{s.stress_flag}` |")
            if not snap.book_fetch_enabled:
                lines.extend([
                    "",
                    "## Book Data",
                    "",
                    "- Disabled by default; enable with `--include-books` only after OANDA book entitlement is confirmed.",
                ])
            else:
                lines.extend(["", "## Position Book Top Clusters", ""])
                position_book_issues = [pb.issue for pb in snap.position_books if pb.issue]
                unique_position_book_issues = sorted({i for i in position_book_issues if i})
                if (
                    len(unique_position_book_issues) == 1
                    and len(position_book_issues) == len(snap.position_books)
                ):
                    lines.append(f"- all pairs issue: {unique_position_book_issues[0]}")
                else:
                    for pb in snap.position_books:
                        if pb.issue:
                            lines.append(f"- `{pb.instrument}` issue: {pb.issue}")
                            continue
                        top_long = ", ".join(f"{b.price}@{b.long_pct:.1f}%" for b in pb.top_long_clusters[:3])
                        top_short = ", ".join(f"{b.price}@{b.short_pct:.1f}%" for b in pb.top_short_clusters[:3])
                        lines.append(f"- `{pb.instrument}` price={pb.price} long_total={pb.long_total_pct:.1f}% short_total={pb.short_total_pct:.1f}% top_long=[{top_long}] top_short=[{top_short}]")
            if snap.issues:
                lines.extend(["", "## Issues", ""] + [f"- {i}" for i in snap.issues])
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text("\n".join(lines) + "\n")
        print(json.dumps({
            "output_path": str(args.output), "report_path": str(args.report),
            "pairs": len(pairs),
            "book_fetch_enabled": snap.book_fetch_enabled,
            "book_fetch_reason": snap.book_fetch_reason,
            "issues": list(snap.issues),
        }, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "currency-strength":
        from quant_rabbit.analysis.strength import build_strength_snapshot
        try:
            client = OandaReadOnlyClient()
        except RuntimeError as exc:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2, sort_keys=True))
            return 2
        snap = build_strength_snapshot(
            client=client, granularity=args.granularity,
            lookback_bars=int(args.lookback_bars), fetch_count=int(args.fetch_count),
        )
        payload = snap.to_dict()
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        if args.report:
            lines = [
                "# Currency Strength",
                "",
                f"- Generated at UTC: `{snap.generated_at_utc}`",
                f"- Granularity: `{snap.granularity}` over `{snap.lookback_bars}` bars",
                f"- Pairs used: {len(snap.pairs_used)} / missing {len(snap.pairs_missing)}",
                f"- Suggested cross: `{snap.strongest_pair_suggestion}`",
                "",
                "| Rank | Currency | Score (%) |",
                "|---|---|---|",
            ]
            for s in snap.scores:
                lines.append(f"| {s.rank} | `{s.currency}` | {s.score_pct:.3f} |")
            if snap.issues:
                lines.extend(["", "## Issues", ""] + [f"- {i}" for i in snap.issues])
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text("\n".join(lines) + "\n")
        print(json.dumps({
            "output_path": str(args.output), "report_path": str(args.report),
            "suggestion": snap.strongest_pair_suggestion,
            "issues": list(snap.issues)[:5],
        }, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "levels-snapshot":
        from quant_rabbit.analysis.levels import build_levels_snapshot
        try:
            client = OandaReadOnlyClient()
        except RuntimeError as exc:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2, sort_keys=True))
            return 2
        pairs = tuple(p.strip().upper() for p in args.pairs.split(",") if p.strip())
        snap = build_levels_snapshot(client=client, pairs=pairs)
        payload = snap.to_dict()
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        if args.report:
            lines = [
                "# Levels Snapshot",
                "",
                f"- Generated at UTC: `{snap.generated_at_utc}`",
                "",
                "## Per-Pair",
                "",
            ]
            for p in snap.pairs:
                lines.append(f"### `{p.pair}`")
                lines.append(f"- last={p.last_close}, PDH={p.pdh}, PDL={p.pdl}, PDC={p.pdc}, daily_open={p.daily_open}, weekly_open={p.weekly_open}, monthly_open={p.monthly_open}")
                for piv in p.pivots:
                    lines.append(f"- `{piv.style}`: PP={piv.pp}, R1={piv.r1}, R2={piv.r2}, R3={piv.r3}, S1={piv.s1}, S2={piv.s2}, S3={piv.s3}")
                for s in p.sessions:
                    lines.append(f"- session `{s.name}`: H={s.high}, L={s.low}, range={s.range_pips}p")
                rn_near = [r for r in p.round_numbers if abs(r.distance_pips) < 50][:5]
                lines.append(f"- nearby round-numbers: " + ", ".join(f"{r.price}({r.distance_pips:.1f}p)" for r in rn_near))
                lines.append("")
            if snap.issues:
                lines.extend(["## Issues", ""] + [f"- {i}" for i in snap.issues])
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text("\n".join(lines) + "\n")
        print(json.dumps({
            "output_path": str(args.output), "report_path": str(args.report),
            "pairs": len(snap.pairs), "issues": list(snap.issues),
        }, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "market-context-matrix":
        from quant_rabbit.analysis.market_context_matrix import (
            build_market_context_matrix,
            write_market_context_matrix_report,
        )

        payload = build_market_context_matrix(
            pair_charts_path=args.pair_charts,
            cross_asset_path=args.cross_asset,
            flow_path=args.flow,
            currency_strength_path=args.currency_strength,
            levels_path=args.levels,
            calendar_path=args.calendar,
            cot_path=args.cot,
            option_skew_path=args.option_skew,
            context_asset_charts_path=args.context_asset_charts,
        )
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        if args.report:
            write_market_context_matrix_report(payload, args.report)
        pair_count = len(payload.get("pairs") or {})
        side_rows = [
            reading
            for side_map in (payload.get("pairs") or {}).values()
            if isinstance(side_map, dict)
            for reading in side_map.values()
            if isinstance(reading, dict)
        ]
        print(json.dumps({
            "output_path": str(args.output),
            "report_path": str(args.report),
            "pairs": pair_count,
            "trade_count_policy": payload.get("trade_count_policy"),
            "supports": sum(int(row.get("support_count") or 0) for row in side_rows),
            "rejects": sum(int(row.get("reject_count") or 0) for row in side_rows),
            "warnings": sum(int(row.get("warning_count") or 0) for row in side_rows),
            "missing": sum(int(row.get("missing_count") or 0) for row in side_rows),
            "issues": list(payload.get("issues") or [])[:10],
        }, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "news-snapshot":
        from quant_rabbit.analysis.news import (
            DEFAULT_DIGEST_ITEMS,
            DEFAULT_FLOW_ENTRIES,
            DEFAULT_LOOKBACK_HOURS,
            DEFAULT_MAX_ITEMS,
            build_news_snapshot,
            write_news_artifacts,
        )

        snap = build_news_snapshot(
            lookback_hours=args.lookback_hours or DEFAULT_LOOKBACK_HOURS,
            max_items=args.max_items or DEFAULT_MAX_ITEMS,
            fetch=not args.no_fetch,
        )
        write_news_artifacts(
            snap,
            output_path=args.output,
            digest_path=args.digest,
            flow_log_path=args.flow_log,
            digest_items=args.digest_items or DEFAULT_DIGEST_ITEMS,
            flow_entries=args.flow_entries or DEFAULT_FLOW_ENTRIES,
        )
        print(
            json.dumps(
                {
                    "output_path": str(args.output),
                    "digest_path": str(args.digest),
                    "flow_log_path": str(args.flow_log),
                    "items": len(snap.items),
                    "issues": list(snap.issues),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "news-health":
        from quant_rabbit.analysis.news_health import build_news_health, write_news_health_report

        now_override = None
        if args.now_utc:
            try:
                now_override = datetime.fromisoformat(str(args.now_utc).replace("Z", "+00:00"))
            except ValueError:
                print(json.dumps({"error": f"invalid --now-utc: {args.now_utc}"}, ensure_ascii=False), file=sys.stdout)
                return 2
        payload = build_news_health(
            news_items_path=args.news_items,
            digest_path=args.digest,
            flow_log_path=args.flow_log,
            market_story_profile_path=args.market_story_profile,
            calendar_path=args.calendar,
            automation_path=args.automation,
            weekend_state_path=args.weekend_state,
            now_utc=now_override,
            verify_fetch=args.verify_fetch,
        )
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        if args.report:
            write_news_health_report(payload, args.report)
        print(
            json.dumps(
                {
                    "output_path": str(args.output),
                    "report_path": str(args.report),
                    "status": payload.get("status"),
                    "market_window": payload.get("market_window"),
                    "issues": list(payload.get("issues") or []),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        if args.strict and payload.get("status") == "BLOCK":
            return 2
        return 0
    if args.command == "trader-prompt-route":
        try:
            route = route_trader_prompts(
                snapshot_path=args.snapshot,
                target_state_path=args.target_state,
                intents_path=args.intents,
                pair_charts_path=args.pair_charts,
                cross_asset_path=args.cross_asset,
                flow_path=args.flow,
                currency_strength_path=args.currency_strength,
                levels_path=args.levels,
                market_context_matrix_path=args.market_context_matrix,
                calendar_path=args.calendar,
                cot_path=args.cot,
                option_skew_path=args.option_skew,
                attack_advice_path=args.attack_advice,
                learning_audit_path=args.learning_audit,
                memory_health_path=args.memory_health,
                strategy_profile_path=args.strategy_profile,
                trader_overrides_path=args.trader_overrides,
                decision_response_path=args.decision_response,
                gpt_decision_path=args.gpt_decision,
                live_order_path=args.live_order,
                position_execution_path=args.position_execution,
                autotrade_report_path=args.autotrade_report,
                include_content=args.include_content,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2, sort_keys=True))
            return 2
        print(json.dumps(route.to_payload(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "economic-calendar":
        from quant_rabbit.analysis.calendar import build_calendar_snapshot
        pairs = tuple(p.strip().upper() for p in args.pairs.split(",") if p.strip())
        impact = tuple(p.strip() for p in args.impact.split(",") if p.strip())
        snap = build_calendar_snapshot(
            pairs=pairs, pre_minutes=int(args.pre_minutes), post_minutes=int(args.post_minutes),
            impact_filter=impact, fetch=not args.no_fetch,
        )
        payload = snap.to_dict()
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        if args.report:
            lines = [
                "# Economic Calendar",
                "",
                f"- Generated at UTC: `{snap.generated_at_utc}`",
                f"- Source: `{snap.source_url}`",
                f"- Events parsed: {len(snap.events)}",
                "",
                "## Pair Windows",
                "",
                "| Pair | In Window | Reason |",
                "|---|---|---|",
            ]
            for w in snap.pair_windows:
                lines.append(f"| `{w.pair}` | {'YES' if w.in_window else 'no'} | {w.reason} |")
            lines.extend(["", "## Upcoming High/Medium Events (first 30)", "", "| Time UTC | Currency | Impact | Title | Forecast | Previous |", "|---|---|---|---|---|---|"])
            for e in list(snap.events)[:30]:
                if e.impact in ("High", "Medium"):
                    lines.append(f"| `{e.timestamp_utc}` | `{e.currency}` | `{e.impact}` | {e.title} | {e.forecast or ''} | {e.previous or ''} |")
            if snap.issues:
                lines.extend(["", "## Issues", ""] + [f"- {i}" for i in snap.issues])
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text("\n".join(lines) + "\n")
        print(json.dumps({
            "output_path": str(args.output), "report_path": str(args.report),
            "events": len(snap.events),
            "in_window_pairs": [w.pair for w in snap.pair_windows if w.in_window],
            "issues": list(snap.issues),
        }, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "cot-snapshot":
        from quant_rabbit.analysis.cot import build_cot_snapshot
        snap = build_cot_snapshot(fetch=not args.no_fetch)
        payload = snap.to_dict()
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        if args.report:
            lines = [
                "# COT Snapshot",
                "",
                f"- Generated at UTC: `{snap.generated_at_utc}`",
                f"- Source: `{snap.source_url}`",
                f"- Reports: {len(snap.reports)}",
                "",
                "| Currency | Report Date | Lev Long | Lev Short | Lev Net | Δw Lev Net | OI |",
                "|---|---|---|---|---|---|---|",
            ]
            for r in snap.reports:
                lines.append(f"| `{r.currency}` | `{r.report_date}` | {r.leveraged_long} | {r.leveraged_short} | {r.leveraged_net} | {r.week_change_leveraged_net} | {r.open_interest} |")
            if snap.issues:
                lines.extend(["", "## Issues", ""] + [f"- {i}" for i in snap.issues])
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text("\n".join(lines) + "\n")
        print(json.dumps({
            "output_path": str(args.output), "report_path": str(args.report),
            "reports": len(snap.reports), "issues": list(snap.issues),
        }, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "daily-review":
        from quant_rabbit.strategy.daily_review import (
            DAILY_REVIEW_LOOKBACK_HOURS,
            compute_daily_review,
            write_trader_overrides,
        )
        lookback = args.lookback_hours if args.lookback_hours is not None else DAILY_REVIEW_LOOKBACK_HOURS
        report = compute_daily_review(args.ledger_db, lookback_hours=lookback)
        if not args.dry_run:
            write_trader_overrides(report, args.output)
        print(json.dumps({
            "output_path": str(args.output) if not args.dry_run else "(dry-run, not written)",
            "expires_at_utc": report.expires_at_utc,
            "narrative_summary": report.narrative_summary,
            "bias_overrides": report.bias_overrides,
            "blocked_lanes": report.blocked_lanes,
            "lookback_hours": lookback,
        }, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "option-skew":
        from quant_rabbit.analysis.options import build_option_skew_snapshot
        pairs = tuple(p.strip().upper() for p in args.pairs.split(",") if p.strip())
        tenors = tuple(t.strip() for t in args.tenors.split(",") if t.strip())
        snap = build_option_skew_snapshot(pairs=pairs, tenors=tenors)
        payload = snap.to_dict()
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        if args.report:
            lines = [
                "# Option Skew Snapshot",
                "",
                f"- Generated at UTC: `{snap.generated_at_utc}`",
                f"- Provider: `{snap.provider}`",
                f"- Enabled: `{snap.enabled}`",
            ]
            if snap.disabled_reason:
                lines.append(f"- Disabled reason: `{snap.disabled_reason}`")
            lines.extend([
                "",
                "| Pair | Tenor | ATM IV | RR 25Δ | BF 25Δ | Source | Issue |",
                "|---|---|---|---|---|---|---|",
            ])
            for r in snap.readings:
                lines.append(f"| `{r.pair}` | `{r.tenor}` | {r.atm_iv} | {r.rr_25d} | {r.bf_25d} | {r.source or ''} | {r.issue or ''} |")
            if snap.issues:
                lines.extend(["", "## Issues", ""] + [f"- {i}" for i in snap.issues])
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text("\n".join(lines) + "\n")
        print(json.dumps({
            "output_path": str(args.output), "report_path": str(args.report),
            "enabled": snap.enabled,
            "disabled_reason": snap.disabled_reason,
            "readings": len(snap.readings), "issues": list(snap.issues),
        }, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "mine-strategy":
        summary = StrategyMiner(
            args.db,
            args.report,
            args.profile,
            loss_cap_jpy=args.loss_cap_jpy,
            target_state_path=args.target_state,
        ).run()
        print(
            json.dumps(
                {
                    "db_path": str(summary.db_path),
                    "report_path": str(summary.report_path),
                    "profile_path": str(summary.profile_path),
                    "profiles": summary.profiles,
                    "blocked": summary.blocked,
                    "candidates": summary.candidates,
                    "risk_repair_candidates": summary.risk_repair_candidates,
                    "mined_missed_edges": summary.mined_missed_edges,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "daily-target-state":
        try:
            summary = DailyTargetLedger(
                state_path=args.state,
                report_path=args.report,
                pace_backtest_path=DEFAULT_AI_TEST_BOT_BACKTEST,
                execution_ledger_path=args.execution_ledger_db,
            ).run(
                start_balance_jpy=args.start_balance,
                target_return_pct=args.target_return_pct,
                target_profit_jpy=args.target_profit_jpy,
                realized_pl_jpy=args.realized_pl,
                daily_risk_budget_jpy=args.daily_risk_budget,
                daily_risk_pct=args.daily_risk_pct,
                target_trades_per_day=args.target_trades_per_day,
                snapshot_path=args.snapshot,
            )
        except ValueError as exc:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2, sort_keys=True))
            return 2
        print(
            json.dumps(
                {
                    "state_path": str(summary.state_path),
                    "report_path": str(summary.report_path),
                    "status": summary.status,
                    "target_jpy": summary.target_jpy,
                    "target_profit_jpy": summary.target_profit_jpy,
                    "progress_jpy": summary.progress_jpy,
                    "progress_pct": summary.progress_pct,
                    "remaining_target_jpy": summary.remaining_target_jpy,
                    "remaining_risk_budget_jpy": summary.remaining_risk_budget_jpy,
                    "target_trades_per_day": summary.target_trades_per_day,
                    "target_trades_per_day_source": summary.target_trades_per_day_source,
                    "target_trades_per_day_basis_return_pct": summary.target_trades_per_day_basis_return_pct,
                    "per_trade_risk_budget_jpy": summary.per_trade_risk_budget_jpy,
                    "unprotected_positions": summary.unprotected_positions,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "replay-backtest":
        summary = ReplayBacktester(
            db_path=args.db,
            output_path=args.output,
            report_path=args.report,
            max_loss_jpy=args.max_loss,
            daily_risk_pct=args.daily_risk_pct,
            target_trades_per_day=args.target_trades_per_day,
            target_state_path=args.target_state,
        ).run(
            start_balance_jpy=args.start_balance,
            target_return_pct=args.target_return_pct,
            max_days=args.max_days,
        )
        print(
            json.dumps(
                {
                    "output_path": str(summary.output_path),
                    "report_path": str(summary.report_path),
                    "days": summary.days,
                    "target_jpy": summary.target_jpy,
                    "historical_target_hits": summary.historical_target_hits,
                    "evidence_target_covered": summary.evidence_target_covered,
                    "risk_repair_days": summary.risk_repair_days,
                    "missed_edge_days": summary.missed_edge_days,
                    "total_historical_net_jpy": summary.total_historical_net_jpy,
                    "total_risk_capped_net_jpy": summary.total_risk_capped_net_jpy,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "ai-test-bot-backtest":
        try:
            source_tables = tuple(item.strip() for item in str(args.source_tables).split(",") if item.strip())
            summary = AITestBotBacktester(
                db_path=args.db,
                output_path=args.output,
                report_path=args.report,
                target_state_path=args.target_state,
                max_loss_jpy=args.max_loss,
                daily_risk_pct=args.daily_risk_pct,
                target_trades_per_day=args.target_trades_per_day,
                training_days=args.training_days,
                min_train_trades=args.min_train_trades,
                min_train_win_rate_pct=args.min_train_win_rate_pct,
                max_active_buckets=args.max_active_buckets,
                source_tables=source_tables,
                execution_ledger_db_path=args.execution_ledger_db,
                dedupe_opportunities=not args.no_dedupe_opportunities,
            ).run(
                start_balance_jpy=args.start_balance,
                target_return_pct=args.target_return_pct,
                max_validation_days=args.max_validation_days,
            )
        except (OSError, sqlite3.Error, json.JSONDecodeError, ValueError) as exc:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2, sort_keys=True))
            return 2
        print(
            json.dumps(
                {
                    "status": summary.status,
                    "output_path": str(summary.output_path),
                    "report_path": str(summary.report_path),
                    "validation_days": summary.validation_days,
                    "traded_days": summary.traded_days,
                    "target_hit_days": summary.target_hit_days,
                    "total_managed_net_jpy": summary.total_managed_net_jpy,
                    "profit_factor": summary.profit_factor,
                    "blockers": summary.blockers,
                    "live_permission": False,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "build-outcome-mart":
        try:
            summary = OutcomeMartBuilder(
                db_path=args.db,
                execution_ledger_db_path=args.execution_ledger_db,
                output_path=args.output,
                report_path=args.report,
            ).run()
        except (OSError, sqlite3.Error, json.JSONDecodeError, ValueError) as exc:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2, sort_keys=True))
            return 2
        print(
            json.dumps(
                {
                    "status": summary.status,
                    "output_path": str(summary.output_path),
                    "report_path": str(summary.report_path),
                    "archive_outcomes": summary.archive_outcomes,
                    "execution_ledger_outcomes": summary.execution_ledger_outcomes,
                    "story_observations": summary.story_observations,
                    "condition_edges": summary.condition_edges,
                    "condition_rollups": summary.condition_rollups,
                    "validated_condition_outcomes": summary.validated_condition_outcomes,
                    "condition_directional_hit_rate_pct": summary.condition_directional_hit_rate_pct,
                    "method_edges": summary.method_edges,
                    "setup_buckets": summary.setup_buckets,
                    "context_feature_edges": summary.context_feature_edges,
                    "context_feature_outcomes": summary.context_feature_outcomes,
                    "context_feature_coverage_pct": summary.context_feature_coverage_pct,
                    "live_permission": False,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if summary.status == "OUTCOME_MART_READY" else 2
    if args.command == "optimize-coverage":
        summary = CoverageOptimizer(
            intents_path=args.intents,
            target_state_path=args.target_state,
            replay_path=args.replay,
            ai_backtest_path=args.ai_backtest,
            strategy_profile_path=args.strategy_profile,
            market_context_matrix_path=args.market_context_matrix,
            output_path=args.output,
            report_path=args.report,
        ).run()
        print(
            json.dumps(
                {
                    "status": summary.status,
                    "output_path": str(summary.output_path),
                    "report_path": str(summary.report_path),
                    "remaining_target_jpy": summary.remaining_target_jpy,
                    "live_ready_reward_jpy": summary.live_ready_reward_jpy,
                    "sequential_ladder_reward_jpy": summary.sequential_ladder_reward_jpy,
                    "sequential_ladder_steps": summary.sequential_ladder_steps,
                    "potential_reward_jpy": summary.potential_reward_jpy,
                    "live_ready_lanes": summary.live_ready_lanes,
                    "promotion_candidate_lanes": summary.promotion_candidate_lanes,
                    "action_items": summary.action_items,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        # Coverage gaps are diagnostic output for the trader refresh packet,
        # not a shell failure. The playbook runs under set -e and still needs
        # downstream advice/audit/memory reports when no LIVE_READY lane exists.
        return 0
    if args.command == "replay-execution":
        try:
            summary = ExecutionReplayer(
                intents_path=args.intents,
                price_path=args.prices,
                output_path=args.output,
                report_path=args.report,
            ).run(target_jpy=args.target_jpy, lane_id=args.lane_id)
        except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2, sort_keys=True))
            return 2
        print(
            json.dumps(
                {
                    "status": summary.status,
                    "output_path": str(summary.output_path),
                    "report_path": str(summary.report_path),
                    "orders": summary.orders,
                    "filled": summary.filled,
                    "closed": summary.closed,
                    "target_hit": summary.target_hit,
                    "net_pl_jpy": summary.net_pl_jpy,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if summary.status in {"TARGET_HIT", "REPLAY_COMPLETE"} else 2
    if args.command == "learn-post-trade":
        summary = PostTradeLearner(
            outcome_path=args.outcome,
            live_order_path=args.live_order,
            position_execution_path=args.position_execution,
            trader_decision_path=args.trader_decision,
            gpt_decision_path=args.gpt_decision,
            output_path=args.output,
            report_path=args.report,
        ).run()
        print(
            json.dumps(
                {
                    "status": summary.status,
                    "output_path": str(summary.output_path),
                    "report_path": str(summary.report_path),
                    "candidates": summary.candidates,
                    "profile_update_candidates": summary.profile_update_candidates,
                    "blockers": summary.blockers,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if summary.status == "READY_FOR_REVIEW" else 2
    if args.command == "execution-ledger-sync":
        try:
            summary = ExecutionLedger(db_path=args.db, report_path=args.report).sync_oanda_transactions(
                OandaExecutionClient(),
                since_transaction_id=args.since_transaction_id,
            )
        except (RuntimeError, OSError, json.JSONDecodeError, sqlite3.Error, ValueError) as exc:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2, sort_keys=True))
            return 2
        print(
            json.dumps(
                {
                    "status": summary.status,
                    "db_path": str(summary.db_path),
                    "report_path": str(summary.report_path),
                    "transactions_seen": summary.transactions_seen,
                    "transactions_inserted": summary.transactions_inserted,
                    "events_inserted": summary.events_inserted,
                    "gateway_receipts_inserted": summary.gateway_receipts_inserted,
                    "baseline_transaction_id": summary.baseline_transaction_id,
                    "last_transaction_id": summary.last_transaction_id,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "verification-ledger-audit":
        try:
            from quant_rabbit.verification_ledger import VerificationLedger

            summary = VerificationLedger(db_path=args.db, output_path=args.output, report_path=args.report).run(
                snapshot_path=args.snapshot,
                order_intents_path=args.order_intents,
                gpt_decision_path=args.gpt_decision,
                live_order_path=args.live_order,
                position_execution_path=args.position_execution,
                thesis_evolution_path=args.thesis_evolution,
                position_thesis_path=args.position_thesis,
                forecast_persistence_path=args.forecast_persistence,
                ai_backtest_path=args.ai_backtest,
                outcome_mart_path=args.outcome_mart,
                post_trade_learning_path=args.post_trade_learning,
                ai_attack_advice_path=args.ai_attack_advice,
                learning_audit_path=args.learning_audit,
                window_hours=args.window_hours,
            )
        except (OSError, json.JSONDecodeError, sqlite3.Error, ValueError) as exc:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2, sort_keys=True))
            return 2
        print(
            json.dumps(
                {
                    "status": summary.status,
                    "db_path": str(summary.db_path),
                    "output_path": str(summary.output_path),
                    "report_path": str(summary.report_path),
                    "observations_inserted": summary.observations_inserted,
                    "measurements_inserted": summary.measurements_inserted,
                    "blocking_observations": summary.blocking_observations,
                    "missing_observations": summary.missing_observations,
                    "closed_trades": summary.closed_trades,
                    "net_jpy": summary.net_jpy,
                    "profit_factor": summary.profit_factor,
                    "win_rate": summary.win_rate,
                    "expectancy_jpy": summary.expectancy_jpy,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "learning-audit":
        try:
            from quant_rabbit.learning_audit import LearningAuditor

            summary = LearningAuditor(
                db_path=args.db,
                output_path=args.output,
                report_path=args.report,
            ).run(
                ai_backtest_path=args.ai_backtest,
                outcome_mart_path=args.outcome_mart,
                post_trade_learning_path=args.post_trade_learning,
                ai_attack_advice_path=args.ai_attack_advice,
                window_hours=args.window_hours,
                min_effect_sample=args.min_effect_sample,
            )
        except (OSError, json.JSONDecodeError, sqlite3.Error, ValueError) as exc:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2, sort_keys=True))
            return 2
        print(
            json.dumps(
                {
                    "status": summary.status,
                    "output_path": str(summary.output_path),
                    "report_path": str(summary.report_path),
                    "db_path": str(summary.db_path),
                    "checks": summary.checks,
                    "blockers": summary.blockers,
                    "warnings": summary.warnings,
                    "influenced_lanes": summary.influenced_lanes,
                    "total_learning_score_delta": summary.total_learning_score_delta,
                    "closed_trades": summary.closed_trades,
                    "net_jpy": summary.net_jpy,
                    "profit_factor": summary.profit_factor,
                    "expectancy_jpy": summary.expectancy_jpy,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if summary.status != "LEARNING_AUDIT_BLOCKED" else 2
    if args.command == "memory-health":
        try:
            from quant_rabbit.memory_health import MemoryHealthAuditor

            summary = MemoryHealthAuditor(
                output_path=args.output,
                report_path=args.report,
            ).run(
                snapshot_path=args.snapshot,
                target_state_path=args.target_state,
                order_intents_path=args.order_intents,
                strategy_profile_path=args.strategy_profile,
                forecast_history_path=args.forecast_history,
                projection_ledger_path=args.projection_ledger,
                learning_audit_path=args.learning_audit,
                entry_thesis_ledger_path=args.entry_thesis_ledger,
                execution_ledger_db_path=args.execution_ledger_db,
            )
        except (OSError, json.JSONDecodeError, sqlite3.Error, ValueError) as exc:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2, sort_keys=True))
            return 2
        print(
            json.dumps(
                {
                    "status": summary.status,
                    "output_path": str(summary.output_path),
                    "report_path": str(summary.report_path),
                    "issues": summary.issues,
                    "blockers": summary.blockers,
                    "warnings": summary.warnings,
                    "layers": summary.layers,
                    "metrics": summary.metrics,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "self-improvement-audit":
        try:
            from quant_rabbit.self_improvement_audit import SelfImprovementAuditor

            summary = SelfImprovementAuditor(
                db_path=args.db,
                history_db_path=args.history_db,
                output_path=args.output,
                report_path=args.report,
            ).run(
                snapshot_path=args.snapshot,
                target_state_path=args.target_state,
                order_intents_path=args.order_intents,
                market_context_matrix_path=args.market_context_matrix,
                memory_health_path=args.memory_health,
                learning_audit_path=args.learning_audit,
                ai_test_bot_backtest_path=args.ai_test_bot_backtest,
                verification_ledger_path=args.verification_ledger,
                forecast_history_path=args.forecast_history,
                projection_ledger_path=args.projection_ledger,
                entry_thesis_ledger_path=args.entry_thesis_ledger,
                gpt_decision_path=args.gpt_decision,
                trader_decision_path=args.trader_decision,
                position_management_path=args.position_management,
                thesis_evolution_path=args.thesis_evolution,
                position_thesis_path=args.position_thesis,
                forecast_persistence_path=args.forecast_persistence,
                coverage_optimization_path=args.coverage_optimization,
                window_hours=args.window_hours,
            )
        except (OSError, json.JSONDecodeError, sqlite3.Error, ValueError) as exc:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2, sort_keys=True))
            return 2
        print(
            json.dumps(
                {
                    "status": summary.status,
                    "db_path": str(summary.db_path),
                    "history_db_path": str(summary.history_db_path),
                    "output_path": str(summary.output_path),
                    "report_path": str(summary.report_path),
                    "findings": summary.findings,
                    "p0_findings": summary.p0_findings,
                    "p1_findings": summary.p1_findings,
                    "p2_findings": summary.p2_findings,
                    "closed_trades": summary.closed_trades,
                    "net_jpy": summary.net_jpy,
                    "profit_factor": summary.profit_factor,
                    "expectancy_jpy": summary.expectancy_jpy,
                    "live_ready_lanes": summary.live_ready_lanes,
                    "open_trader_positions": summary.open_trader_positions,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if summary.status != "SELF_IMPROVEMENT_BLOCKED" else 2
    if args.command == "trailing-sl-update":
        from quant_rabbit.strategy.trailing_sl import apply_trailing_sls
        snapshot_payload = json.loads(args.snapshot.read_text()) if args.snapshot.exists() else {}
        # Reconstruct snapshot object (use the read-only client to parse).
        client = OandaReadOnlyClient()
        snapshot = client.parse_snapshot(snapshot_payload) if hasattr(client, "parse_snapshot") else None
        if snapshot is None:
            # Fall back to direct construction from broker_snapshot.json.
            # NOTE: `datetime` is imported at module level (line 8); a
            # local `from datetime import …` here would shadow that and
            # break sibling commands (`pair-charts` etc.) inside this
            # same `main()` function with UnboundLocalError.
            from quant_rabbit.models import (
                BrokerSnapshot, BrokerPosition, BrokerOrder, AccountSummary, Quote, Owner, Side,
            )
            def _owner(s):
                try:
                    return Owner(s) if s else Owner.UNKNOWN
                except Exception:
                    return Owner.UNKNOWN
            positions = [
                BrokerPosition(
                    trade_id=p.get("trade_id"),
                    pair=p.get("pair", ""),
                    side=Side(p.get("side", "LONG")),
                    units=int(p.get("units", 0)),
                    entry_price=float(p.get("entry_price", 0)),
                    take_profit=p.get("take_profit"),
                    stop_loss=p.get("stop_loss"),
                    owner=_owner(p.get("owner", "unknown")),
                    unrealized_pl_jpy=float(p.get("unrealized_pl_jpy", 0)),
                )
                for p in snapshot_payload.get("positions", []) or []
            ]
            quotes = {
                pair: Quote(pair=pair, bid=q.get("bid", 0.0), ask=q.get("ask", 0.0))
                for pair, q in (snapshot_payload.get("quotes") or {}).items()
                if isinstance(q, dict)
            }
            acc_in = snapshot_payload.get("account") or {}
            account = AccountSummary(
                balance_jpy=float(acc_in.get("balance_jpy", 0.0)),
                nav_jpy=float(acc_in.get("nav_jpy", 0.0)),
                margin_used_jpy=float(acc_in.get("margin_used_jpy", 0.0)),
                margin_available_jpy=float(acc_in.get("margin_available_jpy", 0.0)),
                unrealized_pl_jpy=float(acc_in.get("unrealized_pl_jpy", 0.0)),
                pl_jpy=float(acc_in.get("pl_jpy", 0.0)),
                financing_jpy=float(acc_in.get("financing_jpy", 0.0)),
                hedging_enabled=bool(acc_in.get("hedging_enabled", True)),
                last_transaction_id=str(acc_in.get("last_transaction_id", "")),
            ) if acc_in else None
            snapshot = BrokerSnapshot(
                fetched_at_utc=datetime.now(timezone.utc),
                positions=tuple(positions),
                orders=tuple(),
                quotes=quotes,
                home_conversions=snapshot_payload.get("home_conversions", {}),
                account=account,
            )
        pair_charts_payload = json.loads(args.pair_charts.read_text()) if args.pair_charts.exists() else {}
        updates = apply_trailing_sls(
            snapshot=snapshot,
            pair_charts_payload=pair_charts_payload,
            broker_client=client,
            dry_run=args.dry_run,
        )
        print(json.dumps(
            {
                "status": "OK",
                "dry_run": args.dry_run,
                "updates": [
                    {
                        "trade_id": u.trade_id, "pair": u.pair, "side": u.side,
                        "old_sl": u.old_sl, "new_sl": u.new_sl,
                        "bos_tf": u.bos_tf, "bos_price": u.bos_price,
                        "reason": u.reason, "applied": u.applied,
                    }
                    for u in updates
                ],
                "count": len(updates),
            },
            indent=2,
        ))
        return 0
    if args.command == "position-thesis-check":
        from quant_rabbit.strategy.position_thesis_validator import assess_all_positions
        from quant_rabbit.strategy.projection_ledger import compute_hit_rates
        from quant_rabbit.models import BrokerPosition, Owner, Side
        snapshot_payload = json.loads(args.snapshot.read_text()) if args.snapshot.exists() else {}
        pair_charts_payload = json.loads(args.pair_charts.read_text()) if args.pair_charts.exists() else {}
        charts_by_pair: dict = {}
        for c in pair_charts_payload.get("charts", []) or []:
            if isinstance(c, dict) and c.get("pair"):
                charts_by_pair[c["pair"]] = c
        quotes_by_pair = snapshot_payload.get("quotes") or {}
        # Build BrokerPosition objects
        positions = []
        for p in snapshot_payload.get("positions", []) or []:
            try:
                positions.append(BrokerPosition(
                    trade_id=p.get("trade_id"),
                    pair=p.get("pair"),
                    side=Side(p.get("side")),
                    units=int(p.get("units", 0)),
                    entry_price=float(p.get("entry_price", 0.0)),
                    take_profit=p.get("take_profit"),
                    stop_loss=p.get("stop_loss"),
                    unrealized_pl_jpy=float(p.get("unrealized_pl_jpy", 0.0)),
                    owner=Owner(p.get("owner", "trader")) if p.get("owner") else Owner.UNKNOWN,
                ))
            except Exception:
                continue
        # COT + option_skew payloads
        cot_payload = None
        option_skew_payload = None
        try:
            cot_path = Path("data/cot_snapshot.json")
            if cot_path.exists():
                cot_payload = json.loads(cot_path.read_text())
            opt_path = Path("data/option_skew_snapshot.json")
            if opt_path.exists():
                option_skew_payload = json.loads(opt_path.read_text())
        except Exception:
            pass
        hit_rates = compute_hit_rates(Path("data"))
        assessments = assess_all_positions(
            positions,
            quotes_by_pair=quotes_by_pair,
            pair_charts_full=charts_by_pair,
            cot_payload=cot_payload,
            option_skew_payload=option_skew_payload,
            calendar_path=Path("data/economic_calendar.json"),
            cross_asset_path=Path("data/cross_asset_snapshot.json"),
            hit_rates=hit_rates,
            data_root=Path("data"),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "generated_at_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
            "assessments": [a.to_dict() for a in assessments],
        }, ensure_ascii=False, indent=2))
        # Surface verdict summary to stdout
        print(json.dumps({
            "status": "OK",
            "position_count": len(assessments),
            "verdicts": [
                {"trade_id": a.trade_id, "pair": a.pair, "side": a.side,
                 "score": a.aggregate_score, "verdict": a.verdict}
                for a in assessments
            ],
            "output_path": str(args.output),
        }, indent=2, ensure_ascii=False))
        return 0
    if args.command == "generate-predictive-limits":
        from quant_rabbit.strategy.forward_projection import detect_forward_projections
        from quant_rabbit.strategy.path_projection import detect_paths
        from quant_rabbit.strategy.predictive_limit_orders import (
            apply_limit_orders,
            generate_limits_from_projections,
            serialize_limit_orders,
        )
        snapshot_payload = json.loads(args.snapshot.read_text()) if args.snapshot.exists() else {}
        pair_charts_payload = json.loads(args.pair_charts.read_text()) if args.pair_charts.exists() else {}
        charts_by_pair: dict = {}
        for chart in pair_charts_payload.get("charts", []) or []:
            if isinstance(chart, dict) and chart.get("pair"):
                charts_by_pair[chart["pair"]] = chart
        quotes_by_pair = snapshot_payload.get("quotes") or {}
        all_orders = []
        for pair, chart in charts_by_pair.items():
            q = quotes_by_pair.get(pair) or {}
            bid = float(q.get("bid", 0)); ask = float(q.get("ask", 0))
            if bid <= 0 or ask <= 0:
                continue
            mid = (bid + ask) / 2.0
            projections = detect_forward_projections(
                chart, pair=pair, current_price=mid,
                calendar_path=Path("data/economic_calendar.json"),
                news_digest_path=DEFAULT_NEWS_DIGEST,
                news_items_path=DEFAULT_NEWS_SNAPSHOT,
                cross_asset_path=Path("data/cross_asset_snapshot.json"),
            )
            # Try both directions for paths (we don't know which the trader will pick)
            paths_long = detect_paths(chart, "LONG", mid)
            paths_short = detect_paths(chart, "SHORT", mid)
            paths = list(paths_long) + list(paths_short)
            orders = generate_limits_from_projections(
                pair=pair, pair_chart=chart, current_bid=bid, current_ask=ask,
                projection_signals=projections, paths=paths,
            )
            all_orders.extend(orders)
        results = []
        if all_orders:
            client = OandaExecutionClient() if args.send else None
            results = apply_limit_orders(all_orders, client, dry_run=not args.send)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "generated_at_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
            "dry_run": not args.send,
            "orders": serialize_limit_orders(all_orders),
            "send_results": results,
        }, ensure_ascii=False, indent=2))
        print(json.dumps({
            "status": "OK",
            "orders_count": len(all_orders),
            "sent": args.send,
            "output_path": str(args.output),
        }, indent=2, ensure_ascii=False))
        return 0
    if args.command in {"thesis-evolution-check", "thesis-evolution"}:
        # Compare entry-time thesis vs latest cycle forecast per open
        # position. Reads the canonical latest forecast from
        # data/forecast_history.jsonl (written each cycle by trader_brain)
        # rather than re-running the 17-detector stack — keeps this CLI
        # cheap and aligned with what trader_brain saw.
        from quant_rabbit.strategy.entry_thesis_ledger import (
            evaluate_all_open_positions,
            load_latest_forecast,
            write_thesis_evolution_report,
        )
        from quant_rabbit.models import BrokerPosition, Owner, Side
        snapshot_payload = json.loads(args.snapshot.read_text()) if args.snapshot.exists() else {}
        quotes_by_pair = snapshot_payload.get("quotes") or {}
        pair_charts_payload = json.loads(args.pair_charts.read_text()) if args.pair_charts.exists() else {}
        charts_by_pair: dict = {}
        for c in pair_charts_payload.get("charts", []) or []:
            if isinstance(c, dict) and c.get("pair"):
                charts_by_pair[c["pair"]] = c
        positions = []
        for p in snapshot_payload.get("positions", []) or []:
            try:
                positions.append(BrokerPosition(
                    trade_id=p.get("trade_id"),
                    pair=p.get("pair"),
                    side=Side(p.get("side")),
                    units=int(p.get("units", 0)),
                    entry_price=float(p.get("entry_price", 0.0)),
                    take_profit=p.get("take_profit"),
                    stop_loss=p.get("stop_loss"),
                    unrealized_pl_jpy=float(p.get("unrealized_pl_jpy", 0.0)),
                    owner=Owner(p.get("owner", "trader")) if p.get("owner") else Owner.UNKNOWN,
                ))
            except Exception:
                continue

        data_root = Path("data")
        forecast_refresh = _refresh_current_forecast_history(
            snapshot_payload=snapshot_payload,
            pair_charts_path=args.pair_charts,
            pairs=_trader_position_pairs(positions),
            data_root=data_root,
            cycle_source="position-forecast-refresh",
        )

        class _ForecastShim:
            __slots__ = ("direction", "confidence")
            def __init__(self, direction: str, confidence: float):
                self.direction = direction
                self.confidence = confidence

        forecasts_by_pair: dict = {}
        regimes_by_pair: dict = {}
        for pos in positions:
            pair = pos.pair
            if pair in forecasts_by_pair:
                continue
            f = load_latest_forecast(pair, data_root)
            if f is not None:
                forecasts_by_pair[pair] = _ForecastShim(
                    direction=str(f.get("direction", "UNCLEAR")),
                    confidence=float(f.get("confidence", 0.0)),
                )
            chart = charts_by_pair.get(pair) or {}
            conf = chart.get("confluence") or {}
            regime_raw = str(conf.get("dominant_regime") or "").upper()
            if "TREND" in regime_raw:
                regimes_by_pair[pair] = "TREND"
            elif "RANGE" in regime_raw:
                regimes_by_pair[pair] = "RANGE"
            else:
                regimes_by_pair[pair] = None

        evolutions = evaluate_all_open_positions(
            positions,
            current_forecasts_by_pair=forecasts_by_pair,
            current_regimes_by_pair=regimes_by_pair,
            data_root=data_root,
            quotes_by_pair=quotes_by_pair,
            pair_charts_by_pair=charts_by_pair,
        )
        report_path = write_thesis_evolution_report(
            evolutions,
            data_root=data_root,
            output_path=args.output,
        )
        print(json.dumps({
            "status": "OK",
            "evolution_count": len(evolutions),
            "by_status": {
                "STILL_VALID": sum(1 for e in evolutions if e.status == "STILL_VALID"),
                "WEAKENED": sum(1 for e in evolutions if e.status == "WEAKENED"),
                "BROKEN": sum(1 for e in evolutions if e.status == "BROKEN"),
                "UNVERIFIABLE": sum(1 for e in evolutions if e.status == "UNVERIFIABLE"),
            },
            "forecast_refresh": forecast_refresh,
            "verdicts": [
                {"trade_id": e.trade_id, "pair": e.pair, "side": e.side,
                 "status": e.status, "verdict": e.verdict,
                 "rationale": e.rationale}
                for e in evolutions
            ],
            "output_path": str(report_path),
        }, indent=2, ensure_ascii=False))
        return 0
    if args.command == "forecast-persistence-check":
        # Read trader-owned positions, evaluate forecast persistence for
        # each, emit per-position verdict (HOLD / EXTEND / RECOMMEND_CLOSE).
        # The verdict logic lives in forecast_persistence_tracker — this
        # CLI just surfaces it for the operator.
        from quant_rabbit.strategy.forecast_persistence_tracker import (
            assess_all_positions as _persistence_assess_all,
        )
        from quant_rabbit.models import BrokerPosition, Owner, Side
        snapshot_payload = json.loads(args.snapshot.read_text()) if args.snapshot.exists() else {}
        fresh_after = None
        try:
            raw_fetched = snapshot_payload.get("fetched_at_utc") or (snapshot_payload.get("account") or {}).get("fetched_at_utc")
            if raw_fetched:
                raw_text = str(raw_fetched)
                if raw_text.endswith("Z"):
                    raw_text = raw_text[:-1] + "+00:00"
                fresh_after = datetime.fromisoformat(raw_text)
                if fresh_after.tzinfo is None:
                    fresh_after = fresh_after.replace(tzinfo=timezone.utc)
                else:
                    fresh_after = fresh_after.astimezone(timezone.utc)
        except Exception:
            fresh_after = None
        positions = []
        for p in snapshot_payload.get("positions", []) or []:
            try:
                positions.append(BrokerPosition(
                    trade_id=p.get("trade_id"),
                    pair=p.get("pair"),
                    side=Side(p.get("side")),
                    units=int(p.get("units", 0)),
                    entry_price=float(p.get("entry_price", 0.0)),
                    take_profit=p.get("take_profit"),
                    stop_loss=p.get("stop_loss"),
                    unrealized_pl_jpy=float(p.get("unrealized_pl_jpy", 0.0)),
                    owner=Owner(p.get("owner", "trader")) if p.get("owner") else Owner.UNKNOWN,
                ))
            except Exception:
                continue
        forecast_refresh = _refresh_current_forecast_history(
            snapshot_payload=snapshot_payload,
            pair_charts_path=args.pair_charts,
            pairs=_trader_position_pairs(positions),
            data_root=Path("data"),
            cycle_source="position-forecast-refresh",
        )
        verdicts = _persistence_assess_all(
            positions,
            data_root=Path("data"),
            fresh_after_utc=fresh_after,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "generated_at_utc": __import__("datetime").datetime.now(
                __import__("datetime").timezone.utc
            ).isoformat(),
            "count": len(verdicts),
            "by_verdict": {
                "RECOMMEND_CLOSE": sum(1 for v in verdicts if v.verdict == "RECOMMEND_CLOSE"),
                "EXTEND": sum(1 for v in verdicts if v.verdict == "EXTEND"),
                "HOLD": sum(1 for v in verdicts if v.verdict == "HOLD"),
            },
            "forecast_refresh": forecast_refresh,
            "verdicts": [v.to_dict() for v in verdicts],
        }, ensure_ascii=False, indent=2))
        print(json.dumps({
            "status": "OK",
            "position_count": len(verdicts),
            "by_verdict": {
                "RECOMMEND_CLOSE": sum(1 for v in verdicts if v.verdict == "RECOMMEND_CLOSE"),
                "EXTEND": sum(1 for v in verdicts if v.verdict == "EXTEND"),
                "HOLD": sum(1 for v in verdicts if v.verdict == "HOLD"),
            },
            "forecast_refresh": forecast_refresh,
            "verdicts": [
                {"trade_id": v.trade_id, "pair": v.pair, "side": v.side,
                 "verdict": v.verdict, "reason": v.reason}
                for v in verdicts
            ],
            "output_path": str(args.output),
        }, indent=2, ensure_ascii=False))
        return 0
    if args.command == "position-management":
        from quant_rabbit.strategy.position_manager import PositionManager

        snapshot_payload = json.loads(args.snapshot.read_text()) if args.snapshot.exists() else {}
        snapshot = _snapshot_from_json(snapshot_payload)
        decision = PositionManager(
            trader_decision_path=args.trader_decision,
            pair_charts_path=args.pair_charts,
            output_path=args.output,
            report_path=args.report,
            data_root=args.data_root or args.snapshot.parent,
        ).run(snapshot)
        print(json.dumps({
            "status": "OK",
            "snapshot_fetched_at_utc": snapshot.fetched_at_utc.isoformat(),
            "action": decision.action,
            "position_count": len(decision.positions),
            "positions": [
                {
                    "trade_id": position.trade_id,
                    "pair": position.pair,
                    "side": position.side,
                    "owner": position.owner,
                    "action": position.action,
                    "unrealized_pl_jpy": position.unrealized_pl_jpy,
                    "recommended_stop_loss": position.recommended_stop_loss,
                    "recommended_take_profit": position.recommended_take_profit,
                }
                for position in decision.positions
            ],
            "output_path": str(args.output),
            "report_path": str(args.report),
        }, indent=2, ensure_ascii=False))
        return 0
    if args.command == "verify-projections":
        from quant_rabbit.strategy.projection_ledger import (
            compute_hit_rates,
            load_ledger,
            retryable_truth_timeout_pairs,
            verify_pending,
        )
        from quant_rabbit.analysis.candles import fetch_candles_via_client
        from pathlib import Path as _P
        data_root = _P("data")
        snapshot_payload = json.loads(args.snapshot.read_text()) if args.snapshot.exists() else {}
        quotes_by_pair: dict = {}
        for pair_key, quote_data in (snapshot_payload.get("quotes") or {}).items():
            if isinstance(quote_data, dict):
                quotes_by_pair[pair_key] = {
                    "bid": float(quote_data.get("bid", 0)),
                    "ask": float(quote_data.get("ask", 0)),
                }
        # Per-pair ATR for EITHER / directional thresholding.
        atr_pips_by_pair: dict = {}
        if args.pair_charts.exists():
            pc = json.loads(args.pair_charts.read_text())
            for c in pc.get("charts", []) or []:
                pair = c.get("pair")
                if not pair:
                    continue
                for v in c.get("views", []):
                    if v.get("granularity") == "H1":
                        atr = (v.get("indicators") or {}).get("atr_pips")
                        if atr:
                            atr_pips_by_pair[pair] = float(atr)
                            break
        ledger_entries = load_ledger(data_root)
        pending_pairs = sorted({e.pair for e in ledger_entries if e.resolution_status == "PENDING"})
        retry_timeout_pairs = sorted(retryable_truth_timeout_pairs(ledger_entries))
        verification_pairs = sorted(set(pending_pairs) | set(retry_timeout_pairs))
        candles_by_pair = None
        candle_counts: dict[str, int] = {}
        candle_granularity_counts: dict[str, dict[str, int]] = {}
        candle_errors: dict[str, str] = {}
        if verification_pairs and (args.m1_count > 0 or args.m5_count > 0):
            candles_by_pair = {}
            try:
                client = OandaReadOnlyClient()
            except Exception as exc:
                candle_errors["_client"] = f"{type(exc).__name__}: {str(exc)[:160]}"
                candles_by_pair = None
            else:
                for pair in verification_pairs:
                    pair_candles: dict[str, list[Any]] = {}
                    per_granularity: dict[str, int] = {}
                    for granularity, count in (("M1", int(args.m1_count)), ("M5", int(args.m5_count))):
                        if count <= 0:
                            continue
                        try:
                            candles = list(fetch_candles_via_client(client, pair, granularity, count=count))
                        except Exception as exc:
                            candle_errors[f"{pair}:{granularity}"] = f"{type(exc).__name__}: {str(exc)[:160]}"
                            continue
                        pair_candles[granularity] = candles
                        per_granularity[granularity] = len(candles)
                    if not pair_candles:
                        continue
                    candles_by_pair[pair] = pair_candles
                    candle_counts[pair] = sum(per_granularity.values())
                    candle_granularity_counts[pair] = per_granularity
        counts = verify_pending(
            data_root,
            quotes_by_pair=quotes_by_pair,
            atr_pips_by_pair=atr_pips_by_pair,
            candles_by_pair=candles_by_pair,
        )
        hr = compute_hit_rates(data_root)
        print(json.dumps({
            "status": "OK",
            "resolution_counts": counts,
            "price_truth": {
                "pending_pairs": pending_pairs,
                "retryable_timeout_pairs": retry_timeout_pairs,
                "m1_count_requested": int(args.m1_count),
                "m5_count_requested": int(args.m5_count),
                "candle_counts": candle_counts,
                "candle_granularity_counts": candle_granularity_counts,
                "m1_candles_loaded": {
                    pair: counts.get("M1", 0) for pair, counts in candle_granularity_counts.items()
                },
                "m5_candles_loaded": {
                    pair: counts.get("M5", 0) for pair, counts in candle_granularity_counts.items()
                },
                "candle_errors": candle_errors,
                "m1_errors": {
                    key: value for key, value in candle_errors.items() if key.endswith(":M1") or key == "_client"
                },
            },
            "hit_rates_by_signal": {
                sig: {pair: round(d.get("hit_rate", 0), 3) for pair, d in by_pair.items()}
                for sig, by_pair in hr.items()
            },
        }, indent=2, ensure_ascii=False))
        return 0
    if args.command == "tp-rebalance":
        from quant_rabbit.strategy.tp_rebalancer import (
            apply_tp_adjustments,
            compute_all_tp_adjustments,
            load_close_review_trade_ids,
            load_entry_thesis_blocker_trade_ids,
        )
        from quant_rabbit.strategy.intent_generator import _market_derived_reward_risk
        snapshot_payload = json.loads(args.snapshot.read_text()) if args.snapshot.exists() else {}
        # Reuse the same snapshot-parsing pattern as trailing-sl-update so
        # we share the model decoding without duplicating it.
        from quant_rabbit.models import (
            BrokerSnapshot, BrokerPosition, BrokerOrder, AccountSummary, Quote, Owner, Side,
        )
        def _owner_tp(s):
            try:
                return Owner(s) if s else Owner.UNKNOWN
            except Exception:
                return Owner.UNKNOWN
        positions = []
        for p in snapshot_payload.get("positions", []):
            try:
                positions.append(BrokerPosition(
                    trade_id=p.get("trade_id"),
                    pair=p.get("pair"),
                    side=Side(p.get("side")),
                    units=int(p.get("units", 0)),
                    entry_price=float(p.get("entry_price", 0.0)),
                    take_profit=p.get("take_profit"),
                    stop_loss=p.get("stop_loss"),
                    unrealized_pl_jpy=float(p.get("unrealized_pl_jpy", 0.0)),
                    owner=_owner_tp(p.get("owner")),
                ))
            except Exception:
                continue
        # Pair charts keyed by pair.
        pair_charts_payload = json.loads(args.pair_charts.read_text()) if args.pair_charts.exists() else {}
        pair_charts_keyed: dict = {}
        for chart in pair_charts_payload.get("charts", []) or []:
            if isinstance(chart, dict) and chart.get("pair"):
                pair_charts_keyed[chart["pair"]] = chart
        # Quotes from snapshot.
        quotes_keyed: dict = {}
        for pair_key, quote_data in (snapshot_payload.get("quotes") or {}).items():
            if isinstance(quote_data, dict):
                quotes_keyed[pair_key] = quote_data
        latest_forecasts_by_pair: dict = {}
        try:
            from quant_rabbit.strategy.entry_thesis_ledger import load_latest_forecast

            data_root = Path("data")
            for pair_key in {str(getattr(p, "pair", "")) for p in positions if getattr(p, "pair", None)}:
                latest = load_latest_forecast(pair_key, data_root)
                if latest is not None:
                    latest_forecasts_by_pair[pair_key] = latest
        except Exception:
            latest_forecasts_by_pair = {}
        adjustments = compute_all_tp_adjustments(
            positions=positions,
            quotes=quotes_keyed,
            pair_charts=pair_charts_keyed,
            market_reward_risk_fn=_market_derived_reward_risk,
            latest_forecasts_by_pair=latest_forecasts_by_pair,
            close_review_trade_ids=load_close_review_trade_ids(Path("data")),
            entry_thesis_block_trade_ids=load_entry_thesis_blocker_trade_ids(Path("data")),
        )
        client = None
        if not args.dry_run and adjustments:
            client = OandaExecutionClient()
        results = apply_tp_adjustments(adjustments, client, dry_run=args.dry_run)
        print(json.dumps(
            {
                "status": "OK",
                "dry_run": args.dry_run,
                "adjustments_count": len(adjustments),
                "results": results,
            },
            indent=2,
            ensure_ascii=False,
        ))
        return 0
    if args.command == "adverse-partial-close":
        from quant_rabbit.strategy.adverse_partial_close import (
            apply_partial_closes,
            compute_all_partial_closes,
        )
        snapshot_payload = json.loads(args.snapshot.read_text()) if args.snapshot.exists() else {}
        from quant_rabbit.models import BrokerPosition, Owner, Side
        def _owner_apc(s):
            try:
                return Owner(s) if s else Owner.UNKNOWN
            except Exception:
                return Owner.UNKNOWN
        positions = []
        for p in snapshot_payload.get("positions", []):
            try:
                positions.append(BrokerPosition(
                    trade_id=p.get("trade_id"),
                    pair=p.get("pair"),
                    side=Side(p.get("side")),
                    units=int(p.get("units", 0)),
                    entry_price=float(p.get("entry_price", 0.0)),
                    take_profit=p.get("take_profit"),
                    stop_loss=p.get("stop_loss"),
                    unrealized_pl_jpy=float(p.get("unrealized_pl_jpy", 0.0)),
                    owner=_owner_apc(p.get("owner")),
                ))
            except Exception:
                continue
        pair_charts_payload = json.loads(args.pair_charts.read_text()) if args.pair_charts.exists() else {}
        pair_charts_keyed: dict = {}
        for chart in pair_charts_payload.get("charts", []) or []:
            if isinstance(chart, dict) and chart.get("pair"):
                pair_charts_keyed[chart["pair"]] = chart
        quotes_keyed: dict = {}
        for pair_key, quote_data in (snapshot_payload.get("quotes") or {}).items():
            if isinstance(quote_data, dict):
                quotes_keyed[pair_key] = quote_data
        actions = compute_all_partial_closes(
            positions=positions,
            quotes=quotes_keyed,
            pair_charts=pair_charts_keyed,
        )
        if args.dry_run and args.send:
            print(json.dumps({
                "error": "adverse-partial-close cannot combine --dry-run and --send",
            }, ensure_ascii=False, indent=2, sort_keys=True))
            return 2
        live_send = bool(args.send)
        if live_send and not args.confirm_live:
            print(json.dumps({
                "error": "adverse-partial-close --send requires --confirm-live",
            }, ensure_ascii=False, indent=2, sort_keys=True))
            return 2
        if live_send and os.environ.get("QR_LIVE_ENABLED") != "1":
            print(json.dumps({
                "error": "adverse-partial-close --send requires QR_LIVE_ENABLED=1",
            }, ensure_ascii=False, indent=2, sort_keys=True))
            return 2
        client = None
        if live_send and actions:
            client = OandaExecutionClient()
        results = apply_partial_closes(actions, client, dry_run=not live_send)
        sent_count = sum(1 for result in results if result.get("sent"))
        blocked_count = sum(1 for result in results if result.get("error"))
        if not actions:
            status = "NO_ACTION"
        elif sent_count:
            status = "SENT" if not blocked_count else "PARTIAL_SENT_WITH_BLOCKS"
        elif blocked_count:
            status = "BLOCKED"
        else:
            status = "STAGED"
        payload = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "dry_run": not live_send,
            "send": live_send,
            "send_requested": live_send,
            "confirm_live": args.confirm_live,
            "actions_count": len(actions),
            "sent_count": sent_count,
            "blocked_count": blocked_count,
            "results": results,
            "actions": _partial_close_receipt_actions(
                results,
                management_action="ADVERSE_PARTIAL_CLOSE",
            ),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Adverse Partial Close Report",
            "",
            f"- Generated at UTC: `{payload['generated_at_utc']}`",
            f"- Status: `{status}`",
            f"- Send requested: `{live_send}`",
            f"- Sent count: `{sent_count}`",
            "",
            "## Actions",
            "",
        ]
        if not results:
            lines.append("- none")
        for result in results:
            lines.append(
                f"- `{result['trade_id']}` `{result['pair']}` `{result['side']}` "
                f"close=`{result['close_units']}` remain=`{result['remaining_units']}` "
                f"sent=`{result['sent']}`"
            )
            lines.append(f"  - rationale: {result['rationale']}")
            if result.get("error"):
                lines.append(f"  - BLOCK: {result['error']}")
        lines.extend(
            [
                "",
                "## Contract",
                "",
                "- Adverse partial close is an operator-gated margin relief path, not an implicit full-close path.",
                "- Live send requires `--send --confirm-live` and `QR_LIVE_ENABLED=1`.",
                "- Every result is persisted as a position-execution receipt so market-close outcomes can be attributed.",
            ]
        )
        args.report.write_text("\n".join(lines) + "\n")
        payload["execution_ledger"] = _record_position_execution_receipt(
            output_path=args.output,
            ledger_db_path=args.execution_ledger_db,
            ledger_report_path=args.execution_ledger_report,
        )
        args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
        print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "profit-partial-close":
        from quant_rabbit.strategy.profit_partial_close import (
            apply_profit_partial_closes,
            compute_all_profit_partial_closes,
            load_profit_partial_state,
            save_profit_partial_state_from_results,
        )
        snapshot_payload = json.loads(args.snapshot.read_text()) if args.snapshot.exists() else {}
        from quant_rabbit.models import BrokerPosition, Owner, Side

        def _owner_ppc(s):
            try:
                return Owner(s) if s else Owner.UNKNOWN
            except Exception:
                return Owner.UNKNOWN

        positions = []
        for p in snapshot_payload.get("positions", []):
            try:
                positions.append(
                    BrokerPosition(
                        trade_id=p.get("trade_id"),
                        pair=p.get("pair"),
                        side=Side(p.get("side")),
                        units=int(p.get("units", 0)),
                        entry_price=float(p.get("entry_price", 0.0)),
                        take_profit=p.get("take_profit"),
                        stop_loss=p.get("stop_loss"),
                        unrealized_pl_jpy=float(p.get("unrealized_pl_jpy", 0.0)),
                        owner=_owner_ppc(p.get("owner")),
                    )
                )
            except Exception:
                continue
        pair_charts_payload = json.loads(args.pair_charts.read_text()) if args.pair_charts.exists() else {}
        pair_charts_keyed: dict = {}
        for chart in pair_charts_payload.get("charts", []) or []:
            if isinstance(chart, dict) and chart.get("pair"):
                pair_charts_keyed[chart["pair"]] = chart
        quotes_keyed: dict = {}
        for pair_key, quote_data in (snapshot_payload.get("quotes") or {}).items():
            if isinstance(quote_data, dict):
                quotes_keyed[pair_key] = quote_data
        state = load_profit_partial_state(args.state)
        actions = compute_all_profit_partial_closes(
            positions=positions,
            quotes=quotes_keyed,
            pair_charts=pair_charts_keyed,
            state=state,
        )
        live_enabled = os.environ.get("QR_LIVE_ENABLED") == "1"
        client = OandaExecutionClient() if args.send and actions and live_enabled and args.confirm_live else None
        results = apply_profit_partial_closes(
            actions,
            client,
            send=args.send,
            live_enabled=live_enabled,
            confirm_live=args.confirm_live,
        )
        if args.send:
            state = save_profit_partial_state_from_results(results, path=args.state, state=state)
        sent_count = sum(1 for r in results if r.get("sent"))
        blocked_count = sum(1 for r in results if r.get("error"))
        if not actions:
            status = "NO_ACTION"
        elif sent_count:
            status = "SENT" if not blocked_count else "PARTIAL_SENT_WITH_BLOCKS"
        elif blocked_count:
            status = "BLOCKED"
        else:
            status = "STAGED"
        payload = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "send_requested": args.send,
            "confirm_live": args.confirm_live,
            "live_enabled": live_enabled,
            "actions_count": len(actions),
            "sent_count": sent_count,
            "blocked_count": blocked_count,
            "results": results,
            "actions": _partial_close_receipt_actions(
                results,
                management_action="PROFIT_PARTIAL_CLOSE",
            ),
            "state": state,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Profit Partial Close Report",
            "",
            f"- Generated at UTC: `{payload['generated_at_utc']}`",
            f"- Status: `{status}`",
            f"- Send requested: `{args.send}`",
            f"- Sent count: `{sent_count}`",
            "",
            "## Actions",
            "",
        ]
        if not results:
            lines.append("- none")
        for result in results:
            lines.append(
                f"- `{result['trade_id']}` `{result['pair']}` `{result['side']}` "
                f"close=`{result['close_units']}` remain=`{result['remaining_units']}` "
                f"milestone=`{result['milestone']}` sent=`{result['sent']}`"
            )
            lines.append(f"  - rationale: {result['rationale']}")
            if result.get("error"):
                lines.append(f"  - BLOCK: {result['error']}")
        lines.extend(
            [
                "",
                "## Contract",
                "",
                "- Profit partial close only reduces already-profitable trader-owned or manual/tagless exposure.",
                "- Manual/tagless profit partials never realize an adverse P/L loss and never write stop-loss orders.",
                "- Same trade milestone is persisted in state after a successful send to avoid repeat closes.",
                "- Live send requires `--send --confirm-live` and `QR_LIVE_ENABLED=1`.",
            ]
        )
        args.report.write_text("\n".join(lines) + "\n")
        payload["execution_ledger"] = _record_position_execution_receipt(
            output_path=args.output,
            ledger_db_path=args.execution_ledger_db,
            ledger_report_path=args.execution_ledger_report,
        )
        args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
        print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
        return 0 if status not in {"BLOCKED"} else 2
    if args.command == "certify-dry-run":
        summary = DryRunCertifier(
            coverage_path=args.coverage,
            execution_replay_path=args.execution_replay,
            post_trade_learning_path=args.post_trade_learning,
            order_intents_path=args.order_intents,
            live_order_path=args.live_order,
            position_execution_path=args.position_execution,
            gpt_decision_path=args.gpt_decision,
            output_path=args.output,
            report_path=args.report,
        ).run()
        print(
            json.dumps(
                {
                    "status": summary.status,
                    "output_path": str(summary.output_path),
                    "report_path": str(summary.report_path),
                    "checks": summary.checks,
                    "blockers": summary.blockers,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if summary.status == "CERTIFIED" else 2
    if args.command == "completion-status":
        summary = CompletionAuditor(
            broker_snapshot_path=args.broker_snapshot,
            order_intents_path=args.order_intents,
            target_state_path=args.target_state,
            coverage_path=args.coverage,
            replay_backtest_path=args.replay_backtest,
            execution_replay_path=args.execution_replay,
            dry_run_certification_path=args.dry_run_certification,
            live_order_path=args.live_order,
            output_path=args.output,
            report_path=args.report,
        ).run()
        print(
            json.dumps(
                {
                    "status": summary.status,
                    "output_path": str(summary.output_path),
                    "report_path": str(summary.report_path),
                    "blockers": summary.blockers,
                    "next_actions": summary.next_actions,
                    "live_ready_lanes": summary.live_ready_lanes,
                    "remaining_target_jpy": summary.remaining_target_jpy,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if summary.status == "COMPLETE" else 2
    if args.command == "ai-attack-advice":
        try:
            summary = AttackAdvisor(
                intents_path=args.intents,
                target_state_path=args.target_state,
                ai_backtest_path=args.ai_backtest,
                outcome_mart_path=args.outcome_mart,
                coverage_path=args.coverage,
                output_path=args.output,
                report_path=args.report,
            ).run()
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2, sort_keys=True))
            return 2
        print(
            json.dumps(
                {
                    "status": summary.status,
                    "output_path": str(summary.output_path),
                    "report_path": str(summary.report_path),
                    "live_ready_lanes": summary.live_ready_lanes,
                    "recommended_now_lanes": summary.recommended_now_lanes,
                    "recommended_reward_jpy": summary.recommended_reward_jpy,
                    "coverage_pct": summary.coverage_pct,
                    "blockers": summary.blockers,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        # NO_ATTACK_ADVICE is a normal diagnostic state: the refresh playbook
        # runs under set -e and must continue to memory-health / routing even
        # when the current market packet has no fillable attack lane.
        return 0
    if args.command == "gpt-trader-decision":
        try:
            provider = _static_gpt_provider(
                decision_response=args.decision_response,
                required=True,
            )
            summary = GPTTraderBrain(
                provider=provider,
                intents_path=args.intents,
                campaign_plan_path=args.campaign_plan,
                strategy_profile_path=args.strategy_profile,
                market_story_profile_path=args.market_story_profile,
                market_status_path=args.market_status,
                target_state_path=args.target_state,
                attack_advice_path=args.attack_advice,
                learning_audit_path=args.learning_audit,
                market_context_matrix_path=args.market_context_matrix,
                output_path=args.output,
                report_path=args.report,
                max_lanes=args.max_lanes,
            ).run(snapshot_path=args.snapshot)
        except (RuntimeError, ValueError, OSError, json.JSONDecodeError) as exc:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2, sort_keys=True))
            return 2
        print(
            json.dumps(
                {
                    "status": summary.status,
                    "output_path": str(summary.output_path),
                    "report_path": str(summary.report_path),
                    "action": summary.action,
                    "selected_lane_id": summary.selected_lane_id,
                    "selected_lane_ids": list(summary.selected_lane_ids),
                    "allowed": summary.allowed,
                    "issues": summary.issues,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if summary.allowed else 2
    if args.command == "risk-dry-run":
        intent = _intent_from_json(json.loads(args.intent.read_text()))
        snapshot = _snapshot_from_json(json.loads(args.snapshot.read_text()))
        engine = RiskEngine(
            policy=RiskPolicy(
                max_loss_jpy=_resolve_max_loss_from_args(
                    max_loss_jpy=args.max_loss_jpy,
                    max_loss_pct=args.max_loss_pct,
                    risk_equity_jpy=args.risk_equity_jpy,
                    label="risk-dry-run",
                )
            ),
            live_enabled=os.environ.get("QR_LIVE_ENABLED") == "1",
        )
        decision = engine.validate(intent, snapshot, for_live_send=args.for_live_send)
        strategy_issues = []
        if not args.no_strategy_profile and args.strategy_profile.exists():
            strategy_issues = list(
                StrategyProfile.load(args.strategy_profile).validate(intent, for_live_send=args.for_live_send)
            )
        all_issues = [*decision.issues, *strategy_issues]
        allowed = decision.allowed and not any(issue.severity == "BLOCK" for issue in strategy_issues)
        print(
            json.dumps(
                {
                    "allowed": allowed,
                    "issues": [issue.__dict__ for issue in all_issues],
                    "metrics": decision.metrics.__dict__ if decision.metrics else None,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if allowed else 2
    return 1


def _resolve_max_loss_from_args(
    *,
    max_loss_jpy: float | None,
    max_loss_pct: float | None,
    risk_equity_jpy: float | None,
    label: str,
) -> float:
    # Per AGENT_CONTRACT §3.5: per-trade JPY caps must be equity-derived from
    # the daily target ledger, not from the RiskPolicy literal default
    # (`RiskPolicy.max_loss_jpy = 500.0` is a library default for tests, not a
    # production risk decision). Pull `per_trade_risk_budget_jpy` from
    # `data/daily_target_state.json` first; only fall through to the policy
    # literal when the ledger has no value (first-run / file missing) so the
    # operator gets a non-silent, finite cap and a clear remediation path.
    from quant_rabbit.strategy.intent_generator import _per_trade_risk_from_state
    default_cap = _per_trade_risk_from_state()
    if default_cap is None:
        default_cap = RiskPolicy().max_loss_jpy
    return resolve_max_loss_jpy(
        max_loss_jpy=max_loss_jpy,
        max_loss_pct=max_loss_pct,
        equity_jpy=risk_equity_jpy,
        default_max_loss_jpy=default_cap,
        label=label,
    )


def _snapshot_to_json(snapshot: BrokerSnapshot) -> str:
    payload = {
        "fetched_at_utc": snapshot.fetched_at_utc.isoformat(),
        "positions": [
            {
                "trade_id": pos.trade_id,
                "pair": pos.pair,
                "side": pos.side.value,
                "units": pos.units,
                "entry_price": pos.entry_price,
                "unrealized_pl_jpy": pos.unrealized_pl_jpy,
                "take_profit": pos.take_profit,
                "stop_loss": pos.stop_loss,
                "owner": pos.owner.value,
            }
            for pos in snapshot.positions
        ],
        "orders": [
            {
                "order_id": order.order_id,
                "pair": order.pair,
                "order_type": order.order_type,
                "trade_id": order.trade_id,
                "price": order.price,
                "state": order.state,
                "units": order.units,
                "owner": order.owner.value,
                "raw": snapshot_order_raw(order.raw),
            }
            for order in snapshot.orders
        ],
        "quotes": {
            pair: {
                "bid": quote.bid,
                "ask": quote.ask,
                "timestamp_utc": quote.timestamp_utc.isoformat(),
            }
            for pair, quote in snapshot.quotes.items()
        },
        "home_conversions": snapshot.home_conversions,
    }
    if getattr(snapshot, "account", None) is not None:
        account = snapshot.account
        payload["account"] = {
            "nav_jpy": account.nav_jpy,
            "balance_jpy": account.balance_jpy,
            "unrealized_pl_jpy": account.unrealized_pl_jpy,
            "margin_used_jpy": account.margin_used_jpy,
            "margin_available_jpy": account.margin_available_jpy,
            "pl_jpy": account.pl_jpy,
            "financing_jpy": account.financing_jpy,
            "last_transaction_id": account.last_transaction_id,
            "hedging_enabled": account.hedging_enabled,
            "fetched_at_utc": account.fetched_at_utc.isoformat(),
        }
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)


def _intent_from_json(payload: dict) -> OrderIntent:
    return OrderIntent(
        pair=str(payload["pair"]).upper(),
        side=Side.parse(str(payload["side"])),
        order_type=OrderType.parse(str(payload["order_type"])),
        units=int(payload["units"]),
        entry=float(payload["entry"]) if payload.get("entry") is not None else None,
        tp=float(payload["tp"]),
        sl=float(payload["sl"]),
        thesis=str(payload.get("thesis") or ""),
        reason=str(payload.get("reason") or ""),
        owner=Owner(str(payload.get("owner") or Owner.TRADER.value)),
        market_context=_market_context_from_json(payload.get("market_context")),
        metadata=dict(payload.get("metadata") or {}),
    )


def _market_context_from_json(payload: object) -> MarketContext | None:
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise ValueError("market_context must be an object")
    return MarketContext(
        regime=str(payload.get("regime") or ""),
        narrative=str(payload.get("narrative") or ""),
        chart_story=str(payload.get("chart_story") or ""),
        method=TradeMethod.parse(str(payload.get("method") or "")),
        invalidation=str(payload.get("invalidation") or ""),
        event_risk=str(payload.get("event_risk") or ""),
        session=str(payload.get("session") or ""),
    )


def _snapshot_from_json(payload: dict) -> BrokerSnapshot:
    from quant_rabbit.models import BrokerOrder, BrokerPosition

    positions = []
    for item in payload.get("positions", []) or []:
        positions.append(
            BrokerPosition(
                trade_id=str(item["trade_id"]),
                pair=str(item["pair"]),
                side=Side.parse(str(item["side"])),
                units=int(item["units"]),
                entry_price=float(item["entry_price"]),
                unrealized_pl_jpy=float(item.get("unrealized_pl_jpy") or 0.0),
                take_profit=float(item["take_profit"]) if item.get("take_profit") is not None else None,
                stop_loss=float(item["stop_loss"]) if item.get("stop_loss") is not None else None,
                owner=Owner(str(item.get("owner") or Owner.UNKNOWN.value)),
            )
        )
    orders = []
    for item in payload.get("orders", []) or []:
        orders.append(
            BrokerOrder(
                order_id=str(item["order_id"]),
                pair=item.get("pair"),
                order_type=str(item.get("order_type") or ""),
                trade_id=item.get("trade_id"),
                price=float(item["price"]) if item.get("price") is not None else None,
                state=item.get("state"),
                units=int(item["units"]) if item.get("units") is not None else None,
                owner=Owner(str(item.get("owner") or Owner.UNKNOWN.value)),
                raw=snapshot_payload_order_raw(item),
            )
        )
    quotes = {}
    for pair, item in (payload.get("quotes") or {}).items():
        ts = item.get("timestamp_utc")
        quotes[pair] = Quote(
            pair=pair,
            bid=float(item["bid"]),
            ask=float(item["ask"]),
            timestamp_utc=datetime.fromisoformat(ts) if ts else datetime.now(timezone.utc),
        )
    fetched = payload.get("fetched_at_utc")
    account = _account_summary_from_payload(payload.get("account"))
    return BrokerSnapshot(
        fetched_at_utc=datetime.fromisoformat(fetched) if fetched else datetime.now(timezone.utc),
        positions=tuple(positions),
        orders=tuple(orders),
        quotes=quotes,
        account=account,
        home_conversions={str(k).upper(): float(v) for k, v in (payload.get("home_conversions") or {}).items()},
    )


def _account_summary_from_payload(payload: object):
    from quant_rabbit.models import AccountSummary

    if not isinstance(payload, dict):
        return None
    fetched = payload.get("fetched_at_utc")
    return AccountSummary(
        nav_jpy=float(payload.get("nav_jpy") or 0.0),
        balance_jpy=float(payload.get("balance_jpy") or 0.0),
        unrealized_pl_jpy=float(payload.get("unrealized_pl_jpy") or 0.0),
        margin_used_jpy=float(payload.get("margin_used_jpy") or 0.0),
        margin_available_jpy=float(payload.get("margin_available_jpy") or 0.0),
        pl_jpy=float(payload.get("pl_jpy") or 0.0),
        financing_jpy=float(payload.get("financing_jpy") or 0.0),
        last_transaction_id=str(payload.get("last_transaction_id") or ""),
        hedging_enabled=bool(payload.get("hedging_enabled") or False),
        fetched_at_utc=(
            datetime.fromisoformat(fetched) if isinstance(fetched, str) and fetched else datetime.now(timezone.utc)
        ),
    )


def _static_gpt_provider(
    *,
    decision_response: Path | None,
    required: bool,
) -> StaticTraderProvider | None:
    source = decision_response
    if source is None:
        if required:
            raise ValueError("Codex GPT mode requires --decision-response")
        return None
    return StaticTraderProvider(json.loads(source.read_text()))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
