from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from .improvement import _append_once
from .ledger import CryptoLedger
from .report import atomic_write_json
from .strategies import load_strategy_profiles


def _d(value: object) -> Decimal:
    return Decimal(str(value or "0"))


def _s(value: Decimal) -> str:
    return format(value, "f")


def _json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _profit_factor(rows: list[dict[str, Any]]) -> str | None:
    gains = sum(
        (max(Decimal("0"), _d(row["net_pnl_jpy"])) for row in rows),
        Decimal("0"),
    )
    losses = sum(
        (
            abs(min(Decimal("0"), _d(row["net_pnl_jpy"])))
            for row in rows
        ),
        Decimal("0"),
    )
    return _s(gains / losses) if losses else None


def _drawdown(rows: list[dict[str, Any]]) -> Decimal:
    equity = Decimal("0")
    peak = Decimal("0")
    result = Decimal("0")
    for row in sorted(rows, key=lambda item: str(item["closed_at_utc"])):
        equity += _d(row["net_pnl_jpy"])
        peak = max(peak, equity)
        result = max(result, peak - equity)
    return result


def _trade_turnover(row: dict[str, Any]) -> Decimal:
    quantity = abs(_d(row.get("quantity")))
    entry = _d(row.get("entry_notional_jpy"))
    exit_notional = quantity * _d(row.get("exit_price"))
    return entry + exit_notional


def _contribution(
    rows: list[dict[str, Any]], field: str
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Decimal]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(field) or "UNKNOWN")].append(
            _d(row["net_pnl_jpy"])
        )
    return {
        key: {
            "completed_trades": len(values),
            "net_pnl_jpy": _s(sum(values, Decimal("0"))),
            "expectancy_jpy": _s(
                sum(values, Decimal("0")) / len(values)
            ),
        }
        for key, values in sorted(grouped.items())
    }


def _window_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    completed = len(rows)
    net = sum(
        (_d(row["net_pnl_jpy"]) for row in rows), Decimal("0")
    )
    gross = sum(
        (_d(row["gross_pnl_jpy"]) for row in rows), Decimal("0")
    )
    fees = sum((_d(row["fees_jpy"]) for row in rows), Decimal("0"))
    turnover = sum(
        (_trade_turnover(row) for row in rows),
        Decimal("0"),
    )
    stopped = sum(
        1 for row in rows if row.get("exit_reason") == "STOP_LOSS"
    )
    return {
        "completed_trades": completed,
        "gross_pnl_jpy": _s(gross),
        "fees_jpy": _s(fees),
        "net_pnl_jpy": _s(net),
        "profit_factor_after_cost": _profit_factor(rows),
        "expectancy_jpy": _s(net / completed) if completed else None,
        "max_drawdown_jpy": _s(_drawdown(rows)),
        "turnover_jpy": _s(turnover),
        "fees_per_trade_jpy": (
            _s(fees / completed) if completed else None
        ),
        "stop_out_count": stopped,
        "stop_out_rate": (
            _s(Decimal(stopped) / completed) if completed else None
        ),
    }


def _fill_audit(
    path: Path,
    *,
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metrics = metrics or {}
    if metrics.get("fill_count") is not None:
        fill_events = int(metrics["fill_count"])
        partial = int(metrics.get("partial_fill_count") or 0)
        maker = int(metrics.get("maker_fill_count") or 0)
        taker = int(metrics.get("taker_fill_count") or 0)
        return {
            "fill_events": fill_events,
            "statuses": {
                "PARTIALLY_FILLED": partial,
                "FILLED": max(0, fill_events - partial),
            },
            "order_styles": {
                "PAPER_MAKER_LIMIT": maker,
                "PAPER_TAKER": taker,
            },
            "partial_fill_ratio": (
                partial / fill_events if fill_events else None
            ),
            "source": "PERSISTED_PAPER_STATE",
        }
    if not path.exists():
        return {
            "fill_events": 0,
            "statuses": {},
            "order_styles": {},
            "partial_fill_ratio": None,
        }
    with sqlite3.connect(path) as connection:
        fill_events = int(
            connection.execute(
                "SELECT COUNT(*) FROM crypto_events "
                "WHERE event_type='PAPER_FILL'"
            ).fetchone()[0]
        )
        statuses = Counter(
            {
                str(key): int(count)
                for key, count in connection.execute(
                    "SELECT json_extract(payload_json, '$.status'), COUNT(*) "
                    "FROM crypto_events WHERE event_type='PAPER_FILL' "
                    "GROUP BY json_extract(payload_json, '$.status')"
                )
            }
        )
        styles = Counter(
            {
                str(key): int(count)
                for key, count in connection.execute(
                    "SELECT json_extract(payload_json, '$.order_style'), "
                    "COUNT(*) FROM crypto_events "
                    "WHERE event_type='PAPER_FILL' "
                    "GROUP BY json_extract(payload_json, '$.order_style')"
                )
            }
        )
    return {
        "fill_events": fill_events,
        "statuses": dict(statuses),
        "order_styles": dict(styles),
        "partial_fill_ratio": (
            statuses["PARTIALLY_FILLED"] / fill_events
            if fill_events
            else None
        ),
    }


def _top_causes(
    *,
    completed: int,
    gross: Decimal,
    fees: Decimal,
    spread: Decimal,
    adverse: Decimal,
    fills: dict[str, Any],
    exits: Counter[str],
    wait_reasons: dict[str, Any],
) -> list[dict[str, Any]]:
    causes: list[dict[str, Any]] = []

    def add(
        code: str,
        impact: float,
        confidence: float,
        evidence: dict[str, Any],
    ) -> None:
        causes.append(
            {
                "code": code,
                "impact_score": round(min(100.0, impact), 3),
                "confidence": round(confidence, 3),
                "rank_score": round(
                    min(100.0, impact) * confidence, 3
                ),
                "evidence": evidence,
            }
        )

    if completed and abs(fees) > abs(gross):
        add(
            "FEE_DRAG_DOMINATES_GROSS_EDGE",
            min(
                100.0,
                float(abs(fees) / max(Decimal("0.000001"), abs(gross)))
                * 10,
            ),
            0.99,
            {
                "gross_pnl_jpy": _s(gross),
                "fees_jpy": _s(fees),
                "fee_to_abs_gross_ratio": _s(
                    abs(fees) / max(Decimal("0.000001"), abs(gross))
                ),
            },
        )
    partial_ratio = fills.get("partial_fill_ratio")
    if partial_ratio is not None and partial_ratio > 0.5:
        add(
            "PARTIAL_FILL_CHURN",
            partial_ratio * 100,
            0.98,
            {
                "partial_fill_ratio": partial_ratio,
                "fill_events": fills["fill_events"],
                "statuses": fills["statuses"],
            },
        )
    forced_exits = sum(
        exits[reason]
        for reason in ("MAX_HOLD", "SIGNAL_INVALIDATED", "STOP_LOSS")
    )
    if completed and forced_exits:
        add(
            "MAKER_ENTRY_TAKER_EXIT_OVERTRADING",
            100 * forced_exits / completed,
            0.95,
            {
                "forced_exit_trades": forced_exits,
                "completed_trades": completed,
                "exit_reasons": dict(exits),
                "order_styles": fills["order_styles"],
            },
        )
    if completed and adverse == 0 and spread == 0:
        add(
            "ADVERSE_AND_SPREAD_COST_UNMEASURED",
            60,
            0.9,
            {
                "spread_cost_jpy": _s(spread),
                "adverse_cost_jpy": _s(adverse),
                "warning": "ZERO_RECORDED_IS_NOT_PROOF_OF_ZERO_MARKET_COST",
            },
        )
    if not completed:
        total_waits = sum(int(value) for value in wait_reasons.values())
        add(
            "NO_ACTIONABLE_TRADES",
            100,
            0.95,
            {
                "completed_trades": 0,
                "waits": total_waits,
                "wait_reasons": wait_reasons,
            },
        )
    return sorted(
        causes,
        key=lambda item: (-item["rank_score"], item["code"]),
    )[:3]


class StrategyLabAudit:
    def __init__(
        self,
        runtime_root: Path,
        *,
        baseline_root: Path | None = None,
        strategy_config: Path | None = None,
    ) -> None:
        self.runtime_root = runtime_root
        self.baseline_root = baseline_root
        self.profiles = load_strategy_profiles(strategy_config)
        self.output_root = runtime_root / "audit"
        self.audit_path = self.output_root / "audits.jsonl"
        self.latest_path = self.output_root / "latest.json"

    def run_once(self) -> dict[str, Any]:
        lanes: list[dict[str, Any]] = []
        source_tips: list[str] = []
        previous = _json(self.latest_path)
        checkpoints = {
            str(lane.get("lane_id")): dict(lane.get("ledger") or {})
            for lane in previous.get("strategy_lanes", [])
        }
        for strategy in self.profiles:
            slug = strategy.lower().replace("_", "-")
            for mode in ("spot", "margin"):
                lane_id = f"{strategy}:{mode.upper()}"
                lane = self._lane(
                    strategy,
                    slug,
                    mode,
                    checkpoint=checkpoints.get(lane_id),
                )
                lanes.append(lane)
                source_tips.append(
                    f"{strategy}:{mode}:{lane['ledger']['head_hash']}:"
                    f"{lane['completed_trade_count']}"
                )
        baseline = self._baseline()
        operation_id = hashlib.sha256(
            "|".join(
                ["crypto-strategy-audit-v1", *sorted(source_tips)]
            ).encode()
        ).hexdigest()
        totals = {
            "completed_trades": sum(
                int(lane["completed_trade_count"]) for lane in lanes
            ),
            "net_pnl_jpy": _s(
                sum(
                    (_d(lane["performance"]["net_pnl_jpy"]) for lane in lanes),
                    Decimal("0"),
                )
            ),
            "turnover_jpy": _s(
                sum(
                    (_d(lane["performance"]["turnover_jpy"]) for lane in lanes),
                    Decimal("0"),
                )
            ),
        }
        payload = {
            "schema": "QR_CRYPTO_STRATEGY_LAB_AUDIT_V1",
            "operation_id": operation_id,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "baseline": baseline,
            "strategy_lanes": lanes,
            "experiments": self._experiments(),
            "strategy_totals": totals,
            "metric_contract": {
                "trade_count": (
                    "DEPRECATED_CUMULATIVE_FILL_EVENTS_NOT_COMPLETED_TRADES"
                ),
                "completed_trade_count": (
                    "ONE_FULLY_CLOSED_POSITION_RECORDED_IN_TRADE_OUTBOX"
                ),
                "events_processed": "CURRENT_EPOCH_ONLY",
                "service_events_processed_total": (
                    "MONOTONIC_WITHIN_AND_ACROSS_SERVICE_RESTARTS"
                ),
                "fill_count_vs_epoch_events_comparable": False,
            },
            "adoption_gate": {
                "profit_factor_after_cost": ">1",
                "expectancy_jpy": ">0",
                "max_drawdown": "NON_WORSE_THAN_BASELINE",
                "minimum_completed_trades_per_lane": 30,
                "minimum_unseen_windows": 3,
                "live_promotion_allowed": False,
            },
            "authority": "NONE",
            "live_mutation": False,
        }
        _append_once(self.audit_path, payload)
        atomic_write_json(self.latest_path, payload)
        return payload

    def _experiments(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for strategy, profile in self.profiles.items():
            if not profile.variant_of:
                continue
            variant_slug = strategy.lower().replace("_", "-")
            baseline_slug = profile.variant_of.lower().replace("_", "-")
            for mode in ("spot", "margin"):
                variant_rows = _jsonl(
                    self.runtime_root
                    / variant_slug
                    / mode
                    / "trade_outbox.jsonl"
                )
                if variant_rows:
                    window_start = min(
                        str(row["opened_at_utc"]) for row in variant_rows
                    )
                    baseline_rows = [
                        row
                        for row in _jsonl(
                            self.runtime_root
                            / baseline_slug
                            / mode
                            / "trade_outbox.jsonl"
                        )
                        if str(row["closed_at_utc"]) >= window_start
                    ]
                else:
                    state = _json(
                        self.runtime_root
                        / variant_slug
                        / mode
                        / "state.json"
                    )
                    window_start = str(
                        state.get("service_started_at_utc") or ""
                    )
                    baseline_rows = []
                baseline_metrics = _window_metrics(baseline_rows)
                variant_metrics = _window_metrics(variant_rows)
                baseline_net = _d(baseline_metrics["net_pnl_jpy"])
                variant_net = _d(variant_metrics["net_pnl_jpy"])
                status = "COLLECTING"
                reason = "MINIMUM_30_COMPLETED_TRADES_NOT_REACHED"
                variant_count = int(variant_metrics["completed_trades"])
                baseline_count = int(baseline_metrics["completed_trades"])
                variant_expectancy = variant_metrics["expectancy_jpy"]
                variant_fee = variant_metrics["fees_per_trade_jpy"]
                baseline_fee = baseline_metrics["fees_per_trade_jpy"]
                if (
                    profile.changed_category == "cooldown_ms"
                    and variant_count >= 2
                    and variant_expectancy is not None
                    and _d(variant_expectancy) < 0
                    and variant_fee is not None
                    and (
                        baseline_fee is None
                        or _d(variant_fee) >= _d(baseline_fee) * Decimal("0.9")
                    )
                ):
                    status = "REJECTED_EARLY"
                    reason = (
                        "COOLDOWN_DID_NOT_REDUCE_PER_TRADE_FEE_DRAG"
                    )
                elif (
                    5 <= variant_count < 30
                    and variant_expectancy is not None
                    and _d(variant_expectancy) <= 0
                    and (
                        variant_metrics["profit_factor_after_cost"] is None
                        or _d(
                            variant_metrics[
                                "profit_factor_after_cost"
                            ]
                        )
                        <= 1
                    )
                ):
                    status = "NOT_ADOPTED_CONTINUE_EVIDENCE"
                    reason = "CURRENT_PF_AND_EXPECTANCY_FAIL"
                elif variant_count >= 30 and baseline_count >= 30:
                    variant_pf = variant_metrics[
                        "profit_factor_after_cost"
                    ]
                    if (
                        variant_pf is not None
                        and _d(variant_pf) > 1
                        and variant_expectancy is not None
                        and _d(variant_expectancy) > 0
                        and _d(variant_metrics["max_drawdown_jpy"])
                        <= _d(baseline_metrics["max_drawdown_jpy"])
                    ):
                        status = "HOLD_FOR_REPRODUCIBILITY"
                        reason = "NEEDS_THREE_UNSEEN_WINDOWS"
                    else:
                        status = "REJECTED"
                        reason = "ADOPTION_METRICS_FAILED"
                result.append(
                    {
                        "experiment_id": (
                            f"{strategy}:{mode.upper()}:"
                            f"{window_start or 'PENDING'}"
                        ),
                        "variant": strategy,
                        "baseline": profile.variant_of,
                        "mode": mode.upper(),
                        "changed_category": profile.changed_category,
                        "only_one_category_changed": True,
                        "window_start_utc": window_start or None,
                        "window_end_utc": datetime.now(
                            timezone.utc
                        ).isoformat(),
                        "baseline_metrics": baseline_metrics,
                        "variant_metrics": variant_metrics,
                        "opportunity_loss_vs_baseline_jpy": _s(
                            max(Decimal("0"), baseline_net - variant_net)
                        ),
                        "opportunity_gain_vs_baseline_jpy": _s(
                            max(Decimal("0"), variant_net - baseline_net)
                        ),
                        "opportunity_metric_note": (
                            "FORWARD_WINDOW_AGGREGATE_NOT_TRADE_PAIRED"
                        ),
                        "status": status,
                        "reason": reason,
                        "unseen_window_count": 1,
                        "future_data_used": False,
                        "live_promotion_allowed": False,
                    }
                )
        return result

    def _lane(
        self,
        strategy: str,
        slug: str,
        mode: str,
        *,
        checkpoint: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        root = self.runtime_root / slug / mode
        state = _json(root / "state.json")
        rows = _jsonl(root / "trade_outbox.jsonl")
        metrics = dict(state.get("metrics") or {})
        fills = _fill_audit(root / "ledger.db", metrics=metrics)
        ledger_store = CryptoLedger(
            root / "ledger.db",
            verify_on_open=False,
        )
        if (
            checkpoint
            and checkpoint.get("valid") is True
            and checkpoint.get("event_count") is not None
            and checkpoint.get("head_hash")
        ):
            ledger = ledger_store.verify_incremental(
                event_count=int(checkpoint["event_count"]),
                head_hash=str(checkpoint["head_hash"]),
            )
        else:
            ledger = ledger_store.verify()
        gross = sum(
            (_d(row["gross_pnl_jpy"]) for row in rows), Decimal("0")
        )
        net = sum(
            (_d(row["net_pnl_jpy"]) for row in rows), Decimal("0")
        )
        fees = sum((_d(row["fees_jpy"]) for row in rows), Decimal("0"))
        spread = sum(
            (_d(row["spread_cost_jpy"]) for row in rows), Decimal("0")
        )
        adverse = sum(
            (_d(row["adverse_cost_jpy"]) for row in rows), Decimal("0")
        )
        interest = sum(
            (_d(row["funding_interest_jpy"]) for row in rows),
            Decimal("0"),
        )
        turnover = sum(
            (_trade_turnover(row) for row in rows),
            Decimal("0"),
        )
        exits = Counter(str(row.get("exit_reason")) for row in rows)
        reasons = dict(state.get("reasons") or {})
        completed = len(rows)
        return {
            "lane_id": f"{strategy}:{mode.upper()}",
            "strategy": strategy,
            "variant_of": self.profiles[strategy].variant_of,
            "changed_category": self.profiles[strategy].changed_category,
            "mode": mode.upper(),
            "state": {
                "status": state.get("status"),
                "pid": state.get("service_pid"),
                "run_id": state.get("run_id"),
                "guardian": (state.get("guardian") or {}).get("state"),
                "equity_jpy": (state.get("metrics") or {}).get(
                    "equity_jpy"
                ),
                "net_pnl_jpy_including_open": (
                    state.get("metrics") or {}
                ).get("net_pnl_jpy"),
                "max_drawdown_jpy_including_open": (
                    state.get("metrics") or {}
                ).get("max_drawdown_jpy"),
                "open_positions": (state.get("metrics") or {}).get(
                    "open_position_count"
                ),
                "epoch_events_processed": state.get(
                    "epoch_events_processed",
                    state.get("events_processed"),
                ),
                "service_events_processed_total": state.get(
                    "service_events_processed_total"
                ),
            },
            "ledger": ledger,
            "completed_trade_count": completed,
            "fill_audit": fills,
            "performance": {
                "gross_pnl_jpy": _s(gross),
                "net_pnl_jpy": _s(net),
                "profit_factor_after_cost": _profit_factor(rows),
                "expectancy_jpy": (
                    _s(net / completed) if completed else None
                ),
                "max_drawdown_jpy": _s(_drawdown(rows)),
                "turnover_jpy": _s(turnover),
            },
            "costs": {
                "fees_jpy": _s(fees),
                "spread_jpy": _s(spread),
                "adverse_selection_jpy": _s(adverse),
                "funding_interest_jpy": _s(interest),
            },
            "exit_reasons": dict(exits),
            "contribution": {
                "pair": _contribution(rows, "pair"),
                "side": _contribution(rows, "side"),
                "regime": _contribution(rows, "regime"),
            },
            "root_causes_top3": _top_causes(
                completed=completed,
                gross=gross,
                fees=fees,
                spread=spread,
                adverse=adverse,
                fills=fills,
                exits=exits,
                wait_reasons=reasons,
            ),
        }

    def _baseline(self) -> list[dict[str, Any]]:
        if self.baseline_root is None:
            return []
        result: list[dict[str, Any]] = []
        for mode in ("spot", "margin"):
            state = _json(self.baseline_root / mode / "state.json")
            metrics = state.get("metrics") or {}
            result.append(
                {
                    "mode": mode.upper(),
                    "strategy": "FAST_MICROSTRUCTURE",
                    "status": state.get("status"),
                    "guardian": (state.get("guardian") or {}).get("state"),
                    "completed_trade_count": metrics.get(
                        "completed_trade_count",
                        metrics.get("round_trip_count", 0),
                    ),
                    "net_pnl_jpy": metrics.get("net_pnl_jpy"),
                    "equity_jpy": metrics.get("equity_jpy"),
                    "max_drawdown_jpy": metrics.get("max_drawdown_jpy"),
                }
            )
        return result
