"""Fast, Paper-only A/B replay for archived 2025 strategy workers.

The archived result ledger fixes the entry cohort.  Arm A preserves each
worker's recorded exit.  Arm B sees the same entries and applies one frozen,
model-authored inventory policy using only information available at each
decision bar.  The module has no broker, network, live, or order capability.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


CONTRACT = "QR_DOJO_LEGACY_WORKER_AB_REPLAY_V1"
POLICY_CONTRACT = "QR_DOJO_LEGACY_AI_INVENTORY_POLICY_V1"
AUTHORITY = {
    "paper_replay_only": True,
    "external_broker_mutation_allowed": False,
    "live_permission": False,
    "order_authority": "NONE",
}
ELIGIBLE = ("TrendMA", "PulseBreak", "M1Scalper")
DISCOVERED = ("TrendMA", "PulseBreak", "M1Scalper", "RangeFader")
PIP = 0.01
UTC = timezone.utc


class LegacyWorkerComparisonError(ValueError):
    """The archived cohort, candle source, or policy is unsafe."""


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _time(value: str) -> datetime:
    raw = value.strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(raw)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def load_archived_candles(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in sorted(paths):
        payload = json.loads(path.read_text(encoding="utf-8"))
        candles = payload.get("candles", payload) if isinstance(payload, dict) else payload
        if not isinstance(candles, list):
            raise LegacyWorkerComparisonError(f"invalid candle file: {path}")
        for raw in candles:
            if not isinstance(raw, dict) or "time" not in raw:
                raise LegacyWorkerComparisonError(f"invalid candle row: {path}")
            mid = raw.get("mid") or {}
            ts = _time(str(raw["time"])).isoformat()
            if ts in seen:
                continue
            seen.add(ts)
            try:
                row = {
                    "time": ts,
                    "epoch": int(_time(ts).timestamp()),
                    "open": float(mid.get("o", mid.get("open"))),
                    "high": float(mid.get("h", mid.get("high"))),
                    "low": float(mid.get("l", mid.get("low"))),
                    "close": float(mid.get("c", mid.get("close"))),
                }
            except (TypeError, ValueError) as exc:
                raise LegacyWorkerComparisonError(f"bad candle prices: {path}") from exc
            if not all(math.isfinite(row[key]) and row[key] > 0 for key in ("open", "high", "low", "close")):
                raise LegacyWorkerComparisonError(f"non-finite candle: {path}")
            rows.append(row)
    rows.sort(key=lambda row: row["epoch"])
    if not rows or any(a["epoch"] >= b["epoch"] for a, b in zip(rows, rows[1:])):
        raise LegacyWorkerComparisonError("candle series is empty or non-monotonic")
    return rows


def load_result_ledger(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("trades"), list):
        raise LegacyWorkerComparisonError("archived result ledger is malformed")
    by_strategy = payload.get("by_strategy")
    if not isinstance(by_strategy, dict) or not set(DISCOVERED).issubset(by_strategy):
        raise LegacyWorkerComparisonError("archived strategy denominator is incomplete")
    return payload


def validate_policy(path: Path) -> dict[str, Any]:
    policy = json.loads(path.read_text(encoding="utf-8"))
    if policy.get("contract") != POLICY_CONTRACT or policy.get("authority") != AUTHORITY:
        raise LegacyWorkerComparisonError("AI inventory policy contract/authority invalid")
    params = policy.get("parameters")
    required = {
        "direction_lookback_bars",
        "direction_block_pips",
        "risk_fraction",
        "high_volatility_atr_pips",
        "high_volatility_size_multiple",
        "inventory_same_side_cap",
        "breakeven_trigger_r",
        "partial_trigger_r",
        "partial_fraction",
        "trailing_trigger_r",
        "trailing_atr_multiple",
        "early_exit_opposition_bars",
    }
    if not isinstance(params, dict) or set(params) != required:
        raise LegacyWorkerComparisonError("AI inventory policy parameter schema invalid")
    numeric = {key: float(value) for key, value in params.items()}
    if not all(math.isfinite(value) and value >= 0 for value in numeric.values()):
        raise LegacyWorkerComparisonError("AI inventory policy contains invalid values")
    if not 0 < numeric["risk_fraction"] <= 0.02:
        raise LegacyWorkerComparisonError("risk_fraction must be in (0, 0.02]")
    if not 0 < numeric["high_volatility_size_multiple"] <= 1:
        raise LegacyWorkerComparisonError("size multiple must be trim-only")
    if not 0 < numeric["partial_fraction"] < 1:
        raise LegacyWorkerComparisonError("partial_fraction must be in (0, 1)")
    return policy


def build_loss_window_packet(
    *,
    result_ledger: Mapping[str, Any],
    candles: list[dict[str, Any]],
    result_path: Path,
    code_paths: Mapping[str, Path],
    windows_per_strategy: int = 5,
) -> dict[str, Any]:
    index = {row["epoch"]: idx for idx, row in enumerate(candles)}
    losses: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for trade in result_ledger["trades"]:
        strategy = trade.get("strategy")
        if strategy in ELIGIBLE and float(trade.get("pnl_pips") or 0) < 0:
            losses[strategy].append(trade)
    windows = []
    for strategy in ELIGIBLE:
        ranked = sorted(losses[strategy], key=lambda row: float(row["pnl_pips"]))
        for trade in ranked[:windows_per_strategy]:
            epoch = int(_time(str(trade["entry_time"])).timestamp())
            cursor = index.get(epoch)
            if cursor is None:
                continue
            prior = candles[max(0, cursor - 30) : cursor + 1]
            after = candles[cursor + 1 : cursor + 31]
            windows.append(
                {
                    "strategy": strategy,
                    "entry": {
                        key: trade.get(key)
                        for key in (
                            "side",
                            "entry_time",
                            "entry_price",
                            "tp_price",
                            "sl_price",
                            "timeout_sec",
                        )
                    },
                    "recorded_outcome": {
                        "exit_time": trade.get("exit_time"),
                        "outcome": trade.get("outcome"),
                        "gross_pips": trade.get("gross_pips"),
                        "pnl_pips": trade.get("pnl_pips"),
                    },
                    "causal_prior_bars": prior,
                    "post_entry_review_bars": after,
                    "post_entry_use": "POST_OUTCOME_POLICY_REVIEW_ONLY",
                }
            )
    body = {
        "contract": "QR_DOJO_LEGACY_HIGH_INFORMATION_WINDOWS_V1",
        "authority": AUTHORITY,
        "result_ledger": {
            "path": str(result_path),
            "sha256": file_sha256(result_path),
        },
        "strategy_code": {
            name: {"path": str(path), "sha256": file_sha256(path)}
            for name, path in sorted(code_paths.items())
        },
        "candidate_order": list(ELIGIBLE),
        "windows": windows,
        "rules": {
            "mechanical_replay_first": True,
            "fresh_model_reviews_only_high_information_windows": True,
            "future_bars_for_policy_review_not_entry_decision": True,
            "84_cell_queue_used": False,
        },
    }
    return {**body, "packet_sha256": canonical_sha256(body)}


def _atr(candles: list[dict[str, Any]], cursor: int, period: int = 14) -> float:
    start = max(1, cursor - period + 1)
    true_ranges = []
    for idx in range(start, cursor + 1):
        row = candles[idx]
        prev = candles[idx - 1]["close"]
        true_ranges.append(max(row["high"] - row["low"], abs(row["high"] - prev), abs(row["low"] - prev)))
    return sum(true_ranges) / len(true_ranges) if true_ranges else 0.0


def _exit_jpy(entry: float, exit_price: float, side: str, units: float, cost_pips: float) -> float:
    direction = 1.0 if side == "LONG" else -1.0
    gross = (exit_price - entry) * direction * units
    return gross - cost_pips * PIP * units


def _units(initial_balance_jpy: float, risk_fraction: float, entry: float, stop: float) -> float:
    risk_per_unit = abs(entry - stop)
    if risk_per_unit <= 0:
        return 0.0
    return math.floor(initial_balance_jpy * risk_fraction / risk_per_unit)


def _metrics(rows: list[dict[str, Any]], initial_balance_jpy: float) -> dict[str, Any]:
    pnl = [float(row["net_pnl_jpy"]) for row in rows]
    gains = sum(value for value in pnl if value > 0)
    losses = -sum(value for value in pnl if value < 0)
    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    for value in pnl:
        cumulative += value
        peak = max(peak, cumulative)
        max_dd = max(max_dd, peak - cumulative)
    giveback = 0.0 if peak <= 0 else max(0.0, peak - cumulative) / peak
    return {
        "initial_balance_jpy": round(initial_balance_jpy, 2),
        "net_profit_jpy": round(sum(pnl), 2),
        "profit_factor": None if losses == 0 and gains > 0 else round(gains / losses, 4) if losses else 0.0,
        "expectancy_jpy": round(sum(pnl) / len(pnl), 2) if pnl else 0.0,
        "max_drawdown_jpy": round(max_dd, 2),
        "profit_giveback_ratio": round(giveback, 4),
        "trade_count": len(rows),
        "win_rate": round(sum(value > 0 for value in pnl) / len(pnl), 4) if pnl else 0.0,
        "ending_equity_jpy": round(initial_balance_jpy + sum(pnl), 2),
    }


def _simulate_ai_trade(
    *,
    trade: Mapping[str, Any],
    candles: list[dict[str, Any]],
    entry_cursor: int,
    initial_balance_jpy: float,
    cost_pips: float,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    side = str(trade["side"])
    entry = float(trade["entry_price"])
    original_stop = float(trade["sl_price"])
    original_tp = float(trade["tp_price"])
    risk_distance = abs(entry - original_stop)
    decisions: list[dict[str, Any]] = []
    lookback = int(params["direction_lookback_bars"])
    prior_cursor = max(0, entry_cursor - lookback)
    drift_pips = (candles[entry_cursor]["close"] - candles[prior_cursor]["close"]) / PIP
    opposed = (side == "LONG" and drift_pips < -float(params["direction_block_pips"])) or (
        side == "SHORT" and drift_pips > float(params["direction_block_pips"])
    )
    decisions.append({"action": "ENTRY_DIRECTION_CHECK", "drift_pips": round(drift_pips, 3), "blocked": opposed})
    if opposed:
        return {
            "strategy": trade["strategy"],
            "side": side,
            "entry_time": trade["entry_time"],
            "exit_reason": "AI_ENTRY_SUPPRESSED",
            "units": 0,
            "net_pnl_jpy": 0.0,
            "ai_decisions": decisions,
        }

    entry_atr = _atr(candles, entry_cursor)
    multiple = 1.0
    if entry_atr / PIP >= float(params["high_volatility_atr_pips"]):
        multiple = float(params["high_volatility_size_multiple"])
    base_units = _units(initial_balance_jpy, float(params["risk_fraction"]), entry, original_stop)
    units = math.floor(base_units * multiple)
    decisions.append({"action": "DYNAMIC_LOT", "base_units": base_units, "multiple": multiple, "units": units})
    if units <= 0:
        raise LegacyWorkerComparisonError("AI sizing produced zero units")

    deadline = int(_time(str(trade["entry_time"])).timestamp()) + int(trade["timeout_sec"])
    stop = original_stop
    remaining = float(units)
    realized = 0.0
    favorable = entry
    partial_done = False
    opposition = 0
    exit_reason = "EOD"
    exit_price = candles[-1]["close"]
    for cursor in range(entry_cursor + 1, len(candles)):
        row = candles[cursor]
        if row["epoch"] > deadline:
            exit_reason = "TIME"
            exit_price = row["close"]
            break
        if side == "LONG":
            if row["low"] <= stop:
                exit_reason, exit_price = "AI_STOP", stop
                break
            if row["high"] >= original_tp:
                exit_reason, exit_price = "TP", original_tp
                break
            favorable = max(favorable, row["high"])
            favorable_r = (favorable - entry) / risk_distance
        else:
            if row["high"] >= stop:
                exit_reason, exit_price = "AI_STOP", stop
                break
            if row["low"] <= original_tp:
                exit_reason, exit_price = "TP", original_tp
                break
            favorable = min(favorable, row["low"])
            favorable_r = (entry - favorable) / risk_distance

        if not partial_done and favorable_r >= float(params["partial_trigger_r"]):
            close_units = math.floor(units * float(params["partial_fraction"]))
            if close_units > 0:
                partial_price = entry + risk_distance * float(params["partial_trigger_r"]) * (1 if side == "LONG" else -1)
                realized += _exit_jpy(entry, partial_price, side, close_units, cost_pips)
                remaining -= close_units
                partial_done = True
                decisions.append({"action": "PARTIAL_CLOSE", "units": close_units, "price": round(partial_price, 3)})
        if favorable_r >= float(params["breakeven_trigger_r"]):
            candidate = entry
            if (side == "LONG" and candidate > stop) or (side == "SHORT" and candidate < stop):
                stop = candidate
                decisions.append({"action": "BREAKEVEN", "stop": round(stop, 3)})
        if favorable_r >= float(params["trailing_trigger_r"]):
            distance = max(_atr(candles, cursor) * float(params["trailing_atr_multiple"]), PIP)
            candidate = favorable - distance if side == "LONG" else favorable + distance
            if (side == "LONG" and candidate > stop and candidate < row["close"]) or (
                side == "SHORT" and candidate < stop and candidate > row["close"]
            ):
                stop = candidate
                decisions.append({"action": "TRAIL", "stop": round(stop, 3)})

        if cursor >= 3:
            fast = sum(candles[idx]["close"] for idx in range(cursor - 2, cursor + 1)) / 3
            slow_start = max(0, cursor - 9)
            slow_rows = candles[slow_start : cursor + 1]
            slow = sum(item["close"] for item in slow_rows) / len(slow_rows)
            is_opposed = (side == "LONG" and fast < slow) or (side == "SHORT" and fast > slow)
            opposition = opposition + 1 if is_opposed else 0
            current_r = ((row["close"] - entry) if side == "LONG" else (entry - row["close"])) / risk_distance
            if opposition >= int(params["early_exit_opposition_bars"]) and current_r < 0.25:
                exit_reason, exit_price = "AI_EARLY_EXIT", row["close"]
                decisions.append({"action": "EARLY_EXIT", "opposition_bars": opposition, "current_r": round(current_r, 3)})
                break

    total = realized + _exit_jpy(entry, float(exit_price), side, remaining, cost_pips)
    return {
        "strategy": trade["strategy"],
        "side": side,
        "entry_time": trade["entry_time"],
        "exit_reason": exit_reason,
        "units": units,
        "net_pnl_jpy": round(total, 6),
        "ai_decisions": decisions,
    }


def run_comparison(
    *,
    result_path: Path,
    candle_paths: Iterable[Path],
    policy_path: Path,
    initial_balance_jpy: float = 200_000.0,
    round_trip_cost_pips: float = 0.8,
) -> dict[str, Any]:
    ledger = load_result_ledger(result_path)
    candles = load_archived_candles(candle_paths)
    policy = validate_policy(policy_path)
    index = {row["epoch"]: idx for idx, row in enumerate(candles)}
    by_strategy: dict[str, dict[str, Any]] = {}
    params = policy["parameters"]
    for strategy in ELIGIBLE:
        cohort = [row for row in ledger["trades"] if row.get("strategy") == strategy]
        a_rows = []
        b_rows = []
        for trade in cohort:
            entry_epoch = int(_time(str(trade["entry_time"])).timestamp())
            cursor = index.get(entry_epoch)
            if cursor is None:
                continue
            units = _units(
                initial_balance_jpy,
                float(params["risk_fraction"]),
                float(trade["entry_price"]),
                float(trade["sl_price"]),
            )
            a_rows.append(
                {
                    "strategy": strategy,
                    "entry_time": trade["entry_time"],
                    "exit_reason": trade.get("outcome"),
                    "units": units,
                    "net_pnl_jpy": round(
                        float(trade.get("gross_pips", trade.get("pnl_pips", 0)))
                        * PIP
                        * units
                        - round_trip_cost_pips * PIP * units,
                        6,
                    ),
                }
            )
            b_rows.append(
                _simulate_ai_trade(
                    trade=trade,
                    candles=candles,
                    entry_cursor=cursor,
                    initial_balance_jpy=initial_balance_jpy,
                    cost_pips=round_trip_cost_pips,
                    params=params,
                )
            )
        a_metrics = _metrics(a_rows, initial_balance_jpy)
        b_metrics = _metrics(b_rows, initial_balance_jpy)
        decision_count = sum(len(row["ai_decisions"]) for row in b_rows)
        delta = round(b_metrics["net_profit_jpy"] - a_metrics["net_profit_jpy"], 2)
        if (
            b_metrics["net_profit_jpy"] > 0
            and b_metrics["profit_factor"] is not None
            and b_metrics["profit_factor"] > 1
            and b_metrics["max_drawdown_jpy"] < a_metrics["max_drawdown_jpy"]
            and delta >= 0
        ):
            adoption = "AI_PAPER_CONTINUE"
        elif (
            a_metrics["net_profit_jpy"] > 0
            and (b_metrics["net_profit_jpy"] <= 0 or delta < 0)
        ):
            adoption = "BOT_ONLY_PAPER_CONTINUE_AI_POLICY_REJECT"
        elif (
            delta > 0
            and b_metrics["max_drawdown_jpy"] < a_metrics["max_drawdown_jpy"]
        ):
            adoption = "PAPER_OBSERVE_COST_EDGE_STILL_NEGATIVE"
        else:
            adoption = "PAPER_REJECT"
        by_strategy[strategy] = {
            "A_bot_only": a_metrics,
            "B_ai_inventory": {
                **b_metrics,
                "ai_decision_count": decision_count,
                "ai_cost_jpy": None,
                "ai_cost_status": "PLATFORM_COST_UNAVAILABLE",
            },
            "ai_net_profit_delta_jpy_before_ai_cost": delta,
            "adoption": adoption,
        }
    body = {
        "contract": CONTRACT,
        "authority": AUTHORITY,
        "comparison_design": {
            "same_entry_cohort": True,
            "same_market": True,
            "same_initial_balance": True,
            "same_round_trip_cost": True,
            "initial_balance_jpy": initial_balance_jpy,
            "round_trip_cost_pips": round_trip_cost_pips,
            "mechanical_replay_then_high_information_ai_review": True,
            "84_cell_queue_used": False,
        },
        "source": {
            "result_ledger_path": str(result_path),
            "result_ledger_sha256": file_sha256(result_path),
            "candle_files": [
                {"path": str(path), "sha256": file_sha256(path)}
                for path in sorted(candle_paths)
            ],
            "policy_path": str(policy_path),
            "policy_sha256": file_sha256(policy_path),
        },
        "candidate_priority": list(ELIGIBLE),
        "rejected_or_deferred": {
            "RangeFader": "DEFER_SAMPLE_COUNT_6",
        },
        "by_strategy": by_strategy,
        "economic_status": "UNDETERMINED_AI_COST_MISSING",
    }
    return {**body, "result_sha256": canonical_sha256(body)}
