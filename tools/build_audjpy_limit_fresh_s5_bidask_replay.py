#!/usr/bin/env python3
"""Build read-only AUD_JPY LIMIT S5 bid/ask replay proof artifacts.

The tool reuses scripts/oanda_history_replay_validate.py so the spread model is
identical to the replay validator: SHORT entries use bid and exits use ask.
It reads local files only and never calls broker write endpoints.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
FORECAST_HISTORY = ROOT / "data/forecast_history.jsonl"
HISTORY_DIR = ROOT / "logs/replay/oanda_history/20260705T180445Z"
REPLAY_REPORT = (
    ROOT
    / "logs/reports/forecast_improvement/audjpy_limit_fresh_s5_probe/oanda_history_replay_validate_latest.json"
)
JSON_OUT = ROOT / "data/audjpy_limit_fresh_s5_bidask_replay.json"
MD_OUT = ROOT / "docs/audjpy_limit_fresh_s5_bidask_replay.md"

PAIR = "AUD_JPY"
SIDE = "SHORT"
METHOD = "BREAKOUT_FAILURE"
ORDER_TYPE = "LIMIT"
EXIT_SHAPE = "TP_PROOF_COLLECTION_HARVEST"
FORECAST_DIRECTION = "UP"
TRADE_DIRECTION = "DOWN"
TAKE_PROFIT_PIPS = 10.0
STOP_LOSS_PIPS = 7.0
MIN_SAMPLES = 30
MIN_ACTIVE_DAYS = 3
MAX_DAILY_SAMPLE_SHARE = 0.70
MIN_POSITIVE_DAY_RATE = 2.0 / 3.0


def main() -> int:
    generated_at = _now()
    payload = build_payload(generated_at)
    _write_json(JSON_OUT, payload)
    _write_text(MD_OUT, _markdown(payload))
    print(f"wrote {_rel(JSON_OUT)}")
    print(f"wrote {_rel(MD_OUT)}")
    return 0


def build_payload(generated_at: str) -> dict[str, Any]:
    replay = _load_replay_module()
    report = _load_json(REPLAY_REPORT) if REPLAY_REPORT.exists() else {}

    rows, load_stats = replay._load_forecasts(FORECAST_HISTORY, pairs={PAIR})
    candles_by_pair, candle_stats = replay._load_candles(
        [HISTORY_DIR],
        granularity="S5",
        windows_by_pair=replay._forecast_truth_windows(rows),
    )
    results, score_stats, unscorable_no_market, pending_future_truth = replay._score_forecasts(
        rows,
        candles_by_pair,
        now_utc=datetime.now(timezone.utc),
    )
    contrarian_rows = [
        item for item in (replay._contrarian_row(row) for row in results)
        if item is not None
    ]
    exact_rows = [
        row
        for row in contrarian_rows
        if row.get("pair") == PAIR
        and row.get("forecast_direction") == FORECAST_DIRECTION
        and row.get("direction") == TRADE_DIRECTION
    ]
    precision_rows = [
        row
        for row in exact_rows
        if row.get("horizon_bucket") == "31-60m"
        and row.get("confidence_bucket") == "0.75-0.90"
    ]
    direct_down_rows = [
        row
        for row in results
        if row.get("pair") == PAIR and row.get("direction") == TRADE_DIRECTION
    ]

    exact = _replay_stats(replay, exact_rows)
    precision = _replay_stats(replay, precision_rows)
    direct_down = _replay_stats(replay, direct_down_rows)
    thresholds = _thresholds(exact)
    classification = _classification(exact, thresholds)
    failed = _failed_reasons(thresholds)
    queue_row = _current_queue_row()

    return {
        "generated_at_utc": generated_at,
        "mode": "read_only_audjpy_limit_fresh_s5_bidask_replay",
        "classification": classification,
        "requested_shape": {
            "pair": PAIR,
            "side": SIDE,
            "method": METHOD,
            "order_type": ORDER_TYPE,
            "exit_shape": EXIT_SHAPE,
            "forecast_direction_faded": FORECAST_DIRECTION,
            "trade_direction": TRADE_DIRECTION,
            "take_profit_pips": TAKE_PROFIT_PIPS,
            "stop_loss_pips": STOP_LOSS_PIPS,
        },
        "thresholds": {
            "min_samples": MIN_SAMPLES,
            "min_active_days": MIN_ACTIVE_DAYS,
            "max_daily_sample_share": MAX_DAILY_SAMPLE_SHARE,
            "min_positive_day_rate": round(MIN_POSITIVE_DAY_RATE, 6),
            "expectancy_requires_positive": True,
            "spread_included": True,
            "results": thresholds,
            "failed_reasons": failed,
            "meets_exact_s5_proof_thresholds": all(thresholds.values()),
        },
        "exact_shape_replay": exact,
        "rank_only_precision_subset": {
            "note": (
                "Higher-confidence subset is useful as rank-only evidence, but it cannot create "
                "permission while daily stability fails."
            ),
            **precision,
            "validator_rule": _matching_precision_rule(report),
        },
        "direct_down_reference": {
            "note": "Direct DOWN forecasts are not the requested HARVEST fade shape; included as a reference only.",
            **direct_down,
        },
        "replay_context": {
            "source": _rel(FORECAST_HISTORY),
            "history_dir": _rel(HISTORY_DIR),
            "validator_report": _rel(REPLAY_REPORT) if REPLAY_REPORT.exists() else None,
            "truth_source": report.get(
                "truth_source",
                "local OANDA S5 bid/ask candles; DOWN entry=bid/exit=ask",
            ),
            "price_truth_coverage": report.get("price_truth_coverage"),
            "load_stats": load_stats,
            "candle_stats": candle_stats,
            "score_stats": score_stats,
            "unscorable_no_market_rows": len(unscorable_no_market),
            "pending_future_truth_rows": len(pending_future_truth),
            "live_side_effects": [],
        },
        "current_queue_context": queue_row,
        "remaining_blockers": _remaining_blockers(failed, queue_row),
        "source_artifacts": [
            "data/forecast_history.jsonl",
            _rel(HISTORY_DIR),
            _rel(REPLAY_REPORT),
            "data/as_proof_pack_queue.json",
            "data/as_lane_candidate_board.json",
        ],
        "live_side_effects": [],
    }


def _replay_stats(replay: Any, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    realized = [
        replay._simulate_exit(row, take_profit_pips=TAKE_PROFIT_PIPS, stop_loss_pips=STOP_LOSS_PIPS)
        for row in rows
    ]
    pips = [float(item["pips"]) for item in realized]
    wins = [value for value in pips if value > 0.0]
    losses = [-value for value in pips if value < 0.0]
    summary = replay._exit_summary(
        realized,
        take_profit_pips=TAKE_PROFIT_PIPS,
        stop_loss_pips=STOP_LOSS_PIPS,
    )
    daily = replay._daily_exit_stability(
        rows,
        take_profit_pips=TAKE_PROFIT_PIPS,
        stop_loss_pips=STOP_LOSS_PIPS,
    )
    win_rate = float(summary["win_rate"]) if summary.get("win_rate") is not None else None
    wilson = _wilson_lower_bound(len(wins), len(pips)) if pips else None
    avg_win = sum(wins) / len(wins) if wins else None
    avg_loss = sum(losses) / len(losses) if losses else None
    pessimistic = (
        (wilson * (avg_win or 0.0)) - ((1.0 - wilson) * (avg_loss or 0.0))
        if wilson is not None
        else None
    )
    return {
        "sample_count": len(pips),
        "replay_window_utc": _window(rows),
        "net_pl_pips": _round(sum(pips)) if pips else 0.0,
        "expectancy_pips": _round(summary.get("avg_realized_pips")),
        "win_rate": _round(win_rate),
        "avg_win_pips": _round(avg_win),
        "avg_loss_pips": _round(avg_loss),
        "max_loss_pips": _round(min(pips)) if losses else None,
        "worst_loss_abs_pips": _round(max(losses)) if losses else None,
        "wilson_95_win_rate_lower": _round(wilson),
        "pessimistic_expectancy_pips": _round(pessimistic),
        "profit_factor": _round(summary.get("profit_factor")),
        "tp_rate": _round(summary.get("tp_rate")),
        "sl_rate": _round(summary.get("sl_rate")),
        "timeout_rate": _round(summary.get("timeout_rate")),
        "active_days": daily.get("active_days", 0),
        "max_daily_sample_share": daily.get("max_daily_sample_share"),
        "positive_day_rate": daily.get("positive_day_rate"),
        "daily_distribution": daily.get("daily_summaries") or [],
        "spread_included": True,
    }


def _thresholds(stats: dict[str, Any]) -> dict[str, bool]:
    return {
        "sample_count_floor": int(stats.get("sample_count") or 0) >= MIN_SAMPLES,
        "active_day_floor": int(stats.get("active_days") or 0) >= MIN_ACTIVE_DAYS,
        "daily_stability_floor": _at_most(stats.get("max_daily_sample_share"), MAX_DAILY_SAMPLE_SHARE),
        "positive_day_rate_floor": _at_least(stats.get("positive_day_rate"), MIN_POSITIVE_DAY_RATE),
        "spread_included_expectancy_positive": _at_least(stats.get("expectancy_pips"), 0.0, strict=True),
    }


def _classification(stats: dict[str, Any], thresholds: dict[str, bool]) -> str:
    if int(stats.get("sample_count") or 0) == 0:
        return "EVIDENCE_GAP"
    if all(thresholds.values()):
        return "PROOF_READY"
    if not thresholds["spread_included_expectancy_positive"]:
        return "REJECTED"
    return "REPAIR_REQUIRED"


def _failed_reasons(thresholds: dict[str, bool]) -> list[str]:
    mapping = {
        "sample_count_floor": "S5_SAMPLE_COUNT_FLOOR_NOT_MET",
        "active_day_floor": "S5_ACTIVE_DAY_FLOOR_NOT_MET",
        "daily_stability_floor": "S5_DAILY_SAMPLE_CONCENTRATED",
        "positive_day_rate_floor": "S5_POSITIVE_DAY_RATE_LOW",
        "spread_included_expectancy_positive": "S5_SPREAD_INCLUDED_EXPECTANCY_NOT_POSITIVE",
    }
    return [mapping[key] for key, value in thresholds.items() if not value]


def _remaining_blockers(failed: Sequence[str], queue_row: dict[str, Any] | None) -> list[str]:
    blockers = list(failed)
    if queue_row:
        blockers.extend(str(item) for item in queue_row.get("current_blockers") or [])
        missing = queue_row.get("missing_proof") if isinstance(queue_row.get("missing_proof"), dict) else {}
        blockers.extend(
            _proof_gap_name(key)
            for key, value in missing.items()
            if value is not True
        )
    blockers.extend(
        [
            "FORECAST_EXECUTABLE_PROOF_STILL_REQUIRED_BEFORE_LIVE",
            "PROFITABILITY_ACCEPTANCE_BLOCKED",
        ]
    )
    out: list[str] = []
    for item in blockers:
        if item not in out:
            out.append(item)
    return out


def _proof_gap_name(key: str) -> str:
    return {
        "s5_bidask_spread_included_replay": "S5_BIDASK_SPREAD_INCLUDED_REPLAY_NOT_PROVEN_FOR_LIVE",
        "risk_engine_pass": "RISK_ENGINE_PASS_MISSING",
        "live_order_gateway_pass": "LIVE_ORDER_GATEWAY_PREFLIGHT_MISSING",
        "gpt_verifier_pass": "FRESH_GPT_VERIFIER_TRADE_RECEIPT_MISSING",
        "no_guardian_operator_review_blocker": "GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED",
    }.get(key, f"{key.upper()}_MISSING")


def _matching_precision_rule(report: dict[str, Any]) -> dict[str, Any] | None:
    rules = ((report.get("precision_rules") or {}).get("contrarian_edge_rules") or [])
    for rule in rules:
        if (
            rule.get("pair") == PAIR
            and rule.get("side") == SIDE
            and rule.get("forecast_direction") == FORECAST_DIRECTION
            and rule.get("direction") == TRADE_DIRECTION
            and float(rule.get("optimized_take_profit_pips") or 0.0) == TAKE_PROFIT_PIPS
            and float(rule.get("optimized_stop_loss_pips") or 0.0) == STOP_LOSS_PIPS
        ):
            return rule
    return None


def _current_queue_row() -> dict[str, Any] | None:
    path = ROOT / "data/as_proof_pack_queue.json"
    if not path.exists():
        return None
    data = _load_json(path)
    for row in data.get("queue") or []:
        if (
            row.get("pair") == PAIR
            and row.get("side") == SIDE
            and row.get("method") == METHOD
            and row.get("order_type") == ORDER_TYPE
        ):
            return row
    return None


def _window(rows: Sequence[dict[str, Any]]) -> dict[str, str | None]:
    timestamps = [_timestamp_text(row.get("timestamp_utc")) for row in rows if row.get("timestamp_utc") is not None]
    timestamps = [value for value in timestamps if value is not None]
    if not timestamps:
        return {"first": None, "last": None}
    first = min(timestamps)
    last = max(timestamps)
    return {"first": first, "last": last}


def _timestamp_text(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat().replace("+00:00", "Z")
    text = str(value)
    if text.endswith("+00:00"):
        return text[:-6] + "Z"
    return text


def _wilson_lower_bound(wins: int, total: int, z: float = 1.96) -> float | None:
    if total <= 0:
        return None
    p = wins / total
    denom = 1.0 + (z * z / total)
    center = p + (z * z / (2.0 * total))
    margin = z * math.sqrt((p * (1.0 - p) + (z * z / (4.0 * total))) / total)
    return max(0.0, (center - margin) / denom)


def _load_replay_module() -> Any:
    path = ROOT / "scripts/oanda_history_replay_validate.py"
    spec = importlib.util.spec_from_file_location("oanda_history_replay_validate", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _markdown(payload: dict[str, Any]) -> str:
    exact = payload["exact_shape_replay"]
    thresholds = payload["thresholds"]
    lines = [
        "# AUD_JPY LIMIT Fresh S5 Bid/Ask Replay",
        "",
        f"- Generated: `{payload['generated_at_utc']}`",
        f"- Classification: `{payload['classification']}`",
        f"- Pair/side/method/order type/exit shape: `{PAIR}` `{SIDE}` `{METHOD}` `{ORDER_TYPE}` `{EXIT_SHAPE}`",
        f"- Replay window: `{exact['replay_window_utc']['first']}` to `{exact['replay_window_utc']['last']}`",
        f"- Samples: `{exact['sample_count']}`",
        f"- Active days: `{exact['active_days']}`",
        f"- Max daily sample share: `{exact['max_daily_sample_share']}`",
        f"- Positive day rate: `{exact['positive_day_rate']}`",
        f"- Spread-included net P/L: `{exact['net_pl_pips']}` pips",
        f"- Expectancy: `{exact['expectancy_pips']}` pips/trade",
        f"- Avg win/loss: `{exact['avg_win_pips']}` / `{exact['avg_loss_pips']}` pips",
        f"- Max loss: `{exact['max_loss_pips']}` pips",
        f"- Wilson 95% lower win rate: `{exact['wilson_95_win_rate_lower']}`",
        f"- Pessimistic expectancy: `{exact['pessimistic_expectancy_pips']}` pips/trade",
        f"- Meets exact S5 proof thresholds: `{thresholds['meets_exact_s5_proof_thresholds']}`",
        "",
        "## Thresholds",
        "",
    ]
    for key, value in thresholds["results"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Daily Distribution", ""])
    for row in exact["daily_distribution"]:
        lines.append(
            "- "
            f"`{row.get('date')}` samples `{row.get('samples')}` "
            f"realized `{row.get('realized_pips')}` pips "
            f"avg `{row.get('avg_realized_pips')}` "
            f"win_rate `{row.get('win_rate')}`"
        )
    lines.extend(["", "## Remaining Blockers", ""])
    for item in payload["remaining_blockers"]:
        lines.append(f"- `{item}`")
    return "\n".join(lines) + "\n"


def _round(value: Any, digits: int = 6) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isinf(number):
        return number
    return round(number, digits)


def _at_least(value: Any, threshold: float, *, strict: bool = False) -> bool:
    if value is None:
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number > threshold if strict else number >= threshold


def _at_most(value: Any, threshold: float) -> bool:
    if value is None:
        return False
    try:
        return float(value) <= threshold
    except (TypeError, ValueError):
        return False


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
