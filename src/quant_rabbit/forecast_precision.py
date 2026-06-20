"""Forecast precision helpers for live-entry gating.

The projection ledger can show very high raw hit rates for targets that are
too small to monetize after spread. Live gates therefore need both statistical
confidence and an execution-aware target-width check.
"""

from __future__ import annotations

import math
import re
from typing import Any

from quant_rabbit.instruments import instrument_pip_factor


_PIP_RE = re.compile(r"(?P<pips>\d+(?:\.\d+)?)\s*pip", re.IGNORECASE)

TECHNICAL_HARVEST_PRECISION_RULES: tuple[dict[str, Any], ...] = (
    {
        "name": "EUR_USD_DOWN_M1_ATR_LOW_TP5_SL4",
        "pair": "EUR_USD",
        "side": "SHORT",
        "direction": "DOWN",
        "timeframe": "M1",
        "feature": "M1:atr_low",
        "scalp_tp_pips": 5.0,
        "scalp_stop_pips": 4.0,
        "scalp_tp_first_hit_rate": 0.9770,
        "scalp_tp_first_wilson95_lower": 0.9200,
        "samples": 87,
        "max_m1_atr_percentile_100": 0.25,
        "min_target_pips": 4.8,
        "max_target_pips": 5.5,
        "max_stop_pips": 4.2,
        "audit_report": "logs/reports/forecast_improvement/technical_entry_mining_latest.json",
    },
    {
        "name": "EUR_USD_DOWN_M15_CHOP_TREND_TP5_SL4",
        "pair": "EUR_USD",
        "side": "SHORT",
        "direction": "DOWN",
        "timeframe": "M15",
        "feature": "M15:chop_trend",
        "scalp_tp_pips": 5.0,
        "scalp_stop_pips": 4.0,
        "scalp_tp_first_hit_rate": 0.9737,
        "scalp_tp_first_wilson95_lower": 0.9255,
        "samples": 114,
        "max_m15_choppiness_14": 38.2,
        "min_target_pips": 4.8,
        "max_target_pips": 5.5,
        "max_stop_pips": 4.2,
        "audit_report": "logs/reports/forecast_improvement/technical_entry_mining_latest.json",
    },
    {
        "name": "EUR_USD_DOWN_M15_BB_WIDTH_LOW_TP5_SL4",
        "pair": "EUR_USD",
        "side": "SHORT",
        "direction": "DOWN",
        "timeframe": "M15",
        "feature": "M15:bb_width_low",
        "scalp_tp_pips": 5.0,
        "scalp_stop_pips": 4.0,
        "scalp_tp_first_hit_rate": 0.9818,
        "scalp_tp_first_wilson95_lower": 0.9039,
        "samples": 55,
        "max_m15_bb_width_percentile_100": 0.25,
        "min_target_pips": 4.8,
        "max_target_pips": 5.5,
        "max_stop_pips": 4.2,
        "audit_report": "logs/reports/forecast_improvement/technical_entry_mining_latest.json",
    },
)


def wilson_lower_bound(successes: int, trials: int, *, z: float = 1.96) -> float:
    """Return the Wilson lower confidence bound for a binomial hit rate."""
    if trials <= 0:
        return 0.0
    successes = max(0, min(int(successes), int(trials)))
    p_hat = successes / float(trials)
    denom = 1.0 + (z * z / trials)
    centre = p_hat + (z * z / (2.0 * trials))
    margin = z * math.sqrt((p_hat * (1.0 - p_hat) + (z * z / (4.0 * trials))) / trials)
    return max(0.0, min(1.0, (centre - margin) / denom))


def successes_from_hit_rate(hit_rate: float | None, samples: int | None) -> int | None:
    """Convert a rounded hit-rate bucket back to an integer success count."""
    if hit_rate is None or samples is None or samples <= 0:
        return None
    bounded = max(0.0, min(1.0, float(hit_rate)))
    return max(0, min(int(samples), int(round(bounded * int(samples)))))


def hit_rate_wilson_lower(hit_rate: float | None, samples: int | None) -> float | None:
    successes = successes_from_hit_rate(hit_rate, samples)
    if successes is None or samples is None or samples <= 0:
        return None
    return wilson_lower_bound(successes, int(samples))


def target_pips_from_text(text: str | None) -> float | None:
    """Extract the first '<number>pip' distance from rationale text."""
    if not text:
        return None
    match = _PIP_RE.search(str(text))
    if match is None:
        return None
    try:
        return float(match.group("pips"))
    except (TypeError, ValueError):
        return None


def target_pips_from_payload(payload: Any) -> float | None:
    """Read target-pip distance from a support signal payload."""
    if not isinstance(payload, dict):
        return None
    for key in ("target_pips", "target_distance_pips", "reward_pips"):
        try:
            value = payload.get(key)
        except AttributeError:
            value = None
        if value is None:
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if parsed >= 0.0:
            return parsed
    return target_pips_from_text(str(payload.get("rationale") or ""))


def support_signal_clears_live_precision(
    payload: dict[str, Any],
    *,
    min_wilson_lower: float,
    min_samples: int,
    min_target_pips: float,
) -> bool:
    """Return whether a support signal is statistically and economically usable.

    `liquidity_sweep_*` is target-distance based. If its target-pip width is
    missing or inside the configured floor, the signal is treated as unproven
    for live support even if its raw touch hit-rate is high.
    """
    try:
        samples = int(payload.get("samples", 0) or 0)
    except (TypeError, ValueError):
        return False
    if samples < int(min_samples):
        return False
    try:
        hit_rate = float(payload.get("hit_rate"))
    except (TypeError, ValueError):
        return False
    lower = hit_rate_wilson_lower(hit_rate, samples)
    if lower is None or lower < float(min_wilson_lower):
        return False
    name = str(payload.get("name") or payload.get("calibration_name") or "").lower()
    target_pips = target_pips_from_payload(payload)
    if "liquidity_sweep" in name:
        if target_pips is None:
            return False
        return target_pips >= float(min_target_pips)
    if target_pips is not None and target_pips < float(min_target_pips):
        return False
    return True


def technical_harvest_precision_support(
    metadata: dict[str, Any],
    *,
    pair: str,
    side: str,
    order_type: str | None,
    method: str | None,
    entry: float | None,
    take_profit: float | None,
    stop_loss: float | None,
) -> dict[str, Any] | None:
    """Return audited short-harvest support when current metadata matches it.

    This is intentionally narrower than generic forecast support. It only
    recognizes backtested TP-first scalp shapes whose current chart features,
    pair, side, TP width, and stop width match the audit rule.
    """

    if not isinstance(metadata, dict):
        return None
    if str(order_type or "").upper() == "MARKET":
        return None
    if str(metadata.get("tp_execution_mode") or "").upper() != "ATTACHED_TECHNICAL_TP":
        return None
    if str(metadata.get("tp_target_intent") or "").upper() != "HARVEST":
        return None
    if str(metadata.get("opportunity_mode") or "").upper() not in {"", "HARVEST"}:
        return None
    normalized_pair = str(pair or "").upper()
    normalized_side = str(side or "").upper()
    normalized_direction = str(metadata.get("forecast_direction") or "").upper()
    chart_bias = str(metadata.get("chart_direction_bias") or "").upper()
    if chart_bias and chart_bias != normalized_side:
        return None
    for rule in TECHNICAL_HARVEST_PRECISION_RULES:
        if normalized_pair != rule["pair"]:
            continue
        if normalized_side != rule["side"]:
            continue
        if normalized_direction != rule["direction"]:
            continue
        target_pips = _signed_reward_pips(
            normalized_pair,
            normalized_side,
            entry=entry,
            take_profit=take_profit,
        )
        stop_pips = _signed_stop_pips(
            normalized_pair,
            normalized_side,
            entry=entry,
            stop_loss=stop_loss,
        )
        if target_pips is None or stop_pips is None:
            continue
        if target_pips < float(rule["min_target_pips"]) or target_pips > float(rule["max_target_pips"]):
            continue
        if stop_pips <= 0.0 or stop_pips > float(rule["max_stop_pips"]):
            continue
        feature_values = _technical_rule_feature_values(metadata, rule)
        if feature_values is None:
            continue
        support = {
            "name": rule["name"],
            "pair": rule["pair"],
            "side": rule["side"],
            "direction": rule["direction"],
            "feature": rule["feature"],
            "timeframe": rule["timeframe"],
            "samples": rule["samples"],
            "scalp_tp_first_hit_rate": rule["scalp_tp_first_hit_rate"],
            "scalp_tp_first_wilson95_lower": rule["scalp_tp_first_wilson95_lower"],
            "scalp_tp_pips": rule["scalp_tp_pips"],
            "scalp_stop_pips": rule["scalp_stop_pips"],
            "current_target_pips": round(target_pips, 4),
            "current_stop_pips": round(stop_pips, 4),
            "audit_report": rule["audit_report"],
        }
        support.update(feature_values)
        return support
    return None


def _technical_rule_feature_values(
    metadata: dict[str, Any],
    rule: dict[str, Any],
) -> dict[str, Any] | None:
    out: dict[str, Any] = {}
    if "max_m1_atr_percentile_100" in rule:
        value = _percentile_0_1(metadata.get("m1_atr_percentile_100"), metadata.get("m1_atr_percentile"))
        if value is None or value > float(rule["max_m1_atr_percentile_100"]):
            return None
        out["current_m1_atr_percentile_100"] = round(value, 4)
    if "max_m15_bb_width_percentile_100" in rule:
        value = _percentile_0_1(
            metadata.get("m15_bb_width_percentile_100"),
            metadata.get("m15_bb_width_percentile"),
        )
        if value is None or value > float(rule["max_m15_bb_width_percentile_100"]):
            return None
        out["current_m15_bb_width_percentile_100"] = round(value, 4)
    if "max_m15_choppiness_14" in rule:
        value = _safe_float(metadata.get("m15_choppiness_14"))
        if value is None or value > float(rule["max_m15_choppiness_14"]):
            return None
        out["current_m15_choppiness_14"] = round(value, 4)
    return out or None


def _percentile_0_1(primary: Any, secondary: Any = None) -> float | None:
    value = _safe_float(primary)
    if value is None:
        value = _safe_float(secondary)
        if value is not None and value > 1.0:
            value /= 100.0
    if value is None:
        return None
    return max(0.0, min(1.0, value))


def _signed_reward_pips(
    pair: str,
    side: str,
    *,
    entry: float | None,
    take_profit: float | None,
) -> float | None:
    if entry is None or take_profit is None:
        return None
    factor = instrument_pip_factor(pair)
    if side == "LONG":
        return (float(take_profit) - float(entry)) * factor
    if side == "SHORT":
        return (float(entry) - float(take_profit)) * factor
    return None


def _signed_stop_pips(
    pair: str,
    side: str,
    *,
    entry: float | None,
    stop_loss: float | None,
) -> float | None:
    if entry is None or stop_loss is None:
        return None
    factor = instrument_pip_factor(pair)
    if side == "LONG":
        return (float(entry) - float(stop_loss)) * factor
    if side == "SHORT":
        return (float(stop_loss) - float(entry)) * factor
    return None


def _safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None
