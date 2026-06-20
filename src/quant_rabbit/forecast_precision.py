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

# Ranking weights are advisory only; RiskEngine and LiveOrderGateway remain
# executable authorities. Values are anchored to the 2026-06-20 TP5/SL4 audit:
# a 90%+ Wilson lower-bound bucket gets enough score to outrank ordinary
# history noise, while high-rotation MFE buckets get a smaller nudge because
# they improve basket ordering but are not low-confidence live exceptions.
TECHNICAL_HARVEST_PRECISION_SCORE_BONUS = 24.0
TECHNICAL_HARVEST_PRECISION_EXTRA_MATCH_BONUS = 6.0
TECHNICAL_HARVEST_ROTATION_SCORE_BONUS = 12.0
TECHNICAL_HARVEST_ROTATION_EXTRA_MATCH_BONUS = 3.0
TECHNICAL_HARVEST_NEGATIVE_SCORE_PENALTY = 35.0


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

# Rotation rules are mined bucket descriptors from
# logs/reports/forecast_improvement/technical_entry_mining_latest.json. The
# indicator boundaries mirror the miner's categorical labels (low/mid/high
# percentile zones, Bollinger %B location, RSI zone, MACD sign), so runtime
# still revalidates them against fresh pair_charts instead of treating the
# report row as live permission.
TECHNICAL_HARVEST_ROTATION_RULES: tuple[dict[str, Any], ...] = (
    {
        "name": "EUR_USD_DOWN_M5_BB_WIDTH_LOW_ROTATION_TP5_SL4",
        "pair": "EUR_USD",
        "side": "SHORT",
        "direction": "DOWN",
        "timeframe": "M5",
        "feature": "EUR_USD|DOWN|M5:bb_width_low",
        "samples": 57,
        "scalp_tp_pips": 5.0,
        "scalp_stop_pips": 4.0,
        "scalp_tp_first_hit_rate": 0.9474,
        "scalp_tp_first_wilson95_lower": 0.8563,
        "mfe_ge_2pip_hit_rate": 1.0,
        "mfe_ge_2pip_wilson95_lower": 0.9369,
        "avg_final_pips": 6.84,
        "max_m5_bb_width_percentile_100": 0.25,
        "min_target_pips": 4.8,
        "max_target_pips": 5.5,
        "max_stop_pips": 4.2,
        "audit_report": "logs/reports/forecast_improvement/technical_entry_mining_latest.json",
    },
    {
        "name": "EUR_USD_DOWN_M5_BB_MOMENTUM_ROTATION_TP5_SL4",
        "pair": "EUR_USD",
        "side": "SHORT",
        "direction": "DOWN",
        "timeframe": "M5",
        "feature": "EUR_USD|DOWN|M5:bb_momentum_aligned",
        "samples": 171,
        "scalp_tp_pips": 5.0,
        "scalp_stop_pips": 4.0,
        "scalp_tp_first_hit_rate": 0.8772,
        "scalp_tp_first_wilson95_lower": 0.8196,
        "mfe_ge_2pip_hit_rate": 0.9708,
        "mfe_ge_2pip_wilson95_lower": 0.9334,
        "avg_final_pips": 4.80,
        "max_m5_bb_pct_b": 0.50,
        "min_target_pips": 4.8,
        "max_target_pips": 5.5,
        "max_stop_pips": 4.2,
        "audit_report": "logs/reports/forecast_improvement/technical_entry_mining_latest.json",
    },
    {
        "name": "EUR_USD_DOWN_M1_CHOP_MID_ROTATION_TP5_SL4",
        "pair": "EUR_USD",
        "side": "SHORT",
        "direction": "DOWN",
        "timeframe": "M1",
        "feature": "EUR_USD|DOWN|M1:chop_mid",
        "samples": 174,
        "scalp_tp_pips": 5.0,
        "scalp_stop_pips": 4.0,
        "scalp_tp_first_hit_rate": 0.8678,
        "scalp_tp_first_wilson95_lower": 0.8095,
        "mfe_ge_2pip_hit_rate": 0.9713,
        "mfe_ge_2pip_wilson95_lower": 0.9345,
        "avg_final_pips": 5.04,
        "min_m1_choppiness_14": 38.2,
        "max_m1_choppiness_14": 61.8,
        "min_target_pips": 4.8,
        "max_target_pips": 5.5,
        "max_stop_pips": 4.2,
        "audit_report": "logs/reports/forecast_improvement/technical_entry_mining_latest.json",
    },
    {
        "name": "DOWN_M5_ATR_HIGH_ROTATION_TP5_SL4",
        "side": "SHORT",
        "direction": "DOWN",
        "timeframe": "M5",
        "feature": "DOWN|M5:atr_high",
        "samples": 203,
        "scalp_tp_pips": 5.0,
        "scalp_stop_pips": 4.0,
        "scalp_tp_first_hit_rate": 0.7685,
        "scalp_tp_first_wilson95_lower": 0.7058,
        "mfe_ge_2pip_hit_rate": 0.9606,
        "mfe_ge_2pip_wilson95_lower": 0.9242,
        "avg_final_pips": 4.00,
        "min_m5_atr_percentile_100": 0.75,
        "min_target_pips": 4.8,
        "max_target_pips": 5.5,
        "max_stop_pips": 4.2,
        "audit_report": "logs/reports/forecast_improvement/technical_entry_mining_latest.json",
    },
    {
        "name": "DOWN_M5_BB_MOMENTUM_ROTATION_TP5_SL4",
        "side": "SHORT",
        "direction": "DOWN",
        "timeframe": "M5",
        "feature": "DOWN|M5:bb_momentum_aligned",
        "samples": 276,
        "scalp_tp_pips": 5.0,
        "scalp_stop_pips": 4.0,
        "scalp_tp_first_hit_rate": 0.7826,
        "scalp_tp_first_wilson95_lower": 0.7302,
        "mfe_ge_2pip_hit_rate": 0.9493,
        "mfe_ge_2pip_wilson95_lower": 0.9167,
        "avg_final_pips": 3.03,
        "max_m5_bb_pct_b": 0.50,
        "min_target_pips": 4.8,
        "max_target_pips": 5.5,
        "max_stop_pips": 4.2,
        "audit_report": "logs/reports/forecast_improvement/technical_entry_mining_latest.json",
    },
    {
        "name": "DOWN_M15_BB_LOWER_ROTATION_TP5_SL4",
        "side": "SHORT",
        "direction": "DOWN",
        "timeframe": "M15",
        "feature": "DOWN|M15:bb_lower",
        "samples": 198,
        "scalp_tp_pips": 5.0,
        "scalp_stop_pips": 4.0,
        "scalp_tp_first_hit_rate": 0.7828,
        "scalp_tp_first_wilson95_lower": 0.7203,
        "mfe_ge_2pip_hit_rate": 0.9545,
        "mfe_ge_2pip_wilson95_lower": 0.9159,
        "avg_final_pips": 1.68,
        "max_m15_bb_pct_b": 0.20,
        "min_target_pips": 4.8,
        "max_target_pips": 5.5,
        "max_stop_pips": 4.2,
        "audit_report": "logs/reports/forecast_improvement/technical_entry_mining_latest.json",
    },
    {
        "name": "DOWN_M15_RSI_LOW_ROTATION_TP5_SL4",
        "side": "SHORT",
        "direction": "DOWN",
        "timeframe": "M15",
        "feature": "DOWN|M15:rsi_low",
        "samples": 155,
        "scalp_tp_pips": 5.0,
        "scalp_stop_pips": 4.0,
        "scalp_tp_first_hit_rate": 0.8129,
        "scalp_tp_first_wilson95_lower": 0.7442,
        "mfe_ge_2pip_hit_rate": 0.9548,
        "mfe_ge_2pip_wilson95_lower": 0.9097,
        "avg_final_pips": 2.70,
        "max_m15_rsi_14": 35.0,
        "min_target_pips": 4.8,
        "max_target_pips": 5.5,
        "max_stop_pips": 4.2,
        "audit_report": "logs/reports/forecast_improvement/technical_entry_mining_latest.json",
    },
    {
        "name": "DOWN_CROSS_M5M15_MACD_ROTATION_TP5_SL4",
        "side": "SHORT",
        "direction": "DOWN",
        "timeframe": "M5/M15",
        "feature": "DOWN|cross:M5M15:macd_all_aligned",
        "samples": 199,
        "scalp_tp_pips": 5.0,
        "scalp_stop_pips": 4.0,
        "scalp_tp_first_hit_rate": 0.7638,
        "scalp_tp_first_wilson95_lower": 0.7002,
        "mfe_ge_2pip_hit_rate": 0.9447,
        "mfe_ge_2pip_wilson95_lower": 0.9037,
        "avg_final_pips": 1.55,
        "max_m5_macd_hist": 0.0,
        "max_m15_macd_hist": 0.0,
        "min_target_pips": 4.8,
        "max_target_pips": 5.5,
        "max_stop_pips": 4.2,
        "audit_report": "logs/reports/forecast_improvement/technical_entry_mining_latest.json",
    },
)

TECHNICAL_HARVEST_NEGATIVE_RULES: tuple[dict[str, Any], ...] = (
    {
        "name": "EUR_USD_DOWN_M5_EMA_SLOPE5_OPPOSED_TP5_SL4",
        "pair": "EUR_USD",
        "side": "SHORT",
        "direction": "DOWN",
        "timeframe": "M5",
        "feature": "EUR_USD|DOWN|M5:ema_slope5_opposed",
        "samples": 31,
        "scalp_tp_pips": 5.0,
        "scalp_stop_pips": 4.0,
        "scalp_tp_first_hit_rate": 0.0,
        "scalp_tp_first_wilson95_lower": 0.0,
        "final_hit_rate": 0.0323,
        "avg_final_pips": -0.40,
        "min_m5_ema_slope_5": 0.0,
        "min_target_pips": 4.8,
        "max_target_pips": 5.5,
        "max_stop_pips": 4.2,
        "blocks_live_support": True,
        "audit_report": "logs/reports/forecast_improvement/technical_entry_mining_latest.json",
    },
    {
        "name": "DOWN_M15_ATR_HIGH_TP5_SL4",
        "side": "SHORT",
        "direction": "DOWN",
        "timeframe": "M15",
        "feature": "DOWN|M15:atr_high",
        "samples": 67,
        "scalp_tp_pips": 5.0,
        "scalp_stop_pips": 4.0,
        "scalp_tp_first_hit_rate": 0.4030,
        "scalp_tp_first_wilson95_lower": 0.2939,
        "final_hit_rate": 0.4030,
        "avg_final_pips": -2.41,
        "min_m15_atr_percentile_100": 0.75,
        "min_target_pips": 4.8,
        "max_target_pips": 5.5,
        "max_stop_pips": 4.2,
        "blocks_live_support": False,
        "audit_report": "logs/reports/forecast_improvement/technical_entry_mining_latest.json",
    },
    {
        "name": "DOWN_M15_BB_WIDTH_HIGH_TP5_SL4",
        "side": "SHORT",
        "direction": "DOWN",
        "timeframe": "M15",
        "feature": "DOWN|M15:bb_width_high",
        "samples": 75,
        "scalp_tp_pips": 5.0,
        "scalp_stop_pips": 4.0,
        "scalp_tp_first_hit_rate": 0.3067,
        "scalp_tp_first_wilson95_lower": 0.2139,
        "final_hit_rate": 0.3867,
        "avg_final_pips": -0.45,
        "min_m15_bb_width_percentile_100": 0.75,
        "min_target_pips": 4.8,
        "max_target_pips": 5.5,
        "max_stop_pips": 4.2,
        "blocks_live_support": False,
        "audit_report": "logs/reports/forecast_improvement/technical_entry_mining_latest.json",
    },
    {
        "name": "DOWN_M5_FAMILY_TREND_OPPOSED_TP5_SL4",
        "side": "SHORT",
        "direction": "DOWN",
        "timeframe": "M5",
        "feature": "DOWN|M5:family_trend_opposed",
        "samples": 75,
        "scalp_tp_pips": 5.0,
        "scalp_stop_pips": 4.0,
        "scalp_tp_first_hit_rate": 0.2933,
        "scalp_tp_first_wilson95_lower": 0.2024,
        "final_hit_rate": 0.4933,
        "avg_final_pips": -1.25,
        "min_m5_trend_score": 0.0,
        "min_target_pips": 4.8,
        "max_target_pips": 5.5,
        "max_stop_pips": 4.2,
        "blocks_live_support": False,
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
    assessment = technical_harvest_precision_assessment(
        metadata,
        pair=pair,
        side=side,
        order_type=order_type,
        method=method,
        entry=entry,
        take_profit=take_profit,
        stop_loss=stop_loss,
    )
    support = assessment.get("primary_support")
    return support if isinstance(support, dict) else None


def technical_harvest_precision_assessment(
    metadata: dict[str, Any],
    *,
    pair: str,
    side: str,
    order_type: str | None,
    method: str | None,
    entry: float | None,
    take_profit: float | None,
    stop_loss: float | None,
) -> dict[str, Any]:
    """Return positive and negative audited TP5/SL4 technical evidence.

    The previous live gate used only the winning buckets. Accuracy improves
    only when the same mined surface is also used to reject conditions that
    repeatedly failed the same TP-before-stop audit.
    """

    if not isinstance(metadata, dict):
        return _empty_technical_harvest_assessment()
    if str(order_type or "").upper() == "MARKET":
        return _empty_technical_harvest_assessment()
    if str(metadata.get("tp_execution_mode") or "").upper() != "ATTACHED_TECHNICAL_TP":
        return _empty_technical_harvest_assessment()
    if str(metadata.get("tp_target_intent") or "").upper() != "HARVEST":
        return _empty_technical_harvest_assessment()
    if str(metadata.get("opportunity_mode") or "").upper() not in {"", "HARVEST"}:
        return _empty_technical_harvest_assessment()
    normalized_pair = str(pair or "").upper()
    normalized_side = str(side or "").upper()
    normalized_direction = str(metadata.get("forecast_direction") or "").upper()
    chart_bias = str(metadata.get("chart_direction_bias") or "").upper()
    if chart_bias and chart_bias != normalized_side:
        return _empty_technical_harvest_assessment()
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
    if target_pips is None or stop_pips is None or stop_pips <= 0.0:
        return _empty_technical_harvest_assessment()

    positive_supports: list[dict[str, Any]] = []
    for rule in TECHNICAL_HARVEST_PRECISION_RULES:
        if normalized_pair != rule["pair"]:
            continue
        if normalized_side != rule["side"]:
            continue
        if normalized_direction != rule["direction"]:
            continue
        if target_pips < float(rule["min_target_pips"]) or target_pips > float(rule["max_target_pips"]):
            continue
        if stop_pips > float(rule["max_stop_pips"]):
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
        positive_supports.append(support)

    rotation_supports: list[dict[str, Any]] = []
    for rule in TECHNICAL_HARVEST_ROTATION_RULES:
        if rule.get("pair") and normalized_pair != str(rule["pair"]).upper():
            continue
        if normalized_side != rule["side"]:
            continue
        if normalized_direction != rule["direction"]:
            continue
        if target_pips < float(rule["min_target_pips"]) or target_pips > float(rule["max_target_pips"]):
            continue
        if stop_pips > float(rule["max_stop_pips"]):
            continue
        feature_values = _technical_rule_feature_values(metadata, rule)
        if feature_values is None:
            continue
        support = {
            "name": rule["name"],
            "pair": rule.get("pair") or normalized_pair,
            "side": rule["side"],
            "direction": rule["direction"],
            "feature": rule["feature"],
            "timeframe": rule["timeframe"],
            "samples": rule["samples"],
            "scalp_tp_first_hit_rate": rule["scalp_tp_first_hit_rate"],
            "scalp_tp_first_wilson95_lower": rule["scalp_tp_first_wilson95_lower"],
            "mfe_ge_2pip_hit_rate": rule["mfe_ge_2pip_hit_rate"],
            "mfe_ge_2pip_wilson95_lower": rule["mfe_ge_2pip_wilson95_lower"],
            "avg_final_pips": rule["avg_final_pips"],
            "scalp_tp_pips": rule["scalp_tp_pips"],
            "scalp_stop_pips": rule["scalp_stop_pips"],
            "current_target_pips": round(target_pips, 4),
            "current_stop_pips": round(stop_pips, 4),
            "audit_report": rule["audit_report"],
        }
        support.update(feature_values)
        rotation_supports.append(support)

    negative_matches: list[dict[str, Any]] = []
    for rule in TECHNICAL_HARVEST_NEGATIVE_RULES:
        if rule.get("pair") and normalized_pair != str(rule["pair"]).upper():
            continue
        if normalized_side != rule["side"]:
            continue
        if normalized_direction != rule["direction"]:
            continue
        if target_pips < float(rule["min_target_pips"]) or target_pips > float(rule["max_target_pips"]):
            continue
        if stop_pips > float(rule["max_stop_pips"]):
            continue
        feature_values = _technical_negative_rule_feature_values(metadata, rule)
        if feature_values is None:
            continue
        match = {
            "name": rule["name"],
            "pair": rule.get("pair") or normalized_pair,
            "side": rule["side"],
            "direction": rule["direction"],
            "feature": rule["feature"],
            "timeframe": rule["timeframe"],
            "samples": rule["samples"],
            "scalp_tp_first_hit_rate": rule["scalp_tp_first_hit_rate"],
            "scalp_tp_first_wilson95_lower": rule["scalp_tp_first_wilson95_lower"],
            "final_hit_rate": rule["final_hit_rate"],
            "avg_final_pips": rule["avg_final_pips"],
            "scalp_tp_pips": rule["scalp_tp_pips"],
            "scalp_stop_pips": rule["scalp_stop_pips"],
            "current_target_pips": round(target_pips, 4),
            "current_stop_pips": round(stop_pips, 4),
            "blocks_live_support": bool(rule.get("blocks_live_support")),
            "audit_report": rule["audit_report"],
        }
        match.update(feature_values)
        negative_matches.append(match)

    blocking_negative_matches = [
        item for item in negative_matches if bool(item.get("blocks_live_support"))
    ]
    primary_support = None
    if positive_supports and not blocking_negative_matches:
        primary_support = max(
            positive_supports,
            key=lambda item: (
                float(item.get("scalp_tp_first_wilson95_lower") or 0.0),
                int(item.get("samples") or 0),
            ),
        )
    primary_rotation_support = None
    if rotation_supports and not blocking_negative_matches:
        primary_rotation_support = max(
            rotation_supports,
            key=lambda item: (
                float(item.get("mfe_ge_2pip_wilson95_lower") or 0.0),
                float(item.get("scalp_tp_first_wilson95_lower") or 0.0),
                int(item.get("samples") or 0),
            ),
        )
    score_delta = 0.0
    if positive_supports and not blocking_negative_matches:
        score_delta += TECHNICAL_HARVEST_PRECISION_SCORE_BONUS
        score_delta += max(0, len(positive_supports) - 1) * TECHNICAL_HARVEST_PRECISION_EXTRA_MATCH_BONUS
    if rotation_supports and not blocking_negative_matches:
        score_delta += TECHNICAL_HARVEST_ROTATION_SCORE_BONUS
        score_delta += max(0, len(rotation_supports) - 1) * TECHNICAL_HARVEST_ROTATION_EXTRA_MATCH_BONUS
    if negative_matches:
        score_delta -= len(negative_matches) * TECHNICAL_HARVEST_NEGATIVE_SCORE_PENALTY
    return {
        "eligible_shape": True,
        "primary_support": primary_support,
        "primary_rotation_support": primary_rotation_support,
        "positive_supports": positive_supports,
        "rotation_supports": rotation_supports,
        "negative_matches": negative_matches,
        "blocking_negative_matches": blocking_negative_matches,
        "score_delta": round(score_delta, 4),
    }


def _empty_technical_harvest_assessment() -> dict[str, Any]:
    return {
        "eligible_shape": False,
        "primary_support": None,
        "primary_rotation_support": None,
        "positive_supports": [],
        "rotation_supports": [],
        "negative_matches": [],
        "blocking_negative_matches": [],
        "score_delta": 0.0,
    }


def technical_harvest_negative_precision_issue(
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
    """Return the first live-blocking negative technical precision bucket."""

    assessment = technical_harvest_precision_assessment(
        metadata,
        pair=pair,
        side=side,
        order_type=order_type,
        method=method,
        entry=entry,
        take_profit=take_profit,
        stop_loss=stop_loss,
    )
    blockers = assessment.get("blocking_negative_matches")
    if isinstance(blockers, list) and blockers:
        return blockers[0]
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
    if "min_m1_choppiness_14" in rule or "max_m1_choppiness_14" in rule:
        value = _safe_float(metadata.get("m1_choppiness_14"))
        if value is None:
            return None
        if "min_m1_choppiness_14" in rule and value < float(rule["min_m1_choppiness_14"]):
            return None
        if "max_m1_choppiness_14" in rule and value > float(rule["max_m1_choppiness_14"]):
            return None
        out["current_m1_choppiness_14"] = round(value, 4)
    if "min_m5_atr_percentile_100" in rule:
        value = _percentile_0_1(metadata.get("m5_atr_percentile_100"), metadata.get("m5_atr_percentile"))
        if value is None or value < float(rule["min_m5_atr_percentile_100"]):
            return None
        out["current_m5_atr_percentile_100"] = round(value, 4)
    if "max_m5_bb_width_percentile_100" in rule:
        value = _percentile_0_1(
            metadata.get("m5_bb_width_percentile_100"),
            metadata.get("m5_bb_width_percentile"),
        )
        if value is None or value > float(rule["max_m5_bb_width_percentile_100"]):
            return None
        out["current_m5_bb_width_percentile_100"] = round(value, 4)
    if "max_m5_bb_pct_b" in rule:
        value = _bb_pct_b_0_1(metadata, "m5")
        if value is None or value > float(rule["max_m5_bb_pct_b"]):
            return None
        out["current_m5_bb_pct_b"] = round(value, 4)
    if "max_m15_bb_width_percentile_100" in rule:
        value = _percentile_0_1(
            metadata.get("m15_bb_width_percentile_100"),
            metadata.get("m15_bb_width_percentile"),
        )
        if value is None or value > float(rule["max_m15_bb_width_percentile_100"]):
            return None
        out["current_m15_bb_width_percentile_100"] = round(value, 4)
    if "max_m15_bb_pct_b" in rule:
        value = _bb_pct_b_0_1(metadata, "m15")
        if value is None or value > float(rule["max_m15_bb_pct_b"]):
            return None
        out["current_m15_bb_pct_b"] = round(value, 4)
    if "max_m15_choppiness_14" in rule:
        value = _safe_float(metadata.get("m15_choppiness_14"))
        if value is None or value > float(rule["max_m15_choppiness_14"]):
            return None
        out["current_m15_choppiness_14"] = round(value, 4)
    if "max_m15_rsi_14" in rule:
        value = _safe_float(metadata.get("m15_rsi_14"))
        if value is None or value > float(rule["max_m15_rsi_14"]):
            return None
        out["current_m15_rsi_14"] = round(value, 4)
    if "max_m5_macd_hist" in rule:
        value = _safe_float(metadata.get("m5_macd_hist"))
        if value is None or value >= float(rule["max_m5_macd_hist"]):
            return None
        out["current_m5_macd_hist"] = round(value, 8)
    if "max_m15_macd_hist" in rule:
        value = _safe_float(metadata.get("m15_macd_hist"))
        if value is None or value >= float(rule["max_m15_macd_hist"]):
            return None
        out["current_m15_macd_hist"] = round(value, 8)
    return out or None


def _technical_negative_rule_feature_values(
    metadata: dict[str, Any],
    rule: dict[str, Any],
) -> dict[str, Any] | None:
    out: dict[str, Any] = {}
    if "min_m5_ema_slope_5" in rule:
        value = _safe_float(metadata.get("m5_ema_slope_5"))
        if value is None or value <= float(rule["min_m5_ema_slope_5"]):
            return None
        out["current_m5_ema_slope_5"] = round(value, 4)
    if "min_m15_atr_percentile_100" in rule:
        value = _percentile_0_1(metadata.get("m15_atr_percentile_100"), metadata.get("m15_atr_percentile"))
        if value is None or value < float(rule["min_m15_atr_percentile_100"]):
            return None
        out["current_m15_atr_percentile_100"] = round(value, 4)
    if "min_m15_bb_width_percentile_100" in rule:
        value = _percentile_0_1(
            metadata.get("m15_bb_width_percentile_100"),
            metadata.get("m15_bb_width_percentile"),
        )
        if value is None or value < float(rule["min_m15_bb_width_percentile_100"]):
            return None
        out["current_m15_bb_width_percentile_100"] = round(value, 4)
    if "min_m5_trend_score" in rule:
        value = _safe_float(metadata.get("m5_trend_score"))
        if value is None or value <= float(rule["min_m5_trend_score"]):
            return None
        out["current_m5_trend_score"] = round(value, 4)
    return out or None


def _percentile_0_1(primary: Any, secondary: Any = None) -> float | None:
    value = _safe_float(primary)
    if value is None:
        value = _safe_float(secondary)
    if value is None:
        return None
    if value > 1.0:
        value /= 100.0
    return max(0.0, min(1.0, value))


def _bb_pct_b_0_1(metadata: dict[str, Any], timeframe_prefix: str) -> float | None:
    direct = _safe_float(metadata.get(f"{timeframe_prefix}_bb_pct_b"))
    if direct is not None:
        return max(0.0, min(1.0, direct))
    close = _safe_float(metadata.get(f"{timeframe_prefix}_close"))
    lower = _safe_float(metadata.get(f"{timeframe_prefix}_bb_lower"))
    upper = _safe_float(metadata.get(f"{timeframe_prefix}_bb_upper"))
    if close is None or lower is None or upper is None or upper <= lower:
        return None
    return max(0.0, min(1.0, (close - lower) / (upper - lower)))


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
