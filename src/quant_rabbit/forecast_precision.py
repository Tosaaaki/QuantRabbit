"""Forecast precision helpers for live-entry gating.

The projection ledger can show very high raw hit rates for targets that are
too small to monetize after spread. Live gates therefore need both statistical
confidence and an execution-aware target-width check.
"""

from __future__ import annotations

import functools
import json
import math
import os
import re
from pathlib import Path
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
BIDASK_REPLAY_EDGE_SCORE_BONUS = 18.0
BIDASK_REPLAY_CONTRARIAN_SCORE_BONUS = 18.0
BIDASK_REPLAY_NEGATIVE_SCORE_PENALTY = 70.0
BIDASK_REPLAY_RULES_ENV = "QR_BIDASK_REPLAY_PRECISION_RULES"
DEFAULT_BIDASK_REPLAY_RULES_PATH = Path(__file__).with_name("bidask_replay_precision_rules.json")


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
        "name": "UP_M5_DISAGREE_M15_BB_REVERSION_HOLDOUT_ROTATION_TP5_SL4",
        "side": "LONG",
        "direction": "UP",
        "timeframe": "M5/M15",
        "feature": "direction:UP + M5:family_disagreement_high + M15:bb_reversion_not_aligned",
        "samples": 36,
        "validation_samples": 36,
        "train_samples": 77,
        "all_samples": 113,
        "scalp_tp_pips": 5.0,
        "scalp_stop_pips": 4.0,
        "scalp_tp_first_hit_rate": 0.5833,
        "scalp_tp_first_wilson95_lower": 0.4220,
        "mfe_ge_2pip_hit_rate": 1.0,
        "mfe_ge_2pip_wilson95_lower": 0.9036,
        "avg_final_pips": 11.21,
        "optimized_take_profit_pips": 10.0,
        "optimized_stop_loss_pips": 2.0,
        "optimized_validation_samples": 34,
        "optimized_validation_win_rate": 0.5588,
        "optimized_validation_win_wilson95_lower": 0.3945,
        "optimized_validation_avg_realized_pips": 3.13,
        "optimized_validation_profit_factor": 4.59,
        "optimized_validation_timeout_rate": 0.2647,
        "min_m5_family_disagreement": 0.75,
        "min_exclusive_m15_bb_pct_b": 0.20,
        "min_target_pips": 4.8,
        "max_target_pips": 5.5,
        "max_stop_pips": 4.2,
        "audit_report": "logs/reports/forecast_improvement/technical_entry_mining_latest.json",
    },
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


# Directional forecast history replayed on local OANDA S5 bid/ask candles.
# Runtime loads the audited rule set from bidask_replay_precision_rules.json
# so every pair/direction is judged by the same replay thresholds. The tuples
# below are the built-in fallback for environments missing that committed file.
BIDASK_REPLAY_EDGE_RULES: tuple[dict[str, Any], ...] = (
    {
        "name": "EUR_USD_DOWN_S5_BIDASK_HARVEST_TP5_SL7",
        "pair": "EUR_USD",
        "side": "SHORT",
        "direction": "DOWN",
        "granularity": "S5",
        "samples": 222,
        "directional_hit_rate": 0.7432,
        "avg_final_pips": 2.3225,
        "median_final_pips": 3.7000,
        "avg_mfe_pips": 5.1239,
        "avg_mae_pips": 4.2622,
        "optimized_take_profit_pips": 5.0,
        "optimized_stop_loss_pips": 7.0,
        "optimized_avg_realized_pips": 2.7440,
        "optimized_win_rate": 0.7520,
        "optimized_profit_factor": 3.7170,
        "min_target_pips": 4.8,
        "max_target_pips": 5.5,
        "max_stop_pips": 7.2,
        "audit_report": "logs/reports/forecast_improvement/oanda_history_replay_validate_20260620T155821Z.json",
    },
)

BIDASK_REPLAY_NEGATIVE_RULES: tuple[dict[str, Any], ...] = (
    {
        "name": "AUD_JPY_UP_S5_BIDASK_NEGATIVE_EXPECTANCY",
        "pair": "AUD_JPY",
        "side": "LONG",
        "direction": "UP",
        "granularity": "S5",
        "samples": 124,
        "directional_hit_rate": 0.2016,
        "avg_final_pips": -6.7589,
        "median_final_pips": -4.7000,
        "avg_mfe_pips": 3.7556,
        "avg_mae_pips": 14.3710,
        "reward_side_target_rate": 0.0089,
        "blocks_live_support": True,
        "audit_report": "logs/reports/forecast_improvement/oanda_history_replay_validate_20260620T155821Z.json",
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
        for optional_key in (
            "optimized_take_profit_pips",
            "optimized_stop_loss_pips",
            "optimized_validation_samples",
            "optimized_validation_win_rate",
            "optimized_validation_win_wilson95_lower",
            "optimized_validation_avg_realized_pips",
            "optimized_validation_profit_factor",
            "optimized_validation_timeout_rate",
        ):
            if optional_key in rule:
                support[optional_key] = rule[optional_key]
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


def bidask_replay_precision_support(
    metadata: dict[str, Any],
    *,
    pair: str,
    side: str,
    order_type: str | None,
    method: str | None,
    entry: float | None,
    take_profit: float | None,
    stop_loss: float | None,
    rules_path: str | Path | None = None,
) -> dict[str, Any] | None:
    """Return S5 bid/ask replay support for an executable harvest shape."""

    assessment = bidask_replay_precision_assessment(
        metadata,
        pair=pair,
        side=side,
        order_type=order_type,
        method=method,
        entry=entry,
        take_profit=take_profit,
        stop_loss=stop_loss,
        rules_path=rules_path,
    )
    support = assessment.get("primary_support")
    return support if isinstance(support, dict) else None


def bidask_replay_negative_precision_issue(
    metadata: dict[str, Any],
    *,
    pair: str,
    side: str,
    order_type: str | None,
    method: str | None,
    entry: float | None,
    take_profit: float | None,
    stop_loss: float | None,
    rules_path: str | Path | None = None,
) -> dict[str, Any] | None:
    """Return the first live-blocking S5 bid/ask negative replay bucket."""

    assessment = bidask_replay_precision_assessment(
        metadata,
        pair=pair,
        side=side,
        order_type=order_type,
        method=method,
        entry=entry,
        take_profit=take_profit,
        stop_loss=stop_loss,
        rules_path=rules_path,
    )
    blockers = assessment.get("blocking_negative_matches")
    if isinstance(blockers, list) and blockers:
        return blockers[0]
    return None


def bidask_replay_precision_assessment(
    metadata: dict[str, Any],
    *,
    pair: str,
    side: str,
    order_type: str | None,
    method: str | None,
    entry: float | None,
    take_profit: float | None,
    stop_loss: float | None,
    rules_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return S5 bid/ask replay evidence for the current pair/direction.

    Positive support is intentionally stricter than the negative block: it
    requires a non-market attached-TP HARVEST shape with TP/SL geometry close to
    the replayed profitable exit grid. Negative evidence blocks the same
    pair/direction forecast vehicle even if geometry later tries to paper over
    the direction-level loss.
    """

    if not isinstance(metadata, dict):
        return _empty_bidask_replay_assessment()
    normalized_pair = str(pair or "").upper()
    normalized_side = str(side or "").upper()
    normalized_direction = str(metadata.get("forecast_direction") or "").upper()
    chart_bias = str(metadata.get("chart_direction_bias") or "").upper()
    edge_rules, negative_rules, contrarian_rules, rule_source = _bidask_replay_rule_sets(rules_path)

    negative_matches: list[dict[str, Any]] = []
    for rule in negative_rules:
        if not _bidask_replay_rule_matches(rule, normalized_pair, normalized_side, normalized_direction):
            continue
        match = _bidask_replay_rule_payload(rule)
        match["blocks_live_support"] = bool(rule.get("blocks_live_support"))
        negative_matches.append(match)

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
    positive_supports: list[dict[str, Any]] = []
    contrarian_supports: list[dict[str, Any]] = []
    if (
        target_pips is not None
        and stop_pips is not None
        and stop_pips > 0.0
        and str(order_type or "").upper() != "MARKET"
        and str(metadata.get("tp_execution_mode") or "").upper() == "ATTACHED_TECHNICAL_TP"
        and str(metadata.get("tp_target_intent") or "").upper() == "HARVEST"
        and str(metadata.get("opportunity_mode") or "").upper() in {"", "HARVEST"}
    ):
        if not chart_bias or chart_bias == normalized_side:
            for rule in edge_rules:
                if not _bidask_replay_rule_matches(rule, normalized_pair, normalized_side, normalized_direction):
                    continue
                if target_pips < float(rule["min_target_pips"]) or target_pips > float(rule["max_target_pips"]):
                    continue
                if stop_pips > float(rule["max_stop_pips"]):
                    continue
                support = _bidask_replay_rule_payload(rule)
                support.update(
                    {
                        "current_target_pips": round(target_pips, 4),
                        "current_stop_pips": round(stop_pips, 4),
                    }
                )
                positive_supports.append(support)
        for rule in contrarian_rules:
            if not _bidask_replay_contrarian_rule_matches(
                rule,
                normalized_pair,
                normalized_side,
                normalized_direction,
                metadata,
            ):
                continue
            if target_pips < float(rule["min_target_pips"]) or target_pips > float(rule["max_target_pips"]):
                continue
            if stop_pips > float(rule["max_stop_pips"]):
                continue
            support = _bidask_replay_rule_payload(rule)
            support.update(
                {
                    "current_target_pips": round(target_pips, 4),
                    "current_stop_pips": round(stop_pips, 4),
                }
            )
            contrarian_supports.append(support)

    blocking_negative_matches = [
        item for item in negative_matches if bool(item.get("blocks_live_support"))
    ]
    primary_support = None
    supported_matches = [*positive_supports, *contrarian_supports]
    if supported_matches and not blocking_negative_matches:
        primary_support = max(
            supported_matches,
            key=lambda item: (
                int(item.get("daily_stability_status") == "DAILY_STABLE"),
                float(item.get("optimized_profit_factor") or 0.0),
                float(item.get("avg_final_pips") or 0.0),
                int(item.get("samples") or 0),
                int(bool(item.get("horizon_bucket"))) + int(bool(item.get("confidence_bucket"))),
            ),
        )
    score_delta = 0.0
    if supported_matches and not blocking_negative_matches:
        primary_is_contrarian = bool(primary_support and primary_support.get("contrarian_edge"))
        score_delta += (
            BIDASK_REPLAY_CONTRARIAN_SCORE_BONUS
            if primary_is_contrarian
            else BIDASK_REPLAY_EDGE_SCORE_BONUS
        )
    if negative_matches:
        score_delta -= len(negative_matches) * BIDASK_REPLAY_NEGATIVE_SCORE_PENALTY
    return {
        "eligible_shape": bool(supported_matches or negative_matches),
        "primary_support": primary_support,
        "positive_supports": positive_supports,
        "contrarian_supports": contrarian_supports,
        "negative_matches": negative_matches,
        "blocking_negative_matches": blocking_negative_matches,
        "rule_source": rule_source,
        "score_delta": round(score_delta, 4),
    }


def _empty_bidask_replay_assessment() -> dict[str, Any]:
    return {
        "eligible_shape": False,
        "primary_support": None,
        "positive_supports": [],
        "contrarian_supports": [],
        "negative_matches": [],
        "blocking_negative_matches": [],
        "rule_source": None,
        "score_delta": 0.0,
    }


def _bidask_replay_rule_sets(
    rules_path: str | Path | None = None,
) -> tuple[
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    dict[str, Any],
]:
    explicit = str(rules_path) if rules_path is not None else os.environ.get(BIDASK_REPLAY_RULES_ENV)
    path = Path(explicit).expanduser() if explicit else DEFAULT_BIDASK_REPLAY_RULES_PATH
    loaded_edge, loaded_negative, loaded_contrarian, source = _load_bidask_replay_rule_sets(str(path))
    if loaded_edge or loaded_negative or loaded_contrarian:
        return loaded_edge, loaded_negative, loaded_contrarian, source
    return (
        BIDASK_REPLAY_EDGE_RULES,
        BIDASK_REPLAY_NEGATIVE_RULES,
        (),
        {
            "type": "builtin_fallback",
            "path": str(path),
            "reason": "missing_or_empty_rule_file",
        },
    )


@functools.lru_cache(maxsize=8)
def _load_bidask_replay_rule_sets(
    path_text: str,
) -> tuple[
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    dict[str, Any],
]:
    path = Path(path_text)
    if not path.exists():
        return (), (), (), {"type": "missing_file", "path": str(path)}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return (), (), (), {"type": "unreadable_file", "path": str(path)}
    if not isinstance(payload, dict):
        return (), (), (), {"type": "invalid_schema", "path": str(path)}
    source = {
        "type": "json_file",
        "path": str(path),
        "schema_version": payload.get("schema_version"),
        "generated_at_utc": payload.get("generated_at_utc"),
        "generated_from": payload.get("generated_from"),
    }
    edge = _normalize_bidask_replay_rules(
        payload.get("edge_rules"),
        source=source,
        require_geometry=True,
    )
    negative = _normalize_bidask_replay_rules(
        payload.get("negative_rules"),
        source=source,
        require_geometry=False,
    )
    contrarian = _normalize_bidask_replay_rules(
        payload.get("contrarian_edge_rules"),
        source=source,
        require_geometry=True,
    )
    return edge, negative, contrarian, source


def _normalize_bidask_replay_rules(
    raw_rules: Any,
    *,
    source: dict[str, Any],
    require_geometry: bool,
) -> tuple[dict[str, Any], ...]:
    if not isinstance(raw_rules, list):
        return ()
    out: list[dict[str, Any]] = []
    for raw in raw_rules:
        if not isinstance(raw, dict):
            continue
        rule = dict(raw)
        pair = str(rule.get("pair") or "").upper().strip()
        direction = str(rule.get("direction") or "").upper().strip()
        side = str(rule.get("side") or "").upper().strip() or _side_for_direction(direction)
        if not pair or direction not in {"UP", "DOWN"} or side not in {"LONG", "SHORT"}:
            continue
        faded_direction = str(rule.get("faded_direction") or rule.get("forecast_direction") or "").upper().strip()
        if bool(rule.get("contrarian_edge")) or faded_direction:
            if faded_direction not in {"UP", "DOWN"} or faded_direction == direction:
                continue
            rule["forecast_direction"] = faded_direction
            rule["faded_direction"] = faded_direction
            rule["contrarian_edge"] = True
        try:
            samples = int(rule.get("samples") or 0)
        except (TypeError, ValueError):
            continue
        if samples <= 0:
            continue
        if require_geometry and not _rule_has_positive_geometry(rule):
            continue
        rule["pair"] = pair
        rule["direction"] = direction
        rule["side"] = side
        rule["samples"] = samples
        rule.setdefault("rule_set_generated_at_utc", source.get("generated_at_utc"))
        rule.setdefault("rule_set_source", source.get("generated_from") or source.get("path"))
        out.append(rule)
    return tuple(out)


def _rule_has_positive_geometry(rule: dict[str, Any]) -> bool:
    for key in ("min_target_pips", "max_target_pips", "max_stop_pips"):
        try:
            value = float(rule.get(key))
        except (TypeError, ValueError):
            return False
        if value <= 0.0:
            return False
    return True


def _side_for_direction(direction: str) -> str:
    if direction == "UP":
        return "LONG"
    if direction == "DOWN":
        return "SHORT"
    return ""


def _bidask_replay_rule_matches(
    rule: dict[str, Any],
    pair: str,
    side: str,
    direction: str,
) -> bool:
    return (
        pair == str(rule.get("pair") or "").upper()
        and side == str(rule.get("side") or "").upper()
        and direction == str(rule.get("direction") or "").upper()
    )


def _bidask_replay_contrarian_rule_matches(
    rule: dict[str, Any],
    pair: str,
    side: str,
    forecast_direction: str,
    metadata: dict[str, Any],
) -> bool:
    if not (
        bool(rule.get("contrarian_edge"))
        and pair == str(rule.get("pair") or "").upper()
        and side == str(rule.get("side") or "").upper()
        and forecast_direction == str(rule.get("faded_direction") or rule.get("forecast_direction") or "").upper()
    ):
        return False
    confidence_bucket = str(rule.get("confidence_bucket") or "").strip()
    if confidence_bucket and confidence_bucket != _forecast_confidence_bucket(metadata):
        return False
    horizon_bucket = str(rule.get("horizon_bucket") or "").strip()
    if horizon_bucket and horizon_bucket != _forecast_horizon_bucket(metadata):
        return False
    return True


def _forecast_confidence_bucket(metadata: dict[str, Any]) -> str | None:
    value = _safe_float(
        metadata.get("forecast_confidence")
        if metadata.get("forecast_confidence") is not None
        else metadata.get("confidence")
    )
    if value is None:
        return "missing"
    if value < 0.50:
        return "<0.50"
    if value < 0.65:
        return "0.50-0.65"
    if value < 0.75:
        return "0.65-0.75"
    if value < 0.90:
        return "0.75-0.90"
    return ">=0.90"


def _forecast_horizon_bucket(metadata: dict[str, Any]) -> str | None:
    value = _safe_float(
        metadata.get("forecast_horizon_min")
        if metadata.get("forecast_horizon_min") is not None
        else metadata.get("horizon_min")
    )
    if value is None:
        return None
    if value <= 15:
        return "<=15m"
    if value <= 30:
        return "16-30m"
    if value <= 60:
        return "31-60m"
    if value <= 240:
        return "61-240m"
    return ">240m"


def _bidask_replay_rule_payload(rule: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "name",
        "pair",
        "side",
        "direction",
        "forecast_direction",
        "faded_direction",
        "contrarian_edge",
        "horizon_bucket",
        "confidence_bucket",
        "granularity",
        "samples",
        "source_directional_hit_rate",
        "source_avg_final_pips",
        "source_avg_mfe_pips",
        "source_avg_mae_pips",
        "directional_hit_rate",
        "avg_final_pips",
        "median_final_pips",
        "avg_mfe_pips",
        "avg_mae_pips",
        "reward_side_target_rate",
        "optimized_take_profit_pips",
        "optimized_stop_loss_pips",
        "optimized_avg_realized_pips",
        "optimized_win_rate",
        "optimized_profit_factor",
        "daily_stability_status",
        "campaign_timezone",
        "active_days",
        "first_day",
        "last_day",
        "min_daily_samples",
        "max_daily_samples",
        "avg_daily_samples",
        "max_daily_sample_share",
        "positive_days",
        "negative_days",
        "flat_days",
        "positive_day_rate",
        "avg_daily_realized_pips",
        "worst_daily_realized_pips",
        "best_daily_realized_pips",
        "audit_report",
        "rule_set_generated_at_utc",
        "rule_set_source",
    )
    return {key: rule[key] for key in keys if key in rule}


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
    if "min_m5_family_disagreement" in rule:
        value = _safe_float(metadata.get("m5_family_disagreement"))
        if value is None or value < float(rule["min_m5_family_disagreement"]):
            return None
        out["current_m5_family_disagreement"] = round(value, 4)
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
    if "min_exclusive_m15_bb_pct_b" in rule:
        value = _bb_pct_b_0_1(metadata, "m15")
        if value is None or value <= float(rule["min_exclusive_m15_bb_pct_b"]):
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
