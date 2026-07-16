"""Normalized point-in-time technical context for forecast evaluation.

The directional forecaster already consumes rich chart state, but historical
rows used to retain only its final scores and prose drivers.  This module
freezes a small, deterministic subset of the chart state at forecast creation
time so later replay can test conditions without reconstructing them from
future or mutable artifacts.
"""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from quant_rabbit.instruments import instrument_pip_factor
from quant_rabbit.strategy.failed_break_evidence import (
    build_m5_failed_break_evidence,
    verify_m5_failed_break_evidence,
)
from quant_rabbit.strategy.tf_weights import (
    build_dynamic_tf_classifier_inputs,
    classify_situation_from_classifier_inputs,
    verify_dynamic_tf_policy_evidence,
)
from quant_rabbit.strategy.regime_family_weighting import (
    build_regime_family_weighting_receipt,
    regime_family_state_from_view,
    verify_regime_family_weighting_context_binding,
    verify_regime_family_weighting_receipt,
)


SCHEMA_VERSION = "technical_context_v1"
TECHNICAL_CONTEXT_EVIDENCE_CONTRACT = "QR_FORECAST_TECHNICAL_CONTEXT_EVIDENCE_V1"
CONFIDENCE_SEMANTICS = "CALIBRATED_SCORE_NOT_WIN_PROBABILITY"
PAIR_CHART_SOURCE_CONTRACT = "QR_FORECAST_PAIR_CHART_SOURCE_ROW_V1"
MAX_CONTEXT_BYTES = 16384
# A VALID envelope contains one body already bounded by MAX_CONTEXT_BYTES plus
# fixed contract flags and two SHA-256 strings.  Reserve 2 KiB for that wrapper;
# UNKNOWN envelopes are much smaller because they never retain a body.
MAX_EVIDENCE_BYTES = MAX_CONTEXT_BYTES + 2048
# Reasons are machine-owned diagnostic codes/messages, not an artifact echo or
# a prompt channel.  This character cap also bounds worst-case UTF-8 expansion
# well inside the envelope allowance.
MAX_REASON_CHARS = 256
TOP_LEVEL_FIELDS = {
    "schema_version",
    "identity",
    "regime",
    "volatility",
    "execution",
    "location",
    "structure",
    "families",
    "completeness",
    "context_sha256",
}
# Historical v1 contexts froze only the four execution/primary frames.  Keep
# those receipt-less bodies readable for replay, while every newly built body
# freezes all seven frames used by the weighting receipt.
LEGACY_CONTEXT_TIMEFRAMES = ("M5", "M15", "H1", "H4")
CONTEXT_TIMEFRAMES = ("D", "H4", "H1", "M30", "M15", "M5", "M1")
INTERMEDIATE_TOP_LEVEL_FIELDS = {*TOP_LEVEL_FIELDS, "regime_family_weighting"}
FAILED_BREAK_ONLY_TOP_LEVEL_FIELDS = {
    *TOP_LEVEL_FIELDS,
    "m5_failed_break_evidence",
}
CURRENT_DISPLAY_TOP_LEVEL_FIELDS = {
    *FAILED_BREAK_ONLY_TOP_LEVEL_FIELDS,
    "dynamic_tf_policy_evidence",
}
CURRENT_POLICY_SOURCE_DISPLAY_TOP_LEVEL_FIELDS = {
    *CURRENT_DISPLAY_TOP_LEVEL_FIELDS,
    "dynamic_tf_policy_source_context",
}
INTERMEDIATE_FAILED_BREAK_TOP_LEVEL_FIELDS = {
    *INTERMEDIATE_TOP_LEVEL_FIELDS,
    "m5_failed_break_evidence",
}
INTERMEDIATE_POLICY_TOP_LEVEL_FIELDS = {
    *INTERMEDIATE_TOP_LEVEL_FIELDS,
    "dynamic_tf_policy_evidence",
}
PRE_POLICY_SOURCE_CURRENT_TOP_LEVEL_FIELDS = {
    *INTERMEDIATE_TOP_LEVEL_FIELDS,
    "m5_failed_break_evidence",
    "dynamic_tf_policy_evidence",
}
CURRENT_TOP_LEVEL_FIELDS = {
    *PRE_POLICY_SOURCE_CURRENT_TOP_LEVEL_FIELDS,
    "dynamic_tf_policy_source_context",
}
DYNAMIC_TF_POLICY_SOURCE_CONTEXT_FIELDS = {
    "classifier_inputs",
    "news_source",
    "strategy_profile_source",
    "pair_chart_row_sha256",
    "evaluated_at_utc",
}
REQUIRED_FIELDS = (
    "identity.pair",
    "regime.dominant",
    "volatility.primary_atr_band",
    "execution.spread_band",
    "location.range_location_24h",
    "structure.primary_direction",
)
TECHNICAL_CONTEXT_EVIDENCE_FIELDS = {
    "contract",
    "status",
    "reason",
    "confidence_semantics",
    "technical_context_v1",
    "context_sha256",
    "evidence_sha256",
    "read_only",
    "proof_eligible",
    "live_permission",
}
IDENTITY_FIELDS = {"pair"}
REGIME_FIELDS = {"dominant", "primary", "by_timeframe"}
LEGACY_VOLATILITY_FIELDS = {
    "primary_atr_percentile",
    "primary_atr_band",
    "atr_percentile_by_timeframe",
    "atr_band_by_timeframe",
    "atr_pips_by_timeframe",
}
VOLATILITY_FIELDS = {
    *LEGACY_VOLATILITY_FIELDS,
    "confluence_atr_percentile_24h",
}
EXECUTION_FIELDS = {
    "spread_pips",
    "m5_atr_pips",
    "spread_to_m5_atr",
    "spread_band",
}
LOCATION_FIELDS = {
    "current_price",
    "price_percentile_24h",
    "range_location_24h",
    "price_percentile_7d",
    "range_location_7d",
}
STRUCTURE_FIELDS = {
    "primary_timeframe",
    "primary_kind",
    "primary_direction",
    "by_timeframe",
}
STRUCTURE_ITEM_FIELDS = {"kind", "direction", "index", "timestamp_utc"}
FAMILIES_FIELDS = {"by_timeframe"}
FAMILY_SCORE_FIELDS = {
    "trend_score",
    "mean_reversion_score",
    "breakout_score",
    "disagreement",
}
COMPLETENESS_FIELDS = {"required_fields", "missing_fields", "complete"}
ATR_BANDS = {"UNKNOWN", "LOW", "NORMAL", "HIGH"}
SPREAD_BANDS = {"UNKNOWN", "TIGHT", "NORMAL", "WIDE"}
RANGE_LOCATIONS = {"UNKNOWN", "LOWER", "MIDDLE", "UPPER"}
STRUCTURE_DIRECTIONS = {"UNKNOWN", "UP", "DOWN"}
# Serialization/model-handoff safety ceiling, not a market or entry threshold.
# Prices, pip measures, and candle indexes are many orders below this value;
# bounding them prevents huge finite JSON numbers from reaching GPT consumers.
MAX_CONTEXT_NUMERIC_ABS = 1_000_000_000_000.0
# The live context cohort is approximately -2.27..2.25.  This deliberately
# generous diagnostic bound preserves future score evolution without allowing
# a malformed score to dominate or bloat the bounded GPT packet.
MAX_FAMILY_SCORE_ABS = 100.0


def build_forecast_technical_context(
    pair_chart: Mapping[str, Any] | None,
    *,
    pair: str,
    current_price: float | None,
    spread_pips: float | None,
    calendar_path: Path | None = None,
    strategy_profile_path: Path | None = None,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    """Return a bounded content-addressed forecast-time context.

    Buckets are descriptive audit cohorts, not trading thresholds.  Raw values
    are retained beside every bucket so future evaluators can reproduce the
    classification exactly.
    """

    chart = pair_chart if isinstance(pair_chart, Mapping) else {}
    confluence = chart.get("confluence") if isinstance(chart.get("confluence"), Mapping) else {}
    views = _views_by_timeframe(chart)

    regime_by_timeframe: dict[str, str] = {}
    atr_percentile_by_timeframe: dict[str, float | None] = {}
    atr_band_by_timeframe: dict[str, str] = {}
    atr_pips_by_timeframe: dict[str, float | None] = {}
    structure_by_timeframe: dict[str, dict[str, Any]] = {}
    family_scores_by_timeframe: dict[str, dict[str, float | None]] = {}
    for timeframe in CONTEXT_TIMEFRAMES:
        view = views.get(timeframe) or {}
        reading = view.get("regime_reading") if isinstance(view.get("regime_reading"), Mapping) else {}
        indicators = view.get("indicators") if isinstance(view.get("indicators"), Mapping) else {}
        regime_by_timeframe[timeframe], _regime_confidence = (
            regime_family_state_from_view(view)
        )
        reading_atr_percentile = reading.get("atr_percentile")
        atr_percentile = _round_or_none(
            (
                _bounded_percent_0_100(reading_atr_percentile)
                if reading_atr_percentile is not None
                else _fraction_percentile_to_100(
                    indicators.get("atr_percentile_100")
                )
            ),
            4,
        )
        atr_percentile_by_timeframe[timeframe] = atr_percentile
        atr_band_by_timeframe[timeframe] = _atr_band(atr_percentile)
        atr_pips_by_timeframe[timeframe] = _round_or_none(
            _number(indicators.get("atr_pips")), 6
        )
        structure_by_timeframe[timeframe] = _latest_close_confirmed_structure(view)
        families = view.get("family_scores") if isinstance(view.get("family_scores"), Mapping) else {}
        family_scores_by_timeframe[timeframe] = {
            "trend_score": _round_or_none(
                _family_score_number((families or {}).get("trend_score")), 6
            ),
            "mean_reversion_score": _round_or_none(
                _family_score_number((families or {}).get("mean_rev_score")), 6
            ),
            "breakout_score": _round_or_none(
                _family_score_number((families or {}).get("breakout_score")), 6
            ),
            "disagreement": _round_or_none(
                _family_score_number((families or {}).get("disagreement")), 6
            ),
        }

    dominant_regime = _normalized_regime((confluence or {}).get("dominant_regime"))
    primary_regime = _first_known(
        regime_by_timeframe.get("M15"),
        regime_by_timeframe.get("H1"),
        dominant_regime,
    )
    confluence_atr_percentile_24h = _round_or_none(
        # chart_reader stores this confluence fallback directly from the
        # IndicatorSet's 0..1 quantile rank.
        _fraction_percentile_to_100(
            (confluence or {}).get("atr_percentile_24h")
        ),
        4,
    )
    primary_atr_percentile = _round_or_none(
        _first_number(
            atr_percentile_by_timeframe.get("M15"),
            atr_percentile_by_timeframe.get("M5"),
            confluence_atr_percentile_24h,
        ),
        4,
    )
    primary_atr_band = _atr_band(primary_atr_percentile)
    m5_atr_pips = atr_pips_by_timeframe.get("M5")
    spread = _round_or_none(_number(spread_pips), 6)
    spread_to_m5_atr = _round_or_none(
        _number(spread / m5_atr_pips)
        if spread is not None and m5_atr_pips is not None and m5_atr_pips > 0.0
        else None,
        6,
    )
    location_24h_value = _round_or_none(
        _fraction_0_1((confluence or {}).get("price_percentile_24h")),
        6,
    )
    location_7d_value = _round_or_none(
        _fraction_0_1((confluence or {}).get("price_percentile_7d")),
        6,
    )
    primary_structure = _primary_structure(structure_by_timeframe)
    m5_failed_break_evidence = build_m5_failed_break_evidence(chart)
    weighting_result = build_regime_family_weighting_receipt(
        chart,
        pair=pair,
        calendar_path=calendar_path,
        strategy_profile_path=strategy_profile_path,
        now_utc=now_utc,
        m5_failed_break_evidence=m5_failed_break_evidence,
        include_policy_evidence=True,
    )
    regime_family_weighting, dynamic_tf_policy_evidence = weighting_result
    weighting_source = regime_family_weighting.get("source_identity") or {}
    news_evidence = dynamic_tf_policy_evidence.get("news_evidence") or {}
    strategy_evidence = (
        dynamic_tf_policy_evidence.get("strategy_profile_evidence") or {}
    )
    dynamic_tf_policy_source_context = {
        "classifier_inputs": build_dynamic_tf_classifier_inputs(
            session=weighting_source.get("session"),
            chart_story=str(chart.get("chart_story") or ""),
            dominant_regime=weighting_source.get("dominant_regime"),
        ),
        "news_source": deepcopy(dict(news_evidence.get("source") or {})),
        "strategy_profile_source": deepcopy(
            dict(strategy_evidence.get("source") or {})
        ),
        "pair_chart_row_sha256": forecast_pair_chart_row_sha256(
            chart,
            pair=pair,
        ),
        "evaluated_at_utc": news_evidence.get("evaluated_at_utc"),
    }

    body: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "identity": {"pair": str(pair or "").strip().upper() or "UNKNOWN"},
        "regime": {
            "dominant": dominant_regime,
            "primary": primary_regime,
            "by_timeframe": regime_by_timeframe,
        },
        "volatility": {
            "primary_atr_percentile": primary_atr_percentile,
            "primary_atr_band": primary_atr_band,
            "confluence_atr_percentile_24h": confluence_atr_percentile_24h,
            "atr_percentile_by_timeframe": atr_percentile_by_timeframe,
            "atr_band_by_timeframe": atr_band_by_timeframe,
            "atr_pips_by_timeframe": atr_pips_by_timeframe,
        },
        "execution": {
            "spread_pips": spread,
            "m5_atr_pips": m5_atr_pips,
            "spread_to_m5_atr": spread_to_m5_atr,
            "spread_band": _spread_band(spread_to_m5_atr),
        },
        "location": {
            "current_price": _round_or_none(_number(current_price), 8),
            "price_percentile_24h": location_24h_value,
            "range_location_24h": _range_location(location_24h_value),
            "price_percentile_7d": location_7d_value,
            "range_location_7d": _range_location(location_7d_value),
        },
        "structure": {
            "primary_timeframe": primary_structure.get("timeframe"),
            "primary_kind": primary_structure.get("kind"),
            "primary_direction": primary_structure.get("direction", "UNKNOWN"),
            "by_timeframe": structure_by_timeframe,
        },
        "families": {"by_timeframe": family_scores_by_timeframe},
        "m5_failed_break_evidence": m5_failed_break_evidence,
        "dynamic_tf_policy_evidence": dynamic_tf_policy_evidence,
        "dynamic_tf_policy_source_context": dynamic_tf_policy_source_context,
        "regime_family_weighting": regime_family_weighting,
    }
    missing = _missing_required_fields(body)
    body["completeness"] = {
        "required_fields": list(REQUIRED_FIELDS),
        "missing_fields": missing,
        "complete": not missing,
    }
    body["context_sha256"] = technical_context_sha256(body)
    return body


def verify_forecast_technical_context(
    value: object,
    *,
    pair: str | None = None,
    current_price: float | None = None,
) -> tuple[bool, str | None]:
    if not isinstance(value, Mapping):
        return False, "TECHNICAL_CONTEXT_MISSING"
    if value.get("schema_version") != SCHEMA_VERSION:
        return False, "TECHNICAL_CONTEXT_SCHEMA_INVALID"
    context_fields = set(value)
    if context_fields not in (
        TOP_LEVEL_FIELDS,
        INTERMEDIATE_TOP_LEVEL_FIELDS,
        FAILED_BREAK_ONLY_TOP_LEVEL_FIELDS,
        CURRENT_DISPLAY_TOP_LEVEL_FIELDS,
        CURRENT_POLICY_SOURCE_DISPLAY_TOP_LEVEL_FIELDS,
        INTERMEDIATE_FAILED_BREAK_TOP_LEVEL_FIELDS,
        INTERMEDIATE_POLICY_TOP_LEVEL_FIELDS,
        PRE_POLICY_SOURCE_CURRENT_TOP_LEVEL_FIELDS,
        CURRENT_TOP_LEVEL_FIELDS,
    ):
        return False, "TECHNICAL_CONTEXT_SCHEMA_INVALID"
    try:
        encoded_size = len(
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
    except (TypeError, ValueError):
        return False, "TECHNICAL_CONTEXT_HASH_MISMATCH"
    if encoded_size > MAX_CONTEXT_BYTES:
        return False, "TECHNICAL_CONTEXT_TOO_LARGE"
    stored = str(value.get("context_sha256") or "").strip().lower()
    try:
        expected = technical_context_sha256(value)
    except (TypeError, ValueError):
        return False, "TECHNICAL_CONTEXT_HASH_MISMATCH"
    if len(stored) != 64 or stored != expected:
        return False, "TECHNICAL_CONTEXT_HASH_MISMATCH"
    schema_error = _technical_context_nested_schema_error(value)
    if schema_error is not None:
        return False, schema_error
    identity = value.get("identity") if isinstance(value.get("identity"), Mapping) else {}
    if pair is not None:
        stored_pair = str((identity or {}).get("pair") or "").upper()
        if stored_pair != str(pair or "").strip().upper():
            return False, "TECHNICAL_CONTEXT_PAIR_MISMATCH"
    if current_price is not None:
        location = value.get("location") if isinstance(value.get("location"), Mapping) else {}
        stored_price = _number((location or {}).get("current_price"))
        expected_price = _number(current_price)
        if (
            stored_price is None
            or expected_price is None
        ):
            return False, "TECHNICAL_CONTEXT_PRICE_MISMATCH"
        pair_name = str(pair or (identity or {}).get("pair") or "").upper()
        try:
            quote_unit = 1.0 / (float(instrument_pip_factor(pair_name)) * 10.0)
        except (TypeError, ValueError, ZeroDivisionError):
            quote_unit = 1e-8
        if abs(stored_price - expected_price) > quote_unit + 1e-12:
            return False, "TECHNICAL_CONTEXT_PRICE_MISMATCH"
    return True, None


def forecast_technical_context_is_current_for_allocation(value: object) -> bool:
    """Return whether ``value`` is the exact fresh-allocation context shape.

    Historical four-frame and intermediate v1 bodies remain readable for
    replay/display, but they cannot authorize capital.  Current allocation
    requires all seven independently bound frames plus both raw policy and M5
    failed-break evidence siblings.
    """

    if not isinstance(value, Mapping) or set(value) != CURRENT_TOP_LEVEL_FIELDS:
        return False
    valid, _error = verify_forecast_technical_context(value)
    if not valid:
        return False
    regime = value.get("regime")
    by_timeframe = (
        regime.get("by_timeframe")
        if isinstance(regime, Mapping)
        else None
    )
    return isinstance(by_timeframe, Mapping) and set(by_timeframe) == set(
        CONTEXT_TIMEFRAMES
    )


def _technical_context_nested_schema_error(
    value: Mapping[str, Any],
) -> str | None:
    """Validate the exact builder-owned nested schema and completeness truth.

    The context hash proves byte integrity, not semantic validity.  Recomputing
    a hash over a hand-authored body must therefore not be enough to set the GPT
    allocation context to ``VALID``.  This validator intentionally mirrors the
    bounded shape emitted by :func:`build_forecast_technical_context`.
    """

    context_fields = set(value)
    current_context = context_fields == CURRENT_TOP_LEVEL_FIELDS
    timeframes = (
        CONTEXT_TIMEFRAMES
        if current_context
        else _stored_context_timeframes(value)
    )
    if timeframes is None:
        return "TECHNICAL_CONTEXT_SCHEMA_INVALID"

    identity = value.get("identity")
    if not _mapping_has_exact_fields(identity, IDENTITY_FIELDS):
        return "TECHNICAL_CONTEXT_SCHEMA_INVALID"
    pair = identity.get("pair")
    if not _canonical_nonempty_upper_text(pair):
        return "TECHNICAL_CONTEXT_SCHEMA_INVALID"

    regime = value.get("regime")
    if not _mapping_has_exact_fields(regime, REGIME_FIELDS):
        return "TECHNICAL_CONTEXT_SCHEMA_INVALID"
    if not _canonical_regime(regime.get("dominant")) or not _canonical_regime(
        regime.get("primary")
    ):
        return "TECHNICAL_CONTEXT_SCHEMA_INVALID"
    regime_by_timeframe = regime.get("by_timeframe")
    if not _timeframe_mapping(regime_by_timeframe, timeframes=timeframes):
        return "TECHNICAL_CONTEXT_SCHEMA_INVALID"
    if any(
        not _canonical_regime(regime_by_timeframe.get(timeframe))
        for timeframe in timeframes
    ):
        return "TECHNICAL_CONTEXT_SCHEMA_INVALID"
    if regime.get("primary") != _first_known(
        regime_by_timeframe.get("M15"),
        regime_by_timeframe.get("H1"),
        regime.get("dominant"),
    ):
        return "TECHNICAL_CONTEXT_SCHEMA_INVALID"

    volatility = value.get("volatility")
    if not isinstance(volatility, Mapping):
        return "TECHNICAL_CONTEXT_SCHEMA_INVALID"
    volatility_fields = frozenset(volatility)
    if volatility_fields not in (
        frozenset(LEGACY_VOLATILITY_FIELDS),
        frozenset(VOLATILITY_FIELDS),
    ):
        return "TECHNICAL_CONTEXT_SCHEMA_INVALID"
    has_stored_confluence_atr = volatility_fields == frozenset(VOLATILITY_FIELDS)
    if not _optional_bounded_number(
        volatility.get("primary_atr_percentile"),
        lower=0.0,
        upper=100.0,
    ) or volatility.get("primary_atr_band") not in ATR_BANDS:
        return "TECHNICAL_CONTEXT_SCHEMA_INVALID"
    if has_stored_confluence_atr and not _optional_bounded_number(
        volatility.get("confluence_atr_percentile_24h"),
        lower=0.0,
        upper=100.0,
    ):
        return "TECHNICAL_CONTEXT_SCHEMA_INVALID"
    atr_percentiles = volatility.get("atr_percentile_by_timeframe")
    atr_bands = volatility.get("atr_band_by_timeframe")
    atr_pips = volatility.get("atr_pips_by_timeframe")
    if not all(
        _timeframe_mapping(item, timeframes=timeframes)
        for item in (atr_percentiles, atr_bands, atr_pips)
    ):
        return "TECHNICAL_CONTEXT_SCHEMA_INVALID"
    if any(
        not _optional_bounded_number(
            atr_percentiles.get(timeframe),
            lower=0.0,
            upper=100.0,
        )
        or atr_bands.get(timeframe) not in ATR_BANDS
        or not _optional_nonnegative_number(atr_pips.get(timeframe))
        for timeframe in timeframes
    ):
        return "TECHNICAL_CONTEXT_SCHEMA_INVALID"
    if any(
        atr_bands.get(timeframe)
        != _atr_band(_number(atr_percentiles.get(timeframe)))
        for timeframe in timeframes
    ):
        return "TECHNICAL_CONTEXT_SCHEMA_INVALID"
    expected_primary_atr_percentile = _first_number(
        atr_percentiles.get("M15"),
        atr_percentiles.get("M5"),
        (
            volatility.get("confluence_atr_percentile_24h")
            if has_stored_confluence_atr
            else None
        ),
    )
    if volatility.get("primary_atr_percentile") != expected_primary_atr_percentile:
        return "TECHNICAL_CONTEXT_SCHEMA_INVALID"
    if volatility.get("primary_atr_band") != _atr_band(
        _number(volatility.get("primary_atr_percentile"))
    ):
        return "TECHNICAL_CONTEXT_SCHEMA_INVALID"

    execution = value.get("execution")
    if not _mapping_has_exact_fields(execution, EXECUTION_FIELDS):
        return "TECHNICAL_CONTEXT_SCHEMA_INVALID"
    if (
        not _optional_nonnegative_number(execution.get("spread_pips"))
        or not _optional_nonnegative_number(execution.get("m5_atr_pips"))
        or not _optional_nonnegative_number(execution.get("spread_to_m5_atr"))
        or execution.get("spread_band") not in SPREAD_BANDS
    ):
        return "TECHNICAL_CONTEXT_SCHEMA_INVALID"
    if execution.get("m5_atr_pips") != atr_pips.get("M5"):
        return "TECHNICAL_CONTEXT_SCHEMA_INVALID"
    stored_spread = _number(execution.get("spread_pips"))
    stored_m5_atr = _number(execution.get("m5_atr_pips"))
    expected_spread_to_m5_atr = _round_or_none(
        _number(stored_spread / stored_m5_atr)
        if (
            stored_spread is not None
            and stored_m5_atr is not None
            and stored_m5_atr > 0.0
        )
        else None,
        6,
    )
    if execution.get("spread_to_m5_atr") != expected_spread_to_m5_atr:
        return "TECHNICAL_CONTEXT_SCHEMA_INVALID"
    if execution.get("spread_band") != _spread_band(
        expected_spread_to_m5_atr
    ):
        return "TECHNICAL_CONTEXT_SCHEMA_INVALID"

    location = value.get("location")
    if not _mapping_has_exact_fields(location, LOCATION_FIELDS):
        return "TECHNICAL_CONTEXT_SCHEMA_INVALID"
    if (
        not _optional_positive_number(location.get("current_price"))
        or not _optional_bounded_number(
            location.get("price_percentile_24h"), lower=0.0, upper=1.0
        )
        or not _optional_bounded_number(
            location.get("price_percentile_7d"), lower=0.0, upper=1.0
        )
        or location.get("range_location_24h") not in RANGE_LOCATIONS
        or location.get("range_location_7d") not in RANGE_LOCATIONS
    ):
        return "TECHNICAL_CONTEXT_SCHEMA_INVALID"
    if location.get("range_location_24h") != _range_location(
        _number(location.get("price_percentile_24h"))
    ) or location.get("range_location_7d") != _range_location(
        _number(location.get("price_percentile_7d"))
    ):
        return "TECHNICAL_CONTEXT_SCHEMA_INVALID"

    structure = value.get("structure")
    if not _mapping_has_exact_fields(structure, STRUCTURE_FIELDS):
        return "TECHNICAL_CONTEXT_SCHEMA_INVALID"
    primary_timeframe = structure.get("primary_timeframe")
    if primary_timeframe is not None and primary_timeframe not in timeframes:
        return "TECHNICAL_CONTEXT_SCHEMA_INVALID"
    if (
        not _optional_structure_kind(structure.get("primary_kind"))
        or structure.get("primary_direction") not in STRUCTURE_DIRECTIONS
    ):
        return "TECHNICAL_CONTEXT_SCHEMA_INVALID"
    structure_by_timeframe = structure.get("by_timeframe")
    if not _timeframe_mapping(structure_by_timeframe, timeframes=timeframes):
        return "TECHNICAL_CONTEXT_SCHEMA_INVALID"
    for timeframe in timeframes:
        item = structure_by_timeframe.get(timeframe)
        if not _mapping_has_exact_fields(item, STRUCTURE_ITEM_FIELDS):
            return "TECHNICAL_CONTEXT_SCHEMA_INVALID"
        direction = item.get("direction")
        if (
            direction not in STRUCTURE_DIRECTIONS
            or not _optional_structure_kind(item.get("kind"))
            or not _optional_finite_number(item.get("index"))
            or not _optional_text(item.get("timestamp_utc"))
        ):
            return "TECHNICAL_CONTEXT_SCHEMA_INVALID"
        if direction == "UNKNOWN":
            if any(
                item.get(field) is not None
                for field in ("kind", "index", "timestamp_utc")
            ):
                return "TECHNICAL_CONTEXT_SCHEMA_INVALID"
        else:
            kind = item.get("kind")
            if not isinstance(kind, str) or not kind.endswith(f"_{direction}"):
                return "TECHNICAL_CONTEXT_SCHEMA_INVALID"

    expected_primary = _primary_structure(structure_by_timeframe)
    if (
        primary_timeframe != expected_primary.get("timeframe")
        or structure.get("primary_kind") != expected_primary.get("kind")
        or structure.get("primary_direction") != expected_primary.get("direction")
    ):
        return "TECHNICAL_CONTEXT_SCHEMA_INVALID"

    families = value.get("families")
    if not _mapping_has_exact_fields(families, FAMILIES_FIELDS):
        return "TECHNICAL_CONTEXT_SCHEMA_INVALID"
    family_by_timeframe = families.get("by_timeframe")
    if not _timeframe_mapping(family_by_timeframe, timeframes=timeframes):
        return "TECHNICAL_CONTEXT_SCHEMA_INVALID"
    for timeframe in timeframes:
        scores = family_by_timeframe.get(timeframe)
        if not _mapping_has_exact_fields(scores, FAMILY_SCORE_FIELDS) or any(
            not _optional_family_score(scores.get(field))
            for field in FAMILY_SCORE_FIELDS
        ):
            return "TECHNICAL_CONTEXT_SCHEMA_INVALID"

    completeness = value.get("completeness")
    if not _mapping_has_exact_fields(completeness, COMPLETENESS_FIELDS):
        return "TECHNICAL_CONTEXT_COMPLETENESS_INVALID"
    required_fields = completeness.get("required_fields")
    missing_fields = completeness.get("missing_fields")
    if required_fields != list(REQUIRED_FIELDS) or not isinstance(missing_fields, list):
        return "TECHNICAL_CONTEXT_COMPLETENESS_INVALID"
    if any(not isinstance(item, str) for item in missing_fields):
        return "TECHNICAL_CONTEXT_COMPLETENESS_INVALID"
    expected_missing = _missing_required_fields(value)
    if (
        missing_fields != expected_missing
        or completeness.get("complete") is not (not expected_missing)
    ):
        return "TECHNICAL_CONTEXT_COMPLETENESS_INVALID"
    if "m5_failed_break_evidence" in value:
        failed_break_valid, failed_break_error = verify_m5_failed_break_evidence(
            value.get("m5_failed_break_evidence")
        )
        if not failed_break_valid:
            return failed_break_error or "M5_FAILED_BREAK_EVIDENCE_INVALID"
    if "dynamic_tf_policy_evidence" in value:
        policy_valid, policy_error = verify_dynamic_tf_policy_evidence(
            value.get("dynamic_tf_policy_evidence")
        )
        if not policy_valid:
            return policy_error or "DYNAMIC_TF_EVIDENCE_INVALID"
    if "dynamic_tf_policy_source_context" in value:
        policy_source = value.get("dynamic_tf_policy_source_context")
        if not _mapping_has_exact_fields(
            policy_source,
            DYNAMIC_TF_POLICY_SOURCE_CONTEXT_FIELDS,
        ):
            return "DYNAMIC_TF_POLICY_SOURCE_CONTEXT_INVALID"
        classifier_inputs = policy_source.get("classifier_inputs")
        news_source = policy_source.get("news_source")
        strategy_source = policy_source.get("strategy_profile_source")
        if not all(
            isinstance(item, Mapping)
            for item in (classifier_inputs, news_source, strategy_source)
        ):
            return "DYNAMIC_TF_POLICY_SOURCE_CONTEXT_INVALID"
        pair_chart_sha = policy_source.get("pair_chart_row_sha256")
        evaluated_at_utc = policy_source.get("evaluated_at_utc")
        if (
            not isinstance(pair_chart_sha, str)
            or len(pair_chart_sha) != 64
            or any(character not in "0123456789abcdef" for character in pair_chart_sha)
            or not isinstance(evaluated_at_utc, str)
            or not evaluated_at_utc.strip()
        ):
            return "DYNAMIC_TF_POLICY_SOURCE_CONTEXT_INVALID"
        try:
            classify_situation_from_classifier_inputs(dict(classifier_inputs))
        except (TypeError, ValueError, OverflowError):
            return "DYNAMIC_TF_POLICY_SOURCE_CONTEXT_INVALID"
        policy_evidence = value.get("dynamic_tf_policy_evidence")
        if not isinstance(policy_evidence, Mapping):
            return "DYNAMIC_TF_POLICY_SOURCE_CONTEXT_MISMATCH"
        policy_news = (
            policy_evidence.get("news_evidence")
            if isinstance(policy_evidence.get("news_evidence"), Mapping)
            else {}
        )
        policy_strategy = (
            policy_evidence.get("strategy_profile_evidence")
            if isinstance(policy_evidence.get("strategy_profile_evidence"), Mapping)
            else {}
        )
        if (
            dict(classifier_inputs)
            != dict(policy_evidence.get("classifier_inputs") or {})
        ) or dict(news_source) != dict(policy_news.get("source") or {}) or dict(
            strategy_source
        ) != dict(policy_strategy.get("source") or {}) or evaluated_at_utc != (
            policy_news.get("evaluated_at_utc")
        ):
            return "DYNAMIC_TF_POLICY_SOURCE_CONTEXT_MISMATCH"
    if "regime_family_weighting" in value:
        weighting_valid, weighting_error = verify_regime_family_weighting_receipt(
            value.get("regime_family_weighting"),
            pair=str(identity.get("pair") or ""),
        )
        if not weighting_valid:
            return weighting_error or "REGIME_FAMILY_WEIGHTING_INVALID"
        binding_valid, binding_error = verify_regime_family_weighting_context_binding(
            value.get("regime_family_weighting"),
            technical_context=value,
        )
        if not binding_valid:
            return binding_error or "REGIME_FAMILY_WEIGHTING_CONTEXT_BINDING_INVALID"
    return None


def _mapping_has_exact_fields(value: object, fields: set[str]) -> bool:
    return isinstance(value, Mapping) and set(value) == fields


def _stored_context_timeframes(
    value: Mapping[str, Any],
) -> tuple[str, ...] | None:
    regime = value.get("regime") if isinstance(value.get("regime"), Mapping) else {}
    stored = regime.get("by_timeframe") if isinstance(regime, Mapping) else None
    if not isinstance(stored, Mapping):
        return None
    keys = set(stored)
    if keys == set(LEGACY_CONTEXT_TIMEFRAMES):
        return LEGACY_CONTEXT_TIMEFRAMES
    if keys == set(CONTEXT_TIMEFRAMES):
        return CONTEXT_TIMEFRAMES
    return None


def _timeframe_mapping(
    value: object,
    *,
    timeframes: tuple[str, ...],
) -> bool:
    return isinstance(value, Mapping) and set(value) == set(timeframes)


def _canonical_nonempty_upper_text(value: object) -> bool:
    return isinstance(value, str) and bool(value) and value == value.strip().upper()


def _canonical_regime(value: object) -> bool:
    return (
        _canonical_nonempty_upper_text(value)
        and _normalized_regime(value) == value
    )


def _finite_number(value: object) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return math.isfinite(number) and abs(number) <= MAX_CONTEXT_NUMERIC_ABS


def _positive_number(value: object) -> bool:
    return _finite_number(value) and float(value) > 0.0


def _optional_positive_number(value: object) -> bool:
    return value is None or _positive_number(value)


def _optional_finite_number(value: object) -> bool:
    return value is None or _finite_number(value)


def _optional_family_score(value: object) -> bool:
    return value is None or (
        _finite_number(value) and abs(float(value)) <= MAX_FAMILY_SCORE_ABS
    )


def _optional_nonnegative_number(value: object) -> bool:
    return value is None or (_finite_number(value) and float(value) >= 0.0)


def _optional_bounded_number(
    value: object,
    *,
    lower: float,
    upper: float,
) -> bool:
    return value is None or (
        _finite_number(value) and lower <= float(value) <= upper
    )


def _optional_structure_kind(value: object) -> bool:
    return value is None or _canonical_nonempty_upper_text(value)


def _optional_text(value: object) -> bool:
    return value is None or (isinstance(value, str) and bool(value.strip()))


def technical_context_sha256(value: Mapping[str, Any]) -> str:
    body = {str(key): item for key, item in value.items() if key != "context_sha256"}
    encoded = json.dumps(
        body,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def forecast_pair_chart_row_sha256(
    pair_chart: Mapping[str, Any],
    *,
    pair: str,
) -> str:
    """Digest the exact canonical pair-chart row that seeded a forecast."""

    material = {
        "contract": PAIR_CHART_SOURCE_CONTRACT,
        "pair": str(pair or "").strip().upper(),
        "pair_chart": deepcopy(dict(pair_chart)),
    }
    # Source rows can be malformed before normalization (for example a NaN
    # ATR in a retryable truth-gap regression).  Preserve that exact malformed
    # source identity without putting the non-finite value into the context
    # body itself.  Python's stable NaN/Infinity tokens are used only inside
    # this SHA preimage; every transmitted context still uses strict JSON.
    encoded = json.dumps(
        material,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_forecast_technical_context_evidence(
    value: object,
    *,
    pair: str | None,
    current_price: float | None,
) -> dict[str, Any]:
    """Return a bounded, content-addressed GPT handoff for forecast context.

    The handoff is diagnostic context only.  A missing, incomplete, or invalid
    body is represented as ``UNKNOWN`` without copying unverified content, so
    downstream market-read or allocation code cannot mistake malformed context
    for additional live permission.  ``confidence`` remains a calibrated score;
    sizing continues to use direction-specific economic hit-rate Wilson bounds.
    """

    pair_name = str(pair or "").strip().upper()
    parsed_price = _number(current_price)
    valid, error = verify_forecast_technical_context(
        value,
        pair=pair_name or None,
        current_price=parsed_price,
    )
    if not valid:
        return unknown_forecast_technical_context_evidence(
            error or "TECHNICAL_CONTEXT_INVALID"
        )
    if not pair_name:
        return unknown_forecast_technical_context_evidence(
            "TECHNICAL_CONTEXT_PAIR_MISSING"
        )
    if parsed_price is None:
        return unknown_forecast_technical_context_evidence(
            "TECHNICAL_CONTEXT_PRICE_MISSING"
        )
    context = deepcopy(dict(value)) if isinstance(value, Mapping) else {}
    completeness = (
        context.get("completeness")
        if isinstance(context.get("completeness"), Mapping)
        else {}
    )
    if completeness.get("complete") is not True:
        return unknown_forecast_technical_context_evidence(
            "TECHNICAL_CONTEXT_INCOMPLETE"
        )
    context_sha256 = str(context.get("context_sha256") or "").strip().lower()
    return _technical_context_evidence_with_sha(
        {
            "contract": TECHNICAL_CONTEXT_EVIDENCE_CONTRACT,
            "status": "VALID",
            "reason": None,
            "confidence_semantics": CONFIDENCE_SEMANTICS,
            "technical_context_v1": context,
            "context_sha256": context_sha256,
            "read_only": True,
            "proof_eligible": False,
            "live_permission": False,
        }
    )


def verify_forecast_technical_context_evidence(
    value: object,
    *,
    pair: str | None,
    current_price: float | None,
) -> tuple[bool, str | None]:
    """Verify the context handoff and its nested point-in-time body."""

    if not isinstance(value, Mapping):
        return False, "TECHNICAL_CONTEXT_EVIDENCE_MISSING"
    if set(value) != TECHNICAL_CONTEXT_EVIDENCE_FIELDS:
        return False, "TECHNICAL_CONTEXT_EVIDENCE_SCHEMA_INVALID"
    try:
        encoded_size = len(_canonical_json_bytes(value))
    except (TypeError, ValueError, OverflowError):
        return False, "TECHNICAL_CONTEXT_EVIDENCE_HASH_MISMATCH"
    if encoded_size > MAX_EVIDENCE_BYTES:
        return False, "TECHNICAL_CONTEXT_EVIDENCE_TOO_LARGE"
    if value.get("contract") != TECHNICAL_CONTEXT_EVIDENCE_CONTRACT:
        return False, "TECHNICAL_CONTEXT_EVIDENCE_SCHEMA_INVALID"
    if value.get("confidence_semantics") != CONFIDENCE_SEMANTICS:
        return False, "TECHNICAL_CONTEXT_EVIDENCE_CONFIDENCE_SEMANTICS_INVALID"
    if (
        value.get("read_only") is not True
        or value.get("proof_eligible") is not False
        or value.get("live_permission") is not False
    ):
        return False, "TECHNICAL_CONTEXT_EVIDENCE_PERMISSION_INVALID"
    stored_evidence_sha = str(value.get("evidence_sha256") or "").strip().lower()
    try:
        expected_evidence_sha = _canonical_json_sha256(
            {str(key): item for key, item in value.items() if key != "evidence_sha256"}
        )
    except (TypeError, ValueError):
        return False, "TECHNICAL_CONTEXT_EVIDENCE_HASH_MISMATCH"
    if len(stored_evidence_sha) != 64 or stored_evidence_sha != expected_evidence_sha:
        return False, "TECHNICAL_CONTEXT_EVIDENCE_HASH_MISMATCH"

    status = value.get("status")
    if status == "UNKNOWN":
        reason = value.get("reason")
        if (
            not isinstance(reason, str)
            or not reason.strip()
            or reason != reason.strip()
            or len(reason) > MAX_REASON_CHARS
            or value.get("technical_context_v1") is not None
            or value.get("context_sha256") is not None
        ):
            return False, "TECHNICAL_CONTEXT_EVIDENCE_UNKNOWN_INVALID"
        return True, None
    if status != "VALID" or value.get("reason") is not None:
        return False, "TECHNICAL_CONTEXT_EVIDENCE_STATUS_INVALID"

    pair_name = str(pair or "").strip().upper()
    parsed_price = _number(current_price)
    if not pair_name:
        return False, "TECHNICAL_CONTEXT_EVIDENCE_PAIR_MISSING"
    if parsed_price is None:
        return False, "TECHNICAL_CONTEXT_EVIDENCE_PRICE_MISSING"
    context = value.get("technical_context_v1")
    context_sha = str(value.get("context_sha256") or "").strip().lower()
    if not isinstance(context, Mapping) or context_sha != str(
        context.get("context_sha256") or ""
    ).strip().lower():
        return False, "TECHNICAL_CONTEXT_EVIDENCE_CONTEXT_SHA_MISMATCH"
    valid, error = verify_forecast_technical_context(
        context,
        pair=pair_name,
        current_price=parsed_price,
    )
    if not valid:
        return False, error or "TECHNICAL_CONTEXT_INVALID"
    completeness = (
        context.get("completeness")
        if isinstance(context.get("completeness"), Mapping)
        else {}
    )
    if completeness.get("complete") is not True:
        return False, "TECHNICAL_CONTEXT_INCOMPLETE"
    return True, None


def normalize_forecast_technical_context_evidence(
    value: object,
    *,
    pair: str | None,
    current_price: float | None,
) -> dict[str, Any]:
    """Copy a valid handoff or replace tampered input with bounded UNKNOWN."""

    valid, error = verify_forecast_technical_context_evidence(
        value,
        pair=pair,
        current_price=current_price,
    )
    if not valid:
        return unknown_forecast_technical_context_evidence(
            error or "TECHNICAL_CONTEXT_EVIDENCE_INVALID"
        )
    return deepcopy(dict(value))


def unknown_forecast_technical_context_evidence(reason: str) -> dict[str, Any]:
    """Create a content-addressed UNKNOWN handoff without unverified body data."""

    normalized_reason = (
        reason.strip()
        if isinstance(reason, str) and reason.strip()
        else "TECHNICAL_CONTEXT_UNKNOWN"
    )
    normalized_reason = normalized_reason[:MAX_REASON_CHARS]
    return _technical_context_evidence_with_sha(
        {
            "contract": TECHNICAL_CONTEXT_EVIDENCE_CONTRACT,
            "status": "UNKNOWN",
            "reason": normalized_reason,
            "confidence_semantics": CONFIDENCE_SEMANTICS,
            "technical_context_v1": None,
            "context_sha256": None,
            "read_only": True,
            "proof_eligible": False,
            "live_permission": False,
        }
    )


def _technical_context_evidence_with_sha(material: Mapping[str, Any]) -> dict[str, Any]:
    body = deepcopy(dict(material))
    body["evidence_sha256"] = _canonical_json_sha256(body)
    return body


def _canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _views_by_timeframe(chart: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    out: dict[str, Mapping[str, Any]] = {}
    for raw in chart.get("views") or []:
        if not isinstance(raw, Mapping):
            continue
        timeframe = str(raw.get("granularity") or "").upper()
        if timeframe in CONTEXT_TIMEFRAMES and timeframe not in out:
            out[timeframe] = raw
    return out


def _latest_close_confirmed_structure(view: Mapping[str, Any]) -> dict[str, Any]:
    structure = view.get("structure") if isinstance(view.get("structure"), Mapping) else {}
    latest: tuple[float, dict[str, Any]] | None = None
    for order, raw in enumerate((structure or {}).get("structure_events") or []):
        if not isinstance(raw, Mapping) or not bool(raw.get("close_confirmed")):
            continue
        kind = str(raw.get("kind") or "").upper().split(":", 1)[0]
        direction = "UP" if kind.endswith("_UP") else "DOWN" if kind.endswith("_DOWN") else None
        if direction is None:
            continue
        index = _number(raw.get("index"))
        rank = index if index is not None else float(order)
        item = {
            "kind": kind,
            "direction": direction,
            "index": _round_or_none(index, 6),
            "timestamp_utc": str(raw.get("timestamp") or "").strip() or None,
        }
        if latest is None or rank >= latest[0]:
            latest = (rank, item)
    return latest[1] if latest is not None else {
        "kind": None,
        "direction": "UNKNOWN",
        "index": None,
        "timestamp_utc": None,
    }


def _primary_structure(by_timeframe: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    for timeframe in ("H1", "M15", "M5", "H4"):
        item = by_timeframe.get(timeframe) or {}
        if item.get("direction") in {"UP", "DOWN"}:
            return {"timeframe": timeframe, **dict(item)}
    return {"timeframe": None, "kind": None, "direction": "UNKNOWN"}


def _normalized_regime(value: object) -> str:
    text = str(value or "").upper().strip()
    if not text:
        return "UNKNOWN"
    if "BREAKOUT_PENDING" in text:
        return "BREAKOUT_PENDING"
    if "RANGE" in text:
        return "RANGE"
    if "TREND_STRONG" in text:
        return "TREND_STRONG"
    if "TREND_WEAK" in text:
        return "TREND_WEAK"
    if "TREND_UP" in text or text == "UP":
        return "TREND_UP"
    if "TREND_DOWN" in text or text == "DOWN":
        return "TREND_DOWN"
    if "TRANSITION" in text:
        return "TRANSITION"
    return text[:32]


def _atr_band(value: float | None) -> str:
    if value is None:
        return "UNKNOWN"
    if value <= 25.0:
        return "LOW"
    if value >= 75.0:
        return "HIGH"
    return "NORMAL"


def _spread_band(spread_to_atr: float | None) -> str:
    if spread_to_atr is None:
        return "UNKNOWN"
    if spread_to_atr <= 0.25:
        return "TIGHT"
    if spread_to_atr <= 0.75:
        return "NORMAL"
    return "WIDE"


def _range_location(value: float | None) -> str:
    if value is None:
        return "UNKNOWN"
    if value <= 1.0 / 3.0:
        return "LOWER"
    if value >= 2.0 / 3.0:
        return "UPPER"
    return "MIDDLE"


def _missing_required_fields(body: Mapping[str, Any]) -> list[str]:
    values = {
        "identity.pair": ((body.get("identity") or {}).get("pair")),
        "regime.dominant": ((body.get("regime") or {}).get("dominant")),
        "volatility.primary_atr_band": ((body.get("volatility") or {}).get("primary_atr_band")),
        "execution.spread_band": ((body.get("execution") or {}).get("spread_band")),
        "location.range_location_24h": ((body.get("location") or {}).get("range_location_24h")),
        "structure.primary_direction": ((body.get("structure") or {}).get("primary_direction")),
    }
    return [
        name
        for name in REQUIRED_FIELDS
        if values.get(name) in {None, "", "UNKNOWN"}
    ]


def _number(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(number) or abs(number) > MAX_CONTEXT_NUMERIC_ABS:
        return None
    return number


def _family_score_number(value: object) -> float | None:
    number = _number(value)
    if number is None or abs(number) > MAX_FAMILY_SCORE_ABS:
        return None
    return number


def _bounded_percent_0_100(value: object) -> float | None:
    """Validate a value whose producer contract is already 0..100."""

    number = _number(value)
    if number is None or not 0.0 <= number <= 100.0:
        return None
    return number


def _fraction_percentile_to_100(value: object) -> float | None:
    """Convert IndicatorSet quantile ranks (0..1) to percentage points."""

    number = _number(value)
    if number is None or not 0.0 <= number <= 1.0:
        return None
    return number * 100.0


def _fraction_0_1(value: object) -> float | None:
    number = _number(value)
    if number is None:
        return None
    if number > 1.0 and number <= 100.0:
        number /= 100.0
    if not 0.0 <= number <= 1.0:
        return None
    return number


def _round_or_none(value: float | None, digits: int) -> float | None:
    return round(value, digits) if value is not None else None


def _first_number(*values: object) -> float | None:
    for value in values:
        number = _number(value)
        if number is not None:
            return number
    return None


def _first_known(*values: object) -> str:
    for value in values:
        text = str(value or "").strip().upper()
        if text and text != "UNKNOWN":
            return text
    return "UNKNOWN"
