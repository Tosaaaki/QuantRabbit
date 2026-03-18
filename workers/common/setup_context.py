from __future__ import annotations

from typing import Any, Mapping, Optional

_KNOWN_FLOW_REGIMES = {
    "range_compression",
    "range_fade",
    "transition",
    "trend_long",
    "trend_short",
}


def _to_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def _looks_like_common_microstructure_bucket(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    if text == "unknown":
        return True
    return text.startswith(("tight_", "normal_", "wide_"))


def _parse_common_setup_fingerprint(value: Any) -> dict[str, str]:
    text = str(value or "").strip()
    if not text:
        return {}
    parts = [part.strip() for part in text.split("|")]
    if len(parts) < 4:
        return {}
    flow_regime = parts[2]
    microstructure_bucket = parts[3]
    if flow_regime not in _KNOWN_FLOW_REGIMES:
        return {}
    if not _looks_like_common_microstructure_bucket(microstructure_bucket):
        return {}
    return {
        "setup_fingerprint": text,
        "flow_regime": flow_regime,
        "microstructure_bucket": microstructure_bucket,
    }


def _bucket_label(
    value: Optional[float],
    *,
    cuts: tuple[tuple[float, str], ...],
    default: str,
) -> str:
    if value is None:
        return default
    for threshold, label in cuts:
        if value < threshold:
            return label
    return cuts[-1][1] if cuts else default


def _technical_indicator_frame(
    entry_thesis: Mapping[str, Any], tf: str = "M1"
) -> dict[str, object]:
    technical_context = entry_thesis.get("technical_context")
    if not isinstance(technical_context, Mapping):
        return {}
    indicators = technical_context.get("indicators")
    if not isinstance(indicators, Mapping):
        return {}
    frame = indicators.get(tf)
    return dict(frame) if isinstance(frame, Mapping) else {}


def _frame_ma_pair(frame: Mapping[str, Any]) -> tuple[Optional[float], Optional[float]]:
    ma_fast = _to_float(frame.get("ma10"))
    if ma_fast is None:
        ma_fast = _to_float(frame.get("ema12"))
    if ma_fast is None:
        ma_fast = _to_float(frame.get("ema20"))
    ma_slow = _to_float(frame.get("ma20"))
    if ma_slow is None:
        ma_slow = _to_float(frame.get("ema20"))
    if ma_slow is None:
        ma_slow = _to_float(frame.get("ema24"))
    return ma_fast, ma_slow


def _frame_flow_snapshot(frame: Mapping[str, Any]) -> dict[str, object]:
    if not frame:
        return {}
    atr_pips = _to_float(frame.get("atr_pips"))
    adx = _to_float(frame.get("adx"))
    plus_di = _to_float(frame.get("plus_di"))
    minus_di = _to_float(frame.get("minus_di"))
    di_gap = (
        plus_di - minus_di if plus_di is not None and minus_di is not None else None
    )
    ma_fast, ma_slow = _frame_ma_pair(frame)
    ma_gap_pips = None
    if ma_fast is not None and ma_slow is not None:
        ma_gap_pips = (ma_fast - ma_slow) / 0.01
    gap_ratio = (
        abs(ma_gap_pips) / max(atr_pips or 0.0, 1.0)
        if ma_gap_pips is not None
        else None
    )

    trend_dir = "neutral"
    if di_gap is not None:
        if di_gap > 0.0:
            trend_dir = "long"
        elif di_gap < 0.0:
            trend_dir = "short"
    if trend_dir == "neutral" and ma_gap_pips is not None and abs(ma_gap_pips) >= 0.05:
        trend_dir = "long" if ma_gap_pips > 0.0 else "short"

    flow_regime = "transition"
    strong_gap = gap_ratio is not None and gap_ratio >= 0.55
    medium_gap = gap_ratio is not None and gap_ratio >= 0.35
    strong_di = di_gap is not None and abs(di_gap) >= 6.0
    if trend_dir != "neutral" and (
        (strong_gap and (adx is None or adx >= 17.0))
        or (medium_gap and strong_di and (adx is None or adx >= 14.0))
    ):
        flow_regime = f"trend_{trend_dir}"

    strength = 0.0
    if gap_ratio is not None:
        strength += min(gap_ratio, 1.5)
    if adx is not None:
        strength += max(0.0, min((adx - 14.0) / 16.0, 1.5))
    if di_gap is not None:
        strength += min(abs(di_gap) / 12.0, 1.5)

    return {
        "flow_regime": flow_regime,
        "trend_dir": trend_dir,
        "trend_strength": round(strength, 4),
    }


def _derive_mtf_context(
    entry_thesis: Mapping[str, Any],
    *,
    side_label: str,
) -> dict[str, object]:
    snapshots: dict[str, object] = {}
    trend_votes: list[str] = []
    available = 0
    for tf in ("H1", "H4", "D1"):
        frame = _technical_indicator_frame(entry_thesis, tf)
        snapshot = _frame_flow_snapshot(frame)
        if not snapshot:
            continue
        available += 1
        flow_regime = str(snapshot.get("flow_regime") or "transition")
        trend_strength = _to_float(snapshot.get("trend_strength"))
        snapshots[f"{tf.lower()}_flow_regime"] = flow_regime
        if trend_strength is not None:
            snapshots[f"{tf.lower()}_trend_strength"] = round(trend_strength, 4)
        if flow_regime in {"trend_long", "trend_short"}:
            trend_votes.append(flow_regime.rsplit("_", 1)[-1])
    if available == 0:
        return {}

    vote_long = sum(1 for value in trend_votes if value == "long")
    vote_short = sum(1 for value in trend_votes if value == "short")
    if vote_long > 0 and vote_short > 0:
        macro_flow_regime = "transition"
        mtf_alignment = "mixed"
    elif vote_long > 0:
        macro_flow_regime = "trend_long"
        mtf_alignment = (
            "aligned"
            if side_label == "long"
            else "countertrend" if side_label == "short" else "unknown"
        )
    elif vote_short > 0:
        macro_flow_regime = "trend_short"
        mtf_alignment = (
            "aligned"
            if side_label == "short"
            else "countertrend" if side_label == "long" else "unknown"
        )
    else:
        macro_flow_regime = "transition"
        mtf_alignment = "neutral"

    snapshots["macro_flow_regime"] = macro_flow_regime
    snapshots["mtf_alignment"] = mtf_alignment
    return snapshots


def _resolve_units(units: int, entry_thesis: Mapping[str, Any]) -> float:
    if units != 0:
        return float(units)
    for key in ("entry_units_intent", "units", "raw_units"):
        value = _to_float(entry_thesis.get(key))
        if value is not None and abs(value) > 0.0:
            return value
    return 0.0


def _entry_side_label(units: int, entry_thesis: Mapping[str, Any]) -> str:
    resolved_units = _resolve_units(units, entry_thesis)
    if resolved_units > 0.0:
        return "long"
    if resolved_units < 0.0:
        return "short"
    for key in ("side", "entry_side"):
        value = str(entry_thesis.get(key) or "").strip().lower()
        if value in {"long", "short"}:
            return value
    return "flat"


def derive_live_setup_context(
    entry_thesis: Mapping[str, Any] | None,
    *,
    units: int = 0,
) -> Optional[dict[str, object]]:
    if not isinstance(entry_thesis, Mapping):
        return None
    frame_m1 = _technical_indicator_frame(entry_thesis, "M1")
    technical_context = entry_thesis.get("technical_context")
    ticks_raw = (
        technical_context.get("ticks") if isinstance(technical_context, Mapping) else {}
    )
    ticks = ticks_raw if isinstance(ticks_raw, Mapping) else {}

    atr_pips = _to_float(frame_m1.get("atr_pips"))
    if atr_pips is None:
        atr_pips = _to_float(entry_thesis.get("atr_entry"))
    rsi = _to_float(frame_m1.get("rsi"))
    adx = _to_float(frame_m1.get("adx"))
    plus_di = _to_float(frame_m1.get("plus_di"))
    minus_di = _to_float(frame_m1.get("minus_di"))
    ma_fast = _to_float(frame_m1.get("ma10"))
    ma_slow = _to_float(frame_m1.get("ma20"))
    if ma_fast is None:
        ma_fast = _to_float(frame_m1.get("ema20"))
    if ma_slow is None:
        ma_slow = _to_float(frame_m1.get("ema24"))
    ma_gap_pips = None
    if ma_fast is not None and ma_slow is not None:
        ma_gap_pips = (ma_fast - ma_slow) / 0.01
    gap_ratio = (
        abs(ma_gap_pips) / max(atr_pips or 0.0, 1.0)
        if ma_gap_pips is not None
        else None
    )

    range_mode = str(entry_thesis.get("range_mode") or "").strip().lower()
    range_score = _to_float(entry_thesis.get("range_score"))
    if (
        range_mode == "range"
        and range_score is not None
        and range_score >= 0.45
        and (adx is None or adx < 24.0)
    ):
        flow_regime = "range_compression"
    else:
        trend_dir = "neutral"
        if plus_di is not None and minus_di is not None:
            if plus_di > minus_di:
                trend_dir = "long"
            elif minus_di > plus_di:
                trend_dir = "short"
        if (
            gap_ratio is not None
            and gap_ratio >= 0.85
            and (adx or 0.0) >= 22.0
            and trend_dir != "neutral"
        ):
            flow_regime = f"trend_{trend_dir}"
        elif (
            range_score is not None
            and range_score >= 0.28
            and (adx is None or adx < 32.0)
        ):
            flow_regime = "range_fade"
        else:
            flow_regime = "transition"

    spread_pips = _to_float(ticks.get("spread_pips"))
    if spread_pips is None:
        spread_pips = _to_float(entry_thesis.get("spread_pips"))
    tick_rate = _to_float(ticks.get("tick_rate"))
    if spread_pips is None:
        microstructure_bucket = "unknown"
    else:
        spread_ratio = spread_pips / max(atr_pips or 0.0, 1.0)
        if spread_pips <= 0.9 and spread_ratio <= 0.35:
            spread_bucket = "tight"
        elif spread_pips <= 1.4 and spread_ratio <= 0.55:
            spread_bucket = "normal"
        else:
            spread_bucket = "wide"
        pace_bucket = "normal"
        if tick_rate is not None:
            if tick_rate < 2.0:
                pace_bucket = "thin"
            elif tick_rate >= 8.0:
                pace_bucket = "fast"
        microstructure_bucket = f"{spread_bucket}_{pace_bucket}"

    rsi_bucket = _bucket_label(
        rsi,
        cuts=(
            (35.0, "ext_oversold"),
            (45.0, "oversold"),
            (55.0, "mid"),
            (65.0, "overbought"),
            (1e9, "ext_overbought"),
        ),
        default="unknown",
    )
    atr_bucket = _bucket_label(
        atr_pips,
        cuts=(
            (1.2, "ultra_low"),
            (2.6, "low"),
            (5.5, "mid"),
            (9.0, "high"),
            (1e9, "extreme"),
        ),
        default="unknown",
    )
    if ma_gap_pips is None or gap_ratio is None:
        gap_bucket = "unknown"
    else:
        gap_side = "up" if ma_gap_pips >= 0.0 else "down"
        gap_mag = _bucket_label(
            gap_ratio,
            cuts=((0.35, "flat"), (0.75, "lean"), (1.20, "strong"), (1e9, "extended")),
            default="unknown",
        )
        gap_bucket = f"{gap_side}_{gap_mag}"

    side_label = _entry_side_label(units, entry_thesis)
    mtf_context = _derive_mtf_context(entry_thesis, side_label=side_label)
    macro_flow_regime = str(mtf_context.get("macro_flow_regime") or "").strip()
    mtf_alignment = str(mtf_context.get("mtf_alignment") or "").strip()
    setup_anchor = (
        str(entry_thesis.get("pattern_tag") or entry_thesis.get("range_reason") or "")
        .strip()
        .lower()
    )
    setup_parts = [
        str(
            entry_thesis.get("strategy_tag") or entry_thesis.get("strategy") or ""
        ).strip(),
        side_label,
        flow_regime,
        microstructure_bucket,
        f"rsi:{rsi_bucket}",
        f"atr:{atr_bucket}",
        f"gap:{gap_bucket}",
    ]
    if setup_anchor:
        setup_parts.append(setup_anchor)
    if (
        macro_flow_regime in {"trend_long", "trend_short"}
        and macro_flow_regime != flow_regime
    ):
        setup_parts.append(f"macro:{macro_flow_regime}")
    if mtf_alignment in {"countertrend", "mixed"}:
        setup_parts.append(f"align:{mtf_alignment}")
    summary: dict[str, object] = {}
    live_setup = entry_thesis.get("live_setup_context")
    if isinstance(live_setup, Mapping):
        summary.update(dict(live_setup))
    summary.update(mtf_context)
    summary.update(
        {
            "flow_regime": flow_regime,
            "microstructure_bucket": microstructure_bucket,
            "setup_fingerprint": "|".join(part for part in setup_parts if part),
            "side": side_label,
            "range_mode": range_mode or None,
            "range_score": round(range_score, 4) if range_score is not None else None,
            "atr_pips": round(atr_pips, 4) if atr_pips is not None else None,
            "rsi": round(rsi, 4) if rsi is not None else None,
            "adx": round(adx, 4) if adx is not None else None,
            "spread_pips": round(spread_pips, 4) if spread_pips is not None else None,
            "tick_rate": round(tick_rate, 4) if tick_rate is not None else None,
            "ma_gap_pips": round(ma_gap_pips, 4) if ma_gap_pips is not None else None,
            "gap_ratio": round(gap_ratio, 4) if gap_ratio is not None else None,
            "rsi_bucket": rsi_bucket,
            "atr_bucket": atr_bucket,
            "gap_bucket": gap_bucket,
        }
    )
    explicit_fingerprint = entry_thesis.get("setup_fingerprint")
    if explicit_fingerprint in {None, ""} and isinstance(live_setup, Mapping):
        explicit_fingerprint = live_setup.get("setup_fingerprint")
    parsed_fingerprint = _parse_common_setup_fingerprint(explicit_fingerprint)
    if parsed_fingerprint:
        summary.update(parsed_fingerprint)
    else:
        for key in ("flow_regime", "microstructure_bucket", "setup_fingerprint"):
            explicit = entry_thesis.get(key)
            if explicit in {None, ""} and isinstance(live_setup, Mapping):
                explicit = live_setup.get(key)
            text = str(explicit or "").strip()
            if text:
                summary[key] = text
    return summary


def extract_setup_identity(
    entry_thesis: Mapping[str, Any] | None,
    *,
    units: int = 0,
) -> dict[str, str]:
    if not isinstance(entry_thesis, Mapping):
        return {}
    live_setup = entry_thesis.get("live_setup_context")
    if not isinstance(live_setup, Mapping):
        live_setup = {}
    context: dict[str, str] = {}
    explicit_fingerprint = entry_thesis.get("setup_fingerprint")
    if explicit_fingerprint in {None, ""}:
        explicit_fingerprint = live_setup.get("setup_fingerprint")
    parsed_fingerprint = _parse_common_setup_fingerprint(explicit_fingerprint)
    if parsed_fingerprint:
        context.update(parsed_fingerprint)
    for key in ("setup_fingerprint", "flow_regime", "microstructure_bucket"):
        raw = entry_thesis.get(key)
        if raw in {None, ""}:
            raw = live_setup.get(key)
        text = str(raw or "").strip()
        if text:
            if parsed_fingerprint and key in {"flow_regime", "microstructure_bucket"}:
                continue
            context[key] = text
    if len(context) == 3:
        return context
    derived = derive_live_setup_context(entry_thesis, units=units)
    if isinstance(derived, dict):
        for key in ("setup_fingerprint", "flow_regime", "microstructure_bucket"):
            text = str(derived.get(key) or "").strip()
            if text:
                context.setdefault(key, text)
    return context
