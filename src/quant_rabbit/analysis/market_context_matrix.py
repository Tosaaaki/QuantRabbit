"""Build a pair/side market-context matrix from existing technical artifacts.

The matrix is advisory evidence, not an execution gate. It is designed to
raise decision quality without reducing trade count: every observation is
reported as support/reject/warning/missing for the trader and verifier packet,
but this module never changes lane status, units, risk budget, or live
permission.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SIDES = ("LONG", "SHORT")
# Compact packet limit for advisory evidence refs in intent metadata. This is
# a payload-size guard, not a market threshold: the full matrix stays in
# `data/market_context_matrix.json`, while each intent carries the refs needed
# to attribute later P/L to gold/oil/cross-asset/news context.
SUMMARY_LIST_LIMIT = 8


def build_market_context_matrix_from_payloads(
    *,
    pair_charts: dict[str, Any] | None = None,
    context_asset_charts: dict[str, Any] | None = None,
    cross_asset: dict[str, Any] | None = None,
    flow: dict[str, Any] | None = None,
    currency_strength: dict[str, Any] | None = None,
    levels: dict[str, Any] | None = None,
    calendar: dict[str, Any] | None = None,
    cot: dict[str, Any] | None = None,
    option_skew: dict[str, Any] | None = None,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    charts = _by_key((pair_charts or {}).get("charts"), "pair")
    context_charts = _by_key((context_asset_charts or {}).get("charts"), "pair")
    flow_spreads = _by_key((flow or {}).get("spreads"), "instrument")
    flow_order_books = _by_key((flow or {}).get("order_books"), "instrument")
    flow_position_books = _by_key((flow or {}).get("position_books"), "instrument")
    level_pairs = _by_key((levels or {}).get("pairs"), "pair")
    calendar_windows = _by_key((calendar or {}).get("pair_windows"), "pair")
    strength_scores = _by_key((currency_strength or {}).get("scores"), "currency")
    cot_reports = _by_key((cot or {}).get("reports"), "currency")

    pairs = sorted(
        {
            str(pair)
            for pair in (
                set(charts)
                | set(flow_spreads)
                | set(level_pairs)
                | set(calendar_windows)
                | _option_pairs(option_skew)
            )
            if str(pair).count("_") == 1
        }
    )
    missing_artifacts = [
        name
        for name, payload in (
            ("pair_charts", pair_charts),
            ("context_asset_charts", context_asset_charts),
            ("cross_asset", cross_asset),
            ("flow", flow),
            ("currency_strength", currency_strength),
            ("levels", levels),
            ("calendar", calendar),
            ("cot", cot),
        )
        if not isinstance(payload, dict)
    ]

    matrix_pairs: dict[str, Any] = {}
    for pair in pairs:
        pair_matrix = {side: _empty_side(pair, side) for side in SIDES}
        base, quote = pair.split("_", 1)
        _apply_chart_layer(pair_matrix, pair, charts.get(pair))
        _apply_strength_layer(pair_matrix, pair, base, quote, strength_scores)
        _apply_cross_asset_layer(pair_matrix, pair, base, quote, cross_asset)
        _apply_context_asset_chart_layer(pair_matrix, pair, base, quote, context_charts)
        _apply_flow_layer(
            pair_matrix,
            pair,
            flow_spreads.get(pair),
            flow_order_books.get(pair),
            flow_position_books.get(pair),
        )
        _apply_levels_layer(pair_matrix, pair, level_pairs.get(pair))
        _apply_calendar_layer(pair_matrix, pair, calendar_windows.get(pair))
        _apply_cot_layer(pair_matrix, pair, base, quote, cot_reports)
        _apply_option_skew_layer(pair_matrix, pair, option_skew)
        for side_payload in pair_matrix.values():
            _finalize_side(side_payload)
        matrix_pairs[pair] = pair_matrix

    issues = [f"MISSING_{name.upper()}_ARTIFACT" for name in missing_artifacts]
    for payload in (cross_asset, flow, currency_strength, levels, calendar, cot, option_skew):
        if isinstance(payload, dict):
            issues.extend(str(item) for item in payload.get("issues", []) or [] if str(item).strip())
    return {
        "generated_at_utc": generated_at_utc or datetime.now(timezone.utc).isoformat(),
        "trade_count_policy": "ADVISORY_ONLY_DOES_NOT_BLOCK_OR_DEMOTE_LANES",
        "schema_version": 1,
        "pairs": matrix_pairs,
        "issues": issues[:80],
    }


def build_market_context_matrix(
    *,
    pair_charts_path: Path,
    cross_asset_path: Path,
    flow_path: Path,
    currency_strength_path: Path,
    levels_path: Path,
    calendar_path: Path,
    cot_path: Path,
    option_skew_path: Path,
    context_asset_charts_path: Path | None = None,
) -> dict[str, Any]:
    return build_market_context_matrix_from_payloads(
        pair_charts=_load_optional_json(pair_charts_path),
        context_asset_charts=_load_optional_json(context_asset_charts_path) if context_asset_charts_path else None,
        cross_asset=_load_optional_json(cross_asset_path),
        flow=_load_optional_json(flow_path),
        currency_strength=_load_optional_json(currency_strength_path),
        levels=_load_optional_json(levels_path),
        calendar=_load_optional_json(calendar_path),
        cot=_load_optional_json(cot_path),
        option_skew=_load_optional_json(option_skew_path),
    )


def write_market_context_matrix_report(payload: dict[str, Any], report_path: Path) -> None:
    lines = [
        "# Market Context Matrix",
        "",
        f"- Generated at UTC: `{payload.get('generated_at_utc')}`",
        f"- Trade count policy: `{payload.get('trade_count_policy')}`",
        f"- Pairs: `{len(payload.get('pairs') or {})}`",
        "",
        "## Pair Summary",
        "",
        "| Pair | Side | Supports | Rejects | Warnings | Missing | Strongest reject |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for pair, side_map in sorted((payload.get("pairs") or {}).items()):
        if not isinstance(side_map, dict):
            continue
        for side in SIDES:
            reading = side_map.get(side) if isinstance(side_map.get(side), dict) else {}
            strongest = reading.get("strongest_reject") or ""
            lines.append(
                f"| `{pair}` | `{side}` | {reading.get('support_count', 0)} | "
                f"{reading.get('reject_count', 0)} | {reading.get('warning_count', 0)} | "
                f"{reading.get('missing_count', 0)} | {strongest} |"
            )
    issues = [str(item) for item in payload.get("issues", []) or [] if str(item).strip()]
    if issues:
        lines.extend(["", "## Issues", ""])
        lines.extend(f"- {issue}" for issue in issues[:40])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n")


def matrix_summary_for_intent(matrix: dict[str, Any] | None, pair: str, side: str) -> dict[str, Any]:
    if not isinstance(matrix, dict):
        return {}
    side_payload = (((matrix.get("pairs") or {}).get(pair) or {}).get(side))
    if not isinstance(side_payload, dict):
        return {}
    return {
        "market_context_matrix_ref": side_payload.get("evidence_ref"),
        "matrix_support_count": side_payload.get("support_count", 0),
        "matrix_reject_count": side_payload.get("reject_count", 0),
        "matrix_warning_count": side_payload.get("warning_count", 0),
        "matrix_missing_count": side_payload.get("missing_count", 0),
        "strongest_matrix_support": side_payload.get("strongest_support"),
        "strongest_matrix_reject": side_payload.get("strongest_reject"),
        "strongest_matrix_warning": side_payload.get("strongest_warning"),
        "matrix_support_layers": _layers(side_payload, bucket="supports"),
        "matrix_reject_layers": _layers(side_payload, bucket="rejects"),
        "matrix_warning_layers": _layers(side_payload, bucket="warnings"),
        "matrix_support_context": _cross_asset_context(side_payload, buckets=("supports",)),
        "matrix_reject_context": _cross_asset_context(side_payload, buckets=("rejects",)),
        "matrix_warning_context": _cross_asset_context(side_payload, buckets=("warnings",)),
        "matrix_support_refs": _evidence_refs(side_payload, buckets=("supports",)),
        "matrix_reject_refs": _evidence_refs(side_payload, buckets=("rejects",)),
        "matrix_warning_refs": _evidence_refs(side_payload, buckets=("warnings",)),
        "matrix_context_refs": _evidence_refs(side_payload, buckets=("supports", "rejects", "warnings")),
    }


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _empty_side(pair: str, side: str) -> dict[str, Any]:
    return {
        "evidence_ref": f"matrix:{pair}:{side}",
        "side": side,
        "supports": [],
        "rejects": [],
        "warnings": [],
        "missing": [],
        "horizon_conflicts": [],
    }


def _finalize_side(side_payload: dict[str, Any]) -> None:
    for key, count_key in (
        ("supports", "support_count"),
        ("rejects", "reject_count"),
        ("warnings", "warning_count"),
        ("missing", "missing_count"),
        ("horizon_conflicts", "horizon_conflict_count"),
    ):
        rows = side_payload.get(key) if isinstance(side_payload.get(key), list) else []
        side_payload[count_key] = len(rows)
    side_payload["strongest_support"] = _first_message(side_payload.get("supports"))
    side_payload["strongest_reject"] = _first_message(side_payload.get("rejects"))
    side_payload["strongest_warning"] = _first_message(side_payload.get("warnings"))


def _first_message(rows: Any) -> str | None:
    if not isinstance(rows, list) or not rows:
        return None
    first = rows[0]
    if not isinstance(first, dict):
        return None
    return str(first.get("message") or "") or None


def _layers(side_payload: dict[str, Any], *, bucket: str) -> list[str]:
    rows = side_payload.get(bucket) if isinstance(side_payload.get(bucket), list) else []
    return sorted(
        {
            str(row.get("layer") or "").strip()
            for row in rows
            if isinstance(row, dict) and str(row.get("layer") or "").strip()
        }
    )


def _evidence_refs(side_payload: dict[str, Any], *, buckets: tuple[str, ...]) -> list[str]:
    refs: list[str] = []
    for bucket in buckets:
        rows = side_payload.get(bucket) if isinstance(side_payload.get(bucket), list) else []
        for row in rows:
            if not isinstance(row, dict):
                continue
            for ref in row.get("evidence_refs", []) or []:
                text = str(ref).strip()
                if text and text not in refs:
                    refs.append(text)
                if len(refs) >= SUMMARY_LIST_LIMIT:
                    return refs
    return refs


def _cross_asset_context(side_payload: dict[str, Any], *, buckets: tuple[str, ...]) -> list[str]:
    items: list[str] = []
    for bucket in buckets:
        rows = side_payload.get(bucket) if isinstance(side_payload.get(bucket), list) else []
        for row in rows:
            if not isinstance(row, dict):
                continue
            layer = str(row.get("layer") or "")
            refs = [str(ref) for ref in row.get("evidence_refs", []) or []]
            if layer not in {"cross_asset", "context_asset_chart"} and not any(
                ref.startswith("context_asset:") or ref.startswith("cross:") for ref in refs
            ):
                continue
            code = str(row.get("code") or layer or "context")
            message = str(row.get("message") or "").strip()
            ref_text = f" refs={','.join(refs[:3])}" if refs else ""
            items.append(f"{code}: {message}{ref_text}" if message else f"{code}{ref_text}")
            if len(items) >= SUMMARY_LIST_LIMIT:
                return items
    return items


def _add(side_payload: dict[str, Any], bucket: str, *, code: str, layer: str, message: str, refs: Iterable[str]) -> None:
    side_payload.setdefault(bucket, []).append(
        {
            "code": code,
            "layer": layer,
            "message": message,
            "evidence_refs": [ref for ref in refs if ref],
        }
    )


def _directional(
    pair_matrix: dict[str, dict[str, Any]],
    *,
    support_side: str,
    code: str,
    layer: str,
    message: str,
    refs: Iterable[str],
    opposite_bucket: str = "rejects",
) -> None:
    opposite = "SHORT" if support_side == "LONG" else "LONG"
    _add(pair_matrix[support_side], "supports", code=code, layer=layer, message=message, refs=refs)
    _add(pair_matrix[opposite], opposite_bucket, code=code, layer=layer, message=message, refs=refs)


def _warning_both(pair_matrix: dict[str, dict[str, Any]], *, code: str, layer: str, message: str, refs: Iterable[str]) -> None:
    for side in SIDES:
        _add(pair_matrix[side], "warnings", code=code, layer=layer, message=message, refs=refs)


def _missing_both(pair_matrix: dict[str, dict[str, Any]], *, code: str, layer: str, message: str, refs: Iterable[str] = ()) -> None:
    for side in SIDES:
        _add(pair_matrix[side], "missing", code=code, layer=layer, message=message, refs=refs)


def _apply_chart_layer(pair_matrix: dict[str, dict[str, Any]], pair: str, chart: dict[str, Any] | None) -> None:
    if not isinstance(chart, dict):
        _missing_both(pair_matrix, code="MISSING_PAIR_CHART", layer="chart", message=f"{pair} pair_charts row missing")
        return
    refs = [f"chart:{pair}:structure"]
    confluence = chart.get("confluence") if isinstance(chart.get("confluence"), dict) else {}
    score_balance = str(confluence.get("score_balance") or "").upper()
    if score_balance == "LONG_LEAN":
        _directional(
            pair_matrix,
            support_side="LONG",
            code="CHART_CONFLUENCE_LONG_LEAN",
            layer="chart",
            message=f"{pair} confluence score_balance=LONG_LEAN",
            refs=refs,
        )
    elif score_balance == "SHORT_LEAN":
        _directional(
            pair_matrix,
            support_side="SHORT",
            code="CHART_CONFLUENCE_SHORT_LEAN",
            layer="chart",
            message=f"{pair} confluence score_balance=SHORT_LEAN",
            refs=refs,
        )
    long_score = _float_or_none(chart.get("long_score"))
    short_score = _float_or_none(chart.get("short_score"))
    if long_score is not None and short_score is not None:
        if long_score > short_score:
            _directional(
                pair_matrix,
                support_side="LONG",
                code="CHART_SCORE_LONG_DOMINATES",
                layer="chart",
                message=f"{pair} long_score {long_score:.3f} > short_score {short_score:.3f}",
                refs=refs,
            )
        elif short_score > long_score:
            _directional(
                pair_matrix,
                support_side="SHORT",
                code="CHART_SCORE_SHORT_DOMINATES",
                layer="chart",
                message=f"{pair} short_score {short_score:.3f} > long_score {long_score:.3f}",
                refs=refs,
            )
    dominant_regime = str(confluence.get("dominant_regime") or chart.get("dominant_regime") or "").upper()
    if "TREND_UP" in dominant_regime:
        _directional(
            pair_matrix,
            support_side="LONG",
            code="CHART_DOMINANT_TREND_UP",
            layer="chart",
            message=f"{pair} dominant_regime={dominant_regime}",
            refs=refs,
            opposite_bucket="warnings",
        )
    elif "TREND_DOWN" in dominant_regime:
        _directional(
            pair_matrix,
            support_side="SHORT",
            code="CHART_DOMINANT_TREND_DOWN",
            layer="chart",
            message=f"{pair} dominant_regime={dominant_regime}",
            refs=refs,
            opposite_bucket="warnings",
        )
    if str(confluence.get("higher_tf_alignment") or "").upper() == "OPPOSED":
        _warning_both(
            pair_matrix,
            code="HIGHER_TF_ALIGNMENT_OPPOSED",
            layer="chart",
            message=f"{pair} higher_tf_alignment=OPPOSED; declare counter-trend scope before extending TP",
            refs=refs,
        )


def _apply_strength_layer(
    pair_matrix: dict[str, dict[str, Any]],
    pair: str,
    base: str,
    quote: str,
    strength_scores: dict[str, dict[str, Any]],
) -> None:
    base_score = _float_or_none((strength_scores.get(base) or {}).get("score_pct"))
    quote_score = _float_or_none((strength_scores.get(quote) or {}).get("score_pct"))
    if base_score is None or quote_score is None:
        _missing_both(
            pair_matrix,
            code="MISSING_CURRENCY_STRENGTH",
            layer="strength",
            message=f"{pair} needs both {base} and {quote} strength scores",
            refs=[f"strength:{base}", f"strength:{quote}", f"strength:{pair}"],
        )
        return
    if base_score > quote_score:
        _directional(
            pair_matrix,
            support_side="LONG",
            code="BASE_STRENGTH_EXCEEDS_QUOTE",
            layer="strength",
            message=f"{pair} base {base} strength {base_score:.3f} > quote {quote} {quote_score:.3f}",
            refs=[f"strength:{base}", f"strength:{quote}", f"strength:{pair}"],
        )
    elif quote_score > base_score:
        _directional(
            pair_matrix,
            support_side="SHORT",
            code="QUOTE_STRENGTH_EXCEEDS_BASE",
            layer="strength",
            message=f"{pair} quote {quote} strength {quote_score:.3f} > base {base} {base_score:.3f}",
            refs=[f"strength:{base}", f"strength:{quote}", f"strength:{pair}"],
        )


def _apply_cross_asset_layer(
    pair_matrix: dict[str, dict[str, Any]],
    pair: str,
    base: str,
    quote: str,
    cross_asset: dict[str, Any] | None,
) -> None:
    if not isinstance(cross_asset, dict):
        _missing_both(pair_matrix, code="MISSING_CROSS_ASSET", layer="cross_asset", message="cross_asset_snapshot missing")
        return
    dxy_change = _float_or_none(((cross_asset.get("synthetic_dxy") or {}).get("change_pct_24h")))
    if dxy_change is not None and "USD" in (base, quote) and dxy_change != 0:
        usd_support_side = "LONG" if base == "USD" else "SHORT"
        if dxy_change < 0:
            usd_support_side = "SHORT" if base == "USD" else "LONG"
        _directional(
            pair_matrix,
            support_side=usd_support_side,
            code="DXY_24H_DIRECTION",
            layer="cross_asset",
            message=f"{pair} synthetic DXY 24h change {dxy_change:.3f}% maps to {usd_support_side}",
            refs=["cross:dxy"],
        )
    usb10 = _asset(cross_asset, "USB10Y_USD")
    yield_change = _float_or_none((usb10 or {}).get("change_pct_24h"))
    if yield_change is not None and "JPY" in (base, quote) and yield_change != 0:
        jpy_weak_side = "LONG" if quote == "JPY" else "SHORT"
        support_side = jpy_weak_side if yield_change > 0 else ("SHORT" if jpy_weak_side == "LONG" else "LONG")
        _directional(
            pair_matrix,
            support_side=support_side,
            code="US10Y_JPY_CROSS_DIRECTION",
            layer="cross_asset",
            message=f"{pair} USB10Y_USD 24h change {yield_change:.3f}% maps to {support_side}",
            refs=["cross:USB10Y_USD"],
            opposite_bucket="warnings",
        )
    spx = _asset(cross_asset, "SPX500_USD")
    spx_change = _float_or_none((spx or {}).get("change_pct_24h"))
    if spx_change is not None and quote == "JPY" and spx_change != 0:
        support_side = "LONG" if spx_change > 0 else "SHORT"
        _directional(
            pair_matrix,
            support_side=support_side,
            code="RISK_ASSET_JPY_CROSS_DIRECTION",
            layer="cross_asset",
            message=f"{pair} SPX500_USD 24h change {spx_change:.3f}% maps to {support_side}",
            refs=["cross:spx"],
            opposite_bucket="warnings",
        )
    xau = _asset(cross_asset, "XAU_USD")
    xau_change = _float_or_none((xau or {}).get("change_pct_24h"))
    if xau_change is not None and "USD" in (base, quote) and xau_change != 0:
        usd_pressure_side = "SHORT" if base == "USD" else "LONG"
        support_side = usd_pressure_side if xau_change > 0 else ("SHORT" if usd_pressure_side == "LONG" else "LONG")
        _directional(
            pair_matrix,
            support_side=support_side,
            code="GOLD_USD_PRESSURE_DIRECTION",
            layer="cross_asset",
            message=f"{pair} XAU_USD 24h change {xau_change:.3f}% maps to {support_side} as USD-pressure context",
            refs=["cross:XAU_USD"],
            opposite_bucket="warnings",
        )
    oil = _asset(cross_asset, "WTICO_USD") or _asset(cross_asset, "BCO_USD")
    oil_instrument = str((oil or {}).get("instrument") or "WTICO_USD")
    oil_change = _float_or_none((oil or {}).get("change_pct_24h"))
    if oil_change is not None and "CAD" in (base, quote) and oil_change != 0:
        cad_support_side = "LONG" if base == "CAD" else "SHORT"
        support_side = cad_support_side if oil_change > 0 else ("SHORT" if cad_support_side == "LONG" else "LONG")
        _directional(
            pair_matrix,
            support_side=support_side,
            code="OIL_CAD_DIRECTION",
            layer="cross_asset",
            message=f"{pair} {oil_instrument} 24h change {oil_change:.3f}% maps to {support_side} as CAD commodity context",
            refs=[f"cross:{oil_instrument}"],
            opposite_bucket="warnings",
        )


def _apply_context_asset_chart_layer(
    pair_matrix: dict[str, dict[str, Any]],
    pair: str,
    base: str,
    quote: str,
    context_charts: dict[str, dict[str, Any]],
) -> None:
    gold_direction = _context_asset_direction(context_charts.get("XAU_USD"))
    if gold_direction and "USD" in (base, quote):
        usd_pressure_side = "SHORT" if base == "USD" else "LONG"
        support_side = usd_pressure_side if gold_direction == "UP" else _opposite_side(usd_pressure_side)
        _directional(
            pair_matrix,
            support_side=support_side,
            code="GOLD_CONTEXT_TECHNICAL_DIRECTION",
            layer="context_asset_chart",
            message=(
                f"{pair} XAU_USD technical {_context_asset_label(context_charts.get('XAU_USD'))} "
                f"maps to {support_side} as USD-pressure context"
            ),
            refs=["context_asset:XAU_USD"],
            opposite_bucket="warnings",
        )

    oil_instrument, oil_direction = _first_context_asset_direction(context_charts, ("WTICO_USD", "BCO_USD"))
    if oil_instrument and oil_direction and "CAD" in (base, quote):
        cad_support_side = "LONG" if base == "CAD" else "SHORT"
        support_side = cad_support_side if oil_direction == "UP" else _opposite_side(cad_support_side)
        _directional(
            pair_matrix,
            support_side=support_side,
            code="OIL_CONTEXT_TECHNICAL_DIRECTION",
            layer="context_asset_chart",
            message=(
                f"{pair} {oil_instrument} technical {_context_asset_label(context_charts.get(oil_instrument))} "
                f"maps to {support_side} as CAD commodity context"
            ),
            refs=[f"context_asset:{oil_instrument}"],
            opposite_bucket="warnings",
        )

    spx_direction = _context_asset_direction(context_charts.get("SPX500_USD"))
    if spx_direction and "JPY" in (base, quote):
        jpy_weak_side = "LONG" if quote == "JPY" else "SHORT"
        support_side = jpy_weak_side if spx_direction == "UP" else _opposite_side(jpy_weak_side)
        _directional(
            pair_matrix,
            support_side=support_side,
            code="EQUITY_INDEX_CONTEXT_TECHNICAL_DIRECTION",
            layer="context_asset_chart",
            message=(
                f"{pair} SPX500_USD technical {_context_asset_label(context_charts.get('SPX500_USD'))} "
                f"maps to {support_side} as JPY risk context"
            ),
            refs=["context_asset:SPX500_USD"],
            opposite_bucket="warnings",
        )

    us10y_direction = _context_asset_direction(context_charts.get("USB10Y_USD"))
    if us10y_direction and "JPY" in (base, quote):
        jpy_weak_side = "LONG" if quote == "JPY" else "SHORT"
        support_side = jpy_weak_side if us10y_direction == "UP" else _opposite_side(jpy_weak_side)
        _directional(
            pair_matrix,
            support_side=support_side,
            code="US10Y_CONTEXT_TECHNICAL_DIRECTION",
            layer="context_asset_chart",
            message=(
                f"{pair} USB10Y_USD technical {_context_asset_label(context_charts.get('USB10Y_USD'))} "
                f"maps to {support_side} as yield/JPY context"
            ),
            refs=["context_asset:USB10Y_USD"],
            opposite_bucket="warnings",
        )


def _apply_flow_layer(
    pair_matrix: dict[str, dict[str, Any]],
    pair: str,
    spread: dict[str, Any] | None,
    order_book: dict[str, Any] | None,
    position_book: dict[str, Any] | None,
) -> None:
    if not isinstance(spread, dict):
        _missing_both(pair_matrix, code="MISSING_SPREAD_STATS", layer="flow", message=f"{pair} spread stats missing", refs=[f"flow:{pair}"])
    else:
        flag = str(spread.get("stress_flag") or "").upper()
        current = spread.get("current_pips")
        median = spread.get("median_pips")
        if flag == "STRESSED":
            for side in SIDES:
                _add(
                    pair_matrix[side],
                    "rejects",
                    code="FLOW_SPREAD_STRESSED",
                    layer="flow",
                    message=f"{pair} spread stress=STRESSED current={current} median={median}",
                    refs=[f"flow:{pair}"],
                )
        elif flag == "ELEVATED":
            _warning_both(
                pair_matrix,
                code="FLOW_SPREAD_ELEVATED",
                layer="flow",
                message=f"{pair} spread stress=ELEVATED current={current} median={median}",
                refs=[f"flow:{pair}"],
            )
        elif flag:
            for side in SIDES:
                _add(
                    pair_matrix[side],
                    "supports",
                    code="FLOW_SPREAD_EXECUTABLE",
                    layer="flow",
                    message=f"{pair} spread stress={flag} current={current} median={median}",
                    refs=[f"flow:{pair}"],
                )
    for book, code in ((order_book, "MISSING_ORDER_BOOK"), (position_book, "MISSING_POSITION_BOOK")):
        issue = str((book or {}).get("issue") or "")
        if issue:
            _missing_both(pair_matrix, code=code, layer="flow", message=f"{pair} {issue}", refs=[f"flow:{pair}"])


def _apply_levels_layer(pair_matrix: dict[str, dict[str, Any]], pair: str, levels: dict[str, Any] | None) -> None:
    if not isinstance(levels, dict):
        _missing_both(pair_matrix, code="MISSING_LEVELS", layer="levels", message=f"{pair} levels missing", refs=[f"levels:{pair}"])
        return
    last_close = _float_or_none(levels.get("last_close"))
    for key, label in (("daily_open", "daily open"), ("weekly_open", "weekly open"), ("pdc", "previous daily close")):
        anchor = _float_or_none(levels.get(key))
        if last_close is None or anchor is None or last_close == anchor:
            continue
        support_side = "LONG" if last_close > anchor else "SHORT"
        _directional(
            pair_matrix,
            support_side=support_side,
            code=f"PRICE_VS_{key.upper()}",
            layer="levels",
            message=f"{pair} last_close {last_close} is {'above' if support_side == 'LONG' else 'below'} {label} {anchor}",
            refs=[f"levels:{pair}"],
            opposite_bucket="warnings",
        )
    round_numbers = [item for item in levels.get("round_numbers", []) or [] if isinstance(item, dict)]
    if round_numbers:
        nearest = sorted(round_numbers, key=lambda item: abs(float(item.get("distance_pips") or 0.0)))[0]
        _warning_both(
            pair_matrix,
            code="NEAREST_ROUND_NUMBER_CONTEXT",
            layer="levels",
            message=f"{pair} nearest round number {nearest.get('price')} distance_pips={nearest.get('distance_pips')}",
            refs=[f"levels:{pair}"],
        )


def _apply_calendar_layer(pair_matrix: dict[str, dict[str, Any]], pair: str, window: dict[str, Any] | None) -> None:
    if not isinstance(window, dict):
        _missing_both(pair_matrix, code="MISSING_CALENDAR_WINDOW", layer="calendar", message=f"{pair} calendar window missing", refs=[f"calendar:{pair}"])
        return
    reason = str(window.get("reason") or "")
    if window.get("in_window") is True and _calendar_feed_unavailable(reason):
        _missing_both(
            pair_matrix,
            code="CALENDAR_FEED_UNAVAILABLE",
            layer="calendar",
            message=f"{pair} calendar feed unavailable: {reason}",
            refs=[f"calendar:{pair}"],
        )
        return
    if window.get("in_window") is True:
        for side in SIDES:
            _add(
                pair_matrix[side],
                "rejects",
                code="CALENDAR_EVENT_WINDOW",
                layer="calendar",
                message=f"{pair} calendar in_window=true: {reason}",
                refs=[f"calendar:{pair}"],
            )
    elif reason:
        _warning_both(
            pair_matrix,
            code="NEXT_CALENDAR_EVENT_CONTEXT",
            layer="calendar",
            message=f"{pair} calendar context: {reason}",
            refs=[f"calendar:{pair}"],
        )


def _calendar_feed_unavailable(reason: str) -> bool:
    normalized = reason.upper()
    return "CALENDAR UNAVAILABLE" in normalized or "MISSING_FOREX_FACTORY_FEED" in normalized


def _apply_cot_layer(
    pair_matrix: dict[str, dict[str, Any]],
    pair: str,
    base: str,
    quote: str,
    cot_reports: dict[str, dict[str, Any]],
) -> None:
    base_net = _float_or_none((cot_reports.get(base) or {}).get("leveraged_net"))
    quote_net = _float_or_none((cot_reports.get(quote) or {}).get("leveraged_net"))
    if base_net is None or quote_net is None:
        _missing_both(
            pair_matrix,
            code="MISSING_COT_CURRENCY",
            layer="cot",
            message=f"{pair} COT needs {base} and {quote} leveraged_net",
            refs=[f"cot:{base}", f"cot:{quote}"],
        )
        return
    if base_net == quote_net:
        return
    support_side = "LONG" if base_net > quote_net else "SHORT"
    opposite = "SHORT" if support_side == "LONG" else "LONG"
    _add(
        pair_matrix[support_side],
        "warnings",
        code="COT_LONGER_TERM_ALIGNS",
        layer="cot",
        message=f"{pair} COT leveraged_net base={base_net:.0f} quote={quote_net:.0f} aligns {support_side}; longer-term only",
        refs=[f"cot:{base}", f"cot:{quote}"],
    )
    _add(
        pair_matrix[opposite],
        "warnings",
        code="COT_LONGER_TERM_CONFLICTS",
        layer="cot",
        message=f"{pair} COT leveraged_net base={base_net:.0f} quote={quote_net:.0f} conflicts {opposite}; not an M1/M5 blocker",
        refs=[f"cot:{base}", f"cot:{quote}"],
    )


def _apply_option_skew_layer(pair_matrix: dict[str, dict[str, Any]], pair: str, option_skew: dict[str, Any] | None) -> None:
    if not isinstance(option_skew, dict):
        return
    if option_skew.get("enabled") is False and option_skew.get("disabled_reason"):
        return
    readings = [
        item
        for item in option_skew.get("readings", []) or []
        if isinstance(item, dict) and str(item.get("pair") or "") == pair
    ]
    if not readings:
        _missing_both(pair_matrix, code="MISSING_OPTION_SKEW_PAIR", layer="option_skew", message=f"{pair} option skew reading missing", refs=[f"option:skew:{pair}", "option:skew:unknown"])
        return
    usable = False
    for reading in readings:
        rr = _float_or_none(reading.get("rr_25d"))
        issue = str(reading.get("issue") or "")
        if rr is None:
            if issue:
                _missing_both(pair_matrix, code="MISSING_OPTION_SKEW_FEED", layer="option_skew", message=f"{pair} {issue}", refs=[f"option:skew:{pair}", "option:skew:unknown"])
            continue
        usable = True
        if rr == 0:
            continue
        support_side = "LONG" if rr > 0 else "SHORT"
        _directional(
            pair_matrix,
            support_side=support_side,
            code="OPTION_RR_25D_DIRECTION",
            layer="option_skew",
            message=f"{pair} {reading.get('tenor')} rr_25d={rr} maps to {support_side}",
            refs=[f"option:skew:{pair}"],
            opposite_bucket="warnings",
        )
    if not usable and not any((item.get("issue") if isinstance(item, dict) else None) for item in readings):
        _missing_both(pair_matrix, code="MISSING_OPTION_SKEW_VALUES", layer="option_skew", message=f"{pair} option skew has no rr_25d values", refs=[f"option:skew:{pair}", "option:skew:unknown"])


def _first_context_asset_direction(
    context_charts: dict[str, dict[str, Any]],
    instruments: tuple[str, ...],
) -> tuple[str | None, str | None]:
    for instrument in instruments:
        direction = _context_asset_direction(context_charts.get(instrument))
        if direction:
            return instrument, direction
    return None, None


def _context_asset_direction(chart: dict[str, Any] | None) -> str | None:
    if not isinstance(chart, dict):
        return None
    confluence = chart.get("confluence") if isinstance(chart.get("confluence"), dict) else {}
    score_balance = str(confluence.get("score_balance") or "").upper()
    if score_balance == "LONG_LEAN":
        return "UP"
    if score_balance == "SHORT_LEAN":
        return "DOWN"
    dominant_regime = str(confluence.get("dominant_regime") or chart.get("dominant_regime") or "").upper()
    if "TREND_UP" in dominant_regime or "IMPULSE_UP" in dominant_regime:
        return "UP"
    if "TREND_DOWN" in dominant_regime or "IMPULSE_DOWN" in dominant_regime:
        return "DOWN"
    long_score = _float_or_none(chart.get("long_score"))
    short_score = _float_or_none(chart.get("short_score"))
    if long_score is None or short_score is None or long_score == short_score:
        return None
    return "UP" if long_score > short_score else "DOWN"


def _context_asset_label(chart: dict[str, Any] | None) -> str:
    if not isinstance(chart, dict):
        return "missing"
    confluence = chart.get("confluence") if isinstance(chart.get("confluence"), dict) else {}
    score_balance = str(confluence.get("score_balance") or "").upper()
    dominant_regime = str(confluence.get("dominant_regime") or chart.get("dominant_regime") or "").upper()
    direction = _context_asset_direction(chart) or "UNKNOWN"
    long_score = _float_or_none(chart.get("long_score"))
    short_score = _float_or_none(chart.get("short_score"))
    parts = [f"direction={direction}"]
    if dominant_regime:
        parts.append(f"regime={dominant_regime}")
    if score_balance:
        parts.append(f"score_balance={score_balance}")
    if long_score is not None and short_score is not None:
        parts.append(f"long={long_score:.3f}/short={short_score:.3f}")
    return " ".join(parts)


def _opposite_side(side: str) -> str:
    return "SHORT" if side == "LONG" else "LONG"


def _by_key(rows: Any, key: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for item in rows or []:
        if not isinstance(item, dict):
            continue
        value = str(item.get(key) or "")
        if value:
            out[value] = item
    return out


def _option_pairs(payload: dict[str, Any] | None) -> set[str]:
    return {
        str(item.get("pair") or "")
        for item in (payload or {}).get("readings", []) or []
        if isinstance(item, dict) and item.get("pair")
    }


def _asset(payload: dict[str, Any], instrument: str) -> dict[str, Any] | None:
    for item in payload.get("assets", []) or []:
        if isinstance(item, dict) and item.get("instrument") == instrument:
            return item
    return None


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
