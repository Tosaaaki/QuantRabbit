from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol


def _trader_sl_repair_disabled() -> bool:
    return os.environ.get("QR_TRADER_DISABLE_SL_REPAIR", "").strip() in {"1", "true", "TRUE", "yes", "YES"}


def _missing_tp_repair_enabled() -> bool:
    return os.environ.get("QR_ENABLE_MISSING_TP_REPAIR", "").strip() in {"1", "true", "TRUE", "yes", "YES"}


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_utc(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _operator_close_override_active() -> bool:
    """Emergency override for the CLOSE-discipline gate.

    Set `QR_OPERATOR_CLOSE_OVERRIDE=1` in the operator shell when an out-of-
    band close is urgently needed (broker disconnect, regulatory order,
    user-confirmed structural reversal not yet visible to pair_charts).
    Documented so the override is auditable rather than implicit.
    """
    return os.environ.get("QR_OPERATOR_CLOSE_OVERRIDE", "").strip() in {
        "1", "true", "TRUE", "yes", "YES",
    }


# J (2026-05-13) — Operator close token file. A token-file authorization
# path that the GPT trader receipt cannot self-set. The operator creates
# the file explicitly with `touch data/.operator_close_token`; the verifier
# reads its mtime and rejects if older than the documented freshness window.
#
# 2026-05-12T15:33 UTC mass-close incident proved that
# `operator_close_authorized: true` JSON field is honor-system: the
# trader fills its own receipt. The token file lives in `data/`, a
# directory the trader process can technically write to, but it would
# require the model to identify, name, and `touch` the file — none of
# which match the normal write-this-decision-receipt flow. The
# `operator_close_authorized` JSON field is now treated as advisory
# audit only; it is NOT accepted as Gate B authorization.
OPERATOR_CLOSE_TOKEN_FILENAME = ".operator_close_token"
OPERATOR_CLOSE_TOKEN_FRESH_SECONDS = 300  # 5 minutes documented window


def _operator_close_token_fresh(data_root: Path | None = None) -> bool:
    """Whether a fresh operator close-authorization token file exists.

    The default location is `data/.operator_close_token` under the
    repo root; tests inject their own path. A token older than
    `OPERATOR_CLOSE_TOKEN_FRESH_SECONDS` is treated as stale and the
    gate fails — operators must explicitly re-authorize before each
    CLOSE batch.
    """
    if data_root is None:
        # Lazy import to avoid module-level cwd dependency in tests.
        from quant_rabbit.paths import ROOT
        data_root = ROOT / "data"
    token = data_root / OPERATOR_CLOSE_TOKEN_FILENAME
    if not token.exists():
        return False
    try:
        mtime = token.stat().st_mtime
    except OSError:
        return False
    age = datetime.now(timezone.utc).timestamp() - mtime
    return age <= OPERATOR_CLOSE_TOKEN_FRESH_SECONDS


def _operator_close_gate_authorized() -> bool:
    """Gate B authorization for loss-side CLOSE decisions."""
    return _operator_close_override_active() or _operator_close_token_fresh()


# Matches a single per-timeframe segment in chart_reader's chart_story format.
# Example tokens captured:
#   "M15(RANGE, ADX=15.4 ... struct=CHOCH_UP@113.9900)"        (close confirmed)
#   "M15(RANGE, ADX=15.4 ... struct=BOS_UP@114.1460:wick)"     (wick-only)
# Group 1 = timeframe (M1/M5/M15/M30/H1/H4/D), Group 2 = "BOS"|"CHOCH",
# Group 3 = "UP"|"DOWN", Group 4 = numeric price, Group 5 = ":wick" tag or "".
# The optional ":wick" suffix marks a break whose breaking candle closed
# back inside the prior range — Gate A treats it as a stop-hunt that does
# NOT authorize CLOSE on its own (added 2026-05-13, structure.py
# close_confirmed flag).
_STRUCT_EVENT_RE = re.compile(
    r"\b(M1|M5|M15|M30|H1|H4|D)\([^)]*?struct=(BOS|CHOCH)_(UP|DOWN)@([0-9]+\.?[0-9]*)(:wick)?",
)

# Timeframes consulted when deciding whether the operator-thesis behind a
# trader-owned position has been structurally invalidated. M15 catches the
# fast-fail flip; H4 catches the dominant-tape reversal. Lower TFs (M1/M5)
# would whip the gate around session noise; higher TFs (D/W) would lag past
# the per_trade_risk_budget. Both are anchored by chart_reader.structure_events
# rather than a JPY/pip literal, so they're §6-compliant.
CLOSE_DISCIPLINE_TIMEFRAMES: tuple[str, ...] = ("M15", "H4")

# Layers that can represent directional evidence for/against the position
# thesis. Session safety layers such as flow/calendar are intentionally omitted:
# a normal spread or quiet event window is useful context, but it does not prove
# recovery edge for a same-direction position.
CLOSE_DIRECTIONAL_MATRIX_LAYERS: frozenset[str] = frozenset(
    {
        "chart",
        "strength",
        "currency_strength",
        "cross_asset",
        "context_asset_chart",
        "levels",
        "cot",
        "option_skew",
    }
)


def _parse_struct_events(
    chart_story: str,
) -> dict[str, tuple[str, str, float, bool]]:
    """Return {timeframe: (event_type, direction, price, close_confirmed)} parsed
    from chart_story.

    Silently skips tokens whose price does not parse as a float. The
    parser is intentionally tolerant: chart_reader format drift should
    degrade to "no thesis-invalidation evidence" rather than crash the
    verifier and stall the cycle.

    `close_confirmed` is False when chart_reader appended the `:wick`
    suffix — the breaking candle's close did not clear the prior pivot,
    so the high/low was swept but the close held inside the range. Gate
    A treats wick-only breaks as advisory, not as CLOSE authorization.
    """
    if not chart_story:
        return {}
    out: dict[str, tuple[str, str, float, bool]] = {}
    for tf, event_type, direction, price_str, wick_tag in _STRUCT_EVENT_RE.findall(
        chart_story
    ):
        try:
            close_confirmed = wick_tag != ":wick"
            out[tf] = (event_type, direction, float(price_str), close_confirmed)
        except (TypeError, ValueError):
            continue
    return out


def _pair_chart(packet: dict[str, Any], pair: str) -> dict[str, Any]:
    """Pull the full per-pair chart payload from the verifier packet."""
    pairs_block = (
        ((packet.get("market_context") or {}).get("pairs") or {})
        .get(pair) or {}
    )
    chart = pairs_block.get("chart") if isinstance(pairs_block, dict) else None
    if isinstance(chart, dict):
        return chart
    return {}


def _pair_chart_story(packet: dict[str, Any], pair: str) -> str:
    """Pull the per-pair chart_story string from the verifier packet."""
    return str(_pair_chart(packet, pair).get("chart_story") or "")


def _close_same_direction_matrix_support(
    packet: dict[str, Any],
    pair: str,
    side: str,
) -> tuple[bool, str]:
    side_upper = str(side or "").upper()
    if side_upper not in {"LONG", "SHORT"}:
        return False, ""
    pair_block = (
        ((packet.get("market_context") or {}).get("pairs") or {})
        .get(pair) or {}
    )
    matrix = pair_block.get("matrix") if isinstance(pair_block, dict) else None
    reading = matrix.get(side_upper) if isinstance(matrix, dict) else None
    if not isinstance(reading, dict):
        return False, ""

    supports = _directional_matrix_observations(reading.get("supports"))
    if not supports:
        return False, ""
    rejects = _directional_matrix_observations(reading.get("rejects"))
    if len(rejects) >= len(supports):
        return False, ""

    support_codes = [
        str(item.get("code") or item.get("layer") or "directional_support")
        for item in supports
    ]
    refs: list[str] = []
    for item in supports:
        for ref in item.get("evidence_refs") or []:
            text = str(ref).strip()
            if text and text not in refs:
                refs.append(text)
    reading_ref = str(reading.get("evidence_ref") or "").strip()
    if reading_ref and reading_ref not in refs:
        refs.insert(0, reading_ref)
    ref_text = f"; refs={', '.join(refs[:5])}" if refs else ""
    return (
        True,
        f"{pair} {side_upper} still has directional matrix support "
        f"({', '.join(support_codes[:4])}){ref_text}",
    )


def _close_same_direction_sidecar_support(
    packet: dict[str, Any],
    *,
    trade_id: str | None,
    pair: str,
    side: str,
) -> tuple[bool, str]:
    """Whether fresh position sidecars still support carrying this position.

    The support floor is a categorical majority across the three independent
    position-review sidecars: position_thesis, thesis_evolution, and
    forecast_persistence. This is not a market threshold; it prevents one M15
    internal structure flip from forcing out an H1/H4 swing while most of the
    current position stack still says the thesis is alive.
    """

    sidecars = packet.get("protection_sidecars")
    if not isinstance(sidecars, dict):
        return False, ""
    support = sidecars.get("position_hold_support")
    if not isinstance(support, list):
        return False, ""

    matched: list[dict[str, Any]] = []
    for rec in support:
        if not isinstance(rec, dict):
            continue
        rec_trade = str(rec.get("trade_id") or "")
        if trade_id is not None and rec_trade != str(trade_id):
            continue
        rec_pair = str(rec.get("pair") or "")
        if pair and rec_pair not in {"", pair}:
            continue
        rec_side = str(rec.get("side") or "").upper()
        if side and rec_side not in {"", str(side).upper()}:
            continue
        matched.append(rec)

    sources = sorted({str(rec.get("source") or "") for rec in matched if str(rec.get("source") or "")})
    if len(sources) < 2:
        return False, ""
    summary = ", ".join(
        f"{rec.get('source')}:{rec.get('verdict') or rec.get('status')}"
        for rec in matched
        if rec.get("source")
    )
    return True, f"fresh same-direction position sidecars support HOLD/EXTEND ({summary})"


def _directional_matrix_observations(rows: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in rows or []:
        if not isinstance(item, dict):
            continue
        layer = str(item.get("layer") or "").strip()
        if layer in CLOSE_DIRECTIONAL_MATRIX_LAYERS:
            out.append(item)
    return out


def _trader_position_lookup(packet: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Map trade_id -> position summary for trader-owned open positions."""
    out: dict[str, dict[str, Any]] = {}
    snapshot = packet.get("broker_snapshot") or {}
    for pos in snapshot.get("position_summaries", []) or []:
        if str(pos.get("owner") or "") != "trader":
            continue
        tid = pos.get("trade_id")
        if tid is None:
            continue
        out[str(tid)] = pos
    return out


def _position_close_spread_override_enabled() -> bool:
    return os.environ.get("QR_POSITION_CLOSE_SPREAD_OVERRIDE", "").strip() in {
        "1", "true", "TRUE", "yes", "YES",
    }


def _close_spread_session_tag(packet: dict[str, Any], pair: str) -> str | None:
    market_pairs = (packet.get("market_context") or {}).get("pairs") or {}
    pair_context = market_pairs.get(pair, {})
    if not isinstance(pair_context, dict):
        return None
    chart = pair_context.get("chart")
    if isinstance(chart, dict):
        session = chart.get("session")
        if isinstance(session, dict):
            tag = session.get("current_tag") or session.get("session_current_tag") or session.get("session_bucket")
            if tag:
                return str(tag)
    technical = pair_context.get("technical_context")
    if isinstance(technical, dict):
        tag = technical.get("session_current_tag") or technical.get("session_bucket")
        if tag:
            return str(tag)
    return None


def _close_spread_issues(
    packet: dict[str, Any],
    pair: str,
    *,
    trade_id: str,
) -> tuple["VerificationIssue", ...]:
    if _position_close_spread_override_enabled():
        return ()

    normal_spread = NORMAL_SPREAD_PIPS.get(pair)
    if normal_spread is None:
        return (
            VerificationIssue(
                "POSITION_CLOSE_SPREAD_BASELINE_MISSING",
                f"CLOSE rejected for trade {trade_id} {pair}: missing normal spread baseline",
            ),
        )

    max_spread_multiple = RiskPolicy().max_spread_multiple
    session_tag = _close_spread_session_tag(packet, pair)
    session_mult = _spread_session_multiplier_from_tag(session_tag)
    effective_spread_cap_mult = max_spread_multiple * session_mult
    spread_cap = normal_spread * effective_spread_cap_mult
    issues: list[VerificationIssue] = []

    snapshot = packet.get("broker_snapshot") or {}
    quote = (snapshot.get("quotes") or {}).get(pair)
    if not isinstance(quote, dict):
        issues.append(
            VerificationIssue(
                "POSITION_CLOSE_QUOTE_MISSING",
                f"CLOSE rejected for trade {trade_id} {pair}: missing broker quote",
            )
        )
    else:
        bid = _optional_float(quote.get("bid"))
        ask = _optional_float(quote.get("ask"))
        if bid is None or ask is None:
            issues.append(
                VerificationIssue(
                    "POSITION_CLOSE_QUOTE_MISSING",
                    f"CLOSE rejected for trade {trade_id} {pair}: broker quote lacks bid/ask",
                )
            )
        else:
            spread_pips = abs(ask - bid) * instrument_pip_factor(pair)
            if spread_pips > spread_cap:
                issues.append(
                    VerificationIssue(
                        "POSITION_CLOSE_SPREAD_TOO_WIDE",
                        "CLOSE rejected for trade "
                        f"{trade_id} {pair}: broker quote spread {spread_pips:.2f} pips "
                        f"exceeds close cap {spread_cap:.2f} pips "
                        f"({effective_spread_cap_mult:.2f}x normal {normal_spread:.1f}pip; "
                        f"policy={max_spread_multiple:.1f}x, session={session_tag or 'UNKNOWN'}, "
                        f"session_mult={session_mult:.2f})",
                    )
                )

    market_pairs = (packet.get("market_context") or {}).get("pairs") or {}
    pair_context = market_pairs.get(pair, {})
    flow_spread = (
        ((pair_context.get("flow") or {}).get("spread") or {})
        if isinstance(pair_context, dict)
        else {}
    )
    if isinstance(flow_spread, dict):
        flow_current = _optional_float(flow_spread.get("current_pips"))
        if flow_current is not None and flow_current > spread_cap:
            stress_flag = str(flow_spread.get("stress_flag") or "UNKNOWN")
            issues.append(
                VerificationIssue(
                    "POSITION_CLOSE_FLOW_SPREAD_TOO_WIDE",
                    "CLOSE rejected for trade "
                    f"{trade_id} {pair}: flow spread {flow_current:.2f} pips "
                    f"({stress_flag}) exceeds close cap {spread_cap:.2f} pips "
                    f"({effective_spread_cap_mult:.2f}x normal {normal_spread:.1f}pip; "
                    f"policy={max_spread_multiple:.1f}x, session={session_tag or 'UNKNOWN'}, "
                    f"session_mult={session_mult:.2f})",
                )
            )

    return tuple(issues)


def _close_thesis_invalidated(
    packet: dict[str, Any],
    pair: str,
    side: str,
    *,
    trade_id: str | None = None,
    decision: "GPTTraderDecision | None" = None,
) -> tuple[bool, str]:
    invalidated, reason, _standing_authorized = _close_thesis_invalidation(
        packet,
        pair,
        side,
        trade_id=trade_id,
        decision=decision,
    )
    return invalidated, reason


def _close_thesis_invalidation(
    packet: dict[str, Any],
    pair: str,
    side: str,
    *,
    trade_id: str | None = None,
    decision: "GPTTraderDecision | None" = None,
) -> tuple[bool, str, bool]:
    """Check whether the position's thesis has been invalidated.

    Returns `(invalidated, reason, standing_authorized)`.

    The first two Gate A paths are hard machine evidence and satisfy the
    operator's standing instruction that justified loss-cuts are allowed. The
    sidecar path may be hard (`thesis_evolution`) or soft
    (`position_thesis` / `forecast_persistence`); soft reviews still require
    explicit operator Gate B.

    Acceptance paths (§6-compliant — no JPY/pip/multiplier literals):

      (a) Structural BOS or CHOCH on M15 or H4 printing AGAINST the
          position side. This is the chart_reader.structure_events lens
          that already drives trader_brain price-action scoring; using
          the same signal keeps prefilter and CLOSE-gate aligned.

      (b) The decision receipt's `invalidation_price` + `invalidation_tf`
          fields are populated AND broker-truth quote has cleared the
          level by the configured anti-wick buffer. LONG invalidates
          downward, SHORT upward. Pure prose `invalidation` text alone
          is not enough — the gate requires a machine-checkable price
          hit beyond the buffer plus chart/technical confirmation.

      (c) A fresh position sidecar generated after the current broker
          snapshot marks this trade `REVIEW_CLOSE` / `RECOMMEND_CLOSE`
          from the prediction stack, thesis evolution, or N-cycle
          forecast persistence. This is the machine-checkable
          "no longer likely to recover to plus" path. `thesis_evolution`
          BROKEN / RECOMMEND_CLOSE and `position_thesis` no-ledger/adverse
          technical-loss evidence with multi-TF confirmation are treated as
          hard standing loss-cut authorization; softer sidecars still require
          env/token Gate B.
    """
    side_upper = str(side or "").upper()
    if side_upper not in {"LONG", "SHORT"}:
        return False, "unknown position side", False

    chart_story = _pair_chart_story(packet, pair)
    structs = _parse_struct_events(chart_story)
    counter_direction = "DOWN" if side_upper == "LONG" else "UP"
    m15_supported_pullback_reason: str | None = None
    m15_soft_structural_reason: str | None = None
    for tf in CLOSE_DISCIPLINE_TIMEFRAMES:
        event = structs.get(tf)
        if event and event[1] == counter_direction:
            event_type, direction, price, close_confirmed = event
            if not close_confirmed:
                # Wick-only break (the candle that printed the new pivot
                # closed back inside the prior range). Classic stop-hunt
                # / liquidity-sweep — does NOT authorize Gate A on its
                # own. The structural high/low was tagged but the move
                # was rejected.
                continue
            if tf == "M15":
                supported, support_reason = _close_same_direction_sidecar_support(
                    packet,
                    trade_id=trade_id,
                    pair=pair,
                    side=side_upper,
                )
                if supported:
                    m15_supported_pullback_reason = (
                        f"M15 {event_type}_{direction}@{price:g} prints against "
                        f"{side_upper}, but {support_reason}"
                    )
                    continue
                soft_blocker = _soft_sidecar_blocks_hard_close_authorization(
                    packet,
                    trade_id=trade_id,
                    pair=pair,
                    side=side_upper,
                )
                reason = (
                    f"M15 {event_type}_{direction}@{price:g} prints against "
                    f"{side_upper} thesis (close-confirmed)"
                )
                if soft_blocker:
                    m15_soft_structural_reason = f"{reason}; {soft_blocker}"
                    continue
                m15_soft_structural_reason = (
                    f"{reason}; M15 structure is Gate A evidence but requires explicit Gate B "
                    "unless H4 structure, recorded invalidation, or a hard sidecar also confirms"
                )
                continue
            return True, (
                f"{tf} {event_type}_{direction}@{price:g} prints against "
                f"{side_upper} thesis (close-confirmed)"
            ), True

    if m15_supported_pullback_reason:
        return False, m15_supported_pullback_reason, False

    if decision is not None and decision.invalidation_price is not None:
        snapshot = packet.get("broker_snapshot") or {}
        quotes = snapshot.get("quotes") or {}
        quote = quotes.get(pair)
        if isinstance(quote, dict):
            bid = _optional_float(quote.get("bid"))
            ask = _optional_float(quote.get("ask"))
            level = decision.invalidation_price
            tf = decision.invalidation_tf or "unspecified-TF"
            price = bid if side_upper == "LONG" else ask
            label = "bid" if side_upper == "LONG" else "ask"
            reason = invalidation_price_hit_reason(
                pair=pair,
                side=side_upper,
                current_price=price,
                invalidation_price=level,
                price_label=label,
            )
            if reason:
                technical_reason = technical_invalidation_confirmation_reason(
                    _pair_chart(packet, pair),
                    side=side_upper,
                )
                if technical_reason:
                    soft_blocker = _soft_sidecar_blocks_hard_close_authorization(
                        packet,
                        trade_id=trade_id,
                        pair=pair,
                        side=side_upper,
                    )
                    receipt_reason = f"{reason}; {technical_reason} on {tf} per receipt"
                    if soft_blocker:
                        return True, f"{receipt_reason}; {soft_blocker}", False
                    hold_conflict = _same_direction_hold_support_conflict(
                        packet,
                        trade_id=trade_id,
                        pair=pair,
                        side=side_upper,
                        evidence_label="receipt-level invalidation hit",
                    )
                    if hold_conflict:
                        return True, f"{receipt_reason}; {hold_conflict}", False
                    return True, receipt_reason, True

    sidecar_ok, sidecar_reason, sidecar_standing_authorized = _position_sidecar_close_recommended(
        packet,
        trade_id=trade_id,
        pair=pair,
        side=side_upper,
    )
    if sidecar_ok:
        return True, sidecar_reason, sidecar_standing_authorized

    if m15_soft_structural_reason:
        return True, m15_soft_structural_reason, False

    return False, "", False


def _position_sidecar_close_recommended(
    packet: dict[str, Any],
    *,
    trade_id: str | None,
    pair: str,
    side: str,
) -> tuple[bool, str, bool]:
    sidecars = packet.get("protection_sidecars")
    if not isinstance(sidecars, dict):
        return False, "", False
    recs = sidecars.get("position_close_recommendations")
    if not isinstance(recs, list):
        return False, "", False
    matched = _matching_position_close_sidecars(recs, trade_id=trade_id, pair=pair, side=side)

    if not matched:
        return False, "", False

    # A trade may appear in multiple fresh sidecars in deterministic order
    # (position_thesis before thesis_evolution). Prefer hard evidence so a soft
    # review cannot mask standing structural loss-cut authorization.
    rec = next((item for item in matched if _sidecar_close_standing_authorized(item)), matched[0])
    source = str(rec.get("source") or "position_sidecar")
    verdict = str(rec.get("verdict") or "RECOMMEND_CLOSE")
    reason = str(rec.get("reason") or "prediction no longer supports recovery")
    rec_trade = str(rec.get("trade_id") or "")
    standing_authorized = _sidecar_close_standing_authorized(rec)
    hold_conflict = _sidecar_hold_support_conflict(packet, rec)
    if standing_authorized and hold_conflict:
        standing_authorized = False
        reason = f"{reason}; {hold_conflict}"
    return (
        True,
        f"{source} {verdict} for trade {rec_trade}: {reason}",
        standing_authorized,
    )


def _sidecar_hold_support_conflict(
    packet: dict[str, Any],
    rec: dict[str, Any],
) -> str | None:
    """Downgrade close evidence when current position stack still supports hold.

    `THESIS_EXPIRED` is meant to stop decayed, unsupported theses. If separate
    fresh position sidecars still support the open side, the problem is
    position geometry/repricing, not a hard unattended loss-cut. The same
    conflict applies to recorded-invalidation sidecars unless the reason
    contains a higher-timeframe structural break; a small invalidation hit while
    current forecasts still support the open side is the 2026-06-15 USD_CAD
    loss-close failure mode.
    """

    source = str(rec.get("source") or "").strip()
    reason = str(rec.get("reason") or "")
    if _sidecar_reason_has_h4_structural_break(reason):
        return None
    upper_reason = reason.upper()
    lower_reason = reason.lower()
    if source == "thesis_evolution":
        conflict_label = (
            "THESIS_EXPIRED"
            if "THESIS_EXPIRED" in upper_reason
            else "thesis_evolution close evidence"
        )
    elif source == "position_thesis" and (
        "invalidation hit:" in lower_reason
        or "technical invalidation confirmed against" in lower_reason
    ):
        conflict_label = "position_thesis invalidation evidence"
    elif (
        source in {"position_management", "position_guardian_management"}
        and "entry thesis invalidation hit" in lower_reason
    ):
        conflict_label = f"{source} entry-invalidation evidence"
    else:
        return None
    return _same_direction_hold_support_conflict(
        packet,
        trade_id=str(rec.get("trade_id") or "") or None,
        pair=str(rec.get("pair") or ""),
        side=str(rec.get("side") or ""),
        evidence_label=conflict_label,
    )


def _same_direction_hold_support_conflict(
    packet: dict[str, Any],
    *,
    trade_id: str | None,
    pair: str,
    side: str,
    evidence_label: str,
) -> str | None:
    supported, support_reason = _close_same_direction_sidecar_support(
        packet,
        trade_id=trade_id,
        pair=pair,
        side=side,
    )
    if supported:
        return (
            f"{evidence_label} is downgraded to soft Gate A because {support_reason}; "
            "use HOLD/reprice/TP rebalance unless explicit Gate B authorizes the close"
        )
    matrix_supported, matrix_reason = _close_same_direction_matrix_support(packet, pair, side)
    h4_supported, h4_reason = _close_same_direction_h4_support(packet, pair, side)
    if not matrix_supported or not h4_supported:
        return None
    return (
        f"{evidence_label} is downgraded to soft Gate A because {matrix_reason}; {h4_reason}; "
        "use HOLD/reprice/TP rebalance unless explicit Gate B authorizes the close"
    )


def _close_same_direction_h4_support(
    packet: dict[str, Any],
    pair: str,
    side: str,
) -> tuple[bool, str]:
    side_upper = str(side or "").upper()
    if side_upper not in {"LONG", "SHORT"}:
        return False, ""
    same_direction = "UP" if side_upper == "LONG" else "DOWN"
    event = _parse_struct_events(_pair_chart_story(packet, pair)).get("H4")
    if event is None or event[1] != same_direction:
        return False, ""
    event_type, direction, price, close_confirmed = event
    confirmation = "close-confirmed" if close_confirmed else "wick-only"
    return (
        True,
        f"H4 {event_type}_{direction}@{price:g} still supports {side_upper} ({confirmation})",
    )


def _sidecar_reason_has_h4_structural_break(reason: str) -> bool:
    lowered = str(reason or "").lower()
    if "h4" not in lowered:
        return False
    return any(
        token in lowered
        for token in (
            "bos_",
            "choch_",
            "close-confirmed",
            "structural break",
            "order block",
            "ob broken",
        )
    )


def _sidecar_close_standing_authorized(rec: dict[str, Any]) -> bool:
    """Whether a fresh sidecar is strong enough for standing loss-cut auth.

    `thesis_evolution` compares the entry thesis to current broker truth and
    emits BROKEN / RECOMMEND_CLOSE only after invalidation plus technical
    confirmation. `position_management` can carry the deterministic
    PositionManager REVIEW_EXIT into this GPT CLOSE route; it is hard only when
    its own reasons are structural loss-cut reasons. `position_thesis` may also
    hard-authorize a legacy/no-ledger trade, but only when it records a
    machine-checkable invalidation hit or structural break plus multi-TF
    confirmation. Score-only, adverse-buffer-only position-thesis, soft
    position-management, and persistence reviews remain softer and need
    explicit operator Gate B.
    """
    if rec.get("gate_b_standing_authorized") is True:
        return True
    source = str(rec.get("source") or "").strip()
    verdict = str(rec.get("verdict") or "").strip().upper()
    if source == "thesis_evolution" and verdict in {"BROKEN", "RECOMMEND_CLOSE"}:
        return True
    if source == "position_thesis" and verdict == "REVIEW_CLOSE":
        reason = str(rec.get("reason") or "").lower()
        has_technical_confirmation = "technical invalidation confirmed against" in reason
        has_invalidation_hit = "invalidation hit:" in reason
        has_structural_break = _position_thesis_structural_break_text(reason)
        return has_technical_confirmation and (has_invalidation_hit or has_structural_break)
    if source in {"position_management", "position_guardian_management"} and verdict == "REVIEW_EXIT":
        reason = str(rec.get("reason") or "").lower()
        return "close-confirmed structural break" in reason or "structural ob broken" in reason
    return False


def _matching_position_close_sidecars(
    recs: Any,
    *,
    trade_id: str | None,
    pair: str,
    side: str,
) -> list[dict[str, Any]]:
    if not isinstance(recs, list):
        return []
    matched: list[dict[str, Any]] = []
    for rec in recs:
        if not isinstance(rec, dict):
            continue
        rec_trade = str(rec.get("trade_id") or "")
        if trade_id is not None and rec_trade != str(trade_id):
            continue
        if pair and str(rec.get("pair") or "") not in {"", pair}:
            continue
        if side and str(rec.get("side") or "").upper() not in {"", str(side).upper()}:
            continue
        matched.append(rec)
    return matched


def _soft_sidecar_blocks_hard_close_authorization(
    packet: dict[str, Any],
    *,
    trade_id: str | None,
    pair: str,
    side: str,
) -> str | None:
    """Keep no-ledger/entry-buffer close evidence in the soft Gate-B path.

    M15 internal structure and receipt-level `invalidation_price` can confirm a
    recorded thesis level. They must not launder a fresh soft position_thesis
    fallback (`entry thesis lacks invalidation_price` / `entry-buffer`) into
    standing hard loss-cut authorization.
    """

    sidecars = packet.get("protection_sidecars")
    recs = sidecars.get("position_close_recommendations") if isinstance(sidecars, dict) else None
    matched = _matching_position_close_sidecars(recs, trade_id=trade_id, pair=pair, side=side)
    if not matched:
        return None
    if any(_sidecar_close_standing_authorized(rec) for rec in matched):
        return None
    thesis_context = _matching_entry_thesis_close_context(
        packet,
        trade_id=trade_id,
        pair=pair,
        side=side,
    )
    if thesis_context is not None and not bool(thesis_context.get("has_recorded_invalidation_price")):
        recorded = "recorded" if bool(thesis_context.get("recorded")) else "missing"
        return (
            "matching soft close sidecar has "
            f"{recorded} entry_thesis without a recorded invalidation_price; "
            "receipt-level invalidation_price cannot convert it into standing hard Gate A"
        )
    for rec in matched:
        source = str(rec.get("source") or "").strip()
        reason = str(rec.get("reason") or "").lower()
        if source not in {"position_thesis", "position_management"}:
            continue
        if _entry_buffer_or_unrecorded_invalidation_text(reason):
            return (
                "matching soft close sidecar is entry-buffer / unrecorded-invalidation evidence; "
                "M15/receipt evidence cannot convert it into standing hard Gate A"
            )
    return None


def _matching_entry_thesis_close_context(
    packet: dict[str, Any],
    *,
    trade_id: str | None,
    pair: str,
    side: str,
) -> dict[str, Any] | None:
    sidecars = packet.get("protection_sidecars")
    context_rows = sidecars.get("entry_thesis_close_context") if isinstance(sidecars, dict) else None
    if not isinstance(context_rows, list):
        return None
    for row in context_rows:
        if not isinstance(row, dict):
            continue
        if trade_id is not None and str(row.get("trade_id") or "") != str(trade_id):
            continue
        if pair and str(row.get("pair") or "") not in {"", pair}:
            continue
        if side and str(row.get("side") or "").upper() not in {"", str(side).upper()}:
            continue
        return row
    return None


def _entry_buffer_or_unrecorded_invalidation_text(text: str) -> bool:
    lowered = str(text or "").lower()
    return any(
        token in lowered
        for token in (
            "entry-buffer",
            "entry thesis lacks invalidation_price",
            "no entry thesis",
            "adverse technical loss",
        )
    )


def _position_thesis_structural_break_text(text: str) -> bool:
    lowered = str(text).lower()
    return any(
        token in lowered
        for token in (
            "structural",
            "close-confirmed",
            "order block",
            "ob broken",
        )
    )

from quant_rabbit.analysis.chart_reader import DEFAULT_TIMEFRAMES as DEFAULT_PAIR_CHART_TIMEFRAMES
from quant_rabbit.paths import (
    DEFAULT_AI_ATTACK_ADVICE,
    DEFAULT_BROKER_INSTRUMENTS,
    DEFAULT_CAMPAIGN_PLAN,
    DEFAULT_CALENDAR_SNAPSHOT,
    DEFAULT_CAPTURE_ECONOMICS,
    DEFAULT_CONTEXT_ASSET_CHARTS,
    DEFAULT_COVERAGE_OPTIMIZATION,
    DEFAULT_COT_SNAPSHOT,
    DEFAULT_CROSS_ASSET_SNAPSHOT,
    DEFAULT_CURRENCY_STRENGTH,
    DEFAULT_DAILY_TARGET_STATE,
    DEFAULT_FLOW_SNAPSHOT,
    DEFAULT_GPT_TRADER_DECISION,
    DEFAULT_GPT_TRADER_DECISION_REPORT,
    DEFAULT_LEVELS_SNAPSHOT,
    DEFAULT_LEARNING_AUDIT,
    DEFAULT_MANUAL_MARKET_CONTEXT_AUDIT,
    DEFAULT_MARKET_CONTEXT_MATRIX,
    DEFAULT_MARKET_STATUS,
    DEFAULT_MARKET_STORY_PROFILE,
    DEFAULT_OPTION_SKEW,
    DEFAULT_OPERATOR_PRECEDENT_AUDIT,
    DEFAULT_ORDER_INTENTS,
    DEFAULT_PAIR_CHARTS,
    DEFAULT_PREDICTIVE_LIMIT_ORDERS,
    DEFAULT_SELF_IMPROVEMENT_AUDIT,
    DEFAULT_STRATEGY_PROFILE,
    DEFAULT_VERIFICATION_LEDGER,
)
from quant_rabbit.instruments import (
    DEFAULT_CONTEXT_ASSETS,
    NORMAL_SPREAD_PIPS,
    instrument_pip_factor,
)
from quant_rabbit.risk import RiskPolicy, _spread_session_multiplier_from_tag
from quant_rabbit.strategy.entry_thesis_ledger import (
    invalidation_price_hit_reason,
    technical_invalidation_confirmation_reason,
)


ALLOWED_ACTIONS = ("TRADE", "WAIT", "CANCEL_PENDING", "PROTECT", "TIGHTEN_SL", "CLOSE", "REQUEST_EVIDENCE")
ALLOWED_CONFIDENCE = ("LOW", "MEDIUM", "HIGH")
ALLOWED_METHODS = ("TREND_CONTINUATION", "RANGE_ROTATION", "BREAKOUT_FAILURE", "EVENT_RISK", "POSITION_MANAGEMENT")
ALLOWED_SPECIALIST_ROLES = ("macro_news", "indicator", "flow_levels", "risk_audit", "strategy", "portfolio_context")
OPERATOR_PRECEDENT_EVIDENCE_REF = "operator:precedent"
MANUAL_MARKET_CONTEXT_EVIDENCE_REF = "manual:market_context"
FORBIDDEN_SPECIALIST_AUTHORITY_FIELDS = (
    "action",
    "selected_lane_id",
    "selected_lane_ids",
    "cancel_order_ids",
    "units",
    "tp",
    "sl",
    "entry",
    "risk_budget_jpy",
    "daily_risk_budget_jpy",
    "per_trade_risk_budget_jpy",
    "max_loss_jpy",
    "size_multiple",
    "stage_order",
    "send_order",
    "confirm_live",
)
# Matches the CLI generate-intents breadth used by the scheduled trader. The
# verifier also keeps every LIVE_READY lane even when a smaller cap is passed,
# because the operator may cite any executable lane visible in order_intents.
DEFAULT_GPT_MAX_LANES = 56

# Maximum distinct pairs a verifier-accepted basket should cover when
# ai_attack_advice has recommended lanes across multiple pairs and the
# campaign target is open. Mirrors RiskPolicy.max_portfolio_positions; the
# gateway still re-validates portfolio risk and margin before send.
BASKET_PAIR_COVERAGE_TARGET = 4

# Rank ceiling for "primary attack" lanes when computing basket pair
# coverage. ai_attack_advice sorts recommended_now_lane_ids by descending
# score, so the top N ranks represent the highest-conviction setups for
# the cycle. Pairs whose first appearance in the advised list is below
# this rank are treated as low-conviction repair candidates rather than
# primary-basket coverage requirements; the rank gap is the deterministic
# conviction gate that satisfies AGENT_CONTRACT §5–§6 without forcing the
# trader to paste boilerplate skip-rationale per lower-ranked pair. This
# keeps the bot-grinding defense (single low-conviction lane spam) while
# unblocking high-conviction concentrated attacks (per
# feedback_high_conviction_execution.md).
PRIMARY_ATTACK_RANK_CEILING = 4

# The scheduled trader cadence is approximately one operator decision every
# 20 minutes. This is an operational receipt horizon, not a market threshold,
# JPY cap, pip distance, or reward/risk multiplier. If scheduler cadence
# changes, replace this with scheduler config rather than tuning it from trade
# outcomes.
TRADER_DECISION_HORIZON_MINUTES = 20
ENTRY_DECISION_HORIZON_ACTIONS = ("TRADE", "WAIT", "REQUEST_EVIDENCE")
TWENTY_MINUTE_PLAN_TEXT_FIELDS = (
    "primary_path",
    "failure_path",
    "entry_or_hold_trigger",
    "invalidation_or_cancel_trigger",
    "counterargument",
    "next_cycle_check",
)
SESSION_ONLY_WAIT_PATTERN = re.compile(
    r"(WAIT|PATIENCE|STAY\s+FLAT|HOLD).{0,100}"
    r"(LONDON|NEW\s+YORK|\bNY\b|ASIA|ASIAN|TOKYO|OFF[\s_-]*HOURS|SESSION|KILLZONE)"
    r"|"
    r"(QUIET|THIN|LOW[\s_-]*LIQUIDITY).{0,60}"
    r"(SESSION|ASIA|ASIAN|TOKYO|OFF[\s_-]*HOURS|KILLZONE)"
    r"|"
    r"(SESSION|ASIA|ASIAN|TOKYO|OFF[\s_-]*HOURS|KILLZONE).{0,60}"
    r"(QUIET|THIN|LOW[\s_-]*LIQUIDITY|WAIT|PATIENCE|STAY\s+FLAT)",
    re.IGNORECASE,
)
CONCRETE_WAIT_GATE_PATTERN = re.compile(
    r"\b("
    r"SPREAD|FORECAST|NEWS|EVENT|CPI|FOMC|NFP|RATE|CONFLICT|INVALIDATION|"
    r"MARGIN|REWARD|RR|LIVE_READY|LIVE\s+READY|BOS|CHOCH|ATR|"
    r"VOLATILITY|STRUCTURE|SHELF|BREAK|BROKEN|SUPPORT|RESISTANCE|RETEST|"
    r"PENDING|CAPACITY|TP|SL"
    r")\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class GPTTraderDecision:
    generated_at_utc: str | None
    action: str
    selected_lane_id: str | None
    selected_lane_ids: tuple[str, ...]
    cancel_order_ids: tuple[str, ...]
    confidence: str
    thesis: str
    method: str
    narrative: str
    chart_story: str
    invalidation: str
    rejected_alternatives: tuple[str, ...]
    risk_notes: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    operator_summary: str
    twenty_minute_plan: dict[str, Any] | None = None
    # Operator-directed market close on existing trader-owned positions.
    # Used only with action="CLOSE". A loss-cut and a new entry must still
    # have separate receipts: automation ends the close cycle, then the next
    # scheduled cycle must refresh broker truth and require a fresh verified
    # TRADE receipt before any re-entry.
    close_trade_ids: tuple[str, ...] = ()
    strategy_reviews: tuple[dict[str, Any], ...] = ()
    specialist_reviews: tuple[dict[str, Any], ...] = ()
    # CLOSE-action discipline fields (added 2026-05-12, see
    # `feedback_no_unilateral_close.md`; Gate B split 2026-06-04). A CLOSE
    # receipt must pass Gate A market evidence plus the applicable Gate B:
    # hard Gate A carries standing loss-cut authorization, while soft Gate A
    # requires explicit env/token authorization. The `operator_close_authorized`
    # field remains audit text only and is not accepted as authorization,
    # because the trader can set fields in its own receipt.
    invalidation_price: float | None = None
    invalidation_tf: str | None = None
    operator_close_authorized: bool = False


@dataclass(frozen=True)
class VerificationIssue:
    code: str
    message: str
    severity: str = "BLOCK"


@dataclass(frozen=True)
class VerificationResult:
    allowed: bool
    issues: tuple[VerificationIssue, ...]


@dataclass(frozen=True)
class GPTTraderSummary:
    status: str
    output_path: Path
    report_path: Path
    action: str | None
    selected_lane_id: str | None
    selected_lane_ids: tuple[str, ...]
    cancel_order_ids: tuple[str, ...]
    allowed: bool
    issues: int
    close_trade_ids: tuple[str, ...] = ()


class TraderModelProvider(Protocol):
    def decide(self, input_packet: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]: ...


class StaticTraderProvider:
    def __init__(self, decision: dict[str, Any], *, source_path: Path | None = None) -> None:
        self.decision = decision
        self.source_path = source_path

    def decide(self, input_packet: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
        return dict(self.decision)


class GPTTraderBrain:
    """Build a broker-truth packet and verify a Codex-created decision receipt."""

    def __init__(
        self,
        *,
        provider: TraderModelProvider | None = None,
        intents_path: Path = DEFAULT_ORDER_INTENTS,
        campaign_plan_path: Path = DEFAULT_CAMPAIGN_PLAN,
        strategy_profile_path: Path = DEFAULT_STRATEGY_PROFILE,
        market_story_profile_path: Path = DEFAULT_MARKET_STORY_PROFILE,
        market_status_path: Path = DEFAULT_MARKET_STATUS,
        target_state_path: Path = DEFAULT_DAILY_TARGET_STATE,
        pair_charts_path: Path = DEFAULT_PAIR_CHARTS,
        context_asset_charts_path: Path = DEFAULT_CONTEXT_ASSET_CHARTS,
        broker_instruments_path: Path = DEFAULT_BROKER_INSTRUMENTS,
        cross_asset_path: Path = DEFAULT_CROSS_ASSET_SNAPSHOT,
        flow_path: Path = DEFAULT_FLOW_SNAPSHOT,
        currency_strength_path: Path = DEFAULT_CURRENCY_STRENGTH,
        levels_path: Path = DEFAULT_LEVELS_SNAPSHOT,
        market_context_matrix_path: Path = DEFAULT_MARKET_CONTEXT_MATRIX,
        calendar_path: Path = DEFAULT_CALENDAR_SNAPSHOT,
        cot_path: Path = DEFAULT_COT_SNAPSHOT,
        option_skew_path: Path = DEFAULT_OPTION_SKEW,
        attack_advice_path: Path = DEFAULT_AI_ATTACK_ADVICE,
        capture_economics_path: Path = DEFAULT_CAPTURE_ECONOMICS,
        coverage_optimization_path: Path = DEFAULT_COVERAGE_OPTIMIZATION,
        learning_audit_path: Path = DEFAULT_LEARNING_AUDIT,
        verification_ledger_path: Path = DEFAULT_VERIFICATION_LEDGER,
        self_improvement_audit_path: Path = DEFAULT_SELF_IMPROVEMENT_AUDIT,
        operator_precedent_path: Path = DEFAULT_OPERATOR_PRECEDENT_AUDIT,
        manual_market_context_path: Path = DEFAULT_MANUAL_MARKET_CONTEXT_AUDIT,
        predictive_limits_path: Path = DEFAULT_PREDICTIVE_LIMIT_ORDERS,
        output_path: Path = DEFAULT_GPT_TRADER_DECISION,
        report_path: Path = DEFAULT_GPT_TRADER_DECISION_REPORT,
        max_lanes: int = DEFAULT_GPT_MAX_LANES,
    ) -> None:
        self.provider = provider
        self.intents_path = intents_path
        self.campaign_plan_path = campaign_plan_path
        self.strategy_profile_path = strategy_profile_path
        self.market_story_profile_path = market_story_profile_path
        self.market_status_path = market_status_path
        self.target_state_path = target_state_path
        self.pair_charts_path = pair_charts_path
        self.context_asset_charts_path = context_asset_charts_path
        self.broker_instruments_path = broker_instruments_path
        self.cross_asset_path = cross_asset_path
        self.flow_path = flow_path
        self.currency_strength_path = currency_strength_path
        self.levels_path = levels_path
        self.market_context_matrix_path = market_context_matrix_path
        self.calendar_path = calendar_path
        self.cot_path = cot_path
        self.option_skew_path = option_skew_path
        self.attack_advice_path = attack_advice_path
        self.capture_economics_path = capture_economics_path
        self.coverage_optimization_path = coverage_optimization_path
        self.learning_audit_path = learning_audit_path
        self.verification_ledger_path = verification_ledger_path
        self.self_improvement_audit_path = self_improvement_audit_path
        self.operator_precedent_path = operator_precedent_path
        self.manual_market_context_path = manual_market_context_path
        self.predictive_limits_path = predictive_limits_path
        self.output_path = output_path
        self.report_path = report_path
        self.max_lanes = max_lanes

    def run(self, *, snapshot_path: Path) -> GPTTraderSummary:
        generated_at = datetime.now(timezone.utc).isoformat()
        packet = self._input_packet(snapshot_path)
        if self.provider is None:
            raise RuntimeError("Codex GPT verifier requires a decision response JSON")
        raw_decision = self.provider.decide(packet, GPT_TRADER_SCHEMA)
        decision = _decision_from_payload(raw_decision)
        verification = DecisionVerifier(packet).verify(decision)
        status = "ACCEPTED" if verification.allowed else "REJECTED"
        result = {
            "generated_at_utc": generated_at,
            "status": status,
            "decision": asdict(decision),
            "verification_issues": [asdict(issue) for issue in verification.issues],
            "input_packet": packet,
        }
        self._write_result(result)
        self._write_report(result)
        return GPTTraderSummary(
            status=status,
            output_path=self.output_path,
            report_path=self.report_path,
            action=decision.action,
            selected_lane_id=decision.selected_lane_id,
            selected_lane_ids=decision.selected_lane_ids,
            cancel_order_ids=decision.cancel_order_ids,
            allowed=verification.allowed,
            issues=len(verification.issues),
            close_trade_ids=decision.close_trade_ids,
        )

    def _input_packet(self, snapshot_path: Path) -> dict[str, Any]:
        snapshot = _load_json(snapshot_path)
        intents = _load_json(self.intents_path)
        campaign = _load_json(self.campaign_plan_path)
        strategy = _load_json(self.strategy_profile_path)
        story = _load_json(self.market_story_profile_path)
        market_status = _load_optional_json(self.market_status_path)
        target = _load_json(self.target_state_path) if self.target_state_path.exists() else {}
        lanes = _lane_packet(intents, campaign, strategy, story, max_lanes=self.max_lanes)
        attack_advice = _load_optional_json(self.attack_advice_path)
        capture_economics = _load_optional_json(self.capture_economics_path)
        coverage_optimization = _load_optional_json(self.coverage_optimization_path)
        learning_audit = _load_optional_json(self.learning_audit_path)
        verification_ledger = _load_optional_json(self.verification_ledger_path)
        self_improvement_audit = _load_optional_json(self.self_improvement_audit_path)
        operator_precedent = _load_optional_json(self.operator_precedent_path)
        manual_market_context = _load_optional_json(self.manual_market_context_path)
        predictive_limits = _load_optional_json(self.predictive_limits_path)
        market_context_matrix = _load_optional_json(self.market_context_matrix_path)
        option_skew = _load_optional_json(self.option_skew_path)
        pairs = _pairs_from_lanes_and_positions(lanes, snapshot)
        currencies = _currencies_from_pairs(pairs)
        refs = _allowed_refs(
            snapshot=snapshot,
            target=target,
            lanes=lanes,
            attack_advice=attack_advice,
            capture_economics=capture_economics,
            coverage_optimization=coverage_optimization,
            learning_audit=learning_audit,
            verification_ledger=verification_ledger,
            self_improvement_audit=self_improvement_audit,
            operator_precedent=operator_precedent,
            manual_market_context=manual_market_context,
            predictive_limits=predictive_limits,
            market_status=market_status,
            market_context_matrix=market_context_matrix,
            option_skew=option_skew,
        )
        return {
            "contract": {
                "allowed_actions": list(ALLOWED_ACTIONS),
                "entry_decision_actions": list(ENTRY_DECISION_HORIZON_ACTIONS + ("CANCEL_PENDING",)),
                "trade_requires_live_ready_lane": True,
                "trade_may_select_multiple_live_ready_lanes": True,
                "entry_decisions_require_twenty_minute_plan": True,
                "decision_horizon_minutes": TRADER_DECISION_HORIZON_MINUTES,
                "pending_entries_are_basket_counted_by_gateway": True,
                "protected_trader_position_adds_require_portfolio_validation": True,
                "model_output_is_advisory_until_verified": True,
                "strategy_reviews_must_use_lane_id_not_desk_alias": True,
                "predictive_limits_are_advisory_timing_evidence": True,
                "tp_rebalance_sidecar_blocks_wait": True,
                "entry_thesis_blocker_blocks_trade_and_wait": True,
                "learning_audit_blocks_unsafe_learning_influence": True,
                "verification_ledger_is_read_only_structured_evidence": True,
                "self_improvement_p0_blocks_trade": True,
                "market_status_is_authoritative_calendar_evidence": True,
                "coverage_optimization_is_read_only_gap_evidence": True,
                "operator_precedent_is_advisory_only": True,
                "manual_market_context_gates_only_precedent_usage": True,
                "soft_close_advisory_non_blocking": (
                    "position_close_recommendations include blocks_non_close_actions; "
                    "when false, do not choose CLOSE from that advisory on the entry branch "
                    "unless explicit operator Gate B is present"
                ),
            },
            "artifact_timestamps": {
                "order_intents_generated_at_utc": intents.get("generated_at_utc"),
                "market_context_matrix_generated_at_utc": (
                    market_context_matrix.get("generated_at_utc")
                    if isinstance(market_context_matrix, dict)
                    else None
                ),
                "operator_precedent_generated_at_utc": (
                    operator_precedent.get("generated_at_utc")
                    if isinstance(operator_precedent, dict)
                    else None
                ),
                "manual_market_context_generated_at_utc": (
                    manual_market_context.get("generated_at_utc")
                    if isinstance(manual_market_context, dict)
                    else None
                ),
            },
            "broker_snapshot": _snapshot_packet(snapshot),
            "daily_target": _target_packet(target),
            "lanes": lanes,
            "ai_attack_advice": _attack_advice_packet(attack_advice),
            "capture_economics": _capture_economics_packet(capture_economics),
            "coverage_optimization": _coverage_optimization_packet(coverage_optimization),
            "learning_audit": _learning_audit_packet(learning_audit),
            "verification_ledger": _verification_ledger_packet(verification_ledger),
            "self_improvement_audit": _self_improvement_audit_packet(self_improvement_audit),
            "operator_precedent": _operator_precedent_packet(operator_precedent),
            "manual_market_context": _manual_market_context_packet(manual_market_context),
            "predictive_limits": _predictive_limits_packet(predictive_limits, pairs=pairs),
            "market_status": _market_status_packet(market_status),
            "protection_sidecars": _protection_sidecars_packet(
                snapshot=snapshot,
                snapshot_path=snapshot_path,
                pair_charts_path=self.pair_charts_path,
            ),
            "market_context": _market_context_packet(
                pairs=pairs,
                currencies=currencies,
                pair_charts_path=self.pair_charts_path,
                context_asset_charts_path=self.context_asset_charts_path,
                broker_instruments_path=self.broker_instruments_path,
                cross_asset_path=self.cross_asset_path,
                flow_path=self.flow_path,
                currency_strength_path=self.currency_strength_path,
                levels_path=self.levels_path,
                market_context_matrix_path=self.market_context_matrix_path,
                calendar_path=self.calendar_path,
                cot_path=self.cot_path,
                option_skew_path=self.option_skew_path,
            ),
            "allowed_evidence_refs": refs,
        }

    def _write_result(self, result: dict[str, Any]) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    def _write_report(self, result: dict[str, Any]) -> None:
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        decision = result["decision"]
        lines = [
            "# GPT Trader Decision Report",
            "",
            f"- Generated at UTC: `{result['generated_at_utc']}`",
            f"- Status: `{result['status']}`",
            f"- Action: `{decision.get('action')}`",
            f"- Selected lane: `{decision.get('selected_lane_id')}`",
            f"- Selected basket lanes: `{', '.join(decision.get('selected_lane_ids') or []) or 'none'}`",
            f"- Cancel order ids: `{', '.join(decision.get('cancel_order_ids') or []) or 'none'}`",
            f"- Confidence: `{decision.get('confidence')}`",
            f"- 20m plan: `{decision.get('twenty_minute_plan') or 'missing'}`",
            f"- Specialist reviews: `{len(decision.get('specialist_reviews') or [])}`",
            f"- Operator summary: {decision.get('operator_summary')}",
            "",
            "## Verification Issues",
            "",
        ]
        issues = result.get("verification_issues", [])
        if issues:
            for issue in issues:
                lines.append(f"- `{issue['severity']}` {issue['code']}: {issue['message']}")
        else:
            lines.append("- none")
        lines.extend(
            [
                "",
                "## Decision Contract",
                "",
                "- GPT is the discretionary reasoning layer; deterministic verification remains the execution gate.",
                "- `TRADE` requires known `LIVE_READY` lane(s); pending entries are counted by gateway basket validation.",
                "- `TRADE`/`CANCEL_PENDING` cancel ids must be current trader-owned pending entry orders from broker truth.",
                "- Current `ai_attack_advice` recommendations make generic WAIT invalid while the daily target is open, but never grant live permission.",
                "- Learning may only rank already-live-ready lanes. Any learning-influenced selected lane must be covered by a non-blocked `learning_audit` packet and cite `learning:audit` plus `learning:lane:<lane_id>`.",
                "- `TRADE`, `WAIT`, and `REQUEST_EVIDENCE` receipts must include `twenty_minute_plan`: the next-20-minute primary path, failure path, trigger, invalidation/cancel trigger, strongest counterargument, next-cycle check, and known packet refs. This is a receipt-depth gate, not a new market-risk gate.",
                "- `market_status` is deterministic calendar/session evidence only; broker truth still decides prices, positions, and tradability.",
                "- A deterministic `tp-rebalance` sidecar requirement makes WAIT / REQUEST_EVIDENCE invalid until the sidecar is run.",
                "- A deterministic entry-thesis blocker makes TRADE / WAIT invalid until the unverifiable active position is repaired or reviewed.",
                "- Any self-improvement P0 blocks new `TRADE` receipts until the named blocker is repaired or the trader route explicitly justifies the exception.",
                "- The 2025 operator precedent is advisory only. A `TRADE` that cites `operator:precedent` must also cite `manual:market_context`, at least one selected lane must match the current operator-precedent aligned lane set, and that selected lane must not conflict with the bounded manual technical replay buckets; otherwise the receipt must use current deterministic edge instead of precedent-based aggression.",
                "- Evidence refs must come from the input packet; invented refs reject the decision.",
                "- `CLOSE` requires Gate A plus the applicable Gate B. Hard Gate A (H4 close-confirmed BOS/CHOCH against side, buffered invalidation_price hit with technical confirmation, fresh thesis_evolution BROKEN/RECOMMEND_CLOSE, structural position_management / position_guardian_management REVIEW_EXIT, or position_thesis invalidation-hit/structural-break evidence with multi-TF confirmation) carries standing loss-cut authorization only when it has not been downgraded by fresh same-direction HOLD/EXTEND sidecars. M15 structure is Gate A evidence but not unattended hard Gate B unless H4 / recorded invalidation / hard sidecar also confirms; M15 internal structure or receipt-level `invalidation_price` cannot harden a matching soft entry-buffer / unrecorded-invalidation sidecar. `protection_sidecars.position_close_recommendations[].blocks_non_close_actions=false` means the sidecar is advisory for entry routing: do not write CLOSE merely to test the verifier; evaluate current LIVE_READY entries unless explicit operator Gate B is present. Softer Gate A still needs `QR_OPERATOR_CLOSE_OVERRIDE=1` or a fresh `data/.operator_close_token` when the trader chooses CLOSE. If the same-direction market stack still supports the open position, treat it as TP rebalance / HOLD / profit-side partial / ADD geometry, not loss-side CLOSE plus same-direction re-entry. `TRADE` must not include `close_trade_ids`; automation ends the close cycle, then the next scheduled cycle must refresh broker truth, reprice intents, and require a separate verified `TRADE` receipt. The receipt's `operator_close_authorized` field is advisory only. See AGENT_CONTRACT §10.",
            ]
        )
        self.report_path.write_text("\n".join(lines) + "\n")


class DecisionVerifier:
    def __init__(self, input_packet: dict[str, Any]) -> None:
        self.packet = input_packet
        self.lanes = {str(lane["lane_id"]): lane for lane in input_packet.get("lanes", [])}
        self.allowed_refs = set(str(ref) for ref in input_packet.get("allowed_evidence_refs", []))

    def verify(self, decision: GPTTraderDecision) -> VerificationResult:
        issues: list[VerificationIssue] = []
        self._verify_decision_freshness(decision, issues)
        if decision.action not in ALLOWED_ACTIONS:
            issues.append(VerificationIssue("BAD_ACTION", f"unsupported action {decision.action!r}"))
        if decision.confidence not in ALLOWED_CONFIDENCE:
            issues.append(VerificationIssue("BAD_CONFIDENCE", f"unsupported confidence {decision.confidence!r}"))
        if decision.method not in ALLOWED_METHODS:
            issues.append(VerificationIssue("BAD_METHOD", f"unsupported method {decision.method!r}"))
        if not decision.evidence_refs:
            issues.append(VerificationIssue("MISSING_EVIDENCE_REFS", "decision must cite packet evidence refs"))
        unknown_refs = sorted(set(decision.evidence_refs) - self.allowed_refs)
        if unknown_refs:
            issues.append(VerificationIssue("UNKNOWN_EVIDENCE_REF", f"unknown evidence refs: {', '.join(unknown_refs)}"))
        self._verify_strategy_reviews(decision, issues)
        self._verify_specialist_reviews(decision, issues)
        self._verify_twenty_minute_plan(decision, issues)

        broker = self.packet.get("broker_snapshot", {})
        positions = int(broker.get("positions") or 0)
        selected_lane_ids = _selected_trade_lane_ids(decision)
        primary_lane_id = decision.selected_lane_id or (selected_lane_ids[0] if selected_lane_ids else None)
        tradeable_lanes = _tradeable_live_ready_lanes(self.packet)
        attack_lane_ids = _attack_recommended_tradeable_lane_ids(self.packet, tradeable_lanes)
        exposure_blockers = _trade_exposure_blockers(self.packet)
        entry_thesis_blockers = _entry_thesis_sidecar_reasons(self.packet)
        position_close_reasons = _position_close_sidecar_reasons(self.packet)
        self_improvement_trade_blockers = _self_improvement_trade_blockers(
            self.packet,
            decision_generated_at_utc=decision.generated_at_utc,
        )
        self_improvement_entry_blockers = _self_improvement_trade_blockers(
            self.packet,
            decision_generated_at_utc=decision.generated_at_utc,
            include_decision_history_stale=False,
        )

        if decision.action == "TRADE":
            if not selected_lane_ids:
                issues.append(VerificationIssue("LANE_REQUIRED", "TRADE requires selected_lane_id or selected_lane_ids"))
            if decision.close_trade_ids:
                issues.append(
                    VerificationIssue(
                        "CLOSE_REENTRY_SAME_RECEIPT",
                        "TRADE must not include close_trade_ids. Loss-cut first with action=CLOSE, then "
                        "end the close cycle, rerun broker-snapshot / intents on the next scheduled cycle, and re-enter only from a "
                        "separate verified TRADE receipt if a fresh LIVE_READY lane survives.",
                    )
                )
            if position_close_reasons:
                _append_position_close_required_issue(
                    issues,
                    packet=self.packet,
                    action="TRADE",
                    reasons=position_close_reasons,
                )
            if exposure_blockers:
                issues.append(VerificationIssue("EXPOSURE_BLOCKS_TRADE", "; ".join(exposure_blockers[:3])))
            if entry_thesis_blockers:
                issues.append(
                    VerificationIssue(
                        "ENTRY_THESIS_REPAIR_REQUIRED",
                        "TRADE rejected while active trader position(s) have unverifiable entry thesis: "
                        + "; ".join(entry_thesis_blockers[:3]),
                    )
                )
            if self_improvement_trade_blockers:
                issues.append(
                    VerificationIssue(
                        "SELF_IMPROVEMENT_P0_BLOCKS_TRADE",
                        "TRADE rejected while self-improvement audit carries P0 blocker(s): "
                        + "; ".join(self_improvement_trade_blockers[:3]),
                    )
                )
            issues.extend(_learning_audit_trade_issues(self.packet, selected_lane_ids, decision.evidence_refs))
            issues.extend(_manual_precedent_trade_issues(self.packet, selected_lane_ids, decision.evidence_refs))
            if _target_requires_entry(self.packet) and attack_lane_ids and not exposure_blockers:
                selected_attack_lanes = [lane_id for lane_id in selected_lane_ids if lane_id in attack_lane_ids]
                if not selected_attack_lanes:
                    issues.append(
                        VerificationIssue(
                            "ATTACK_ADVICE_IGNORED",
                            "ai_attack_advice recommends current tradeable LIVE_READY lane(s); the selected "
                            "basket must include at least one recommended lane or the advice must be regenerated: "
                            f"{', '.join(attack_lane_ids[:3])}",
                        )
                    )
                else:
                    priority_lane_id = attack_lane_ids[0]
                    if priority_lane_id not in selected_lane_ids:
                        issues.append(
                            VerificationIssue(
                                "ATTACK_PRIORITY_SKIPPED",
                                "ai_attack_advice ranks current tradeable LIVE_READY lanes in execution order; "
                                "the selected basket must include the first-ranked lane instead of skipping "
                                f"to lower-ranked advice: {priority_lane_id}",
                            )
                        )
                    advised_pairs = []
                    for advised_lane_id in attack_lane_ids:
                        pair = _pair_from_lane_id(advised_lane_id)
                        if pair and pair not in advised_pairs:
                            advised_pairs.append(pair)
                    # Restrict primary basket coverage to pairs whose top-ranked
                    # advised lane sits within the rank ceiling. attack_lane_ids
                    # is already sorted by descending score, so a pair whose
                    # first appearance is rank > ceiling has been ranked below
                    # PRIMARY_ATTACK_RANK_CEILING higher-conviction lanes — the
                    # rank gap itself is the deterministic conviction gate, and
                    # bot-grinding the lower-ranked pair would defeat the very
                    # ranking ai_attack_advice exists to express.
                    primary_advised_pairs: list[str] = []
                    for rank, advised_lane_id in enumerate(attack_lane_ids):
                        if rank >= PRIMARY_ATTACK_RANK_CEILING:
                            break
                        pair = _pair_from_lane_id(advised_lane_id)
                        if pair and pair not in primary_advised_pairs:
                            primary_advised_pairs.append(pair)
                    selected_pairs: set[str] = set()
                    for chosen_lane_id in selected_lane_ids:
                        pair = _pair_from_lane_id(chosen_lane_id)
                        if pair:
                            selected_pairs.add(pair)
                    expected_basket_pairs = min(
                        len(primary_advised_pairs),
                        BASKET_PAIR_COVERAGE_TARGET,
                    )
                    if len(selected_pairs) < expected_basket_pairs:
                        skipped_pairs = [
                            pair for pair in primary_advised_pairs if pair not in selected_pairs
                        ][:expected_basket_pairs]
                        issues.append(
                            VerificationIssue(
                                "BASKET_PAIR_COVERAGE_INCOMPLETE",
                                "ai_attack_advice recommends top-ranked tradeable LIVE_READY lanes across "
                                f"{len(primary_advised_pairs)} primary pair(s) "
                                f"({', '.join(primary_advised_pairs)}); basket only covers "
                                f"{len(selected_pairs) or 'no'} pair(s) "
                                f"({', '.join(sorted(selected_pairs)) or 'none'}). Per AGENT_CONTRACT "
                                "§5–§6 campaign exposure occupancy: include one lane per primary "
                                "advised pair (those whose top-ranked lane is within rank "
                                f"{PRIMARY_ATTACK_RANK_CEILING}), or cite a named deterministic gate "
                                f"per skipped primary pair in risk_notes: {', '.join(skipped_pairs)}. "
                                "Pairs whose top advised lane ranks lower than the ceiling are gated "
                                "by the rank/conviction gap itself.",
                                severity="WARN",
                            )
                        )
                    if "attack:advice" not in decision.evidence_refs:
                        issues.append(
                            VerificationIssue(
                                "ATTACK_ADVICE_EVIDENCE_MISSING",
                                "TRADE selecting an ai_attack_advice recommended lane must cite attack:advice",
                            )
                        )
                    for lane_id in selected_attack_lanes:
                        if f"attack:lane:{lane_id}" not in decision.evidence_refs:
                            issues.append(
                                VerificationIssue(
                                    "ATTACK_ADVICE_LANE_EVIDENCE_MISSING",
                                    f"TRADE selecting recommended lane {lane_id} must cite attack:lane:{lane_id}",
                                )
                            )
            if decision.selected_lane_id and decision.selected_lane_id not in selected_lane_ids:
                issues.append(
                    VerificationIssue(
                        "PRIMARY_LANE_NOT_IN_BASKET",
                        "selected_lane_id must also appear in selected_lane_ids",
                    )
                )
            for selected_lane_id in selected_lane_ids:
                selected_lane = self.lanes.get(selected_lane_id)
                if selected_lane is None:
                    issues.append(VerificationIssue("UNKNOWN_LANE", f"selected lane is not in packet: {selected_lane_id}"))
                    continue
                if selected_lane.get("status") != "LIVE_READY":
                    issues.append(VerificationIssue("LANE_NOT_LIVE_READY", f"{selected_lane_id} status is {selected_lane.get('status')}"))
                if selected_lane_id == primary_lane_id and selected_lane.get("method") != decision.method:
                    issues.append(VerificationIssue("METHOD_MISMATCH", "decision method does not match selected primary lane"))
                if selected_lane.get("risk_blockers") or selected_lane.get("strategy_blockers") or selected_lane.get("live_blockers"):
                    issues.append(VerificationIssue("LANE_HAS_BLOCKERS", f"{selected_lane_id} still carries blockers"))
                forecast_issue = _lane_forecast_direction_issue(selected_lane)
                if forecast_issue is not None:
                    issues.append(forecast_issue)
                if str(selected_lane.get("evidence_ref") or "") not in decision.evidence_refs:
                    issues.append(
                        VerificationIssue(
                            "SELECTED_LANE_EVIDENCE_MISSING",
                            f"selected lane {selected_lane_id} must be cited in evidence_refs",
                        )
                    )
            for field_name, value in (
                ("thesis", decision.thesis),
                ("narrative", decision.narrative),
                ("chart_story", decision.chart_story),
                ("invalidation", decision.invalidation),
            ):
                if not value.strip():
                    issues.append(VerificationIssue("INCOMPLETE_TRADE_DECISION", f"TRADE missing {field_name}"))
            self._verify_cancel_order_ids(decision, issues, action="TRADE")
        elif decision.action in {"WAIT", "REQUEST_EVIDENCE"}:
            if decision.selected_lane_id is not None:
                issues.append(VerificationIssue("WAIT_SELECTED_LANE", f"{decision.action} must not select a lane"))
            if decision.action == "WAIT" and _wait_is_session_only(decision):
                issues.append(
                    VerificationIssue(
                        "SESSION_ONLY_WAIT_REJECTED",
                        "WAIT cannot be justified by time-of-day, quiet-session, or London/NY timing alone; "
                        "cite a current spread, forecast, structure, event, broker-truth, close, or risk gate.",
                    )
                )
            if decision.action == "WAIT" and entry_thesis_blockers:
                issues.append(
                    VerificationIssue(
                        "ENTRY_THESIS_REPAIR_REQUIRED",
                        "WAIT rejected while active trader position(s) have unverifiable entry thesis: "
                        + "; ".join(entry_thesis_blockers[:3]),
                    )
                )
            tp_rebalance_reasons = _tp_rebalance_sidecar_reasons(self.packet)
            if tp_rebalance_reasons:
                issues.append(
                    VerificationIssue(
                        "TP_REBALANCE_REQUIRED",
                        "WAIT / REQUEST_EVIDENCE cannot complete while deterministic tp-rebalance has "
                        f"executable adjustment(s): {tp_rebalance_reasons[0]}",
                    )
                )
            if position_close_reasons:
                _append_position_close_required_issue(
                    issues,
                    packet=self.packet,
                    action=decision.action,
                    reasons=position_close_reasons,
                )
            if (
                not position_close_reasons
                and _target_requires_entry(self.packet)
                and not exposure_blockers
                and not self_improvement_entry_blockers
                and attack_lane_ids
            ):
                issues.append(
                    VerificationIssue(
                        "ATTACK_ADVICE_REQUIRES_TRADE",
                        "ai_attack_advice recommends current tradeable LIVE_READY lane(s) while the daily target "
                        "is still open. Protected trader exposure is not a no-trade gate; choose TRADE or rerun "
                        f"the advice after a named hard blocker fires: {', '.join(attack_lane_ids[:3])}",
                    )
                )
            if (
                not position_close_reasons
                and _target_requires_entry(self.packet)
                and not exposure_blockers
                and not self_improvement_entry_blockers
                and tradeable_lanes
            ):
                if not _trader_exposure_present(self.packet):
                    issues.append(
                        VerificationIssue(
                            "CAMPAIGN_EXPOSURE_REQUIRED",
                            "daily target is still open, no trader-owned position or pending entry is active, "
                            "and tradeable LIVE_READY lanes exist; choose TRADE instead of leaving the "
                            f"campaign flat: {', '.join(tradeable_lanes[:3])}",
                        )
                    )
                cited_live_ready = _cited_live_ready_lanes(decision, tradeable_lanes)
                if decision.action == "REQUEST_EVIDENCE":
                    issues.append(
                        VerificationIssue(
                            "REQUEST_EVIDENCE_WITH_LIVE_READY_LANES",
                            "REQUEST_EVIDENCE is stale or contradictory because the packet already contains "
                            f"tradeable LIVE_READY lanes: {', '.join(tradeable_lanes[:3])}",
                        )
                    )
                elif not cited_live_ready:
                    issues.append(
                        VerificationIssue(
                            "WAIT_MISSING_LIVE_READY_REJECTION",
                            "WAIT must cite at least one current LIVE_READY lane evidence ref when clean "
                            "tradeable lanes exist and the daily target is still open",
                        )
                    )
        elif decision.action == "CANCEL_PENDING":
            if decision.selected_lane_id is not None:
                issues.append(VerificationIssue("CANCEL_SELECTED_LANE", "CANCEL_PENDING must not select a trade lane"))
            if not _pending_entry_order_ids(self.packet):
                issues.append(VerificationIssue("NO_PENDING_ENTRY", "CANCEL_PENDING requires a pending entry order"))
            if not decision.cancel_order_ids:
                issues.append(
                    VerificationIssue(
                        "MISSING_CANCEL_ORDER_IDS",
                        "CANCEL_PENDING must name the pending entry order ids to cancel",
                    )
                )
            self._verify_cancel_order_ids(decision, issues, action="CANCEL_PENDING")
            if (
                _target_requires_entry(self.packet)
                and not position_close_reasons
                and not exposure_blockers
                and tradeable_lanes
            ):
                issues.append(
                    VerificationIssue(
                        "CANCEL_PENDING_WITH_LIVE_READY_LANES",
                        "CANCEL_PENDING is stale or contradictory because the packet already contains "
                        "tradeable LIVE_READY lane(s). If pending entries must be retired while the daily "
                        "target is still open, choose TRADE with selected lane(s) and optional cancel_order_ids "
                        f"in the same receipt: {', '.join(tradeable_lanes[:3])}",
                    )
                )
        elif decision.action in {"PROTECT", "TIGHTEN_SL", "CLOSE"}:
            if positions <= 0:
                issues.append(VerificationIssue("NO_OPEN_POSITION", f"{decision.action} requires an open position"))
            if decision.action in {"PROTECT", "TIGHTEN_SL"} and position_close_reasons:
                _append_position_close_required_issue(
                    issues,
                    packet=self.packet,
                    action=decision.action,
                    reasons=position_close_reasons,
                )
            if decision.action == "CLOSE":
                soft_advisory_reasons = _soft_nonblocking_close_advisory_reasons(
                    self.packet,
                    decision.close_trade_ids,
                )
                if (
                    soft_advisory_reasons
                    and _target_requires_entry(self.packet)
                    and not self_improvement_entry_blockers
                    and tradeable_lanes
                ):
                    issues.append(
                        VerificationIssue(
                            "SOFT_CLOSE_ADVISORY_DOES_NOT_PREEMPT_ENTRY",
                            "CLOSE selected a non-blocking soft close advisory while the daily target is open "
                            "and tradeable LIVE_READY lane(s) exist. Treat the advisory as HOLD/reprice/TP "
                            "monitoring unless explicit operator Gate B is present; write an entry-branch "
                            f"TRADE/CANCEL/WAIT receipt instead. Soft advisory: {'; '.join(soft_advisory_reasons[:3])}. "
                            f"Tradeable lane(s): {', '.join(tradeable_lanes[:3])}",
                        )
                    )
                self._verify_close_trade_ids(decision, issues)
                self._verify_close_discipline(decision, issues)
        # A TRADE receipt may not execute close+reentry in one packet,
        # but still validate any supplied close_trade_ids so the report shows
        # every violation instead of hiding bad ids or missing Gate A/B behind
        # the receipt-level reentry blocker.
        if decision.action == "TRADE" and decision.close_trade_ids:
            self._verify_close_trade_ids(decision, issues)
            self._verify_close_discipline(decision, issues)

        return VerificationResult(allowed=not any(issue.severity == "BLOCK" for issue in issues), issues=tuple(issues))

    def _verify_decision_freshness(self, decision: GPTTraderDecision, issues: list[VerificationIssue]) -> None:
        if not decision.generated_at_utc:
            issues.append(
                VerificationIssue(
                    "MISSING_DECISION_TIMESTAMP",
                    "decision receipt must include generated_at_utc so broker-snapshot and market-packet freshness can be verified",
                )
            )
            return
        decision_ts = _parse_utc(decision.generated_at_utc)
        if decision_ts is None:
            issues.append(
                VerificationIssue(
                    "BAD_DECISION_TIMESTAMP",
                    f"generated_at_utc is not parseable: {decision.generated_at_utc}",
                )
            )
            return
        broker = self.packet.get("broker_snapshot", {})
        snapshot_ts = _parse_utc(broker.get("fetched_at_utc") if isinstance(broker, dict) else None)
        freshness_checks = [
            (
                snapshot_ts,
                "broker snapshot",
                "current broker truth, position sidecars, and order intents are rebuilt",
            )
        ]
        artifact_timestamps = self.packet.get("artifact_timestamps")
        if isinstance(artifact_timestamps, dict):
            freshness_checks.extend(
                [
                    (
                        _parse_utc(artifact_timestamps.get("order_intents_generated_at_utc")),
                        "order intents",
                        "current order intents are rebuilt from the latest market packet",
                    ),
                    (
                        _parse_utc(artifact_timestamps.get("market_context_matrix_generated_at_utc")),
                        "market_context_matrix",
                        "current market_context_matrix is reflected into the order intents and decision receipt",
                    ),
                ]
            )
        for artifact_ts, label, refresh_hint in freshness_checks:
            if artifact_ts is None:
                continue
            if decision_ts < artifact_ts:
                issues.append(
                    VerificationIssue(
                        "STALE_DECISION_RECEIPT",
                        f"decision receipt predates the {label} used for verification; refresh the "
                        f"decision after {refresh_hint}",
                    )
                )

    def _verify_cancel_order_ids(
        self,
        decision: GPTTraderDecision,
        issues: list[VerificationIssue],
        *,
        action: str,
    ) -> None:
        if not decision.cancel_order_ids:
            return
        pending_order_ids = set(_pending_entry_order_ids(self.packet))
        unknown_cancel_ids = sorted(set(decision.cancel_order_ids) - pending_order_ids)
        if unknown_cancel_ids:
            issues.append(
                VerificationIssue(
                    "UNKNOWN_CANCEL_ORDER_ID",
                    f"{action} cancel_order_ids must match current trader-owned pending entry orders: "
                    + ", ".join(unknown_cancel_ids),
                )
            )

    def _verify_twenty_minute_plan(
        self,
        decision: GPTTraderDecision,
        issues: list[VerificationIssue],
    ) -> None:
        if decision.action not in ENTRY_DECISION_HORIZON_ACTIONS:
            return
        plan = decision.twenty_minute_plan if isinstance(decision.twenty_minute_plan, dict) else {}
        if not plan:
            issues.append(
                VerificationIssue(
                    "SHALLOW_DECISION_HORIZON",
                    "TRADE / WAIT / REQUEST_EVIDENCE receipts must include twenty_minute_plan so the "
                    "operator states the next-cycle path before acting.",
                )
            )
            return

        horizon = _optional_float(plan.get("horizon_minutes"))
        if horizon is None or abs(horizon - TRADER_DECISION_HORIZON_MINUTES) > 0.01:
            issues.append(
                VerificationIssue(
                    "BAD_DECISION_HORIZON_MINUTES",
                    f"twenty_minute_plan.horizon_minutes must match the scheduled trader cadence "
                    f"({TRADER_DECISION_HORIZON_MINUTES} minutes), not an invented holding period.",
                )
            )

        missing_fields = [
            field
            for field in TWENTY_MINUTE_PLAN_TEXT_FIELDS
            if not str(plan.get(field) or "").strip()
        ]
        if missing_fields:
            issues.append(
                VerificationIssue(
                    "INCOMPLETE_TWENTY_MINUTE_PLAN",
                    "twenty_minute_plan is missing required reasoning fields: "
                    + ", ".join(missing_fields),
                )
            )

        raw_refs = plan.get("evidence_refs")
        if isinstance(raw_refs, list):
            plan_refs = tuple(str(ref) for ref in raw_refs if str(ref))
        else:
            plan_refs = ()
        if len(plan_refs) < 2:
            issues.append(
                VerificationIssue(
                    "TWENTY_MINUTE_PLAN_REFS_MISSING",
                    "twenty_minute_plan.evidence_refs must cite at least two packet refs used by the next-cycle path.",
                )
            )
        unknown_refs = sorted(set(plan_refs) - self.allowed_refs)
        if unknown_refs:
            issues.append(
                VerificationIssue(
                    "UNKNOWN_TWENTY_MINUTE_PLAN_REF",
                    "twenty_minute_plan uses unknown evidence refs: " + ", ".join(unknown_refs),
                )
            )

        tradeable_lanes = _tradeable_live_ready_lanes(self.packet)
        if (decision.action == "TRADE" or tradeable_lanes) and not any(
            ref.startswith("chart:") for ref in plan_refs
        ):
            issues.append(
                VerificationIssue(
                    "TWENTY_MINUTE_PLAN_CHART_REF_MISSING",
                    "twenty_minute_plan must cite chart evidence when deciding from current tradeable lanes.",
                )
            )

        if decision.action == "TRADE":
            missing_lane_refs = [
                lane_id
                for lane_id in _selected_trade_lane_ids(decision)
                if f"intent:{lane_id}" not in plan_refs
            ]
            if missing_lane_refs:
                issues.append(
                    VerificationIssue(
                        "TWENTY_MINUTE_PLAN_LANE_REF_MISSING",
                        "twenty_minute_plan must cite the selected intent ref(s): "
                        + ", ".join(missing_lane_refs),
                    )
                )

    def _verify_close_trade_ids(
        self,
        decision: GPTTraderDecision,
        issues: list[VerificationIssue],
    ) -> None:
        if not decision.close_trade_ids:
            if decision.action == "CLOSE":
                issues.append(
                    VerificationIssue(
                        "MISSING_CLOSE_TRADE_IDS",
                        "CLOSE must name the trader-owned trade ids to close",
                    )
                )
            return
        snapshot = self.packet.get("broker_snapshot", {}) or {}
        trader_trade_ids: set[str] = set()
        for position in snapshot.get("position_summaries", []) or []:
            if str(position.get("owner") or "") == "trader":
                tid = position.get("trade_id")
                if tid is not None:
                    trader_trade_ids.add(str(tid))
        unknown = sorted(set(decision.close_trade_ids) - trader_trade_ids)
        if unknown:
            issues.append(
                VerificationIssue(
                    "UNKNOWN_CLOSE_TRADE_ID",
                    "close_trade_ids must match current trader-owned open positions: "
                    + ", ".join(unknown),
                )
            )

    def _verify_close_discipline(
        self,
        decision: GPTTraderDecision,
        issues: list[VerificationIssue],
    ) -> None:
        """Two-gate CLOSE discipline (see feedback_no_unilateral_close.md and
        AGENT_CONTRACT §10):

        Gate A — market evidence: every named trade must have its thesis
        invalidated by either a structural BOS/CHOCH on M15 or H4 against
        the position side, OR by an explicit `invalidation_price` +
        `invalidation_tf` that broker truth confirms has been hit. Prose
        `invalidation` text alone does not count.

        Gate B — standing hard loss-cut authorization or explicit operator
        authorization. Hard Gate A is enough for justified loss-cuts; softer
        Gate A still needs one of:

          1. `QR_OPERATOR_CLOSE_OVERRIDE=1` in the operator shell, OR
          2. A fresh `data/.operator_close_token` file (mtime within
             OPERATOR_CLOSE_TOKEN_FRESH_SECONDS = 5 minutes), created
             by `touch data/.operator_close_token` before each softer CLOSE
             batch.

        The `operator_close_authorized` JSON field remains in the
        schema for backward compatibility and audit trail (the verifier
        logs whether the trader claimed authorization) but is treated
        as advisory only.

        Both gates must pass; failing either rejects the receipt.
        """
        if not decision.close_trade_ids:
            return

        position_by_tid = _trader_position_lookup(self.packet)

        # Gate A: thesis-still-valid check, applied to every named trade.
        # Hard Gate A also carries the operator's standing authorization for a
        # justified loss-cut; softer Gate A still needs env/token Gate B.
        still_valid: list[str] = []
        same_direction_supported: list[str] = []
        needs_explicit_gate_b: list[str] = []
        for tid in decision.close_trade_ids:
            pos = position_by_tid.get(str(tid))
            if pos is None:
                # `_verify_close_trade_ids` already flagged the unknown id;
                # do not double-report.
                continue
            pair = str(pos.get("pair") or "")
            side = str(pos.get("side") or "")
            issues.extend(_close_spread_issues(self.packet, pair, trade_id=str(tid)))
            invalidated, _reason, standing_authorized = _close_thesis_invalidation(
                self.packet,
                pair,
                side,
                trade_id=str(tid),
                decision=decision,
            )
            if not invalidated:
                still_valid.append(
                    f"{tid} ({pair} {side})"
                )
            elif not standing_authorized:
                supported, support_reason = _close_same_direction_matrix_support(
                    self.packet,
                    pair,
                    side,
                )
                unrealized_pl_jpy = _optional_float(pos.get("unrealized_pl_jpy"))
                if supported and (unrealized_pl_jpy is None or unrealized_pl_jpy <= 0):
                    same_direction_supported.append(
                        f"{tid} ({support_reason})"
                    )
                    continue
                needs_explicit_gate_b.append(f"{tid} ({pair} {side})")

        if still_valid:
            issues.append(
                VerificationIssue(
                    "CLOSE_THESIS_STILL_VALID",
                    "CLOSE rejected: thesis still valid (no BOS/CHOCH against "
                    "side on M15/H4, no buffered invalidation_price hit with chart/technical confirmation, and no fresh "
                    "position sidecar REVIEW_CLOSE/RECOMMEND_CLOSE) for: "
                    + ", ".join(still_valid),
                )
            )

        if same_direction_supported:
            issues.append(
                VerificationIssue(
                    "CLOSE_SAME_DIRECTION_MARKET_SUPPORT",
                    "CLOSE rejected: softer close evidence conflicts with "
                    "same-direction directional market_context_matrix support. "
                    "Contract §10 requires HOLD/reprice/TP rebalance while the "
                    "market stack still supports a loss-side open position, "
                    "unless hard invalidation evidence is present. Blocked for: "
                    + ", ".join(same_direction_supported),
                )
            )

        # Gate B (repaired 2026-06-04): hard machine-confirmed loss cuts
        # satisfy the operator's standing "妥当な損切りならやっていい" directive.
        # Softer sidecar-only reviews still require explicit env/token
        # authorization. The JSON receipt field remains advisory-only.
        if needs_explicit_gate_b and not _operator_close_gate_authorized():
            issues.append(
                VerificationIssue(
                    "CLOSE_OPERATOR_AUTH_REQUIRED",
                    "CLOSE rejected for softer close evidence: requires "
                    "QR_OPERATOR_CLOSE_OVERRIDE=1 in the operator shell OR a fresh data/.operator_close_token "
                    f"(mtime within {OPERATOR_CLOSE_TOKEN_FRESH_SECONDS}s). The "
                    "receipt's `operator_close_authorized` field is advisory "
                    "only and is no longer accepted as authorization "
                    "(2026-05-12T15:33 UTC mass-close incident, "
                    "feedback_no_unilateral_close.md). Standing structural close authorization covered no-token "
                    "hard loss-cuts only; explicit Gate B is still missing for: "
                    + ", ".join(needs_explicit_gate_b),
                )
            )

    def _verify_strategy_reviews(self, decision: GPTTraderDecision, issues: list[VerificationIssue]) -> None:
        for review in decision.strategy_reviews:
            lane_id = str(review.get("lane_id") or "")
            method = str(review.get("method") or "")
            verdict = str(review.get("verdict") or "")
            if not lane_id:
                issues.append(VerificationIssue("STRATEGY_REVIEW_LANE_REQUIRED", "strategy review requires lane_id"))
                continue
            lane = self.lanes.get(lane_id)
            if lane is None:
                issues.append(VerificationIssue("UNKNOWN_STRATEGY_REVIEW_LANE", f"review lane is not in packet: {lane_id}"))
                continue
            if method not in ALLOWED_METHODS:
                issues.append(VerificationIssue("BAD_STRATEGY_REVIEW_METHOD", f"unsupported strategy review method {method!r}"))
            elif lane.get("method") != method:
                issues.append(
                    VerificationIssue(
                        "STRATEGY_REVIEW_METHOD_MISMATCH",
                        f"review method {method} does not match lane {lane_id} method {lane.get('method')}",
                    )
                )
            if verdict and verdict not in {"SUPPORTS", "REJECTS", "BLOCKED", "WATCH"}:
                issues.append(VerificationIssue("BAD_STRATEGY_REVIEW_VERDICT", f"unsupported strategy review verdict {verdict!r}"))

    def _verify_specialist_reviews(self, decision: GPTTraderDecision, issues: list[VerificationIssue]) -> None:
        for review in decision.specialist_reviews:
            role = str(review.get("role") or "")
            verdict = str(review.get("verdict") or "")
            lane_id = str(review.get("lane_id") or "")
            method = str(review.get("method") or "")
            cited_refs = tuple(str(ref) for ref in review.get("cited_evidence_refs", []) or [])
            if role not in ALLOWED_SPECIALIST_ROLES:
                issues.append(VerificationIssue("BAD_SPECIALIST_REVIEW_ROLE", f"unsupported specialist review role {role!r}"))
            if verdict not in {"SUPPORTS", "REJECTS", "BLOCKED", "WATCH"}:
                issues.append(VerificationIssue("BAD_SPECIALIST_REVIEW_VERDICT", f"unsupported specialist review verdict {verdict!r}"))
            if review.get("read_only") is not True:
                issues.append(
                    VerificationIssue(
                        "SPECIALIST_REVIEW_NOT_READ_ONLY",
                        "specialist reviews are processed observation only and must declare read_only=true",
                    )
                )
            if review.get("live_permission") is not False:
                issues.append(
                    VerificationIssue(
                        "SPECIALIST_REVIEW_LIVE_PERMISSION",
                        "specialist reviews must declare live_permission=false; only the final verified trader receipt can authorize execution",
                    )
                )
            forbidden = sorted(field for field in FORBIDDEN_SPECIALIST_AUTHORITY_FIELDS if field in review)
            if forbidden:
                issues.append(
                    VerificationIssue(
                        "SPECIALIST_REVIEW_AUTHORITY_FIELD",
                        "specialist reviews must not carry execution authority fields: " + ", ".join(forbidden),
                    )
                )
            if not cited_refs:
                issues.append(
                    VerificationIssue(
                        "SPECIALIST_REVIEW_REFS_REQUIRED",
                        "specialist reviews must cite packet evidence refs",
                    )
                )
            unknown_refs = sorted(set(cited_refs) - self.allowed_refs)
            if unknown_refs:
                issues.append(
                    VerificationIssue(
                        "UNKNOWN_SPECIALIST_REVIEW_REF",
                        f"specialist review uses unknown evidence refs: {', '.join(unknown_refs)}",
                    )
                )
            if lane_id:
                lane = self.lanes.get(lane_id)
                if lane is None:
                    issues.append(VerificationIssue("UNKNOWN_SPECIALIST_REVIEW_LANE", f"specialist review lane is not in packet: {lane_id}"))
                elif method and method != str(lane.get("method") or ""):
                    issues.append(
                        VerificationIssue(
                            "SPECIALIST_REVIEW_METHOD_MISMATCH",
                            f"specialist review method {method} does not match lane {lane_id} method {lane.get('method')}",
                        )
                    )


GPT_TRADER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "generated_at_utc",
        "action",
        "selected_lane_id",
        "confidence",
        "thesis",
        "method",
        "narrative",
        "chart_story",
        "invalidation",
        "rejected_alternatives",
        "risk_notes",
        "evidence_refs",
        "operator_summary",
        "twenty_minute_plan",
    ],
    "properties": {
        "generated_at_utc": {"type": "string"},
        "action": {"type": "string", "enum": list(ALLOWED_ACTIONS)},
        "selected_lane_id": {"type": ["string", "null"]},
        "selected_lane_ids": {"type": "array", "items": {"type": "string"}},
        "cancel_order_ids": {"type": "array", "items": {"type": "string"}},
        "close_trade_ids": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "string", "enum": list(ALLOWED_CONFIDENCE)},
        "thesis": {"type": "string"},
        "method": {"type": "string", "enum": list(ALLOWED_METHODS)},
        "narrative": {"type": "string"},
        "chart_story": {"type": "string"},
        "invalidation": {"type": "string"},
        "invalidation_price": {"type": ["number", "null"]},
        "invalidation_tf": {"type": ["string", "null"]},
        "operator_close_authorized": {"type": "boolean"},
        "rejected_alternatives": {"type": "array", "items": {"type": "string"}},
        "risk_notes": {"type": "array", "items": {"type": "string"}},
        "evidence_refs": {"type": "array", "items": {"type": "string"}},
        "operator_summary": {"type": "string"},
        "twenty_minute_plan": {
            "type": ["object", "null"],
            "additionalProperties": False,
            "required": [
                "horizon_minutes",
                "primary_path",
                "failure_path",
                "entry_or_hold_trigger",
                "invalidation_or_cancel_trigger",
                "counterargument",
                "next_cycle_check",
                "evidence_refs",
            ],
            "properties": {
                "horizon_minutes": {"type": "number"},
                "primary_path": {"type": "string"},
                "failure_path": {"type": "string"},
                "entry_or_hold_trigger": {"type": "string"},
                "invalidation_or_cancel_trigger": {"type": "string"},
                "counterargument": {"type": "string"},
                "next_cycle_check": {"type": "string"},
                "evidence_refs": {"type": "array", "items": {"type": "string"}},
            },
        },
        "strategy_reviews": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["lane_id", "method", "verdict", "summary"],
                "properties": {
                    "lane_id": {"type": "string"},
                    "method": {"type": "string", "enum": list(ALLOWED_METHODS)},
                    "verdict": {"type": "string", "enum": ["SUPPORTS", "REJECTS", "BLOCKED", "WATCH"]},
                    "summary": {"type": "string"},
                },
            },
        },
        "specialist_reviews": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "role",
                    "verdict",
                    "summary",
                    "cited_evidence_refs",
                    "read_only",
                    "live_permission",
                ],
                "properties": {
                    "role": {"type": "string", "enum": list(ALLOWED_SPECIALIST_ROLES)},
                    "lane_id": {"type": ["string", "null"]},
                    "method": {"type": ["string", "null"], "enum": [*ALLOWED_METHODS, None]},
                    "verdict": {"type": "string", "enum": ["SUPPORTS", "REJECTS", "BLOCKED", "WATCH"]},
                    "summary": {"type": "string"},
                    "cited_evidence_refs": {"type": "array", "items": {"type": "string"}},
                    "hard_gate_codes": {"type": "array", "items": {"type": "string"}},
                    "read_only": {"type": "boolean"},
                    "live_permission": {"type": "boolean"},
                },
            },
        },
    },
}


def _decision_from_payload(payload: dict[str, Any]) -> GPTTraderDecision:
    selected_lane_id = payload.get("selected_lane_id")
    selected_lane_ids = tuple(
        dict.fromkeys(str(item) for item in payload.get("selected_lane_ids", []) or [] if str(item))
    )
    if not selected_lane_ids and selected_lane_id is not None:
        selected_lane_ids = (str(selected_lane_id),)
    return GPTTraderDecision(
        generated_at_utc=(
            str(payload.get("generated_at_utc")) if payload.get("generated_at_utc") else None
        ),
        action=str(payload.get("action") or ""),
        selected_lane_id=str(selected_lane_id) if selected_lane_id is not None else None,
        selected_lane_ids=selected_lane_ids,
        cancel_order_ids=tuple(str(item) for item in payload.get("cancel_order_ids", []) or []),
        confidence=str(payload.get("confidence") or ""),
        thesis=str(payload.get("thesis") or ""),
        method=str(payload.get("method") or ""),
        narrative=str(payload.get("narrative") or ""),
        chart_story=str(payload.get("chart_story") or ""),
        invalidation=str(payload.get("invalidation") or ""),
        rejected_alternatives=tuple(str(item) for item in payload.get("rejected_alternatives", []) or []),
        risk_notes=tuple(str(item) for item in payload.get("risk_notes", []) or []),
        evidence_refs=tuple(str(item) for item in payload.get("evidence_refs", []) or []),
        operator_summary=str(payload.get("operator_summary") or ""),
        twenty_minute_plan=(
            dict(payload.get("twenty_minute_plan"))
            if isinstance(payload.get("twenty_minute_plan"), dict)
            else None
        ),
        close_trade_ids=tuple(str(item) for item in payload.get("close_trade_ids", []) or []),
        invalidation_price=_optional_float(payload.get("invalidation_price")),
        invalidation_tf=(
            str(payload.get("invalidation_tf")) if payload.get("invalidation_tf") else None
        ),
        operator_close_authorized=bool(payload.get("operator_close_authorized", False)),
        strategy_reviews=tuple(
            dict(item)
            for item in payload.get("strategy_reviews", []) or []
            if isinstance(item, dict)
        ),
        specialist_reviews=tuple(
            dict(item)
            for item in payload.get("specialist_reviews", []) or []
            if isinstance(item, dict)
        ),
    )


def _selected_trade_lane_ids(decision: GPTTraderDecision) -> tuple[str, ...]:
    lane_ids = tuple(dict.fromkeys(lane_id for lane_id in decision.selected_lane_ids if lane_id))
    if lane_ids:
        return lane_ids
    return (decision.selected_lane_id,) if decision.selected_lane_id else ()


def _lane_packet(
    intents: dict[str, Any],
    campaign: dict[str, Any],
    strategy: dict[str, Any],
    story: dict[str, Any],
    *,
    max_lanes: int,
) -> list[dict[str, Any]]:
    campaign_index = {f"{lane.get('desk')}:{lane.get('pair')}:{lane.get('direction')}:{lane.get('method')}": lane for lane in campaign.get("lanes", []) or []}
    strategy_index = {(item.get("pair"), item.get("direction")): item for item in strategy.get("profiles", []) or []}
    story_index = {item.get("pair"): item for item in story.get("pair_profiles", []) or []}
    lanes: list[dict[str, Any]] = []
    for result in intents.get("results", []) or []:
        if not isinstance(result, dict) or not isinstance(result.get("intent"), dict):
            continue
        intent = result["intent"]
        lane_id = str(result.get("lane_id") or "")
        pair = str(intent.get("pair") or "")
        direction = str(intent.get("side") or "")
        context = intent.get("market_context") or {}
        metadata = intent.get("metadata") if isinstance(intent.get("metadata"), dict) else {}
        risk_blockers = _block_issues(result.get("risk_issues"))
        strategy_blockers = _block_issues(result.get("strategy_issues"))
        lanes.append(
            {
                "lane_id": lane_id,
                "evidence_ref": f"intent:{lane_id}",
                "status": result.get("status"),
                "pair": pair,
                "direction": direction,
                "method": context.get("method") or "",
                "order_type": intent.get("order_type"),
                "entry": intent.get("entry"),
                "tp": intent.get("tp"),
                "sl": intent.get("sl"),
                "units": intent.get("units"),
                "risk_metrics": _small_dict(
                    result.get("risk_metrics"),
                    ("entry_price", "loss_pips", "reward_pips", "risk_jpy", "reward_jpy", "reward_risk", "spread_pips", "jpy_per_pip"),
                ),
                "thesis": intent.get("thesis"),
                "narrative": context.get("narrative") or "",
                "chart_story": context.get("chart_story") or "",
                "invalidation": context.get("invalidation") or "",
                "risk_blockers": risk_blockers,
                "strategy_blockers": strategy_blockers,
                "live_blockers": list(result.get("live_blockers", []) or []),
                "campaign": _small_dict(
                    campaign_index.get(lane_id) or campaign_index.get(_parent_lane_id(lane_id)),
                    ("adoption", "campaign_role", "required_receipt"),
                ),
                "strategy": _small_dict(
                    strategy_index.get((pair, direction)),
                    (
                        "status",
                        "pretrade_net_jpy",
                        "live_net_jpy",
                        "live_worst_jpy",
                        "seat_pl_n",
                        "seat_net_jpy",
                        "seat_win_rate_pct",
                        "required_fix",
                    ),
                ),
                "story": _small_dict(story_index.get(pair), ("methods", "themes", "examples")),
                "forecast": _lane_forecast_packet(intent.get("metadata")),
                "opportunity": _small_dict(
                    metadata,
                    (
                        "opportunity_mode",
                        "opportunity_mode_reason",
                        "opportunity_mode_reward_risk",
                        "tp_execution_mode",
                        "tp_target_intent",
                        "tp_target_source",
                    ),
                ),
                "position_building": _small_dict(
                    metadata,
                    (
                        "position_intent",
                        "position_fill",
                        "same_pair_add_type",
                        "same_pair_existing_entries",
                        "same_pair_existing_units",
                        "same_pair_existing_avg_entry",
                        "same_pair_add_entry",
                        "same_pair_add_distance_from_avg_pips",
                        "same_pair_adverse_add_pips",
                        "same_pair_with_move_add_pips",
                        "hedge_suppressed_reason",
                    ),
                ),
                "market_context_matrix": _small_dict(
                    metadata,
                    (
                        "market_context_matrix_ref",
                        "matrix_support_count",
                        "matrix_reject_count",
                        "matrix_warning_count",
                        "matrix_missing_count",
                        "strongest_matrix_support",
                        "strongest_matrix_reject",
                    ),
                ),
                "technical_context": _small_dict(
                    metadata,
                    (
                        "session_bucket",
                        "session_current_tag",
                        "entry_price_percentile_24h",
                        "entry_price_percentile_7d",
                        "price_percentile_24h",
                        "price_percentile_7d",
                        "range_24h_sigma_multiple",
                        "tf_agreement_score",
                        "chart_direction_bias",
                        "h1_regime",
                        "h1_adx",
                        "m5_regime",
                        "m5_regime_quantile",
                        "current_price_mid",
                    ),
                ),
            }
        )
    if max_lanes <= 0 or len(lanes) <= max_lanes:
        return lanes
    capped = lanes[:max_lanes]
    capped_ids = {str(lane.get("lane_id") or "") for lane in capped}
    for lane in lanes[max_lanes:]:
        lane_id = str(lane.get("lane_id") or "")
        if lane_id and lane_id not in capped_ids and lane.get("status") == "LIVE_READY":
            capped.append(lane)
            capped_ids.add(lane_id)
    return capped


def _snapshot_packet(snapshot: dict[str, Any]) -> dict[str, Any]:
    orders = snapshot.get("orders", []) or []
    quotes_in = snapshot.get("quotes") or {}
    # Scope quotes to pairs we actually have positions on (or are likely
    # to reference in this cycle) so the verifier packet stays compact
    # but the CLOSE-discipline gate has bid/ask available to check
    # `invalidation_price` hits against current broker truth.
    relevant_pairs: set[str] = set()
    for item in snapshot.get("positions", []) or []:
        if item.get("pair"):
            relevant_pairs.add(str(item["pair"]))
    for order in orders:
        if order.get("pair"):
            relevant_pairs.add(str(order["pair"]))
    scoped_quotes = {
        pair: {
            "bid": q.get("bid"),
            "ask": q.get("ask"),
            "timestamp_utc": q.get("timestamp_utc"),
        }
        for pair, q in quotes_in.items()
        if pair in relevant_pairs and isinstance(q, dict)
    }
    return {
        "evidence_ref": "broker:snapshot",
        "fetched_at_utc": snapshot.get("fetched_at_utc"),
        "positions": len(snapshot.get("positions", []) or []),
        "orders": len(orders),
        "position_summaries": [
            {
                "trade_id": item.get("trade_id"),
                "pair": item.get("pair"),
                "side": item.get("side"),
                "units": item.get("units"),
                "unrealized_pl_jpy": item.get("unrealized_pl_jpy"),
                "take_profit": item.get("take_profit"),
                "stop_loss": item.get("stop_loss"),
                "owner": item.get("owner"),
            }
            for item in (snapshot.get("positions", []) or [])
        ],
        "pending_orders": _pending_order_packet(orders),
        "quotes": scoped_quotes,
    }


def _pending_order_packet(orders: list[Any]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(item: Any) -> None:
        if not isinstance(item, dict):
            return
        order_id = str(item.get("order_id") or "")
        if order_id in seen:
            return
        if order_id:
            seen.add(order_id)
        selected.append(
            {
                "order_id": item.get("order_id"),
                "pair": item.get("pair"),
                "order_type": item.get("order_type"),
                "trade_id": item.get("trade_id"),
                "price": item.get("price"),
                "units": item.get("units"),
                "owner": item.get("owner"),
            }
        )

    for item in orders[:5]:
        add(item)
    for item in orders:
        if _is_pending_entry_order_payload(item):
            add(item)
    return selected


def _is_pending_entry_order_payload(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    if item.get("trade_id"):
        return False
    order_type = str(item.get("order_type") or "").upper()
    return order_type in {"LIMIT", "STOP", "MARKET_IF_TOUCHED", "MARKET_IF_TOUCHED_ORDER"}


def _parent_lane_id(lane_id: str) -> str:
    if lane_id.endswith(":MARKET"):
        return lane_id[: -len(":MARKET")]
    return lane_id


def _target_packet(target: dict[str, Any]) -> dict[str, Any]:
    if not target:
        return {"evidence_ref": "target:daily", "status": "missing"}
    return {
        "evidence_ref": "target:daily",
        "status": target.get("status"),
        "target_jpy": target.get("target_jpy"),
        "progress_jpy": target.get("progress_jpy"),
        "account_progress_jpy": target.get("account_progress_jpy"),
        "account_progress_pct": target.get("account_progress_pct"),
        "account_unrealized_pl_jpy": target.get("account_unrealized_pl_jpy"),
        "current_equity_jpy": target.get("current_equity_jpy"),
        "remaining_target_jpy": target.get("remaining_target_jpy"),
        "remaining_risk_budget_jpy": target.get("remaining_risk_budget_jpy"),
    }


def _protection_sidecars_packet(
    *,
    snapshot: dict[str, Any],
    snapshot_path: Path,
    pair_charts_path: Path,
) -> dict[str, Any]:
    from quant_rabbit.trader_prompts import (
        _fresh_close_recommendations,
        _fresh_entry_thesis_blockers,
        _fresh_position_hold_support,
        _tp_rebalance_reasons,
    )

    position_close_recommendations = list(
        _fresh_close_recommendations(snapshot, data_root=snapshot_path.parent)
    )
    position_hold_support = list(
        _fresh_position_hold_support(snapshot, data_root=snapshot_path.parent)
    )
    entry_thesis_blockers = list(
        _fresh_entry_thesis_blockers(snapshot, data_root=snapshot_path.parent)
    )
    entry_thesis_close_context = _entry_thesis_close_context(
        snapshot,
        data_root=snapshot_path.parent,
    )
    if not pair_charts_path.exists():
        sidecars = {
            "tp_rebalance": {
                "required": False,
                "reasons": [],
                "issue": f"missing pair_charts: {pair_charts_path}",
            },
            "position_close_recommendations": position_close_recommendations,
            "position_hold_support": position_hold_support,
            "entry_thesis_blockers": entry_thesis_blockers,
            "entry_thesis_close_context": entry_thesis_close_context,
        }
        sidecars["position_close_recommendations"] = _annotated_position_close_recommendations(
            snapshot,
            sidecars,
        )
        return sidecars
    pair_charts = _load_json(pair_charts_path)

    reasons = _tp_rebalance_reasons(
        snapshot,
        pair_charts,
        snapshot_path=snapshot_path,
    )
    sidecars = {
        "tp_rebalance": {
            "required": bool(reasons),
            "reasons": list(reasons),
        },
        "position_close_recommendations": position_close_recommendations,
        "position_hold_support": position_hold_support,
        "entry_thesis_blockers": entry_thesis_blockers,
        "entry_thesis_close_context": entry_thesis_close_context,
    }
    sidecars["position_close_recommendations"] = _annotated_position_close_recommendations(
        snapshot,
        sidecars,
    )
    return sidecars


def _annotated_position_close_recommendations(
    snapshot: dict[str, Any],
    sidecars: dict[str, Any],
) -> list[dict[str, Any]]:
    packet = {
        "broker_snapshot": _snapshot_packet(snapshot),
        "protection_sidecars": sidecars,
    }
    annotated: list[dict[str, Any]] = []
    for rec in sidecars.get("position_close_recommendations") or []:
        if not isinstance(rec, dict):
            continue
        item = dict(rec)
        blocks_non_close = _position_close_recommendation_blocks_non_close_action(packet, item)
        item["blocks_non_close_actions"] = blocks_non_close
        item["routing_effect"] = (
            "BLOCKS_NON_CLOSE_ACTIONS"
            if blocks_non_close
            else "SOFT_ADVISORY_NON_BLOCKING"
        )
        if not blocks_non_close:
            item["entry_decision_guidance"] = (
                "Do not choose CLOSE from this advisory on the entry branch unless "
                "explicit operator Gate B is present; evaluate current LIVE_READY lanes."
            )
            hold_conflict = _sidecar_hold_support_conflict(packet, item)
            if hold_conflict:
                item["non_blocking_reason"] = hold_conflict
            elif not _operator_close_gate_authorized():
                item["non_blocking_reason"] = (
                    "soft close evidence lacks explicit operator Gate B and therefore "
                    "does not preempt entry routing"
                )
        annotated.append(item)
    return annotated


def _entry_thesis_close_context(snapshot: dict[str, Any], *, data_root: Path) -> list[dict[str, Any]]:
    from quant_rabbit.strategy.entry_thesis_ledger import load_entry_thesis

    rows: list[dict[str, Any]] = []
    for position in snapshot.get("positions", []) or []:
        if not isinstance(position, dict) or str(position.get("owner") or "") != "trader":
            continue
        trade_id = str(position.get("trade_id") or "")
        pair = str(position.get("pair") or "")
        side = str(position.get("side") or "").upper()
        if not trade_id:
            continue
        thesis = load_entry_thesis(trade_id, data_root)
        invalidation = getattr(thesis, "invalidation_price", None) if thesis is not None else None
        rows.append(
            {
                "trade_id": trade_id,
                "pair": pair,
                "side": side,
                "recorded": thesis is not None,
                "has_recorded_invalidation_price": invalidation is not None,
                "recorded_invalidation_price": invalidation,
            }
        )
    return rows


def _allowed_refs(
    *,
    snapshot: dict[str, Any],
    target: dict[str, Any],
    lanes: list[dict[str, Any]],
    attack_advice: dict[str, Any] | None,
    capture_economics: dict[str, Any] | None,
    coverage_optimization: dict[str, Any] | None,
    learning_audit: dict[str, Any] | None,
    verification_ledger: dict[str, Any] | None,
    self_improvement_audit: dict[str, Any] | None,
    operator_precedent: dict[str, Any] | None,
    manual_market_context: dict[str, Any] | None,
    predictive_limits: dict[str, Any] | None,
    market_status: dict[str, Any] | None,
    market_context_matrix: dict[str, Any] | None,
    option_skew: dict[str, Any] | None,
) -> list[str]:
    # Per docs/SKILL_trader.md the playbook prescribes a richer set of evidence
    # refs than the base broker/target/lane triple — the trader is required to
    # cite per-pair charts, cross-asset, flow, levels, currency strength,
    # economic calendar, and COT data. The verifier therefore must accept these
    # refs as known; otherwise every well-formed decision is rejected with
    # UNKNOWN_EVIDENCE_REF and the cycle never reaches the gateway.
    timeframes = DEFAULT_PAIR_CHART_TIMEFRAMES
    structure_keys = ("structure",)
    cross_assets = (
        "dxy",
        "USB10Y_USD",
        "USB02Y_USD",
        "SPX500_USD",
        "XAU_USD",
        "WTICO_USD",
        "BTC_USD",
        "spx",
        "gold",
        "oil",
        "btc",
    )
    refs = ["broker:snapshot", "target:daily", "verification:ledger"]
    if isinstance(market_status, dict):
        refs.append(str(market_status.get("evidence_ref") or "market:status"))
        pairs: set[str] = set()
        currencies: set[str] = set()
    for position in snapshot.get("positions", []) or []:
        if not isinstance(position, dict) or str(position.get("owner") or "") != "trader":
            continue
        pair = str(position.get("pair") or "")
        if pair:
            pairs.add(pair)
            for currency in pair.split("_"):
                if currency:
                    currencies.add(currency)
        trade_id = str(position.get("trade_id") or "")
        if not trade_id:
            continue
        refs.extend(
            [
                f"position:thesis:{trade_id}",
                f"position:evolution:{trade_id}",
                f"position:management:{trade_id}",
                f"position:guardian_management:{trade_id}",
                f"position:persistence:{trade_id}",
            ]
        )
    for lane in lanes:
        lane_id = lane["lane_id"]
        pair = str(lane.get("pair") or "")
        direction = str(lane.get("direction") or "")
        if pair:
            pairs.add(pair)
            for currency in pair.split("_"):
                if currency:
                    currencies.add(currency)
        refs.extend(
            [
                str(lane["evidence_ref"]),
                f"campaign:{lane_id}",
                f"strategy:{pair}:{direction}",
                f"story:{pair}",
                f"intent:{lane_id}",
            ]
        )
    for pair in pairs:
        for tf in timeframes:
            refs.append(f"chart:{pair}:{tf}")
        for key in structure_keys:
            refs.append(f"chart:{pair}:{key}")
        pair_refs = [
            f"flow:{pair}",
            f"levels:{pair}",
            f"calendar:{pair}",
            f"strength:{pair}",
            f"cross:correlations:{pair}",
        ]
        if _option_skew_enabled(option_skew):
            pair_refs.append(f"option:skew:{pair}")
        refs.extend(pair_refs)
        refs.append(f"matrix:{pair}")
        side_map = ((market_context_matrix or {}).get("pairs") or {}).get(pair)
        if isinstance(side_map, dict):
            for side in ("LONG", "SHORT"):
                if isinstance(side_map.get(side), dict):
                    refs.append(str(side_map[side].get("evidence_ref") or f"matrix:{pair}:{side}"))
        else:
            refs.extend([f"matrix:{pair}:LONG", f"matrix:{pair}:SHORT"])
    for currency in currencies:
        refs.append(f"cot:{currency}")
        refs.append(f"strength:{currency}")
        refs.append(f"calendar:{currency}")
    for asset in cross_assets:
        refs.append(f"cross:{asset}")
    refs.extend(["cross:dxy", "cross:correlations"])
    refs.append("broker:instruments")
    refs.extend(f"context_asset:{asset}" for asset in DEFAULT_CONTEXT_ASSETS)
    if _option_skew_enabled(option_skew):
        refs.extend(["option:skew", "option:skew:unknown"])
    if attack_advice:
        refs.append("attack:advice")
        for lane_id in attack_advice.get("recommended_now_lane_ids", []) or []:
            refs.append(f"attack:lane:{lane_id}")
        for lane_id in attack_advice.get("watchlist_lane_ids", []) or []:
            refs.append(f"attack:lane:{lane_id}")
    if capture_economics:
        refs.append("capture:economics")
        by_exit = capture_economics.get("by_exit_reason")
        if isinstance(by_exit, dict):
            for reason in by_exit:
                reason_key = str(reason or "").strip()
                if reason_key:
                    refs.append(f"capture:exit_reason:{reason_key}")
    if coverage_optimization:
        refs.append("coverage:optimization")
        diagnostics = (
            coverage_optimization.get("artifact_diagnostics")
            if isinstance(coverage_optimization.get("artifact_diagnostics"), dict)
            else {}
        )
        bucket_diag = (
            diagnostics.get("profitable_bucket_coverage")
            if isinstance(diagnostics.get("profitable_bucket_coverage"), dict)
            else {}
        )
        for edge in bucket_diag.get("top_edges", []) or []:
            if not isinstance(edge, dict):
                continue
            pair = str(edge.get("pair") or "")
            direction = str(edge.get("direction") or "")
            if pair and direction:
                refs.append(f"coverage:profitable_bucket:{pair}:{direction}")
        for edge in bucket_diag.get("matrix_supported_repair_queue", []) or []:
            if not isinstance(edge, dict):
                continue
            pair = str(edge.get("pair") or "")
            direction = str(edge.get("direction") or "")
            if pair and direction:
                refs.append(f"coverage:profitable_bucket:{pair}:{direction}")
    if learning_audit:
        refs.append("learning:audit")
        influence = learning_audit.get("learning_influence") if isinstance(learning_audit.get("learning_influence"), dict) else {}
        for lane in influence.get("lanes", []) or []:
            if not isinstance(lane, dict):
                continue
            lane_id = str(lane.get("lane_id") or "")
            if lane_id:
                refs.append(f"learning:lane:{lane_id}")
        effect = learning_audit.get("effect_metrics") if isinstance(learning_audit.get("effect_metrics"), dict) else {}
        exit_reasons = effect.get("exit_reason_metrics") if isinstance(effect.get("exit_reason_metrics"), dict) else {}
        for reason in exit_reasons:
            reason_key = str(reason or "").strip()
            if reason_key:
                refs.append(f"learning:exit_reason:{reason_key}")
    if verification_ledger:
        refs.extend(["verification:blockers", "verification:effect:all"])
        for key in ("blocking_evidence", "missing_artifacts", "learning_evidence", "measurements"):
            rows = verification_ledger.get(key) if isinstance(verification_ledger.get(key), list) else []
            for item in rows:
                if not isinstance(item, dict):
                    continue
                ref = str(item.get("evidence_ref") or "").strip()
                if ref:
                    refs.append(ref)
    if self_improvement_audit:
        refs.extend(["self_improvement:audit", "self_improvement:profitability"])
        for finding in self_improvement_audit.get("findings", []) or []:
            if not isinstance(finding, dict):
                continue
            layer = str(finding.get("layer") or "").strip()
            if layer:
                refs.append(f"self_improvement:{layer}")
            code = str(finding.get("code") or "").strip()
            if code:
                refs.append(f"self_improvement:finding:{code}")
    if operator_precedent:
        refs.append(OPERATOR_PRECEDENT_EVIDENCE_REF)
    if manual_market_context:
        refs.append(MANUAL_MARKET_CONTEXT_EVIDENCE_REF)
    if predictive_limits:
        refs.append("predictive:limits")
        for item in predictive_limits.get("orders", []) or []:
            if not isinstance(item, dict):
                continue
            pair = str(item.get("pair") or "")
            side = str(item.get("side") or "")
            if pair and side:
                refs.append(f"predictive:limit:{pair}:{side}")
    return sorted(set(refs))


def _option_skew_enabled(payload: dict[str, Any] | None) -> bool:
    if not isinstance(payload, dict):
        return False
    if payload.get("enabled") is False and payload.get("disabled_reason"):
        return False
    return bool(payload.get("readings") or payload.get("issues") or payload.get("provider"))


def _attack_advice_packet(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {"evidence_ref": "attack:advice", "status": "missing"}
    recommended_ids = {str(item) for item in payload.get("recommended_now_lane_ids", []) or [] if str(item).strip()}
    lane_summaries: list[dict[str, Any]] = []
    learning_influenced_lane_ids: list[str] = []
    for lane in payload.get("lanes", []) or []:
        if not isinstance(lane, dict):
            continue
        lane_id = str(lane.get("lane_id") or "")
        if not lane_id or lane_id not in recommended_ids:
            continue
        influences = [str(item) for item in (lane.get("learning_influences") or []) if str(item).strip()]
        lane_summary = {
            "lane_id": lane_id,
            "learning_influences": influences,
            "learning_score_delta": lane.get("learning_score_delta"),
        }
        lane_summaries.append(lane_summary)
        if influences:
            learning_influenced_lane_ids.append(lane_id)
    return {
        "evidence_ref": "attack:advice",
        "status": payload.get("status"),
        "read_only": payload.get("read_only"),
        "live_permission": payload.get("live_permission"),
        "coverage_pct": payload.get("coverage_pct"),
        "recommended_now_lane_ids": list(payload.get("recommended_now_lane_ids", []) or []),
        "recommended_now_reward_jpy": payload.get("recommended_now_reward_jpy"),
        "recommended_now_risk_jpy": payload.get("recommended_now_risk_jpy"),
        "required_additional_reward_jpy": payload.get("required_additional_reward_jpy"),
        "recommended_lane_learning": lane_summaries[:20],
        "learning_influenced_lane_ids": learning_influenced_lane_ids[:20],
        "settings_advice": payload.get("settings_advice") if isinstance(payload.get("settings_advice"), dict) else {},
    }


def _capture_economics_packet(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {
            "evidence_ref": "capture:economics",
            "status": "missing",
            "overall": {},
            "by_exit_reason": {},
        }
    by_exit: dict[str, Any] = {}
    raw_by_exit = payload.get("by_exit_reason")
    if isinstance(raw_by_exit, dict):
        for reason, metrics in raw_by_exit.items():
            if not isinstance(metrics, dict):
                continue
            by_exit[str(reason)] = {
                "evidence_ref": f"capture:exit_reason:{reason}",
                **_small_dict(
                    metrics,
                    (
                        "trades",
                        "wins",
                        "losses",
                        "win_rate",
                        "payoff_ratio",
                        "breakeven_payoff_at_win_rate",
                        "expectancy_jpy_per_trade",
                        "net_jpy",
                        "avg_win_jpy",
                        "avg_loss_jpy",
                    ),
                ),
            }
    return {
        "evidence_ref": "capture:economics",
        "generated_at_utc": payload.get("generated_at_utc"),
        "status": payload.get("status"),
        "overall": _small_dict(
            payload.get("overall"),
            (
                "trades",
                "wins",
                "losses",
                "win_rate",
                "payoff_ratio",
                "breakeven_payoff_at_win_rate",
                "expectancy_jpy_per_trade",
                "net_jpy",
                "avg_win_jpy",
                "avg_loss_jpy",
            ),
        ),
        "by_exit_reason": by_exit,
    }


def _coverage_optimization_packet(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {
            "evidence_ref": "coverage:optimization",
            "status": "missing",
            "live_permission": False,
            "profitable_bucket_coverage": {},
            "action_items": [],
        }
    diagnostics = payload.get("artifact_diagnostics") if isinstance(payload.get("artifact_diagnostics"), dict) else {}
    bucket_diag = (
        diagnostics.get("profitable_bucket_coverage")
        if isinstance(diagnostics.get("profitable_bucket_coverage"), dict)
        else {}
    )
    return {
        "evidence_ref": "coverage:optimization",
        "status": payload.get("status"),
        "generated_at_utc": payload.get("generated_at_utc"),
        "live_permission": False,
        **_small_dict(
            payload,
            (
                "remaining_target_jpy",
                "remaining_risk_budget_jpy",
                "live_ready_reward_jpy",
                "live_ready_risk_jpy",
                "potential_reward_jpy",
                "coverage_pct",
                "potential_coverage_pct",
                "sequential_ladder_reward_jpy",
                "sequential_ladder_steps",
            ),
        ),
        "diagnostics": _small_dict(
            diagnostics,
            (
                "intents_artifact_stale",
                "all_lanes_spread_blocked",
                "spread_normalized_candidate_count",
                "spread_normalized_candidate_reward_jpy",
                "spread_normalized_no_live_blocker_count",
                "spread_normalized_no_live_blocker_reward_jpy",
                "market_context_matrix_missing",
            ),
        ),
        "profitable_bucket_coverage": _profitable_bucket_coverage_packet(bucket_diag),
        "opportunity_modes": _opportunity_modes_packet(payload.get("opportunity_modes")),
        "runner_candidate_diagnostics": _runner_candidate_diagnostics_packet(
            payload.get("runner_candidate_diagnostics")
        ),
        "action_items": [str(item) for item in (payload.get("action_items") or [])[:8] if str(item).strip()],
    }


def _runner_candidate_diagnostics_packet(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    return {
        "status": payload.get("status"),
        "trend_candidate_lanes": payload.get("trend_candidate_lanes"),
        "runner_qualified_lanes": payload.get("runner_qualified_lanes"),
        "attached_harvest_lanes": payload.get("attached_harvest_lanes"),
        "status_counts": payload.get("status_counts") if isinstance(payload.get("status_counts"), dict) else {},
        "top_demotion_reasons": [
            {
                "reason": str(item.get("reason") or ""),
                "count": item.get("count"),
            }
            for item in (payload.get("top_demotion_reasons") or [])[:5]
            if isinstance(item, dict) and str(item.get("reason") or "").strip()
        ],
        "top_issue_codes": [
            {
                "code": str(item.get("code") or ""),
                "count": item.get("count"),
            }
            for item in (payload.get("top_issue_codes") or [])[:5]
            if isinstance(item, dict) and str(item.get("code") or "").strip()
        ],
        "top_lanes": [
            {
                "lane_id": str(item.get("lane_id") or ""),
                "status": item.get("status"),
                "opportunity_mode": item.get("opportunity_mode"),
                "tp_execution_mode": item.get("tp_execution_mode"),
                "tp_attach_reason": item.get("tp_attach_reason"),
                "reward_risk": item.get("reward_risk"),
            }
            for item in (payload.get("top_lanes") or [])[:5]
            if isinstance(item, dict) and str(item.get("lane_id") or "").strip()
        ],
    }


def _opportunity_modes_packet(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    out: dict[str, Any] = {}
    for mode in ("HARVEST", "RUNNER", "BALANCED"):
        item = payload.get(mode)
        if not isinstance(item, dict):
            continue
        out[mode] = {
            "lanes": item.get("lanes"),
            "live_ready_lanes": item.get("live_ready_lanes"),
            "promotion_candidate_lanes": item.get("promotion_candidate_lanes"),
            "reward_jpy": item.get("reward_jpy"),
            "live_ready_reward_jpy": item.get("live_ready_reward_jpy"),
            "potential_reward_jpy": item.get("potential_reward_jpy"),
            "coverage_pct": item.get("coverage_pct"),
            "potential_coverage_pct": item.get("potential_coverage_pct"),
            "diagnostic_candidate_lanes": item.get("diagnostic_candidate_lanes"),
            "demoted_to_harvest_lanes": item.get("demoted_to_harvest_lanes"),
            "runner_qualified_lanes": item.get("runner_qualified_lanes"),
            "diagnostic_status": item.get("diagnostic_status"),
            "top_demotion_reasons": list(item.get("top_demotion_reasons") or [])[:5],
            "top_issue_codes": list(item.get("top_issue_codes") or [])[:5],
            "top_blockers": list(item.get("top_blockers") or [])[:3],
            "top_lanes": list(item.get("top_lanes") or [])[:3],
        }
    return out


def _profitable_bucket_coverage_packet(payload: dict[str, Any]) -> dict[str, Any]:
    if not payload:
        return {}
    top_edges: list[dict[str, Any]] = []
    for edge in payload.get("top_edges", []) or []:
        if not isinstance(edge, dict):
            continue
        pair = str(edge.get("pair") or "")
        direction = str(edge.get("direction") or "")
        evidence_ref = f"coverage:profitable_bucket:{pair}:{direction}" if pair and direction else None
        top_edges.append(
            {
                "evidence_ref": evidence_ref,
                "pair": pair,
                "direction": direction,
                "coverage_state": edge.get("coverage_state"),
                "managed_net_jpy": edge.get("managed_net_jpy"),
                "raw_net_jpy": edge.get("raw_net_jpy"),
                "trades": edge.get("trades"),
                "days": edge.get("days"),
                "current_lane_count": edge.get("current_lane_count"),
                "current_status_counts": edge.get("current_status_counts") if isinstance(edge.get("current_status_counts"), dict) else {},
                "current_best_reward_jpy": edge.get("current_best_reward_jpy"),
                "spread_normalized_candidate_count": edge.get("spread_normalized_candidate_count"),
                "spread_normalized_no_live_blocker_count": edge.get("spread_normalized_no_live_blocker_count"),
                "top_blockers": [str(item) for item in (edge.get("top_blockers") or [])[:5] if str(item).strip()],
                "strategy_profile_status": edge.get("strategy_profile_status"),
                "strategy_profile_required_fix": edge.get("strategy_profile_required_fix"),
                "strategy_profile_blocks_live": edge.get("strategy_profile_blocks_live"),
                "strategy_profile_live_net_jpy": edge.get("strategy_profile_live_net_jpy"),
                "strategy_profile_pretrade_net_jpy": edge.get("strategy_profile_pretrade_net_jpy"),
                "strategy_profile_seat_net_jpy": edge.get("strategy_profile_seat_net_jpy"),
                "strategy_profile_seat_win_rate_pct": edge.get("strategy_profile_seat_win_rate_pct"),
                "matrix_ref": edge.get("matrix_ref"),
                "matrix_support_count": edge.get("matrix_support_count"),
                "matrix_reject_count": edge.get("matrix_reject_count"),
                "matrix_warning_count": edge.get("matrix_warning_count"),
                "matrix_strongest_support": edge.get("matrix_strongest_support"),
                "matrix_strongest_reject": edge.get("matrix_strongest_reject"),
                "matrix_cross_asset_context": [
                    str(item) for item in (edge.get("matrix_cross_asset_context") or [])[:4] if str(item).strip()
                ],
            }
        )
        if len(top_edges) >= 12:
            break
    return {
        "source_status": payload.get("source_status"),
        "live_permission": False,
        "positive_pair_directions": payload.get("positive_pair_directions"),
        "positive_managed_net_jpy": payload.get("positive_managed_net_jpy"),
        "positive_trade_count": payload.get("positive_trade_count"),
        "state_counts": payload.get("state_counts") if isinstance(payload.get("state_counts"), dict) else {},
        "top_edges": top_edges,
        "matrix_supported_repair_queue": _matrix_supported_repair_queue_packet(
            payload.get("matrix_supported_repair_queue")
            if isinstance(payload.get("matrix_supported_repair_queue"), list)
            else []
        ),
    }


def _matrix_supported_repair_queue_packet(rows: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        pair = str(item.get("pair") or "")
        direction = str(item.get("direction") or "")
        evidence_ref = f"coverage:profitable_bucket:{pair}:{direction}" if pair and direction else None
        out.append(
            {
                "evidence_ref": evidence_ref,
                "pair": pair,
                "direction": direction,
                "coverage_state": item.get("coverage_state"),
                "managed_net_jpy": item.get("managed_net_jpy"),
                "top_blockers": [str(value) for value in (item.get("top_blockers") or [])[:4] if str(value).strip()],
                "strategy_profile_status": item.get("strategy_profile_status"),
                "matrix_ref": item.get("matrix_ref"),
                "matrix_support_count": item.get("matrix_support_count"),
                "matrix_reject_count": item.get("matrix_reject_count"),
                "matrix_support_layers": [
                    str(value) for value in (item.get("matrix_support_layers") or [])[:6] if str(value).strip()
                ],
                "matrix_support_context": [
                    str(value) for value in (item.get("matrix_support_context") or [])[:4] if str(value).strip()
                ],
            }
        )
        if len(out) >= 8:
            break
    return out


def _learning_audit_packet(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {
            "evidence_ref": "learning:audit",
            "status": "missing",
            "blockers": [],
            "warnings": [],
            "learning_influence": {
                "influenced_lanes": 0,
                "total_learning_score_delta": 0.0,
                "lanes": [],
            },
        }
    influence = payload.get("learning_influence") if isinstance(payload.get("learning_influence"), dict) else {}
    lanes: list[dict[str, Any]] = []
    for lane in influence.get("lanes", []) or []:
        if not isinstance(lane, dict):
            continue
        lanes.append(
            {
                "evidence_ref": f"learning:lane:{lane.get('lane_id')}",
                "lane_id": lane.get("lane_id"),
                "learning_influences": list(lane.get("learning_influences", []) or []),
                "learning_score_delta": lane.get("learning_score_delta"),
            }
        )
    effect = payload.get("effect_metrics") if isinstance(payload.get("effect_metrics"), dict) else {}
    effect_packet = _small_dict(
        effect,
        (
            "closed_trades",
            "net_jpy",
            "profit_factor",
            "expectancy_jpy",
        ),
    )
    exit_reason_metrics = _learning_exit_reason_metrics(effect)
    if exit_reason_metrics:
        effect_packet["exit_reason_metrics"] = exit_reason_metrics
    return {
        "evidence_ref": "learning:audit",
        "generated_at_utc": payload.get("generated_at_utc"),
        "status": payload.get("status"),
        "blockers": list(payload.get("blockers", []) or [])[:12],
        "warnings": list(payload.get("warnings", []) or [])[:12],
        "effect_metrics": effect_packet,
        "learning_influence": {
            "influenced_lanes": influence.get("influenced_lanes", 0),
            "total_learning_score_delta": influence.get("total_learning_score_delta", 0.0),
            "lanes": lanes[:20],
        },
    }


def _learning_exit_reason_metrics(effect: dict[str, Any]) -> dict[str, dict[str, Any]]:
    exit_reasons = effect.get("exit_reason_metrics") if isinstance(effect.get("exit_reason_metrics"), dict) else {}
    rows: list[tuple[float, str, dict[str, Any]]] = []
    for reason, metrics in exit_reasons.items():
        reason_key = str(reason or "").strip()
        if not reason_key or not isinstance(metrics, dict):
            continue
        compact = _small_dict(
            metrics,
            (
                "closed_trades",
                "net_jpy",
                "gross_profit_jpy",
                "gross_loss_jpy",
                "profit_factor",
                "win_rate",
                "expectancy_jpy",
            ),
        )
        compact["evidence_ref"] = f"learning:exit_reason:{reason_key}"
        net = _optional_float(metrics.get("net_jpy"))
        rows.append((net if net is not None else 0.0, reason_key, compact))
    return {reason: compact for _net, reason, compact in sorted(rows, key=lambda item: item[0])[:8]}


def _verification_ledger_packet(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {
            "evidence_ref": "verification:ledger",
            "status": "missing",
            "blocking_observations": 0,
            "missing_observations": 0,
            "effect_metrics": {},
            "blocking_evidence": [],
            "learning_evidence": [],
        }
    effect = payload.get("effect_metrics") if isinstance(payload.get("effect_metrics"), dict) else {}
    return {
        "evidence_ref": "verification:ledger",
        "generated_at_utc": payload.get("generated_at_utc"),
        "status": payload.get("status"),
        "db_path": payload.get("db_path"),
        "report_path": payload.get("report_path"),
        "blocking_observations": payload.get("blocking_observations"),
        "missing_observations": payload.get("missing_observations"),
        "effect_metrics": _small_dict(
            effect,
            (
                "window_hours",
                "closed_trades",
                "net_jpy",
                "profit_factor",
                "win_rate",
                "expectancy_jpy",
                "sample_warning",
            ),
        ),
        "blocking_evidence": _verification_rows(payload.get("blocking_evidence")),
        "missing_artifacts": _verification_rows(payload.get("missing_artifacts")),
        "learning_evidence": _verification_rows(payload.get("learning_evidence")),
        "measurements": _verification_rows(payload.get("measurements")),
        "contract": _small_dict(
            payload.get("contract"),
            (
                "read_only",
                "live_permission",
                "sqlite_tables",
                "json_packet_is_trader_readable",
                "markdown_report_is_operator_readable",
                "learning_cannot_override_risk_or_gateway_gates",
            ),
        ),
    }


def _verification_rows(rows: object) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not isinstance(rows, list):
        return out
    for item in rows[:12]:
        if not isinstance(item, dict):
            continue
        out.append(
            _small_dict(
                item,
                (
                    "evidence_ref",
                    "source",
                    "subject_type",
                    "subject_id",
                    "check_name",
                    "status",
                    "severity",
                    "metric_value",
                    "metric_unit",
                    "evidence",
                ),
            )
        )
    return out


def _self_improvement_audit_packet(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {
            "evidence_ref": "self_improvement:audit",
            "status": "missing",
            "p0_findings": 0,
            "p0_blockers": [],
            "profitability_blockers": [],
        }
    blockers: list[dict[str, Any]] = []
    p0_blockers: list[dict[str, Any]] = []
    for finding in payload.get("findings", []) or []:
        if not isinstance(finding, dict):
            continue
        code = str(finding.get("code") or "")
        priority = str(finding.get("priority") or "").upper()
        layer = str(finding.get("layer") or "")
        evidence = finding.get("evidence") if isinstance(finding.get("evidence"), dict) else {}
        if priority == "P0":
            p0_blockers.append(
                {
                    "evidence_ref": f"self_improvement:finding:{code}",
                    "code": code,
                    "layer": layer,
                    "message": finding.get("message"),
                    "next_action": finding.get("next_action"),
                    "current_streak": evidence.get("current_streak"),
                    "count": evidence.get("count"),
                    "examples": list(evidence.get("examples", []) or [])[:3],
                }
            )
        if priority == "P0" and layer == "profitability" and code == "PERSISTENT_PROFITABILITY_DISCIPLINE_BLOCKED":
            evidence = finding.get("evidence") if isinstance(finding.get("evidence"), dict) else {}
            system_evidence = evidence.get("system_defect_evidence") if isinstance(evidence.get("system_defect_evidence"), dict) else {}
            blockers.append(
                {
                    "evidence_ref": f"self_improvement:finding:{code}",
                    "code": code,
                    "message": finding.get("message"),
                    "next_action": finding.get("next_action"),
                    "current_streak": evidence.get("current_streak"),
                    "profit_factor": system_evidence.get("profit_factor"),
                    "expectancy_jpy": system_evidence.get("expectancy_jpy"),
                    "avg_win_jpy": system_evidence.get("avg_win_jpy"),
                    "avg_loss_jpy_abs": system_evidence.get("avg_loss_jpy_abs"),
                    "worst_segments": list(system_evidence.get("worst_segments", []) or [])[:5],
                }
            )
    return {
        "evidence_ref": "self_improvement:audit",
        "generated_at_utc": payload.get("generated_at_utc"),
        "status": payload.get("status"),
        "p0_findings": payload.get("p0_findings"),
        "p1_findings": payload.get("p1_findings"),
        "p2_findings": payload.get("p2_findings"),
        "effect_metrics": _small_dict(
            payload.get("effect_metrics"),
            (
                "closed_trades",
                "net_jpy",
                "profit_factor",
                "expectancy_jpy",
                "avg_win_jpy",
                "avg_loss_jpy_abs",
            ),
        ),
        "p0_blockers": p0_blockers,
        "profitability_blockers": blockers,
    }


def _operator_precedent_packet(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {
            "evidence_ref": OPERATOR_PRECEDENT_EVIDENCE_REF,
            "status": "missing",
            "operator_claim": {},
            "winning_shape": {},
            "runtime_alignment": {},
            "warnings": [],
            "blockers": [],
        }
    precedent = payload.get("precedent") if isinstance(payload.get("precedent"), dict) else {}
    return {
        "evidence_ref": OPERATOR_PRECEDENT_EVIDENCE_REF,
        "generated_at_utc": payload.get("generated_at_utc"),
        "status": payload.get("status"),
        "operator_claim": _small_dict(
            payload.get("operator_claim"),
            ("claim", "required_return_pct", "verified"),
        ),
        "winning_shape": _small_dict(
            precedent.get("winning_shape"),
            (
                "primary_pair",
                "primary_direction",
                "primary_sessions",
                "positive_sessions",
                "negative_sessions",
                "expectancy_jpy_per_exit",
                "median_hold_hours",
                "payoff",
            ),
        ),
        "failure_shape": _small_dict(
            (precedent.get("failure_shape") or {}).get("margin_closeout"),
            ("trades", "net_jpy", "win_rate", "median_hold_hours"),
        ),
        "runtime_alignment": _small_dict(
            payload.get("runtime_alignment"),
            (
                "live_ready_lanes",
                "aligned_live_ready_lanes",
                "aligned_lanes",
                "manual_context_alignment",
                "manual_exit_events_per_calendar_day",
                "target_trades_per_day",
                "alignment_contract",
            ),
        ),
        "warnings": list(payload.get("warnings") or [])[:5],
        "blockers": list(payload.get("blockers") or [])[:5],
    }


def _manual_market_context_packet(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {
            "evidence_ref": MANUAL_MARKET_CONTEXT_EVIDENCE_REF,
            "status": "missing",
            "sample": {},
            "guidance": {},
            "bounded_replay_profile": {},
            "excluded_tail_profile": {},
            "position_building": {},
            "warnings": [],
            "blockers": [],
        }
    bounded = payload.get("bounded_replay_profile") if isinstance(payload.get("bounded_replay_profile"), dict) else {}
    excluded = payload.get("excluded_tail_profile") if isinstance(payload.get("excluded_tail_profile"), dict) else {}
    building = payload.get("position_building_profile") if isinstance(payload.get("position_building_profile"), dict) else {}
    return {
        "evidence_ref": MANUAL_MARKET_CONTEXT_EVIDENCE_REF,
        "generated_at_utc": payload.get("generated_at_utc"),
        "status": payload.get("status"),
        "sample": _small_dict(
            payload.get("sample"),
            ("pair", "manual_trades", "analyzed_trades", "coverage_pct"),
        ),
        "guidance": payload.get("guidance") if isinstance(payload.get("guidance"), dict) else {},
        "bounded_replay_profile": {
            "overall": bounded.get("overall") if isinstance(bounded.get("overall"), dict) else {},
            "by_h1_alignment": _profile_rows(bounded.get("by_h1_alignment")),
            "by_side_h1_alignment": _profile_rows(bounded.get("by_side_h1_alignment")),
            "by_side_entry_location_24h": _profile_rows(bounded.get("by_side_entry_location_24h")),
            "by_session_jst": _profile_rows(bounded.get("by_session_jst")),
        },
        "excluded_tail_profile": {
            "by_hold_bucket": _profile_rows(excluded.get("by_hold_bucket")),
            "by_close_reason": _profile_rows(excluded.get("by_close_reason")),
        },
        "position_building": {
            "basis": building.get("basis"),
            "overall": _small_dict(
                building.get("overall"),
                (
                    "clusters",
                    "multi_entry_clusters",
                    "entries",
                    "net_jpy",
                    "win_rate",
                    "expectancy_jpy",
                    "max_entries",
                    "adverse_adds",
                    "pyramid_adds",
                    "avg_adverse_add_pips",
                ),
            ),
            "bounded_lt_12h_excluding_margin_closeout": _small_dict(
                building.get("bounded_lt_12h_excluding_margin_closeout"),
                (
                    "clusters",
                    "multi_entry_clusters",
                    "entries",
                    "net_jpy",
                    "win_rate",
                    "expectancy_jpy",
                    "max_entries",
                    "adverse_adds",
                    "pyramid_adds",
                    "avg_adverse_add_pips",
                ),
            ),
            "adverse_adds": _small_dict(
                building.get("adverse_adds"),
                (
                    "clusters",
                    "entries",
                    "net_jpy",
                    "win_rate",
                    "expectancy_jpy",
                    "max_entries",
                    "adverse_adds",
                    "avg_adverse_add_pips",
                ),
            ),
            "bounded_by_build_type": _position_building_rows(building.get("bounded_by_build_type")),
            "largest_adverse_add_winners": _position_building_examples(
                ((building.get("examples") or {}).get("largest_adverse_add_winners"))
                if isinstance(building.get("examples"), dict)
                else None
            ),
            "contract": _small_dict(
                building.get("contract"),
                (
                    "advisory_only",
                    "nanpin_is_not_live_permission",
                    "requires_current_basket_risk_validation",
                    "forbidden_to_use_for_unbounded_martingale",
                ),
            ),
        },
        "contract": _small_dict(
            payload.get("contract"),
            (
                "advisory_only",
                "may_gate_use_of_operator_precedent_as_aggression_reason",
                "does_not_override_current_risk_geometry",
                "does_not_grant_live_permission",
            ),
        ),
        "warnings": list(payload.get("warnings") or [])[:5],
        "blockers": list(payload.get("blockers") or [])[:5],
    }


def _position_building_rows(rows: object, *, limit: int = 8) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        return []
    out: list[dict[str, Any]] = []
    for item in rows[:limit]:
        if not isinstance(item, dict):
            continue
        out.append(
            _small_dict(
                item,
                (
                    "bucket",
                    "clusters",
                    "multi_entry_clusters",
                    "entries",
                    "net_jpy",
                    "win_rate",
                    "expectancy_jpy",
                    "median_entries",
                    "max_entries",
                    "adverse_adds",
                    "pyramid_adds",
                    "avg_adverse_add_pips",
                ),
            )
        )
    return out


def _position_building_examples(rows: object, *, limit: int = 5) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        return []
    out: list[dict[str, Any]] = []
    for item in rows[:limit]:
        if not isinstance(item, dict):
            continue
        out.append(
            _small_dict(
                item,
                (
                    "cluster_id",
                    "side",
                    "build_type",
                    "entries",
                    "trade_ids",
                    "session_jst",
                    "hold_hours",
                    "realized_pl",
                    "initial_price",
                    "final_weighted_avg",
                    "adverse_add_count",
                    "pyramid_add_count",
                    "close_reasons",
                ),
            )
        )
    return out


def _profile_rows(rows: object, *, limit: int = 8) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        return []
    out: list[dict[str, Any]] = []
    for item in rows[:limit]:
        if not isinstance(item, dict):
            continue
        out.append(
            _small_dict(
                item,
                (
                    "bucket",
                    "trades",
                    "net_jpy",
                    "win_rate",
                    "expectancy_jpy",
                    "median_hold_hours",
                    "avg_h1_adx",
                    "median_entry_price_percentile_24h",
                ),
            )
        )
    return out


def _predictive_limits_packet(payload: dict[str, Any] | None, *, pairs: set[str]) -> dict[str, Any]:
    if not payload:
        return {"evidence_ref": "predictive:limits", "status": "missing", "orders_count": 0}
    orders_in = [item for item in payload.get("orders", []) or [] if isinstance(item, dict)]
    relevant_pairs = set(pairs)

    def priority(item: dict[str, Any]) -> tuple[int, int]:
        grade = str(item.get("grade") or "").upper()
        pair = str(item.get("pair") or "")
        return (0 if grade == "A" else 1, 0 if pair in relevant_pairs else 1)

    selected = sorted(orders_in, key=priority)[:12]
    return {
        "evidence_ref": "predictive:limits",
        "status": "DRY_RUN" if payload.get("dry_run", True) else "SENT_OR_ATTEMPTED",
        "generated_at_utc": payload.get("generated_at_utc"),
        "dry_run": payload.get("dry_run"),
        "orders_count": len(orders_in),
        "orders": [
            {
                "evidence_ref": f"predictive:limit:{item.get('pair')}:{item.get('side')}",
                "pair": item.get("pair"),
                "side": item.get("side"),
                "grade": item.get("grade"),
                "limit_price": item.get("limit_price"),
                "take_profit_price": item.get("take_profit_price"),
                "units": item.get("units"),
                "source": item.get("source"),
                "gtd_utc": item.get("gtd_utc"),
                "rationale": item.get("rationale"),
            }
            for item in selected
        ],
    }


def _block_issues(items: object) -> list[str]:
    blockers: list[str] = []
    for item in items or []:
        if isinstance(item, dict) and item.get("severity") == "BLOCK":
            blockers.append(str(item.get("message") or item.get("code") or "block"))
    return blockers


def _small_dict(payload: object, keys: tuple[str, ...]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    return {key: payload.get(key) for key in keys if key in payload}


def _lane_forecast_packet(metadata: object) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        return {}
    return _small_dict(
        metadata,
        (
            "forecast_direction",
            "forecast_confidence",
            "forecast_target_price",
            "forecast_invalidation_price",
            "forecast_horizon_min",
            "forecast_rationale",
        ),
    )


def _has_pending_entry_order(packet: dict[str, Any]) -> bool:
    return bool(_pending_entry_order_ids(packet))


def _pending_entry_order_ids(packet: dict[str, Any]) -> list[str]:
    snapshot = packet.get("broker_snapshot", {})
    order_ids: list[str] = []
    for order in snapshot.get("pending_orders", []) or []:
        owner = str(order.get("owner") or "")
        if owner in {"manual", "unknown"}:
            continue
        if order.get("trade_id"):
            continue
        order_type = str(order.get("order_type") or "").upper()
        order_id = str(order.get("order_id") or "")
        if order_id and order_type in {"LIMIT", "STOP", "MARKET_IF_TOUCHED", "MARKET_IF_TOUCHED_ORDER"}:
            order_ids.append(order_id)
    return order_ids


def _trade_exposure_blockers(packet: dict[str, Any]) -> list[str]:
    snapshot = packet.get("broker_snapshot", {})
    blockers: list[str] = []
    sl_free_active = _trader_sl_repair_disabled()
    for position in snapshot.get("position_summaries", []) or []:
        owner = str(position.get("owner") or "")
        if owner in {"manual", "unknown"}:
            continue
        # SL-free regime: trader-owned SL=None is intentional, and missing
        # broker TP is a no-broker-TP runner unless repair is explicitly
        # enabled. Exposure still reaches LiveOrderGateway for margin,
        # hedging, and portfolio validation.
        sl_ok = position.get("stop_loss") is not None or sl_free_active
        tp_ok = position.get("take_profit") is not None or (
            owner == "trader" and sl_free_active and not _missing_tp_repair_enabled()
        )
        if owner == "trader" and tp_ok and sl_ok:
            continue
        blockers.append(
            f"non-layerable position {position.get('pair')} {position.get('side')} id={position.get('trade_id')}"
        )
    return blockers


def _trader_exposure_present(packet: dict[str, Any]) -> bool:
    snapshot = packet.get("broker_snapshot", {})
    for position in snapshot.get("position_summaries", []) or []:
        if str(position.get("owner") or "") == "trader":
            return True
    return _has_pending_entry_order(packet)


def _target_requires_entry(packet: dict[str, Any]) -> bool:
    target = packet.get("daily_target", {})
    remaining = target.get("remaining_target_jpy")
    try:
        return float(remaining or 0.0) > 0.0 and target.get("status") != "TARGET_REACHED_PROTECT"
    except (TypeError, ValueError):
        return False


def _tp_rebalance_sidecar_reasons(packet: dict[str, Any]) -> list[str]:
    sidecars = packet.get("protection_sidecars")
    if not isinstance(sidecars, dict):
        return []
    tp_rebalance = sidecars.get("tp_rebalance")
    if not isinstance(tp_rebalance, dict) or not tp_rebalance.get("required"):
        return []
    return [
        str(reason)
        for reason in (tp_rebalance.get("reasons") or [])
        if str(reason).strip()
    ]


def _entry_thesis_sidecar_reasons(packet: dict[str, Any]) -> list[str]:
    sidecars = packet.get("protection_sidecars")
    if not isinstance(sidecars, dict):
        return []
    blockers = sidecars.get("entry_thesis_blockers")
    if not isinstance(blockers, list):
        return []
    reasons: list[str] = []
    for item in blockers:
        if not isinstance(item, dict):
            continue
        trade_id = str(item.get("trade_id") or "").strip()
        pair = str(item.get("pair") or "").strip()
        side = str(item.get("side") or "").strip()
        reason = str(item.get("reason") or "original entry thesis is not machine-verifiable").strip()
        label = " ".join(part for part in (pair, side, f"id={trade_id}" if trade_id else "") if part)
        reasons.append(f"{label}: {reason}" if label else reason)
    return reasons


def _position_close_sidecar_reasons(
    packet: dict[str, Any],
    *,
    blocking_only: bool = True,
) -> list[str]:
    sidecars = packet.get("protection_sidecars")
    if not isinstance(sidecars, dict):
        return []
    recommendations = sidecars.get("position_close_recommendations")
    if not isinstance(recommendations, list):
        return []
    reasons: list[str] = []
    for item in recommendations:
        if not isinstance(item, dict):
            continue
        blocks_non_close_action = _position_close_recommendation_blocks_non_close_action(packet, item)
        if blocking_only and not blocks_non_close_action:
            continue
        if not blocking_only and blocks_non_close_action:
            continue
        trade_id = str(item.get("trade_id") or "").strip()
        pair = str(item.get("pair") or "").strip()
        side = str(item.get("side") or "").strip()
        source = str(item.get("source") or "position_sidecar").strip()
        verdict = str(item.get("verdict") or "RECOMMEND_CLOSE").strip()
        reason = str(item.get("reason") or "position recovery edge is broken").strip()
        label = " ".join(part for part in (pair, side, f"id={trade_id}" if trade_id else "") if part)
        prefix = f"{source} {verdict}"
        reasons.append(f"{prefix} {label}: {reason}" if label else f"{prefix}: {reason}")
    return reasons


def _soft_nonblocking_close_advisory_reasons(
    packet: dict[str, Any],
    trade_ids: list[str] | tuple[str, ...],
) -> list[str]:
    sidecars = packet.get("protection_sidecars")
    if not isinstance(sidecars, dict):
        return []
    recommendations = sidecars.get("position_close_recommendations")
    if not isinstance(recommendations, list):
        return []
    selected = {str(trade_id) for trade_id in trade_ids if str(trade_id)}
    reasons: list[str] = []
    for item in recommendations:
        if not isinstance(item, dict):
            continue
        trade_id = str(item.get("trade_id") or "").strip()
        if selected and trade_id not in selected:
            continue
        if _position_close_recommendation_blocks_non_close_action(packet, item):
            continue
        pair = str(item.get("pair") or "").strip()
        side = str(item.get("side") or "").strip()
        source = str(item.get("source") or "position_sidecar").strip()
        verdict = str(item.get("verdict") or "RECOMMEND_CLOSE").strip()
        reason = str(
            item.get("non_blocking_reason")
            or item.get("reason")
            or "soft close evidence does not block non-CLOSE actions"
        ).strip()
        label = " ".join(part for part in (pair, side, f"id={trade_id}" if trade_id else "") if part)
        reasons.append(f"{source} {verdict} {label}: {reason}" if label else f"{source} {verdict}: {reason}")
    return reasons


def _position_close_recommendation_blocks_non_close_action(
    packet: dict[str, Any],
    rec: dict[str, Any],
) -> bool:
    """Whether a close sidecar must preempt TRADE/WAIT/PROTECT.

    Hard Gate A, or soft Gate A with explicit operator Gate B, requires a CLOSE
    receipt first. Soft-only sidecars remain usable Gate A evidence if the
    trader chooses CLOSE, but they do not freeze TP-managed exposure and block
    fresh all-horizon entries while authorization is absent.
    """
    if bool(rec.get("gate_b_standing_authorized")):
        if _sidecar_hold_support_conflict(packet, rec):
            return _operator_close_gate_authorized()
        return True
    trade_id = str(rec.get("trade_id") or "")
    if trade_id:
        pos = _trader_position_lookup(packet).get(trade_id)
        if pos is not None:
            pair = str(pos.get("pair") or rec.get("pair") or "")
            side = str(pos.get("side") or rec.get("side") or "")
            invalidated, _reason, standing_authorized = _close_thesis_invalidation(
                packet,
                pair,
                side,
                trade_id=trade_id,
                decision=None,
            )
            if invalidated and standing_authorized:
                return True
    return _operator_close_gate_authorized()


def _append_position_close_required_issue(
    issues: list[VerificationIssue],
    *,
    packet: dict[str, Any],
    action: str,
    reasons: list[str],
) -> None:
    reason_text = "; ".join(reasons[:3])
    if _operator_close_gate_authorized() or _standing_close_authorization_available(packet):
        issues.append(
            VerificationIssue(
                "POSITION_CLOSE_REQUIRED",
                f"{action} rejected while fresh position sidecar Gate A close evidence exists and close authorization is available. "
                "Emit action=CLOSE for the named trade(s) first; after the accepted CLOSE receipt is handled, "
                "end that cycle and refresh broker truth / intents on the next scheduled cycle before any separate TRADE receipt: "
                f"{reason_text}",
            )
        )
        return
    issues.append(
        VerificationIssue(
            "CLOSE_OPERATOR_AUTH_REQUIRED",
            f"{action} rejected while fresh position sidecar Gate A close evidence exists, but Gate B "
            "operator authorization is missing for the softer close evidence. This is not a recovery-confidence hold; require "
            "QR_OPERATOR_CLOSE_OVERRIDE=1 in the operator shell or a fresh data/.operator_close_token "
            f"before a loss-side CLOSE can be verified: {reason_text}",
        )
    )


def _standing_close_authorization_available(packet: dict[str, Any]) -> bool:
    position_by_tid = _trader_position_lookup(packet)
    sidecars = packet.get("protection_sidecars")
    recs = sidecars.get("position_close_recommendations") if isinstance(sidecars, dict) else None
    if not isinstance(recs, list):
        return False
    for rec in recs:
        if not isinstance(rec, dict):
            continue
        trade_id = str(rec.get("trade_id") or "")
        pos = position_by_tid.get(trade_id)
        if pos is None:
            continue
        pair = str(pos.get("pair") or rec.get("pair") or "")
        side = str(pos.get("side") or rec.get("side") or "")
        invalidated, _reason, standing_authorized = _close_thesis_invalidation(
            packet,
            pair,
            side,
            trade_id=trade_id,
            decision=None,
        )
        if invalidated and standing_authorized:
            return True
    return False


def _tradeable_live_ready_lanes(packet: dict[str, Any]) -> list[str]:
    lanes: list[str] = []
    for lane in packet.get("lanes", []) or []:
        if not isinstance(lane, dict):
            continue
        if lane.get("status") != "LIVE_READY":
            continue
        if lane.get("risk_blockers") or lane.get("strategy_blockers") or lane.get("live_blockers"):
            continue
        lane_id = str(lane.get("lane_id") or "")
        if lane_id:
            lanes.append(lane_id)
    return lanes


def _attack_recommended_tradeable_lane_ids(
    packet: dict[str, Any],
    tradeable_lanes: list[str] | None = None,
) -> list[str]:
    advice = packet.get("ai_attack_advice")
    if not isinstance(advice, dict):
        return []
    current = set(tradeable_lanes if tradeable_lanes is not None else _tradeable_live_ready_lanes(packet))
    lane_ids: list[str] = []
    for raw_lane_id in advice.get("recommended_now_lane_ids", []) or []:
        lane_id = str(raw_lane_id or "")
        learning_allowed, _reason = _learning_audit_allows_influenced_lane(packet, lane_id)
        if lane_id and lane_id in current and lane_id not in lane_ids and learning_allowed:
            lane_ids.append(lane_id)
    return lane_ids


def _attack_lane_learning_influenced(packet: dict[str, Any], lane_id: str) -> bool:
    advice = packet.get("ai_attack_advice")
    if not isinstance(advice, dict):
        return False
    influenced = {str(item) for item in advice.get("learning_influenced_lane_ids", []) or []}
    if lane_id in influenced:
        return True
    for lane in advice.get("recommended_lane_learning", []) or []:
        if not isinstance(lane, dict) or str(lane.get("lane_id") or "") != lane_id:
            continue
        return bool([item for item in (lane.get("learning_influences") or []) if str(item).strip()])
    return False


def _learning_audit_lane_ids(packet: dict[str, Any]) -> set[str]:
    audit = packet.get("learning_audit")
    if not isinstance(audit, dict):
        return set()
    influence = audit.get("learning_influence") if isinstance(audit.get("learning_influence"), dict) else {}
    out: set[str] = set()
    for lane in influence.get("lanes", []) or []:
        if not isinstance(lane, dict):
            continue
        lane_id = str(lane.get("lane_id") or "")
        if lane_id:
            out.add(lane_id)
    return out


def _learning_audit_allows_influenced_lane(packet: dict[str, Any], lane_id: str) -> tuple[bool, str]:
    if not _attack_lane_learning_influenced(packet, lane_id):
        return True, ""
    audit = packet.get("learning_audit")
    if not isinstance(audit, dict):
        return False, "learning audit packet is missing"
    status = str(audit.get("status") or "missing")
    if status in {"", "missing"}:
        return False, "learning audit is missing"
    if status == "LEARNING_AUDIT_BLOCKED":
        blockers = [str(item) for item in (audit.get("blockers") or []) if str(item).strip()]
        suffix = f": {'; '.join(blockers[:3])}" if blockers else ""
        return False, f"learning audit is blocked{suffix}"
    if lane_id not in _learning_audit_lane_ids(packet):
        return False, f"learning audit does not cover influenced lane {lane_id}"
    return True, ""


def _learning_audit_trade_issues(
    packet: dict[str, Any],
    selected_lane_ids: tuple[str, ...],
    evidence_refs: tuple[str, ...],
) -> list[VerificationIssue]:
    issues: list[VerificationIssue] = []
    refs = set(evidence_refs)
    for lane_id in selected_lane_ids:
        if not _attack_lane_learning_influenced(packet, lane_id):
            continue
        allowed, reason = _learning_audit_allows_influenced_lane(packet, lane_id)
        if not allowed:
            code = "LEARNING_AUDIT_BLOCKED"
            audit = packet.get("learning_audit")
            status = str(audit.get("status") or "") if isinstance(audit, dict) else ""
            if status in {"", "missing"}:
                code = "LEARNING_AUDIT_REQUIRED"
            elif "does not cover" in reason:
                code = "LEARNING_AUDIT_STALE"
            issues.append(
                VerificationIssue(
                    code,
                    f"TRADE rejected for learning-influenced lane {lane_id}: {reason}",
                )
            )
            continue
        if "learning:audit" not in refs:
            issues.append(
                VerificationIssue(
                    "LEARNING_AUDIT_EVIDENCE_MISSING",
                    f"TRADE selecting learning-influenced lane {lane_id} must cite learning:audit",
                )
            )
        lane_ref = f"learning:lane:{lane_id}"
        if lane_ref not in refs:
            issues.append(
                VerificationIssue(
                    "LEARNING_LANE_EVIDENCE_MISSING",
                    f"TRADE selecting learning-influenced lane {lane_id} must cite {lane_ref}",
                )
            )
    return issues


def _manual_precedent_trade_issues(
    packet: dict[str, Any],
    selected_lane_ids: tuple[str, ...],
    evidence_refs: tuple[str, ...],
) -> list[VerificationIssue]:
    refs = set(evidence_refs)
    if OPERATOR_PRECEDENT_EVIDENCE_REF not in refs:
        return []
    issues: list[VerificationIssue] = []
    if MANUAL_MARKET_CONTEXT_EVIDENCE_REF not in refs:
        issues.append(
            VerificationIssue(
                "MANUAL_CONTEXT_EVIDENCE_MISSING",
                "TRADE citing the 2025 operator precedent must also cite "
                f"{MANUAL_MARKET_CONTEXT_EVIDENCE_REF} so the manual technical context is checked.",
            )
        )
    precedent = packet.get("operator_precedent")
    if not isinstance(precedent, dict) or precedent.get("status") == "missing":
        issues.append(
            VerificationIssue(
                "OPERATOR_PRECEDENT_PACKET_MISSING",
                "TRADE cites operator precedent but the decision packet has no readable operator precedent audit.",
            )
        )
        return issues
    claim = precedent.get("operator_claim") if isinstance(precedent.get("operator_claim"), dict) else {}
    if claim.get("verified") is not True:
        issues.append(
            VerificationIssue(
                "OPERATOR_PRECEDENT_UNVERIFIED",
                "TRADE cites operator precedent but the manual-history claim is not verified in the audit packet.",
            )
        )
    manual_context = packet.get("manual_market_context")
    if not isinstance(manual_context, dict) or manual_context.get("status") == "missing":
        issues.append(
            VerificationIssue(
                "MANUAL_CONTEXT_PACKET_MISSING",
                "TRADE cites operator precedent but the decision packet has no readable manual market-context audit.",
            )
        )
    elif str(manual_context.get("status") or "") != "MANUAL_MARKET_CONTEXT_PASS":
        issues.append(
            VerificationIssue(
                "MANUAL_CONTEXT_NOT_PASSING",
                "TRADE cites operator precedent but manual-market-context audit is not passing: "
                f"{manual_context.get('status')}",
            )
        )

    runtime = precedent.get("runtime_alignment") if isinstance(precedent.get("runtime_alignment"), dict) else {}
    aligned_lane_ids = {
        str(item.get("lane_id") or "")
        for item in (runtime.get("aligned_lanes") or [])
        if isinstance(item, dict) and str(item.get("lane_id") or "").strip()
    }
    if not selected_lane_ids:
        return issues
    if not aligned_lane_ids:
        issues.append(
            VerificationIssue(
                "OPERATOR_PRECEDENT_NO_CURRENT_ALIGNMENT",
                "TRADE cites the 2025 operator precedent, but the current operator-precedent audit has no "
                "LIVE_READY lane aligned to the manual pair/direction/session shape. Cite current deterministic "
                "edge instead of using the manual precedent as an aggression reason.",
            )
        )
        return issues
    selected_aligned = [lane_id for lane_id in selected_lane_ids if lane_id in aligned_lane_ids]
    if not selected_aligned:
        issues.append(
            VerificationIssue(
                "OPERATOR_PRECEDENT_SELECTED_LANE_NOT_ALIGNED",
                "TRADE cites the 2025 operator precedent, but none of the selected lane(s) are aligned to the "
                "manual precedent shape: "
                f"selected={', '.join(selected_lane_ids)} aligned={', '.join(sorted(aligned_lane_ids))}. "
                "Use current forecast/risk/matrix evidence for this trade instead.",
            )
        )
        return issues
    lane_by_id = {
        str(lane.get("lane_id") or ""): lane
        for lane in packet.get("lanes", []) or []
        if isinstance(lane, dict) and str(lane.get("lane_id") or "").strip()
    }
    selected_with_move_adds = []
    for lane_id in selected_aligned:
        lane = lane_by_id.get(lane_id) or {}
        building = lane.get("position_building") if isinstance(lane.get("position_building"), dict) else {}
        if str(building.get("same_pair_add_type") or "").upper() == "PYRAMID_WITH_MOVE":
            selected_with_move_adds.append(lane_id)
    if selected_with_move_adds:
        issues.append(
            VerificationIssue(
                "OPERATOR_PRECEDENT_POSITION_BUILDING_CONFLICT",
                "TRADE cites the 2025 operator precedent, but the selected same-pair add is "
                "PYRAMID_WITH_MOVE. The bounded manual replay supports only selective adverse "
                "retest/add behavior as precedent; cite current deterministic edge instead: "
                f"{', '.join(selected_with_move_adds)}",
            )
        )
    manual_alignment = (
        runtime.get("manual_context_alignment")
        if isinstance(runtime.get("manual_context_alignment"), dict)
        else {}
    )
    conflicting_rows = [
        row
        for row in (manual_alignment.get("conflicting_lanes") or [])
        if isinstance(row, dict) and str(row.get("lane_id") or "").strip()
    ]
    conflicting_lane_ids = {str(row.get("lane_id") or "") for row in conflicting_rows}
    selected_conflicting = [lane_id for lane_id in selected_aligned if lane_id in conflicting_lane_ids]
    if selected_conflicting:
        conflict_details = [
            f"{row.get('lane_id')}:{','.join(str(bucket) for bucket in (row.get('conflicting_buckets') or []))}"
            for row in conflicting_rows
            if str(row.get("lane_id") or "") in selected_conflicting
        ]
        issues.append(
            VerificationIssue(
                "OPERATOR_PRECEDENT_TECHNICAL_CONTEXT_CONFLICT",
                "TRADE cites the 2025 operator precedent, but the selected lane conflicts with the bounded manual "
                "technical replay context. Cite current deterministic edge instead, or choose a lane whose H1/M5/"
                f"24h-location context matches the manual precedent: {'; '.join(conflict_details)}",
            )
        )
    return issues


def _self_improvement_trade_blockers(
    packet: dict[str, Any],
    *,
    decision_generated_at_utc: str | None = None,
    include_decision_history_stale: bool = True,
) -> list[str]:
    audit = packet.get("self_improvement_audit")
    if not isinstance(audit, dict):
        return []
    audit_generated_at = _parse_utc(audit.get("generated_at_utc"))
    receipt_generated_at = (
        _parse_utc(decision_generated_at_utc) if decision_generated_at_utc else None
    )
    out: list[str] = []
    blockers = audit.get("p0_blockers", []) or audit.get("profitability_blockers", []) or []
    for blocker in blockers:
        if not isinstance(blocker, dict):
            continue
        code = str(blocker.get("code") or "SELF_IMPROVEMENT_P0")
        if code == "LATEST_GPT_DECISION_STALE" and not include_decision_history_stale:
            continue
        if (
            code == "LATEST_GPT_DECISION_STALE"
            and receipt_generated_at is not None
            and audit_generated_at is not None
            and receipt_generated_at > audit_generated_at
        ):
            # The audit predates the receipt under verification, so its
            # stale-decision verdict is about an older receipt. Writing one
            # current receipt is exactly the repair that finding demands;
            # rejecting the repair receipt with its own staleness streak
            # would re-create the deadlock the streak exemption documents.
            continue
        if _self_improvement_non_trade_blocker(code, blocker):
            continue
        layer = str(blocker.get("layer") or "").strip()
        message = str(blocker.get("message") or "").strip()
        streak = blocker.get("current_streak")
        pf = blocker.get("profit_factor")
        expectancy = blocker.get("expectancy_jpy")
        avg_loss = blocker.get("avg_loss_jpy_abs")
        avg_win = blocker.get("avg_win_jpy")
        count = blocker.get("count")
        details = []
        if layer:
            details.append(f"layer={layer}")
        if streak is not None:
            details.append(f"streak={streak}")
        if count is not None:
            details.append(f"count={count}")
        if pf is not None:
            details.append(f"PF={pf}")
        if expectancy is not None:
            details.append(f"expectancy={expectancy}")
        if avg_loss is not None and avg_win is not None:
            details.append(f"avg_loss={avg_loss} vs avg_win={avg_win}")
        suffix = f" ({', '.join(details)})" if details else ""
        out.append(f"{code}{suffix}: {message or 'self-improvement P0 blocks new risk'}")
    return out


def _self_improvement_non_trade_blocker(code: str, blocker: dict[str, Any]) -> bool:
    if code not in SELF_IMPROVEMENT_NON_TRADE_BLOCKER_CODES:
        return False
    streak = _optional_int(blocker.get("current_streak"))
    if streak is None:
        return True
    return streak < SELF_IMPROVEMENT_STALE_DECISION_PERSISTENT_STREAK


# A single stale prior GPT decision means "rewrite/verify the receipt against
# the latest packet"; blocking that fresh verifier pass is circular. Once the
# same finding persists across audit runs, it is no longer just repair-in-flight:
# new risk must stop until the route proves the stale decision is cleared.
SELF_IMPROVEMENT_NON_TRADE_BLOCKER_CODES = frozenset({"LATEST_GPT_DECISION_STALE"})
SELF_IMPROVEMENT_STALE_DECISION_PERSISTENT_STREAK = 2


def _lane_forecast_direction_issue(lane: dict[str, Any]) -> VerificationIssue | None:
    forecast = lane.get("forecast")
    if not isinstance(forecast, dict):
        return None
    direction = str(forecast.get("forecast_direction") or "").upper()
    if direction not in {"UP", "DOWN"}:
        return None
    confidence = _optional_float(forecast.get("forecast_confidence"))
    if confidence is None or confidence < _forecast_confidence_min():
        return None
    forecast_side = "LONG" if direction == "UP" else "SHORT"
    lane_side = str(lane.get("direction") or "").upper()
    if lane_side == forecast_side:
        return None
    return VerificationIssue(
        "FORECAST_DIRECTION_CONFLICT",
        (
            f"{lane.get('lane_id')} is {lane_side} but current pair forecast is "
            f"{direction} conf={confidence:.2f}; verifier refuses forecast-opposite TRADE."
        ),
    )


def _forecast_confidence_min() -> float:
    try:
        from quant_rabbit.strategy.directional_forecaster import ENTRY_CONFIDENCE_MIN

        return float(ENTRY_CONFIDENCE_MIN)
    except Exception:
        return 1.0


def _cited_live_ready_lanes(decision: GPTTraderDecision, lane_ids: list[str]) -> list[str]:
    refs = set(decision.evidence_refs)
    return [lane_id for lane_id in lane_ids if f"intent:{lane_id}" in refs]


def _wait_is_session_only(decision: GPTTraderDecision) -> bool:
    text = " ".join(
        part
        for part in (
            decision.thesis,
            decision.narrative,
            decision.chart_story,
            decision.invalidation,
            decision.operator_summary,
            " ".join(decision.rejected_alternatives),
            " ".join(decision.risk_notes),
        )
        if part
    )
    if not SESSION_ONLY_WAIT_PATTERN.search(text):
        return False
    return CONCRETE_WAIT_GATE_PATTERN.search(text) is None


def _pair_from_lane_id(lane_id: str) -> str:
    """Extract the pair token from a `desk:pair:side:method[:MARKET]` lane id."""
    if not lane_id:
        return ""
    parts = lane_id.split(":")
    if len(parts) >= 2 and parts[1]:
        return parts[1]
    return ""


def _pairs_from_lanes(lanes: list[dict[str, Any]]) -> tuple[str, ...]:
    return tuple(sorted({str(lane.get("pair") or "") for lane in lanes if lane.get("pair")}))


def _pairs_from_lanes_and_positions(lanes: list[dict[str, Any]], snapshot: dict[str, Any]) -> tuple[str, ...]:
    pairs = {str(lane.get("pair") or "") for lane in lanes if lane.get("pair")}
    for position in snapshot.get("positions", []) or []:
        if not isinstance(position, dict):
            continue
        pair = str(position.get("pair") or "")
        if pair:
            pairs.add(pair)
    return tuple(sorted(pairs))


def _currencies_from_pairs(pairs: tuple[str, ...]) -> tuple[str, ...]:
    currencies: set[str] = set()
    for pair in pairs:
        for currency in pair.split("_"):
            if currency:
                currencies.add(currency)
    return tuple(sorted(currencies))


def _market_status_packet(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {
            "status": "MISSING",
            "evidence_ref": None,
            "is_fx_open": None,
            "active_sessions": [],
            "issues": ["MISSING_MARKET_STATUS_ARTIFACT"],
        }
    return {
        "status": "AVAILABLE",
        "evidence_ref": str(payload.get("evidence_ref") or "market:status"),
        "generated_at_utc": payload.get("generated_at_utc"),
        "weekday": payload.get("weekday"),
        "weekday_index": payload.get("weekday_index"),
        "is_fx_open": payload.get("is_fx_open"),
        "closed_reason": payload.get("closed_reason"),
        "active_sessions": list(payload.get("active_sessions") or []),
        "minutes_to_next_open": payload.get("minutes_to_next_open"),
        "minutes_to_next_close": payload.get("minutes_to_next_close"),
        "contract": payload.get("contract") if isinstance(payload.get("contract"), dict) else {},
        "issues": [],
    }


def _market_context_packet(
    *,
    pairs: tuple[str, ...],
    currencies: tuple[str, ...],
    pair_charts_path: Path,
    context_asset_charts_path: Path,
    broker_instruments_path: Path,
    cross_asset_path: Path,
    flow_path: Path,
    currency_strength_path: Path,
    levels_path: Path,
    market_context_matrix_path: Path,
    calendar_path: Path,
    cot_path: Path,
    option_skew_path: Path,
) -> dict[str, Any]:
    artifacts = {
        "pair_charts": _load_optional_json(pair_charts_path),
        "context_asset_charts": _load_optional_json(context_asset_charts_path),
        "broker_instruments": _load_optional_json(broker_instruments_path),
        "cross_asset": _load_optional_json(cross_asset_path),
        "flow": _load_optional_json(flow_path),
        "currency_strength": _load_optional_json(currency_strength_path),
        "levels": _load_optional_json(levels_path),
        "market_context_matrix": _load_optional_json(market_context_matrix_path),
        "calendar": _load_optional_json(calendar_path),
        "cot": _load_optional_json(cot_path),
        "option_skew": _load_optional_json(option_skew_path),
    }
    missing = [
        f"MISSING_{name.upper()}_ARTIFACT"
        for name, payload in artifacts.items()
        if payload is None and name != "option_skew"
    ]
    pair_payloads = {
        pair: {
            "chart": _chart_summary(artifacts["pair_charts"], pair),
            "flow": _flow_summary(artifacts["flow"], pair),
            "levels": _levels_summary(artifacts["levels"], pair),
            "matrix": _matrix_pair_summary(artifacts["market_context_matrix"], pair),
            "calendar": _calendar_summary(artifacts["calendar"], pair),
            "option_skew": _option_skew_summary(artifacts["option_skew"], pair),
            "cross_correlations": _cross_correlations(artifacts["cross_asset"], pair),
        }
        for pair in pairs
    }
    issues = list(missing)
    for payload in artifacts.values():
        if isinstance(payload, dict):
            issues.extend(str(issue) for issue in payload.get("issues", [])[:12])
    return {
        "pairs": pair_payloads,
        "context_assets": _context_asset_charts_summary(
            artifacts["context_asset_charts"],
            artifacts["broker_instruments"],
        ),
        "broker_tradeability": _broker_tradeability_summary(artifacts["broker_instruments"]),
        "cross_asset": _cross_asset_summary(artifacts["cross_asset"]),
        "matrix_issues": _matrix_issues(artifacts["market_context_matrix"]),
        "currency_strength": _currency_strength_summary(artifacts["currency_strength"], currencies),
        "cot": _cot_summary(artifacts["cot"], currencies),
        "issues": issues[:40],
    }


def _context_asset_charts_summary(
    payload: dict[str, Any] | None,
    broker_instruments: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"status": "missing", "assets": {}, "issues": ["MISSING_CONTEXT_ASSET_CHARTS_ARTIFACT"]}
    tradeable = set()
    not_tradeable = set()
    if isinstance(broker_instruments, dict):
        tradeable = {str(item) for item in broker_instruments.get("context_assets_tradeable", []) or []}
        not_tradeable = {str(item) for item in broker_instruments.get("context_assets_not_tradeable", []) or []}
    assets: dict[str, Any] = {}
    for chart in payload.get("charts", []) or []:
        if not isinstance(chart, dict):
            continue
        instrument = str(chart.get("pair") or "")
        if not instrument:
            continue
        assets[instrument] = {
            "broker_tradeable": instrument in tradeable,
            "broker_tradeability": "TRADEABLE" if instrument in tradeable else ("NOT_TRADEABLE" if instrument in not_tradeable else "UNKNOWN"),
            "evidence_ref": f"context_asset:{instrument}",
            "chart": _chart_summary(payload, instrument),
        }
    return {
        "status": "present",
        "role": payload.get("role"),
        "generated_at_utc": payload.get("generated_at_utc"),
        "assets": assets,
        "issues": [str(issue) for issue in payload.get("issues", [])[:12]],
    }


def _broker_tradeability_summary(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {
            "status": "missing",
            "evidence_ref": "broker:instruments",
            "issues": ["MISSING_BROKER_INSTRUMENTS_ARTIFACT"],
        }
    return {
        "status": payload.get("status"),
        "evidence_ref": "broker:instruments",
        "tradeability_policy": payload.get("tradeability_policy"),
        "tradeable_count": len(payload.get("tradeable_instruments") or []),
        "context_assets_tradeable": list(payload.get("context_assets_tradeable") or []),
        "context_assets_not_tradeable": list(payload.get("context_assets_not_tradeable") or [])[:24],
        "trader_pairs_missing": list(payload.get("trader_pairs_missing") or [])[:24],
        "issues": [str(issue) for issue in payload.get("issues", [])[:12]],
    }


def _chart_summary(payload: dict[str, Any] | None, pair: str) -> dict[str, Any]:
    chart = _first_by_key(payload, "charts", "pair", pair)
    if not chart:
        return {}
    views: dict[str, Any] = {}
    for view in chart.get("views", []) or []:
        if not isinstance(view, dict):
            continue
        granularity = str(view.get("granularity") or "")
        if not granularity:
            continue
        indicators = view.get("indicators") if isinstance(view.get("indicators"), dict) else {}
        regime = view.get("regime_reading") if isinstance(view.get("regime_reading"), dict) else {}
        family = view.get("family_scores") if isinstance(view.get("family_scores"), dict) else {}
        stat = view.get("stat_filters") if isinstance(view.get("stat_filters"), dict) else {}
        structure = view.get("structure") if isinstance(view.get("structure"), dict) else {}
        last_event = structure.get("last_event") if isinstance(structure.get("last_event"), dict) else {}
        views[granularity] = {
            **_small_dict(
                indicators,
                (
                    "atr_pips",
                    "atr_percentile_100",
                    "adx_14",
                    "adx_percentile_100",
                    "rsi_14",
                    "williams_r_14",
                    "mfi_14",
                    "choppiness_14",
                    "bb_width_percentile_100",
                    "hurst_100",
                    "close",
                    "macd_hist",
                    "supertrend_dir",
                    "ichimoku_cloud_pos",
                    "plus_di_14",
                    "minus_di_14",
                ),
            ),
            "regime": view.get("regime"),
            "regime_state": regime.get("state"),
            "regime_confidence": regime.get("confidence"),
            "trend_score": family.get("trend_score"),
            "mean_rev_score": family.get("mean_rev_score"),
            "breakout_score": family.get("breakout_score"),
            "disagreement": family.get("disagreement"),
            "last_jump_bars_ago": stat.get("last_jump_bars_ago"),
            "lag1_autocorr": stat.get("lag1_autocorr"),
            "structure": {
                "last_event": _small_dict(last_event, ("kind", "close_confirmed", "broken_pivot_price")),
            },
        }
    return {
        "dominant_regime": chart.get("dominant_regime"),
        "long_score": chart.get("long_score"),
        "short_score": chart.get("short_score"),
        "chart_story": chart.get("chart_story"),
        "session": _small_dict(
            chart.get("session"),
            (
                "current_tag",
                "jp_holiday",
                "holiday_name",
                "judas_armed",
                "ny_midnight_open_price",
                "next_killzone",
                "minutes_to_next_killzone",
            ),
        ),
        "views": views,
    }


def _flow_summary(payload: dict[str, Any] | None, pair: str) -> dict[str, Any]:
    spread = _first_by_key(payload, "spreads", "instrument", pair)
    return {"spread": _small_dict(spread, ("current_pips", "median_pips", "p90_pips", "stress_flag", "sample_size"))}


def _levels_summary(payload: dict[str, Any] | None, pair: str) -> dict[str, Any]:
    levels = _first_by_key(payload, "pairs", "pair", pair)
    if not levels:
        return {}
    standard_pivot = None
    for pivot in levels.get("pivots", []) or []:
        if isinstance(pivot, dict) and pivot.get("style") == "STANDARD":
            standard_pivot = pivot
            break
    return {
        **_small_dict(levels, ("daily_open", "weekly_open", "monthly_open", "pdh", "pdl", "pdc", "last_close")),
        "standard_pivot": _small_dict(standard_pivot, ("pp", "r1", "r2", "s1", "s2")),
        "nearest_round_numbers": [
            _small_dict(item, ("price", "distance_pips"))
            for item in (levels.get("round_numbers", []) or [])[:3]
            if isinstance(item, dict)
        ],
    }


def _matrix_pair_summary(payload: dict[str, Any] | None, pair: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"status": "missing", "evidence_ref": f"matrix:{pair}"}
    side_map = ((payload.get("pairs") or {}).get(pair) or {})
    if not isinstance(side_map, dict):
        return {"status": "missing_pair", "evidence_ref": f"matrix:{pair}"}
    out: dict[str, Any] = {"status": "present", "evidence_ref": f"matrix:{pair}"}
    for side in ("LONG", "SHORT"):
        reading = side_map.get(side) if isinstance(side_map.get(side), dict) else {}
        out[side] = {
            "evidence_ref": reading.get("evidence_ref") or f"matrix:{pair}:{side}",
            "support_count": reading.get("support_count", 0),
            "reject_count": reading.get("reject_count", 0),
            "warning_count": reading.get("warning_count", 0),
            "missing_count": reading.get("missing_count", 0),
            "strongest_support": reading.get("strongest_support"),
            "strongest_reject": reading.get("strongest_reject"),
            "strongest_warning": reading.get("strongest_warning"),
            "supports": _compact_observations(reading.get("supports")),
            "rejects": _compact_observations(reading.get("rejects")),
            "warnings": _compact_observations(reading.get("warnings")),
        }
    return out


def _compact_observations(rows: Any, *, limit: int = 3) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in rows or []:
        if not isinstance(item, dict):
            continue
        out.append(_small_dict(item, ("code", "layer", "message", "evidence_refs")))
        if len(out) >= limit:
            break
    return out


def _matrix_issues(payload: dict[str, Any] | None) -> list[str]:
    if not isinstance(payload, dict):
        return ["MISSING_MARKET_CONTEXT_MATRIX_ARTIFACT"]
    return [str(item) for item in payload.get("issues", [])[:12] if str(item).strip()]


def _calendar_summary(payload: dict[str, Any] | None, pair: str) -> dict[str, Any]:
    window = _first_by_key(payload, "pair_windows", "pair", pair)
    if not window:
        return {}
    next_event = window.get("next_event") if isinstance(window.get("next_event"), dict) else {}
    return {
        "in_window": window.get("in_window"),
        "reason": window.get("reason"),
        "next_event": _small_dict(next_event, ("timestamp_utc", "currency", "impact", "title")),
    }


def _option_skew_summary(payload: dict[str, Any] | None, pair: str) -> dict[str, Any]:
    if isinstance(payload, dict) and payload.get("enabled") is False and payload.get("disabled_reason"):
        return {
            "enabled": False,
            "disabled_reason": payload.get("disabled_reason"),
            "readings": [],
        }
    readings = [
        _small_dict(item, ("tenor", "rr_25d", "atm_iv", "bf_25d", "issue"))
        for item in (payload or {}).get("readings", []) or []
        if isinstance(item, dict) and item.get("pair") == pair
    ]
    return {"enabled": bool(readings), "readings": readings[:3]}


def _cross_asset_summary(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    assets = {}
    for asset in payload.get("assets", []) or []:
        if not isinstance(asset, dict):
            continue
        instrument = str(asset.get("instrument") or "")
        if instrument in {"USB10Y_USD", "USB02Y_USD", "SPX500_USD", "XAU_USD", "WTICO_USD", "BTC_USD"}:
            assets[instrument] = _small_dict(asset, ("last_price", "trend_label", "change_pct_24h", "z_score_60", "issue"))
    return {
        "synthetic_dxy": _small_dict(payload.get("synthetic_dxy"), ("last_value", "change_pct_24h", "change_pct_5d")),
        "yield_spreads": [
            _small_dict(item, ("name", "spread_last", "spread_change_24h", "issue"))
            for item in (payload.get("yield_spreads", []) or [])[:3]
            if isinstance(item, dict)
        ],
        "assets": assets,
    }


def _cross_correlations(payload: dict[str, Any] | None, pair: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    correlations = payload.get("correlations")
    if not isinstance(correlations, dict):
        return {}
    return _small_dict(
        correlations.get(pair),
        ("USB10Y_USD", "USB02Y_USD", "SPX500_USD", "XAU_USD", "WTICO_USD", "BTC_USD"),
    )


def _currency_strength_summary(payload: dict[str, Any] | None, currencies: tuple[str, ...]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    summaries: dict[str, Any] = {
        "strongest_pair_suggestion": payload.get("strongest_pair_suggestion"),
    }
    wanted = set(currencies)
    for item in payload.get("scores", []) or []:
        if not isinstance(item, dict):
            continue
        currency = str(item.get("currency") or "")
        if currency in wanted:
            summaries[currency] = _small_dict(item, ("rank", "score_pct"))
    return summaries


def _cot_summary(payload: dict[str, Any] | None, currencies: tuple[str, ...]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    wanted = set(currencies)
    summaries: dict[str, Any] = {}
    for item in payload.get("reports", []) or []:
        if not isinstance(item, dict):
            continue
        currency = str(item.get("currency") or "")
        if currency in wanted:
            summaries[currency] = _small_dict(
                item,
                ("report_date", "leveraged_net", "week_change_leveraged_net", "asset_mgr_net", "open_interest"),
            )
    return summaries


def _first_by_key(payload: dict[str, Any] | None, list_key: str, item_key: str, value: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    for item in payload.get(list_key, []) or []:
        if isinstance(item, dict) and item.get(item_key) == value:
            return item
    return {}


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())
