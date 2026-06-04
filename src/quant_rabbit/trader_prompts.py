from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _trader_sl_repair_disabled() -> bool:
    return os.environ.get("QR_TRADER_DISABLE_SL_REPAIR", "").strip() in {"1", "true", "TRUE", "yes", "YES"}


def _missing_tp_repair_enabled() -> bool:
    return os.environ.get("QR_ENABLE_MISSING_TP_REPAIR", "").strip() in {"1", "true", "TRUE", "yes", "YES"}


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

from quant_rabbit.paths import (
    ROOT,
    DEFAULT_AI_ATTACK_ADVICE,
    DEFAULT_BROKER_SNAPSHOT,
    DEFAULT_CALENDAR_SNAPSHOT,
    DEFAULT_CODEX_TRADER_DECISION_RESPONSE,
    DEFAULT_COT_SNAPSHOT,
    DEFAULT_CROSS_ASSET_SNAPSHOT,
    DEFAULT_CURRENCY_STRENGTH,
    DEFAULT_DAILY_TARGET_STATE,
    DEFAULT_FLOW_SNAPSHOT,
    DEFAULT_GPT_TRADER_DECISION,
    DEFAULT_LIVE_ORDER_REQUEST,
    DEFAULT_LEVELS_SNAPSHOT,
    DEFAULT_LEARNING_AUDIT,
    DEFAULT_OPTION_SKEW,
    DEFAULT_ORDER_INTENTS,
    DEFAULT_PAIR_CHARTS,
    DEFAULT_POSITION_EXECUTION,
    DEFAULT_TRADER_PROMPTS_DIR,
)


BRANCH_REFRESH = "refresh_market_context"
BRANCH_ENTRY = "entry_decision"
BRANCH_POSITION = "position_management"
BRANCH_VERIFY = "verify_execute"
BRANCH_LEARNING = "learning_gap"
DEFAULT_AUTOTRADE_REPORT = ROOT / "docs" / "autotrade_cycle_report.md"


@dataclass(frozen=True)
class PromptDoc:
    key: str
    path: Path
    exists: bool
    content: str | None = None

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "key": self.key,
            "path": str(self.path),
            "exists": self.exists,
        }
        if self.content is not None:
            payload["content"] = self.content
        return payload


@dataclass(frozen=True)
class PromptRoute:
    branch: str
    reasons: tuple[str, ...]
    read_order: tuple[PromptDoc, ...]

    def to_payload(self) -> dict[str, Any]:
        return {
            "branch": self.branch,
            "reasons": list(self.reasons),
            "read_order": [doc.to_payload() for doc in self.read_order],
        }


@dataclass(frozen=True)
class _TPProbePosition:
    trade_id: str
    pair: str
    side: str
    units: int
    entry_price: float
    take_profit: float | None
    stop_loss: float | None
    unrealized_pl_jpy: float
    owner: str


PROMPT_FILES: dict[str, Path] = {
    "contract": ROOT / "docs" / "AGENT_CONTRACT.md",
    "entry": ROOT / "docs" / "SKILL_trader.md",
    "router": DEFAULT_TRADER_PROMPTS_DIR / "00_router.md",
    "precheck_refresh": DEFAULT_TRADER_PROMPTS_DIR / "10_precheck_refresh.md",
    "market_packet": DEFAULT_TRADER_PROMPTS_DIR / "20_market_packet.md",
    "entry_decision": DEFAULT_TRADER_PROMPTS_DIR / "30_entry_decision.md",
    "position_management": DEFAULT_TRADER_PROMPTS_DIR / "35_position_management.md",
    "verify_execute": DEFAULT_TRADER_PROMPTS_DIR / "40_verify_execute.md",
    "learning_gap": DEFAULT_TRADER_PROMPTS_DIR / "50_learning_gap.md",
    "decision_schema": DEFAULT_TRADER_PROMPTS_DIR / "90_decision_receipt_schema.md",
}

BRANCH_READ_ORDER: dict[str, tuple[str, ...]] = {
    BRANCH_REFRESH: ("contract", "entry", "router", "precheck_refresh"),
    BRANCH_ENTRY: ("contract", "entry", "router", "market_packet", "entry_decision", "decision_schema"),
    BRANCH_POSITION: ("contract", "entry", "router", "market_packet", "position_management", "decision_schema"),
    BRANCH_VERIFY: ("contract", "entry", "router", "verify_execute"),
    BRANCH_LEARNING: ("contract", "entry", "router", "market_packet", "learning_gap"),
}


def route_trader_prompts(
    *,
    snapshot_path: Path = DEFAULT_BROKER_SNAPSHOT,
    target_state_path: Path = DEFAULT_DAILY_TARGET_STATE,
    intents_path: Path = DEFAULT_ORDER_INTENTS,
    pair_charts_path: Path = DEFAULT_PAIR_CHARTS,
    cross_asset_path: Path = DEFAULT_CROSS_ASSET_SNAPSHOT,
    flow_path: Path = DEFAULT_FLOW_SNAPSHOT,
    currency_strength_path: Path = DEFAULT_CURRENCY_STRENGTH,
    levels_path: Path = DEFAULT_LEVELS_SNAPSHOT,
    calendar_path: Path = DEFAULT_CALENDAR_SNAPSHOT,
    cot_path: Path = DEFAULT_COT_SNAPSHOT,
    option_skew_path: Path = DEFAULT_OPTION_SKEW,
    attack_advice_path: Path = DEFAULT_AI_ATTACK_ADVICE,
    learning_audit_path: Path = DEFAULT_LEARNING_AUDIT,
    decision_response_path: Path | None = DEFAULT_CODEX_TRADER_DECISION_RESPONSE,
    gpt_decision_path: Path = DEFAULT_GPT_TRADER_DECISION,
    live_order_path: Path | None = DEFAULT_LIVE_ORDER_REQUEST,
    position_execution_path: Path | None = DEFAULT_POSITION_EXECUTION,
    autotrade_report_path: Path | None = DEFAULT_AUTOTRADE_REPORT,
    include_content: bool = False,
) -> PromptRoute:
    artifacts = {
        "broker_snapshot": snapshot_path,
        "daily_target_state": target_state_path,
        "pair_charts": pair_charts_path,
        "cross_asset_snapshot": cross_asset_path,
        "flow_snapshot": flow_path,
        "currency_strength": currency_strength_path,
        "levels_snapshot": levels_path,
        "economic_calendar": calendar_path,
        "cot_snapshot": cot_path,
        "option_skew_snapshot": option_skew_path,
        "order_intents": intents_path,
        "ai_attack_advice": attack_advice_path,
        "learning_audit": learning_audit_path,
    }
    missing_artifacts = tuple(name for name, path in artifacts.items() if not path.exists())
    if missing_artifacts:
        return _build_route(
            BRANCH_REFRESH,
            (f"missing required artifact(s): {', '.join(missing_artifacts)}",),
            include_content=include_content,
        )

    snapshot = _load_json(snapshot_path)
    target_state = _load_json(target_state_path)
    intents = _load_json(intents_path)
    pair_charts = _load_json(pair_charts_path)

    position_reasons = _position_management_reasons(snapshot)
    tp_rebalance_reasons = _tp_rebalance_reasons(
        snapshot,
        pair_charts,
        snapshot_path=snapshot_path,
    )
    close_review_reasons = _position_close_recommendation_reasons(
        snapshot,
        snapshot_path=snapshot_path,
    )
    entry_thesis_block_reasons = _entry_thesis_blocker_reasons(
        snapshot,
        snapshot_path=snapshot_path,
    )
    if position_reasons or tp_rebalance_reasons or close_review_reasons or entry_thesis_block_reasons:
        return _build_route(
            BRANCH_POSITION,
            (
                *position_reasons,
                *tp_rebalance_reasons,
                *close_review_reasons,
                *entry_thesis_block_reasons,
            ),
            include_content=include_content,
        )

    decision_state = _decision_receipt_state(
        decision_response_path=decision_response_path,
        snapshot_path=snapshot_path,
        intents_path=intents_path,
        gpt_decision_path=gpt_decision_path,
        live_order_path=live_order_path,
        position_execution_path=position_execution_path,
        autotrade_report_path=autotrade_report_path,
    )
    if decision_state.pending:
        return _build_route(
            BRANCH_VERIFY,
            decision_state.reasons,
            include_content=include_content,
        )
    carry_reasons = decision_state.reasons

    live_ready_lanes = _live_ready_lane_ids(intents)
    if _target_open(target_state) and live_ready_lanes:
        return _build_route(
            BRANCH_ENTRY,
            (*carry_reasons, f"daily target open with {len(live_ready_lanes)} current LIVE_READY lane(s)"),
            include_content=include_content,
        )
    if _target_open(target_state):
        return _build_route(
            BRANCH_LEARNING,
            (*carry_reasons, "daily target open but no current LIVE_READY lane is available"),
            include_content=include_content,
        )

    return _build_route(
        BRANCH_POSITION,
        (*carry_reasons, "daily target is closed or protected; review exposure before adding risk"),
        include_content=include_content,
    )


def _build_route(branch: str, reasons: tuple[str, ...], *, include_content: bool) -> PromptRoute:
    docs = tuple(_prompt_doc(key, include_content=include_content) for key in BRANCH_READ_ORDER[branch])
    return PromptRoute(branch=branch, reasons=reasons, read_order=docs)


def _prompt_doc(key: str, *, include_content: bool) -> PromptDoc:
    path = PROMPT_FILES[key]
    exists = path.exists()
    content = path.read_text() if include_content and exists else None
    return PromptDoc(key=key, path=path, exists=exists, content=content)


@dataclass(frozen=True)
class DecisionReceiptState:
    pending: bool
    reasons: tuple[str, ...] = ()


def _decision_receipt_state(
    *,
    decision_response_path: Path | None,
    snapshot_path: Path,
    intents_path: Path,
    gpt_decision_path: Path | None,
    live_order_path: Path | None,
    position_execution_path: Path | None,
    autotrade_report_path: Path | None,
) -> DecisionReceiptState:
    if decision_response_path is None or not decision_response_path.exists():
        return DecisionReceiptState(pending=False)

    decision_mtime_ns = decision_response_path.stat().st_mtime_ns
    for path, label in (
        (snapshot_path, "refreshed broker snapshot"),
        (intents_path, "repriced order intents"),
    ):
        if _artifact_newer(path, decision_mtime_ns):
            return DecisionReceiptState(
                pending=False,
                reasons=(f"decision response predates {label}; refresh decision from broker truth: {path}",),
            )

    gpt_pending = _accepted_gpt_trade_pending_gateway(
        gpt_decision_path=gpt_decision_path,
        decision_mtime_ns=decision_mtime_ns,
        live_order_path=live_order_path,
        position_execution_path=position_execution_path,
        autotrade_report_path=autotrade_report_path,
    )
    if gpt_pending is not None:
        return gpt_pending

    for path, label in (
        (live_order_path, "live order gateway receipt"),
        (position_execution_path, "position gateway receipt"),
        (autotrade_report_path, "autotrade cycle report"),
    ):
        if _artifact_newer(path, decision_mtime_ns):
            return DecisionReceiptState(
                pending=False,
                reasons=(f"decision response already consumed by {label}: {path}",),
            )

    gpt_state = _gpt_decision_terminal_state(gpt_decision_path, decision_mtime_ns)
    if gpt_state:
        return DecisionReceiptState(pending=False, reasons=(gpt_state,))

    return DecisionReceiptState(
        pending=True,
        reasons=(f"unconsumed decision response exists: {decision_response_path}",),
    )


def _accepted_gpt_trade_pending_gateway(
    *,
    gpt_decision_path: Path | None,
    decision_mtime_ns: int,
    live_order_path: Path | None,
    position_execution_path: Path | None,
    autotrade_report_path: Path | None,
) -> DecisionReceiptState | None:
    if gpt_decision_path is None or not gpt_decision_path.exists():
        return None
    gpt_mtime_ns = gpt_decision_path.stat().st_mtime_ns
    if gpt_mtime_ns <= decision_mtime_ns:
        return None
    try:
        payload = _load_json(gpt_decision_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    status = str(payload.get("status") or "")
    decision = payload.get("decision") if isinstance(payload.get("decision"), dict) else {}
    action = str(decision.get("action") or "")
    if status != "ACCEPTED" or action != "TRADE":
        return None

    for path, label in (
        (live_order_path, "live order gateway receipt"),
        (position_execution_path, "position gateway receipt"),
        (autotrade_report_path, "autotrade cycle report"),
    ):
        if path is not None and path.exists() and path.stat().st_mtime_ns > gpt_mtime_ns:
            return DecisionReceiptState(
                pending=False,
                reasons=(f"accepted TRADE decision already consumed by {label}: {path}",),
            )

    return DecisionReceiptState(
        pending=True,
        reasons=(
            "accepted TRADE decision has no newer gateway receipt; "
            f"run exactly one gateway cycle now: {gpt_decision_path}",
        ),
    )


def _artifact_newer(path: Path | None, decision_mtime_ns: int) -> bool:
    return path is not None and path.exists() and path.stat().st_mtime_ns > decision_mtime_ns


def _gpt_decision_terminal_state(path: Path | None, decision_mtime_ns: int) -> str | None:
    if path is None or not path.exists() or path.stat().st_mtime_ns <= decision_mtime_ns:
        return None
    try:
        payload = _load_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    status = str(payload.get("status") or "")
    decision = payload.get("decision") if isinstance(payload.get("decision"), dict) else {}
    action = str(decision.get("action") or "")
    if status != "ACCEPTED":
        return f"decision response already verified as {status or 'UNKNOWN'}: {path}"
    if action and action != "TRADE":
        return f"decision response already verified as non-executable {action}: {path}"
    return None


def _position_management_reasons(snapshot: dict[str, Any]) -> tuple[str, ...]:
    sl_free_active = _trader_sl_repair_disabled()
    reasons: list[str] = []
    for position in snapshot.get("positions", []) or []:
        if not isinstance(position, dict):
            continue
        owner = str(position.get("owner") or "")
        if owner in {"manual", "unknown"}:
            # Manual/tagless exposure is observed but must not block fresh
            # trader entries. Route to TP-only management only when there is
            # profit to insure/bank, or when the operator explicitly opts into
            # missing-TP repair. Underwater manual runners remain held.
            upl = _optional_float(position.get("unrealized_pl_jpy"))
            if position.get("take_profit") is None and (_missing_tp_repair_enabled() or (upl is not None and upl > 0)):
                reasons.append(
                    "manual/tagless position needs TP-only profit management: "
                    f"{position.get('pair')} {position.get('side')} id={position.get('trade_id')}"
                )
            continue
        tp_missing = position.get("take_profit") is None
        sl_missing = position.get("stop_loss") is None
        # SL-free regime (`QR_TRADER_DISABLE_SL_REPAIR=1`, user directive
        # 「SLいらない」 / 「損失を出さないで稼ぎまくる」 2026-05-07): trader-owned
        # SL=None is intentional. Missing broker TP is also preserved as a
        # no-broker-TP runner unless explicit repair is enabled, so it should
        # not trap the operator in BRANCH_POSITION and starve fresh entries.
        if (
            sl_free_active
            and owner == "trader"
            and (not tp_missing or not _missing_tp_repair_enabled())
        ):
            continue
        if tp_missing or sl_missing:
            reasons.append(
                "trader-owned position needs protection repair: "
                f"{position.get('pair')} {position.get('side')} id={position.get('trade_id')}"
            )
    return tuple(reasons)


def _position_close_recommendation_reasons(
    snapshot: dict[str, Any],
    *,
    snapshot_path: Path,
) -> tuple[str, ...]:
    """Route fresh loss-cut evidence to the position-management prompt branch.

    These sidecars are generated from the current broker snapshot and market
    packet. A stale close recommendation is worse than no recommendation, so
    reports older than the current snapshot are ignored.
    """
    data_root = snapshot_path.parent
    reasons: list[str] = []
    for rec in _fresh_close_recommendations(snapshot, data_root=data_root):
        reasons.append(
            "loss-cut review required: "
            f"{rec['source']} {rec['verdict']} for "
            f"{rec.get('pair') or '?'} {rec.get('side') or '?'} id={rec.get('trade_id')}: "
            f"{rec.get('reason') or 'prediction no longer supports recovery'}"
        )
    return tuple(reasons)


def _entry_thesis_blocker_reasons(
    snapshot: dict[str, Any],
    *,
    snapshot_path: Path,
) -> tuple[str, ...]:
    """Route unverifiable active positions to position management.

    This is deliberately separate from `_fresh_close_recommendations`: a
    missing entry thesis is not Gate A close evidence, but normal WAIT/new-risk
    routing must not proceed while the position cannot be audited.
    """
    data_root = snapshot_path.parent
    reasons: list[str] = []
    for rec in _fresh_entry_thesis_blockers(snapshot, data_root=data_root):
        reasons.append(
            "entry-thesis repair required: "
            f"{rec.get('pair') or '?'} {rec.get('side') or '?'} id={rec.get('trade_id')}: "
            f"{rec.get('reason') or 'original entry thesis is not machine-verifiable'}"
        )
    return tuple(reasons)


def _fresh_close_recommendations(snapshot: dict[str, Any], *, data_root: Path) -> tuple[dict[str, Any], ...]:
    fetched_at = _parse_utc(snapshot.get("fetched_at_utc"))
    if fetched_at is None:
        return ()
    active_positions = _active_trader_positions_by_trade_id(snapshot)
    recs: list[dict[str, Any]] = []
    recs.extend(_position_thesis_recommendations(data_root / "position_thesis_report.json", fetched_at))
    recs.extend(_thesis_evolution_recommendations(data_root / "thesis_evolution_report.json", fetched_at))
    recs.extend(_forecast_persistence_recommendations(data_root / "forecast_persistence_report.json", fetched_at))
    out: list[dict[str, Any]] = []
    for rec in recs:
        trade_id = str(rec.get("trade_id") or "")
        active = active_positions.get(trade_id)
        if not active:
            continue
        rec_pair = str(rec.get("pair") or active["pair"])
        rec_side = str(rec.get("side") or active["side"]).upper()
        if active["pair"] and rec_pair and rec_pair != active["pair"]:
            continue
        if active["side"] and rec_side and rec_side != active["side"]:
            continue
        out.append({**rec, "pair": rec_pair or active["pair"], "side": rec_side or active["side"]})
    return tuple(out)


def _fresh_entry_thesis_blockers(snapshot: dict[str, Any], *, data_root: Path) -> tuple[dict[str, Any], ...]:
    fetched_at = _parse_utc(snapshot.get("fetched_at_utc"))
    if fetched_at is None:
        return ()
    payload = _fresh_report_payload(data_root / "thesis_evolution_report.json", fetched_at)
    if not payload:
        return ()
    active_positions = _active_trader_positions_by_trade_id(snapshot)
    out: list[dict[str, Any]] = []
    for item in payload.get("evolutions", []) or []:
        if not isinstance(item, dict):
            continue
        status = str(item.get("status") or "").upper()
        verdict = str(item.get("verdict") or "").upper()
        if status != "UNVERIFIABLE" and verdict != "REQUIRE_THESIS_REPAIR":
            continue
        trade_id = str(item.get("trade_id") or "")
        active = active_positions.get(trade_id)
        if not active:
            continue
        rec_pair = str(item.get("pair") or active["pair"])
        rec_side = str(item.get("side") or active["side"]).upper()
        if active["pair"] and rec_pair and rec_pair != active["pair"]:
            continue
        if active["side"] and rec_side and rec_side != active["side"]:
            continue
        out.append(
            {
                "source": "entry_thesis",
                "evidence_ref": f"position:evolution:{trade_id}",
                "trade_id": trade_id,
                "pair": rec_pair or active["pair"],
                "side": rec_side or active["side"],
                "verdict": "REQUIRE_THESIS_REPAIR",
                "status": status or "UNVERIFIABLE",
                "reason": item.get("rationale") or "original entry thesis is not machine-verifiable",
            }
        )
    return tuple(out)


def _active_trader_positions_by_trade_id(snapshot: dict[str, Any]) -> dict[str, dict[str, str]]:
    active: dict[str, dict[str, str]] = {}
    for position in snapshot.get("positions", []) or []:
        if not isinstance(position, dict) or str(position.get("owner") or "") != "trader":
            continue
        trade_id = str(position.get("trade_id") or "")
        if not trade_id:
            continue
        active[trade_id] = {
            "pair": str(position.get("pair") or ""),
            "side": str(position.get("side") or "").upper(),
        }
    return active


def _position_thesis_recommendations(path: Path, fetched_at: datetime) -> list[dict[str, Any]]:
    payload = _fresh_report_payload(path, fetched_at)
    if not payload:
        return []
    out: list[dict[str, Any]] = []
    for item in payload.get("assessments", []) or []:
        if not isinstance(item, dict) or str(item.get("verdict") or "") != "REVIEW_CLOSE":
            continue
        trade_id = str(item.get("trade_id") or "")
        reason_parts = [
            str(x)
            for x in (list(item.get("rationale_lines") or []) + list(item.get("context_notes") or []))
            if str(x)
        ]
        out.append(
            {
                "source": "position_thesis",
                "evidence_ref": f"position:thesis:{trade_id}",
                "trade_id": trade_id,
                "pair": item.get("pair"),
                "side": item.get("side"),
                "verdict": "REVIEW_CLOSE",
                "reason": "; ".join(reason_parts[:3])
                or f"aggregate_score={item.get('aggregate_score')}",
            }
        )
    return out


def _thesis_evolution_recommendations(path: Path, fetched_at: datetime) -> list[dict[str, Any]]:
    payload = _fresh_report_payload(path, fetched_at)
    if not payload:
        return []
    out: list[dict[str, Any]] = []
    for item in payload.get("evolutions", []) or []:
        if not isinstance(item, dict):
            continue
        verdict = str(item.get("verdict") or "")
        status = str(item.get("status") or "")
        if verdict != "RECOMMEND_CLOSE" and status != "BROKEN":
            continue
        trade_id = str(item.get("trade_id") or "")
        out.append(
            {
                "source": "thesis_evolution",
                "evidence_ref": f"position:evolution:{trade_id}",
                "trade_id": trade_id,
                "pair": item.get("pair"),
                "side": item.get("side"),
                "verdict": verdict or status,
                "reason": item.get("rationale") or f"status={status}",
            }
        )
    return out


def _forecast_persistence_recommendations(path: Path, fetched_at: datetime) -> list[dict[str, Any]]:
    payload = _fresh_report_payload(path, fetched_at)
    if not payload:
        return []
    out: list[dict[str, Any]] = []
    for item in payload.get("verdicts", []) or []:
        if not isinstance(item, dict) or str(item.get("verdict") or "") != "RECOMMEND_CLOSE":
            continue
        trade_id = str(item.get("trade_id") or "")
        out.append(
            {
                "source": "forecast_persistence",
                "evidence_ref": f"position:persistence:{trade_id}",
                "trade_id": trade_id,
                "pair": item.get("pair"),
                "side": item.get("side"),
                "verdict": "RECOMMEND_CLOSE",
                "reason": item.get("reason") or "persistent forecast no longer supports the position",
            }
        )
    return out


def _fresh_report_payload(path: Path, fetched_at: datetime) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = _load_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    generated_at = _parse_utc(payload.get("generated_at_utc"))
    if generated_at is None or generated_at < fetched_at:
        return None
    return payload


def _parse_utc(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _tp_rebalance_reasons(
    snapshot: dict[str, Any],
    pair_charts_payload: dict[str, Any],
    *,
    snapshot_path: Path,
) -> tuple[str, ...]:
    """Return position-management reasons for executable TP adjustments.

    The router only probes in dry-run memory: it never writes broker state.
    This keeps prompt routing deterministic while forcing cycles with stale
    profitable TP geometry back into the protection branch instead of letting
    an accepted WAIT stop the sidecar work.
    """
    positions = tuple(_tp_probe_positions(snapshot))
    if not positions:
        return ()

    pair_charts_keyed = _pair_charts_by_pair(pair_charts_payload)
    if not pair_charts_keyed:
        return ()

    quotes_keyed = _quotes_by_pair(snapshot)
    if not quotes_keyed:
        return ()

    from quant_rabbit.strategy.entry_thesis_ledger import load_latest_forecast
    from quant_rabbit.strategy.intent_generator import _market_derived_reward_risk
    from quant_rabbit.strategy.tp_rebalancer import (
        compute_all_tp_adjustments,
        load_entry_thesis_blocker_trade_ids,
    )

    data_root = snapshot_path.parent
    latest_forecasts_by_pair: dict[str, dict[str, Any]] = {}
    for pair in sorted({position.pair for position in positions if position.pair}):
        latest = load_latest_forecast(pair, data_root)
        if isinstance(latest, dict):
            latest_forecasts_by_pair[pair] = latest

    adjustments = compute_all_tp_adjustments(
        positions=positions,
        quotes=quotes_keyed,
        pair_charts=pair_charts_keyed,
        market_reward_risk_fn=_market_derived_reward_risk,
        latest_forecasts_by_pair=latest_forecasts_by_pair,
        entry_thesis_block_trade_ids=load_entry_thesis_blocker_trade_ids(data_root),
    )
    reasons: list[str] = []
    for adjustment in adjustments:
        reasons.append(
            "TP rebalance required before WAIT/entry routing: "
            f"{adjustment.pair} {adjustment.side} id={adjustment.trade_id} "
            f"{adjustment.current_tp}->{adjustment.new_tp}; {adjustment.rationale}"
        )
    return tuple(reasons)


def _tp_probe_positions(snapshot: dict[str, Any]) -> tuple[_TPProbePosition, ...]:
    positions: list[_TPProbePosition] = []
    for raw in snapshot.get("positions", []) or []:
        if not isinstance(raw, dict):
            continue
        trade_id = str(raw.get("trade_id") or "")
        pair = str(raw.get("pair") or "")
        side = str(raw.get("side") or "").upper()
        entry_price = _optional_float(raw.get("entry_price"))
        if not trade_id or not pair or side not in {"LONG", "SHORT"} or entry_price is None:
            continue
        try:
            units = int(raw.get("units") or 0)
        except (TypeError, ValueError):
            units = 0
        positions.append(
            _TPProbePosition(
                trade_id=trade_id,
                pair=pair,
                side=side,
                units=units,
                entry_price=entry_price,
                take_profit=_optional_float(raw.get("take_profit")),
                stop_loss=_optional_float(raw.get("stop_loss")),
                unrealized_pl_jpy=_optional_float(raw.get("unrealized_pl_jpy")) or 0.0,
                owner=str(raw.get("owner") or ""),
            )
        )
    return tuple(positions)


def _pair_charts_by_pair(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    keyed: dict[str, dict[str, Any]] = {}
    charts = payload.get("charts") if isinstance(payload, dict) else None
    for chart in charts or []:
        if not isinstance(chart, dict):
            continue
        pair = str(chart.get("pair") or "")
        if pair:
            keyed[pair] = chart
    return keyed


def _quotes_by_pair(snapshot: dict[str, Any]) -> dict[str, dict[str, Any]]:
    keyed: dict[str, dict[str, Any]] = {}
    quotes = snapshot.get("quotes") if isinstance(snapshot, dict) else None
    for pair, quote in (quotes or {}).items():
        if isinstance(quote, dict):
            keyed[str(pair)] = quote
    return keyed


def _live_ready_lane_ids(intents: dict[str, Any]) -> tuple[str, ...]:
    lane_ids: list[str] = []
    for result in intents.get("results", []) or []:
        if not isinstance(result, dict):
            continue
        if result.get("status") != "LIVE_READY":
            continue
        if _block_issues(result.get("risk_issues")):
            continue
        if _block_issues(result.get("strategy_issues")):
            continue
        if result.get("live_blockers"):
            continue
        lane_id = str(result.get("lane_id") or "")
        if lane_id:
            lane_ids.append(lane_id)
    return tuple(dict.fromkeys(lane_ids))


def _target_open(target_state: dict[str, Any]) -> bool:
    # Zero is the ledger boundary: a positive remaining target means the
    # campaign still needs exposure; this is not a discretionary risk threshold.
    remaining = target_state.get("remaining_target_jpy")
    try:
        return float(remaining or 0.0) > 0.0 and target_state.get("status") != "TARGET_REACHED_PROTECT"
    except (TypeError, ValueError):
        return False


def _block_issues(items: object) -> tuple[str, ...]:
    blockers: list[str] = []
    for item in items or []:
        if isinstance(item, dict) and item.get("severity") == "BLOCK":
            blockers.append(str(item.get("code") or item.get("message") or "BLOCK"))
    return tuple(blockers)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        return payload
    raise ValueError(f"{path} must contain a JSON object")


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return _load_json(path)
