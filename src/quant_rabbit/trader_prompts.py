from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _trader_sl_repair_disabled() -> bool:
    return os.environ.get("QR_TRADER_DISABLE_SL_REPAIR", "").strip() in {"1", "true", "TRUE", "yes", "YES"}

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

    position_reasons = _position_management_reasons(snapshot)
    if position_reasons:
        return _build_route(BRANCH_POSITION, position_reasons, include_content=include_content)

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
            continue
        tp_missing = position.get("take_profit") is None
        sl_missing = position.get("stop_loss") is None
        # SL-free regime (`QR_TRADER_DISABLE_SL_REPAIR=1`, user directive
        # 「SLいらない」 / 「損失を出さないで稼ぎまくる」 2026-05-07): trader-owned
        # SL=None is intentional and should NOT route the operator into
        # BRANCH_POSITION repair mode every cycle. TP-only is the SL-free
        # design — only flag a real protection gap (missing TP, or any
        # missing protection on owner=trader when SL-repair is enabled).
        if sl_free_active and owner == "trader" and not tp_missing:
            continue
        if tp_missing or sl_missing:
            reasons.append(
                "trader-owned position needs protection repair: "
                f"{position.get('pair')} {position.get('side')} id={position.get('trade_id')}"
            )
    return tuple(reasons)


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
