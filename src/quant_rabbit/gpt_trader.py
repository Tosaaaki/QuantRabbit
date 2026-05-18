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


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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


def _pair_chart_story(packet: dict[str, Any], pair: str) -> str:
    """Pull the per-pair chart_story string from the verifier packet."""
    pairs_block = (
        ((packet.get("market_context") or {}).get("pairs") or {})
        .get(pair) or {}
    )
    chart = pairs_block.get("chart") if isinstance(pairs_block, dict) else None
    if isinstance(chart, dict):
        return str(chart.get("chart_story") or "")
    return ""


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


def _close_thesis_invalidated(
    packet: dict[str, Any],
    pair: str,
    side: str,
    *,
    decision: "GPTTraderDecision | None" = None,
) -> tuple[bool, str]:
    """Check whether the position's thesis has been invalidated.

    Two acceptance paths (§6-compliant — no JPY/pip/multiplier literals):

      (a) Structural BOS or CHOCH on M15 or H4 printing AGAINST the
          position side. This is the chart_reader.structure_events lens
          that already drives trader_brain price-action scoring; using
          the same signal keeps prefilter and CLOSE-gate aligned.

      (b) The decision receipt's `invalidation_price` + `invalidation_tf`
          fields are populated AND broker-truth quote shows the level
          has traded through. LONG invalidates downward, SHORT upward.
          Pure prose `invalidation` text alone is not enough — the gate
          requires a machine-checkable price hit.
    """
    side_upper = str(side or "").upper()
    if side_upper not in {"LONG", "SHORT"}:
        return False, "unknown position side"

    chart_story = _pair_chart_story(packet, pair)
    structs = _parse_struct_events(chart_story)
    counter_direction = "DOWN" if side_upper == "LONG" else "UP"
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
            return True, (
                f"{tf} {event_type}_{direction}@{price:g} prints against "
                f"{side_upper} thesis (close-confirmed)"
            )

    if decision is not None and decision.invalidation_price is not None:
        snapshot = packet.get("broker_snapshot") or {}
        quotes = snapshot.get("quotes") or {}
        quote = quotes.get(pair)
        if isinstance(quote, dict):
            bid = _optional_float(quote.get("bid"))
            ask = _optional_float(quote.get("ask"))
            level = decision.invalidation_price
            tf = decision.invalidation_tf or "unspecified-TF"
            if side_upper == "LONG" and bid is not None and bid <= level:
                return True, (
                    f"bid {bid:g} reached LONG invalidation {level:g} "
                    f"on {tf} per receipt"
                )
            if side_upper == "SHORT" and ask is not None and ask >= level:
                return True, (
                    f"ask {ask:g} reached SHORT invalidation {level:g} "
                    f"on {tf} per receipt"
                )

    return False, ""

from quant_rabbit.analysis.chart_reader import DEFAULT_TIMEFRAMES as DEFAULT_PAIR_CHART_TIMEFRAMES
from quant_rabbit.paths import (
    DEFAULT_AI_ATTACK_ADVICE,
    DEFAULT_CAMPAIGN_PLAN,
    DEFAULT_CALENDAR_SNAPSHOT,
    DEFAULT_COT_SNAPSHOT,
    DEFAULT_CROSS_ASSET_SNAPSHOT,
    DEFAULT_CURRENCY_STRENGTH,
    DEFAULT_DAILY_TARGET_STATE,
    DEFAULT_FLOW_SNAPSHOT,
    DEFAULT_GPT_TRADER_DECISION,
    DEFAULT_GPT_TRADER_DECISION_REPORT,
    DEFAULT_LEVELS_SNAPSHOT,
    DEFAULT_MARKET_STORY_PROFILE,
    DEFAULT_OPTION_SKEW,
    DEFAULT_ORDER_INTENTS,
    DEFAULT_PAIR_CHARTS,
    DEFAULT_STRATEGY_PROFILE,
)


ALLOWED_ACTIONS = ("TRADE", "WAIT", "CANCEL_PENDING", "PROTECT", "TIGHTEN_SL", "CLOSE", "REQUEST_EVIDENCE")
ALLOWED_CONFIDENCE = ("LOW", "MEDIUM", "HIGH")
ALLOWED_METHODS = ("TREND_CONTINUATION", "RANGE_ROTATION", "BREAKOUT_FAILURE", "EVENT_RISK", "POSITION_MANAGEMENT")
ALLOWED_SPECIALIST_ROLES = ("macro_news", "indicator", "flow_levels", "risk_audit", "strategy", "portfolio_context")
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


@dataclass(frozen=True)
class GPTTraderDecision:
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
    # Operator-directed market close on existing trader-owned positions.
    # Used with action="CLOSE" or attached to TRADE when an entry should
    # also retire prior positions in the same cycle. Each id must match a
    # current trader-owned open trade in the broker snapshot.
    close_trade_ids: tuple[str, ...] = ()
    strategy_reviews: tuple[dict[str, Any], ...] = ()
    specialist_reviews: tuple[dict[str, Any], ...] = ()
    # CLOSE-action discipline fields (added 2026-05-12, see
    # `feedback_no_unilateral_close.md`). A CLOSE receipt must pass two
    # gates: (A) market evidence that the thesis is invalidated, either
    # via structural BOS/CHOCH against the position side on M15 or H4
    # (parsed from pair_charts.chart_story) or via an explicit
    # `invalidation_price` + `invalidation_tf` that broker truth confirms
    # has been hit; (B) explicit operator authorization via
    # `QR_OPERATOR_CLOSE_OVERRIDE=1` or a fresh
    # `data/.operator_close_token`. The `operator_close_authorized` field
    # remains audit text only and is not accepted as authorization, because
    # the trader can set fields in its own receipt.
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
    def __init__(self, decision: dict[str, Any]) -> None:
        self.decision = decision

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
        target_state_path: Path = DEFAULT_DAILY_TARGET_STATE,
        pair_charts_path: Path = DEFAULT_PAIR_CHARTS,
        cross_asset_path: Path = DEFAULT_CROSS_ASSET_SNAPSHOT,
        flow_path: Path = DEFAULT_FLOW_SNAPSHOT,
        currency_strength_path: Path = DEFAULT_CURRENCY_STRENGTH,
        levels_path: Path = DEFAULT_LEVELS_SNAPSHOT,
        calendar_path: Path = DEFAULT_CALENDAR_SNAPSHOT,
        cot_path: Path = DEFAULT_COT_SNAPSHOT,
        option_skew_path: Path = DEFAULT_OPTION_SKEW,
        attack_advice_path: Path = DEFAULT_AI_ATTACK_ADVICE,
        output_path: Path = DEFAULT_GPT_TRADER_DECISION,
        report_path: Path = DEFAULT_GPT_TRADER_DECISION_REPORT,
        max_lanes: int = DEFAULT_GPT_MAX_LANES,
    ) -> None:
        self.provider = provider
        self.intents_path = intents_path
        self.campaign_plan_path = campaign_plan_path
        self.strategy_profile_path = strategy_profile_path
        self.market_story_profile_path = market_story_profile_path
        self.target_state_path = target_state_path
        self.pair_charts_path = pair_charts_path
        self.cross_asset_path = cross_asset_path
        self.flow_path = flow_path
        self.currency_strength_path = currency_strength_path
        self.levels_path = levels_path
        self.calendar_path = calendar_path
        self.cot_path = cot_path
        self.option_skew_path = option_skew_path
        self.attack_advice_path = attack_advice_path
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
        target = _load_json(self.target_state_path) if self.target_state_path.exists() else {}
        lanes = _lane_packet(intents, campaign, strategy, story, max_lanes=self.max_lanes)
        attack_advice = _load_optional_json(self.attack_advice_path)
        refs = _allowed_refs(snapshot=snapshot, target=target, lanes=lanes, attack_advice=attack_advice)
        pairs = _pairs_from_lanes(lanes)
        currencies = _currencies_from_pairs(pairs)
        return {
            "contract": {
                "allowed_actions": list(ALLOWED_ACTIONS),
                "trade_requires_live_ready_lane": True,
                "trade_may_select_multiple_live_ready_lanes": True,
                "pending_entries_are_basket_counted_by_gateway": True,
                "protected_trader_position_adds_require_portfolio_validation": True,
                "model_output_is_advisory_until_verified": True,
                "strategy_reviews_must_use_lane_id_not_desk_alias": True,
            },
            "broker_snapshot": _snapshot_packet(snapshot),
            "daily_target": _target_packet(target),
            "lanes": lanes,
            "ai_attack_advice": _attack_advice_packet(attack_advice),
            "market_context": _market_context_packet(
                pairs=pairs,
                currencies=currencies,
                pair_charts_path=self.pair_charts_path,
                cross_asset_path=self.cross_asset_path,
                flow_path=self.flow_path,
                currency_strength_path=self.currency_strength_path,
                levels_path=self.levels_path,
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
                "- Evidence refs must come from the input packet; invented refs reject the decision.",
                "- `CLOSE` (or TRADE+`close_trade_ids`) requires gate A (structural BOS/CHOCH against side on M15/H4, or `invalidation_price`+`invalidation_tf` hit on broker truth) AND gate B (`QR_OPERATOR_CLOSE_OVERRIDE=1` or a fresh `data/.operator_close_token`). The receipt's `operator_close_authorized` field is advisory only. See AGENT_CONTRACT §10.",
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

        broker = self.packet.get("broker_snapshot", {})
        positions = int(broker.get("positions") or 0)
        selected_lane_ids = _selected_trade_lane_ids(decision)
        primary_lane_id = decision.selected_lane_id or (selected_lane_ids[0] if selected_lane_ids else None)
        tradeable_lanes = _tradeable_live_ready_lanes(self.packet)
        attack_lane_ids = _attack_recommended_tradeable_lane_ids(self.packet, tradeable_lanes)
        exposure_blockers = _trade_exposure_blockers(self.packet)

        if decision.action == "TRADE":
            if not selected_lane_ids:
                issues.append(VerificationIssue("LANE_REQUIRED", "TRADE requires selected_lane_id or selected_lane_ids"))
            if exposure_blockers:
                issues.append(VerificationIssue("EXPOSURE_BLOCKS_TRADE", "; ".join(exposure_blockers[:3])))
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
            if _target_requires_entry(self.packet) and not exposure_blockers and attack_lane_ids:
                issues.append(
                    VerificationIssue(
                        "ATTACK_ADVICE_REQUIRES_TRADE",
                        "ai_attack_advice recommends current tradeable LIVE_READY lane(s) while the daily target "
                        "is still open. Protected trader exposure is not a no-trade gate; choose TRADE or rerun "
                        f"the advice after a named hard blocker fires: {', '.join(attack_lane_ids[:3])}",
                    )
                )
            if _target_requires_entry(self.packet) and not exposure_blockers and tradeable_lanes:
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
        elif decision.action in {"PROTECT", "TIGHTEN_SL", "CLOSE"}:
            if positions <= 0:
                issues.append(VerificationIssue("NO_OPEN_POSITION", f"{decision.action} requires an open position"))
            if decision.action == "CLOSE":
                self._verify_close_trade_ids(decision, issues)
                self._verify_close_discipline(decision, issues)
        # CLOSE attached to TRADE: validate close_trade_ids and discipline
        # even when action is TRADE (a TRADE receipt may also retire prior
        # positions, and those closures must clear the same gates).
        if decision.action == "TRADE" and decision.close_trade_ids:
            self._verify_close_trade_ids(decision, issues)
            self._verify_close_discipline(decision, issues)

        return VerificationResult(allowed=not any(issue.severity == "BLOCK" for issue in issues), issues=tuple(issues))

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

        Gate B — operator authorization (hardened 2026-05-13 after
        the 00:33 JST mass-close incident proved the JSON-field
        version was honor-system). The decision-receipt
        `operator_close_authorized` boolean is NO LONGER ACCEPTED as
        authorization on its own — a trader that fills its own JSON
        receipt can set the flag. Authorization must come from a
        path the trader code cannot self-set within the receipt-write
        flow:

          1. `QR_OPERATOR_CLOSE_OVERRIDE=1` in the operator shell, OR
          2. A fresh `data/.operator_close_token` file (mtime within
             OPERATOR_CLOSE_TOKEN_FRESH_SECONDS = 5 minutes), created
             by `touch data/.operator_close_token` before each CLOSE
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
        still_valid: list[str] = []
        for tid in decision.close_trade_ids:
            pos = position_by_tid.get(str(tid))
            if pos is None:
                # `_verify_close_trade_ids` already flagged the unknown id;
                # do not double-report.
                continue
            pair = str(pos.get("pair") or "")
            side = str(pos.get("side") or "")
            invalidated, _reason = _close_thesis_invalidated(
                self.packet,
                pair,
                side,
                decision=decision,
            )
            if not invalidated:
                still_valid.append(
                    f"{tid} ({pair} {side})"
                )

        if still_valid:
            issues.append(
                VerificationIssue(
                    "CLOSE_THESIS_STILL_VALID",
                    "CLOSE rejected: thesis still valid (no BOS/CHOCH against "
                    "side on M15/H4 and no invalidation_price hit) for: "
                    + ", ".join(still_valid),
                )
            )

        # Gate B (hardened 2026-05-13): operator authorization can come
        # only from env or token file — NEVER from the JSON receipt
        # field. The trader fills its own decision payload, so the
        # boolean was a self-authorizing escape that drove the 15:33
        # UTC mass-close.
        if not _operator_close_override_active() and not _operator_close_token_fresh():
            issues.append(
                VerificationIssue(
                    "CLOSE_OPERATOR_AUTH_REQUIRED",
                    "CLOSE rejected: requires QR_OPERATOR_CLOSE_OVERRIDE=1 in "
                    "the operator shell OR a fresh data/.operator_close_token "
                    f"(mtime within {OPERATOR_CLOSE_TOKEN_FRESH_SECONDS}s). The "
                    "receipt's `operator_close_authorized` field is advisory "
                    "only and is no longer accepted as authorization "
                    "(2026-05-12T15:33 UTC mass-close incident, "
                    "feedback_no_unilateral_close.md).",
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
    ],
    "properties": {
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
                "strategy": _small_dict(strategy_index.get((pair, direction)), ("status", "pretrade_net_jpy", "live_net_jpy", "live_worst_jpy", "required_fix")),
                "story": _small_dict(story_index.get(pair), ("methods", "themes", "examples")),
                "forecast": _lane_forecast_packet(intent.get("metadata")),
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
        "remaining_target_jpy": target.get("remaining_target_jpy"),
        "remaining_risk_budget_jpy": target.get("remaining_risk_budget_jpy"),
    }


def _allowed_refs(
    *,
    snapshot: dict[str, Any],
    target: dict[str, Any],
    lanes: list[dict[str, Any]],
    attack_advice: dict[str, Any] | None,
) -> list[str]:
    # Per docs/SKILL_trader.md the playbook prescribes a richer set of evidence
    # refs than the base broker/target/lane triple — the trader is required to
    # cite per-pair charts, cross-asset, flow, levels, currency strength,
    # economic calendar, and COT data. The verifier therefore must accept these
    # refs as known; otherwise every well-formed decision is rejected with
    # UNKNOWN_EVIDENCE_REF and the cycle never reaches the gateway.
    timeframes = DEFAULT_PAIR_CHART_TIMEFRAMES
    structure_keys = ("structure",)
    cross_assets = ("dxy", "USB10Y_USD", "USB02Y_USD", "spx", "gold", "oil", "btc")
    refs = ["broker:snapshot", "target:daily"]
    pairs: set[str] = set()
    currencies: set[str] = set()
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
        refs.extend(
            [
                f"flow:{pair}",
                f"levels:{pair}",
                f"calendar:{pair}",
                f"strength:{pair}",
                f"option:skew:{pair}",
                f"cross:correlations:{pair}",
            ]
        )
    for currency in currencies:
        refs.append(f"cot:{currency}")
        refs.append(f"strength:{currency}")
        refs.append(f"calendar:{currency}")
    for asset in cross_assets:
        refs.append(f"cross:{asset}")
    refs.extend(["cross:dxy", "cross:correlations", "option:skew", "option:skew:unknown"])
    if attack_advice:
        refs.append("attack:advice")
        for lane_id in attack_advice.get("recommended_now_lane_ids", []) or []:
            refs.append(f"attack:lane:{lane_id}")
        for lane_id in attack_advice.get("watchlist_lane_ids", []) or []:
            refs.append(f"attack:lane:{lane_id}")
    return sorted(set(refs))


def _attack_advice_packet(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {"evidence_ref": "attack:advice", "status": "missing"}
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
        "settings_advice": payload.get("settings_advice") if isinstance(payload.get("settings_advice"), dict) else {},
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
        # SL-free regime: trader-owned TP-only positions are layerable
        # (user directive 「SLいらない」 / 「損失を出さないで稼ぎまくる」).
        sl_ok = position.get("stop_loss") is not None or sl_free_active
        if (
            owner == "trader"
            and position.get("take_profit") is not None
            and sl_ok
        ):
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
        if lane_id and lane_id in current and lane_id not in lane_ids:
            lane_ids.append(lane_id)
    return lane_ids


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


def _currencies_from_pairs(pairs: tuple[str, ...]) -> tuple[str, ...]:
    currencies: set[str] = set()
    for pair in pairs:
        for currency in pair.split("_"):
            if currency:
                currencies.add(currency)
    return tuple(sorted(currencies))


def _market_context_packet(
    *,
    pairs: tuple[str, ...],
    currencies: tuple[str, ...],
    pair_charts_path: Path,
    cross_asset_path: Path,
    flow_path: Path,
    currency_strength_path: Path,
    levels_path: Path,
    calendar_path: Path,
    cot_path: Path,
    option_skew_path: Path,
) -> dict[str, Any]:
    artifacts = {
        "pair_charts": _load_optional_json(pair_charts_path),
        "cross_asset": _load_optional_json(cross_asset_path),
        "flow": _load_optional_json(flow_path),
        "currency_strength": _load_optional_json(currency_strength_path),
        "levels": _load_optional_json(levels_path),
        "calendar": _load_optional_json(calendar_path),
        "cot": _load_optional_json(cot_path),
        "option_skew": _load_optional_json(option_skew_path),
    }
    missing = [
        f"MISSING_{name.upper()}_ARTIFACT"
        for name, payload in artifacts.items()
        if payload is None
    ]
    pair_payloads = {
        pair: {
            "chart": _chart_summary(artifacts["pair_charts"], pair),
            "flow": _flow_summary(artifacts["flow"], pair),
            "levels": _levels_summary(artifacts["levels"], pair),
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
        "cross_asset": _cross_asset_summary(artifacts["cross_asset"]),
        "currency_strength": _currency_strength_summary(artifacts["currency_strength"], currencies),
        "cot": _cot_summary(artifacts["cot"], currencies),
        "issues": issues[:40],
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
                ),
            ),
            "regime_state": regime.get("state"),
            "regime_confidence": regime.get("confidence"),
            "trend_score": family.get("trend_score"),
            "mean_rev_score": family.get("mean_rev_score"),
            "breakout_score": family.get("breakout_score"),
            "disagreement": family.get("disagreement"),
            "last_jump_bars_ago": stat.get("last_jump_bars_ago"),
            "lag1_autocorr": stat.get("lag1_autocorr"),
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
    readings = [
        _small_dict(item, ("tenor", "rr_25d", "atm_iv", "bf_25d", "issue"))
        for item in (payload or {}).get("readings", []) or []
        if isinstance(item, dict) and item.get("pair") == pair
    ]
    return {"readings": readings[:3]}


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
