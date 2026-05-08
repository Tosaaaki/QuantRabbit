from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quant_rabbit.models import BrokerPosition, BrokerSnapshot, Owner, Side
from quant_rabbit.paths import (
    DEFAULT_DAILY_TARGET_STATE,
    DEFAULT_PAIR_CHARTS,
    DEFAULT_POSITION_MANAGEMENT,
    DEFAULT_POSITION_MANAGEMENT_REPORT,
    DEFAULT_TRADER_DECISION,
)
from quant_rabbit.risk import RiskPolicy
from quant_rabbit.strategy.intent_generator import (
    GEOMETRY_ATR_TIMEFRAME,
    GEOMETRY_SPREAD_FLOOR_MULT,
    _atr_pips_for,
    _load_pair_charts,
    _session_bucket_for,
)


ACTION_HOLD_PROTECTED = "HOLD_PROTECTED"
ACTION_HOLD_SL_FREE = "HOLD_SL_FREE"
ACTION_PROFIT_PROTECT = "PROFIT_PROTECT_REQUIRED"
ACTION_REVIEW_EXIT = "REVIEW_EXIT"
ACTION_REPAIR_PROTECTION = "REPAIR_PROTECTION_REQUIRED"
# Adaptive TP management actions (user 2026-05-08「ミクロとマクロの視点が
# ないとできない」「確実に利益を取って」「伸ばすとこは伸ばす、限界なら
# 見極める」). Each action carries a recommended_take_profit that the
# position_execution gateway issues as a DEPENDENT_ORDER_REPLACE TP update.
ACTION_HARVEST_TP = "HARVEST_TP"        # Pull TP near current price to lock profit fast
ACTION_NARROW_TP = "NARROW_TP"          # Pull TP partway (halfway) toward current price
ACTION_EXTEND_TP = "EXTEND_TP"          # Push TP further out when momentum keeps running


# When `QR_TRADER_DISABLE_SL_REPAIR=1` the protection gateway treats a missing
# SL on a trader-owned position as deliberate (the user's "SLいらない" directive,
# `feedback_no_tight_sl_thin_market.md`) and emits HOLD_SL_FREE instead of
# REPAIR_PROTECTION_REQUIRED. TP repair, profit protection, and contradiction
# exits still apply.
def _trader_sl_repair_disabled() -> bool:
    return os.environ.get("QR_TRADER_DISABLE_SL_REPAIR", "").strip() in {"1", "true", "TRUE", "yes", "YES"}

# Profit protection must not move SL to breakeven while the market is still
# inside ordinary execution noise. Use the same spread floor as entry geometry
# plus one current M5 ATR: this is market-derived noise room, not a profit gate.
PROFIT_PROTECTION_NOISE_ATR_MULT = 1.0
PROFIT_PROTECTION_SPREAD_MULT = GEOMETRY_SPREAD_FLOOR_MULT


@dataclass(frozen=True)
class ManagedPosition:
    trade_id: str
    pair: str
    side: str
    units: int
    action: str
    unrealized_pl_jpy: float
    remaining_risk_jpy: float | None
    remaining_reward_jpy: float | None
    same_direction_score: float | None
    opposite_direction_score: float | None
    recommended_stop_loss: float | None
    recommended_take_profit: float | None
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class PositionManagementDecision:
    generated_at_utc: str
    action: str
    positions: tuple[ManagedPosition, ...]


class PositionManager:
    """Manage open exposure as a trader decision, not a passive monitor."""

    def __init__(
        self,
        *,
        trader_decision_path: Path = DEFAULT_TRADER_DECISION,
        pair_charts_path: Path = DEFAULT_PAIR_CHARTS,
        output_path: Path = DEFAULT_POSITION_MANAGEMENT,
        report_path: Path = DEFAULT_POSITION_MANAGEMENT_REPORT,
    ) -> None:
        self.trader_decision_path = trader_decision_path
        self.pair_charts_path = pair_charts_path
        self.output_path = output_path
        self.report_path = report_path

    def run(self, snapshot: BrokerSnapshot) -> PositionManagementDecision:
        generated_at = datetime.now(timezone.utc).isoformat()
        scores = _load_scores(self.trader_decision_path)
        pair_charts = _load_pair_charts(self.pair_charts_path)
        trader_positions = tuple(position for position in snapshot.positions if position.owner == Owner.TRADER)
        managed = tuple(self._manage_position(position, snapshot, scores, pair_charts) for position in trader_positions)
        action = _aggregate_action(managed)
        decision = PositionManagementDecision(generated_at_utc=generated_at, action=action, positions=managed)
        self._write(decision)
        return decision

    def _manage_position(
        self,
        position: BrokerPosition,
        snapshot: BrokerSnapshot,
        scores: dict[tuple[str, str], float],
        pair_charts: dict[str, dict[str, Any]] | None,
    ) -> ManagedPosition:
        same_score = scores.get((position.pair, position.side.value))
        opposite_score = scores.get((position.pair, _opposite(position.side)))
        remaining_risk = _remaining_risk_jpy(position, snapshot.quotes, snapshot.home_conversions)
        remaining_reward = _remaining_reward_jpy(position, snapshot.quotes, snapshot.home_conversions)
        reasons: list[str] = []
        quote = snapshot.quotes.get(position.pair)
        recommended_stop_loss: float | None = None
        recommended_take_profit: float | None = None
        reasons.extend(_session_protection_notes(position, quote, pair_charts))

        sl_free_hold = (
            position.stop_loss is None
            and position.owner == Owner.TRADER
            and _trader_sl_repair_disabled()
        )
        sl_free_owned = (
            position.owner == Owner.TRADER and _trader_sl_repair_disabled()
        )

        # Contradiction-based auto-REVIEW_EXIT was producing churn loops on
        # SL-free trader-owned positions: chart regime flips frequently on M1/M5,
        # so a -500 JPY EUR_USD SHORT got auto-closed and a fresh SHORT was
        # opened in the same minute by the basket layer (2026-05-08 470395 →
        # 470415, ~2 min apart). User directives 「SLいらない」「無駄な損切り
        # はしない」「損失を出さないで稼ぎまくる」 (`feedback_no_tight_sl_thin_market.md`,
        # `feedback_offense_sizing.md`) make operator-driven exits via
        # `close_trade_ids` (commit 32deccb) authoritative. Suppress the
        # auto-contradiction trigger on SL-free trader-owned positions; let the
        # operator decide when a SHORT is genuinely contradicted vs noise.
        contradicted = (
            not sl_free_owned
            and opposite_score is not None
            and same_score is not None
            and opposite_score >= same_score + 20
            and position.unrealized_pl_jpy < 0
        )
        if not contradicted and not sl_free_owned:
            contradicted = _chart_regime_contradicted(position, pair_charts)
        if contradicted:
            if opposite_score is not None and same_score is not None:
                reasons.append(f"opposite thesis score {opposite_score:.1f} materially exceeds same-direction {same_score:.1f}")
            reasons.append(f"chart regime contradicts {position.side.value} (losing {position.unrealized_pl_jpy:.1f} JPY)")
            action = ACTION_REVIEW_EXIT
        elif position.stop_loss is None or position.take_profit is None:
            missing = []
            if position.take_profit is None:
                missing.append("TP")
            if position.stop_loss is None:
                missing.append("SL")
            reasons.append(f"missing {'/'.join(missing)}")
            if position.stop_loss is None:
                if sl_free_hold:
                    reasons.append("trader SL-repair disabled (QR_TRADER_DISABLE_SL_REPAIR=1); discretionary SL-free hold")
                    action = ACTION_HOLD_SL_FREE
                    # Adaptive TP: when the SL-free position is profitable
                    # and chart_story available, evaluate EXTEND/HARVEST/
                    # NARROW/EXIT (user 2026-05-08「ミクロとマクロの視点」).
                    adaptive_action, adaptive_tp, adaptive_reasons = _adaptive_tp_action(
                        position, quote, pair_charts
                    )
                    reasons.extend(adaptive_reasons)
                    if adaptive_action != ACTION_HOLD_PROTECTED:
                        if adaptive_action == ACTION_REVIEW_EXIT:
                            action = ACTION_REVIEW_EXIT
                        elif adaptive_tp is not None:
                            recommended_take_profit = adaptive_tp
                            action = adaptive_action
                else:
                    recommended_stop_loss = _repair_stop_loss(position, quote, snapshot.quotes, snapshot.home_conversions)
                    if recommended_stop_loss is None:
                        reasons.append("no market-valid capped SL repair is available; exposure needs exit review")
                        action = ACTION_REVIEW_EXIT
                    else:
                        reasons.append(f"repair SL candidate {recommended_stop_loss:.5f}")
                        action = ACTION_REPAIR_PROTECTION
            else:
                action = ACTION_REPAIR_PROTECTION
            if position.take_profit is None:
                basis_stop = recommended_stop_loss if recommended_stop_loss is not None else position.stop_loss
                recommended_take_profit = _repair_take_profit(position, basis_stop, quote)
                if recommended_take_profit is not None:
                    reasons.append(f"repair TP candidate {recommended_take_profit:.5f}")
        else:
            profit_protection_needed, profit_reasons = _profit_protection_needed(
                position,
                remaining_risk,
                quote,
                snapshot.quotes,
                snapshot.home_conversions,
                pair_charts,
            )
            reasons.extend(profit_reasons)
            sl_free_global = (
                position.owner == Owner.TRADER and _trader_sl_repair_disabled()
            )
            if profit_protection_needed and sl_free_global:
                # SL-free directive: do not auto-tighten SL even on profit.
                # The operator decides when to harvest. TP stays as the auto
                # exit; auto-added BE-stop is exactly the noise-hunt vector
                # the user told us to stop generating ("意図的じゃないSLは
                # 生成するな" 2026-05-07).
                reasons.append("profit-protect skipped (QR_TRADER_DISABLE_SL_REPAIR=1); operator-managed harvest")
                action = ACTION_HOLD_SL_FREE
            elif profit_protection_needed:
                reasons.append("profit clears remaining risk plus current session noise")
                recommended_stop_loss = _break_even_stop(position, quote)
                if recommended_stop_loss is None:
                    reasons.append("break-even SL is not market-valid yet")
                else:
                    reasons.append(f"break-even SL candidate {recommended_stop_loss:.5f}")
                action = ACTION_PROFIT_PROTECT
            elif (
                not sl_free_owned
                and opposite_score is not None
                and same_score is not None
                and opposite_score >= same_score + 20
                and position.unrealized_pl_jpy < 0
            ):
                reasons.append(f"opposite thesis score {opposite_score:.1f} materially exceeds same-direction {same_score:.1f}")
                action = ACTION_REVIEW_EXIT
            else:
                reasons.append("TP/SL present and current thesis is not contradicted enough to force exit")
                action = ACTION_HOLD_PROTECTED

        if remaining_risk is not None:
            reasons.append(f"remaining risk about {remaining_risk:.0f} JPY")
        elif position.stop_loss is not None:
            reasons.append("remaining risk cannot be converted to JPY from current broker snapshot")
        if remaining_reward is not None:
            reasons.append(f"remaining reward about {remaining_reward:.0f} JPY")
        elif position.take_profit is not None:
            reasons.append("remaining reward cannot be converted to JPY from current broker snapshot")

        return ManagedPosition(
            trade_id=position.trade_id,
            pair=position.pair,
            side=position.side.value,
            units=position.units,
            action=action,
            unrealized_pl_jpy=round(position.unrealized_pl_jpy, 4),
            remaining_risk_jpy=round(remaining_risk, 2) if remaining_risk is not None else None,
            remaining_reward_jpy=round(remaining_reward, 2) if remaining_reward is not None else None,
            same_direction_score=same_score,
            opposite_direction_score=opposite_score,
            recommended_stop_loss=round(recommended_stop_loss, _price_precision(position.pair))
            if recommended_stop_loss is not None
            else None,
            recommended_take_profit=round(recommended_take_profit, _price_precision(position.pair))
            if recommended_take_profit is not None
            else None,
            reasons=tuple(reasons),
        )

    def _write(self, decision: PositionManagementDecision) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(asdict(decision), ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Position Management Report",
            "",
            f"- Generated at UTC: `{decision.generated_at_utc}`",
            f"- Action: `{decision.action}`",
            f"- Positions: `{len(decision.positions)}`",
            "",
            "## Positions",
            "",
        ]
        if not decision.positions:
            lines.append("- none")
        for item in decision.positions:
            lines.append(
                f"- `{item.trade_id}` `{item.pair} {item.side}` units=`{item.units}` "
                f"action=`{item.action}` upl=`{item.unrealized_pl_jpy:.1f}`"
            )
            lines.append(f"  - scores: same=`{item.same_direction_score}` opposite=`{item.opposite_direction_score}`")
            lines.append(
                f"  - protection plan: sl=`{item.recommended_stop_loss}` tp=`{item.recommended_take_profit}`"
            )
            for reason in item.reasons:
                lines.append(f"  - reason: {reason}")
        lines.extend(
            [
                "",
                "## Management Contract",
                "",
                "- Existing positions are managed before any new entry is considered.",
                "- Operator-managed manual/tagless positions are observed in broker truth but ignored by this gateway.",
                "- Missing TP/SL is a repair requirement, not a passive monitor state.",
                "- Profit protection is required once open profit clears remaining stop risk plus current session noise.",
                "- A materially stronger opposite thesis triggers exit review; the gateway still prevents fresh stacking.",
            ]
        )
        self.report_path.write_text("\n".join(lines) + "\n")


import re as _re

_PM_TF_BLOCK = _re.compile(r"\b(D|H4|H1|M30|M15|M5|M1)\(([^)]+)\)")
_PM_STRUCT = _re.compile(r"struct=(BOS|CHOCH)_(UP|DOWN)@")
_PM_ADX = _re.compile(r"ADX=([\d.]+)")
_PM_ST = _re.compile(r"ST=([+-])")
_PM_REGIME_UP = {"TREND_UP", "IMPULSE_UP", "BULL"}
_PM_REGIME_DOWN = {"TREND_DOWN", "IMPULSE_DOWN", "BEAR"}
_PM_REGIME_RANGE = {"RANGE", "UNCLEAR", "TRANSITION", "FAILURE_RISK"}


def _parse_tf(chart_story: str, tf: str) -> dict[str, Any] | None:
    """Pull the per-timeframe block out of an inline chart_story."""
    for m in _PM_TF_BLOCK.finditer(chart_story or ""):
        if m.group(1) != tf:
            continue
        body = m.group(2)
        head, _, rest = body.partition(",")
        out: dict[str, Any] = {"regime": head.strip()}
        adx_m = _PM_ADX.search(rest)
        if adx_m:
            try:
                out["adx"] = float(adx_m.group(1))
            except ValueError:
                pass
        st_m = _PM_ST.search(rest)
        if st_m:
            out["st"] = "UP" if st_m.group(1) == "+" else "DOWN"
        s_m = _PM_STRUCT.search(rest)
        if s_m:
            out["struct_dir"] = s_m.group(2)
        return out
    return None


def _classify_micro(chart_story: str, lane_dir: str) -> str:
    """Return ALIVE / DYING / DEAD for the M1+M5 momentum vs lane direction.

    ALIVE  – M5 ADX≥22 with regime/struct/ST aligned in lane direction.
    DEAD   – any of M1/M5 carries opposite struct, opposite ST, or
             opposite regime token (the lane is being walked into).
    DYING  – everything else (ADX falling, RANGE/FAILURE_RISK, mixed).
    """
    if lane_dir not in {"LONG", "SHORT"}:
        return "DYING"
    target_up = lane_dir == "LONG"
    m1 = _parse_tf(chart_story, "M1") or {}
    m5 = _parse_tf(chart_story, "M5") or {}

    def _flip(t: dict[str, Any]) -> bool:
        # struct opposite, ST opposite, or regime opposite — any of these is a flip
        sd = t.get("struct_dir")
        if sd and ((sd == "UP") != target_up):
            return True
        st = t.get("st")
        if st and ((st == "UP") != target_up):
            return True
        regime = str(t.get("regime") or "")
        if (regime in _PM_REGIME_UP and not target_up) or (regime in _PM_REGIME_DOWN and target_up):
            return True
        return False

    if _flip(m1) or _flip(m5):
        return "DEAD"

    m5_adx = m5.get("adx") if isinstance(m5.get("adx"), (int, float)) else 0.0
    m5_regime = str(m5.get("regime") or "")
    m5_aligned = (
        m5_regime in (_PM_REGIME_UP if target_up else _PM_REGIME_DOWN)
        or (m5.get("struct_dir") == ("UP" if target_up else "DOWN"))
    )
    if m5_aligned and m5_adx >= 22.0:
        return "ALIVE"
    return "DYING"


def _classify_macro(chart_story: str, lane_dir: str) -> str:
    """Return ALIGNED / WEAKENING / REVERSED for H1+H4+D vs lane direction.

    ALIGNED   – H1 and H4 both in lane direction, D not opposing.
    REVERSED  – H1 and H4 both opposite (or D explicitly opposite).
    WEAKENING – partially aligned (e.g. H1 yes but H4 RANGE/UNCLEAR).
    """
    if lane_dir not in {"LONG", "SHORT"}:
        return "WEAKENING"
    target_up = lane_dir == "LONG"

    def _bias(tf: str) -> str:
        # Macro bias relies on the TF's regime label only. Struct events
        # (BOS/CHOCH) are timestamped at the last swing event and can be
        # days old on D/H4 — using them as a tie-breaker on RANGE/UNCLEAR
        # regimes mis-classifies positions with old downward struct as
        # REVERSED while H1/H4 still trend up (2026-05-08 EUR_USD case).
        t = _parse_tf(chart_story, tf) or {}
        regime = str(t.get("regime") or "")
        if regime in _PM_REGIME_UP:
            return "UP"
        if regime in _PM_REGIME_DOWN:
            return "DOWN"
        # RANGE / UNCLEAR / TRANSITION / FAILURE_RISK: no directional bias.
        return "MIXED"

    h1 = _bias("H1")
    h4 = _bias("H4")
    d = _bias("D")

    aligned = "UP" if target_up else "DOWN"
    opposite = "DOWN" if target_up else "UP"

    if h1 == aligned and h4 == aligned and d != opposite:
        return "ALIGNED"
    if h1 == opposite and h4 == opposite:
        return "REVERSED"
    if d == opposite:
        return "REVERSED"
    return "WEAKENING"


def _adaptive_tp_action(
    position: BrokerPosition,
    quote,
    pair_charts: dict[str, dict[str, Any]] | None,
) -> tuple[str, float | None, list[str]]:
    """Decide adaptive TP action for an SL-free, profitable, TP-set position.

    Combines micro (M1/M5) momentum and macro (H1/H4/D) alignment to choose
    between EXTEND / HARVEST / NARROW / EXIT / HOLD. Returns
    (action, recommended_tp, reasons).

    Conservative gates:
    - Position must be currently profitable (`unrealized_pl_jpy > 0`).
    - Position must already have a TP (operator-set during entry).
    - Decision falls back to HOLD_PROTECTED on any data gap.

    User directive 2026-05-08: 「確実に利益を取って。伸ばすとこは伸ばす、
    限界なら見極める。ミクロとマクロの視点」.
    """
    reasons: list[str] = []
    if position.take_profit is None or quote is None:
        return ACTION_HOLD_PROTECTED, None, reasons
    if position.unrealized_pl_jpy <= 0:
        # Don't manage TPs on losing positions; let market structure decide
        # (operator may still propose CLOSE via close_trade_ids on real reversal).
        return ACTION_HOLD_PROTECTED, None, reasons

    pc = (pair_charts or {}).get(position.pair) if pair_charts else None
    chart_story = str(pc.get("chart_story") or "") if isinstance(pc, dict) else ""
    if not chart_story:
        return ACTION_HOLD_PROTECTED, None, reasons

    lane_dir = position.side.value
    target_up = lane_dir == "LONG"
    micro = _classify_micro(chart_story, lane_dir)
    macro = _classify_macro(chart_story, lane_dir)
    reasons.append(f"micro={micro} macro={macro}")

    # Decision matrix (rows = macro, cols = micro)
    matrix = {
        ("ALIGNED", "ALIVE"): ACTION_EXTEND_TP,
        ("ALIGNED", "DYING"): ACTION_HARVEST_TP,
        ("ALIGNED", "DEAD"): ACTION_HARVEST_TP,
        ("WEAKENING", "ALIVE"): ACTION_HARVEST_TP,
        ("WEAKENING", "DYING"): ACTION_NARROW_TP,
        ("WEAKENING", "DEAD"): ACTION_REVIEW_EXIT,
        ("REVERSED", "ALIVE"): ACTION_REVIEW_EXIT,
        ("REVERSED", "DYING"): ACTION_REVIEW_EXIT,
        ("REVERSED", "DEAD"): ACTION_REVIEW_EXIT,
    }
    action = matrix.get((macro, micro), ACTION_HOLD_PROTECTED)

    pip_factor = 100.0 if position.pair.endswith("_JPY") else 10000.0
    pip = 1.0 / pip_factor
    cur = quote.bid if not target_up else quote.ask
    # Distance bookkeeping (positive when TP still ahead)
    if target_up:
        tp_pips_remaining = (position.take_profit - cur) * pip_factor
    else:
        tp_pips_remaining = (cur - position.take_profit) * pip_factor

    new_tp: float | None = None

    if action == ACTION_EXTEND_TP:
        # If already > 60% to TP and still alive+aligned, extend by 50% of
        # the original distance so winners run further. Skip when the move
        # is already late (<20% remaining might just be TP fill imminent).
        if tp_pips_remaining > 5.0:
            extra_pips = max(10.0, tp_pips_remaining * 0.5)
            new_tp = (position.take_profit + extra_pips * pip) if target_up else (position.take_profit - extra_pips * pip)
            reasons.append(f"extend +{extra_pips:.1f}p (winners run, micro alive macro aligned)")
        else:
            # TP imminent — just hold
            action = ACTION_HOLD_PROTECTED
            reasons.append(f"TP {tp_pips_remaining:.1f}p imminent; hold")
    elif action == ACTION_HARVEST_TP:
        # Pull TP to current price + small safety buffer (5p) so the broker
        # accepts it and the next routine bar locks the profit. Don't move
        # backwards (wider) — bail to HOLD if our buffered TP is worse.
        buffer_pips = 5.0
        candidate = (cur + buffer_pips * pip) if target_up else (cur - buffer_pips * pip)
        # Only narrow (move TP closer to current price)
        if (target_up and candidate < position.take_profit) or (not target_up and candidate > position.take_profit):
            new_tp = candidate
            reasons.append(f"harvest TP→cur+{buffer_pips:.0f}p (lock profit, momentum dying)")
        else:
            action = ACTION_HOLD_PROTECTED
            reasons.append(f"harvest skipped (current {cur} already closer than buffered TP)")
    elif action == ACTION_NARROW_TP:
        # Bring TP halfway between current price and existing TP.
        midpoint = (cur + position.take_profit) / 2.0
        if (target_up and midpoint < position.take_profit) or (not target_up and midpoint > position.take_profit):
            new_tp = midpoint
            reasons.append(f"narrow TP→midpoint {midpoint:.5f} (macro weakening, micro dying)")
        else:
            action = ACTION_HOLD_PROTECTED
            reasons.append("narrow skipped (midpoint not closer)")
    elif action == ACTION_REVIEW_EXIT:
        # MARKET close handled by position_execution.py path on REVIEW_EXIT;
        # no recommended_tp needed.
        reasons.append("EXIT (macro reversed or weak+dead micro)")
    else:
        reasons.append("HOLD (mixed signal)")

    return action, new_tp, reasons


def _load_scores(path: Path) -> dict[tuple[str, str], float]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    scores: dict[tuple[str, str], float] = {}
    for item in payload.get("scores", []) or []:
        if not isinstance(item, dict):
            continue
        pair = str(item.get("pair") or "")
        direction = str(item.get("direction") or "")
        if not pair or not direction:
            continue
        key = (pair, direction)
        score = float(item.get("score") or 0.0)
        if score > scores.get(key, -10_000.0):
            scores[key] = score
    return scores


def _aggregate_action(positions: tuple[ManagedPosition, ...]) -> str:
    actions = {position.action for position in positions}
    if ACTION_REPAIR_PROTECTION in actions:
        return ACTION_REPAIR_PROTECTION
    if ACTION_REVIEW_EXIT in actions:
        return ACTION_REVIEW_EXIT
    if ACTION_PROFIT_PROTECT in actions:
        return ACTION_PROFIT_PROTECT
    if positions:
        return ACTION_HOLD_PROTECTED
    return "NO_POSITION"


def _remaining_risk_jpy(position: BrokerPosition, quotes, home_conversions=None) -> float | None:
    if position.stop_loss is None:
        return None
    pips = (position.entry_price - position.stop_loss) * _pip_factor(position.pair)
    if position.side == Side.SHORT:
        pips = (position.stop_loss - position.entry_price) * _pip_factor(position.pair)
    jpy_per_pip = _jpy_per_pip(position, quotes, home_conversions or {})
    if jpy_per_pip is None:
        return None
    return max(0.0, pips) * jpy_per_pip


def _remaining_reward_jpy(position: BrokerPosition, quotes, home_conversions=None) -> float | None:
    if position.take_profit is None:
        return None
    pips = (position.take_profit - position.entry_price) * _pip_factor(position.pair)
    if position.side == Side.SHORT:
        pips = (position.entry_price - position.take_profit) * _pip_factor(position.pair)
    jpy_per_pip = _jpy_per_pip(position, quotes, home_conversions or {})
    if jpy_per_pip is None:
        return None
    return max(0.0, pips) * jpy_per_pip


def _profit_protection_needed(
    position: BrokerPosition,
    remaining_risk: float | None,
    quote,
    quotes,
    home_conversions,
    pair_charts: dict[str, dict[str, Any]] | None,
) -> tuple[bool, tuple[str, ...]]:
    if position.stop_loss is None:
        return True, ("profit protection requires a stop loss first",)
    if _stop_is_break_even_or_better(position):
        return False, ("SL already at/through break-even",)
    if remaining_risk is None or remaining_risk <= 0:
        return False, ("profit protection deferred: remaining risk cannot be measured",)

    noise_jpy = _profit_protection_noise_jpy(position, quote, quotes, home_conversions, pair_charts)
    if noise_jpy is None:
        return False, ("profit protection deferred until session ATR/spread noise can be measured",)

    threshold = remaining_risk + noise_jpy
    if position.unrealized_pl_jpy < threshold:
        return (
            False,
            (
                f"profit protection deferred: upl {position.unrealized_pl_jpy:.0f} JPY < "
                f"remaining risk {remaining_risk:.0f} + session noise {noise_jpy:.0f}",
            ),
        )
    return (
        True,
        (
            f"profit protection trigger: upl {position.unrealized_pl_jpy:.0f} JPY >= "
            f"remaining risk {remaining_risk:.0f} + session noise {noise_jpy:.0f}",
        ),
    )


def _profit_protection_noise_jpy(
    position: BrokerPosition,
    quote,
    quotes,
    home_conversions,
    pair_charts: dict[str, dict[str, Any]] | None,
) -> float | None:
    jpy_per_pip = _jpy_per_pip(position, quotes, home_conversions or {})
    if jpy_per_pip is None or jpy_per_pip <= 0:
        return None
    noise_pips: list[float] = []
    atr_pips = _atr_pips_for(position.pair, pair_charts, GEOMETRY_ATR_TIMEFRAME)
    if atr_pips is not None and atr_pips > 0:
        noise_pips.append(atr_pips * PROFIT_PROTECTION_NOISE_ATR_MULT)
    spread_pips = _spread_pips(position.pair, quote)
    if spread_pips is not None and spread_pips > 0:
        noise_pips.append(spread_pips * PROFIT_PROTECTION_SPREAD_MULT)
    if not noise_pips:
        return None
    return max(noise_pips) * jpy_per_pip


def _session_protection_notes(
    position: BrokerPosition,
    quote,
    pair_charts: dict[str, dict[str, Any]] | None,
) -> tuple[str, ...]:
    notes: list[str] = []
    session_bucket = _session_bucket_for(position.pair, pair_charts)
    if session_bucket:
        notes.append(f"session bucket {session_bucket}")
    else:
        notes.append("session bucket unavailable")
    atr_pips = _atr_pips_for(position.pair, pair_charts, GEOMETRY_ATR_TIMEFRAME)
    if atr_pips is not None:
        notes.append(f"{GEOMETRY_ATR_TIMEFRAME} ATR about {atr_pips:.1f}pip")
    else:
        notes.append(f"{GEOMETRY_ATR_TIMEFRAME} ATR unavailable for session SL/TP noise evaluation")
    spread_pips = _spread_pips(position.pair, quote)
    if spread_pips is not None:
        notes.append(f"current spread about {spread_pips:.1f}pip")
    sl_pips = _position_sl_pips(position)
    if sl_pips is not None:
        if atr_pips is not None and atr_pips > 0:
            notes.append(f"SL distance {sl_pips:.1f}pip = {sl_pips / atr_pips:.1f}x {GEOMETRY_ATR_TIMEFRAME} ATR")
        else:
            notes.append(f"SL distance {sl_pips:.1f}pip")
    tp_pips = _position_tp_pips(position)
    if tp_pips is not None:
        if atr_pips is not None and atr_pips > 0:
            notes.append(f"TP distance {tp_pips:.1f}pip = {tp_pips / atr_pips:.1f}x {GEOMETRY_ATR_TIMEFRAME} ATR")
        else:
            notes.append(f"TP distance {tp_pips:.1f}pip")
    return tuple(notes)


def _stop_is_break_even_or_better(position: BrokerPosition) -> bool:
    if position.stop_loss is None:
        return False
    if position.side == Side.LONG:
        return position.stop_loss >= position.entry_price
    return position.stop_loss <= position.entry_price


def _position_sl_pips(position: BrokerPosition) -> float | None:
    if position.stop_loss is None:
        return None
    distance = position.entry_price - position.stop_loss if position.side == Side.LONG else position.stop_loss - position.entry_price
    return max(0.0, distance * _pip_factor(position.pair))


def _position_tp_pips(position: BrokerPosition) -> float | None:
    if position.take_profit is None:
        return None
    distance = position.take_profit - position.entry_price if position.side == Side.LONG else position.entry_price - position.take_profit
    return max(0.0, distance * _pip_factor(position.pair))


def _spread_pips(pair: str, quote) -> float | None:
    if quote is None:
        return None
    spread = (quote.ask - quote.bid) * _pip_factor(pair)
    return spread if spread > 0 else None


def _break_even_stop(position: BrokerPosition, quote) -> float | None:
    if quote is None:
        return position.entry_price
    if position.side == Side.LONG:
        if quote.bid <= position.entry_price:
            return None
        return position.entry_price
    if quote.ask >= position.entry_price:
        return None
    return position.entry_price


def _repair_stop_loss(position: BrokerPosition, quote, quotes, home_conversions=None) -> float | None:
    jpy_per_pip = _jpy_per_pip(position, quotes, home_conversions or {})
    if jpy_per_pip is None or jpy_per_pip <= 0:
        return None
    cap = _repair_loss_cap_jpy()
    if cap is None:
        return None
    cap_pips = cap / jpy_per_pip
    repair_pips = min(cap_pips, _default_repair_stop_pips(position.pair))
    distance = repair_pips / _pip_factor(position.pair)
    candidate = position.entry_price - distance if position.side == Side.LONG else position.entry_price + distance
    if quote is None:
        return candidate
    if not _market_valid_stop(position, candidate, quote):
        return None
    return candidate


def _repair_take_profit(position: BrokerPosition, stop_loss: float | None, quote) -> float | None:
    if stop_loss is None:
        return None
    risk_distance = abs(position.entry_price - stop_loss)
    if risk_distance <= 0:
        return None
    candidate = (
        position.entry_price + risk_distance * 1.5
        if position.side == Side.LONG
        else position.entry_price - risk_distance * 1.5
    )
    if quote is None:
        return candidate
    if position.side == Side.LONG and candidate <= quote.ask:
        return None
    if position.side == Side.SHORT and candidate >= quote.bid:
        return None
    return candidate


def _market_valid_stop(position: BrokerPosition, stop_loss: float, quote) -> bool:
    if position.side == Side.LONG:
        return stop_loss < quote.bid
    return stop_loss > quote.ask


def _default_repair_stop_pips(pair: str) -> float:
    return 10.0 if pair.endswith("_JPY") else 8.0


def _chart_regime_contradicted(position: BrokerPosition, pair_charts: dict[str, dict[str, Any]] | None) -> bool:
    """Check if pair-chart composite regime strongly contradicts the position side.

    A LONG position is contradicted when short_score >= 0.8 and long_score < 0.2
    (and losing). A SHORT position is contradicted by the reverse. This catches
    regime contradiction when trader_decision.json lane scores don't show a wide
    enough gap but the chart composite clearly disagrees with the position.
    """
    if pair_charts is None or position.unrealized_pl_jpy >= 0:
        return False
    chart = pair_charts.get(position.pair)
    if chart is None:
        return False
    long_score = chart.get("long_score")
    short_score = chart.get("short_score")
    if long_score is None or short_score is None:
        return False
    if position.side == Side.LONG:
        return short_score >= 0.8 and long_score < 0.2
    return long_score >= 0.8 and short_score < 0.2


def _opposite(side: Side) -> str:
    return Side.SHORT.value if side == Side.LONG else Side.LONG.value


def _pip_factor(pair: str) -> int:
    return 100 if pair.endswith("_JPY") else 10000


def _price_precision(pair: str) -> int:
    return 3 if pair.endswith("_JPY") else 5


def _jpy_per_pip(position: BrokerPosition, quotes, home_conversions) -> float | None:
    if position.pair.endswith("_JPY"):
        return position.units / 100
    quote_ccy = position.pair.split("_", 1)[1]
    home_conversion = home_conversions.get(quote_ccy)
    if home_conversion is not None and home_conversion > 0:
        return (position.units / _pip_factor(position.pair)) * float(home_conversion)
    conversion_quote = quotes.get(f"{quote_ccy}_JPY")
    if conversion_quote is None:
        return None
    return (position.units / _pip_factor(position.pair)) * max(conversion_quote.bid, conversion_quote.ask)


def _repair_loss_cap_jpy() -> float | None:
    """Return the current per-trade cap for capped SL repair.

    Position repair must not widen exposure from a stale literal. Prefer the
    daily target ledger's equity-derived per-trade cap; use RiskPolicy's
    documented library default only when the ledger is absent in tests/ad-hoc
    runs.
    """
    if DEFAULT_DAILY_TARGET_STATE.exists():
        try:
            payload = json.loads(DEFAULT_DAILY_TARGET_STATE.read_text())
            value = float(payload.get("per_trade_risk_budget_jpy") or 0.0)
            if value > 0:
                return value
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None
    policy_cap = RiskPolicy().max_loss_jpy
    return float(policy_cap) if policy_cap is not None and policy_cap > 0 else None
