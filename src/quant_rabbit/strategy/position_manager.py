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
from quant_rabbit.strategy.intent_generator import (
    GEOMETRY_ATR_TIMEFRAME,
    GEOMETRY_SPREAD_FLOOR_MULT,
    _atr_pips_for,
    _load_pair_charts,
    _market_derived_reward_risk,
    _session_bucket_for,
)
from quant_rabbit.strategy.entry_thesis_ledger import (
    load_entry_thesis,
    load_latest_forecast,
    technical_invalidation_confirmation_reason,
    thesis_invalidation_hit_reason,
)
from quant_rabbit.strategy.price_action import (
    aggregate_price_action_score,
    classify_order_block_proximity,
    structural_tp_target,
)
from quant_rabbit.strategy.tp_rebalancer import (
    MAX_TP_DISTANCE_ATR_MULT,
    MIN_TP_TO_MARKET_PIPS,
    _forecast_runner_drag_reasons,
    _technical_harvest_pressure,
)


def _load_full_pair_charts(charts_path: Path = DEFAULT_PAIR_CHARTS) -> dict[str, dict[str, Any]]:
    """Load pair_charts.json keyed by pair, preserving the full views array.

    `_load_pair_charts` (intent_generator) flattens views into per-TF
    keys for ATR / regime extraction. The price-action lens needs the
    raw `views` list because it inspects swings, structure_events,
    liquidity, order_blocks, dealing_range — which the flat shape drops.
    Returns {} on missing/malformed file so callers can degrade gracefully.
    """
    if not charts_path.exists():
        return {}
    try:
        payload = json.loads(charts_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for chart in payload.get("charts", []) or []:
        pair = chart.get("pair")
        if isinstance(pair, str):
            out[pair] = chart
    return out


ACTION_HOLD_PROTECTED = "HOLD_PROTECTED"
ACTION_HOLD_SL_FREE = "HOLD_SL_FREE"
ACTION_BREAK_EVEN_STOP = "BREAK_EVEN_STOP"
ACTION_PROFIT_PROTECT = "PROFIT_PROTECT_REQUIRED"
ACTION_REVIEW_EXIT = "REVIEW_EXIT"
ACTION_TAKE_PROFIT_MARKET = "TAKE_PROFIT_MARKET"
ACTION_REPAIR_PROTECTION = "REPAIR_PROTECTION_REQUIRED"
ACTION_REPAIR_TAKE_PROFIT = "REPAIR_TAKE_PROFIT_REQUIRED"
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


def _manual_take_profit_owner(owner: Owner) -> bool:
    return owner in {Owner.MANUAL, Owner.UNKNOWN}


def _missing_tp_repair_enabled() -> bool:
    return os.environ.get("QR_ENABLE_MISSING_TP_REPAIR", "").strip() in {"1", "true", "TRUE", "yes", "YES"}


def _position_management_owner(owner: Owner) -> bool:
    return owner == Owner.TRADER or _manual_take_profit_owner(owner)

# Profit protection must not move SL to breakeven while the market is still
# inside ordinary execution noise. Use the same spread floor as entry geometry
# plus one current M5 ATR: this is market-derived noise room, not a profit gate.
PROFIT_PROTECTION_NOISE_ATR_MULT = 1.0
PROFIT_PROTECTION_SPREAD_MULT = GEOMETRY_SPREAD_FLOOR_MULT
# Soft forecast-persistence close evidence should not force a market close by
# itself. It does need to remove HOLD support when the remaining payoff is badly
# asymmetric, otherwise a stale loser can keep blocking capital recycle.
FORECAST_PERSISTENCE_CLOSE_REVIEW_MAX_REWARD_RISK = 0.5
# SL-free break-even/profit-lock is not initial SL repair. It is a profit-only
# escape hatch after executable MFE clears current micro-noise. M5 is the live
# management timeframe already exposed by pair_charts. The spread multiplier is
# the identity value: profit locking only needs to clear the observed spread
# itself, not the much wider new-entry SL geometry floor.
PROFIT_BREAK_EVEN_ATR_TIMEFRAME = GEOMETRY_ATR_TIMEFRAME
PROFIT_BREAK_EVEN_NOISE_ATR_MULT = PROFIT_PROTECTION_NOISE_ATR_MULT
PROFIT_BREAK_EVEN_SPREAD_MULT = 1.0
# Profit-lock SL must not convert a planned TP trade into a micro-scalp.
# A broker-side stop is allowed only after the executable move has captured a
# majority of the current broker TP distance. The 60% gate represents "most of
# the planned reward has already developed"; before that point the TP is still
# the profit-taking mechanism and the SL-free directive should remain intact.
PROFIT_BREAK_EVEN_MIN_TP_PROGRESS = float(os.environ.get("QR_PROFIT_BREAK_EVEN_MIN_TP_PROGRESS", "0.60"))
# Adaptive TP contraction shares the same "majority of planned reward" idea as
# SL-free profit-lock. A HARVEST/NARROW TP may be pulled closer early only when
# the existing broker TP is already stale-wide versus the operating ATR cap.
ADAPTIVE_TP_CONTRACTION_MIN_PROGRESS = float(
    os.environ.get("QR_ADAPTIVE_TP_CONTRACTION_MIN_PROGRESS", str(PROFIT_BREAK_EVEN_MIN_TP_PROGRESS))
)
ADAPTIVE_TP_STALE_DISTANCE_ATR_MULT = MAX_TP_DISTANCE_ATR_MULT
# Quick volatility for broker-side BE/profit-lock must be recent enough that
# it describes the noise around the current quote, not a stale previous cycle.
# Granularity seconds are broker timeframe definitions; the quick window is the
# current management timeframe (M5) so M1/M5 realized range reacts before a
# slower ATR has fully adjusted.
QUICK_VOL_TIMEFRAMES = ("M1", "M5")
QUICK_VOL_GRANULARITY_SECONDS = {"M1": 60, "M5": 300}
QUICK_VOL_WINDOW_TIMEFRAME = PROFIT_BREAK_EVEN_ATR_TIMEFRAME
# Bollinger %B rail checks use the same edge convention as pattern_signals:
# the outer 15% of the band is "at rail". These are indicator-convention
# boundaries, not pair-specific tuning. Oscillator confirmations use standard
# market thresholds: Williams %R overbought/oversold (-20/-80), MFI (80/20),
# and StochRSI rail values (0.8/0.2 or 80/20 depending on source scale).
BB_RAIL_EDGE_FRACTION = 0.85
BB_RAIL_TIMEFRAMES = ("M1", "M5", "M15")
STOCH_RSI_HIGH = 0.8
STOCH_RSI_LOW = 0.2
STOCH_RSI_PERCENT_SCALE_HIGH = 80.0
STOCH_RSI_PERCENT_SCALE_LOW = 20.0
WILLIAMS_OVERBOUGHT = -20.0
WILLIAMS_OVERSOLD = -80.0
MFI_OVERBOUGHT = 80.0
MFI_OVERSOLD = 20.0
TEMPORARY_EXTREME_LOOKBACK_BARS = int(os.environ.get("QR_TEMPORARY_EXTREME_LOOKBACK_BARS", "12"))
TEMPORARY_EXTREME_PULLBACK_ATR_MULT = float(os.environ.get("QR_TEMPORARY_EXTREME_PULLBACK_ATR_MULT", "1.0"))
TEMPORARY_EXTREME_MIN_PROFIT_NOISE_MULT = float(os.environ.get("QR_TEMPORARY_EXTREME_MIN_PROFIT_NOISE_MULT", "1.0"))
# The temporary-extreme detector already requires a market-location context
# (distribution edge, rail touch, or local M1 swing extreme). Once that context
# exists, three independent reversal readings are the smallest majority that
# can catch the first confirmed rollover before the MFE has already been given
# back. Requiring four waited for a fuller micro trend flip and missed the
# 2026-06-12 USD_CHF local-top harvest while the trade was still +6 pips.
TEMPORARY_EXTREME_MIN_EVIDENCE = int(os.environ.get("QR_TEMPORARY_EXTREME_MIN_EVIDENCE", "3"))
TEMPORARY_EXTREME_DISTRIBUTION_PCT = float(os.environ.get("QR_TEMPORARY_EXTREME_DISTRIBUTION_PCT", "0.80"))
# Giveback guard: once recent M1 bid/ask-executable MFE has given back at least
# half of its move and the position is still currently profitable, bank it
# before it becomes the next red MARKET_ORDER_TRADE_CLOSE. The half-giveback
# default is intentionally earlier than broker-side profit-lock progress
# because it is paired with independent reversal evidence; waiting for a fuller
# giveback has shown up as late loss-close lag in execution_timing_audit.
MFE_GIVEBACK_TAKE_FRACTION = float(os.environ.get("QR_MFE_GIVEBACK_TAKE_FRACTION", "0.50"))
MFE_GIVEBACK_MIN_EVIDENCE = int(os.environ.get("QR_MFE_GIVEBACK_MIN_EVIDENCE", "2"))


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
    owner: str = Owner.TRADER.value
    close_review_action: str | None = None


@dataclass(frozen=True)
class PositionManagementDecision:
    generated_at_utc: str
    snapshot_fetched_at_utc: str | None
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
        data_root: Path | None = None,
    ) -> None:
        self.trader_decision_path = trader_decision_path
        self.pair_charts_path = pair_charts_path
        self.output_path = output_path
        self.report_path = report_path
        self.data_root = data_root or output_path.parent

    def run(self, snapshot: BrokerSnapshot) -> PositionManagementDecision:
        generated_at = datetime.now(timezone.utc).isoformat()
        scores = _load_scores(self.trader_decision_path)
        pair_charts = _load_pair_charts(self.pair_charts_path)
        full_pair_charts = _load_full_pair_charts(self.pair_charts_path)
        manageable_positions = tuple(
            position for position in snapshot.positions if _position_management_owner(position.owner)
        )
        managed = tuple(
            self._manage_position(position, snapshot, scores, pair_charts, full_pair_charts)
            for position in manageable_positions
        )
        # Global kill switch (2026-05-15): when `QR_DISABLE_AUTO_CLOSE=1`,
        # demote every REVIEW_EXIT to HOLD_PROTECTED before aggregation.
        # The 2026-05-14 24h cycle leaked -¥8,983 through deterministic
        # PositionManager REVIEW_EXIT firings on five separate branches
        # (macro=REVERSED OB break, profitable-matrix REVERSED, etc.).
        # The user's explicit directive 「資産減らしてるだけ」 means we
        # stop the auto-close pathway entirely; CLOSE decisions must
        # now flow through gpt_trader Gate A/B with operator token.
        if os.environ.get("QR_DISABLE_AUTO_CLOSE", "").strip() in {"1", "true", "TRUE", "yes", "YES"}:
            demoted: list[ManagedPosition] = []
            data_root = self.data_root
            for m in managed:
                if m.action == ACTION_REVIEW_EXIT:
                    if _structural_auto_close_enabled() and _next_generation_structural_auto_close_allowed(m, data_root):
                        new_reasons = tuple(list(m.reasons) + [
                            "next-generation entry thesis ledger present → structural loss-cut remains executable under QR_DISABLE_AUTO_CLOSE=1",
                        ])
                        demoted.append(ManagedPosition(
                            trade_id=m.trade_id, pair=m.pair, side=m.side,
                            units=m.units, action=m.action,
                            unrealized_pl_jpy=m.unrealized_pl_jpy,
                            remaining_risk_jpy=m.remaining_risk_jpy,
                            remaining_reward_jpy=m.remaining_reward_jpy,
                            same_direction_score=m.same_direction_score,
                            opposite_direction_score=m.opposite_direction_score,
                            reasons=new_reasons,
                            recommended_stop_loss=m.recommended_stop_loss,
                            recommended_take_profit=m.recommended_take_profit,
                            owner=m.owner,
                            close_review_action=m.close_review_action,
                        ))
                        continue
                    if _structural_auto_close_enabled():
                        demotion_reason = (
                            "QR_DISABLE_AUTO_CLOSE=1 → REVIEW_EXIT demoted to HOLD_PROTECTED; "
                            "QR_ALLOW_STRUCTURAL_AUTO_CLOSE=1 applies only to next-generation hard structural "
                            "loss-cuts (H1/H4 close-confirmed break or multi-TF structural OB break); "
                            "soft entry-thesis / forecast-collapse evidence must go through gpt_trader Gate A/B"
                        )
                    else:
                        demotion_reason = (
                            "QR_DISABLE_AUTO_CLOSE=1 → REVIEW_EXIT demoted to HOLD_PROTECTED; "
                            "loss-side close must go through gpt_trader Gate A/B unless "
                            "QR_ALLOW_STRUCTURAL_AUTO_CLOSE=1 is explicitly set for hard structural evidence"
                        )
                    new_reasons = tuple(list(m.reasons) + [demotion_reason])
                    demoted.append(ManagedPosition(
                        trade_id=m.trade_id, pair=m.pair, side=m.side,
                        units=m.units, action=ACTION_HOLD_PROTECTED,
                        unrealized_pl_jpy=m.unrealized_pl_jpy,
                        remaining_risk_jpy=m.remaining_risk_jpy,
                        remaining_reward_jpy=m.remaining_reward_jpy,
                        same_direction_score=m.same_direction_score,
                        opposite_direction_score=m.opposite_direction_score,
                        reasons=new_reasons,
                        recommended_stop_loss=m.recommended_stop_loss,
                        recommended_take_profit=m.recommended_take_profit,
                        owner=m.owner,
                        close_review_action=ACTION_REVIEW_EXIT,
                    ))
                else:
                    demoted.append(m)
            managed = tuple(demoted)
        action = _aggregate_action(managed)
        decision = PositionManagementDecision(
            generated_at_utc=generated_at,
            snapshot_fetched_at_utc=snapshot.fetched_at_utc.isoformat(),
            action=action,
            positions=managed,
        )
        self._write(decision)
        return decision

    def _manage_position(
        self,
        position: BrokerPosition,
        snapshot: BrokerSnapshot,
        scores: dict[tuple[str, str], float],
        pair_charts: dict[str, dict[str, Any]] | None,
        full_pair_charts: dict[str, dict[str, Any]] | None,
    ) -> ManagedPosition:
        if _manual_take_profit_owner(position.owner):
            return self._manage_manual_take_profit_position(position, snapshot, scores, pair_charts)

        same_score = scores.get((position.pair, position.side.value))
        opposite_score = scores.get((position.pair, _opposite(position.side)))
        remaining_risk = _remaining_risk_jpy(position, snapshot.quotes, snapshot.home_conversions)
        remaining_reward = _remaining_reward_jpy(position, snapshot.quotes, snapshot.home_conversions)
        reasons: list[str] = []
        quote = snapshot.quotes.get(position.pair)
        recommended_stop_loss: float | None = None
        recommended_take_profit: float | None = None
        close_review_action: str | None = None
        reasons.extend(_session_protection_notes(position, quote, pair_charts))
        latest_forecast = _latest_forecast_for_position(position.pair, self.data_root)
        if latest_forecast:
            reasons.append(
                f"latest forecast {str(latest_forecast.get('direction') or 'UNCLEAR').upper()} "
                f"conf={_to_float(latest_forecast.get('confidence')) or 0.0:.2f}"
            )

        sl_free_hold = (
            position.stop_loss is None
            and position.owner == Owner.TRADER
            and _trader_sl_repair_disabled()
        )
        sl_free_owned = (
            position.owner == Owner.TRADER and _trader_sl_repair_disabled()
        )

        entry_invalidation_review, entry_invalidation_reasons = _entry_thesis_invalidation_review(
            position=position,
            quote=quote,
            full_pair_charts=full_pair_charts,
            data_root=self.data_root,
            latest_forecast=latest_forecast,
        )
        thesis_break_review, thesis_break_reasons = _fresh_broken_thesis_close_review(
            position=position,
            snapshot=snapshot,
            data_root=self.data_root,
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
        if entry_invalidation_review:
            reasons.extend(entry_invalidation_reasons)
            action = ACTION_REVIEW_EXIT
        elif thesis_break_review:
            reasons.extend(thesis_break_reasons)
            action = ACTION_REVIEW_EXIT
        elif contradicted:
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
                        position, quote, pair_charts, full_pair_charts, latest_forecast
                    )
                    reasons.extend(adaptive_reasons)
                    if adaptive_action != ACTION_HOLD_PROTECTED:
                        if adaptive_action == ACTION_REVIEW_EXIT:
                            action = ACTION_REVIEW_EXIT
                        elif adaptive_action == ACTION_TAKE_PROFIT_MARKET:
                            action = ACTION_TAKE_PROFIT_MARKET
                        elif adaptive_tp is not None:
                            recommended_take_profit = adaptive_tp
                            action = adaptive_action
                    break_even_stop, break_even_reasons = _sl_free_profit_lock_stop_candidate(
                        position, quote, pair_charts
                    )
                    reasons.extend(break_even_reasons)
                    if break_even_stop is not None:
                        recommended_stop_loss = break_even_stop
                        if action == ACTION_HOLD_SL_FREE:
                            action = ACTION_BREAK_EVEN_STOP
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
            adaptive_action = ACTION_HOLD_PROTECTED
            adaptive_tp: float | None = None
            if position.unrealized_pl_jpy > 0 and position.take_profit is not None:
                adaptive_action, adaptive_tp, adaptive_reasons = _adaptive_tp_action(
                    position, quote, pair_charts, full_pair_charts, latest_forecast
                )
                reasons.extend(adaptive_reasons)

            if adaptive_action == ACTION_TAKE_PROFIT_MARKET:
                action = ACTION_TAKE_PROFIT_MARKET
            elif adaptive_action in {ACTION_HARVEST_TP, ACTION_NARROW_TP, ACTION_EXTEND_TP} and adaptive_tp is not None:
                recommended_take_profit = adaptive_tp
                action = adaptive_action
            elif adaptive_action == ACTION_REVIEW_EXIT and not sl_free_owned:
                action = ACTION_REVIEW_EXIT
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
                    rollover_review_reason = _rollover_spread_impaired_close_review_reason(
                        position,
                        quote,
                        pair_charts,
                        remaining_risk=remaining_risk,
                        remaining_reward=remaining_reward,
                    )
                    if rollover_review_reason is not None:
                        reasons.append(rollover_review_reason)
                        reasons.append(
                            "loss-side market close still requires GPT CLOSE Gate A/B; "
                            "keep broker TP/SL live while close review is pending"
                        )
                        close_review_action = ACTION_REVIEW_EXIT
                    else:
                        persistence_review_reason = (
                            _fresh_forecast_persistence_poor_rr_close_review_reason(
                                position=position,
                                snapshot=snapshot,
                                data_root=self.data_root,
                                remaining_risk=remaining_risk,
                                remaining_reward=remaining_reward,
                            )
                        )
                        if persistence_review_reason is not None:
                            reasons.append(persistence_review_reason)
                            reasons.append(
                                "loss-side market close still requires GPT CLOSE Gate A/B; "
                                "keep broker TP/SL live while close review is pending"
                            )
                            close_review_action = ACTION_REVIEW_EXIT
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
            owner=position.owner.value,
            close_review_action=close_review_action,
        )

    def _manage_manual_take_profit_position(
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
        quote = snapshot.quotes.get(position.pair)
        recommended_take_profit: float | None = None
        reasons: list[str] = []
        reasons.extend(_session_protection_notes(position, quote, pair_charts))
        reasons.append(
            "manual/tagless position: TP-only profit management enabled; SL and loss-close management disabled"
        )

        if position.take_profit is None:
            if not _missing_tp_repair_enabled():
                action = ACTION_HOLD_SL_FREE
                reasons.append(
                    "manual/tagless broker TP absent; missing-TP repair disabled "
                    "(QR_ENABLE_MISSING_TP_REPAIR!=1), preserving no-broker-TP runner"
                )
            else:
                recommended_take_profit, tp_reason = _market_take_profit_repair_candidate(
                    position, quote, pair_charts
                )
                if recommended_take_profit is None:
                    action = ACTION_HOLD_SL_FREE
                    reasons.append(f"manual/tagless TP repair skipped: {tp_reason}")
                else:
                    action = ACTION_REPAIR_TAKE_PROFIT
                    reasons.append(f"manual/tagless TP repair candidate {recommended_take_profit:.5f}: {tp_reason}")
        else:
            action = ACTION_HOLD_SL_FREE
            reasons.append("manual/tagless take-profit already present; stop-loss untouched")

        if remaining_risk is not None:
            reasons.append(f"remaining risk about {remaining_risk:.0f} JPY (observed only; no SL action)")
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
            recommended_stop_loss=None,
            recommended_take_profit=round(recommended_take_profit, _price_precision(position.pair))
            if recommended_take_profit is not None
            else None,
            reasons=tuple(reasons),
            owner=position.owner.value,
        )

    def _write(self, decision: PositionManagementDecision) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(asdict(decision), ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Position Management Report",
            "",
            f"- Generated at UTC: `{decision.generated_at_utc}`",
            f"- Broker snapshot fetched at UTC: `{decision.snapshot_fetched_at_utc or 'unknown'}`",
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
                f"- `{item.trade_id}` `{item.pair} {item.side}` owner=`{item.owner}` units=`{item.units}` "
                f"action=`{item.action}` upl=`{item.unrealized_pl_jpy:.1f}`"
            )
            if item.close_review_action:
                lines.append(f"  - close review: `{item.close_review_action}`")
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
                "- Operator-managed manual/tagless positions are eligible for TP-only profit management.",
                "- Manual/tagless positions must never receive SL repair, SL tightening, or loss-close actions.",
                "- Missing TP/SL is a repair requirement, not a passive monitor state.",
                "- Profit protection is required once open profit clears remaining stop risk plus current session noise.",
                "- SL-free break-even/profit-lock is allowed only after executable profit clears M5 ATR/spread micro-noise.",
                "- Profit-only TAKE_PROFIT_MARKET is separate from loss-side REVIEW_EXIT Gate A/B discipline.",
                "- A materially stronger opposite thesis triggers exit review; the gateway still prevents fresh stacking.",
                "- With QR_DISABLE_AUTO_CLOSE=1, deterministic REVIEW_EXIT stays advisory unless QR_ALLOW_STRUCTURAL_AUTO_CLOSE=1 explicitly opts into structural auto-close; GPT CLOSE Gate A+B remains executable.",
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
    full_pair_charts: dict[str, dict[str, Any]] | None = None,
    latest_forecast: dict[str, Any] | None = None,
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

    pc = (pair_charts or {}).get(position.pair) if pair_charts else None
    chart_story = str(pc.get("chart_story") or "") if isinstance(pc, dict) else ""
    if not chart_story:
        return ACTION_HOLD_PROTECTED, None, reasons

    lane_dir = position.side.value
    target_up = lane_dir == "LONG"
    micro = _classify_micro(chart_story, lane_dir)
    macro = _classify_macro(chart_story, lane_dir)
    reasons.append(f"micro={micro} macro={macro}")

    # Price-action lens (multi-TF SMC: swings, BOS/CHOCH events, dealing
    # range, order blocks, liquidity touches). Returns a small ±delta that
    # complements the macro/micro classification; primary action selection
    # below still uses the matrix, but PA delta surfaces real market
    # context in the rationale and gates ambiguous EXTEND decisions.
    pip_factor = 100.0 if position.pair.endswith("_JPY") else 10000.0
    cur_price = quote.bid if not target_up else quote.ask
    full_charts = full_pair_charts if full_pair_charts is not None else _load_full_pair_charts()
    pair_chart = full_charts.get(position.pair)
    chart_context = _adaptive_tp_chart_context(pair_chart, pc)
    pa_delta_lane, pa_reasons = aggregate_price_action_score(
        pair_chart, lane_dir, cur_price, pip_factor
    )
    if pa_reasons:
        reasons.append(f"price-action ({pa_delta_lane:+.1f}): " + " ; ".join(pa_reasons[:2]))
    bb_delta_lane, bb_reasons = _bb_rail_pressure(position.pair, lane_dir, pair_charts)
    if bb_reasons:
        reasons.append(f"BB rail ({bb_delta_lane:+.1f}): " + " ; ".join(bb_reasons[:2]))

    # Losing positions: EXIT only on a market-derived structural break,
    # never on a hardcoded NAV-percent threshold (AGENT_CONTRACT §3.5
    # forbids thoughtless hardcodes / fallbacks; user 2026-05-08「決済も
    # 市況によって動的。ハードコードとフォールバックはなし」).
    #
    # Two structural triggers:
    #   1. macro=REVERSED — H1/H4 confirmed turn against the lane direction
    #      (computed from the MTF chart_story by `_classify_macro`).
    #   2. price has broken past the lane-relevant order block: for LONG,
    #      below the nearest unmitigated BULL OB low (structural support
    #      gone); for SHORT, above the nearest unmitigated BEAR OB high.
    #
    # If neither structural trigger fires, HOLD. Pullback noise inside
    # the trend backbone is the exact regime SL-free was designed for —
    # cutting at −5 pips manufactures the "open → red → close" churn the
    # user flagged. The exit level comes from the market itself (OB
    # levels priced by SMC), not from a literal JPY floor.
    if position.unrealized_pl_jpy <= 0:
        if macro == "REVERSED":
            # 2026-05-14 fix: require CLOSE-CONFIRMED structural event
            # before auto-exit. The `_classify_macro` label flips on
            # noisy intraday regime swings (M15/M30/H1 regime tags
            # change on wick prints), causing the trader to exit
            # positions within 10 minutes of entry on a temporary
            # macro re-classification. Recent live incidents
            # (471061 EUR/JPY closed -154 in 1h, 471077 EUR/JPY closed
            # -217 in 9min) both fired here with no real structural
            # break to back them up. Mirrors the Gate A close_confirmed
            # filter on gpt_trader (feedback_structure_close_vs_wick.md).
            expected_struct_dir = "DOWN" if target_up else "UP"
            close_confirmed_break = False
            if isinstance(pair_chart, dict):
                for view in pair_chart.get("views") or []:
                    tf = str(view.get("granularity") or "").upper()
                    if tf not in ("H1", "H4"):
                        continue
                    events = ((view.get("structure") or {}).get("structure_events") or [])
                    for ev in events[-4:]:
                        if not isinstance(ev, dict):
                            continue
                        if not ev.get("close_confirmed"):
                            continue
                        kind = str(ev.get("kind") or "")
                        if expected_struct_dir in kind:
                            close_confirmed_break = True
                            break
                    if close_confirmed_break:
                        break
            if not close_confirmed_break:
                reasons.append(
                    f"macro=REVERSED on label but no H1/H4 close-confirmed "
                    f"BOS/CHOCH against {lane_dir} → HOLD pending structural confirm "
                    f"(2026-05-14 anti-churn fix)"
                )
                # Fall through to OB-break check below; if that also
                # doesn't confirm, the function reaches the default
                # HOLD at the bottom.
            else:
                reasons.append(
                    f"loss-cut: macro REVERSED + H1/H4 close-confirmed "
                    f"structural break against {lane_dir} "
                    f"({position.unrealized_pl_jpy:+.0f} JPY)"
                )
                return ACTION_REVIEW_EXIT, None, reasons

        # Multi-TF structural OB break check (user 2026-05-11「H1とH4でしか
        # みてない？」: previous version only consulted M30, missing real
        # breaks visible on M15/H1/H4 and overcounting M30-only false breaks).
        # Iterate M15+M30+H1+H4 — a break on ≥2 TFs counts as confirmed
        # structural invalidation; a single-TF break (often noise or
        # measurement artifact) HOLDs.
        ob_break_tfs: list[tuple[str, float]] = []  # [(tf, broken_level)]
        if isinstance(pair_chart, dict):
            for view in pair_chart.get("views") or []:
                gran = str(view.get("granularity") or "")
                if gran not in {"M15", "M30", "H1", "H4"}:
                    continue
                obs = classify_order_block_proximity(view, cur_price, pip_factor)
                if target_up and obs.nearest_bull_low is not None:
                    if cur_price < obs.nearest_bull_low:
                        ob_break_tfs.append((gran, obs.nearest_bull_low))
                elif not target_up and obs.nearest_bear_high is not None:
                    if cur_price > obs.nearest_bear_high:
                        ob_break_tfs.append((gran, obs.nearest_bear_high))

        if len(ob_break_tfs) >= 2:
            tfs_summary = ", ".join(f"{tf}@{lvl:.5f}" for tf, lvl in ob_break_tfs)
            reasons.append(
                f"loss-cut: structural OB broken across {len(ob_break_tfs)} TFs "
                f"({tfs_summary}) ({position.unrealized_pl_jpy:+.0f} JPY)"
            )
            return ACTION_REVIEW_EXIT, None, reasons
        if ob_break_tfs:
            # Single-TF break — note in rationale but HOLD; needs 2nd TF
            # confirmation before acting.
            tf, lvl = ob_break_tfs[0]
            reasons.append(
                f"single-TF OB break ({tf}@{lvl:.5f}) — HOLD pending confirmation"
            )

        # No structural break — HOLD and let the pullback resolve. The
        # SL-free philosophy: the operator placed this position, no
        # auto-rule should dump it on intra-trend noise.
        reasons.append(
            f"hold underwater ({position.unrealized_pl_jpy:+.0f} JPY): "
            f"macro={macro} micro={micro} — no structural break detected"
        )
        return ACTION_HOLD_PROTECTED, None, reasons

    temporary_profit_take, temporary_profit_reasons = _temporary_extreme_profit_take_signal(
        position=position,
        quote=quote,
        full_pair_charts=full_pair_charts,
        latest_forecast=latest_forecast,
    )
    reasons.extend(temporary_profit_reasons)
    if temporary_profit_take:
        profit_take_blocker = _profit_market_take_noise_blocker(
            position=position,
            quote=quote,
            pair_chart=pair_chart,
        )
        if profit_take_blocker is not None:
            reasons.append(profit_take_blocker)
            return ACTION_HOLD_PROTECTED, None, reasons
        return ACTION_TAKE_PROFIT_MARKET, None, reasons

    mfe_giveback_profit_take, mfe_giveback_reasons = _mfe_giveback_profit_take_signal(
        position=position,
        quote=quote,
        full_pair_charts=full_pair_charts,
        latest_forecast=latest_forecast,
    )
    reasons.extend(mfe_giveback_reasons)
    if mfe_giveback_profit_take:
        profit_take_blocker = _profit_market_take_noise_blocker(
            position=position,
            quote=quote,
            pair_chart=pair_chart,
        )
        if profit_take_blocker is not None:
            reasons.append(profit_take_blocker)
            return ACTION_HOLD_PROTECTED, None, reasons
        return ACTION_TAKE_PROFIT_MARKET, None, reasons

    # Decision matrix (rows = macro, cols = micro). Profitable positions only.
    # WEAKENING+DEAD demoted from REVIEW_EXIT to HARVEST_TP: MARKET close pays
    # full spread, but a narrow TP fills at the broker's TP price (no spread).
    # Same outcome (lock profit) cheaper. REVERSED uses a profit-only market
    # take action because we want OUT immediately, not to wait for a TP fill.
    matrix = {
        ("ALIGNED", "ALIVE"): ACTION_EXTEND_TP,
        ("ALIGNED", "DYING"): ACTION_HARVEST_TP,
        ("ALIGNED", "DEAD"): ACTION_HARVEST_TP,
        ("WEAKENING", "ALIVE"): ACTION_HARVEST_TP,
        ("WEAKENING", "DYING"): ACTION_NARROW_TP,
        ("WEAKENING", "DEAD"): ACTION_HARVEST_TP,
        ("REVERSED", "ALIVE"): ACTION_TAKE_PROFIT_MARKET,
        ("REVERSED", "DYING"): ACTION_TAKE_PROFIT_MARKET,
        ("REVERSED", "DEAD"): ACTION_TAKE_PROFIT_MARKET,
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

    # Gate EXTEND on positive price-action: don't push TP further when the
    # lens reads weakening structure. Demote to HARVEST so we lock profit
    # instead of chasing a TP the structure says won't print
    # (user 2026-05-08「限界なら見極める」).
    if action == ACTION_EXTEND_TP and pa_delta_lane < -3.0:
        action = ACTION_HARVEST_TP
        reasons.append(f"PA delta {pa_delta_lane:+.1f} demotes EXTEND→HARVEST")

    if action == ACTION_EXTEND_TP:
        # Market-derived EXTEND target — push TP to the next-but-one
        # structural level beyond the current one (skip the anchor that
        # sits closest to current price, target the one after). No
        # hardcoded "+X pips" extension.
        candidate, anchor_reason = structural_tp_target(
            pair_chart, side=lane_dir, current_price=cur,
            pip_factor=pip_factor, intent="EXTEND",
        )
        if candidate is None:
            action = ACTION_HOLD_PROTECTED
            reasons.append(f"extend skipped: {anchor_reason}")
        elif (target_up and candidate > position.take_profit) or (
            not target_up and candidate < position.take_profit
        ):
            new_tp = candidate
            reasons.append(f"extend TP→{candidate:.5f} ({anchor_reason})")
        else:
            # New anchor isn't farther than existing TP — TP already past
            # the next structural level, just hold and let it fill.
            action = ACTION_HOLD_PROTECTED
            reasons.append(f"extend skipped: anchor {anchor_reason} not farther than current TP")
    elif action == ACTION_HARVEST_TP:
        if bb_delta_lane > 0 and macro == "ALIGNED":
            tp_on_rail, tp_rail_reason = _existing_tp_at_opposite_bb_rail(position, pair_charts)
            if tp_on_rail:
                action = ACTION_HOLD_PROTECTED
                reasons.append(
                    f"BB rail supports {lane_dir}; keep existing TP ({tp_rail_reason}) and use BE/profit-lock sidecar"
                )
                return action, new_tp, reasons
        # Market-derived TP target (no hardcoded buffer, no fallback).
        # AGENT_CONTRACT §3.5 + user 2026-05-08「TPの設定は市況をみてる？
        # テクニカル等」「ハードコードとフォールバックはなし」.
        # `structural_tp_target` returns the nearest opposing structural
        # level (cross-TF liquidity / OB edge / Fib node). Skip if no
        # anchor was found — better to HOLD than place a TP at a guess.
        candidate, anchor_reason = structural_tp_target(
            pair_chart, side=lane_dir, current_price=cur,
            pip_factor=pip_factor, intent="HARVEST",
        )
        if candidate is None:
            action = ACTION_HOLD_PROTECTED
            reasons.append(f"harvest skipped: {anchor_reason}")
        elif (target_up and candidate < position.take_profit) or (
            not target_up and candidate > position.take_profit
        ):
            allowed, gate_reason = _adaptive_tp_contraction_allowed(
                position,
                quote,
                pair_charts,
                ACTION_HARVEST_TP,
                latest_forecast=latest_forecast,
                chart_context=chart_context,
            )
            reasons.append(gate_reason)
            if allowed:
                new_tp = candidate
                reasons.append(f"harvest TP→{candidate:.5f} ({anchor_reason})")
            else:
                action = ACTION_HOLD_PROTECTED
        else:
            action = ACTION_HOLD_PROTECTED
            reasons.append(f"harvest skipped: structural anchor {anchor_reason} not closer than current TP")
    elif action == ACTION_NARROW_TP:
        # Same market-derived path with NARROW intent (midpoint between
        # current price and the nearest structural anchor).
        candidate, anchor_reason = structural_tp_target(
            pair_chart, side=lane_dir, current_price=cur,
            pip_factor=pip_factor, intent="NARROW",
        )
        if candidate is None:
            action = ACTION_HOLD_PROTECTED
            reasons.append(f"narrow skipped: {anchor_reason}")
        elif (target_up and candidate < position.take_profit) or (
            not target_up and candidate > position.take_profit
        ):
            allowed, gate_reason = _adaptive_tp_contraction_allowed(
                position,
                quote,
                pair_charts,
                ACTION_NARROW_TP,
                latest_forecast=latest_forecast,
                chart_context=chart_context,
            )
            reasons.append(gate_reason)
            if allowed:
                new_tp = candidate
                reasons.append(f"narrow TP→{candidate:.5f} ({anchor_reason})")
            else:
                action = ACTION_HOLD_PROTECTED
        else:
            action = ACTION_HOLD_PROTECTED
            reasons.append(f"narrow skipped: anchor {anchor_reason} not closer")
    elif action == ACTION_TAKE_PROFIT_MARKET:
        noise_blocker = _profit_market_take_noise_blocker(
            position=position,
            quote=quote,
            pair_chart=pair_chart,
            protect_reachable_tp=True,
        )
        if noise_blocker is not None:
            action = ACTION_HOLD_PROTECTED
            reasons.append(noise_blocker)
            return action, new_tp, reasons
        # MARKET close handled by position_execution.py; this is deliberately
        # separate from loss-side REVIEW_EXIT / Gate A/B close discipline.
        reasons.append(
            f"profit-harvest market close: macro={macro} micro={micro} while position is profitable"
        )
    else:
        reasons.append("HOLD (mixed signal)")

    return action, new_tp, reasons


def _temporary_extreme_profit_take_signal(
    *,
    position: BrokerPosition,
    quote,
    full_pair_charts: dict[str, dict[str, Any]] | None,
    latest_forecast: dict[str, Any] | None = None,
) -> tuple[bool, list[str]]:
    """Detect a local top/bottom where banking profit beats waiting for TP.

    This is intentionally profit-side only. It does not authorize loss-side
    closes, and it does not create a re-entry in the same pass. The re-entry
    discipline is: close the local MFE when microstructure rolls over, refresh
    broker truth, then let the next fresh-entry cycle decide whether the
    pullback has rebuilt a LIVE_READY lane.
    """

    reasons: list[str] = []
    if position.unrealized_pl_jpy <= 0:
        return False, reasons
    if quote is None:
        return False, ["temporary extreme profit-take skipped: quote missing"]
    pair_chart = (full_pair_charts or {}).get(position.pair)
    if not isinstance(pair_chart, dict):
        return False, ["temporary extreme profit-take skipped: pair chart missing"]

    m1 = _view_by_timeframe(pair_chart, "M1")
    m5 = _view_by_timeframe(pair_chart, "M5")
    if not isinstance(m1, dict):
        return False, ["temporary extreme profit-take skipped: M1 chart missing"]

    side = position.side.value
    side_up = side.upper()
    if side_up not in {"LONG", "SHORT"}:
        return False, reasons
    long_side = side_up == "LONG"
    pip_factor = _pip_factor(position.pair)
    exit_price = quote.bid if long_side else quote.ask
    profit_pips = _executable_profit_pips(position, quote)
    spread_pips = _spread_pips(position.pair, quote)
    m1_atr = _indicator_float(m1, "atr_pips")
    if profit_pips is None or profit_pips <= 0:
        return False, ["temporary extreme profit-take skipped: executable profit is not positive"]
    if spread_pips is None or spread_pips <= 0:
        return False, ["temporary extreme profit-take skipped: spread missing"]
    if m1_atr is None or m1_atr <= 0:
        return False, ["temporary extreme profit-take skipped: M1 ATR missing"]

    min_profit_pips = max(spread_pips * TEMPORARY_EXTREME_MIN_PROFIT_NOISE_MULT, m1_atr)
    if profit_pips < min_profit_pips:
        return False, [
            f"temporary extreme profit-take skipped: profit {profit_pips:.1f}pip < "
            f"market noise floor {min_profit_pips:.1f}pip"
        ]

    candles = _closed_recent_candles(m1)
    lookback = max(4, TEMPORARY_EXTREME_LOOKBACK_BARS)
    evidence_recent = candles[-lookback:]
    if len(evidence_recent) < 4:
        return False, ["temporary extreme profit-take skipped: insufficient M1 closed candles"]

    # Use the full bounded M1 sample from chart_reader for the local extreme
    # context, while keeping reversal evidence on the shorter live-execution
    # window. A 20-minute scheduler can otherwise drop the actual top/bottom
    # just outside the last 12 M1 candles and then miss the rollover that is
    # happening right now.
    context_recent = candles
    highs = [_candle_float(candle, "h", "high") for candle in context_recent]
    lows = [_candle_float(candle, "l", "low") for candle in context_recent]
    if any(value is None for value in highs + lows):
        return False, ["temporary extreme profit-take skipped: malformed M1 candle prices"]

    typed_highs = [float(value) for value in highs if value is not None]
    typed_lows = [float(value) for value in lows if value is not None]
    if long_side:
        extreme = max(typed_highs)
        pullback_pips = (extreme - exit_price) * pip_factor
        pullback_label = "top pullback"
    else:
        extreme = min(typed_lows)
        pullback_pips = (exit_price - extreme) * pip_factor
        pullback_label = "bottom bounce"
    context_reasons = _temporary_extreme_context_reasons(
        side_up=side_up,
        pair_chart=pair_chart,
        m1=m1,
        m5=m5,
        recent=context_recent,
        extreme=extreme,
        pip_factor=pip_factor,
        atr_pips=m1_atr,
    )
    if not context_reasons:
        return False, ["temporary extreme profit-take skipped: no upper/lower rail or distribution extreme"]

    evidence = _temporary_extreme_reversal_evidence(
        side_up=side_up,
        m1=m1,
        m5=m5,
        recent=evidence_recent,
        extreme=extreme,
        latest_forecast=latest_forecast,
    )
    if len(evidence) < TEMPORARY_EXTREME_MIN_EVIDENCE:
        return False, [
            "temporary extreme profit-take skipped: reversal evidence "
            f"{len(evidence)}/{TEMPORARY_EXTREME_MIN_EVIDENCE}: " + "; ".join(evidence[:4])
        ]

    pullback_floor = max(spread_pips, m1_atr * TEMPORARY_EXTREME_PULLBACK_ATR_MULT)
    strong_early_rollover = _strong_temporary_extreme_rollover(side_up, evidence)
    if pullback_pips < pullback_floor and not strong_early_rollover:
        return False, [
            f"temporary extreme profit-take skipped: {pullback_label} {pullback_pips:.1f}pip < "
            f"floor {pullback_floor:.1f}pip"
        ]

    label = "temporary top" if long_side else "temporary bottom"
    floor_note = (
        f"{pullback_label} {pullback_pips:.1f}pip < floor {pullback_floor:.1f}pip "
        "but close-confirmed M1 rollover stack is already complete"
        if pullback_pips < pullback_floor
        else f"{pullback_label} {pullback_pips:.1f}pip >= floor {pullback_floor:.1f}pip"
    )
    reasons.append(
        f"{label} profit-take: profit {profit_pips:.1f}pip, "
        f"{floor_note}; "
        + "; ".join((context_reasons + evidence)[:7])
    )
    reasons.append(
        "post-close re-entry discipline: refresh broker truth and require a fresh LIVE_READY pullback/retest lane before re-entering"
    )
    return True, reasons


def _mfe_giveback_profit_take_signal(
    *,
    position: BrokerPosition,
    quote,
    full_pair_charts: dict[str, dict[str, Any]] | None,
    latest_forecast: dict[str, Any] | None = None,
) -> tuple[bool, list[str]]:
    """Bank profit when recent executable MFE is being given back.

    This guard handles the audit pattern where a trade was briefly positive,
    the local top/bottom detector lacked a rail/distribution context, and the
    later close was red. It is profit-side only: current executable P/L must
    still be positive, and it never authorizes loss-side CLOSE.
    """

    if position.unrealized_pl_jpy <= 0:
        return False, []
    if quote is None:
        return False, ["MFE giveback profit-take skipped: quote missing"]
    pair_chart = (full_pair_charts or {}).get(position.pair)
    if not isinstance(pair_chart, dict):
        return False, ["MFE giveback profit-take skipped: pair chart missing"]
    m1 = _view_by_timeframe(pair_chart, "M1")
    m5 = _view_by_timeframe(pair_chart, "M5")
    if not isinstance(m1, dict):
        return False, ["MFE giveback profit-take skipped: M1 chart missing"]

    side_up = position.side.value.upper()
    if side_up not in {"LONG", "SHORT"}:
        return False, []
    profit_pips = _executable_profit_pips(position, quote)
    spread_pips = _spread_pips(position.pair, quote)
    m1_atr = _indicator_float(m1, "atr_pips")
    if profit_pips is None or profit_pips <= 0:
        return False, ["MFE giveback profit-take skipped: executable profit is not positive"]
    if spread_pips is None or spread_pips <= 0:
        return False, ["MFE giveback profit-take skipped: spread missing"]
    if m1_atr is None or m1_atr <= 0:
        return False, ["MFE giveback profit-take skipped: M1 ATR missing"]

    candles = _closed_recent_candles(m1)
    lookback = max(4, TEMPORARY_EXTREME_LOOKBACK_BARS)
    recent = candles[-lookback:]
    if len(recent) < 4:
        return False, ["MFE giveback profit-take skipped: insufficient M1 closed candles"]

    pip_factor = _pip_factor(position.pair)
    highs = [_candle_float(candle, "h", "high") for candle in recent]
    lows = [_candle_float(candle, "l", "low") for candle in recent]
    if any(value is None for value in highs + lows):
        return False, ["MFE giveback profit-take skipped: malformed M1 candle prices"]
    typed_highs = [float(value) for value in highs if value is not None]
    typed_lows = [float(value) for value in lows if value is not None]
    if side_up == "LONG":
        recent_mfe_pips = max(0.0, (max(typed_highs) - position.entry_price) * pip_factor)
        extreme = max(typed_highs)
    else:
        recent_mfe_pips = max(0.0, (position.entry_price - min(typed_lows)) * pip_factor)
        extreme = min(typed_lows)

    mfe_floor = max(spread_pips, m1_atr)
    if recent_mfe_pips < mfe_floor:
        return False, [
            f"MFE giveback profit-take skipped: recent MFE {recent_mfe_pips:.1f}pip < "
            f"market noise floor {mfe_floor:.1f}pip"
        ]
    giveback_pips = max(0.0, recent_mfe_pips - profit_pips)
    fraction = max(0.0, min(1.0, MFE_GIVEBACK_TAKE_FRACTION))
    if giveback_pips < recent_mfe_pips * fraction:
        return False, [
            f"MFE giveback profit-take skipped: giveback {giveback_pips:.1f}pip < "
            f"{fraction:.2f}× recent MFE {recent_mfe_pips:.1f}pip"
        ]

    evidence = _temporary_extreme_reversal_evidence(
        side_up=side_up,
        m1=m1,
        m5=m5,
        recent=recent,
        extreme=extreme,
        latest_forecast=latest_forecast,
    )
    required = max(1, MFE_GIVEBACK_MIN_EVIDENCE)
    if len(evidence) < required:
        return False, [
            f"MFE giveback profit-take skipped: reversal evidence {len(evidence)}/{required}: "
            + "; ".join(evidence[:4])
        ]

    return True, [
        (
            f"MFE giveback profit-take: recent MFE {recent_mfe_pips:.1f}pip, "
            f"current profit {profit_pips:.1f}pip, giveback {giveback_pips:.1f}pip "
            f">= {fraction:.2f}× MFE; " + "; ".join(evidence[:5])
        ),
        "post-close re-entry discipline: refresh broker truth and require a fresh LIVE_READY pullback/retest lane before re-entering",
    ]


def _profit_market_take_noise_blocker(
    *,
    position: BrokerPosition,
    quote,
    pair_chart: dict[str, Any] | None,
    protect_reachable_tp: bool = False,
) -> str | None:
    """Require profit market takes to clear executable noise and TP progress.

    Paying the full spread to bank a tiny positive tick worsens the average
    win/average-loss asymmetry that capture-economics audits. A broker TP that
    is still mostly ahead should remain the harvest mechanism unless the
    executable move has captured most of the planned reward.
    """

    if quote is None:
        return "profit-harvest market close skipped: quote missing for executable noise floor"
    if not isinstance(pair_chart, dict):
        return "profit-harvest market close skipped: pair chart missing for executable noise floor"
    m1 = _view_by_timeframe(pair_chart, "M1")
    if not isinstance(m1, dict):
        return "profit-harvest market close skipped: M1 chart missing for executable noise floor"
    profit_pips = _executable_profit_pips(position, quote)
    spread_pips = _spread_pips(position.pair, quote)
    m1_atr = _indicator_float(m1, "atr_pips")
    if profit_pips is None or profit_pips <= 0:
        return "profit-harvest market close skipped: executable profit is not positive"
    if spread_pips is None or spread_pips <= 0:
        return "profit-harvest market close skipped: spread missing for executable noise floor"
    if m1_atr is None or m1_atr <= 0:
        return "profit-harvest market close skipped: M1 ATR missing for executable noise floor"
    min_profit_pips = max(spread_pips * TEMPORARY_EXTREME_MIN_PROFIT_NOISE_MULT, m1_atr)
    if profit_pips < min_profit_pips:
        return (
            f"profit-harvest market close skipped: executable profit {profit_pips:.1f}pip < "
            f"market noise floor {min_profit_pips:.1f}pip"
        )
    progress_blocker = _profit_market_take_tp_progress_blocker(position, profit_pips)
    if progress_blocker is not None:
        return progress_blocker
    if protect_reachable_tp:
        reachable_tp_blocker = _profit_market_take_reachable_tp_blocker(
            position=position,
            quote=quote,
            pair_chart=pair_chart,
            spread_pips=spread_pips,
        )
        if reachable_tp_blocker is not None:
            return reachable_tp_blocker
    return None


def _profit_market_take_tp_progress_blocker(
    position: BrokerPosition,
    profit_pips: float,
) -> str | None:
    tp_pips = _position_tp_pips(position)
    if tp_pips is None or tp_pips <= 0:
        return None
    progress_gate = tp_pips * PROFIT_BREAK_EVEN_MIN_TP_PROGRESS
    if profit_pips >= progress_gate:
        return None
    progress = max(0.0, profit_pips / tp_pips)
    return (
        f"profit-harvest market close skipped: TP progress {progress:.0%} "
        f"({profit_pips:.1f}/{tp_pips:.1f}pip) < {PROFIT_BREAK_EVEN_MIN_TP_PROGRESS:.0%}; "
        "keep broker TP instead of micro-scalping attached-TP harvest"
    )


def _profit_market_take_reachable_tp_blocker(
    *,
    position: BrokerPosition,
    quote,
    pair_chart: dict[str, Any],
    spread_pips: float,
) -> str | None:
    remaining_pips = _tp_remaining_pips(position, quote)
    if remaining_pips is None or remaining_pips <= 0:
        return None
    m5 = _view_by_timeframe(pair_chart, PROFIT_BREAK_EVEN_ATR_TIMEFRAME)
    if not isinstance(m5, dict):
        return None
    m5_atr = _indicator_float(m5, "atr_pips")
    if m5_atr is None or m5_atr <= 0:
        return None
    reachable_window = m5_atr + spread_pips
    if remaining_pips > reachable_window:
        return None
    return (
        f"profit-harvest market close skipped: broker TP is reachable within "
        f"{PROFIT_BREAK_EVEN_ATR_TIMEFRAME} ATR + spread "
        f"({remaining_pips:.1f}pip <= {m5_atr:.1f}+{spread_pips:.1f}pip); "
        "keep broker TP instead of replacing a near fill with a spread-paid market close"
    )


def _view_by_timeframe(pair_chart: dict[str, Any], timeframe: str) -> dict[str, Any] | None:
    for view in pair_chart.get("views") or []:
        if not isinstance(view, dict):
            continue
        if str(view.get("granularity") or view.get("timeframe") or "").upper() == timeframe:
            return view
    return None


def _indicator_float(view: dict[str, Any] | None, key: str) -> float | None:
    if not isinstance(view, dict):
        return None
    indicators = view.get("indicators") if isinstance(view.get("indicators"), dict) else {}
    value = indicators.get(key)
    if value is None:
        value = view.get(key)
    return _to_float(value)


def _closed_recent_candles(view: dict[str, Any]) -> list[dict[str, Any]]:
    raw = view.get("recent_candles")
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for candle in raw:
        if not isinstance(candle, dict):
            continue
        if candle.get("complete") is False:
            continue
        out.append(candle)
    return out


def _candle_float(candle: dict[str, Any], short_key: str, long_key: str) -> float | None:
    return _to_float(candle.get(short_key) if candle.get(short_key) is not None else candle.get(long_key))


def _temporary_extreme_context_reasons(
    *,
    side_up: str,
    pair_chart: dict[str, Any],
    m1: dict[str, Any],
    m5: dict[str, Any] | None,
    recent: list[dict[str, Any]],
    extreme: float,
    pip_factor: int,
    atr_pips: float,
) -> list[str]:
    reasons: list[str] = []
    confluence = pair_chart.get("confluence") if isinstance(pair_chart.get("confluence"), dict) else {}
    pct_24h = _to_float(confluence.get("price_percentile_24h"))
    pct_7d = _to_float(confluence.get("price_percentile_7d"))
    if side_up == "LONG":
        if pct_24h is not None and pct_24h >= TEMPORARY_EXTREME_DISTRIBUTION_PCT:
            reasons.append(f"24h upper distribution pct={pct_24h:.2f}")
        if pct_7d is not None and pct_7d >= TEMPORARY_EXTREME_DISTRIBUTION_PCT:
            reasons.append(f"7d upper distribution pct={pct_7d:.2f}")
        rail_note = _upper_rail_touch_reason(m1, m5, extreme, pip_factor, atr_pips)
    else:
        if pct_24h is not None and pct_24h <= 1.0 - TEMPORARY_EXTREME_DISTRIBUTION_PCT:
            reasons.append(f"24h lower distribution pct={pct_24h:.2f}")
        if pct_7d is not None and pct_7d <= 1.0 - TEMPORARY_EXTREME_DISTRIBUTION_PCT:
            reasons.append(f"7d lower distribution pct={pct_7d:.2f}")
        rail_note = _lower_rail_touch_reason(m1, m5, extreme, pip_factor, atr_pips)
    if rail_note:
        reasons.append(rail_note)
    local_swing = _local_swing_extreme_reason(
        side_up=side_up,
        recent=recent,
        extreme=extreme,
        pip_factor=pip_factor,
        atr_pips=atr_pips,
    )
    if local_swing:
        reasons.append(local_swing)
    return reasons


def _local_swing_extreme_reason(
    *,
    side_up: str,
    recent: list[dict[str, Any]],
    extreme: float,
    pip_factor: int,
    atr_pips: float,
) -> str | None:
    """Identify a local M1 swing extreme without requiring a Bollinger rail.

    The required swing size is one current M1 ATR: this is the market's own
    one-minute noise unit, so the detector can catch a temporary top/bottom
    like the operator's USD/CAD screenshot even when the broader 24h/7d
    distribution is not at an edge.
    """
    if atr_pips <= 0 or len(recent) < 4:
        return None
    highs = [_candle_float(candle, "h", "high") for candle in recent]
    lows = [_candle_float(candle, "l", "low") for candle in recent]
    if any(value is None for value in highs + lows):
        return None
    typed_highs = [float(value) for value in highs if value is not None]
    typed_lows = [float(value) for value in lows if value is not None]
    if side_up == "LONG":
        extreme_index = max(range(len(typed_highs)), key=lambda idx: typed_highs[idx])
        if extreme_index >= len(typed_highs) - 1:
            return None
        runup_pips = (extreme - min(typed_lows[: extreme_index + 1])) * pip_factor
        if runup_pips >= atr_pips:
            return f"M1 local swing top run-up {runup_pips:.1f}pip >= M1 ATR {atr_pips:.1f}pip"
    else:
        extreme_index = min(range(len(typed_lows)), key=lambda idx: typed_lows[idx])
        if extreme_index >= len(typed_lows) - 1:
            return None
        rundown_pips = (max(typed_highs[: extreme_index + 1]) - extreme) * pip_factor
        if rundown_pips >= atr_pips:
            return f"M1 local swing bottom run-down {rundown_pips:.1f}pip >= M1 ATR {atr_pips:.1f}pip"
    return None


def _upper_rail_touch_reason(
    m1: dict[str, Any],
    m5: dict[str, Any] | None,
    high: float,
    pip_factor: int,
    atr_pips: float,
) -> str | None:
    tolerance = max(atr_pips * 0.25, 0.0) / pip_factor
    for tf, view in (("M1", m1), ("M5", m5)):
        if not isinstance(view, dict):
            continue
        bb_upper = _indicator_float(view, "bb_upper")
        donchian_high = _indicator_float(view, "donchian_high")
        if bb_upper is not None and high >= bb_upper - tolerance:
            return f"{tf} upper rail touched high={high:.5f} bb_upper={bb_upper:.5f}"
        if donchian_high is not None and high >= donchian_high - tolerance:
            return f"{tf} Donchian high touched high={high:.5f} donchian={donchian_high:.5f}"
    return None


def _lower_rail_touch_reason(
    m1: dict[str, Any],
    m5: dict[str, Any] | None,
    low: float,
    pip_factor: int,
    atr_pips: float,
) -> str | None:
    tolerance = max(atr_pips * 0.25, 0.0) / pip_factor
    for tf, view in (("M1", m1), ("M5", m5)):
        if not isinstance(view, dict):
            continue
        bb_lower = _indicator_float(view, "bb_lower")
        donchian_low = _indicator_float(view, "donchian_low")
        if bb_lower is not None and low <= bb_lower + tolerance:
            return f"{tf} lower rail touched low={low:.5f} bb_lower={bb_lower:.5f}"
        if donchian_low is not None and low <= donchian_low + tolerance:
            return f"{tf} Donchian low touched low={low:.5f} donchian={donchian_low:.5f}"
    return None


def _temporary_extreme_reversal_evidence(
    *,
    side_up: str,
    m1: dict[str, Any],
    m5: dict[str, Any] | None,
    recent: list[dict[str, Any]],
    extreme: float,
    latest_forecast: dict[str, Any] | None,
) -> list[str]:
    evidence: list[str] = []
    long_side = side_up == "LONG"
    candle_note = _recent_candle_rollover_note(side_up, recent)
    if candle_note:
        evidence.append(candle_note)
    micro_note = _micro_opposes_note("M1", m1, side_up)
    if micro_note:
        evidence.append(micro_note)
    m5_note = _micro_opposes_note("M5", m5, side_up)
    if m5_note:
        evidence.append(m5_note)
    fvg_note = _fresh_unfilled_fvg_note(m1, "DOWN" if long_side else "UP")
    if fvg_note:
        evidence.append(fvg_note)
    break_note = _post_extreme_candle_break_note(side_up, recent, extreme)
    if break_note:
        evidence.append(break_note)
    forecast_drag = _forecast_runner_drag_reasons(side=side_up, latest_forecast=latest_forecast)
    if forecast_drag:
        evidence.append(forecast_drag[0])
    if _last_close_moved_away_from_extreme(side_up, recent, extreme):
        evidence.append("latest M1 close moved away from the local extreme")
    return evidence


def _post_extreme_candle_break_note(side_up: str, recent: list[dict[str, Any]], extreme: float) -> str | None:
    if len(recent) < 2:
        return None
    if side_up == "LONG":
        highs = [_candle_float(candle, "h", "high") for candle in recent]
        if any(value is None for value in highs):
            return None
        extreme_index = max(range(len(highs)), key=lambda idx: float(highs[idx]))
        if extreme_index >= len(recent) - 1:
            return None
        extreme_low = _candle_float(recent[extreme_index], "l", "low")
        last_close = _candle_float(recent[-1], "c", "close")
        if extreme_low is not None and last_close is not None and float(last_close) < float(extreme_low):
            return "latest M1 close broke below the extreme candle low"
    else:
        lows = [_candle_float(candle, "l", "low") for candle in recent]
        if any(value is None for value in lows):
            return None
        extreme_index = min(range(len(lows)), key=lambda idx: float(lows[idx]))
        if extreme_index >= len(recent) - 1:
            return None
        extreme_high = _candle_float(recent[extreme_index], "h", "high")
        last_close = _candle_float(recent[-1], "c", "close")
        if extreme_high is not None and last_close is not None and float(last_close) > float(extreme_high):
            return "latest M1 close broke above the extreme candle high"
    return None


def _strong_temporary_extreme_rollover(side_up: str, evidence: list[str]) -> bool:
    if side_up == "LONG":
        candle_turn = any(item.startswith("M1 rollover:") for item in evidence)
        candle_break = any("broke below the extreme candle low" in item for item in evidence)
    else:
        candle_turn = any(item.startswith("M1 rebound:") for item in evidence)
        candle_break = any("broke above the extreme candle high" in item for item in evidence)
    micro_flip = any(item.startswith("M1 opposes") for item in evidence)
    close_away = any("latest M1 close moved away from the local extreme" in item for item in evidence)
    return candle_turn and micro_flip and (candle_break or close_away)


def _recent_candle_rollover_note(side_up: str, recent: list[dict[str, Any]]) -> str | None:
    last = recent[-4:]
    closes = [_candle_float(candle, "c", "close") for candle in last]
    opens = [_candle_float(candle, "o", "open") for candle in last]
    if any(value is None for value in closes + opens):
        return None
    typed_closes = [float(value) for value in closes if value is not None]
    typed_opens = [float(value) for value in opens if value is not None]
    if side_up == "LONG":
        body_count = sum(1 for open_, close in zip(typed_opens, typed_closes) if close < open_)
        step_count = sum(1 for prev, cur in zip(typed_closes, typed_closes[1:]) if cur < prev)
        if body_count >= 2 and step_count >= 2:
            return f"M1 rollover: {body_count}/4 bearish bodies and {step_count}/3 lower closes"
    else:
        body_count = sum(1 for open_, close in zip(typed_opens, typed_closes) if close > open_)
        step_count = sum(1 for prev, cur in zip(typed_closes, typed_closes[1:]) if cur > prev)
        if body_count >= 2 and step_count >= 2:
            return f"M1 rebound: {body_count}/4 bullish bodies and {step_count}/3 higher closes"
    return None


def _micro_opposes_note(tf: str, view: dict[str, Any] | None, side_up: str) -> str | None:
    if not isinstance(view, dict):
        return None
    long_side = side_up == "LONG"
    indicators = view.get("indicators") if isinstance(view.get("indicators"), dict) else {}
    pieces: list[str] = []
    regime = str(view.get("regime") or indicators.get("regime") or "").upper()
    if long_side and regime in _PM_REGIME_DOWN:
        pieces.append(f"regime={regime}")
    elif not long_side and regime in _PM_REGIME_UP:
        pieces.append(f"regime={regime}")
    supertrend = _to_float(indicators.get("supertrend_dir"))
    if supertrend is not None:
        if long_side and supertrend < 0:
            pieces.append("SuperTrend=-")
        elif not long_side and supertrend > 0:
            pieces.append("SuperTrend=+")
    psar = _to_float(indicators.get("psar_dir"))
    if psar is not None:
        if long_side and psar < 0:
            pieces.append("PSAR=-")
        elif not long_side and psar > 0:
            pieces.append("PSAR=+")
    long_bias = _to_float(view.get("long_bias"))
    short_bias = _to_float(view.get("short_bias"))
    if long_bias is not None and short_bias is not None:
        if long_side and short_bias > long_bias:
            pieces.append(f"short_bias>{long_bias:.2f}")
        elif not long_side and long_bias > short_bias:
            pieces.append(f"long_bias>{short_bias:.2f}")
    if pieces:
        return f"{tf} opposes {side_up}: " + "/".join(pieces[:3])
    return None


def _fresh_unfilled_fvg_note(view: dict[str, Any], direction: str) -> str | None:
    structure = view.get("structure") if isinstance(view.get("structure"), dict) else {}
    gaps = structure.get("fair_value_gaps") if isinstance(structure.get("fair_value_gaps"), list) else []
    for gap in reversed(gaps[-6:]):
        if not isinstance(gap, dict):
            continue
        if str(gap.get("direction") or "").upper() != direction:
            continue
        if bool(gap.get("filled")):
            continue
        return f"M1 unfilled {direction} FVG after extreme"
    return None


def _last_close_moved_away_from_extreme(side_up: str, recent: list[dict[str, Any]], extreme: float) -> bool:
    last_close = _candle_float(recent[-1], "c", "close") if recent else None
    if last_close is None:
        return False
    if side_up == "LONG":
        return float(last_close) < extreme
    return float(last_close) > extreme


def _next_generation_structural_auto_close_allowed(m: ManagedPosition, data_root: Path) -> bool:
    """Allow auto-close only for post-change entries with hard structural evidence.

    `QR_DISABLE_AUTO_CLOSE=1` remains the default safety brake. Structural
    deterministic loss-cuts require the explicit operator/runtime opt-in
    `QR_ALLOW_STRUCTURAL_AUTO_CLOSE=1`; otherwise loss-side closes must flow
    through GPT CLOSE Gate A/B. The entry-thesis ledger is only a generation
    marker, not sufficient live permission by itself.
    """
    if m.action != ACTION_REVIEW_EXIT:
        return False
    if m.unrealized_pl_jpy >= 0:
        return False
    if not _structural_loss_cut_reason(m.reasons):
        return False
    return load_entry_thesis(m.trade_id, data_root) is not None


def _fresh_broken_thesis_close_review(
    *,
    position: BrokerPosition,
    snapshot: BrokerSnapshot,
    data_root: Path,
) -> tuple[bool, list[str]]:
    """Carry fresh thesis_evolution BROKEN into the close-review path.

    PositionManager runs after thesis-evolution in the cycle. If it ignores a
    fresh BROKEN/RECOMMEND_CLOSE row and emits a plain HOLD_PROTECTED, router
    and GPT packets treat that new HOLD as same-direction support and can bury
    the hard close evidence. This helper does not close by itself; under the
    live default QR_DISABLE_AUTO_CLOSE=1 it is demoted to
    close_review_action=REVIEW_EXIT and still must pass GPT CLOSE Gate A/B.
    """

    fetched_at = _parse_utc_datetime(snapshot.fetched_at_utc)
    payload = _fresh_report(data_root / "thesis_evolution_report.json", fetched_at)
    if not payload:
        return False, []
    for item in payload.get("evolutions", []) or []:
        if not isinstance(item, dict):
            continue
        trade_id = str(item.get("trade_id") or "")
        if trade_id and trade_id != str(position.trade_id):
            continue
        pair = str(item.get("pair") or "")
        if pair and pair != position.pair:
            continue
        side = str(item.get("side") or "").upper()
        if side and side != position.side.value:
            continue
        status = str(item.get("status") or "").upper()
        verdict = str(item.get("verdict") or "").upper()
        if status != "BROKEN" and verdict != "RECOMMEND_CLOSE":
            continue
        rationale = str(item.get("rationale") or f"status={status} verdict={verdict}")
        return (
            True,
            [
                f"thesis_evolution BROKEN/RECOMMEND_CLOSE for trade {position.trade_id}: {rationale}",
                (
                    "loss-side market close still requires GPT CLOSE Gate A/B; "
                    "keep broker TP/SL live while close review is pending"
                ),
            ],
        )
    return False, []


def _fresh_forecast_persistence_poor_rr_close_review_reason(
    *,
    position: BrokerPosition,
    snapshot: BrokerSnapshot,
    data_root: Path,
    remaining_risk: float | None,
    remaining_reward: float | None,
) -> str | None:
    if position.owner != Owner.TRADER:
        return None
    if position.unrealized_pl_jpy >= 0:
        return None
    if remaining_risk is None or remaining_reward is None:
        return None
    if remaining_risk <= 0 or remaining_reward <= 0:
        return None
    reward_risk = remaining_reward / remaining_risk
    if reward_risk >= FORECAST_PERSISTENCE_CLOSE_REVIEW_MAX_REWARD_RISK:
        return None

    fetched_at = _parse_utc_datetime(snapshot.fetched_at_utc)
    payload = _fresh_report(data_root / "forecast_persistence_report.json", fetched_at)
    if not payload:
        return None
    for item in payload.get("verdicts", []) or []:
        if not isinstance(item, dict):
            continue
        trade_id = str(item.get("trade_id") or "")
        if trade_id and trade_id != str(position.trade_id):
            continue
        pair = str(item.get("pair") or "")
        if pair and pair != position.pair:
            continue
        side = str(item.get("side") or "").upper()
        if side and side != position.side.value:
            continue
        verdict = str(item.get("verdict") or "").upper()
        if verdict != "RECOMMEND_CLOSE":
            continue
        reason = str(item.get("reason") or "forecast persistence recommends capital recycle")
        return (
            "close-review: forecast_persistence RECOMMEND_CLOSE for trade "
            f"{position.trade_id}: {reason}; remaining reward/risk "
            f"{reward_risk:.2f} < {FORECAST_PERSISTENCE_CLOSE_REVIEW_MAX_REWARD_RISK:.2f} "
            f"({remaining_reward:.0f} JPY reward vs {remaining_risk:.0f} JPY risk)"
        )
    return None


def _fresh_report(path: Path, fetched_at: datetime | None) -> dict[str, Any] | None:
    if fetched_at is None or not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    generated_at = _parse_utc_datetime(payload.get("generated_at_utc"))
    if generated_at is None or generated_at < fetched_at:
        return None
    return payload


def _structural_auto_close_enabled() -> bool:
    return os.environ.get("QR_ALLOW_STRUCTURAL_AUTO_CLOSE", "").strip().lower() in {"1", "true", "yes"}


def _structural_loss_cut_reason(reasons: tuple[str, ...]) -> bool:
    """Return true only for deterministic structural loss-cut evidence.

    Entry-thesis invalidation hits and forecast-confidence collapses are useful
    Gate A material, but they are not deterministic structural auto-close
    permission. They must remain in the GPT CLOSE Gate A/B path so same-side
    recovery context, matrix support, and operator authorization can be checked.
    """
    for reason in reasons:
        text = str(reason)
        if not text.startswith("loss-cut:"):
            continue
        lowered = text.lower()
        if "close-confirmed structural break" in lowered:
            return True
        if "structural ob broken" in lowered:
            return True
    return False


def _thesis_confidence_collapse_ratio() -> float:
    try:
        return max(0.0, min(1.0, float(os.environ.get("QR_THESIS_CONFIDENCE_COLLAPSE_RATIO", "0.50"))))
    except ValueError:
        return 0.50


def _entry_thesis_invalidation_review(
    *,
    position: BrokerPosition,
    quote,
    full_pair_charts: dict[str, dict[str, Any]] | None,
    data_root: Path,
    latest_forecast: dict[str, Any] | None = None,
) -> tuple[bool, list[str]]:
    if position.owner != Owner.TRADER or position.unrealized_pl_jpy >= 0:
        return False, []
    thesis = load_entry_thesis(position.trade_id, data_root)
    if thesis is None:
        return False, []
    if quote is None:
        return False, ["entry thesis invalidation check deferred: missing current quote"]

    side = position.side.value
    price = quote.bid if side == "LONG" else quote.ask
    label = "bid" if side == "LONG" else "ask"
    chart = (full_pair_charts or {}).get(position.pair)
    technical_reason = technical_invalidation_confirmation_reason(chart, side=side)
    hit_reason = (
        thesis_invalidation_hit_reason(
            thesis,
            side=side,
            current_price=price,
            price_label=label,
        )
        if thesis.invalidation_price is not None
        else None
    )
    if hit_reason:
        if not technical_reason:
            return False, [f"{hit_reason}; waiting for chart/technical confirmation"]
        return True, [
            (
                "loss-cut: entry thesis invalidation hit: "
                f"{hit_reason}; {technical_reason} ({position.unrealized_pl_jpy:+.0f} JPY)"
            )
        ]

    collapse_reasons = _entry_thesis_confidence_collapse_review(
        position=position,
        thesis_forecast_direction=thesis.forecast_direction,
        thesis_forecast_confidence=thesis.forecast_confidence,
        latest_forecast=latest_forecast,
        technical_reason=technical_reason,
    )
    if collapse_reasons:
        return True, collapse_reasons
    return False, []


def _entry_thesis_confidence_collapse_review(
    *,
    position: BrokerPosition,
    thesis_forecast_direction: str,
    thesis_forecast_confidence: float,
    latest_forecast: dict[str, Any] | None,
    technical_reason: str | None,
) -> list[str]:
    if not latest_forecast or not technical_reason:
        return []
    aligned_direction = "UP" if position.side == Side.LONG else "DOWN"
    thesis_direction = str(thesis_forecast_direction or "").upper()
    latest_direction = str(latest_forecast.get("direction") or "").upper()
    if thesis_direction != aligned_direction or latest_direction != aligned_direction:
        return []
    latest_confidence = _to_float(latest_forecast.get("confidence"))
    if latest_confidence is None or thesis_forecast_confidence <= 0:
        return []
    collapse_ratio = _thesis_confidence_collapse_ratio()
    collapse_threshold = thesis_forecast_confidence * collapse_ratio
    if latest_confidence >= collapse_threshold:
        return []
    return [
        (
            "loss-cut: entry thesis confidence collapse: "
            f"entry {thesis_direction} conf={thesis_forecast_confidence:.2f} → "
            f"latest {latest_direction} conf={latest_confidence:.2f} "
            f"(< {collapse_ratio:.2f}× entry); {technical_reason} "
            f"({position.unrealized_pl_jpy:+.0f} JPY)"
        )
    ]


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
    if ACTION_REPAIR_TAKE_PROFIT in actions:
        return ACTION_REPAIR_TAKE_PROFIT
    if ACTION_TAKE_PROFIT_MARKET in actions:
        return ACTION_TAKE_PROFIT_MARKET
    if ACTION_REVIEW_EXIT in actions:
        return ACTION_REVIEW_EXIT
    if ACTION_BREAK_EVEN_STOP in actions:
        return ACTION_BREAK_EVEN_STOP
    if ACTION_PROFIT_PROTECT in actions:
        return ACTION_PROFIT_PROTECT
    if ACTION_HARVEST_TP in actions:
        return ACTION_HARVEST_TP
    if ACTION_NARROW_TP in actions:
        return ACTION_NARROW_TP
    if ACTION_EXTEND_TP in actions:
        return ACTION_EXTEND_TP
    if positions:
        return ACTION_HOLD_PROTECTED
    return "NO_POSITION"


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bb_rail_pressure(
    pair: str,
    lane_dir: str,
    pair_charts: dict[str, dict[str, Any]] | None,
) -> tuple[float, list[str]]:
    per_tf = (pair_charts or {}).get(pair) if pair_charts else None
    if not isinstance(per_tf, dict):
        return 0.0, []
    target_up = lane_dir == "LONG"
    score = 0.0
    reasons: list[str] = []
    for tf in BB_RAIL_TIMEFRAMES:
        ind = per_tf.get(tf)
        if not isinstance(ind, dict):
            continue
        bb_pos = _bb_position(ind)
        if bb_pos is None:
            continue
        overbought = _oscillator_overbought(ind)
        oversold = _oscillator_oversold(ind)
        if bb_pos >= BB_RAIL_EDGE_FRACTION and overbought:
            supports_up = False
            if supports_up == target_up:
                score += 1.0
                reasons.append(f"{tf} upper BB rail/overbought supports {lane_dir}: %B={bb_pos:.2f}, {overbought}")
            else:
                score -= 1.0
                reasons.append(f"{tf} upper BB rail/overbought opposes {lane_dir}: %B={bb_pos:.2f}, {overbought}")
        elif bb_pos <= 1.0 - BB_RAIL_EDGE_FRACTION and oversold:
            supports_up = True
            if supports_up == target_up:
                score += 1.0
                reasons.append(f"{tf} lower BB rail/oversold supports {lane_dir}: %B={bb_pos:.2f}, {oversold}")
            else:
                score -= 1.0
                reasons.append(f"{tf} lower BB rail/oversold opposes {lane_dir}: %B={bb_pos:.2f}, {oversold}")
    return score, reasons


def _existing_tp_at_opposite_bb_rail(
    position: BrokerPosition,
    pair_charts: dict[str, dict[str, Any]] | None,
) -> tuple[bool, str]:
    if position.take_profit is None:
        return False, "no existing TP"
    per_tf = (pair_charts or {}).get(position.pair) if pair_charts else None
    if not isinstance(per_tf, dict):
        return False, "BB unavailable"
    # M5 is the position-management rail: M1 is too noisy for TP placement,
    # M15 is often too slow for a short-lived MFE capture.
    ind = per_tf.get("M5")
    if not isinstance(ind, dict):
        return False, "M5 BB unavailable"
    tp_pos = _bb_position(ind, price=position.take_profit)
    if tp_pos is None:
        return False, "M5 BB invalid"
    if position.side == Side.SHORT and tp_pos <= 1.0 - BB_RAIL_EDGE_FRACTION:
        return True, f"M5 TP %B={tp_pos:.2f} at lower rail"
    if position.side == Side.LONG and tp_pos >= BB_RAIL_EDGE_FRACTION:
        return True, f"M5 TP %B={tp_pos:.2f} at upper rail"
    return False, f"M5 TP %B={tp_pos:.2f} not at opposite rail"


def _bb_position(indicators: dict[str, Any], *, price: float | None = None) -> float | None:
    close = _to_float(price if price is not None else indicators.get("close"))
    upper = _to_float(indicators.get("bb_upper"))
    lower = _to_float(indicators.get("bb_lower"))
    if close is None or upper is None or lower is None or upper <= lower:
        return None
    return (close - lower) / (upper - lower)


def _oscillator_overbought(indicators: dict[str, Any]) -> str | None:
    parts: list[str] = []
    stoch = _to_float(indicators.get("stoch_rsi"))
    if stoch is not None and (
        (stoch <= 1.5 and stoch >= STOCH_RSI_HIGH)
        or (stoch > 1.5 and stoch >= STOCH_RSI_PERCENT_SCALE_HIGH)
    ):
        parts.append(f"StochRSI={stoch:.2f}")
    williams = _to_float(indicators.get("williams_r_14"))
    if williams is not None and williams >= WILLIAMS_OVERBOUGHT:
        parts.append(f"%R={williams:.1f}")
    mfi = _to_float(indicators.get("mfi_14"))
    if mfi is not None and mfi >= MFI_OVERBOUGHT:
        parts.append(f"MFI={mfi:.1f}")
    return "/".join(parts) if parts else None


def _oscillator_oversold(indicators: dict[str, Any]) -> str | None:
    parts: list[str] = []
    stoch = _to_float(indicators.get("stoch_rsi"))
    if stoch is not None and (
        stoch <= STOCH_RSI_LOW or (stoch > 1.0 and stoch <= STOCH_RSI_PERCENT_SCALE_LOW)
    ):
        parts.append(f"StochRSI={stoch:.2f}")
    williams = _to_float(indicators.get("williams_r_14"))
    if williams is not None and williams <= WILLIAMS_OVERSOLD:
        parts.append(f"%R={williams:.1f}")
    mfi = _to_float(indicators.get("mfi_14"))
    if mfi is not None and mfi <= MFI_OVERSOLD:
        parts.append(f"MFI={mfi:.1f}")
    return "/".join(parts) if parts else None


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


def _sl_free_profit_lock_stop_candidate(
    position: BrokerPosition,
    quote,
    pair_charts: dict[str, dict[str, Any]] | None,
) -> tuple[float | None, tuple[str, ...]]:
    """Return a BE-or-better stop only after SL-free profit clears live micro-noise."""
    if position.owner != Owner.TRADER or position.stop_loss is not None or not _trader_sl_repair_disabled():
        return None, ()
    if position.unrealized_pl_jpy <= 0:
        return None, ("SL-free profit-lock deferred: position is not profitable",)
    break_even = _break_even_stop(position, quote)
    if break_even is None:
        return None, ("SL-free profit-lock deferred: entry is not reward-side of the current executable price",)
    profit_pips = _executable_profit_pips(position, quote)
    if profit_pips is None or profit_pips <= 0:
        return None, ("SL-free profit-lock deferred: executable profit pips cannot be measured from broker quote",)
    noise_pips, noise_basis = _profit_break_even_noise_pips(position.pair, quote, pair_charts)
    if noise_pips is None:
        return None, (f"SL-free profit-lock deferred until fresh volatility can be measured ({noise_basis})",)
    if profit_pips < noise_pips:
        return (
            None,
            (
                f"SL-free profit-lock deferred: executable profit {profit_pips:.1f}pip < "
                f"micro-noise {noise_pips:.1f}pip ({noise_basis})",
            ),
        )
    tp_gate, tp_gate_note = _profit_lock_tp_progress_gate_pips(position)
    if tp_gate is not None and profit_pips < tp_gate:
        return (
            None,
            (
                f"SL-free profit-lock deferred: executable profit {profit_pips:.1f}pip < "
                f"TP-progress gate {tp_gate:.1f}pip ({tp_gate_note})",
            ),
        )
    lock_stop = _profit_lock_stop(position, quote, noise_pips)
    if lock_stop is None:
        return None, ("SL-free profit-lock deferred: market-valid stop cannot be computed",)
    locked_pips = _locked_profit_pips(position, lock_stop)
    lock_label = "break-even" if locked_pips <= 0 else f"+{locked_pips:.1f}pip"
    return (
        lock_stop,
        (
            f"SL-free profit-lock trigger: executable profit {profit_pips:.1f}pip >= "
            f"micro-noise {noise_pips:.1f}pip ({noise_basis}); "
            f"{tp_gate_note}; stop {lock_stop:.5f} ({lock_label})",
        ),
    )


def _profit_lock_tp_progress_gate_pips(position: BrokerPosition) -> tuple[float | None, str]:
    tp_pips = _position_tp_pips(position)
    if tp_pips is None or tp_pips <= 0:
        return None, "no broker TP progress gate"
    gate = tp_pips * PROFIT_BREAK_EVEN_MIN_TP_PROGRESS
    return gate, f"TP progress gate {PROFIT_BREAK_EVEN_MIN_TP_PROGRESS:.0%} of {tp_pips:.1f}pip target"


def _latest_forecast_for_position(pair: str, data_root: Path) -> dict[str, Any] | None:
    try:
        latest = load_latest_forecast(pair, data_root)
    except Exception:
        return None
    return latest if isinstance(latest, dict) else None


def _adaptive_tp_chart_context(
    pair_chart: dict[str, Any] | None,
    flat_pair_chart: dict[str, Any] | None,
) -> dict[str, Any]:
    confluence: dict[str, Any] = {}
    session: dict[str, Any] = {}
    indicators_by_tf: dict[str, dict[str, Any]] = {}
    if isinstance(pair_chart, dict):
        raw_confluence = pair_chart.get("confluence")
        if isinstance(raw_confluence, dict):
            confluence = raw_confluence
        raw_session = pair_chart.get("session")
        if isinstance(raw_session, dict):
            session = raw_session
        for view in pair_chart.get("views") or []:
            if not isinstance(view, dict):
                continue
            granularity = str(view.get("granularity") or "")
            indicators = view.get("indicators")
            if granularity and isinstance(indicators, dict):
                indicators_by_tf[granularity] = indicators
    if isinstance(flat_pair_chart, dict):
        if not confluence and isinstance(flat_pair_chart.get("confluence"), dict):
            confluence = flat_pair_chart["confluence"]
        if not session and isinstance(flat_pair_chart.get("session"), dict):
            session = flat_pair_chart["session"]
        for tf in ("M15", "M30", "H1"):
            indicators = flat_pair_chart.get(tf)
            if tf not in indicators_by_tf and isinstance(indicators, dict):
                indicators_by_tf[tf] = indicators
    return {
        "confluence": confluence,
        "session": session,
        "indicators_by_tf": indicators_by_tf,
    }


def _forecast_technical_mfe_risk_reasons(
    position: BrokerPosition,
    *,
    profit_pips: float,
    latest_forecast: dict[str, Any] | None,
    chart_context: dict[str, Any] | None,
) -> tuple[tuple[str, ...], str]:
    forecast_reasons = _forecast_runner_drag_reasons(
        side=position.side.value,
        latest_forecast=latest_forecast,
    )
    technical_reasons = _technical_harvest_pressure(
        side=position.side.value,
        chart_context=chart_context,
    )
    if not forecast_reasons:
        if latest_forecast:
            direction = str(latest_forecast.get("direction") or "UNCLEAR").upper()
            confidence = _to_float(latest_forecast.get("confidence")) or 0.0
            return (), f"forecast {direction} conf={confidence:.2f} does not weaken {position.side.value} runner"
        return (), "latest forecast unavailable; cannot prove runner edge has weakened"
    if len(technical_reasons) < 2:
        return (), (
            "forecast drag present but technical MFE-risk stack is incomplete "
            f"({len(technical_reasons)}/2): " + "; ".join((forecast_reasons + technical_reasons)[:3])
        )
    reasons = tuple((forecast_reasons + technical_reasons)[:5])
    return reasons, "forecast + technical MFE-risk: " + "; ".join(reasons)


def _adaptive_tp_contraction_allowed(
    position: BrokerPosition,
    quote,
    pair_charts: dict[str, dict[str, Any]] | None,
    action: str,
    *,
    latest_forecast: dict[str, Any] | None = None,
    chart_context: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    """Gate HARVEST/NARROW TP contractions with forecast and technical evidence."""
    label = action.lower().replace("_", "-")
    tp_pips = _position_tp_pips(position)
    if tp_pips is None or tp_pips <= 0:
        return False, f"{label} skipped: current broker TP distance cannot be measured"
    profit_pips = _executable_profit_pips(position, quote)
    if profit_pips is None or profit_pips <= 0:
        return False, f"{label} skipped: executable profit pips cannot be measured from broker quote"

    atr_pips = _atr_pips_for(position.pair, pair_charts, GEOMETRY_ATR_TIMEFRAME)
    if atr_pips is not None and atr_pips > 0:
        stale_wide_gate = ADAPTIVE_TP_STALE_DISTANCE_ATR_MULT * atr_pips
        if tp_pips > stale_wide_gate:
            mfe_reasons, mfe_note = _forecast_technical_mfe_risk_reasons(
                position,
                profit_pips=profit_pips,
                latest_forecast=latest_forecast,
                chart_context=chart_context,
            )
            if profit_pips < atr_pips:
                return (
                    True,
                    f"{label} allowed: existing TP {tp_pips:.1f}pip > "
                    f"stale-wide {ADAPTIVE_TP_STALE_DISTANCE_ATR_MULT:.1f}× "
                    f"{GEOMETRY_ATR_TIMEFRAME} ATR {atr_pips:.1f}pip; "
                    f"profit {profit_pips:.1f}pip < operating ATR {atr_pips:.1f}pip, "
                    "so original TP is not behaving like a reachable runner",
                )
            if mfe_reasons:
                return (
                    True,
                    f"{label} allowed: existing TP {tp_pips:.1f}pip > "
                    f"stale-wide {ADAPTIVE_TP_STALE_DISTANCE_ATR_MULT:.1f}× "
                    f"{GEOMETRY_ATR_TIMEFRAME} ATR {atr_pips:.1f}pip; {mfe_note}",
                )
            return (
                False,
                f"{label} skipped: existing TP is stale-wide but current profit "
                f"{profit_pips:.1f}pip >= operating ATR {atr_pips:.1f}pip and {mfe_note}",
            )
        atr_note = (
            f"; existing TP {tp_pips:.1f}pip is within "
            f"{ADAPTIVE_TP_STALE_DISTANCE_ATR_MULT:.1f}× {GEOMETRY_ATR_TIMEFRAME} ATR {atr_pips:.1f}pip"
        )
    else:
        atr_note = f"; {GEOMETRY_ATR_TIMEFRAME} ATR unavailable for stale-wide TP exception"

    mfe_reasons, mfe_note = _forecast_technical_mfe_risk_reasons(
        position,
        profit_pips=profit_pips,
        latest_forecast=latest_forecast,
        chart_context=chart_context,
    )
    if not mfe_reasons:
        return False, f"{label} skipped: reachable TP contraction needs market-read MFE risk; {mfe_note}{atr_note}"

    if atr_pips is not None and atr_pips > 0 and profit_pips > atr_pips:
        return (
            True,
            f"{label} allowed: executable profit {profit_pips:.1f}pip > "
            f"{GEOMETRY_ATR_TIMEFRAME} ATR {atr_pips:.1f}pip under market-read MFE risk; "
            f"{mfe_note}{atr_note}",
        )

    gate = tp_pips * ADAPTIVE_TP_CONTRACTION_MIN_PROGRESS
    if profit_pips < gate:
        return (
            False,
            f"{label} skipped: executable profit {profit_pips:.1f}pip < "
            f"adaptive TP-progress gate {gate:.1f}pip "
            f"({ADAPTIVE_TP_CONTRACTION_MIN_PROGRESS:.0%} of {tp_pips:.1f}pip target{atr_note}); {mfe_note}",
        )
    return (
        True,
        f"{label} allowed: executable profit {profit_pips:.1f}pip >= "
        f"adaptive TP-progress gate {gate:.1f}pip "
        f"({ADAPTIVE_TP_CONTRACTION_MIN_PROGRESS:.0%} of {tp_pips:.1f}pip target{atr_note}); {mfe_note}",
    )


def _profit_lock_stop(position: BrokerPosition, quote, noise_pips: float) -> float | None:
    if quote is None or noise_pips <= 0:
        return None
    pip_size = 1.0 / _pip_factor(position.pair)
    distance = noise_pips * pip_size
    if position.side == Side.LONG:
        candidate = max(position.entry_price, quote.bid - distance)
    else:
        candidate = min(position.entry_price, quote.ask + distance)
    if not _market_valid_stop(position, candidate, quote):
        return None
    return candidate


def _locked_profit_pips(position: BrokerPosition, stop_loss: float) -> float:
    if position.side == Side.LONG:
        return max(0.0, (stop_loss - position.entry_price) * _pip_factor(position.pair))
    return max(0.0, (position.entry_price - stop_loss) * _pip_factor(position.pair))


def _profit_break_even_noise_pips(
    pair: str,
    quote,
    pair_charts: dict[str, dict[str, Any]] | None,
) -> tuple[float | None, str]:
    vol_candidates: list[tuple[float, str]] = []
    atr_pips = _atr_pips_for(pair, pair_charts, PROFIT_BREAK_EVEN_ATR_TIMEFRAME)
    atr_fresh, atr_fresh_reason = _volatility_source_fresh(pair, quote, pair_charts, PROFIT_BREAK_EVEN_ATR_TIMEFRAME)
    if atr_pips is not None and atr_pips > 0 and atr_fresh:
        value = atr_pips * PROFIT_BREAK_EVEN_NOISE_ATR_MULT
        vol_candidates.append((value, f"fresh {PROFIT_BREAK_EVEN_ATR_TIMEFRAME} ATR {atr_pips:.1f}pip ({atr_fresh_reason})"))
    quick_pips, quick_basis = _quick_realized_volatility_pips(pair, quote, pair_charts)
    if quick_pips is not None and quick_pips > 0:
        vol_candidates.append((quick_pips, quick_basis))
    if not vol_candidates:
        return None, "missing fresh M1/M5 quick volatility and fresh M5 ATR"

    candidates = list(vol_candidates)
    spread_pips = _spread_pips(pair, quote)
    if spread_pips is not None and spread_pips > 0:
        value = spread_pips * PROFIT_BREAK_EVEN_SPREAD_MULT
        candidates.append((value, f"spread {spread_pips:.1f}pip x {PROFIT_BREAK_EVEN_SPREAD_MULT:.1f}"))
    return max(candidates, key=lambda item: item[0])


def _volatility_source_fresh(
    pair: str,
    quote,
    pair_charts: dict[str, dict[str, Any]] | None,
    timeframe: str,
) -> tuple[bool, str]:
    quote_ts = _quote_timestamp_utc(quote)
    if quote_ts is None or pair_charts is None:
        return False, "quote or pair-charts timestamp unavailable"
    per_tf = pair_charts.get(pair)
    if not per_tf:
        return False, "pair chart missing"
    latest_candle = _latest_recent_candle_time(per_tf.get(f"{timeframe}__recent_candles"))
    window = _quick_vol_window_seconds()
    if latest_candle is not None:
        age_seconds = (quote_ts - latest_candle).total_seconds()
        if 0 <= age_seconds <= window:
            return True, f"latest {timeframe} candle age {age_seconds:.0f}s <= quick window {window:.0f}s"
        return False, f"latest {timeframe} candle stale age {age_seconds:.0f}s > quick window {window:.0f}s"
    generated_at = _parse_utc_datetime(per_tf.get("generated_at_utc"))
    if generated_at is None:
        return False, "pair chart generated_at missing"
    age_seconds = (quote_ts - generated_at).total_seconds()
    if 0 <= age_seconds <= window:
        return True, f"pair-charts generated age {age_seconds:.0f}s <= quick window {window:.0f}s"
    return False, f"pair-charts generated stale age {age_seconds:.0f}s > quick window {window:.0f}s"


def _quick_realized_volatility_pips(
    pair: str,
    quote,
    pair_charts: dict[str, dict[str, Any]] | None,
) -> tuple[float | None, str]:
    quote_ts = _quote_timestamp_utc(quote)
    if quote_ts is None or pair_charts is None:
        return None, "quote or pair-charts timestamp unavailable for quick volatility"
    per_tf = pair_charts.get(pair)
    if not per_tf:
        return None, "pair chart missing for quick volatility"
    window = _quick_vol_window_seconds()
    candidates: list[tuple[float, str]] = []
    for timeframe in QUICK_VOL_TIMEFRAMES:
        candles = _fresh_recent_candles(per_tf.get(f"{timeframe}__recent_candles"), quote_ts, window)
        if not candles:
            continue
        highs = [float(item["h"]) for item in candles]
        lows = [float(item["l"]) for item in candles]
        range_pips = (max(highs) - min(lows)) * _pip_factor(pair)
        if range_pips > 0:
            latest_age = (quote_ts - candles[-1]["t"]).total_seconds()
            candidates.append((range_pips, f"quick {timeframe} range {range_pips:.1f}pip/{len(candles)} bars age {latest_age:.0f}s"))
    if not candidates:
        return None, "no fresh recent M1/M5 candles inside quick volatility window"
    return max(candidates, key=lambda item: item[0])


def _fresh_recent_candles(raw_candles: object, quote_ts: datetime, window_seconds: float) -> list[dict[str, Any]]:
    if not isinstance(raw_candles, list):
        return []
    out: list[dict[str, Any]] = []
    for raw in raw_candles:
        if not isinstance(raw, dict):
            continue
        ts = _parse_utc_datetime(raw.get("t") or raw.get("timestamp_utc") or raw.get("time"))
        high = _float_or_none(raw.get("h") or raw.get("high"))
        low = _float_or_none(raw.get("l") or raw.get("low"))
        if ts is None or high is None or low is None or high <= low:
            continue
        age_seconds = (quote_ts - ts).total_seconds()
        if 0 <= age_seconds <= window_seconds:
            out.append({"t": ts, "h": high, "l": low})
    out.sort(key=lambda item: item["t"])
    return out


def _latest_recent_candle_time(raw_candles: object) -> datetime | None:
    if not isinstance(raw_candles, list):
        return None
    times: list[datetime] = []
    for raw in raw_candles:
        if isinstance(raw, dict):
            parsed = _parse_utc_datetime(raw.get("t") or raw.get("timestamp_utc") or raw.get("time"))
            if parsed is not None:
                times.append(parsed)
    return max(times) if times else None


def _quick_vol_window_seconds() -> float:
    return float(QUICK_VOL_GRANULARITY_SECONDS[QUICK_VOL_WINDOW_TIMEFRAME])


def _quote_timestamp_utc(quote) -> datetime | None:
    if quote is None:
        return None
    return _parse_utc_datetime(getattr(quote, "timestamp_utc", None))


def _parse_utc_datetime(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc)
    text = str(value or "")
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        return datetime.fromisoformat(text).astimezone(timezone.utc)
    except ValueError:
        return None


def _float_or_none(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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


def _rollover_spread_impaired_close_review_reason(
    position: BrokerPosition,
    quote,
    pair_charts: dict[str, dict[str, Any]] | None,
    *,
    remaining_risk: float | None,
    remaining_reward: float | None,
) -> str | None:
    if position.unrealized_pl_jpy >= 0:
        return None
    if position.take_profit is None or position.stop_loss is None or quote is None:
        return None
    if _session_bucket_for(position.pair, pair_charts) != "ROLLOVER":
        return None
    spread_pips = _spread_pips(position.pair, quote)
    tp_pips = _position_tp_pips(position)
    if spread_pips is None or tp_pips is None or tp_pips <= 0:
        return None
    if spread_pips < tp_pips:
        return None
    if (
        remaining_risk is None
        or remaining_reward is None
        or remaining_reward <= 0
        or remaining_risk <= remaining_reward
    ):
        return None
    return (
        f"close-review: ROLLOVER spread {spread_pips:.1f}pip >= planned TP distance "
        f"{tp_pips:.1f}pip; short-horizon reward is inside closeout spread while "
        f"remaining risk {remaining_risk:.0f} JPY > remaining reward {remaining_reward:.0f} JPY"
    )


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


def _tp_remaining_pips(position: BrokerPosition, quote) -> float | None:
    if position.take_profit is None or quote is None:
        return None
    if position.side == Side.LONG:
        distance = position.take_profit - quote.bid
    else:
        distance = quote.ask - position.take_profit
    return max(0.0, distance * _pip_factor(position.pair))


def _spread_pips(pair: str, quote) -> float | None:
    if quote is None:
        return None
    spread = (quote.ask - quote.bid) * _pip_factor(pair)
    return spread if spread > 0 else None


def _executable_profit_pips(position: BrokerPosition, quote) -> float | None:
    if quote is None:
        return None
    if position.side == Side.LONG:
        return (quote.bid - position.entry_price) * _pip_factor(position.pair)
    return (position.entry_price - quote.ask) * _pip_factor(position.pair)


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


def _market_take_profit_repair_candidate(
    position: BrokerPosition,
    quote,
    pair_charts: dict[str, dict[str, Any]] | None,
) -> tuple[float | None, str]:
    if quote is None:
        return None, "quote unavailable; cannot enforce current-price TP safety margin"
    chart_context = (pair_charts or {}).get(position.pair) if pair_charts else None
    if not chart_context:
        return None, "pair chart context unavailable; no silent TP fallback"
    atr_pips = _atr_pips_for(position.pair, pair_charts, GEOMETRY_ATR_TIMEFRAME)
    if atr_pips is None or atr_pips <= 0:
        return None, f"{GEOMETRY_ATR_TIMEFRAME} ATR unavailable; no silent TP fallback"
    reward_risk, rr_reasons = _market_derived_reward_risk(chart_context)
    if reward_risk <= 0:
        return None, "market-derived reward_risk is not positive"

    pip_factor = _pip_factor(position.pair)
    pip_size = 1.0 / pip_factor
    distance_pips = min(reward_risk * atr_pips, MAX_TP_DISTANCE_ATR_MULT * atr_pips)
    safety_distance = MIN_TP_TO_MARKET_PIPS * pip_size
    if position.side == Side.LONG:
        entry_candidate = position.entry_price + distance_pips * pip_size
        market_candidate = quote.ask + safety_distance
        candidate = max(entry_candidate, market_candidate)
        if candidate <= position.entry_price:
            return None, "computed LONG TP is not on reward side of entry"
    else:
        entry_candidate = position.entry_price - distance_pips * pip_size
        market_candidate = quote.bid - safety_distance
        candidate = min(entry_candidate, market_candidate)
        if candidate >= position.entry_price:
            return None, "computed SHORT TP is not on reward side of entry"

    rationale = (
        f"{GEOMETRY_ATR_TIMEFRAME} ATR {atr_pips:.1f}pip × reward_risk {reward_risk:.2f} "
        f"capped at {MAX_TP_DISTANCE_ATR_MULT:.1f}×ATR; "
        f"current-price safety {MIN_TP_TO_MARKET_PIPS:.1f}pip"
    )
    if rr_reasons:
        rationale += "; rr: " + "; ".join(rr_reasons[:2])
    return candidate, rationale


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
    daily target ledger's equity-derived per-trade cap; return None when the
    ledger is absent instead of inventing a JPY fallback.
    """
    if DEFAULT_DAILY_TARGET_STATE.exists():
        try:
            payload = json.loads(DEFAULT_DAILY_TARGET_STATE.read_text())
            value = float(payload.get("per_trade_risk_budget_jpy") or 0.0)
            if value > 0:
                return value
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None
    return None
