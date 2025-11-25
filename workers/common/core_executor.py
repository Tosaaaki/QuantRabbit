"""Common executor that turns PocketPlan objects into orders."""

from __future__ import annotations

import datetime
import logging
import os
from typing import Dict, Tuple

from analytics.insight_client import InsightClient
from execution.exit_manager import ExitManager, ExitDecision
from execution.managed_positions import filter_bot_managed_positions
from execution.order_ids import build_client_order_id
from execution.order_manager import (
    close_trade,
    market_order,
    min_units_for_pocket,
    plan_partial_reductions,
    update_dynamic_protections,
)
from execution.pocket_limits import POCKET_ENTRY_MIN_INTERVAL, POCKET_LOSS_COOLDOWNS, cooldown_for_pocket
from execution.risk_guard import can_trade, clamp_sl_tp
from execution.stage_rules import compute_stage_lot
from execution.stage_tracker import StageTracker
from execution.position_manager import PositionManager
from workers.common.pocket_plan import PocketPlan, PocketType

LOG = logging.getLogger(__name__)


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


_USD_LONG_CAP_LOT = _env_float("EXPOSURE_USD_LONG_MAX_LOT", 2.5)


class PocketPlanExecutor:
    def __init__(self, pocket: PocketType, *, log_prefix: str) -> None:
        self.pocket = pocket
        self.log_prefix = log_prefix
        self.stage_tracker = StageTracker()
        self.exit_manager = ExitManager()
        self.pos_manager = PositionManager()
        self.insight = InsightClient()
        self._last_insight_refresh = datetime.datetime.min
        self._stage_empty_since: Dict[Tuple[str, str], datetime.datetime] = {}

    async def process_plan(self, plan: PocketPlan) -> None:
        now = datetime.datetime.utcnow()
        self.stage_tracker.clear_expired(now)
        self.stage_tracker.update_loss_streaks(now=now, cooldown_map=POCKET_LOSS_COOLDOWNS)
        self._maybe_refresh_insight(now)

        open_positions = self.pos_manager.get_open_positions()
        managed_positions = filter_bot_managed_positions(open_positions)

        try:
            update_dynamic_protections(managed_positions, plan.factors_m1, plan.factors_h4)
        except Exception as exc:  # pragma: no cover - defensive
            LOG.warning("%s protection update failed: %s", self.log_prefix, exc)

        try:
            partials = plan_partial_reductions(
                managed_positions,
                plan.factors_m1,
                plan.factors_h4,
                range_mode=plan.range_active,
                now=now,
            )
        except Exception as exc:  # pragma: no cover - defensive
            LOG.warning("%s partial plan failed: %s", self.log_prefix, exc)
            partials = []
        partial_closed = False
        for pocket, trade_id, reduce_units in partials:
            if pocket != self.pocket:
                continue
            ok = await close_trade(trade_id, reduce_units)
            if ok:
                LOG.info(
                    "%s partial pocket=%s trade=%s units=%s",
                    self.log_prefix,
                    pocket,
                    trade_id,
                    reduce_units,
                )
                partial_closed = True
        if partial_closed:
            open_positions = self.pos_manager.get_open_positions()
            managed_positions = filter_bot_managed_positions(open_positions)

        try:
            resets = self.stage_tracker.expire_stages_if_flat(
                open_positions, now=now, grace_seconds=180
            )
            if resets:
                LOG.info("%s auto-reset %d stale stages", self.log_prefix, resets)
        except Exception as exc:  # pragma: no cover - defensive
            LOG.debug("%s stage reset check failed: %s", self.log_prefix, exc)

        self._update_stage_resets(managed_positions, now)
        block_entries = await self._handle_exits(managed_positions, plan, now)
        await self._handle_entries(open_positions, managed_positions, plan, now, block_entries)

    def close(self) -> None:
        try:
            self.stage_tracker.close()
        except Exception:  # pragma: no cover - defensive
            pass
        try:
            self.pos_manager.close()
        except Exception:  # pragma: no cover - defensive
            pass

    def _maybe_refresh_insight(self, now: datetime.datetime) -> None:
        if (now - self._last_insight_refresh).total_seconds() < 300:
            return
        try:
            self.insight.refresh()
            self._last_insight_refresh = now
        except Exception:  # pragma: no cover - defensive
            pass

    def _update_stage_resets(self, positions: Dict[str, Dict], now: datetime.datetime) -> None:
        info = positions.get(self.pocket)
        if not info:
            return
        for direction, key_units in (("long", "long_units"), ("short", "short_units")):
            units_value = int(info.get(key_units, 0) or 0)
            tracker_stage = self.stage_tracker.get_stage(self.pocket, direction)
            key = (self.pocket, direction)
            if units_value == 0 and tracker_stage > 0:
                empty_since = self._stage_empty_since.get(key)
                if empty_since is None:
                    self._stage_empty_since[key] = now
                elif (now - empty_since).total_seconds() >= 180:
                    LOG.info(
                        "%s stage reset pocket=%s dir=%s",
                        self.log_prefix,
                        self.pocket,
                        direction,
                    )
                    self.stage_tracker.reset_stage(self.pocket, direction, now=now)
                    self._stage_empty_since.pop(key, None)
            elif units_value != 0:
                self._stage_empty_since.pop(key, None)

    async def _handle_exits(
        self,
        managed_positions: Dict[str, Dict],
        plan: PocketPlan,
        now: datetime.datetime,
    ) -> bool:
        block_entries = False
        exit_decisions = self.exit_manager.plan_closures(
            managed_positions,
            plan.signals,
            plan.factors_m1,
            plan.factors_h4,
            plan.event_soon,
            plan.range_active,
            now=now,
            stage_tracker=self.stage_tracker,
        )
        for decision in exit_decisions:
            if decision.pocket != self.pocket:
                continue
            block_entries |= not decision.allow_reentry or decision.reason == "range_cooldown"
            await self._execute_exit(decision, managed_positions, plan, now)
        return block_entries

    async def _execute_exit(
        self,
        decision: ExitDecision,
        managed_positions: Dict[str, Dict],
        plan: PocketPlan,
        now: datetime.datetime,
    ) -> None:
        pocket = decision.pocket
        remaining = abs(decision.units)
        target_side = "long" if decision.units < 0 else "short"
        trades = (managed_positions.get(pocket, {}) or {}).get("open_trades", [])
        trades = [t for t in trades if t.get("side") == target_side]
        try:
            available = sum(abs(int(t.get("units", 0) or 0)) for t in trades)
            remaining = min(remaining, available) if available > 0 else 0
        except Exception:
            pass
        for tr in trades:
            if remaining <= 0:
                break
            trade_units = abs(int(tr.get("units", 0) or 0))
            if trade_units == 0:
                continue
            close_amount = min(remaining, trade_units)
            sign = 1 if target_side == "long" else -1
            trade_id = tr.get("trade_id")
            if not trade_id:
                continue
            ok = await close_trade(trade_id, sign * close_amount)
            if ok:
                LOG.info(
                    "%s exit trade=%s pocket=%s units=%s reason=%s",
                    self.log_prefix,
                    trade_id,
                    pocket,
                    sign * close_amount,
                    decision.reason,
                )
                remaining -= close_amount
        if remaining > 0:
            client_id = build_client_order_id(plan.focus_tag, decision.tag)
            fallback_units = -remaining if decision.units < 0 else remaining
            trade_id = await market_order(
                "USD_JPY",
                fallback_units,
                None,
                None,
                pocket,
                client_order_id=client_id,
                reduce_only=True,
            )
            if trade_id:
                LOG.info(
                    "%s exit-fallback trade=%s pocket=%s units=%s reason=%s",
                    self.log_prefix,
                    trade_id,
                    pocket,
                    fallback_units,
                    decision.reason,
                )
                remaining = 0
            else:
                LOG.error(
                    "%s exit-fallback failed pocket=%s units=%s reason=%s",
                    self.log_prefix,
                    pocket,
                    decision.units,
                    decision.reason,
                )
        if remaining <= 0:
            cooldown_seconds = cooldown_for_pocket(pocket, range_mode=plan.range_active)
            self.stage_tracker.set_cooldown(
                pocket,
                target_side,
                reason=decision.reason,
                seconds=cooldown_seconds,
                now=now,
            )
            if decision.reason == "reverse_signal":
                opposite = "short" if target_side == "long" else "long"
                flip_cd = min(240, max(60, cooldown_seconds // 2))
                self.stage_tracker.set_cooldown(
                    pocket,
                    opposite,
                    reason="flip_guard",
                    seconds=flip_cd,
                    now=now,
                )

    async def _handle_entries(
        self,
        open_positions: Dict[str, Dict],
        managed_positions: Dict[str, Dict],
        plan: PocketPlan,
        now: datetime.datetime,
        block_entries: bool,
    ) -> None:
        if not plan.signals:
            LOG.info("%s no entry signals", self.log_prefix)
            return
        if block_entries:
            LOG.info("%s entries blocked due to recent exit", self.log_prefix)
            return
        if plan.spread_gate_active:
            LOG.info(
                "%s spread gate active (%s, %s)",
                self.log_prefix,
                plan.spread_gate_reason,
                plan.spread_log_context,
            )
            return
        if plan.range_active and self.pocket == "macro":
            LOG.info("%s range mode active, skip macro entry", self.log_prefix)
            return
        if plan.event_soon and self.pocket == "scalp":
            LOG.info("%s event window active, skip scalp entry", self.log_prefix)
            return
        if not can_trade(self.pocket):
            LOG.info("%s pocket disabled by DD guard", self.log_prefix)
            return
        total_lot = plan.lot_allocation or 0.0
        if total_lot <= 0:
            LOG.info("%s no lot allocation (%.4f)", self.log_prefix, total_lot)
            return
        usd_long_cap_lot = float(plan.notes.get("usd_long_cap_lot") or _USD_LONG_CAP_LOT)
        if usd_long_cap_lot < 0:
            usd_long_cap_lot = _USD_LONG_CAP_LOT
        usd_long_cap_units = int(usd_long_cap_lot * 100000)
        net_units = int(plan.notes.get("net_units") or open_positions.get("__net__", {}).get("units", 0) or 0)
        projected_usd_long_units = max(0, net_units)
        executed = False
        for signal in plan.signals:
            if executed:
                break
            action = signal.get("action")
            if action not in {"OPEN_LONG", "OPEN_SHORT"}:
                LOG.info("%s skip non-entry action=%s", self.log_prefix, action)
                continue
            pocket = signal.get("pocket") or self.pocket
            factors = plan.factors_m1 if pocket in {"micro", "scalp"} else plan.factors_h4
            if factors:
                try:
                    close_price = float(factors.get("close"))
                    ema20 = float(factors.get("ema20") or factors.get("ma20"))
                except Exception:
                    close_price = None
                    ema20 = None
                rsi_val = None
                try:
                    rsi_raw = factors.get("rsi")
                    rsi_val = float(rsi_raw) if rsi_raw is not None else None
                except Exception:
                    rsi_val = None
                if close_price is not None and ema20 is not None:
                    trend_gap = close_price - ema20
                    strong_thr = 0.007 if pocket in {"micro", "scalp"} else 0.015
                    soft_thr = 0.003 if pocket in {"micro", "scalp"} else 0.006
                    is_long = action == "OPEN_LONG"
                    counter_ok = False
                    if rsi_val is not None:
                        if (not is_long and rsi_val >= 70) or (is_long and rsi_val <= 30):
                            counter_ok = True
                    if abs(trend_gap) <= soft_thr:
                        counter_ok = True
                    if trend_gap >= strong_thr and not is_long and not counter_ok:
                        LOG.info(
                            "%s skip %s short vs uptrend (close-ema=%.3f rsi=%s)",
                            self.log_prefix,
                            pocket,
                            trend_gap,
                            rsi_val,
                        )
                        continue
                    if trend_gap <= -strong_thr and is_long and not counter_ok:
                        LOG.info(
                            "%s skip %s long vs downtrend (close-ema=%.3f rsi=%s)",
                            self.log_prefix,
                            pocket,
                            trend_gap,
                            rsi_val,
                        )
                        continue
                # ADXが極端に低い（レンジ弱）かつBBW広めならトレンド系を抑制
                if adx_val is not None and adx_val < 10 and bbw_val is not None and bbw_val > 0.20:
                    LOG.info("%s skip %s entry due to low ADX=%.1f bbw=%.2f", self.log_prefix, pocket, adx_val, bbw_val)
                    continue
            # 強制下限: micro は SL/TP と min_hold を底上げして即死・即利確を避ける
            if self.pocket == "micro":
                fac_m1 = plan.factors_m1 or {}
                atr_raw = fac_m1.get("atr_pips") or ((fac_m1.get("atr") or 0.0) * 100)
                try:
                    atr_pips = float(atr_raw)
                except Exception:
                    atr_pips = 0.0
                min_sl = max(5.0, atr_pips * 1.2) if atr_pips > 0 else 5.0
                min_tp = 3.5
                if signal.get("sl_pips") is None or signal["sl_pips"] < min_sl:
                    signal["sl_pips"] = round(min_sl, 2)
                if signal.get("tp_pips") is None or signal["tp_pips"] < min_tp:
                    signal["tp_pips"] = round(min_tp, 2)
                hold = signal.get("min_hold_sec") or signal.get("min_hold_seconds")
                try:
                    hold_val = float(hold) if hold is not None else 0.0
                except Exception:
                    hold_val = 0.0
                if hold_val < 120.0:
                    signal["min_hold_sec"] = 120.0
            confidence = max(0, min(100, signal.get("confidence", 50)))
            confidence_factor = max(0.2, confidence / 100.0)
            confidence_target = round(total_lot * confidence_factor, 3)
            if confidence_target <= 0:
                LOG.info(
                    "%s skip signal=%s reason=confidence_zero target=%.3f",
                    self.log_prefix,
                    signal.get("tag") or signal.get("strategy"),
                    confidence_target,
                )
                continue
            side = "LONG" if action == "OPEN_LONG" else "SHORT"
            try:
                m = float(self.insight.get_multiplier(self.pocket, side))
            except Exception:
                m = 1.0
            if abs(m - 1.0) > 1e-3:
                confidence_target = round(confidence_target * m, 3)
            open_info = open_positions.get(self.pocket, {}) or {}
            price = plan.factors_m1.get("close")
            if action == "OPEN_LONG":
                open_units = int(open_info.get("long_units", 0))
                ref_price = open_info.get("long_avg_price")
                direction = "long"
            else:
                open_units = int(open_info.get("short_units", 0))
                ref_price = open_info.get("short_avg_price")
                direction = "short"
            size_factor = self.stage_tracker.size_multiplier(self.pocket, direction)
            confidence_target = round(confidence_target * size_factor, 3)
            if confidence_target <= 0:
                continue
            blocked, remain_sec, reason = self.stage_tracker.is_blocked(self.pocket, direction, now)
            if blocked:
                LOG.info(
                    "%s cooldown pocket=%s dir=%s remain=%s reason=%s",
                    self.log_prefix,
                    self.pocket,
                    direction,
                    remain_sec,
                    reason or "cooldown",
                )
                continue
            stage_context = dict(open_info)
            if ref_price is None or (ref_price == 0.0 and open_units == 0):
                ref_price = price
            if ref_price is not None:
                stage_context["avg_price"] = ref_price
            staged_lot, stage_idx = compute_stage_lot(
                self.pocket,
                confidence_target,
                open_units,
                action,
                plan.factors_m1,
                plan.factors_h4,
                stage_context,
            )
            if staged_lot <= 0:
                continue
            units = int(round(staged_lot * 100000)) * (1 if action == "OPEN_LONG" else -1)
            if units == 0:
                continue
            min_units_required = max(0, min_units_for_pocket(self.pocket))
            if min_units_required > 0 and 0 < abs(units) < min_units_required:
                clamped_units = min_units_required if units > 0 else -min_units_required
                LOG.info(
                    "%s units clamped to pocket minimum pocket=%s requested=%d -> %d stage=%s",
                    self.log_prefix,
                    self.pocket,
                    units,
                    clamped_units,
                    stage_idx + 1,
                )
                units = clamped_units
                staged_lot = abs(units) / 100000.0
            min_units_required = max(0, min_units_for_pocket(self.pocket))
            if min_units_required > 0 and 0 < abs(units) < min_units_required:
                clamped_units = min_units_required if units > 0 else -min_units_required
                LOG.info(
                    "%s units below pocket minimum pocket=%s requested=%d -> %d stage=%s",
                    self.log_prefix,
                    self.pocket,
                    units,
                    clamped_units,
                    stage_idx + 1,
                )
                units = clamped_units
                staged_lot = abs(units) / 100000.0
            if (
                usd_long_cap_units > 0
                and action == "OPEN_LONG"
                and (projected_usd_long_units + max(0, units)) > usd_long_cap_units
            ):
                LOG.info(
                    "%s USD long cap %.2f lot reached (projected %.2f)",
                    self.log_prefix,
                    usd_long_cap_lot,
                    (projected_usd_long_units + max(0, units)) / 100000.0,
                )
                continue
            sl_pips = signal.get("sl_pips")
            tp_pips = signal.get("tp_pips")
            if sl_pips is None or tp_pips is None:
                LOG.info("%s skip signal=%s reason=missing_sl_tp", self.log_prefix, signal.get("tag"))
                continue
            if price is None:
                LOG.info("%s skip signal=%s reason=no_price", self.log_prefix, signal.get("tag"))
                continue
            LOG.info(
                "%s entry plan strategy=%s pocket=%s action=%s units=%d sl=%.2fp tp=%.2fp price=%.3f conf=%d%% stage=%s",
                self.log_prefix,
                signal.get("strategy") or signal.get("tag"),
                self.pocket,
                action,
                units,
                sl_pips,
                tp_pips,
                price,
                confidence,
                stage_idx + 1,
            )
            sl, tp = clamp_sl_tp(
                price,
                price - sl_pips / 100,
                price + tp_pips / 100,
                action == "OPEN_LONG",
            )
            client_id = build_client_order_id(plan.focus_tag, signal.get("tag", "plan"))
            entry_thesis = {
                "strategy_tag": signal.get("tag"),
                "strategy": signal.get("strategy"),
                "pocket": self.pocket,
                "profile": signal.get("profile"),
                "min_hold_sec": signal.get("min_hold_sec"),
                "loss_guard_pips": signal.get("loss_guard_pips"),
                "target_tp_pips": signal.get("target_tp_pips") or tp_pips,
                "sl_pips": sl_pips,
                "tp_pips": tp_pips,
            }
            trade_id = await market_order(
                "USD_JPY",
                units,
                sl,
                tp,
                self.pocket,
                client_order_id=client_id,
                strategy_tag=signal.get("tag"),
                entry_thesis=entry_thesis,
                meta={"entry_price": price},
            )
            if not trade_id:
                LOG.error("%s order failed strategy=%s", self.log_prefix, signal.get("strategy"))
                continue
            LOG.info(
                "%s order trade=%s strategy=%s lot=%.3f sl=%.3f tp=%.3f conf=%d",
                self.log_prefix,
                trade_id,
                signal.get("strategy"),
                staged_lot,
                sl if sl is not None else -1.0,
                tp if tp is not None else -1.0,
                confidence,
            )
            self.pos_manager.register_open_trade(trade_id, self.pocket, client_id)
            self.stage_tracker.set_stage(self.pocket, direction, stage_idx + 1, now=now)
            entry_cd = POCKET_ENTRY_MIN_INTERVAL.get(self.pocket, 120)
            self.stage_tracker.set_cooldown(
                self.pocket,
                direction,
                reason="entry_rate_limit",
                seconds=entry_cd,
                now=now,
            )
            info = open_positions.setdefault(
                self.pocket,
                {
                    "units": 0,
                    "avg_price": price or 0.0,
                    "trades": 0,
                    "long_units": 0,
                    "long_avg_price": 0.0,
                    "short_units": 0,
                    "short_avg_price": 0.0,
                },
            )
            info["units"] = info.get("units", 0) + units
            info["trades"] = info.get("trades", 0) + 1
            if price is not None:
                info["avg_price"] = price
                if units > 0:
                    prev_units = info.get("long_units", 0)
                    new_units = prev_units + units
                    if new_units > 0:
                        if prev_units == 0:
                            info["long_avg_price"] = price
                        else:
                            info["long_avg_price"] = (
                                info.get("long_avg_price", price) * prev_units + price * units
                            ) / new_units
                    info["long_units"] = new_units
                else:
                    trade_size = abs(units)
                    prev_units = info.get("short_units", 0)
                    new_units = prev_units + trade_size
                    if new_units > 0:
                        if prev_units == 0:
                            info["short_avg_price"] = price
                        else:
                            info["short_avg_price"] = (
                                info.get("short_avg_price", price) * prev_units + price * trade_size
                            ) / new_units
                    info["short_units"] = new_units
            open_positions.setdefault("__net__", {})["units"] = open_positions.get("__net__", {}).get("units", 0) + units
            projected_usd_long_units += max(0, units)
            executed = True
