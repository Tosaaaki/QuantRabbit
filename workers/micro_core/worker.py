"""Worker that owns micro-pocket strategy execution."""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import logging
import os
import time
from typing import Any, Dict, List, Optional, Set

from analysis import policy_bus
from analysis.range_guard import detect_range_mode
from analysis.summary_ingestor import get_latest_news
from analytics.realtime_metrics_client import (
    ConfidencePolicy,
    RealtimeMetricsClient,
)
from execution.order_manager import market_order
from execution.position_manager import PositionManager
from execution.risk_guard import allowed_lot, can_trade, update_dd_context
from execution.stage_tracker import StageTracker
from indicators.factor_cache import all_factors
from market_data import spread_monitor
from strategies.mean_reversion.bb_rsi import BBRsi
from strategies.news.spike_reversal import NewsSpikeReversal
from utils.oanda_account import get_account_snapshot
from analysis.perf_monitor import snapshot as get_perf

from . import config

LOG = logging.getLogger(__name__)

POCKET = "micro"
PIP = 0.01
DEFAULT_STRATEGIES = ["BB_RSI", "NewsSpikeReversal"]
POCKET_STRATEGY_MAP = {"micro": {"BB_RSI", "NewsSpikeReversal"}}
BASE_RISK_PCT = 0.02
FALLBACK_EQUITY = 10_000.0
ENTRY_COOLDOWN_SEC = config.COOLDOWN_BASE_SEC
STRATEGY_CLASSES = {
    "BB_RSI": BBRsi,
    "NewsSpikeReversal": NewsSpikeReversal,
}


def _build_client_order_id(strategy: str) -> str:
    ts_ms = int(time.time() * 1000)
    digest = hashlib.sha1(
        f"{ts_ms}-{strategy}-{os.getpid()}".encode("utf-8")
    ).hexdigest()[:9]
    return f"qr-micro-{ts_ms}-{strategy[:4]}-{digest}"


def _ma_bias(factors: Optional[dict], *, threshold: float = 0.002) -> str:
    if not factors:
        return "neutral"
    try:
        ma10 = float(factors.get("ma10"))
        ma20 = float(factors.get("ma20"))
    except (TypeError, ValueError):
        return "neutral"
    diff = ma10 - ma20
    if diff > threshold:
        return "long"
    if diff < -threshold:
        return "short"
    return "neutral"


def _default_micro_policy() -> Dict[str, Any]:
    return {
        "enabled": True,
        "bias": "neutral",
        "confidence": 0.0,
        "units_cap": None,
        "entry_gates": {
            "allow_new": True,
            "require_retest": False,
            "spread_ok": True,
            "event_ok": True,
        },
        "exit_profile": {
            "reverse_threshold": 70,
            "allow_negative_exit": False,
        },
        "be_profile": {
            "enabled": True,
            "trigger_pips": 2.4,
            "cooldown_sec": 45.0,
            "lock_ratio": 0.5,
            "min_lock_pips": 0.35,
        },
        "partial_profile": {
            "thresholds_pips": [1.6, 3.0],
            "fractions": [0.45, 0.30],
            "min_units": 40,
        },
        "strategies": [],
        "pending_orders": [],
    }


async def micro_core_worker() -> None:
    if not config.ENABLED:
        LOG.info("%s disabled by configuration", config.LOG_PREFIX)
        return

    stage_tracker = StageTracker()
    pos_manager = PositionManager()
    metrics_client = RealtimeMetricsClient()
    confidence_policy = ConfidencePolicy()
    last_published_version = 0
    cooldown_rsi: Dict[str, float] = {}

    try:
        while True:
            factors = all_factors()
            fac_m1 = dict(factors.get("M1") or {})
            fac_h4 = factors.get("H4") or {}
            if not fac_m1.get("close"):
                await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                continue

            try:
                blocked, _, spread_state, reason = spread_monitor.is_blocked()
            except Exception:
                blocked = False
                spread_state = None
                reason = ""
            if blocked:
                LOG.info(
                    "%s spread gate active (%s)",
                    config.LOG_PREFIX,
                    reason or "guard",
                )
                await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                continue
            spread_pips = 0.0
            if spread_state and isinstance(spread_state, dict):
                try:
                    spread_pips = float(spread_state.get("spread_pips") or 0.0)
                except (TypeError, ValueError):
                    spread_pips = 0.0
            fac_m1["spread_pips"] = spread_pips

            plan_snapshot = policy_bus.latest()
            strategies_hint: List[str] = []
            if plan_snapshot and getattr(plan_snapshot, "pockets", None):
                raw = plan_snapshot.pockets.get(POCKET)
                if isinstance(raw, dict):
                    hint = raw.get("strategies")
                    if isinstance(hint, list):
                        strategies_hint = [str(s) for s in hint if str(s)]
                last_published_version = plan_snapshot.version

            ranked_strategies = strategies_hint or list(DEFAULT_STRATEGIES)
            news_cache = get_latest_news()
            perf_cache = get_perf()
            range_ctx = detect_range_mode(fac_m1, fac_h4)

            positions = pos_manager.get_open_positions()
            micro_info = positions.get(POCKET) or {}
            open_trades = micro_info.get("open_trades") or []

            signals = _collect_micro_signals(
                ranked_strategies,
                fac_m1,
                news_cache,
                metrics_client,
                confidence_policy,
            )

            if not signals or not can_trade(POCKET):
                await _publish_micro_policy(
                    fac_m1,
                    range_ctx,
                    strategies=ranked_strategies,
                    confidence=0.0,
                    open_trades=open_trades,
                    base_version=last_published_version,
                )
                await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                continue

            try:
                snapshot = get_account_snapshot()
                equity = snapshot.nav or snapshot.balance or FALLBACK_EQUITY
                margin_available = snapshot.margin_available
                margin_rate = snapshot.margin_rate
            except Exception:
                equity = FALLBACK_EQUITY
                margin_available = None
                margin_rate = None

            avg_sl = max(1.0, sum(s["sl_pips"] for s in signals) / len(signals))
            lot_total = allowed_lot(
                equity,
                sl_pips=avg_sl,
                margin_available=margin_available,
                price=fac_m1.get("close"),
                margin_rate=margin_rate,
                risk_pct_override=BASE_RISK_PCT,
            )
            update_dd_context(equity, weight_macro=0.0, scalp_share=0.0)

            executed_ids: Set[str] = set()
            executed_any = False
            now = datetime.datetime.utcnow()
            atr_pips = fac_m1.get("atr_pips")
            if atr_pips is None:
                atr_raw = fac_m1.get("atr")
                atr_pips = (atr_raw or 0.0) * 100.0
            try:
                atr_pips = float(atr_pips or 0.0)
            except (TypeError, ValueError):
                atr_pips = 0.0
            atr_ratio = atr_pips / config.COOLDOWN_ATR_REF_PIPS if config.COOLDOWN_ATR_REF_PIPS else 1.0
            atr_ratio = max(config.COOLDOWN_ATR_MIN_FACTOR, min(config.COOLDOWN_ATR_MAX_FACTOR, atr_ratio or 0.0))
            dynamic_cooldown = max(
                5,
                int(round(config.COOLDOWN_BASE_SEC * atr_ratio)),
            )

            for signal in signals:
                strategy = signal["strategy"]
                confidence = max(0, min(100, signal["confidence"]))
                confidence_factor = max(0.2, confidence / 100.0)
                staged_lot = lot_total * confidence_factor
                units = int(round(staged_lot * 100000))
                if units < 1000:
                    continue

                action = signal["action"]
                direction = "long" if action == "OPEN_LONG" else "short"
                blocked_cd, remain, reason = stage_tracker.is_blocked(
                    POCKET, direction, now
                )
                if blocked_cd:
                    key = f"{POCKET}:{direction}"
                    rsi_val = fac_m1.get("rsi")
                    release = False
                    if (
                        remain
                        and remain >= config.COOLDOWN_RELEASE_MIN_REMAIN_SEC
                        and rsi_val is not None
                        and key in cooldown_rsi
                        and spread_pips <= config.COOLDOWN_RELEASE_MAX_SPREAD
                    ):
                        try:
                            current_rsi = float(rsi_val)
                        except (TypeError, ValueError):
                            current_rsi = None
                        if current_rsi is not None:
                            prev_rsi = cooldown_rsi.get(key)
                            if prev_rsi is not None and abs(current_rsi - prev_rsi) >= config.COOLDOWN_RELEASE_RSI_DELTA:
                                if stage_tracker.clear_cooldown(POCKET, direction):
                                    cooldown_rsi.pop(key, None)
                                    LOG.info(
                                        "%s cooldown cleared due to RSI reversal dir=%s delta=%.2f",
                                        config.LOG_PREFIX,
                                        direction,
                                        current_rsi - prev_rsi,
                                    )
                                    blocked_cd = False
                                else:
                                    LOG.debug("%s cooldown clear requested but no record found", config.LOG_PREFIX)
                    if blocked_cd:
                        LOG.info(
                            "%s cooldown pocket=%s dir=%s remain=%s reason=%s",
                            config.LOG_PREFIX,
                            POCKET,
                            direction,
                            remain,
                            reason or "cooldown",
                        )
                        continue

                if open_trades:
                    LOG.debug(
                        "%s open trades present; skip additional entry",
                        config.LOG_PREFIX,
                    )
                    continue

                client_id = _build_client_order_id(strategy)
                if client_id in executed_ids:
                    continue

                sl = signal["sl_pips"] * PIP
                tp = signal["tp_pips"] * PIP
                price = float(fac_m1.get("close") or 0.0)
                if action == "OPEN_LONG":
                    sl_price = round(price - sl, 3) if signal["sl_pips"] > 0 else None
                    tp_price = round(price + tp, 3) if signal["tp_pips"] > 0 else None
                    units_signed = units
                else:
                    sl_price = round(price + sl, 3) if signal["sl_pips"] > 0 else None
                    tp_price = round(price - tp, 3) if signal["tp_pips"] > 0 else None
                    units_signed = -units

                try:
                    trade_id = await market_order(
                        "USD_JPY",
                        units_signed,
                        sl_price,
                        tp_price,
                        POCKET,
                        client_order_id=client_id,
                        entry_thesis={
                            "strategy_tag": strategy,
                            "reason": signal["tag"],
                        },
                    )
                except Exception as exc:  # noqa: BLE001
                    LOG.error(
                        "%s order error strategy=%s err=%s",
                        config.LOG_PREFIX,
                        strategy,
                        exc,
                    )
                    continue

                if not trade_id:
                    LOG.error(
                        "%s order rejected strategy=%s",
                        config.LOG_PREFIX,
                        strategy,
                    )
                    continue

                LOG.info(
                    "%s entry trade=%s strategy=%s units=%s sl=%s tp=%s",
                    config.LOG_PREFIX,
                    trade_id,
                    strategy,
                    units_signed,
                    f"{sl_price:.3f}" if sl_price is not None else "None",
                    f"{tp_price:.3f}" if tp_price is not None else "None",
                )
                pos_manager.register_open_trade(trade_id, POCKET, client_id)
                stage_tracker.set_stage(POCKET, direction, 1, now=now)
                stage_tracker.set_cooldown(
                    POCKET,
                    direction,
                    reason="entry_rate_limit",
                    seconds=dynamic_cooldown,
                    now=now,
                )
                try:
                    cooldown_rsi[f"{POCKET}:{direction}"] = float(fac_m1.get("rsi") or 50.0)
                except (TypeError, ValueError):
                    cooldown_rsi[f"{POCKET}:{direction}"] = 50.0
                executed_ids.add(client_id)
                executed_any = True
                break

            await _publish_micro_policy(
                fac_m1,
                range_ctx,
                strategies=ranked_strategies,
                confidence=sum(s.get("confidence", 0) for s in signals)
                / max(1, len(signals)),
                open_trades=open_trades,
                executed=executed_any,
                base_version=last_published_version,
            )

            await asyncio.sleep(config.LOOP_INTERVAL_SEC)
    except asyncio.CancelledError:
        LOG.info("%s worker cancelled", config.LOG_PREFIX)
        raise
    finally:
        try:
            stage_tracker.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            pos_manager.close()
        except Exception:  # noqa: BLE001
            pass


def _collect_micro_signals(
    ranked: List[str],
    fac_m1: dict,
    news_cache: dict,
    metrics_client: RealtimeMetricsClient,
    confidence_policy: ConfidencePolicy,
) -> List[dict]:
    signals: List[dict] = []
    news_short = news_cache.get("short", [])

    for sname in ranked:
        cls = STRATEGY_CLASSES.get(sname)
        if not cls:
            continue
        if sname not in POCKET_STRATEGY_MAP["micro"]:
            continue
        if sname == "NewsSpikeReversal":
            raw_signal = cls.check(fac_m1, news_short)
        else:
            raw_signal = cls.check(fac_m1)

        if not raw_signal:
            continue
        action = raw_signal.get("action")
        if action not in {"OPEN_LONG", "OPEN_SHORT"}:
            continue
        signal = {
            "strategy": sname,
            "action": action,
            "sl_pips": float(raw_signal.get("sl_pips") or 0.0),
            "tp_pips": float(raw_signal.get("tp_pips") or 0.0),
            "confidence": int(raw_signal.get("confidence", 50) or 50),
            "tag": raw_signal.get("tag", sname),
        }
        # Apply strategy health scaling
        confidence_policy.reset()
        health = metrics_client.evaluate(sname, POCKET)
        health = confidence_policy.apply(health)
        if not health.allowed:
            continue
        scaled_conf = int(signal["confidence"] * health.confidence_scale)
        signal["confidence"] = max(0, min(100, scaled_conf))
        signals.append(signal)
    return signals


async def _publish_micro_policy(
    fac_m1: dict,
    range_ctx,
    *,
    strategies: List[str],
    confidence: float,
    open_trades: List[dict],
    executed: bool = False,
    base_version: int = 0,
) -> None:
    snapshot = policy_bus.latest()
    if snapshot:
        data = snapshot.to_dict()
    else:
        data = {
            "version": base_version,
            "generated_ts": time.time(),
            "air_score": getattr(range_ctx, "score", 0.0),
            "uncertainty": 0.0,
            "event_lock": False,
            "range_mode": getattr(range_ctx, "active", False),
            "notes": {},
            "pockets": {},
        }

    pockets = data.setdefault("pockets", {})
    micro_entry = pockets.get(POCKET) or _default_micro_policy()
    micro_entry.update(
        {
            "bias": _ma_bias(fac_m1),
            "confidence": max(0.0, min(1.0, confidence / 100.0)),
            "strategies": list(strategies),
            "pending_orders": [],
            "open_trades": len(open_trades),
            "executed": executed,
        }
    )
    pockets[POCKET] = micro_entry
    data["version"] = max(base_version, int(data.get("version", 0))) + 1
    data["generated_ts"] = time.time()
    policy_bus.publish(data)
