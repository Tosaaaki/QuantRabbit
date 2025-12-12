"""Shadow worker for the one-pip maker (S1) strategy."""

from __future__ import annotations

import asyncio
import time
import logging
from dataclasses import dataclass
from typing import Dict, Optional

import json
from pathlib import Path

from analytics import cost_guard
from analysis.range_guard import detect_range_mode
from indicators.factor_cache import all_factors
from market_data import orderbook_state, spread_monitor, tick_window
from workers.common.quality_gate import news_block_active
from execution.order_manager import limit_order, cancel_order
from execution.risk_guard import allowed_lot, can_trade, clamp_sl_tp
from utils.oanda_account import get_account_snapshot
from utils.metrics_logger import log_metric

from . import config

LOG = logging.getLogger(__name__)


@dataclass(slots=True)
class _ShadowEvent:
    epoch_ts: float
    direction: str
    price: float
    imbalance: float
    spread_pips: float
    drift_pips: float
    instant_move_pips: float
    cost_snapshot: Dict[str, float]
    provider: Optional[str]
    latency_ms: Optional[float]
    reason: str = "ready"


def _build_client_id(direction: str) -> str:
    ts_ms = int(time.time() * 1000)
    return f"qr-onepip-s1-{ts_ms}-{direction.lower()[0]}"


def _safe_log_metric(metric: str, value: float, *, tags: Optional[dict] = None) -> None:
    try:
        log_metric(metric, value, tags=tags)
    except Exception as exc:  # pragma: no cover - defensive
        LOG.debug("%s metric drop %s: %s", config.LOG_PREFIX, metric, exc)


async def _cancel_after(order_id: str, ttl_ms: float, pocket: str, client_id: str) -> None:
    await asyncio.sleep(max(ttl_ms, 100.0) / 1000.0)
    success = await cancel_order(order_id=order_id, pocket=pocket, client_order_id=client_id, reason="ttl_cancel")
    if success:
        _safe_log_metric(
            "onepip_maker_ttl_cancel",
            1.0,
            tags={"order_id": order_id, "pocket": pocket},
        )


class _ShadowWriter:
    def __init__(self, path: Path, log_all: bool) -> None:
        self._path = path
        self._log_all = log_all
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: dict) -> None:
        try:
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(event, ensure_ascii=False) + "\n")
        except Exception as exc:  # noqa: BLE001
            LOG.warning("%s shadow log write failed: %s", config.LOG_PREFIX, exc)

    def maybe_write(self, event: dict, *, ready: bool) -> None:
        if ready or self._log_all:
            self.write(event)


def _compute_momentum(seconds: float) -> tuple[float, float]:
    ticks = tick_window.recent_ticks(seconds=seconds, limit=24)
    if len(ticks) < 2:
        return 0.0, 0.0
    first = ticks[0]
    last = ticks[-1]
    drift = (float(last["mid"]) - float(first["mid"])) / config.PIP_VALUE
    prev_mid = float(ticks[-2]["mid"])
    instant = (float(last["mid"]) - prev_mid) / config.PIP_VALUE
    return drift, instant


async def onepip_maker_s1_worker() -> None:
    if not config.ENABLED:
        LOG.info("%s disabled", config.LOG_PREFIX)
        return

    shadow_mode = bool(config.SHADOW_MODE)
    LOG.info("%s worker starting (shadow=%s)", config.LOG_PREFIX, shadow_mode)
    pattern = "logs/oanda/transactions_*.jsonl"
    bootstrap = cost_guard.bootstrap_from_logs(
        pattern,
        max_files=config.COST_BOOTSTRAP_FILES,
        max_lines=config.COST_BOOTSTRAP_LINES,
    )
    if bootstrap:
        LOG.info("%s bootstrapped %d cost samples from logs.", config.LOG_PREFIX, bootstrap)

    shadow_writer = _ShadowWriter(config.SHADOW_LOG_PATH, config.SHADOW_LOG_ALL)
    cooldown_until = 0.0
    last_snapshot_warn = 0.0
    last_news_warn = 0.0
    last_cost_warn = 0.0
    last_spread_warn = 0.0
    session_reset_at = time.monotonic() + config.SESSION_RESET_SEC
    last_bootstrap_check = time.monotonic()
    last_account_refresh = 0.0
    account_snapshot = None
    last_risk_warn = 0.0
    last_depth_warn = 0.0
    last_latency_warn = 0.0

    try:
        while True:
            await asyncio.sleep(config.LOOP_INTERVAL_SEC)
            now_mono = time.monotonic()
            now_epoch = time.time()

            if now_mono >= session_reset_at:
                cooldown_until = 0.0
                session_reset_at = now_mono + config.SESSION_RESET_SEC

            if now_mono < cooldown_until:
                continue

            if now_mono - last_bootstrap_check >= 180.0:
                last_bootstrap_check = now_mono
                age = cost_guard.latest_sample_age_sec()
                threshold = max(config.COST_WINDOW_SEC, 600.0)
                if age is None or age > threshold:
                    loaded = cost_guard.bootstrap_from_logs(
                        pattern,
                        max_files=config.COST_BOOTSTRAP_FILES,
                        max_lines=config.COST_BOOTSTRAP_LINES,
                    )
                    if loaded:
                        LOG.info(
                            "%s refreshed %d cost samples (age %.0fs).",
                            config.LOG_PREFIX,
                            loaded,
                            age or -1.0,
                        )

            news_gate = news_block_active(
                config.NEWS_BLOCK_MINUTES,
                min_impact=config.NEWS_BLOCK_MIN_IMPACT,
            )
            if news_gate:
                if now_mono - last_news_warn > 30.0:
                    LOG.info("%s gated by news window.", config.LOG_PREFIX)
                    last_news_warn = now_mono
                continue

            snapshot = orderbook_state.get_latest(max_age_ms=config.MAX_SNAPSHOT_AGE_MS)
            if snapshot is None:
                if now_mono - last_snapshot_warn > 10.0:
                    LOG.info("%s waiting for orderbook snapshot (age>%.0fms).", config.LOG_PREFIX, config.MAX_SNAPSHOT_AGE_MS)
                    last_snapshot_warn = now_mono
                continue

            if snapshot.latency_ms is not None and snapshot.latency_ms > config.MAX_SNAPSHOT_LATENCY_MS:
                if now_mono - last_latency_warn > 10.0:
                    LOG.debug(
                        "%s snapshot latency %.1fms > limit %.1fms",
                        config.LOG_PREFIX,
                        snapshot.latency_ms,
                        config.MAX_SNAPSHOT_LATENCY_MS,
                    )
                    last_latency_warn = now_mono
                _safe_log_metric(
                    "onepip_maker_skip",
                    1.0,
                    tags={"reason": "latency", "latency_ms": snapshot.latency_ms},
                )
                continue

            if not orderbook_state.has_sufficient_depth(
                snapshot,
                depth=config.DEPTH_LEVELS,
                min_size=config.MIN_TOP_LIQUIDITY,
            ):
                if now_mono - last_depth_warn > 10.0:
                    LOG.debug(
                        "%s insufficient depth depth=%d min_size=%.0f",
                        config.LOG_PREFIX,
                        config.DEPTH_LEVELS,
                        config.MIN_TOP_LIQUIDITY,
                    )
                    last_depth_warn = now_mono
                _safe_log_metric(
                    "onepip_maker_skip",
                    1.0,
                    tags={"reason": "depth"},
                )
                continue

            spread_state = spread_monitor.get_state()
            spread_pips = snapshot.spread / config.PIP_VALUE
            if spread_pips <= 0.0 and spread_state:
                spread_pips = float(spread_state.get("spread_pips") or 0.0)

            if spread_pips > config.MAX_SPREAD_PIPS:
                if now_mono - last_spread_warn > 15.0:
                    LOG.debug(
                        "%s spread %.3fp > limit %.3fp.",
                        config.LOG_PREFIX,
                        spread_pips,
                        config.MAX_SPREAD_PIPS,
                    )
                    last_spread_warn = now_mono
                continue

            # ベースラインスプレッド（移動窓）での適応ゲート
            if spread_state and spread_state.get("baseline_ready"):
                b_avg = spread_state.get("baseline_avg_pips")
                b_p95 = spread_state.get("baseline_p95_pips")
                bad = False
                reason = None
                try:
                    if b_avg is not None and float(b_avg) > config.BASELINE_SPREAD_P50_MAX:
                        bad = True; reason = f"baseline_p50>{config.BASELINE_SPREAD_P50_MAX:.2f}"
                    if b_p95 is not None and float(b_p95) > config.BASELINE_SPREAD_P95_MAX:
                        bad = True; reason = (reason or "") + ",baseline_p95"
                except Exception:
                    bad = False
                if bad:
                    _safe_log_metric("onepip_maker_skip", 1.0, tags={"reason": reason or "baseline_spread"})
                    continue

            imbalance = orderbook_state.queue_imbalance(
                snapshot, depth=config.QUEUE_IMBALANCE_DEPTH
            )
            if imbalance is None or abs(imbalance) < config.QUEUE_IMBALANCE_MIN:
                continue

            if config.RANGE_ONLY:
                factors = all_factors()
                fac_m1 = factors.get("M1")
                fac_h4 = factors.get("H4")
                if not fac_m1 or not fac_h4:
                    continue
                range_ctx = detect_range_mode(fac_m1, fac_h4)
                if not range_ctx.active:
                    continue

            cost_ok, cost_reason = cost_guard.allow_entry(
                config.MAX_COST_PIPS, window_sec=config.COST_WINDOW_SEC
            )
            if not cost_ok:
                if now_mono - last_cost_warn > 25.0:
                    LOG.debug("%s cost guard blocked: %s", config.LOG_PREFIX, cost_reason)
                    last_cost_warn = now_mono
                continue

            cost_snapshot = cost_guard.snapshot(window_sec=config.COST_WINDOW_SEC)
            if cost_snapshot.get("count", 0) < config.MIN_COST_SAMPLES:
                _safe_log_metric(
                    "onepip_maker_skip",
                    1.0,
                    tags={"reason": "cost_samples"},
                )
                continue

            drift_pips, instant_pips = _compute_momentum(config.MICRO_WINDOW_SEC)
            if abs(drift_pips) > config.MICRO_DRIFT_MAX_PIPS:
                continue
            if abs(instant_pips) > config.MICRO_INSTANT_MOVE_MAX_PIPS:
                continue

            direction = "BUY" if imbalance > 0 else "SELL"
            best_bid = snapshot.bid_levels[0].price if snapshot.bid_levels else None
            best_ask = snapshot.ask_levels[0].price if snapshot.ask_levels else None
            price = best_bid if direction == "BUY" else best_ask
            if price is None:
                continue
            price = round(float(price), 3)
            event = _ShadowEvent(
                epoch_ts=now_epoch,
                direction=direction,
                price=float(price),
                imbalance=float(imbalance),
                spread_pips=float(spread_pips),
                drift_pips=float(drift_pips),
                instant_move_pips=float(instant_pips),
                cost_snapshot=cost_snapshot,
                provider=snapshot.provider,
                latency_ms=snapshot.latency_ms,
            )
            shadow_writer.maybe_write(
                {
                    "ts": event.epoch_ts,
                    "direction": event.direction,
                    "imbalance": event.imbalance,
                    "spread_pips": event.spread_pips,
                    "drift_pips": event.drift_pips,
                    "instant_move_pips": event.instant_move_pips,
                    "price": event.price,
                    "cost": event.cost_snapshot,
                    "provider": event.provider,
                    "latency_ms": event.latency_ms,
                    "mode": "shadow" if shadow_mode else "live",
                },
                ready=True,
            )

            if shadow_mode:
                # Impose a short cooldown to avoid flooding when conditions linger.
                cooldown_until = now_mono + config.TTL_MS / 1000.0
                continue

            if not can_trade(config.POCKET):
                if now_mono - last_risk_warn > 20.0:
                    LOG.info("%s pocket risk guard blocked pocket=%s", config.LOG_PREFIX, config.POCKET)
                    last_risk_warn = now_mono
                _safe_log_metric(
                    "onepip_maker_skip",
                    1.0,
                    tags={"reason": "risk_guard", "pocket": config.POCKET},
                )
                continue

            if account_snapshot is None or (now_mono - last_account_refresh) >= config.ACCOUNT_REFRESH_SEC:
                try:
                    account_snapshot = get_account_snapshot()
                    last_account_refresh = now_mono
                    _safe_log_metric(
                        "onepip_maker_account_nav",
                        float(account_snapshot.nav),
                        tags={"pocket": config.POCKET},
                    )
                except Exception as exc:  # noqa: BLE001
                    LOG.warning("%s account snapshot failed: %s", config.LOG_PREFIX, exc)
                    _safe_log_metric(
                        "onepip_maker_account_snapshot_fail",
                        1.0,
                        tags={"error": type(exc).__name__},
                    )
                    shadow_mode = True
                    continue

            equity = float(getattr(account_snapshot, "nav", 0.0) or getattr(account_snapshot, "balance", 0.0))
            if equity <= 0.0:
                _safe_log_metric(
                    "onepip_maker_skip",
                    1.0,
                    tags={"reason": "equity_zero"},
                )
                continue

            # Margin free guard – block entries when free margin is too low
            free_ratio = getattr(account_snapshot, "free_margin_ratio", None)
            try:
                free_ratio_val = float(free_ratio) if free_ratio is not None else None
            except Exception:
                free_ratio_val = None
            if free_ratio_val is not None and free_ratio_val < config.MARGIN_FREE_MIN:
                _safe_log_metric(
                    "onepip_maker_skip",
                    1.0,
                    tags={"reason": "margin_low", "free": free_ratio_val},
                )
                # Apply a cooldown to avoid busy-loop when margin is constrained
                cooldown_until = now_mono + max(2.0, config.COOLDOWN_AFTER_CANCEL_MS / 2000.0)
                continue

            lot_allowed = allowed_lot(
                equity,
                config.SL_PIPS,
                margin_available=getattr(account_snapshot, "margin_available", None),
                price=price,
                margin_rate=getattr(account_snapshot, "margin_rate", None),
                pocket=config.POCKET,
            )
            units_allowed = int(round(lot_allowed * 100000))
            units_allowed = min(units_allowed, config.MAX_UNITS, config.ENTRY_UNITS)
            if units_allowed < config.MIN_UNITS:
                _safe_log_metric(
                    "onepip_maker_skip",
                    1.0,
                    tags={"reason": "units_low", "units": units_allowed},
                )
                continue

            sl_price = tp_price = None
            if config.SL_PIPS > 0.0 and config.TP_PIPS > 0.0:
                # 実コストに合わせて TP を引き上げ（net + MIN_NET_GAIN_PIPS を確保）
                avg_cost = float(cost_snapshot.get("avg_cost_pips") or 0.0)
                tp_eff = max(config.TP_PIPS, avg_cost + config.MIN_NET_GAIN_PIPS)
                tp_eff = min(tp_eff, config.TP_PIPS_MAX)
                if direction == "BUY":
                    sl_candidate = price - config.SL_PIPS * config.PIP_VALUE
                    tp_candidate = price + tp_eff * config.PIP_VALUE
                else:
                    sl_candidate = price + config.SL_PIPS * config.PIP_VALUE
                    tp_candidate = price - tp_eff * config.PIP_VALUE
                sl_price, tp_price = clamp_sl_tp(
                    price,
                    sl_candidate,
                    tp_candidate,
                    direction == "BUY",
                )

            units = units_allowed if direction == "BUY" else -units_allowed
            client_id = _build_client_id(direction)
            trade_id, order_id = await limit_order(
                "USD_JPY",
                units,
                price,
                sl_price,
                tp_price,
                config.POCKET,
                current_bid=best_bid,
                current_ask=best_ask,
                require_passive=True,
                client_order_id=client_id,
                ttl_ms=config.TTL_MS,
                entry_thesis={
                    "imbalance": event.imbalance,
                    "spread": event.spread_pips,
                    "drift": event.drift_pips,
                    "instant": event.instant_move_pips,
                },
            )
            if trade_id:
                shadow_writer.write(
                    {
                        "ts": now_epoch,
                        "mode": "live",
                        "direction": direction,
                        "price": price,
                        "trade_id": trade_id,
                        "client_id": client_id,
                        "result": "filled",
                        "units": units,
                    }
                )
                _safe_log_metric(
                    "onepip_maker_order_submitted",
                    1.0,
                    tags={"result": "filled", "units": abs(units)},
                )
                cooldown_until = now_mono + config.COOLDOWN_AFTER_CANCEL_MS / 1000.0
                continue

            if order_id:
                shadow_writer.write(
                    {
                        "ts": now_epoch,
                        "mode": "live",
                        "direction": direction,
                        "price": price,
                        "order_id": order_id,
                        "client_id": client_id,
                        "result": "submitted",
                        "units": units,
                    }
                )
                _safe_log_metric(
                    "onepip_maker_order_submitted",
                    1.0,
                    tags={"result": "pending", "units": abs(units)},
                )
                if config.TTL_MS > 0.0:
                    asyncio.create_task(
                        _cancel_after(order_id, config.TTL_MS, config.POCKET, client_id)
                    )
                cooldown_until = now_mono + config.COOLDOWN_AFTER_CANCEL_MS / 1000.0
                continue

            LOG.warning("%s limit order submission returned no identifiers; entering shadow mode.", config.LOG_PREFIX)
            _safe_log_metric(
                "onepip_maker_skip",
                1.0,
                tags={"reason": "submit_failed"},
            )
            shadow_mode = True
    except asyncio.CancelledError:  # pragma: no cover
        raise
    except Exception as exc:  # noqa: BLE001
        LOG.exception("%s worker crashed: %s", config.LOG_PREFIX, exc)
        _safe_log_metric(
            "onepip_maker_worker_exit",
            1.0,
            tags={"reason": "unexpected", "error": str(exc)},
        )


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )
    LOG.info("%s worker boot", config.LOG_PREFIX)
    try:
        asyncio.run(onepip_maker_s1_worker())
    except KeyboardInterrupt:  # pragma: no cover
        LOG.info("%s worker stopped by KeyboardInterrupt", config.LOG_PREFIX)
