"""Live wiring helpers for addon/placeholder workers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import asyncio
import datetime as dt
import logging
import os
import time

from execution.order_ids import build_client_order_id
from execution.strategy_entry import cancel_order, limit_order, market_order
from execution.risk_guard import allowed_lot, can_trade, clamp_sl_tp
from indicators.factor_cache import all_factors, get_candles_snapshot
from utils.market_hours import is_market_open
from utils.oanda_account import get_account_snapshot

try:  # optional: tick cache might not be present in minimal runs
    from market_data import tick_window
except Exception:  # pragma: no cover - optional
    tick_window = None  # type: ignore[assignment]

LOG = logging.getLogger(__name__)
PIP_VALUE = 0.01


def _env_bool(key: str, default: Optional[bool]) -> Optional[bool]:
    raw = os.getenv(key)
    if raw is None:
        return default
    val = raw.strip().lower()
    if val in {"1", "true", "yes", "on"}:
        return True
    if val in {"0", "false", "no", "off"}:
        return False
    return default


def _env_float(key: str, default: Optional[float]) -> Optional[float]:
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _env_int(key: str, default: Optional[int]) -> Optional[int]:
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _env_csv(key: str) -> Optional[List[str]]:
    raw = os.getenv(key)
    if not raw:
        return None
    parts = [p.strip() for p in raw.replace(";", ",").split(",") if p.strip()]
    return parts or None


def is_live_enabled(prefix: str) -> bool:
    return bool(_env_bool("ADDON_LIVE_MODE", False) or _env_bool(f"{prefix}_LIVE", False))


def normalize_symbol(symbol: Optional[str]) -> str:
    if not symbol:
        return "USD_JPY"
    sym = symbol.strip().upper().replace("/", "_")
    if sym == "USDJPY":
        sym = "USD_JPY"
    return sym


def normalize_timeframe(tf: Optional[str]) -> str:
    if not tf:
        return "M5"
    raw = tf.strip()
    up = raw.upper()
    if up in {"M1", "M5", "H1", "H4", "D1"}:
        return up
    if up.startswith("M") and up[1:].isdigit():
        candidate = f"M{int(up[1:])}"
        return candidate if candidate in {"M1", "M5"} else "M5"
    if up.endswith("M") and up[:-1].isdigit():
        minutes = int(up[:-1])
        if minutes <= 1:
            return "M1"
        if minutes <= 5:
            return "M5"
        if minutes <= 15:
            return "M5"
        if minutes <= 60:
            return "H1"
        if minutes <= 240:
            return "H4"
        return "D1"
    if up.endswith("H") and up[:-1].isdigit():
        hours = int(up[:-1])
        if hours <= 1:
            return "H1"
        if hours <= 4:
            return "H4"
        return "D1"
    if up.endswith("D") and up[:-1].isdigit():
        return "D1"
    return "M5"


def _parse_ts(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        ts = float(value)
        if ts > 1e12:
            return ts / 1000.0
        if ts > 1e10:
            return ts / 1000.0
        return ts
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            parsed = dt.datetime.fromisoformat(raw)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=dt.timezone.utc)
            else:
                parsed = parsed.astimezone(dt.timezone.utc)
            return parsed.timestamp()
        except Exception:
            return None
    return None


def apply_env_overrides(
    prefix: str,
    base_cfg: Optional[Dict[str, Any]],
    *,
    default_universe: Optional[List[str]] = None,
    default_pocket: Optional[str] = None,
    default_loop: float = 15.0,
    default_exit: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    cfg: Dict[str, Any] = dict(base_cfg or {})

    if default_universe and not cfg.get("universe"):
        cfg["universe"] = list(default_universe)
    env_universe = _env_csv(f"{prefix}_UNIVERSE")
    if env_universe is not None:
        cfg["universe"] = env_universe

    env_pocket = os.getenv(f"{prefix}_POCKET")
    if env_pocket:
        cfg["pocket"] = env_pocket.strip().lower()
    elif default_pocket and not cfg.get("pocket"):
        cfg["pocket"] = default_pocket

    if "loop_interval_sec" not in cfg:
        cfg["loop_interval_sec"] = float(default_loop)
    loop_override = _env_float(f"{prefix}_LOOP_INTERVAL_SEC", None)
    if loop_override is not None:
        cfg["loop_interval_sec"] = loop_override

    for key in ("timeframe", "timeframe_entry", "timeframe_trend"):
        env_val = os.getenv(f"{prefix}_{key.upper()}")
        if env_val:
            cfg[key] = env_val.strip()

    budget_bps = _env_float(f"{prefix}_BUDGET_BPS", None)
    if budget_bps is not None:
        cfg["budget_bps"] = budget_bps
    size_bps = _env_float(f"{prefix}_SIZE_BPS", None)
    if size_bps is not None:
        cfg["size_bps"] = size_bps

    exit_cfg = dict(cfg.get("exit") or {})
    if not exit_cfg and default_exit:
        exit_cfg = dict(default_exit)
    stop_pips = _env_float(f"{prefix}_STOP_PIPS", None)
    if stop_pips is None:
        stop_pips = _env_float(f"{prefix}_SL_PIPS", None)
    if stop_pips is not None:
        exit_cfg["stop_pips"] = float(stop_pips)
    tp_pips = _env_float(f"{prefix}_TP_PIPS", None)
    if tp_pips is not None:
        exit_cfg["tp_pips"] = float(tp_pips)
    stop_atr = _env_float(f"{prefix}_STOP_ATR", None)
    if stop_atr is not None:
        exit_cfg["stop_atr"] = float(stop_atr)
    tp_atr = _env_float(f"{prefix}_TP_ATR", None)
    if tp_atr is not None:
        exit_cfg["tp_atr"] = float(tp_atr)
    if exit_cfg:
        cfg["exit"] = exit_cfg

    ttl_ms = _env_float(f"{prefix}_TTL_MS", None)
    if ttl_ms is not None:
        cfg["ttl_ms"] = ttl_ms
    require_passive = _env_bool(f"{prefix}_REQUIRE_PASSIVE", None)
    if require_passive is not None:
        cfg["require_passive"] = require_passive
    atr_len = _env_int(f"{prefix}_ATR_LEN", None)
    if atr_len is not None:
        cfg["atr_len"] = atr_len
    refresh_sec = _env_float(f"{prefix}_REFRESH_SEC", None)
    if refresh_sec is not None:
        cfg["refresh_sec"] = refresh_sec
    replace_bp = _env_float(f"{prefix}_REPLACE_BP", None)
    if replace_bp is not None:
        cfg["replace_bp"] = replace_bp
    max_quote_age_sec = _env_float(f"{prefix}_MAX_QUOTE_AGE_SEC", None)
    if max_quote_age_sec is not None:
        cfg["max_quote_age_sec"] = max_quote_age_sec

    live_enabled = is_live_enabled(prefix)
    place_orders = _env_bool(f"{prefix}_PLACE_ORDERS", None)
    if place_orders is None:
        place_orders = _env_bool("ADDON_PLACE_ORDERS", None)
    if place_orders is not None:
        cfg["place_orders"] = place_orders
    elif live_enabled:
        cfg["place_orders"] = True
    cfg["live_enabled"] = live_enabled
    return cfg


class LiveDataFeed:
    def __init__(
        self,
        *,
        default_timeframe: str = "M5",
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.default_timeframe = normalize_timeframe(default_timeframe)
        self.log = logger or LOG

    def get_bars(self, symbol: str, tf: str, n: int) -> List[Dict[str, float]]:
        tf_norm = normalize_timeframe(tf or self.default_timeframe)
        limit = int(n) if n and n > 0 else 200
        candles = get_candles_snapshot(tf_norm, limit=limit)
        bars: List[Dict[str, float]] = []
        for row in candles:
            if not isinstance(row, dict):
                continue
            try:
                o = float(row.get("open", row.get("o", 0.0)))
                h = float(row.get("high", row.get("h", 0.0)))
                l = float(row.get("low", row.get("l", 0.0)))
                c = float(row.get("close", row.get("c", 0.0)))
            except (TypeError, ValueError):
                continue
            bar: Dict[str, float] = {"open": o, "high": h, "low": l, "close": c}
            ts = _parse_ts(row.get("timestamp") or row.get("ts") or row.get("time"))
            if ts is not None:
                bar["timestamp"] = float(ts)
            bars.append(bar)
        return bars

    def last(self, symbol: str) -> float:
        if tick_window is not None:
            ticks = tick_window.recent_ticks(seconds=15.0, limit=1)
            if ticks:
                try:
                    return float(ticks[0].get("mid") or 0.0)
                except Exception:
                    pass
        bars = self.get_bars(symbol, self.default_timeframe, 1)
        if bars:
            try:
                return float(bars[-1]["close"])
            except Exception:
                return 0.0
        return 0.0

    def best_bid_ask(self, symbol: str) -> Optional[Tuple[float, float]]:
        if tick_window is None:
            return None
        ticks = tick_window.recent_ticks(seconds=8.0, limit=1)
        if not ticks:
            return None
        bid = ticks[0].get("bid")
        ask = ticks[0].get("ask")
        if bid is None or ask is None:
            return None
        try:
            return float(bid), float(ask)
        except Exception:
            return None


class AddonLiveBroker:
    def __init__(
        self,
        *,
        worker_id: str,
        pocket: str,
        datafeed: LiveDataFeed,
        exit_cfg: Optional[Dict[str, Any]] = None,
        atr_len: int = 14,
        atr_timeframe: Optional[str] = None,
        default_budget_bps: Optional[float] = None,
        default_size_bps: Optional[float] = None,
        ttl_ms: float = 800.0,
        require_passive: bool = True,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.worker_id = worker_id
        self.pocket = pocket
        self.datafeed = datafeed
        self.exit_cfg = dict(exit_cfg or {})
        self.atr_len = max(1, int(atr_len or 14))
        self.atr_timeframe = normalize_timeframe(atr_timeframe or datafeed.default_timeframe)
        self.default_budget_bps = default_budget_bps
        self.default_size_bps = default_size_bps
        self.ttl_ms = float(ttl_ms or 800.0)
        self.require_passive = bool(require_passive)
        self.log = logger or LOG
        self._limit_orders: Dict[str, Dict[str, str]] = {}

    def send(self, order: Dict[str, Any]) -> Optional[str]:
        if not isinstance(order, dict):
            return None

        symbol = normalize_symbol(order.get("symbol"))
        side_raw = str(order.get("side") or "").lower()
        if side_raw in {"buy", "long"}:
            side = "buy"
            side_norm = "long"
        elif side_raw in {"sell", "short"}:
            side = "sell"
            side_norm = "short"
        else:
            self.log.debug("addon_live: skip order (unknown side) order=%s", order)
            return None

        order_type = str(order.get("type") or "market").lower()
        entry_price = order.get("price")
        if entry_price is None:
            entry_price = self.datafeed.last(symbol)
        try:
            entry_price = float(entry_price)
        except (TypeError, ValueError):
            entry_price = 0.0
        if entry_price <= 0:
            self.log.debug("addon_live: skip order (invalid price) order=%s", order)
            return None

        meta = order.get("meta") or order.get("metadata") or {}
        if not isinstance(meta, dict):
            meta = {}
        intent = meta.get("intent")
        if not isinstance(intent, dict):
            intent = {}

        pocket = str(order.get("pocket") or self.pocket or "micro").lower()
        strategy_tag = str(
            meta.get("worker_id")
            or meta.get("strategy")
            or order.get("strategy")
            or self.worker_id
        )
        sl_pips, tp_pips, atr_pips = self._resolve_exits(symbol, intent)
        if sl_pips <= 0.0:
            self.log.info(
                "addon_live: skip order (missing SL) worker=%s symbol=%s",
                self.worker_id,
                symbol,
            )
            return None
        if tp_pips <= 0.0:
            tp_pips = max(1.0, sl_pips * 1.5)

        sl_price, tp_price = self._price_from_pips(entry_price, sl_pips, tp_pips, side == "buy")
        sl_price, tp_price = clamp_sl_tp(price=entry_price, sl=sl_price, tp=tp_price, is_buy=side == "buy")

        risk_pct = self._risk_pct(order)
        snap = get_account_snapshot()
        strategy_tag_raw = order.get("strategy_tag") or order.get("tag") or ""
        strategy_tag = str(strategy_tag_raw).strip() or None
        factors = all_factors()
        fac_m1 = factors.get("M1") or {}
        fac_h4 = factors.get("H4") or {}
        lot = allowed_lot(
            float(snap.nav or 0.0),
            float(sl_pips),
            margin_available=float(snap.margin_available or 0.0),
            price=entry_price,
            margin_rate=float(snap.margin_rate or 0.0),
            risk_pct_override=risk_pct,
            pocket=pocket,
            side=side_norm,
            strategy_tag=strategy_tag,
            fac_m1=fac_m1,
            fac_h4=fac_h4,
        )
        units = int(round(lot * 100000))
        if units <= 0:
            self.log.debug("addon_live: skip order (units=0) worker=%s", self.worker_id)
            return None
        if side == "sell":
            units = -abs(units)

        focus_tag = pocket if pocket != "scalp_fast" else "scalp"
        client_id = build_client_order_id(focus_tag, strategy_tag)
        entry_thesis = {
            "strategy_tag": strategy_tag,
            "entry_price": entry_price,
            "sl_pips": round(sl_pips, 3),
            "tp_pips": round(tp_pips, 3),
            "hard_stop_pips": round(sl_pips, 3),
            "worker_id": self.worker_id,
        }
        order_probability = order.get("entry_probability")
        if isinstance(order_probability, (int, float)):
            entry_thesis["entry_probability"] = max(0.0, min(1.0, float(order_probability)))
        elif isinstance(intent.get("entry_probability"), (int, float)):
            entry_thesis["entry_probability"] = max(0.0, min(1.0, float(intent.get("entry_probability"))))

        entry_units_intent = order.get("entry_units_intent")
        if entry_units_intent is None:
            entry_units_intent = intent.get("entry_units_intent")
        if isinstance(entry_units_intent, (int, float)) and float(entry_units_intent) >= 0:
            entry_thesis["entry_units_intent"] = max(0, int(round(float(entry_units_intent))))
        if atr_pips > 0:
            entry_thesis["atr_pips"] = round(atr_pips, 3)
        if intent:
            entry_thesis["intent"] = intent

        meta_payload = {"worker_id": self.worker_id, "order": order}

        if order_type == "limit":
            current = self.datafeed.best_bid_ask(symbol)
            current_bid = current[0] if current else None
            current_ask = current[1] if current else None
            limit_pocket = pocket if pocket != "scalp_fast" else "scalp"
            try:
                trade_id, order_id = asyncio.run(
                    limit_order(
                        instrument=symbol,
                        units=units,
                        price=float(entry_price),
                        sl_price=sl_price,
                        tp_price=tp_price,
                        pocket=limit_pocket,
                        current_bid=current_bid,
                        current_ask=current_ask,
                        require_passive=self.require_passive,
                        client_order_id=client_id,
                        ttl_ms=self.ttl_ms,
                        entry_thesis=entry_thesis,
                        meta=meta_payload,
                    )
                )
            except RuntimeError:
                trade_id, order_id = None, None
            local_id = order.get("id")
            if local_id and order_id:
                self._limit_orders[str(local_id)] = {
                    "order_id": str(order_id),
                    "client_order_id": client_id,
                    "pocket": limit_pocket,
                }
            return trade_id or order_id

        try:
            ticket = asyncio.run(
                market_order(
                    instrument=symbol,
                    units=units,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    pocket=pocket,
                    client_order_id=client_id,
                    strategy_tag=strategy_tag,
                    entry_thesis=entry_thesis,
                    meta=meta_payload,
                )
            )
        except RuntimeError:
            ticket = None
        return ticket

    def cancel(self, order_id: str) -> Optional[bool]:
        if not order_id:
            return None
        stored = self._limit_orders.get(str(order_id))
        if not stored:
            return None
        try:
            ok = asyncio.run(
                cancel_order(
                    order_id=stored["order_id"],
                    pocket=stored.get("pocket"),
                    client_order_id=stored.get("client_order_id"),
                    reason="addon_cancel",
                )
            )
        except RuntimeError:
            ok = False
        if ok:
            self._limit_orders.pop(str(order_id), None)
        return ok

    def _risk_pct(self, order: Dict[str, Any]) -> float:
        size = order.get("size")
        try:
            size = float(size) if size is not None else 0.0
        except (TypeError, ValueError):
            size = 0.0
        if size <= 0 and self.default_budget_bps is not None:
            size = float(self.default_budget_bps) / 10000.0
        if size <= 0 and self.default_size_bps is not None:
            size = float(self.default_size_bps) / 10000.0
        if size > 1.0:
            if size <= 10000:
                size = size / 10000.0
            else:
                size = size / 100000.0
        return max(0.0005, min(float(size or 0.0), 0.4))

    def _resolve_exits(self, symbol: str, intent: Dict[str, Any]) -> Tuple[float, float, float]:
        sl_pips = 0.0
        tp_pips = 0.0

        for key in ("sl_pips", "stop_pips", "stop_loss_pips"):
            if key in intent:
                try:
                    sl_pips = float(intent[key])
                except (TypeError, ValueError):
                    sl_pips = 0.0
                if sl_pips > 0:
                    break
        for key in ("tp_pips", "take_profit_pips"):
            if key in intent:
                try:
                    tp_pips = float(intent[key])
                except (TypeError, ValueError):
                    tp_pips = 0.0
                if tp_pips > 0:
                    break

        if sl_pips <= 0.0:
            try:
                sl_pips = float(self.exit_cfg.get("stop_pips") or 0.0)
            except Exception:
                sl_pips = 0.0
        if tp_pips <= 0.0:
            try:
                tp_pips = float(self.exit_cfg.get("tp_pips") or 0.0)
            except Exception:
                tp_pips = 0.0

        atr_pips = self._intent_atr_pips(intent)
        if atr_pips <= 0.0:
            atr_pips = self._atr_from_feed(symbol)

        if sl_pips <= 0.0:
            try:
                stop_atr = float(self.exit_cfg.get("stop_atr") or 0.0)
            except Exception:
                stop_atr = 0.0
            if stop_atr > 0.0 and atr_pips > 0.0:
                sl_pips = stop_atr * atr_pips

        if tp_pips <= 0.0:
            try:
                tp_atr = float(self.exit_cfg.get("tp_atr") or 0.0)
            except Exception:
                tp_atr = 0.0
            if tp_atr > 0.0 and atr_pips > 0.0:
                tp_pips = tp_atr * atr_pips

        return float(sl_pips or 0.0), float(tp_pips or 0.0), float(atr_pips or 0.0)

    def _intent_atr_pips(self, intent: Dict[str, Any]) -> float:
        meta = intent.get("meta")
        if isinstance(meta, dict):
            val = meta.get("atr")
            if val:
                try:
                    return float(val) / PIP_VALUE
                except Exception:
                    pass
        for key in ("atr", "atr_pips"):
            if key in intent:
                try:
                    val = float(intent[key])
                    return val if key == "atr_pips" else val / PIP_VALUE
                except Exception:
                    continue
        return 0.0

    def _atr_from_feed(self, symbol: str) -> float:
        bars = self.datafeed.get_bars(symbol, self.atr_timeframe, self.atr_len + 2)
        if len(bars) < self.atr_len + 1:
            return 0.0
        trs = []
        for i in range(1, self.atr_len + 1):
            h = bars[-i]["high"]
            l = bars[-i]["low"]
            prev = bars[-i - 1]["close"]
            trs.append(max(h - l, abs(h - prev), abs(l - prev)))
        if not trs:
            return 0.0
        return (sum(trs) / len(trs)) / PIP_VALUE

    def _price_from_pips(
        self, entry: float, sl_pips: float, tp_pips: float, is_buy: bool
    ) -> Tuple[Optional[float], Optional[float]]:
        if entry <= 0:
            return None, None
        if is_buy:
            sl = round(entry - sl_pips * PIP_VALUE, 3) if sl_pips > 0 else None
            tp = round(entry + tp_pips * PIP_VALUE, 3) if tp_pips > 0 else None
        else:
            sl = round(entry + sl_pips * PIP_VALUE, 3) if sl_pips > 0 else None
            tp = round(entry - tp_pips * PIP_VALUE, 3) if tp_pips > 0 else None
        return sl, tp


def run_loop(
    worker: Any,
    *,
    loop_interval_sec: float,
    pocket: str,
    pass_now: bool = False,
    logger: Optional[logging.Logger] = None,
) -> None:
    log = logger or LOG
    interval = max(1.0, float(loop_interval_sec))
    pocket_val = str(pocket or "micro").lower()
    log.info(
        "addon_live loop start worker=%s interval=%.1fs pocket=%s",
        worker.__class__.__name__,
        interval,
        pocket_val,
    )
    try:
        while True:
            time.sleep(interval)
            if not is_market_open():
                continue
            if not can_trade(pocket_val):
                continue
            try:
                if pass_now:
                    worker.run_once(time.time())
                else:
                    worker.run_once()
            except Exception as exc:  # pragma: no cover - defensive
                log.warning("addon_live worker error: %s", exc, exc_info=True)
    except KeyboardInterrupt:
        log.info("addon_live loop stopped by keyboard")
