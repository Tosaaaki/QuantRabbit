"""
execution.order_manager
~~~~~~~~~~~~~~~~~~~~~~~
OANDA REST で成行・指値を発注。
• clientExtensions.tag = "pocket=micro" などを付与
"""

from __future__ import annotations

import asyncio
import time
import json
import logging
import sqlite3
import pathlib
from datetime import datetime, timezone, timedelta
from typing import Literal, Optional, Tuple

from oandapyV20 import API
from oandapyV20.exceptions import V20Error
from oandapyV20.endpoints.orders import OrderCancel, OrderCreate
from oandapyV20.endpoints.trades import TradeCRCDO, TradeClose

from utils.secrets import get_secret
from utils.market_hours import is_market_open

# ---------- 読み込み：env.toml ----------
TOKEN = get_secret("oanda_token")
ACCOUNT = get_secret("oanda_account_id")
try:
    PRACTICE_FLAG = get_secret("oanda_practice").lower() == "true"
except KeyError:
    PRACTICE_FLAG = False  # デフォルトは本番環境

try:
    HEDGING_ENABLED = get_secret("oanda_hedging_enabled").lower() == "true"
except KeyError:
    HEDGING_ENABLED = False

ENVIRONMENT = "practice" if PRACTICE_FLAG else "live"
POSITION_FILL = "OPEN_ONLY" if HEDGING_ENABLED else "DEFAULT"

api = API(access_token=TOKEN, environment=ENVIRONMENT)

if HEDGING_ENABLED:
    logging.info("[ORDER] Hedging mode enabled (positionFill=OPEN_ONLY).")

_LAST_PROTECTIONS: dict[str, Tuple[Optional[float], Optional[float]]] = {}
MACRO_BE_GRACE_SECONDS = 45
_MARGIN_REJECT_UNTIL: dict[str, float] = {}

# ---------- orders logger (logs/orders.db) ----------
_ORDERS_DB_PATH = pathlib.Path("logs/orders.db")


def _ensure_orders_schema() -> sqlite3.Connection:
    _ORDERS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(_ORDERS_DB_PATH)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS orders (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          ts TEXT,
          pocket TEXT,
          instrument TEXT,
          side TEXT,
          units INTEGER,
          sl_price REAL,
          tp_price REAL,
          client_order_id TEXT,
          status TEXT,
          attempt INTEGER,
          ticket_id TEXT,
          executed_price REAL,
          error_code TEXT,
          error_message TEXT,
          request_json TEXT,
          response_json TEXT
        )
        """
    )
    # Useful indexes
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_orders_client ON orders(client_order_id)"
    )
    con.execute("CREATE INDEX IF NOT EXISTS idx_orders_ts ON orders(ts)")
    con.commit()
    return con


def _orders_con() -> sqlite3.Connection:
    # Singleton-ish connection
    global _ORDERS_CON
    try:
        return _ORDERS_CON
    except NameError:
        _ORDERS_CON = _ensure_orders_schema()
        return _ORDERS_CON


def _ensure_utc(candidate: Optional[datetime]) -> datetime:
    if candidate is None:
        return datetime.now(timezone.utc)
    if candidate.tzinfo is None:
        return candidate.replace(tzinfo=timezone.utc)
    return candidate.astimezone(timezone.utc)


def _log_order(
    *,
    pocket: Optional[str],
    instrument: Optional[str],
    side: Optional[str],
    units: Optional[int],
    sl_price: Optional[float],
    tp_price: Optional[float],
    client_order_id: Optional[str],
    status: str,
    attempt: int,
    ticket_id: Optional[str] = None,
    executed_price: Optional[float] = None,
    error_code: Optional[str] = None,
    error_message: Optional[str] = None,
    request_payload: Optional[dict] = None,
    response_payload: Optional[dict] = None,
) -> None:
    try:
        con = _orders_con()
        ts = datetime.now(timezone.utc).isoformat()
        con.execute(
            """
            INSERT INTO orders (
              ts, pocket, instrument, side, units, sl_price, tp_price,
              client_order_id, status, attempt, ticket_id, executed_price,
              error_code, error_message, request_json, response_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts,
                pocket,
                instrument,
                side,
                int(units) if units is not None else None,
                float(sl_price) if sl_price is not None else None,
                float(tp_price) if tp_price is not None else None,
                client_order_id,
                status,
                int(attempt),
                ticket_id,
                float(executed_price) if executed_price is not None else None,
                error_code,
                error_message,
                json.dumps(request_payload, ensure_ascii=False) if request_payload else None,
                json.dumps(response_payload, ensure_ascii=False) if response_payload else None,
            ),
        )
        con.commit()
    except Exception as exc:  # noqa: BLE001
        logging.warning("[ORDER][LOG] failed to persist orders log: %s", exc)


def _estimate_entry_price(
    *, units: int, sl_price: Optional[float], tp_price: Optional[float], meta: Optional[dict]
) -> Optional[float]:
    """Estimate entry price for margin preflight.

    Prefer caller-provided meta['entry_price']. As a fallback, try a naive
    inference from SL/TP if片方のみ与えられている場合は精度が落ちるため None を返す。
    """
    try:
        if meta and isinstance(meta, dict) and meta.get("entry_price"):
            return float(meta["entry_price"])
    except Exception:
        pass
    # Heuristic fallback (avoid guessing if insufficient data)
    if sl_price is not None and tp_price is not None:
        # Mid between SL and TP as a last resort
        return float((sl_price + tp_price) / 2.0)
    return None


def _preflight_units(
    *,
    estimated_price: float,
    requested_units: int,
    margin_buffer: float = 0.92,
) -> Tuple[int, float]:
    """Return (allowed_units, required_margin_estimate).

    allowed_units may be 0 if margin is insufficient even for minimum size.
    """
    try:
        from utils.oanda_account import get_account_snapshot  # lazy import

        snap = get_account_snapshot()
        margin_avail = float(getattr(snap, "margin_available", 0.0) or 0.0)
        margin_rate = float(getattr(snap, "margin_rate", 0.0) or 0.0)
    except Exception:
        # If snapshot fails, do not block the order here.
        return (requested_units, 0.0)

    if margin_rate <= 0.0 or estimated_price <= 0.0:
        return (requested_units, 0.0)

    # Required margin ≈ |units| * price * marginRate (JPY)
    req = abs(requested_units) * estimated_price * margin_rate
    if req * 1.0 <= margin_avail * margin_buffer:
        return (requested_units, req)

    max_units = int((margin_avail * margin_buffer) / (estimated_price * margin_rate))
    if max_units <= 0:
        return (0, req)
    # Keep sign
    allowed = max_units if requested_units > 0 else -max_units
    # Round to nearest 100 units (OANDA accepts 1, but we prefer coarse steps)
    if allowed > 0:
        allowed = int((allowed // 100) * 100)
    else:
        allowed = -int((abs(allowed) // 100) * 100)
    return (allowed, req)


def _is_passive_price(
    *,
    units: int,
    price: float,
    current_bid: Optional[float],
    current_ask: Optional[float],
    min_buffer: float = 0.0001,
) -> bool:
    if units == 0:
        return False
    if units > 0:
        if current_ask is None:
            return False
        return price <= (current_ask - min_buffer)
    if current_bid is None:
        return False
    return price >= (current_bid + min_buffer)
_PARTIAL_STAGE: dict[str, int] = {}

_PARTIAL_THRESHOLDS = {
    # トレンド場面（通常時）の段階利確トリガー（pip）
    # 直近の実測では micro/scalp は数 pip の含み益が頻出する一方、macro は伸びが限定的。
    # 小さめのしきい値で部分利確を優先し、ランナーは trailing に委ねる。
    "macro": (5.0, 10.0),
    "micro": (3.0, 6.0),
    "scalp": (2.0, 4.0),
}
_PARTIAL_THRESHOLDS_RANGE = {
    # レンジ場面ではマクロのみ利確幅をやや引き上げ、慌てた縮小を防ぐ
    "macro": (4.0, 7.0),
    "micro": (2.0, 4.0),
    "scalp": (1.5, 3.0),
}
_PARTIAL_FRACTIONS = (0.4, 0.3)
# micro の平均建玉（~160u）でも段階利確が動作するよう下限を緩和
_PARTIAL_MIN_UNITS = 50
_PARTIAL_RANGE_MACRO_MIN_AGE_MIN = 6.0


def _extract_trade_id(response: dict) -> Optional[str]:
    """
    OANDA の orderFillTransaction から tradeID を抽出する。
    tradeOpened が無い場合でも tradeReduced / tradesClosed を拾って決済扱いにする。
    """
    fill = response.get("orderFillTransaction") or {}

    opened = fill.get("tradeOpened")
    if opened and opened.get("tradeID"):
        return str(opened["tradeID"])

    reduced = fill.get("tradeReduced")
    if reduced and reduced.get("tradeID"):
        return str(reduced["tradeID"])

    for closed in fill.get("tradesClosed") or []:
        trade_id = closed.get("tradeID")
        if trade_id:
            return str(trade_id)

    # tradeReduced が複数のケース（現状 API 仕様では単一辞書）に備えて念のため走査
    for reduced_item in fill.get("tradesReduced") or []:
        trade_id = reduced_item.get("tradeID")
        if trade_id:
            return str(trade_id)

    return None


async def cancel_order(
    *,
    order_id: str,
    pocket: Optional[str] = None,
    client_order_id: Optional[str] = None,
    reason: str = "user_cancel",
) -> bool:
    try:
        endpoint = OrderCancel(accountID=ACCOUNT, orderID=order_id)
        api.request(endpoint)
        _log_order(
            pocket=pocket,
            instrument=None,
            side=None,
            units=None,
            sl_price=None,
            tp_price=None,
            client_order_id=client_order_id,
            status=reason,
            attempt=0,
            request_payload={"order_id": order_id},
            response_payload=endpoint.response,
        )
        logging.info("[ORDER] Cancelled order %s (%s).", order_id, reason)
        return True
    except V20Error as exc:
        logging.warning("[ORDER] Cancel failed for %s: %s", order_id, exc)
    except Exception as exc:  # noqa: BLE001
        logging.warning("[ORDER] Cancel exception for %s: %s", order_id, exc)
    return False


def _parse_trade_open_time(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    candidate = value.strip()
    try:
        if candidate.endswith("Z"):
            candidate = candidate[:-1] + "+00:00"
        if "." in candidate:
            head, frac = candidate.split(".", 1)
            frac_digits = "".join(ch for ch in frac if ch.isdigit())
            if len(frac_digits) > 6:
                frac_digits = frac_digits[:6]
            tz_part = ""
            if "+" in candidate:
                tz_part = candidate[candidate.rfind("+") :]
            if not tz_part:
                tz_part = "+00:00"
            candidate = f"{head}.{frac_digits}{tz_part}"
        elif "+" not in candidate:
            candidate = f"{candidate}+00:00"
        dt = datetime.fromisoformat(candidate)
        return dt.astimezone(timezone.utc)
    except ValueError:
        try:
            trimmed = candidate.split(".", 1)[0]
            if not trimmed.endswith("+00:00"):
                trimmed = trimmed.rstrip("Z") + "+00:00"
            dt = datetime.fromisoformat(trimmed)
            return dt.astimezone(timezone.utc)
        except ValueError:
            return None


def _maybe_update_protections(
    trade_id: str,
    sl_price: Optional[float],
    tp_price: Optional[float],
) -> None:
    if not trade_id or (sl_price is None and tp_price is None):
        return

    data: dict[str, dict[str, str]] = {}
    if sl_price is not None:
        data["stopLoss"] = {
            "price": f"{sl_price:.3f}",
            "timeInForce": "GTC",
        }
    if tp_price is not None:
        data["takeProfit"] = {
            "price": f"{tp_price:.3f}",
            "timeInForce": "GTC",
        }

    if not data:
        return

    previous = _LAST_PROTECTIONS.get(trade_id)
    current = (
        round(sl_price, 3) if sl_price is not None else None,
        round(tp_price, 3) if tp_price is not None else None,
    )
    if previous == current:
        return

    try:
        req = TradeCRCDO(accountID=ACCOUNT, tradeID=trade_id, data=data)
        api.request(req)
        _LAST_PROTECTIONS[trade_id] = current
    except Exception as exc:  # noqa: BLE001
        logging.warning(
            "[ORDER] Failed to update protections trade=%s sl=%s tp=%s: %s",
            trade_id,
            sl_price,
            tp_price,
            exc,
        )


async def close_trade(trade_id: str, units: Optional[int] = None) -> bool:
    if units is None:
        data: Optional[dict[str, str]] = {"units": "ALL"}
    else:
        rounded_units = int(units)
        if rounded_units == 0:
            return True
        # OANDA expects the absolute size; the trade side is derived from trade_id.
        data = {"units": str(abs(rounded_units))}
    req = TradeClose(accountID=ACCOUNT, tradeID=trade_id, data=data)
    try:
        _log_order(
            pocket=None,
            instrument=None,
            side=None,
            units=units,
            sl_price=None,
            tp_price=None,
            client_order_id=None,
            status="close_request",
            attempt=1,
            ticket_id=str(trade_id),
            executed_price=None,
            request_payload={"trade_id": trade_id, "data": data or {}},
        )
        api.request(req)
        _LAST_PROTECTIONS.pop(trade_id, None)
        _PARTIAL_STAGE.pop(trade_id, None)
        _log_order(
            pocket=None,
            instrument=None,
            side=None,
            units=units,
            sl_price=None,
            tp_price=None,
            client_order_id=None,
            status="close_ok",
            attempt=1,
            ticket_id=str(trade_id),
            executed_price=None,
        )
        return True
    except V20Error as exc:
        error_payload = {}
        try:
            error_payload = json.loads(exc.msg or "{}")
        except json.JSONDecodeError:
            error_payload = {"errorMessage": exc.msg}
        error_code = error_payload.get("errorCode")
        logging.warning(
            "[ORDER] TradeClose rejected trade=%s units=%s code=%s",
            trade_id,
            units,
            error_code or exc.code,
        )
        log_error_code = (
            str(error_code) if error_code is not None else str(exc.code)
        )
        _log_order(
            pocket=None,
            instrument=None,
            side=None,
            units=units,
            sl_price=None,
            tp_price=None,
            client_order_id=None,
            status="close_failed",
            attempt=1,
            ticket_id=str(trade_id),
            executed_price=None,
            error_code=log_error_code,
            error_message=error_payload.get("errorMessage") or str(exc),
            response_payload=error_payload if error_payload else None,
        )
        # OANDA returns a few variants for missing/closed trades; treat as idempotent success
        benign_missing = {
            "TRADE_DOES_NOT_EXIST",
            "TRADE_DOESNT_EXIST",
            "NOT_FOUND",
        }
        benign_reduce_only = {"NO_POSITION_TO_REDUCE"}
        if error_code in benign_missing or error_code in benign_reduce_only:
            logging.info(
                "[ORDER] Close benign rejection for trade %s (code=%s) – treating as success.",
                trade_id,
                error_code,
            )
            return True
        return False
    except Exception as exc:  # noqa: BLE001
        logging.warning(
            "[ORDER] Failed to close trade %s units=%s: %s", trade_id, units, exc
        )
        _log_order(
            pocket=None,
            instrument=None,
            side=None,
            units=units,
            sl_price=None,
            tp_price=None,
            client_order_id=None,
            status="close_failed",
            attempt=1,
            ticket_id=str(trade_id),
            executed_price=None,
            error_message=str(exc),
        )
        return False


def update_dynamic_protections(
    open_positions: dict,
    fac_m1: dict,
    fac_h4: dict,
) -> None:
    if not open_positions:
        return
    atr_m1 = fac_m1.get("atr_pips")
    if atr_m1 is None:
        atr_m1 = (fac_m1.get("atr") or 0.0) * 100
    atr_h4 = fac_h4.get("atr_pips")
    if atr_h4 is None:
        atr_h4 = (fac_h4.get("atr") or atr_m1 or 0.0)
    current_price = fac_m1.get("close")
    defaults = {
        "macro": (max(25.0, (atr_h4 or atr_m1 or 0.0) * 1.1), 2.2),
        "micro": (max(8.0, (atr_m1 or 0.0) * 0.9), 1.9),
        # Scalpは原則として建玉時のSL/TPを尊重する。
        # 必要最小限の安全網のみ。mirror-s5 等のスキャル戦略では広げない。
        "scalp": (max(6.0, (atr_m1 or 0.0) * 1.2), 1.0),
    }
    # ポケット別の BE/トレーリング開始閾値（pip）
    # micro/scalp は早めに建値超えへ移行し、利確を積み上げる方針
    per_pocket_triggers = {
        "macro": max(8.0, (atr_h4 or atr_m1 or 0.0) * 1.5),
        "micro": max(3.0, (atr_m1 or 0.0) * 0.8),
        # スキャルは break-even を行わない
        "scalp": float("inf"),
    }
    lock_ratio = 0.6
    per_pocket_min_lock = {"macro": 3.0, "micro": 2.0, "scalp": 0.9}
    pip = 0.01
    now_ts = datetime.now(timezone.utc)
    for pocket, info in open_positions.items():
        if pocket == "__net__":
            continue
        base = defaults.get(pocket)
        if not base:
            continue
        base_sl, tp_ratio = base
        trail_trigger = per_pocket_triggers.get(pocket, per_pocket_triggers["macro"])
        min_lock = per_pocket_min_lock.get(pocket, 3.0)
        trades = info.get("open_trades") or []
        for tr in trades:
            trade_id = tr.get("trade_id")
            price = tr.get("price")
            side = tr.get("side")
            if not trade_id or price is None or not side:
                continue
            # mirror-s5（client_id が 'qr-mirror-s5-' プレフィクス）については
            # ここでの動的SL更新をスキップし、エントリー時のSL/TPを維持する。
            if pocket == "scalp":
                client_id = str(tr.get("client_id") or "")
                if client_id.startswith("qr-mirror-s5-"):
                    continue
            entry = float(price)
            sl_pips = max(1.0, base_sl)
            tp_pips = max(sl_pips * tp_ratio, sl_pips + 5.0) if pocket != "scalp" else None
            opened_at = _parse_trade_open_time(tr.get("open_time"))
            hold_seconds = 0.0
            if opened_at:
                hold_seconds = max(0.0, (now_ts - opened_at).total_seconds())
            gain_pips = 0.0
            if side == "long":
                gain_pips = (current_price or entry) - entry
                gain_pips *= 100
                sl_price = round(entry - sl_pips * pip, 3)
                tp_price = round(entry + tp_pips * pip, 3) if tp_pips is not None else None
                allow_lock = True
                if (
                    pocket == "macro"
                    and hold_seconds < MACRO_BE_GRACE_SECONDS
                ):
                    momentum = (fac_m1.get("close") or 0.0) - (
                        fac_m1.get("ema20") or fac_m1.get("ma20") or 0.0
                    )
                    if momentum >= 0.0:
                        allow_lock = False
                if gain_pips > trail_trigger and allow_lock and pocket != "scalp":
                    lock_pips = max(min_lock, gain_pips * lock_ratio)
                    be_price = entry + lock_pips * pip
                    sl_price = max(sl_price, round(be_price, 3))
                    tp_price = round(entry + max(gain_pips + sl_pips, tp_pips) * pip, 3)
                    if tp_price <= sl_price + 0.002:
                        tp_price = round(sl_price + max(0.004, tp_pips * pip), 3)
                if current_price is not None and sl_price >= current_price:
                    sl_price = round(current_price - 0.003, 3)
            else:
                gain_pips = entry - (current_price or entry)
                gain_pips *= 100
                sl_price = round(entry + sl_pips * pip, 3)
                tp_price = round(entry - tp_pips * pip, 3) if tp_pips is not None else None
                allow_lock = True
                if (
                    pocket == "macro"
                    and hold_seconds < MACRO_BE_GRACE_SECONDS
                ):
                    momentum = (fac_m1.get("close") or 0.0) - (
                        fac_m1.get("ema20") or fac_m1.get("ma20") or 0.0
                    )
                    if momentum <= 0.0:
                        allow_lock = False
                if gain_pips > trail_trigger and allow_lock and pocket != "scalp":
                    lock_pips = max(min_lock, gain_pips * lock_ratio)
                    be_price = entry - lock_pips * pip
                    sl_price = min(sl_price, round(be_price, 3))
                    tp_price = round(entry - max(gain_pips + sl_pips, tp_pips) * pip, 3)
                    if tp_price >= sl_price - 0.002:
                        tp_price = round(sl_price - max(0.004, tp_pips * pip), 3)
                if current_price is not None and sl_price <= current_price:
                    sl_price = round(current_price + 0.003, 3)
            _maybe_update_protections(trade_id, sl_price, tp_price)


async def set_trade_protections(
    trade_id: str,
    *,
    sl_price: Optional[float],
    tp_price: Optional[float],
) -> bool:
    """
    Legacy compatibility layer – update SL/TP for an open trade and report success.
    """
    if not trade_id:
        return False
    try:
        _maybe_update_protections(trade_id, sl_price, tp_price)
        return True
    except Exception as exc:  # noqa: BLE001
        logging.warning(
            "[ORDER] set_trade_protections failed trade=%s sl=%s tp=%s exc=%s",
            trade_id,
            sl_price,
            tp_price,
            exc,
        )
        return False


def _macro_partial_profile(
    fac_h4: Optional[dict],
    range_mode: bool,
) -> tuple[Tuple[float, float], Tuple[float, float]]:
    if range_mode:
        return (2.8, 4.6), (0.75, 0.18)
    if not fac_h4:
        return (3.2, 5.6), (0.78, 0.12)
    adx = float(fac_h4.get("adx", 0.0) or 0.0)
    ma10 = fac_h4.get("ma10", 0.0) or 0.0
    ma20 = fac_h4.get("ma20", 0.0) or 0.0
    atr_raw = fac_h4.get("atr") or 0.0
    atr_pips = atr_raw * 100.0
    gap_pips = abs(ma10 - ma20) * 100.0
    strength_ratio = gap_pips / atr_pips if atr_pips > 1e-6 else 0.0
    if strength_ratio >= 0.9 or adx >= 28.0:
        return (4.0, 6.8), (1.0, 0.0)
    if strength_ratio >= 0.6 or adx >= 24.0:
        return (3.6, 6.2), (0.98, 0.0)
    return (3.2, 5.2), (0.95, 0.0)


def plan_partial_reductions(
    open_positions: dict,
    fac_m1: dict,
    fac_h4: Optional[dict] = None,
    *,
    range_mode: bool = False,
    now: Optional[datetime] = None,
) -> list[tuple[str, str, int]]:
    price = fac_m1.get("close")
    if price is None:
        return []
    pip_scale = 100
    current_time = _ensure_utc(now)
    actions: list[tuple[str, str, int]] = []

    for pocket, info in open_positions.items():
        if pocket == "__net__":
            continue
        thresholds = _PARTIAL_THRESHOLDS.get(pocket)
        fractions = _PARTIAL_FRACTIONS
        if pocket == "macro":
            thresholds, fractions = _macro_partial_profile(fac_h4, range_mode)
        elif range_mode:
            thresholds = _PARTIAL_THRESHOLDS_RANGE.get(pocket, thresholds)
        if not thresholds:
            continue
        trades = info.get("open_trades") or []
        for tr in trades:
            trade_id = tr.get("trade_id")
            side = tr.get("side")
            entry = tr.get("price")
            units = int(tr.get("units", 0) or 0)
            if not trade_id or not side or entry is None or units == 0:
                continue
            opened_at = _parse_trade_open_time(tr.get("open_time"))
            age_minutes = None
            if opened_at:
                age_minutes = max(0.0, (current_time - opened_at).total_seconds() / 60.0)
            if range_mode and pocket == "macro":
                if age_minutes is None or age_minutes < _PARTIAL_RANGE_MACRO_MIN_AGE_MIN:
                    continue
            current_stage = _PARTIAL_STAGE.get(trade_id, 0)
            gain_pips = 0.0
            if side == "long":
                gain_pips = (price - entry) * pip_scale
            else:
                gain_pips = (entry - price) * pip_scale
            if gain_pips <= thresholds[0]:
                continue
            stage = 0
            for idx, threshold in enumerate(thresholds, start=1):
                if gain_pips >= threshold:
                    stage = idx
            if stage <= current_stage:
                continue
            fraction_idx = min(stage - 1, len(fractions) - 1)
            fraction = fractions[fraction_idx]
            reduce_units = int(abs(units) * fraction)
            if reduce_units < _PARTIAL_MIN_UNITS:
                continue
            reduce_units = min(reduce_units, abs(units))
            sign = 1 if units > 0 else -1
            actions.append((pocket, trade_id, sign * reduce_units))
            _PARTIAL_STAGE[trade_id] = stage
    return actions


async def market_order(
    instrument: str,
    units: int,
    sl_price: Optional[float],
    tp_price: Optional[float],
    pocket: Literal["micro", "macro", "scalp"],
    *,
    client_order_id: Optional[str] = None,
    reduce_only: bool = False,
    entry_thesis: Optional[dict] = None,
    meta: Optional[dict] = None,
) -> Optional[str]:
    """
    units : +10000 = buy 0.1 lot, ‑10000 = sell 0.1 lot
    returns order ticket id（決済のみの fill でも tradeID を返却）
    """
    # Pocket-level cooldown after margin rejection
    try:
        if pocket and _MARGIN_REJECT_UNTIL.get(pocket, 0.0) > time.monotonic():
            _log_order(
                pocket=pocket,
                instrument=instrument,
                side="buy" if units > 0 else "sell",
                units=units,
                sl_price=sl_price,
                tp_price=tp_price,
                client_order_id=client_order_id,
                status="margin_cooldown",
                attempt=0,
                request_payload={"cooldown_until": _MARGIN_REJECT_UNTIL.get(pocket)},
            )
            return None
    except Exception:
        pass

    if not is_market_open():
        logging.info(
            "[ORDER] Market closed window. Skip order pocket=%s units=%s client_id=%s",
            pocket,
            units,
            client_order_id,
        )
        attempt_payload: dict = {"reason": "market_closed"}
        if entry_thesis is not None:
            attempt_payload["entry_thesis"] = entry_thesis
        if meta is not None:
            attempt_payload["meta"] = meta
        _log_order(
            pocket=pocket,
            instrument=instrument,
            side="buy" if units > 0 else "sell",
            units=units,
            sl_price=sl_price,
            tp_price=tp_price,
            client_order_id=client_order_id,
            status="market_closed",
            attempt=0,
            request_payload=attempt_payload,
        )
        return None

    # Margin preflight (new entriesのみ)
    preflight_units = units
    if not reduce_only:
        est_price = _estimate_entry_price(
            units=units, sl_price=sl_price, tp_price=tp_price, meta=meta
        )
        if est_price is not None:
            allowed_units, req_margin = _preflight_units(
                estimated_price=est_price, requested_units=units
            )
            if allowed_units == 0:
                _log_order(
                    pocket=pocket,
                    instrument=instrument,
                    side="buy" if units > 0 else "sell",
                    units=units,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    client_order_id=client_order_id,
                    status="insufficient_margin_skip",
                    attempt=0,
                    request_payload={
                        "estimated_price": est_price,
                        "required_margin": req_margin,
                    },
                )
                return None
            if allowed_units != units:
                logging.info(
                    "[ORDER] Preflight scaled units %s -> %s (pocket=%s)",
                    units,
                    allowed_units,
                    pocket,
                )
                preflight_units = allowed_units

    order_data = {
        "order": {
            "type": "MARKET",
            "instrument": instrument,
            "units": str(preflight_units),
            "timeInForce": "FOK",
            "positionFill": "REDUCE_ONLY" if reduce_only else POSITION_FILL,
            "clientExtensions": {"tag": f"pocket={pocket}"},
            "tradeClientExtensions": {"tag": f"pocket={pocket}"},
        }
    }
    if client_order_id:
        order_data["order"]["clientExtensions"]["id"] = client_order_id
        order_data["order"]["tradeClientExtensions"]["id"] = client_order_id
    if sl_price is not None:
        order_data["order"]["stopLossOnFill"] = {"price": f"{sl_price:.3f}"}
    if tp_price is not None:
        order_data["order"]["takeProfitOnFill"] = {"price": f"{tp_price:.3f}"}

    side = "buy" if preflight_units > 0 else "sell"
    units_to_send = preflight_units
    for attempt in range(2):
        payload = order_data.copy()
        payload["order"] = dict(order_data["order"], units=str(units_to_send))
        # Log attempt payload (include non-OANDA context for analytics)
        attempt_payload: dict = {"oanda": payload}
        if entry_thesis is not None:
            attempt_payload["entry_thesis"] = entry_thesis
        if meta is not None:
            attempt_payload["meta"] = meta
        _log_order(
            pocket=pocket,
            instrument=instrument,
            side=side,
            units=units_to_send,
            sl_price=sl_price,
            tp_price=tp_price,
            client_order_id=client_order_id,
            status="submit_attempt",
            attempt=attempt + 1,
            request_payload=attempt_payload,
        )
        r = OrderCreate(accountID=ACCOUNT, data=payload)
        try:
            api.request(r)
            response = r.response
            reject = response.get("orderRejectTransaction") or response.get(
                "orderCancelTransaction"
            )
            if reject:
                reason = reject.get("rejectReason") or reject.get("reason")
                logging.error(
                    "OANDA order rejected (attempt %d) pocket=%s units=%s reason=%s",
                    attempt + 1,
                    pocket,
                    units_to_send,
                    reason,
                )
                logging.error("Reject payload: %s", reject)
                _log_order(
                    pocket=pocket,
                    instrument=instrument,
                    side=side,
                    units=units_to_send,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    client_order_id=client_order_id,
                    status="rejected",
                    attempt=attempt + 1,
                    error_code=reject.get("errorCode"),
                    error_message=reject.get("errorMessage") or reason,
                    response_payload=response,
                )
                if str(reason).upper() == "INSUFFICIENT_MARGIN" and pocket:
                    _MARGIN_REJECT_UNTIL[pocket] = time.monotonic() + 120.0
                if attempt == 0 and abs(units_to_send) >= 2000:
                    units_to_send = int(units_to_send * 0.5)
                    if units_to_send == 0:
                        break
                    logging.info(
                        "Retrying order with reduced units=%s (half).", units_to_send
                    )
                    continue
                return None

            trade_id = _extract_trade_id(response)
            if trade_id:
                fill_type = response.get("orderFillTransaction", {}).get(
                    "tradeOpened"
                )
                if not fill_type:
                    logging.info(
                        "OANDA order filled by adjusting existing trade(s): %s", trade_id
                    )
                # Extract executed_price if present
                executed_price = None
                ofill = response.get("orderFillTransaction") or {}
                if ofill.get("price"):
                    try:
                        executed_price = float(ofill.get("price"))
                    except Exception:
                        executed_price = None
                _log_order(
                    pocket=pocket,
                    instrument=instrument,
                    side=side,
                    units=units_to_send,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    client_order_id=client_order_id,
                    status="filled",
                    attempt=attempt + 1,
                    ticket_id=trade_id,
                    executed_price=executed_price,
                    response_payload=response,
                )
                _maybe_update_protections(trade_id, sl_price, tp_price)
                return trade_id

            logging.error(
                "OANDA order fill lacked trade identifiers (attempt %d): %s",
                attempt + 1,
                response,
            )
            _log_order(
                pocket=pocket,
                instrument=instrument,
                side=side,
                units=units_to_send,
                sl_price=sl_price,
                tp_price=tp_price,
                client_order_id=client_order_id,
                status="fill_no_tradeid",
                attempt=attempt + 1,
                response_payload=response,
            )
            return None
        except V20Error as e:
            logging.error(
                "OANDA API Error (attempt %d) pocket=%s units=%s: %s",
                attempt + 1,
                pocket,
                units_to_send,
                e,
            )
            resp = getattr(e, "response", None)
            if resp:
                logging.error("OANDA response: %s", resp)
            _log_order(
                pocket=pocket,
                instrument=instrument,
                side=side,
                units=units_to_send,
                sl_price=sl_price,
                tp_price=tp_price,
                client_order_id=client_order_id,
                status="api_error",
                attempt=attempt + 1,
                error_message=str(e),
                response_payload=resp if isinstance(resp, dict) else None,
            )

            if attempt == 0 and abs(units_to_send) >= 2000:
                units_to_send = int(units_to_send * 0.5)
                if units_to_send == 0:
                    break
                logging.info(
                    "Retrying order with reduced units=%s (half).", units_to_send
                )
                continue
            return None
        except Exception as exc:
            logging.exception(
                "Unexpected error submitting order (attempt %d): %s",
                attempt + 1,
                exc,
            )
            _log_order(
                pocket=pocket,
                instrument=instrument,
                side=side,
                units=units_to_send,
                sl_price=sl_price,
                tp_price=tp_price,
                client_order_id=client_order_id,
                status="unexpected_error",
                attempt=attempt + 1,
                error_message=str(exc),
            )
            return None
    return None


async def limit_order(
    instrument: str,
    units: int,
    price: float,
    sl_price: Optional[float],
    tp_price: Optional[float],
    pocket: Literal["micro", "macro", "scalp"],
    *,
    current_bid: Optional[float] = None,
    current_ask: Optional[float] = None,
    require_passive: bool = True,
    client_order_id: Optional[str] = None,
    reduce_only: bool = False,
    ttl_ms: float = 800.0,
    entry_thesis: Optional[dict] = None,
    meta: Optional[dict] = None,
) -> tuple[Optional[str], Optional[str]]:
    """Place a passive limit order. Returns (trade_id, order_id)."""

    if units == 0:
        return None, None

    if require_passive and not _is_passive_price(
        units=units,
        price=price,
        current_bid=current_bid,
        current_ask=current_ask,
    ):
        logging.info(
            "[ORDER] Passive guard blocked limit order pocket=%s units=%s price=%.3f bid=%s ask=%s",
            pocket,
            units,
            price,
            f"{current_bid:.3f}" if current_bid is not None else "NA",
            f"{current_ask:.3f}" if current_ask is not None else "NA",
        )
        return None, None

    if not is_market_open():
        _log_order(
            pocket=pocket,
            instrument=instrument,
            side="buy" if units > 0 else "sell",
            units=units,
            sl_price=sl_price,
            tp_price=tp_price,
            client_order_id=client_order_id,
            status="market_closed",
            attempt=0,
            request_payload={"price": price, "reason": "market_closed"},
        )
        return None, None

    ttl_sec = max(0.0, ttl_ms / 1000.0)
    time_in_force = "GTC"
    gtd_time = None
    if ttl_sec >= 1.0:
        expiry = datetime.now(timezone.utc) + timedelta(seconds=ttl_sec)
        gtd_time = expiry.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        time_in_force = "GTD"

    payload = {
        "order": {
            "type": "LIMIT",
            "instrument": instrument,
            "units": str(units),
            "price": f"{price:.3f}",
            "timeInForce": time_in_force,
            "positionFill": "REDUCE_ONLY" if reduce_only else POSITION_FILL,
            "clientExtensions": {"tag": f"pocket={pocket}"},
            "tradeClientExtensions": {"tag": f"pocket={pocket}"},
        }
    }
    if gtd_time:
        payload["order"]["gtdTime"] = gtd_time
    if client_order_id:
        payload["order"]["clientExtensions"]["id"] = client_order_id
        payload["order"]["tradeClientExtensions"]["id"] = client_order_id
    if sl_price is not None:
        payload["order"]["stopLossOnFill"] = {"price": f"{sl_price:.3f}"}
    if tp_price is not None:
        payload["order"]["takeProfitOnFill"] = {"price": f"{tp_price:.3f}"}

    attempt_payload: dict = {"oanda": payload}
    if entry_thesis is not None:
        attempt_payload["entry_thesis"] = entry_thesis
    if meta is not None:
        attempt_payload["meta"] = meta

    _log_order(
        pocket=pocket,
        instrument=instrument,
        side="buy" if units > 0 else "sell",
        units=units,
        sl_price=sl_price,
        tp_price=tp_price,
        client_order_id=client_order_id,
        status="submit_attempt",
        attempt=1,
        request_payload=attempt_payload,
    )

    endpoint = OrderCreate(accountID=ACCOUNT, data=payload)
    try:
        api.request(endpoint)
    except V20Error as exc:
        logging.error("[ORDER] Limit order error: %s", exc)
        _log_order(
            pocket=pocket,
            instrument=instrument,
            side="buy" if units > 0 else "sell",
            units=units,
            sl_price=sl_price,
            tp_price=tp_price,
            client_order_id=client_order_id,
            status="api_error",
            attempt=1,
            error_code=getattr(exc, "code", None),
            error_message=str(exc),
            request_payload=attempt_payload,
        )
        return None, None
    except Exception as exc:  # noqa: BLE001
        logging.exception("[ORDER] Limit order unexpected error: %s", exc)
        _log_order(
            pocket=pocket,
            instrument=instrument,
            side="buy" if units > 0 else "sell",
            units=units,
            sl_price=sl_price,
            tp_price=tp_price,
            client_order_id=client_order_id,
            status="unexpected_error",
            attempt=1,
            error_message=str(exc),
            request_payload=attempt_payload,
        )
        return None, None

    response = endpoint.response
    reject = response.get("orderRejectTransaction")
    if reject:
        reason = reject.get("rejectReason") or reject.get("reason")
        logging.error("[ORDER] Limit order rejected: %s", reason)
        _log_order(
            pocket=pocket,
            instrument=instrument,
            side="buy" if units > 0 else "sell",
            units=units,
            sl_price=sl_price,
            tp_price=tp_price,
            client_order_id=client_order_id,
            status="rejected",
            attempt=1,
            error_code=reject.get("errorCode"),
            error_message=reject.get("errorMessage") or reason,
            response_payload=response,
        )
        return None, None

    trade_id = _extract_trade_id(response)
    executed_price = None
    if response.get("orderFillTransaction") and response["orderFillTransaction"].get(
        "price"
    ):
        try:
            executed_price = float(response["orderFillTransaction"]["price"])
        except Exception:
            executed_price = None

    order_id = None
    create_tx = response.get("orderCreateTransaction")
    if create_tx and create_tx.get("id") is not None:
        order_id = str(create_tx["id"])

    status = "submitted"
    if trade_id:
        status = "filled"

    _log_order(
        pocket=pocket,
        instrument=instrument,
        side="buy" if units > 0 else "sell",
        units=units,
        sl_price=sl_price,
        tp_price=tp_price,
        client_order_id=client_order_id,
        status=status,
        attempt=1,
        ticket_id=trade_id,
        executed_price=executed_price,
        response_payload=response,
    )

    if trade_id:
        return trade_id, order_id

    return None, order_id
