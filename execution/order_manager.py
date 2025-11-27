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
import os
from typing import Any, Literal, Optional, Tuple
import requests

from oandapyV20 import API
from oandapyV20.exceptions import V20Error
from oandapyV20.endpoints.orders import OrderCancel, OrderCreate
from oandapyV20.endpoints.trades import TradeCRCDO, TradeClose, TradeDetails

from execution.order_ids import build_client_order_id

from analysis import policy_bus
from utils.secrets import get_secret
from utils.market_hours import is_market_open
from utils.metrics_logger import log_metric
from execution import strategy_guard

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

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _as_float(value: object, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

_DEFAULT_MIN_UNITS = _env_int("ORDER_MIN_UNITS_DEFAULT", 1000)
_MACRO_MIN_UNITS_DEFAULT = max(_DEFAULT_MIN_UNITS * 4, 10000)
_MIN_UNITS_BY_POCKET: dict[str, int] = {
    "micro": _env_int("ORDER_MIN_UNITS_MICRO", _DEFAULT_MIN_UNITS),
    "macro": _env_int("ORDER_MIN_UNITS_MACRO", _MACRO_MIN_UNITS_DEFAULT),
    "scalp": _env_int("ORDER_MIN_UNITS_SCALP", _DEFAULT_MIN_UNITS),
}
# If true, do not attach stopLossOnFill (TP is still sent). Honors either
# ORDER_DISABLE_STOP_LOSS or DISABLE_STOP_LOSS env var.
_DISABLE_STOP_LOSS = (
    os.getenv("ORDER_DISABLE_STOP_LOSS")
    or os.getenv("DISABLE_STOP_LOSS")
    or "true"
).lower() in {"1", "true", "yes", "on"}


def min_units_for_pocket(pocket: Optional[str]) -> int:
    if not pocket:
        return _DEFAULT_MIN_UNITS
    return int(_MIN_UNITS_BY_POCKET.get(pocket, _DEFAULT_MIN_UNITS))


_DYNAMIC_SL_ENABLE = os.getenv("ORDER_DYNAMIC_SL_ENABLE", "true").lower() in {
    "1",
    "true",
    "yes",
}
_DYNAMIC_SL_POCKETS = {
    token.strip().lower()
    for token in os.getenv("ORDER_DYNAMIC_SL_POCKETS", "micro,macro").split(",")
    if token.strip()
}
_DYNAMIC_SL_RATIO = float(os.getenv("ORDER_DYNAMIC_SL_RATIO", "1.2"))
_DYNAMIC_SL_MAX_PIPS = float(os.getenv("ORDER_DYNAMIC_SL_MAX_PIPS", "8.0"))

_LAST_PROTECTIONS: dict[str, dict[str, float | None]] = {}
MACRO_BE_GRACE_SECONDS = 45
_MARGIN_REJECT_UNTIL: dict[str, float] = {}
_PROTECTION_MIN_BUFFER = max(0.0005, float(os.getenv("ORDER_PROTECTION_MIN_BUFFER", "0.003")))
_PROTECTION_MIN_SEPARATION = max(0.001, float(os.getenv("ORDER_PROTECTION_MIN_SEPARATION", "0.006")))
_PROTECTION_FALLBACK_PIPS = max(0.02, float(os.getenv("ORDER_PROTECTION_FALLBACK_PIPS", "0.12")))
_PROTECTION_RETRY_REASONS = {
    "STOP_LOSS_ON_FILL_LOSS",
    "STOP_LOSS_ON_FILL_INVALID",
    "STOP_LOSS_LOSS",
    "TAKE_PROFIT_ON_FILL_LOSS",
}
_PARTIAL_CLOSE_RETRY_CODES = {
    "CLOSE_TRADE_UNITS_EXCEED_TRADE_SIZE",
    "POSITION_TO_REDUCE_TOO_SMALL",
}
_ORDER_SPREAD_BLOCK_PIPS = float(os.getenv("ORDER_SPREAD_BLOCK_PIPS", "1.6"))

# Max units per new entry order (reduce_only orders are exempt)
try:
    _MAX_ORDER_UNITS = int(os.getenv("MAX_ORDER_UNITS", "20000"))
except Exception:
    _MAX_ORDER_UNITS = 20000
# Hard safety cap even after dynamic boosts
try:
    _MAX_ORDER_UNITS_HARD = int(os.getenv("MAX_ORDER_UNITS_HARD", "40000"))
except Exception:
    _MAX_ORDER_UNITS_HARD = 40000

# 最低発注単位（AGENT.me 6.1 に準拠）。
# リスク計算・ステージ係数適用後の“最終 units”に対して適用する最終ゲート。
# reduce_only（決済）では適用しない。
_MIN_ORDER_UNITS = 1000

# ---------- orders logger (logs/orders.db) ----------
_ORDERS_DB_PATH = pathlib.Path("logs/orders.db")

_DEFAULT_MIN_HOLD_SEC = {
    "macro": 360.0,
    "micro": 150.0,
    "scalp": 75.0,
}


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
          stage_index INTEGER,
          ticket_id TEXT,
          executed_price REAL,
          error_code TEXT,
          error_message TEXT,
          request_json TEXT,
          response_json TEXT
        )
        """
    )
    cols = {row[1] for row in con.execute("PRAGMA table_info(orders)")}
    if "stage_index" not in cols:
        con.execute("ALTER TABLE orders ADD COLUMN stage_index INTEGER")
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


def _fetch_quote(instrument: str) -> dict[str, float] | None:
    """Fetch a single pricing snapshot to derive bid/ask/spread."""
    try:
        url = f"https://api-fxtrade.oanda.com/v3/accounts/{ACCOUNT}/pricing"
        headers = {"Authorization": f"Bearer {TOKEN}"}
        resp = requests.get(
            url,
            params={"instruments": instrument},
            headers=headers,
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
        price = (data.get("prices") or [{}])[0]
        bid = _as_float(price.get("bids", [{}])[0].get("price"))
        ask = _as_float(price.get("asks", [{}])[0].get("price"))
        if bid is None or ask is None:
            return None
        spread_pips = (ask - bid) / 0.01
        return {
            "bid": bid,
            "ask": ask,
            "mid": (bid + ask) / 2.0,
            "spread_pips": spread_pips,
            "ts": price.get("time") or datetime.now(timezone.utc).isoformat(),
        }
    except Exception:
        return None


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
    stage_index: Optional[int] = None,
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
              client_order_id, status, attempt, stage_index, ticket_id, executed_price,
              error_code, error_message, request_json, response_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                int(stage_index) if stage_index is not None else None,
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


def _console_order_log(
    event: str,
    *,
    pocket: Optional[str],
    strategy_tag: Optional[str],
    side: Optional[str],
    units: Optional[int],
    sl_price: Optional[float],
    tp_price: Optional[float],
    client_order_id: Optional[str],
    ticket_id: Optional[str] = None,
    note: Optional[str] = None,
) -> None:
    def _fmt(value: Optional[float]) -> str:
        if value is None:
            return "-"
        try:
            return f"{float(value):.3f}"
        except (TypeError, ValueError):
            return str(value)

    logging.info(
        "[ORDER][%s] pocket=%s strategy=%s side=%s units=%s sl=%s tp=%s client=%s ticket=%s note=%s",
        event,
        pocket or "-",
        strategy_tag or "-",
        side or "-",
        units or "-",
        _fmt(sl_price),
        _fmt(tp_price),
        client_order_id or "-",
        ticket_id or "-",
        note or "-",
    )


def _normalize_protections(
    estimated_price: Optional[float],
    sl_price: Optional[float],
    tp_price: Optional[float],
    is_buy: bool,
) -> tuple[Optional[float], Optional[float], bool]:
    """Ensure SL/TP are on the correct side of the entry with a minimal buffer."""

    if estimated_price is None:
        return sl_price, tp_price, False
    changed = False
    price = float(estimated_price)
    buffer = max(_PROTECTION_MIN_BUFFER, 0.0005)
    separation = max(_PROTECTION_MIN_SEPARATION, buffer * 2)
    if sl_price is not None:
        if is_buy and sl_price >= price - buffer:
            sl_price = round(price - buffer, 3)
            changed = True
        elif (not is_buy) and sl_price <= price + buffer:
            sl_price = round(price + buffer, 3)
            changed = True
    if tp_price is not None:
        if is_buy and tp_price <= price + buffer:
            tp_price = round(price + buffer, 3)
            changed = True
        elif (not is_buy) and tp_price >= price - buffer:
            tp_price = round(price - buffer, 3)
            changed = True
    if sl_price is not None and tp_price is not None:
        gap = tp_price - sl_price if is_buy else sl_price - tp_price
        if gap < separation:
            sl_delta = separation / 2.0
            if is_buy:
                sl_price = round(price - sl_delta, 3)
                tp_price = round(price + sl_delta, 3)
            else:
                sl_price = round(price + sl_delta, 3)
                tp_price = round(price - sl_delta, 3)
            changed = True
    return sl_price, tp_price, changed


def _fallback_protections(
    baseline_price: Optional[float],
    *,
    is_buy: bool,
    has_sl: bool,
    has_tp: bool,
    sl_gap_pips: Optional[float],
    tp_gap_pips: Optional[float],
) -> tuple[Optional[float], Optional[float]]:
    """Return wider SL/TP values used after STOP_LOSS_ON_FILL rejects."""

    if baseline_price is None:
        return None, None
    gap_sl = _PROTECTION_FALLBACK_PIPS
    gap_tp = _PROTECTION_FALLBACK_PIPS
    try:
        if sl_gap_pips is not None:
            gap_sl = max(gap_sl, float(sl_gap_pips) * 0.01)
    except (TypeError, ValueError):
        pass
    try:
        if tp_gap_pips is not None:
            gap_tp = max(gap_tp, float(tp_gap_pips) * 0.01)
    except (TypeError, ValueError):
        pass
    if is_buy:
        sl_price = round(baseline_price - gap_sl, 3) if has_sl else None
        tp_price = round(baseline_price + gap_tp, 3) if has_tp else None
    else:
        sl_price = round(baseline_price + gap_sl, 3) if has_sl else None
        tp_price = round(baseline_price - gap_tp, 3) if has_tp else None
    return sl_price, tp_price


def _derive_fallback_basis(
    estimated_price: Optional[float],
    sl_price: Optional[float],
    tp_price: Optional[float],
    is_buy: bool,
) -> Optional[float]:
    if estimated_price is not None:
        return estimated_price
    if sl_price is not None and tp_price is not None:
        return float((sl_price + tp_price) / 2.0)
    gap = _PROTECTION_FALLBACK_PIPS
    if sl_price is not None:
        return float(sl_price + (gap if is_buy else -gap))
    if tp_price is not None:
        return float(tp_price - (gap if is_buy else -gap))
    return None


def _current_trade_units(trade_id: str) -> Optional[int]:
    try:
        req = TradeDetails(accountID=ACCOUNT, tradeID=trade_id)
        api.request(req)
        trade = req.response.get("trade") or {}
        units_raw = trade.get("currentUnits")
        if units_raw is None:
            return None
        return abs(int(float(units_raw)))
    except Exception as exc:  # noqa: BLE001
        logging.warning("[ORDER] Failed to fetch trade units trade=%s err=%s", trade_id, exc)
        return None


def _retry_close_with_actual_units(
    trade_id: str,
    requested_units: Optional[int],
) -> bool:
    actual_units = _current_trade_units(trade_id)
    if actual_units is None or actual_units <= 0:
        logging.info(
            "[ORDER] Trade %s already flat or missing when retrying close; treat as success.",
            trade_id,
        )
        return True
    target_units = actual_units
    if requested_units is not None:
        try:
            target_units = min(actual_units, abs(int(requested_units)))
        except Exception:
            target_units = actual_units
    if target_units <= 0:
        return True
    data = {"units": str(target_units)}
    try:
        req = TradeClose(accountID=ACCOUNT, tradeID=trade_id, data=data)
        api.request(req)
        _LAST_PROTECTIONS.pop(trade_id, None)
        _PARTIAL_STAGE.pop(trade_id, None)
        _log_order(
            pocket=None,
            instrument=None,
            side=None,
            units=target_units,
            sl_price=None,
            tp_price=None,
            client_order_id=None,
            status="close_retry_ok",
            attempt=2,
            ticket_id=str(trade_id),
            executed_price=None,
            request_payload={"retry": True, "data": data},
        )
        _console_order_log(
            "CLOSE_OK",
            pocket=None,
            strategy_tag=None,
            side=None,
            units=target_units,
            sl_price=None,
            tp_price=None,
            client_order_id=None,
            ticket_id=str(trade_id),
            note="retry",
        )
        return True
    except Exception as exc:  # noqa: BLE001
        logging.warning(
            "[ORDER] Retry close failed trade=%s units=%s err=%s",
            trade_id,
            target_units,
            exc,
        )
        return False

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
    # 超短期（fast scalp）は利幅が小さいため閾値も縮小
    # まずは小刻みにヘッジしてランナーのみを残す方針
    "scalp_fast": (1.2, 2.0),
}
_PARTIAL_THRESHOLDS_RANGE = {
    # AGENT.me 仕様（3.5.1）に合わせ、レンジ時は段階利確を引き延ばしすぎず早期ヘッジ。
    # macro 16/22, micro 10/16, scalp 6/10 pips
    "macro": (16.0, 22.0),
    "micro": (10.0, 16.0),
    "scalp": (6.0, 10.0),
    # fast scalp はさらに近い利確で早めに在庫を薄くする
    "scalp_fast": (0.9, 1.4),
}
_PARTIAL_FRACTIONS = (0.4, 0.3)
# micro の平均建玉（~160u）でも段階利確が動作するよう下限を緩和
_PARTIAL_MIN_UNITS = 20
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


def _coerce_entry_thesis(meta: Any) -> dict:
    if isinstance(meta, dict):
        return meta
    if isinstance(meta, str):
        try:
            parsed = json.loads(meta)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {}
    return {}


def _trade_min_hold_seconds(trade: dict, pocket: str) -> float:
    thesis = _coerce_entry_thesis(trade.get("entry_thesis"))
    hold = thesis.get("min_hold_sec") or thesis.get("min_hold_seconds")
    try:
        hold_val = float(hold)
    except (TypeError, ValueError):
        hold_val = _DEFAULT_MIN_HOLD_SEC.get(pocket, 60.0)
    if hold_val <= 0.0:
        return _DEFAULT_MIN_HOLD_SEC.get(pocket, 60.0)
    return hold_val


def _trade_age_seconds(trade: dict, now: datetime) -> Optional[float]:
    opened_at = _parse_trade_open_time(trade.get("open_time"))
    if not opened_at:
        return None
    delta = (now - opened_at).total_seconds()
    return max(0.0, delta)


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

    previous = _LAST_PROTECTIONS.get(trade_id) or {}
    prev_sl = previous.get("sl")
    prev_tp = previous.get("tp")
    current_sl = round(sl_price, 3) if sl_price is not None else None
    current_tp = round(tp_price, 3) if tp_price is not None else None
    if prev_sl == current_sl and prev_tp == current_tp:
        return

    try:
        req = TradeCRCDO(accountID=ACCOUNT, tradeID=trade_id, data=data)
        api.request(req)
        _LAST_PROTECTIONS[trade_id] = {
            "sl": current_sl,
            "tp": current_tp,
            "ts": time.time(),
        }
    except Exception as exc:  # noqa: BLE001
        logging.warning(
            "[ORDER] Failed to update protections trade=%s sl=%s tp=%s: %s",
            trade_id,
            sl_price,
            tp_price,
            exc,
        )


async def set_trade_protections(
    trade_id: str,
    *,
    sl_price: Optional[float],
    tp_price: Optional[float],
) -> bool:
    if not trade_id or (sl_price is None and tp_price is None):
        return False
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
        return False
    try:
        req = TradeCRCDO(accountID=ACCOUNT, tradeID=trade_id, data=data)
        api.request(req)
        _LAST_PROTECTIONS[trade_id] = (
            round(sl_price, 3) if sl_price is not None else None,
            round(tp_price, 3) if tp_price is not None else None,
        )
        _log_order(
            pocket=None,
            instrument=None,
            side=None,
            units=None,
            sl_price=sl_price,
            tp_price=tp_price,
            client_order_id=None,
            status="protection_update",
            attempt=1,
            stage_index=None,
            ticket_id=trade_id,
            executed_price=None,
            request_payload={"trade_id": trade_id, "data": data},
        )
        return True
    except Exception as exc:  # noqa: BLE001
        logging.warning(
            "[ORDER] protection update failed trade=%s sl=%s tp=%s: %s",
            trade_id,
            sl_price,
            tp_price,
            exc,
        )
        _log_order(
            pocket=None,
            instrument=None,
            side=None,
            units=None,
            sl_price=sl_price,
            tp_price=tp_price,
            client_order_id=None,
            status="protection_update_failed",
            attempt=1,
            stage_index=None,
            ticket_id=trade_id,
            executed_price=None,
            error_message=str(exc),
            request_payload={"trade_id": trade_id, "data": data},
        )
        return False


async def close_trade(trade_id: str, units: Optional[int] = None) -> bool:
    data: Optional[dict[str, str]] = None
    if units is None:
        data = {"units": "ALL"}
    else:
        rounded_units = int(units)
        if rounded_units == 0:
            return True
        target_units = abs(rounded_units)
        actual_units = _current_trade_units(trade_id)
        if actual_units is not None:
            if actual_units <= 0:
                logging.info("[ORDER] Close skipped trade=%s already flat.", trade_id)
                return True
            if target_units > actual_units:
                logging.info(
                    "[ORDER] Clamping close units trade=%s requested=%s available=%s",
                    trade_id,
                    target_units,
                    actual_units,
                )
                target_units = actual_units
        # OANDA expects the absolute size; the trade side is derived from trade_id.
        data = {"units": str(target_units)}
    req = TradeClose(accountID=ACCOUNT, tradeID=trade_id, data=data)
    _console_order_log(
        "CLOSE_REQ",
        pocket=None,
        strategy_tag=None,
        side=None,
        units=units,
        sl_price=None,
        tp_price=None,
        client_order_id=None,
        ticket_id=str(trade_id),
    )
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
        _console_order_log(
            "CLOSE_OK",
            pocket=None,
            strategy_tag=None,
            side=None,
            units=units,
            sl_price=None,
            tp_price=None,
            client_order_id=None,
            ticket_id=str(trade_id),
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
        log_error_code = str(error_code) if error_code is not None else str(exc.code)
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
        _console_order_log(
            "CLOSE_FAIL",
            pocket=None,
            strategy_tag=None,
            side=None,
            units=units,
            sl_price=None,
            tp_price=None,
            client_order_id=None,
            ticket_id=str(trade_id),
            note=f"code={log_error_code}",
        )
        if (log_error_code or "").upper() in _PARTIAL_CLOSE_RETRY_CODES:
            if _retry_close_with_actual_units(trade_id, units):
                return True
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
        _console_order_log(
            "CLOSE_FAIL",
            pocket=None,
            strategy_tag=None,
            side=None,
            units=units,
            sl_price=None,
            tp_price=None,
            client_order_id=None,
            ticket_id=str(trade_id),
            note="exception",
        )
        return False


def update_dynamic_protections(
    open_positions: dict,
    fac_m1: dict,
    fac_h4: dict,
) -> None:
    if _DISABLE_STOP_LOSS:
        return
    if not open_positions:
        return
    _apply_dynamic_protections_v2(open_positions, fac_m1, fac_h4)
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
        # スキャルも一定の含み益で建値超えに移行（戻り負けの抑制）
        "scalp": max(2.4, (atr_m1 or 0.0) * 1.2),
    }
    lock_ratio = 0.6
    per_pocket_min_lock = {"macro": 3.0, "micro": 2.0, "scalp": 0.6}
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
            client_id = str(tr.get("client_id") or "")
            # Safety: skip manual/unknown trades unless explicitly managed by the bot
            if not client_id.startswith("qr-"):
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


def _apply_dynamic_protections_v2(
    open_positions: dict,
    fac_m1: dict,
    fac_h4: dict,
) -> None:
    policy = policy_bus.latest()
    pockets_policy = policy.pockets if policy else {}
    now_ts = time.time()
    current_price = fac_m1.get("close")
    pip = 0.01

    def _coerce(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    defaults = {
        "macro": {"trigger": 6.8, "lock_ratio": 0.55, "min_lock": 2.6, "cooldown": 90.0},
        "micro": {"trigger": 2.2, "lock_ratio": 0.48, "min_lock": 0.45, "cooldown": 60.0},
        "scalp": {"trigger": 1.3, "lock_ratio": 0.45, "min_lock": 0.20, "cooldown": 20.0},
    }

    for pocket, info in open_positions.items():
        if pocket == "__net__":
            continue
        trades = info.get("open_trades") or []
        if not trades:
            continue

        plan = pockets_policy.get(pocket) if isinstance(pockets_policy, dict) else {}
        be_profile = plan.get("be_profile", {}) if isinstance(plan, dict) else {}
        default_cfg = defaults.get(pocket, defaults["macro"])
        trigger = _coerce(be_profile.get("trigger_pips"), default_cfg["trigger"])
        lock_ratio = max(0.0, min(1.0, _coerce(be_profile.get("lock_ratio"), default_cfg["lock_ratio"])))
        min_lock = max(0.0, _coerce(be_profile.get("min_lock_pips"), default_cfg["min_lock"]))
        cooldown_sec = max(0.0, _coerce(be_profile.get("cooldown_sec"), default_cfg["cooldown"]))

        for tr in trades:
            trade_id = tr.get("trade_id")
            price = tr.get("price")
            side = tr.get("side")
            if not trade_id or price is None or not side:
                continue

            client_id = str(tr.get("client_id") or "")
            if not client_id.startswith("qr-"):
                continue
            if pocket == "scalp" and client_id.startswith("qr-mirror-s5-"):
                continue

            entry = float(price)
            trade_tp_info = tr.get("take_profit") or {}
            try:
                trade_tp = float(trade_tp_info.get("price"))
            except (TypeError, ValueError):
                trade_tp = None
            trade_sl_info = tr.get("stop_loss") or {}
            try:
                trade_sl = float(trade_sl_info.get("price"))
            except (TypeError, ValueError):
                trade_sl = None

            if side == "long":
                gain_pips = ((current_price or entry) - entry) * 100.0
            else:
                gain_pips = (entry - (current_price or entry)) * 100.0

            if gain_pips < trigger:
                continue

            last_record = _LAST_PROTECTIONS.get(trade_id) or {}
            last_ts = float(last_record.get("ts") or 0.0)
            if cooldown_sec > 0.0 and (now_ts - last_ts) < cooldown_sec:
                continue

            lock_pips = max(min_lock, gain_pips * lock_ratio)
            if side == "long":
                desired_sl = entry + lock_pips * pip
                if current_price is not None:
                    desired_sl = min(desired_sl, float(current_price) - 0.003)
                if trade_sl is not None and desired_sl <= trade_sl + 1e-6:
                    continue
            else:
                desired_sl = entry - lock_pips * pip
                if current_price is not None:
                    desired_sl = max(desired_sl, float(current_price) + 0.003)
                if trade_sl is not None and desired_sl >= trade_sl - 1e-6:
                    continue

            desired_sl = round(desired_sl, 3)
            tp_price = trade_tp

            _maybe_update_protections(trade_id, desired_sl, tp_price)


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
    stage_state: Optional[dict[str, dict[str, int]]] = None,
    pocket_profiles: Optional[dict[str, dict[str, float]]] = None,
    now: Optional[datetime] = None,
    threshold_overrides: Optional[dict[str, tuple[float, float]]] = None,
) -> list[tuple[str, str, int]]:
    price = fac_m1.get("close")
    if price is None:
        return []
    pip_scale = 100
    current_time = _ensure_utc(now)
    actions: list[tuple[str, str, int]] = []
    policy = policy_bus.latest()
    pockets_policy = policy.pockets if policy else {}

    for pocket, info in open_positions.items():
        if pocket == "__net__":
            continue
        thresholds = _PARTIAL_THRESHOLDS.get(pocket)
        fractions = _PARTIAL_FRACTIONS
        if pocket == "macro":
            thresholds, fractions = _macro_partial_profile(fac_h4, range_mode)
        elif range_mode:
            thresholds = _PARTIAL_THRESHOLDS_RANGE.get(pocket, thresholds)
        plan = pockets_policy.get(pocket) if isinstance(pockets_policy, dict) else {}
        partial_plan = plan.get("partial_profile", {}) if isinstance(plan, dict) else {}
        min_units_override: Optional[int] = None
        if isinstance(partial_plan, dict):
            plan_thresholds = partial_plan.get("thresholds_pips")
            if isinstance(plan_thresholds, (list, tuple)) and plan_thresholds:
                try:
                    thresholds = [float(x) for x in plan_thresholds]
                except (TypeError, ValueError):
                    pass
            plan_fractions = partial_plan.get("fractions")
            if isinstance(plan_fractions, (list, tuple)) and plan_fractions:
                try:
                    fractions = tuple(float(x) for x in plan_fractions)
                except (TypeError, ValueError):
                    pass
            override_units = partial_plan.get("min_units")
            try:
                min_units_override = int(override_units) if override_units is not None else None
            except (TypeError, ValueError):
                min_units_override = None
        if not thresholds:
            continue
        pocket_stage = (stage_state or {}).get(pocket, {})
        pocket_profile = (pocket_profiles or {}).get(pocket, {})
        trades = info.get("open_trades") or []
        for tr in trades:
            trade_id = tr.get("trade_id")
            side = tr.get("side")
            entry = tr.get("price")
            units = int(tr.get("units", 0) or 0)
            if not trade_id or not side or entry is None or units == 0:
                continue
            client_id = str(tr.get("client_id") or "")
            if not client_id.startswith("qr-"):
                continue
            opened_at = _parse_trade_open_time(tr.get("open_time"))
            age_minutes = None
            if opened_at:
                age_minutes = max(0.0, (current_time - opened_at).total_seconds() / 60.0)
            if range_mode and pocket == "macro":
                if age_minutes is None or age_minutes < _PARTIAL_RANGE_MACRO_MIN_AGE_MIN:
                    continue
            current_stage = _PARTIAL_STAGE.get(trade_id, 0)
            min_hold_sec = _trade_min_hold_seconds(tr, pocket)
            age_seconds = _trade_age_seconds(tr, current_time)
            if age_seconds is not None and age_seconds < min_hold_sec:
                if range_mode:
                    thesis = _coerce_entry_thesis(tr.get("entry_thesis"))
                    strategy_tag = thesis.get("strategy_tag") or tr.get("strategy_tag")
                    tags = {
                        "pocket": pocket,
                        "strategy": strategy_tag or "unknown",
                    }
                    log_metric("partial_hold_guard", 1.0, tags=tags)
                continue
            gain_pips = 0.0
            stage_level = (pocket_stage or {}).get(side, 0)
            profile = pocket_profile or {}
            effective_thresholds = list(thresholds)
            if stage_level >= 3:
                effective_thresholds = [max(2.0, t * 0.75) for t in effective_thresholds]
            elif stage_level >= 1:
                effective_thresholds = [max(2.0, t * 0.85) for t in effective_thresholds]
            if profile.get("win_rate", 0.0) >= 0.55:
                effective_thresholds = [max(2.0, t * 0.9) for t in effective_thresholds]
            if profile.get("avg_loss_pips", 0.0) > 5.0:
                effective_thresholds = [max(1.5, t * 0.8) for t in effective_thresholds]
            thresholds_eff = tuple(effective_thresholds)
            if side == "long":
                gain_pips = (price - entry) * pip_scale
            else:
                gain_pips = (entry - price) * pip_scale
            if gain_pips <= thresholds_eff[0]:
                continue
            stage = 0
            for idx, threshold in enumerate(thresholds_eff, start=1):
                if gain_pips >= threshold:
                    stage = idx
            if stage <= current_stage:
                continue
            fraction_idx = min(stage - 1, len(fractions) - 1)
            fraction = fractions[fraction_idx]
            reduce_units = int(abs(units) * fraction)
            min_units_threshold = min_units_override if min_units_override is not None else _PARTIAL_MIN_UNITS
            if reduce_units < min_units_threshold:
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
    pocket: Literal["micro", "macro", "scalp", "scalp_fast", "manual"],
    *,
    client_order_id: Optional[str] = None,
    strategy_tag: Optional[str] = None,
    reduce_only: bool = False,
    entry_thesis: Optional[dict] = None,
    meta: Optional[dict] = None,
    confidence: Optional[int] = None,
) -> Optional[str]:
    """
    units : +10000 = buy 0.1 lot, ‑10000 = sell 0.1 lot
    returns order ticket id（決済のみの fill でも tradeID を返却）
    """
    if strategy_tag is not None:
        strategy_tag = str(strategy_tag)
        if not strategy_tag:
            strategy_tag = None
    else:
        strategy_tag = None
    thesis_sl_pips: Optional[float] = None
    thesis_tp_pips: Optional[float] = None
    if isinstance(entry_thesis, dict):
        raw_tag = entry_thesis.get("strategy_tag") or entry_thesis.get("strategy")
        if raw_tag and not strategy_tag:
            strategy_tag = str(raw_tag)
        try:
            value = entry_thesis.get("sl_pips")
            if value is not None:
                thesis_sl_pips = float(value)
        except (TypeError, ValueError):
            thesis_sl_pips = None
        try:
            value = entry_thesis.get("tp_pips") or entry_thesis.get("target_tp_pips")
            if value is not None:
                thesis_tp_pips = float(value)
        except (TypeError, ValueError):
            thesis_tp_pips = None
    side_label = "buy" if units > 0 else "sell"

    generated_client_id = False
    if not client_order_id:
        strategy_hint = None
        if isinstance(entry_thesis, dict):
            strategy_hint = entry_thesis.get("strategy_tag") or entry_thesis.get("strategy")
        if not strategy_hint and strategy_tag:
            strategy_hint = strategy_tag
        focus_hint = pocket or "hybrid"
        client_order_id = build_client_order_id(focus_hint, str(strategy_hint or "fallback"))
        generated_client_id = True
        logging.warning(
            "[ORDER] Missing client_order_id; generated %s (pocket=%s strategy=%s)",
            client_order_id,
            pocket,
            strategy_hint or strategy_tag or "unknown",
        )

    entry_price_meta = _as_float((meta or {}).get("entry_price"))

    if strategy_tag and not reduce_only:
        blocked, remain, reason = strategy_guard.is_blocked(strategy_tag)
        if blocked:
            _console_order_log(
                "OPEN_SKIP",
                pocket=pocket,
                strategy_tag=strategy_tag,
                side=side_label,
                units=units,
                sl_price=sl_price,
                tp_price=tp_price,
                client_order_id=client_order_id,
                note=f"strategy_cooldown:{reason}",
            )
            _log_order(
                pocket=pocket,
                instrument=instrument,
                side="buy" if units > 0 else "sell",
                units=units,
                sl_price=sl_price,
                tp_price=tp_price,
                client_order_id=client_order_id,
                status="strategy_cooldown",
                attempt=0,
                request_payload={
                    "strategy_tag": strategy_tag,
                    "cooldown_reason": reason,
                    "cooldown_remaining_sec": remain,
                },
            )
            return None

    # Pocket-level cooldown after margin rejection
    try:
        if pocket and _MARGIN_REJECT_UNTIL.get(pocket, 0.0) > time.monotonic():
            _console_order_log(
                "OPEN_SKIP",
                pocket=pocket,
                strategy_tag=strategy_tag,
                side=side_label,
                units=units,
                sl_price=sl_price,
                tp_price=tp_price,
                client_order_id=client_order_id,
                note="margin_cooldown_active",
            )
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

    if _DISABLE_STOP_LOSS:
        sl_price = None

    if (
        _DYNAMIC_SL_ENABLE
        and (pocket or "").lower() in _DYNAMIC_SL_POCKETS
        and not reduce_only
        and entry_price_meta is not None
        and not _DISABLE_STOP_LOSS
    ):
        loss_guard = None
        sl_hint = None
        if isinstance(entry_thesis, dict):
            loss_guard = _as_float(entry_thesis.get("loss_guard_pips"))
            if loss_guard is None:
                loss_guard = _as_float(entry_thesis.get("loss_guard"))
            sl_hint = _as_float(entry_thesis.get("sl_pips"))
        target_pips = 0.0
        for cand in (sl_hint, loss_guard):
            if cand and cand > target_pips:
                target_pips = cand
        if target_pips > 0.0:
            dynamic_pips = min(_DYNAMIC_SL_MAX_PIPS, target_pips * _DYNAMIC_SL_RATIO)
            target_pips = max(target_pips, dynamic_pips)
            offset = round(target_pips * 0.01, 3)
            if units > 0:
                sl_price = round(entry_price_meta - offset, 3)
            else:
                sl_price = round(entry_price_meta + offset, 3)

    if not is_market_open():
        logging.info(
            "[ORDER] Market closed window. Skip order pocket=%s units=%s client_id=%s",
            pocket,
            units,
            client_order_id,
        )
        _console_order_log(
            "OPEN_SKIP",
            pocket=pocket,
            strategy_tag=strategy_tag,
            side=side_label,
            units=units,
            sl_price=sl_price,
            tp_price=tp_price,
            client_order_id=client_order_id,
            note="market_closed",
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

    quote = _fetch_quote(instrument)
    if quote and quote.get("spread_pips") is not None:
        if quote["spread_pips"] >= _ORDER_SPREAD_BLOCK_PIPS and not reduce_only:
            note = f"spread_block:{quote['spread_pips']:.2f}p"
            _console_order_log(
                "OPEN_SKIP",
                pocket=pocket,
                strategy_tag=strategy_tag,
                side=side_label,
                units=units,
                sl_price=sl_price,
                tp_price=tp_price,
                client_order_id=client_order_id,
                note=note,
            )
            _log_order(
                pocket=pocket,
                instrument=instrument,
                side="buy" if units > 0 else "sell",
                units=units,
                sl_price=sl_price,
                tp_price=tp_price,
                client_order_id=client_order_id,
                status="spread_block",
                attempt=0,
                request_payload={"quote": quote, "threshold": _ORDER_SPREAD_BLOCK_PIPS},
            )
            return None

    estimated_entry = _estimate_entry_price(
        units=units, sl_price=sl_price, tp_price=tp_price, meta=meta
    )
    entry_basis = None
    if quote:
        entry_basis = quote["ask"] if units > 0 else quote["bid"]
        estimated_entry = entry_basis
    if entry_basis is None and entry_price_meta is not None:
        entry_basis = entry_price_meta

    # Recalculate SL/TP from thesis gaps using live quote to preserve intended RR
    if entry_basis is not None:
        if thesis_sl_pips is not None:
            if units > 0:
                sl_price = round(entry_basis - thesis_sl_pips * 0.01, 3)
            else:
                sl_price = round(entry_basis + thesis_sl_pips * 0.01, 3)
        if thesis_tp_pips is not None:
            if units > 0:
                tp_price = round(entry_basis + thesis_tp_pips * 0.01, 3)
            else:
                tp_price = round(entry_basis - thesis_tp_pips * 0.01, 3)

    # Margin preflight (new entriesのみ)
    preflight_units = units
    min_allowed_units = min_units_for_pocket(pocket)
    requested_units = units
    clamped_to_minimum = False
    if (
        not reduce_only
        and min_allowed_units > 0
        and 0 < abs(requested_units) < min_allowed_units
    ):
        requested_units = min_allowed_units if requested_units > 0 else -min_allowed_units
        clamped_to_minimum = True
        units = requested_units
        logging.info(
            "[ORDER] units clamped to pocket minimum pocket=%s requested=%s -> %s",
            pocket,
            preflight_units,
            requested_units,
        )

    if not reduce_only and estimated_entry is not None:
        allowed_units, req_margin = _preflight_units(
            estimated_price=estimated_entry, requested_units=requested_units
        )
        if allowed_units == 0:
            _console_order_log(
                "OPEN_SKIP",
                pocket=pocket,
                strategy_tag=strategy_tag,
                side=side_label,
                units=units,
                sl_price=sl_price,
                tp_price=tp_price,
                client_order_id=client_order_id,
                note="insufficient_margin",
            )
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
                    "estimated_price": estimated_entry,
                    "required_margin": req_margin,
                },
            )
            return None
        if (
            min_allowed_units > 0
            and 0 < abs(allowed_units) < min_allowed_units
            and not reduce_only
        ):
            _console_order_log(
                "OPEN_SKIP",
                pocket=pocket,
                strategy_tag=strategy_tag,
                side=side_label,
                units=allowed_units,
                sl_price=sl_price,
                tp_price=tp_price,
                client_order_id=client_order_id,
                note="insufficient_margin_min_units",
            )
            _log_order(
                pocket=pocket,
                instrument=instrument,
                side="buy" if allowed_units > 0 else "sell",
                units=allowed_units,
                sl_price=sl_price,
                tp_price=tp_price,
                client_order_id=client_order_id,
                status="min_unit_margin_skip",
                attempt=0,
                request_payload={
                    "requested_units": requested_units,
                    "min_units": min_allowed_units,
                    "estimated_price": estimated_entry,
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
        if clamped_to_minimum and abs(preflight_units) >= min_allowed_units:
            logging.info(
                "[ORDER] Raised %s pocket units to minimum %s from %s",
                pocket,
                min_allowed_units,
                units,
            )

    if not reduce_only and estimated_entry is not None:
        norm_sl = None if _DISABLE_STOP_LOSS else sl_price
        norm_tp = tp_price
        norm_sl, norm_tp, normalized = _normalize_protections(
            estimated_entry,
            norm_sl,
            norm_tp,
            units > 0,
        )
        sl_price = None if _DISABLE_STOP_LOSS else norm_sl
        tp_price = norm_tp
        if normalized:
            logging.debug(
                "[ORDER] normalized SL/TP client=%s sl=%s tp=%s entry=%.3f",
                client_order_id,
                f"{sl_price:.3f}" if sl_price is not None else "None",
                f"{tp_price:.3f}" if tp_price is not None else "None",
                estimated_entry,
            )

    # Cap new-entry order size with dynamic boost for high-confidence profiles
    if not reduce_only:
        cap_multiplier = 1.0
        try:
            if confidence is None and isinstance(entry_thesis, dict):
                confidence_val = entry_thesis.get("confidence")
            else:
                confidence_val = confidence
            if confidence_val is not None:
                c = float(confidence_val)
                if c >= 90:
                    cap_multiplier = 1.6
                elif c >= 80:
                    cap_multiplier = max(cap_multiplier, 1.3)
        except Exception:
            pass
        try:
            profile = None
            if isinstance(entry_thesis, dict):
                profile = entry_thesis.get("profile") or entry_thesis.get("pocket_profile")
            if profile and str(profile).lower() in {"aggressive", "momentum"}:
                cap_multiplier = max(cap_multiplier, 1.4)
        except Exception:
            pass
        dynamic_cap = int(_MAX_ORDER_UNITS * cap_multiplier)
        dynamic_cap = min(max(_MAX_ORDER_UNITS, dynamic_cap), _MAX_ORDER_UNITS_HARD)
        abs_units = abs(int(preflight_units))
        if abs_units > dynamic_cap:
            capped = (1 if preflight_units > 0 else -1) * dynamic_cap
            _log_order(
                pocket=pocket,
                instrument=instrument,
                side="buy" if preflight_units > 0 else "sell",
                units=preflight_units,
                sl_price=sl_price,
                tp_price=tp_price,
                client_order_id=client_order_id,
                status="units_cap_applied",
                attempt=0,
                request_payload={
                    "from_units": preflight_units,
                    "to_units": capped,
                    "cap_multiplier": cap_multiplier,
                    "max_order_units": _MAX_ORDER_UNITS,
                    "hard_cap": _MAX_ORDER_UNITS_HARD,
                },
            )
            preflight_units = capped

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
    if (not _DISABLE_STOP_LOSS) and sl_price is not None:
        order_data["order"]["stopLossOnFill"] = {"price": f"{sl_price:.3f}"}
    if tp_price is not None:
        order_data["order"]["takeProfitOnFill"] = {"price": f"{tp_price:.3f}"}

    side = "buy" if preflight_units > 0 else "sell"
    units_to_send = preflight_units
    _console_order_log(
        "OPEN_REQ",
        pocket=pocket,
        strategy_tag=strategy_tag,
        side=side,
        units=units_to_send,
        sl_price=sl_price,
        tp_price=tp_price,
        client_order_id=client_order_id,
        note="reduce_only" if reduce_only else None,
    )
    protection_fallback_applied = False
    for attempt in range(2):
        payload = order_data.copy()
        payload["order"] = dict(order_data["order"], units=str(units_to_send))
        # Log attempt payload (include non-OANDA context for analytics)
        attempt_payload: dict = {"oanda": payload}
        if entry_thesis is not None:
            attempt_payload["entry_thesis"] = entry_thesis
        if meta is not None:
            attempt_payload["meta"] = meta
        if quote:
            attempt_payload["quote"] = quote
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
                reason_key = str(reason or "").upper()
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
                    stage_index=stage_index,
                    error_code=reject.get("errorCode"),
                    error_message=reject.get("errorMessage") or reason,
                    response_payload=response,
                )
                if reason_key == "INSUFFICIENT_MARGIN" and pocket:
                    _MARGIN_REJECT_UNTIL[pocket] = time.monotonic() + 120.0
                if (
                    attempt == 0
                    and not reduce_only
                    and not protection_fallback_applied
                    and reason_key in _PROTECTION_RETRY_REASONS
                    and (sl_price is not None or tp_price is not None)
                ):
                    fallback_basis = _derive_fallback_basis(
                        estimated_entry,
                        sl_price,
                        tp_price,
                        units_to_send > 0,
                    )
                    fallback_sl, fallback_tp = _fallback_protections(
                        fallback_basis,
                        is_buy=units_to_send > 0,
                        has_sl=sl_price is not None,
                        has_tp=tp_price is not None,
                        sl_gap_pips=thesis_sl_pips,
                        tp_gap_pips=thesis_tp_pips,
                    )
                    if fallback_sl is not None or fallback_tp is not None:
                        if fallback_sl is not None:
                            order_data["order"]["stopLossOnFill"] = {
                                "price": f"{fallback_sl:.3f}"
                            }
                            sl_price = fallback_sl
                        elif "stopLossOnFill" in order_data["order"]:
                            order_data["order"].pop("stopLossOnFill", None)
                        if fallback_tp is not None:
                            order_data["order"]["takeProfitOnFill"] = {
                                "price": f"{fallback_tp:.3f}"
                            }
                            tp_price = fallback_tp
                        elif "takeProfitOnFill" in order_data["order"]:
                            order_data["order"].pop("takeProfitOnFill", None)
                        protection_fallback_applied = True
                        logging.warning(
                            "[ORDER] protection fallback applied client=%s reason=%s",
                            client_order_id,
                            reason_key,
                        )
                        continue
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
                    stage_index=stage_index,
                    ticket_id=trade_id,
                    executed_price=executed_price,
                    response_payload=response,
                )
                _console_order_log(
                    "OPEN_FILLED",
                    pocket=pocket,
                    strategy_tag=strategy_tag,
                    side=side,
                    units=units_to_send,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    client_order_id=client_order_id,
                    ticket_id=trade_id,
                    note=f"attempt={attempt+1}",
                )
                target_sl = None if _DISABLE_STOP_LOSS else sl_price
                _maybe_update_protections(trade_id, target_sl, tp_price)
                return trade_id

            logging.error(
                "OANDA order fill lacked trade identifiers (attempt %d): %s",
                attempt + 1,
                response,
            )
            _console_order_log(
                "OPEN_FAIL",
                pocket=pocket,
                strategy_tag=strategy_tag,
                side=side,
                units=units_to_send,
                sl_price=sl_price,
                tp_price=tp_price,
                client_order_id=client_order_id,
                note=f"no_trade_id attempt={attempt+1}",
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
                stage_index=stage_index,
                response_payload=response,
            )
            return None, None
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
                stage_index=stage_index,
                error_message=str(e),
                response_payload=resp if isinstance(resp, dict) else None,
            )

            if (
                attempt == 0
                and abs(units_to_send) >= 2000
                and pocket != "scalp_fast"
            ):
                units_to_send = int(units_to_send * 0.5)
                if units_to_send == 0:
                    break
                logging.info(
                    "Retrying order with reduced units=%s (half).", units_to_send
                )
                continue
            return None, None
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
                stage_index=stage_index,
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
