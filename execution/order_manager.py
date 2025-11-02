"""
execution.order_manager
~~~~~~~~~~~~~~~~~~~~~~~
OANDA REST で成行・指値を発注。
• clientExtensions.tag = "pocket=micro" などを付与
"""

from __future__ import annotations
from typing import Literal, Optional, Tuple
import sqlite3
import json
import pathlib
from datetime import datetime, timezone

from oandapyV20 import API
from oandapyV20.exceptions import V20Error
from oandapyV20.endpoints.orders import OrderCreate
from oandapyV20.endpoints.trades import TradeCRCDO, TradeClose

import logging

from utils.secrets import get_secret

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

# 最低発注単位（AGENT.me 6.1 に準拠）。
# リスク計算・ステージ係数適用後の“最終 units”に対して適用する最終ゲート。
# reduce_only（決済）では適用しない。
_MIN_ORDER_UNITS = 1000

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
        if error_code in {"TRADE_DOES_NOT_EXIST", "NOT_FOUND"}:
            logging.info(
                "[ORDER] Trade %s already closed upstream, skipping fallback.",
                trade_id,
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
        "scalp": (max(4.0, (atr_m1 or 0.0) * 0.6), 1.4),
    }
    # ポケット別の BE/トレーリング開始閾値（pip）
    # micro/scalp は早めに建値超えへ移行し、利確を積み上げる方針
    per_pocket_triggers = {
        "macro": max(6.0, (atr_h4 or atr_m1 or 0.0) * 1.2),
        "micro": max(3.0, (atr_m1 or 0.0) * 0.8),
        "scalp": max(2.0, (atr_m1 or 0.0) * 0.6),
    }
    lock_ratio = 0.6
    per_pocket_min_lock = {"macro": 3.0, "micro": 2.0, "scalp": 1.2}
    pip = 0.01
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
            entry = float(price)
            sl_pips = max(1.0, base_sl)
            tp_pips = max(sl_pips * tp_ratio, sl_pips + 5.0)
            gain_pips = 0.0
            if side == "long":
                gain_pips = (current_price or entry) - entry
                gain_pips *= 100
                sl_price = round(entry - sl_pips * pip, 3)
                tp_price = round(entry + tp_pips * pip, 3)
                if gain_pips > trail_trigger:
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
                tp_price = round(entry - tp_pips * pip, 3)
                if gain_pips > trail_trigger:
                    lock_pips = max(min_lock, gain_pips * lock_ratio)
                    be_price = entry - lock_pips * pip
                    sl_price = min(sl_price, round(be_price, 3))
                    tp_price = round(entry - max(gain_pips + sl_pips, tp_pips) * pip, 3)
                    if tp_price >= sl_price - 0.002:
                        tp_price = round(sl_price - max(0.004, tp_pips * pip), 3)
                if current_price is not None and sl_price <= current_price:
                    sl_price = round(current_price + 0.003, 3)
            _maybe_update_protections(trade_id, sl_price, tp_price)


def plan_partial_reductions(
    open_positions: dict,
    fac_m1: dict,
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

    for pocket, info in open_positions.items():
        if pocket == "__net__":
            continue
        thresholds = None
        if threshold_overrides and pocket in threshold_overrides:
            thresholds = threshold_overrides[pocket]
        if thresholds is None:
            thresholds = _PARTIAL_THRESHOLDS.get(pocket)
        if range_mode:
            thresholds = _PARTIAL_THRESHOLDS_RANGE.get(pocket, thresholds)
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
            opened_at = _parse_trade_open_time(tr.get("open_time"))
            age_minutes = None
            if opened_at:
                age_minutes = max(0.0, (current_time - opened_at).total_seconds() / 60.0)
            if range_mode and pocket == "macro":
                if age_minutes is None or age_minutes < _PARTIAL_RANGE_MACRO_MIN_AGE_MIN:
                    continue
            current_stage = _PARTIAL_STAGE.get(trade_id, 0)
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
            fraction_idx = min(stage - 1, len(_PARTIAL_FRACTIONS) - 1)
            fraction = _PARTIAL_FRACTIONS[fraction_idx]
            if stage_level >= 3:
                fraction = min(0.65, fraction + 0.15)
            elif stage_level == 2:
                fraction = min(0.6, fraction + 0.1)
            if profile.get("win_rate", 0.0) >= 0.55:
                fraction = min(0.6, fraction + 0.05)
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
    pocket: Literal["micro", "macro", "scalp", "scalp_fast", "manual"],
    *,
    client_order_id: Optional[str] = None,
    reduce_only: bool = False,
    entry_thesis: Optional[dict] = None,
    stage_index: Optional[int] = None,
) -> tuple[Optional[str], Optional[float]]:
    """
    units : +10000 = buy 0.1 lot, ‑10000 = sell 0.1 lot
    returns order ticket id（決済のみの fill でも tradeID を返却）
    """
    # 最終ゲート: エントリー（reduce_only=False）で units の絶対値が下限未満の場合は発注しない
    if not reduce_only and abs(int(units)) < _MIN_ORDER_UNITS:
        _log_order(
            pocket=pocket,
            instrument=instrument,
            side=("buy" if units > 0 else "sell"),
            units=units,
            sl_price=sl_price,
            tp_price=tp_price,
            client_order_id=client_order_id,
            status="skipped_tiny_units",
            attempt=0,
            stage_index=stage_index,
            request_payload={
                "reason": "units_below_min",
                "min_units": _MIN_ORDER_UNITS,
                "entry_thesis": entry_thesis or {},
            },
        )
        logging.info(
            "[ORDER] Skip entry due to tiny units=%s (<%s), pocket=%s",
            units,
            _MIN_ORDER_UNITS,
            pocket,
        )
        return None, None

    order_data = {
        "order": {
            "type": "MARKET",
            "instrument": instrument,
            "units": str(units),
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

    side = "buy" if units > 0 else "sell"
    units_to_send = units
    for attempt in range(2):
        payload = order_data.copy()
        payload["order"] = dict(order_data["order"], units=str(units_to_send))
        # Log attempt payload (append meta with entry_thesis if provided)
        log_payload = payload
        if entry_thesis is not None:
            log_payload = {"order": payload.get("order"), "meta": {"entry_thesis": entry_thesis}}
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
            stage_index=stage_index,
            request_payload=log_payload,
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
                    stage_index=stage_index,
                    error_code=reject.get("errorCode"),
                    error_message=reject.get("errorMessage") or reason,
                    response_payload=response,
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
                _maybe_update_protections(trade_id, sl_price, tp_price)
            return trade_id, executed_price

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
            return None, None
    return None, None
