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
import math
from typing import Any, Literal, Optional, Tuple
import requests

from oandapyV20 import API
from oandapyV20.exceptions import V20Error
from oandapyV20.endpoints.orders import OrderCancel, OrderCreate
from oandapyV20.endpoints.trades import TradeCRCDO, TradeClose, TradeDetails

from execution.order_ids import build_client_order_id
from execution.stop_loss_policy import stop_loss_disabled, trailing_sl_allowed
from execution.section_axis import attach_section_axis
from execution.entry_guard import evaluate_entry_guard

from analysis import policy_bus
from analysis.technique_engine import evaluate_entry_techniques
from utils.secrets import get_secret
from utils.market_hours import is_market_open
from utils.metrics_logger import log_metric
from execution import strategy_guard
from execution.position_manager import PositionManager, agent_client_prefixes
from execution.risk_guard import POCKET_MAX_RATIOS, MAX_LEVERAGE
from workers.common import perf_guard
from utils import signal_bus
from utils.oanda_account import get_account_snapshot
try:
    from market_data import tick_window
except Exception:  # pragma: no cover - optional
    tick_window = None

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

# エントリーが詰まらないようデフォルトの最小ユニットを下げる。
_DEFAULT_MIN_UNITS = _env_int("ORDER_MIN_UNITS_DEFAULT", 1_000)
# Macro も環境変数で可変（デフォルト 2,000 units、最低でも DEFAULT_MIN を確保）
_MACRO_MIN_UNITS_DEFAULT = max(_env_int("ORDER_MIN_UNITS_MACRO", 2_000), _DEFAULT_MIN_UNITS)
_MIN_UNITS_BY_POCKET: dict[str, int] = {
    "micro": _env_int("ORDER_MIN_UNITS_MICRO", _DEFAULT_MIN_UNITS),
    "macro": _env_int("ORDER_MIN_UNITS_MACRO", _MACRO_MIN_UNITS_DEFAULT),
    # scalp 系も同じ下限を使う（環境変数で上書き可）
    "scalp": _env_int("ORDER_MIN_UNITS_SCALP", _DEFAULT_MIN_UNITS),
}
# Raise scalp floor to avoid tiny entries; can override via env ORDER_MIN_UNITS_SCALP
# If true, do not attach stopLossOnFill (TP is still sent).
STOP_LOSS_DISABLED = stop_loss_disabled()
TRAILING_SL_ALLOWED = trailing_sl_allowed()


def min_units_for_pocket(pocket: Optional[str]) -> int:
    if not pocket:
        return _DEFAULT_MIN_UNITS
    return int(_MIN_UNITS_BY_POCKET.get(pocket, _DEFAULT_MIN_UNITS))


def _apply_tech_units(units: int, multiplier: float, pocket: str) -> int:
    if units == 0:
        return units
    sign = 1 if units > 0 else -1
    target = int(round(abs(units) * multiplier))
    if target <= 0:
        target = abs(units)
    min_units = min_units_for_pocket(pocket)
    if min_units > 0 and target < min_units and abs(units) >= min_units:
        target = min_units
    return sign * target


_EXIT_NO_NEGATIVE_CLOSE = os.getenv("EXIT_NO_NEGATIVE_CLOSE", "1").strip().lower() not in {"", "0", "false", "no"}
_EXIT_EMERGENCY_ALLOW_NEGATIVE = os.getenv("EXIT_EMERGENCY_ALLOW_NEGATIVE", "1").strip().lower() not in {
    "",
    "0",
    "false",
    "no",
}
_EXIT_EMERGENCY_HEALTH_BUFFER = max(
    0.0, float(os.getenv("EXIT_EMERGENCY_HEALTH_BUFFER", "0.07"))
)
_EXIT_EMERGENCY_CACHE_TTL_SEC = max(
    0.5, float(os.getenv("EXIT_EMERGENCY_CACHE_TTL_SEC", "2.0"))
)
_LAST_EMERGENCY_LOG_TS: float = 0.0

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
# ワーカーのオーダーをメインの関所に転送するフラグ（reduce_only は除外）
_FORWARD_TO_SIGNAL_GATE = (
    os.getenv("ORDER_FORWARD_TO_SIGNAL_GATE", "1").strip().lower()
    not in {"", "0", "false", "no"}
)
# コメントを付けるとリジェクトが発生する場合に強制オフにするトグル（デフォルトで無効化）
_DISABLE_CLIENT_COMMENT = os.getenv("ORDER_DISABLE_CLIENT_COMMENT", "1").strip().lower() not in {
    "",
    "0",
    "false",
    "no",
}

# Max units per new entry order (reduce_only orders are exempt)
try:
    _MAX_ORDER_UNITS = int(os.getenv("MAX_ORDER_UNITS", "40000"))
except Exception:
    _MAX_ORDER_UNITS = 40000
# Hard safety cap even after dynamic boosts
try:
    _MAX_ORDER_UNITS_HARD = int(os.getenv("MAX_ORDER_UNITS_HARD", "60000"))
except Exception:
    _MAX_ORDER_UNITS_HARD = 60000

# 最低発注単位（AGENT.me 6.1 に準拠）。
# リスク計算・ステージ係数適用後の“最終 units”に対して適用する最終ゲート。
# reduce_only（決済）では適用しない。
_MIN_ORDER_UNITS = 500

# Directional exposure cap (dynamic; scales units instead of rejecting)
_DIR_CAP_ENABLE = os.getenv("DIR_CAP_ENABLE", "1").strip().lower() not in {"", "0", "false", "no"}
_DIR_CAP_RATIO = float(os.getenv("DIR_CAP_RATIO", "0.70"))
_DIR_CAP_WARN_RATIO = float(os.getenv("DIR_CAP_WARN_RATIO", "0.98"))
# Floor multiplier to avoid crushing frequency when shrinking; 0.0 to disable
_DIR_CAP_MIN_FRACTION = float(os.getenv("DIR_CAP_MIN_FRACTION", "0.15"))
_DIR_CAP_CACHE: Optional[PositionManager] = None

# ---------- orders logger (logs/orders.db) ----------
_ORDERS_DB_PATH = pathlib.Path("logs/orders.db")
_ORDER_DB_JOURNAL_MODE = os.getenv("ORDER_DB_JOURNAL_MODE", "WAL")
_ORDER_DB_SYNCHRONOUS = os.getenv("ORDER_DB_SYNCHRONOUS", "NORMAL")
_ORDER_DB_BUSY_TIMEOUT_MS = int(os.getenv("ORDER_DB_BUSY_TIMEOUT_MS", "5000"))
_ORDER_DB_WAL_AUTOCHECKPOINT_PAGES = int(
    os.getenv("ORDER_DB_WAL_AUTOCHECKPOINT_PAGES", "500")
)
_ORDER_DB_JOURNAL_SIZE_LIMIT_BYTES = int(
    os.getenv("ORDER_DB_JOURNAL_SIZE_LIMIT_BYTES", "67108864")
)
_ORDER_DB_CHECKPOINT_ENABLE = (
    os.getenv("ORDER_DB_CHECKPOINT_ENABLE", "1").strip().lower()
    not in {"", "0", "false", "no"}
)
_ORDER_DB_CHECKPOINT_INTERVAL_SEC = float(
    os.getenv("ORDER_DB_CHECKPOINT_INTERVAL_SEC", "60")
)
_ORDER_DB_CHECKPOINT_MIN_WAL_BYTES = int(
    os.getenv("ORDER_DB_CHECKPOINT_MIN_WAL_BYTES", "33554432")
)
_ORDERS_DB_WAL_PATH = _ORDERS_DB_PATH.with_suffix(_ORDERS_DB_PATH.suffix + "-wal")
_LAST_ORDER_DB_CHECKPOINT = 0.0

_DEFAULT_MIN_HOLD_SEC = {
    "macro": 360.0,
    "micro": 150.0,
    "scalp": 75.0,
}


def _configure_orders_sqlite(con: sqlite3.Connection) -> sqlite3.Connection:
    """Apply SQLite PRAGMAs for WAL size control."""
    try:
        con.execute(f"PRAGMA journal_mode={_ORDER_DB_JOURNAL_MODE}")
    except sqlite3.Error:
        pass
    try:
        con.execute(f"PRAGMA synchronous={_ORDER_DB_SYNCHRONOUS}")
    except sqlite3.Error:
        pass
    try:
        con.execute(f"PRAGMA busy_timeout={_ORDER_DB_BUSY_TIMEOUT_MS}")
    except sqlite3.Error:
        pass
    try:
        con.execute(
            f"PRAGMA wal_autocheckpoint={_ORDER_DB_WAL_AUTOCHECKPOINT_PAGES}"
        )
    except sqlite3.Error:
        pass
    try:
        con.execute(
            f"PRAGMA journal_size_limit={_ORDER_DB_JOURNAL_SIZE_LIMIT_BYTES}"
        )
    except sqlite3.Error:
        pass
    return con


def _maybe_checkpoint_orders_db(con: sqlite3.Connection) -> None:
    """Best-effort WAL checkpoint to avoid runaway orders.db-wal growth."""
    if not _ORDER_DB_CHECKPOINT_ENABLE:
        return
    global _LAST_ORDER_DB_CHECKPOINT
    now = time.monotonic()
    if now - _LAST_ORDER_DB_CHECKPOINT < _ORDER_DB_CHECKPOINT_INTERVAL_SEC:
        return
    try:
        wal_size = _ORDERS_DB_WAL_PATH.stat().st_size
    except FileNotFoundError:
        _LAST_ORDER_DB_CHECKPOINT = now
        return
    except Exception as exc:  # pragma: no cover - defensive
        logging.debug("[ORDER][DB] WAL size check failed: %s", exc)
        _LAST_ORDER_DB_CHECKPOINT = now
        return
    if wal_size < _ORDER_DB_CHECKPOINT_MIN_WAL_BYTES:
        _LAST_ORDER_DB_CHECKPOINT = now
        return
    try:
        row = con.execute("PRAGMA wal_checkpoint(PASSIVE)").fetchone()
        busy = row[0] if row else 0
        if busy == 0:
            con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    except sqlite3.Error as exc:
        logging.info("[ORDER][DB] wal checkpoint failed: %s", exc)
    _LAST_ORDER_DB_CHECKPOINT = now


def _ensure_orders_schema() -> sqlite3.Connection:
    _ORDERS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(_ORDERS_DB_PATH)
    _configure_orders_sqlite(con)
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


def _estimate_price(meta: Optional[dict]) -> Optional[float]:
    if not meta:
        return None
    for key in ("entry_price", "price", "mid_price"):
        try:
            val = float(meta.get(key))
            if val > 0:
                return val
        except Exception:
            continue
    return None


def _latest_mid_price() -> Optional[float]:
    if tick_window is None:
        return None
    try:
        ticks = tick_window.recent_ticks(seconds=3.0, limit=1)
    except Exception:
        return None
    if not ticks:
        return None
    tick = ticks[-1]
    try:
        if tick.get("mid") is not None:
            return float(tick.get("mid"))
    except Exception:
        pass
    bid = tick.get("bid")
    ask = tick.get("ask")
    try:
        if bid is not None and ask is not None:
            return (float(bid) + float(ask)) / 2.0
    except Exception:
        return None
    try:
        if bid is not None:
            return float(bid)
    except Exception:
        pass
    try:
        if ask is not None:
            return float(ask)
    except Exception:
        pass
    return None


def _entry_price_hint(entry_thesis: Optional[dict], meta: Optional[dict]) -> Optional[float]:
    if isinstance(entry_thesis, dict):
        for key in ("entry_price", "price", "entry_ref"):
            try:
                val = float(entry_thesis.get(key))
                if val > 0:
                    return val
            except Exception:
                continue
    est = _estimate_price(meta)
    if est is not None:
        return est
    return _latest_mid_price()


def _strategy_tag_from_thesis(entry_thesis: Optional[dict]) -> Optional[str]:
    if not isinstance(entry_thesis, dict):
        return None
    raw_tag = entry_thesis.get("strategy_tag") or entry_thesis.get("strategy")
    if raw_tag:
        return str(raw_tag)
    return None


def _scaled_thresholds(
    pocket: str,
    base: tuple[float, ...],
    atr_m1: float,
    vol_5m: float,
) -> tuple[float, ...]:
    """
    Scale partial/lock thresholds by short-term volatility/ATR.
    Bound the scale to avoid extreme shrink/expansion.
    """
    scale = 1.0
    if pocket in {"micro", "scalp", "scalp_fast"}:
        if vol_5m < 0.8:
            scale *= 0.9  # 早めに利確
        elif vol_5m > 1.6:
            scale *= 1.2  # 伸ばす
        if atr_m1 > 3.0:
            scale *= 1.15
        elif atr_m1 < 1.2:
            scale *= 0.92
    elif pocket == "macro":
        if vol_5m > 1.6 or atr_m1 > 3.0:
            scale *= 1.12
    scale = max(0.7, min(1.4, scale))
    return tuple(max(1.0, t * scale) for t in base)


def _apply_directional_cap(
    units: int,
    pocket: str,
    side_label: str,
    meta: Optional[dict],
) -> int:
    """
    Dynamically scale units to keep same-direction exposure within a cap derived
    from NAV and pocket ratio. Never rejects outright; scales down to remaining.
    """
    if not _DIR_CAP_ENABLE or units == 0 or pocket is None:
        return units
    try:
        from utils.oanda_account import get_account_snapshot
    except Exception:
        return units
    try:
        snap = get_account_snapshot(cache_ttl_sec=3.0)
    except Exception as exc:  # pragma: no cover - defensive
        logging.warning("[DIR_CAP] snapshot fetch failed: %s", exc)
        return units
    price = _estimate_price(meta) or 0.0
    if price <= 0:
        return units
    if snap.margin_rate <= 0 or snap.nav <= 0:
        return units
    try:
        pocket_ratio = POCKET_MAX_RATIOS.get(pocket, 1.0)
    except Exception:
        pocket_ratio = 1.0
    # cap derived from notional limit (NAV * leverage) and pocket share
    notional_cap_units = (snap.nav / price) * MAX_LEVERAGE * _DIR_CAP_RATIO * pocket_ratio
    if notional_cap_units <= 0:
        return units
    # fetch cached PositionManager to avoid repeated instantiation
    global _DIR_CAP_CACHE
    if _DIR_CAP_CACHE is None:
        _DIR_CAP_CACHE = PositionManager()
    positions = _DIR_CAP_CACHE.get_open_positions()
    info = positions.get(pocket) or {}
    current_same_dir = 0
    try:
        if side_label == "buy":
            current_same_dir = int(info.get("long_units", 0) or 0)
        else:
            current_same_dir = int(info.get("short_units", 0) or 0)
    except Exception:
        current_same_dir = 0
    remaining = max(0.0, notional_cap_units - current_same_dir)
    if remaining <= 0:
        logging.warning(
            "[DIR_CAP] pocket=%s side=%s blocked at cap cap_units=%.0f current=%s",
            pocket,
            side_label,
            notional_cap_units,
            current_same_dir,
        )
        return 0
    target = abs(units)
    if remaining < target:
        scaled = max(remaining, target * _DIR_CAP_MIN_FRACTION)
        new_units = int(max(0, round(scaled)))
        logging.info(
            "[DIR_CAP] scale pocket=%s side=%s units=%d -> %d (current=%s cap=%.0f)",
            pocket,
            side_label,
            units,
            new_units if units > 0 else -new_units,
            current_same_dir,
            notional_cap_units,
        )
        return new_units if units > 0 else -new_units
    # near-cap warning
    if current_same_dir + target >= _DIR_CAP_WARN_RATIO * notional_cap_units:
        logging.warning(
            "[DIR_CAP] near cap pocket=%s side=%s current=%s pending=%d cap=%.0f",
            pocket,
            side_label,
            current_same_dir,
            target,
            notional_cap_units,
        )
    return units


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


def _safe_json(payload: Optional[dict]) -> str:
    """
    Serialize payload safely; never raises. None -> "{}".
    Coerces non-serializable objects to string to avoid dropping the payload.
    """
    def _coerce(obj: object):
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return obj
        if isinstance(obj, dict):
            return {str(k): _coerce(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [_coerce(v) for v in obj]
        try:
            return str(obj)
        except Exception:
            return repr(obj)

    if payload is None:
        return "{}"
    try:
        coerced = _coerce(payload)
        return json.dumps(coerced, ensure_ascii=False)
    except Exception as exc:  # pragma: no cover - defensive
        logging.warning("[ORDER][LOG] failed to serialize payload: %s", exc)
        return "{}"


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
                _safe_json(request_payload),
                _safe_json(response_payload),
            ),
        )
        con.commit()
        _maybe_checkpoint_orders_db(con)
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


def _current_trade_unrealized_pl(trade_id: str) -> Optional[float]:
    try:
        req = TradeDetails(accountID=ACCOUNT, tradeID=trade_id)
        api.request(req)
        trade = req.response.get("trade") or {}
        pl = trade.get("unrealizedPL")
        if pl is None:
            return None
        return float(pl)
    except Exception as exc:  # noqa: BLE001
        logging.warning("[ORDER] Failed to fetch unrealized PL trade=%s err=%s", trade_id, exc)
        return None


def _should_allow_negative_close(client_order_id: Optional[str]) -> bool:
    if not _EXIT_EMERGENCY_ALLOW_NEGATIVE:
        return False
    if not client_order_id or not client_order_id.startswith(agent_client_prefixes):
        return False
    if _EXIT_EMERGENCY_HEALTH_BUFFER <= 0:
        return False
    try:
        snapshot = get_account_snapshot(cache_ttl_sec=_EXIT_EMERGENCY_CACHE_TTL_SEC)
    except Exception as exc:  # noqa: BLE001
        logging.debug("[ORDER] emergency health check failed: %s", exc)
        return False
    hb = snapshot.health_buffer
    if hb is None:
        return False
    if hb <= _EXIT_EMERGENCY_HEALTH_BUFFER:
        global _LAST_EMERGENCY_LOG_TS
        now = time.time()
        if now - _LAST_EMERGENCY_LOG_TS >= 30.0:
            log_metric(
                "close_emergency_allow_negative",
                float(hb),
                tags={"threshold": _EXIT_EMERGENCY_HEALTH_BUFFER},
            )
            _LAST_EMERGENCY_LOG_TS = now
        return True
    return False


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
    Netting-aware: if the requested units shrink current net exposure, allow them
    even when free margin is low.
    """
    try:
        from utils.oanda_account import (  # lazy import
            get_account_snapshot,
            get_position_summary,
        )

        snap = get_account_snapshot()
        margin_avail = float(getattr(snap, "margin_available", 0.0) or 0.0)
        margin_used = float(getattr(snap, "margin_used", 0.0) or 0.0)
        margin_rate = float(getattr(snap, "margin_rate", 0.0) or 0.0)
    except Exception as exc:
        logging.warning("[ORDER] preflight snapshot failed: %s", exc)
        try:
            log_metric(
                "order_margin_block",
                1.0,
                tags={"reason": "preflight_snapshot_failed"},
            )
        except Exception:
            pass
        return (0, 0.0)

    if margin_rate <= 0.0 or estimated_price <= 0.0:
        logging.warning(
            "[ORDER] preflight missing margin data rate=%.4f price=%.4f",
            margin_rate,
            estimated_price,
        )
        try:
            log_metric(
                "order_margin_block",
                1.0,
                tags={"reason": "preflight_missing_margin_data"},
            )
        except Exception:
            pass
        return (0, 0.0)

    per_unit_margin = estimated_price * margin_rate
    if per_unit_margin <= 0.0:
        return (0, 0.0)

    long_units: float | None = None
    short_units: float | None = None
    try:
        long_units, short_units = get_position_summary(timeout=3.0)
        long_units = max(0.0, float(long_units or 0.0))
        short_units = max(0.0, float(short_units or 0.0))
    except Exception as exc:
        logging.debug("[ORDER] preflight position summary unavailable: %s", exc)
        long_units = short_units = None

    if long_units is not None and short_units is not None:
        net_before = long_units - short_units
        net_after = net_before + requested_units
        margin_after = abs(net_after) * per_unit_margin
        # If the order reduces or keeps net margin, allow it regardless of free margin.
        if margin_after <= margin_used * 1.0005:
            return (requested_units, margin_after)

        budget = margin_used + margin_avail * margin_buffer
        if margin_after <= budget:
            return (requested_units, margin_after)

        # Scale down to stay within budget while preserving direction.
        target_net = budget / per_unit_margin
        clamped_net_after = max(-target_net, min(target_net, net_after))
        allowed_units = clamped_net_after - net_before
        if requested_units > 0:
            allowed_units = min(max(0.0, allowed_units), float(requested_units))
        else:
            allowed_units = -min(max(0.0, -allowed_units), float(abs(requested_units)))
        allowed_units_int = int(round(allowed_units))
        margin_allowed = abs(net_before + allowed_units_int) * per_unit_margin
        return (allowed_units_int, margin_allowed)

    # Fallback: use free margin only (no position breakdown available)
    req = abs(requested_units) * per_unit_margin
    if req * 1.0 <= margin_avail * margin_buffer:
        return (requested_units, req)

    max_units = int((margin_avail * margin_buffer) / per_unit_margin)
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


def _projected_usage_with_netting(
    nav: float,
    margin_rate: float,
    side_label: str,
    units: int,
    margin_used: float | None = None,
    meta: Optional[dict] = None,
) -> Optional[float]:
    """
    Estimate margin usage after applying `units`, accounting for netting.
    Returns None if estimation is not possible.
    """
    if nav <= 0 or margin_rate <= 0 or units == 0:
        return None
    try:
        from utils.oanda_account import get_position_summary

        long_u, short_u = get_position_summary()
    except Exception:
        return None

    price_hint = _estimate_price(meta) or 0.0
    net_before = float(long_u) - float(short_u)
    # If price is missing, infer from current margin usage as a last resort.
    if price_hint <= 0 and margin_used and net_before:
        try:
            price_hint = (margin_used / abs(net_before)) / margin_rate
        except Exception:
            price_hint = 0.0
    if price_hint <= 0:
        return None

    new_long = float(long_u)
    new_short = float(short_u)
    if side_label.lower() == "buy":
        new_long += abs(units)
    else:
        new_short += abs(units)
    projected_net_units = abs(new_long - new_short)
    projected_used = projected_net_units * price_hint * margin_rate
    return projected_used / nav


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
    "micro": (2.0, 4.2),
    # 早めに小利を拾いつつ、2段目は少し先にしてランナーを残す
    "scalp": (1.6, 3.6),
    # 超短期（fast scalp）は利幅が小さいため閾値も縮小
    # まずは小刻みにヘッジしてランナーのみを残す方針
    "scalp_fast": (1.0, 1.8),
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


def _soft_tp_mode(thesis: Optional[dict]) -> bool:
    if not isinstance(thesis, dict):
        return False
    mode = thesis.get("tp_mode")
    if not mode and isinstance(thesis.get("execution"), dict):
        mode = thesis.get("execution", {}).get("tp_mode")
    return str(mode or "").lower() in {"soft_zone", "soft"}


def _encode_thesis_comment(entry_thesis: Optional[dict]) -> Optional[str]:
    """
    Serialize a minimal subset of the thesis into OANDA clientExtensions.comment
    so exit側がリスタート後も fast_cut/kill メタを参照できるようにする。
    """
    if _DISABLE_CLIENT_COMMENT:
        return None
    if not isinstance(entry_thesis, dict):
        return None
    keys = (
        "strategy_tag",
        "profile",
        "tag",
        "fast_cut_pips",
        "fast_cut_time_sec",
        "fast_cut_hard_mult",
        "kill_switch",
        "loss_guard_pips",
        "min_hold_sec",
        "target_tp_pips",
    )
    compact: dict[str, object] = {}
    for key in keys:
        val = entry_thesis.get(key)
        if val in (None, "", False):
            continue
        compact[key] = val
    if not compact:
        return None
    try:
        text = json.dumps(compact, ensure_ascii=True, separators=(",", ":"))
    except Exception:
        return None
    # OANDA comment max 256 chars
    return text[:255]


def _trade_min_hold_seconds(trade: dict, pocket: str) -> float:
    thesis = _coerce_entry_thesis(trade.get("entry_thesis"))
    hold = thesis.get("min_hold_sec") or thesis.get("min_hold_seconds")
    if hold is None:
        try:
            hold = float(thesis.get("min_hold_min") or thesis.get("min_hold_minutes")) * 60.0
        except Exception:
            hold = None
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
    *,
    context: str = "auto",
    ref_price: Optional[float] = None,
) -> None:
    if STOP_LOSS_DISABLED:
        sl_price = None
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

    def _fmt(val: Optional[float]) -> str:
        return "-" if val is None else f"{val:.3f}"

    logging.info(
        "[PROTECT][%s] trade=%s sl=%s->%s tp=%s->%s ref=%s",
        context,
        trade_id,
        _fmt(prev_sl),
        _fmt(current_sl),
        _fmt(prev_tp),
        _fmt(current_tp),
        _fmt(ref_price),
    )

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
            "[ORDER] Failed to update protections trade=%s sl=%s tp=%s ctx=%s: %s",
            trade_id,
            sl_price,
            tp_price,
            context,
            exc,
        )


async def close_trade(
    trade_id: str,
    units: Optional[int] = None,
    client_order_id: Optional[str] = None,
    allow_negative: bool = False,
) -> bool:
    data: Optional[dict[str, str]] = None
    # close 側も client_order_id を必須化。欠損かつ agent 管理外の建玉はスキップして無駄打ちを防ぐ。
    if not client_order_id:
        log_metric("close_skip_missing_client_id", 1.0, tags={"trade_id": str(trade_id)})
        logging.info("[ORDER] skip close trade=%s missing client_id (likely manual/external)", trade_id)
        return False
    if _EXIT_NO_NEGATIVE_CLOSE and not allow_negative:
        pl = _current_trade_unrealized_pl(trade_id)
        if pl is not None and pl <= 0:
            if _should_allow_negative_close(client_order_id):
                allow_negative = True
            if not allow_negative:
                log_metric(
                    "close_blocked_negative",
                    float(pl),
                    tags={"trade_id": str(trade_id)},
                )
                _console_order_log(
                    "CLOSE_REJECT",
                    pocket=None,
                    strategy_tag=None,
                    side=None,
                    units=units,
                    sl_price=None,
                    tp_price=None,
                    client_order_id=client_order_id,
                    ticket_id=str(trade_id),
                    note="no_negative_close",
                )
                _log_order(
                    pocket=None,
                    instrument=None,
                    side=None,
                    units=units,
                    sl_price=None,
                    tp_price=None,
                    client_order_id=client_order_id,
                    status="close_reject_no_negative",
                    attempt=0,
                    ticket_id=str(trade_id),
                    executed_price=None,
                    request_payload={"trade_id": trade_id, "data": {"unrealized_pl": pl}},
                )
                return False
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
    if not client_order_id:
        log_metric("close_missing_client_id", 1.0, tags={"trade_id": str(trade_id)})
        client_order_id = None
    _console_order_log(
        "CLOSE_REQ",
        pocket=None,
        strategy_tag=None,
        side=None,
        units=units,
        sl_price=None,
        tp_price=None,
        client_order_id=client_order_id,
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
            client_order_id=client_order_id,
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
            client_order_id=client_order_id,
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
            client_order_id=client_order_id,
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
            client_order_id=client_order_id,
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
            client_order_id=client_order_id,
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
            client_order_id=client_order_id,
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
            client_order_id=client_order_id,
            ticket_id=str(trade_id),
            note="exception",
        )
        return False


def update_dynamic_protections(
    open_positions: dict,
    fac_m1: dict,
    fac_h4: dict,
) -> None:
    if not TRAILING_SL_ALLOWED:
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
            _maybe_update_protections(
                trade_id,
                sl_price,
                tp_price,
                context="dynamic_protection_v1",
                ref_price=current_price,
            )


def _apply_dynamic_protections_v2(
    open_positions: dict,
    fac_m1: dict,
    fac_h4: dict,
) -> None:
    policy = policy_bus.latest()
    pockets_policy = policy.pockets if policy else {}
    try:
        atr_m1 = float(fac_m1.get("atr_pips") or (fac_m1.get("atr") or 0.0) * 100.0)
    except Exception:
        atr_m1 = 0.0
    try:
        vol_5m = float(fac_m1.get("vol_5m") or 1.0)
    except Exception:
        vol_5m = 1.0

    now_ts = time.time()
    current_price = fac_m1.get("close")
    pip = 0.01
    # 市況（ATR/短期ボラ）を拾ってトリガーを動的調整
    def _coerce(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    atr_m1 = _coerce(fac_m1.get("atr_pips"), _coerce(fac_m1.get("atr"), 0.0) * 100.0)
    vol_5m = _coerce(fac_m1.get("vol_5m"), 1.0)

    defaults = {
        "macro": {"trigger": 6.8, "lock_ratio": 0.55, "min_lock": 2.6, "cooldown": 90.0},
        "micro": {"trigger": 2.8, "lock_ratio": 0.42, "min_lock": 0.60, "cooldown": 60.0},
        # スキャルは利幅を伸ばせるようにトリガー/ロックを緩める
        "scalp": {"trigger": 2.5, "lock_ratio": 0.25, "min_lock": 0.60, "cooldown": 30.0},
    }
    # トレーリング開始を ATR/ボラで動的に決める（micro/scalp）
    base_start_delay = {"micro": 55.0, "scalp": 45.0}

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

        # ATR/ボラに応じて動的スケール
        atr_val = atr_m1
        if pocket == "macro":
            # マクロは大きめのATRを見て少しだけ拡げる
            trigger = max(trigger, atr_val * (1.2 if vol_5m < 1.0 else 1.35 if vol_5m < 1.8 else 1.5))
            min_lock = max(min_lock, atr_val * 0.3)
            lock_ratio = max(lock_ratio, 0.50 if vol_5m >= 1.5 else 0.45)
        elif pocket == "micro":
            if vol_5m < 0.8:
                trigger = max(trigger, atr_val * 1.1)
                lock_ratio = max(lock_ratio, 0.4)
            elif vol_5m > 1.6:
                trigger = max(trigger, atr_val * 1.4)
                lock_ratio = max(lock_ratio, 0.35)
            else:
                trigger = max(trigger, atr_val * 1.25)
                lock_ratio = max(lock_ratio, 0.38)
            min_lock = max(min_lock, atr_val * 0.25)
        elif pocket == "scalp":
            if vol_5m < 0.8:
                trigger = max(trigger, atr_val * 1.0)
                lock_ratio = max(lock_ratio, 0.22)
            elif vol_5m > 1.6:
                trigger = max(trigger, atr_val * 1.3)
                lock_ratio = max(lock_ratio, 0.28)
            else:
                trigger = max(trigger, atr_val * 1.15)
                lock_ratio = max(lock_ratio, 0.25)
            min_lock = max(min_lock, atr_val * 0.25)
        # 経過時間に応じてロック強度を少し引き上げる
        def _age_scaled_lock(age_sec: float, base_ratio: float) -> float:
            if age_sec <= 0:
                return base_ratio
            bump = min(0.2, (age_sec / 180.0) * 0.2)  # 3分で+20%まで
            return min(0.65, base_ratio * (1.0 + bump))

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
            thesis = _coerce_entry_thesis(tr.get("entry_thesis"))
            if _soft_tp_mode(thesis):
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

            # micro/scalp: ATR/ボラに応じた開始遅延＋経過時間でロック強化
            opened_at = _parse_trade_open_time(tr.get("open_time"))
            if pocket in {"micro", "scalp"} and opened_at:
                age_sec = max(0.0, (datetime.now(timezone.utc) - opened_at).total_seconds())
                start_delay = base_start_delay.get(pocket, 45.0)
                start_delay = max(start_delay, atr_val * (22.0 if pocket == "micro" else 18.0))
                if vol_5m > 1.6:
                    start_delay *= 0.85
                elif vol_5m < 0.8:
                    start_delay *= 1.1
                if age_sec < start_delay:
                    continue
                lock_ratio = _age_scaled_lock(age_sec - start_delay, lock_ratio)

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

            _maybe_update_protections(
                trade_id,
                desired_sl,
                tp_price,
                context="dynamic_protection_v2",
                ref_price=current_price,
            )


async def set_trade_protections(
    trade_id: str,
    *,
    sl_price: Optional[float],
    tp_price: Optional[float],
) -> bool:
    """
    Legacy compatibility layer – update SL/TP for an open trade and report success.
    """
    if STOP_LOSS_DISABLED:
        sl_price = None
    if not TRAILING_SL_ALLOWED and sl_price is not None:
        return False
    if not trade_id:
        return False
    try:
        _maybe_update_protections(
            trade_id,
            sl_price,
            tp_price,
            context="legacy_set_trade_protections",
        )
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
    try:
        atr_m1 = float(fac_m1.get("atr_pips") or (fac_m1.get("atr") or 0.0) * 100.0)
    except Exception:
        atr_m1 = 0.0
    try:
        vol_5m = float(fac_m1.get("vol_5m") or 1.0)
    except Exception:
        vol_5m = 1.0
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
        if thresholds and range_mode:
            try:
                atr_hint = float(atr_m1 or 0.0)
            except Exception:
                atr_hint = 0.0
            if atr_hint <= 2.0:
                thresholds = (1.6, 2.6) if pocket == "scalp" else (2.2, 3.6)
            elif atr_hint <= 3.0:
                thresholds = (min(thresholds[0], 2.2), min(thresholds[1], 3.8))
        if thresholds:
            thresholds = _scaled_thresholds(pocket, thresholds, atr_m1, vol_5m)
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
            thesis = _coerce_entry_thesis(tr.get("entry_thesis"))
            if _soft_tp_mode(thesis):
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
    stage_index: Optional[int] = None,
    arbiter_final: bool = False,
) -> Optional[str]:
    """
    units : +10000 = buy 0.1 lot, ‑10000 = sell 0.1 lot
    returns order ticket id（決済のみの fill でも tradeID を返却）
    """
    # client_order_id は必須。欠損したまま送れば OANDA 側で空白になり、追跡不能となる。
    if not client_order_id:
        _console_order_log(
            "OPEN_REJECT",
            pocket=pocket,
            strategy_tag=strategy_tag or "unknown",
            side="buy" if units > 0 else "sell",
            units=units,
            sl_price=sl_price,
            tp_price=tp_price,
            client_order_id="",
            note="missing_client_order_id",
        )
        _log_order(
            pocket=pocket,
            instrument=instrument,
            side="buy" if units > 0 else "sell",
            units=units,
            sl_price=sl_price,
            tp_price=tp_price,
            client_order_id=None,
            status="missing_client_order_id",
            attempt=0,
            request_payload={
                "strategy_tag": strategy_tag,
                "meta": meta,
                "entry_thesis": entry_thesis,
            },
        )
        log_metric(
            "order_missing_client_id",
            1.0,
            tags={"pocket": pocket, "strategy": strategy_tag or "unknown"},
        )
        return None
    if strategy_tag is not None:
        strategy_tag = str(strategy_tag)
        if not strategy_tag:
            strategy_tag = None
    else:
        strategy_tag = None
    virtual_sl_price: Optional[float] = None
    virtual_tp_price: Optional[float] = None
    side_label = "buy" if units > 0 else "sell"

    def _merge_virtual(payload: Optional[dict] = None) -> dict:
        base: dict = {}
        if virtual_sl_price is not None:
            base["virtual_sl_price"] = virtual_sl_price
        if virtual_tp_price is not None:
            base["virtual_tp_price"] = virtual_tp_price
        if payload:
            base.update(payload)
        return base

    def log_order(**kwargs):
        kwargs["request_payload"] = _merge_virtual(kwargs.get("request_payload"))
        return _log_order(**kwargs)
    thesis_sl_pips: Optional[float] = None
    thesis_tp_pips: Optional[float] = None
    soft_tp = _soft_tp_mode(entry_thesis)
    if stage_index is None and isinstance(entry_thesis, dict):
        try:
            raw_stage = entry_thesis.get("stage_index")
            if raw_stage is not None:
                stage_index = int(raw_stage)
        except Exception:
            stage_index = None
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
    if soft_tp:
        tp_price = None
        thesis_tp_pips = None

    # strategy_tag は必須: entry_thesis から補完しても欠損なら拒否
    raw_tag = None
    if isinstance(entry_thesis, dict):
        raw_tag = entry_thesis.get("strategy_tag") or entry_thesis.get("strategy")
    if not strategy_tag and raw_tag:
        strategy_tag = str(raw_tag)
    if isinstance(entry_thesis, dict) and strategy_tag and not entry_thesis.get("strategy_tag"):
        entry_thesis = dict(entry_thesis)
        entry_thesis["strategy_tag"] = strategy_tag
    if isinstance(entry_thesis, dict) and not reduce_only:
        entry_thesis = attach_section_axis(entry_thesis, pocket=pocket)

    # strategy_tag も必須。entry_thesis から補完した上で欠損なら拒否。
    if not strategy_tag:
        _console_order_log(
            "OPEN_REJECT",
            pocket=pocket,
            strategy_tag="missing",
            side="buy" if units > 0 else "sell",
            units=units,
            sl_price=sl_price,
            tp_price=tp_price,
            client_order_id=client_order_id,
            note="missing_strategy_tag",
        )
        _log_order(
            pocket=pocket,
            instrument=instrument,
            side="buy" if units > 0 else "sell",
            units=units,
            sl_price=sl_price,
            tp_price=tp_price,
            client_order_id=client_order_id,
            status="missing_strategy_tag",
            attempt=0,
            request_payload={"meta": meta, "entry_thesis": entry_thesis},
        )
        log_metric(
            "order_missing_strategy_tag",
            1.0,
            tags={"pocket": pocket, "strategy": "missing"},
        )
        return None

    entry_price = None
    if not reduce_only and pocket != "manual":
        entry_price = _entry_price_hint(entry_thesis, meta)
        if entry_price is not None:
            guard = evaluate_entry_guard(
                entry_price=entry_price,
                side="long" if units > 0 else "short",
                pocket=pocket,
                strategy_tag=strategy_tag,
                entry_thesis=entry_thesis,
            )
            if guard.allowed and guard.reason:
                log_metric(
                    "entry_guard_bypass",
                    1.0,
                    tags={
                        "pocket": pocket,
                        "strategy": strategy_tag or "unknown",
                        "reason": guard.reason,
                    },
                )
            if not guard.allowed:
                _console_order_log(
                    "OPEN_REJECT",
                    pocket=pocket,
                    strategy_tag=strategy_tag or "unknown",
                    side=side_label,
                    units=units,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    client_order_id=client_order_id,
                    note=guard.reason or "entry_guard",
                )
                log_order(
                    pocket=pocket,
                    instrument=instrument,
                    side=side_label,
                    units=units,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    client_order_id=client_order_id,
                    status="entry_guard_block",
                    attempt=0,
                    request_payload={
                        "reason": guard.reason,
                        "entry_price": entry_price,
                        "entry_guard": guard.debug,
                        "entry_thesis": entry_thesis,
                        "meta": meta,
                    },
                )
                log_metric(
                    "entry_guard_block",
                    1.0,
                    tags={
                        "pocket": pocket,
                        "strategy": strategy_tag or "unknown",
                        "reason": guard.reason or "entry_guard",
                    },
            )
            return None

    if not reduce_only and pocket != "manual":
        tech_entry_price = entry_price or _entry_price_hint(entry_thesis, meta)
        if tech_entry_price is not None:
            tech = evaluate_entry_techniques(
                entry_price=tech_entry_price,
                side="long" if units > 0 else "short",
                pocket=pocket,
                strategy_tag=strategy_tag,
                entry_thesis=entry_thesis,
            )
            if not tech.allowed:
                _console_order_log(
                    "OPEN_REJECT",
                    pocket=pocket,
                    strategy_tag=strategy_tag or "unknown",
                    side=side_label,
                    units=units,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    client_order_id=client_order_id,
                    note="tech_entry_block",
                )
                log_order(
                    pocket=pocket,
                    instrument=instrument,
                    side=side_label,
                    units=units,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    client_order_id=client_order_id,
                    status="entry_tech_block",
                    attempt=0,
                    request_payload={
                        "entry_price": tech_entry_price,
                        "tech_decision": tech.debug,
                        "tech_reasons": list(tech.reasons),
                        "entry_thesis": entry_thesis,
                        "meta": meta,
                    },
                )
                log_metric(
                    "entry_tech_block",
                    1.0,
                    tags={
                        "pocket": pocket,
                        "strategy": strategy_tag or "unknown",
                        "reason": "score_block",
                    },
                )
                return None
            if isinstance(entry_thesis, dict):
                entry_thesis = dict(entry_thesis)
                entry_thesis["tech_entry"] = {
                    "score": round(tech.score, 3),
                    "multiplier": round(tech.size_multiplier, 3),
                    "reasons": list(tech.reasons)[:6],
                }
            if tech.size_multiplier and abs(tech.size_multiplier - 1.0) >= 0.01:
                before_units = units
                units = _apply_tech_units(units, tech.size_multiplier, pocket)
                if units != before_units:
                    log_metric(
                        "entry_tech_multiplier",
                        tech.size_multiplier,
                        tags={"pocket": pocket, "strategy": strategy_tag or "unknown"},
                    )
            log_metric(
                "entry_tech_score",
                tech.score,
                tags={"pocket": pocket, "strategy": strategy_tag or "unknown"},
            )

    # 成績ガード（直近 PF/勝率が悪いタグは全ポケットでブロック。manual は除外）
    perf_guard_enabled = os.getenv("PERF_GUARD_GLOBAL_ENABLED", "0").strip().lower() not in {
        "",
        "0",
        "false",
        "no",
    }
    if perf_guard_enabled and pocket != "manual" and strategy_tag:
        try:
            current_hour = datetime.now(timezone.utc).hour
        except Exception:
            current_hour = None
        decision = perf_guard.is_allowed(str(strategy_tag), pocket, hour=current_hour)
        if not decision.allowed:
            note = f"perf_block:{decision.reason}"
            _console_order_log(
                "OPEN_REJECT",
                pocket=pocket,
                strategy_tag=strategy_tag,
                side=side_label,
                units=units,
                sl_price=sl_price,
                tp_price=tp_price,
                client_order_id=client_order_id,
                note=note,
            )
            log_order(
                pocket=pocket,
                instrument=instrument,
                side=side_label,
                units=units,
                sl_price=sl_price,
                tp_price=tp_price,
                client_order_id=client_order_id,
                status="perf_block",
                attempt=0,
                stage_index=stage_index,
                request_payload={"strategy_tag": strategy_tag, "meta": meta, "entry_thesis": entry_thesis},
            )
            log_metric(
                "order_perf_block",
                1.0,
                tags={
                    "pocket": pocket,
                    "strategy": str(strategy_tag),
                    "reason": decision.reason,
                },
            )
            return None

    exec_cfg = None
    if isinstance(entry_thesis, dict):
        exec_cfg = entry_thesis.get("execution")
    if exec_cfg is None and isinstance(meta, dict):
        exec_cfg = meta.get("execution")
    if (
        not reduce_only
        and isinstance(exec_cfg, dict)
        and exec_cfg.get("order_policy") == "market_guarded"
    ):
        ideal = _as_float(exec_cfg.get("ideal_entry"))
        chase_max = _as_float(exec_cfg.get("chase_max"))
        price_hint = (
            _as_float((meta or {}).get("entry_price"))
            or _as_float((entry_thesis or {}).get("entry_ref") if isinstance(entry_thesis, dict) else None)
            or _estimate_price(meta)
            or _latest_mid_price()
        )
        if ideal is not None and chase_max is not None and price_hint is not None:
            if abs(price_hint - ideal) > chase_max:
                _console_order_log(
                    "OPEN_SKIP",
                    pocket=pocket,
                    strategy_tag=str(strategy_tag or "unknown"),
                    side=side_label,
                    units=units,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    client_order_id=client_order_id,
                    note="market_guarded",
                )
                log_order(
                    pocket=pocket,
                    instrument=instrument,
                    side=side_label,
                    units=units,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    client_order_id=client_order_id,
                    status="market_guarded_skip",
                    attempt=0,
                    stage_index=stage_index,
                    request_payload={
                        "price_hint": price_hint,
                        "ideal_entry": ideal,
                        "chase_max": chase_max,
                        "execution": exec_cfg,
                    },
                )
                return None

    # 強制マージンガード（reduce_only 以外）。直近スナップショットから
    # 現在の使用率と注文後の想定使用率を確認し、上限超えは即リジェクト。
    if not reduce_only:
        try:
            from utils.oanda_account import get_account_snapshot
        except Exception:
            get_account_snapshot = None  # type: ignore
        if get_account_snapshot is not None:
            try:
                snap = get_account_snapshot(cache_ttl_sec=1.0)
            except Exception as exc:
                note = "margin_snapshot_failed"
                logging.warning("[ORDER] margin guard snapshot failed: %s", exc)
                _console_order_log(
                    "OPEN_REJECT",
                    pocket=pocket,
                    strategy_tag=strategy_tag or "unknown",
                    side=side_label,
                    units=units,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    client_order_id=client_order_id,
                    note=note,
                )
                log_order(
                    pocket=pocket,
                    instrument=instrument,
                    side=side_label,
                    units=units,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    client_order_id=client_order_id,
                    status=note,
                    attempt=0,
                    stage_index=stage_index,
                    request_payload={
                        "strategy_tag": strategy_tag,
                        "meta": meta,
                        "entry_thesis": entry_thesis,
                        "error": str(exc),
                    },
                )
                log_metric(
                    "order_margin_block",
                    1.0,
                    tags={
                        "pocket": pocket,
                        "strategy": strategy_tag or "unknown",
                        "reason": note,
                    },
                )
                return None
            try:
                nav = float(snap.nav or 0.0)
                margin_used = float(snap.margin_used or 0.0)
                margin_rate = float(snap.margin_rate or 0.0)
                soft_cap = min(float(os.getenv("MAX_MARGIN_USAGE", "0.92") or 0.92), 0.99)
                hard_cap = min(float(os.getenv("MAX_MARGIN_USAGE_HARD", "0.96") or 0.96), 0.995)
                cap = min(hard_cap, max(soft_cap, 0.0))
                net_reducing = False
                net_before_units = 0.0
                try:
                    from utils.oanda_account import get_position_summary

                    long_u, short_u = get_position_summary()
                    net_before_units = float(long_u) - float(short_u)
                    net_after_units = (
                        net_before_units + abs(units) if side_label.lower() == "buy" else net_before_units - abs(units)
                    )
                    net_reducing = abs(net_after_units) < abs(net_before_units)
                except Exception:
                    net_reducing = False
                    net_before_units = 0.0
                if nav > 0:
                    usage = margin_used / nav
                    projected_usage = _projected_usage_with_netting(
                        nav,
                        margin_rate,
                        side_label,
                        units,
                        margin_used=margin_used,
                        meta=meta,
                    )
                    usage_for_cap = projected_usage if projected_usage is not None else usage
                    if (
                        usage_for_cap >= hard_cap * 0.995
                        and not (
                            net_reducing
                            and projected_usage is not None
                            and usage is not None
                            and projected_usage < usage
                        )
                    ):
                        price_hint = _estimate_price(meta) or _latest_mid_price() or 0.0
                        scaled_units = 0
                        cap_target = hard_cap * 0.99
                        if projected_usage and projected_usage > 0 and abs(units) > 0:
                            factor = cap_target / projected_usage
                            scaled_units = int(math.floor(abs(units) * factor))
                        elif nav > 0 and margin_rate > 0 and price_hint > 0:
                            try:
                                allowed_net = (cap_target * nav) / (price_hint * margin_rate)
                                room = allowed_net - abs(net_before_units)
                                scaled_units = int(math.floor(min(abs(units), room)))
                            except Exception:
                                scaled_units = 0
                        if scaled_units > 0:
                            new_units = scaled_units if units > 0 else -scaled_units
                            logging.info(
                                "[ORDER] margin cap scale units %s -> %s usage=%.3f cap=%.3f",
                                units,
                                new_units,
                                usage_for_cap,
                                cap_target,
                            )
                            units = new_units
                        else:
                            note = "margin_usage_exceeds_cap"
                            _console_order_log(
                                "OPEN_REJECT",
                                pocket=pocket,
                                strategy_tag=strategy_tag or "unknown",
                                side=side_label,
                                units=units,
                                sl_price=sl_price,
                                tp_price=tp_price,
                                client_order_id=client_order_id,
                                note=note,
                            )
                            log_order(
                                pocket=pocket,
                                instrument=instrument,
                                side=side_label,
                                units=units,
                                sl_price=sl_price,
                                tp_price=tp_price,
                                client_order_id=client_order_id,
                                status=note,
                                attempt=0,
                                stage_index=stage_index,
                                request_payload={
                                    "strategy_tag": strategy_tag,
                                    "meta": meta,
                                    "entry_thesis": entry_thesis,
                                    "margin_usage": usage,
                                    "projected_usage": projected_usage,
                                    "cap": hard_cap,
                                },
                            )
                            log_metric(
                                "order_margin_block",
                                1.0,
                                tags={
                                    "pocket": pocket,
                                    "strategy": strategy_tag or "unknown",
                                    "reason": note,
                                },
                            )
                            return None
                    if (
                        usage_for_cap >= hard_cap * 0.995
                        and net_reducing
                        and projected_usage is not None
                        and usage is not None
                        and projected_usage < usage
                    ):
                        logging.info(
                            "[ORDER] allow net-reducing order usage=%.3f->%.3f cap=%.3f units=%d",
                            usage,
                            projected_usage,
                            hard_cap,
                            units,
                        )
                price_hint = _estimate_price(meta) or 0.0
                projected_usage = None
                if nav > 0 and margin_rate > 0:
                    projected_usage = _projected_usage_with_netting(
                        nav,
                        margin_rate,
                        side_label,
                        units,
                        margin_used=margin_used,
                        meta=meta,
                    )
                    if projected_usage is None and price_hint > 0:
                        # フォールバック: 片側加算のみ
                        projected_used = margin_used + abs(units) * price_hint * margin_rate
                        projected_usage = projected_used / nav

                if (
                    projected_usage is not None
                    and projected_usage >= cap
                    and not (net_reducing and usage is not None and projected_usage < usage)
                ):
                    price_hint = _estimate_price(meta) or _latest_mid_price() or 0.0
                    scaled_units = 0
                    cap_target = cap * 0.99
                    try:
                        factor = cap_target / projected_usage if projected_usage > 0 else 0.0
                        if factor > 0 and abs(units) > 0:
                            scaled_units = int(math.floor(abs(units) * factor))
                        elif nav > 0 and margin_rate > 0 and price_hint > 0:
                            allowed_net = (cap_target * nav) / (price_hint * margin_rate)
                            room = allowed_net - abs(net_before_units)
                            scaled_units = int(math.floor(min(abs(units), room)))
                    except Exception:
                        scaled_units = 0
                    if scaled_units > 0:
                        new_units = scaled_units if units > 0 else -scaled_units
                        logging.info(
                            "[ORDER] projected margin scale units %s -> %s usage=%.3f cap=%.3f",
                            units,
                            new_units,
                            projected_usage,
                            cap_target,
                        )
                        units = new_units
                    else:
                        note = "margin_usage_projected_cap"
                        _console_order_log(
                            "OPEN_REJECT",
                            pocket=pocket,
                            strategy_tag=strategy_tag or "unknown",
                            side=side_label,
                            units=units,
                            sl_price=sl_price,
                            tp_price=tp_price,
                            client_order_id=client_order_id,
                            note=note,
                        )
                        log_order(
                            pocket=pocket,
                            instrument=instrument,
                            side=side_label,
                            units=units,
                            sl_price=sl_price,
                            tp_price=tp_price,
                            client_order_id=client_order_id,
                            status=note,
                            attempt=0,
                            stage_index=stage_index,
                            request_payload={
                                "strategy_tag": strategy_tag,
                                "meta": meta,
                                "entry_thesis": entry_thesis,
                                "projected_usage": projected_usage,
                                "cap": cap,
                            },
                        )
                        log_metric(
                            "order_margin_block",
                            1.0,
                            tags={
                                "pocket": pocket,
                                "strategy": strategy_tag or "unknown",
                                "reason": note,
                            },
                        )
                        return None
                if (
                    projected_usage is not None
                    and projected_usage >= cap
                    and net_reducing
                    and usage is not None
                    and projected_usage < usage
                ):
                    logging.info(
                        "[ORDER] allow net-reducing projected usage=%.3f->%.3f cap=%.3f units=%d",
                        usage,
                        projected_usage,
                        cap,
                        units,
                    )
            except Exception as exc:  # pragma: no cover - defensive
                note = "margin_guard_error"
                logging.warning("[ORDER] margin guard error: %s", exc)
                _console_order_log(
                    "OPEN_REJECT",
                    pocket=pocket,
                    strategy_tag=strategy_tag or "unknown",
                    side=side_label,
                    units=units,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    client_order_id=client_order_id,
                    note=note,
                )
                log_order(
                    pocket=pocket,
                    instrument=instrument,
                    side=side_label,
                    units=units,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    client_order_id=client_order_id,
                    status=note,
                    attempt=0,
                    stage_index=stage_index,
                    request_payload={
                        "strategy_tag": strategy_tag,
                        "meta": meta,
                        "entry_thesis": entry_thesis,
                        "error": str(exc),
                    },
                )
                log_metric(
                    "order_margin_block",
                    1.0,
                    tags={
                        "pocket": pocket,
                        "strategy": strategy_tag or "unknown",
                        "reason": note,
                    },
                )
                return None
    if not strategy_tag:
        _console_order_log(
            "OPEN_REJECT",
            pocket=pocket,
            strategy_tag="missing_tag",
            side="buy" if units > 0 else "sell",
            units=units,
            sl_price=sl_price,
            tp_price=tp_price,
            client_order_id=client_order_id,
            note="missing_strategy_tag",
        )
        log_order(
            pocket=pocket,
            instrument=instrument,
            side="buy" if units > 0 else "sell",
            units=units,
            sl_price=sl_price,
            tp_price=tp_price,
            client_order_id=client_order_id,
            status="missing_strategy_tag",
            attempt=0,
            request_payload={
                "strategy_tag": strategy_tag,
                "meta": meta,
                "entry_thesis": entry_thesis,
            },
        )
        log_metric(
            "order_missing_strategy_tag",
            1.0,
            tags={"pocket": pocket, "side": "buy" if units > 0 else "sell"},
        )
        return None
    side_label = "buy" if units > 0 else "sell"

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
            log_order(
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
            log_order(
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

    if STOP_LOSS_DISABLED:
        sl_price = None

    if (
        _FORWARD_TO_SIGNAL_GATE
        and not reduce_only
        and not arbiter_final
    ):
        price_hint = entry_price_meta or _estimate_price(meta) or _latest_mid_price()
        sl_pips_hint = _as_float((entry_thesis or {}).get("sl_pips")) if isinstance(entry_thesis, dict) else None
        if sl_pips_hint is None and isinstance(entry_thesis, dict):
            for alt in ("profile_sl_pips", "loss_guard_pips", "hard_stop_pips"):
                alt_val = _as_float(entry_thesis.get(alt))
                if alt_val:
                    sl_pips_hint = alt_val
                    break
        tp_pips_hint = _as_float((entry_thesis or {}).get("tp_pips")) if isinstance(entry_thesis, dict) else None
        if tp_pips_hint is None and isinstance(entry_thesis, dict):
            for alt in ("profile_tp_pips", "target_tp_pips"):
                alt_val = _as_float(entry_thesis.get(alt))
                if alt_val:
                    tp_pips_hint = alt_val
                    break
        if sl_pips_hint is None and price_hint is not None and sl_price is not None:
            sl_pips_hint = abs(price_hint - sl_price) / 0.01
        if tp_pips_hint is None and price_hint is not None and tp_price is not None:
            tp_pips_hint = abs(price_hint - tp_price) / 0.01
        try:
            conf_val = int(confidence if confidence is not None else (entry_thesis or {}).get("confidence", 50))
        except Exception:
            conf_val = 50
        conf_val = max(0, min(100, conf_val))
        payload = {
            "source": "order_manager",
            "strategy": strategy_tag,
            "pocket": pocket,
            "action": "OPEN_LONG" if units > 0 else "OPEN_SHORT",
            "confidence": conf_val,
            "sl_pips": sl_pips_hint,
            "tp_pips": tp_pips_hint,
            "sl_price": sl_price,
            "tp_price": tp_price,
            "entry_price": price_hint,
            "client_order_id": client_order_id,
            "proposed_units": abs(units),
            "entry_type": (entry_thesis or {}).get("entry_type"),
            "entry_thesis": entry_thesis,
            "meta": meta or {},
        }
        try:
            signal_bus.enqueue(payload)
            log_order(
                pocket=pocket,
                instrument=instrument,
                side="buy" if units > 0 else "sell",
                units=units,
                sl_price=sl_price,
                tp_price=tp_price,
                client_order_id=client_order_id,
                status="queued_to_gate",
                attempt=0,
                stage_index=stage_index,
                request_payload=payload,
            )
            _console_order_log(
                "OPEN_QUEUE",
                pocket=pocket,
                strategy_tag=strategy_tag or "unknown",
                side=side_label,
                units=units,
                sl_price=sl_price,
                tp_price=tp_price,
                client_order_id=client_order_id,
                note="signal_gate",
            )
            return None
        except Exception as exc:
            logging.warning("[ORDER_GATE] enqueue failed, fall back to live order: %s", exc)

    if (
        _DYNAMIC_SL_ENABLE
        and (pocket or "").lower() in _DYNAMIC_SL_POCKETS
        and not reduce_only
        and entry_price_meta is not None
        and not STOP_LOSS_DISABLED
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
        log_order(
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
            log_order(
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
    original_units = units
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
        preflight_units = requested_units
        logging.info(
            "[ORDER] units clamped to pocket minimum pocket=%s requested=%s -> %s",
            pocket,
            original_units,
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
            log_order(
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
        norm_sl = None if STOP_LOSS_DISABLED else sl_price
        norm_tp = tp_price
        norm_sl, norm_tp, normalized = _normalize_protections(
            estimated_entry,
            norm_sl,
            norm_tp,
            units > 0,
        )
        sl_price = None if STOP_LOSS_DISABLED else norm_sl
        tp_price = norm_tp
        if normalized:
            logging.debug(
                "[ORDER] normalized SL/TP client=%s sl=%s tp=%s entry=%.3f",
                client_order_id,
                f"{sl_price:.3f}" if sl_price is not None else "None",
                f"{tp_price:.3f}" if tp_price is not None else "None",
                estimated_entry,
            )

    # Virtual SL/TP logging (even if SL is disabled)
    if estimated_entry is not None:
        if thesis_sl_pips is not None:
            virtual_sl_price = round(
                estimated_entry - thesis_sl_pips * 0.01, 3
            ) if units > 0 else round(estimated_entry + thesis_sl_pips * 0.01, 3)
        if thesis_tp_pips is not None:
            virtual_tp_price = round(
                estimated_entry + thesis_tp_pips * 0.01, 3
            ) if units > 0 else round(estimated_entry - thesis_tp_pips * 0.01, 3)
    if sl_price is not None:
        virtual_sl_price = sl_price
    if tp_price is not None:
        virtual_tp_price = tp_price

    # Directional exposure cap: scale down instead of rejecting
    if not reduce_only:
        adjusted = _apply_directional_cap(preflight_units, pocket, side_label, meta)
        if adjusted == 0:
            _console_order_log(
                "OPEN_SKIP",
                pocket=pocket,
                strategy=strategy_tag,
                side=side_label,
                units=preflight_units,
                sl_price=sl_price,
                tp_price=tp_price,
                client_order_id=client_order_id,
                note="dir_cap",
            )
            return None
        if adjusted != preflight_units:
            logging.info(
                "[ORDER] dir_cap scaled units %s -> %s pocket=%s side=%s",
                preflight_units,
                adjusted,
                pocket,
                side_label,
            )
            preflight_units = adjusted

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
            log_order(
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

    comment = _encode_thesis_comment(entry_thesis)
    client_ext = {"tag": f"pocket={pocket}"}
    trade_ext = {"tag": f"pocket={pocket}"}
    if comment:
        client_ext["comment"] = comment
        trade_ext["comment"] = comment
    order_data = {
        "order": {
            "type": "MARKET",
            "instrument": instrument,
            "units": str(preflight_units),
            "timeInForce": "FOK",
            "positionFill": "REDUCE_ONLY" if reduce_only else POSITION_FILL,
            "clientExtensions": client_ext,
            "tradeClientExtensions": trade_ext,
        }
    }
    if client_order_id:
        order_data["order"]["clientExtensions"]["id"] = client_order_id
        order_data["order"]["tradeClientExtensions"]["id"] = client_order_id
    if (not STOP_LOSS_DISABLED) and sl_price is not None and not _EXIT_NO_NEGATIVE_CLOSE:
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
        attempt_payload = _merge_virtual(attempt_payload)
        log_order(
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
                log_order(
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
                    if STOP_LOSS_DISABLED:
                        fallback_sl = None
                    if fallback_sl is not None or fallback_tp is not None:
                        if fallback_sl is not None and not _EXIT_NO_NEGATIVE_CLOSE:
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
                log_order(
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
                    # Keep the original request for post-hoc analysis (side/units/TP含む)
                    request_payload=attempt_payload,
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
                target_sl = None if STOP_LOSS_DISABLED else sl_price
                _maybe_update_protections(
                    trade_id,
                    target_sl,
                    tp_price,
                    context="on_fill_protection",
                    ref_price=executed_price,
                )
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
            log_order(
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
            log_order(
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
            return None
        except Exception as exc:
            logging.exception(
                "Unexpected error submitting order (attempt %d): %s",
                attempt + 1,
                exc,
            )
            log_order(
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

    if _soft_tp_mode(entry_thesis):
        tp_price = None

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

    if not reduce_only and pocket != "manual":
        strategy_tag = _strategy_tag_from_thesis(entry_thesis)
        guard = evaluate_entry_guard(
            entry_price=price,
            side="long" if units > 0 else "short",
            pocket=pocket,
            strategy_tag=strategy_tag,
            entry_thesis=entry_thesis,
        )
        if guard.allowed and guard.reason:
            log_metric(
                "entry_guard_bypass",
                1.0,
                tags={
                    "pocket": pocket,
                    "strategy": strategy_tag or "unknown",
                    "reason": guard.reason,
                },
            )
        if not guard.allowed:
            _log_order(
                pocket=pocket,
                instrument=instrument,
                side="buy" if units > 0 else "sell",
                units=units,
                sl_price=sl_price,
                tp_price=tp_price,
                client_order_id=client_order_id,
                status="entry_guard_block",
                attempt=0,
                request_payload={
                    "reason": guard.reason,
                    "entry_price": price,
                    "entry_guard": guard.debug,
                    "entry_thesis": entry_thesis,
                    "meta": meta,
                },
            )
            log_metric(
                "entry_guard_block",
                1.0,
                tags={
                    "pocket": pocket,
                    "strategy": strategy_tag or "unknown",
                    "reason": guard.reason or "entry_guard",
                },
            )
            return None, None

        tech = evaluate_entry_techniques(
            entry_price=price,
            side="long" if units > 0 else "short",
            pocket=pocket,
            strategy_tag=strategy_tag,
            entry_thesis=entry_thesis,
        )
        if not tech.allowed:
            _log_order(
                pocket=pocket,
                instrument=instrument,
                side="buy" if units > 0 else "sell",
                units=units,
                sl_price=sl_price,
                tp_price=tp_price,
                client_order_id=client_order_id,
                status="entry_tech_block",
                attempt=0,
                request_payload={
                    "entry_price": price,
                    "tech_decision": tech.debug,
                    "tech_reasons": list(tech.reasons),
                    "entry_thesis": entry_thesis,
                    "meta": meta,
                },
            )
            log_metric(
                "entry_tech_block",
                1.0,
                tags={
                    "pocket": pocket,
                    "strategy": strategy_tag or "unknown",
                    "reason": "score_block",
                },
            )
            return None, None
        if isinstance(entry_thesis, dict):
            entry_thesis = dict(entry_thesis)
            entry_thesis["tech_entry"] = {
                "score": round(tech.score, 3),
                "multiplier": round(tech.size_multiplier, 3),
                "reasons": list(tech.reasons)[:6],
            }
        if tech.size_multiplier and abs(tech.size_multiplier - 1.0) >= 0.01:
            before_units = units
            units = _apply_tech_units(units, tech.size_multiplier, pocket)
            if units != before_units:
                log_metric(
                    "entry_tech_multiplier",
                    tech.size_multiplier,
                    tags={"pocket": pocket, "strategy": strategy_tag or "unknown"},
                )
        log_metric(
            "entry_tech_score",
            tech.score,
            tags={"pocket": pocket, "strategy": strategy_tag or "unknown"},
        )

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

    comment = _encode_thesis_comment(entry_thesis)
    client_ext = {"tag": f"pocket={pocket}"}
    trade_ext = {"tag": f"pocket={pocket}"}
    if comment:
        client_ext["comment"] = comment
        trade_ext["comment"] = comment

    payload = {
        "order": {
            "type": "LIMIT",
            "instrument": instrument,
            "units": str(units),
            "price": f"{price:.3f}",
            "timeInForce": time_in_force,
            "positionFill": "REDUCE_ONLY" if reduce_only else POSITION_FILL,
            "clientExtensions": client_ext,
            "tradeClientExtensions": trade_ext,
        }
    }
    if gtd_time:
        payload["order"]["gtdTime"] = gtd_time
    if client_order_id:
        payload["order"]["clientExtensions"]["id"] = client_order_id
        payload["order"]["tradeClientExtensions"]["id"] = client_order_id
    if (not STOP_LOSS_DISABLED) and sl_price is not None and not _EXIT_NO_NEGATIVE_CLOSE:
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
        request_payload=attempt_payload,
        response_payload=response,
    )

    if trade_id:
        return trade_id, order_id

    return None, order_id
