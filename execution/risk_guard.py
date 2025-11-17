"""
execution.risk_guard
~~~~~~~~~~~~~~~~~~~~
• 許容 lot 計算
• SL/TP クランプ
• Pocket 別ドローダウン監視
• グローバルエクスポージャ（手動玉＋bot）を 92% 以下へ抑制
"""

from __future__ import annotations

import logging
import os
import pathlib
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

from utils.metrics_logger import log_metric
from utils.secrets import get_secret

# --- risk params ---
MAX_LEVERAGE = 20.0  # 1:20
MAX_LOT = 3.0  # 1 lot = 100k 通貨
POCKET_DD_LIMITS = {"micro": 0.05, "macro": 0.15, "scalp": 0.03}  # equity 比 (%)
GLOBAL_DD_LIMIT = 0.20  # 全体ドローダウン 20%
# Pocketごとの口座配分上限（DD母数の推定に使用）
# Macro は 30% に制限（以前は 80%）。
POCKET_MAX_RATIOS = {"macro": 0.3, "micro": 0.6, "scalp": 0.25}
_DEFAULT_BASE_EQUITY = {"macro": 8000.0, "micro": 6000.0, "scalp": 2500.0}
_LOOKBACK_DAYS = 7

_DISABLE_POCKET_DD = os.getenv("DISABLE_POCKET_DD", "false").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_DISABLE_GLOBAL_DD = os.getenv("DISABLE_GLOBAL_DD", "false").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
EXPOSURE_MAX_RATIO = float(os.getenv("EXPOSURE_MAX_RATIO", "0.93"))

_DB = pathlib.Path("logs/trades.db")
con = sqlite3.connect(_DB)
con.row_factory = sqlite3.Row

_POCKET_EQUITY_HINT: Dict[str, float] = {
    pocket: _DEFAULT_BASE_EQUITY[pocket] for pocket in _DEFAULT_BASE_EQUITY
}

_LOSS_COOLDOWN_CACHE: Dict[str, float] = {}


def _parse_close_time(value: str | None) -> datetime | None:
    if not value:
        return None
    candidate = value.strip()
    if not candidate:
        return None
    candidate = candidate.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(candidate).astimezone(timezone.utc)
    except ValueError:
        # Try trimming fractional seconds
        try:
            if "." in candidate:
                head, tail = candidate.split(".", 1)
                if "+" in tail:
                    frac, tz = tail.split("+", 1)
                    tz = "+" + tz
                elif "-" in tail[6:]:
                    frac, tz = tail.split("-", 1)
                    tz = "-" + tz
                else:
                    frac, tz = tail, "+00:00"
                frac = frac[:6].ljust(6, "0")
                return datetime.fromisoformat(f"{head}.{frac}{tz}").astimezone(timezone.utc)
        except Exception:
            return None
    return None


def update_dd_context(account_equity: float, weight_macro: float, scalp_share: float) -> None:
    """最新の口座残高とポケット配分ヒントを共有し、DD 判定の母数を更新する。"""
    if account_equity <= 0:
        return

    macro_ratio = min(max(weight_macro, 0.0), POCKET_MAX_RATIOS["macro"])
    remainder = max(1.0 - macro_ratio, 0.0)
    scalp_ratio = min(max(scalp_share, 0.0) * remainder, POCKET_MAX_RATIOS["scalp"])
    micro_ratio = max(remainder - scalp_ratio, 0.0)
    micro_ratio = min(micro_ratio, POCKET_MAX_RATIOS["micro"])

    ratios = {
        "macro": macro_ratio,
        "micro": micro_ratio,
        "scalp": scalp_ratio,
    }

    for pocket, ratio in ratios.items():
        if ratio <= 0:
            # 直近で配分ゼロなら既存値を維持（オープンポジションがある可能性がある）
            continue
        _POCKET_EQUITY_HINT[pocket] = max(account_equity * ratio, 1.0)


def _pocket_dd(pocket: str) -> float:
    rows = con.execute(
        """
        SELECT COALESCE(SUM(realized_pl), 0) AS loss_sum
        FROM trades
        WHERE pocket = ?
          AND realized_pl < 0
          AND substr(close_time, 1, 10) >= date('now', ?)
        """,
        (pocket, f"-{_LOOKBACK_DAYS} day"),
    ).fetchone()
    loss_jpy = abs(rows["loss_sum"]) if rows else 0.0
    equity = _POCKET_EQUITY_HINT.get(pocket, _DEFAULT_BASE_EQUITY[pocket])
    if equity <= 0:
        return 1.0
    return loss_jpy / equity


def check_global_drawdown() -> bool:
    """口座全体のドローダウンが閾値を超えているかチェック"""
    if _DISABLE_GLOBAL_DD:
        return False
    # 全ての取引の損益合計を取得
    rows = con.execute("SELECT SUM(pl_pips) FROM trades").fetchone()
    total_pl_pips = rows[0] or 0.0

    # 損失の場合のみドローダウンとして計算
    if total_pl_pips >= 0:
        return False  # 利益が出ているか、損失がない場合はドローダウンなし

    # 10万pips = 100% equity と近似してドローダウン率を計算
    drawdown_percentage = abs(total_pl_pips) / 100000.0
    print(
        f"[RISK] Global Drawdown: {drawdown_percentage:.2%} (Limit: {GLOBAL_DD_LIMIT:.2%})"
    )
    return drawdown_percentage >= GLOBAL_DD_LIMIT


def can_trade(pocket: str) -> bool:
    if _DISABLE_POCKET_DD:
        return True
    return _pocket_dd(pocket) < POCKET_DD_LIMITS[pocket]


def allowed_lot(
    equity: float,
    sl_pips: float,
    *,
    margin_available: float | None = None,
    price: float | None = None,
    margin_rate: float | None = None,
    risk_pct_override: float | None = None,
) -> float:
    """
    口座全体の許容ロットを概算する。
    sl_pips: 損切り幅（pip単位）
    margin_available: 利用可能証拠金
    price: 現在値（USD/JPY mid）
    margin_rate: OANDA口座の証拠金率
    """
    if sl_pips <= 0:
        return 0.0

    # Allow override from config/env or environment: key "risk_pct" (e.g. 0.01 = 1%)
    try:
        risk_pct_str = get_secret("risk_pct")
        risk_pct = float(risk_pct_str)
        if not (0.0001 <= risk_pct <= 0.2):
            raise ValueError("out_of_range")
    except Exception:
        # safer default under drawdown pressure
        risk_pct = 0.01
    if risk_pct_override is not None:
        risk_pct = max(0.0005, min(risk_pct_override, 0.25))
    risk_amount = equity * risk_pct
    lot = risk_amount / (sl_pips * 1000)  # USD/JPYの1lotは1000JPY/pip ≒ 1000

    if margin_available is not None and price is not None and margin_rate:
        margin_per_lot = price * margin_rate * 100000
        if margin_per_lot > 0:
            margin_budget = margin_available
            try:
                usage_cap = float(get_secret("max_margin_usage"))
                if 0.0 < usage_cap <= 1.0:
                    margin_budget = margin_available * usage_cap
            except Exception:
                pass
            lot = min(lot, margin_budget / margin_per_lot)

    lot = min(lot, MAX_LOT)
    return round(max(lot, 0.0), 3)


def _max_notional_units(equity: float | None, price: float | None) -> float:
    if equity is None or price is None:
        return 0.0
    try:
        equity_val = float(equity)
        price_val = float(price)
        if equity_val <= 0.0 or price_val <= 0.0:
            return 0.0
        return (equity_val / price_val) * MAX_LEVERAGE
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


def _abs_units(info: Dict) -> int:
    if not info:
        return 0
    trades = info.get("open_trades") or []
    total = 0
    if trades:
        for tr in trades:
            try:
                total += abs(int(tr.get("units", 0) or 0))
            except (TypeError, ValueError):
                continue
        return total
    try:
        return abs(int(info.get("units", 0) or 0))
    except (TypeError, ValueError):
        return 0


@dataclass(slots=True)
class ExposureState:
    cap_ratio: float
    max_units: float
    manual_units: float
    bot_units: float

    def limit_units(self) -> float:
        return max(0.0, self.cap_ratio * self.max_units)

    def ratio(self) -> float:
        limit = self.limit_units()
        if limit <= 0:
            return 0.0
        return min(2.0, (self.manual_units + self.bot_units) / limit)

    def manual_ratio(self) -> float:
        if self.max_units <= 0:
            return 0.0
        return min(2.0, self.manual_units / self.max_units)

    def bot_ratio(self) -> float:
        if self.max_units <= 0:
            return 0.0
        return min(2.0, self.bot_units / self.max_units)

    def available_units(self) -> float:
        used = self.manual_units + self.bot_units
        return max(0.0, self.limit_units() - used)

    def would_exceed(self, units: int) -> bool:
        if units == 0:
            return False
        future = self.manual_units + self.bot_units + abs(units)
        return future > self.limit_units()

    def allocate(self, units: int) -> None:
        if units == 0:
            return
        self.bot_units += abs(units)


def build_exposure_state(
    open_positions: Dict[str, Dict],
    *,
    equity: float | None,
    price: float | None,
    cap_ratio: float | None = None,
) -> Optional[ExposureState]:
    """現在の手動玉＋bot玉の使用率を計算し、追加エントリー可否を判定する。"""
    cap = float(cap_ratio or EXPOSURE_MAX_RATIO)
    if cap <= 0:
        return None
    max_units = _max_notional_units(equity, price)
    if max_units <= 0:
        return None
    manual_units = 0
    bot_units = 0
    for pocket, info in (open_positions or {}).items():
        if pocket in {"__net__", "__meta__"}:
            continue
        units = _abs_units(info or {})
        if units <= 0:
            continue
        if pocket.startswith("manual") or pocket in {"manual", "unknown"}:
            manual_units += units
        else:
            bot_units += units
    state = ExposureState(
        cap_ratio=cap,
        max_units=max_units,
        manual_units=float(manual_units),
        bot_units=float(bot_units),
    )
    _log_exposure_metrics(state)
    return state


def _log_exposure_metrics(state: ExposureState) -> None:
    ratio = state.ratio()
    tags = {
        "cap": f"{state.cap_ratio:.2f}",
        "manual_ratio": f"{state.manual_ratio():.3f}",
        "bot_ratio": f"{state.bot_ratio():.3f}",
    }
    log_metric("exposure_ratio", ratio, tags=tags)
    if ratio >= 1.0:
        logging.warning(
            "[EXPOSURE] ratio=%.3f manual=%.2f%% bot=%.2f%% cap=%.2f%%",
            ratio,
            state.manual_ratio() * 100,
            state.bot_ratio() * 100,
            state.cap_ratio * 100,
        )


def loss_cooldown_status(
    pocket: str, *, max_losses: int, cooldown_minutes: float
) -> tuple[bool, float]:
    """
    Evaluate whether the given pocket should be temporarily blocked because of a
    consecutive loss streak.

    Returns (blocked_flag, remaining_seconds).
    """
    if max_losses <= 0 or cooldown_minutes <= 0:
        return False, 0.0

    key = f"{pocket}:{max_losses}:{cooldown_minutes}"
    expiry_mono = _LOSS_COOLDOWN_CACHE.get(key)
    now_mono = time.monotonic()
    if expiry_mono:
        if now_mono < expiry_mono:
            remaining = max(0.0, expiry_mono - now_mono)
            return True, remaining
        _LOSS_COOLDOWN_CACHE.pop(key, None)

    rows = con.execute(
        """
        SELECT pl_pips, close_time
        FROM trades
        WHERE pocket = ?
          AND close_time IS NOT NULL
        ORDER BY datetime(close_time) DESC
        LIMIT ?
        """,
        (pocket, max_losses),
    ).fetchall()

    if not rows:
        return False, 0.0

    streak = 0
    latest_time: datetime | None = None
    latest_raw: str | None = None
    for row in rows:
        try:
            pl = float(row["pl_pips"] or 0.0)
        except Exception:
            pl = 0.0
        if pl < 0:
            streak += 1
            if latest_time is None:
                latest_raw = row["close_time"]
                latest_time = _parse_close_time(latest_raw)
        else:
            break

    if streak < max_losses or latest_time is None:
        return False, 0.0

    expiry_dt = latest_time + timedelta(minutes=cooldown_minutes)
    remaining = (expiry_dt - datetime.now(timezone.utc)).total_seconds()
    if remaining <= 0:
        return False, 0.0

    expiry_mono = now_mono + remaining
    _LOSS_COOLDOWN_CACHE[key] = expiry_mono
    return True, remaining


def clamp_sl_tp(
    price: float, sl: float, tp: float, is_buy: bool
) -> tuple[float, float]:
    """
    SL/TP が逆転していないかチェックし妥当な値を返す
    """
    if is_buy:
        if sl >= price:  # 異常
            sl = price - 0.1
        if tp <= price:
            tp = price + 0.1
    else:
        if sl <= price:
            sl = price + 0.1
        if tp >= price:
            tp = price - 0.1
    return round(sl, 3), round(tp, 3)
