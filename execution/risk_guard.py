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
POCKET_DD_LIMITS = {
    "micro": 0.05,
    "macro": 0.15,
    "scalp": 0.03,
    "scalp_fast": 0.02,
}  # equity 比 (%)
GLOBAL_DD_LIMIT = 0.20  # 全体ドローダウン 20%
# Pocketごとの口座配分上限（DD母数の推定に使用）
# Macro は 30% に制限（以前は 80%）。
POCKET_MAX_RATIOS = {
    "macro": 0.3,
    "micro": 0.6,
    "scalp": 0.25,
    "scalp_fast": 0.1,
}
_DEFAULT_BASE_EQUITY = {
    "macro": 8000.0,
    "micro": 6000.0,
    "scalp": 2500.0,
    "scalp_fast": 2000.0,
}
_LOOKBACK_DAYS = 7
MAX_MARGIN_USAGE = float(os.getenv("MAX_MARGIN_USAGE", "0.92"))

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
EXPOSURE_WARN_THRESHOLD = float(os.getenv("EXPOSURE_WARN_THRESHOLD", "1.05"))
_MIN_LOT_BY_POCKET = {
    "macro": max(0.0, float(os.getenv("RISK_MIN_LOT_MACRO", "0.1"))),
    "micro": max(0.0, float(os.getenv("RISK_MIN_LOT_MICRO", "0.0"))),
    # scalp は最小 0.05 lot (=5k units) に緩和（環境変数で上書き可）
    "scalp": max(0.0, float(os.getenv("RISK_MIN_LOT_SCALP", "0.05"))),
}
_EXPOSURE_IGNORE_POCKETS = {
    token.strip().lower()
    for token in os.getenv("EXPOSURE_IGNORE_POCKETS", "unknown").split(",")
    if token.strip()
}

_DB = pathlib.Path("logs/trades.db")
con = sqlite3.connect(_DB)
con.row_factory = sqlite3.Row


def _guard_enabled() -> bool:
    flag = os.getenv("ENABLE_DRAWDOWN_GUARD", "0").strip().lower()
    return flag not in {"", "0", "false", "off"}

_FAST_SCALP_SHARE = 0.35
try:
    _FAST_SCALP_SHARE = max(
        0.0, min(1.0, float(os.getenv("FAST_SCALP_SHARE_HINT", "0.35")))
    )
except Exception:
    _FAST_SCALP_SHARE = 0.35

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


def update_dd_context(
    account_equity: float,
    weight_macro: float,
    weight_scalp: Optional[float] = None,
    scalp_share: float = 0.0,
    *,
    atr_pips: Optional[float] = None,
    free_margin_ratio: Optional[float] = None,
    perf_hint: Optional[dict] = None,
) -> None:
    """最新の口座残高とポケット配分ヒントを共有し、DD 判定の母数を更新する。"""
    if account_equity <= 0:
        return

    macro_cap = POCKET_MAX_RATIOS["macro"]
    micro_cap = POCKET_MAX_RATIOS["micro"]
    scalp_cap = POCKET_MAX_RATIOS["scalp"]
    scalp_fast_cap = POCKET_MAX_RATIOS["scalp_fast"]

    # ATRでダイナミックに揺らす（高ボラで上限を少し緩め、低ボラで絞る）
    if atr_pips is not None:
        try:
            atr_val = float(atr_pips)
        except Exception:
            atr_val = None
        if atr_val is not None and atr_val > 0:
            if atr_val >= 3.0:
                scale = 1.2
            elif atr_val <= 1.0:
                scale = 0.8
            else:
                scale = 1.0
            macro_cap = min(1.0, macro_cap * scale)
            micro_cap = min(1.0, micro_cap * scale)
            scalp_cap = min(1.0, scalp_cap * scale)
            scalp_fast_cap = min(1.0, scalp_fast_cap * scale)

    # 成績連動（pf中心）で微調整
    if isinstance(perf_hint, dict):
        def _scale_from_pf(pf: Optional[float]) -> float:
            try:
                val = float(pf)
            except Exception:
                return 1.0
            if val <= 0.8:
                return 0.7
            if val <= 1.0:
                return 0.9
            if val >= 1.3:
                return 1.15
            if val >= 1.1:
                return 1.05
            return 1.0

        macro_cap *= _scale_from_pf((perf_hint.get("macro") or {}).get("pf"))
        micro_cap *= _scale_from_pf((perf_hint.get("micro") or {}).get("pf"))
        scalp_cap *= _scale_from_pf((perf_hint.get("scalp") or {}).get("pf"))

    # 手動ポジ含めた負荷を free_margin_ratio でざっくり反映
    if free_margin_ratio is not None:
        try:
            fmr = float(free_margin_ratio)
        except Exception:
            fmr = None
        if fmr is not None:
            if fmr < 0.15:
                macro_cap *= 0.6
                micro_cap *= 0.6
                scalp_cap *= 0.6
                scalp_fast_cap *= 0.6
            elif fmr < 0.25:
                macro_cap *= 0.8
                micro_cap *= 0.8
                scalp_cap *= 0.8
                scalp_fast_cap *= 0.8

    macro_cap = min(max(macro_cap, 0.0), 1.0)
    micro_cap = min(max(micro_cap, 0.0), 1.0)
    scalp_cap = min(max(scalp_cap, 0.0), 1.0)
    scalp_fast_cap = min(max(scalp_fast_cap, 0.0), 1.0)

    macro_ratio = min(max(weight_macro, 0.0), macro_cap)
    scalp_ratio = 0.0
    if weight_scalp is not None:
        scalp_ratio = min(max(weight_scalp, 0.0), scalp_cap)
        remainder = max(1.0 - macro_ratio - scalp_ratio, 0.0)
    else:
        remainder = max(1.0 - macro_ratio, 0.0)
        share = max(scalp_share, 0.0)
        scalp_ratio = min(share * remainder, scalp_cap)
        remainder = max(remainder - scalp_ratio, 0.0)

    micro_ratio = min(remainder, micro_cap)

    scalp_fast_ratio = min(
        max(scalp_ratio * _FAST_SCALP_SHARE, 0.0),
        scalp_fast_cap,
    )
    scalp_ratio = max(0.0, min(scalp_cap, scalp_ratio - scalp_fast_ratio))

    ratios = {
        "macro": macro_ratio,
        "micro": micro_ratio,
        "scalp": scalp_ratio,
        "scalp_fast": scalp_fast_ratio,
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
    # DD ガードを撤廃し、常に取引を許可
    return True


def allowed_lot(
    equity: float,
    sl_pips: float,
    *,
    margin_available: float | None = None,
    price: float | None = None,
    margin_rate: float | None = None,
    risk_pct_override: float | None = None,
    pocket: str | None = None,
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
        # safer default under drawdown pressure; slightly more assertive sizing
        risk_pct = 0.02
    if risk_pct_override is not None:
        risk_pct = max(0.0005, min(risk_pct_override, 0.25))
    risk_amount = equity * risk_pct
    lot = risk_amount / (sl_pips * 1000)  # USD/JPYの1lotは1000JPY/pip ≒ 1000

    if margin_available is not None and price is not None and margin_rate:
        margin_per_lot = price * margin_rate * 100000
        if margin_per_lot > 0:
            margin_cap = max(0.0, min(MAX_MARGIN_USAGE, 1.0))
            try:
                usage_cap = float(get_secret("max_margin_usage"))
                if 0.0 < usage_cap <= 1.0:
                    margin_cap = max(margin_cap, usage_cap)
            except Exception:
                pass
            # guard下でも92%までは使う
            margin_cap = max(margin_cap, 0.92)
            margin_budget = margin_available * margin_cap
            lot = min(lot, margin_budget / margin_per_lot)

    lot = min(lot, MAX_LOT)
    min_lot = _MIN_LOT_BY_POCKET.get((pocket or "").lower(), 0.0)
    if min_lot > 0.0 and lot > 0.0 and lot < min_lot:
        logging.debug(
            "[RISK] min lot clamp pocket=%s lot=%.3f -> %.3f",
            pocket,
            lot,
            min_lot,
        )
        lot = min(min_lot, MAX_LOT)
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
    margin_pool: float | None = None
    manual_margin: float | None = None
    bot_margin: float | None = None
    unit_margin_cost: float | None = None

    def limit_units(self) -> float:
        return max(0.0, self.cap_ratio * self.max_units)

    def limit_margin(self) -> float | None:
        return self.margin_pool

    def ratio(self) -> float:
        if self.margin_pool and self.margin_pool > 0 and self.manual_margin is not None:
            return min(
                2.0,
                ((self.manual_margin or 0.0) + (self.bot_margin or 0.0))
                / self.margin_pool,
            )
        limit = self.limit_units()
        if limit <= 0:
            return 0.0
        return min(2.0, (self.manual_units + self.bot_units) / limit)

    def manual_ratio(self) -> float:
        if self.margin_pool and self.margin_pool > 0 and self.manual_margin is not None:
            return min(2.0, (self.manual_margin or 0.0) / self.margin_pool)
        if self.max_units <= 0:
            return 0.0
        return min(2.0, self.manual_units / self.max_units)

    def bot_ratio(self) -> float:
        if self.margin_pool and self.margin_pool > 0 and self.bot_margin is not None:
            return min(2.0, (self.bot_margin or 0.0) / self.margin_pool)
        if self.max_units <= 0:
            return 0.0
        return min(2.0, self.bot_units / self.max_units)

    def available_units(self) -> float:
        if (
            self.margin_pool
            and self.unit_margin_cost
            and self.manual_margin is not None
        ):
            remaining = self.margin_pool - (
                (self.manual_margin or 0.0) + (self.bot_margin or 0.0)
            )
            return max(0.0, remaining / self.unit_margin_cost)
        used = self.manual_units + self.bot_units
        return max(0.0, self.limit_units() - used)

    def would_exceed(self, units: int) -> bool:
        if units == 0:
            return False
        if (
            self.margin_pool
            and self.unit_margin_cost
            and self.manual_margin is not None
        ):
            projected = (
                (self.manual_margin or 0.0)
                + (self.bot_margin or 0.0)
                + abs(units) * self.unit_margin_cost
            )
            return projected > self.margin_pool
        future = self.manual_units + self.bot_units + abs(units)
        return future > self.limit_units()

    def allocate(self, units: int) -> None:
        if units == 0:
            return
        self.bot_units += abs(units)
        if self.bot_margin is not None and self.unit_margin_cost:
            self.bot_margin += abs(units) * self.unit_margin_cost


def build_exposure_state(
    open_positions: Dict[str, Dict],
    *,
    equity: float | None,
    price: float | None,
    margin_used: float | None = None,
    margin_available: float | None = None,
    margin_rate: float | None = None,
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
    ignore_manual = "manual" in _EXPOSURE_IGNORE_POCKETS
    for pocket, info in (open_positions or {}).items():
        if pocket in {"__net__", "__meta__"}:
            continue
        units = _abs_units(info or {})
        if units <= 0:
            continue
        pocket_key = (pocket or "").lower()
        if pocket_key in _EXPOSURE_IGNORE_POCKETS:
            logging.debug("[EXPOSURE] ignoring pocket=%s units=%s per config", pocket, units)
            continue
        if pocket_key.startswith("manual") and ignore_manual:
            logging.debug("[EXPOSURE] ignoring manual pocket=%s units=%s", pocket, units)
            continue
        if pocket_key.startswith("manual") or pocket_key == "manual":
            manual_units += units
        else:
            bot_units += units
    unit_margin_cost = None
    margin_pool = None
    manual_margin = None
    bot_margin = None
    if (
        price is not None
        and margin_rate is not None
        and margin_rate > 0
        and margin_used is not None
        and margin_available is not None
    ):
        unit_margin_cost = price * margin_rate
        manual_margin = manual_units * unit_margin_cost
        bot_margin = bot_units * unit_margin_cost
        margin_pool = cap * (margin_used + margin_available)

    state = ExposureState(
        cap_ratio=cap,
        max_units=max_units,
        manual_units=float(manual_units),
        bot_units=float(bot_units),
        margin_pool=margin_pool,
        manual_margin=manual_margin,
        bot_margin=bot_margin,
        unit_margin_cost=unit_margin_cost,
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
    if ratio >= EXPOSURE_WARN_THRESHOLD:
        logging.warning(
            "[EXPOSURE] ratio=%.3f manual=%.2f%% bot=%.2f%% cap=%.2f%%",
            ratio,
            state.manual_ratio() * 100,
            state.bot_ratio() * 100,
            state.cap_ratio * 100,
        )
    else:
        logging.info(
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
    price: float,
    sl: Optional[float],
    tp: Optional[float],
    is_buy: bool,
) -> Tuple[Optional[float], Optional[float]]:
    """
    SL/TP が逆転していないかチェックし妥当な値を返す
    """
    adj_sl = sl
    adj_tp = tp
    if adj_tp is not None:
        if is_buy and adj_tp <= price:
            adj_tp = price + 0.1
        if not is_buy and adj_tp >= price:
            adj_tp = price - 0.1
    if adj_sl is not None:
        if is_buy and adj_sl >= price:
            adj_sl = price - 0.1
        if not is_buy and adj_sl <= price:
            adj_sl = price + 0.1
    return (
        round(adj_sl, 3) if adj_sl is not None else None,
        round(adj_tp, 3) if adj_tp is not None else None,
    )
