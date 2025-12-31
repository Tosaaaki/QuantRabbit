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
MAX_LOT = float(os.getenv("RISK_MAX_LOT", "10.0"))  # aggressive上限（envで絞れる）
POCKET_DD_LIMITS = {
    "micro": 0.05,
    "macro": 0.15,
    "scalp": 0.03,
    "scalp_fast": 0.02,
}  # equity 比 (%)
GLOBAL_DD_LIMIT = 0.20  # 全体ドローダウン 20%
# Pocketごとの口座配分上限（DD母数の推定に使用）
POCKET_MAX_RATIOS = {
    "macro": float(os.getenv("POCKET_MAX_RATIO_MACRO", "0.82")),
    "micro": float(os.getenv("POCKET_MAX_RATIO_MICRO", "0.82")),
    "scalp": float(os.getenv("POCKET_MAX_RATIO_SCALP", "0.82")),
    "scalp_fast": float(os.getenv("POCKET_MAX_RATIO_SCALP_FAST", "0.82")),
}
_DEFAULT_BASE_EQUITY = {
    "macro": 8000.0,
    "micro": 6000.0,
    "scalp": 2500.0,
    "scalp_fast": 2000.0,
}
_LOOKBACK_DAYS = 7
MAX_MARGIN_USAGE = float(os.getenv("MAX_MARGIN_USAGE", "0.85"))

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
EXPOSURE_MAX_RATIO = float(os.getenv("EXPOSURE_MAX_RATIO", "0.90"))
EXPOSURE_WARN_THRESHOLD = float(os.getenv("EXPOSURE_WARN_THRESHOLD", "0.95"))
_MIN_LOT_BY_POCKET = {
    "macro": max(0.0, float(os.getenv("RISK_MIN_LOT_MACRO", "0.1"))),
    "micro": max(0.0, float(os.getenv("RISK_MIN_LOT_MICRO", "0.0"))),
    # scalp系も0を許容し、リスク計算結果に全面委ねる（envで上書き可）
    "scalp": max(0.0, float(os.getenv("RISK_MIN_LOT_SCALP", "0.0"))),
    "scalp_fast": max(0.0, float(os.getenv("RISK_MIN_LOT_SCALP_FAST", "0.0"))),
}
# 手動ポジはエクスポージャ計算から除外するのをデフォルトにする。
# manual ポジも含めて総エクスポージャを見たいので、既定で manual は除外しない。
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


def _trading_paused() -> bool:
    return os.getenv("TRADING_PAUSED", "0").strip().lower() not in {"", "0", "false", "off"}

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

    # enforce lower bound to avoid over-shrinking when metrics are soft
    MIN_CAP = 0.92
    macro_cap = min(max(macro_cap, MIN_CAP), 1.0)
    micro_cap = min(max(micro_cap, MIN_CAP), 1.0)
    scalp_cap = min(max(scalp_cap, MIN_CAP), 1.0)
    scalp_fast_cap = min(max(scalp_fast_cap, MIN_CAP), 1.0)

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
    # グローバル停止フラグ（環境変数 TRADING_PAUSED=1 で全ポケット停止）
    if _trading_paused():
        return False
    # DD ガードは撤廃したまま
    return True


def allowed_lot(
    equity: float,
    sl_pips: float,
    *,
    margin_available: float | None = None,
    margin_used: float | None = None,
    price: float | None = None,
    margin_rate: float | None = None,
    risk_pct_override: float | None = None,
    pocket: str | None = None,
    side: str | None = None,
    open_long_units: float | None = None,
    open_short_units: float | None = None,
) -> float:
    """
    口座全体の許容ロットを概算する。
    sl_pips: 損切り幅（pip単位）。SLなしの場合は margin ベースで算出。
    margin_available: 利用可能証拠金
    price: 現在値（USD/JPY mid）
    margin_rate: OANDA口座の証拠金率
    """

    # Allow override from config/env or environment: key "risk_pct" (e.g. 0.01 = 1%)
    try:
        risk_pct_str = get_secret("risk_pct")
        risk_pct = float(risk_pct_str)
        if not (0.0001 <= risk_pct <= 0.4):
            raise ValueError("out_of_range")
    except Exception:
        # aggressiveデフォルトに寄せる
        risk_pct = 0.04
    if risk_pct_override is not None:
        risk_pct = max(0.0005, min(risk_pct_override, 0.4))

    # free margin が枯渇していれば発注自体を止める
    min_free_margin_ratio = max(0.005, float(os.getenv("MIN_FREE_MARGIN_RATIO", "0.01") or 0.01))
    if equity > 0 and margin_available is not None:
        free_ratio = margin_available / equity if equity > 0 else 0.0
        if free_ratio < min_free_margin_ratio:
            allow_low_margin = os.getenv("ALLOW_HEDGE_ON_LOW_MARGIN", "1").strip().lower() not in {
                "",
                "0",
                "false",
                "off",
                "no",
            }
            if not allow_low_margin:
                logging.info(
                    "[RISK] free margin low: ratio=%.3f < min=%.3f -> skip",
                    free_ratio,
                    min_free_margin_ratio,
                )
                return 0.0
            logging.warning(
                "[RISK] free margin low but ALLOW_HEDGE_ON_LOW_MARGIN enabled: ratio=%.3f < min=%.3f (continue)",
                free_ratio,
                min_free_margin_ratio,
            )

    risk_amount = equity * risk_pct
    lot_risk = MAX_LOT
    if sl_pips > 0:
        lot_risk = risk_amount / (sl_pips * 1000)  # USD/JPYの1lotは1000JPY/pip ≒ 1000
    lot = lot_risk

    # 証拠金情報が欠損している場合は安全側（発注しない）
    if margin_available is None or price is None or not margin_rate:
        return 0.0

    margin_cap = None
    side_norm = (side or "").lower()
    used: float | None = None

    def _net_margin_after(add_units: float) -> float:
        """
        追加後のネット証拠金使用額（netting を考慮）をざっくり推定する。
        price/margin_rate が無い場合は現在値を返す。
        """
        if price is None or not margin_rate:
            if used is not None:
                return used
            if margin_used is not None:
                return float(margin_used)
            if equity > 0 and margin_available is not None:
                return max(0.0, equity - margin_available)
            return 0.0
        long_u = float(open_long_units or 0.0)
        short_u = float(open_short_units or 0.0)
        if side_norm == "long":
            long_u += add_units
        else:
            short_u += add_units
        net_units = abs(long_u - short_u)
        return net_units * price * margin_rate

    if margin_available is not None and price is not None and margin_rate:
        margin_per_lot = price * margin_rate * 100000
        if margin_per_lot > 0:
            # 基準: MAX_MARGIN_USAGE を絶対上限（デフォルト0.88、運用上限0.92にクランプ）
            margin_cap = max(0.0, min(MAX_MARGIN_USAGE, 0.92))
            hard_margin_cap = margin_cap
            # margin_cap=0 の場合は強制停止
            if margin_cap <= 0.0:
                return 0.0
            # 既存利用分を踏まえて「総使用率が cap を超えない」ように残余枠を計算
            used = margin_used
            if used is None:
                used = max(0.0, equity - margin_available)

            margin_budget_total = equity * margin_cap
            # 少し余白（0.5%）を残す
            margin_budget_safe = margin_budget_total * 0.995

            # provisional margin clamp
            lot_margin = (margin_budget_safe - used) / margin_per_lot

            # If netting would reduce usage (opposite side), allow larger lot up to cap.
            proposed_units = lot * 100000.0
            net_used_after = _net_margin_after(abs(proposed_units))
            # 片側 flatten までの必要ロットを最低限許容する
            gap_units = 0.0
            if side_norm == "long":
                gap_units = max(0.0, float(open_short_units or 0.0) - float(open_long_units or 0.0))
            elif side_norm == "short":
                gap_units = max(0.0, float(open_long_units or 0.0) - float(open_short_units or 0.0))
            gap_lot = gap_units / 100000.0 if gap_units > 0 else 0.0
            if net_used_after < used:
                # nettingで使用率が下がる場合は cap を緩和し、flatten分を優先して通す
                hedge_cap = max(margin_cap, 0.995)
                hedge_budget_safe = equity * hedge_cap * 0.999
                lot_margin = max(lot_margin, (hedge_budget_safe - net_used_after) / margin_per_lot)
                lot_margin = max(lot_margin, gap_lot)
                # margin_budget を超えていてもネット縮小なら最低ロットは通す
                if lot_margin <= 0 and gap_lot > 0:
                    lot_margin = gap_lot

            # すでに cap 超なら新規はゼロ
            current_usage = used / equity if equity > 0 else 0.0
            if current_usage >= margin_cap * 0.995 and net_used_after >= used:
                lot_margin = 0.0
            lot = min(lot, lot_margin)
    # margin_rate が取れない場合でも、free_ratio から強制ガードを入れる
    if margin_cap is None and margin_available is not None and equity > 0:
        hard_margin_cap = float(os.getenv("MAX_MARGIN_USAGE_HARD", "0.83") or 0.83)
        margin_used = max(0.0, equity - margin_available)
        current_usage = margin_used / equity
        if current_usage >= hard_margin_cap:
            lot = 0.0

    # 余剰を見込んでサイズを一段絞る（INSUFFICIENT_MARGIN の頻発を防ぐ）
    try:
        margin_safety = float(os.getenv("MARGIN_SAFETY_FACTOR", "0.9") or 0.9)
    except Exception:
        margin_safety = 0.9
    margin_safety = min(max(margin_safety, 0.1), 1.0)
    lot *= margin_safety

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

    # Free margin ベースでロット幅を追加スケール（残余証拠金が少なければ線形に絞る）
    # netting で使用証拠金が減る方向のポジションは、投下後の free_ratio を基準に緩和する
    if margin_available is not None and equity > 0:
        free_ratio = max(0.0, margin_available / equity)
        projected_used = _net_margin_after(abs(lot * 100000.0))
        free_ratio_after = None
        if equity > 0 and projected_used is not None:
            free_ratio_after = max(0.0, (equity - projected_used) / equity)
        netting_reduce = projected_used is not None and used is not None and projected_used < used
        try:
            soft = float(os.getenv("FREE_MARGIN_SOFT_RATIO", "0.35") or 0.35)
            hard = float(os.getenv("FREE_MARGIN_HARD_RATIO", "0.2") or 0.2)
        except Exception:
            soft, hard = 0.35, 0.2
        hard = max(0.0, min(hard, soft))
        # netting 減少なら「投下後」の free ratio を優先
        ratio_for_scale = free_ratio
        if free_ratio_after is not None and netting_reduce:
            ratio_for_scale = max(free_ratio, free_ratio_after)
        if ratio_for_scale <= hard and not netting_reduce:
            lot = 0.0
        elif ratio_for_scale < soft and lot > 0.0:
            scale = (ratio_for_scale - hard) / (soft - hard) if soft > hard else 0.0
            scale = max(0.0, min(1.0, scale))
            # netting 減少ならスケールを少し緩める
            if netting_reduce and scale < 1.0:
                scale = max(scale, 0.5)
            lot *= scale
        # netting 減少で lot が 0 まで絞られた場合、最小ロット(0.01)は確保する
        if netting_reduce and lot <= 0.0:
            lot = max(lot, 0.01)

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
    long_units_total = 0
    short_units_total = 0
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
        try:
            long_units_total += int(info.get("long_units", 0) or 0)
            short_units_total += int(info.get("short_units", 0) or 0)
        except Exception:
            # fallback: sign of net units if detailed breakdown is missing
            net_units = info.get("units")
            try:
                net_units = int(net_units or 0)
            except Exception:
                net_units = 0
            if net_units > 0:
                long_units_total += net_units
            elif net_units < 0:
                short_units_total += abs(net_units)
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
    _log_exposure_metrics(state, long_units_total=long_units_total, short_units_total=short_units_total)
    return state


def _log_exposure_metrics(
    state: ExposureState,
    *,
    long_units_total: float = 0.0,
    short_units_total: float = 0.0,
) -> None:
    ratio = state.ratio()
    cap_units = state.limit_units()
    available = state.available_units()
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
    try:
        log_metric(
            "exposure_ratio",
            ratio,
            tags={
                "cap_units": f"{cap_units:.0f}",
                "available_units": f"{available:.0f}",
                "manual_units": f"{state.manual_units:.0f}",
                "bot_units": f"{state.bot_units:.0f}",
                "long_units": f"{long_units_total:.0f}",
                "short_units": f"{short_units_total:.0f}",
            },
        )
    except Exception:
        pass


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
