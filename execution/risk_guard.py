"""
execution.risk_guard
~~~~~~~~~~~~~~~~~~~~
• 許容 lot 計算
• SL/TP クランプ
• Pocket 別ドローダウン監視
"""

from __future__ import annotations
import os
import sqlite3
import pathlib
from typing import Dict, Optional, Tuple
from utils.secrets import get_secret

# --- risk params ---
MAX_LEVERAGE = 20.0  # 1:20
MAX_LOT = 3.0  # 1 lot = 100k 通貨
POCKET_DD_LIMITS = {"micro": 0.05, "macro": 0.15, "scalp": 0.03}  # equity 比 (%)
GLOBAL_DD_LIMIT = 0.20  # 全体ドローダウン 20%
POCKET_MAX_RATIOS = {"macro": 0.8, "micro": 0.6, "scalp": 0.25}
_DEFAULT_BASE_EQUITY = {"macro": 8000.0, "micro": 6000.0, "scalp": 2500.0}
_LOOKBACK_DAYS = 7

_DB = pathlib.Path("logs/trades.db")
con = sqlite3.connect(_DB)
con.row_factory = sqlite3.Row


def _guard_enabled() -> bool:
    flag = os.getenv("ENABLE_DRAWDOWN_GUARD", "0").strip().lower()
    return flag not in {"", "0", "false", "off"}

_POCKET_EQUITY_HINT: Dict[str, float] = {
    pocket: _DEFAULT_BASE_EQUITY[pocket] for pocket in _DEFAULT_BASE_EQUITY
}


def update_dd_context(
    account_equity: float,
    weight_macro: float,
    weight_scalp: Optional[float] = None,
    scalp_share: float = 0.0,
) -> None:
    """最新の口座残高とポケット配分ヒントを共有し、DD 判定の母数を更新する。"""
    if account_equity <= 0:
        return

    macro_ratio = min(max(weight_macro, 0.0), POCKET_MAX_RATIOS["macro"])
    scalp_ratio = 0.0
    if weight_scalp is not None:
        scalp_ratio = min(max(weight_scalp, 0.0), POCKET_MAX_RATIOS["scalp"])
        remainder = max(1.0 - macro_ratio - scalp_ratio, 0.0)
    else:
        remainder = max(1.0 - macro_ratio, 0.0)
        share = max(scalp_share, 0.0)
        scalp_ratio = min(share * remainder, POCKET_MAX_RATIOS["scalp"])
        remainder = max(remainder - scalp_ratio, 0.0)

    micro_ratio = min(remainder, POCKET_MAX_RATIOS["micro"])

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
    if not _guard_enabled():
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
    if not _guard_enabled():
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


from typing import Optional, Tuple


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
