"""
execution.risk_guard
~~~~~~~~~~~~~~~~~~~~
• 許容 lot 計算
• SL/TP クランプ
• Pocket 別ドローダウン監視
"""

from __future__ import annotations
import sqlite3
import pathlib

# --- risk params ---
MAX_LEVERAGE = 20.0  # 1:20
MAX_LOT = 1.0  # 1 lot = 100k 通貨
POCKET_DD_LIMITS = {"micro": 0.05, "macro": 0.15, "scalp": 0.03}  # equity 比 (%)
GLOBAL_DD_LIMIT = 0.20  # 全体ドローダウン 20%

_DB = pathlib.Path("logs/trades.db")
con = sqlite3.connect(_DB)


def _pocket_dd(pocket: str) -> float:
    rows = con.execute(
        """
SELECT SUM(pl_pips) FROM trades
WHERE pocket=? AND date(close_time)>=date('now','-7 day')
""",
        (pocket,),
    ).fetchone()
    total = rows[0] or 0.0
    # equity は外部取得だが 10 万 pips = 100% として近似
    return abs(total) / 100000.0


def check_global_drawdown() -> bool:
    """口座全体のドローダウンが閾値を超えているかチェック"""
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
    return _pocket_dd(pocket) < POCKET_DD_LIMITS[pocket]


def allowed_lot(equity: float, sl_pips: float) -> float:
    risk_pct = 0.01
    lot = (equity * risk_pct) / sl_pips / 10  # $10/pip で 1 lot
    lot = min(lot, MAX_LOT)
    return round(lot, 3)


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
