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
POCKET_DD_LIMITS = {"micro": 0.05, "macro": 0.15}  # equity 比 (%)
MAX_LOSS_STREAK = {"micro": 3, "macro": 2}  # 直近の連敗で一時停止
GLOBAL_DD_LIMIT = 0.20  # 全体ドローダウン 20%

_DB = pathlib.Path("logs/trades.db")
_DB.parent.mkdir(exist_ok=True)
con = sqlite3.connect(_DB)

# テーブルを必ず用意（position_manager/ perf_monitor と統一）
con.execute(
    """
CREATE TABLE IF NOT EXISTS trades (
  id INTEGER PRIMARY KEY,
  ticket_id TEXT UNIQUE,
  pocket TEXT,
  instrument TEXT,
  units INTEGER,
  entry_price REAL,
  close_price REAL,
  pl_pips REAL,
  entry_time TEXT,
  close_time TEXT
)
"""
)
con.commit()


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


def _recent_loss_streak(pocket: str, hours: int = 24) -> int:
    rows = con.execute(
        """
SELECT pl_pips FROM trades
WHERE pocket=? AND datetime(close_time) >= datetime('now', ?)
ORDER BY datetime(close_time) DESC
""",
        (pocket, f"-{hours} hour"),
    ).fetchall()
    streak = 0
    for (pl,) in rows:
        try:
            if pl is not None and float(pl) < 0:
                streak += 1
            else:
                break
        except Exception:
            break
    return streak


def recent_loss_streak(pocket: str, hours: int = 24) -> int:
    """公開API: 直近の連敗数を返す（安全側で使用）"""
    return _recent_loss_streak(pocket, hours)


def can_trade(pocket: str) -> bool:
    if _pocket_dd(pocket) >= POCKET_DD_LIMITS[pocket]:
        return False
    if _recent_loss_streak(pocket) >= MAX_LOSS_STREAK[pocket]:
        print(f"[RISK] Loss streak pause for {pocket}")
        return False
    return True


def allowed_lot(equity: float, sl_pips: float, *, risk_pct: float = 0.02) -> float:
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
