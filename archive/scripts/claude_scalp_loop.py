#!/usr/bin/env python3
"""
Claude Scalp Loop — MTF分析ベースの裁量スキャルプ自動ループ
目標: 30分で1000 JPY
手法: H1トレンド方向 × M5タイミング × M1微調整
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import sys
import time
from datetime import datetime, timezone, timedelta
from typing import Optional, Literal

# Project root
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import requests
from utils.secrets import get_secret

# ─── Config ───
TOKEN = get_secret("oanda_token")
ACCOUNT = get_secret("oanda_account_id")
HOST = "https://api-fxtrade.oanda.com"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}
INSTRUMENT = "USD_JPY"

# Trading params
BASE_UNITS = 5000  # ~50 JPY/pip at USD/JPY ~159
SL_PIPS = 4.0  # Stop loss
TP_PIPS = 6.0  # Take profit (RR 1.5:1)
MAX_POSITIONS = 1  # Max concurrent trades
LOOP_INTERVAL_SEC = 15  # Analysis interval
SESSION_TARGET_JPY = 1000  # 30-min target
MAX_DRAWDOWN_JPY = -500  # Max session drawdown before pause

LOG_PATH = PROJECT_ROOT / "logs" / "claude_scalp_loop.jsonl"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("scalp_loop")


# ─── OANDA API Helpers ───


def get_price() -> tuple[float, float]:
    """Returns (bid, ask)"""
    resp = requests.get(
        f"{HOST}/v3/accounts/{ACCOUNT}/pricing?instruments={INSTRUMENT}",
        headers=HEADERS,
        timeout=10,
    )
    resp.raise_for_status()
    p = resp.json()["prices"][0]
    return float(p["bids"][0]["price"]), float(p["asks"][0]["price"])


def get_candles(tf: str, count: int) -> list[dict]:
    resp = requests.get(
        f"{HOST}/v3/instruments/{INSTRUMENT}/candles?granularity={tf}&count={count}",
        headers=HEADERS,
        timeout=10,
    )
    resp.raise_for_status()
    candles = resp.json()["candles"]
    result = []
    for c in candles:
        if not c.get("complete", True) and tf != "M1":
            continue  # Skip incomplete except M1
        mid = c["mid"]
        result.append(
            {
                "time": c["time"],
                "o": float(mid["o"]),
                "h": float(mid["h"]),
                "l": float(mid["l"]),
                "c": float(mid["c"]),
                "vol": c["volume"],
                "complete": c["complete"],
            }
        )
    return result


def get_account_nav() -> float:
    resp = requests.get(
        f"{HOST}/v3/accounts/{ACCOUNT}/summary",
        headers=HEADERS,
        timeout=10,
    )
    resp.raise_for_status()
    return float(resp.json()["account"]["NAV"])


def get_open_trades() -> list[dict]:
    resp = requests.get(
        f"{HOST}/v3/accounts/{ACCOUNT}/openTrades",
        headers=HEADERS,
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json().get("trades", [])


def place_market_order(
    units: int,
    sl_price: float,
    tp_price: float,
    reason: str = "",
) -> Optional[str]:
    """Place market order. units > 0 = buy, < 0 = sell. Returns trade ID or None."""
    body = {
        "order": {
            "type": "MARKET",
            "instrument": INSTRUMENT,
            "units": str(units),
            "timeInForce": "FOK",
            "positionFill": "OPEN_ONLY",
            "stopLossOnFill": {"price": f"{sl_price:.3f}", "timeInForce": "GTC"},
            "takeProfitOnFill": {"price": f"{tp_price:.3f}", "timeInForce": "GTC"},
            "clientExtensions": {
                "tag": "pocket=manual",
                "comment": f"claude_scalp:{reason[:60]}",
            },
        }
    }
    resp = requests.post(
        f"{HOST}/v3/accounts/{ACCOUNT}/orders",
        headers={**HEADERS, "Content-Type": "application/json"},
        json=body,
        timeout=10,
    )
    data = resp.json()

    if "orderFillTransaction" in data:
        trade_id = data["orderFillTransaction"].get("tradeOpened", {}).get("tradeID")
        fill_price = data["orderFillTransaction"].get("price", "?")
        log.info(
            f"FILLED: {units}u @ {fill_price} | SL={sl_price:.3f} TP={tp_price:.3f} | {reason}"
        )
        return trade_id
    elif "orderCancelTransaction" in data:
        cancel_reason = data["orderCancelTransaction"].get("reason", "unknown")
        log.warning(f"ORDER CANCELLED: {cancel_reason}")
        return None
    else:
        log.warning(
            f"ORDER UNKNOWN RESPONSE: {json.dumps(data, ensure_ascii=False)[:300]}"
        )
        return None


def close_trade(trade_id: str) -> Optional[float]:
    """Close a trade. Returns realized PL."""
    resp = requests.put(
        f"{HOST}/v3/accounts/{ACCOUNT}/trades/{trade_id}/close",
        headers={**HEADERS, "Content-Type": "application/json"},
        json={"units": "ALL"},
        timeout=10,
    )
    data = resp.json()
    if "orderFillTransaction" in data:
        pl = float(data["orderFillTransaction"].get("pl", 0))
        log.info(f"CLOSED trade {trade_id}: PL={pl:.1f} JPY")
        return pl
    else:
        log.warning(f"CLOSE FAILED: {json.dumps(data)[:200]}")
        return None


def modify_trade_sl_tp(
    trade_id: str, sl_price: Optional[float] = None, tp_price: Optional[float] = None
):
    """Modify SL/TP on an existing trade."""
    body = {}
    if sl_price is not None:
        body["stopLoss"] = {"price": f"{sl_price:.3f}", "timeInForce": "GTC"}
    if tp_price is not None:
        body["takeProfit"] = {"price": f"{tp_price:.3f}", "timeInForce": "GTC"}
    if not body:
        return
    resp = requests.put(
        f"{HOST}/v3/accounts/{ACCOUNT}/trades/{trade_id}/orders",
        headers={**HEADERS, "Content-Type": "application/json"},
        json=body,
        timeout=10,
    )
    if resp.status_code == 200:
        log.info(f"Modified trade {trade_id}: SL={sl_price} TP={tp_price}")
    else:
        log.warning(f"Modify failed: {resp.text[:200]}")


# ─── Technical Analysis ───


def compute_ema(values: list[float], period: int) -> list[float]:
    if not values:
        return []
    k = 2 / (period + 1)
    ema = [values[0]]
    for v in values[1:]:
        ema.append(v * k + ema[-1] * (1 - k))
    return ema


def compute_rsi(closes: list[float], period: int = 14) -> Optional[float]:
    if len(closes) < period + 1:
        return None
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [max(0, d) for d in deltas[-period:]]
    losses = [max(0, -d) for d in deltas[-period:]]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def compute_atr(candles: list[dict], period: int = 14) -> Optional[float]:
    if len(candles) < period + 1:
        return None
    trs = []
    for i in range(1, len(candles)):
        h, l, pc = candles[i]["h"], candles[i]["l"], candles[i - 1]["c"]
        tr = max(h - l, abs(h - pc), abs(l - pc))
        trs.append(tr)
    return sum(trs[-period:]) / period


def detect_trend(candles: list[dict], ema_fast: int = 8, ema_slow: int = 21) -> str:
    """Returns 'up', 'down', or 'range'"""
    closes = [c["c"] for c in candles]
    if len(closes) < ema_slow:
        return "range"
    fast = compute_ema(closes, ema_fast)
    slow = compute_ema(closes, ema_slow)
    diff = fast[-1] - slow[-1]
    diff_prev = fast[-2] - slow[-2] if len(fast) > 1 else diff
    threshold = 0.015  # ~1.5 pips
    if diff > threshold and diff >= diff_prev:
        return "up"
    elif diff < -threshold and diff <= diff_prev:
        return "down"
    return "range"


def detect_m5_signal(candles: list[dict], h1_trend: str) -> tuple[Optional[str], str]:
    """
    Returns (signal, reason) where signal is 'buy', 'sell', or None.
    Uses pullback entries in trend direction.
    """
    if len(candles) < 20:
        return None, "insufficient_data"

    closes = [c["c"] for c in candles]
    rsi = compute_rsi(closes, 14)
    atr = compute_atr(candles, 14)

    if rsi is None or atr is None:
        return None, "indicator_calc_failed"

    last = candles[-1]
    prev = candles[-2]
    prev2 = candles[-3]

    ema8 = compute_ema(closes, 8)
    ema21 = compute_ema(closes, 21)

    # Trend alignment check
    ema_bullish = ema8[-1] > ema21[-1]
    ema_bearish = ema8[-1] < ema21[-1]

    # Pullback detection: price dipped below EMA8 then recovered
    pullback_buy = (
        h1_trend == "up"
        and ema_bullish
        and prev["l"] <= ema8[-2]  # Dipped to EMA
        and last["c"] > ema8[-1]  # Recovered above
        and rsi < 65  # Not overbought
        and last["c"] > last["o"]  # Bullish candle
    )

    pullback_sell = (
        h1_trend == "down"
        and ema_bearish
        and prev["h"] >= ema8[-2]
        and last["c"] < ema8[-1]
        and rsi > 35
        and last["c"] < last["o"]  # Bearish candle
    )

    # Momentum breakout: strong candle in trend direction
    body = abs(last["c"] - last["o"])
    is_strong = body > atr * 0.6

    momentum_buy = (
        h1_trend in ("up", "range")
        and last["c"] > last["o"]
        and is_strong
        and last["c"] > max(c["h"] for c in candles[-5:-1])  # New 5-bar high
        and rsi > 50
        and rsi < 75
    )

    momentum_sell = (
        h1_trend in ("down", "range")
        and last["c"] < last["o"]
        and is_strong
        and last["c"] < min(c["l"] for c in candles[-5:-1])  # New 5-bar low
        and rsi < 50
        and rsi > 25
    )

    # V-shape reversal (from previous session's winning pattern)
    recent_low = min(c["l"] for c in candles[-6:])
    recent_high = max(c["h"] for c in candles[-6:])
    range_pips = (recent_high - recent_low) * 100

    v_reversal_buy = (
        range_pips > 8  # Significant move
        and candles[-3]["c"] < candles[-3]["o"]  # Down candle
        and candles[-2]["c"] < candles[-2]["o"]  # Down candle
        and last["c"] > last["o"]  # Reversal up
        and last["c"] > candles[-2]["h"]  # Engulfing
        and rsi < 45  # Was oversold
    )

    v_reversal_sell = (
        range_pips > 8
        and candles[-3]["c"] > candles[-3]["o"]
        and candles[-2]["c"] > candles[-2]["o"]
        and last["c"] < last["o"]
        and last["c"] < candles[-2]["l"]
        and rsi > 55
    )

    if pullback_buy:
        return "buy", f"pullback_buy RSI={rsi:.0f} EMA8={ema8[-1]:.3f}"
    if pullback_sell:
        return "sell", f"pullback_sell RSI={rsi:.0f} EMA8={ema8[-1]:.3f}"
    if momentum_buy:
        return "buy", f"momentum_break RSI={rsi:.0f} body={body*100:.1f}pip"
    if momentum_sell:
        return "sell", f"momentum_break RSI={rsi:.0f} body={body*100:.1f}pip"
    if v_reversal_buy:
        return "buy", f"v_reversal_buy RSI={rsi:.0f} range={range_pips:.1f}pip"
    if v_reversal_sell:
        return "sell", f"v_reversal_sell RSI={rsi:.0f} range={range_pips:.1f}pip"

    return None, f"no_signal RSI={rsi:.0f} H1={h1_trend} ATR={atr*100:.1f}pip"


def compute_dynamic_sl_tp(
    side: str,
    entry_price: float,
    atr_m5: float,
    h1_trend: str,
) -> tuple[float, float]:
    """Compute SL/TP based on ATR and trend strength."""
    # Scale SL with ATR but clamp to bounds
    sl_pips = max(3.0, min(SL_PIPS, atr_m5 * 100 * 1.5))

    # TP is wider when aligned with H1 trend
    if (side == "buy" and h1_trend == "up") or (side == "sell" and h1_trend == "down"):
        tp_pips = max(5.0, min(10.0, atr_m5 * 100 * 2.5))
    else:
        tp_pips = max(4.0, min(7.0, atr_m5 * 100 * 1.8))

    pip = 0.01  # USD/JPY pip

    if side == "buy":
        sl = entry_price - sl_pips * pip
        tp = entry_price + tp_pips * pip
    else:
        sl = entry_price + sl_pips * pip
        tp = entry_price - tp_pips * pip

    return round(sl, 3), round(tp, 3)


# ─── Position Management ───


def manage_open_position(trade: dict, bid: float, ask: float, atr_m5: float):
    """Trail stop and manage existing position."""
    trade_id = trade["id"]
    units = float(trade["currentUnits"])
    entry = float(trade["price"])
    unrealized = float(trade["unrealizedPL"])

    is_long = units > 0
    current_price = bid if is_long else ask
    pips_profit = (
        (current_price - entry) if is_long else (entry - current_price)
    ) * 100

    # Get existing SL
    existing_sl = None
    sl_order = trade.get("stopLossOrder")
    if sl_order:
        existing_sl = float(sl_order["price"])

    pip = 0.01

    # Trail stop when in profit > 2 pips
    if pips_profit > 2.0 and existing_sl is not None:
        trail_distance = max(1.5, atr_m5 * 100 * 0.8)  # Dynamic trail

        if is_long:
            new_sl = current_price - trail_distance * pip
            if new_sl > existing_sl + 0.002:  # Only move up
                log.info(
                    f"TRAIL: trade {trade_id} SL {existing_sl:.3f} → {new_sl:.3f} (profit={pips_profit:.1f}pip)"
                )
                modify_trade_sl_tp(trade_id, sl_price=new_sl)
        else:
            new_sl = current_price + trail_distance * pip
            if new_sl < existing_sl - 0.002:  # Only move down
                log.info(
                    f"TRAIL: trade {trade_id} SL {existing_sl:.3f} → {new_sl:.3f} (profit={pips_profit:.1f}pip)"
                )
                modify_trade_sl_tp(trade_id, sl_price=new_sl)

    # Emergency close if held too long with small profit
    open_time_str = trade.get("openTime", "")
    if open_time_str:
        try:
            open_time = datetime.fromisoformat(
                open_time_str.replace("Z", "+00:00").split(".")[0] + "+00:00"
            )
            hold_secs = (datetime.now(timezone.utc) - open_time).total_seconds()
            if hold_secs > 600 and pips_profit > 0.5:  # 10min+ with small profit
                log.info(
                    f"TIME EXIT: trade {trade_id} held {hold_secs:.0f}s, profit={pips_profit:.1f}pip"
                )
                close_trade(trade_id)
        except Exception:
            pass


# ─── Session Tracking ───


class SessionTracker:
    def __init__(self):
        self.start_time = datetime.now(timezone.utc)
        self.start_nav = get_account_nav()
        self.trades: list[dict] = []
        self.current_30min_start = self.start_time
        self.current_30min_start_nav = self.start_nav
        self.total_trades = 0
        self.wins = 0
        self.losses = 0
        self.total_pl = 0.0
        self.reflections: list[str] = []

    def record_trade(self, trade_info: dict):
        self.trades.append(trade_info)
        pl = trade_info.get("pl", 0)
        self.total_trades += 1
        self.total_pl += pl
        if pl > 0:
            self.wins += 1
        elif pl < 0:
            self.losses += 1

    def check_30min_window(self) -> dict:
        """Check 30-min P&L window and return status."""
        now = datetime.now(timezone.utc)
        elapsed = (now - self.current_30min_start).total_seconds()

        if elapsed >= 1800:  # 30 minutes
            current_nav = get_account_nav()
            window_pl = current_nav - self.current_30min_start_nav

            result = {
                "window_start": self.current_30min_start.isoformat(),
                "window_end": now.isoformat(),
                "pl": window_pl,
                "target": SESSION_TARGET_JPY,
                "achieved": window_pl >= SESSION_TARGET_JPY,
                "trades": self.total_trades,
                "wins": self.wins,
                "losses": self.losses,
            }

            # Reflect on performance
            if not result["achieved"]:
                self._reflect(result)

            # Reset window
            self.current_30min_start = now
            self.current_30min_start_nav = current_nav
            return result

        return {
            "elapsed_sec": elapsed,
            "remaining_sec": 1800 - elapsed,
            "current_session_pl": self.total_pl,
        }

    def _reflect(self, window: dict):
        """Analyze what went wrong when target not met."""
        pl = window["pl"]
        trades = window["trades"]

        reasons = []
        if trades == 0:
            reasons.append(
                "NO_TRADES: シグナルが出なかった。フィルター条件を緩和すべきか検討"
            )
        elif trades < 3:
            reasons.append(f"LOW_FREQUENCY: {trades}回のみ。エントリー機会が少なすぎ")

        if pl < 0:
            wr = self.wins / max(1, self.total_trades)
            if wr < 0.5:
                reasons.append(f"LOW_WINRATE: {wr:.0%}。トレンド方向の精度を上げる必要")
            reasons.append(
                f"NEGATIVE_PL: {pl:.0f}JPY。SL/TP比率やエントリー精度を見直す"
            )

        if 0 <= pl < SESSION_TARGET_JPY:
            reasons.append(
                f"INSUFFICIENT_GAIN: {pl:.0f}/{SESSION_TARGET_JPY}JPY。ユニット数かTP幅の拡大を検討"
            )

        reflection = f"[30min振り返り] PL={pl:.0f}JPY, Trades={trades} | " + " | ".join(
            reasons
        )
        self.reflections.append(reflection)
        log.warning(reflection)

    def summary(self) -> str:
        wr = self.wins / max(1, self.total_trades)
        return (
            f"Session: {self.total_trades} trades, {self.wins}W/{self.losses}L "
            f"(WR={wr:.0%}), PL={self.total_pl:.0f} JPY"
        )


# ─── Log to file ───


def log_to_file(entry: dict):
    entry["timestamp"] = datetime.now(timezone.utc).isoformat()
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ─── Main Loop ───


def main():
    log.info("=" * 60)
    log.info("Claude Scalp Loop START")
    log.info(f"Target: {SESSION_TARGET_JPY} JPY / 30min")
    log.info(f"Units: {BASE_UNITS}, SL: {SL_PIPS}pip, TP: {TP_PIPS}pip")
    log.info("=" * 60)

    tracker = SessionTracker()
    log.info(f"Starting NAV: {tracker.start_nav:.0f} JPY")

    consecutive_no_signal = 0
    cooldown_until = 0.0
    last_trade_side: Optional[str] = None
    last_trade_time = 0.0

    while True:
        try:
            now = time.time()

            # Cooldown after trade
            if now < cooldown_until:
                time.sleep(LOOP_INTERVAL_SEC)
                continue

            # 1. Get market data
            bid, ask = get_price()
            spread = (ask - bid) * 100
            mid = (bid + ask) / 2

            # Skip if spread too wide
            if spread > 2.0:
                log.info(f"Wide spread: {spread:.1f}pip, skipping")
                time.sleep(LOOP_INTERVAL_SEC)
                continue

            # 2. Check open positions
            open_trades = get_open_trades()
            our_trades = [t for t in open_trades if t.get("instrument") == INSTRUMENT]

            # Manage existing positions
            if our_trades:
                candles_m5 = get_candles("M5", 20)
                atr_m5 = compute_atr(candles_m5, 14) or 0.05
                for t in our_trades:
                    manage_open_position(t, bid, ask, atr_m5)

                # Check if trade closed (TP/SL hit)
                time.sleep(LOOP_INTERVAL_SEC)
                continue

            # 3. Analyze market (no position open)
            candles_h1 = get_candles("H1", 30)
            candles_m5 = get_candles("M5", 48)
            candles_m1 = get_candles("M1", 30)

            h1_trend = detect_trend(candles_h1)
            m5_trend = detect_trend(candles_m5)
            atr_m5 = compute_atr(candles_m5, 14)
            rsi_m5 = compute_rsi([c["c"] for c in candles_m5], 14)

            # 4. Generate signal
            signal, reason = detect_m5_signal(candles_m5, h1_trend)

            # M1 confirmation: ensure M1 is not against the signal
            if signal and candles_m1:
                m1_trend = detect_trend(candles_m1, ema_fast=5, ema_slow=13)
                if signal == "buy" and m1_trend == "down":
                    log.info(f"M1 conflict: signal={signal} but M1={m1_trend}, waiting")
                    signal = None
                    reason = "m1_conflict"
                elif signal == "sell" and m1_trend == "up":
                    log.info(f"M1 conflict: signal={signal} but M1={m1_trend}, waiting")
                    signal = None
                    reason = "m1_conflict"

            # Avoid same-direction re-entry too quickly
            if signal and signal == last_trade_side and (now - last_trade_time) < 120:
                log.info(f"Same direction cooldown: {signal}, waiting")
                signal = None
                reason = "same_direction_cooldown"

            if signal is None:
                consecutive_no_signal += 1
                if consecutive_no_signal % 20 == 0:
                    log.info(
                        f"No signal x{consecutive_no_signal} | "
                        f"H1={h1_trend} M5={m5_trend} RSI={rsi_m5:.0f if rsi_m5 else '?'} "
                        f"ATR={atr_m5*100:.1f}pip | mid={mid:.3f} spread={spread:.1f}pip"
                    )
                time.sleep(LOOP_INTERVAL_SEC)

                # Check 30-min window
                window = tracker.check_30min_window()
                if "achieved" in window:
                    log_to_file({"type": "30min_window", **window})
                    log.info(
                        f"30min window: PL={window['pl']:.0f} JPY {'ACHIEVED' if window['achieved'] else 'MISSED'}"
                    )

                continue

            consecutive_no_signal = 0

            # 5. Execute trade
            entry_price = ask if signal == "buy" else bid
            sl, tp = compute_dynamic_sl_tp(
                signal, entry_price, atr_m5 or 0.05, h1_trend
            )

            units = BASE_UNITS if signal == "buy" else -BASE_UNITS

            log.info(f"SIGNAL: {signal.upper()} | {reason}")
            log.info(f"  Entry={entry_price:.3f} SL={sl:.3f} TP={tp:.3f} Units={units}")
            log.info(
                f"  H1={h1_trend} M5={m5_trend} RSI={rsi_m5:.0f if rsi_m5 else '?'}"
            )

            trade_id = place_market_order(units, sl, tp, reason)

            if trade_id:
                last_trade_side = signal
                last_trade_time = now
                cooldown_until = now + 30  # 30s cooldown after entry

                log_to_file(
                    {
                        "type": "entry",
                        "signal": signal,
                        "reason": reason,
                        "entry_price": entry_price,
                        "sl": sl,
                        "tp": tp,
                        "units": units,
                        "trade_id": trade_id,
                        "h1_trend": h1_trend,
                        "m5_trend": m5_trend,
                        "rsi": rsi_m5,
                        "atr_m5": atr_m5,
                    }
                )
            else:
                cooldown_until = now + 60  # Wait longer on failure

            time.sleep(LOOP_INTERVAL_SEC)

            # Check 30-min window
            window = tracker.check_30min_window()
            if "achieved" in window:
                log_to_file({"type": "30min_window", **window})
                log.info(
                    f"30min window: PL={window['pl']:.0f} JPY {'ACHIEVED' if window['achieved'] else 'MISSED'}"
                )

            # Session drawdown check
            current_nav = get_account_nav()
            session_pl = current_nav - tracker.start_nav
            if session_pl < MAX_DRAWDOWN_JPY:
                log.error(
                    f"SESSION DRAWDOWN LIMIT: {session_pl:.0f} JPY < {MAX_DRAWDOWN_JPY}. PAUSING 5min."
                )
                log_to_file({"type": "drawdown_pause", "session_pl": session_pl})
                time.sleep(300)

        except KeyboardInterrupt:
            log.info("Interrupted by user")
            break
        except requests.exceptions.RequestException as e:
            log.error(f"API error: {e}")
            time.sleep(30)
        except Exception as e:
            log.error(f"Unexpected error: {e}", exc_info=True)
            time.sleep(30)

    # Session summary
    log.info("=" * 60)
    log.info(f"Session ended. {tracker.summary()}")
    final_nav = get_account_nav()
    log.info(
        f"NAV: {tracker.start_nav:.0f} → {final_nav:.0f} JPY (PL={final_nav - tracker.start_nav:.0f})"
    )
    if tracker.reflections:
        log.info("Reflections:")
        for r in tracker.reflections:
            log.info(f"  {r}")


if __name__ == "__main__":
    main()
