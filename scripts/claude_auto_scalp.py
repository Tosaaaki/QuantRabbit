#!/usr/bin/env python3
"""
Claude Auto Scalp — 改良版自律スキャルパー
レンジ/トレンド自動判別 + マルチテクニカル
"""

import json
import logging
import os
import pathlib
import sys
import time
from datetime import datetime, timezone, timedelta

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import requests
from utils.secrets import get_secret

TOKEN = get_secret("oanda_token")
ACCOUNT = get_secret("oanda_account_id")
HOST = "https://api-fxtrade.oanda.com"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}
INSTRUMENT = "USD_JPY"

UNITS = 5000
SL_PIPS = 2.0
TP_PIPS = 3.0
MAX_DRAWDOWN_SESSION = -500
LOOP_SEC = 12
COOLDOWN_AFTER_TRADE = 45
COOLDOWN_AFTER_LOSS = 90
END_TIME_UTC = 15  # 00:00 JST

LOG_PATH = PROJECT_ROOT / "logs" / "claude_auto_scalp.jsonl"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("auto_scalp")


# ─── API ───


def get_price():
    r = requests.get(
        f"{HOST}/v3/accounts/{ACCOUNT}/pricing?instruments={INSTRUMENT}",
        headers=HEADERS,
        timeout=10,
    )
    r.raise_for_status()
    p = r.json()["prices"][0]
    return float(p["bids"][0]["price"]), float(p["asks"][0]["price"])


def get_candles(tf, count):
    r = requests.get(
        f"{HOST}/v3/instruments/{INSTRUMENT}/candles?granularity={tf}&count={count}",
        headers=HEADERS,
        timeout=10,
    )
    r.raise_for_status()
    out = []
    for c in r.json()["candles"]:
        mid = c["mid"]
        out.append(
            {
                "o": float(mid["o"]),
                "h": float(mid["h"]),
                "l": float(mid["l"]),
                "c": float(mid["c"]),
                "vol": c["volume"],
                "complete": c["complete"],
                "time": c["time"],
            }
        )
    return out


def get_nav():
    r = requests.get(
        f"{HOST}/v3/accounts/{ACCOUNT}/summary", headers=HEADERS, timeout=10
    )
    r.raise_for_status()
    return float(r.json()["account"]["NAV"])


def get_open_trades():
    r = requests.get(
        f"{HOST}/v3/accounts/{ACCOUNT}/openTrades", headers=HEADERS, timeout=10
    )
    r.raise_for_status()
    return [t for t in r.json().get("trades", []) if t.get("instrument") == INSTRUMENT]


def place_order(units, sl, tp, comment=""):
    body = {
        "order": {
            "type": "MARKET",
            "instrument": INSTRUMENT,
            "units": str(units),
            "timeInForce": "FOK",
            "positionFill": "OPEN_ONLY",
            "stopLossOnFill": {"price": f"{sl:.3f}", "timeInForce": "GTC"},
            "takeProfitOnFill": {"price": f"{tp:.3f}", "timeInForce": "GTC"},
            "clientExtensions": {"tag": "pocket=manual", "comment": comment[:60]},
        }
    }
    r = requests.post(
        f"{HOST}/v3/accounts/{ACCOUNT}/orders",
        headers={**HEADERS, "Content-Type": "application/json"},
        json=body,
        timeout=10,
    )
    data = r.json()
    if "orderFillTransaction" in data:
        fill = data["orderFillTransaction"]
        tid = fill.get("tradeOpened", {}).get("tradeID")
        fp = float(fill.get("price", 0))
        return tid, fp
    return None, 0.0


def modify_sl(trade_id, new_sl):
    r = requests.put(
        f"{HOST}/v3/accounts/{ACCOUNT}/trades/{trade_id}/orders",
        headers={**HEADERS, "Content-Type": "application/json"},
        json={"stopLoss": {"price": f"{new_sl:.3f}", "timeInForce": "GTC"}},
        timeout=10,
    )
    return r.status_code == 200


# ─── Technicals ───


def ema(values, period):
    k = 2 / (period + 1)
    e = [values[0]]
    for v in values[1:]:
        e.append(v * k + e[-1] * (1 - k))
    return e


def rsi(closes, period=14):
    if len(closes) < period + 1:
        return 50.0
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [max(0, d) for d in deltas[-period:]]
    losses = [max(0, -d) for d in deltas[-period:]]
    ag = sum(gains) / period
    al = sum(losses) / period
    if al == 0:
        return 100.0
    return 100.0 - 100.0 / (1.0 + ag / al)


def atr(candles, period=14):
    if len(candles) < period + 1:
        return 0.05
    trs = []
    for i in range(1, len(candles)):
        h, l, pc = candles[i]["h"], candles[i]["l"], candles[i - 1]["c"]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    return sum(trs[-period:]) / period


def bollinger_bands(closes, period=20, std_mult=2.0):
    if len(closes) < period:
        mid = closes[-1]
        return mid - 0.05, mid, mid + 0.05, 0.10
    window = closes[-period:]
    mid = sum(window) / period
    variance = sum((x - mid) ** 2 for x in window) / period
    std = variance**0.5
    width = (std * std_mult * 2) * 100  # in pips
    return mid - std * std_mult, mid, mid + std * std_mult, width


def detect_regime(m5_candles):
    """Returns 'trend_up', 'trend_down', or 'range'"""
    closes = [c["c"] for c in m5_candles]
    if len(closes) < 21:
        return "range"

    e8 = ema(closes, 8)
    e21 = ema(closes, 21)
    diff = e8[-1] - e21[-1]

    _, _, _, bb_width = bollinger_bands(closes)

    # Tight BB = range, wide BB = trend
    if bb_width < 6.0:  # < 6 pips BB width = range
        return "range"
    if diff > 0.020 and e8[-1] > e8[-3]:
        return "trend_up"
    if diff < -0.020 and e8[-1] < e8[-3]:
        return "trend_down"
    return "range"


def find_signal(m5_candles, m1_candles, regime):
    """Returns (signal, reason) where signal is 'buy', 'sell', or None"""
    m5_closes = [c["c"] for c in m5_candles]
    m1_closes = [c["c"] for c in m1_candles]

    if len(m5_closes) < 21 or len(m1_closes) < 13:
        return None, "insufficient_data"

    m5_rsi = rsi(m5_closes)
    m5_atr = atr(m5_candles)
    m5_e8 = ema(m5_closes, 8)
    m5_e21 = ema(m5_closes, 21)

    m1_e5 = ema(m1_closes, 5)
    m1_e13 = ema(m1_closes, 13)
    m1_up = m1_e5[-1] > m1_e13[-1] + 0.005
    m1_dn = m1_e5[-1] < m1_e13[-1] - 0.005

    bb_lower, bb_mid, bb_upper, bb_width = bollinger_bands(m5_closes)

    last_m1 = m1_candles[-1]
    m1_bullish = last_m1["c"] > last_m1["o"]
    m1_bearish = last_m1["c"] < last_m1["o"]
    current = m5_closes[-1]

    if regime == "range":
        # Range: buy at lower BB, sell at upper BB
        if current <= bb_lower + 0.005 and m1_up and m1_bullish and m5_rsi < 40:
            return "buy", f"range_bb_lower RSI={m5_rsi:.0f} BB_L={bb_lower:.3f}"
        if current >= bb_upper - 0.005 and m1_dn and m1_bearish and m5_rsi > 60:
            return "sell", f"range_bb_upper RSI={m5_rsi:.0f} BB_U={bb_upper:.3f}"
        return None, f"range_no_signal RSI={m5_rsi:.0f} BBw={bb_width:.0f}"

    elif regime == "trend_up":
        # Trend up: buy when M1 aligns bullish, RSI NOT overbought
        if 30 < m5_rsi < 60 and m1_up and m1_bullish:
            return "buy", f"trend_up_m1_align RSI={m5_rsi:.0f}"
        # Also: buy on M1 pullback to M1 EMA13 bounce
        if 30 < m5_rsi < 60 and m1_bullish:
            prev_m1 = m1_candles[-2]
            if prev_m1["l"] <= m1_e13[-2] + 0.003 and last_m1["c"] > m1_e13[-1]:
                return "buy", f"trend_up_m1_ema_bounce RSI={m5_rsi:.0f}"
        return None, f"trend_up_wait RSI={m5_rsi:.0f}"

    elif regime == "trend_down":
        # Trend down: sell when M1 aligns bearish, RSI NOT oversold
        if 40 < m5_rsi < 70 and m1_dn and m1_bearish:
            return "sell", f"trend_dn_m1_align RSI={m5_rsi:.0f}"
        # Also: sell on M1 rally rejection (bounce to M1 EMA13 then rejected)
        if 40 < m5_rsi < 70 and m1_bearish:
            prev_m1 = m1_candles[-2]
            if prev_m1["h"] >= m1_e13[-2] - 0.003 and last_m1["c"] < m1_e13[-1]:
                return "sell", f"trend_dn_m1_ema_reject RSI={m5_rsi:.0f}"
        return None, f"trend_dn_wait RSI={m5_rsi:.0f}"

    return None, "unknown_regime"


def log_trade(entry):
    entry["timestamp"] = datetime.now(timezone.utc).isoformat()
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ─── Main ───


def main():
    start_nav = get_nav()
    log.info("=" * 50)
    log.info(f"Auto Scalp START | NAV={start_nav:.0f} | Until {END_TIME_UTC}:00 UTC")
    log.info("=" * 50)

    session_pl = 0.0
    total_trades = 0
    wins = 0
    losses = 0
    cooldown_until = 0.0
    consecutive_losses = 0

    while True:
        try:
            now_utc = datetime.now(timezone.utc)
            if now_utc.hour >= END_TIME_UTC:
                log.info("Session end time reached")
                break

            if time.time() < cooldown_until:
                time.sleep(LOOP_SEC)
                continue

            # Check drawdown
            current_nav = get_nav()
            session_pl = current_nav - start_nav
            if session_pl < MAX_DRAWDOWN_SESSION:
                log.error(f"MAX DRAWDOWN: {session_pl:.0f} JPY. Pausing 5min.")
                time.sleep(300)
                consecutive_losses = 0
                continue

            # Check spread
            bid, ask = get_price()
            spread = (ask - bid) * 100
            if spread > 2.0:
                time.sleep(LOOP_SEC)
                continue

            # Manage open trades
            open_trades = get_open_trades()
            if open_trades:
                m5_data = get_candles("M5", 20)
                m5_atr = atr(m5_data)
                for t in open_trades:
                    _manage_trade(t, bid, ask, m5_atr)
                time.sleep(LOOP_SEC)
                continue

            # No position — analyze
            m5_data = get_candles("M5", 48)
            m1_data = get_candles("M1", 30)

            regime = detect_regime(m5_data)
            signal, reason = find_signal(m5_data, m1_data, regime)

            if signal is None:
                if total_trades == 0 or total_trades % 20 == 0:
                    log.info(f"No signal | regime={regime} | {reason} | bid={bid:.3f}")
                time.sleep(LOOP_SEC)
                continue

            # Execute
            pip = 0.01
            if signal == "buy":
                entry_price = ask
                sl = round(entry_price - SL_PIPS * pip, 3)
                tp = round(entry_price + TP_PIPS * pip, 3)
                units = UNITS
            else:
                entry_price = bid
                sl = round(entry_price + SL_PIPS * pip, 3)
                tp = round(entry_price - TP_PIPS * pip, 3)
                units = -UNITS

            log.info(f"SIGNAL: {signal.upper()} | {reason} | regime={regime}")
            log.info(f"  Entry~{entry_price:.3f} SL={sl:.3f} TP={tp:.3f}")

            tid, fill_price = place_order(units, sl, tp, f"{regime}:{reason[:40]}")

            if tid:
                total_trades += 1
                log_trade(
                    {
                        "type": "entry",
                        "signal": signal,
                        "regime": regime,
                        "reason": reason,
                        "entry": fill_price,
                        "sl": sl,
                        "tp": tp,
                    }
                )
                # Wait for trade to close, then check result
                pre_nav = get_nav()
                cooldown_until = time.time() + COOLDOWN_AFTER_TRADE
                # Monitor until closed
                for _ in range(120):  # max 6 min
                    time.sleep(3)
                    ot = get_open_trades()
                    if not ot:
                        break
                    # Trail stop management
                    bid_now, ask_now = get_price()
                    m5_now = get_candles("M5", 20)
                    atr_now = atr(m5_now)
                    for t in ot:
                        _manage_trade(t, bid_now, ask_now, atr_now)
                post_nav = get_nav()
                trade_pl = post_nav - pre_nav
                if trade_pl > 0:
                    wins += 1
                    log.info(
                        f"  WIN: +{trade_pl:.0f} JPY | Total: {post_nav-start_nav:+.0f}"
                    )
                    cooldown_until = time.time() + COOLDOWN_AFTER_TRADE
                    consecutive_losses = 0
                else:
                    losses += 1
                    consecutive_losses += 1
                    log.info(
                        f"  LOSS: {trade_pl:.0f} JPY | Total: {post_nav-start_nav:+.0f}"
                    )
                    cooldown_until = time.time() + COOLDOWN_AFTER_LOSS
                    if consecutive_losses >= 3:
                        log.warning("3 consecutive losses. Extra cooldown 3min.")
                        cooldown_until = time.time() + 180
                        consecutive_losses = 0
            else:
                log.warning("Order failed")
                cooldown_until = time.time() + 60

            time.sleep(LOOP_SEC)

        except KeyboardInterrupt:
            break
        except requests.exceptions.RequestException as e:
            log.error(f"API error: {e}")
            time.sleep(30)
        except Exception as e:
            log.error(f"Error: {e}", exc_info=True)
            time.sleep(30)

    # Summary
    final_nav = get_nav()
    session_pl = final_nav - start_nav
    wr = wins / max(1, total_trades)
    log.info("=" * 50)
    log.info(f"Session END | {total_trades} trades | {wins}W/{losses}L (WR={wr:.0%})")
    log.info(f"NAV: {start_nav:.0f} → {final_nav:.0f} (PL={session_pl:+.0f} JPY)")
    log.info("=" * 50)


def _manage_trade(trade, bid, ask, m5_atr_val):
    """Trail stop on open position."""
    tid = trade["id"]
    units = float(trade["currentUnits"])
    entry = float(trade["price"])
    is_long = units > 0
    current = bid if is_long else ask
    pips = ((current - entry) if is_long else (entry - current)) * 100
    pip = 0.01

    sl_order = trade.get("stopLossOrder")
    if not sl_order:
        return
    existing_sl = float(sl_order["price"])

    # Trail at +1.2 pip: move to BE+
    if pips >= 1.2:
        if is_long:
            new_sl = entry + 0.002
            if new_sl > existing_sl:
                modify_sl(tid, new_sl)
        else:
            new_sl = entry - 0.002
            if new_sl < existing_sl:
                modify_sl(tid, new_sl)

    # Tighter trail at +2.0 pip
    if pips >= 2.0:
        trail = max(0.008, m5_atr_val * 0.6)
        if is_long:
            new_sl = current - trail
            if new_sl > existing_sl + 0.002:
                modify_sl(tid, round(new_sl, 3))
        else:
            new_sl = current + trail
            if new_sl < existing_sl - 0.002:
                modify_sl(tid, round(new_sl, 3))


if __name__ == "__main__":
    main()
