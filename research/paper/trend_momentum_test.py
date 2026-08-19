"""TrendMomentumMicro over six years instead of seven days.

This is the one bot in the whole ledger whose day-level interval excluded zero:
`micro/システム`, +27,735 JPY, day CI [+1,084, +7,027], 5/7 winning days. It could
not be raised under `QR_AI_STRATEGY_ALLOCATION_V1` for a specific reason — the
`strategy` column was empty, and the amendment refuses to size a strategy it
cannot name.

It can be named now. The client_order_id prefix `qr-micro-<ts>-Tren-<hash>` leads
to `archive/strategies/micro/trend_momentum.py`, class `TrendMomentumMicro`,
which survived the v8 archive intact. The decision rule is ported here verbatim:
MA10/MA20 gap, ADX floor, EMA slope agreement, 15m drift agreement, pullback cap,
Bollinger-width floor, spread cap, and ATR-scaled SL with TP = 1.7 x SL.

Two things this can settle that seven live days could not:
  * does it survive six years, or was that week the market being kind
  * does it work on JPY pairs other than USD_JPY

`PIP = 0.01` is hardcoded in the original, so the rule is JPY-quoted only.
Running it on EUR_USD would divide gaps by 0.01 instead of 0.0001 and make every
threshold 100x too loose — the same class of defect as the previous session's #4.
JPY pairs only, deliberately.

Exit is the strategy's own TP/SL, not the engine's fixed hold, because TP = 1.7 x
SL is part of the rule. Same-bar TP and SL touches are charged as SL, matching
`AGENT_CONTRACT`'s stop-first rule for temporally ambiguous fills.
"""
import os, statistics, sys
from collections import defaultdict, deque

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from engine import load, lower_bound, TRAIN_END, MIN_DAYS  # noqa: E402

PIP = 0.01
JPY_PAIRS = ["USD_JPY", "EUR_JPY", "GBP_JPY", "AUD_JPY", "CAD_JPY", "CHF_JPY", "NZD_JPY"]
MAX_HOLD = 480          # bars; the original had no cap, this bounds the test
WARM = 1500

# thresholds, verbatim from the archived class
MIN_GAP_PIPS = 0.28
MIN_ADX = 12.0
MIN_SLOPE = 0.03
MAX_PULLBACK = 1.2
MIN_ATR_PIPS = 0.9
MIN_BBW = 0.10
SPREAD_PIPS_MAX = 1.2
SPREAD_ATR_RATIO_MAX = 0.30


def indicators(h, l, c, sp):
    n = len(c)
    ma10 = [0.0] * n; ma20 = [0.0] * n; ema20 = [0.0] * n
    atr = [0.0] * n; adx = [0.0] * n; bbw = [0.0] * n
    s10 = s20 = 0.0
    sq20 = 0.0
    x = c[0]; a = None
    tr_s = pdm = ndm = 0.0
    for i in range(n):
        s10 += c[i]
        if i >= 10:
            s10 -= c[i - 10]
        s20 += c[i]; sq20 += c[i] * c[i]
        if i >= 20:
            s20 -= c[i - 20]; sq20 -= c[i - 20] * c[i - 20]
        if i >= 9:
            ma10[i] = s10 / 10
        if i >= 19:
            m = s20 / 20
            ma20[i] = m
            var = max(sq20 / 20 - m * m, 0.0)
            sd = var ** 0.5
            # archive/tools/trend_bot.py:685 defines bbw as `bb_upper - bb_lower`,
            # i.e. the RAW price width, not a normalised one. Dividing by the
            # middle band makes it ~0.0005 on USD_JPY, so MIN_BBW=0.10 rejects
            # every bar and the strategy silently never fires.
            bbw[i] = 4 * sd
        x += (c[i] - x) * (2 / 21); ema20[i] = x
        if i:
            tr = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
            a = tr if a is None else a + (tr - a) / 14.0
            atr[i] = a
            up = h[i] - h[i - 1]; dn = l[i - 1] - l[i]
            p = up if (up > dn and up > 0) else 0.0
            q = dn if (dn > up and dn > 0) else 0.0
            tr_s = tr if tr_s == 0 else tr_s + (tr - tr_s) / 14.0
            pdm = p if pdm == 0 else pdm + (p - pdm) / 14.0
            ndm = q if ndm == 0 else ndm + (q - ndm) / 14.0
            if tr_s > 0:
                pdi = 100 * pdm / tr_s; ndi = 100 * ndm / tr_s
                tot = pdi + ndi
                adx[i] = 100 * abs(pdi - ndi) / tot if tot else 0.0
    return ma10, ma20, ema20, atr, adx, bbw


def check(i, c, ma10, ma20, ema20, atr, adx, bbw, spread_pips):
    """Verbatim port of TrendMomentumMicro.check for a JPY-quoted pair."""
    if i < 20 or ma10[i] == 0 or ma20[i] == 0 or atr[i] <= 0:
        return None
    atr_pips = atr[i] / PIP
    cap = max(SPREAD_PIPS_MAX, atr_pips * SPREAD_ATR_RATIO_MAX)
    if spread_pips > cap:
        return None
    if bbw[i] and bbw[i] < MIN_BBW:
        return None
    diff = (ma10[i] - ma20[i]) / PIP
    if diff >= MIN_GAP_PIPS and adx[i] >= MIN_ADX:
        side = +1
    elif diff <= -MIN_GAP_PIPS and adx[i] >= MIN_ADX:
        side = -1
    else:
        return None
    ema_gap = (ma10[i] - ema20[i]) / PIP
    if side > 0 and ema_gap < MIN_SLOPE:
        return None
    if side < 0 and ema_gap > -MIN_SLOPE:
        return None
    drift = (c[i] - c[i - 15]) / PIP if i >= 15 else 0.0
    if side > 0 and drift < -1.3:
        return None
    if side < 0 and drift > 1.3:
        return None
    pullback = (c[i] - ma10[i]) / PIP
    if side > 0 and pullback < -MAX_PULLBACK:
        return None
    if side < 0 and pullback > MAX_PULLBACK:
        return None
    ap = max(MIN_ATR_PIPS, min(atr_pips, 12.0))
    sl = round(max(2.8, ap * 1.05), 2)
    tp = round(max(sl * 1.7, sl + max(ap * 0.8, 1.2)), 2)
    return side, sl, tp


def run(pair):
    ts, h, l, c, sp = load(pair)
    n = len(c)
    if n < 200000:
        return None
    half = statistics.median(sp) / 2
    half_p = half / PIP
    ma10, ma20, ema20, atr, adx, bbw = indicators(h, l, c, sp)
    byday = defaultdict(float)
    trades = wins = 0
    i = WARM
    while i < n - MAX_HOLD:
        got = check(i, c, ma10, ma20, ema20, atr, adx, bbw, sp[i] / PIP)
        if not got:
            i += 1
            continue
        side, sl, tp = got
        px = c[i] + side * half            # pay half the spread entering
        r = None
        for j in range(i + 1, i + MAX_HOLD + 1):
            if side > 0:
                fav = (h[j] - half - px) / PIP
                adv = (l[j] - half - px) / PIP
            else:
                fav = (px - (l[j] + half)) / PIP
                adv = (px - (h[j] + half)) / PIP
            if adv <= -sl:                 # stop-first on same-bar ambiguity
                r = -sl; break
            if fav >= tp:
                r = tp; break
        if r is None:
            j = i + MAX_HOLD
            r = ((c[j] - half - px) if side > 0 else (px - (c[j] + half))) / PIP
        byday[ts[i][:10]] += r
        trades += 1; wins += (r > 0)
        i = j + 1                          # flat before the next signal
    return byday, trades, wins, half_p


print(f"TrendMomentumMicro（アーカイブ版を逐語移植）を円ペアで6年検証")
print(f"出口は戦略本来の TP/SL（TP = 1.7 x SL、ATR連動）。同足のTP/SL同時到達はSL扱い。\n")
print(f"{'pair':9s} {'取引':>7s} {'勝率':>5s} | {'TR日':>5s} {'TR日次':>8s} | "
      f"{'TE日':>5s} {'TE日次':>8s} {'TE下限':>8s} {'TE勝日':>7s} {'合格':>5s}")
print("-" * 96)
allday = {"TRAIN": defaultdict(float), "TEST": defaultdict(float)}
for pair in (sys.argv[1:] or JPY_PAIRS):
    got = run(pair)
    if not got:
        print(f"{pair:9s} バー数不足"); continue
    byday, trades, wins, half_p = got
    tr = [v for d, v in byday.items() if d < TRAIN_END]
    te = [v for d, v in byday.items() if d >= TRAIN_END]
    for d, v in byday.items():
        allday["TRAIN" if d < TRAIN_END else "TEST"][(pair, d)] += v
    if not tr or not te:
        print(f"{pair:9s} 期間不足"); continue
    elb = lower_bound(te)
    ok = len(te) >= MIN_DAYS and sum(te) > 0 and elb > 0
    print(f"{pair:9s} {trades:7d} {100*wins/max(trades,1):4.0f}% | "
          f"{len(tr):5d} {statistics.mean(tr):8.2f} | {len(te):5d} {statistics.mean(te):8.2f} "
          f"{elb:8.2f} {100*sum(1 for x in te if x>0)/len(te):6.0f}% {'○' if ok else '×':>5s}")

tr = list(allday["TRAIN"].values()); te = list(allday["TEST"].values())
if tr and te:
    elb = lower_bound(te)
    print(f"\n=== 円ペア合算（ポートフォリオ） ===")
    print(f"TRAIN {len(tr)}日  日次 {statistics.mean(tr):+.2f} pips")
    print(f"TEST  {len(te)}日  日次 {statistics.mean(te):+.2f} pips  片側95%下限 {elb:+.2f}  "
          f"勝ち日 {100*sum(1 for x in te if x>0)/len(te):.0f}%")
    print(f"合格: {'○' if (len(te)>=MIN_DAYS and sum(te)>0 and elb>0) else '×'}")
