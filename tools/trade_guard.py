#!/usr/bin/env python3
"""Run every cycle: auto-detect trade quality issues and warn.
Used alongside margin_alert.py. Embed in the trader's Bash ② / next-cycle Bash.

Policy: do not mechanically BLOCK. Present the facts and force awareness.
"""
import json, re, os, sys
from datetime import datetime, timezone, timedelta

LOG_PATH = 'logs/live_trade_log.txt'
JST = timezone(timedelta(hours=9))


def payoff_metrics(closes):
    """Realized intraday payoff quality."""
    pls = [c['pl'] for c in closes]
    if not pls:
        return None

    wins = [pl for pl in pls if pl > 0]
    losses = [pl for pl in pls if pl < 0]
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    if avg_loss < 0:
        rr = avg_win / abs(avg_loss)
    elif avg_win > 0 and not losses:
        rr = float('inf')
    else:
        rr = 0.0

    break_even_wr = None
    if avg_win > 0 and avg_loss < 0:
        break_even_wr = abs(avg_loss) / (avg_win + abs(avg_loss))
    elif avg_win > 0 and not losses:
        break_even_wr = 0.0

    return {
        'count': len(pls),
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': len(wins) / len(pls),
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'rr': rr,
        'expectancy': sum(pls) / len(pls),
        'break_even_wr': break_even_wr,
    }

def parse_trades_today():
    """Parse today's CLOSE trades"""
    today = datetime.now(JST).strftime('%Y-%m-%d')
    closes = []
    if not os.path.exists(LOG_PATH):
        return closes

    for line in open(LOG_PATH):
        if 'CLOSE' not in line:
            continue
        m = re.search(r'PL=([+-]?[\d,.]+)', line)
        if not m:
            continue
        pl = float(m.group(1).replace(',', ''))

        is_today = False
        if today in line:
            is_today = True
        yesterday_utc = (datetime.now(JST) - timedelta(hours=9)).strftime('%Y-%m-%d')
        if yesterday_utc in line:
            time_m = re.search(r'(\d{2}):(\d{2})', line)
            if time_m and int(time_m.group(1)) >= 15:
                is_today = True

        if is_today:
            pair_m = re.search(r'CLOSE (\w+_\w+)', line)
            pair = pair_m.group(1) if pair_m else '?'
            # Also extract reason
            reason_m = re.search(r'reason=(.+?)(?:\||$)', line)
            reason = reason_m.group(1).strip()[:60] if reason_m else ''
            closes.append({'pl': pl, 'pair': pair, 'reason': reason, 'line': line.strip()[:100]})

    return closes


def check_garbage_wins(closes):
    """Detect garbage wins (tiny profit takes)"""
    garbage = [c for c in closes if 0 < c['pl'] <= 30]
    if len(garbage) >= 3:
        total = sum(c['pl'] for c in garbage)
        return f'🗑️ {len(garbage)} closes today at 30 JPY or less (total +{total:.0f} JPY). Is this profit worth the spread + effort?'
    return None


def check_recent_losses():
    """Display the reasons behind recent consecutive stop losses (fact presentation, not mechanical panic detection)"""
    if not os.path.exists(LOG_PATH):
        return None

    recent_closes = []
    for line in open(LOG_PATH):
        if 'CLOSE' not in line:
            continue
        m = re.search(r'PL=([+-]?[\d,.]+)', line)
        if not m:
            continue
        pl = float(m.group(1).replace(',', ''))
        pair_m = re.search(r'CLOSE (\w+_\w+)', line)
        pair = pair_m.group(1) if pair_m else '?'
        reason_m = re.search(r'reason=(.+?)(?:\||$)', line)
        reason = reason_m.group(1).strip()[:60] if reason_m else 'no reason'
        recent_closes.append({'pl': pl, 'pair': pair, 'reason': reason})

    if len(recent_closes) < 3:
        return None

    # Get the streak of consecutive stop losses from the end
    streak = []
    for c in reversed(recent_closes):
        if c['pl'] < 0:
            streak.append(c)
        else:
            break

    if len(streak) < 3:
        return None

    streak.reverse()
    total = sum(c['pl'] for c in streak)
    lines = [f'  {c["pair"]} {c["pl"]:+,.0f} JPY | {c["reason"]}' for c in streak[-5:]]  # show up to 5

    result = f'⚡ {len(streak)} consecutive stop losses (total {total:+,.0f} JPY). Check the reasons:\n'
    result += '\n'.join(lines)
    result += '\n  → If all are "H1 structure change" that is correct judgment. If the same reason repeats, that is panic'
    return result


def check_payoff_quality(closes):
    """Present realized payoff quality as a fact, not just planned R:R."""
    metrics = payoff_metrics(closes)
    if not metrics or metrics['wins'] < 3 or metrics['losses'] < 3:
        return None
    be_wr = metrics['break_even_wr']
    be_text = f"{be_wr:.0%}" if be_wr is not None else '?'
    if metrics['expectancy'] < 0:
        return (
            f"📊 Payoff: WR {metrics['win_rate']:.0%} vs break-even {be_text} | "
            f"avg win +{metrics['avg_win']:.0f} JPY vs avg loss {metrics['avg_loss']:.0f} JPY "
            f"(R:R={metrics['rr']:.2f}) | EV {metrics['expectancy']:+,.0f} JPY/trade"
        )
    if metrics['rr'] < 0.7:
        return (
            f"📊 Payoff: avg win +{metrics['avg_win']:.0f} JPY vs avg loss {metrics['avg_loss']:.0f} JPY "
            f"(R:R={metrics['rr']:.2f}) | WR {metrics['win_rate']:.0%} | EV {metrics['expectancy']:+,.0f}/trade"
        )
    return None


def check_pair_loss(closes):
    """Cumulative loss on a specific pair"""
    pair_pl = {}
    for c in closes:
        pair = c['pair']
        if pair not in pair_pl:
            pair_pl[pair] = 0
        pair_pl[pair] += c['pl']

    alerts = []
    for pair, pl in pair_pl.items():
        if pl < -1000:
            alerts.append(f'📊 {pair}: {pl:+,.0f} JPY today. Factor this into your judgment')
    return alerts


def main():
    closes = parse_trades_today()
    if not closes:
        return

    alerts = []

    # Recent consecutive stop losses (with reasons)
    recent = check_recent_losses()
    if recent:
        alerts.append(recent)

    # Garbage wins
    garbage = check_garbage_wins(closes)
    if garbage:
        alerts.append(garbage)

    # Payoff quality
    payoff = check_payoff_quality(closes)
    if payoff:
        alerts.append(payoff)

    # Per-pair cumulative loss
    pair_alerts = check_pair_loss(closes)
    alerts.extend(pair_alerts)

    if alerts:
        print('=== 🛡️ TRADE GUARD ===')
        for a in alerts:
            print(a)
        print('======================')


if __name__ == '__main__':
    main()
