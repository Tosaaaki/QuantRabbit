#!/usr/bin/env python3
"""Pre-close reflection. Not a BLOCK — a reminder.
Same philosophy as pretrade_check: surface past lessons to improve decision quality.
Does not override judgment. Presents the information needed to make the call.

Usage:
    python3 tools/preclose_check.py EUR_USD SHORT 12000 -612

Output: Facts needed for the decision. Final judgment is the trader's.
"""
import sys
import os
import re
import time
from datetime import datetime, timezone, timedelta

JST = timezone(timedelta(hours=9))
LOG_PATH = 'logs/live_trade_log.txt'
STATE_PATH = 'collab_trade/state.md'


def get_consecutive_losses():
    """Number of consecutive losses from the most recent close."""
    if not os.path.exists(LOG_PATH):
        return 0, 0
    closes = []
    for line in open(LOG_PATH):
        if 'CLOSE' not in line:
            continue
        m = re.search(r'PL=([+-]?[\d,.]+)', line)
        if m:
            closes.append(float(m.group(1).replace(',', '')))
    streak = 0
    streak_sum = 0
    for pl in reversed(closes):
        if pl < 0:
            streak += 1
            streak_sum += pl
        else:
            break
    return streak, streak_sum


def get_pair_pl_today(pair):
    """Realized P&L for this pair today."""
    total = 0
    if not os.path.exists(LOG_PATH):
        return total
    for line in open(LOG_PATH):
        if 'CLOSE' not in line or pair not in line:
            continue
        m = re.search(r'PL=([+-]?[\d,.]+)', line)
        if m:
            total += float(m.group(1).replace(',', ''))
    return total


def get_thesis_from_state(pair):
    """Extract the thesis and conviction for this pair from state.md."""
    if not os.path.exists(STATE_PATH):
        return None
    content = open(STATE_PATH).read()
    # Find the section containing the pair name
    lines = content.split('\n')
    section = []
    in_section = False
    for line in lines:
        if pair in line and ('###' in line or '##' in line):
            in_section = True
            section.append(line)
        elif in_section:
            if line.startswith('###') or line.startswith('---'):
                break
            section.append(line)
    return '\n'.join(section) if section else None


def get_today_stats():
    """Today's win/loss statistics."""
    wins = 0
    losses = 0
    win_sum = 0
    loss_sum = 0
    if not os.path.exists(LOG_PATH):
        return wins, losses, win_sum, loss_sum
    for line in open(LOG_PATH):
        if 'CLOSE' not in line:
            continue
        m = re.search(r'PL=([+-]?[\d,.]+)', line)
        if m:
            pl = float(m.group(1).replace(',', ''))
            if pl > 0:
                wins += 1
                win_sum += pl
            else:
                losses += 1
                loss_sum += pl
    return wins, losses, win_sum, loss_sum


def main():
    if len(sys.argv) < 5:
        print("Usage: python3 tools/preclose_check.py PAIR SIDE UNITS UNREALIZED_PL")
        sys.exit(1)

    pair = sys.argv[1]
    side = sys.argv[2]
    units = int(sys.argv[3])
    upl = float(sys.argv[4])

    print(f'=== 📋 PRECLOSE REFLECTION: {pair} {side} {units}u (unrealized P&L: {upl:+,.0f} JPY) ===')
    print()

    # === 1. Remember your thesis ===
    thesis = get_thesis_from_state(pair)
    if thesis:
        print('[YOUR THESIS]')
        # Extract invalidation conditions from the thesis
        for line in thesis.split('\n'):
            if '転換条件' in line or 'テーゼ' in line or '読み' in line or '根拠' in line or 'CONVICTION' in line:
                print(f'  {line.strip()}')
        print()

    # === 2. Present the facts ===
    facts = []

    # Consecutive loss streak
    streak, streak_sum = get_consecutive_losses()
    if streak >= 2 and upl < 0:
        facts.append(f'⚡ Last {streak} closes were losses (total {streak_sum:+,.0f} JPY). Closing now would be loss #{streak+1}')

    # Pair cumulative loss today
    pair_pl = get_pair_pl_today(pair)
    if pair_pl < -500 and upl < 0:
        facts.append(f'📊 {pair} is already down {pair_pl:+,.0f} JPY realized today')

    # Garbage take-profit
    if 0 < upl <= 30:
        facts.append(f'📊 This close would lock in +{upl:.0f} JPY. After spread, essentially break-even. What is today\'s average win?')

    # Large full close
    if units >= 8000 and upl < -500:
        facts.append(f'📊 {units}u position. Besides full close, have you considered closing half?')

    # Small unrealized loss
    if -100 < upl < 0:
        facts.append(f'📊 Unrealized P&L of {upl:+,.0f} JPY is noise-level. Are you certain the thesis has collapsed?')

    if facts:
        print('[FACTS]')
        for f in facts:
            print(f'  {f}')
        print()

    # === 3. Questions to answer ===
    print('[ANSWER THESE BEFORE CLOSING]')
    if upl < 0:
        print('  1. Has the H1 structure changed? (DI+/DI- crossover, ADX shift)')
        print('  2. Does this match the thesis invalidation condition?')
        print('  3. Is this panic, or a judgment based on analysis?')
    elif 0 < upl <= 50:
        print('  1. What % of ATR have you reached? If under 50%, is this too early?')
        print('  2. What % of the thesis TP target have you hit?')
        print('  3. Is this profit worth the spread + effort?')
    else:
        print('  → Log the reason for closing (what technically changed?)')

    print()
    print('If you can answer these, close — but record the reason in live_trade_log.')
    print('================================================')


if __name__ == '__main__':
    main()
