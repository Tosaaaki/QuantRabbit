#!/usr/bin/env python3
"""決済前リフレクション。BLOCKではなく「思い出させる」。
pretrade_checkと同じ哲学: 過去の教訓を思い出させて判断の質を上げる。
判断を奪わない。判断に必要な情報を突きつける。

Usage:
    python3 tools/preclose_check.py EUR_USD SHORT 12000 -612

出力: 判断に必要な事実。最終判断はトレーダーが下す。
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
    """末尾からの連続損切り数"""
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
    """今日のペア別確定損益"""
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
    """state.mdからそのペアのテーゼ・覚悟を抽出"""
    if not os.path.exists(STATE_PATH):
        return None
    content = open(STATE_PATH).read()
    # ペア名を含むセクションを探す
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
    """今日の勝敗統計"""
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

    print(f'=== 📋 PRECLOSE REFLECTION: {pair} {side} {units}u ({upl:+,.0f}円) ===')
    print()

    # === 1. お前のテーゼを思い出せ ===
    thesis = get_thesis_from_state(pair)
    if thesis:
        print('【お前が書いたテーゼ】')
        # テーゼから転換条件を抽出
        for line in thesis.split('\n'):
            if '転換条件' in line or 'テーゼ' in line or '読み' in line or '根拠' in line or 'CONVICTION' in line:
                print(f'  {line.strip()}')
        print()

    # === 2. 事実を突きつける ===
    facts = []

    # 連続損切り状況
    streak, streak_sum = get_consecutive_losses()
    if streak >= 2 and upl < 0:
        facts.append(f'⚡ 直近{streak}連続損切り（合計{streak_sum:+,.0f}円）。今切ると{streak+1}連続目')

    # ペア累積損失
    pair_pl = get_pair_pl_today(pair)
    if pair_pl < -500 and upl < 0:
        facts.append(f'📊 {pair}は今日すでに{pair_pl:+,.0f}円確定損')

    # ゴミ利確の事実
    if 0 < upl <= 30:
        facts.append(f'📊 この利確は+{upl:.0f}円。スプレッド差し引くとほぼ±0円。今日の平均勝ちは？')

    # 大量一括
    if units >= 8000 and upl < -500:
        facts.append(f'📊 {units}u一括。全部切る以外に半分だけ切る選択肢は？')

    # 小さい含み損
    if -100 < upl < 0:
        facts.append(f'📊 含み損{upl:+,.0f}円はノイズ幅。テーゼ崩壊と確信してるか？')

    if facts:
        print('【事実】')
        for f in facts:
            print(f'  {f}')
        print()

    # === 3. 問い ===
    print('【決済前に答えろ】')
    if upl < 0:
        print('  1. H1構造は変わったか？（DI+/DI-逆転、ADX変化）')
        print('  2. テーゼの転換条件に該当してるか？')
        print('  3. これはパニックか、それとも分析に基づく判断か？')
    elif 0 < upl <= 50:
        print('  1. ATRの何%に達してる？50%未満なら早すぎないか？')
        print('  2. テーゼのTP目標に対して何%か？')
        print('  3. この利確額はスプレッド+手間に見合うか？')
    else:
        print('  → 利確理由をログに明記しろ（「テクニカルの何が変わったか」）')

    print()
    print('答えた上で決済するなら、live_trade_logに理由を明記。')
    print('================================================')


if __name__ == '__main__':
    main()
