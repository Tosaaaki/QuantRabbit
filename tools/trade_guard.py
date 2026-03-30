#!/usr/bin/env python3
"""毎サイクル実行: トレード品質を自動検知して警告。
margin_alert.pyと併用。traderのBash②/次サイクルBashに組み込む。

方針: 機械的にBLOCKしない。事実を突きつけて認識させる。
"""
import json, re, os, sys
from datetime import datetime, timezone, timedelta

LOG_PATH = 'logs/live_trade_log.txt'
JST = timezone(timedelta(hours=9))

def parse_trades_today():
    """今日のCLOSEトレードをパース"""
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
            # 理由も抽出
            reason_m = re.search(r'reason=(.+?)(?:\||$)', line)
            reason = reason_m.group(1).strip()[:60] if reason_m else ''
            closes.append({'pl': pl, 'pair': pair, 'reason': reason, 'line': line.strip()[:100]})

    return closes


def check_garbage_wins(closes):
    """ゴミ利確の検知"""
    garbage = [c for c in closes if 0 < c['pl'] <= 30]
    if len(garbage) >= 3:
        total = sum(c['pl'] for c in garbage)
        return f'🗑️ 今日{len(garbage)}件の30円以下利確（合計+{total:.0f}円）。この利確額はスプレッド+手間に見合ってるか？'
    return None


def check_recent_losses():
    """直近の連続損切りの理由を表示（機械的パニック判定ではなく、事実の提示）"""
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
        reason = reason_m.group(1).strip()[:60] if reason_m else '理由なし'
        recent_closes.append({'pl': pl, 'pair': pair, 'reason': reason})

    if len(recent_closes) < 3:
        return None

    # 末尾からの連続損切りを取得
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
    lines = [f'  {c["pair"]} {c["pl"]:+,.0f}円 | {c["reason"]}' for c in streak[-5:]]  # 最大5件表示

    result = f'⚡ 直近{len(streak)}連続損切り（合計{total:+,.0f}円）。理由を確認しろ:\n'
    result += '\n'.join(lines)
    result += '\n  → 全部「H1構造変化」なら正しい判断。同じ理由の繰り返しならパニック'
    return result


def check_rr_ratio(closes):
    """RR比の事実提示"""
    wins = [c['pl'] for c in closes if c['pl'] > 0]
    losses = [c['pl'] for c in closes if c['pl'] < 0]

    if len(wins) < 5 or len(losses) < 3:
        return None

    avg_win = sum(wins) / len(wins)
    avg_loss = sum(losses) / len(losses)
    rr = avg_win / abs(avg_loss)

    if rr < 0.7:
        return f'📊 RR: 平均勝ち+{avg_win:.0f}円 vs 平均負け{avg_loss:.0f}円（RR={rr:.2f}）。勝ちを伸ばす意識を持て'
    return None


def check_pair_loss(closes):
    """特定ペアでの累積損失"""
    pair_pl = {}
    for c in closes:
        pair = c['pair']
        if pair not in pair_pl:
            pair_pl[pair] = 0
        pair_pl[pair] += c['pl']

    alerts = []
    for pair, pl in pair_pl.items():
        if pl < -1000:
            alerts.append(f'📊 {pair}: 今日{pl:+,.0f}円。この事実を踏まえた上で判断しろ')
    return alerts


def main():
    closes = parse_trades_today()
    if not closes:
        return

    alerts = []

    # 直近の連続損切り（理由付き）
    recent = check_recent_losses()
    if recent:
        alerts.append(recent)

    # ゴミ利確
    garbage = check_garbage_wins(closes)
    if garbage:
        alerts.append(garbage)

    # RR比
    rr = check_rr_ratio(closes)
    if rr:
        alerts.append(rr)

    # ペア別累積損失
    pair_alerts = check_pair_loss(closes)
    alerts.extend(pair_alerts)

    if alerts:
        print('=== 🛡️ TRADE GUARD ===')
        for a in alerts:
            print(a)
        print('======================')


if __name__ == '__main__':
    main()
