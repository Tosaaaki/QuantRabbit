#!/usr/bin/env python3
"""日次サマリーを #qr-daily に投稿する
Usage: python3 scripts/trader_tools/slack_daily_summary.py [--date YYYY-MM-DD]
"""
import urllib.request, json, sys, os, argparse, glob
from datetime import datetime, timedelta


def load_config():
    cfg = {}
    path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'env.toml')
    for line in open(path):
        line = line.strip()
        if '=' in line and not line.startswith('#'):
            k, v = line.split('=', 1)
            cfg[k.strip()] = v.strip().strip('"')
    return cfg


def post(text, channel_id, token):
    payload = {"channel": channel_id, "text": text}
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(
        "https://slack.com/api/chat.postMessage",
        data=data,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8"
        }
    )
    resp = json.loads(urllib.request.urlopen(req).read())
    if not resp.get('ok'):
        print(f"ERROR: {resp.get('error', 'unknown')}", file=sys.stderr)
        sys.exit(1)
    return resp


def get_account_summary(cfg):
    token = cfg['oanda_token']
    acct = cfg['oanda_account_id']
    req = urllib.request.Request(
        f"https://api-fxtrade.oanda.com/v3/accounts/{acct}/summary",
        headers={"Authorization": f"Bearer {token}"}
    )
    data = json.loads(urllib.request.urlopen(req).read())['account']
    return {
        'balance': float(data['balance']),
        'nav': float(data['NAV']),
        'unrealized_pl': float(data['unrealizedPL']),
        'margin_used': float(data['marginUsed']),
        'open_positions': int(data['openPositionCount']),
        'open_trades': int(data['openTradeCount']),
    }


def get_daily_trades(date_str):
    """trades.mdからその日のエントリー/決済を集計"""
    base = os.path.join(os.path.dirname(__file__), '..', '..', 'collab_trade', 'daily', date_str)
    trades_path = os.path.join(base, 'trades.md')
    if not os.path.exists(trades_path):
        return None

    content = open(trades_path).read()
    entries = content.count('ENTRY') + content.count('entry') + content.count('Entry')
    closes = content.count('CLOSE') + content.count('close') + content.count('Close') + content.count('決済')
    return {'entries': entries, 'closes': closes, 'has_data': True}


def get_daily_pl_from_log(date_str):
    """live_trade_log.txtからその日の確定損益を集計"""
    log_path = os.path.join(os.path.dirname(__file__), '..', '..', 'logs', 'live_trade_log.txt')
    if not os.path.exists(log_path):
        return 0.0

    total_pl = 0.0
    for line in open(log_path):
        if date_str not in line:
            continue
        # PLの抽出: PL=+123.45 or PL=-67.89 パターン
        if 'PL=' in line:
            try:
                pl_part = line.split('PL=')[1].split()[0].replace('円', '').replace(',', '')
                total_pl += float(pl_part)
            except (ValueError, IndexError):
                pass
        # realized_pl パターン
        elif 'realized_pl' in line:
            try:
                import re
                m = re.search(r'realized_pl["\s:=]+([+-]?[\d.]+)', line)
                if m:
                    total_pl += float(m.group(1))
            except (ValueError, AttributeError):
                pass
    return total_pl


def get_performance_summary():
    """trade_performance.pyの結果からWR等を取得"""
    import subprocess
    perf_path = os.path.join(os.path.dirname(__file__), 'trade_performance.py')
    try:
        result = subprocess.run(
            [sys.executable, perf_path],
            capture_output=True, text=True, timeout=30,
            cwd=os.path.join(os.path.dirname(__file__), '..', '..')
        )
        output = result.stdout
        # Overall行を探す
        for line in output.split('\n'):
            if 'Overall' in line and 'WR=' in line:
                return line.strip()
        return None
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', default=None, help='YYYY-MM-DD (default: yesterday)')
    args = parser.parse_args()

    if args.date:
        target_date = args.date
    else:
        target_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

    cfg = load_config()
    acct = get_account_summary(cfg)
    daily_pl = get_daily_pl_from_log(target_date)
    trades_info = get_daily_trades(target_date)
    perf_line = get_performance_summary()

    # サマリー構築
    lines = []
    lines.append(f"\U0001f4ca *Daily Summary: {target_date}*")
    lines.append("")

    # 日次損益
    pl_icon = "\U0001f7e2" if daily_pl >= 0 else "\U0001f534"
    lines.append(f"{pl_icon} *日次確定損益: {daily_pl:+,.0f}円*")

    # トレード数
    if trades_info and trades_info['has_data']:
        lines.append(f"\U0001f4dd エントリー: {trades_info['entries']}件 | 決済: {trades_info['closes']}件")
    else:
        lines.append(f"\U0001f4dd トレード記録なし")

    lines.append("")

    # 口座状態
    lines.append("*口座状態:*")
    lines.append(f"  Balance: {acct['balance']:,.0f}円")
    lines.append(f"  NAV: {acct['nav']:,.0f}円")
    lines.append(f"  未実現損益: {acct['unrealized_pl']:+,.0f}円")
    lines.append(f"  オープン: {acct['open_trades']}本 ({acct['open_positions']}ペア)")

    # パフォーマンス
    if perf_line:
        lines.append("")
        lines.append(f"*通算:* {perf_line}")

    message = "\n".join(lines)
    channel = cfg.get('slack_channel_daily', cfg.get('slack_channel_id', ''))

    post(message, channel, cfg['slack_bot_token'])
    print(f"Posted daily summary for {target_date} to #{channel}")


if __name__ == '__main__':
    main()
