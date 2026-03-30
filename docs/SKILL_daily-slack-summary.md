---
name: daily-slack-summary
description: 毎朝7:00 JSTにSlack #qr-dailyへ日次トレードサマリーを自動投稿
---

日次トレードサマリーをSlackの #qr-daily チャンネルに投稿する。

手順:
1. 以下のスクリプトを実行:
```bash
cd /Users/tossaki/App/QuantRabbit && python3 tools/slack_daily_summary.py
```
2. 成功したら完了。エラーが出たら内容を確認して修正を試みる。