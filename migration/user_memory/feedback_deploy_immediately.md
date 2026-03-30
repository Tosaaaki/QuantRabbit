---
name: コード変更は即デプロイ
description: live_monitor.pyやスクリプトを変更したら確認を挟まず即座に反映(launchctl restart等)。作ったのに動いてない状態を作るな
type: feedback
---

コード変更したら即デプロイ。「反映させる？」と聞くな。反映するために作ってる。

**Why:** live_monitor.pyの変更(cs_flow追加、registry修正、momentum_close)を作ったのに、旧コードのmonitorが動き続けていた。その間にEUR_USDが旧ロジックで負けた。修正コードが動いていれば防げた可能性がある。

**How to apply:** live_monitor.py、プロンプト、スクリプトを変更したら:
1. テスト実行(python -c で基本動作確認)
2. 即座に `launchctl stop/start` でmonitor再起動
3. err.logとsummary.jsonで反映確認
この3ステップを変更直後に自動的にやれ。ユーザーに「反映する？」と聞くな。
