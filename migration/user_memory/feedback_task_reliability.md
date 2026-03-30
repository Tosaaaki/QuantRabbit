---
name: task_reliability_lessons
description: Scheduled Task信頼性の教訓 - ディスク容量・ロック・データ鮮度・トークン節約・プロセス管理
type: feedback
---

## タスク停止の原因パターン

1. **ディスク容量98%超** → ログ書き込み失敗 → タスクフリーズ
   - 対策: market-radarにディスク保守ロジック追加済み (80%でクリーンアップ、90%でCRITICAL)

2. **ロック解放漏れ** → タスクがエラーで死ぬとロックがrunning:trueのまま残る
   - 対策: タイムアウト5分で強制解放（PID生存でも超過なら解放）
   - reaper (`reap_stale_agents.sh`) が孤立ロックも自動解放

3. **Maxプラン使用量超過** → タスクが実行されなくなる
   - 対策: タスク間隔を広げる + 不要タスク停止 + プロンプト英語化(トークン40-50%削減)

4. **shared_stateとOANDAの不整合** → クローズ済みポジションをopenとして分析
   - 対策: OANDA API is SINGLE SOURCE OF TRUTH。shared_stateは参考値

5. **タイムスタンプの日付ズレ** → `date -u`使用義務化で修正中

6. **並列プロセス蓄積 (2026-03-19発見)** → Claude Desktopが各タスクを独立プロセスで起動。完了後もプロセスが数分残存し、300-400MB/個のメモリを消費
   - 原因: claude-codeプロセスのライフサイクルがClaude Desktop側で管理され、タスク完了後すぐにexitしない
   - 対策:
     a. **グローバルロック** (`global_agent`): 全タスクが同一ロックを共有。常に1タスクのみ実行
     b. **互いに素な間隔**: scalp=5分, radar=7分, secretary=11分, macro=19分 → 同時起動の衝突を最小化
     c. **reaper** (`scripts/trader_tools/reap_stale_agents.sh`): launchd 1分間隔で実行。4分超のclaude-codeプロセスをkill + 孤立ロック解放。`--resume`付き（ユーザー会話）は保護
     d. **ローテーションyield**: `--caller`で同一タスクの連続実行を30秒防止

## 重要ファイル
- ロック: `scripts/trader_tools/task_lock.py` (グローバルロック `global_agent`)
- reaper: `scripts/trader_tools/reap_stale_agents.sh`
- reaper plist: `~/Library/LaunchAgents/com.quantrabbit.reaper.plist`
- ロック状態: `logs/locks/` ディレクトリ

**Why:** 夜間自動運用で複数問題が同時発生し、タスク停止・メモリ圧迫が繰り返された。

**How to apply:** タスク設定変更時は上記6点をチェック。特に間隔変更時はLCM計算で衝突頻度を確認。reaperが動いているかも `logs/reaper.log` で確認。
