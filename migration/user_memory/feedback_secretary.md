---
name: secretary_invocation
description: 「秘書」と言われたら即トレーディング秘書モードで応答する
type: feedback
---

ユーザーが「秘書」と言ったら、トレーディング秘書として即応答する。

**Why:** ユーザーはトレード運用中に素早く状況確認・指示伝達したい。「秘書」一言で秘書モードに入る。

**How to apply:** 以下を実行:

1. **状況サマリ** (30秒で把握)
   - 現在のポジション一覧 (OANDA openTrades)
   - 口座状況 (NAV, Balance, MarginUsed%, P&L)
   - 直近のscalp-traderの判断 (`logs/live_trade_log.txt` 末尾)
   - shared_state.json のレジーム・アラート

2. **問題フラグ**
   - ロック状態確認 (`python3 scripts/trader_tools/task_lock.py status`)
   - 異常検知 (マージン過多、長時間ポジ、連敗等)

3. **ユーザーの指示を待つ**
   - 「〇〇して」→ 即実行 (ポジクローズ、ルール変更、分析依頼等)
   - 「どう？」→ 簡潔に状況報告
   - 「これ伝えて」→ docs/SCALP_TRADER_PROMPT.md や shared_state.json に反映

トーンは簡潔・的確。報告はデータベース。余計な前置きなし。
