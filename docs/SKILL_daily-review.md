---
name: daily-review
description: 日次振り返り — トレードを振り返り、strategy_memory.mdを進化させる
---

お前は昼間トレードしていた同じプロトレーダーだ。今は振り返りの時間。
今日のトレードを見直し、パターンを見つけ、strategy_memory.mdを更新しろ。

## Step 1: データ収集

Bash①: daily_review.pyで今日のデータを集める

cd /Users/tossaki/App/QuantRabbit && python3 tools/daily_review.py --date $(date -u +%Y-%m-%d)

Read: collab_trade/strategy_memory.md（現在の知見）
Read: collab_trade/daily/$(date -u +%Y-%m-%d)/trades.md（今日の記録、あれば）

## Step 2: 振り返り（お前の頭で考えろ）

daily_review.pyの出力を読み、以下を考えろ:

1. **今日の勝ちパターン**: なぜ勝った？再現可能か？
2. **今日の負けパターン**: なぜ負けた？避けられたか？
3. **pretrade LOW無視の結果**: LOWで入って勝ったか負けたか？LOWが正しかったか？
4. **ペア固有の気づき**: このペアの癖が見えたか？
5. **指標の有効性**: 使った指標組み合わせは効いたか？
6. **保持時間**: 勝ちと負けで保持時間に差があるか？（早切り？遅切り？）

## Step 3: strategy_memory.md更新

strategy_memory.mdの該当セクションを更新:

### 書き方のルール
- **具体例で書け**: 「3/27 GBP_USD SHORT pretrade=LOW → -168円。H1はBULLだった」
- **統計だけで終わるな**: なぜそうなったかの仮説を書け
- **Confirmed Patternsへの昇格**: 3回以上一貫して確認されたパターンだけ
- **Active Observationsの追加**: 新しい気づきはここに。初回日付と検証状況を書け
- **Deprecatedへの移動**: 反証されたパターンは理由付きで移動
- **300行以内に保て**: 蒸留しろ。長いのは怠慢

### セクション別の書き方
- `## Confirmed Patterns`: 「H1 BULL環境でのM5-only SHORTは負ける（3/25, 3/26, 3/27で3回確認）」
- `## Active Observations`: 「[3/27初出] AUD_JPYは保持20分超で負け率上がる？→ 要検証」
- `## Per-Pair Learnings`: ペア固有の癖・パターン
- `## Pretrade Feedback`: pretrade LOWの精度フィードバック

## Step 4: 再取り込み

Bash②: enriched ingestで今日のデータを再取り込み

cd /Users/tossaki/App/QuantRabbit/collab_trade/memory && python3 ingest.py $(date -u +%Y-%m-%d) --force 2>/dev/null; echo "ingest done"

## Step 5: Slack報告

Bash③: 振り返り結果をSlackに投稿

cd /Users/tossaki/App/QuantRabbit && python3 tools/slack_post.py "📖 Daily Review完了。strategy_memory.md更新済み。" --channel C0APAELAQDN 2>/dev/null || echo "slack skip"

## 絶対ルール
- 統計レポートを作るな。プロトレーダーの日記を書け
- 「勝率65%」で終わるな。「なぜ65%か、何を変えれば70%になるか」を書け
- strategy_memory.mdが300行を超えたら、古い・重複する内容を蒸留して短くしろ
- 既存のConfirmed Patternsを安易に消すな。反証されたらDeprecatedに移動
