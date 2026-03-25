# Claude Codeに投げるプロンプト

以下をClaude Codeにそのまま貼り付けて、定期タスク「trader」を更新してもらう。

---

## 投げるプロンプト

定期タスク「trader」を更新して。30分間隔。Opusモデル。SKILL.mdは `~/.claude/scheduled-tasks/trader/SKILL.md` に直接配置済み。

旧タスク（scalp-trader, market-radar, macro-intel, secretary, analyst等）があれば全部無効化して。v6ではtraderだけがClaude Codeで動く。Coworkのanalyst/secretary/newsは全て廃止済み。

## v6アーキテクチャ

- **trader 1本**: 分析・ニュース・ポジション管理を全部1人でやる
- **30分間隔**: セッション間でハンドオフプロトコルにより状態を引き継ぐ
- **shared_state.json廃止**: traderが直接OANDA API・WebSearch・refresh_factor_cacheでデータ取得

## ハンドオフプロトコル

セッション間の引き継ぎをファイルベースで実現:

1. 新セッション起動 → `logs/.trader_handoff_request` を書く
2. 旧セッションがサイクル末尾でこれを検知 → state.md保存 → `logs/.trader_handoff_complete` を書く → 終了
3. 新セッションが `handoff_complete` を検知 → クリーンアップ → state.md読んでトレード開始

これにより:
- 旧セッションは最新のstate.mdを必ず書いてから終わる
- 新セッションはその最新のstate.mdを読んで開始できる
- セッション間のデータロスがない
