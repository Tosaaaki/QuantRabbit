---
name: jam-deploy
description: 毎日13時 — git push + seed実行 + HANDOFF消化
---

jam-session-map-package リポジトリの日次デプロイタスク。

## 手順

1. **STATE.md を読んで現在の状態を把握する**
2. **HANDOFF.md を確認** — 未対応タスク（✅でないもの）があれば対応する
   - seed/route.ts への会場追加
   - venue-patrol.json の更新
   - DB への INSERT（単発イベント等）
   - その他 Cowork からの引き継ぎ作業
3. **未コミットの変更があれば git commit する**
4. **git push origin main** — Vercel に自動デプロイされる
5. **seed API を実行**（変更があった場合のみ）
   - `curl -X POST https://enso-music.jp/api/seed`
   - レスポンスの venuesInserted / eventsInserted を確認
6. **HANDOFF.md の対応済みタスクに ✅ をつける**
7. **STATE.md を更新** — 件数・直近作業ログ・git push待ちを最新に
8. **docs/ に作業ログを残す**（実施内容がある場合）

## 注意事項
- seed と venue-patrol.json は必ずセットで更新する
- 会場追加時はセッション開催実績を必ず確認する（docs/43_会場セッション確認フロー.md 参照）
- 変更がなければ「変更なし、スキップ」と報告して終了
- エラーが出たら HANDOFF.md に記録してゆうきさんに報告