---
name: ando-nightly-check
description: 毎日22時 — YORIAI/Ando の typecheck + export:web 自動検証
---

YORIAI/Ando の夜間ヘルスチェック。

## 手順

1. **YORIAI リポジトリに移動**
   - パス候補: `/Users/tossaki/事業/YORIAI`
   - 見つからない場合は HANDOFF.md にパス不明と報告して終了
2. **typecheck を実行**
   ```
   cd apps/ando && npm run typecheck
   ```
3. **export:web を実行**
   ```
   npm run export:web
   ```
4. **結果を判定**
   - 両方成功 → 簡潔に「Ando nightly check OK」と報告
   - 失敗あり → エラー内容を確認し、修正を試みる
     - 修正できた場合: commit して報告
     - 修正できない場合: HANDOFF.md にエラー詳細を記録し、ゆうきさんへのアクション依頼を書く

## 注意事項
- ディスク容量不足エラー（ENOSPC）が過去に発生している。その場合は HANDOFF.md に記録
- YORIAI リポジトリが存在しない場合は無理に探さず報告のみ