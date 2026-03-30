---
name: feedback_secretary_live_first
description: 秘書モードでは必ずOANDA APIライブデータを最初に取得。shared_stateはキャッシュとして古い可能性がある
type: feedback
---

秘書モードの状況レポートでは、**shared_state.jsonを鵜呑みにせず、必ずOANDA APIからライブデータを最初に取得する**。

**Why:** 2026-03-19にshared_stateが21時間前のデータのまま更新停止していたが、そのまま読み上げてAUD/USDの含み損を-9.9pipと報告。実際は-17.8pipまで悪化していた。ユーザーに指摘されて初めてライブ確認した。

**How to apply:** 「秘書」モード起動時、最初のアクションは常にOANDA API直接取得（口座サマリー + openTrades）。shared_stateは補助情報として使う。updated_atが10分以上古ければ「データが古い」と明示する。
