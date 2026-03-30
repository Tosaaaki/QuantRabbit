---
name: active_trading_strategy
description: v4.4 SL-free裁量トレード戦略 — 棚卸し中心、災害SLのみ、TP動的管理。Phase 1検証中
type: project
---

## 現在の戦略: SL-free裁量アプローチ (v4.4, 2026-03-20~)

### 核心思想
- **固定SLは入れない。** ノイズで狩られて即反発のパターンを排除
- **Claudeの最大の価値 = 棚卸し（position housekeeping）。** 毎サイクル全ポジを裁量評価
- **TPは動的。** 初期値を入れるが市況に応じて調整し続ける

### 3層防衛体制
1. **OANDA災害SL (-25pip)** — 相場崩壊時のみ発動する保険
2. **monitor機械管理 (30秒)** — trail, partial, momentum_close, BE移動。cut_at_pip=-20(災害レベル)
3. **Claude裁量棚卸し (2-3分)** — 本命。テーゼ生存判定で閉じるか続けるか決める

### 棚卸しの判断基準
- **テーゼ崩壊 = 閉じる**: H1構造変化、cs_flow反転、マクロ変化、エントリー根拠全消失
- **ノイズ = 閉じるな**: micro_vel一時反転、-2〜-5pip逆行、M1指標の一瞬の動き、**髭形成の途中**
- 「この逆行は髭になるんじゃないか？」を必ず考える

### 根拠（ユーザーのボット実績）
- TP有/SL無/高頻度/狭TP → 5時間で資産+10%達成
- しかし棚卸し不足でそれ以上の損失 → Claude裁量棚卸しが解決策
- 理論的に日次15-20%が射程

### ロードマップ
- **Phase 1（現在）: Claude entry + Claude棚卸し** — 棚卸し精度の実戦検証。ノイズで閉じない/テーゼ死亡で閉じる、が安定するまで
- **Phase 2（棚卸し安定後）: monitor entry + Claude棚卸し専任** — monitorに30秒entryロジック追加。ClaudeがGO/CAUTION/STOPレジーム制御。Claudeはentry一切やらず棚卸し専任。ボットの速度+Claudeの目

### 今日の教訓 (2026-03-20)
- USD_JPY SHORT -2.4pip: micro_vel一時反転で閉じたがテーゼ(H1 BEAR+天井打ち)は生存。上髭形成の途中だった → ノイズvsテーゼ崩壊の区別をプロンプトに焼き込み済
- registry登録漏れ(EUR_USD): 注文→登録→確認を1動作としてプロンプトに明記済

### 変遷
- v4.0 (03-20): 5→3エージェント(trader/analyst/secretary)
- v4.1: スコア廃止、生データ提供、原則ベース
- v4.2: 禁止→適応転換(5原則)、MTFリバージョン、セッション適応
- v4.3: 予測ファースト、cs_flow追加、momentum_close実装
- **v4.4: SL-free裁量。固定SL廃止、棚卸しが生命線、ノイズvsテーゼ崩壊の区別、Phase 1/2ロードマップ**
