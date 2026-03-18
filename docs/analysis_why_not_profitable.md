# QuantRabbit 収益性阻害要因の網羅的分析

**作成日**: 2026-03-11
**分析対象**: QuantRabbit ローカルV2運用（USD/JPY 自律トレーディングエージェント）
**根拠**: `docs/TRADE_FINDINGS.md`、`AGENTS.md`、`docs/RISK_AND_EXECUTION.md`、`docs/SL_POLICY.md`、`execution/order_manager.py`、`execution/risk_guard.py`、各ワーカー実装、ローカルログ実測値

---

## 結論（サマリ）

直近24時間の実測値は **346 trades / net_jpy=-440.9 / PF=0.452 / win_rate=0.289** であり、前日は **398 trades / net_jpy=-1642.1 / PF=0.558**。システムは「大量に薄い負けトレードを打ち続ける」構造になっている。根本原因は単一ではなく、以下の7カテゴリに分類される構造的問題の複合である。

---

## 1. エントリー品質の低さ（最大要因）

### 1.1 RangeFader の低品質 fade が大量に通過

- **実測**: RangeFader 211 trades / -156.0 JPY / expectancy=-0.7 JPY/trade（24h）
- `RangeFader|long|neutral-fade|range_fade|p0`: 18 trades / -21.6 JPY / win_rate=0.111
- `RangeFader|long|buy-fade|range_fade|p0`: 16 trades / -8.2 JPY / win_rate=0.000
- `entry_probability 0.29-0.30` の薄い fade がフィルターを通過し、spread 負けで終わる
- **直近10分の reject/fill 比率**: `entry_probability_reject=128` vs `filled=3`（reject が圧倒的だが、通過した3件も期待値が低い）

### 1.2 PrecisionLowVol の hostile short lane

- **実測**: 8 trades / -92.2 JPY / avg -1.387 pips / win_rate=0.125
- `projection.score=-0.125/-0.14` の negative projection で short fade を打ち続けていた
- `setup_quality=0.262-0.378` と低品質
- broker SL が付いていても、entry quality 自体が負のため STOP_LOSS_ORDER を連打

### 1.3 MomentumBurst の short side 偏損

- **実測（前日24h）**: 24 trades / -617.21 JPY / win_rate=50.0% / PF=0.675
- short は 11 trades / -598.47 JPY / win_rate=36.4%（ほぼ全損失の本体）
- oversold 帯で bearish marubozu をそのまま売り追いする pattern が上位損失
- `rsi=23-33` の極端な oversold で non-reaccel short を打っていた

### 根本原因

各戦略が「打てる条件」を広く取りすぎており、**期待値が負の setup でもエントリーを通過させる**構造になっている。strategy-local の quality guard が後追いで次々と追加されているのは、元来のエントリー条件が粗いことの裏返し。

---

## 2. ストップロス管理の構造的欠陥

### 2.1 broker SL の欠落（致命的バグ、3/11修正中）

- **実測**: `WickReversalBlend avg_loss_vs_sl=3.84x, max=5.46x`
- **実測**: `VwapRevertS avg_loss_vs_sl=3.66x, max=7.61x`
- `VwapRevertS` は `sl_pips=1.8` に対して `-13.7p / -12.4p` の実損失
- **原因**: worker 側 dedicated env の `ORDER_FIXED_SL_MODE=1` が、発注主体の `quant-order-manager` に伝播していなかった
- global baseline `ORDER_FIXED_SL_MODE=0` が優先され、live order から `stopLossOnFill` が抜けていた

### 2.2 SL設定の複雑さによる設定ミスリスク

- SL決定に関わるレイヤーが5段階以上ある:
  1. Worker の `entry_thesis.sl_pips`
  2. order_manager の dynamic SL / min RR / TP cap
  3. `stopLossOnFill` の strategy override
  4. fill 後の保護更新
  5. exit_worker の独自判定
- 設定キーが `ORDER_FIXED_SL_MODE`, `ORDER_ALLOW_STOP_LOSS_ON_FILL_STRATEGY_*`, `ORDER_ENTRY_HARD_STOP_PIPS*`, `ORDER_DYNAMIC_SL_*` 等、10種類以上散在
- **結果**: 意図した SL が実際に適用されているかの検証が困難で、バグが長期間残存する

---

## 3. EXIT ロジックの非最適性

### 3.1 「損失ではクローズしない」ルールの副作用

- 全ワーカーが `PnL<=0 は原則ホールド` を採用
- 方向性が完全に間違っているトレードでも min_hold（scalp>=10s, micro>=15-20s）まで強制保有
- broker SL が無い場合（上記2.1の欠陥と合わさると）、SL到達前に大幅な含み損を抱える

### 3.2 static exit の残存

- `RangeFader` の exit が `setup_quality` と連動していなかった（3/11に修正開始）
- `entry_probability 0.29-0.30` の低品質 trade も、高品質 trade と同じ exit threshold で処理
- hostile setup でも `max_hold_loss` まで保有させていた

---

## 4. 過剰なフィルター/ゲート層による Entry 数の激減

### 4.1 reject >> fill の構造

- **実測（24h）**: `preflight_start=2112, filled=404, entry_probability_reject=2242, perf_block=1672`
- fill 率はおよそ **8-15%**（残り85%以上がフィルターで拒否）
- `risk_mult_perf=0.55`, `order_probability_scale=0.3781` が掛け合わさり、ほぼすべてを落とす

### 4.2 フィルター層の多重化

以下のフィルターが直列に存在:
1. strategy-local quality guard
2. shared `participation_alloc`（units trim + probability offset）
3. `dynamic_alloc` multiplier
4. `strategy_feedback` advice
5. `auto_canary` override
6. `pattern_gate`（opt-in）
7. `perf_guard`（performance block）
8. `profit_guard`
9. `forecast_gate`
10. `brain` preflight（Ollama）
11. `strategy_entry.py` の blackboard coordination
12. `order_manager` の probability reject / min units / margin check

**問題**: 期待値の高い trade まで巻き添えで reject されている可能性が高い。winner lane の `boost_participation` は追加されたが、そもそもの reject 率が高すぎる。

---

## 5. スプレッドコスト vs 期待値の構造的不整合

### 5.1 スプレッド負けの累積

- USD/JPY スプレッド: 平常時 **0.8 pips**
- scalp 系ワーカーの TP は 1-3 pips 程度
- 往復スプレッドコスト（0.8 pips）は TP の **26-80%** に相当
- **win_rate=0.289** では、1勝あたりの利益がスプレッドコストを吸収できない

### 5.2 大量の薄利トレードの累積損

- `MARKET_ORDER_TRADE_CLOSE` による微益決済が多い一方、`STOP_LOSS_ORDER` は SL 幅相当の損失
- **非対称性**: 勝ちは小さく、負けは SL 幅まで大きい
- MomentumBurst: winner avg=+2.238p vs loser avg=-4.044p（約1:1.8の非対称）

---

## 6. システム複雑性による運用品質の低下

### 6.1 ワーカー数の爆発

- 現在 **40以上** の dedicated ワーカー（scalp_*, micro_* 等）が存在
- 各ワーカーが独自の entry/exit/config を持ち、相互作用が把握困難
- **同一口座** で多数の戦略が同時に稼働し、ポジション管理が複雑化

### 6.2 設定の散在

- `ops/env/` 配下に戦略ごとの .env ファイルが大量に存在
- `config/` 配下の JSON/YAML が自動生成・手動編集の混在
- `participation_alloc.json`, `dynamic_alloc.json`, `strategy_feedback.json`, `pattern_book.json`, `auto_canary_overrides.json` 等が相互に影響

### 6.3 改善の「モグラ叩き」化

- TRADE_FINDINGS を見ると、**1つの戦略の1つのsetupを修正→別のsetupが露出→修正→...** のサイクル
- 根本的なエントリー品質の改善ではなく、「loser cluster を個別に潰す」アプローチ
- 3/11だけで5件以上の strategy-local guard が追加されている

---

## 7. 戦略設計の構造的問題

### 7.1 過学習リスク

- `WFO_OVERFIT_REPORT.md` で過学習検知の仕組みはあるが、実運用での feedback loop が速すぎる
- `auto_canary`, `strategy_feedback`, `dynamic_alloc` が数時間〜24時間の窓で戦略パラメータを動的変更
- 短期の noise に反応して trim/boost を繰り返す可能性

### 7.2 レンジ相場への過剰適応

- 全体の構造が「レンジでの逆張り（mean reversion / range fade）」に偏っている
- `RangeFader` が最多取引数（211/346 = 61%）を占める
- トレンド相場では range fade の損失が累積する構造

### 7.3 risk/reward 比の設計不良

- `risk_guard.py` の `POCKET_DD_LIMITS`: scalp=3%, micro=5%（equity比）
- 個別トレードの risk は小さいが、大量の負けトレードが累積してドローダウンを形成
- 1トレードあたりの期待値がゼロ近傍なら、スプレッド分だけ確実に負ける

---

## 改善の優先順位（提案）

| 優先度 | カテゴリ | 具体策 |
|--------|---------|--------|
| **1** | SL管理 | broker SL の全戦略確実適用（3/11修正中）。設定の一本化 |
| **2** | エントリー品質 | 戦略数を絞り、期待値が実証された setup のみを稼働させる |
| **3** | スプレッド対策 | TP/SL 比率を見直し、スプレッドコストを吸収できる最低 RR を強制 |
| **4** | フィルター簡素化 | 12段のフィルターを整理し、reject 率を下げつつ品質を維持 |
| **5** | 戦略集中 | 40+ワーカーを5-10に絞り、実績のある戦略のみに資源集中 |
| **6** | EXIT最適化 | setup quality に応じた dynamic exit の全戦略適用 |
| **7** | 過学習対策 | feedback loop の反応速度を落とし、統計的に有意な窓でのみ調整 |

---

## 出典

- `docs/TRADE_FINDINGS.md`: 2026-03-09〜03-11 の全エントリ
- `AGENTS.md`: セクション1-10
- `docs/RISK_AND_EXECUTION.md`: エントリー/EXIT/リスク制御
- `docs/SL_POLICY.md`: SL生成順序と分岐
- `execution/risk_guard.py`: リスクパラメータ
- `execution/order_manager.py`: 発注ロジック
- `docs/TODO_EXIT_ALIGNMENT.md`: EXIT整合性チェック
- `docs/WFO_OVERFIT_REPORT.md`: 過学習検知
