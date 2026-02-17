# 型（Kata）設計: `scalp_m1scalper`（M1Scalper）

このドキュメントは、`workers/scalp_m1scalper` の「型（Kata）」を **再現可能に作る**ための設計書です。
型は、過去トレードから抽出したパターンを DB/JSON に継続学習し、次のエントリーで `block / scale`（拒否/縮小/増強）に接続します。

## ゴール
- M1Scalper の「良い形/悪い形」を履歴から定量化し、次の発注でサイズ調整に反映する（主に scale）。
- 型は戦略ごとに最適化する（全戦略共通の一律型にしない）。
- 互換性を壊す変更（`pattern_id` の分割軸追加/変更）は、移行計画込みで扱う。

## 1. 全体フロー（VM常時運用）
1. `logs/trades.db` のクローズ済みトレードを材料にする
2. `scripts/pattern_book_worker.py` が `logs/patterns.db` を更新
3. deep 分析（`analysis/pattern_deep.py`）で `pattern_scores / drift / cluster` を更新
4. `execution/order_manager.py` の preflight で `workers/common/pattern_gate.py` を評価
5. opt-in のエントリーに限り、型に応じて `block/scale` する

systemd（VM）:
- `systemd/quant-pattern-book.service`
- `systemd/quant-pattern-book.timer`（5分周期）

## 2. 何を「型」と呼ぶか（pattern_id）
型の最小単位は `pattern_id`（文字列）です。トレードは必ずどれか 1 つの `pattern_id` に割り当てられます。

`pattern_id` は `analysis/pattern_book.py:build_pattern_id()` で生成され、次のトークンを `|` 区切りで連結します。

- `st:` strategy tag（戦略名）
- `pk:` pocket（資金/口座内の区分）
- `sd:` direction（long/short）
- `sg:` signal mode（strategy 固有のモード。`signal_mode/entry_mode/mode`）
- `mtf:` MTF gate（`mtf_regime_gate/mtf_gate`）
- `hz:` horizon gate（`horizon_gate`）
- `ex:` extrema gate reason（`extrema_gate_reason`）
- `rg:` range bucket（レンジ内の位置; `entry_range_bucket` または `section_axis + entry_ref` から導出）
- `pt:` pattern tag（戦略が任意で付ける追加タグ; `pattern_tag`）

重要:
- `workers/common/pattern_gate.py` は **情報量がゼロの pattern_id（sg/mtf/hz/ex/rg/pt が全部 `na`）を “generic” とみなして no-op** にします。
- つまり M1Scalper で Pattern Gate を効かせるには、少なくとも `pt` または `rg`（または `sg/mtf/hz/ex`）を “低カーディナリティ” で埋める必要があります。

## 3. M1Scalper の「型」を構成する情報源（entry_thesis）
M1Scalper はエントリー直前に `entry_thesis` を組み立てて `market_order(..., entry_thesis=...)` / `limit_order(..., entry_thesis=...)` に渡します。

起点:
- `workers/scalp_m1scalper/worker.py`（`entry_thesis = {...}`）

現行（コード上）で `pattern_id` に関与しうるキー:
- `pattern_tag`（`pt:`）
  - `analysis.pattern_stats:derive_pattern_signature()` 由来の追加タグ（ローソク形状系）
- `entry_price`（`_range_bucket()` の fallback 候補）
  - limit のみ付与。`section_axis` が無い限り `rg:` は `na` のまま

現行で “まだ entry_thesis に入れていない” ため `pattern_id` 側が常に `na` になるキー:
- `signal_mode`（`sg:`）
- `mtf_regime_gate` / `mtf_gate`（`mtf:`）
- `horizon_gate`（`hz:`）
- `extrema_gate_reason`（`ex:`）
- `section_axis` / `entry_ref` / `entry_range_bucket`（`rg:`）

## 4. 現状のサンプル厚み（VM実データ）
スナップショット日: 2026-02-12

閉じたトレード数（VM `logs/trades.db`）:
- `M1Scalper`: 3453 trades（last_close: 2026-02-10T12:31:49Z）

Pattern Gate が評価対象にできる “非generic” の割合（VM `logs/patterns.db`）:
- total=3435
- generic=2267
- informative=1168

直近（2026-02-09 以降）の M1Scalper は 7 trades 全て informative（generic 0）。

確認コマンド（VM）:
```bash
scripts/vm.sh -p quantrabbit -z asia-northeast1-a -m fx-trader-vm -t sql \
  -f /home/tossaki/QuantRabbit/logs/patterns.db \
  -q "SELECT COUNT(*) AS total, SUM(CASE WHEN pattern_id LIKE '%|sg:na|mtf:na|hz:na|ex:na|rg:na|pt:na' THEN 1 ELSE 0 END) AS generic, SUM(CASE WHEN pattern_id NOT LIKE '%|sg:na|mtf:na|hz:na|ex:na|rg:na|pt:na' THEN 1 ELSE 0 END) AS informative FROM pattern_trade_features WHERE strategy_tag='M1Scalper';"
```

## 5. エントリー制御（Pattern Gate）の仕様（M1Scalper適用）
実装:
- `workers/common/pattern_gate.py`
- 呼び出し元: `execution/order_manager.py`（preflight）

重要: **opt-in したエントリーだけに適用**
- グローバル強制: `ORDER_PATTERN_GATE_GLOBAL_OPT_IN=1`（通常は 0 を維持）
- M1Scalper 側: `entry_thesis["pattern_gate_opt_in"]=...` を付与して opt-in

M1Scalper の opt-in（導入）:
- `workers/scalp_m1scalper/worker.py` に `entry_thesis["pattern_gate_opt_in"]` を追加し、`SCALP_M1SCALPER_PATTERN_GATE_OPT_IN`（default: true）で ON/OFF する
  - 導入PR: `codex/patterngate-m1scalper-optin`

ブロック:
- `quality in ORDER_PATTERN_GATE_BLOCK_QUALITIES`（既定: `avoid`）
- `trades >= ORDER_PATTERN_GATE_BLOCK_MIN_TRADES`（既定: 90）
- `robust_score <= ORDER_PATTERN_GATE_BLOCK_MAX_SCORE`（既定: -0.9）
- `p_value <= ORDER_PATTERN_GATE_BLOCK_MAX_PVALUE`（既定: 0.35）

スケール:
- `suggested_multiplier` をベースに `ORDER_PATTERN_GATE_SCALE_MIN/MAX` へクリップ
- “weak” は縮小寄りに倒す（`ORDER_PATTERN_GATE_REDUCE_FALLBACK_SCALE`）
- ドリフト悪化時は縮小上限を下げる（`ORDER_PATTERN_GATE_DRIFT_*`）

無効化（即時ロールバック）:
- `ORDER_PATTERN_GATE_ENABLED=0`（全戦略の Pattern Gate を停止）
- `SCALP_M1SCALPER_PATTERN_GATE_OPT_IN=0`（M1Scalper だけ opt-in を止める）

## 6. M1Scalper の型を “強くする” 方針（設計ルール）
M1Scalper は「ローソク形状（`pt`）だけ」に依存すると、次のリスクがあります。
- `pt:na` が多い期間は generic になり、Gate が何もしない
- `pt` を増やしすぎるとパターンが分裂し `learn_only` が増える

推奨方針（段階的）:
1. まずは現行の `pt:` を活かして opt-in し、Gate の評価ログ（orders.db）を観測する
2. “generic が多い” 期間が問題なら、追加の分割軸を 1 つだけ入れる
   - 例: `sg:` に `m1scalper` のサブモード（低カーディナリティ: 3-6 程度）
   - 例: `mtf:` に MTF gate（低カーディナリティ: 3-5 程度）
3. `rg:`（追いかけ/押し目の位置）を入れる場合は、必ずバケット化する
   - `entry_range_bucket` を `bot/low/mid/high/top` の 5 分割で付ける
   - `section_axis` + `entry_ref` の整合（古いトレードの欠損をどう扱うか）を移行計画に含める

注意（互換性）:
- `pattern_id` の分割軸を追加すると、既存の `pattern_scores` と一致しなくなり、反映直後は `learn_only` が増える。
- 変更するなら “どのくらい learn_only が増えて、いつ効き始めるか” を `docs/KATA_PROGRESS.md` に記録する。

## 7. 運用チェック（VM）
pattern book が効いているかは VM の実データで確認する。

deep 側の確認:
```bash
scripts/vm.sh -p quantrabbit -z asia-northeast1-a -m fx-trader-vm -t sql \
  -f /home/tossaki/QuantRabbit/logs/patterns.db \
  -q "SELECT quality, COUNT(*) AS patterns, SUM(trades) AS trades_sum FROM pattern_scores WHERE strategy_tag='M1Scalper' GROUP BY quality ORDER BY trades_sum DESC;"
```

Gate が発注に反映されているか（orders.db の pattern_gate payload を点検）:
- `pattern_gate.allowed/scale/reason/pattern_id` を確認する
- “generic で no-op” の場合は payload 自体が付かない

## 8. 今回追加実装（2026-02-17）

### 8.1 戦略A: ブレイク → リテスト順張り（`breakout_retest`）

エントリー前提:
- 直近1本（`candles[-2]`）で直近レンジを明確に上抜け/下抜け
- その後の価格がバンド内に再接近したときに、順方向で再エントリー

実装キー:
- `M1Scalper._breakout_retest_signal(...)`
- `note.mode = breakout_retest`

主な設定:
- `M1SCALP_BREAKOUT_RETEST_LOOKBACK`
- `M1SCALP_BREAKOUT_RETEST_BREAKOUT_MOVE_PIPS`
- `M1SCALP_BREAKOUT_RETEST_RETEST_BAND_PIPS`
- `M1SCALP_BREAKOUT_RETEST_MOMENTUM_PIPS`
- `M1SCALP_BREAKOUT_RETEST_LIMIT_TTL_SEC`

### 8.2 戦略B: 急変動V字初期反発（`vshape_rebound`）

エントリー前提:
- 直近窓内の局所安値/高値を起点とした急落→第一反発、または急騰→第一押し
- RSIが極端過ぎない方向を優先
- `ADX` が上限超過なら抑止

実装キー:
- `M1Scalper._vshape_rebound_signal(...)`
- `note.mode = vshape_rebound`

主な設定:
- `M1SCALP_VSHAPE_REBOUND_LOOKBACK`
- `M1SCALP_VSHAPE_DROP_PIPS`
- `M1SCALP_VSHAPE_BODY_PIPS`
- `M1SCALP_VSHAPE_RETEST_PIPS`
- `M1SCALP_VSHAPE_LONG_RSI_MAX`
- `M1SCALP_VSHAPE_SHORT_RSI_MIN`
- `M1SCALP_VSHAPE_MAX_ADX`

運用時の観測ポイント:
- `orders.db` の `entry-skip` 理由に `entry_probability_below_min_units` が増えないか
- `signal` ノートに `notes.mode` が入り、`signal` で採点が継続的に上振れしているか
- 直近 `nwave`、`fallback` の比率が極端に下がらないこと
