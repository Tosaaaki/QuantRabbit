# Pattern Book / Pattern Gate（戦略opt-in）

このドキュメントは「トレード履歴 + ローソク/テクニカル由来の特徴」から **型（pattern）** を蓄積し、エントリー時に **block / scale（縮小・拡大）** を行う仕組みを、運用で迷わない粒度まで落としたメモです。

対象はまず `scalp_ping_5s`（`strategy_tag=scalp_ping_5s_live`）を「型の作り方の基準ケース」として扱います。

## 非交渉ルール（重要）

- Pattern Gate は **戦略ごとの opt-in** を前提にする（全戦略一律適用は危険）。
- 本番の真実は VM の `/home/tossaki/QuantRabbit/logs/*` と OANDA。ローカルの `logs/*.db` だけで断定しない。

## 全体フロー

```
workers/* (entry_thesis を作る)
  -> execution/order_manager.py preflight
    -> workers/common/pattern_gate.py decide()
      -> block / scale / no-op
        -> OANDA order

VM logs/trades.db
  -> scripts/pattern_book_worker.py (systemd timer)
    -> logs/patterns.db (features + stats + deep scores)
    -> config/pattern_book.json / config/pattern_book_deep.json
      -> pattern_gate が参照（DB優先、無ければJSON fallback）
```

## 用語

- `entry_thesis`: 各戦略ワーカーがエントリー時に作る JSON（`trades.db` の `entry_thesis` に保存され、後段分析の一次データになる）。
- `pattern_id`: Pattern Book/Gate が使う **型の主キー**。`analysis/pattern_book.py:build_pattern_id()` で生成される `"k:v|k:v|..."` 形式の文字列。
- `pattern_tag` / `pattern_meta`: ローソク/テクニカル由来のサブ特徴（文字列/メタJSON）。`pattern_id` の `pt:`（pattern_tag）に入る想定。戦略ごとに設計して良い。
- `quality`: deep分析（`analysis/pattern_deep.py`）で付与される定性的なラベル。代表: `learn_only` / `weak` / `neutral` / `candidate` / `robust` / `avoid`
- `suggested_multiplier`: deep分析が出す推奨倍率（`units` に掛ける）。Gate はこれを `SCALE_MIN..SCALE_MAX` へクランプして使う。

## Pattern ID 仕様（共通）

生成元: `analysis/pattern_book.py:build_pattern_id()`

`pattern_id` は下記トークンで構成されます。

```
st:<strategy> | pk:<pocket> | sd:<side> | sg:<signal_mode> | mtf:<mtf_gate> |
hz:<horizon_gate> | ex:<extrema_reason> | rg:<range_bucket> | pt:<pattern_tag>
```

`pattern_id` は **トークンが増えるほど型が増え、1型あたりのサンプルが薄くなる** ので、戦略に合わせて `entry_thesis` 側で粒度を調整します。

`rg:`（range_bucket）は下記のどちらかで決まります。

1. `entry_thesis.entry_range_bucket` があればそれを優先（推奨: `bot/low/mid/high/top` の5段階）
2. 無ければ `entry_thesis.section_axis{high,low,...}` と `entry_thesis.entry_ref`（または `entry_price/ideal_entry/entry_mean`）から自動算出

## Pattern Gate（共通）

実体: `workers/common/pattern_gate.py`  
接続: `execution/order_manager.py` の preflight（`market_order` の直前）

### Gate が「何もしない」条件

- `ORDER_PATTERN_GATE_ENABLED=0`
- `pocket` が allowlist に居ない（`ORDER_PATTERN_GATE_POCKET_ALLOWLIST`）
- `entry_thesis` が無い / opt-in が無い（後述）
- `units=0` / `side` 解釈不能
- `pattern_id` が generic 判定（`sg/mtf/hz/ex/rg/pt` が全部 `na/unknown` 相当）
- `patterns.db`（または deep JSON）に `pattern_id` が存在しない

### opt-in 仕様（重要）

Gate は次のどれかで opt-in とみなします。

1. `ORDER_PATTERN_GATE_GLOBAL_OPT_IN=1`（原則OFF）
2. `entry_thesis.pattern_gate_opt_in=true`（推奨）
3. `entry_thesis.use_pattern_gate=true` / `entry_thesis.pattern_gate_enabled=true`
4. `meta` 側の同名キー（main側から渡す場合）

### block / scale の挙動

優先順位は以下です。

1. block（`quality=avoid` 等 + サンプル十分 + スコア条件）
2. scale（`suggested_multiplier` と drift penalty で倍率を決める）
3. pass（scale=1.0）

block 条件（デフォルト）はコード上こうなっています（値は env で変更可）。

- `quality in ORDER_PATTERN_GATE_BLOCK_QUALITIES`（既定 `avoid`）
- `trades >= ORDER_PATTERN_GATE_BLOCK_MIN_TRADES`（既定 `90`）
- `robust_score <= ORDER_PATTERN_GATE_BLOCK_MAX_SCORE`（既定 `-0.9`）
- `p_value <= ORDER_PATTERN_GATE_BLOCK_MAX_PVALUE`（既定 `0.35`）

scale は `suggested_multiplier` をベースに、下記を順に適用します。

- `trades < ORDER_PATTERN_GATE_SCALE_MIN_TRADES`（既定 `30`）なら強制で 1.0（= no-op）
- `quality in ORDER_PATTERN_GATE_REDUCE_QUALITIES`（既定 `weak`）なのに multiplier>=1.0 の場合は fallback で縮小寄りに補正
- drift が `deterioration/soft_deterioration` の場合は上限をさらに下げる（`ORDER_PATTERN_GATE_DRIFT_*`）
- 最後に `ORDER_PATTERN_GATE_SCALE_MIN..MAX`（既定 `0.70..1.20`）へクランプ

`execution/order_manager.py` 側の最終挙動:

- `allowed=false`: `orders.db status="pattern_block"` で記録し、エントリーを拒否する。
- `scale != 1.0`: `units = round(abs(units) * scale)` を計算し、pocket最小 `min_units_for_pocket()` 未満なら `pattern_scale_below_min` として拒否する。そうでなければ `units` をスケールして発注する。

## Pattern Book（共通）

更新ジョブ: `scripts/pattern_book_worker.py`  
systemd: `systemd/quant-pattern-book.service` + `systemd/quant-pattern-book.timer`（5分周期）

### 出力（VM）

- DB: `/home/tossaki/QuantRabbit/logs/patterns.db`
- JSON（集計）: `/home/tossaki/QuantRabbit/config/pattern_book.json`
- JSON（deep）: `/home/tossaki/QuantRabbit/config/pattern_book_deep.json`

### patterns.db のテーブル

最低限これだけ押さえれば運用できます。

- `pattern_trade_features`: trades.db から増分取り込みした「トレード1件=1行」の特徴ログ（`pattern_id` を含む）。
- `pattern_stats` / `pattern_actions`: `pattern_id` ごとの簡易集計と、簡易ルールによる `boost/reduce/block/neutral/learn_only`。
- `pattern_scores` / `pattern_drift` / `pattern_clusters` / `pattern_cluster_summary`: deep分析（`analysis/pattern_deep.py`）の出力。Gate は原則ここを読む（無ければ `pattern_actions` にフォールバック）。

### deep分析の前提

`analysis/pattern_deep.py` は `numpy/pandas/scipy/sklearn` が必要です。VM の `quant-pattern-book.service` は venv Python を使う前提なので、依存が入っていないと deep が空になります。

### 重要: pattern_id 仕様変更時の「再計算」

`scripts/pattern_book_worker.py` は **増分取り込み** です。`pattern_trade_features.transaction_id` の最大値より大きい trade だけを取り込みます。

そのため、次のような変更を入れると `patterns.db` 内に **旧pattern_idと新pattern_idが混在** します。

- `analysis/pattern_book.py:build_pattern_id()` の変更
- 戦略ワーカー側で `entry_thesis` に入れる key/value を変える（`rg:`/`pt:` を追加する等）

「過去分も含めて同一ルールで再集計したい」場合は、VM上で `patterns.db` を作り直す運用が必要です（`trades.db` が元データ）。

## 5秒スキャ（scalp_ping_5s）の「型」設計

### いま実装されている opt-in

- `workers/scalp_ping_5s/config.py`: `SCALP_PING_5S_PATTERN_GATE_OPT_IN`（既定 true）
- `workers/scalp_ping_5s/worker.py`: `entry_thesis["pattern_gate_opt_in"] = bool(config.PATTERN_GATE_OPT_IN)` を必ず付与

つまり **`scalp_ping_5s_live` は gate の対象になり得る** 状態です（ただし `patterns.db` に該当 `pattern_id` が無い/サンプル不足なら no-op）。

### scalp_ping_5s の pattern_id に入る情報

`workers/scalp_ping_5s/worker.py` の `entry_thesis` は以下を持ち、`pattern_id` に反映されます。

- `st:` = `entry_thesis.strategy_tag`（例: `scalp_ping_5s_live`）
- `pk:` = pocket（例: `scalp_fast`）
- `sd:` = units の符号から `long/short`
- `sg:` = `entry_thesis.signal_mode`（`momentum` / `revert`）
- `mtf:` = `entry_thesis.mtf_regime_gate`（MTFレジームゲートの結果）
- `hz:` = `entry_thesis.horizon_gate`（Horizon bias ゲートの結果）
- `ex:` = `entry_thesis.extrema_gate_reason`（extrema ゲートの結果）

現状 `scalp_ping_5s` は `pattern_tag`（`pt:`）と `range_bucket`（`rg:`）を明示していないため、両方とも `na` になりやすいです。

### 「完璧にする」ときの指針（scalp_ping_5s）

目的は2つです。

1. `pattern_id` が「この戦略のセットアップ差」を表現できる
2. 型が増えすぎてサンプルが薄くならない（= gate が一生 learn_only にならない）

`scalp_ping_5s` は tick主導なので、いきなり多次元（momentum/spread/tick_rate/imbalance…）を全部 `pattern_id` に入れると型が爆発します。まずは以下の順で足すのが安全です。

1. `rg:`（range_bucket）を入れる。
`entry_thesis.entry_range_bucket` を `bot/low/mid/high/top` の5段階で付与する。`extrema_m1_pos` 等（0..1）から bucket 化すると実装が簡単で、かつ chase_risk 判定が活きる。
2. `pt:`（pattern_tag）を入れる。
M1のローソク形状 + トレンド/RSI/ATR の荒いbucket（既に他戦略が `pattern_tag` を持っているので形式を揃える）を付与する。raw値（小数）を埋め込まず、必ず bucket/string に落とす。

`pattern_tag` を入れる場合の最小要件:

- 文字列は `analysis/pattern_book.py:_norm_token()` で正規化される（英数+`_`）前提で設計する
- 1トレードごとに値がランダムに変わらない（時刻/乱数/小数第n位のブレを入れない）

## VMで「型の蓄積量」を確認する（運用）

以下は *VMの patterns.db / JSON を直接見る* 前提の確認コマンド例です。

### pattern_book の最新状態（JSON）

```bash
scripts/vm.sh -p quantrabbit -z asia-northeast1-a -m fx-trader-vm -t exec -- "python3 - <<'PY'
import json
from pathlib import Path
b=Path('/home/tossaki/QuantRabbit/config/pattern_book.json')
d=Path('/home/tossaki/QuantRabbit/config/pattern_book_deep.json')
print('book_exists', b.exists())
if b.exists():
  x=json.loads(b.read_text())
  print('book_as_of', x.get('as_of'))
  print('patterns_total', x.get('patterns_total'))
  print('action_counts', x.get('action_counts'))
  print('processed_new_rows', x.get('processed_new_rows'))
print('deep_exists', d.exists())
if d.exists():
  y=json.loads(d.read_text())
  print('deep_as_of', y.get('as_of'))
  print('deep_rows_total', y.get('rows_total'))
  print('deep_patterns_scored', y.get('patterns_scored'))
  print('deep_quality_counts', y.get('quality_counts'))
  print('deep_drift_alerts_len', len(y.get('drift_alerts') or []))
PY"
```

### scalp_ping_5s_live の quality 分布（patterns.db）

```bash
scripts/vm.sh -p quantrabbit -z asia-northeast1-a -m fx-trader-vm -t sql \
  -f /home/tossaki/QuantRabbit/logs/patterns.db \
  -q "SELECT strategy_tag, quality, COUNT(*) FROM pattern_scores WHERE strategy_tag='scalp_ping_5s_live' GROUP BY strategy_tag, quality ORDER BY COUNT(*) DESC;"
```

### 戦略別の「サンプル厚み」ざっくり（patterns.db）

```bash
scripts/vm.sh -p quantrabbit -z asia-northeast1-a -m fx-trader-vm -t sql \
  -f /home/tossaki/QuantRabbit/logs/patterns.db \
  -q "SELECT strategy_tag, COUNT(*) AS patterns, SUM(trades) AS trades_sum, SUM(CASE WHEN trades>=30 THEN 1 ELSE 0 END) AS ge30, SUM(CASE WHEN trades>=90 THEN 1 ELSE 0 END) AS ge90 FROM pattern_scores GROUP BY strategy_tag ORDER BY trades_sum DESC LIMIT 15;"
```

### scalp_ping_5s_live の bucket（どれだけ「使える型」が増えたか）

```bash
scripts/vm.sh -p quantrabbit -z asia-northeast1-a -m fx-trader-vm -t sql \
  -f /home/tossaki/QuantRabbit/logs/patterns.db \
  -q "SELECT CASE WHEN trades>=120 THEN 'ge120' WHEN trades>=90 THEN 'ge90' WHEN trades>=60 THEN 'ge60' WHEN trades>=30 THEN 'ge30' WHEN trades>=15 THEN 'ge15' ELSE 'lt15' END AS bucket, COUNT(*) FROM pattern_scores WHERE strategy_tag='scalp_ping_5s_live' GROUP BY bucket ORDER BY CASE bucket WHEN 'ge120' THEN 1 WHEN 'ge90' THEN 2 WHEN 'ge60' THEN 3 WHEN 'ge30' THEN 4 WHEN 'ge15' THEN 5 ELSE 6 END;"
```

## 参照ファイル（コード）

- `analysis/pattern_book.py`（pattern_id と action の基本）
- `analysis/pattern_deep.py`（quality / suggested_multiplier / drift / cluster）
- `scripts/pattern_book_worker.py`（patterns.db 更新・JSON出力）
- `systemd/quant-pattern-book.service`
- `systemd/quant-pattern-book.timer`
- `workers/common/pattern_gate.py`（block/scale/no-op の判定）
- `execution/order_manager.py`（preflight 接続と orders.db 記録）
- `workers/scalp_ping_5s/config.py`
- `workers/scalp_ping_5s/worker.py`
