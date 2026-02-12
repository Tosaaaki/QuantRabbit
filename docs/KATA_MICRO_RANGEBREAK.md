# 型（Kata）設計: `MicroRangeBreak`（micro_multi）

このドキュメントは、`workers/micro_multistrat` 内の `MicroRangeBreak` の「型（Kata）」を **再現可能に作る**ための設計書です。
型は、過去トレードから抽出したパターンを DB/JSON に継続学習し、次のエントリーで `block / scale`（拒否/縮小/増強）に接続します。

## ゴール
- micro ポケットの `MicroRangeBreak` を「型」で可視化し、損な型は縮小/拒否、良い型は必要なら増強できる状態にする。
- ただし、型は戦略ごとに作る（全戦略一律の強制ゲートはしない）。
- 互換性を壊す変更（`pattern_id` の分割軸変更）は、移行計画込みで扱う。

## 1. 稼働形態（VM）
`MicroRangeBreak` は単独 unit ではなく、micro の複合ワーカー内で稼働します。

systemd（VM）:
- `quant-micro-multi.service`（`ExecStart: -m workers.micro_multistrat.worker`）

allowlist（VM）例:
- `MICRO_STRATEGY_ALLOWLIST=MicroLevelReactor,MicroRangeBreak,MicroVWAPRevert,MomentumPulse`

## 2. 何を「型」と呼ぶか（pattern_id）
型の最小単位は `pattern_id`（文字列）です。トレードは必ずどれか 1 つの `pattern_id` に割り当てられます。

`pattern_id` は `analysis/pattern_book.py:build_pattern_id()` で生成され、次のトークンを `|` 区切りで連結します。

- `st:` strategy tag（戦略名）
- `pk:` pocket（資金/口座内の区分）
- `sd:` direction（long/short）
- `sg:` signal mode（`signal_mode/entry_mode/mode`）
- `mtf:` MTF gate（`mtf_regime_gate/mtf_gate`）
- `hz:` horizon gate（`horizon_gate`）
- `ex:` extrema gate reason（`extrema_gate_reason`）
- `rg:` range bucket（`entry_range_bucket` または `section_axis + entry_ref`）
- `pt:` pattern tag（`pattern_tag`）

## 3. `MicroRangeBreak` の entry_thesis と分割軸
起点:
- `workers/micro_multistrat/worker.py`（`entry_thesis = {...}`）

現状の `MicroRangeBreak` は、典型的には次の傾向です（= `pattern_id` 側の出方）:
- `signal_mode/entry_mode` を入れていないため `sg:na`
- `section_axis/entry_ref` を入れていないため `rg:na`
- `pattern_tag`（ローソク+簡易テックのシグネチャ）を入れているため `pt:*` が立ちやすい
  - `pt:` は `analysis.pattern_stats:derive_pattern_signature()` 由来（長い場合は `build_pattern_id()` 側で短縮/ハッシュ化される）

重要:
- `Pattern Gate` は “情報量ゼロの pattern_id（sg/mtf/hz/ex/rg/pt が全部 `na`）” を generic とみなし、デフォルトでは no-op します。
- `MicroRangeBreak` は `pt:` が付くなら generic ではありませんが、`pt:na` になる局面があるなら `pattern_gate_allow_generic` の opt-in を検討します。

## 4. 現状のサンプル厚み（VM実データ）
スナップショット日: 2026-02-12

deep（`logs/patterns.db:pattern_scores`）の品質分布:
- `robust`: 1 pattern / trades_sum=86
- `candidate`: 1 pattern / trades_sum=63
- `learn_only`: 8 patterns / trades_sum=23

確認コマンド（VM）:
```bash
scripts/vm.sh -p quantrabbit -z asia-northeast1-a -m fx-trader-vm -t sql \
  -f /home/tossaki/QuantRabbit/logs/patterns.db \
  -q "SELECT quality, COUNT(*) AS patterns, SUM(trades) AS trades_sum FROM pattern_scores WHERE strategy_tag='MicroRangeBreak' GROUP BY quality ORDER BY trades_sum DESC;"
```

注意（読み方）:
- `pt:na` の direction-only パターンが厚くても、直近トレードが `pt:*` に寄っていると型が分裂して `learn_only` が増えやすい。
- まずは gate opt-in を入れて「どの pattern_id が実際に発注で使われているか（orders.db の entry_thesis）」を観測し、必要なら `pt:` の粒度を落とす。

## 5. エントリー制御（Pattern Gate）の接続（戦略 opt-in）
実装:
- `workers/micro_multistrat/worker.py`

opt-in 仕様（`MicroRangeBreak` のみ）:
- `entry_thesis["pattern_gate_opt_in"]=...` を付与（env: `MICRO_RANGEBREAK_PATTERN_GATE_OPT_IN`, default: true）
- `pt:na` などで generic になりうる場合に備え、必要なら `entry_thesis["pattern_gate_allow_generic"]=true` を付与（env: `MICRO_RANGEBREAK_PATTERN_GATE_ALLOW_GENERIC`, default: true）

重要:
- opt-in は `MicroRangeBreak` のみに付与し、micro_multi 内の他戦略へは波及させない（共通一律適用を避ける）。

## 6. 運用チェック（VM）
Gate が発注に反映されているか（orders.db）:
```bash
scripts/vm.sh -p quantrabbit -z asia-northeast1-a -m fx-trader-vm -t sql \
  -f /home/tossaki/QuantRabbit/logs/orders.db \
  -q "SELECT ts, status, client_order_id FROM orders WHERE status IN ('pattern_block','pattern_scale_below_min') ORDER BY ts DESC LIMIT 30;"
```

deep 側（patterns.db）の確認:
```bash
scripts/vm.sh -p quantrabbit -z asia-northeast1-a -m fx-trader-vm -t sql \
  -f /home/tossaki/QuantRabbit/logs/patterns.db \
  -q "SELECT pattern_id, trades, quality, ROUND(suggested_multiplier,3) FROM pattern_scores WHERE strategy_tag='MicroRangeBreak' ORDER BY trades DESC LIMIT 30;"
```

即時ロールバック:
- `MICRO_RANGEBREAK_PATTERN_GATE_OPT_IN=0`（MicroRangeBreak だけ停止）
- もしくは `ORDER_PATTERN_GATE_ENABLED=0`（Pattern Gate 全停止）

## 7. 次の改善候補（設計ルール）
- `learn_only` が長期化する場合は、`pt:` の粒度を落として「型の分裂」を抑える（例: `rsi/vol/atr` を外してローソク+方向だけにする、など）。
- `sg:` を 3-6 程度の低カーディナリティで導入し、局面別（勢い/押し目）に最低限分ける。
- 互換性を壊す変更（`pattern_id` 分割軸の追加/変更）は `docs/KATA_PROGRESS.md` に移行計画と learn_only 増加期間を記録する。

