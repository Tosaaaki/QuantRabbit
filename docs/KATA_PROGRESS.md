# 型（Kata）進捗ログ

このドキュメントは「型（Kata）= 戦略ごとの `pattern_id` 設計 + Pattern Book/Pattern Gate への接続」の進捗を記録します。
結論や判断は、必ず VM 実データ（`logs/*.db` / systemd / OANDA）で裏取りした上で追記します。

## 原則（崩さない）
- 型は戦略ごとに作る（共通層の一律強制はしない）。
- Pattern Gate は戦略 opt-in のみ（`ORDER_PATTERN_GATE_GLOBAL_OPT_IN=0` を維持）。
- `pattern_id` の構成を変える変更は互換性を壊す（変更する場合は「移行計画」「効果が落ちる期間（learn_only増）」を明記し、必要に応じて `logs/patterns.db` の rebuild/backfill を検討する）。
- 運用判断はローカルのスナップショットで断定しない（VM を正とする）。

## 現状（VMスナップショット）
スナップショット日: 2026-02-12

### 稼働中の戦略（systemd）
VM で確認した running service（再現コマンドは下）:
- `quant-impulse-retest-s5.service`
- `quant-m1scalper.service`
- `quant-micro-multi.service`
- `quant-scalp-ping-5s.service`
- `quant-scalp-precision-squeeze-pulse-break.service`
- `quant-scalp-precision-tick-imbalance.service`
- `quant-scalp-precision-wick-reversal-blend.service`
- `quant-scalp-precision-wick-reversal-pro.service`
- `quant-session-open.service`
- `quant-trend-reclaim-long.service`
- `quantrabbit.service`

再現コマンド:
```bash
scripts/vm.sh -p quantrabbit -z asia-northeast1-a -m fx-trader-vm -t exec -- \
  "systemctl list-units --type=service --state=running --no-pager"
```

### Pattern Gate opt-in 状況
コード上の opt-in 実装（main）:
- `scalp_ping_5s`: `workers/scalp_ping_5s/worker.py`
- `scalp_m1scalper`: `workers/scalp_m1scalper/worker.py`
- `TickImbalance`（scalp_precision）: `workers/scalp_precision/worker.py`（必要なら `pattern_gate_allow_generic` も付与）
- `MicroRangeBreak`（micro_multi）: `workers/micro_multistrat/worker.py`（必要なら `pattern_gate_allow_generic` も付与）

確認コマンド（リポジトリ）:
```bash
rg -n "pattern_gate_opt_in|use_pattern_gate|pattern_gate_enabled" workers
```

### パターン蓄積（deep: `pattern_scores`）
deep のサンプル厚みが大きい順（例）:
- `M1Scalper`: trades_sum=3435
- `TickImbalance`: trades_sum=2430
- `scalp_ping_5s_live`: trades_sum=1576（patterns=79, ge30=14, ge90=4）

補足:
- Pattern Gate は “情報量ゼロの pattern_id（sg/mtf/hz/ex/rg/pt が全部 `na`）” を generic とみなし **デフォルト no-op** にする。
- ただし opt-in 戦略のみ `pattern_gate_allow_generic=true` を付けることで、generic pattern_id でも gate を評価できる（粗い型で block/scale したい戦略向け）。
- `TickImbalance` は direction-only の deep サンプルが厚いため、`pattern_gate_allow_generic` を使う方針。

確認コマンド（VM）:
```bash
scripts/vm.sh -p quantrabbit -z asia-northeast1-a -m fx-trader-vm -t sql \
  -f /home/tossaki/QuantRabbit/logs/patterns.db \
  -q "SELECT strategy_tag, COUNT(*) AS patterns, SUM(trades) AS trades_sum, SUM(CASE WHEN trades>=30 THEN 1 ELSE 0 END) AS ge30, SUM(CASE WHEN trades>=90 THEN 1 ELSE 0 END) AS ge90 FROM pattern_scores GROUP BY strategy_tag ORDER BY trades_sum DESC LIMIT 25;"
```

### `scalp_ping_5s_live` の `rg:`（range bucket）分布
`rg:na` は約 15%（235/1561）だが、深掘り上は少数パターンに集中している。

確認コマンド（VM）:
```bash
scripts/vm.sh -p quantrabbit -z asia-northeast1-a -m fx-trader-vm -t sql \
  -f /home/tossaki/QuantRabbit/logs/patterns.db \
  -q "SELECT CASE WHEN pattern_id LIKE '%|rg:bot|%' THEN 'bot' WHEN pattern_id LIKE '%|rg:low|%' THEN 'low' WHEN pattern_id LIKE '%|rg:mid|%' THEN 'mid' WHEN pattern_id LIKE '%|rg:high|%' THEN 'high' WHEN pattern_id LIKE '%|rg:top|%' THEN 'top' ELSE 'na' END AS rg, COUNT(*) AS trades, ROUND(AVG(pl_pips),4) AS avg_pips, ROUND(AVG(CASE WHEN pl_pips>0 THEN 1.0 ELSE 0.0 END),3) AS win_rate FROM pattern_trade_features WHERE strategy_tag='scalp_ping_5s_live' GROUP BY rg ORDER BY trades DESC;"
```

補足（原因の切り分け）:
- `rg:na` のサンプルを `trades.db` で確認すると、`section_axis` は付いているが `entry_ref`（および `entry_price/ideal_entry/entry_mean`）が `entry_thesis` に入っていない古いトレードが混ざっている。
- その場合 `analysis/pattern_book.py:_range_bucket()` は `entry_ref` を取れず `na` になる（`section_axis` 欠損が主因ではない）。
- 現行の `workers/scalp_ping_5s/worker.py` は `entry_thesis["entry_ref"]` を入れているため、新規トレードは自然に改善する見込み。

確認コマンド（VM、`rg:na` の entry_thesis を点検）:
```bash
scripts/vm.sh -p quantrabbit -z asia-northeast1-a -m fx-trader-vm -t exec -- "
cd /home/tossaki/QuantRabbit && python3 - <<'PY'
import json, sqlite3

conp = sqlite3.connect('logs/patterns.db')
conp.row_factory = sqlite3.Row
txs = [r['transaction_id'] for r in conp.execute(
  "SELECT transaction_id FROM pattern_trade_features WHERE strategy_tag='scalp_ping_5s_live' AND pattern_id LIKE '%|rg:na|%' ORDER BY close_time DESC LIMIT 20"
).fetchall()]

cont = sqlite3.connect('logs/trades.db')
cont.row_factory = sqlite3.Row
q = 'SELECT transaction_id, close_time, entry_thesis FROM trades WHERE transaction_id IN ({})'.format(','.join(['?']*len(txs)))
for r in cont.execute(q, txs).fetchall():
  th = json.loads(r['entry_thesis']) if r['entry_thesis'] else {}
  axis = th.get('section_axis')
  print(r['transaction_id'], r['close_time'], 'has_axis', isinstance(axis, dict), 'entry_ref', th.get('entry_ref'))
PY"
```

## 次のアクション（優先順）
### P0: `scalp_ping_5s` を「壊さず強くする」（おすすめ方針）
- いま `rg:na` にも deep 上の `avoid` が存在し、Pattern Gate がブロックに使えている。
- ここで `pattern_id` の構成を大きく変えると、既存の `avoid` ブロックが一時的に効かなくなるリスクがある。

やること:
1. `rg:na` が出る原因を VM 実データで切り分け（`entry_ref` 欠損 / `section_axis` 欠損 / データ不整合）。
2. `avoid/weak` を「回避・縮小」する用途を優先して継続運用（ブーストは慎重）。
3. `entry_range_bucket` の導入は “移行計画あり” で後段に回す（必要なら「旧 `pattern_id` 参照のフォールバック」など、ブロック消失を避ける手当を先に入れる）。

### P1: 他戦略への kata 展開（稼働中 + サンプル厚い順）
候補（deep の trades_sum が大きい順、かつ稼働中ユニット優先）:
1. `MicroRangeBreak`（micro_multi）
2. `TickImbalance`（scalp_precision）
3. `M1Scalper`
4. `scalp_ping_5s_live`（継続で厚くする）

各戦略でやること:
- `entry_thesis` の設計（低カーディナリティで分割軸を決める）
- Pattern Gate opt-in の追加（戦略ごと）
- `pattern_scores` の quality 分布を見てブロック/スケールの効き方を確認

## 進捗ログ
### 2026-02-12
- `docs/KATA_SCALP_PING_5S.md` が main に存在（commit: `7e111d7a`）。
- VM `pattern_scores` で `scalp_ping_5s_live` は patterns=73 / trades_sum=1562 / ge30=14 / ge90=4 を確認。
- `rg:na` が 235 trades（約15%）あるが、deep 上は少数パターンに集中していることを確認。
- `rg:na` の原因は「`section_axis` 欠損」ではなく「古いトレードの `entry_thesis` に `entry_ref` が入っていない」ケースがあることを確認（`trades.db` サンプル点検）。
- 方針決定: `pattern_id` を急に変えず、まずは `avoid/weak` の回避・縮小に効かせる（P0）。

追記（同日、main/VM ロールアウト）:
- `scalp_m1scalper` opt-in + docs 追加: `docs/KATA_SCALP_M1SCALPER.md`（commit: `847d6463`） / `workers/scalp_m1scalper/worker.py`（commit: `39c611ae`） / env例（commit: `959628b8`）
- generic pattern_id の opt-in ゲート許可: `workers/common/pattern_gate.py`（commit: `ab61d00f`）
- `TickImbalance` opt-in（generic許可）: `workers/scalp_precision/worker.py`（commit: `948752cb`）
- `MicroRangeBreak` opt-in（generic許可）+ docs 追加: `workers/micro_multistrat/worker.py`（commit: `61b0a19d`） / `docs/KATA_MICRO_RANGEBREAK.md`（commit: `43ac2f86`） / `AGENTS.md` 動線（commit: `3f605189`） / env例（commit: `710fcab1`）
- VM状況（2026-02-12）: TickImbalance は `reentry_block` が強く Pattern Gate まで届いていない（`logs/orders.db`）。`scalp_ping_5s` は pattern_id が細かく分岐するため、deep に存在しない新pattern_idでは gate が no-op になる（`logs/patterns.db` 側に row が出るまで None）。
- VM 反映確認: `/home/tossaki/QuantRabbit` の `HEAD == origin/main == fd4a185c`。
- `quantrabbit.service` 再起動: 2026-02-12T13:46:06Z（`Application started!`: 2026-02-12T13:46:14Z）。
- `quant-micro-multi.service` 再起動: 2026-02-12T13:48:06Z。
