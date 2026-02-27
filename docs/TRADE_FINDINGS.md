# Trade Findings Ledger (Single Source of Truth)

このファイルは、QuantRabbit の「改善記録」と「敗因記録」の単一台帳です。
以後、同種の記録は必ずここに追記し、他の分散ファイルは作らないこと。

## Rules (Read First)
- 記録先はこのファイルのみ（`docs/TRADE_FINDINGS.md` 固定）。
- 新規の改善/敗因分析を行ったら、必ず 1 エントリ以上追記する。
- 追記順は「新しいものを上」に統一する。
- 事実は VM/OANDA 実測を優先し、日時は UTC と JST を明記する。
- 推測は `Hypothesis` と明示し、事実 (`Fact`) と混在させない。
- 最低限の記載項目:
  - `Period`（集計期間）
  - `Fact`（数値）
  - `Failure Cause`（敗因）
  - `Improvement`（改善施策）
  - `Verification`（確認方法/判定基準）
  - `Status`（open/in_progress/done）

## Entry Template
```
## YYYY-MM-DD HH:MM UTC / YYYY-MM-DD HH:MM JST - <short title>
Period:
- ...

Fact:
- ...

Failure Cause:
1. ...
2. ...

Improvement:
1. ...
2. ...

Verification:
1. ...
2. ...

Status:
- open | in_progress | done
```

## 2026-02-27 15:38 UTC / 2026-02-28 00:38 JST - `scalp_ping_5s_c_live` の `entry_probability_reject` 閾値を再緩和
Period:
- 観測: 2026-02-27 15:34:41-15:37:20 UTC（long leading reject 無効化後）

Fact:
- `REJECT_BELOW=0.00` 反映後、long 側 `entry_leading_profile_reject` は減少した一方、
  `entry_probability_reject` が主因化。
- Cログは `prob=0.81〜0.89` のシグナルでも
  `err=entry_probability_reject_threshold` を連発。
- 同期間の C は `orders.db` 上で送出が細く、露出回復が不足。
- `quant-order-manager` 実効envは
  `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_C[_LIVE]=0.72`。

Failure Cause:
1. leading reject を外した後も preserve-intent 側の確率閾値 `0.72` が高く、
   C の実効確率帯（0.8前後）で閾値下振れが起きやすい。
2. C worker と order-manager の reject_under が同水準で、同時に拒否寄りへ働いた。

Improvement:
1. `ops/env/scalp_ping_5s_c.env` の
   `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_C_LIVE`
   を `0.72 -> 0.58`。
2. `ops/env/quant-order-manager.env` の
   `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_C[_LIVE]`
   を `0.72 -> 0.58` へ同期。

Verification:
1. 再デプロイ後 15 分で C の
   `market_order rejected ... entry_probability_reject` 件数が減少すること。
2. 同期間 `orders.db` で `scalp_ping_5s_c_live` の
   `submit_attempt`/`filled` が再開・増加すること。
3. `metrics.db` の `order_perf_block` hard reason 再発がないこと。

Status:
- in_progress

## 2026-02-27 15:29 UTC / 2026-02-28 00:29 JST - `scalp_ping_5s_c_live` long 側 leading reject を無効化（short維持）
Period:
- 観測: 2026-02-27 15:25:36-15:28:54 UTC（前回緩和反映後）

Fact:
- 前回緩和後も `journalctl` で long 側の `entry_leading_profile_reject` が継続し、
  `open mode=...` の直後に reject が多発。
- 直近ログでも `side=long` の reject が連続し、C の露出回復が不十分。
- 同時に `orders.db` では反映直後に `scalp_ping_5s_c_live` の `submit_attempt=3 / filled=3` まで再開しており、
  方向性としては「hard reject をさらに減らせば約定回復が進む」状態。

Failure Cause:
1. `REJECT_BELOW=0.56` でも long 側の `adjusted_probability` が閾値を下回る局面が多く、hard reject が継続。
2. C は long bias 運用なのに long 側も hard reject で止まり、機会損失が残った。

Improvement:
1. `ops/env/scalp_ping_5s_c.env` の
   `SCALP_PING_5S_C_ENTRY_LEADING_PROFILE_REJECT_BELOW` を `0.56 -> 0.00` へ変更し、
   long 側 hard reject を無効化。
2. `SCALP_PING_5S_C_ENTRY_LEADING_PROFILE_REJECT_BELOW_SHORT=0.80` は維持し、
   short 側は従来どおり強く抑制。
3. 既存の `PENALTY_MAX=0.14` / `UNITS_MIN_MULT=0.58` / `entry_probability` ガードで
   低品質シグナルは縮小・拒否を継続。

Verification:
1. 再デプロイ後 15 分で `journalctl` の `market_order rejected ... entry_leading_profile_reject`（long）が大幅減少すること。
2. 同期間 `orders.db` で `scalp_ping_5s_c_live` の `submit_attempt`/`filled` が前回より増加すること。
3. `metrics.db` で `order_perf_block` hard reason（failfast/sl_loss_rate）が再発しないこと。

Status:
- in_progress

## 2026-02-27 15:35 UTC / 2026-02-28 00:35 JST - 市況プレイブックを policy_overlay へ自動適用（no-delta抑止付き）

Period:
- 実装時点（単発検証）

Source:
- `scripts/gpt_ops_report.py`
- `analytics/policy_apply.py`
- `tests/scripts/test_gpt_ops_report.py`

Fact:
- 既存の `gpt_ops_report --policy` は `deterministic_playbook_only` の `no_change` を常に出力し、
  `--apply-policy` でも overlay へ実反映しない実装だった。
- これにより、プレイブックの A/B/C シナリオと `order_manager` の entry gate が接続されず、
  「分析は更新されるが執行条件は固定」の状態になっていた。

Failure Cause:
1. policy diff 生成がスタブ固定（`no_change=true`）で、方向バイアス/イベント/データ鮮度が反映されない。
2. `--apply-policy` でも `apply_policy_diff_to_paths` を呼んでいないため、導線が未接続。
3. 同値判定がないため、将来 patch 適用を追加しても毎サイクル version 増加ノイズが発生しやすい。

Improvement:
1. `gpt_ops_report` に deterministic translator を追加し、`short_term.bias`、`direction_confidence_pct`、
   scenario gap、`event_soon/event_active_window`、`factor_stale`、`reject_rate` から
   pocket別 `entry_gates.allow_new` / `bias` / `confidence` を生成。
2. `--apply-policy` で `apply_policy_diff_to_paths` を実行し、
   `policy_overlay` / `policy_latest` / `policy_history` を更新。
3. 現在 overlay と patch の deep-subset 比較で no-delta 判定を入れ、
   同値時は `no_change=true` として不要な version 連番更新を回避。

Verification:
1. `pytest -q tests/scripts/test_gpt_ops_report.py tests/scripts/test_run_market_playbook_cycle.py` -> `13 passed`
2. ローカル実行:
   - `python3 scripts/gpt_ops_report.py ... --policy --apply-policy ...`
   - `INFO [OPS_POLICY] applied=True ...` を確認。
3. no-delta再計算時に `no_change=true` になるユニットテストを追加済み。

Status:
- done

## 2026-02-27 15:50 UTC / 2026-02-28 00:50 JST - policy適用は回っていたが order_manager gate がOFFだったため本番ON

Period:
- 監査: 2026-02-27 15:42-15:49 UTC

Source:
- VM `journalctl -u quant-ops-policy.service`
- VM `/home/tossaki/QuantRabbit/ops/env/quant-v2-runtime.env`
- VM `/home/tossaki/QuantRabbit/ops/env/quant-order-manager.env`
- VM `execution/order_manager.py`（`_POLICY_GATE_ENABLED`）

Fact:
- `quant-ops-policy.service` は `applied=True` で `policy_overlay` を更新（15:42 UTC時点で確認）。
- ただし runtime/order-manager env に `ORDER_POLICY_GATE_ENABLED` が存在せず、
  order_manager の policy gate が default false のまま。
- 直近 orders サマリでも `policy_*` 系 reject は観測されず、
  preflight 適用が実運用に接続されていない状態だった。

Failure Cause:
1. プレイブック→overlay 導線の実装後、order_manager 側の有効化フラグを本番envでONにしていなかった。

Improvement:
1. `ops/env/quant-order-manager.env` に `ORDER_POLICY_GATE_ENABLED=1` を追加。
2. `quant-order-manager.service` 再起動後、process env と journal で有効化を確認する。

Verification:
1. VMで `systemctl restart quant-order-manager.service` 後、`/proc/<pid>/environ` に
   `ORDER_POLICY_GATE_ENABLED=1` が存在すること。
2. `quant-ops-policy.service` の次回 `applied=True` 更新後、order_manager が同overlayを読み込むこと。
3. 直近 orders で `policy_allow_new_false` / `policy_bias_*` が必要局面で発生することを継続監視。

Status:
- in_progress

## 2026-02-27 15:24 UTC / 2026-02-28 00:24 JST - `scalp_ping_5s_c_live` の `entry_leading_profile_reject` 過多を C専用で緩和
Period:
- 観測: 2026-02-27 15:20-15:24 UTC（再起動直後）

Fact:
- `quant-scalp-ping-5s-c.service` は `2026-02-27 15:19:59 UTC` に再起動後 active。
- 同期間ログ集計で `open=59` に対し `entry_leading_profile_reject=56`。
- 同期間 `orders.db` では `scalp_ping_5s_c_live` の `submit_attempt/filled` が 0 件。
- 一方で `metrics.db` の `order_perf_block` は 15:18 UTC 以降 0 件で、主阻害要因が `perf_block` から `entry_leading_profile_reject` へ移行。

Failure Cause:
1. C の leading profile が `REJECT_BELOW=0.64` + `PENALTY_MAX=0.20` で逆風時にゼロ化しやすく、`entry_leading_profile_reject` が多発。
2. reject 条件が先に成立して `order_manager` 送出前に止まり、約定再開に繋がらなかった。

Improvement:
1. `ops/env/scalp_ping_5s_c.env` の `SCALP_PING_5S_C_ENTRY_LEADING_PROFILE_REJECT_BELOW` を `0.64 -> 0.56` へ緩和。
2. `SCALP_PING_5S_C_ENTRY_LEADING_PROFILE_PENALTY_MAX` を `0.20 -> 0.14` へ緩和。
3. reject 回避後のリスク抑制として `SCALP_PING_5S_C_ENTRY_LEADING_PROFILE_UNITS_MIN_MULT` を `0.72 -> 0.58` に下げ、低確度帯は縮小で通す。

Verification:
1. デプロイ後 15 分で `journalctl` 集計の `entry_leading_profile_reject/open` 比率が低下すること。
2. 同期間 `orders.db` で `scalp_ping_5s_c_live` の `submit_attempt` と `filled` が再開すること。
3. `metrics.db` で `order_perf_block` の hard `failfast/sl_loss_rate` 再発がないこと。

Status:
- in_progress

## 2026-02-27 15:15 UTC / 2026-02-28 00:15 JST - `scalp_ping_5s_c_live` の hard `perf_block` 主因を failfast/sl_loss_rate と特定して緩和
Period:
- 集計: 2026-02-27 14:12-15:12 UTC（直近60分）

Fact:
- VM `logs/metrics.db` の `order_perf_block` は 134 件。
- 内訳は `scalp_ping_5s_c_live` が 119 件で、理由は
  `hard:hour14:sl_loss_rate=0.68 pf=0.39 n=41` が 94 件、
  `hard:hour15:failfast:pf=0.12 win=0.28 n=43` が 25 件。
- 同時間帯の `orders.db` では C は `filled=4 / perf_block=119` で、B は `filled=24 / perf_block=15`。

Failure Cause:
1. `PERF_GUARD_FAILFAST_PF/WIN` を下げても、`PERF_GUARD_FAILFAST_HARD_PF` の既定値で hard block が継続していた。
2. `PERF_GUARD_SL_LOSS_RATE_MAX=0.55` が C の時間帯成績（sl_loss_rate=0.68）に対して厳しすぎ、hour14 で hard block が連発した。

Improvement:
1. `ops/env/quant-order-manager.env` と `ops/env/scalp_ping_5s_c.env` に
   `SCALP_PING_5S_C_PERF_GUARD_FAILFAST_HARD_PF=0.00` を追加（fallback `SCALP_PING_5S_*` も同時設定）。
2. 同2ファイルで `SCALP_PING_5S_C_PERF_GUARD_SL_LOSS_RATE_MAX` を `0.55 -> 0.70` へ緩和
   （fallback `SCALP_PING_5S_*` も同時に `0.70` へ更新）。

Verification:
1. デプロイ後、`metrics.db` の `order_perf_block` を15分監視し、Cの `hard:hour*:failfast` / `hard:hour*:sl_loss_rate` が再発しないことを確認。
2. 同期間で `orders.db` の C `filled/perf_block` 比率が改善（`filled` 増・`perf_block` 減）することを確認。

Status:
- in_progress

## 2026-02-27 17:10 UTC / 2026-02-28 02:10 JST - Counterfactual auto-improve を noise+pattern LCB で昇格判定
Period:
- 実装/テスト: 2026-02-27（ローカル）

Source:
- `analysis/trade_counterfactual_worker.py`
- `analysis/replay_quality_gate_worker.py`
- `tests/analysis/test_trade_counterfactual_worker.py`
- `tests/analysis/test_replay_quality_gate_worker.py`

Fact:
- 既存 auto-improve は `policy_hints.block_jst_hours` のみを採用条件にしており、
  ノイズ局面で時間帯ブロックへ寄る導線だった。
- replay→counterfactual の昇格判定に spread/stuck/OOS のノイズ補正と
  pattern book 事前確率が未統合だった。

Failure Cause:
1. 採用条件が時間帯ブロック中心で、reentry 品質の調整（cooldown/reentry距離）へ接続されていなかった。
2. 候補ランクが期待値中心で、ノイズ耐性（LCB）と pattern prior が弱かった。

Improvement:
1. `trade_counterfactual_worker` に `noise_penalty`（spread/stuck/OOS）と
  `pattern_book_deep` 事前確率を統合し、`quality_score` で候補を再ランキング。
2. `policy_hints.reentry_overrides`（tighten/loosen, multiplier, confidence, lcb_uplift）を追加。
3. `replay_quality_gate_worker` は `reentry_overrides` の
  `confidence`/`lcb_uplift_pips` をゲートにして
  `worker_reentry.yaml` の `cooldown_* / same_dir_reentry_pips / return_wait_bias` を更新。
4. `block_jst_hours` の自動適用は `REPLAY_QUALITY_GATE_AUTO_IMPROVE_APPLY_BLOCK_HOURS=1`
  を明示した場合のみ許可（既定 0）。

Verification:
1. `pytest -q tests/analysis/test_trade_counterfactual_worker.py tests/analysis/test_replay_quality_gate_worker.py`
   で回帰テストが通過すること。
2. auto-improve 実行後の `logs/replay_quality_gate_latest.json.auto_improve.strategy_runs[*]` で
   `reentry_mode/confidence/lcb` と `accepted_update.reentry_overrides` が記録されること。
3. `worker_reentry.yaml` の更新が時間帯ブロックでなく
   `cooldown_* / same_dir_reentry_pips / return_wait_bias` 中心であること。

Status:
- in_progress

## 2026-02-27 14:20 UTC / 2026-02-27 23:20 JST - `scalp_extrema_reversal_live` の取り残し（SL欠損 + loss_cut未発火）対策
Period:
- 直近7日（orders/trades 集計）
- 2026-02-27 13:49 UTC 時点の open trade

Source:
- VM `/home/tossaki/QuantRabbit/logs/orders.db`
- VM `/home/tossaki/QuantRabbit/logs/trades.db`
- VM `PositionManager.get_open_positions()`
- Repo/VM `ops/env/quant-order-manager.env`, `config/strategy_exit_protections.yaml`

Fact:
- `scalp_extrema_reversal_live` の直近7日 `filled=47` のうち、
  `stopLossOnFill` 付きは `4`（約 `8.5%`）。
- 同時点 open trade は `3件` すべて `scalp_extrema_reversal_live` で、
  `stop_loss=null`（TPのみ）。
- `trade_id=408076` は `hold≈526min / -49.5pips` まで逆行保持。

Failure Cause:
1. `quant-order-manager` 実効envで `scalp_extrema_reversal_live` の
   `ORDER_ALLOW_STOP_LOSS_ON_FILL_STRATEGY_*` が未設定だった。
2. `strategy_exit_protections` に `scalp_extrema_reversal_live` の個別 `exit_profile` がなく、
   defaults（`loss_cut_enabled=false`, `loss_cut_require_sl=true`）へフォールバックしていた。
3. その結果「SLなしで建つと、負け側を deterministic に閉じる条件が弱い」状態が発生した。

Improvement:
1. `ops/env/quant-order-manager.env`
  - `ORDER_ALLOW_STOP_LOSS_ON_FILL_STRATEGY_SCALP_EXTREMA_REVERSAL_LIVE=1`
  - `ORDER_ALLOW_STOP_LOSS_ON_FILL_STRATEGY_SCALP_EXTREMA_REVERSAL=1`
2. `config/strategy_exit_protections.yaml`
  - `scalp_extrema_reversal_live.exit_profile` を追加:
    `loss_cut_enabled=true`, `loss_cut_require_sl=false`,
    `loss_cut_hard_pips=7.0`, `loss_cut_reason_hard=m1_structure_break`,
    `loss_cut_max_hold_sec=900`, `loss_cut_cooldown_sec=4`

Verification:
1. 反映後 24h で `scalp_extrema_reversal_live` の `filled` について
   `stopLossOnFill` 付与率が改善していること（目標: `>=0.90`）。
2. open trade 監査で `scalp_extrema_reversal_live` の `stop_loss=null` が常態化しないこと。
3. 逆行保持（例: `-20pips` 超かつ `hold>20min`）の滞留件数が低下すること。

Status:
- in_progress

## 2026-02-27 15:05 UTC / 2026-02-28 00:05 JST - 市況プレイブックの stale factor 誤判定を是正（外部価格フォールバック）

Period:
- ローカル再現（`run_market_playbook_cycle.py --force`）

Source:
- `tmp/gpt_ops_report_user_now.json`
- `scripts/gpt_ops_report.py`
- `tests/scripts/test_gpt_ops_report.py`

Fact:
- `factor_cache` M1 の時刻が `2025-10-29T23:58:00+00:00` の stale 条件でも、
  従来の `gpt_ops_report` は `snapshot.current_price` と方向スコアに stale 値を直接利用していた。
- 同時に `market_context.pairs.usd_jpy` は外部取得値（当日値）で更新されるため、
  `snapshot` と `market_context` の価格整合が崩れるケースが発生していた。

Failure Cause:
1. `gpt_ops_report` に factor 鮮度判定が無く、`M1 close` を無条件採用していた。
2. stale 時の信頼度減衰ルールがなく、シナリオ確率が過信方向に寄る余地があった。

Improvement:
1. `OPS_PLAYBOOK_FACTOR_MAX_AGE_SEC`（default 900s）で M1 factor 鮮度を判定。
2. stale 時は `snapshot.current_price` を外部 `USD/JPY` にフォールバック。
3. `snapshot.factor_stale/factor_age_m1_sec/current_price_source` を追加し、判定根拠を可視化。
4. stale 時に `direction_score` と `direction_confidence_pct` を減衰し、`break_points/if_then_rules` に鮮度ガードを追加。
5. `execution/order_manager._factor_age_seconds()` で `timestamp/ts/time` を受理し、
   `ENTRY_FACTOR_MAX_AGE_SEC` の stale block が ts系 factor でも確実に発火するよう補正。

Verification:
1. `pytest -q tests/scripts/test_gpt_ops_report.py tests/scripts/test_run_market_playbook_cycle.py` で `11 passed`。
2. 再実行で `factor_stale=true` の場合 `current_price_source=external_snapshot` を確認。
3. 同ケースで `direction_confidence_pct` が自動で低下（過信抑制）することを確認。

Status:
- in_progress

## 2026-02-27 15:20 UTC / 2026-02-28 00:20 JST - `quant-ops-policy` 新導線の依存欠損（bs4）復旧

Period:
- VM 反映直後（`quant-ops-policy.service` 更新後）

Source:
- VM `journalctl -u quant-ops-policy.service`
- `scripts/fetch_market_snapshot.py`
- `requirements.txt`

Fact:
- `quant-ops-policy.service` を `run_market_playbook_cycle.py` 実行に切り替えた直後、
  `ModuleNotFoundError: No module named 'bs4'` で service が失敗した。

Failure Cause:
1. `fetch_market_snapshot.py` が `from bs4 import BeautifulSoup` を使用。
2. `requirements.txt` に `beautifulsoup4` が未定義で、VM venv へ導入されていなかった。

Improvement:
1. `requirements.txt` に `beautifulsoup4==4.12.3` を追加。
2. VM で venv へ依存を導入後、`quant-ops-policy.service` を再起動して復旧する。

Verification:
1. `quant-ops-policy.service` が失敗せず完走すること。
2. `logs/gpt_ops_report.json` が更新され、`snapshot.factor_stale` と `current_price_source` が出力されること。

Status:
- in_progress

## 2026-02-27 15:00 UTC / 2026-02-28 00:00 JST - `scalp_ping_5s_c` 第13ラウンド（failfast hard block の下限を調整）

Period:
- Round12 反映直後（`2026-02-27T14:59:55+00:00` 以降）

Source:
- VM `journalctl -u quant-scalp-ping-5s-c.service`
- VM `/home/tossaki/QuantRabbit/logs/orders.db`
- Repo/VM `ops/env/scalp_ping_5s_c.env`, `ops/env/quant-order-manager.env`

Fact:
- Round12 後も C は `perf_block` が残存（直近で `perf_block=1` を確認）。
- C ログで reject 原因が明示:
  - `note=perf_block:hard:hour15:failfast:pf=0.12 win=0.28 n=43`
- 同時間帯で B は `submit_attempt=2`, `filled=2` と継続約定。

Failure Cause:
1. setup guard は緩和できたが、C の hour15 failfast（PF 下限 0.20）が先に発火。
2. C は `mapped_prefix=SCALP_PING_5S` を使うため、fallback failfast も同時に満たす必要がある。

Improvement:
1. `ops/env/scalp_ping_5s_c.env`:
  - `SCALP_PING_5S_C_PERF_GUARD_FAILFAST_PF: 0.20 -> 0.10`
  - `SCALP_PING_5S_C_PERF_GUARD_FAILFAST_WIN: 0.20 -> 0.25`
  - `SCALP_PING_5S_PERF_GUARD_FAILFAST_PF: 0.20 -> 0.10`
  - `SCALP_PING_5S_PERF_GUARD_FAILFAST_WIN: 0.20 -> 0.25`
2. `ops/env/quant-order-manager.env`:
  - 上記 C/fallback failfast 値を同値へ同期。

Verification:
1. 反映後 30 分で C の `order_reject:perf_block` が減少し、`submit_attempt/filled` が再出現すること。
2. `failfast:pf=...` 理由の reject が連続しないこと。
3. 24hで C の `sum(realized_pl)` が急悪化しないこと（setup/SL系ガードは維持）。

Status:
- in_progress

## 2026-02-27 14:58 UTC / 2026-02-27 23:58 JST - `scalp_ping_5s_c` 第12ラウンド（setup perf guard を failfast中心へ寄せる）

Period:
- Round11 反映直後（`2026-02-27T14:56:53+00:00` 以降）

Source:
- VM `/home/tossaki/QuantRabbit/logs/orders.db`
- VM `journalctl -u quant-scalp-ping-5s-c.service`
- Repo/VM `ops/env/scalp_ping_5s_c.env`, `ops/env/quant-order-manager.env`

Fact:
- Round11 後、B は約定再開:
  - `submit_attempt=4`, `filled=4`, `avg_units=121.5`
- C は `perf_block` が残存:
  - `perf_block=12`, `slo_block=1`（同期間、strategy tag 抽出）
  - 例: `14:57:40 UTC` の C `entry-skip summary total=31` で `order_reject:perf_block=10`
- C の `RISK multiplier` は `pf=0.47 / win=0.48` で、setup guard の `PF_MIN=0.90` が主にボトルネック。

Failure Cause:
1. C の setup guard が現在性能帯（PF 0.47 前後）より高すぎ、`perf_block` が継続。
2. C は `mapped_prefix=SCALP_PING_5S` を使うため、`SCALP_PING_5S_*` fallback も同時に厳しいとブロックが残る。

Improvement:
1. `ops/env/scalp_ping_5s_c.env`（C + fallback）:
  - `SETUP_MIN_TRADES: 16 -> 24`
  - `SETUP_PF_MIN: 0.90 -> 0.45`
  - `SETUP_WIN_MIN: 0.45 -> 0.40`
2. `ops/env/quant-order-manager.env`（C + fallback）:
  - `SETUP_MIN_TRADES: 16 -> 24`
  - `SETUP_PF_MIN: 0.90 -> 0.45`
  - `SETUP_WIN_MIN: 0.45 -> 0.40`

Verification:
1. 反映後 30 分で C の `order_reject:perf_block` 比率が低下すること。
2. `orders.db` で C の `submit_attempt/filled` が再出現すること。
3. 24h で C の `sum(realized_pl)` が急悪化しないこと（failfastは維持）。

Status:
- in_progress

## 2026-02-27 14:53 UTC / 2026-02-27 23:53 JST - `quant-order-manager` 第11ラウンド（B/C 閾値ドリフト同期）

Period:
- Round10 後の直近 120 分

Source:
- VM `/home/tossaki/QuantRabbit/logs/orders.db`
- VM `journalctl -u quant-scalp-ping-5s-b.service`
- VM `journalctl -u quant-scalp-ping-5s-c.service`
- Repo/VM `ops/env/quant-order-manager.env`, `ops/env/scalp_ping_5s_b.env`, `ops/env/scalp_ping_5s_c.env`

Fact:
- `orders.db` 直近120分（B/C strategy tag）は `perf_block` のみ:
  - `scalp_ping_5s_b_live: 57`
  - `scalp_ping_5s_c_live: 131`
- worker ログ主因:
  - B（14:51:23 UTC）: `no_signal:revert_not_found=17`, `extrema_block=15`, `rate_limited=22`, `order_reject:perf_block=1`
  - C（14:51:43 UTC）: `no_signal:revert_not_found=32`, `extrema_block=18`, `order_reject:entry_leading_profile_reject=7`
- `quant-order-manager.env` が worker env より厳しいまま残存:
  - B preserve-intent: `0.78/0.20/0.32`（worker `0.76/0.40/0.42`）
  - B min units: `10`（worker `1`）
  - B/C setup/hourly guard の `min_trades=6` や `setup pf/win=0.95/0.50` が worker より strict

Failure Cause:
1. order-manager 側 env ドリフトが worker 側緩和を上書きし、`perf_block` が過多。
2. B の `ORDER_MIN_UNITS=10` が縮小後通過を阻害し、送信機会を削減。
3. C/common perf guard の setup/hourly 閾値が高く、短期窓で block 判定が先行。

Improvement:
1. `ops/env/quant-order-manager.env` を worker 現行値へ同期。
2. B 同期:
  - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER=0.76`
  - `MIN_SCALE/MAX_SCALE=0.40/0.42`
  - `ORDER_MIN_UNITS=1`
  - `PERF_GUARD_HOURLY_MIN_TRADES=10`, `SETUP_MIN_TRADES=10`
  - `SETUP_PF/WIN=0.88/0.44`, `FAILFAST_PF/WIN=0.10/0.27`
3. C + fallback 同期:
  - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER=0.72`
  - `SCALP_PING_5S[_C]_PERF_GUARD_HOURLY_MIN_TRADES=16`
  - `SCALP_PING_5S[_C]_PERF_GUARD_SETUP_MIN_TRADES=16`
  - `SCALP_PING_5S[_C]_PERF_GUARD_SETUP_PF/WIN=0.90/0.45`
  - `SCALP_PING_5S[_C]_PERF_GUARD_PF/WIN_MIN=0.92/0.49`
  - `SCALP_PING_5S[_C]_PERF_GUARD_SL_LOSS_RATE_MAX=0.55`

Verification:
1. 反映後 30 分/2h で `orders.db` が `perf_block only` から脱し、`submit_attempt/filled` が再出現すること。
2. `entry-skip summary` の `order_reject:perf_block` 比率が低下すること。
3. 24h で B/C の `sum(realized_pl)` が悪化せず、`avg_loss_pips` 再拡大がないこと。

Status:
- in_progress

## 2026-02-27 14:03 UTC / 2026-02-27 23:03 JST - `scalp_ping_5s_c` 第10ラウンド（約定再開後のロット底上げ）

Period:
- Round8b 反映直後: `2026-02-27T14:00:00+00:00` 以降

Source:
- VM `/home/tossaki/QuantRabbit/logs/orders.db`
- VM `journalctl -u quant-scalp-ping-5s-c.service`

Fact:
- Round8b 後に `perf_block` は解消し、`orders.db` で `submit_attempt=3`, `filled=3` を確認。
- 一方で再開直後の約定は `buy` 偏重かつ `avg_units=1.3`（min=1, max=2）と小口化。
- `filled` 行の `entry_thesis.entry_units_intent` も `1-2` が中心で、long 露出不足が継続。

Failure Cause:
1. C worker の `BASE_ENTRY_UNITS=80` / `MIN_UNITS=1` が、現行の多段縮小下で実効 1-2 units へ収束。
2. `ENTRY_LEADING_PROFILE_UNITS_MAX_MULT=0.85` と `ALLOW_HOURS_OUTSIDE_UNITS_MULT=0.55` がロット回復を抑制。

Improvement:
1. `ops/env/scalp_ping_5s_c.env`
  - `BASE_ENTRY_UNITS 80 -> 140`
  - `MIN_UNITS 1 -> 5`
  - `MAX_UNITS 160 -> 260`
  - `ALLOW_HOURS_OUTSIDE_UNITS_MULT 0.55 -> 0.70`
  - `ENTRY_LEADING_PROFILE_UNITS_MIN_MULT 0.58 -> 0.72`
  - `ENTRY_LEADING_PROFILE_UNITS_MAX_MULT 0.85 -> 1.00`

Verification:
1. 反映後30分で C `filled` の `avg(abs(units))` が `>=5` へ上昇すること。
2. `filled` 継続（ゼロ化しない）を維持しつつ、`perf_block` が再燃しないこと。
3. 24hで C long の `sum(realized_pl)` と `avg_units` を併記し、収益立ち上がりを評価すること。

Status:
- in_progress

## 2026-02-27 13:56 UTC / 2026-02-27 22:56 JST - `scalp_ping_5s_b/c` 第9ラウンド（revert/leading の過剰拒否を小幅緩和）

Period:
- Round8 反映後（直近 120 分）
- 24h 集計（`julianday(close_time) >= julianday('now','-24 hours')`）

Source:
- VM `journalctl -u quant-scalp-ping-5s-b.service`
- VM `journalctl -u quant-scalp-ping-5s-c.service`
- VM `/home/tossaki/QuantRabbit/logs/orders.db`
- VM `/home/tossaki/QuantRabbit/logs/trades.db`

Fact:
- 24h は依然マイナス:
  - `scalp_ping_5s_b_live`: `582 trades / -670.8 JPY / -352.3 pips / avg_win=1.165 / avg_loss=1.994`
  - `scalp_ping_5s_c_live`: `359 trades / -139.1 JPY / -284.7 pips / avg_win=1.090 / avg_loss=1.849`
- 直近 120 分 `orders.db`（strategy tag filter）は `perf_block` 偏重:
  - B: `perf_block=50`
  - C: `perf_block=110`
- worker ログの skip 主因:
  - B（13:54:26 UTC）: `total=92`, `revert_not_found=32`, `rate_limited=21`, `entry_probability_reject=3`
  - C（13:54:36 UTC）: `total=108`, `revert_not_found=41`, `rate_limited=10`, `entry_leading_profile_reject=9`

Failure Cause:
1. `revert_not_found` が B/C 共通で高止まりし、シグナル段階での取りこぼしが継続。
2. C は `entry_leading_profile_reject` と `entry_probability` 側の閾値が重なり、送信前 reject が多い。
3. B は第5ラウンドで絞った `MAX_ORDERS_PER_MINUTE` の影響が残り、`rate_limited` が高め。

Improvement:
1. `ops/env/scalp_ping_5s_b.env`
   - `MAX_ORDERS_PER_MINUTE: 7 -> 8`
   - `REVERT_MIN_TICK_RATE: 0.50 -> 0.45`
   - `REVERT_RANGE_MIN_PIPS: 0.05 -> 0.04`
   - `REVERT_BOUNCE_MIN_PIPS: 0.008 -> 0.006`
   - `REVERT_CONFIRM_RATIO_MIN: 0.18 -> 0.15`
   - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER: 0.77 -> 0.76`
   - `ENTRY_LEADING_PROFILE_REJECT_BELOW: 0.68 -> 0.67`
2. `ops/env/scalp_ping_5s_c.env`
   - `ENTRY_PROBABILITY_ALIGN_FLOOR: 0.72 -> 0.70`
   - `REVERT_MIN_TICK_RATE: 0.50 -> 0.45`
   - `REVERT_RANGE_MIN_PIPS: 0.05 -> 0.04`
   - `REVERT_BOUNCE_MIN_PIPS: 0.008 -> 0.006`
   - `REVERT_CONFIRM_RATIO_MIN: 0.18 -> 0.15`
   - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER: 0.74 -> 0.72`
   - `ENTRY_LEADING_PROFILE_REJECT_BELOW: 0.66 -> 0.64`
   - `ENTRY_LEADING_PROFILE_REJECT_BELOW_SHORT: 0.82 -> 0.80`

Verification:
1. 反映後 30 分/2h で `entry-skip summary` の `revert_not_found` 比率が B/C とも低下すること。
2. C で `entry_leading_profile_reject` が減り、`orders.db` の `submit_attempt/filled` が再出現すること。
3. 24h で B/C の `sum(realized_pl)` が悪化せず、`avg_loss_pips` の再拡大がないこと。

Status:
- in_progress

## 2026-02-27 13:54 UTC / 2026-02-27 22:54 JST - `scalp_ping_5s_c` 第8ラウンド（order-manager env乖離の是正）

Period:
- Round7 反映後: `2026-02-27T13:47:22+00:00` 以降

Source:
- VM `journalctl -u quant-scalp-ping-5s-c.service`
- VM `/home/tossaki/QuantRabbit/logs/orders.db`
- VM python 実測（`workers.common.perf_guard.is_allowed`）

Fact:
- Round7 後の C 注文状態（strategy tag filter）は `perf_block=39`, `entry_probability_reject=11`, `probability_scaled=9`, `filled=0`。
- ログは `market_order rejected ... reason=perf_block` が連続し、long の送信前段で停止。
- 同時点の `perf_guard` 実測:
  - order-manager 実効env（`quant-v2-runtime` + `quant-order-manager`）では  
    `allowed=False`, `reason='hard:hour13:failfast:pf=0.32 win=0.36 n=22'`
  - worker env（`scalp_ping_5s_c.env`）も加えると  
    `allowed=True`, `reason='warn:margin_closeout_soft...'`
- failfast 同期後の再計測では reject 理由が  
  `hard:sl_loss_rate=0.50 pf=0.32 n=22` へ移行し、`perf_block` は継続。

Failure Cause:
1. `quant-order-manager.service` が読む env 側で C failfast 閾値が旧値（`min_trades=8`, `pf=0.90`, `win=0.48`）のまま残存し、hard block 化。
2. worker 側で緩めた preserve-intent 閾値（`reject_under=0.74`）が order-manager 側へ未同期で、`entry_probability_reject` が過多。

Improvement:
1. `ops/env/quant-order-manager.env` の C preserve-intent を worker 側と同期:
  - `REJECT_UNDER 0.76 -> 0.74`
  - `MIN_SCALE 0.24 -> 0.34`
  - `MAX_SCALE 0.50 -> 0.56`
2. `ops/env/quant-order-manager.env` の `SCALP_PING_5S[_C]_PERF_GUARD_FAILFAST_*` を worker 側と同期:
  - `MIN_TRADES 8 -> 30`
  - `PF 0.90 -> 0.20`
  - `WIN 0.48 -> 0.20`
3. `ops/env/quant-order-manager.env` の `SCALP_PING_5S[_C]_PERF_GUARD_SL_LOSS_RATE_*` を warmup寄りへ更新:
  - `MIN_TRADES 16 -> 30`
  - `MAX 0.55/0.50 -> 0.68`

Verification:
1. 再起動後の `perf_guard.is_allowed(..., env_prefix=SCALP_PING_5S_C)` が `allowed=True` となること。
2. 反映後30分で `orders.db` の C `submit_attempt/filled` が再出現すること。
3. `entry-skip summary` の `order_reject:perf_block` 比率が低下すること。

Status:
- in_progress

## 2026-02-27 13:30 UTC / 2026-02-27 22:30 JST - `scalp_ping_5s_b/c` 第4ラウンド（rate-limit/revert/perf同時緩和）

Period:
- Round3 反映後: `2026-02-27T13:07:28+00:00` 以降（主に `13:21-13:26 UTC`）

Source:
- VM `journalctl -u quant-scalp-ping-5s-b.service`
- VM `journalctl -u quant-scalp-ping-5s-c.service`
- VM `/home/tossaki/QuantRabbit/logs/orders.db`
- VM `/home/tossaki/QuantRabbit/logs/trades.db`

Fact:
- Round3 反映後も `entry-skip summary` の上位は `rate_limited` と `no_signal:revert_not_found`。
  - B 例（13:25:30 UTC）: `total=82`, `rate_limited=49`, `revert_not_found=14`
  - C 例（13:25:18 UTC）: `total=106`, `rate_limited=48`, `revert_not_found=25`
- `entry_leading_profile_reject` は依然 long 側にも発生（B/C とも継続）。
- 反映後直近（約 19 分）で `orders.db` は B long のみ約定:
  - `7 fills`, `avg_units=57.9`, `avg_sl=1.23 pips`, `avg_tp=1.17 pips`, `tp/sl=0.95`
  - C は同期間で `filled` が確認できず、long ロット回復が不十分。

Failure Cause:
1. `MAX_ORDERS_PER_MINUTE=6` が高頻度シグナル区間で飽和し、長短とも通過機会を失っている。
2. `REVERT_*` がまだ厳しく、`revert_not_found` による no-signal 落ちが継続。
3. long 側の `entry_leading_profile` / `preserve-intent` / setup perf guard が重なり、C で特に通過率が低い。

Improvement:
1. B/C 共通で `MAX_ORDERS_PER_MINUTE` を `10` へ引き上げ。
2. B/C 共通で `REVERT_*` を追加緩和:
   - `RANGE_MIN 0.08->0.05`, `SWEEP_MIN 0.04->0.02`, `BOUNCE_MIN 0.01->0.008`, `CONFIRM_RATIO_MIN 0.22->0.18`
3. long 通過率とサイズ下限を追加緩和:
   - B: `ENTRY_LEADING_PROFILE_REJECT_BELOW 0.68->0.64`, `UNITS_MIN_MULT 0.70->0.76`,
     `PRESERVE_INTENT_REJECT_UNDER 0.78->0.74`, `MIN_SCALE 0.34->0.40`
   - C: `ENTRY_LEADING_PROFILE_REJECT_BELOW 0.68->0.64`, `UNITS_MIN_MULT 0.68->0.74`,
     `PRESERVE_INTENT_REJECT_UNDER 0.76->0.72`, `MIN_SCALE 0.38->0.44`
4. setup perf guard の早期ブロックを緩和（B/C + Cのfallbackキー）:
   - `HOURLY_MIN_TRADES 6->10`, `SETUP_MIN_TRADES 6->10`, `SETUP_PF_MIN`/`WIN_MIN` を小幅緩和
5. `MIN_UNITS_RESCUE` 閾値を引き下げ、long の極小ロット化を抑制。

Verification:
1. 反映後30分/2hで `entry-skip summary` の `rate_limited` と `revert_not_found` 比率が低下すること。
2. C の `filled` 再開と、B/C long の `avg_units` 上昇を確認すること。
3. `perf_block` の過剰増加がないことを確認しつつ、`tp/sl` 改善方向を維持すること。

Status:
- in_progress

## 2026-02-27 13:32 UTC / 2026-02-27 22:32 JST - `scalp_ping_5s_b/c` 第5ラウンド調整（損失側圧縮 + 低品質約定の抑制）

Period:
- 直近24h（`julianday(close_time) >= julianday('now','-24 hours')`）
- 直近注文ログ（`orders.db` 最新30件）

Source:
- VM `/home/tossaki/QuantRabbit/logs/trades.db`
- VM `/home/tossaki/QuantRabbit/logs/orders.db`

Fact:
- 24h 戦略別:
  - `scalp_ping_5s_b_live`: `587 trades / -679.0 JPY / -358.8 pips / avg_win=1.158 / avg_loss=1.998 / avg_units=106.1`
  - `scalp_ping_5s_c_live`: `372 trades / -146.5 JPY / -294.6 pips / avg_win=1.075 / avg_loss=1.844 / avg_units=26.7`
- close reason（24h）:
  - B: `STOP_LOSS_ORDER 310 trades / -1094.6 JPY`, `TAKE_PROFIT_ORDER 190 / +336.8 JPY`, `MARKET_ORDER_TRADE_CLOSE 87 / +78.8 JPY`
  - C: `STOP_LOSS_ORDER 175 / -113.5 JPY`, `MARKET_ORDER_TRADE_CLOSE 111 / -89.3 JPY`, `TAKE_PROFIT_ORDER 86 / +56.2 JPY`
- 最新注文ログは `perf_block` が上位で、Cは `units=2-5` の小ロット通過が中心。

Failure Cause:
1. B/C とも `avg_loss_pips > avg_win_pips` が継続し、RR が負のまま。
2. B は `STOP_LOSS_ORDER` 側損失が過大で、勝ちトレードで吸収できていない。
3. C は通過時ユニットが小さく、低品質約定の churn で収益復元が遅い。

Improvement:
1. `ops/env/scalp_ping_5s_b.env`
   - 通過頻度抑制: `MAX_ORDERS_PER_MINUTE 10 -> 5`
   - 入口品質を引き上げ: `MIN_UNITS_RESCUE_MIN_ENTRY_PROBABILITY 0.54 -> 0.58`, `MIN_UNITS_RESCUE_MIN_CONFIDENCE 75 -> 78`
   - RR再設計: `TP_BASE/MAX 0.75/2.2 -> 0.90/2.6`, `SL_BASE/MAX 1.20/1.8 -> 1.00/1.5`, `FORCE_EXIT_MAX_FLOATING_LOSS 1.5 -> 1.2`
   - preserve-intent/leading-profile を厳格化: `REJECT_UNDER 0.74 -> 0.80`, `ENTRY_LEADING_PROFILE_REJECT_BELOW 0.64 -> 0.70`
2. `ops/env/scalp_ping_5s_c.env`
   - 絶対エクスポージャ抑制: `BASE_ENTRY_UNITS/MAX_UNITS 120/260 -> 80/160`, `MAX_ORDERS_PER_MINUTE 10 -> 6`
   - 入口品質を引き上げ: `MIN_UNITS_RESCUE_MIN_ENTRY_PROBABILITY 0.56 -> 0.60`, `MIN_UNITS_RESCUE_MIN_CONFIDENCE 78 -> 82`
   - RR再設計: `TP_BASE/MAX 0.60/1.8 -> 0.85/2.3`, `SL_BASE/MAX 1.05/1.7 -> 0.90/1.4`, `FORCE_EXIT_MAX_FLOATING_LOSS 0.8 -> 0.6`
   - preserve-intent/leading-profile を厳格化: `REJECT_UNDER 0.72 -> 0.82`, `ENTRY_LEADING_PROFILE_REJECT_BELOW 0.64 -> 0.74`

Verification:
1. 反映後2h/24hで B/C の `avg_loss_pips` と `STOP_LOSS_ORDER` の `sum(realized_pl)` が低下すること。
2. `orders.db` で B/C の `perf_block` 比率が維持されつつ、`filled` がゼロ化しないこと。
3. 24hで B/C の `sum(realized_pl)` が改善方向（損失縮小）へ転じること。

Status:
- in_progress

## 2026-02-27 13:40 UTC / 2026-02-27 22:40 JST - `scalp_ping_5s_c` spread guard が約定阻害（第5ラウンド）

Period:
- Round4 反映後: `2026-02-27T13:29:51+00:00` 以降

Source:
- VM `journalctl -u quant-scalp-ping-5s-c.service`
- VM `/home/tossaki/QuantRabbit/logs/orders.db`

Fact:
- C の反映後ログで skip 主因が `spread_blocked` に集中:
  - `13:30:35 UTC`: `entry-skip summary total=143, spread_blocked=134`
  - `13:31:05 UTC`: `entry-skip summary total=110, spread_blocked=64`
- ガード理由は `spread_med ... >= limit 1.00p` で、実勢例は `med=0.85p, p95=1.16p, max=1.20p`。
- 同期間は `orders.db` で C の `filled` を確認できず、通過不足が継続。

Failure Cause:
1. C の spread guard 閾値 `limit=1.00p` が現行マーケット実勢より低く、入口で継続ブロックされる。
2. `hot_spread_now` と `spread_med` の連鎖でクールダウンが重なり、エントリー機会が枯渇する。

Improvement:
1. `ops/env/scalp_ping_5s_c.env` に C 専用 `spread_guard_*` を追加:
   - `spread_guard_max_pips=1.30`
   - `spread_guard_release_pips=1.05`
   - `spread_guard_hot_trigger_pips=1.50`
   - `spread_guard_hot_cooldown_sec=6`
2. B は `SPREAD_GUARD_DISABLE=1` 運用を維持し、今回の調整対象外とする。

Verification:
1. 反映後30分/2hで C の `entry-skip summary` における `spread_blocked` 比率が低下すること。
2. C の `orders.db status=filled` が再開すること。
3. `spread block remain` ログの連続発生が短縮/減少すること。

Status:
- in_progress

## 2026-02-27 13:46 UTC / 2026-02-27 22:46 JST - `scalp_ping_5s_c` 第6ラウンド（rate-limit/perf_block 縮小）

Period:
- 第5ラウンド反映後: `2026-02-27T13:39:35+00:00` 以降

Source:
- VM `journalctl -u quant-scalp-ping-5s-c.service`
- VM `/home/tossaki/QuantRabbit/logs/orders.db`

Fact:
- 第5ラウンド後、`spread_blocked` はほぼ消失した一方で skip 主因が移行:
  - `13:40:24 UTC`: `total=110`, `revert_not_found=38`, `rate_limited=24`, `perf_block=5`
  - `13:40:54 UTC`: `total=118`, `rate_limited=53`, `revert_not_found=27`
- `orders.db`（同期間）は `perf_block` と `probability_scaled` のみで `filled=0`。

Failure Cause:
1. `MAX_ORDERS_PER_MINUTE=6` で高頻度区間に飽和し、`rate_limited` が先に上位化。
2. perf guard の `*_MIN_TRADES=10` が短期ノイズで発火し、`perf_block` が継続。
3. C の通過閾値（preserve-intent / leading profile）が厳しめで、注文送信まで届きにくい。

Improvement:
1. `ops/env/scalp_ping_5s_c.env`
   - `MAX_ORDERS_PER_MINUTE 6 -> 10`
   - `SCALP_PING_5S_C_PERF_GUARD_HOURLY_MIN_TRADES 10 -> 16`
   - `SCALP_PING_5S_C_PERF_GUARD_SETUP_MIN_TRADES 10 -> 16`
   - fallback `SCALP_PING_5S_PERF_GUARD_HOURLY_MIN_TRADES 10 -> 16`
   - fallback `SCALP_PING_5S_PERF_GUARD_SETUP_MIN_TRADES 10 -> 16`
   - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER 0.82 -> 0.78`
   - `ENTRY_LEADING_PROFILE_REJECT_BELOW 0.74 -> 0.70`

Verification:
1. 反映後30分/2hで `rate_limited` 比率が低下すること。
2. C の `orders.db status=filled` が再開すること。
3. `perf_block` が優位理由でなくなること。

Status:
- in_progress

## 2026-02-27 13:49 UTC / 2026-02-27 22:49 JST - `scalp_ping_5s_c` 第7ラウンド（rate_limited 優位の追加緩和）

Period:
- 第6ラウンド反映後: `2026-02-27T13:43:11+00:00` 以降

Source:
- VM `journalctl -u quant-scalp-ping-5s-c.service`
- VM `/home/tossaki/QuantRabbit/logs/orders.db`

Fact:
- 第6ラウンド後、`spread_blocked` は実質解消したが `rate_limited` が主因で残存:
  - `13:45:22 UTC`: `entry-skip summary total=107, rate_limited=65`
  - 近接窓で `revert_not_found` も継続（`12-42`程度）
- `orders.db`（同期間）は `perf_block/probability_scaled` のみで `filled=0`。

Failure Cause:
1. C の `MAX_ORDERS_PER_MINUTE=10` と `ENTRY_COOLDOWN_SEC=1.6` が高頻度局面で依然ボトルネック。
2. preserve-intent / leading profile の閾値が高く、レート制限解除後も通過率が伸びにくい。
3. `min_units_rescue` 閾値が高めで、long 側の極小シグナル救済が不足。

Improvement:
1. `ops/env/scalp_ping_5s_c.env`
   - `ENTRY_COOLDOWN_SEC 1.6 -> 1.2`
   - `MAX_ORDERS_PER_MINUTE 10 -> 16`
   - `MIN_UNITS_RESCUE_MIN_ENTRY_PROBABILITY 0.60 -> 0.56`
   - `MIN_UNITS_RESCUE_MIN_CONFIDENCE 82 -> 78`
   - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER 0.78 -> 0.74`
   - `ENTRY_LEADING_PROFILE_REJECT_BELOW 0.70 -> 0.66`

Verification:
1. 反映後30分/2hで `rate_limited` 比率が低下すること。
2. C の `orders.db status=filled` が再開すること。
3. `entry_probability_reject` と `entry_leading_profile_reject` が急増しないこと。

Status:
- in_progress

## 2026-02-27 08:52 UTC / 2026-02-27 17:52 JST - M1系 spread 閾値を 1.00 に統一
Period:
- Adjustment window: `2026-02-27 17:46` ～ `17:52` JST
- Source: VM `journalctl`, `ops/env/quant-*.env`

Fact:
- 新規3戦略ログで `blocked by spread spread=0.80p` が連続。
- 分離3戦略の spread 上限は `0.35/0.40/0.45` と不統一だった。

Failure Cause:
1. ワーカー追加時の個別チューニングで spread 上限が規定運用から外れた。
2. 実勢スプレッドに対して閾値が低すぎ、entry判定まで進まなかった。

Improvement:
1. M1系4ワーカー（既存M1 + 分離3戦略）を `M1SCALP_MAX_SPREAD_PIPS=1.00` に統一。
2. 分離3戦略に `spread_guard_max_pips=1.00` を追加し、`spread_monitor` 側のガード閾値も同一化。
3. 既存 `quant-m1scalper.env` に spread 上限を明示し、暗黙 default 依存を排除。

Verification:
1. VM反映後に `printenv/EnvironmentFile` と `journalctl` で実効 spread 上限を確認。
2. `blocked by spread` の頻度低下と、`preflight_start -> submit_attempt` 遷移を比較。

Status:
- in_progress

## 2026-02-27 13:07 UTC / 2026-02-27 22:07 JST - `order_manager` の duplicate回復失敗と orders.db スレッド競合を是正（VM実測）

Period:
- 直近24h（`2026-02-26 13:07 UTC` 以降）

Source:
- VM `systemctl status quant-order-manager.service`
- VM `journalctl -u quant-order-manager.service`
- VM `/home/tossaki/QuantRabbit/logs/orders.db`
- VM `/home/tossaki/QuantRabbit/logs/trades.db`
- VM `scripts/oanda_open_trades.py`

Fact:
- 24h損益は `1296 trades / -851.8 pips / -3285.14 JPY`。
- 大きな下押しは `scalp_ping_5s_c_live (-3431.03 JPY)` と `scalp_ping_5s_b_live (-696.34 JPY)`。
- orders最終状態（client_order_id単位）で `perf_block=1535`, `margin_usage_projected_cap=514`, `filled=491`, `entry_probability_reject=392`, `rejected=190`。
- `status='rejected'` の内訳は `CLIENT_TRADE_ID_ALREADY_EXISTS=294`, `LOSING_TAKE_PROFIT=14`, `STOP_LOSS_ON_FILL_LOSS=6`, `TAKE_PROFIT_ON_FILL_LOSS=1`。
- `quant-order-manager` で `SQLite objects created in a thread can only be used in that same thread` が継続発生し、ordersログ永続化が断続的に失敗。

Failure Cause:
1. `orders.db` 接続がプロセス内で単一接続共有になっており、複数スレッド利用で SQLite thread affinity 例外が発生。
2. duplicate復旧（`CLIENT_TRADE_ID_ALREADY_EXISTS`）は `orders.db` の filled レコード依存で、書き込み失敗時に復旧できず reject 化。
3. duplicate復旧のキャッシュ情報に `ticket_id` が保持されず、DB失敗時の回復余地が狭い。

Improvement:
1. `execution/order_manager.py`
   - `orders.db` 接続を global singleton から thread-local へ変更。
   - `_cache_order_status` に `ticket_id` を保持。
   - duplicate復旧で `orders.db` → cache → `trades.db` の順で trade_id を回収するフォールバックを追加。

Verification:
1. `python3 -m py_compile execution/order_manager.py` が成功すること。
2. 反映後VMで `journalctl -u quant-order-manager.service` の SQLite thread例外が減少/消失すること。
3. `orders.db` で `status='rejected' AND error_message='CLIENT_TRADE_ID_ALREADY_EXISTS'` が減少し、`duplicate_recovered` が増えること。
4. 24hで `reject_rate` と `scalp_ping_5s_b/c` の実効約定品質（filled比率）が改善方向になること。

Status:
- in_progress

## 2026-02-27 13:10 UTC / 2026-02-27 22:10 JST - `scalp_ping_5s_b/c` 第3ラウンド（RR再補正 + longロット押上げ）

Period:
- 直近6h（`datetime(ts) >= now - 6 hours`）を主観測

Source:
- VM `/home/tossaki/QuantRabbit/logs/orders.db`
- VM `/home/tossaki/QuantRabbit/logs/trades.db`
- VM `journalctl -u quant-scalp-ping-5s-b.service`
- VM `journalctl -u quant-scalp-ping-5s-c.service`

Fact:
- Round2 後の skip 主因は `entry_leading_profile_reject` に移行し、`rate_limited` と `revert_not_found` は直近集計で優位でない。
  - B: `entry_leading_profile_reject=39`, `entry_probability_reject=5`（直近800行）
  - C: `entry_leading_profile_reject=40`, `entry_probability_reject=3`（直近800行）
- 直近6h `orders.db`（filled）:
  - `scalp_ping_5s_b_live long`: `203 fills`, `avg_units=78.8`, `avg_sl=1.84 pips`, `avg_tp=0.99 pips`, `tp/sl=0.54`
  - `scalp_ping_5s_b_live short`: `117 fills`, `avg_units=130.4`
  - `scalp_ping_5s_c_live long`: `46 fills`, `avg_units=31.5`, `avg_sl=1.31 pips`, `avg_tp=0.96 pips`, `tp/sl=0.74`
  - `scalp_ping_5s_c_live short`: `88 fills`, `avg_units=37.2`
- 直近6h `trades.db`:
  - `B long`: `203 trades`, `sum_realized_pl=-97.4 JPY`, `avg_win=1.017 pips`, `avg_loss=1.859 pips`
  - `C long`: `46 trades`, `sum_realized_pl=-12.1 JPY`, `avg_win=0.859 pips`, `avg_loss=1.952 pips`

Failure Cause:
1. `rate_limit/revert` 問題はほぼ解消した一方、`entry_leading_profile_reject` が強く、long通過ロットが依然不足。
2. B/C long とも `tp/sl < 1` が継続し、`avg_loss_pips > avg_win_pips` の負け非対称が残存。
3. preserve-intent と leading profile の下限設定が、long側のサイズ回復を抑制している。

Improvement:
1. `ops/env/scalp_ping_5s_b.env`
   - long RR補正: `TP_BASE/MAX 0.55/1.9 -> 0.75/2.2`, `SL_BASE/MAX 1.35/2.0 -> 1.20/1.8`
   - net最小利幅引上げ: `TP_NET_MIN 0.45 -> 0.65`, `TP_TIME_MULT_MIN 0.55 -> 0.72`
   - lot押上げ: `BASE_ENTRY_UNITS 260 -> 300`, `MAX_UNITS 900 -> 1000`
   - 通過下限緩和: `ORDER_MANAGER_PRESERVE_INTENT_MIN_SCALE 0.30 -> 0.34`
   - leading profile: `REJECT_BELOW 0.70 -> 0.68`, `UNITS_MIN/MAX 0.64/0.95 -> 0.70/1.00`
2. `ops/env/scalp_ping_5s_c.env`
   - long RR補正: `TP_BASE/MAX 0.45/1.5 -> 0.60/1.8`, `SL_BASE/MAX 1.15/1.9 -> 1.05/1.7`
   - net最小利幅引上げ: `TP_NET_MIN 0.40 -> 0.55`, `TP_TIME_MULT_MIN 0.55 -> 0.70`
   - lot押上げ: `BASE_ENTRY_UNITS 95 -> 120`, `MAX_UNITS 220 -> 260`
   - 通過下限緩和: `ORDER_MANAGER_PRESERVE_INTENT_MIN_SCALE 0.34 -> 0.38`
   - leading profile: `REJECT_BELOW 0.70 -> 0.68`, `UNITS_MIN/MAX 0.62/0.90 -> 0.68/0.95`

Verification:
1. 反映後2h/24hで B/C long の `avg_tp/avg_sl` が上昇し、`tp/sl` が改善すること。
2. 反映後2h/24hで `avg_units(long)` が増加し、`entry_leading_profile_reject` の件数比率が低下すること。
3. 24hで `scalp_ping_5s_b/c long` の `sum(realized_pl)` が改善方向へ向かうこと。

Status:
- in_progress

## 2026-02-27 09:06 UTC / 2026-02-27 18:06 JST - split worker の spread 二重判定を解消（single-source化）

Period:
- Log validation window: 2026-02-27 08:45 UTC - 09:06 UTC

Source:
- VM `journalctl -u quant-scalp-{trend-breakout,pullback-continuation,failed-break-reverse}.service`
- VM `/home/tossaki/QuantRabbit/ops/env/quant-scalp-*.env`
- Repo `workers/scalp_*_*/worker.py`, `market_data/spread_monitor.py`

Fact:
- 08:45-08:55 UTC に split 3 worker で
  `blocked by spread spread=0.80p reason=guard_active` が連続発生。
- env は `M1SCALP_MAX_SPREAD_PIPS=1.00` / `spread_guard_max_pips=1.00` に揃っていたが、
  worker 側に `spread_pips > M1SCALP_MAX_SPREAD_PIPS` の別判定があり、
  設定ズレ時に entry skip が再発しうる構造だった。
- 08:57 UTC 以降の直近ブロック要因は spread ではなく
  `skip_nwave_long_late` / `tag_filter_block` / `reversion_range_block`。

Failure Cause:
1. spread 入口判定が worker と spread_monitor で重複し、構成差分時の挙動が不透明化。
2. 判定 reason の一元性が不足し、監査時に「どちらで止まったか」を切り分けにくい。

Improvement:
1. split/m1 worker で spread 判定を整理し、既定は `spread_monitor` を単一ソース化。
2. `M1SCALP_LOCAL_SPREAD_CAP_ENABLED` を追加し、
   必要時のみ local cap を有効化する運用へ変更。
3. `SPREAD_GUARD_DISABLE=1` 運用（quant-m1scalper）は local cap を自動fallbackで維持。
4. runtime env へ `M1SCALP_LOCAL_SPREAD_CAP_ENABLED` を明示し、新規worker追加時のズレを防止。

Verification:
1. split 3 worker で `blocked by spread` の件数を反映後 2h/24h で監視。
2. 同期間の block reason 内訳で spread 以外（tag/range/timing）へ収束することを確認。
3. `M1SCALP_LOCAL_SPREAD_CAP_ENABLED` が split=0 / m1=1 でロードされていることを確認。

Status:
- in_progress

## 2026-02-27 08:45 UTC / 2026-02-27 17:45 JST - 3分離戦略を既存M1と同時起動へ切替
Period:
- Activation window: `2026-02-27 17:40` ～ `17:45` JST
- Source: `ops/env/quant-scalp-*.env`, VM `systemctl` / `journalctl`

Fact:
- 直前のVM状態は、`quant-m1scalper*` のみ active、新規3戦略は disabled/inactive。
- ユーザー指定は「既存停止なしで全部起動」。

Failure Cause:
1. 新規3戦略は `M1SCALP_ENABLED=0` の安全初期値のまま。
2. unit は導入済みでも enable/start 未実施だった。

Improvement:
1. 3戦略envの `M1SCALP_ENABLED=1` に更新。
2. `quant-scalp-{trend-breakout,pullback-continuation,failed-break-reverse}` と各 exit の計6 unit を `enable --now`。
3. 既存 `quant-m1scalper*` は停止せず同時稼働を維持。

Verification:
1. `systemctl is-active` で既存M1 + 新規6unit が `active`。
2. `systemctl is-enabled` で新規6unit が `enabled`。
3. `journalctl -u quant-scalp-*-*.service` で起動直後ログを確認。

Status:
- in_progress

## 2026-02-27 08:40 UTC / 2026-02-27 17:40 JST - M1シナリオ3戦略の独立性をロジック単位まで引き上げ
Period:
- Implementation window: `2026-02-27 17:20` ～ `17:40` JST
- Source: `workers/scalp_{trend_breakout,pullback_continuation,failed_break_reverse}/*`, `tests/workers/test_m1scalper_split_workers.py`

Fact:
- 先行版は service 分離済みでも、entry/exit ロジック本体が `workers.scalp_m1scalper` 依存のラッパー構成だった。
- そのため、`m1scalper` の単一変更が3戦略へ同時波及する構造だった。

Failure Cause:
1. 「独立ワーカー」の定義が process 分離止まりで、ロジック独立まで達していなかった。
2. 戦略別のデフォルト（タグ・allowlist）が wrapper の `os.environ.setdefault` に依存していた。

Improvement:
1. 3戦略へ `m1scalper` の entry/exit 実体モジュールを複製し、パッケージ内で完結させた。
2. 各戦略 `config.py` に戦略別 default（tag/side policy）を埋め込み、wrapper依存を削除。
3. 各戦略 `exit_worker.py` の `_DEFAULT_ALLOWED_TAGS` を専用タグへ変更。
4. テストを更新し、`workers.scalp_m1scalper` 直importなし・戦略別 default 反映を検証。

Verification:
1. `pytest -q tests/workers/test_m1scalper_split_workers.py tests/workers/test_m1scalper_config.py tests/workers/test_m1scalper_quickshot.py`
2. `rg \"workers\\.scalp_m1scalper\" workers/scalp_trend_breakout workers/scalp_pullback_continuation workers/scalp_failed_break_reverse`
3. `python3 -m py_compile` で新規/更新モジュールを検証。

Status:
- in_progress

## 2026-02-27 05:35 UTC / 2026-02-27 14:35 JST - 全体監査で B/C 損失源を再圧縮し、勝ち筋へ再配分（timeout 再劣化の再発防止込み）
Period:
- VM実測: `2026-02-26 17:35 UTC` 〜 `2026-02-27 05:35 UTC`（直近12h）
- 参考窓: 24h（`julianday(close_time) >= julianday('now','-24 hours')`）
- Source: `trades.db`, `orders.db`, `metrics.db`, `journalctl -u quant-scalp-ping-5s-{b,c}.service`, `journalctl -u quant-order-manager.service`, `scripts/oanda_open_trades.py`

Fact:
- 24h realized P/L:
  - `scalp_ping_5s_c_live`: `473 trades / -1455.4 JPY`
  - `scalp_ping_5s_b_live`: `425 trades / -592.3 JPY`
  - `WickReversalBlend`: `7 trades / +332.3 JPY`
  - `scalp_extrema_reversal_live`: `14 trades / +30.9 JPY`
- 12h side別（`b_live/c_live`）:
  - `b_live buy`: `189 trades / -309.7 JPY`
  - `b_live sell`: `111 trades / -71.3 JPY`
  - `c_live buy`: `136 trades / -34.3 JPY`
  - `c_live sell`: `88 trades / -24.3 JPY`
- 12h orders（`scalp_fast/scalp/micro`）では、`filled=565` に対して
  `margin_usage_exceeds_cap=217`, `margin_usage_projected_cap=210`, `slo_block=109` が同時多発。
- `quant-v2-runtime.env` の実効値は `ORDER_MANAGER_SERVICE_TIMEOUT=60.0`。
  同期間の B/C ログには `order_manager service call failed ... Read timed out (45.0)` が `165件/12h`（`85件/2h`）発生。
- 直近稼働中の open trade でも、`B/C` と `Extrema` が混在し、損失源と勝ち筋の配分最適化余地が継続。

Failure Cause:
1. `b_live` が高頻度・高比率で負け寄与を継続（特に buy 側）。
2. `c_live` はサイズが小さい一方で頻度が高く、合算で負けを積み上げる構造が残存。
3. service timeout が長いため、order-manager 遅延時に worker が長時間ブロックされ、取りこぼしと偏りを増幅。
4. 正寄与戦略（Wick/Extrema）の配分が相対的に不足。

Hypothesis:
- B/C の頻度・上限倍率を追加圧縮し、Wick/Extrema の発火枠を増やすことで、
  `STOP_LOSS_ORDER` 優位の負け勾配を下げつつ、純益の期待値を上げられる。

Improvement:
1. B 圧縮（`ops/env/scalp_ping_5s_b.env`）:
   - `MAX_ORDERS_PER_MINUTE: 8 -> 6`
   - `BASE_ENTRY_UNITS: 380 -> 300`
   - `MAX_UNITS: 1400 -> 1100`
   - `DIRECTION_BIAS_SHORT_OPPOSITE_UNITS_MULT: 0.58 -> 0.45`
   - `DIRECTION_BIAS_LONG_OPPOSITE_UNITS_MULT: 0.68 -> 0.55`
   - `SIDE_BIAS_BLOCK_THRESHOLD: 0.00 -> 0.12`
   - `ENTRY_PROBABILITY_BAND_ALLOC_SIDE_METRICS_MAX_MULT: 0.95 -> 0.82`
2. C 圧縮（`ops/env/scalp_ping_5s_c.env`）:
   - `MAX_ORDERS_PER_MINUTE: 8 -> 6`
   - `BASE_ENTRY_UNITS: 140 -> 110`
   - `MAX_UNITS: 320 -> 240`
   - `DIRECTION_BIAS_SHORT_OPPOSITE_UNITS_MULT: 0.62 -> 0.56`
   - `DIRECTION_BIAS_LONG_OPPOSITE_UNITS_MULT: 0.72 -> 0.62`
   - `SIDE_BIAS_BLOCK_THRESHOLD: 0.10 -> 0.16`
   - `ENTRY_PROBABILITY_BAND_ALLOC_SIDE_METRICS_MAX_MULT: 1.00 -> 0.88`
   - fallback local 整合:
     `ORDER_MANAGER_PRESERVE_INTENT_(REJECT_UNDER/MIN_SCALE/MAX_SCALE)` を
     `0.68/0.35/0.72` へ同期
3. 共通 preflight 圧縮（`ops/env/quant-order-manager.env`）:
   - B: `REJECT_UNDER 0.68 -> 0.72`, `MAX_SCALE 0.50 -> 0.42`
   - C: `REJECT_UNDER 0.66 -> 0.68`, `MIN_SCALE 0.40 -> 0.35`, `MAX_SCALE 0.78 -> 0.72`
   - `ops/env/scalp_ping_5s_b.env` 側の fallback local 値も
     `REJECT_UNDER 0.68 -> 0.72`, `MAX_SCALE 0.55 -> 0.42` へ同期
4. service timeout 再発防止（`ops/env/quant-v2-runtime.env`）:
   - `ORDER_MANAGER_SERVICE_TIMEOUT: 60.0 -> 12.0`
   - `ORDER_MANAGER_SERVICE_TIMEOUT_RECOVERY_WAIT_SEC: 10.0 -> 4.0`
5. 勝ち筋再配分:
   - `ops/env/quant-scalp-wick-reversal-blend.env`
     - `MAX_OPEN_TRADES: 3 -> 4`
     - `UNIT_BASE_UNITS: 10200 -> 11200`
   - `ops/env/quant-scalp-extrema-reversal.env`
     - `COOLDOWN_SEC: 35 -> 30`
     - `MAX_OPEN_TRADES: 2 -> 3`
     - `BASE_UNITS: 12000 -> 13000`
     - `MIN_ENTRY_CONF: 57 -> 54`
6. B unit override 競合を解消（`systemd/quant-scalp-ping-5s-b.service`）:
   - unit直書きの `BASE_ENTRY_UNITS` / `MAX_UNITS` /
     `ORDER_MANAGER_PRESERVE_INTENT_*` を削除し、
     `ops/env/scalp_ping_5s_b.env` と `ops/env/quant-order-manager.env` を唯一の実効値に統一。

Verification:
1. デプロイ後、VMで `HEAD == origin/main` と `quant-order-manager`, `quant-scalp-ping-5s-{b,c}`, `quant-scalp-{wick-reversal-blend,extrema-reversal}` の再起動完了を確認。
2. 30-120分窓で `b_live/c_live` の `realized_pl`、`STOP_LOSS_ORDER` 比率、`filled`/`submit_attempt` を反映前と比較。
3. 同窓で `order_manager service call failed` と `slow_request` の再発有無を監査。
4. 同窓で Wick/Extrema の `filled` と `realized_pl` が増加/維持し、全体損益勾配が改善することを確認。

Status:
- in_progress

## 2026-02-27 17:50 JST - quickshot 判定のローカル回帰テストを追加
Period:
- Validation window: `2026-02-27 17:40` ～ `17:50` JST
- Source: `tests/workers/test_m1scalper_config.py`, `tests/workers/test_m1scalper_quickshot.py`

Fact:
- quickshot 判定の主要分岐（allow / JSTメンテ時間 block / side mismatch block）をユニットテスト化した。
- quickshot 設定値（`M1SCALP_USDJPY_QUICKSHOT_*`）の env 読込を config テストで監査可能にした。

Improvement:
1. 判定ロジックの改修時に、回帰で「誤って全拒否/全通過」になるリスクを抑制。
2. `target_jpy` 逆算ロジックの単位崩れ（pips換算）をテストで即検出できる状態へ固定。

Verification:
1. `pytest -q tests/workers/test_m1scalper_config.py tests/workers/test_m1scalper_quickshot.py`
   - `7 passed`

Status:
- done (local)

## 2026-02-27 18:20 JST - 場面別3戦略（Trend/Pullback/FailedBreak）を専用ワーカーへ分離
Period:
- Design window: `2026-02-27 18:00` ～ `18:20` JST
- Source: `workers/scalp_*`, `systemd/quant-scalp-*.service`, `ops/env/quant-scalp-*.env`

Fact:
- 既存 `M1Scalper` は single worker 内で複数シグナルを扱うため、戦略単位で on/off・監査・exit閉域化が難しかった。
- EXIT 側は固定 allowlist（`M1Scalper,m1scalper,m1_scalper`）で、strategy_tag分離に追従できなかった。

Improvement:
1. `TrendBreakout` / `PullbackContinuation` / `FailedBreakReverse` を ENTRY/EXIT ペアで新設。
2. 各 ENTRY は signal タグを固定し、strategy_tag を専用名へ分離。
3. EXIT は `M1SCALP_EXIT_TAG_ALLOWLIST` で対象タグを閉域化。
4. 既存ワーカーとの競合回避のため、新規 env は `M1SCALP_ENABLED=0` を初期値にした。

Verification:
1. 新規ラッパー module の import と env 固定化をユニットテストで確認。
2. `M1SCALP_EXIT_TAG_ALLOWLIST` の反映をユニットテストで確認。

Status:
- in_progress

## 2026-02-27 17:35 JST - M1Scalper quickshot（M5 breakout + M1 pullback + 100円逆算）を導入
Period:
- Design/implementation window: `2026-02-27 16:55` ～ `17:35` JST
- Source: `workers/scalp_m1scalper/*`, `ops/env/quant-m1scalper.env`

Fact:
- 既存 `M1Scalper` は `breakout_retest` シグナルを持つが、最終執行側で
  「100円目標のロット逆算」や「JSTメンテ時間の quickshot block」は持っていなかった。
- `entry_probability` / `entry_units_intent` 契約は既に worker 側で維持されている。

Failure Cause:
1. シグナル品質が良くても、ロットが目標利益に対して過大/過小になりやすかった。
2. 即時トレード用の追加条件（spread上限、M5方向一致、pullback成立）が統合されていなかった。

Improvement:
1. `M1SCALP_USDJPY_QUICKSHOT_*` を追加し、`M5 breakout + M1 pullback` を機械判定化。
2. `tp/sl` を ATR 連動で決定し、`target_jpy` を `tp_pips` で逆算した `target_units` を採用。
3. `entry_thesis.usdjpy_quickshot` に `setup_score/entry_probability/target_units` を保存し、監査可能化。

Verification:
1. ユニットテストで allow/block（JST7時/side mismatch）を確認。
2. VM反映後に `orders.db` の `entry_thesis.usdjpy_quickshot` と block 理由を集計。
3. 24h 集計で `avg_win_jpy` / `avg_loss_jpy` のバランス悪化がないことを確認。

Status:
- in_progress

## 2026-02-27 07:46 UTC / 2026-02-27 16:46 JST - 早利確/ロット偏りの深掘り（M1 lock_floor + B payoff是正）
Period:
- 直近48h/2h（VM live `logs/trades.db`, `logs/orders.db`）
- Source: VM `fx-trader-vm`（`sudo -u tossaki` で直接照会）

Fact:
- `scalp_ping_5s_b_live`（48h）:
  - long: `win avg_units=436.2`, `loss avg_units=632.5`（勝ち側のロットが相対的に小さい）
  - close reason別:
    - `long TAKE_PROFIT_ORDER win=118 avg_pips=+0.888 avg_units=216.2`
    - `long STOP_LOSS_ORDER loss=438 avg_pips=-2.025 avg_units=642.1`
  - 直近2hでも `STOP_LOSS_ORDER loss=119 avg_pips=-2.102` に対し
    `TAKE_PROFIT_ORDER win=99 avg_pips=+1.033` で、Rが負側に偏る。
- `M1Scalper-M1`（直近2h）:
  - `MARKET_ORDER_TRADE_CLOSE`: `loss=11 avg_pips=-1.936`, `win=9 avg_pips=+1.244`
  - `tp_pips` 比率（`pl_pips/tp_pips`）は勝ちでも `0.218` と低く、TP到達前のクローズが主体。
  - `orders.db` の `close_request.exit_reason` は `lock_floor`/`m1_rsi_fade` が主。
    - `lock_floor win=6 avg_pips=+0.983`
    - `m1_rsi_fade loss=8 avg_pips=-1.25`
- 直近2hの B/M1 発注では `filled/preflight_start` は大差なし
  （B: win `0.888`, loss `0.899`; M1: win/loss とも `1.0`）。

Failure Cause:
1. `M1Scalper` は `lock_floor` 発火が早く、`tp_hint` まで伸ばす前に利を確定しやすい。
2. `M1Scalper` の `m1_rsi_fade` が逆行初期で多発し、反発余地のある局面も早期クローズする。
3. `scalp_ping_5s_b_live` は TPが浅く（+1p台）SL側が重い（-2p台）ため、勝率が維持されても期待値が伸びにくい。

Improvement:
1. `workers/scalp_m1scalper/exit_worker.py`
  - lock/trail 関連を env で調整可能化:
    - `M1SCALP_EXIT_LOCK_TRIGGER_FROM_TP_RATIO`
    - `M1SCALP_EXIT_LOCK_TRIGGER_MIN_PIPS`
    - 既存 hard-coded の `profit_take/trail/lock` も env 化。
2. `ops/env/quant-m1scalper-exit.env`
  - `M1SCALP_EXIT_RSI_FADE_LONG=40`
  - `M1SCALP_EXIT_RSI_FADE_SHORT=60`
  - `M1SCALP_EXIT_LOCK_FROM_TP_RATIO=0.70`
  - `M1SCALP_EXIT_LOCK_TRIGGER_FROM_TP_RATIO=0.55`
  - `M1SCALP_EXIT_LOCK_TRIGGER_MIN_PIPS=1.00`
3. `ops/env/scalp_ping_5s_b.env`
  - TP/SL再調整: `TP_BASE/TP_MAX` を上げ、`SL_BASE` と force-exit loss を圧縮。
  - `ENTRY_LEADING_PROFILE_UNITS_MAX_MULT` を `0.72 -> 0.80` に引き上げ、良化局面の過小サイズを緩和。

Verification:
1. 反映後 2h/24h で `M1Scalper-M1` の `exit_reason` 分布を再集計し、
   `lock_floor` 比率低下と `take_profit` 比率上昇を確認。
2. 反映後 2h/24h で `scalp_ping_5s_b_live` の
   `avg_win_pips / avg_loss_pips` と `TAKE_PROFIT_ORDER` 平均pipsを前窓比較。
3. `orders.db` で B/M1 の `filled/preflight_start` 比率を監視し、約定率の悪化がないことを確認。

Status:
- in_progress

## 2026-02-27 14:20 UTC / 2026-02-27 23:20 JST - duplicate CID + exit disable 連鎖の実装対策（order_manager）
Period:
- Analysis window: `2026-02-20` ～ `2026-02-27`
- Source: VM `orders.db`, `trades.db`, `strategy_control.db`

Fact:
- `strategy_control_exit_disabled` が短時間に集中し、同一 trade/client で close reject が連鎖。
- `CLIENT_TRADE_ID_ALREADY_EXISTS` が多発し、filled 復元不能時に同一CID再送の再拒否ループが残っていた。
- `entry_probability` が欠損/不正な entry_thesis 経路でも、order-manager 側で reject せず通る余地があった。

Failure Cause:
1. close preflight で `strategy_control` 拒否が続くと、緊急状態でも fail-open 経路が無かった。
2. duplicate CID reject 時、filled 復元不可のケースで CID を更新せず次の reject を誘発。
3. `entry_thesis` の必須意図項目（`entry_probability`, `entry_units_intent`, `strategy_tag`）の order-manager 側検証が弱い。

Improvement:
1. `order_manager.close_trade` に連続ブロック監視 + emergency fail-open 条件を追加。
2. market/limit の duplicate CID reject で再採番リトライを追加（filled復元不可時）。
3. entry-intent guard を追加し、必須項目欠損を `entry_intent_guard_reject` で拒否。
4. reject ログへ `request_payload` を必須付与して追跡精度を上げた。

Verification:
1. ユニットテスト:
   - `tests/execution/test_order_manager_log_retry.py`
   - `tests/execution/test_order_manager_exit_policy.py`
   - `tests/workers/test_scalp_ping_5s_worker.py`
2. 反映後VM監査（予定）:
   - `orders.status='strategy_control_exit_disabled'` と `close_bypassed_strategy_control` の推移
   - `orders.status='rejected' and error_code='CLIENT_TRADE_ID_ALREADY_EXISTS'` の再発率
   - `orders.status='entry_intent_guard_reject'` の戦略別件数

Status:
- in_progress

## 2026-02-27 07:33 UTC / 2026-02-27 16:33 JST - `scalp_ping_5s_b_live` の `close_reject_no_negative` 連発停止
Period:
- 24h: `datetime(ts) >= datetime('now','-24 hours')`
- 6h: `datetime(ts) >= datetime('now','-6 hours')`
- Source: VM `/home/tossaki/QuantRabbit/logs/orders.db`, `/home/tossaki/QuantRabbit/logs/strategy_control.db`, `journalctl -u quant-strategy-control.service`

Fact:
- 稼働状態:
  - `quant-strategy-control`, `quant-order-manager`, `quant-position-manager`, `quant-market-data-feed` は全て `active`。
  - `journalctl` heartbeat は `global(entry=True, exit=True, lock=False)` を継続。
- `strategy_control.db`:
  - `strategy_control_flags` は全行 `entry_enabled=1` かつ `exit_enabled=1`（`entry=1 & exit=0` は 0 件）。
- `orders.db`（24h）:
  - `close_reject_no_negative=37`
  - `strategy_control_exit_disabled=0`（ステータス上位に非出現）
  - `client_order_id LIKE '%scalp_ping_5s_b_live%'` が `35` 件、`wick` 系が `2` 件。
- 直近拒否サンプルでは `exit_reason=candle_*` / `take_profit` で
  `status=close_reject_no_negative` が反復し、EXITシグナルが通過していない。

Failure Cause:
1. `scalp_ping_5s_b(_live)` が `scalp_ping_5s` の `neg_exit.strict_no_negative=true` を継承していた。
2. B系の exit_reason（`candle_*`, `take_profit`）が strict allow と整合せず、
   no-negative ガードが実運用で EXIT 詰まりを発生させた。
3. `strategy_control_exit_disabled` は解消済みで、今回の主因は strategy-control ではなく `neg_exit` ポリシー側。

Improvement:
1. `config/strategy_exit_protections.yaml`:
   - `scalp_ping_5s_b` / `scalp_ping_5s_b_live` に
     `neg_exit.strict_no_negative=false`
     `neg_exit.allow_reasons=["*"]`
     `neg_exit.deny_reasons=[]`
     を追加し、B系を no-block 運用へ統一。

Verification:
1. デプロイ後、`orders.db` 1h/6h で `close_reject_no_negative` の総数と
   `LIKE '%scalp_ping_5s_b_live%'` 件数が連続減少すること。
2. `close_ok` が維持され、`strategy_control_exit_disabled` が 0 を維持すること。
3. 24hで B系の負け玉平均保有時間（close遅延）が短縮すること。

Status:
- in_progress

## 2026-02-27 06:41 UTC / 2026-02-27 15:41 JST - 方向精度劣化（B/C）+ 単発大損（M1/MACD）を同時圧縮
Period:
- 集計時刻: `2026-02-27 06:41 UTC`（`15:41 JST`）
- 期間: 直近 `6h` / `24h`
- Source: VM `fx-trader-vm` (`/home/tossaki/QuantRabbit/logs/trades.db`, `orders.db`, `metrics.db`) + `scripts/oanda_open_trades.py`

Fact:
- 24h（主要赤字）:
  - `scalp_ping_5s_c_live`: `444 trades`, `-3394.6 JPY`, `-434.7 pips`
  - `scalp_ping_5s_b_live`: `264 trades`, `-519.9 JPY`, `-186.5 pips`
- 6h（主要赤字）:
  - `scalp_macd_rsi_div_live`: `1 trade`, `-279.4 JPY`, `-6.4 pips`
  - `scalp_ping_5s_b_live`: `223 trades`, `-129.2 JPY`, `-115.9 pips`
  - `M1Scalper-M1`: `20 trades`, `-90.1 JPY`, `-10.1 pips`
  - `scalp_ping_5s_c_live`: `122 trades`, `-36.1 JPY`, `-99.7 pips`
- close reason（6h）:
  - B: `STOP_LOSS_ORDER 112 trades / -265.1 JPY`、`TAKE_PROFIT_ORDER 94 / +116.5 JPY`
  - C: `STOP_LOSS_ORDER 64 / -28.7 JPY`、`MARKET_ORDER_TRADE_CLOSE 25 / -20.9 JPY`
  - M1: `MARKET_ORDER_TRADE_CLOSE 20 / -90.1 JPY`
  - MACD: `MARKET_ORDER_TRADE_CLOSE 1 / -279.4 JPY`
- 執行系:
  - `order_success_rate(avg)=0.952`, `reject_rate(avg)=0.048`
  - `decision_latency_ms(avg)=194.357`, `data_lag_ms(avg)=2032.046`
- open trades:
  - extrema short 3本のみ（`-1122/-187/-433 units`）、3本とも `stopLoss=null`。

Failure Cause:
1. B/C は勝ち負け混在でも `SL側の損失幅` が優位で、方向ミス時のpayoff非対称が継続。
2. M1（long固定）とMACD（大ロット）の単発逆行で、短時間に資産毀損を増幅。
3. preflight は動作しているが、低品質エントリーの通過ロットがなお大きい。

Improvement:
1. B: `max orders/min`, `base/max units`, `conf floor`, `entry_probability_align floor`, `preserve-intent` を厳格化。
2. C: 同様に `頻度/ロット/確率閾値` を引き上げ、`force-exit hold/loss` を短縮・厳格化。
3. order-manager: B/C の `preserve-intent` と `forecast gate` を強化し、service実効値を同期。
4. M1: `SIDE_FILTER=none` に戻しつつ `PERF_GUARD_ENABLED=1`、`base/min units` を圧縮。
5. MACD: `range必須化`, `divergence閾値強化`, `spread上限厳格化`, `base/min units` を縮小。

Verification:
1. デプロイ後 `quant-order-manager` / B / C / M1 / MACD 各serviceの実効envを `/proc/<pid>/environ` で照合。
2. 直近2h/6hで B/C の `STOP_LOSS_ORDER 比率` と `avg_loss_jpy` が低下するか確認。
3. M1/MACD の単発損失（`realized_pl`）が縮小し、`-200 JPY` 超級の再発頻度が下がるか監査。

Status:
- in_progress

## 2026-02-27 05:55 UTC / 14:55 JST - B/C 継続赤字に対する追加圧縮（損失幅優先）
Period:
- Audit window: 24h / 6h（VM `trades.db` / `orders.db` / `metrics.db`）
- Post-deploy spot check: `2026-02-27T05:46:00+00:00` 以降（`fe400c8` 反映後）

Purpose:
- B/C を停止せず稼働継続したまま、期待値の主因である「平均損失過大」をまず縮小する。

Hypothesis:
1. 執行コスト（spread/slip/submit latency）は許容範囲で、主因は entry 品質と損切り幅。
2. B は `avg_win < avg_loss` が明確なので、SL/force-exit の縮小でマイナス勾配を圧縮できる。
3. C は低品質エントリー通過が残っているため、prob/conf floor と頻度を絞ると赤字を抑えられる。

Fact:
- 24h realized:
  - `scalp_ping_5s_c_live`: `453 trades / -1077.6`
  - `scalp_ping_5s_b_live`: `454 trades / -607.3`
  - `WickReversalBlend`: `7 trades / +332.3`
  - `scalp_extrema_reversal_live`: `14 trades / +30.9`
- 6h realized:
  - `scalp_ping_5s_b_live`: `199 trades / -121.6`（`avg_win=+1.28`, `avg_loss=-2.52`）
  - `scalp_ping_5s_c_live`: `124 trades / -36.0`（`avg_win=+0.40`, `avg_loss=-0.68`）
- 24h orders status:
  - `margin_usage_projected_cap=548`, `margin_usage_exceeds_cap=368`, `rejected=314`, `entry_probability_reject=233`, `slo_block=136`
- execution precision（`analyze_entry_precision.py`）:
  - `spread_pips mean ≈ 0.80`, `slip p95 ≈ 0.20`, `latency_submit p50 ≈ 203-206ms`
  - コストよりも strategy-side の期待値設計がボトルネック。
- post-deploy spot（`05:46+00:00` 以降）:
  - B fill 3件で `avg_preflight_ms=125`, `avg_submit_to_fill_ms=213`（timeout起因の長遅延は未再発）

Failure Cause:
1. B/C とも pay-off が負（特に B は損失幅が利益幅を大きく上回る）。
2. B/C の margin cap 多発は「意図サイズ過大」を示し、期待値の低い試行が多い。
3. B/C の perf guard が disable のままで、悪化局面の縮小が効いていない。

Improvement:
1. `ops/env/scalp_ping_5s_b.env`
   - `SCALP_PING_5S_B_MAX_ACTIVE_TRADES=6`（from `10`）
   - `SCALP_PING_5S_B_MAX_PER_DIRECTION=4`（from `6`）
   - `SCALP_PING_5S_B_PERF_GUARD_ENABLED=1`（from `0`）
   - `SCALP_PING_5S_B_SL_BASE_PIPS=1.8`（from `2.2`）
   - `SCALP_PING_5S_B_SL_MAX_PIPS=2.4`（from `3.0`）
   - `SCALP_PING_5S_B_SHORT_SL_BASE_PIPS=1.7`（from `2.0`）
   - `SCALP_PING_5S_B_SHORT_SL_MAX_PIPS=2.2`（from `2.8`）
   - `SCALP_PING_5S_B_FORCE_EXIT_MAX_FLOATING_LOSS_PIPS=2.0`（from `2.6`）
   - `SCALP_PING_5S_B_SHORT_FORCE_EXIT_MAX_FLOATING_LOSS_PIPS=1.8`（from `2.2`）
   - `SCALP_PING_5S_B_FORCE_EXIT_FLOATING_LOSS_MIN_HOLD_SEC=14`（from `20`）
   - `SCALP_PING_5S_B_FORCE_EXIT_RECOVERY_WINDOW_SEC=55`（from `75`）
   - `SCALP_PING_5S_B_FORCE_EXIT_RECOVERABLE_LOSS_PIPS=0.80`（from `1.05`）
2. `ops/env/scalp_ping_5s_c.env`
   - `SCALP_PING_5S_C_MAX_ORDERS_PER_MINUTE=5`（from `6`）
   - `SCALP_PING_5S_C_BASE_ENTRY_UNITS=90`（from `110`）
   - `SCALP_PING_5S_C_MAX_UNITS=200`（from `240`）
   - `SCALP_PING_5S_C_PERF_GUARD_ENABLED=1`（from `0`）
   - `SCALP_PING_5S_PERF_GUARD_ENABLED=1`（from `0`、fallback local同期）
   - `SCALP_PING_5S_C_CONF_FLOOR=78`（from `76`）
   - `SCALP_PING_5S_C_ENTRY_PROBABILITY_ALIGN_FLOOR_RAW_MIN=0.74`（from `0.70`）
   - `SCALP_PING_5S_C_ENTRY_PROBABILITY_ALIGN_FLOOR=0.64`（from `0.61`）

Impact Scope:
- 対象: `quant-scalp-ping-5s-b.service`, `quant-scalp-ping-5s-c.service`
- 非対象: V2導線（order-manager/position-manager/strategy-control）の共通ロジック変更なし。

Verification:
1. VMで env 反映確認（`/proc/<pid>/environ`）:
   - B/C の `PERF_GUARD_ENABLED=1`, `B SL/force-exit`, `C units/prob floor` を確認。
2. 反映後 2h/6h 監査:
   - B/C の `avg_loss` 縮小（B目標 `>-2.0`, C目標 `>-0.60`）
   - B/C の `realized_pl` 勾配改善（赤字幅の縮小）
   - `margin_usage_projected_cap`, `margin_usage_exceeds_cap` の件数低下
3. 執行品質監査:
   - `analyze_entry_precision.py` で `spread/slip/latency_submit` が悪化していないことを確認。

Status:
- in_progress

## 2026-02-27 06:05 UTC / 15:05 JST - order-manager 実効設定ズレ是正（B/C perf guard 未適用の修正）
Period:
- Audit window: deploy後 spot check（`2026-02-27T05:57:00+00:00` 以降）
- Source: VM `/proc/<order-manager-pid>/environ`, `orders.db`, `trades.db`

Purpose:
- 直近チューニングが service 経路で実効化されていることを保証し、B/C の悪化局面で縮小運転を確実に発火させる。

Hypothesis:
1. `quant-order-manager` 側の `PERF_GUARD_ENABLED=0` が残っていると、worker側で有効化しても実際の preflight で効かない。
2. preserve-intent 閾値を service 側と worker 側で同値にしないと、経路差で通過品質がブレる。

Fact:
- VM 実測（`quant-order-manager` process env）で以下を確認:
  - `SCALP_PING_5S_B_PERF_GUARD_ENABLED=0`
  - `SCALP_PING_5S_C_PERF_GUARD_ENABLED=0`
  - `SCALP_PING_5S_PERF_GUARD_ENABLED=0`
  - preserve-intent も旧閾値（B `0.72/0.25/0.42`, C `0.68/0.35/0.72`）が残存
- 一方で worker env では `PERF_GUARD_ENABLED=1` へ更新済みで、service/local で実効値が不一致だった。

Improvement:
1. `ops/env/quant-order-manager.env`
   - B preserve-intent: `REJECT_UNDER=0.74`, `MAX_SCALE=0.38`
   - C preserve-intent: `REJECT_UNDER=0.72`, `MIN_SCALE=0.30`, `MAX_SCALE=0.64`
   - `SCALP_PING_5S_B_PERF_GUARD_ENABLED=1`
   - `SCALP_PING_5S_C_PERF_GUARD_MODE=reduce`, `SCALP_PING_5S_C_PERF_GUARD_ENABLED=1`
   - `SCALP_PING_5S_PERF_GUARD_MODE=reduce`, `SCALP_PING_5S_PERF_GUARD_ENABLED=1`
2. `ops/env/scalp_ping_5s_b.env` / `ops/env/scalp_ping_5s_c.env`
   - preserve-intent 閾値を service 側と同値に同期（B `0.74/0.25/0.38`, C `0.72/0.30/0.64`）

Impact Scope:
- 対象: `quant-order-manager.service` preflight（B/C）
- 非対象: 共通導線仕様（V2役割分離）は変更なし

Verification:
1. デプロイ後 `quant-order-manager` の `/proc/<pid>/environ` で上記キーを確認。
2. `orders.db` で B/C の `entry_probability_reject` 増加と `margin_usage_*cap` 減少傾向を確認。
3. 2h/6h の B/C `avg_loss` が改善することを継続監査。

Status:
- in_progress

## 2026-02-27 01:30 UTC / 2026-02-27 10:30 JST - B/C 非エントリーの直接因子を解除（revert復帰 + rate limit緩和 + service timeout短縮）
Period:
- VM実測: `2026-02-27 00:56-01:26 UTC`
- Source: `journalctl -u quant-scalp-ping-5s-{b,c}.service`, `journalctl -u quant-order-manager.service`, `/home/tossaki/QuantRabbit/logs/orders.db`, `/home/tossaki/QuantRabbit/ops/env/*.env`

Fact:
- B/C worker の startup ログで `SCALP_PING_5S_REVERT_ENABLED is OFF` が継続。
- `ops/env/scalp_ping_5s_{b,c}.env` の実値が
  - `SCALP_PING_5S_{B,C}_REVERT_ENABLED=0`
  - `SCALP_PING_5S_{B,C}_MAX_ORDERS_PER_MINUTE=4`
- 直近10分ログカウント:
  - B: `order_manager_none=39`, `revert_disabled=17`, `rate_limited=39`
  - C: `order_manager_none=49`, `revert_disabled=27`, `rate_limited=46`
- `quant-v2-runtime.env` で `ORDER_MANAGER_SERVICE_TIMEOUT=20.0`（fallback local有効）を確認。

Failure Cause:
1. `REVERT_ENABLED=0` により `no_signal:revert_disabled` が恒常化し、シグナル生成が大きく欠損。
2. `MAX_ORDERS_PER_MINUTE=4` が過抑制となり、候補シグナルの大半が `rate_limited` で棄却。
3. order-manager service timeout が長く、応答遅延時に `order_manager_none` を誘発してエントリー密度をさらに低下。

Improvement:
1. `ops/env/scalp_ping_5s_b.env`
   - `SCALP_PING_5S_B_REVERT_ENABLED: 0 -> 1`
   - `SCALP_PING_5S_B_MAX_ORDERS_PER_MINUTE: 4 -> 24`
2. `ops/env/scalp_ping_5s_c.env`
   - `SCALP_PING_5S_C_REVERT_ENABLED: 0 -> 1`
   - `SCALP_PING_5S_C_MAX_ORDERS_PER_MINUTE: 4 -> 24`
3. `ops/env/quant-v2-runtime.env`
   - `ORDER_MANAGER_SERVICE_TIMEOUT: 20.0 -> 8.0`
4. `execution/strategy_entry.py`
   - 協調/パターンゲートで `coordinated_units=0` になった場合に、
     `client_order_id` へ reject理由を `order_status` キャッシュ記録するよう変更。
   - `order_manager_none` に潰れていた reject 内訳（`coordination_*` / `pattern_gate_*`）を可視化。

Verification:
1. デプロイ後ログで `revert_enabled=1` を確認し、`revert_disabled` 件数が減少していること。
2. 直近10分の `rate_limited` 件数が B/C ともに低下していること。
3. `orders.db` で `submit_attempt` と `filled` の発生密度が維持/改善していること。
4. `order_manager_none` と `CLIENT_TRADE_ID_ALREADY_EXISTS` が逓減していること。
5. B/C ログの reject reason が `order_manager_none` から実理由（`coordination_*` 等）へ置換されること。

Status:
- in_progress

## 2026-02-27 05:55 UTC / 2026-02-27 14:55 JST - 発火頻度不足の補正
Period:
- Post-adjust short window: `2026-02-27 05:06` 以降
- Source: VM `orders.db`, `journalctl`

Fact:
- B/C の圧縮後、timeout/none は 0 を維持。
- 一方で短時間窓の約定が薄く、Wick/Extrema の寄与立ち上がりが不足。

Failure Cause:
1. 収益側戦略の監視間隔/クールダウンが相対的に長く、短期機会を拾い切れていない。

Improvement:
1. Extrema:
   - `LOOP_INTERVAL_SEC=1.5`
   - `COOLDOWN_SEC=35`
   - `LOOKBACK=24`
   - `HIGH/LOW_BAND_PIPS=0.9`
2. Wick:
   - `LOOP_INTERVAL_SEC=2.0`
   - `COOLDOWN_SEC=4`
   - `WICK_BLEND_BBW_MAX=0.0018`

Verification:
1. 再反映後 10-20 分窓で Wick/Extrema の `submit_attempt` と `filled` 増加を確認。
2. 同窓で timeout/none が再増加していないことを確認。

Status:
- in_progress

## 2026-02-27 05:40 UTC / 2026-02-27 14:40 JST - 即時収益寄せの第2段（発火不足解消）
Period:
- Analysis window: 直近60分（VM `trades.db` / `orders.db`）
- Source: `trades.db`, `orders.db`, `quant-scalp-{ping_5s_b,ping_5s_c,wick,extrema}` env

Fact:
- 直近60分:
  - `scalp_ping_5s_b_live`: `39 trades / -20.7 JPY`
  - `scalp_ping_5s_c_live`: `21 trades / -11.0 JPY`
- Wick/Extrema は稼働中だが、直近窓で新規寄与が薄く、利益側の回転不足。

Failure Cause:
1. B/C の頻度がまだ高く、短期の負け寄与を削り切れていない。
2. Wick/Extrema の閾値・クールダウンが相対的に厳しく、相場適合時の発火数が不足。

Improvement:
1. B/C 頻度を追加圧縮:
   - `MAX_ORDERS_PER_MINUTE: 12 -> 8`（B/C）
2. Extrema 発火緩和:
   - `COOLDOWN_SEC=45`, `MAX_OPEN_TRADES=2`, `MIN_ENTRY_CONF=57`
   - spread/range/rsi/leading-profile 閾値を緩和。
3. Wick 発火緩和:
   - `COOLDOWN_SEC=5`, `MAX_OPEN_TRADES=3`
   - range/adx/tick/leading-profile 閾値を緩和。

Verification:
1. デプロイ後 10-30 分窓で
   - B/C の `submit_attempt` 減少
   - Wick/Extrema の `submit_attempt` / `filled` 増加
   - 合算 realized P/L の改善
2. timeout 系 (`Read timed out`, `order_manager_none`) が増えていないことを確認。

Status:
- in_progress

## 2026-02-27 05:20 UTC / 2026-02-27 14:20 JST - B/C負け寄与の即圧縮 + Wick再配分
Period:
- Analysis window: 24h (`datetime(close_time) >= now - 24 hours`)
- Source: VM `trades.db`, `orders.db`, worker env/systemd overrides

Fact:
- 24h realized P/L:
  - `scalp_ping_5s_c_live`: `493 trades / -1984.2 JPY`
  - `scalp_ping_5s_b_live`: `415 trades / -588.6 JPY`
  - `WickReversalBlend`: `7 trades / +332.3 JPY`
- `orders.db` 24h では B/C の試行が高く、`submit_attempt` は
  `b=600`, `c=608`。高頻度・低EVの積み上げが継続していた。

Failure Cause:
1. B/C が no-stop のまま高頻度運転（`MAX_ORDERS_PER_MINUTE=24`）で負け寄与を増幅。
2. preserve-intent / leading-profile の閾値が緩く、低品質通過が残存。
3. B は systemd override（`BASE_ENTRY_UNITS=520`）が env より強く、圧縮意図とズレていた。

Improvement:
1. B/Cの頻度・サイズを即圧縮:
   - `MAX_ORDERS_PER_MINUTE: 24 -> 12`
   - `BASE_ENTRY_UNITS: B 450->380, C 170->140`
2. B/Cの通過閾値を引き上げ:
   - preserve-intent reject: `B 0.68`, `C 0.66`
   - leading profile reject: `B 0.68/0.74`, `C 0.66/0.72`
   - confidence / align floor も引き上げ。
3. B service override を同値化:
   - `BASE_ENTRY_UNITS=420`, `MAX_UNITS=780`,
     `ORDER_MANAGER_PRESERVE_INTENT_*` を env 同値へ同期。
4. 勝ち寄与へ小幅再配分:
   - `WickReversalBlend` base units `9500 -> 10200`
   - cooldown `8 -> 7`

Verification:
1. デプロイ後に `HEAD == origin/main` と各 service `active` を確認。
2. 10-30分窓で以下を比較:
   - B/C `submit_attempt` と `filled` の絶対数（過剰頻度が落ちること）
   - B/C realized P/L の損失勾配
   - `WickReversalBlend` の約定寄与

Status:
- in_progress

## 2026-02-27 03:20 UTC / 2026-02-27 12:20 JST - order-manager API の event-loop 詰まり対策
Period:
- Analysis window: 直近の `Read timed out (45.0)` 多発区間（`2026-02-27` UTC）
- Source: `workers/order_manager/worker.py`, `execution/order_manager.py`, `orders.db` 集計ログ

Fact:
- strategy worker 側で `order_manager service call failed ... Read timed out (45.0)` が継続。
- `ORDER_MANAGER_SERVICE_WORKERS=6` に増やした後も timeout 警告が残った。
- `workers/order_manager/worker.py` は `async def` endpoint で
  `execution.order_manager.*` を直接 await していた。
- `execution.order_manager` の実処理は OANDA API / SQLite への同期I/Oを含むため、
  endpoint event loop 占有が発生しやすい。

Failure Cause:
1. service worker の event loop が同期I/O処理を抱え、同時リクエスト時に head-of-line blocking が起きる。
2. RPC 応答遅延が client 側 timeout（45秒）を引き起こし、fallback/再送連鎖のトリガーになる。

Improvement:
1. `workers/order_manager/worker.py` に `_run_order_manager_call` を追加し、
   `execution.order_manager.*` の実行を `asyncio.to_thread(... asyncio.run(...))` に統一。
2. `cancel_order / close_trade / set_trade_protections / market_order / coordinate_entry_intent / limit_order`
   の全 endpoint を同ヘルパー経由へ切替。
3. `ORDER_MANAGER_SERVICE_SLOW_REQUEST_WARN_SEC`（default 8秒）を追加し、
   遅いリクエストを `slow_request` ログで監査可能化。

Verification:
1. ローカル回帰:
   - `python3 -m py_compile workers/order_manager/worker.py`（OK）
   - `pytest -q tests/execution/test_order_manager_safe_json.py tests/execution/test_order_manager_preflight.py`（29 passed）
2. VM確認（実施予定）:
   - `quant-order-manager.service` 再起動後 10分窓で
     `Read timed out` 件数と `order_manager_none` 件数の減少を確認。
   - `orders.db` の `filled/rejected/duplicate_recovered` 比率を同一窓で比較。

Status:
- in_progress（この時点では VM SSH が `Permission denied (publickey)` で未検証）

## 2026-02-27 04:55 UTC / 2026-02-27 13:55 JST - timeout 閾値を実測遅延に合わせて再調整
Period:
- Post-deploy check: `2026-02-27 04:47` ～ `04:55` UTC
- Source: VM `quant-order-manager.service` journald, `orders.db`, strategy worker logs

Fact:
- `quant-order-manager.service` 再起動後に
  `slow_request op=market_order elapsed=49.047s` を確認。
- 同時点で strategy worker 側に
  `order_manager service call failed ... Read timed out` が短時間で残存。

Failure Cause:
1. service client timeout `45.0s` が、49秒級の正常処理を timeout 扱いしていた。
2. timeout 判定が local fallback/再送を誘発し、reject ノイズと遅延を増幅しうる。

Improvement:
1. `ops/env/quant-v2-runtime.env` で
   `ORDER_MANAGER_SERVICE_TIMEOUT=60.0`（from `45.0`）へ更新。
2. `slow_request` 監査ログは継続し、閾値超過の実数を追跡する。

Verification:
1. デプロイ後に `quant-order-manager.service` 再起動と health 応答 (`/health=200`) を確認。
2. `orders.db` 直近窓で `filled` 継続・`rejected` 0 を確認（短時間窓）。
3. 次の監視ポイント:
   - `Read timed out` 件数の減少
   - `duplicate_recovered` / `rejected` 比率の改善

Status:
- in_progress

## 2026-02-27 03:25 UTC / 2026-02-27 12:25 JST - `coordination_reject` 誤ラベルと短期 sell 偏重の同時是正
Period:
- VM確認: `2026-02-27 02:55` ～ `03:20` UTC
- Source: VM `orders.db` / `trades.db` / `journalctl` (`quant-scalp-ping-5s-b/c`, `quant-order-manager`)

Fact:
- 直近30分の `orders.db` では `scalp_ping_5s_b/c` は `sell` 約定のみ（`filled=23`）だったが、2時間窓では `buy/sell` 両側の約定履歴あり。
- `journalctl` 側では `order_reject:coordination_reject` が多発していた一方、`orders.db` には同時刻の `rejected` が乖離しており、拒否理由の可観測性にズレがあった。
- `strategy_entry.market_order/limit_order` は `forecast_fusion` / `entry_leading_profile` で `units=0` になっても coordination へ進み、最終的に `coordination_reject` として記録されうる実装だった。

Failure Cause:
1. 前段拒否（forecast/leading/feedback）と coordination拒否の原因ラベルが混線し、実際の方向判定失敗点が見えない。
2. C の momentum trigger が `long=0.18 / short=0.08` と非対称で、短期的に sell 偏重へ寄りやすい設定だった。

Improvement:
1. `execution/strategy_entry.py`
   - `units=0` を前段で検知した時点で即 return し、`strong_contra_forecast` / `entry_leading_profile_reject` などの実理由を `_cache_order_status` へ記録。
   - side 記録を `units==0` 時でも要求方向（requested units）基準に統一。
   - coordination拒否と前段拒否を分離し、`coordination_reject` の過大計上を抑制。
2. `ops/env/scalp_ping_5s_b.env`
   - `SCALP_PING_5S_B_SHORT_MOMENTUM_TRIGGER_PIPS=0.09`（from `0.08`）
3. `ops/env/scalp_ping_5s_c.env`
   - `SCALP_PING_5S_C_SHORT_MOMENTUM_TRIGGER_PIPS=0.10`（from `0.08`）
   - `SCALP_PING_5S_C_LONG_MOMENTUM_TRIGGER_PIPS=0.12`（from `0.18`）

Verification:
1. Unit test: `pytest -q tests/execution/test_strategy_entry_forecast_fusion.py`（17 passed）
2. VM反映後、`journalctl` の `order_reject:*` が `strong_contra_forecast` / `entry_leading_profile_reject` などに分解されること。
3. 反映後30～60分で B/C の side 分布（buy/sell）と `pl_pips` 偏りを再集計し、sell固定化が緩和しているか監査する。

Status:
- in_progress

## 2026-02-27 01:35 UTC / 2026-02-27 10:35 JST - order_manager timeout起点の重複CIDを回収し、エントリー取りこぼしを削減
Period:
- 調査窓: `2026-02-27 01:00` ～ `01:33` UTC（`10:00` ～ `10:33` JST）
- Source: VM `journalctl`（`quant-order-manager`, `quant-scalp-ping-5s-b/c`, `quant-scalp-extrema-reversal`）, `orders.db`, `trades.db`

目的:
- `Read timed out (20s)` で service call が落ち、同一 `client_order_id` 再送から
  `CLIENT_TRADE_ID_ALREADY_EXISTS` が連鎖する経路を解消する。

仮説:
1. 20秒RPC timeout が短く、order_manager 側で実際に約定済みでも caller が失敗扱いして再送している。
2. 同一CIDの「filled済み」を再送側で回収できれば、reject扱いを成功に変換できる。

Fact:
- 直近30分 `orders.db` の `rejected=11` は全て `CLIENT_TRADE_ID_ALREADY_EXISTS`。
- 同一CIDで `filled` の後に `rejected` が発生する実例を確認:
  - `qr-1772155441497-scalp_ping_5s_c_live-l4d5f062d`
    - `01:24:51 filled ticket=406075`
    - `01:25:11 rejected CLIENT_TRADE_ID_ALREADY_EXISTS`
- B/C/Extrema で `order_manager service call failed ... Read timed out. (read timeout=20.0)` を継続観測。

Improvement:
1. `execution/order_manager.py`
   - `CLIENT_TRADE_ID_ALREADY_EXISTS` 発生時に、同一 `client_order_id` の既存 `filled` 行を `orders.db` から逆引きし、`trade_id` を回収して `duplicate_recovered` として成功返却する導線を追加。
   - service timeout 後の local fallback 前に同一CIDの `orders.db` 状態を最大10秒ポーリングし、`filled/rejected` など終端状態を先に回収して二重送信を抑止。
2. `ops/env/quant-v2-runtime.env`
   - `ORDER_MANAGER_SERVICE_TIMEOUT=45.0`（from `8.0`）
   - `ORDER_MANAGER_SERVICE_TIMEOUT_RECOVERY_WAIT_SEC=10.0`
   - `ORDER_MANAGER_SERVICE_TIMEOUT_RECOVERY_POLL_SEC=0.5`
3. `ops/env/quant-order-manager.env`
   - `ORDER_MANAGER_SERVICE_WORKERS=6`（from `4`）

影響範囲:
- order_manager の新規注文導線（`market_order`）のみ。
- EXIT共通ロジックや戦略ローカル判定には非侵襲。

検証手順:
1. デプロイ後15分で `orders.db` の `status='rejected'` うち `CLIENT_TRADE_ID_ALREADY_EXISTS` 件数を前窓比較。
2. 同一CIDに `filled + rejected` が残っても、worker側で `order_manager_none` が減ることを journal で確認。
3. 30分窓で `scalp_ping_5s_b/c` の `realized_pl` と `filled/submit_attempt` を再集計して改善有無を確認。

Status:
- in_progress

## 2026-02-26 12:19 UTC / 2026-02-26 21:19 JST - PDCA深掘り（`perf_block` 固定化 + `orders.db` ロック + SLO劣化の重畳）
Period:
- 監査期間: 直近24h（`2026-02-25 12:11` ～ `2026-02-26 12:19` UTC）
- ロック集中窓: `2026-02-26 11:26` ～ `11:46` UTC（`20:26` ～ `20:46` JST）
- Source: VM `journalctl -u quant-order-manager.service`, `journalctl -u quant-bq-sync.service`, `journalctl -u quant-position-manager.service`, `/home/tossaki/QuantRabbit/logs/{orders,metrics}.db`, `lsof`, `systemctl cat quant-bq-sync.service`

Fact:
- `orders.db` 24h集計（`rows=44862`）:
  - `preflight_start=17121`, `perf_block=16389`, `probability_scaled=7348`, `entry_probability_reject=594`
  - `submit_attempt=545`, `filled=541`（filled/submit `=99.3%`）
  - 戦略別 `perf_block`: `scalp_ping_5s_c_live=7032`, `scalp_ping_5s_b_live=4577`, `scalp_ping_5s_flow_live=3577`, `M1Scalper-M1=577`
- `perf_block` の実ログ内訳（`[ORDER][OPEN_REJECT]` 行）:
  - 合計 `191`
  - `scalp_ping_5s_b_live=157`, `M1Scalper-M1=34`
  - 主因ノート: `perf_block:hard:hour9:failfast:pf=0.56`（88件）, `hour11:failfast:pf=0.15`（36件）, `failfast:pf=0.38`（34件）
- `entry_probability` skip は `RangeFader` 系中心（sell/buy/neutral 合計74件）、`scalp_ping_5s_c_live=5`, `scalp_ping_5s_b_live=2`。
- `database is locked` は 24hで `83` 件。発生分は `35` 分に集中し、最多は `11:31/11:34`（各5件）。
- lock集中分（例 `11:26-11:42`）は `submit_attempt/filled=0` かつ `manual_margin_pressure` 併発分が多く、発注可否判定の再試行だけが増える形になっていた。
- 同時点で `lsof /home/tossaki/QuantRabbit/logs/orders.db` は PID `3400`（`run_sync_pipeline.py --interval 60 --bq-interval 300 --limit 1200`）が `40+` FD を保持。加えて戦略ワーカー PID `682/706` が保持。
- `orders.db` 自体は `6.7G` / `921,528 rows`（`2025-12-29` ～ `2026-02-26`）まで増大。
- `quant-bq-sync.service` は 60秒周期で常時 `sync_trades start`。env は `POSITION_MANAGER_SERVICE_ENABLED=1`, `POSITION_MANAGER_SERVICE_FALLBACK_LOCAL=1`, `PIPELINE_DB_READ_TIMEOUT_SEC=2.0`。
- `quant-position-manager.service` は 24hで timeout/busy 警告が複数（`fetch_recent_trades timeout`, `sync_trades timeout`, `position manager busy`）。
- `metrics.db` 24h:
  - `data_lag_ms`: `p50=768.9`, `p90=1958.7`, `p95=3603.5`, `p99=13325.1`, `max=225164.5`、閾値超過（`>1500ms`）`16.21%`
  - `decision_latency_ms`: `p50=24.4`, `p90=99.4`, `p95=678.8`, `p99=11947.6`, `max=37865.4`、閾値超過（`>1200ms`）`3.98%`
- `manual_margin_pressure=36` 件（24h）で、`scalp_ping_5s_b_live=29`, `scalp_ping_5s_c_live=7`。サンプルに `manual_net_units=-8500`, `margin_available_jpy=4677.279` を確認。

Failure Cause:
1. `perf_block` は一時的ノイズではなく、`scalp_ping_5s_b_live` と `M1Scalper-M1` の failfast 判定が時間帯別に固定化している。
2. `orders.db` への高頻度書き込み（order-manager）と、高頻度読み取り（bq-sync + position-manager fallback local）が重なり、ロック競合を増幅している。
3. `data_lag_ms` のテール悪化と `manual_margin_pressure` が同時に存在し、通るべき注文の密度を下げて改善ループが遅れる。

Hypothesis:
- `run_sync_pipeline.py` の orders 参照導線（毎分複数クエリ）と `POSITION_MANAGER_SERVICE_FALLBACK_LOCAL=1` の併用で、`orders.db` ハンドル保持が増えやすく、ロック競合を誘発している可能性が高い（A/B確認が必要）。
- `preflight_start` 行の `request_json` が `{"note":"preflight_start",...}` のみで戦略情報を持たず、ボトルネック時間帯の戦略別分解を難化させている。

Improvement:
1. P0: `quant-bq-sync` の `POSITION_MANAGER_SERVICE_FALLBACK_LOCAL` を無効化し、service timeout 時は stale キャッシュ優先でローカルDBフォールバックを抑制する。
2. P0: `orders.db` へ `preflight_start/perf_block` 記録時の `strategy_tag` と `reject_note` 永続化を必須化する（監査盲点の解消）。
3. P1: `run_sync_pipeline.py` の SQLite 読み取りを明示 close に統一し、read-only URI（`mode=ro`）・短timeout・接続数上限を導入する。
4. P1: `replay_quality_gate` / `trade_counterfactual` の worker 対象を `TrendMA/BB_RSI` 以外（`scalp_ping_5s_b_live`, `M1Scalper-M1` など）へ拡張し、failfast固定化へ直接フィードバックする。
5. P1: `manual_margin_pressure` 発火時の段階的建玉縮退（manual併走含む）を優先し、`margin_available_jpy` の下限回復を先に実行する。

Verification:
1. `journalctl -u quant-order-manager.service --since "1 hour ago" | grep -c "database is locked"` が `<=5`。
2. `lsof /home/tossaki/QuantRabbit/logs/orders.db` で PID `3400` の FD 本数が `<=10` に低下。
3. `metrics.db` で `data_lag_ms p95 < 1500`, `decision_latency_ms p95 < 1200` を継続達成。
4. `perf_block` 主因（`hour9/hour11 failfast`）の件数が日次で逓減し、`submit_attempt`/`filled` 密度が回復する。
5. `manual_margin_pressure` が 24h 連続で `0`（または明確な逓減）になる。

Status:
- in_progress

## 2026-02-27 01:25 UTC / 2026-02-27 10:25 JST - `position_manager sync_trades` 過負荷を設定で緩和
Period:
- VM実測: `2026-02-27 00:55-01:20 UTC`
- Source: `journalctl -u quant-position-manager.service`, `journalctl -u quant-scalp-wick-reversal-blend.service`

Fact:
- `position_manager` で `sync_trades timeout (8.0s)` / `position manager busy` が高頻度発生。
- WickBlend 側にも `/position/sync_trades` 失敗警告が継続し、処理遅延を誘発。

Failure Cause:
1. `sync_trades` の取得上限・呼び出し間隔・キャッシュ窓が短く、負荷集中時に timeout 連鎖していた。

Improvement:
1. `ops/env/quant-v2-runtime.env`
   - `POSITION_MANAGER_MAX_FETCH=600`（new）
   - `POSITION_MANAGER_SYNC_MIN_INTERVAL_SEC=4.0`（from `2.0`）
   - `POSITION_MANAGER_SYNC_CACHE_WINDOW_SEC=4.0`（from `1.5`）
   - `POSITION_MANAGER_WORKER_SYNC_TRADES_TIMEOUT_SEC=12.0`（from `8.0`）
   - `POSITION_MANAGER_WORKER_SYNC_TRADES_CACHE_TTL_SEC=3.0`（new）
   - `POSITION_MANAGER_WORKER_SYNC_TRADES_STALE_MAX_AGE_SEC=120.0`（from `60.0`）
   - `POSITION_MANAGER_WORKER_SYNC_TRADES_MAX_FETCH=600`（from `1000`）

Verification:
1. 再起動後に `quant-position-manager` の `sync_trades timeout` / `position manager busy` 件数が減少。
2. WickBlend の `position_manager service call failed path=/position/sync_trades` が減少。
3. `orders.db` の `submit_attempt -> filled` 変換率が維持/改善。

Status:
- in_progress

## 2026-02-27 01:15 UTC / 2026-02-27 10:15 JST - B/Cを追加圧縮（停止なしで損失勾配をさらに低減）
Period:
- VM実測: 直近30分 `scalp_ping_5s_b_live=-35.7 JPY`, `scalp_ping_5s_c_live=-18.4 JPY`
- Source: `logs/trades.db`

Improvement:
1. `ops/env/scalp_ping_5s_b.env`
   - `SCALP_PING_5S_B_BASE_ENTRY_UNITS=450`（from `600`）
   - `SCALP_PING_5S_B_MAX_ORDERS_PER_MINUTE=4`（from `5`）
2. `ops/env/scalp_ping_5s_c.env`
   - `SCALP_PING_5S_C_BASE_ENTRY_UNITS=170`（from `220`）
   - `SCALP_PING_5S_C_MAX_ORDERS_PER_MINUTE=4`（from `5`）

Verification:
1. 反映後15分の strategy_tag 付き損益で B/C 合算損失が前窓より縮小すること。
2. `filled` 件数を維持しつつ `rejected` 比率が悪化しないこと。

Status:
- in_progress

## 2026-02-27 01:20 UTC / 2026-02-27 10:20 JST - B/C 方向精度リセット（sell固定解除 + 低確率遮断強化）
Period:
- 直近24h（`close_time >= now-24h`）
- post-check（`close_time >= 2026-02-27T00:36:34Z`）
- Source: VM `/home/tossaki/QuantRabbit/logs/orders.db`, `/home/tossaki/QuantRabbit/logs/trades.db`

Fact:
- post-check で `scalp_ping_5s_b_live` / `scalp_ping_5s_c_live` は実質 `sell` のみ。
  - `B sell: 27 trades / acc 37.0% / -11.8 pips`
  - `C sell: 22 trades / acc 40.9% / -10.3 pips`
- 24h の `entry_probability` 帯別では、低確率帯が大量通過して負け寄与。
  - `B [0.55,0.60): 57 trades / acc 40.4% / -46.4 pips`
  - `C [0.00,0.55): 185 trades / acc 38.4% / -129.9 pips`
  - `C [0.55,0.60): 38 trades / acc 28.9% / -56.2 pips`
- 稼働中プロセス環境（`/proc/<pid>/environ`）で
  `SCALP_PING_5S_{B,C}_SIDE_FILTER=sell`,
  `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER` が
  `B=0.48 / C=0.46` だった。

Failure Cause:
1. B/C の side が `sell` 固定になっており、方向選択の自由度を失っていた。
2. `REJECT_UNDER` が緩く、低 edge の entry が継続通過していた。
3. `entry_leading_profile` が無効で、strategy_entry 側の追加フィルタが働いていなかった。

Improvement:
1. `ops/env/scalp_ping_5s_b.env`, `ops/env/scalp_ping_5s_c.env`
   - `SIDE_FILTER=none`
   - `ALLOW_NO_SIDE_FILTER=1`
2. 低確率遮断を引き上げ
   - `B: ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER...=0.64`
   - `C: ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER...=0.62`
   - 反映先を `scalp_ping_5s_{b,c}.env` と `quant-order-manager.env` の両方で同値化
3. strategy_entry の追加ゲート有効化
   - `SCALP_PING_5S_B_ENTRY_LEADING_PROFILE_ENABLED=1`
   - `SCALP_PING_5S_C_ENTRY_LEADING_PROFILE_ENABLED=1`
   - `REJECT_BELOW` を B/C で引き上げ（B: `0.64/0.70`, C: `0.62/0.68`）

Verification:
1. VM反映後に `/proc/<pid>/environ` で上記キーが新値へ更新されていること。
2. `orders.db` の `entry_probability_reject` 件数が増え、`probability_scaled` の低帯通過が減ること。
3. post-check で `buy/sell` の両方向が再出現し、B/C の方向一致率が `>50%` へ回復すること。

Status:
- in_progress

## 2026-02-27 01:12 UTC / 2026-02-27 10:12 JST - WickBlendを`StageTracker`初期化失敗時も継続稼働に変更
Period:
- VM実測: `2026-02-27 00:59 UTC` で `quant-scalp-wick-reversal-blend.service` が再停止
- Source: `journalctl -u quant-scalp-wick-reversal-blend.service`

Fact:
- `stage_tracker` の `sqlite3.OperationalError: database is locked` が再発し、WickBlend が `failed`。
- 例外は `StageTracker()` 初期化時に発生し、worker プロセス自体が終了していた。

Failure Cause:
1. `stage_state.db` のロック競合が強い瞬間に、`StageTracker` 初期化例外がプロセス致命化していた。

Improvement:
1. `workers/scalp_wick_reversal_blend/worker.py`
   - `StageTracker()` を `try/except` 化。
   - 初期化失敗時は `_NoopStageTracker` にフォールバックし、worker 本体は稼働継続。

Verification:
1. `python3 -m py_compile workers/scalp_wick_reversal_blend/worker.py` が pass。
2. VM反映後に `quant-scalp-wick-reversal-blend.service` の `active` と `Application started!` 継続を確認。

Status:
- in_progress

## 2026-02-27 01:10 UTC / 2026-02-27 10:10 JST - order_manager API詰まり緩和（service workerを2並列化）
Period:
- VM実測: `2026-02-27 00:57-00:58 UTC`
- Source: `journalctl -u quant-scalp-ping-5s-b.service`, `journalctl -u quant-scalp-ping-5s-c.service`, `journalctl -u quant-order-manager.service`

Fact:
- B/C worker で `order_manager service call failed ... Read timed out (read timeout=20.0)` が発生。
- 同時間帯の `quant-order-manager` は active だが、`ORDER_MANAGER_SERVICE_WORKERS=1` で単一処理。
- timeout 後の再試行で `CLIENT_TRADE_ID_ALREADY_EXISTS` reject が混在し、約定効率が低下。

Failure Cause:
1. order_manager が単一workerのため、OANDA API待ちが重なると localhost API 応答が遅延。
2. strategy worker 側の service timeout 到達で request 経路が不安定化し、重複リクエストが発生しやすい。

Improvement:
1. `ops/env/quant-order-manager.env` の `ORDER_MANAGER_SERVICE_WORKERS` を `1 -> 4` へ段階引き上げ。
2. order_manager の同時処理能力を増やし、localhost API timeout と reject の発生頻度を下げる。

Verification:
1. `quant-order-manager.service` 再起動後に active 維持。
2. 直後ウィンドウで `Read timed out` 警告件数が減少することを `journalctl` で確認。
3. `orders.db` の `filled/submit_attempt` 比率をデプロイ前後で比較する。

Status:
- in_progress

## 2026-02-27 01:00 UTC / 2026-02-27 10:00 JST - `StageTracker` 起動時ロック再発をテーブル存在確認で抑止
Period:
- VM実測: `2026-02-27 00:46-00:50 UTC`
- Source: `journalctl -u quant-scalp-wick-reversal-blend.service`, `logs/stage_state.db`

Fact:
- `quant-scalp-wick-reversal-blend.service` が起動直後に
  `sqlite3.OperationalError: database is locked` で連続停止。
- 例外位置は `execution/stage_tracker.py` の初期DDL（`CREATE TABLE IF NOT EXISTS ...`）。
- 同時に `stage_state.db` は複数 worker (`order_manager`, `scalp_ping_5s`, `tick_imbalance` 等) が共有。

Failure Cause:
1. `StageTracker.__init__` が毎回スキーマDDLを書き込み実行し、共有DBのスキーマロック競合時に起動失敗する。
2. 既存テーブルでも DDL/ALTER を実行するため、起動時ロック競合の露出面が広い。

Improvement:
1. `execution/stage_tracker.py` に `_table_exists` / `_column_exists` を追加。
2. DDLは `_ensure_table` / `_ensure_column` 経由で「不足時のみ」実行へ変更。
3. 既存テーブル環境では起動時DDLを書き込まないため、スキーマロック競合を回避。

Verification:
1. `python3 -m py_compile execution/stage_tracker.py` が pass。
2. `pytest -q tests/test_stage_tracker.py` が `3 passed`。
3. VM反映後に `quant-scalp-wick-reversal-blend.service` の `Application started!` と連続稼働を確認する。

Status:
- in_progress

## 2026-02-27 00:48 UTC / 2026-02-27 09:48 JST - `WickReversalBlend` が `stage_tracker` ロックで停止する障害を修正
Period:
- Incident window: `2026-02-27 00:46` ～ `00:48` UTC
- Source: VM `journalctl -u quant-scalp-wick-reversal-blend.service`, repo `execution/stage_tracker.py`

Fact:
- `quant-scalp-wick-reversal-blend.service` が起動直後に `failed` へ遷移。
- 直近ログで `execution/stage_tracker.py` 初期化中に
  `sqlite3.OperationalError: database is locked` を確認。
- 同時に B/C ワーカーは稼働継続しており、勝ち筋戦略だけが停止していた。

Failure Cause:
1. `StageTracker.__init__` の schema 作成が単発 `execute` で、ロック競合時に即例外終了していた。
2. `stage_state.db` に `busy_timeout` / `WAL` / retry の耐性が不足していた。

Improvement:
1. `execution/stage_tracker.py`
   - `STAGE_DB_BUSY_TIMEOUT_MS` / `STAGE_DB_LOCK_RETRY` /
     `STAGE_DB_LOCK_RETRY_SLEEP_SEC` を追加。
   - 接続を `busy_timeout + WAL + autocommit` で初期化。
   - schema作成 SQL を lock retry 付き `_execute_with_lock_retry()` 経由へ変更。

Verification:
1. `python3 -m py_compile execution/stage_tracker.py` が pass。
2. `pytest -q tests/test_stage_tracker.py` が pass（`3 passed`）。
3. 反映後 `quant-scalp-wick-reversal-blend.service` が `active` を維持すること。

Status:
- in_progress

## 2026-02-27 00:33 UTC / 2026-02-27 09:33 JST - 勝ち筋寄せ再配分（WickBlend増量 + B/C縮小）
Period:
- Snapshot window: `2026-02-27 00:30` ～ `00:33` UTC
- Source: VM `/home/tossaki/QuantRabbit/logs/trades.db`

Fact:
- 自動戦略24h（strategy_tag あり）:
  - `WickReversalBlend`: `3 trades / +114.7 JPY`
  - `scalp_ping_5s_b_live`: `292 trades / -540.2 JPY`
  - `scalp_ping_5s_c_live`: `466 trades / -3403.0 JPY`
- 直近15分（自動のみ）は `-91.9 JPY` で、依然マイナス。

Failure Cause:
1. 損失寄与の大きい B/C の約定量が、勝ち筋に対して過大。
2. WickBlend は勝っているが発火頻度と配分が低く、寄与不足。

Improvement:
1. `ops/env/quant-scalp-wick-reversal-blend.env`
   - `SCALP_PRECISION_UNIT_BASE_UNITS=9500`（from `7000`）
   - `SCALP_PRECISION_UNIT_CAP_MAX=0.65`（from `0.55`）
   - `SCALP_PRECISION_COOLDOWN_SEC=8`（from `12`）
   - `SCALP_PRECISION_MAX_OPEN_TRADES=2`（from `1`）
   - `WICK_BLEND_RANGE_SCORE_MIN=0.40`（from `0.45`）
   - `WICK_BLEND_ADX_MIN/MAX=14/28`（from `16/24`）
   - `WICK_BLEND_BB_TOUCH_RATIO=0.18`（from `0.22`）
   - `WICK_BLEND_TICK_MIN_STRENGTH=0.30`（from `0.40`）
2. `ops/env/scalp_ping_5s_b.env`
   - `SCALP_PING_5S_B_BASE_ENTRY_UNITS=600`（from `720`）
   - `SCALP_PING_5S_B_MAX_ORDERS_PER_MINUTE=5`（from `6`）
3. `ops/env/scalp_ping_5s_c.env`
   - `SCALP_PING_5S_C_BASE_ENTRY_UNITS=220`（from `260`）
   - `SCALP_PING_5S_C_MAX_ORDERS_PER_MINUTE=5`（from `6`）

Verification:
1. 反映後10分で `WickReversalBlend` の `filled` 件数が増えること。
2. 反映後15分で自動損益（strategy_tagあり）が 0 以上へ改善すること。
3. 同窓で B/C の損失寄与（JPY）が縮小すること。

Status:
- in_progress

## 2026-02-27 01:05 UTC / 2026-02-27 10:05 JST - `margin_usage_projected_cap` 誤拒否（side cap と net-reducing の不整合）
Period:
- VM `journalctl -u quant-order-manager.service`（00:11〜00:20 UTC）
- VM `/home/tossaki/QuantRabbit/logs/orders.db`（`margin_usage_projected_cap` 行）

Fact:
- `quant-order-manager` で `margin_usage_projected_cap` が連発し、B/C の `sell` シグナルが約定前に落ちていた。
- 同時刻ログに `projected margin scale ... usage=0.921~0.946` が記録され、cap 付近として扱われていた。
- 一方で口座スナップショット（同VM実測）は `usage_total` が低位（例: 約 `0.03` 台）で、総ネット余力とは乖離していた。
- 乖離は「総ネット使用率」は低いが「同方向 side 使用率」だけが高い局面で顕在化した。

Failure Cause:
1. `MARGIN_SIDE_CAP_ENABLED=1` 経路で `usage/projected_usage` を side ベースへ上書きした後、net-reducing 例外も同じ side 値で判定していた。
2. そのため「総ネット使用率を下げる注文」でも `projected_usage < usage` 条件を満たせず、`margin_usage_projected_cap` として誤拒否されていた。
3. 拒否ログに total/side の両指標が十分残っておらず、現場切り分けコストが高かった。

Improvement:
1. `execution/order_manager.py` に `_is_net_reducing_usage(...)` を追加し、例外判定を常に total usage（netting）基準へ固定。
2. side-cap 経路で `usage/projected_usage` を side 用に使っても、拒否可否は `usage_total` と `projected_usage_total` で評価するよう修正（market/limit 両経路）。
3. `margin_usage_projected_cap` ログ payload に `projected_usage_total / margin_usage_total / side_usage / side_projected` を追加し、再発時の即時判別を可能化。

Verification:
1. ローカル回帰:
   - `pytest -q tests/execution/test_order_manager_preflight.py tests/execution/test_order_manager_log_retry.py`
   - `35 passed`
2. 新規テスト:
   - `test_is_net_reducing_usage_*`（純粋判定）
   - `test_limit_order_allows_net_reducing_under_side_cap`（side cap 高負荷でも net-reducing を許可）
3. VM検証（次段）:
   - 反映後 `margin_usage_projected_cap` の連発が収束し、同条件シグナルで `filled/submitted` が再開することを確認。

Status:
- implemented_local_verified

## 2026-02-27 00:25 UTC / 2026-02-27 09:25 JST - 自動損益の実測確認と `scalp_ping_5s_b_live` 即効デリスク
Period:
- Snapshot window: `2026-02-27 00:22` ～ `00:25` UTC
- Source: VM `/home/tossaki/QuantRabbit/logs/trades.db`, `/home/tossaki/QuantRabbit/logs/orders.db`

Fact:
- 直近15分:
  - 全体: `17 trades / +1311.9 JPY`
  - 自動のみ（`strategy_tag != null`）: `16 trades / -65.1 JPY`
  - `scalp_ping_5s_b_live + scalp_ping_5s_c_live`: `14 trades / +6.9 JPY`
- JST当日（`2026-02-27 00:00 JST` 以降）:
  - 全体: `455 trades / +731.4 JPY`
  - 自動のみ: `454 trades / -645.6 JPY`
  - B/C 内訳:
    - `scalp_ping_5s_b_live`: `244 trades / -428.1 JPY`
    - `scalp_ping_5s_c_live`: `204 trades / -97.4 JPY`
- 全体プラスの主因は `strategy_tag=null` の単発決済
  （ticket `400470`, `+1377.0 JPY`, `MARKET_ORDER_TRADE_CLOSE`）。
- 直近2時間の B は long 側損失偏重:
  - long `34 trades / -41.1 JPY`
  - short `3 trades / -6.2 JPY`

Failure Cause:
1. 「全体損益」は手動/タグ欠損の単発利益に引っ張られ、自動戦略の実態が見えにくい。
2. B は long 側で低品質エントリーが残り、stop 系損失が先行している。

Improvement:
1. `ops/env/scalp_ping_5s_b.env` を即時デリスク:
   - `SCALP_PING_5S_B_BASE_ENTRY_UNITS=720`（from `900`）
   - `SCALP_PING_5S_B_CONF_FLOOR=75`（from `72`）
   - `SCALP_PING_5S_B_ENTRY_PROBABILITY_ALIGN_FLOOR_RAW_MIN=0.74`（from `0.70`）
   - `SCALP_PING_5S_B_ENTRY_PROBABILITY_ALIGN_FLOOR=0.60`（from `0.54`）
2. 停止なし方針を維持しつつ、B の低確度 long 発火を抑制して損失勾配を圧縮する。

Verification:
1. 反映後30分で `scalp_ping_5s_b_live` の `realized_pl` がゼロ超へ改善すること。
2. `orders.db` で B の `probability_scaled` / `rejected` 比率が低下すること。
3. JST当日の自動損益（`strategy_tag != null`）がマイナス幅縮小に転じること。

Status:
- in_progress

## 2026-02-26 13:30 UTC / 2026-02-26 22:30 JST - `SIDE_FILTER=none` が wrapper で `sell` 強制され、B/C entry が詰まる問題を修正
Period:
- Analysis/patch window: `2026-02-26 13:12` ～ `13:30` UTC
- Source: `workers/scalp_ping_5s_b/worker.py`, `workers/scalp_ping_5s_c/worker.py`, `tests/workers/test_scalp_ping_5s_b_worker_env.py`, `ops/env/scalp_ping_5s_{b,c}.env`

Fact:
- `ops/env/scalp_ping_5s_b.env` は `SCALP_PING_5S_B_SIDE_FILTER=none` だったが、
  wrapper 側の fail-closed 実装により不正値扱いで `sell` に上書きされていた。
- `ops/env/scalp_ping_5s_c.env` の `SCALP_PING_5S_C_SIDE_FILTER=none` と
  `SCALP_PING_5S_C_ALLOW_NO_SIDE_FILTER=1` も同様に `sell` へ上書きされていた。
- このため `SIDE_FILTER=none` を設定しても実効せず、`side_filter_block` が残る状態だった。

Failure Cause:
1. wrapper で `ALLOW_NO_SIDE_FILTER` が実装されておらず、`none` が常に invalid 扱いだった。
2. env の意図値と実効値が乖離し、skip要因の切り分けを難しくしていた。

Improvement:
1. `workers/scalp_ping_5s_b/worker.py`, `workers/scalp_ping_5s_c/worker.py`
   - `ALLOW_NO_SIDE_FILTER=1` かつ `SIDE_FILTER in {"", "none", "off", "disabled"}` のとき、
     `SCALP_PING_5S_SIDE_FILTER=""` を許可する正規化を追加。
   - 上記以外の未設定/不正値は従来どおり `sell` へ fail-closed。
2. `ops/env/scalp_ping_5s_b.env`
   - `SCALP_PING_5S_B_ALLOW_NO_SIDE_FILTER=1` を追加。
3. `tests/workers/test_scalp_ping_5s_b_worker_env.py`
   - B/C の `ALLOW_NO_SIDE_FILTER=1` で空 side filter が通る検証を追加/更新。

Verification:
1. `pytest -q tests/workers/test_scalp_ping_5s_b_worker_env.py -k "side_filter"` → `8 passed`
2. `python3 -m py_compile workers/scalp_ping_5s_b/worker.py workers/scalp_ping_5s_c/worker.py` → pass
3. VM 反映後に `entry-skip summary` の `side_filter_block` 比率が低下することを確認する。

Status:
- in_progress

## 2026-02-26 13:05 UTC / 2026-02-26 22:05 JST - 方向精度再崩れの根本対策（C no-side-filter封鎖 + side-filter復元）
Period:
- Analysis/patch window: `2026-02-26 12:40` ～ `13:05` UTC
- Source: repository `workers/scalp_ping_5s*`, `ops/env/scalp_ping_5s_c.env`, unit tests

Fact:
- `workers/scalp_ping_5s_c/worker.py` には `SCALP_PING_5S_C_ALLOW_NO_SIDE_FILTER=1` かつ `SIDE_FILTER=none` で
  side filter を未設定扱いにできる分岐が存在した。
- `ops/env/scalp_ping_5s_c.env` は実際に `SCALP_PING_5S_C_SIDE_FILTER=none`,
  `SCALP_PING_5S_C_ALLOW_NO_SIDE_FILTER=1` だった。
- `workers/scalp_ping_5s/worker.py` は初段で side_filter を通した後に
  ルーティングで side が反転した場合、最終 `side_filter_final_block` で no-entry になる設計だった。

Failure Cause:
1. C に no-side-filter 例外があり、方向固定の前提が運用設定で破れる。
2. side_filter を通過したシグナルでも、後段flipで反転すると発注前に消失し、エントリー密度が不安定化する。

Improvement:
1. C ラッパーで no-side-filter 例外を廃止し、invalid/missing を常に `sell` へ fail-closed。
2. 本体ワーカーへ `_resolve_final_signal_for_side_filter()` を追加し、
   後段反転時は初段の side-filter 適合シグナルへ復元して発注経路を維持。
3. 運用envを `SCALP_PING_5S_C_SIDE_FILTER=sell`, `SCALP_PING_5S_C_ALLOW_NO_SIDE_FILTER=0` に固定。
4. ロット計算は途中丸めを廃止して最終丸めへ統一し、
   `units_below_min` による 0 化でシグナルが消える経路を縮小。
5. `MIN_UNITS_RESCUE`（確率/信頼度/リスクcap条件付き）を導入し、
   最終段だけ 1unit 救済して実約定導線を維持。

Verification:
1. 対象テスト（10件）:
   - `tests/workers/test_scalp_ping_5s_b_worker_env.py`
   - `tests/workers/test_scalp_ping_5s_worker.py -k resolve_final_signal_for_side_filter`
2. 結果: `10 passed`

Status:
- in_progress

## 2026-02-26 12:35 UTC / 2026-02-26 21:35 JST - quote 問題の実測再監査と執行層ハードニング
Period:
- 直近24h / 7d（VM実DB）
- Source: VM `/home/tossaki/QuantRabbit/logs/orders.db`, `journalctl -u quant-order-manager.service`

Fact:
- 24h `orders` 合計: `50365`
- 24h status 上位:
  - `preflight_start=18629`
  - `perf_block=16971`
  - `probability_scaled=7719`
  - `entry_probability_reject=1101`
- 24h error:
  - `TRADE_DOESNT_EXIST=8`、`OFF_QUOTES/PRICE_*` は 0
- 7d `quant-order-manager` journal でも `quote_unavailable`/`quote_retry`/`OFF_QUOTES`/`PRICE_*` は 0

Failure Cause:
1. 現在の損失・機会損失の主因は quote 不足ではなく、`perf_block` と確率/余力ガード側。
2. ただし再クオート要求が急増する局面では既定 `FETCH=2 + RETRY=1` が薄く、将来的な取り逃しリスクは残る。

Improvement:
1. `ops/env/quant-order-manager.env` で quote 再取得耐性を強化。
2. `ORDER_SUBMIT_MAX_ATTEMPTS=1` は維持し、quote 専用リトライだけ増やして戦略判定を汚さない。

Verification:
1. 反映後24hで `quote_unavailable`/`quote_retry`/`OFF_QUOTES`/`PRICE_*` の件数を再計測。
2. 同期間で `filled / submit_attempt` 比率の悪化がないことを確認。
3. `perf_block` が依然主因なら quote ではなく戦略側改善を優先継続。

Status:
- in_progress

## 2026-02-26 12:25 UTC / 2026-02-26 21:25 JST - no-stop維持で「無約定化」を解消する再配線
Period:
- Incident window: `2026-02-26 11:40` ～ `12:25` UTC
- Source: VM `/home/tossaki/QuantRabbit/logs/orders.db`, `/home/tossaki/QuantRabbit/logs/trades.db`, `journalctl -u quant-scalp-ping-5s-{b,c}.service`

Fact:
- `orders.db` の `orders` は直近30分で `0 rows`（`datetime(substr(ts,1,19)) >= now-30m`）。
- `quant-order-manager` は `coordinate_entry_intent` 受信を継続しているが、`preflight_start` 以降の新規発注イベントが停止。
- `entry_intent_board` の直近45分は `1件` のみで、`scalp_extrema_reversal_live` が `below_min_units_after_scale`（`raw=45`, `min_units=1000`）で reject。
- B/C ワーカーは稼働継続しているが、`entry-skip summary` は
  `no_signal:revert_not_found` が最多、次点で `no_signal:side_filter_block` / `units_below_min` が継続。

Failure Cause:
1. no-stop方針の下で `SIDE_FILTER=sell` と逆風ドリフト縮小が重なり、B/C が local 判定段階で枯渇。
2. 共通 `POLICY_HEURISTIC_PERF_BLOCK` が有効のままで、他戦略側の再起動余地も狭い。
3. 最小ロット閾値が小口シグナルの通過率を削り、intent が `order_manager` まで届かない局面が残る。

Improvement:
1. 共通 hard reject の解除:
   - `ops/env/quant-v2-runtime.env`
   - `POLICY_HEURISTIC_PERF_BLOCK_ENABLED=0`（from `1`）
2. B の通過率回復:
   - `SCALP_PING_5S_B_MIN_UNITS=1`（from `5`）
   - `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_B(_LIVE)=1`（from `5`）
   - `SHORT_MOMENTUM_TRIGGER_PIPS=0.08`（from `0.10`）
   - `DIRECTION_BIAS_SHORT_OPPOSITE_UNITS_MULT=0.58`（from `0.42`）
   - `SIDE_BIAS_SCALE_GAIN/FLOOR=0.35/0.28`（from `0.50/0.18`）
3. C の無約定解消（停止ではなく両方向縮小運転）:
   - `SCALP_PING_5S_C_SIDE_FILTER=none`（from `sell`）
   - `SCALP_PING_5S_C_ALLOW_NO_SIDE_FILTER=1`
   - `SCALP_PING_5S_C_MIN_UNITS=1`（from `5`）
   - `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_C(_LIVE)=1`（from `5`）
   - `SHORT/LONG_MOMENTUM_TRIGGER_PIPS=0.08/0.18`（from `0.10/0.10`）
   - `DIRECTION_BIAS_SHORT_OPPOSITE_UNITS_MULT=0.62`（from `0.45`）
   - `SIDE_BIAS_SCALE_GAIN/FLOOR=0.35/0.28`（from `0.50/0.18`）
4. C の reject/rate-limit 緩和:
   - `SCALP_PING_5S_C_MAX_ORDERS_PER_MINUTE=3`（from `1`）
   - `SCALP_PING_5S_C_ENTRY_PROBABILITY_ALIGN_FLOOR_RAW_MIN=0.68`（from `0.76`）
   - `SCALP_PING_5S_C_ENTRY_PROBABILITY_ALIGN_FLOOR=0.58`（from `0.70`）
   - `SCALP_PING_5S_C_ENTRY_PROBABILITY_ALIGN_FLOOR_MAX_COUNTER=0.38`（from `0.24`）
   - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_C_LIVE=0.52`（from `0.62`）
   - `SCALP_PING_5S_C_PERF_GUARD_FAILFAST_MIN_TRADES=30`（from `6`）
   - `SCALP_PING_5S_C_PERF_GUARD_FAILFAST_PF/WIN=0.20/0.20`（from `0.90/0.48`）
   - `SCALP_PING_5S_PERF_GUARD_FAILFAST_MIN_TRADES=30`（from `6`）
   - `SCALP_PING_5S_PERF_GUARD_FAILFAST_PF/WIN=0.20/0.20`（from `0.90/0.48`）
   - `SCALP_PING_5S_C_PERF_GUARD_ENABLED=0`
   - `SCALP_PING_5S_PERF_GUARD_ENABLED=0`
   - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_C_LIVE=0.46`（from `0.52`）
   - `ops/env/quant-order-manager.env` の C上書き値を同値同期
     - `REJECT_UNDER=0.46`, `MIN/MAX_SCALE=0.40/0.85`, `BOOST_PROBABILITY=0.85`
     - `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_C(_LIVE)=1`
     - `SCALP_PING_5S(_C)_PERF_GUARD_MODE=off`, `SCALP_PING_5S(_C)_PERF_GUARD_ENABLED=0`
   - `scalp_extrema_reversal_live` の協調拒否解消:
     - `ORDER_MIN_UNITS_STRATEGY_SCALP_EXTREMA_REVERSAL(_LIVE)=30`（runtime/order-manager 両env）

Verification:
1. 反映後 30分で `orders.db` の `preflight_start` と `filled` が再出現すること。
2. `entry-skip summary` の `units_below_min` 比率が低下すること。
3. 反映後 60分で `trades.db` の `realized_pl` 増分が `scalp_ping_5s_c_live` 単独で急悪化しないこと（損失勾配監視）。

Status:
- in_progress

## 2026-02-26 12:20 UTC / 2026-02-26 21:20 JST - `scalp_ping_5s_b_live/c_live` 方向劣化の再発防止（side filter fail-closed）
Period:
- Direction audit window: `datetime(close_time) >= now - 24 hours`
- Runtime env check: `quant-scalp-ping-5s-b.service` MainPID/child PID
- Source: VM `/home/tossaki/QuantRabbit/logs/orders.db`, `/home/tossaki/QuantRabbit/logs/trades.db`, systemd process env

Fact:
- 24h集計（`trades.db`）で `scalp_ping_5s_b_live` の buy 側が劣化:
  - buy `72 trades`, win rate `20.83%`, avg `-0.936 pips`
  - `entry_probability>=0.75` の buy でも win rate `18.84%`, avg `-1.051 pips`
- 実行中プロセス（起動 `2026-02-26 12:10:55 UTC`）では
  `SCALP_PING_5S_B_SIDE_FILTER=sell` と
  `SCALP_PING_5S_SIDE_FILTER=sell` が有効。
- ただし過去履歴には buy 発注が残るため、env 欠落/不正時に再発しうる。

Failure Cause:
1. B/C variant の方向制御が env 設定依存で、設定欠落時に fail-open になり得る。
2. 高確率帯 buy の実績悪化と整合しない方向シグナルが通過した履歴が存在する。

Improvement:
1. `workers/scalp_ping_5s_b/worker.py` と `workers/scalp_ping_5s_c/worker.py` で side filter を fail-closed 化。
2. `SCALP_PING_5S_SIDE_FILTER` が未設定/不正値なら `sell` を強制。
3. 起動ログへ `side_filter` を明示出力し、監査を容易化。
4. `tests/workers/test_scalp_ping_5s_b_worker_env.py` に
   B/C それぞれの `missing/invalid/valid` ケースを追加。

Verification:
1. `pytest -q tests/workers/test_scalp_ping_5s_b_worker_env.py` が全緑（14 passed）。
2. VMで `quant-scalp-ping-5s-b` の子プロセス環境に
   `SCALP_PING_5S_SIDE_FILTER=sell` が存在することを確認。

Status:
- in_progress

## 2026-02-26 12:09 UTC / 2026-02-26 21:09 JST - B側の `units_below_min` 残りを削るため最小ロット閾値を再調整
Period:
- Snapshot: `2026-02-26 12:06:59` ～ `12:08:10` UTC（`21:06:59` ～ `21:08:10` JST）
- Source: VM `journalctl -u quant-scalp-ping-5s-{b,c}.service`, `/home/tossaki/QuantRabbit/ops/env/{scalp_ping_5s_b.env,quant-order-manager.env}`

Fact:
- 12:06:59 UTC 再起動後、`C` は `units_below_min=0` まで低下。
- 同条件で `B` は `entry-skip summary side=short total=4 units_below_min=4` が残存。
- `RISK multiplier` は引き続き `mult=0.40` で、縮小局面の最終ユニットが閾値を割り込みやすい。

Failure Cause:
1. Bは strategy 側と order-manager 側の min units が `20` で揃っており、縮小ロットの通過余地が不足していた。

Improvement:
1. `ops/env/scalp_ping_5s_b.env`
   - `SCALP_PING_5S_B_MIN_UNITS: 20 -> 10`
   - `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_B_LIVE: 20 -> 10`
   - `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_B: 20 -> 10`
2. `ops/env/quant-order-manager.env`
   - `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_B_LIVE: 20 -> 10`
   - `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_B: 20 -> 10`

Verification:
1. 反映後15分で `journalctl -u quant-scalp-ping-5s-b.service` の `units_below_min` が 0 になること。
2. `orders.db` で `scalp_ping_5s_b_live` の `submit_attempt` / `filled` が再出現すること。
3. `manual_margin_pressure` / `slo_block` が再増加しないこと。

Status:
- in_progress

## 2026-02-26 12:20 UTC / 2026-02-26 21:20 JST - short-only後の無約定ボトルネック（revert_not_found + short units_below_min）
Period:
- Observation window: `2026-02-26 12:00:00` 以降（VM journal）
- Source: `journalctl -u quant-scalp-ping-5s-b.service`, `journalctl -u quant-scalp-ping-5s-c.service`, `journalctl -u quant-order-manager.service`

Fact:
- `SCALP_PING_5S_B/C_SIDE_FILTER=sell` は機能し、long は `no_signal:side_filter_block` / `side_filter_final_block` で継続遮断。
- 一方で約定が止まり、`quant-order-manager` の `OPEN_REJECT/OPEN_FILLED` は同窓で実質発生なし。
- B/C の skip 内訳は `no_signal:revert_not_found` が最大（各30秒集計で概ね `40-94` 件）。
- short 側は `units_below_min` が継続（B: `3-17`, C: `1-6` / 30秒集計）。

Failure Cause:
1. short-only化後、短期反転検知（revert）が成立せず `revert_not_found` に集中。
2. 成立した short シグナルも、動的縮小後ユニットが最小ロット未満になり通過不能。
3. long遮断は効いているが、short化の再配線が不足し、取引密度が0近傍に落ちた。

Improvement:
1. B/C の `revert` 閾値を同時緩和（`REVERT_RANGE/SWEEP/BOUNCE/CONFIRM_RATIO`, `REVERT_SHORT_WINDOW`）。
2. short 最小通過ロットを引き下げ（`SCALP_PING_5S_{B,C}_MIN_UNITS`, `ORDER_MIN_UNITS_STRATEGY_*` を `5`）。
3. C は short 発火側を追加緩和（`SHORT_MIN_TICKS`, `SHORT_MIN_SIGNAL_TICKS`）。
4. long→short 変換を有効化（B/C `EXTREMA_GATE_ENABLED=1`, `EXTREMA_REVERSAL_ALLOW_LONG_TO_SHORT=1`, `LONG_TO_SHORT_MIN_SCORE` 緩和）。
5. C は flip系を再稼働（`SIDE_METRICS_DIRECTION_FLIP_ENABLED=1`）し、short側への再配線を強化。

Verification:
1. 反映後30分で `entry-skip summary side=short` の `units_below_min` 比率が低下すること。
2. `orders.db` で `filled` が再発し、`strategy in {scalp_ping_5s_b_live, scalp_ping_5s_c_live}` の short約定が出ること。
3. `trades.db` で B/C の新規closeにおける `realized_pl` の負勾配が反転または鈍化すること。

Status:
- in_progress

## 2026-02-26 12:06 UTC / 2026-02-26 21:06 JST - 損失主因の再監査（執行品質より制御異常が支配）
Period:
- 24h監査: `orders.db` / `trades.db` / `metrics.db`（`datetime(ts/close_time) >= now - 24 hours`）
- 事故窓監査: `2026-02-24 02:20:16` ～ `09:13:14` UTC（`11:20:16` ～ `18:13:14` JST）
- Source: VM `systemd`, `journalctl`, `/home/tossaki/QuantRabbit/logs/{orders,trades,metrics}.db`, `oanda_*_live.json`

Fact:
- V2主導線は稼働中（`quant-market-data-feed/order-manager/position-manager/strategy-control` は active/running）。
- 直近24hの `orders`:
  - `preflight_start=18629`, `perf_block=16971`, `probability_scaled=7719`, `submit_attempt=1221`, `filled=1202`
  - `rejected=18`, `slo_block=12`, `margin_usage_projected_cap=190`, `manual_margin_pressure=36`
- 直近24hの `trades`: `1283件`, `-4417.54 JPY`。
  - close reason: `STOP_LOSS_ORDER=390件/-5232.48 JPY`, `MARKET_ORDER_MARGIN_CLOSEOUT=3件/-2163.76 JPY`
  - 戦略別下位: `scalp_ping_5s_c_live=-7025.83 JPY`, `scalp_ping_5s_b_live=-3144.31 JPY`
- 執行品質（`analyze_entry_precision.py --limit 1200`）:
  - `slip p95=0.300 pips`, `cost_vs_mid p95=0.700 pips`, `submit latency p95=276.9 ms`, `missing quote=0`
- レイテンシ/SLO:
  - `data_lag_ms p95=4485.9`, `decision_latency_ms p95=8835.9`
  - 閾値超過率: `data_lag_ms>1500ms = 22.8%`, `decision_latency_ms>2000ms = 10.71%`
  - `journalctl quant-order-manager` に `slo_block:data_lag_p95_exceeded` を確認（2026-02-26 11:42～11:45 UTC）
- 事故窓の確定事実:
  - `strategy_control_exit_disabled=10277`（2026-02-24 JST日単位）
  - `micro-micropul*` 4注文で各 `2044～2045` 回の exit disabled 後、`MARKET_ORDER_MARGIN_CLOSEOUT`
  - 上記4件合計 `-16837.43 JPY`
- 直近口座スナップショット（2026-02-26 12:00:35 UTC）:
  - `nav=56970.299`, `margin_used=53060.740`, `margin_available=3943.559`, `health_buffer=0.06918`
  - `USD_JPY short_units=8500`
- `entry_thesis` 必須欠損:
  - 直近24hで `entry_probability/entry_units_intent` 欠損 `80/1283` 件（主に `scalp_ping_5s_*` 派生タグ行）

Failure Cause:
1. 大損の一次原因は、戦略予測精度より `exit不能（strategy_control_exit_disabled）` と `margin closeout tail` のシステム異常。
2. manual併走を含む高マージン使用状態で、`margin_*` ガードと closeout が発火しやすい口座状態が継続。
3. `data_lag_ms` スパイクにより `slo_block` が増え、良い局面のエントリーが欠落して収益回復を阻害。
4. `entry_thesis` 欠損により、意図協調/監査の完全性が崩れ、異常時の切り分けと制御精度を落としている。
5. B/Cは執行コスト劣化より `STOP_LOSS_ORDER` と手仕舞い設計由来の期待値悪化が支配的。

Improvement:
1. `strategy_control` ガード強化:
   - `entry=1 & exit=0` を検知即時で自動修正（`entry=0 & exit=1`）し、監査ログとアラートを必須化。
2. exit不能の再発防止:
   - `strategy_control_exit_disabled` が同一 `client_order_id` で閾値超過したら強制エスカレーション（exit優先モード）。
3. マージン防衛:
   - manual玉込みで `health_buffer` / `margin_available` 連動の新規抑制を強化し、先に建玉縮退を実行。
4. SLO復旧:
   - `data_lag_ms` スパイク時間帯の `market-data-feed` / DB遅延要因を切り分け、`slo_block` は hard reject 連打ではなく段階縮小中心へ。
5. `entry_thesis` スキーマ強制:
   - 非manual注文は `entry_probability` と `entry_units_intent` 欠損時に reject（`status=missing_entry_thesis_fields`）。
6. 収益回復の優先順:
   - B/Cは方向・手仕舞いロジックを再調整し、`STOP_LOSS_ORDER` 依存を減らす。

Verification:
1. `orders.db` で `strategy_control_exit_disabled` が 24h連続 `0`。
2. `trades.db` で `MARKET_ORDER_MARGIN_CLOSEOUT` が 7日連続 `0`。
3. `metrics.db` で `data_lag_ms p95 < 1500`, `decision_latency_ms p95 < 2000` を継続達成。
4. `orders.db` で `slo_block` の連続発火（同一時間帯クラスタ）が解消。
5. `trades.db` で `entry_thesis` 必須2項目の欠損が `0`。
6. B/Cの 24h `jpy` と `PF` が改善方向（少なくとも連続悪化が停止）。

Status:
- in_progress

## 2026-02-26 12:02 UTC / 2026-02-26 21:02 JST - B/C sell限定運用で `units_below_min` が主因化し、発注ゼロ化したため最小ロット閾値を緩和
Period:
- Snapshot: `2026-02-26 11:59:57` ～ `12:01:39` UTC（`20:59:57` ～ `21:01:39` JST）
- Source: VM `journalctl -u quant-scalp-ping-5s-{b,c}.service`, `/home/tossaki/QuantRabbit/logs/orders.db`, `/home/tossaki/QuantRabbit/ops/env/scalp_ping_5s_{b,c}.env`

Fact:
- `quant-scalp-ping-5s-b.service` / `quant-scalp-ping-5s-c.service` は `11:59:57 UTC` に再起動済み（active/running）。
- しかし `orders.db` は `ts >= 2026-02-26T11:59:57Z` で `0件`（新規発注フロー未到達）。
- ワーカーログで共通して以下を確認:
  - `SCALP_PING_5S_{B,C}_SIDE_FILTER=sell` により long 候補が `side_filter_block`。
  - short 候補は `units_below_min` が発生（B: `10件`, C: `8件`）。
  - `RISK multiplier` は `mult=0.40` で推移し、縮小後ユニットが閾値未満へ落ちやすい。

Failure Cause:
1. long を遮断する side filter 自体は意図どおりだが、short 側の最終ユニットが `min_units=30` を下回り、注文生成まで到達できない。
2. strategy 側 min units と order 側 min units の閾値差（Cは 30 固定）が、縮小運転時の失効を助長している。

Improvement:
1. `ops/env/scalp_ping_5s_b.env`
   - `SCALP_PING_5S_B_MIN_UNITS: 30 -> 20`
2. `ops/env/scalp_ping_5s_c.env`
   - `SCALP_PING_5S_C_MIN_UNITS: 30 -> 20`
   - `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_C_LIVE: 30 -> 20`
   - `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_C: 30 -> 20`

Verification:
1. 反映後30分で `orders.db` に `submit_attempt` / `filled` が B/C で再出現すること。
2. `journalctl -u quant-scalp-ping-5s-{b,c}.service` の `units_below_min` 件数が減少すること。
3. `orders.db` で `manual_margin_pressure` / `slo_block` の再悪化がないこと。

Status:
- in_progress

## 2026-02-26 11:50 UTC / 2026-02-26 20:50 JST - PDCA導線の実運用監査（稼働中だが改善ループに断点あり）
Period:
- Snapshot: `2026-02-26 11:39` ～ `11:50` UTC（`20:39` ～ `20:50` JST）
- Source: VM `systemd`, `journalctl`, `/home/tossaki/QuantRabbit/logs/{orders,trades,metrics}.db`, `/home/tossaki/QuantRabbit/logs/*_latest.json`, OANDA account summary/open positions

Fact:
- V2主導線は稼働中:
  - `quant-market-data-feed`, `quant-strategy-control`, `quant-order-manager`, `quant-position-manager`, `quant-forecast` は `active(running)`。
  - `quantrabbit.service` は存在せず、VMリポジトリは `HEAD == origin/main == 0c0caae2c05295cbccd7113454fb24cb7f8afda3`。
- 分析/改善タイマーは起動:
  - `quant-pattern-book`, `quant-dynamic-alloc`, `quant-policy-guard`, `quant-replay-quality-gate`, `quant-trade-counterfactual`, `quant-forecast-improvement-audit` が schedule され実行履歴あり。
- ただし改善ループは市場オープン中に停止:
  - `quant-replay-quality-gate.service` は `skipped: market_open` を連続出力。
  - `quant-trade-counterfactual.service` も `skipped: market_open` を連続出力。
- 発注導線の品質劣化:
  - `quant-order-manager` 直近1時間で `database is locked` が `67` 回。
- 24hの orders 状態:
  - `preflight_start=18629`, `submit_attempt=1221`, `filled=1202`, `perf_block=16971`, `entry_probability_reject=1101`。
- 監査ログ差分:
  - `logs/ops_v2_audit_latest.json` で `POLICY_HEURISTIC_PERF_BLOCK_ENABLED expected=0 actual=1` の `warn`。

Failure Cause:
1. `replay` と `counterfactual` が market open 時に止まるため、改善施策の入力が日中に更新されない。
2. replay quality gate の対象が `TrendMA/BB_RSI` 中心で、現行の主要 scalp/micro 群を十分に監査できていない。
3. auto-improve が `block_jst_hours` を `worker_reentry.yaml` へ自動反映する設定で、方針「停止より改善優先」と衝突しやすい。
4. `orders.db` lock 競合が残り、preflight->submit の実効通過率を毀損している。
5. pattern gate の実効キー (`ORDER_MANAGER_PATTERN_GATE_ENABLED`) と運用キー (`ORDER_PATTERN_GATE_ENABLED`) が分岐し、設定意図と実動作がずれる余地がある。

Improvement:
1. `orders.db` 競合の再抑制を最優先（busy timeout/retry/lock制御と write 経路の再点検）。
2. `REPLAY_QUALITY_GATE_SKIP_WHEN_MARKET_OPEN=0` と `COUNTERFACTUAL_SKIP_WHEN_MARKET_OPEN=0` へ変更し、改善ループを 24/7 化。
3. `config/replay_quality_gate_main.yaml` の `workers` を現行稼働戦略へ拡張し、PDCA対象を本番導線へ合わせる。
4. `REPLAY_QUALITY_GATE_AUTO_IMPROVE_APPLY_REENTRY=0` にして、自動時間帯ブロック反映を停止（提案のみ運用）。
5. pattern gate の env キーを一本化し、order-manager 側の参照名に統一。
6. `POLICY_HEURISTIC_PERF_BLOCK_ENABLED` の期待値と実運用値を監査基準に合わせる。

Verification:
1. `journalctl -u quant-order-manager` の `database is locked` が 1h あたり 0 件へ収束。
2. `replay_quality_gate_latest.json` と `trade_counterfactual_latest.json` の `generated_at` が市場オープン中も更新される。
3. replay report の対象戦略に、稼働中 scalp/micro 群が含まれる。
4. `worker_reentry.yaml` への自動 `block_jst_hours` 書き換えが停止し、改善提案だけが残る。
5. `orders.db` の `preflight_start -> submit_attempt -> filled` の通過率が改善する。

Status:
- in_progress

## 2026-02-26 10:12 UTC / 2026-02-26 19:12 JST - 停止なし・時間帯停止なし運用へ再構成
Period:
- Snapshot: `2026-02-26 09:52` ～ `10:12` UTC（`18:52` ～ `19:12` JST）
- Source: VM `/home/tossaki/QuantRabbit/logs/trades.db`, `/home/tossaki/QuantRabbit/logs/orders.db`, `/home/tossaki/QuantRabbit/logs/strategy_control.db`, `config/worker_reentry.yaml`, OANDA `openTrades`

Fact:
- 24h は `1120 trades`, `-11510.6 JPY`, `PF=0.455`（主損失: `scalp_ping_5s_c_live`, `scalp_ping_5s_b_live`, `M1Scalper-M1`）。
- 直近監査で `entry=1 & exit=0` は `0` 件、`strategy_control_exit_disabled` も直近1hで `0` 件。
- ただし運用設定には「停止相当」の制約が残存:
  - `scalp_ping_5s_{b,c}` の `ALLOW_HOURS_JST`（時間帯限定）
  - `worker_reentry` の `block_jst_hours`（`M1Scalper`, `MicroPullbackEMA`, `MicroLevelReactor`, `scalp_ping_5s_{b,c,d}_live`）

Failure Cause:
1. 収益改善を「停止/時間帯限定」に寄せると、運用方針（常時動的トレード）と矛盾し、再現性が崩れる。
2. `scalp_ping_5s_c_live` は `close_reject_no_negative` が残ると損失玉の解放が遅れやすい。

Improvement:
1. `ops/env/scalp_ping_5s_{b,c}.env`:
   - `ALLOW_HOURS_JST=`（時間帯停止を撤去）
   - `PERF_GUARD_MODE=reduce`（停止ではなく縮小）
2. `config/worker_reentry.yaml`:
   - `M1Scalper`, `MicroPullbackEMA`, `MicroLevelReactor`, `scalp_ping_5s_{b,c,d}_live` の `block_jst_hours` を空配列化。
3. `config/strategy_exit_protections.yaml`:
   - `scalp_ping_5s_{c,c_live}.neg_exit` に `strict_no_negative: false`, `deny_reasons: []` を追加し、EXIT詰まりを解消。
4. `strategy_control_flags` は `entry=1 & exit=1` を基本状態として維持し、停止は緊急安全時のみ許可。

Verification:
1. `orders.db` で `strategy_control_entry_disabled` / `strategy_control_exit_disabled` が発生しないこと。
2. `orders.db` で `close_reject_no_negative` が `scalp_ping_5s_c_live` で減少すること。
3. 24hで `B/C` の `jpy PF` が改善方向（`>=1.0` へ接近）すること。
4. `worker_reentry` 由来の時間帯ブロック理由で待機しないこと。

Status:
- in_progress

## 2026-02-26 11:50 UTC / 2026-02-26 20:50 JST - `slo_block(data_lag_p95_exceeded)` を緩和し、M1配分を引き上げ
Period:
- 30m: `datetime(ts) >= now - 30 minutes`
- 24h: `datetime(close_time) >= now - 24 hours`
- Source: VM `/home/tossaki/QuantRabbit/logs/orders.db`, `/home/tossaki/QuantRabbit/logs/trades.db`

Fact:
- 30m `orders.db`:
  - `preflight_start=110`, `probability_scaled=33`
  - `slo_block=10`（latest `2026-02-26T11:45:09Z`）
  - `manual_margin_pressure=25`（ただし latest `2026-02-26T11:34:06Z` で新規発生停止）
- `slo_block` の reason は全件 `data_lag_p95_exceeded`。
  - `data_lag_p95_ms` は `~7152ms`、現行閾値 `5000ms` を超過。
- 24h損益:
  - `M1Scalper-M1` は直近で `-3.1 JPY`（ほぼ横ばい、下振れ縮小）
  - B/C は side_filter=sell 反映後、B/C の `buy` 注文は 0 件。

Failure Cause:
1. strategy-control 側の data lag p95 スパイクが SLO 閾値を超え、scalp_fast が連続 reject。
2. M1 は止めずに残しているが、配分が低く利益寄与の伸びが不足。

Improvement:
1. `ops/env/quant-order-manager.env`
   - `ORDER_SLO_GUARD_DATA_LAG_P95_MAX_MS: 5000 -> 9000`
2. `ops/env/quant-v2-runtime.env`（worker local `order_manager` 経路）
   - `ORDER_SLO_GUARD_*` を明示追加し、`ORDER_SLO_GUARD_DATA_LAG_P95_MAX_MS=9000`
3. `ops/env/quant-m1scalper.env`
   - `M1SCALP_BASE_UNITS: 3000 -> 4500`

Verification:
1. 反映後 30m で `slo_block` 最新時刻が更新されないこと（再発停止）。
2. `orders.db` の `preflight_start -> probability_scaled/filled` の遷移比率が改善すること。
3. `M1Scalper-M1` の 6h / 24h `realized_pl` が正方向へ改善すること。

Status:
- in_progress

## 2026-02-26 11:45 UTC / 2026-02-26 20:45 JST - B/C の side 偏りを実損で是正（long 逆風遮断）
Period:
- 24h: `datetime(close_time) >= now - 24 hours`
- Source: VM `/home/tossaki/QuantRabbit/logs/trades.db`

Fact:
- `scalp_ping_5s_c_live`:
  - long `347 trades / -5380.5 JPY / -353.4 pips`
  - short `75 trades / -88.8 JPY / -52.2 pips`
- `scalp_ping_5s_b_live`:
  - long `128 trades / -1300.8 JPY / -141.3 pips`
  - short `26 trades / -7.7 JPY / -0.5 pips`
- 直近損失の主因は B/C ともに long 側へ偏在。

Failure Cause:
1. no-stop 継続を優先して `SIDE_FILTER` を空に戻したことで、逆風側（long）の負けが継続拡大。
2. B/C の短期ローカル判定は機能しているが、方向バイアスの切替が遅く実損に追随できていない。

Improvement:
1. `ops/env/scalp_ping_5s_b.env`
   - `SCALP_PING_5S_B_SIDE_FILTER=sell`
2. `ops/env/scalp_ping_5s_c.env`
   - `SCALP_PING_5S_C_SIDE_FILTER=sell`

Verification:
1. 反映後 30m で `trades.db` の B/C long 新規クローズ件数が 0 であること。
2. B/C の short 側 `avg_lose_jpy` と `total_jpy` が long 全開時より改善すること。
3. `orders.db` の `preflight_start -> probability_scaled/filled` の遷移を継続確認し、`perf_block` 再発がないこと。

Status:
- in_progress

## 2026-02-26 11:30 UTC / 2026-02-26 20:30 JST - no-stop阻害点を `perf_block` + `manual_margin_pressure` に限定して除去
Period:
- 15m: `datetime(ts) >= now - 15 minutes`
- Source: VM `/home/tossaki/QuantRabbit/logs/orders.db`, live account snapshot via `execution.order_manager.get_account_snapshot()`

Fact:
- 直近15分の status 集計:
  - `preflight_start=68`
  - `perf_block=52`
  - `probability_scaled=33`
  - `manual_margin_pressure=8`
  - `entry_probability_reject=4`
  - `slo_block=1`
- strategy別（同窓）:
  - `M1Scalper-M1 | perf_block=25`
  - `scalp_ping_5s_b_live | perf_block=24`
  - `scalp_ping_5s_b_live | manual_margin_pressure=10`
- 口座スナップショット:
  - `nav=57,556.799`, `margin_used=53,037.280`, `margin_available=4,553.519`, `health_buffer=0.07907`
  - manual 建玉: `-8500 units`, `1 trade`

Failure Cause:
1. no-stop向けに failfast を緩めても、`perf_guard` の hard reason 判定で B/M1 が preflight reject を継続。
2. manual 併走時の `manual_margin_guard` が小ロット再開局面でも `manual_margin_pressure` を発火させ、B の通過を削る。

Improvement:
1. `ops/env/quant-order-manager.env`:
   - `ORDER_MANUAL_MARGIN_GUARD_MIN_FREE_RATIO=0.00`
   - `ORDER_MANUAL_MARGIN_GUARD_MIN_HEALTH_BUFFER=0.00`
   - `ORDER_MANUAL_MARGIN_GUARD_MIN_AVAILABLE_JPY=0`
   - `SCALP_PING_5S_B_PERF_GUARD_ENABLED=0`
   - `M1SCALP_PERF_GUARD_ENABLED=0`
2. `ops/env/quant-v2-runtime.env`（worker local `order_manager` 経路へ実効反映）:
   - `ORDER_MANUAL_MARGIN_GUARD_MIN_FREE_RATIO=0.00`
   - `ORDER_MANUAL_MARGIN_GUARD_MIN_HEALTH_BUFFER=0.00`
   - `ORDER_MANUAL_MARGIN_GUARD_MIN_AVAILABLE_JPY=0`
   - `SCALP_PING_5S_B_PERF_GUARD_ENABLED=0`
   - `M1SCALP_PERF_GUARD_ENABLED=0`
3. 戦略env側も整合:
   - `ops/env/scalp_ping_5s_b.env`: `SCALP_PING_5S_B_PERF_GUARD_ENABLED=0`
   - `ops/env/quant-m1scalper.env`: `M1SCALP_PERF_GUARD_ENABLED=0`

Verification:
1. `orders.db` 15分窓で `status in ('perf_block','manual_margin_pressure')` が B/M1 で減少すること。
2. 同窓で `filled` と `probability_scaled` の増加が確認できること。
3. `MARKET_ORDER_MARGIN_CLOSEOUT` が増えないこと（24h監査継続）。

Status:
- in_progress

## 2026-02-26 11:02 UTC / 2026-02-26 20:02 JST - `manual_margin_pressure` が B エントリー再開の最終ボトルネック
Period:
- Source: VM `/home/tossaki/QuantRabbit/logs/orders.db`, OANDA account snapshot/openTrades
- Window: `datetime(ts) >= now - 15 minutes`（UTC）

Fact:
- 全体（15分）: `entry_probability_reject=13`, `preflight_start=7`, `probability_scaled=3`, `manual_margin_pressure=3`, `perf_block=1`
- `scalp_ping_5s_b_live`（15分）: `preflight_start=6`, `probability_scaled=3`, `manual_margin_pressure=3`
- `manual_margin_pressure` 3件はすべて B の `scalp_fast` エントリー（`units=139/140/181`）
- 口座実測（2026-02-26 11:01 UTC）:
  - `NAV=57,930.80`, `margin_used=53,022.32`, `margin_available=4,942.48`, `health_buffer=0.0853`
  - open trade: `USD_JPY -8500`（`id=400470`, `TP/SLなし`）

Failure Cause:
1. failfast/forecast/leading-profile を緩和した後、B の最終拒否が `manual_margin_pressure` に収束。
2. guard閾値（`free_ratio>=0.05`, `health_buffer>=0.07`, `available>=3000`）が、手動玉併走時の小ロット意図まで遮断。
3. その結果、B の `probability_scaled` 後の通過意図が実発注に到達しない。

Improvement:
1. `ops/env/quant-order-manager.env` の manual margin guard を no-stop 方針向けに再調整:
   - `ORDER_MANUAL_MARGIN_GUARD_MIN_FREE_RATIO: 0.05 -> 0.01`
   - `ORDER_MANUAL_MARGIN_GUARD_MIN_HEALTH_BUFFER: 0.07 -> 0.02`
   - `ORDER_MANUAL_MARGIN_GUARD_MIN_AVAILABLE_JPY: 3000 -> 500`
2. guard 自体は維持（`ORDER_MANUAL_MARGIN_GUARD_ENABLED=1`）し、極端な near-closeout だけを継続遮断。

Verification:
1. `quant-order-manager` 再起動後に process env で3閾値の反映を確認する。
2. 反映後 15 分で `manual_margin_pressure` 件数が減少し、`submit_attempt/filled` が増えることを確認する。
3. `margin_usage_projected_cap` と `MARKET_ORDER_MARGIN_CLOSEOUT` が増加しないことを同時監視する。

Status:
- in_progress

## 2026-02-26 11:12 UTC / 2026-02-26 20:12 JST - no-stop 維持のまま「負け源圧縮 + 勝ち源増量」へ再配分
Period:
- Source: VM `/home/tossaki/QuantRabbit/logs/orders.db`, `/home/tossaki/QuantRabbit/logs/trades.db`
- Window:
  - 拒否分析: `datetime(ts) >= now - 30 minutes`
  - 損益分析: `datetime(close_time) >= now - 7 days`（補助で 6h）

Fact:
- 直近30分の拒否内訳:
  - `entry_probability_reject=21`（全件 `rangefader`）
  - `preflight_start=7`, `probability_scaled=3`, `manual_margin_pressure=3`, `perf_block=1`
- `rangefader` 拒否理由は `entry_probability_below_min_units` に収束。
  - 直近サンプルの `entry_probability` は約 `0.40`
  - 確率スケール後ユニットが pocket 最小ユニット未満で落ちる状態。
- 7日損益（strategy別、主なもの）:
  - `MomentumBurst`: `+1613.7 JPY`（n=7）
  - `MicroRangeBreak`: `+662.3 JPY`（n=32, PF=3.05）
  - `scalp_ping_5s_b_live`: `-9475.8 JPY`（n=2422, PF=0.43）
  - `scalp_ping_5s_c_live`: `-2735.5 JPY`（n=894, PF=0.86）
  - `M1Scalper-M1`: `-1627.3 JPY`（n=284, PF=0.64）
- 直近6hでも `scalp_ping_5s_c_live` は `-1859.9 JPY`（n=125）。

Failure Cause:
1. 発注経路は稼働しているが、`rangefader` が最小ユニット条件で連続 reject され、約定機会が失われた。
2. B/C と M1 が数量面で重く、no-stop 運用時に損失寄与が勝ち寄与を上回る配分になっていた。

Improvement:
1. `RangeFader` の通過回復（停止ではなく通過条件調整）:
   - `ORDER_MIN_UNITS_STRATEGY_SCALP_RANGEFAD=300` を `quant-order-manager` に追加。
2. 負け源の即時圧縮（B/C/M1 を継続稼働のまま減速）:
   - B: `BASE 1800->900`, `MAX 3600->1800`, `MAX_ORDERS_PER_MINUTE 6->4`, `CONF_FLOOR 74->78`
   - C: `BASE 400->220`, `MAX 900->500`, `MAX_ORDERS_PER_MINUTE 2->1`, `CONF_FLOOR 82->86`
   - M1: `BASE 10000->6000->3000`, `MAX_OPEN_TRADES 2->1`
   - B 追加ホットフィックス: `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_B(_LIVE) 30->20`
     （縮小後の確率スケールで `below_min_units` 連発したため）
   - B/M1 追加ホットフィックス: `hard:failfast` 連続拒否を避けるため
     `PERF_GUARD_FAILFAST_*` を soft 警告側へ再設定（Bは `FAILFAST_PF=0.10`, `HARD_PF=0.00`。
     M1は `MODE=reduce`, `FAILFAST_PF/WIN=0.30/0.35`, `HARD_PF=0.20`）。
3. 勝ち源の増量:
   - `MicroRangeBreak` と `MomentumBurstMicro` の `MICRO_MULTI_BASE_UNITS 42000->52000`
   - 上記2戦略の breakout 発火閾値を緩和:
     - `MIN_ADX 20.0->16.0`, `MIN_RANGE_SCORE 0.42->0.34`, `MIN_ATR 1.2->0.9`
     - `LOOP_INTERVAL_SEC 4.0->3.0`
   - `RangeFader` は過大化を避けるため `RANGEFADER_BASE_UNITS 13000->11000`

Verification:
1. 反映後30分で `rangefader` の `entry_probability_below_min_units` が減少し、`submit_attempt/filled` が発生すること。
2. 反映後2時間で B/C/M1 の `realized_pl` ドローダウン勾配が低下すること。
3. 同時に `MicroRangeBreak` / `MomentumBurst` の filled 数と `realized_pl` を増分監視すること。

Status:
- in_progress

## 2026-02-26 10:31 UTC / 2026-02-26 19:31 JST - Bがhard failfastで全面停止して約定不足
Period:
- `datetime(ts) >= now - 30 minutes`
- Source: VM `/home/tossaki/QuantRabbit/logs/orders.db`, `journalctl -u quant-order-manager.service`

Fact:
- 直近30分の `orders.db`:
  - `scalp_ping_5s_b_live`: `perf_block=32`
  - `scalp_ping_5s_c_live`: `perf_block=4`
  - `RangeFader-sell-fade`: `entry_probability_reject=23`
- `quant-order-manager` 拒否ログ（B）:
  - `perf_block:hard:hour10:failfast:pf=0.62 win=0.29 n=191` が連続発生
- 口座状態:
  - open trades はフラット（新規約定不足）

Failure Cause:
1. `SCALP_PING_5S_B_PERF_GUARD_FAILFAST_PF/WIN` が現状実績より高く、`reduce` 設定でも hard 理由で新規を全面拒否。
2. `SCALP_PING_5S_B_SIDE_FILTER=buy` / `SCALP_PING_5S_C_SIDE_FILTER=buy` で方向固定が残り、相場方向とのズレ時に機会損失が拡大。

Improvement:
1. Bの hard failfast 閾値を実績連動に下げる:
   - `quant-order-manager` 内では `ORDER_MANAGER_SERVICE_ENABLED=0` に固定（自己HTTP再入を停止）
   - `SCALP_PING_5S_B_PERF_GUARD_FAILFAST_PF: 0.88 -> 0.58`
   - `SCALP_PING_5S_B_PERF_GUARD_FAILFAST_WIN: 0.48 -> 0.27`
   - `SCALP_PING_5S_B_PERF_GUARD_SL_LOSS_RATE_MAX: 0.52 -> 0.75`
   - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER(B/C): 0.64/0.76 -> 0.48/0.62`
   - 反映先: `ops/env/quant-order-manager.env`, `ops/env/scalp_ping_5s_b.env`, `ops/env/scalp_ping_5s_c.env`
2. B/C の方向固定を解除:
   - `SCALP_PING_5S_B_SIDE_FILTER=`
   - `SCALP_PING_5S_C_SIDE_FILTER=`
   - 反映先: `ops/env/scalp_ping_5s_b.env`, `ops/env/scalp_ping_5s_c.env`
3. strategy_entry の forecast 逆行一律拒否を解除:
   - `STRATEGY_FORECAST_FUSION_STRONG_CONTRA_REJECT_ENABLED=0`
   - `STRATEGY_FORECAST_FUSION_WEAK_CONTRA_REJECT_ENABLED=0`
   - 反映先: `ops/env/quant-v2-runtime.env`
4. B/C の strategy_entry leading profile 拒否を解除:
   - `SCALP_PING_5S_B_ENTRY_LEADING_PROFILE_ENABLED=0`
   - `SCALP_PING_5S_C_ENTRY_LEADING_PROFILE_ENABLED=0`
   - 反映先: `ops/env/scalp_ping_5s_b.env`, `ops/env/scalp_ping_5s_c.env`

Verification:
1. `orders.db` 30分窓で `scalp_ping_5s_b_live` の `perf_block` 比率が低下すること。
2. 同窓で `submit_attempt` / `filled` が再出現すること。
3. `trades.db` の当日 `scalp_ping_5s_b_live` / `scalp_ping_5s_c_live` 実現JPYが改善方向に転じること。

Status:
- in_progress

## 2026-02-26 10:06 UTC / 2026-02-26 19:06 JST - 即日止血（C停止 + strategy_control 衝突解消）
Period:
- Snapshot: `2026-02-26 09:52` ～ `10:06` UTC（`18:52` ～ `19:06` JST）
- Source: VM `/home/tossaki/QuantRabbit/logs/trades.db`, `/home/tossaki/QuantRabbit/logs/orders.db`, `/home/tossaki/QuantRabbit/logs/strategy_control.db`, OANDA `openTrades`

Fact:
- 24h: `1120 trades`, `-1009.3 pips`, `-11510.6 JPY`, `PF=0.455`
- 7d: `4189 trades`, `-2127.9 pips`, `-25067.5 JPY`, `PF=0.656`
- 24h 主損失は `scalp_ping_5s_c_live=-5633.2 JPY`, `scalp_ping_5s_b_live=-3102.6 JPY`, `M1Scalper-M1=-1265.8 JPY`
- `entry=1 & exit=0` の残骸は是正前に複数残存（`micropullbackema*`, `microtrendretest*`, `scalp_ping_5s_flow_live`）
- `openTradeCount=0`（OANDA、手動玉含めフラット）

Failure Cause:
1. C/B が負EVのまま高回転し、日次損失の主因を継続している。
2. `strategy_control` の stale flag（`entry=1 & exit=0`）が再発時の closeout テール要因になりうる。
3. 直近は `close_reject_no_negative` も C 側で多発し、EXIT遅延を誘発していた。

Improvement:
1. VM運用ガードを即時適用:
   - `strategy_control_flags` の `entry=1 & exit=0` を全解消（`entry=0 & exit=1` に補正）。
2. 即日止血:
   - `scalp_ping_5s_c` を `entry=0 & exit=1` として新規停止（EXITは許可）。
3. 収益機会の残し方:
   - `microtrendretest` / `microtrendretest-long` は `entry=1 & exit=1` へ再有効化。
4. 高リスク再発系は停止維持:
   - `micropullbackema*` と `scalp_ping_5s_flow*` は `entry=0 & exit=1` を維持。

Verification:
1. `strategy_control.db` 現在値: `entry=1 & exit=0` が `0` 件。
2. 直近1h: `orders.db` の `strategy_control_exit_disabled=0`、`close_reject_no_negative=0`。
3. OANDA `openTrades=[]` を確認（含み損ポジション持越しなし）。

Status:
- in_progress

## 2026-02-26 09:36 UTC / 2026-02-26 18:36 JST - 改善策統合プラン（P0/P1/P2）
Period:
- Synthesis window: `2026-02-24` ～ `2026-02-26`（既存3エントリ統合）
- Source: VM `/home/tossaki/QuantRabbit/logs/trades.db`, `/home/tossaki/QuantRabbit/logs/orders.db`, `/home/tossaki/QuantRabbit/logs/strategy_control.db`, `/home/tossaki/QuantRabbit/logs/metrics.db`, OANDA account/openTrades

Fact:
- 24h: `1211 trades`, `-5155.7 JPY`, `PF=0.773`
- 7d: `4277 trades`, `-25750.3 JPY`, `PF=0.650`
- 7d 主損失は `MARKET_ORDER_MARGIN_CLOSEOUT=-19124.7 JPY` と `STOP_LOSS_ORDER=-18393.1 JPY`
- `strategy_control_exit_disabled=10277`（`2026-02-24` 単日）
- 24h 主損失戦略: `scalp_ping_5s_c_live=-6094.2 JPY`, `scalp_ping_5s_b_live=-3141.3 JPY`
- core unit churn: `quant-order-manager starts=62`, `quant-market-data-feed starts=22`, `quant-scalp-ping-5s-c starts=30`（24h）

Failure Cause:
1. `entry=1 & exit=0` と broker `stopLossOnFill` 欠損が重なると、EXIT封鎖から closeout テールへ直結する。
2. `scalp_ping_5s_b/c` は勝率寄り運用で、`avg_win/avg_loss` と `jpy PF` が負のまま件数が先行している。
3. 手動玉込み余力監視と 1-trade loss cap の運用が弱く、大損1発の耐性が不足している。
4. core unit の再起動多発で、執行品質（reject/timeout/latency）の再現性が落ちる。

Improvement:
1. P0（当日）安全不変条件:
   - `entry=1 & exit=0` を自動補正で禁止（検知時は `entry=0` へ強制 + alert）。
   - 非manual新規注文は broker `stopLossOnFill` 必須化（欠損は `missing_broker_sl` reject）。
   - 手動玉込み余力で新規抑制を先行適用し、`MARKET_ORDER_MARGIN_CLOSEOUT` を即時遮断。
2. P1（24-48h）収益反転:
   - `scalp_ping_5s_c_live` は停止ではなく「低頻度・低サイズ運用」に固定し、`jpy PF>=1` まで上限解除しない。
   - `scalp_ping_5s_b_live` も同一KPI（`jpy PF`, `avg_win/avg_loss`, `avg_loss_jpy`）でロット再配分。
   - `dynamic_alloc` は pips優先を避け、`realized_jpy_per_1k_units` と `jpy_downside_share` 主導に固定。
3. P1（24-48h）EXIT詰まり解消:
   - `close_reject_no_negative` を strategy tag × exit_reason で棚卸しし、拒否条件の過剰部分を除去。
   - 負け玉の平均保有時間を短縮するため、reject系ステータスの即時監査を追加。
4. P2（72h）運用品質:
   - core unit ごとの restart budget を設定し、閾値超過時は新規改善より先に安定化を実施。
   - 日次で `orders/trades/strategy_control` を定型監査し、本台帳へ同フォーマット追記。

Verification:
1. `strategy_control.db` で `entry=1 & exit=0` が 0 件、`orders.db` で `strategy_control_exit_disabled` が 24h 連続 0 件。
2. 非manual filled の broker `stopLossOnFill` 設定率 `>= 99%`。
3. `MARKET_ORDER_MARGIN_CLOSEOUT` が 7日連続で件数 0 / JPY損失 0。
4. `scalp_ping_5s_b/c` の 24h `jpy PF >= 1.00` かつ `avg_win/abs(avg_loss) >= 1.00`。
5. core unit の start/stop が各 24h で `<= 5`。
6. 全体 `all_24h JPY` が 3日連続でプラス。

Status:
- in_progress

## 2026-02-26 09:25 UTC / 2026-02-26 18:25 JST - 回転点まとめ（資産曲線を反転させる条件）
Period:
- 24h/7d の実績再集計: `datetime(close_time) >= now - 1 day / 7 day`
- Exit封鎖イベント確認: `2026-02-24`（`orders.db` / `strategy_control.db`）
- Source: VM `/home/tossaki/QuantRabbit/logs/trades.db`, `/home/tossaki/QuantRabbit/logs/orders.db`, `/home/tossaki/QuantRabbit/logs/strategy_control.db`, `/home/tossaki/QuantRabbit/logs/metrics.db`, OANDA account summary

Fact:
- 24h 合計: `1211 trades`, `+525.6 pips`, `-5155.7 JPY`, `PF=0.773`
- 7d 合計: `4277 trades`, `-2287.7 pips`, `-25750.3 JPY`, `PF=0.650`
- 24h の主損失戦略:
  - `scalp_ping_5s_c_live`: `545 trades`, `-6094.2 JPY`（勝率 `53.0%` でも `avg_win=+20.5 JPY < avg_loss=-48.1 JPY`）
  - `scalp_ping_5s_b_live`: `480 trades`, `-3141.3 JPY`
- 24h の反事実:
  - `all_24h = -5155.7 JPY`
  - `exclude C = +938.5 JPY`
  - `exclude B/C = +4079.7 JPY`
  - `exclude margin_closeout = -2991.9 JPY`
- 7d の損失内訳:
  - `MARKET_ORDER_MARGIN_CLOSEOUT=-19124.7 JPY`（負け総額の `51.0%`）
  - `STOP_LOSS_ORDER=-18393.1 JPY`（同 `49.0%`）
- `2026-02-24` 単日に `strategy_control_exit_disabled=10277`。同日 `micropullbackema` で `margin closeout 4件 = -16837.4 JPY`。
- 現在も `strategy_control.db` に `exit_enabled=0` が残存:
  - `micropullbackema`, `micropullbackema-short`, `microtrendretest`, `microtrendretest-long`, `scalp_ping_5s_flow_live`
- 24h の unit churn（journal）:
  - `quant-order-manager starts=62 / stops=30`
  - `quant-market-data-feed starts=22 / stops=20`
  - `quant-scalp-ping-5s-c starts=30 / stops=28`

Failure Cause:
1. C/B の期待値が負のまま件数が多く、日次損失のベースを作っている。
2. Exit封鎖（`entry=1 & exit=0`）が実ポジに刺さると、損失玉が閉じられず closeout テールが発生する。
3. 再起動多発で order/position 連携が不安定化し、執行品質とリスク制御の再現性が落ちる。
4. `perf_block` と `probability_scaled` が大きく、良い局面の約定密度が不足する。

Improvement:
1. **回転点A（資産反転の最低条件）**: `entry=1 & exit=0` を本番で禁止（自動修正 + アラート）。
2. **回転点B（収益反転の主条件）**: `scalp_ping_5s_c_live` の EV を 0 以上にするまで、件数/サイズを強制的に抑える。
3. **回転点C（ドローダウン抑制）**: `MARKET_ORDER_MARGIN_CLOSEOUT` を連続 0 件に固定（manual玉込み余力ガード）。
4. **回転点D（実運用品質）**: core unit の restart/stop を抑制し、連続運転で検証する。

Verification:
1. `orders.db` で `strategy_control_exit_disabled` が 24h 連続で 0 件。
2. `trades.db` で `MARKET_ORDER_MARGIN_CLOSEOUT` が 7日連続 0 件。
3. `scalp_ping_5s_c_live` の 24h `ev_jpy >= 0` かつ `avg_loss_jpy` の絶対値縮小。
4. 日次反事実で `exclude C` が不要な状態（実績 `all_24h` が単独でプラス）へ移行。

Status:
- in_progress

## 2026-02-26 12:58 UTC / 2026-02-26 21:58 JST - `stage_state.db` ロックで market_order が失敗しエントリー停止
Period:
- Window: `2026-02-26 11:42` ～ `12:56` UTC
- Source: VM `journalctl -u quant-order-manager.service`, `logs/orders.db`, `logs/trades.db`

Fact:
- `quant-order-manager` は active だが、`[ORDER_MANAGER_WORKER] request failed: database is locked` が連続発生。
- `orders.db` 直近2h: `preflight_start=237`, `probability_scaled=99`, `perf_block=78`, `entry_probability_reject=19`, `slo_block=11`。
- `trades.db` 直近2h の新規 `entry_time` は `0` 件、open trades も `0` 件。
- 失敗直前ログは `OPEN_SCALE` / `projected margin scale` まで進み、その後に `database is locked` で abort している。

Failure Cause:
1. `execution/strategy_guard.py` が `logs/stage_state.db` に対してロック耐性なし（busy_timeout/retryなし）でアクセス。
2. 同DBを `stage_tracker` 系と共有するため、短時間の書き込み競合で `sqlite OperationalError: database is locked` が発生。
3. 例外が `market_order` 経路まで伝播し、preflight通過後でも発注まで到達しない。

Improvement:
1. `strategy_guard` に busy_timeout + WAL + lock retry を追加。
2. lock時は fail-open（`is_blocked=False`）で返し、エントリーを DB 競合で止めない。
3. `set_block` / `clear_expired` / expired削除も lock耐性化して例外伝播を防止。

Verification:
1. `pytest -q tests/test_stage_tracker.py` pass（3 passed）。
2. デプロイ後、`journalctl -u quant-order-manager.service` で `request failed: database is locked` が再発しないこと。
3. `orders.db` で `preflight_start` のみ増える状態が解消し、`filled` が復帰すること。

Status:
- in_progress

## 2026-02-26 13:10 UTC / 2026-02-26 22:10 JST - B/C エントリー枯渇に対する no-signal 緩和（revert依存解除 + side filter開放）
Period:
- Window: `2026-02-26 13:08` ～ `13:10` UTC
- Source: VM `journalctl -u quant-scalp-ping-5s-b.service`, `quant-scalp-ping-5s-c.service`

Fact:
- `quant-order-manager` の lockエラーは再発していないが、B/C ワーカーは `entry-skip` が継続。
- 主因は `no_signal:revert_not_found` と `units_below_min`（B/C とも short 側で 8〜15件/集計周期）。
- `side_filter_block` も残り、sell固定のままでは通過率が伸びにくい。

Improvement:
1. `ops/env/scalp_ping_5s_b.env`
   - `SIDE_FILTER=none`, `REVERT_ENABLED=0`
   - `MAX_ORDERS_PER_MINUTE=6`, `BASE_ENTRY_UNITS=900`
   - `MIN_UNITS_RESCUE_MIN_ENTRY_PROBABILITY=0.45`, `MIN_UNITS_RESCUE_MIN_CONFIDENCE=65`
   - `CONF_FLOOR=72`
2. `ops/env/scalp_ping_5s_c.env`
   - `SIDE_FILTER=none`, `ALLOW_NO_SIDE_FILTER=1`, `REVERT_ENABLED=0`
   - `MAX_ORDERS_PER_MINUTE=6`, `BASE_ENTRY_UNITS=260`
   - `MIN_UNITS_RESCUE_MIN_ENTRY_PROBABILITY=0.45`, `MIN_UNITS_RESCUE_MIN_CONFIDENCE=70`
   - `CONF_FLOOR=74`

Verification:
1. デプロイ後 10〜20 分で `orders.db` の `preflight_start` だけでなく `filled` が増えること。
2. `entry-skip summary` の `revert_not_found` 比率が低下すること。
3. `units_below_min` が `min_units_rescue` に置換され、0件へ収束すること。

Status:
- in_progress

## 2026-02-26 13:20 UTC / 2026-02-26 22:20 JST - B/C side filter を sell 再固定（精度優先）
Period:
- Analysis/retune window: `2026-02-26 13:10` ～ `13:20` UTC
- Source: repository `ops/env/scalp_ping_5s_b.env`, `ops/env/scalp_ping_5s_c.env`

Fact:
- 最新 `main` では B/C とも `SIDE_FILTER=none` + `ALLOW_NO_SIDE_FILTER=1` になっていた。
- 同時に `MIN_UNITS_RESCUE` は導入済みで、no-entry 側は rescue で緩和可能な状態だった。

Failure Cause:
1. side filter 開放により、方向精度劣化の再発リスク（buy 再流入）を残していた。

Improvement:
1. B/C を `SIDE_FILTER=sell`, `ALLOW_NO_SIDE_FILTER=0` へ再固定。
2. `MIN_UNITS_RESCUE` は維持しつつ閾値を再引き上げ
   - B: `prob>=0.58`, `conf>=78`
   - C: `prob>=0.60`, `conf>=82`

Verification:
1. デプロイ後ログで `env mapped ... side_filter=sell` を確認。
2. restart 後の B/C journal で `side_filter_fallback:long->short` が継続して出ることを確認。

Status:
- in_progress

## 2026-02-26 09:15 UTC / 2026-02-26 18:15 JST - EXIT封鎖とSL未実装の複合で margin closeout が連鎖
Period:
- Incident window: `2026-02-24 02:20:16` ～ `09:13:14` UTC（`2026-02-24 11:20:16` ～ `18:13:14` JST）
- P/L window: `datetime(close_time) >= now - 14 day`
- Source: VM `/home/tossaki/QuantRabbit/logs/strategy_control.db`, `/home/tossaki/QuantRabbit/logs/orders.db`, `/home/tossaki/QuantRabbit/logs/trades.db`, OANDA account/openTrades

Fact:
- `strategy_control_flags` に `entry=1 & exit=0` が残存（`note=manual_hold_current_positions_20260224`）:
  - `micropullbackema`, `micropullbackema-short`
  - `microtrendretest`, `microtrendretest-long`
  - `scalp_ping_5s_flow_live`
- `orders.db` の `strategy_control_exit_disabled` は合計 `10,277` 件。
  - 戦略別: `MicroPullbackEMA-short=8,177`, `MicroTrendRetest-long=2,068`, `scalp_ping_5s_flow_live=32`
- `MicroPullbackEMA` の margin closeout 4件（`384420/384425/384430/384435`）は、
  各 ticket で `strategy_control_exit_disabled` が `2,044～2,045` 回連続した後に強制クローズ。
  - 合計 `-16,837.4 JPY` / `-582.6 pips`
- 上記4件の filled 注文は `takeProfitOnFill` はある一方、`stopLossOnFill` が未設定（broker SLなし）。
- 14日合計: `-93,081.6 JPY`
  - `MARKET_ORDER_MARGIN_CLOSEOUT=-49,370.5 JPY`
  - 非closeoutでも `-43,711.0 JPY`
- 直近口座状態（2026-02-26 09:04 UTC）:
  - `NAV=57,731.21`, `margin_used=53,020.96`, `margin_available=4,744.25`
  - open trade に `TP/SL なし` の `USD_JPY -8500` が残存

Failure Cause:
1. `entry=1 & exit=0` の危険状態を作る運用が残り、保有調整を詰まらせた。
2. close経路が `strategy_control` により拒否され続け、損失玉が放置された。
3. broker側 `stopLossOnFill` 未設定の玉は、ローカルEXIT依存となり封鎖時に脆弱。
4. margin closeout を除いても `scalp_ping_5s_b/c` を中心に期待値が負で、資産減少が継続。

Improvement:
1. 運用ルール固定: `entry=1 & exit=0` を禁止し、保有維持が必要なときは `entry=0 & exit=1` のみ許可。
2. 自動ガード追加: `strategy_control` に `entry=1 & exit=0` が入った時点でアラート + 自動修正（`entry=0` へ強制）。
3. 発注必須化: 非manualの新規注文は `stopLossOnFill` 必須。欠損時は reject して `status=missing_broker_sl` を記録。
4. 既存残骸の解消: `manual_hold_current_positions_20260224` の stale flag を全解除し、原因ノート付きで再設定履歴を残す。
5. 平常時改善: `scalp_ping_5s_b/c` は `win_rate` ではなく `jpy PF` と `avg_win/avg_loss` を主指標に再設計。

Verification:
1. `strategy_control.db` で `entry=1 & exit=0` 行が 0 件。
2. `orders.db` の `strategy_control_exit_disabled` が 24h で 0 件。
3. 7日連続で `MARKET_ORDER_MARGIN_CLOSEOUT` の件数 0 / JPY損失 0。
4. 非manual filled の `stopLossOnFill` 設定率が `>= 95%`（戦略別に監査）。
5. `scalp_ping_5s_b/c` の14日 `jpy PF >= 1.0` かつ `avg_loss_jpy` の絶対値縮小を確認。

Status:
- in_progress

## 2026-02-26 08:40 UTC / 2026-02-26 17:40 JST - Margin Closeout Tail が資産毀損の主因
Period:
- 24h: `datetime(close_time) >= now - 1 day`
- 7d: `datetime(close_time) >= now - 7 day`
- Source: VM `/home/tossaki/QuantRabbit/logs/trades.db`, `/home/tossaki/QuantRabbit/logs/orders.db`

Fact:
- 24h 合計: `1213 trades`, `+555.4 pips`, `-4652 JPY`
- 7d 合計: `4283 trades`, `-2281.8 pips`, `-25242 JPY`
- 7d close reason:
  - `STOP_LOSS_ORDER`: `1752`, `-3980.5 pips`, `-18436 JPY`
  - `MARKET_ORDER_MARGIN_CLOSEOUT`: `17`, `-602.5 pips`, `-17415 JPY`
  - `MARKET_ORDER_TRADE_CLOSE`: `2217`, `+1805.6 pips`, `+315 JPY`
- 7d 戦略別（下位）:
  - `MicroPullbackEMA`: `46`, `-520.3 pips`, `-15527 JPY`（`margin_closeout 4件 = -16837 JPY`）
  - `scalp_ping_5s_c_live`: `855`, `-1006.7 pips`, `-10554 JPY`
  - `scalp_ping_5s_b_live`: `2093`, `-1879.5 pips`, `-9448 JPY`
- 口座スナップショット（VM実測）:
  - `nav=56813.21`, `margin_used=53057.68`, `margin_available=3789.53`, `health_buffer=0.06666`
  - open positions: `manual short -8500 units`
- 稼働確認:
  - `quant-scalp-ping-5s-b.service`: running, `SCALP_PING_5S_B_ENABLED=1`
  - `quant-scalp-ping-5s-c.service`: running, `SCALP_PING_5S_C_ENABLED=1`
  - `quant-scalp-ping-5s-d.service`: running, `SCALP_PING_5S_D_ENABLED=0`

Failure Cause:
1. `Margin closeout` を許す残余証拠金状態で稼働を継続し、少数の巨大損失が累積。
2. `scalp_ping_5s_c_live` は勝率が一定でもペイオフ負け（平均損失が平均利益を上回る）。
3. `close_reject_no_negative` が多発し、負け玉解放が遅れる経路が残る。
4. DD判定は存在するが、エントリー停止ガードへの直接連携が弱く、資産保全の最終防衛線になっていない。
5. 1トレード損失上限（loss cap）機能はあるが、運用値未設定で未活用。

Improvement:
1. 最優先は `margin closeout 回避`:
   - 余力逼迫時の新規抑制条件を先に適用（手動玉込みで判定）。
2. 次に `1トレード損失上限` を明示有効化:
   - `ORDER_ENTRY_LOSS_CAP_*` 系を strategy/pocket 単位で設定。
3. `close_reject_no_negative` の原因別棚卸し:
   - `exit_reason` と `neg_exit policy` の不整合を strategy tag 単位で解消。
4. `scalp_ping_5s_b/c` は勝率ではなく `avg_win/avg_loss` と `jpy PF` を第一指標に調整。
5. 24hごとに同一フォーマットで本台帳へ追記し、改善の効き/副作用を継続監査。

Verification:
1. `MARKET_ORDER_MARGIN_CLOSEOUT` の 24h 件数・JPY損失が連続減少。
2. 24h 合計で `pips` と `JPY` の符号乖離（`+pips / -JPY`）が解消。
3. `close_reject_no_negative` の件数が減少し、負け玉の平均保有時間が短縮。
4. `scalp_ping_5s_b/c` の `jpy PF` と `avg_loss_jpy` が改善。

Status:
- in_progress

## 2026-02-27 08:28 UTC / 2026-02-27 17:28 JST - `WickReversalBlend` の成功例へ寄せる閾値更新（entry probability + exit buffer）

Period:
- Analysis window: 24h / 7d / 14d（`trades.db`）
- Tick validate window: 2026-02-27 UTC（`USD_JPY_ticks_20260227.jsonl`）

Source:
- VM `/home/tossaki/QuantRabbit/logs/trades.db`
- VM `/home/tossaki/QuantRabbit/logs/orders.db`
- VM `/home/tossaki/QuantRabbit/logs/replay/USD_JPY/USD_JPY_ticks_20260227.jsonl`
- `~/.codex/skills/qr-tick-entry-validate/scripts/tick_entry_validate.py`

Fact:
- `WickReversalBlend` 24h は `16 trades / +403.5 JPY`、14d は `24 trades / +255.8 JPY`。
- `entry_probability` 閾値シミュレーション:
  - 現状相当（`>=0.70`）: `24h +403.5 JPY`, `14d +362.8 JPY`
  - `>=0.78`: `24h +497.8 JPY`, `14d +457.1 JPY`（件数は `16 -> 12` / `20 -> 16`）
  - `>=0.80`: `24h +490.9 JPY`, `14d +450.2 JPY`
- 24h の損失 3件はいずれも `MARKET_ORDER_TRADE_CLOSE` で、`hold_sec=349/446/794` と長期化。
- `orders.db` 実測:
  - `ticket 408599` で `close_reject_profit_buffer`（`est_pips=0.5 < min_profit_pips=0.6`）が発生後、`max_adverse` で `-322.0 JPY` へ拡大。
- Tick照合（同ticket）:
  - `sl_hit_s=132`、`hold_sec=794`、`MAE_300s=3.9 pips` と逆行継続。

Failure Cause:
1. `entry_probability 0.75-0.80` 帯の期待値が悪く、勝ちパターンへの集中度が不足。
2. `min_profit_pips=0.6` が lock系 close を弾き、`max_adverse` まで保持して損失が拡大。
3. `loss_cut_max_hold_sec=900` が長く、逆行ポジの解放が遅い。

Improvement:
1. `ops/env/quant-scalp-wick-reversal-blend.env`
   - `SCALP_PRECISION_ENTRY_LEADING_PROFILE_REJECT_BELOW=0.78`（from `0.46`）
2. `config/strategy_exit_protections.yaml`（`WickReversalBlend`）
   - `min_profit_pips: 0.45`（from `0.6`）
   - `loss_cut_max_hold_sec: 420`（from `900`）

Verification:
1. 反映後 2h/24h で `WickReversalBlend` の `entry_probability` 分布が `>=0.78` に収束すること。
2. `orders.db` の `close_reject_profit_buffer`（WickReversalBlend）が減少すること。
3. `trades.db` で `hold_sec>=420` かつ負けの件数が減ること。
4. `WickReversalBlend` の 24h `sum(realized_pl)` が改善または維持されること。

Status:
- in_progress

## 2026-02-27 09:20 UTC / 2026-02-27 18:20 JST - `scalp_ping_5s_b/c` の「SL過大・利幅不足・long lot圧縮」是正（VM実測ベース）

Period:
- 直近24h（`julianday(now, '-24 hours')`）

Source:
- VM `/home/tossaki/QuantRabbit/logs/trades.db`
- VM `/home/tossaki/QuantRabbit/logs/orders.db`

Fact:
- side別（24h）:
  - `long`: `755 trades / -416.9 pips / -730.1 JPY / avg_units=185.8`
  - `short`: `277 trades / -129.5 pips / +1448.6 JPY / avg_units=338.9`
- `scalp_ping_5s_b_live`:
  - `long`: `426 trades / -565.8 JPY / avg_win=1.165 pips / avg_loss=2.035 pips / sl_reason_rate=0.507`
  - `filled long` の実効距離: `avg_sl=2.03 pips`, `avg_tp=0.99 pips`, `tp/sl=0.49`
- `scalp_ping_5s_c_live`:
  - `long`: `266 trades / -117.8 JPY / avg_win=1.159 pips / avg_loss=1.857 pips / sl_reason_rate=0.444`
  - `filled long` の実効距離: `avg_sl=1.30 pips`, `avg_tp=0.90 pips`, `tp/sl=0.69`

Failure Cause:
1. B/C とも long 側で `avg_loss_pips > avg_win_pips` が継続し、RR が負け越し。
2. B/C の lot 圧縮（base units + preserve-intent max scale + leading profile units上限）が重なり、収益復元が遅延。
3. `TP_ENABLED=0` 運用下でも virtual target が短く、実効 `tp/sl` が 1.0 未満に張り付く局面が多い。

Improvement:
1. `ops/env/scalp_ping_5s_b.env`
   - long RR改善: `SL_BASE 1.6 -> 1.35`, `SL_MAX 2.4 -> 2.0`, `TP_BASE 0.35 -> 0.55`, `TP_MAX 1.4 -> 1.9`, `TP_NET_MIN 0.35 -> 0.45`
   - short維持: `SHORT_TP_BASE=0.35`, `SHORT_TP_MAX=1.4` を明示
   - lot復元: `BASE_ENTRY_UNITS 220 -> 260`, `MAX_UNITS 750 -> 900`
   - 圧縮緩和: `ORDER_MANAGER_PRESERVE_INTENT_MAX_SCALE 0.32 -> 0.42`, `ENTRY_LEADING_PROFILE_UNITS_MAX_MULT 0.80 -> 0.95`
2. `ops/env/scalp_ping_5s_c.env`
   - long RR改善: `SL_BASE 1.3 -> 1.15`, `SL_MIN=0.85`, `SL_MAX=1.9`, `TP_BASE 0.20 -> 0.45`, `TP_MAX 1.0 -> 1.5`, `TP_NET_MIN 0.25 -> 0.40`
   - short維持: `SHORT_TP_BASE=0.20`, `SHORT_TP_MAX=1.0`, `SHORT_SL_BASE=1.30`, `SHORT_SL_MIN=0.95`, `SHORT_SL_MAX=2.0`
   - lot復元: `BASE_ENTRY_UNITS 70 -> 95`, `MAX_UNITS 160 -> 220`
   - 圧縮緩和: `ORDER_MANAGER_PRESERVE_INTENT_MAX_SCALE 0.50 -> 0.62`, `ENTRY_LEADING_PROFILE_UNITS_MAX_MULT 0.75 -> 0.90`

Verification:
1. 反映後2h/24hで `scalp_ping_5s_b/c long` の `avg_loss_pips` が低下し、`tp/sl` が改善すること。
2. `orders.db` で `scalp_ping_5s_b/c` の `filled avg_units` が増加しつつ、`perf_block` と `rejected` が急増しないこと。
3. 24hで `long` 側 `sum(realized_pl)` が改善方向へ転じること。

Status:
- in_progress

## 2026-02-27 10:50 UTC / 2026-02-27 19:50 JST - `scalp_ping_5s_b/c` 反映後の無約定化を追加補正（VMログ）

Period:
- 反映後（`>= 2026-02-27T09:29:00+00:00`）

Source:
- VM `journalctl -u quant-scalp-ping-5s-b.service`
- VM `journalctl -u quant-scalp-ping-5s-c.service`
- VM `/home/tossaki/QuantRabbit/logs/orders.db`

Fact:
- 反映後ログ主因（直近300行）:
  - B: `no_signal:revert_not_found=278`, `rate_limited=123`
  - C: `no_signal:revert_not_found=289`, `rate_limited=92`
- 反映後注文は C long の小ロット約定が散発（`units=18`）で、Bの約定密度が戻っていない。

Failure Cause:
1. `revert` 判定が厳しすぎて signal 化前に落ちる。
2. `MAX_ORDERS_PER_MINUTE=4` でレート制限が先に効き、通過機会を失う。
3. long側は side-metrics と preserve-intent の下限が低く、最終unitsが縮み過ぎる。

Improvement:
1. B/C 共通で `MAX_ORDERS_PER_MINUTE` を `6` へ引き上げ。
2. B/C 共通で `REVERT_*` 閾値を小幅緩和（`MIN_TICKS=1`, `RANGE/SWEEP/BOUNCE` 最小幅を低減）。
3. B/C 共通で long圧縮下限を引き上げ（`SIDE_METRICS_MIN_MULT` と `ORDER_MANAGER_PRESERVE_INTENT_MIN_SCALE` を上方修正）。

Verification:
1. 反映後2hで `entry-skip` に占める `revert_not_found` と `rate_limited` の比率が低下すること。
2. `orders.db` で `scalp_ping_5s_b/c` の `filled` 件数と `avg_units(long)` が回復すること。
3. 24hで `scalp_ping_5s_b/c long` の `sum(realized_pl)` が改善方向へ向かうこと。

Status:
- in_progress
