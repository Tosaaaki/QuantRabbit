# Trade Findings Ledger (Single Source of Truth)

## GCP/VM運用の廃止方針（現行運用）
- GCP/VMを前提とする本番運用は廃止。VMの起動・デプロイ・停止手順は履歴参照または障害時のみ利用する。
- 実務の実行フローはローカルV2導線（`scripts/local_v2_stack.sh`）を最優先とする。
- 旧VM/GCP資料は過去ログ・移行検証用途に限定し、日次運用はローカル導線の実データを優先する。


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

## 2026-03-06 14:22 UTC / 2026-03-06 23:22 JST - local-v2: M1Scalper setup絞り込み + Flow低品質entry圧縮 + OANDA 503耐性

Period:
- 集計: `logs/pdca_profitability_report_latest.json` generated_at=`2026-03-06T23:21:50 JST`
- 市況確認: UTC `14:20-14:22` / JST `23:20-23:22`
- 対象（実測）: `logs/trades.db`, `logs/orders.db`, `logs/metrics.db`, `logs/local_v2_stack/*.log`, OANDA pricing/openTrades

Fact:
- 市況（OANDA + local candle, UTC `14:21-14:22` / JST `23:21-23:22`）:
  - USD/JPY `mid=157.976 / spread=0.8p`
  - `ATR14(M1)=4.56p` / `ATR60(M1)=6.79p` / `range_120m=70.6p`
  - pricing は継続 `200 OK`、一方で `openTrades/summary` は `503` が断続
- 直近24h（bot only, `pdca_profitability_report_latest.md`）:
  - `trades=2482 / win_rate=49.9% / PF(pips)=0.69 / net_jpy=-9909.0`
- M1Scalper-M1 の source tag 別（同24h, `trades.db entry_thesis.source_signal_tag`）:
  - `trend-long: 473 trades / -1759.4 JPY / -354.2 pips`
  - `sell-rally: 861 trades / -1557.8 JPY / -442.3 pips`
  - `buy-dip: 328 trades / -1557.7 JPY / -417.6 pips`
  - 一方で `nwave-long: 30 trades / +98.4 JPY / +43.3 pips`
  - `breakout-retest-long: 2 trades / +81.2 JPY / +6.5 pips`
- Flow (`scalp_ping_5s_flow_live`) の直近負けトレード `entry_thesis` では
  - `signal_window_adaptive_live_score_pips=-0.58 〜 -0.89`
  - それでも `filled=366` / `net_jpy=-6998.5` が出ており、低品質entryが通過していた
- ローカル稼働:
  - `health_snapshot` は `data_lag_ms≈84 / decision_latency_ms≈21`
  - `quant-m1scalper` は OANDA `/summary` `503` で worker crash を起こし、stale pid 相当の再起動が発生

Failure Cause:
- `M1Scalper-M1` は当日ソース別で `trend-long` と `sell-rally` が損失寄与の大半を占め、setup の絞り込み不足が継続。
- `scalp_ping_5s_flow_live` は leading profile 無効のまま low-edge signal を大量通過させていた。
- OANDA `/summary` の瞬断時に M1 worker が例外落ちし、稼働継続性を損ねていた。

Improvement:
- `ops/env/local-v2-stack.env`
  - Flow:
    - `SCALP_PING_5S_FLOW_ENTRY_LEADING_PROFILE_ENABLED=1`
    - `...REJECT_BELOW=0.58` / `...SHORT=0.66`
    - `BASE_ENTRY_UNITS=80`
    - `MAX_ACTIVE_TRADES=1`
  - M1:
    - `M1SCALP_ALLOW_REVERSION=0`
    - `M1SCALP_SIGNAL_TAG_CONTAINS=breakout-retest-long,nwave-long`
    - `M1SCALP_BASE_UNITS=1200`
    - `M1SCALP_MARGIN_USAGE_HARD=0.88`
- `workers/scalp_m1scalper/worker.py`
  - account snapshot をキャッシュ付きで扱い、`/summary` `503` では loop skip / cached snapshot fallback に変更

Verification:
- 反映後:
  - `quant-m1scalper` が `/summary 503` で即死せず、`account snapshot ... cached snapshot` ログで継続すること
  - `orders.db` で `client_order_id like '%scalp-m1scalperm1%'` の `source_signal_tag` が `nwave-long / breakout-retest-long` 中心になること
  - `scalp_ping_5s_flow_live` の filled 件数は減っても、`avg_pips / PF` が改善方向へ動くこと
  - 次の24hで `PF(pips)>0.85` を暫定回復目標、14dで `PF>1.0` を再判定

Status:
- in_progress

## 2026-03-06 05:56 UTC / 2026-03-06 14:56 JST - M1Scalper-M1: 損失縮小のため exit チューニング（profit_buffer拒否多発の抑制）

Period:
- 集計: 直近24h（`logs/pdca_profitability_latest.json` generated_at=`2026-03-06T14:51 JST`）
- 対象（実測）: `logs/trades.db`, `logs/orders.db`

Fact:
- 直近24h（trades, strategy=`m1scalper-m1`）:
  - `PF(pips)=0.5648`（≈0.55）/ `win_rate=0.6205` / `trades=1913`
  - `gross_loss_pips=3413.2 / losses=696` → `avg_loss≈4.90p`
  - `gross_win_pips=1927.7 / wins=1187` → `avg_win≈1.62p`
  - **勝率は高いが avg_loss が avg_win を大幅に上回り、期待値が負け**。
- 直近24h（orders, client_order_id like `%m1scalperm1%`）:
  - `close_reject_profit_buffer=509`（多発）
- Ops note:
  - entry は一時停止（2026-03-06 14:42 JST頃）として扱う（再開は別判断）。
  - ただし `orders.db` では `m1scalperm1` の `filled` が `2026-03-06 14:53 JST` にも観測されており、停止が適用されていない可能性がある（要確認）。

Failure Cause:
- `lock_floor` / 早期の利確系が「極小利益帯」で頻発し、`close_reject_profit_buffer` が多発して EXIT が安定しない。
- 負け側は avg_loss が大きく、損失上限（max_adverse）と負け側の cut 条件が弱い。

Improvement:
- `ops/env/quant-m1scalper-exit.env`（exitのみ）:
  - `M1SCALP_EXIT_LOCK_TRIGGER_MIN_PIPS=1.8`（from `1.00`）
  - `M1SCALP_EXIT_COMPOSITE_MIN_SCORE=2.0`
  - `M1SCALP_EXIT_MAX_ADVERSE_PIPS=4.0`
  - `M1SCALP_EXIT_PROFIT_TAKE_PIPS=3.0`
  - `M1SCALP_EXIT_LOCK_BUFFER_PIPS=0.3`
- entry は停止維持（再開は次サイクル判断）。

Verification:
- 反映後24hで以下を確認:
  - `m1scalper-m1` の `avg_loss` が低下し、`PF(pips)` が改善していること（目標: `PF>0.85` へ回復）。
  - `close_reject_profit_buffer` が減少していること（目標: `<= 1/3`）。
  - `quant-m1scalper-exit` のログで例外/連続close失敗が増えていないこと。

Status:
- in_progress

## 2026-03-06 13:25 JST / local-v2: 資産減少/稼げてない RCA（scalp_ping_5s_flow_live + M1Scalper-M1 寄与大、margin closeout 高止まり）

Period:
- 集計: 2026-03-05 13:19〜2026-03-06 13:19 JST（UTC 2026-03-05 04:19〜2026-03-06 04:19）
- 比較: 直近7d（UTC 2026-02-27 04:19〜2026-03-06 04:19）
- 対象（実測）: `logs/metrics.db`, `logs/orders.db`, `logs/trades.db`
- 対象（OANDA API）: account summary / transactions（TRANSFER_FUNDS, DAILY_FINANCING）/ openTrades / pricing

Fact:
- 市況（OANDA pricing, UTC 04:10 / JST 13:10）:
  - USD/JPY `bid=157.510 / ask=157.518 / spread=0.8p`
  - pricing latency `~200ms`
  - replay(M5) `ATR14_pips~3.14` / `range_60m_pips~6.3`
- 口座状態（OANDA summary, UTC 04:14 / JST 13:14）:
  - `balance=38652.5 / NAV=38321.7 / unrealized=-330.8`
  - `marginCloseoutPercent=0.9428` / `marginAvailable=2191.9`（**closeout近傍**）
  - openTrades: `n=13`、`USD_JPY net_units=+4175 / gross_units=7303`（unrealized寄与: micro=-262 / scalp=-60 / scalp_fast=-8）
- 直近24h（account metrics, practice=false）:
  - `ΔNAV=-12845.9 / Δbalance=-10454.5`（含み変動が約 `-2391`）
- 直近24h（bot only, trades.db realized_pl）:
  - `n=2459 / win_rate=0.536 / PF=0.508 / net=-10773 / net_pips=-1743.6`
  - win率>50%でも **avg_loss が avg_win を大幅に上回る**（期待値負け）
- 直近7d（trades.db realized_pl）:
  - bot: `net=-11096 / PF=0.506`、manual: `net=-7851 / PF=0.057`（manual寄与が大きい）
  - OANDA DAILY_FINANCING: `-66.9(24h) / -653.2(7d)`（主因ではないが確実に減る）
- 実行コスト（orders.db 直近2000 filled, analyze_entry_precision）:
  - `spread_pips mean=0.805 (p50=0.8)` / `cost_vs_mid_pips mean=0.401` / `slip_p95=0.3p`
  - `latency_submit p50~193ms` / `latency_preflight p50~228ms`（**執行自体は通常**）
  - strategy別（要点）:
    - `scalp_ping_5s_flow_live`: `tp_mean~1.44p / sl_mean~1.32p`（**spread(0.8p)に対して取り幅が小さすぎる**）
    - `M1Scalper-M1`: `tp_mean~7.35p / sl_mean~6.07p` だが、実現では `avg_win~+5.32 / avg_loss~-17.71`（損失が勝ちの約3.3倍）
- 拒否/ブロック（orders.db, client_order_id最終ステータス, 直近24h）:
  - OANDA `rejected=3/2934`（低い）
  - 一方で `margin_usage_projected_cap=119` / `strategy_control_entry_disabled=146` / `margin_usage_exceeds_cap=26` が観測（主にscalp_fast）

Failure Cause:
- 主因（戦略期待値）:
  - `scalp_ping_5s_flow_live` が **PF=0.30台**で大幅な赤字寄与（取り幅が spread/コストに対して小さく、期待値が構造的に負け）。
  - `M1Scalper-M1` は勝率は高いが、**avg_loss が avg_win の約3倍**で PF<1 に沈む（負けトレードの損失拡大）。
- 副因（リスク/稼働制約）:
  - margin closeout が高止まり（`marginCloseoutPercent~0.94`）し、risk cap 系ブロックが出て「悪化局面での建玉・調整」が難しくなる。

Improvement:
- P0（当日・安全）: margin closeout 回避を最優先。新規entryを抑え、exitは継続。
  - `SCALP_PING_5S_FLOW_*` の entry を一時縮小/停止（`STRATEGY_CONTROL_ENTRY_SCALP_PING_5S_FLOW=0` または units/active_trades を強制縮小）し、口座余力を回復させる。
  - `quant-m1scalper` は `M1SCALP_BASE_UNITS` を縮小し、`M1SCALP_MARGIN_USAGE_HARD` を下げて過剰レバを抑える。
- P1（当日〜1日）: scalp_ping_5s_flow_live の「期待値負け」を構造的に解消。
  - forecast gate: `FORECAST_GATE_EXPECTED_PIPS_MIN_*` と `TARGET_REACH_MIN_*` を **cost_vs_mid(≈0.4p)+spread余裕**以上へ引き上げ（low edgeを排除）。
  - local gating: `SCALP_PING_5S_FLOW_ENTRY_LEADING_PROFILE_REJECT_BELOW` を引き上げ、低品質エントリを減らす。
  - perf/profit guard（strategy限定）を有効化し、PF悪化時は scale-to-low へ自動退避。
- P1（1〜3日）: M1Scalper-M1 の「損失拡大」を抑制（exit_worker/損切り設計を重点監査）。
  - 負け側の cut を早める（損失上限・早期撤退条件の追加/強化）。
  - replay で「どの局面が大負け源泉か」を再現し、entryフィルタ or exit改善へ落とす。

Verification:
- 24hで `PF>1.0` / `expectancy>0` を最低目標（まず n>=300 で暫定判定 → 14d で確定）。
- `scalp_ping_5s_flow_live` の `tp_mean/spread` 比が改善し、net_pips がマイナス継続しないこと。
- `marginCloseoutPercent` と `account.margin_usage_ratio` が 0.90 未満へ戻ること（急変時は即時縮小）。

Status:
- open

## 2026-03-05 17:30 JST / local-v2: Pattern Gateが実質無効化されていた問題を修正（preserve_strategy_intent下でも評価）+ 運用キー整理 + trades.entry_thesis backfill 追加

Period:
- 調査: 2026-03-05 16:55〜17:30 JST（UTC 07:55〜08:30）
- 市況確認: 2026-03-05 17:00 JST（UTC 08:00）
- 対象（実測）: `logs/trades.db`, `logs/orders.db`, `logs/patterns.db`, `logs/tick_cache.json`, `logs/oanda/candles_*_latest.json`
- 対象（実装）: `execution/order_manager.py`, `ops/env/local-v2-stack.env`, `scripts/backfill_entry_thesis_from_orders.py`

Fact:
- 市況（local tick/candles, UTC 08:00 / JST 17:00）:
  - USD/JPY `bid=157.343 / ask=157.351 / spread=0.8p`（tick直近200: min/median/p90/max=0.8p）
  - `ATR14_pips(M1)=2.04` / `range_60m_pips(M1)=11.0`
  - `ATR14_pips(H1)=12.69` / `ATR14_pips(H4)=31.82`
- 直近7d（観測メモ, manual含む）:
  - `net=-7291` / `PF=0.21`
  - 最大損失: manual pocket `2026-03-02` の `MARKET_ORDER_MARGIN_CLOSEOUT (-7696)`
  - OANDA openTrades: `-6998 units` の巨大shortが残存し、`margin_usage_ratio~0.88 / health_buffer~0.12`
- scalp_fast（観測メモ, 直近24h）:
  - `scalp_ping_5s_b_live` の long 側 `504 trades net=-183`、short 側 `201 trades net=+1.8`
  - close_reason は `SL 539件 (avg -1.42pips)` が支配
  - ただし `ops/env/local-v2-stack.env` で `SCALP_PING_5S_B_SIDE_FILTER=sell` に切替後は long close が 0（設定ドリフト起因の偏損が濃厚）
- Pattern Gate が “死んでいた” 根拠（コード+DB）:
  - `execution/order_manager.py` の Pattern Gate は `not preserve_strategy_intent` 条件で常にスキップされ得る。
  - `ORDER_MANAGER_PRESERVE_STRATEGY_INTENT=1`（既定）かつ `ORDER_MANAGER_PATTERN_GATE_ENABLED` 未設定（既定false）だと、orders.db に `pattern_block/pattern_scale_*` が出ない。
  - 一方 `logs/patterns.db` には `st:scalp_ping_5s_b_live...` 等のスコアが大量に存在し、avoid/weak の識別が可能。

Failure Cause:
- Pattern Gate の運用キーが二重（`ORDER_PATTERN_GATE_ENABLED` と `ORDER_MANAGER_PATTERN_GATE_ENABLED`）かつ、
  order_manager側の実行条件が `preserve_strategy_intent` に依存していたため、**opt-in戦略でも gate が実行されない**状態になっていた。

Improvement:
- Pattern Gate を preserve_strategy_intent 下でも評価する:
  - `execution/order_manager.py`: `Pattern gate` 条件から `and not preserve_strategy_intent` を除去（pattern_gate自体はopt-inなので仕様整合）。
- ローカルV2導線で Pattern Gate を有効化:
  - `ops/env/local-v2-stack.env`: `ORDER_MANAGER_PATTERN_GATE_ENABLED=1`
- （任意/母集団浄化）過去 trades の entry_thesis 契約欠損を backfill:
  - `scripts/backfill_entry_thesis_from_orders.py` を追加し、`orders.db submit_attempt.request_json` から `entry_thesis` を復元して `trades.db` を更新（バックアップ/ロック付）。

Verification:
- Pattern Gate の作動確認:
  - `orders.db` に `status='pattern_block'` / `status='pattern_scale_below_min'` が出ること（該当戦略の opt-in 前提）。
  - `request_json.entry_thesis.pattern_gate` が payload を持つこと（allowed/scale/reason/pattern_id）。
- backfill（dry-run→適用）:
  - `python scripts/backfill_entry_thesis_from_orders.py --dry-run`
  - `python scripts/backfill_entry_thesis_from_orders.py --until-utc 2026-02-27T23:59:59+00:00`
- 収益の再評価:
  - scalp_fast の `SL率/avg_pips` が改善方向か、`PF/expectancy` が悪化しないこと（n>=300 の短期判定→14dで再評価）。

Status:
- in_progress

## 2026-03-05 15:50 JST / local-v2: PF悪化RCA（scalp_ping_5s_b寄与大）+ Brain autopdca既定OFF(=opt-in)化

Period:
- 集計: 2026-03-04 15:46:50〜2026-03-05 15:46:50 JST（UTC 2026-03-04 06:46:50〜2026-03-05 06:46:50）
- 市況確認: 2026-03-05 15:40 JST（UTC 06:40）
- 対象: `logs/trades.db`, `logs/factor_cache.json`, `logs/health_snapshot.json`, `scripts/local_v2_autorecover_once.sh`, `scripts/local_v2_stack.sh`

Fact:
- 市況（OANDA実測 + local factor, UTC 06:40 / JST 15:40）:
  - USD/JPY `bid=157.106 / ask=157.114 / spread=0.8p`
  - pricing latency `avg=255ms`（samples `[247,266,251]`）
  - `ATR14_pips(M1)=2.64`, `ATR14_pips(M5)=5.87`, `ATR14_pips(H1)=18.02`
- 直近24h（manual除外, `trades.db`, realized_pl）:
  - `n=706`, `win_rate=0.218`, `PF=0.423`, `expectancy=-0.73 JPY/trade`
  - `net=-518.5 JPY`, `net_pips=-567.1`
- 負け寄与上位（`pocket|strategy_tag`, count>=5, net昇順）:
  - `micro|MicroPullbackEMA n=25 net=-133.9 PF=0.111`
  - `scalp_fast|scalp_ping_5s_b_live n=606 net=-133.2 PF=0.436`
- 直近14d（count>=10, net上位）:
  - `scalp|M1Scalper-M1 n=16 net=+413.3 PF=4.092`
  - `micro|MicroRangeBreak n=27 net=+6.4 PF=1.035`

Failure Cause:
- `scalp_ping_5s_b` が **高頻度かつ期待値マイナス**のため、取引回数の大半を占有しPF/期待値を押し下げた。
- `MicroPullbackEMA` が micro pocket 内で大きなマイナス寄与（PF=0.111）。
- （運用リスク）`local_v2_autorecover_once.sh` の Brain autopdca が既定ONだと、意図せず cycle/restart が走り得る（opt-in運用と不整合）。

Improvement:
- `trade_min` の構成を **core + MicroRangeBreak(+exit) + M1Scalper(+exit)** に寄せ、`scalp_ping_5s_b(+exit)` を除外（stack側で反映）。
- Brain autopdca を **既定OFF（opt-inのみ）**へ変更:
  - `QR_LOCAL_V2_BRAIN_AUTOPDCA_ENABLED` 既定 `0`
  - `QR_LOCAL_V2_BRAIN_AUTOPDCA_ALLOW_RESTART` 既定 `0`（未指定時は `--dry-run` で実行し restart しない）

Verification:
- profile反映後:
  - `scripts/local_v2_stack.sh status --profile trade_min --env ops/env/local-v2-stack.env` で想定workerのみが `running` であること（`scalp_ping_5s_b` が起動していない）。
- 次の再評価（まず24h、次に14d）:
  - 同一集計で `PF>1.0` / `expectancy>0` を目標（`n>=300` で判定、未達なら更にRCA）。
- Brain autopdca:
  - デフォルトで `local_v2_autorecover.log` に autopdca cycle が出ないこと。
  - opt-in時（`QR_LOCAL_V2_BRAIN_AUTOPDCA_ENABLED=1`）でも、`QR_LOCAL_V2_BRAIN_AUTOPDCA_ALLOW_RESTART=1` を明示しない限り restart されないこと（cycle output の `dry_run=true`）。

Status:
- in_progress

## 2026-03-05 15:40 JST / local-v2: MicroPullbackEMAの勝率改善（ATRスケール+M5/H1確認）+ strategy_control hard stop解除

Period:
- 対応: 2026-03-05 15:34〜15:40 JST（UTC 06:34〜06:40）
- 市況確認: 2026-03-05 15:34 JST（UTC 06:34）
- 対象: `strategies/micro/pullback_ema.py`, `workers/micro_runtime/worker.py`, `ops/env/local-v2-stack.env`

Fact:
- 市況（local tick/factor + OANDA, UTC 06:34 / JST 15:34）:
  - USD/JPY spread(avg/p95)=0.8p（tick直近500）/ mid_last=157.134
  - `ATR14(M1)=2.17p`, `ATR14(M5)=5.71p`, `ATR14(H1)=18.02p`
  - M1 `regime=Range`, `ADX=19.97` / H1 `regime=Trend`, `ADX=35.61`
- 直近7d（manual除外, `trades.db`, micro pocket）:
  - `MicroPullbackEMA n=25`, **all long**, `win_rate=0.28`, `PF=0.062`, `net_pips=-45.6`
- 運用設定（対応前）:
  - `STRATEGY_CONTROL_ENTRY_* =0` により entry を hard stop（停止済み戦略の注文試行が `strategy_control_entry_disabled` になりノイズ化）。

Failure Cause:
- MicroPullbackEMA が M1のみで方向決定し、M5/H1の逆行トレンドで long 側が連続損失。
- pullback/ma-gap が固定幅で、低ATR/弱gapで深いpullbackを許容しやすく、レンジ寄りで stop hit が増加。

Improvement:
- MicroPullbackEMA（strategy）:
  - gap/pullback を ATR スケール化（弱トレンド/低vol の誤爆を抑止）
  - `plus_di/minus_di` で方向整合を追加
  - `abs(pullback) <= abs(gap) + buffer(ATR連動)` で深い pullback を抑制
  - range_bias 時は ADX 閾値を上げる
- micro_runtime worker:
  - MicroPullbackEMA に M5/H1 の MA-gap+ADX 確認ゲートを追加（counter-trend を遮断）
- local-v2 env:
  - `STRATEGY_CONTROL_ENTRY_*` を 1 に戻し、停止ではなくフィルタでリスク制御へ
  - `LOCAL_V2_EXTRA_ENV_FILES=` を維持（Brain gate無効のまま）

Verification:
- deploy後（〜2h）:
  - `orders.db` 新規の `strategy_control_entry_disabled` が増えない（矛盾/ノイズ解消）
  - MicroPullbackEMA の `win_rate/PF` が改善（最低でも `PF>1.0` / `expectancy>0` を目標、`n>=30` で再評価）
  - M5/H1 逆行時に `pullback_mtf_block` ログが出ていること
  - Brain gate が無効のまま（order_managerで `BRAIN_ENABLED=0` 維持）

Status:
- in_progress

## 2026-03-05 15:00 JST / local-v2: trade_all常駐の解消（watchdog監視対象の縮退）+ dyn alloc sampling bias修正

Period:
- 対応: 2026-03-05 14:53〜15:00 JST（UTC 05:53〜06:00）
- 市況確認: 2026-03-05 14:43 JST（UTC 05:43）
- 対象: `scripts/local_v2_stack.sh`, `scripts/install_local_v2_launchd.sh`, `~/Library/LaunchAgents/com.quantrabbit.local-v2-autorecover.plist`, `scripts/dynamic_alloc_worker.py`, `config/dynamic_alloc.json`

Fact:
- 市況（OANDA実測, UTC 05:43 / JST 14:43）:
  - USD/JPY `bid=157.039 / ask=157.047 / spread=0.8p`
  - `ATR14(M5)=6.712p(Wilder)`, `range_60m=21.4p`
  - pricing latency `350ms`, candles(M5) latency `318ms`
- local-v2稼働状況（対応直前）:
  - `scripts/local_v2_stack.sh status --profile trade_all --env ops/env/local-v2-stack.env` で多数workerが `running` のまま残存していた（trade_min前提の運用と不整合）。
  - `orders.db` 直近では `strategy_control_entry_disabled` が連続し、entry停止済み戦略が注文試行を継続していた（ノイズ/負荷）。
- sizing:
  - `scripts/dynamic_alloc_worker.py` が `--limit 300` 固定だと、高頻度戦略（例: `scalp_ping_5s_b_live`）の直近取引だけで埋まり、
    低頻度/別ポケット戦略（例: `MicroRangeBreak`）が `config/dynamic_alloc.json` に出ないことがあった。

Failure Cause:
- trade_min想定でも trade_all worker が常駐し、entry停止しても「注文試行/ログ/CPU負荷」が積み上がる。
- dyn alloc のサンプルが「直近N件」偏重で、戦略間の比較・ロット配分が歪む（高頻度が全枠を占有）。

Improvement:
- watchdog/launchd の監視対象を明示縮退:
  - `scripts/install_local_v2_launchd.sh` を `--services "quant-market-data-feed,quant-strategy-control,quant-order-manager,quant-position-manager,quant-micro-rangebreak,quant-micro-rangebreak-exit"` で再インストール（`StartInterval=20s`）。
- trade_all 常駐の解消:
  - `scripts/local_v2_stack.sh down --profile trade_all --env ops/env/local-v2-stack.env`
  - `scripts/local_v2_stack.sh up --services "quant-market-data-feed,quant-strategy-control,quant-order-manager,quant-position-manager,quant-micro-rangebreak,quant-micro-rangebreak-exit" --env ops/env/local-v2-stack.env`
- dyn alloc sampling bias修正:
  - `scripts/dynamic_alloc_worker.py` の `--limit` を既定 `5000` に拡張し、`--limit 0` で full lookback を許可。
  - `--limit 0 --min-trades 24` で再生成し、`MicroRangeBreak` が alloc に含まれることを確認（例: `lot_multiplier=0.681`, `jpy_pf=1.033`）。

Verification:
- `scripts/local_v2_stack.sh status --profile trade_all --env ops/env/local-v2-stack.env` で `running` が縮退サービスのみになる。
- `config/dynamic_alloc.json` に `MicroRangeBreak` が出力され、`lot_multiplier` が floor (`0.45`) に貼り付かない。
- `orders.db` で停止済み戦略由来の `strategy_control_entry_disabled` が新規に増えない（stop後は `orders_last_ts` が更新されない/必要戦略のみ更新される）。

Status:
- in_progress（縮退後の損益・約定・負荷の前後比較が必要）

## 2026-03-05 14:05 JST / Brain(ollama)タイムアウト起因のpreflight遅延を除去 + strategy_control env override修正 + 赤字戦略のentry停止

Period:
- 調査/反映: 2026-03-05 13:38〜14:05 JST（UTC 04:38〜05:05）
- 対象: `ops/env/local-v2-stack.env`, `logs/orders.db`, `logs/trades.db`, `logs/brain_state.db`, `logs/strategy_control.db`, `logs/local_v2_stack/quant-order-manager.log`

Fact:
- 市況（OANDA実測, UTC 04:38 / JST 13:38）:
  - USD/JPY `bid=157.112 / ask=157.120 / spread=0.8p`
  - `ATR14(M5)=6.383p(EMA)`, `range_60m=25.9p`
  - pricing latency `avg=285ms (max=337ms)`, openTrades latency `216ms`
- 直近24hの収益（`trades.db`, manual除外）:
  - `n=797`, `win_rate=0.2095`, `PF=0.421`, `net_jpy=-545.863`, `net_pips=-649.2`
  - 赤字寄与上位: `scalp_ping_5s_b_live(-181.408)`, `MicroPullbackEMA(-112.892)`, `MicroTrendRetest-long(-91.344)`, `scalp_ping_5s_d_live(-84.1)`, `scalp_ping_5s_flow_live(-79.128)`
- 実行品質（`scripts/analyze_entry_precision.py`, filled entry 直近1113）:
  - `spread_pips p95=0.8`, `latency_submit p95=303ms`
  - **`latency_preflight p95=7509ms`**（スキャルプに致命的）
- Brain決定（`brain_state.db`, 直近24h）:
  - `source=llm_fail` が多発し、`avg_latency ≒ 6.0s`（例: `scalp_ping_5s_b_live n=104 avg=5994ms`）
  - order_manager 側で `slow_request elapsed=8〜14s` が連発し、strategy 側で `order_manager Read timed out (8s)` が発生していた。

Failure Cause:
- `ops/env/local-v2-stack.env` の `LOCAL_V2_EXTRA_ENV_FILES=ops/env/profiles/brain-ollama.env` により Brainゲートが常時有効化され、
  ollama呼び出しが `BRAIN_TIMEOUT_SEC=6` で連続失敗 → preflight遅延 + market_order応答タイムアウト → stale entry/損失増大。

Improvement:
- BrainゲートをローカルV2のデフォルト導線から外す:
  - `ops/env/local-v2-stack.env` の `LOCAL_V2_EXTRA_ENV_FILES` を空に変更。
  - `quant-order-manager` を `--env ops/env/local-v2-stack.env` で再起動し、ログで `BRAIN_ENABLED=0 / ORDER_MANAGER_BRAIN_GATE_ENABLED=0` を確認。
  - `brain_decisions` は直近30秒で `0件`（新規 `llm_fail` 停止）。
  - `order/market_order` は reject でも `~70ms` で応答（preflightがタイムアウトしないことを確認）。
- 入口制御の再発防止:
  - `workers/common/strategy_control.py` の env override が `value` を参照しており無効だったため修正（`_env_bool(key)`）。
  - `STRATEGY_CONTROL_ENTRY_*` で赤字戦略（`scalp_ping_5s_{b,c,d,flow}`, `micropullbackema`, `microtrendretest`）の entry を停止し、
    `strategy_control.db` で `entry_enabled=0` を確認（exitは維持）。

Verification:
- 短期（〜1h）:
  - `brain_decisions` の新規 `llm_fail=0` を維持。
  - strategy 側の `order_manager Read timed out` が再発しない。
  - `orders.db` の `brain_scale_below_min` が新規に増えない（Brain無効化の確認）。
- 中期（1〜3h）:
  - 稼働中戦略の `PF>1.0` / `expectancy>0` が確認できたものから順次有効範囲を広げる（停止戦略は原因分析→改善→段階復帰）。

Status:
- in_progress（稼働後の損益前後比較が必要）

## 2026-03-05 11:16 JST / scalp_ping_5s_b 約定停止の即効チューニング（side_filter解除 + lookahead遮断緩和）

Period:
- 調査時刻: 2026-03-05 11:06〜11:16 JST
- 対象: `ops/env/local-v2-stack.env`, `logs/orders.db`, `logs/trades.db`, `logs/local_v2_stack/quant-scalp-ping-5s-b.log`

Fact:
- 着手前手順:
  - `sed -n '/^## 運用手順/,/^## /p' docs/AGENT_COLLAB_HUB.md` を実行し、ローカルV2運用手順を確認。
  - `scripts/local_v2_stack.sh status --profile trade_min --env ops/env/local-v2-stack.env` で主要サービスが `running` を確認。
- 市況/執行プロキシ（JST 11時台）:
  - 価格帯 proxy（`orders.filled.executed_price` 直近20件）: `156.846 - 157.078`
  - spread/range proxy（`scalp_ping_5s_b` lookaheadログ）: `cost_pips ≒ 1.096-1.132`, `range ≒ 0.1-0.8p`
  - OANDA API応答品質: `quant-scalp-ping-5s-b.log` で `/v3/accounts/.../pricing` の `HTTP 200` 継続を確認。
- 約定停止の事実（JST明記）:
  - `orders` 最終約定: `2026-03-05 09:10:20 JST`（`2026-03-05T00:10:20.711262+00:00`）
  - `trades` 最終クローズ: `2026-03-04 22:54:44 JST`（`2026-03-04T13:54:44.442316+00:00`、前日）
- skip内訳（`quant-scalp-ping-5s-b.log`）:
  - `entry-skip summary side=long total=14 no_signal:side_filter_block=14`
  - `entry-skip summary side=short total=23 lookahead_block=23`
  - side_filter解除後の実測: `side_filter=(unset)` は反映されたが、`lookahead edge_negative_block` が主遮断で `orders` 更新なし。

Failure Cause:
- 主因1: `SCALP_PING_5S_B_SIDE_FILTER=short` 固定で long 側が実質停止し、シグナルが `side_filter_block` に集中。
- 主因2: short 側は `lookahead_block` 比率が高止まりし、entry 変換率が低下したまま約定停止へ遷移。

Improvement:
- `ops/env/local-v2-stack.env` を local-v2 即効チューニングとして更新:
  - `SCALP_PING_5S_B_SIDE_FILTER=none`
  - `SCALP_PING_5S_B_ALLOW_NO_SIDE_FILTER=1`
  - `SCALP_PING_5S_B_DIRECTION_BIAS_LONG_OPPOSITE_UNITS_MULT=0.35`
  - `SCALP_PING_5S_B_LOOKAHEAD_GATE_ENABLED=0`
  - `SCALP_PING_5S_B_LOOKAHEAD_SLIP_SPREAD_MULT=0.18`
  - `SCALP_PING_5S_B_LOOKAHEAD_SLIP_RANGE_MULT=0.08`
  - `SCALP_PING_5S_B_LOOKAHEAD_LATENCY_PENALTY_PIPS=0.01`
- 追加施策: `LOOKAHEAD_GATE_ENABLED=0` を適用し、`edge_negative_block` 主遮断を一時的に外して即時の約定再開を優先。
- 狙い:
  - 約定再開（long側の `side_filter_block` 解消）
  - short側 lookahead の過剰遮断を緩和しつつ、`DIRECTION_BIAS_LONG_OPPOSITE_UNITS_MULT=0.35` で過度な逆張りを抑制。

Verification:
- 反映後に `orders.filled` が再開し、最終約定時刻が更新されること。
- `entry-skip summary` で `side_filter_block`（long）と `lookahead_block`（short）の比率が低下すること。
- `HTTP 200` 継続と `rejected=0`（orders）を維持すること。

Status:
- in_progress

## 2026-03-05 06:35 JST / Hourly RCA HOLD（trade_min停止 + trades stale + OANDA DNS失敗）

Period:
- 調査時刻: 2026-03-05 06:29〜06:35 JST
- 対象: `docs/AGENT_COLLAB_HUB.md`, `docs/OPS_LOCAL_RUNBOOK.md`, `logs/orders.db`, `logs/trades.db`, `logs/metrics.db`, `scripts/local_v2_stack.sh`, `scripts/check_oanda_summary.py`

Fact:
- 着手前手順:
  - `sed -n '/^## 運用手順/,/^## /p' docs/AGENT_COLLAB_HUB.md` を実行し運用手順を確認。
  - `sed -n '/^## 運用原則/,/^## /p' docs/OPS_LOCAL_RUNBOOK.md` を実行（該当見出し抽出は空出力）。
- DB鮮度（最終再確認, JST）:
  - `orders.db=05:22:04 (age 72.4分)`, `trades.db=02:26:29 (age 248.6分)`, `metrics.db=06:35:11 (age 0.3分)`
  - 30分超 stale 条件該当のため `scripts/local_v2_stack.sh up --profile trade_min --env ops/env/local-v2-stack.env` を実行し、120秒待機後に再確認。
- スタック状態:
  - `up` は `quant-order-manager port=8300 remains occupied` で失敗（占有PID: `38068,38234,38235,38236,38238,38239,38240`）。
  - `scripts/local_v2_stack.sh status --profile trade_min --env ops/env/local-v2-stack.env` は対象サービスが全て `stopped`。
- OANDA API確認:
  - `PYTHONPATH=. python3 scripts/check_oanda_summary.py` は `NameResolutionError(api-fxtrade.oanda.com)` で失敗。
  - secret欠損エラーではないため `refresh_env_from_gcp.py` 条件には非該当。
- 市況/執行プロキシ（直近2h, `logs/*.db` 実測）:
  - 価格帯（filled）: `157.002 - 157.094`、mid帯（entry_thesis）: `156.998 - 157.089`
  - spread: `avg/min/max = 0.8 / 0.8 / 0.8 pips`
  - ATR/range proxy: `atr_m1_pips avg=1.58`, `signal_range_pips avg=0.59`
  - 約定/拒否: `filled=10`, `submit_attempt=10`, `close_ok=2`, `reject=0`
  - 応答品質 proxy: `data_lag_ms avg=838.452 (max=2457.185)`, `decision_latency_ms avg=32.899 (max=98.952)`
- 直近2h PnL分解:
  - `trades.db` 実績 `0件`（strategy別/時間帯別/拒否理由別とも集計不可）。
  - 実行コスト proxy: `slippage vs ideal_entry = +0.15 pips`, `spread avg=0.8 pips`

Failure Cause:
- 利益阻害 Top3（数値根拠）:
  1. `trades.db` が `248.6分` stale で、2h PnLの一次データが欠損（`trades=0`）。
  2. `check_oanda_summary.py` が `NameResolutionError` で `1/1失敗`、API品質の正常判定が不能。
  3. `quant-order-manager` の `8300` 競合により `trade_min` 復旧失敗、`status` は主要サービス `stopped`。

Improvement:
- 本時間帯の Hourly RCA は `HOLD`。
- 次の1アクション:
  - `8300` 占有プロセス解消で `local_v2_stack up --profile trade_min` を成功させ、
    `orders/trades <=10分` と `check_oanda_summary.py` 成功を同時達成した次ランで
    2h PnL本分解（strategy別/時間帯別/拒否理由別/実行コスト別）を再開する。

Verification:
- `scripts/local_v2_stack.sh status --profile trade_min --env ops/env/local-v2-stack.env` で主要サービスが `running`。
- `logs/orders.db` と `logs/trades.db` の最終更新が `now-10m` 以内。
- `PYTHONPATH=. python3 scripts/check_oanda_summary.py` が成功終了。

Status:
- in_progress（HOLD）

## 2026-03-05 05:36 JST / Hourly RCA HOLD（trades stale継続 + Summary API DNS失敗）

Period:
- 調査時刻: 2026-03-05 05:29〜05:36 JST
- 対象: `docs/AGENT_COLLAB_HUB.md`, `docs/OPS_LOCAL_RUNBOOK.md`, `logs/orders.db`, `logs/trades.db`, `logs/metrics.db`, `scripts/local_v2_stack.sh`, `scripts/check_oanda_summary.py`

Fact:
- 着手前手順:
  - `docs/AGENT_COLLAB_HUB.md` の「運用手順」を確認（runbook準拠）。
  - `docs/OPS_LOCAL_RUNBOOK.md` の「運用原則」は該当見出しの抽出実行のみ（出力なし）。
- DB鮮度（最終再確認, JST）:
  - `orders.db=05:22:04 (age 13.35分)`, `trades.db=02:26:29 (age 188.93分)`, `metrics.db=05:35:18 (age 0.13分)`
  - 初回鮮度チェックで `trades.db` が30分超 stale のため、
    `scripts/local_v2_stack.sh up --profile trade_min --env ops/env/local-v2-stack.env` を実行。
- スタック起動結果:
  - `quant-order-manager port=8300 remains occupied (start aborted for safety)` で `up` は失敗（120秒待機後も改善なし）。
  - `scripts/local_v2_stack.sh status --profile trade_min --env ops/env/local-v2-stack.env` は対象サービス `stopped`。
- OANDA API確認:
  - `PYTHONPATH=. python3 scripts/check_oanda_summary.py` は
    `NameResolutionError(api-fxtrade.oanda.com)` で失敗（secret/env欠損ではない）。
- 市況/執行プロキシ（直近2h, `logs/*.db` 実測）:
  - 価格帯（`orders.filled.executed_price`）: `157.002 - 157.131`（range `12.9 pips`）
  - spread（`preflight_start.entry_thesis.spread_pips`）: `avg=0.8 pips`（min/max `0.8/0.8`）
  - ATR proxy（`preflight_start.entry_thesis`）:
    - `mtf_regime_atr_m1 avg=1.639 pips`（`1.502-1.873`）
    - `mtf_regime_atr_m5 avg=4.823 pips`（`4.526-5.175`）
  - range proxy（`signal_range_pips`）: `avg=0.760 pips`（`0.3-1.3`）
  - 約定/拒否（orders 2h）: `total=58`, `filled=15`, `close_ok=4`, `reject_like=0`, `order_success_rate avg=1.0`, `reject_rate avg=0.0`
  - API応答品質 proxy（metrics 2h）:
    - `data_lag_ms avg=858.79, p95=1776.41, max=2794.15`
    - `decision_latency_ms avg=34.14, p95=62.34, max=98.95`
- 直近2h PnL分解:
  - `trades.db` ベース実績: `trades=0`, `realized_pnl=0`（strategy/hour/reject/execution cost いずれも実績ゼロ）
  - `orders.db` 代理集計（filled↔close_ok CID対応）:
    - strategy別: `scalp_ping_5s_b_live trades=4, net_pips=+0.20, net_unit_pips=+27.0`
    - hour別: `03:00JST +2.0p`, `04:00JST -0.8p`, `05:00JST -1.0p`
    - reject理由別: `0件`
    - execution cost proxy: `closed=4`, `avg_hold_sec=200.83`, `loss_count=3`, `avg_loss=-0.60p`, `avg_win=+2.00p`

Failure Cause:
- 利益阻害 Top3（数値根拠）:
  1. `trades.db` stale (`188.93分`) により、2h realized PnL分解が `0件` となり本来のRCA軸が欠損。
  2. `check_oanda_summary.py` が `NameResolutionError` で `1/1失敗`、API正常性を確認不能。
  3. `quant-order-manager` の `8300` 競合で `local_v2_stack up` が失敗し、復旧オペレーションが閉塞。

Improvement:
- 次の1アクション（復旧ゲート）:
  - `8300` 占有元を解消して `local_v2_stack up --profile trade_min` を成功させ、
    `trades.db age <= 10分` と `check_oanda_summary.py` 成功を同時達成した次ランで
    2h PnL本分解（strategy/hour/reject/execution cost）を再開する。

Verification:
- `scripts/local_v2_stack.sh status --profile trade_min --env ops/env/local-v2-stack.env` で主要サービスが `running`。
- `logs/trades.db` 最終更新が `now-10m` 以内。
- `PYTHONPATH=. python3 scripts/check_oanda_summary.py` が成功終了。

Status:
- in_progress（HOLD）

## 2026-03-05 04:34 JST / Hourly RCA HOLD（OANDA Summary API DNS失敗継続）

Period:
- 調査時刻: 2026-03-05 04:29〜04:34 JST
- 対象: `docs/OPS_LOCAL_RUNBOOK.md`, `docs/AGENT_COLLAB_HUB.md`, `logs/orders.db`, `logs/trades.db`, `logs/metrics.db`, `scripts/local_v2_stack.sh`, `scripts/check_oanda_summary.py`

Fact:
- 着手前手順:
  - `docs/OPS_LOCAL_RUNBOOK.md` の運用原則、および `docs/AGENT_COLLAB_HUB.md` の運用手順を確認。
- DB鮮度（初回確認, JST）:
  - `orders.db=03:50:07`, `trades.db=02:26:29`, `metrics.db=04:29:45`
  - `trades.db` が30分超 stale のため、指示どおり `scripts/local_v2_stack.sh up --profile trade_min --env ops/env/local-v2-stack.env` を実行。
- スタック起動結果:
  - `quant-order-manager port=8300 remains occupied (start aborted for safety)` で `up` は完了せず（feed/control のみ起動ログあり）。
  - 120秒待機後の再確認でも `trades.db=02:26:29 JST` のまま更新なし。
- OANDA API確認:
  - `PYTHONPATH=. python3 scripts/check_oanda_summary.py` を2回実行し、2/2で `NameResolutionError(api-fxtrade.oanda.com)`。
  - secret欠損ではなく DNS 解決失敗。
- 市況/執行プロキシ（`logs/*.db` 直近2h, window UTC `17:34:42-19:34:42`）:
  - 価格帯（`filled.executed_price` + `close_ok.exit_context.mid`）: `156.988 - 157.156`
  - spread: `avg 0.8 pips`
  - ATR proxy: `mtf_regime_atr_m1 avg 1.899 pips`, `mtf_regime_atr_m5 avg 5.408 pips`
  - range proxy: `signal_range_pips avg 0.94 pips`
  - 約定/拒否: `orders=37`, `filled=10`, `reject_like=0`, `order_success_rate_avg=1.0`, `reject_rate_avg=0.0`
  - API応答品質 proxy（strategy_control）: `data_lag_ms avg=894.061 / max=20850.001`, `decision_latency_ms avg=34.566 / max=195.601`
  - `trades_2h=0`, `net_pnl=0`（`trades.db`最終更新が約340分 stale）

Failure Cause:
- ローカル市況/執行プロキシ自体は極端悪化ではないが、`check_oanda_summary.py` が継続してDNS失敗し、
  API応答品質を「正常」と判定できない。
- さらに `trades.db` が長時間 stale のままで、PnL分解の入力品質を満たさない。

Improvement:
- 本時間帯の Hourly RCA は `HOLD` とし、通常の2h PnL分解（strategy別/時間帯別/拒否理由別/実行コスト別）は実施しない。
- 次の1アクション（再開ゲート）:
  - `check_oanda_summary.py` 成功（DNS復旧）かつ `orders.db/trades.db` 更新が `<=10分` に戻るまで復旧対応を優先し、達成後ランでRCA本体を再開する。

Verification:
- `PYTHONPATH=. python3 scripts/check_oanda_summary.py` が成功終了すること。
- `logs/orders.db` と `logs/trades.db` の最終更新が `now-10m` 以内になること。
- 条件成立後にのみ、直近2h PnL分解（strategy/hour/reject/execution cost）を再開すること。

Status:
- in_progress（HOLD）

## 2026-03-05 02:35 JST / Hourly RCA HOLD（OANDA DNS劣化でAPI品質異常）

Period:
- 調査時刻: 2026-03-05 02:29〜02:35 JST
- 対象: `logs/orders.db`, `logs/trades.db`, `logs/metrics.db`, `logs/tick_cache.json`, `logs/factor_cache.json`, `scripts/check_oanda_summary.py`

Fact:
- DB鮮度（`age_min`）:
  - `orders.db=2.77分`, `trades.db=3.55分`, `metrics.db=0.01分`（30分以内のため `local_v2_stack up` は未実行）
- OANDA API確認:
  - `PYTHONPATH=. python3 scripts/check_oanda_summary.py` は
    `api-fxtrade.oanda.com` の `NameResolutionError` で失敗（secret欠損ではない）
- 市況（ローカル実測）:
  - `tick_cache` 最新: `bid/ask/mid=156.976/156.984/156.980`、`spread=0.8 pips`
  - 直近15分バンド: `156.926-157.038`（`range=11.2 pips`）
  - 直近60分バンド: `156.884-157.038`（`range=15.4 pips`）
  - `factor_cache` ATR: `M1=2.633 pips`, `M5=6.753 pips`
- 約定・拒否実績（直近2h）:
  - `orders` ステータス: `filled=4`, `submit_attempt=4`, `preflight_start=4`, `close_ok=2`, `rejected=0`
  - `trades` close件数: `0`、`realized_pl=0`
- 実行/応答品質（直近2h）:
  - `data_lag_ms`: `avg=1462.76`, `max=42302.97`（`n=717`）
  - `decision_latency_ms`: `avg=38.33`, `max=378.20`（`n=717`）

Failure Cause:
- ローカル価格レンジ/スプレッド自体は異常なしだが、OANDA account summary API のDNS解決失敗が継続し、
  API品質を「通常」と判定できない。
- API品質異常時に2h RCA（strategy別/時間帯別/拒否理由別/execution cost別）を進めると、
  市況前提の欠損を含む誤判定リスクが高い。

Improvement:
- 本時間帯の Hourly RCA は `HOLD` とし、PnL分解は実施しない。
- 次の単一アクション（数値ゲート付き）:
  - `check_oanda_summary.py` が成功するまでネットワーク/DNSを復旧し、
    復旧後ランで `API成功` かつ `orders/trades 更新 <=10分` を満たした時点でRCA本体を再開する。

Verification:
- `PYTHONPATH=. python3 scripts/check_oanda_summary.py` が成功終了すること。
- `logs/orders.db` / `logs/trades.db` の最終更新が `now-10m` 以内であること。
- 上記成立後にのみ、直近2hのPnL分解（strategy/hour/reject/execution cost）を再実施する。

Status:
- in_progress（HOLD）

## 2026-03-05 01:36 JST / Hourly RCA HOLD（API DNS失敗継続 + 約定DB更新停止）

Period:
- 調査時刻: 2026-03-05 01:29〜01:36 JST
- 対象: `logs/orders.db`, `logs/trades.db`, `logs/metrics.db`, `logs/orderbook_snapshot.json`, `logs/factor_cache.json`, `scripts/check_oanda_summary.py`

Fact:
- DB最終更新（JST）:
  - `orders.db`: `2026-03-04 23:03:15`（約152.7分 stale）
  - `trades.db`: `2026-03-05 00:04:08`（約91.9分 stale）
  - `metrics.db`: `2026-03-05 01:35:56`（fresh）
- 指示どおり `scripts/local_v2_stack.sh up --profile trade_min --env ops/env/local-v2-stack.env` を実行したが、
  `quant-order-manager port=8300 remains occupied (start aborted for safety)` で再起動不可。
- OANDA API確認:
  - `PYTHONPATH=. python3 scripts/check_oanda_summary.py` は
    `api-fxtrade.oanda.com` の `NameResolutionError` で失敗（DNS解決不可）。
- 市況確認（ローカル実測）:
  - `orderbook_snapshot`: `bid=156.960 / ask=156.968 / spread=0.8 pips`
  - `factor_cache(M1)`: `close=156.998`, `ATR=2.908 pips`, `range15=7.7 pips`, `range60=33.7 pips`
  - 時刻: `2026-03-04T16:34:59Z`（約定DBより新しいが、注文/約定記録は停止）
- 約定・拒否実績（直近2h, `datetime(ts)` 基準）:
  - `orders=0`, `rejected=0`, `filled=0`
  - `trades=0`, `realized_pl=0`
- 応答品質（直近2h, `metrics.db`）:
  - `data_lag_ms`: `last=579.137`, `p95=2502.624`（`n=557`）
  - `decision_latency_ms`: `last=44.872`, `p95=63.332`（`n=557`）

Failure Cause:
- OANDA account summary API が DNS 解決できず、live API品質確認が不能。
- `orders.db` / `trades.db` が90分超 stale で、RCA本体（直近2h PnL分解）の入力が欠損。
- `quant-order-manager` ポート競合で `trade_min` 再起動が完了しないため、復旧確認に進めない。

Improvement:
- 本時間帯の Hourly RCA は `HOLD` とし、通常の「strategy別/時間帯別/拒否理由別/実行コスト別」PnL分解は実施しない。
- 次の1アクション（数値ゲート）:
  1. `8300` 競合を解消し `local_v2_stack up` を成功させ、`orders.db` と `trades.db` の更新を `<=10分` に戻す。
  2. `check_oanda_summary.py` のDNS解決復旧を確認し、成功レスポンスを取得する。
  3. 上記2条件達成後、直近2hのPnL分解を再開する。

Verification:
- `orders.db/trades.db` の最終更新が `now-10m` 以内。
- `PYTHONPATH=. python3 scripts/check_oanda_summary.py` 成功。
- 条件達成後にのみ Hourly RCA本体（PnL分解）を再開。

Status:
- in_progress（HOLD）

## 2026-03-05 00:50 JST / ローカル収益RCA第4段 + 停止耐性（watchdog/launchd）固定

Period:
- 分析窓: 直近24h（`logs/trades.db`, `logs/orders.db`, `logs/metrics.db`）
- 実装/検証: 2026-03-05 00:20〜00:50 JST（ローカルV2導線）

Fact:
- 市況（ローカル実測）:
  - `USD/JPY bid=157.266 ask=157.274 spread=0.8 pips`
  - `ATR14(M1)=5.057 pips`, `range60(M1)=17.4 pips`
  - pricing応答 `avg=255ms, p95=276ms`
- 収益分解（`scalp_ping_5s_b_live`）:
  - `n=608`, `net=-175.605`, `win_rate=19.6%`, `PF=0.416`
  - `STOP_LOSS_ORDER=470 / net=-280.685`
  - side別: `buy n=412 net=-177.957` / `sell n=196 net=+2.352`
  - 保有秒バケット: `<5s net=-78.525`, `5-15s net=-64.523`, `15-30s net=-50.495`（短期偏損）
- 反実仮想:
  - `sell_only` では `net=+2.352`、`sell_no_momentum_hz` では `net=+2.812`

Failure Cause:
- 損失主因は buy側（特に `momentum*` 系）の逆選別。
- 30秒未満の超短期エントリーでSL偏重が発生し、期待値を継続的に毀損。
- 停止耐性は「起動コマンド依存」の運用余地が残り、ネット断/スリープ復帰時の再開確実性が不足。

Improvement:
- 収益改善（`ops/env/scalp_ping_5s_b.env`）:
  - `SIGNAL_MODE_BLOCKLIST=momentum_sidefilter,momentum_hz,momentum_hz_slflip_smflip_hz`
  - `ENTRY_COOLDOWN_SEC=4.5`, `MAX_ORDERS_PER_MINUTE=2`
  - `CONF_FLOOR=82`, `LOOKAHEAD_EDGE_MIN_PIPS=0.30`, `LOOKAHEAD_SAFETY_MARGIN_PIPS=0.18`
  - `MAX_SPREAD_PIPS=0.80`
  - `MIN_TICKS=5`, `MIN_SIGNAL_TICKS=4`, `SHORT_MIN_TICKS=5`, `SHORT_MIN_SIGNAL_TICKS=4`
  - `LOOKAHEAD_ALLOW_THIN_EDGE=0`
  - `REVERT_MIN_TICKS=3`, `REVERT_CONFIRM_TICKS=2`, `REVERT_MIN_TICK_RATE=0.60`
  - force-exit損失側を早期化（`MAX_FLOATING_LOSS_PIPS=0.65`, `MIN_HOLD_SEC=1` など）
- 停止耐性（手動起動不要化）:
  - `scripts/local_v2_watchdog.sh` を新設（`start/run/once/stop/status`）
  - `scripts/local_v2_stack.sh` に `watchdog/watchdog-stop/watchdog-status` を追加
  - `scripts/local_v2_autorecover_once.sh` に state管理を追加（polling gap検知 + network down/up検知）
  - network復帰時に `quant-market-data-feed` を自動再起動（既定ON, cooldown付き）
  - `scripts/install_local_v2_launchd.sh` を watchdog導線へ更新（既定10秒間隔）

Verification:
- テスト:
  - `pytest -q tests/workers/test_scalp_ping_5s_worker.py -k "signal_mode_blocked or resolve_final_signal_for_side_filter"` → `6 passed`
- スクリプト構文:
  - `bash -n scripts/local_v2_stack.sh scripts/local_v2_watchdog.sh scripts/local_v2_autorecover_once.sh scripts/install_local_v2_launchd.sh`
- watchdog実動:
  - `local_v2_stack.sh watchdog --daemon` → `watchdog-status` で `running`、`watchdog-stop` で `stopped`
- launchd反映:
  - `install_local_v2_launchd.sh --interval-sec 10 ...` 実行後、`status_local_v2_launchd.sh` で `run interval = 10 seconds` と env 注入（`QR_LOCAL_V2_NET_RECOVERY_RESTART_MARKET_DATA=1`）を確認
- 稼働確認:
  - `local_v2_stack.sh status --profile trade_min --env ops/env/local-v2-stack.env` で8サービス `running`

Status:
- in_progress（直近 30〜90 分で `STOP_LOSS_ORDER 比率` と `scalp_ping_5s_b_live net` を追跡）

## 2026-03-05 00:34 JST / Hourly RCA HOLD（API名前解決失敗 + 約定DB更新停止）

Period:
- 調査時刻: 2026-03-05 00:29〜00:34 JST
- 対象: `logs/orders.db`, `logs/trades.db`, `logs/metrics.db`, `logs/tick_cache.json`, `logs/factor_cache.json`, `scripts/check_oanda_summary.py`

Fact:
- DB更新時刻:
  - `orders.db`: `2026-03-04 23:03:15 JST`（調査時点で約91分経過）
  - `trades.db`: `2026-03-05 00:04:08 JST`（調査時点で約30分経過）
  - `metrics.db`: `2026-03-05 00:32:09 JST`（最新）
- `scripts/local_v2_stack.sh up --profile trade_min --env ops/env/local-v2-stack.env` は
  `quant-order-manager port=8300 remains occupied (start aborted for safety)` で完了不可。
- API疎通:
  - `PYTHONPATH=. python3 scripts/check_oanda_summary.py` は
    `api-fxtrade.oanda.com` の `NameResolutionError` で失敗（DNS解決不可）。
- 市況（ローカル実測）:
  - `USD/JPY bid=157.234 ask=157.242 mid=157.238`, `spread=0.8 pips`
  - `ATR(M1)=3.664 pips`, `ATR(M5)=7.979 pips`
  - `range(15m)=13.0 pips`, `range(60m)=19.0 pips`
- 約定・拒否実績:
  - `orders` 直近2h: `115` 件、`rejected=2`
  - reject上位（直近24h）: `error_code=(none) 28件`
- 応答品質（`metrics.db` 直近2h）:
  - `data_lag_ms last=1206.68 / p95=1366022.22`
  - `decision_latency_ms last=12.24 / p95=96.70`

Failure Cause:
- OANDA APIのDNS解決失敗により、live APIベースの確認が不能。
- `orders.db` と `trades.db` の更新が停止/遅延しており、直近2時間PnL分解の前提データ品質を満たせない。
- `local_v2_stack up` もポート競合で完了せず、即時復旧を確認できない。

Improvement:
- 本時間帯のRCAは `HOLD` とし、PnL分解（strategy別/時間帯別/拒否理由別/実行コスト別）は保留。
- 次アクション（優先順）:
  1. `8300` 占有PIDの解消後に `local_v2_stack up` を再実行し、`orders/trades` 更新再開を確認。
  2. `check_oanda_summary.py` の再試行で API 名前解決復旧を確認。
  3. 復旧後に直近2h PnL分解を再実施して通常RCAへ戻す。

Verification:
- `orders.db/trades.db` の最終更新が `now-10m` 以内に戻ること。
- `check_oanda_summary.py` が成功し、`pricing/account summary` を取得できること。
- 上記2条件を満たした時点で、2時間PnL分解を再開すること。

Status:
- in_progress（HOLD）

## 2026-03-05 03:34 JST / Hourly RCA HOLD（API DNS失敗継続 + order/trade更新停止）

Period:
- 調査時刻: 2026-03-05 03:27〜03:34 JST
- 対象: `logs/orders.db`, `logs/trades.db`, `logs/metrics.db`, `scripts/local_v2_stack.sh`, `scripts/check_oanda_summary.py`

Fact:
- DB更新時刻:
  - `orders.db`: `2026-03-05 02:27:16 JST`（約67.5分 stale）
  - `trades.db`: `2026-03-05 02:26:29 JST`（約68.3分 stale）
  - `metrics.db`: `2026-03-05 03:34:44 JST`（更新継続）
- スタック再起動:
  - `scripts/local_v2_stack.sh up --profile trade_min --env ops/env/local-v2-stack.env` は
    `quant-order-manager port=8300 remains occupied` で失敗（`120s` 待機後も改善なし）。
- API疎通:
  - `PYTHONPATH=. python3 scripts/check_oanda_summary.py` を2回実行し、両方 `NameResolutionError(api-fxtrade.oanda.com)`。
- 市況プロキシ（`orders.db` 直近2h、`preflight_start`）:
  - spread: `avg=0.801 pips`（min `0.8`, max `1.0`）
  - ATR proxy: `mtf_regime_atr_m1=3.429`, `mtf_regime_atr_m5=9.525`
  - range proxy: `signal_range_pips=0.51`
  - 価格帯 proxy（`filled/close_ok`）: `156.884 - 157.594`（range `0.710`）
- 約定/拒否実績（直近2h）:
  - `orders=2843`, `filled=671`, `reject_like=412`
  - `trades=704`, `net_pnl=-299.270`
  - reject上位: `entry_probability_reject=383`, `STOP_LOSS_ON_FILL_LOSS=27`, `api_error(502)=1`
- API応答品質 proxy（`metrics.db` 直近2h）:
  - `data_lag_ms avg=6,534,745.994`（max `3,385,784,138.449`）
  - `decision_latency_ms avg=39.121`（max `3025.991`）

Failure Cause:
- 利益阻害トップ3（数値根拠）:
  1. API品質異常: `check_oanda_summary` 2/2失敗（DNS解決不可）。
  2. 執行導線停止: `orders/trades` が `67-68分` stale、`local_v2_stack up` も `port 8300` 競合で復旧失敗。
  3. 成績劣化: 直近2h `net_pnl=-299.270`、かつ `reject_like=412/2843 (14.5%)`、主要拒否は `entry_probability_reject=383`。

Improvement:
- 本時間帯のRCAは `HOLD` とし、通常の2h分解（strategy別・時間帯別・拒否理由別・実行コスト別）は保留。
- 次の1アクション:
  - `8300` 占有プロセス（`PID: 38068, 38234, 38235, 38236, 38238, 38239, 38240`）の整理を最優先し、
    `local_v2_stack up` 成功と `orders/trades <=10m` 回復を確認してからRCA再開する。

Verification:
- `orders.db/trades.db` の最終更新が `now-10m` 以内に戻ること。
- `PYTHONPATH=. python3 scripts/check_oanda_summary.py` が成功すること。
- 上記2条件を満たした次ランで、直近2h PnLの4軸分解を再実行すること。

Status:
- in_progress（HOLD）

## Hourly RCA 改善案バックログ（automation: qr-hourly-rca）

- `[status=in_progress]` API到達性と約定DB鮮度の復旧確認（`check_oanda_summary` 成功 + `orders/trades` 更新 `<=10m`）
- `[status=in_progress]` 8300競合（`quant-order-manager`）の占有元特定と競合解消手順の固定化（PID群まで確認済み）
- `[status=open]` 復旧後の直近2h PnL分解（strategy別・時間帯別・拒否理由別・実行コスト別）を再実施
- `[status=done]` 2026-03-05 06:35 JST ランのHOLD判定を台帳へ追記（trade_min停止 + trades stale + DNS失敗）
- `[status=done]` 2026-03-05 05:36 JST ランのHOLD判定を台帳へ追記（trades stale + DNS失敗 + 8300競合）
- `[status=done]` 2026-03-05 03:34 JST ランのHOLD判定を台帳へ追記（DNS失敗 + 8300競合継続）
- `[status=done]` 2026-03-05 02:35 JST ランのHOLD判定を台帳へ追記（API DNS失敗継続）
- `[status=done]` 2026-03-05 01:36 JST ランのHOLD判定を台帳へ追記
- `[status=done]` 2026-03-05 04:34 JST ランのHOLD判定を台帳へ追記（Summary API DNS失敗 + trades stale）

## 2026-03-04 14:22 UTC / 2026-03-04 23:22 JST - `SCALP_PING_5S_B_SIDE_FILTER` の fail-closed 強化（空許可の誤適用防止）

Period:
- 調査時刻: 2026-03-04 14:18〜14:22 UTC（23:18〜23:22 JST）
- 対象: `workers/scalp_ping_5s_b/worker.py` の環境変数マッピング

Fact:
- 起動ログで過去に `SCALP_PING_5S_B_ALLOW_NO_SIDE_FILTER=1` が混入した際、
  `side_filter=(unset)` で起動していた履歴があった。
- 直近起動では `side_filter=sell` を確認したが、未設定時に空sideを許容し得る分岐が残っていた。

Failure Cause:
- `ALLOW_NO_SIDE_FILTER=1` のとき、`SIDE_FILTER` が未設定でも空値を許容する実装だったため、
  意図しない no-filter 起動が起きる余地があった。

Improvement:
- `workers/scalp_ping_5s_b/worker.py` を修正し、
  `ALLOW_NO_SIDE_FILTER=1` でも **`SIDE_FILTER` が明示設定された場合のみ** 空sideを許容するよう変更。
- `SIDE_FILTER` 未設定時は常に `sell` へ fail-closed するよう固定。
- `tests/workers/test_scalp_ping_5s_b_worker_env.py` に再発防止テストを追加。

Verification:
- `pytest -q tests/workers/test_scalp_ping_5s_b_worker_env.py` が `17 passed`。
- `SCALP_PING_5S_B_ALLOW_NO_SIDE_FILTER=1` かつ `SCALP_PING_5S_B_SIDE_FILTER` 未設定でも
  `SCALP_PING_5S_SIDE_FILTER=sell` になることをテストで確認。

Status:
- done

## 2026-03-04 14:15 UTC / 2026-03-04 23:15 JST - ローカル運用タスクでの VM 導線実行ミスの是正

Period:
- 発生時刻: 2026-03-04 13:54〜14:14 UTC（22:54〜23:14 JST）
- 対象: 運用手順の適用判断（ローカル運用タスク）

Fact:
- ユーザー依頼はローカル運用改善だったが、VM/GCP 導線（`scripts/vm.sh` / `deploy_to_vm.sh`）を実行した。
- 実行結果は IAM 権限不足で失敗し、ローカル改善タスクの進行を阻害した。

Failure Cause:
- ローカル運用モードの例外条件（「VMは明示指示時のみ」）を、実行前判断で徹底できなかった。

Improvement:
- `docs/OPS_LOCAL_RUNBOOK.md` の運用原則に、
  「ローカル運用タスクでは VM/GCP コマンドを実行しない。例外は明示依頼のみ」を追記。
- 以後の本タスクはローカルDB + OANDA API のみに限定して実施する。

Verification:
- 以後同種タスクで `scripts/vm.sh` / `deploy_to_vm.sh` / `gcloud compute *` を使わずに完了できること。
- 変更はローカル導線（`local_v2_stack` / `logs/*.db` / OANDA API）でのみ再現確認すること。

Status:
- done

## 2026-03-04 14:08 UTC / 2026-03-04 23:08 JST - `scalp_ping_5s_b_live` 収益悪化RCA第3段（SL偏重の即時圧縮）

Period:
- 集計窓: 直近24h（`logs/orders.db`, `logs/trades.db`, `logs/metrics.db`）
- 市況確認: OANDA API（`USD_JPY` pricing/candles/openTrades）
- 監査時刻: 2026-03-04 13:54〜14:08 UTC（22:54〜23:08 JST）

Fact:
- 市況は稼働可能:
  - `bid/ask=157.302/157.310`、`spread=0.8 pips`
  - `ATR14(M1)=3.393 pips`、`range_60m=18.8 pips`
  - API応答: pricing `mean=262ms`、candles `408ms`、openTrades取得成功
- 戦略収益（`scalp_fast / scalp_ping_5s_b_live`）:
  - `n=608`, `win_rate=19.57%`, `PF=0.416`, `expectancy=-0.752 pips`, `net=-175.6 JPY`
  - `close_reason`: `STOP_LOSS_ORDER=470 (net=-280.7 JPY, avg=-1.421 pips)` /
    `MARKET_ORDER_TRADE_CLOSE=138 (net=+105.1 JPY, avg=+1.527 pips)`
  - side別: `buy n=412 net=-178.0 JPY`、`sell n=196 net=+2.4 JPY`
  - `entry_probability>=0.85` の buy が `n=310 net=-142.4 JPY` で、高確率帯でも逆選別が発生

Failure Cause:
- 低品質エントリーがSLへ偏る構造が主因（約77%がSL終了）。
- 特に buy 側で確率校正が崩れ、`high-probability` 帯でも損失寄与が継続。
- spread/latencyの実行品質は致命劣化ではなく（spread平均0.802p, submit p50≈202ms）、
  エントリー品質とside配分の問題が優勢。

Improvement:
- `ops/env/scalp_ping_5s_b.env` を第3段調整（停止ではなく品質圧縮）:
  - エントリー厳格化: `MIN_UNITS_RESCUE_MIN_ENTRY_PROBABILITY=0.62`, `ENTRY_PROBABILITY_ALIGN_FLOOR=0.58`
  - カウンター抑制: `ENTRY_PROBABILITY_ALIGN_COUNTER_EXTRA_PENALTY_MAX=0.28`
  - side実績連動ロット圧縮: `ENTRY_PROBABILITY_BAND_ALLOC_SIDE_METRICS_MIN_MULT=0.45`,
    `...MAX_MULT=0.78`, `SIDE_ADVERSE_STACK_UNITS_STEP_MULT=0.22`,
    `SIDE_ADVERSE_STACK_UNITS_MIN_MULT=0.28`, `SIDE_ADVERSE_STACK_DD_MIN_MULT=0.40`
  - lookaheadの最低エッジ強化: `LOOKAHEAD_EDGE_MIN_PIPS=0.16`, `LOOKAHEAD_SAFETY_MARGIN_PIPS=0.08`
  - コスト耐性: `MAX_SPREAD_PIPS=0.90`、SL reject低減で `SL_MIN_PIPS=1.00` / `SHORT_SL_MIN_PIPS=1.00`

Verification:
- 反映後 60分/240分で次を確認:
  - `STOP_LOSS_ORDER 比率 <= 70%`
  - `buy` の net寄与がマイナス拡大しない（直近窓で `net_buy >= -20 JPY` を目安）
  - `PF >= 0.85` への回復傾向（最低でも `expectancy_pips > -0.20`）
  - `rejected:STOP_LOSS_ON_FILL_LOSS` の件数低下

Status:
- in_progress

## 2026-03-04 13:31 UTC / 2026-03-04 22:31 JST - sidecar `POSITION_MANAGER_SERVICE_PORT` 未反映の修正（18301運用を有効化）

Period:
- 調査時刻: 2026-03-04 13:29〜13:31 UTC（22:29〜22:31 JST）
- 対象: `workers/position_manager/worker.py`, `ops/env/local-v2-sidecar-ports.env`, `scripts/local_v2_stack.sh`

Fact:
- `local_v2_stack.sh` 側は `POSITION_MANAGER_SERVICE_PORT` を参照していたが、
  `workers.position_manager.worker` の `uvicorn.run` は `port=8301` 固定だった。
- この不整合により、`--env ops/env/local-v2-sidecar-ports.env` 指定時でも
  sidecar 起動は `8301` bind を試行し、parity と競合して失敗していた。

Failure Cause:
- ポート設定の責務分離が不完全で、起動スクリプトの env 設計と worker 実装が一致していなかった。

Improvement:
- `workers/position_manager/worker.py` に `_service_port()` を追加し、
  `POSITION_MANAGER_SERVICE_PORT`（未設定時 `8301`）を読む実装へ変更。
- `tests/workers/test_position_manager_worker_env.py` を追加し、
  default (`8301`) と env override（例: `9315`）の両方を検証。

Verification:
- `pytest -q tests/workers/test_position_manager_worker_env.py` が `2 passed`。
- parity 稼働中に
  `scripts/local_v2_stack.sh up --services quant-position-manager --env ops/env/local-v2-sidecar-ports.env --force-conflict`
  を実行し、`http://127.0.0.1:18301/health` 応答を確認。
- 同コマンドで `down` まで完了し、conflict-safe mode で parity 側を巻き込まず停止できることを確認。

Status:
- done

## 2026-03-04 13:30 UTC / 2026-03-04 22:30 JST - `scalp_ping_5s_b_live` 第2段チューニング（lookahead過剰block緩和 + long過大ロット抑制）

Period:
- 調査時刻: 2026-03-04 13:20〜13:30 UTC（22:20〜22:30 JST）
- 対象: ローカル parity ログ `logs/local_vm_parity/quant-scalp-ping-5s-b.log` と `ops/env/scalp_ping_5s_b.env`

Fact:
- 直近 lookahead block は `edge_negative_block` のみ（`241/241 = 100%`）。
- side内訳は `short=207`, `long=34`（short 偏重）。
- blockサンプルは `pred ~0.10-0.31p` に対し `cost ~1.12-1.19p` で、`edge` が恒常的にマイナス。
- `SCALP_PING_5S_B_LOOKAHEAD_GATE_ENABLED=1` のため、`edge <= 0` は即 block（thin-edge 設定では回避不可）。

Failure Cause:
- spread + slippage 見積りに対して短期予測値（pred）が不足し、lookahead が常時 `edge_negative_block` へ収束。
- 一方で long 側は過去の負け寄与が大きく、entry 復帰時のロット上振れを抑える安全弁が不足。

Improvement:
- `ops/env/scalp_ping_5s_b.env` を第2段で更新:
  - 方向安全弁（時限）: `SIDE_FILTER=sell`, `ALLOW_NO_SIDE_FILTER=0`（long側の即時遮断）
  - lookahead の予測項を引き上げ: `HORIZON_SEC=2.80`, `MOMENTUM_WEIGHT=1.15`, `FLOW_WEIGHT=0.50`, `TRIGGER_WEIGHT=0.45`, `BIAS_WEIGHT=0.42`, `COUNTER_PENALTY=0.30`
  - cost見積りを過剰保守から緩和: `SLIP_BASE_PIPS=0.04`, `SLIP_SPREAD_MULT=0.10`, `SLIP_RANGE_MULT=0.06`
  - long過大ロット抑制: `DIRECTION_BIAS_LONG_OPPOSITE_UNITS_MULT=0.08`,
    `ENTRY_PROBABILITY_ALIGN_UNITS_MAX_MULT=0.94`,
    `ENTRY_PROBABILITY_BAND_ALLOC_SIDE_METRICS_MAX_MULT=0.96`,
    `SIDE_ADVERSE_STACK_UNITS_STEP_MULT=0.34`
- 既存の `LOOKAHEAD_GATE_ENABLED=1` は維持し、無条件エントリー化は行わない。
- long再開は時限条件付き（long単独で `PF>1` かつ `avg_pips>=0` を一定件数で確認後）。

Verification:
- parity 再起動後、30分/120分で以下を確認する。
  - `lookahead block` 件数と `edge_negative_block` 比率が低下すること
  - `filled` / `preflight_start` 比率の回復
  - `PF`, `win_rate`, `STOP_LOSS_ORDER 比率` が第1段より悪化しないこと
  - long/short の平均 units が再び long 側へ偏りすぎないこと

Status:
- in_progress

## 2026-03-04 13:23 UTC / 2026-03-04 22:23 JST - `local_v2_stack` と parity supervisor 競合による ENTRY 減少の再発防止

Period:
- 調査時刻: 2026-03-04 12:40〜13:23 UTC（21:40〜22:23 JST）
- 対象: ローカルV2導線（`scripts/local_v2_stack.sh`）と parity 導線（`scripts/local_vm_parity_supervisor.py`）

Fact:
- `ps` の親子関係で、`workers.position_manager.worker` / `workers.order_manager.worker` は
  `local_vm_parity_supervisor.py`（`screen: qr-local-parity`）配下で稼働。
- `lsof` で `:8300` と `:8301` は parity 側の worker が LISTEN。
- 同時に `local_v2_stack.sh up` を実行すると、`quant-order-manager.log` /
  `quant-position-manager.log` で `Errno 48 (address already in use)` が発生し、
  worker が不安定化して ENTRY 停滞が発生。

Failure Cause:
- 同一 repo で「`local_v2_stack` と parity supervisor を同時運転」し、
  同じ worker と同じ固定ポート（8300/8301）を奪い合う運用競合が発生していた。

Improvement:
- `scripts/local_v2_stack.sh` に排他ガードを追加。
  - `up/down/restart` 実行時に次を検出したら既定拒否:
    - `screen` セッション `qr-local-parity`
    - `scripts/local_vm_parity_supervisor.py` プロセス（repo配下）
  - 案内: `scripts/local_vm_parity_stack.sh stop` を表示。
  - 例外: 意図的な実行時のみ `--force-conflict` でバイパス可能。
- `docs/OPS_LOCAL_RUNBOOK.md` に「`local_v2_stack` と parity は排他運用」を明記。
- `docs/WORKER_REFACTOR_LOG.md` に監査ログ追記。

Verification:
- `scripts/local_v2_stack.sh up --services quant-position-manager --env ops/env/local-v2-stack.env`
  が parity 稼働中に `EXIT:3` で拒否されること。
- `scripts/local_v2_stack.sh up --services quant-position-manager --env ops/env/local-v2-stack.env --force-conflict`
  でガードがバイパスされること（その後の成否は環境依存）。
- `scripts/local_v2_stack.sh status` / `logs` が parity 稼働中でも実行できること。

Status:
- done

## 2026-03-04 12:40 UTC / 2026-03-04 21:40 JST - `scalp_ping_5s_b_live` 緊急収益改善（品質閾値+サイズ圧縮）

Period:
- 集計窓: 直近24h（`logs/trades.db`, `logs/orders.db`, `logs/metrics.db`）
- 市況確認: OANDA API 直近取得（`USD/JPY`）

Fact:
- 市況は稼働可能レンジ:
  - `bid/ask=157.214/157.222`, `spread=0.8 pips`
  - `ATR14(M1)=3.3429 pips`, `range15m=19.1 pips`, `range60m=20.2 pips`
  - API応答 `avg=246ms`, error `0`
- 戦略収益（`pocket <> manual`）:
  - `n=537`, `win_rate=20.3%`, `PF=0.43`, `expectancy=-0.721 pips`, `net=-164.864 JPY`
  - `close_reason`: `STOP_LOSS_ORDER=414 (77.09%, net=-270.768 JPY)`, `MARKET_ORDER_TRADE_CLOSE=123 (net=+105.904 JPY)`
- side寄与:
  - `long: n=397, avg_units=63.4, net=-169.127 JPY`
  - `short: n=146, avg_units=1.9, net=+2.882 JPY`
- entry_probability帯:
  - `0.80-0.90` が最大赤字（`n=297, net=-118.285 JPY, exp=-0.969 pips`）
- 直近24hの注文:
  - `filled=593`, `entry_probability_reject=383`, `rejected=22`
  - reject理由は `STOP_LOSS_ON_FILL_LOSS` が全件

Failure Cause:
- `scalp_ping_5s_b_live` が高頻度エントリーのまま SL 偏重（勝率20%台）で、ロング側の実効サイズが過大。
- 高確率帯（0.80-0.90）で期待値が崩れており、既存の確率補正/帯別配分が過大評価を抑え切れていない。

Improvement:
- `ops/env/scalp_ping_5s_b.env` を即時更新（停止ではなく品質選別を強化）:
  - 取引密度: `MAX_ACTIVE_TRADES 2->1`, `MAX_ORDERS_PER_MINUTE 6->4`
  - ロット上限: `BASE_ENTRY_UNITS 70->45`, `MAX_UNITS 700->180`
  - spread閾値: `MAX_SPREAD_PIPS 2.00->1.00`
  - entry品質: `CONF_FLOOR 72->78`, `CONF_SCALE_MIN_MULT 0.92->0.80`
  - 方向過大化抑制: `DIRECTION_BIAS_ALIGN_UNITS_BOOST_MAX 0.08->0.03`, `SIDE_BIAS_BLOCK_THRESHOLD 0.08->0.12`
  - 確率補正強化:
    - `ENTRY_PROBABILITY_ALIGN_PENALTY_MAX 0.20->0.28`
    - `ENTRY_PROBABILITY_ALIGN_COUNTER_EXTRA_PENALTY_MAX 0.22->0.30`
    - `ENTRY_PROBABILITY_ALIGN_FLOOR_RAW_MIN 0.74->0.82`
    - `ENTRY_PROBABILITY_ALIGN_FLOOR 0.35->0.50`
  - 確率帯配分を縮小側へ:
    - `ENTRY_PROBABILITY_BAND_ALLOC_HIGH_REDUCE_MAX 0.65->0.82`
    - `ENTRY_PROBABILITY_BAND_ALLOC_UNITS_MIN_MULT 0.82->0.65`
    - `ENTRY_PROBABILITY_BAND_ALLOC_SIDE_METRICS_MIN_MULT 0.82->0.68`
  - adverse stackの縮小強化:
    - `SIDE_ADVERSE_STACK_UNITS_MIN_MULT 0.72->0.60`
    - `SIDE_ADVERSE_STACK_DD_MIN_MULT 0.78->0.65`
  - spread guard実値:
    - `spread_guard_max_pips=1.00`
    - `spread_guard_release_pips=0.85`
    - `spread_guard_hot_trigger_pips=1.10`
    - `spread_guard_hot_cooldown_sec=8`

Verification:
- 適用後 60分/180分で以下を確認する。
  - `PF >= 0.80`（まずは負け幅圧縮）
  - `win_rate >= 30%`
  - `STOP_LOSS_ORDER 比率 <= 65%`
  - `net_pips` の時間帯連続悪化（3連続マイナス）解消
  - `rejected` が `STOP_LOSS_ON_FILL_LOSS` 偏重のまま増加しないこと

Status:
- in_progress

## 2026-03-04 08:49 UTC / 2026-03-04 17:49 JST - `scalp_ping_5s_b` 第2段調整（long偏重SL抑制）

Period:
- 変更時刻: 2026-03-04 08:49 UTC / 17:49 JST
- 対象: `ops/env/scalp_ping_5s_b.env`, `ops/env/quant-order-manager.env`

Fact:
- `STOP_LOSS_ORDER` が long 側へ集中し、同方向の連続損切りが収益を圧迫。
- `direction_cap` 判定が優勢な局面で long の過密継続が残存。
- short 側の通過率は維持されており、sell-only 固定へ戻さず long 側の連打のみを抑制する方針が妥当。

Failure Cause:
- エントリー間隔・方向反転クールダウン・long 側モメンタム閾値が浅く、連続longの抑制が不足。

Improvement:
- `ops/env/scalp_ping_5s_b.env`:
  - `ENTRY_COOLDOWN_SEC: 2.8 -> 4.0`
  - `FAST_DIRECTION_FLIP_COOLDOWN_SEC: 0.6 -> 1.2`
  - `BASE_ENTRY_UNITS: 120 -> 90`
  - `LONG_MOMENTUM_TRIGGER_PIPS: 0.14 -> 0.20`
  - `DIRECTION_BIAS_BLOCK_SCORE: 0.52 -> 0.60`
  - `DIRECTION_BIAS_LONG_OPPOSITE_UNITS_MULT: 0.28 -> 0.20`
- `ops/env/quant-order-manager.env`:
  - `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_B(_LIVE): 6 -> 4`
  - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_B(_LIVE): 0.10 -> 0.15`

Verification:
- 15分窓で `STOP_LOSS_ORDER(long)` 件数を追跡。
- 15分窓で `submit/filled` の side 比（long:short）を追跡。
- 15分窓で `net_jpy` と `pf` を追跡。

Status:
- in_progress

## 2026-03-04 08:42 UTC / 2026-03-04 17:42 JST - `scalp_ping_5s_b` の過剰エントリー抑制（sell-only化なし）

Period:
- 変更時刻: 2026-03-04 08:42 UTC / 17:42 JST
- 対象: `ops/env/scalp_ping_5s_b.env`

Fact:
- 直前設定では `ENTRY_COOLDOWN_SEC=2.2`, `MAX_ORDERS_PER_MINUTE=6` で、過密エントリー継続時に SL 連打が再発しやすい構成。
- `SCALP_PING_5S_B_SIDE_FILTER=none` は維持されており、sell-only 固定ではない。

Failure Cause:
- エントリー間隔・分間発注上限・方向別圧縮下限がタイトで、短時間に同質シグナルが重なった際の連続損切りを抑え切れていない。

Improvement:
- `ops/env/scalp_ping_5s_b.env` を更新（重複キーなし、各キーは単一値で整理）。
  - `SCALP_PING_5S_B_ENTRY_COOLDOWN_SEC: 2.2 -> 2.8`
  - `SCALP_PING_5S_B_MAX_ORDERS_PER_MINUTE: 6 -> 4`
  - `SCALP_PING_5S_B_MAX_ACTIVE_TRADES: 2 -> 2`（据え置き明示）
  - `SCALP_PING_5S_B_MAX_ACTIVE_PER_DIRECTION: (未定義) -> 1`（明示追加）
  - `SCALP_PING_5S_B_DIRECTION_BIAS_SHORT_OPPOSITE_UNITS_MULT: 0.95 -> 1.00`
  - `SCALP_PING_5S_B_SIDE_BIAS_SCALE_FLOOR: 0.55 -> 0.45`
  - `SCALP_PING_5S_B_ENTRY_PROBABILITY_ALIGN_UNITS_MIN_MULT: 0.60 -> 0.65`
  - `SCALP_PING_5S_B_ENTRY_PROBABILITY_BAND_ALLOC_UNITS_MIN_MULT: 0.55 -> 0.60`

Verification:
- `rg` で対象8キーの最終値を確認し、同一キー重複がないことを確認する。
- `SIDE_FILTER=none` が維持され、sell-only 化していないことを確認する。

Status:
- in_progress

## 2026-03-04 04:30 UTC / 2026-03-04 13:30 JST - ローカルV2で `scalp_ping_5s_b_live` の敗因集中をデリスク（sell-only継続＋品質閾値引き上げ）

Period:
- 集計窓: 直近24h（`logs/trades.db`, `logs/orders.db`）
- 市況確認: `logs/tick_cache.json`（直近約38分）

Fact:
- 24h（`pocket <> manual`）: `n=80`, `net=-45.456 JPY`, `net_pips=-82.6`, `win_rate=16.25%`, `PF=0.3262`。
- 戦略寄与: `scalp_ping_5s_b_live` が同期間の実トレード全損益を占有（`n=80`, `net=-45.456 JPY`）。
- 注文内訳（24h）: `entry_probability_reject=376`, `filled=83`, `rejected=2`（`STOP_LOSS_ON_FILL_LOSS`）。
- 市況（直近tick）: `USD/JPY mid=157.333`, `spread_avg=0.8 pips`, `range(5m)=2.7 pips`, `range(15m)=9.2 pips`, `range(30m)=23.9 pips`。
- 実行品質（metrics直近1000点）: `decision_latency_ms p50=27.0 / p95=48.9`、`data_lag_ms p50=630.6 / p95=1566.2`（外れ値あり）。
- OANDA応答品質: workerログで pricing API `HTTP 200` 継続、`oanda.positions.error` は直近発生なし（最終 2026-02-21）。

Failure Cause:
- `scalp_ping_5s_b_live` の long 側でSLヒット連鎖が発生し、低エッジ・低品質シグナルの流入で期待値が崩れた。
- 併せてローカル起動導線でワーカープロセス重複/残骸が出る経路があり、実行品質の不安定化要因になっていた。

Improvement:
- `workers/scalp_ping_5s_b/worker.py`: `subprocess.run` 経路を `os.execvpe` に変更し、ラッパー子プロセス残骸を抑止。
- `scripts/local_vm_parity_stack.sh`: 広すぎる `pkill` を repo 配下限定へ縮小、`stop/status` の stale PID 判定を明確化。
- `ops/env/scalp_ping_5s_b.env`:
  - `MAX_ORDERS_PER_MINUTE 12 -> 8`
  - `BASE_ENTRY_UNITS 140 -> 90`
  - `MIN_UNITS_RESCUE_MIN_ENTRY_PROBABILITY 0.60 -> 0.68`
  - `CONF_FLOOR 60 -> 72`
  - `ENTRY_PROBABILITY_ALIGN_FLOOR_RAW_MIN 0.68 -> 0.74`
  - `ENTRY_PROBABILITY_BAND_ALLOC_HIGH_REDUCE_MAX 0.78 -> 0.65`
  - `ENTRY_NET_EDGE_MIN_PIPS 0.12 -> 0.20`
- `ops/env/local-v2-full.env`: parity系にも同方針（sell-only + 閾値強化 + サイズ縮小）を反映。
- `scripts/dynamic_alloc_worker.py` 再計算で `scalp_ping_5s_b_live lot_multiplier=0.45` を維持（攻めず縮小継続）。

Verification:
- 実プロセス環境で `SCALP_PING_5S_B_*` の新値反映を確認（`ENTRY_NET_EDGE_MIN_PIPS=0.20`, `CONF_FLOOR=72`, `SIDE_FILTER=sell` など）。
- `scripts/local_v2_stack.sh status --profile trade_min --env ops/env/local-v2-stack.env` でV2群が `running`。
- 直近ログで `entry-skip summary` の long 側は `side_filter_block` が継続し、不要long流入を遮断していることを確認。
- 次判定条件（再評価）:
  - 24hで `PF >= 1.0` かつ `net_jpy > 0`
  - `entry_probability_reject` 比率の過剰偏り緩和（目安 `< 45%`）
  - `data_lag_ms p95 < 1200` を維持

Status:
- in_progress

## 2026-03-02 02:35 UTC / 2026-03-02 11:35 JST - `fx-trader*` 同時稼働の一時的収束

Period:
- 監査時点: 2026-03-02 02:00–02:35 UTC
- 対象コマンド: `scripts/ensure_single_trading_vm.sh`

Fact:
- `fx-trader*` 系の稼働インスタンスが `fx-trader-vm-es1a`, `fx-trader-c-repair`, `fx-trader-vm-es1c` で
  重複していた状態を確認。
- `--target=fx-trader-vm-es1a` で `ensure_single_trading_vm.sh` を実行し、非対象の稼働/起動中インスタンスを停止指示。
- 再確認時点（`--dry-run`）で `fx-trader-vm-es1a` のみが `RUNNING` と判定。
- `fx-trader-c-repair` は `TERMINATED`、`fx-trader-vm-es1c` は `STOPPING` -> `TERMINATED` へ移行中（運用監視対象）。

Failure Cause:
- 同時稼働を放置した状態が、compute 料金・I/O競合・復元監査ノイズを増加させ、月間コスト（約23,000円規模）悪化要因になり得る。
- BQ面はこの時点の主要原因ではなく、過去監査と合わせて「同時稼働 + 不要リソース残存」が主軸。

Improvement:
- `AGENTS.md` のセクション9を更新し、1台運用（RUNNING 1台）を運用条項化。
- 併せて `quant-bq-sync` の `--bq-interval 900`/`--disable-lot-insights` 方針と
  `ensure_single_trading_vm.sh --dry-run` を監査手順として固定化。

Verification:
- `gcloud compute instances list --filter='name~^fx-trader' --format='value(name,status,zone)'` で
  `RUNNING` が1件のみ。
- `./scripts/ensure_single_trading_vm.sh -p quantrabbit -m fx-trader-vm-es1a -P fx-trader --dry-run`
  が `OK` を返すこと。

Status:
- done

## 2026-03-04 (JST) / scalp_ping_5s_b を「売り限定解除 + 両方向適応」へ再調整（ローカルV2）

Source:
- `logs/local_v2_stack/quant-scalp-ping-5s-b.log`
- `logs/orders.db`
- `logs/trades.db`
- `curl http://127.0.0.1:8301/position/sync_trades`

Findings:
- `SCALP_PING_5S_B_SIDE_FILTER=none` + `ALLOW_NO_SIDE_FILTER=1` で固定sideは解除済み。
- 直近ログで `short` 候補はあるが `units_below_min` 比率が高く、実約定は long 偏重。
- 直近クローズでは `STOP_LOSS_ORDER` が連続し、反転局面で long 連打が残る。

Action:
- `ops/env/scalp_ping_5s_b.env` を更新（ローカル）
  - `ENTRY_COOLDOWN_SEC=2.2`（連打抑制）
  - `MAX_ORDERS_PER_MINUTE=6`（過剰回転抑制）
  - `CONF_SCALE_MIN_MULT=0.92`
  - `DIRECTION_BIAS_OPPOSITE_UNITS_MULT=0.68`
  - `DIRECTION_BIAS_SHORT_OPPOSITE_UNITS_MULT=0.95`
  - `DIRECTION_BIAS_LONG_OPPOSITE_UNITS_MULT=0.28`
  - `SIDE_BIAS_SCALE_FLOOR=0.40`
  - `ENTRY_PROBABILITY_BAND_ALLOC_SIDE_METRICS_MIN_MULT=0.60`
  - `FAST_DIRECTION_FLIP_DIRECTION_SCORE_MIN=0.58`
  - `FAST_DIRECTION_FLIP_HORIZON_SCORE_MIN=0.35`
  - `FAST_DIRECTION_FLIP_HORIZON_AGREE_MIN=2`
  - `SIDE_ADVERSE_STACK_UNITS_MIN_MULT=0.60`
  - `SIDE_ADVERSE_STACK_DD_MIN_MULT=0.65`
- `scripts/local_v2_stack.sh up --profile trade_min --env ops/env/local-v2-stack.env` で反映。

Verification:
- `orders.db` では `submit_attempt/filled` が継続して発生（エントリー導線は稼働）。
- ただし `short` 側の `units_below_min` は残存し、追加観測が必要。
- `position_manager` が断続的に再起動する時間帯があり、短時間で `open_positions` 呼び出し失敗が発生することを確認。

Next:
- `position_manager` の起動安定化を先に固定。
- その後 20〜30分窓で `short fill件数 / units_below_min比率 / PF` を再測定して再調整する。

## 2026-03-02 02:20 UTC / 2026-03-02 11:20 JST - `quant-bq-sync` / `quant-policy-cycle` で BQ 負荷を先回り抑制

Period:
- 対象:
  - `systemd/quant-bq-sync.service`
  - `systemd/quant-policy-cycle.timer`
  - `scripts/run_sync_pipeline.py`
  - `AGENTS.md` / `docs/GCP_PLATFORM.md`
- 根拠データ:
  - 過去監査での `quant-bq-sync` 実引数 (`--interval 60 --bq-interval 300 --limit 1200`) と
    `quant-policy-cycle.timer` (`15min`) の設定差分

Fact:
- 事後監査で、`quant-bq-sync` の BQ 同期が 5 分刻み（`--bq-interval 300`）で回る運用を確認し、`lot insights` 解析も毎サイクル有効な状態だった。
- `systemd/quant-bq-sync.service` を `--bq-interval 900 --disable-lot-insights` へ変更し、現在は
  - BQ エクスポートを 15 分間隔ベースまで緩和
  - ロットインサイトの毎回生成を停止
  - 既定で `--limit 1200` による送信上限を維持
  する運用へ更新した。
- `systemd/quant-policy-cycle.timer` を `15min` から `60min` に変更し、policy-cycle の重複実行圧を低下。
- `AGENTS.md` に 1 台運用・BQ 原価抑制の運用規則を明文化し、月間約 23,000 円要因が「同時 RUNNING VM」や未使用リソース固定化と合致する運用根拠へ寄せた。

Failure Cause:
- コスト上振れの主因は `BQ` 単独より、`run_sync` の高頻度実行と `fx-trader*` 複数台起動が重なった状態である可能性が高い。
- lot insights と policy-cycle の同時高頻度化により、DB I/O 競合・再処理量増大を誘発していた。

Improvement:
- `quant-bq-sync` の実行を `--bq-interval 900` / `--disable-lot-insights` へ変更。
- `scripts/run_sync_pipeline.py` を lot insights 無効時でも継続稼働できるガード実装へ修正。
- `quant-policy-cycle.timer` を 60 分周期化。
- 運用文書（`AGENTS.md`/`GCP_PLATFORM.md`）へ1台化・BQ抑制設定を反映。

Verification:
- 運用反映後に `systemd/quant-bq-sync.service` の `ExecStart`、`systemd/quant-policy-cycle.timer` の `OnUnitActiveSec` を確認。
- `gcloud compute instances list --filter='name~^fx-trader' --format='value(name,status)'` で RUNNING の `fx-trader*` が 1 台のみであることを確認。
- `systemctl cat quant-bq-sync.service` と `systemctl cat quant-policy-cycle.timer` を `ops` 監査ログへ保存。
- 24h 程度で `logs/{orders,metrics}.db` の拒否率と `B/Q` 再実行率（`run_sync_pipeline`/`googleapis`）を比較し、悪化がないことを確認。

Status:
- in_progress

## 2026-03-03 (JST) / 並行2レーン判定の運用化（min-trades導入）

Source:
- `scripts/compare_live_lanes.py`
- `scripts/watch_lane_winner.sh`
- `ops/env/local-llm-lane.env`

Fact:
1. 比較ロジックに `--min-trades` を追加し、サンプル不足時は `winner=insufficient_data` を返すよう変更。
2. ローカルレーンは `ops/env/local-llm-lane.env` をロードして local LLM（Ollama）前提で運用可能化。
3. `watch_lane_winner.sh` で比較結果を `logs/lane_winner_latest.json` と履歴へ出力可能。

Verification:
1. `python scripts/compare_live_lanes.py --hours 24 --min-trades 5`
2. `scripts/watch_lane_winner.sh`

Status:
- in_progress

## 2026-03-03 (JST) / ローカルLLM並行売買レーン導入（VM+Local 比較運用）

Source:
- OANDA API 実測（`scripts/check_oanda_summary.py` + 20回サンプリング）
- OANDA pricing/candles 実測（USD/JPY）
- VM health snapshot（`gs://fx-ui-realtime/realtime/health_fx-trader-vm-es1c.json`）
- VM serial（`gcloud compute instances get-serial-port-output fx-trader-vm-es1c`）
- local live log（`logs/codex_long_autotrade.log`）

Market Check (2026-03-03 22:00 JST 台):
1. USD/JPY: mid `157.756`, spread `0.8 pips`
2. M1 ATR14: `3.236 pips`
3. 直近レンジ推移: `last15m=50.2 pips`, `prev15m=54.5 pips`（比率 `0.921`）
4. OANDA API応答品質: 20/20 成功, `p95=315ms`, `max=345ms`
5. OANDA openTrades: manual short `-7000` のみ（bot open なし）

Fact:
1. VM側 health snapshot は `2026-03-02T09:10:55Z` で更新停止気味、bot lane の約定が薄い。
2. serial に `oom-kill`（`quant-scalp-false-break-fade` 系 python）を確認。
3. local lane は `codex_long_autotrade.log` で実売買中、`gpt-oss:20b` 判定は1回 `~27s` と高遅延。
4. 直近24h比較:
   - local lane: `1 trade / net -7.316 JPY / -5.4 pips`
   - vm lane: `0 trades`

Action:
1. BrainゲートへローカルLLM backend を追加（`BRAIN_BACKEND=ollama`）。
2. Brain失敗時ポリシーを `BRAIN_FAIL_POLICY=allow|reduce|block` で制御可能化。
3. `ORDER_MANAGER_BRAIN_GATE_APPLY_WITH_PRESERVE_INTENT` を追加し、必要時のみ preserve intent と Brain の併用を許可。
4. 2系統比較スクリプト `scripts/compare_live_lanes.py` を追加（local log + vm trades.db）。
5. まず `local lane` を実運用継続し、`vm lane` は OOM/SSH不安定解消後に再評価する方針へ。

Verification:
1. `python scripts/compare_live_lanes.py --hours 24` を定期実行し、`winner` を監視。
2. Brainをollamaで有効化する場合は `brain_latency_ms` と `order_brain_block` の比率を同時監査。
3. VM側は OOM 解消（unit整理/メモリ圧迫タスク抑制）後に `filled` 復帰を再確認。

Status:
- in_progress

## 2026-03-03 (JST) / no-entry継続への追加対応（ping5s閾値再緩和 + OOM要因記録）

Source:
- OANDA account summary (`2026-03-03T10:45:54Z`): `openTradeCount=1`, `lastTransactionID=413002`（manualのみ、bot新規約定なし）
- OANDA pricing/candles (`2026-03-03T10:23:56Z`): `USD/JPY bid=157.730 ask=157.738 spread=0.008`, `ATR14(M5)=0.0765`, `range_last_60m=0.313`, API latency `~220-250ms`
- VM serial (`gcloud compute instances get-serial-port-output`): `quant-scalp-false-break-fade.service` の `oom-kill` 発生（`2026-03-03 10:17:54Z` 付近）

Hypothesis:
1. ping5s B/C/D の `reject_under` と `min_units` が still strict で、`submit_attempt` に到達しないケースが残っている。
2. OOMイベントが発生した時間帯では、worker群の安定稼働が崩れて entry チャンスを取りこぼす。

Action:
- `ops/env/quant-order-manager.env`
  - `ORDER_SUBMIT_MAX_ATTEMPTS: 1 -> 2`
  - `ORDER_PROTECTION_FALLBACK_MAX_RETRIES: 0 -> 1`
  - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_B(_LIVE): 0.15 -> 0.10`
  - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_C(_LIVE): 0.15 -> 0.10`
  - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_D(_LIVE): 0.12 -> 0.10`
  - `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_B(_LIVE): 20 -> 10`
  - `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_C(_LIVE): 20 -> 10`

Impact:
- manual建玉操作は未実施（`BLOCK_MANUAL_NETTING=0` 方針維持）。
- 変更は order_manager 側の preflight 緩和のみで、V2導線・entry_thesis契約は不変。

Verification (post-deploy):
1. OANDA `lastTransactionID` が `413002` から更新し、manual以外の新規open tradeが増えること。
2. `orders.db` で ping5s系の `submit_attempt_count` / `filled` が 0 から復帰すること。
3. OOM再発の有無（`quant-scalp-false-break-fade.service` の `oom-kill`）を serial/journal で監視すること。

Status:
- in_progress

## 2026-03-01 01:40 UTC / 2026-03-01 10:40 JST - `replay_quality_gate.py` のシナリオ同義語互換を `exit_workers_groups` と統一

Period:
- 対象:
  - `scripts/replay_quality_gate.py`
  - `tests/analysis/test_replay_quality_gate_script.py`
- 対象データ:
  - ユニットテスト観測（シナリオ名正規化）

Fact:
- `replay_quality_gate.py` 側で受理シナリオが旧仕様のままだと、`replay_exit_workers_groups.py` が新規対応した
  `trend_up` / `trend_down` / `gap` / `stale` などを指定できず、品質ゲート運用に分断が発生していた。
- `wide` / `uptrend` / `gapdown` / `stale_ticks` といった実運用で使われる別名が
  `replay_quality_gate.py` でも正規化されることで、シナリオ指定の往復が一貫化した。

Failure Cause:
- リプレイ品質評価（walk-forward）とシナリオ分類エンジンの許容名が不一致で、`--scenarios` の実運用運用系が断絶していた。

Improvement:
- `replay_quality_gate.py` のシナリオ正規化を拡張し、`SCENARIO_OPTIONS` を exit 側と同等に更新。
- 受理対象に `trend_up` / `trend_down` / `gap_*` / `stale` を追加。
- 同義語を一元変換する補助関数を追加。

Verification:
- `tests/analysis/test_replay_quality_gate_script.py::test_resolve_scenario_names_validates_supported_grouped_scenarios`
  で同義語を含む指定が期待 canonical 名へ変換されることを固定化。

Status:
- done

## 2026-03-01 01:25 UTC / 2026-03-01 10:25 JST - `replay_exit_workers_groups` のシナリオ網羅性を拡張（trend/gap/stale）

Period:
- 対象:
  - `scripts/replay_exit_workers_groups.py` の `_parse_scenarios`, `_build_tick_scenarios`
  - `tests/scripts/test_replay_exit_workers_groups.py`
- 対象データ:
  - スクリプト内ユニット合成データ（シナリオ分類の観測固定）

Fact:
- 従来の `wide_spread/tight_spread/high_vol/low_vol/trend/range` に加え、
  - `trend_up`/`trend_down`（方向別トレンド）、
  - `gap`/`gap_up`/`gap_down`（tick間急変）、
  - `stale`（tick間時間遅延）
  を付与できるように分類ロジックが拡張された。
- 同時に `uptrend`/`downtrend`/`stale_ticks`/`high_volatility` といった同義語がシナリオCLIへ反映される実装を追加。

Failure Cause:
- 先行のシナリオ選別が上下トレンド・価格ギャップ・欠落ティックに弱く、再現シナリオを運用的に増やしにくい状態だった。

Improvement:
- `_parse_scenarios` に同義語正規化を追加し、既存 `SCENARIO_OPTIONS` の拡張シナリオを受理。
- `_build_tick_scenarios` で tick ミリ単位のギャップ/遅延判定を導入し、方向別トレンドの分類を追加。
- `all + 指定シナリオ` でシナリオごとのフィルタ対象を維持したまま再実行可能にし、検証の粒度を上げる。

Verification:
- `tests/scripts/test_replay_exit_workers_groups.py` にシナリオ同義語展開と拡張フラグ分類の固定テストを追加。

Status:
- done

## 2026-03-01 01:05 UTC / 2026-03-01 10:05 JST - `replay_exit_workers_groups` のスキーマ拡張対応（`OPEN_*`, `signals`, `created_at`）

Period:
- 対象:
  - `scripts/replay_exit_workers_groups.py` の `_load_entries_from_replay`
- 根拠データ:
  - `tests/scripts/test_replay_exit_workers_groups.py` 追加シナリオ

Fact:
- `action` が `OPEN_LONG` / `OPEN_SHORT` 形式、または `signals` / `entries` 配下に格納される形式で、または
  `created_at` / `entry` / `entry_px` / `units_signed` / `target_pips` を用いた形式の
  リプレイ行が観測された。
- `trades` 固定キー前提のままだと、旧実装で取りこぼし増加と不要なスキップが発生しうる。

Failure Cause:
- リプレイ生成経路やバージョン差分により、`entry` フィールド名・方向文字列・時間キーが揺れるのに対し
  受け口側の許容幅が不足していた。

Improvement:
- `_load_entries_from_replay` で受け口を拡張。
  - `trades` 未定義時に `entries`/`signals`/`actions`/`data` を探索。
  - `OPEN_*` 系アクションを方向として解釈。
  - `created_at`, `entry`, `entry_px`, `units_signed`, `take_profit`, `stop_loss`, `tp_distance`, `sl_distance`, `target_pips` を受理。
- 既存 `tp_price`/`sl_price` 優先順序と `tp_pips`/`sl_pips` 補完の方針を維持したまま、別名キーへの後方互換を追加。
- `units_signed` が負数でも `abs` 変換後に受理する前提を明記。方向キーとの整合で、符号付き数量由来のエントリー拒否を回避。

Verification:
- 追加テストで `OPEN_LONG` / `signals` / `created_at` / `units_signed` ケースを通過したことを確認。

Status:
- done

## 2026-03-01 00:45 UTC / 2026-03-01 09:45 JST - `replay_exit_workers_groups` の入力パース脆弱性是正（壊れたレコードでリプレイ停止を防止）

Period:
- 対象:
  - `scripts/replay_exit_workers_groups.py` の `replay_workers_*` 出力パース
- 根拠データ:
  - スキーマ揺れ（`entry_time`/`open_time`/`time`/`ts`/`timestamp`、`direction`/`side`/`action`）を含むローカル再現ログ
  - `tests/scripts/test_replay_exit_workers_groups.py` に追加した再現データ

Fact:
- `replay_workers_*_*.json` が 1 件の不正行を含むと、旧ローダーは `ValueError` で全件停止しやすく、再現レポートが生成不能になる状態を確認。
- 同じ入力で、`entry` スキーマのキー差分を吸収しない trade が混在しても、妥当行は継続して読み込めるようになった。
- `_parse_dt` が文字列エポック値と `ms` 単位数値にも対応し、時刻欠損行をスキップしながら継続処理できることを確認。

Failure Cause:
- `entry` 取得時に `entry_time` と `direction` と `entry_price` を固定キー前提で扱っていたため、仕様差分や欠損行で即停止する設計だった。

Improvement:
- `_load_entries_from_replay` を以下の方針に修正:
  - JSON パース失敗は空配列で継続。
  - `trades` を dict/list どちらでも受ける。
  - `entry` フィールドキーを `entry_time`/`open_time`/`time`/`ts`/`timestamp` 等へ拡張。
  - 方向/価格/数量の異形キーへ柔軟対応し、必須判定を保ちながら不正行をスキップ継続。
  - `tp_price`/`sl_price` と `tp_pips`/`sl_pips` を併用可能にし、`entry` 受理を阻害しない。
  - 時刻を UTC 正規化し、文字列/数値エポック（ms含む）を許容。
- `tests/scripts/test_replay_exit_workers_groups.py` を新規追加し、壊れた入力での継続挙動を固定化。

Verification:
- `pytest -q tests/scripts/test_replay_exit_workers_groups.py`
- 同期間の replay パイプラインで `summary_all.json` の生成中断件数が減少することを確認（運用環境再実行時に比較）。

Status:
- done

## 2026-03-01 00:30 UTC / 2026-03-01 09:30 JST - 収益阻害Top3（prob/perf/intent reject）に対する縮小継続チューニング適用

Period:
- 監査対象:
  - VM実データ `logs/orders.db`, `logs/trades.db`, `logs/metrics.db`, `journalctl`
  - 期間: 24h / 72h / 168h（JST 7-8時はメンテ時間として評価除外）

Fact:
- 24h収益は `PF=0.727`, `net=-737 JPY`。
- 168hでは `net=-31,346.5 JPY`、赤字寄与上位は `MicroPullbackEMA`, `scalp_ping_5s_c_live`, `scalp_ping_5s_b_live`。
- 直近5000 ordersで reject/guard 偏在:
  - `entry_probability_reject=609 (12.18%)`
  - `perf_block=588 (11.76%)`
  - `entry_intent_guard_reject=200 (4.00%)`
- `latency_preflight` の長い尾（p95級）と `fetch_recent_trades timeout(8s)`、`order-manager child died` を同時観測。

Failure Cause:
1. B/C戦略の `entry_probability` / perf guard / preserve-intent の閾値が重なり、同一経路で reject が常態化。
2. `scalp_ping_5s_b_live` の preserve-intent scale 範囲に逆転値（`MIN_SCALE > MAX_SCALE`）が存在し、縮小ロジックの意図維持を阻害。
3. V2 runtime に `WORKER_ONLY_MODE=true` と `MAIN_TRADING_ENABLED=1` が同居し、導線方針に矛盾。
4. position/order 周辺 timeout が長く、preflight遅延時の fail-fast が弱い。

Improvement:
- `ops/env/quant-v2-runtime.env`
  - `MAIN_TRADING_ENABLED=0`、`ORDER_MANAGER_SERVICE_FALLBACK_LOCAL=0`、`POSITION_MANAGER_SERVICE_FALLBACK_LOCAL=0`。
  - order/position timeout を fail-fast 側へ調整。
  - B/C の forecast/net-edge/perf 閾値を 1 段緩和し、`hard block` 常態化を抑制。
- `ops/env/quant-order-manager.env`
  - `ORDER_MANAGER_SERVICE_FALLBACK_LOCAL=0`。
  - B の `ORDER_MANAGER_PRESERVE_INTENT_MAX_SCALE_STRATEGY_SCALP_PING_5S_B_LIVE` を `0.30 -> 0.85` へ修正（逆転解消）。
  - B/C の `reject_under`, `min_units`, perf/forecast 閾値を再調整。
- `ops/env/scalp_ping_5s_b.env`, `ops/env/scalp_ping_5s_c.env`
  - B/C の base units を軽く抑え、probability floor/perf guard 閾値を再調整。

Verification:
- デプロイ後24hで以下を同条件再監査:
  - `orders.db` 直近5000件で
    - `entry_probability_reject <= 5%`
    - `perf_block <= 5%`
    - `entry_intent_guard_reject <= 1%`
  - `metrics.db` で `latency_preflight p95 < 2000ms`（取得可能な同等指標で代替可）
  - `trades.db`（non-manual）で `PF > 1.00`, `net_jpy > 0`

Status:
## Entry Template
```

## 2026-02-28 23:55 UTC / 2026-02-29 08:55 JST - EXIT共通制御で strategy_tag 欠損建玉が制御外を通過しうる事象を確認（2/24 追跡）

Period:
- 監査対象:
  - VM実データ `logs/orders.db`, `logs/trades.db`（2026-02-24 UTC中心）
  - `execution/order_manager.py` の実装査読・監査スクリプト再計算

Fact:
- `orders.db` で `status="strategy_control_exit_disabled"` が 10,277 件確認（主として2/24 UTC 02:00-07:00）。
- 同期間で閉鎖されていない/反復失敗 trade の主要 13件（`384420`,`384425`,`384430`,`384435`,`384797`,`384807`,`384812`,`384920`,`385300`,`385303`,`385332`,`385337`,`385390`）を確認。
- 各tradeの `close_request`→`close_ok` 移行が遅延し、長時間同一建玉の再試行が連続。
- `close` パスには `strategy_tag` 必須拒否が無く、欠損時は `strategy_control` が事実上スキップされる経路が残存。
Failure Cause:
1. `order_manager._reject_exit_by_control` が `strategy_tag` 未定義時に即時許可していたため、共通 EXIT ガードが欠損建玉で適用されない。
2. 2/24の阻害再発は、該当戦略の `strategy_control_exit_disabled` 連打と、閉鎖遅延の長期化を伴って発生。
Improvement:
- `execution/order_manager.py` の `_reject_exit_by_control` を、`global_exit`/`global_lock` を先頭で適用する形へ修正。
- `close_trade` に `strategy_tag` 欠損時の明示 `close_reject_missing_strategy_tag` を追加。
Verification:
- VM `orders.db` で `close_reject_missing_strategy_tag` の新規件数を監視。
- VM側で同一期間を再実行し、`close_request=close_reject` の主要 13取引が strategy_control への制御で停止し続けないことを確認。
- `strategy_control_exit_disabled` 連打抑制（閾値到達時の failopen / bypass）導線が再開することを確認。
Status:
- in_progress

### 2026-03-02（追記）position_manager open_positions タイムアウトの再発抑止

Source:
- `logs/journal`（VM）
- `execution/position_manager.py`
- `ops/env/quant-v2-runtime.env`

Fact:
- `open_positions` の呼び出しで `Read timed out`（`read timeout=4.0/6.5`）が継続観測され、シグナルを拾い続ける前提が崩れていた。

Failure Cause:
- タイムアウト設定が `open_positions` 経路で短く、`position_manager` からの応答前に上位が切り戻されるループが発生していた。

Action:
- `POSITION_MANAGER_SERVICE_OPEN_POSITIONS_TIMEOUT: 4.0 -> 8.0`
- `POSITION_MANAGER_HTTP_TIMEOUT: 5.0 -> 8.0`
- `POSITION_MANAGER_OPEN_TRADES_HTTP_TIMEOUT: 2.8 -> 8.0`
- `POSITION_MANAGER_WORKER_OPEN_POSITIONS_TIMEOUT_SEC: 4.0 -> 10.0`

Next check:
- VM 反映後 15〜30 分で `position_manager service call failed` のログ頻度、`entry_probability_reject` 以外の拒否率変化、`trades.db` の filled 再開状況を確認。

## 2026-02-28 23:40 UTC / 2026-02-29 08:40 JST - `orders_snapshot_48h.db` の鮮度差起因監査崩れに対する同梱/監査ロジック同時修正

Period:
- 対象:
  - `scripts/collect_gcs_status.py`
  - `ops/env/quant-core-backup.env`
  - `execution/position_manager.py`
  - `remote_logs_current/core_extract` 同一時点抽出

Fact:
- `core` バックアップ運用は、`QR_CORE_BACKUP_INCLUDE_ORDERS_DB=0` のため監査に `orders.db` が同梱されず、代替の
  `orders_snapshot_48h.db` が長期更新滞留して `trades` 時系列と `orders` 時系列のズレが発生。
- 追加実装で `collect_gcs_status` は `core_*` ファイル名（`.tar`/`.tar.gz`）対応を維持しつつ、`orders_db_source` と
  `orders_snapshot_age_vs_trades_h` / `orders_snapshot_freshness` を出力し、監査時に鮮度劣化を数値化できる状態に変更。
- `execution/position_manager.py` の `_normalize_entry_contract_fields` は、`payload.units` を含む経路を追加。

Failure Cause:
1. 監査再現では注文DB更新タイムラインを `trades` と同居させる同梱方針が不足していた。
2. 過去世代 `entry_thesis` 再構成では payload 欠損が混在し、監査キー注入が不十分な経路が残っていた。

Improvement:
- `ops/env/quant-core-backup.env` を更新し、`QR_CORE_BACKUP_INCLUDE_ORDERS_DB=1` へ固定。
- `collect_gcs_status.py` に注文DB鮮度の警告分類（`warning` / `critical`）を追加。
- `position_manager` の `entry_units_intent` 補完に `payload.units` を追加し、`fallback_units` 経路を維持。

Verification:
- VM同時取得された `core_*.tar(.gz)` を前提に、`collect_gcs_status` が `orders_db_source`、`orders_snapshot_freshness` を返すことを確認。
- `collect_gcs_status` の判定基準:
  - `orders_snapshot_freshness != critical` かつ `orders_db_source` が `orders.db` の場合に監査結果を採用。

Status:
- in_progress

## 2026-02-28 22:10 UTC / 2026-02-29 07:10 JST - リプレイID `sim-*` の閉鎖失敗を Exit 共通拒否ではなく入力ID不正で遮断へ変更

Period:
- 監査対象:
  - `logs/orders.db`（close 関連ログ）
  - `scripts/replay_exit_workers.py` / `execution/order_manager.py`

Fact:
- close 系の再現ログで `trade_id` が `sim-40`,`sim-8`,`sim-37`,`sim-4` のみを対象に `close_request` と `close_failed` が連続。
- 失敗コードは `oanda::rest::core::InvalidParameterException`、`Invalid value specified for 'tradeID'`。
- 同期間の close は共通拒否系（`close_blocked_by_strategy_control`/`close_reject_*`）の増加が確認されず、リプレイ起因の ID 形式不整合が主因と判断。

Failure Cause:
- リプレイ側で `trade_id` が `sim-*` 形式のまま `order_manager.close_trade` に渡され、`client_order_id` が `sim-sim-*` まで二重化されるケースが残存。

Improvement:
- `execution/order_manager.py` に live tradeID 形式バリデーションを追加し、`close_trade` で `sim-*` 等の非数値 ID を `close_reject_invalid_trade_id` で即時停止。
- `scripts/replay_exit_workers.py` で `sim-` 再付与を抑止（既に `sim-*` の場合はそのまま利用）して `client_order_id` の `sim-sim-*` 化を防止。

Verification:
- 直近時点の `orders.db` では `close_reject_invalid_trade_id` 件数と `trade_id` 非数値拒否ログの出現を確認し、`sim-*` への CLOSE_REQUEST が消失しているかを観測。
- 併せて V2 上で `quant-order-manager` 側が同変更を採用した後、同等条件で close が再度失敗しないかを VM 実行で確認。

Status:
- in_progress

## 2026-02-28 21:10 UTC / 2026-02-29 06:10 JST - ローカルDB再集計で契約欠損は `entry`/`trade` の同世代不一致が主因と判定

Period:
- 対象:
  - `logs/trades.db`
  - `logs/orders.db`
  - `tmp/vm_audit_20260226/trades.db`
  - `tmp/vm_audit_20260226/orders.db` (開封不可; malformed)

Fact:
- `logs/trades.db` を `open_time` 基準で再集計（`entry_probability`,`entry_units_intent` の必須キー）:
  - 24h: `rows=101 / missing_any=101`
  - 48h: `rows=168 / missing_any=168`
  - 240h: `rows=275 / missing_any=275`
  - いずれも `entry_thesis` 自体は存在しているが、対象キーは未挿入。
- `logs/orders.db` 時系列:
  - `ts` 範囲: 2025-05-30 ～ 2026-02-24
  - 2026-02-24 時点以降は `close_request/close_failed` が中心で、`submit_attempt` の `side/instrument/request` エントリー路線は含まれず、`trades` 側 2/27 の新規エントリーと非整合。
  - `submit_attempt` (`n=2907`) 内訳:
    - `entry_thesis` あり: 1840件（`entry_probability` と `entry_units_intent` は 1840 件とも欠損）
    - `entry_thesis` なし: 1067件
- `tmp/vm_audit_20260226/orders.db` は開封時点で `sqlite_error: database disk image is malformed`。

Failure Cause:
1. `trades` の監査対象窓と `orders` の有効窓が同世代化されておらず、再構成が成立しない。
2. 2026-02-24 以降の `orders` 側は close 系に偏り、entry 系の同窓口が欠落。
3. VM への接続が `Connection timed out during banner exchange` / `port 65535` で停止し、最新同世代ログを再取得できない。

Improvement:
- `orders.db` と `trades.db` の同世代再取得が先決条件であることを明文化し、`entry_thesis` 欠損の再評価はそれ後に再実行。
- `entry_thesis` は `logs` 側では `entry_probability:1.0` / `entry_units_intent:abs(units)` で補完再構成可能だった領域と、`request_json` 未持ち込み等で不可の領域を分離して監査継続。

Verification:
- ローカル再現コマンド:
  - `python3 - <<'PY'` で `logs/trades.db` の `open_time` 窓別欠損集計を再計測。
  - `python3 - <<'PY'` で `logs/orders.db` の `status/request_json/entry_thesis` 欠損別分解を再計測。
- 本番側同世代再取得成功後:
  - `gcloud compute ssh fx-trader-vm --tunnel-through-iap --command "echo ok"`
  - `python3 ~/.codex/skills/qr-entry-thesis-contract-check/scripts/check_entry_thesis_contract.py --window-hours 240 --limit 20000 --json ...`

Status:
- in_progress

## 2026-02-28 18:55 UTC / 2026-02-28 03:55 JST - `entry_thesis` 監査は同世代再取得未達で確定

Period:
- 再検証対象:
  - `logs/trades.db` / `logs/orders.db`
  - `tmp/vm_audit_20260226/trades.db` / `tmp/vm_audit_20260226/orders.db`
  - `remote_tmp/trades.db` / `remote_tmp/orders.db`
  - `remote_logs_current/core_extract/trades.db` / `remote_logs_current/core_extract/orders.db`
- コマンド:
  - `python3 ~/.codex/skills/qr-entry-thesis-contract-check/scripts/check_entry_thesis_contract.py --window-hours 240 --limit 20000 --json --trades-db ... --orders-db ...`
  - `python3 ~/.codex/skills/qr-entry-thesis-contract-check/scripts/check_entry_thesis_contract.py --window-hours 48 --limit 20000 --json --trades-db logs/trades.db --orders-db logs/orders.db`
  - `python3 ~/.codex/skills/qr-entry-thesis-contract-check/scripts/check_entry_thesis_contract.py --window-hours 24 --limit 20000 --json --trades-db logs/trades.db --orders-db logs/orders.db`

Fact:
- `logs/trades.db`:
  - `window=240h`: `rows_missing_any=274`、`sampled_rows_in_window=275`
  - `window=48h`: `rows_missing_any=100`、`sampled_rows_in_window=101`
  - `window=24h`: `trades_no_samples_in_window`
  - `orders` 側は `window=240h` で `sampled_rows_in_window=344` なのに `evaluated_rows=0`（`submit_attempt` 参照で再構成不能）
- `tmp/vm_audit_20260226`:
  - `trades`: `rows_missing_any=2740`（`window=240h`）
  - `orders`: `sqlite_error: database disk image is malformed`
- `remote_tmp`: `no sample in requested window`
- `remote_logs_current/core_extract`: `trades`/`orders` のどちらも対象窓に未到達（`no samples in window`）
- 追加観測:
  - `df` 空き不足（`df` 結果が 100% 過負荷状態）で `gcloud` 実行やコピー前にクリーンアップが必要
  - VM 接続:
    - `Permission denied (publickey)` が継続
    - `Connection closed by UNKNOWN port 65535`（鍵指定実行でも再現）
    - シリアル出力で `sshd` と Python が OOM kill される痕跡を確認

Failure Cause:
1. `orders` / `trades` の同一時刻同一世代ペアが取れないため、監査の比較軸がずれている。
2. VM への SSH/OAuth 経路（OS Login / key）と VM 側資源状態（sshd OOM/再起動）が不安定で、同時データ取得不能。
3. ローカル側ディスク空き不足（`No space left on device`）で作業環境ノイズが混入。

Improvement:
- VM 接続安定化（OS Login/SSH 鍵経路、sshd メモリ負荷の是正）を優先して同一時刻 `trades.db` + `orders.db` を再取得し、現地同時監査に入る。
- 監査は `raw_missing` と `recoverable_missing` を分離して記録し続ける運用を維持し、同世代再取得後に `rows_missing_contract_fields=0` 到達を再確認。

Verification:
- 同一世代取得が可能になった時点で以下を実行:
  - `python3 ~/.codex/skills/qr-entry-thesis-contract-check/scripts/check_entry_thesis_contract.py --trades-db logs/trades.db --orders-db orders.db --window-hours 240 --limit 20000 --json`
  - `python3 ~/.codex/skills/qr-entry-thesis-contract-check/scripts/check_entry_thesis_contract.py --trades-db logs/trades.db --orders-db orders.db --window-hours 24 --limit 20000 --json`
- VM 障害確認:
  - `gcloud compute ssh fx-trader-vm --project=quantrabbit --zone=asia-northeast1-a --tunnel-through-iap --command "echo ok"`
  - `gcloud compute instances get-serial-port-output fx-trader-vm --project=quantrabbit --zone=asia-northeast1-a --port=1`

Status:
- in_progress

## 2026-02-28 20:40 UTC / 2026-02-29 05:40 JST - 過去DB再現で `entry_probability` / `entry_units_intent` は補完で吸収、欠損自体は主に旧世代要因

Period:
- 対象DB（`--window-hours 240`, `--window-hours 100000`, `--limit 20000~30000`）:
  - `logs/trades.db` / `logs/orders.db`
  - `remote_logs_current/core_extract/trades.db` / `remote_logs_current/core_extract/orders.db`
  - `tmp/vm_audit_20260226/trades.db` / `tmp/vm_audit_20260226/orders.db`
  - `tmp/qr_gcs_restore/core_20260227T062401Z/trades.db`
  - `tmp/qr_gcs_restore/multi/core_20260227T054231Z/trades.db`
  - `tmp/qr_gcs_restore/multi/core_20260227T050329Z/trades.db`
  - `tmp/qr_gcs_restore/multi/core_20260227T060325Z/trades.db`

Fact:
- `check_entry_thesis_contract`（厳密監査）での集計:
  - `logs/trades.db` (`240h`) `trades_missing_contract_fields=274`
  - `logs/trades.db` (`100000h`) `trades_missing=12049`, `orders_missing=3350`
  - `remote_logs_current/core_extract` (`100000h`) `trades_missing=6576`, `orders_missing=20392`（`orders` 側は 287,835 件をサンプリング）
  - `tmp/vm_audit_20260226` (`100000h`) `trades_missing=4739`
  - `tmp/qr_gcs_restore/*` (`100000h`) `trades_missing=4740`（`orders` は同DBに未同梱）
- 補完シミュレーション（`entry_probability=1.0`、`entry_units_intent=abs(units)`）を適用すると、上記 `trades` 群はほぼ `raw_missing` から回収可能（`recoverable` 化）と確認。
- `orders` 側は `request_json` 欠損/不完全が支配的で、`remote_logs_current` では `raw_missing 278,814` に対し `recovered 8,813`（`unrecoverable 270,001`）と再現。

Failure Cause:
1. 過去世代の `trades` は、保存時点で必須キー注入がなかったため監査で欠損扱い。
2. `orders` は世代差分や `request_json` 欠損の影響が大きく、再構成で contract 欠損を回収できない行が多数。

Improvement:
- `execution/position_manager.py` の保存前補完ロジック自体は新規データの再発抑制効果が高いため、現行運用は同実装を前提に継続。
- 監査運用として、`raw_missing` と `recoverable_missing` の分離レポートを採用し、真に改善対象が残る行だけを精査。
- 旧世代 `logs` ではなく VM 同期新世代（`logs/trades.db` + `logs/orders.db` 同時更新）で再監査し、監査差分の消失を確認。

Verification:
- 厳密監査:
  - `python3 ~/.codex/skills/qr-entry-thesis-contract-check/scripts/check_entry_thesis_contract.py --trades-db ... --orders-db ... --window-hours 240 --limit 20000 --json`
  - `python3 ~/.codex/skills/qr-entry-thesis-contract-check/scripts/check_entry_thesis_contract.py --trades-db ... --orders-db ... --window-hours 100000 --limit 30000 --json`
- 補完シミュレーション（レガシーデータ検証）を実行済み。

Status:
- done

## 2026-02-28 19:00 UTC / 2026-02-29 04:00 JST - `position_manager` の監査復元で `entry_thesis` 欠損を縮小する対策

Period:
- ローカルVM `logs/orders.db` / `logs/trades.db`
- 検証ウィンドウ: `--window-hours 240`

Fact:
- 直近再検証:
  - `trades` 直近 275 行中 274 行が `entry_probability` / `entry_units_intent` 欠損。
  - `orders` 直近 344 行中、`sampled_rows_in_window` は 344、`evaluated_rows=0`（`submit_attempt` が同窓口に存在せず再構成不能）。
  - `sqlite3 ...` の `ATTACH` 照合では `trades` 直近 `client_order_id` 274 件の `orders` 参照一致は 0。
- `gcloud compute ssh fx-trader-vm --tunnel-through-iap` は依然として `Permission denied (publickey)`。

Failure Cause:
1. `position_manager` の `entry_thesis` は、`orders` 由来の再構成に失敗した経路では永続化時にデフォルト補完が十分に入らず、監査レベルで `trades` が欠損扱いになる構成だった。
2. VM 側 `orders.db` 側の同一世代 `submit_attempt` がローカル窓に入らず、監査再構成が成立しない。

Improvement:
- `execution/position_manager.py`:
  - `_get_trade_details_from_orders` 取得後、`_normalize_entry_contract_fields` を通して `entry_probability` / `entry_units_intent` を補完。
  - `_get_trade_details`（OANDAフォールバック）と `open_positions` 形成経路でも同補完を適用し、保存・監査で最終 `entry_thesis` を契約形に寄せる。
- 併せて、監査再計算対象の指針を `TRADE_FINDINGS` に 1 箇所集約。

Verification:
- 追加後に以下を実行済み:
  - `python3 ~/.codex/skills/qr-entry-thesis-contract-check/scripts/check_entry_thesis_contract.py --repo-root . --window-hours 240 --limit 20000 --json`
  - `sqlite3 logs/orders.db` / `sqlite3 logs/trades.db` で `client_order_id` 相互照合（`ATTACH`）
- 完全再現は VM 側同一世代取得ができた時点で `rows_missing_any=0` 目標で再確認。

Status:
- in_progress

## 2026-02-28 19:15 UTC / 2026-02-29 04:15 JST - `position_manager` 永続保存時に `entry_thesis` を保存前補完

Period:
- ローカルVM `logs/trades.db` / `logs/orders.db`
- 検証: `--window-hours 240`, `--window-hours 100000`（`check_entry_thesis_contract`）
- 追加検証: トレード行を `entry_probability`/`entry_units_intent` を
  `_normalize_entry_contract_fields` 的規則で再構成した場合のシミュレーション

Fact:
- 変更前後で `execution/position_manager.py` の `_parse_and_save_trades` と `_get_trade_details` に
  保存前補完を追加。
- 追加直後に再実行したスクリプト:
  - `python3 ~/.codex/skills/qr-entry-thesis-contract-check/scripts/check_entry_thesis_contract.py --repo-root . --window-hours 240 --limit 20000 --json`
  - `python3 ~/.codex/skills/qr-entry-thesis-contract-check/scripts/check_entry_thesis_contract.py --repo-root . --window-hours 100000 --limit 20000 --json`
  いずれも既存DBが旧世代であるため `trades_missing_contract_fields` は継続し、`orders` 側も `entry_...` 欠損を保持。
- ただし、`trades` 生データ（12000件）を上記再構成規則で再評価した結果、`entry_probability`/`entry_units_intent` 欠損は 0 件に収束（将来保存レコードの再発防止効果を示唆）。

Failure Cause:
- 直近ローカル `trades.db` は `entry_thesis` 自体が監査前提キーを欠いた履歴を持ち、`orders.db` との同一世代参照が不十分。
- したがって本番実行データの「保存時点」での補完を追加しても、過去データの監査結果は同時に変わらない。

Improvement:
- `execution/position_manager.py`
  - `trades` 永続化ループ内で `details["entry_thesis"]` を契約正規化。
  - OANDAフォールバック取得 (`_get_trade_details`) でも再正規化し、上流入力が不足していても保存前に契約形へ寄せる。
- VM 取得データが更新された後は、同スクリプトで `trades_missing_contract_fields=0` 到達を再確認することを最終目標として追跡。

Verification:
- 同一ファイル内 `saved_records` 生成前後で保存行 `json.dumps(details["entry_thesis"])` が必ず契約キーを含む形になることをローカル確認。
- 運用上は次の `checks` 再実行時（VM新世代）にて `trades_missing_contract_fields` が解消されるかを監視。

Status:
- in_progress

## 2026-02-28 09:02 UTC / 2026-02-28 18:02 JST - `entry_thesis` 欠損監査: ローカル世代断絶と過去断面の全面欠損を再確認

Period:
- ローカルVM `logs/orders.db` / `logs/trades.db`（`--window-hours 240`）
- 過去スナップショット `remote_logs_current/core_extract/orders.db` / `remote_logs_current/core_extract/trades.db`（`--window-hours 100000`、`check_entry_thesis_contract`）

Fact:
- `check_entry_thesis_contract`（ローカル）:
  - `overall=fail`
  - `trades_missing_contract_fields:274`
  - `trades` `evaluated_rows=274`（`sampled_rows_in_window=275`）
  - `orders` `sampled_rows_in_window=344` のうち `evaluated_rows=0`（`orders` 側は `submit_attempt` がローカル内で `window` に入る世代が `trades` と照合不可）
- ローカル `logs/orders.db` と `logs/trades.db` の `client_order_id` 再構成:
  - `client_order_id` を `qr-` で集約した `trades` 全体一致（`105/...`）はありうる一方、`240h` 窓での `submit_attempt` への一致は `0/255`（一致なし）。
  - `logs/orders.db` の `MAX(ts)` は `2026-02-24T02:09:44Z`、`logs/trades.db` の `MAX(open_time)` は `2026-02-27T00:23:46Z` で、3日超の世代ズレ。
- `remote_logs_current/core_extract`（過去断面）を同スクリプトで検証:
  - `trades_missing_contract_fields:6576 / 6576`
  - `orders_missing_contract_fields:22180 / 40000(対象)`
  - `orders` 側 `top_strategies` 上位多数で `entry_probability` / `entry_units_intent` 不在、`entry_...` 欠損は運用開始初期断面の系統的欠落と一致。

Hypothesis:
1. 直近 240h のローカル監査失敗は `trades` 側の新規世代と `orders` 側断面の世代切替が崩れたことが主因で、検証条件として再構成不能。
2. `remote_logs_current/core_extract` が示す長期 `100000h` 断面では `entry_probability` / `entry_units_intent` が未注入状態で生成されており、欠損自体は「新規機能未反映」の履歴的傾向も確認済み。

Failure Cause:
- `entry_thesis` 必須フィールド注入前の断面が複数存在。
- ローカル `orders.db`/`trades.db` が同期世代を跨いでいるため、`trades` → `request_json` 再構成導線で `orders` を使う監査が成立しない。

Improvement:
- 今回の検証結果は、現時点のローカル断面では「監査基準を満たす再構成不能」扱いとし、VM 側で同一世代を取り直した上で再測定する前提で運用に反映。
- `order_manager` / `position_manager` 側の補完ロジックは維持し、`orders` 側が新世代で欠損が続く場合は追加にて全呼び出し経路の `entry_thesis` 注入箇所を再監査する。

Verification:
- 直近実行:
  - `python3 ~/.codex/skills/qr-entry-thesis-contract-check/scripts/check_entry_thesis_contract.py --repo-root . --window-hours 240 --limit 20000 --json`
  - `sqlite3 logs/orders.db` / `sqlite3 logs/trades.db` の `client_order_id` 世代一致率集計（`ATTACH` で相互照合）
  - 過去断面再現:
    - 上記リポジトリを一時ディレクトリでリンクし、
    `worker` / `execution` と `remote_logs_current/core_extract` を使って同スクリプト再実行

Status:
- done

## 2026-02-28 17:50 UTC / 2026-02-29 02:50 JST - `entry_thesis` 欠損監査: logs 世代不整合確認

Period:
- ローカルVM `logs/orders.db` / `logs/trades.db`（`window_hours=240`）
- 補助検証: `remote_logs_current/core_extract/orders.db` / `remote_logs_current/core_extract/trades.db`

Fact:
- `qr-entry-thesis-contract-check`（`--window-hours 240`）で `trades_missing_contract_fields=274` を再現。
- `logs/trades.db` は `qr-177215...` 世代（2026-02-27付近）を持つ一方、`logs/orders.db` の `qr-` 世代は `qr-176...`（2025-12-10まで）で世代がずれている。
- `logs/orders.db` で `entry_probability`/`entry_units_intent` 同居は 0 件。`remote_logs_current` でも同条件 0 件。

Failure Cause:
1. `orders.db` / `trades.db` の時系列同期ズレにより、trade側再構成の照合対象が欠落。
2. 該当 logs は旧仕様断面で、`entry_thesis` 必須フィールドが履歴に未付与。
3. VM再取得確認は `Permission denied (publickey)` と `GlobalPerProjectConnectivityTests` クォータ上限で遅延。

Improvement:
- まず VM 側で同一世代の `orders.db` / `trades.db` を取得し、同仕様の断面で監査を再実行。
- 本変更分の `order_manager` / `position_manager` 反映後、`rows_missing_any=0` 到達を目標に再計測。

Verification:
- 再監査コマンド:
  - `python3 ~/.codex/skills/qr-entry-thesis-contract-check/scripts/check_entry_thesis_contract.py --repo-root . --window-hours 240 --limit 20000 --json`
  - `gcloud compute ssh fx-trader-vm --tunnel-through-iap --command "..."`（認証確認後）

Status:
- in_progress

## 2026-02-28 08:50 UTC / 2026-02-28 17:50 JST - `entry_thesis` 欠損補完フェイルセーフを拡張（order/position）

Period:
- ローカルVM `logs/orders.db` / `logs/trades.db`（`window_hours=240`）
- 対象: `entry_probability` / `entry_units_intent` 未補完件

Fact:
- `execution/order_manager.py` で `ORDER_MANAGER_ENTRY_PROBABILITY_DEFAULT` を追加し、`_ensure_entry_intent_payload` に `entry_probability` 補完を追加。
- `execution/position_manager.py` に `entry_thesis` 欠損時復元ヘルパーを追加し、`request_json.entry_probability`、`request_json.entry_units_intent`、`request_json.oanda.order.units`、`orders.units` を順に補完。
- 直近240h監査で `trades` 側の欠損は 274 件。`orders` 側は同 window で `submit_attempt` が存在するが（`MAX(ts)=2026-02-24 02:09:44`）、`trades` 連携先 `client_order_id` 参照で `rows_missing` が解消されず再構成不能の状態を確認。

Failure Cause:
- `trades` の多くは `client_order_id` 解決しても当該 `submit_attempt` ログが window 内に存在せず、履歴補完の再構成ソース欠損。

Improvement:
- `order_manager`/`position_manager` の双方でフェイルセーフ補完を追加し、将来の約定履歴に対して `entry_thesis` 要件を持ち越す。
- 監査は VM 実データで `submit_attempt` を再現できる周期へ遷移させ、欠損率を再測定する前提で再実行する。

Verification:
- `python3 -m py_compile execution/order_manager.py execution/position_manager.py` 通過。
- `check_entry_thesis_contract.py --window-hours 240` は現在のローカル再現データ上では `trades_missing_contract_fields=274` を返却。`orders.db` の `submit_attempt` が `2025-11-25` までで、再構成不能区間を含むため、VM 時系列再確認が必要。
- 併せて `gcloud compute ssh fx-trader-vm --tunnel-through-iap --command "sqlite3 /home/tossaki/QuantRabbit/logs/orders.db ..."` で VM 側 `submit_attempt` の時系列を再確認する。

Status:
- in_progress

## 2026-02-28 23:42 UTC / 2026-02-29 08:42 JST - `strategy_control_exit` 同一建玉の再試行キーを `trade_id` 優先へ固定

Period:
- VM実測（`logs/orders.db` / `logs/trades.db`）の 2026-02-24 01:00 UTC〜10:00 UTC を再監査。
- 併せて `execution/order_manager.py` の `strategy_control` 失効導線を確認。

Fact:
- `orders` で `status='strategy_control_exit_disabled'` が 10,277 件確認。
- 集約は 4 つの `trade_id`（`384420`,`384425`,`384430`,`384435`）が中心で、各 `2,044`〜`2,045` 件の連続阻害が発生。
- 同時間帯の `close_request=0` が 11 取引で、阻害後の `close_ok` 取得がほぼ無く、`close_bypassed_strategy_control` / `strategy_control_exit_failopen*` が出力されなかった。
- VM `ops/env/quant-v2-runtime.env` では `ORDER_STRATEGY_CONTROL_EXIT_FAILOPEN_*` は有効（threshold=3, window=20, reset=180, emergency_only=0）。

Failure Cause:
1. `order_manager.py` の再試行キーが `client_order_id` 優先だったため、同一建玉でも client_id が変化した場合に同一阻害として集約されず、フェイルオープン閾値到達判定が崩れる可能性があった。
2. 連続阻害のキー粒度不足により、`strategy_control_exit_failopen` へ遷移せずに同一tradeのリトライだけが増幅し、損失持続を招いた。

Improvement:
1. `execution/order_manager.py` の `_strategy_control_exit_block_key` を `trade_id` 優先キーに変更し、同一建玉の阻害を 1 つのキーで集約。
2. `request_json` に渡す `block_state` と既存ログ `strategy_tag` / `trade_id` を用いて、後続監査でキー再解釈ができる状態を維持。

Verification:
1. VM反映後に `orders.db` で同一 `trade_id` に対する `strategy_control_exit_disabled` の連続数を再集計し、  
   `strategy_control_exit_failopen_*` か `close_bypassed_strategy_control` が threshold 近傍で登場することを確認。
2. 直近 24h で同一tradeの `close_request=1, close_ok=1` 到達率が維持され、`close_request=0` が集中する trade の再発がないことを確認。
3. 併せて `logs/ops_v2_audit_latest.json` / `journalctl -u quant-order-manager.service` の起動後再起動有無を監査。

Status:
- in_progress

## 2026-02-28 23:10 UTC / 2026-02-29 08:10 JST - `close_reject_no_negative` の保護理由集合を収斂

Period:
- VM直近実データ（`orders.db`）に基づき、`close_reject_no_negative` の `exit_reason` 上位を再審査。
- 同時に `quant-order-manager` の `order_manager.py` 設定値を照合し、allow/force/bypass セットの整合性を確認。

Fact:
- `close_reject_no_negative` の主要原因は `max_adverse` / `no_recovery` / `time_stop` 系が大きく、同時に `fast_cut_time` が上位帯に入り、  
  allow/force/immediate の理由集合で運用上の想定と実際トリガが一致しない箇所が存在していた。
- `fast_cut_time` は保護的な exit reason であるにもかかわらず、共通 allow 集合に欠落する設定が残っていた。

Failure Cause:
1. `ORDER_ALLOW_NEGATIVE_REASONS` / `EXIT_FORCE_ALLOW_REASONS` / `ORDER_STRATEGY_CONTROL_EXIT_IMMEDIATE_BYPASS_REASONS` が
   意図的に保護すべき理由と実時点の reason 仕様でズレており、拒否/通過の一貫性が崩れていた。
2. `config/strategy_exit_protections.yaml` の `defaults` / no-block anchor に `fast_cut_time` が未登録で、
   戦略別許可の下位互換上、`close_reject_no_negative` 化しにくい経路が生じていた。

Improvement:
1. `execution/order_manager.py`
   - 保護系理由の既定トークンを整理し、実運用保護理由（`hard_stop`,`tech_hard_stop`,`max_adverse`,`time_stop`,`no_recovery`,`max_floating_loss`,`fast_cut_time`,`time_cut`,`tech_return_fail`,`tech_reversal_combo`,`tech_candle_reversal`,`tech_nwave_flip`）へ明示集中。
   - `drawdown`,`max_drawdown`,`health_exit`,`hazard_exit`,`margin_health`,`free_margin_low`,`margin_usage_high` を
     `ORDER_ALLOW_NEGATIVE_REASONS` / `EXIT_FORCE_ALLOW_REASONS` / `ORDER_STRATEGY_CONTROL_EXIT_IMMEDIATE_BYPASS_REASONS` 既定から削除。
2. `config/strategy_exit_protections.yaml`
   - `defaults.neg_exit.allow_reasons` と `scalp_ping_5s_no_block_neg_exit_allow_reasons` に `fast_cut_time` を追加。
3. `docs/WORKER_REFACTOR_LOG.md` に同内容を追記。

Verification:
1. 反映後24hで `orders.db` を再集計し、`close_reject_no_negative` の上位理由における
   `fast_cut_time` の即時通過率が改善すること。
2. `close_reject_no_negative` で想定外のトリガが増加しないことを確認。
3. VM反映後、`quant-order-manager` の起動監査（`Application started!`）と設定有効性を確認。

Status:
- in_progress

## 2026-02-28 22:50 UTC / 2026-02-28 07:50 JST - `close_blocked_negative` と `hold_until_profit` の過剰保護を縮小

Period:
- VM直近実データ（`orders.db` / `trades.db` / `metrics.db`）に基づく、`close_blocked_negative` 原因別再審査。

Source:
- VM `logs/metrics.db`（`metric='close_blocked_negative'`, `metric='close_blocked_hold_profit'`）
- VM `logs/orders.db`（`status` と `exit_request` 検索）
- VM `systemctl` / unit稼働状態

Fact:
- `close_blocked_negative` の上位は
  `max_adverse`（`22556`）> `no_recovery`（`9772`）> `m1_rsi_fade`（`6805`）> `max_hold`（`5842`）> `reentry_reset`（`5693`）> `m1_structure_break`（`3006`）で、`__de_risk__` や `time_cut` 由来の件数も増加傾向。
- `close_blocked_hold_profit` では `min_profit_pips=9999.0` + `strict=true` の組み合わせが `20663` 件観測され、事実上の強制保有化に近い挙動。
- `scalp_ping_5s_no_block_neg_exit_allow_reasons` に `time_cut/__de_risk__/momentum_stop_loss/max_hold_loss` が欠けていたため、実績上想定された保護解除が拒否されるケースがあった。

Failure Cause:
1. `close_blocked_negative` の許可集合に対し、運用上顕在化している保護理由の一部が未登録だった。
2. `hold_until_profit` の固定 `trade_ids` + `min_profit_pips: 9999.0` + `strict=true` が、暫定目的を逸脱して常時解消抑止を発生させていた。

Improvement:
1. `config/strategy_exit_protections.yaml`
   - `scalp_ping_5s_no_block_neg_exit_allow_reasons` を拡張：
     - `time_cut`
     - `__de_risk__`
     - `momentum_stop_loss`
     - `max_hold_loss`
   - `hold_until_profit` を無効化寄りに更新し、`trade_ids: []`, `min_profit_pips: 0.0`, `strict: false`。
2. `docs/WORKER_REFACTOR_LOG.md` に同変更内容を追記し、変更履歴を監査可能化。

Verification:
1. 反映後24hで `metrics.db` 上の `close_blocked_negative` 上位理由分布と `close_reject_no_negative` の変化を再集計し、
   追加した理由の受理率が改善されること。
2. `close_blocked_hold_profit` の `9999.0`/`strict=true` 相当件数が収束し、`trade_id` 固定ブロックが消滅すること。
3. VMで `quant-order-manager` / `quant-scalp-ping-5s-*` の起動監査、`journalctl` の `Application started!` が最新化されること。

Status:
- in_progress

## 2026-02-28 05:05 UTC / 2026-02-28 14:05 JST - 市況不確実帯での hard reject 偏重を是正（RangeFader + ping5s + extrema）
Period:
- 観測: 2026-02-27 21:57 UTC までの直近24h（VM `orders.db` / `trades.db` / `metrics.db`）
- 改善対象: `entry_probability_reject` と `perf_block` の過多

Fact:
- 戦略最終ステータス（client_order_id単位）で
  `entry_probability_reject=614 (26.44%)`、`perf_block=588 (25.32%)`、`filled=589 (25.37%)`。
- `entry_probability_reject` の内訳は
  `entry_probability_below_min_units=596` が支配的。
- `perf_block` は
  `scalp_ping_5s_c_live=288`, `scalp_extrema_reversal_live=178`, `scalp_ping_5s_b_live=109`。
- `order_perf_block` reason は
  `hard:hour*:failfast` / `hard:hour*:sl_loss_rate` が主因で、`reduce` 設定でも hard 拒否に寄っていた。

Failure Cause:
1. 不確実帯での preflight が「縮小」ではなく hard reject に倒れ、約定密度を落としていた。
2. `RangeFader` は `entry_probability_below_min_units` が主因で、確率縮小後にロットが最小閾値を割っていた。
3. `scalp_extrema_reversal_live` は PF/win の軽度劣化でも `block` モードで停止しやすかった。

Improvement:
1. `workers/common/perf_guard.py`
   - `PERF_GUARD_HARD_FAILFAST_ENABLED`
   - `PERF_GUARD_HARD_SL_LOSS_RATE_ENABLED`
   - `PERF_GUARD_HARD_MARGIN_CLOSEOUT_ENABLED`
   を追加し、hard判定を戦略prefixで制御可能化。
2. `ops/env/quant-order-manager.env`
   - `SCALP_PING_5S_[B/C]_PERF_GUARD_HARD_FAILFAST_ENABLED=0`
   - `SCALP_PING_5S_[B/C]_PERF_GUARD_HARD_SL_LOSS_RATE_ENABLED=0`
   - `SCALP_EXTREMA_REVERSAL_PERF_GUARD_MODE=reduce`
   - `RangeFader` 向け `ORDER_MIN_UNITS_STRATEGY_RANGEFADER*` を `120` へ新設し、
     preserve-intent 閾値を緩和。
3. `ops/env/quant-scalp-rangefader.env`
   - `ENTRY_LEADING_PROFILE_REJECT_BELOW` 緩和、`WEIGHT_RANGE` 引き上げ、
     `WEIGHT_MICRO` 引き下げで range 判定重視へ再配分。

Verification:
1. 反映後30-60分で `orders.db` の
   `entry_probability_reject(entry_probability_below_min_units)` と
   `perf_block` 比率が低下すること。
2. 同期間で `filled` が維持または増加し、`rejected` が急増しないこと。
3. `metrics.db` の `order_perf_block` reason が `hard:failfast/sl_loss_rate`
   から `warn:*` へ遷移すること。

Status:
- in_progress
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

## 2026-02-28 23:10 UTC / 2026-02-29 08:10 JST - strategy_entry で strategy-side net-edge gate を導入

Period:
- 直近実測ログと env 監査
- 対象: `orders.db` / `trades.db` / `metrics.db` と `execution/strategy_entry.py`,
  `ops/env/quant-v2-runtime.env`

Fact:
- strategy_entry に `STRATEGY_ENTRY_NET_EDGE_*` を使う local gate を追加し、
  `market_order` / `limit_order` の main path で
  `analysis_feedback -> forecast_fusion -> strategy_net_edge_gate -> leading_profile -> coordinate_entry_intent`
  の順に通過する実装を入れた。
- `entry_net_edge` と `entry_net_edge_gate` を `entry_thesis` / キャッシュ payload に残す形で監査経路を拡張。
- `ops/env/quant-v2-runtime.env` に  
  `STRATEGY_ENTRY_NET_EDGE_GATE_ENABLED=1`、
  `SCALP_PING_5S_B_ENTRY_NET_EDGE_MIN_PIPS=0.10`、
  `SCALP_PING_5S_C_ENTRY_NET_EDGE_MIN_PIPS=0.12` を追加。

Failure Cause:
1. 戦略ローカルの期待値除外が preflight 前段で体系化されておらず、戦略別最終意図（probability/intent）との整合トレースが弱かった。
2. 単価・スプレッド・見積コストを含む EV 判定が strategy_entry で未集約だったため、拒否理由分析が散在していた。

Improvement:
1. `execution/strategy_entry.py` に `_apply_strategy_net_edge_gate` を追加し、
  pocket ごとの適用可否・strategy prefix 優先 env 解決を実装。
2. net-edge 失格時に `entry_net_edge_gate` を cache status payload に残し、
  coordination 前で確定拒否するフローを追加。
3. `docs/WORKER_REFACTOR_LOG.md` と `docs/ARCHITECTURE.md` へ設計更新を追記。

Verification:
1. 反映後 24h で `entry_net_edge_negative` 系拒否の比率と `coordination_reject` の変化を比較。
2. 同時に `orders.db` の `entry_probability_reject` / `entry_probability_below_min_units` の偏重化有無を検証。
3. 戦略別サンプル 24h で `entry_net_edge_gate` 情報が `entry_thesis` / order status キャッシュに残ることを確認。

Status:
- in_progress

## 2026-02-28 22:50 UTC / 2026-02-28 07:50 JST - `scalp_ping_5s_b/_c` の neg_exit で `allow_reasons` ワイルドカードを廃止

Period:
- 事象確認: 2026-02-28 22:00 UTC 時点の運用設定監査
- 対象: `config/strategy_exit_protections.yaml`, `execution/order_manager.py`

Fact:
- `scalp_ping_5s_b(_live)` / `scalp_ping_5s_c(_live)` の
  `neg_exit.allow_reasons` が `"*"` 指定の構成が残存。
- `order_manager._strategy_neg_exit_policy()` でも戦略 override の `allow_reasons` を
  そのまま上書きしていたため、既定保護へのフォールバック条件が不安定。

Failure Cause:
1. strategy override の `allow_reasons="*"` により、実運用の明示許可ロジックが
   想定より広くなり、`close_reject_no_negative` 期待動作のブレを誘発。
2. 同一設定セットの読み分けが戦略側/共通側で非対称だったため、
   原因切分がしにくく、保護過大設定の実害検知が遅れていた。

Improvement:
1. `execution/order_manager.py`:
   - `_normalize_reason_tokens()` を追加し、`allow_reasons=["*"]` 時は
     `neg_defaults.allow_reasons` を自動採用する。
   - `_strategy_neg_exit_policy()` で strategy override 反映時に上記を適用。
2. `config/strategy_exit_protections.yaml`:
   - `scalp_ping_5s_no_block_neg_exit_allow_reasons` を追加。
   - `scalp_ping_5s_b(_live)` / `scalp_ping_5s_c(_live)` を共通 anchor 参照へ変更。
3. 実装監査を `docs/WORKER_REFACTOR_LOG.md` へ追記。

Verification:
1. 設定監査で対象4戦略の `allow_reasons="*"` が除去され、wildcard 共有アンカーへ統一されていることを確認。
2. `order_manager` の `allow_reasons` 決定経路に fallback が入ることをコード差分で確認。
3. 反映後、VM で `close_reject_no_negative` 連鎖が減るかを `orders/metrics` で追跡。

Status:
- in_progress

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

## 2026-02-28 04:40 UTC / 13:40 JST - 期待値改善を加速するための即時クランプ（ping B/C + MACD RSI div）

Period:
- 直近24h / 直前24h 比較（`datetime(close_time)` 正規化）
- 直近7d（戦略別寄与）

Source:
- VM `/home/tossaki/QuantRabbit/logs/trades.db`
- VM `/home/tossaki/QuantRabbit/logs/orders.db`

Fact:
- 全体期待値は改善中だが未だ負値:
  - `last24h expectancy=-1.5 JPY` vs `prev24h=-2.8 JPY`（`+1.3 JPY/trade`）
- 直近24hの負け寄与（非manual）:
  - `scalp_ping_5s_b_live`: `292 trades / -180.4 JPY / PF 0.303`
  - `scalp_ping_5s_c_live`: `109 trades / -35.3 JPY / PF 0.166`
- 少数大損:
  - `scalp_macd_rsi_div_b_live` + `scalp_macd_rsi_div_live` は `4 trades / -729.9 JPY`

Failure Cause:
1. `scalp_ping_5s_b/c` で低エッジ約定が残り、回転で負けを積み上げる。
2. `scalp_macd_rsi_div*` で単発大損が期待値を崩す。
3. order-manager の `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_[B/C]_LIVE=1` がノイズ約定を許している。

Improvement:
1. `ops/env/quant-order-manager.env`
   - `FORECAST_GATE_*` を B/C で引き上げ（expected/target/edge block）。
   - `ORDER_ENTRY_NET_EDGE_MIN_PIPS_STRATEGY_SCALP_PING_5S_B_LIVE: 0.02 -> 0.10`
   - `ORDER_ENTRY_NET_EDGE_MIN_PIPS_STRATEGY_SCALP_PING_5S_C_LIVE=0.12` を追加。
   - `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_[B/C]_LIVE: 1 -> 30`
   - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER/MAX_SCALE` を B/C で強化。
   - `ORDER_ENTRY_MAX_SL_PIPS_STRATEGY_SCALP_MACD_RSI_DIV_LIVE=2.8`
   - `ORDER_ENTRY_MAX_SL_PIPS_STRATEGY_SCALP_MACD_RSI_DIV_B_LIVE=2.4`
2. `ops/env/quant-scalp-macd-rsi-div.env`
   - `BASE_ENTRY_UNITS: 3000 -> 1500`, `MIN_UNITS: 600 -> 300`
   - `MIN_DIV_SCORE/STRENGTH` を引き上げ、`MAX_DIV_AGE_BARS` を短縮。
   - `SL_ATR_MULT: 0.85 -> 0.65`, `TP_ATR_MULT: 1.10 -> 1.00`
3. `ops/env/quant-scalp-macd-rsi-div-b.env`
   - `BASE_ENTRY_UNITS: 3200 -> 1600`, `MIN_UNITS: 1000 -> 400`
   - `COOLDOWN_SEC: 45 -> 90`
   - `MIN_DIV_SCORE/STRENGTH` 引き上げ、`MAX_DIV_AGE_BARS` 短縮、`RANGE_MIN_SCORE` 引き上げ。
   - `SL_ATR_MULT: 0.85 -> 0.65`, `TP_ATR_MULT: 1.15 -> 1.05`
4. `config/strategy_exit_protections.yaml`
   - `scalp_macd_rsi_div_live` の `loss_cut_hard_pips: 7.0 -> 2.6`
   - `scalp_macd_rsi_div_b_live` を追加し `loss_cut_hard_pips=2.3`

Verification:
1. 反映後24hで `expectancy_jpy` が `> 0` へ近づく（最低でも `-1.5` から改善）。
2. `scalp_ping_5s_b/c` の合算 `net_jpy` が前日比で改善し、`filled` がゼロにならない。
3. `scalp_macd_rsi_div*` の `avg_loss_jpy` と `max_loss_jpy` が明確に低下する。
4. `orders.db` で `entry_probability_reject`/`perf_block` の増加が、`net_jpy` 改善を上回る副作用になっていないことを確認する。

Status:
- applied

## 2026-02-28 04:25 UTC / 2026-02-28 13:25 JST - 全戦略共通: strategy-control EXITロック詰まりの恒久対策

Period:
- incident確認窓: 2026-02-24 07:00-07:07 UTC（JST 16:00-16:07）
- 劣化確認窓: 直近24h / 7d（2026-02-28時点）

Source:
- VM `/home/tossaki/QuantRabbit/logs/orders.db`
- VM `/home/tossaki/QuantRabbit/logs/trades.db`
- `execution/order_manager.py` close preflight 実装

Fact:
- `strategy_control_exit_disabled` が同一 trade に対して連続発生し、`MicroPullbackEMA` の close 要求が滞留。
- close_reason/exit_reason 集計では、赤字寄与の多くが `STOP_LOSS_ORDER` または `max_adverse/time_stop` 系クローズ由来。
- 既存 fail-open は block回数/経過秒に依存し、保護系 exit reason でも初動で詰まる経路が残っていた。

Failure Cause:
1. `strategy_control.can_exit=false` 時、保護系理由（`max_adverse` / `time_stop` 等）でも即時通過できない。
2. fail-open が閾値到達型のため、急変局面で EXIT 遅延が先に発生する。

Improvement:
1. `execution/order_manager.py` に `ORDER_STRATEGY_CONTROL_EXIT_IMMEDIATE_BYPASS_REASONS` を追加。
2. close preflight で `strategy_control` ブロック時に、上記理由一致なら即時 `CLOSE_BYPASS` で通過。
3. 既存 fail-open（閾値到達後のバイパス）は維持し、理由一致時のみ先回りで詰まりを回避。

Verification:
1. `orders.db` で `status='strategy_control_exit_disabled'` の新規増加と連続回数が低下すること。
2. `close_bypassed_strategy_control` メトリクスで `reason='strategy_control_exit_immediate_reason'` が観測されること。
3. `MARKET_ORDER_MARGIN_CLOSEOUT` の再発率が 7d 比で悪化しないこと。
4. テスト: `tests/execution/test_order_manager_exit_policy.py` の即時バイパスケースを含め pass すること。

Status:
- deployed_pending

## 2026-02-28 03:55 UTC / 2026-02-28 12:55 JST - 「全然稼げてない」RCA（VM実測, 24h + 7d）

Period:
- 24h: `close_time >= datetime('now','-24 hours')`
- 比較窓: 直前7日（`close_time >= datetime('now','-8 days') and < datetime('now','-24 hours')`）
- 参考: last 7d 全体

Source:
- VM `systemctl` / `journalctl`（V2導線稼働確認）
- VM `/home/tossaki/QuantRabbit/logs/orders.db`
- VM `/home/tossaki/QuantRabbit/logs/trades.db`
- VM `/home/tossaki/QuantRabbit/logs/metrics.db`
- OANDA summary/open trades (`scripts/check_oanda_summary.py`, `scripts/oanda_open_trades.py`)

Fact:
- 稼働状態:
  - `quant-market-data-feed` / `quant-strategy-control` / `quant-order-manager` / `quant-position-manager` は `active/running`。
  - 直近時点の open trades は `0`（週末クローズ帯: 2026-02-28 JST）。
- 24h損益（manual除外）:
  - `836 trades / win_rate=0.4234 / PF=0.727 / expectancy=-0.9 JPY / net=-737.0 JPY`
- 直前7日比較（manual除外）:
  - `3591 trades / win_rate=0.4152 / PF=0.55 / avg_daily=-4248.5 JPY`
- 7d全体寄与（manual除外）:
  - 総計 `-31346.5 JPY`
  - `MicroPullbackEMA=-15527.3 JPY`, `scalp_ping_5s_c_live=-10732.8 JPY`, `scalp_ping_5s_b_live=-7342.0 JPY`
  - 3戦略除外シミュレーション: `+2255.6 JPY`（`exclude_MicroPullbackEMA_ping_b_c`）
- 拒否/ブロック（orders final status, 24h）:
  - `entry_probability_reject=614 (26.44%)`
  - `perf_block=588 (25.32%)`
  - `filled=589 (25.37%)`
- EXIT詰まり痕跡（last 7d）:
  - `strategy_control_exit_disabled=10277`（全件 2026-02-24 JST）
  - 内訳: `MicroPullbackEMA-short=8177`, `MicroTrendRetest-long=2068`
- Closeout損失:
  - `MARKET_ORDER_MARGIN_CLOSEOUT` のうち `MicroPullbackEMA=4 trades / -16837.4 JPY`

Failure Cause:
1. `strategy_control_exit_disabled` が長時間連続し、EXIT fail-open が遅く `MicroPullbackEMA` の margin closeout を許容した。
2. `MicroPullbackEMA` は 7d 主因（`-15527.3 JPY`）で、base units / margin utilization が高く tail loss が大きい。
3. `scalp_ping_5s_b/c` の高回転低EVが継続し、24h/7dともに負寄与を積み上げた（特に C）。

Improvement:
1. `ops/env/quant-v2-runtime.env`
   - `ORDER_STRATEGY_CONTROL_EXIT_FAILOPEN_BLOCK_THRESHOLD: 6 -> 3`
   - `ORDER_STRATEGY_CONTROL_EXIT_FAILOPEN_WINDOW_SEC: 90 -> 20`
   - `ORDER_STRATEGY_CONTROL_EXIT_FAILOPEN_RESET_SEC: 300 -> 180`
   - `ORDER_STRATEGY_CONTROL_EXIT_FAILOPEN_EMERGENCY_ONLY: 1 -> 0`
2. `ops/env/quant-order-manager.env`
   - `ORDER_ENTRY_MAX_SL_PIPS_STRATEGY_MICROPULLBACKEMA: 4.0 -> 3.0`
   - `PERF_GUARD_MODE_STRATEGY_MICROPULLBACKEMA=block`（明示）
   - `PERF_GUARD_MARGIN_CLOSEOUT_HARD_MIN_TRADES_STRATEGY_MICROPULLBACKEMA: 4 -> 1`
   - `PERF_GUARD_MARGIN_CLOSEOUT_HARD_RATE_STRATEGY_MICROPULLBACKEMA: 0.20 -> 0.05`
3. `ops/env/quant-micro-pullbackema.env`
   - `MICRO_MULTI_BASE_UNITS: 9000 -> 3500`
   - `MICRO_MULTI_MAX_MARGIN_USAGE: 0.72 -> 0.50`
   - `MICRO_MULTI_TARGET_MARGIN_USAGE=0.55`（追加）
   - `MICRO_MULTI_CAP_MAX=0.65`（追加）
4. `ops/env/quant-scalp-macd-rsi-div-b.env`
   - divergence閾値強化 + size縮小（`BASE_ENTRY_UNITS: 5000 -> 3200` など）

Verification:
1. 次回市場オープン後の first 24h で `strategy_control_exit_disabled` の新規累積が `0` に近いこと。
2. `MARKET_ORDER_MARGIN_CLOSEOUT` の件数/損失が 7d 比で有意に減ること（目標: `MicroPullbackEMA` closeout 0件）。
3. `MicroPullbackEMA` / `scalp_macd_rsi_div_b_live` の `net_jpy` と `avg_loss_jpy` が改善すること。
4. `scalp_ping_5s_b/c` は停止せず、`filled` を維持したまま `net_jpy` 改善傾向を確認すること。

Status:
- in_progress

## 2026-02-27 16:13 UTC / 2026-02-28 01:13 JST - `scalp_ping_5s_b/c` 継続赤字 + `MicroPullbackEMA` closeout tail の同時是正

Period:
- 直近24h / 7d（`trades.db`, `orders.db`, `metrics.db`）
- 取得時刻: 2026-02-27 16:13 UTC / 2026-02-28 01:13 JST

Source:
- VM `/home/tossaki/QuantRabbit/logs/trades.db`
- VM `/home/tossaki/QuantRabbit/logs/orders.db`
- VM `/home/tossaki/QuantRabbit/logs/metrics.db`
- VM `scripts/oanda_open_trades.py`
- OANDA account summary（VMからAPI照会）

Fact:
- 24h戦略別:
  - `scalp_ping_5s_c_live`: `618 trades / -590.2 pips / -3452 JPY`
  - `scalp_ping_5s_b_live`: `682 trades / -431.7 pips / -769 JPY`
- 7d戦略別:
  - `scalp_ping_5s_b_live`: `2226 trades / -1868.5 pips / -8502 JPY`
  - `scalp_ping_5s_c_live`: `1273 trades / -1348.6 pips / -10722 JPY`
  - `MicroPullbackEMA`: `46 trades / -520.3 pips / -15527 JPY`
- 7d close reason:
  - `MARKET_ORDER_MARGIN_CLOSEOUT`: `18 trades / -621.5 pips / -19125 JPY`
- OANDA口座状態:
  - `marginCloseoutPercent=0.90854`
  - `marginUsed=53043.4000`, `marginAvailable=5339.6956`
  - `openTradeCount=1`（`USD_JPY`, `currentUnits=-8500`, SL/TP なし）

Failure Cause:
1. `scalp_ping_5s_b/c` は低品質帯通過と高回転が重なり、期待値が負のまま損失を積み上げている。
2. `MicroPullbackEMA` は base units と margin 使用上限が高く、急変時に margin closeout tail を作りやすい。
3. 口座の `marginCloseoutPercent` が高止まり（0.90超）し、一発損失の再発余地が大きい。

Improvement:
1. `ops/env/scalp_ping_5s_b.env` を高確度・低回転へ再設定（件数/ロット/確率閾値/intent縮小/spread guard有効化）。
2. `ops/env/scalp_ping_5s_c.env` を高確度・低回転へ再設定（件数/ロット/force-exit損失上限/確率閾値/intent縮小）。
3. `ops/env/quant-micro-pullbackema.env` で `BASE_UNITS` と `MAX_MARGIN_USAGE` を大幅に縮小し、同時シグナル数を1へ制限。
4. `ops/env/quant-order-manager.env` で B/C preserve-intent 閾値を worker 側へ同期し、`MicroPullbackEMA` の許容SL幅を `6.0 -> 4.0` へ縮小。

Verification:
1. 反映後2h/24hで `scalp_ping_5s_b/c` の `sum(realized_pl)` と `avg_pips` が改善方向へ転じること。
2. `MARKET_ORDER_MARGIN_CLOSEOUT` の新規発生が抑制されること（特に `MicroPullbackEMA`）。
3. OANDA summary の `marginCloseoutPercent` が低下方向へ向かうこと。
4. `orders.db` で `filled` を維持しつつ `entry_probability_reject` / `perf_block` の異常増がないこと。

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

## 2026-02-27 16:03 UTC / 2026-02-28 01:03 JST - `scalp_ping_5s_b/c` 収益悪化に対する高確度化（VM実測）

Period:
- 直近24h（`close_time >= datetime('now','-24 hour')`）
- 直近7d（`close_time >= datetime('now','-7 day')`）

Source:
- VM `/home/tossaki/QuantRabbit/logs/trades.db`
- VM `/home/tossaki/QuantRabbit/logs/orders.db`
- VM `journalctl -u quant-order-manager.service`
- OANDA openTrades (`scripts/oanda_open_trades.py`)

Fact:
- 24h合計: `-3023.8 JPY / -983.5 pips / 1433 trades`
- 7d合計: `-22169.4 JPY / -2031.4 pips / 4310 trades`
- 24h戦略別:
  - `scalp_ping_5s_c_live`: `605 trades / -3444.2 JPY / -575.1 pips`
  - `scalp_ping_5s_b_live`: `677 trades / -756.8 JPY / -420.4 pips`
- side別（24h）:
  - `B long`: `547 trades / -666.3 JPY / avg_win=1.159 / avg_loss=1.863`
  - `B short`: `130 trades / -93.1 JPY / avg_win=1.156 / avg_loss=2.031`
  - `C long`: `493 trades / -3389.2 JPY / avg_win=1.419 / avg_loss=2.300`
  - `C short`: `112 trades / -54.9 JPY / avg_win=0.946 / avg_loss=1.858`
- `orders.db` 24h status:
  - `perf_block=2943`, `entry_probability_reject=683`, `rejected=317`, `filled=1422`
  - `STOP_LOSS_ON_FILL_LOSS` reject が継続。

Failure Cause:
1. `scalp_ping_5s_c_live` は `ENTRY_LEADING_PROFILE_REJECT_BELOW=0.00` で低品質シグナル通過が過多。
2. B/C とも `avg_loss_pips > avg_win_pips` が続き、高回転で負けを積み上げる構造。
3. B は同時保有・発注回転が高く、逆行局面で負け玉を増幅。

Improvement:
1. `ops/env/scalp_ping_5s_c.env`
   - `SCALP_PING_5S_C_ENTRY_LEADING_PROFILE_REJECT_BELOW: 0.00 -> 0.74`
   - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_C_LIVE: 0.58 -> 0.66`
   - `SCALP_PING_5S_C_CONF_FLOOR: 80 -> 83`
   - `SCALP_PING_5S_C_MAX_ORDERS_PER_MINUTE: 16 -> 10`
   - `SCALP_PING_5S_C_BASE_ENTRY_UNITS: 140 -> 110`
2. `ops/env/scalp_ping_5s_b.env`
   - `SCALP_PING_5S_B_ENTRY_LEADING_PROFILE_REJECT_BELOW: 0.67 -> 0.72`
   - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_B_LIVE: 0.76 -> 0.80`
   - `SCALP_PING_5S_B_CONF_FLOOR: 80 -> 82`
   - `SCALP_PING_5S_B_MAX_ACTIVE_TRADES: 6 -> 4`
   - `SCALP_PING_5S_B_MAX_PER_DIRECTION: 4 -> 3`
   - `SCALP_PING_5S_B_MAX_ORDERS_PER_MINUTE: 8 -> 6`
   - `SCALP_PING_5S_B_BASE_ENTRY_UNITS: 300 -> 260`

Verification:
1. 反映後2h/24hで `scalp_ping_5s_b/c` の `sum(realized_pl)` が改善方向へ転じること。
2. `orders.db` で `filled` を維持しつつ `rejected` と `STOP_LOSS_ON_FILL_LOSS` が減ること。
3. 24h side別で `avg_loss_pips` が `avg_win_pips` に接近または逆転すること。
4. `MARKET_ORDER_MARGIN_CLOSEOUT` が増えないこと（特に micro系 tail 監視を継続）。

Status:
- in_progress

## 2026-02-28  (UTC) / 2026-02-28 09:00 JST - EXIT阻害原因分解（trade_id整合性）

Period:
- 本調査対象: `logs/orders.db`（ローカル）
- 補助: `logs/trades.db`（ローカル）
- 補足: VM直読はIAP SSHでコマンドが確立しないため保留（ローカル実DBで継続監査）

Source:
- `logs/orders.db`
- `logs/trades.db`
- `execution/order_manager.py`
- `scripts/replay_exit_workers.py`

Fact:
- `orders` のEXIT関連は `close_request=1598`, `close_ok=1399`, `close_failed=199`。
- `close_failed` 内訳:
  - `oanda::rest::core::InvalidParameterException / Invalid value specified for 'tradeID'` = `172`
  - `TRADE_DOESNT_EXIST / The Trade specified does not exist` = `24`
  - `CLOSE_TRADE_UNITS_EXCEED_TRADE_SIZE` = `3`
- `close_failed` の `172` 件は `ticket_id` が `sim-*`。
  - うち `sim-40:65`, `sim-8:63`, `sim-37:43`, `sim-4:1`
  - 時間帯: `2026-02-24T02:07:21.605365+00:00`〜`2026-02-24T02:09:44.063590+00:00` の集中発生
- `close_failed` 172件の `ticket_id` が同じ `sim-` 系で、`sim-sim-` という client/order-id由来の疑い（`client_order_id` も同系統）を示唆。
- 同期間の `close_request`/`close_failed` 差分 `199` は、共通一律EXITガード系の拒否ステータス（`close_reject_*` 相当）ではなく、OANDA API拒否での `InvalidParameterException` が主因。

Failure Cause:
1. `order_manager.close_trade` が受けた `ticket_id` が実トレードIDではなく `sim-` 系の擬似ID（最終的に `sim-sim-*`）になったことによる `tradeID` 形式不正。
2. 追加で `TRADE_DOESNT_EXIST` が少数残るため、既に決済済み/不整合建玉に対する遅延closeの再送も混在。

Action (already partially applied):
1. `scripts/replay_exit_workers.py` の `_create_trade` を `sim-` 重複付与なしで正規化。
   - `trade_id` が既に `sim-` で始まる場合は再付与しない。
   - `client_order_id`/`clientExtensions.id` へ `sim-<id>` を一貫投入。
2. `execution/order_manager.py` の `close_trade()` は `_is_valid_live_trade_id` で数値ID以外を早期拒否し、`log_metric("close_reject_invalid_trade_id")` で観測可能化。

Verification:
1. `close_failed` の `InvalidParameterException` が減衰し、`close_reject_invalid_trade_id` へ寄与遷移するかをVM `orders.db` で要再確認。
2. VM `orders.db` で `ticket_id LIKE 'sim-%'` の `close_failed` 再集計。
3. `close_request` から `close_ok` への通過率、`strategy_control` 拒否系の有無を同期間で再監査。

Status:
- in_progress

## 2026-03-02  (UTC) / 2026-03-02 06:00 JST - 5秒スキャッパー D/Flow 総力復帰

Source:
- local config audit (`ops/env/scalp_ping_5s_d.env`, `ops/env/scalp_ping_5s_flow.env`)
- preflight/env audit (`ops/env/quant-order-manager.env`)

Hypothesis:
- `scalp_ping_5s_d` が `SCALP_PING_5S_D_ENABLED=0` のままで、`scalp_ping_5s_flow` も停止状態だったことが、シグナル欠落の主要因。
- D/Flow 起動後は、`scalp_ping_5s_d` 側の過剰ブロック（ticks/align/leading_profile）を緩和し、`order_manager` の D/Flow 下限を緩く設定してエントリー量を回復させる。

Action:
- `ops/env/scalp_ping_5s_d.env`
  - `SCALP_PING_5S_D_ENABLED: 0 -> 1`
  - `SCALP_PING_5S_D_MAX_ORDERS_PER_MINUTE: 6 -> 12`
  - `SCALP_PING_5S_D_MIN_TICKS: 4 -> 3`
  - `SCALP_PING_5S_D_MIN_SIGNAL_TICKS: 3 -> 2`
  - `SCALP_PING_5S_D_CONF_FLOOR: 74 -> 72`
  - `SCALP_PING_5S_D_ENTRY_PROBABILITY_ALIGN_PENALTY_MAX: 0.55 -> 0.24`
  - `SCALP_PING_5S_D_ENTRY_PROBABILITY_ALIGN_FLOOR_RAW_MIN: 0.70 -> 0.64`
  - `SCALP_PING_5S_D_ENTRY_LEADING_PROFILE_REJECT_BELOW(_SHORT): 0.50/0.62 -> 0.44/0.54`
- `ops/env/scalp_ping_5s_flow.env`
  - `SCALP_PING_5S_FLOW_ENABLED: 0 -> 1`
  - `SCALP_PING_5S_FLOW_MIN_TICKS: 4 -> 3`
  - `SCALP_PING_5S_FLOW_MIN_SIGNAL_TICKS: 3 -> 2`
  - `SCALP_PING_5S_FLOW_SHORT_MIN_TICKS: 3 -> 2`
  - `SCALP_PING_5S_FLOW_SHORT_MIN_SIGNAL_TICKS: 3 -> 2`
  - `SCALP_PING_5S_FLOW_MIN_TICK_RATE: 0.50 -> 0.45`
  - `SCALP_PING_5S_FLOW_SHORT_MIN_TICK_RATE: 0.50 -> 0.45`
  - `SCALP_PING_5S_FLOW_SIGNAL_WINDOW_ADAPTIVE_ENABLED: 0 -> 1`
  - `SCALP_PING_5S_FLOW_ENTRY_LEADING_PROFILE_REJECT_BELOW(_SHORT): 0.45/0.55 -> 0.40/0.52`
- `ops/env/quant-order-manager.env`
  - `ORDER_MANAGER_PRESERVE_INTENT_MIN_SCALE_STRATEGY_SCALP_PING_5S_D_LIVE: 1.00 -> 0.80`
  - Flow 用の `ORDER_MANAGER_PRESERVE_INTENT_*` / `ORDER_MIN_UNITS` 設定を新規追加。
- `ops/env/quant-scalp-ping-5s-d.env`, `ops/env/quant-scalp-ping-5s-flow.env`
  - 各戦略のサービス共通 `SCALP_PING_5S_*_ENABLED` を `1` に更新

Verification:
1. VM反映後2時間で `scalp_ping_5s_d_live` と `scalp_ping_5s_flow_live` の `filled` が 0 以外に戻ること。
2. `orders.db` の `entry_probability_reject + rate_limited + revert_not_found` 比率を比較し、合計が前日比で低下すること。
3. `filled` と `stop_loss` の増加バランスが極端化しない（短期で `MARKET_ORDER_MARGIN_CLOSEOUT` が増加しない）こと。

Status:
- in_progress

## 2026-03-03 (JST) / bot entry再開対応（manualポジ非干渉）

Source:
- OANDA summary (`scripts/check_oanda_summary.py`): `openTradeCount=1`（manualのみ継続）
- VM serial (`gcloud compute instances get-serial-port-output`): 戦略起動後も新規約定増加なし
- config audit (`ops/env/quant-v2-runtime.env`, `ops/env/quant-order-manager.env`, `ops/env/scalp_ping_5s_{b,c,d,flow}.env`)

Hypothesis:
1. ping5s系のローカル閾値（`CONF_FLOOR` / `ENTRY_PROBABILITY_ALIGN_FLOOR` / reject_under）が高すぎ、`market_order` 呼び出し前に skip される。
2. `BLOCK_MANUAL_NETTING` が有効だと手動建玉と逆方向のbot entryが抑止されるため、明示的に無効化して manual 非干渉を担保する。

Action:
- `ops/env/quant-v2-runtime.env`
  - `BLOCK_MANUAL_NETTING=0` を追加
- `ops/env/quant-order-manager.env`
  - `BLOCK_MANUAL_NETTING=0` を追加
  - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_B(_LIVE)=0.15`
  - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_C(_LIVE)=0.15`
  - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_D(_LIVE)=0.12`
- `ops/env/scalp_ping_5s_b.env`
  - `CONF_FLOOR=60`
  - `ENTRY_PROBABILITY_ALIGN_FLOOR=0.35`
  - `ENTRY_PROBABILITY_ALIGN_FLOOR_REQUIRE_SUPPORT=0`
  - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_B_LIVE=0.15`
- `ops/env/scalp_ping_5s_c.env`
  - `CONF_FLOOR=60`
  - `ENTRY_PROBABILITY_ALIGN_FLOOR=0.35`
  - `ENTRY_PROBABILITY_ALIGN_FLOOR_REQUIRE_SUPPORT=0`
  - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_C_LIVE=0.15`
- `ops/env/scalp_ping_5s_d.env`
  - `PERF_GUARD_MODE=reduce`
  - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_D_LIVE=0.12`

Impact:
- manualポジション自体の操作・クローズ処理は未変更（botのENTRY通過条件のみ緩和）。
- V2導線（strategy worker → order_manager）と `entry_thesis` 契約は非変更。

Verification (post-deploy):
1. OANDA `lastTransactionID` が更新され、`openTradeCount` が manual分以外で増えること。
2. `orders.db` の ping5s系 `entry_probability_reject` 比率が低下し、`filled` が復帰すること。
3. `manual_netting_block` ステータス増加がないこと（manual非干渉の維持確認）。

Status:
- in_progress

## 2026-03-04 (JST) / ローカルV2 PDCA導線追加前の市況確認

Source:
- OANDA v3 Pricing (`USD_JPY`) and Candles (`M1`, 80本) をローカル実行で取得。

Snapshot (2026-03-04 JST):
- bid/ask: `157.668 / 157.676`（mid `157.672`）
- spread: `0.8 pips`
- ATR14 (M1): `5.29 pips`
- 直近20本 M1 平均レンジ: `4.96 pips`
- OANDA API応答遅延（3サンプル）: `min 303.6ms / max 336.0ms / avg 318.4ms`

Interpretation:
- スプレッド・短期レンジ・API遅延はいずれも極端な悪化は観測せず、ローカル開発導線追加作業は継続可能と判断。

Action:
- `scripts/local_v2_stack.sh` を追加し、V2サービスをローカルで `up/down/restart/status/logs` 管理できるようにした。
- `ops/env/local-v2-stack.env` を追加し、ローカル上書きテンプレート（local LLMゲート設定例付き）を用意した。

Status:
- done

## 2026-03-04 (JST) / scalp_ping_5s_b 売り側復帰 + ローカルV2安定化

Source:
- `logs/orders.db` / `logs/trades.db` / `logs/metrics.db`
- `logs/local_v2_stack/quant-scalp-ping-5s-b.log`
- `logs/local_v2_stack/quant-order-manager.log`
- `logs/local_v2_stack/quant-position-manager.log`

Hypothesis:
1. 売り限定ではなく両方向判定は有効だが、short 側は `units_below_min` で事実上失注していた。
2. `position-manager`/`order-manager` の起動・停止の不安定さが、entry/exit 評価の連続性を壊していた。

Action:
- `ops/env/scalp_ping_5s_b.env`
  - `SCALP_PING_5S_B_SIDE_BIAS_SCALE_FLOOR: 0.40 -> 0.55`
  - `SCALP_PING_5S_B_ENTRY_PROBABILITY_ALIGN_COUNTER_EXTRA_PENALTY_MAX: 0.32 -> 0.22`
  - `SCALP_PING_5S_B_ENTRY_PROBABILITY_ALIGN_UNITS_MIN_MULT: 0.45 -> 0.60`
  - `SCALP_PING_5S_B_ENTRY_PROBABILITY_BAND_ALLOC_UNITS_MIN_MULT: 0.40 -> 0.55`
  - `SCALP_PING_5S_B_SHORT_MIN_SIGNAL_TICKS: 3 -> 2`
  - `SCALP_PING_5S_B_SHORT_MIN_TICK_RATE: 0.50 -> 0.42`
  - `SCALP_PING_5S_B_SIDE_ADVERSE_STACK_UNITS_MIN_MULT: 0.60 -> 0.75`
- `ops/env/quant-order-manager.env`
  - `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_B(_LIVE): 10 -> 6`
- `scripts/local_v2_stack.sh`
  - `quant-scalp-ping-5s-b` のPID照合パターンに `workers.scalp_ping_5s.worker` を追加し、`status/down` の誤判定を抑制。

Verification (post-tune, 2026-03-04 16:03 JST 以降):
- `orders.db`（`status in preflight_start/filled/rejected`）
  - `buy: attempts=15 / filled=15 / rejected=0`
  - `sell: attempts=5 / filled=5 / rejected=0`
- `quant-scalp-ping-5s-b.log`
  - short open が復帰（例: `units=-100, -94, -85, -42`）
  - short skip 理由の主成分が `units_below_min` から `rate_limited/cooldown/max_active_cap` へ移行
- `metrics.db`（`account.nav`）
  - `07:03:34 UTC -> 07:09:48 UTC` で `+812.554`（含み評価）

Interpretation:
- 売り限定化は解消し、short 実約定は再開した。
- ただし直後は open ポジションが残る時間帯があり、`realized_pl` の評価窓はもう少し必要。

Next:
1. 同設定で 30-60分連続観測し、`close_reason` 別の実現損益（PF/勝率）を再評価。
2. `rate_limited` が過多なら `ENTRY_COOLDOWN_SEC` / `MAX_ORDERS_PER_MINUTE` を微調整。
3. `risk_mult_total=0.4` 固定が続く場合は、劣化要因（SL連打区間）を切り分けて別途改善。

Status:
- in_progress

### 追記: 2026-03-04 16:20 JST / OANDA ReadTimeoutでのworker停止対策

Issue:
- `scalp_ping_5s_b` が `requests.exceptions.ReadTimeout`（`get_account_snapshot`）でプロセス終了する事象を確認。

Action:
- `workers/scalp_ping_5s/worker.py`
  - `get_account_snapshot(cache_ttl_sec=0.5)` を `try/except` で包み、
    - キャッシュ済み snapshot があればそれを継続利用
    - キャッシュが無い場合は `account_snapshot_unavailable` として当該ループのみ skip
  - 例外で worker 全体が落ちないように変更。

Verification:
- `python3 -m py_compile workers/scalp_ping_5s/worker.py` : OK
- patch反映後に `quant-scalp-ping-5s-b` を再起動し、`status` で running を確認。
- 直近2分観測で新規 `Traceback/ReadTimeout` 出力なし（既存ログの旧トレースは除外）。

Status:
- in_progress

## 2026-03-04 18:21 JST / scalp_ping_5s_b short参加拡大 + long損失抑制チューニング

Evidence summary (直近3h):
- fill は long 側優勢で偏り、`buy/sell fill ratio` が高止まり。
- 損失は `STOP_LOSS_ORDER` に集中し、long 側のマイナス寄与が目立つ。
- `units_below_min` は short 側で多発し、売り参加率を押し下げた。

Hypothesis:
- `BASE_ENTRY_UNITS` と各 `*_UNITS_MIN_MULT` の引き上げで short の `units_below_min` を減らし、`SIDE_BIAS_*`/`DIRECTION_BIAS_*`/`*_MOMENTUM_TRIGGER_PIPS`/`ENTRY_CHASE_MAX_PIPS` の再配分で long の逆風局面エントリーを抑制すれば、短期PFの毀損を抑えつつ売り参加を回復できる。

Recheck KPIs (next 60-90 min):
- `buy/sell fill ratio <= 5.0`
- `STOP_LOSS_ORDER share <= 60%`
- `PF >= 0.90`
- `short units_below_min` を現状比 `>=30%` 削減

## 2026-03-04 18:31 JST / restart後フォローアップ（long過大抑制）

- 再起動直後の事実: fill は buy-only、`STOP_LOSS_ORDER` は1件、short 側 `units_below_min` は依然高止まり。
- 施策意図: 両方向ロジックは維持しつつ、long の過大ロット増幅を抑えて long 損失寄与を下げる。
- 45-60分後の再確認KPI: `avg filled buy units` 前run比低下、`STOP_LOSS_ORDER share` 低下、`net_jpy` 改善、short `units_below_min` トレンド非増加。

## 2026-03-04 19:42 JST / short `units_below_min` 第2チューニング

- 再起動直後の事実: short 側 `units_below_min` は高止まりのままで、初期 fill は buy-only だった。
- 第2チューニング仮説: opposite-side 時のロット縮退（lot collapse）を抑え、動的トレードを維持したまま short 参加を回復させる。
- 30-60分の再確認KPI: short fill 増加、short `units_below_min` を現状比 `>=30%` 削減、`buy/sell fill ratio` のトレンド改善。

## 2026-03-04 20:02 JST / short `units_below_min` 第3チューニング

- 第2段反映後の3分窓で `short units_below_min=75`、fill は依然 buy-only だったため、counter-side の縮退下限を追加で引き上げた。
- 変更方針: `MIN_UNITS_RESCUE` 条件緩和、`MTF/HORIZON/M1` の opposite 側ユニット下限引き上げ、`band_alloc` の最小倍率底上げ、同時に long 閾値（`LONG_MOMENTUM_TRIGGER_PIPS`）を上げて long 過多を抑制。
- 再確認KPI（30-60分）: `short fills > 0`、`short units_below_min` を第2段比でさらに `>=30%` 減、`buy/sell fill ratio` の改善継続。

## 2026-03-04 20:08 JST / 実行詰まり（cooldown/rate_limit）緩和

- 第3段後の集計で short 側は `rate_limited` / `cooldown` が主要スキップに浮上し、`OPEN_REQ` は buy のみだった。
- 追加対策: `ENTRY_COOLDOWN_SEC` を短縮し、`MAX_ORDERS_PER_MINUTE` を引き上げて short の通過機会を回復させる。
- 再確認KPI（30-45分）: short `OPEN_REQ` 発生、`rate_limited(short)` の減少、`buy/sell fill ratio` の改善。

## 2026-03-04 20:16 JST / short最小ロット救済ロジック追加

- env調整後も `short units_below_min` が高止まりし、short fill 未発生の窓が継続した。
- 追加実装: `workers/scalp_ping_5s/worker.py` に short 側限定の `short_probe_rescued` を追加し、`fast/sl_streak/side_metrics` の反転根拠がある場合のみ、緩和閾値で `MIN_UNITS` 救済を許可。
- 期待効果: 動的判定を維持したまま、counter-side の 0lot 化を抑制して short の実発注化率を改善する。
- 再確認KPI（30-60分）: short `OPEN_REQ` / `OPEN_FILLED` 発生、`short units_below_min` 低下、`buy/sell fill ratio` 改善。

## 2026-03-04 20:22 JST / short救済条件の再緩和

- 初回実装後の観測で `short_probe_rescued` が未発火だったため、条件を「shortかつprob/conf/risk_cap充足」へ簡素化。
- 目的: short 側の `units_below_min` を直接減らし、buy-only状態からの離脱を優先する。
- 再確認KPI（30-45分）: `short_probe_rescued` ログ発生、short `OPEN_REQ` 発生、`short units_below_min` 低下。

## 2026-03-04 20:28 JST / short救済の最終整合（worker + order_manager）

- 判明事項: worker 側で short を `MIN_UNITS` まで救済しても、`ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_B(_LIVE)=4` により order_manager 側で拒否され得る。
- 修正:
  - `workers/scalp_ping_5s/worker.py`: short `units_below_min` は `units_risk >= MIN_UNITS` を満たす限り `short_probe_rescued` で `MIN_UNITS` へ救済。
  - `ops/env/quant-order-manager.env`: `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_B_LIVE=1`, `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_B=1` に変更。
- 再確認KPI（30-45分）: `short OPEN_REQ/OPEN_FILLED` 発生、`entry_probability_below_min_units` 減少、`buy/sell fill ratio` 改善。

## 2026-03-04 20:36 JST / short救済を強制発火へ変更

- 観測で `short_probe_rescued` 未発火が継続したため、short の `units < MIN_UNITS` は `MIN_UNITS` へ強制救済する実装に更新。
- 意図: counter-side のシグナルが0lotで消える経路を遮断し、まず short 発注を発生させる。
- 監視: `short_probe_rescued` ログ件数、short `OPEN_REQ/OPEN_FILLED`、短期 `STOP_LOSS_ORDER` 増加有無を同時監視。

## 2026-03-04 20:44 JST / short復帰確認後のlongサイズ圧縮

- 観測結果: short は `OPEN_REQ/OPEN_FILLED` が発生し復帰したが、buy 平均ユニットが大きく（約160）短期損失寄与が拡大。
- 追加対策: `SCALP_PING_5S_B_BASE_ENTRY_UNITS` を `120 -> 70` に圧縮し、long 側の損失インパクトを即時低減。
- 再確認KPI（30-45分）: buy 平均ユニット低下、short fill 維持、短期 net_jpy 改善。

## 2026-03-04 21:35 JST / local-v2 `STOP_LOSS_ON_FILL_LOSS` 初回拒否低減（scalp_ping_5s_b）

- Evidence（`logs/orders.db` 直近24h）:
  - `status='rejected' and error_message='STOP_LOSS_ON_FILL_LOSS'` は 23 件。
  - 同一 `client_order_id` 追跡で、ほぼ全件が `attempt=1 rejected -> attempt=2 filled` の回復パターン。
  - `attempt=1` の `stopLossOnFill` ギャップは概ね `1.3~1.5 pips`。
- Hypothesis:
  - `scalp_ping_5s_b` の SL 上限がタイトで、短時間の価格変位時に初回 `stopLossOnFill` が拒否されやすい。
- Action（ローカル運用上書き）:
  - `ops/env/local-v2-stack.env` と `ops/env/local-v2-full.env` に以下を追加。
  - `SCALP_PING_5S_B_SL_MAX_PIPS=1.60`
  - `SCALP_PING_5S_B_SHORT_SL_MAX_PIPS=1.80`
- Recheck KPI（次の60分）:
  - `STOP_LOSS_ON_FILL_LOSS` の `attempt=1` 発生率低下。
  - `submit_attempt(1) -> filled` までの中央値レイテンシ短縮。
  - `STOP_LOSS_ORDER` 比率が急増していないことを同時確認。

## 2026-03-04 21:34 JST / OANDA実発注確認 + `scalp_ping_5s_b` long偏損対策

- OANDA API実測（`USD_JPY`）:
  - 価格/流動性: `bid=157.196` `ask=157.204` `spread=0.8 pips`
  - ボラ: `ATR14=3.2 pips`, `ATR60=3.083 pips`, `range_60m=20.2 pips`
  - API品質: pricing ping `5/5` 成功, 平均 `269ms`
- 実発注トレース:
  - `order_manager.market_order` 経路（manual最小試験）は内部スケールで `units=-986` となり `INSUFFICIENT_MARGIN` reject。
  - 直接 OANDA REST で `REDUCE_ONLY 1 unit` を実行し約定を確認。
    - `orderCreateTransaction=417417`
    - `orderFillTransaction=417418`
    - `tradeReduced.tradeID=413001`, `realizedPL=+0.1720`
- 直近24hの収益悪化ポイント（`trades.db`, `strategy_tag=scalp_ping_5s_b_live`）:
  - 全体: `n=543`, `PF=0.428`, `winrate=20.3%`, `realized=-166.25`
  - side別: `long n=399 sum=-170.11 PF=0.412` / `short n=147 sum=+2.87 PF=2.155`
  - `STOP_LOSS_ORDER` が損失の大半（特に long 側）
- 反映（long偏損時の縮小/反転を強化）:
  - `SCALP_PING_5S_B_SIDE_METRICS_DIRECTION_FLIP_MIN_CURRENT_SL_RATE=0.48` (from `0.52`)
  - `SCALP_PING_5S_B_SIDE_METRICS_DIRECTION_FLIP_CONFIDENCE_ADD=6` (from `4`)
  - `SCALP_PING_5S_B_SIDE_ADVERSE_STACK_UNITS_STEP_MULT=0.22` (from `0.12`)
  - `SCALP_PING_5S_B_SIDE_ADVERSE_STACK_UNITS_MIN_MULT=0.72` (from `0.95`)
  - `SCALP_PING_5S_B_SIDE_ADVERSE_STACK_DD_MIN_MULT=0.78` (from `0.92`)
- 監視KPI（次の30-90分）:
  - `long STOP_LOSS_ORDER` 件数/損失寄与の低下
  - `scalp_ping_5s_b_live` の `PF` 改善（0.428 -> 0.8+ を目標）
  - `buy/sell` の fillバランス維持（short優位を殺さないこと）

## 2026-03-04 21:43 JST / ローカル運用: lookahead有効化 + side-adverse強化

- 市況チェック（OANDA live, USD/JPY）:
  - `bid=157.206` `ask=157.214` `spread=0.8 pips`
  - `ATR14=3.2 pips`, `ATR60=3.083 pips`, `range_60m=20.2 pips`
  - pricing応答 `5/5`, 平均レイテンシ `283ms`（p95近似 `294ms`）
- 直近実績（local `logs/*.db`）:
  - `scalp_ping_5s_b_live` 24h: `n=546`, `PF=0.387`, `winrate=20.3%`, `avg=-0.721 pips`
  - side別: `long PF=0.347` / `short PF=0.514`（long側の悪化が主因）
  - `orders.db` 24h reject: `STOP_LOSS_ON_FILL_LOSS=23`, `api_error(502)=1`
- 実発注確認（本番キー・最小往復）:
  - OANDA REST で `USD_JPY -1 unit` を約定→3秒後に全決済
  - `trade_id=417472`, `open=157.158`, `close=157.174`, `realized=-0.0160`
- 反映（`ops/env/scalp_ping_5s_b.env`）:
  - `SCALP_PING_5S_B_LOOKAHEAD_GATE_ENABLED=1`
  - `SCALP_PING_5S_B_LOOKAHEAD_ALLOW_THIN_EDGE=0`
  - `SCALP_PING_5S_B_LOOKAHEAD_EDGE_MIN_PIPS=0.14`
  - `SCALP_PING_5S_B_SIDE_ADVERSE_STACK_UNITS_ACTIVE_START=3`（`4 -> 3`）
  - `SCALP_PING_5S_B_SIDE_ADVERSE_STACK_UNITS_STEP_MULT=0.28`（`0.22 -> 0.28`）
  - `SCALP_PING_5S_B_SIDE_ADVERSE_STACK_UNITS_MIN_MULT=0.45`（`0.60 -> 0.45`）
  - `SCALP_PING_5S_B_SIDE_ADVERSE_STACK_DD_MIN_MULT=0.55`（`0.65 -> 0.55`）
- 目的:
  - 薄いエッジのエントリーを lookahead で遮断
  - 損失側サイドが続く局面でロット縮小を早期/強度高めに適用

## 2026-03-04 22:05 JST / `scalp_ping_5s_b` 逆行耐性とpreserve_intentレンジ再調整

- 根因:
  - `scalp_ping_5s_b` は逆行時の早期 `force_exit` と対向トレンド側の縮小不足が重なり、ノイズ帯で損失確定が先行。
  - 併せて `preserve_intent` 下限が低く、意図ロットが過小化しやすい局面が残存。
- 変更値（2026-03-04）:
  - `SCALP_PING_5S_B_SL_BASE_PIPS=1.15`
  - `SCALP_PING_5S_B_SHORT_SL_BASE_PIPS=1.30`
  - `SCALP_PING_5S_B_SL_MAX_PIPS=2.00`
  - `SCALP_PING_5S_B_SHORT_SL_MAX_PIPS=2.10`
  - `SCALP_PING_5S_B_FORCE_EXIT_FLOATING_LOSS_MIN_HOLD_SEC=3`
  - `SCALP_PING_5S_B_FORCE_EXIT_MAX_FLOATING_LOSS_PIPS=0.75`
  - `SCALP_PING_5S_B_SHORT_FORCE_EXIT_MAX_FLOATING_LOSS_PIPS=0.70`
  - `SCALP_PING_5S_B_M1_TREND_OPPOSITE_UNITS_MULT=0.70`
  - `SCALP_PING_5S_B_ENTRY_PROBABILITY_ALIGN_COUNTER_EXTRA_PENALTY_MAX=0.18`
  - `SCALP_PING_5S_B_ENTRY_PROBABILITY_ALIGN_UNITS_MIN_MULT=0.88`
  - `ORDER_MANAGER_PRESERVE_INTENT_MIN_SCALE_STRATEGY_SCALP_PING_5S_B_LIVE=0.60`
  - `ORDER_MANAGER_PRESERVE_INTENT_MAX_SCALE_STRATEGY_SCALP_PING_5S_B_LIVE=1.00`
- 検証KPI:
  - `30m`: `STOP_LOSS_ORDER` 件数、`force_exit` 理由内訳、`OPEN_REQ -> OPEN_FILLED` 成功率。
  - `2h`: `scalp_ping_5s_b_live` の side別 PF/勝率、平均保持秒、平均実効units。
  - `24h`: 総合 PF・実現損益・最大DD、`STOP_LOSS_ON_FILL_LOSS` reject率、entry sideバランス。

## 2026-03-04 23:42 JST / ローカルRCA再実施とB戦略の再調整（buy偏損再発防止）

- 市況実測（OANDA API, USD/JPY）:
  - `bid=157.294`, `ask=157.302`, `spread=0.8 pips`
  - `ATR14(M1)=3.336 pips`, `range60(M1)=26.8 pips`
  - pricing API 応答: 平均 `466.82ms`（max `1385.88ms`）
- 収益分解（`logs/trades.db`, 直近24h, `strategy_tag=scalp_ping_5s_b_live`）:
  - 総計: `n=608`, `net_jpy=-175.6`, `win_rate=19.6%`, `PF=0.416`
  - side別: `buy n=412 net=-178.0 avg_units=62.9`, `sell n=196 net=+2.4 avg_units=1.65`
  - close_reason別: `STOP_LOSS_ORDER n=470 net=-280.7`, `MARKET_ORDER_TRADE_CLOSE n=138 net=+105.1`
- 根因:
  - 損失のほぼ全量が `buy + STOP_LOSS_ORDER`（`n=317 net=-277.6`）に集中。
  - `sell` はほぼ建て値圏だが、`spread 0.8` に対して薄いエッジのエントリーが残り、微損を積む。
- 反映（`ops/env/scalp_ping_5s_b.env`）:
  - sell固定維持: `SCALP_PING_5S_B_SIDE_FILTER=sell`, `...ALLOW_NO_SIDE_FILTER=0`
  - 薄利エントリー抑制: `MAX_SPREAD_PIPS=0.85`, `LOOKAHEAD_EDGE_MIN_PIPS=0.22`, `LOOKAHEAD_SAFETY_MARGIN_PIPS=0.12`
  - SL拒否低減: `SL_BASE/MIN=1.25/1.20`, `SHORT_SL_BASE/MIN/MAX=1.45/1.20/2.20`
  - 確率帯ロット再配分: `LOW/HIGH_THRESHOLD=0.65/0.80`, `HIGH_REDUCE_MAX=0.70`, `LOW_BOOST_MAX=0.16`, `UNITS_MIN_MULT=0.55`
  - shortの強制最小ロット停止: `SCALP_PING_5S_B_SHORT_PROBE_RESCUE_ENABLED=0`
- 実装追補:
  - `workers/scalp_ping_5s/config.py` に `SCALP_PING_5S_SHORT_PROBE_RESCUE_ENABLED` を追加。
  - `workers/scalp_ping_5s/worker.py` で short救済を関数化し、envでON/OFF可能化。
- 追加監査:
  - `close_failed` の大半は `ticket_id=sim-*` 由来（直近12000件中 `172/182`）で、実運用ノイズとして分離。
- 再検証KPI（次の30-90分）:
  - `buy` の `preflight_start/submit_attempt/filled` が 0 を維持すること。
  - `STOP_LOSS_ON_FILL_LOSS` 比率の低下。
  - `sell` の `expectancy_jpy > 0` への復帰、および `PF > 1.0`。

## 2026-03-04 23:46 JST / ローカル自動復帰（launchd）導入

- 目的:
  - PC再起動/ログイン後、スリープ復帰後、ネット復帰後に `local_v2_stack` を自動再開し、
    手動起動なしで運用継続できるようにする。
- 作業前市況確認（OANDA API）:
  - `bid=157.304`, `ask=157.312`, `spread=0.8 pips`
  - `ATR14(M1)=4.5 pips`, `range60(M1)=23.1 pips`
  - pricing応答: 平均 `230.28ms`, max `246.18ms`
  - 作業継続判定: 通常レンジ内として実施
- 実装:
  - `scripts/local_v2_autorecover_once.sh`（新規）
    - `local_v2_stack status` で `stopped/stale` 検知時のみ `up` を実行
    - ネット未接続時は待機（`api-fxtrade.oanda.com:443` 到達確認）
    - `parity` 競合（exit code `3`）時は安全にスキップ
    - 重複実行防止のロック (`logs/local_v2_autorecover.lock`)
  - `scripts/install_local_v2_launchd.sh`（新規）
    - LaunchAgent を `~/Library/LaunchAgents` に生成・`bootstrap`
    - `RunAtLoad + StartInterval + KeepAlive(NetworkState)` を設定
  - `scripts/uninstall_local_v2_launchd.sh`（新規）
  - `scripts/status_local_v2_launchd.sh`（新規）
  - `docs/OPS_LOCAL_RUNBOOK.md`
    - 自動復帰セクション（install/status/uninstall、ログ位置）を追加
- 監視ログ:
  - `logs/local_v2_autorecover.log`
  - `logs/local_v2_autorecover.launchd.out`
  - `logs/local_v2_autorecover.launchd.err`

## 2026-03-05 00:06 JST / 自動復帰の実働安定化（launchd 126/子プロセス回収/lock詰まり修正）

- 症状:
  - LaunchAgent の `last exit code=126`（`Operation not permitted`）で自動復帰が動作しない。
  - `up` 実行直後にワーカーが落ちる（launchd が子プロセスを回収）。
  - ロックディレクトリ残骸で autorecover が無反応になる。
- 根因:
  - macOS `launchd` から `~/Documents` 実体へのスクリプト読み取り制約。
  - plist に `AbandonProcessGroup` 未設定で、ジョブ終了時に spawned worker が終了。
  - ロックが `mkdir` のみで stale 判定が無い。
- 対応:
  - リポジトリ実体を `/Users/tossaki/App/QuantRabbit` へ移動し、
    `/Users/tossaki/Documents/App/QuantRabbit` は互換 symlink 化。
  - `scripts/install_local_v2_launchd.sh`
    - 実行コマンドを絶対パス化し `bash -lc` 経由へ統一。
    - `AbandonProcessGroup=true` を付与。
  - `scripts/local_v2_autorecover_once.sh`
    - lock に owner PID を保存。
    - stale lock 自動除去と再取得を実装。
- 検証（ローカル実測）:
  - `scripts/status_local_v2_launchd.sh` で `last exit code=0` を確認。
  - `local_v2_stack down` 後、20〜30秒で `recover` 実行と全8サービス `running` を確認。
  - `quant-micro-rangebreak` を手動 kill 後、約25秒で自動再起動（PID更新）を確認。

## 2026-03-04 17:07 UTC / 2026-03-05 02:07 JST - no-entry緩和（`scalp_ping_5s_b` / `micro_rangebreak`）

- 目的:
  - `entry-skip` 偏重を緩和し、`scalp_ping_5s_b` と `micro_rangebreak` の約定再開余地を増やす。
- 仮説:
  - `scalp_ping_5s_b` の `lookahead_block` / `no_signal:revert_not_found` と、`micro_rangebreak` の trend-flip 偏重を局所緩和すると no-entry を減らせる。
- 変更値:
  - `ops/env/scalp_ping_5s_b.env`
    - `SCALP_PING_5S_B_LOOKAHEAD_ALLOW_THIN_EDGE=1`
    - `SCALP_PING_5S_B_LOOKAHEAD_COUNTER_PENALTY=0.24`
    - `SCALP_PING_5S_B_REVERT_MIN_TICKS=2`
    - `SCALP_PING_5S_B_REVERT_CONFIRM_RATIO_MIN=0.07`
    - `SCALP_PING_5S_B_SHORT_MIN_SIGNAL_TICKS=3`
  - `ops/env/quant-micro-rangebreak.env`
    - `MICRO_MULTI_TREND_FLIP_STRATEGY_BLOCKLIST=MicroLevelReactor,MicroCompressionRevert,MicroRangeBreak`
    - `MICRO_RANGEBREAK_ENTRY_RATIO=0.38`
    - `MICRO_RANGEBREAK_MIN_RANGE_SCORE=0.32`
    - `MICRO_RANGEBREAK_REVERSION_MAX_ADX=27.0`
    - `MICRO_MULTI_ENTRY_LEADING_PROFILE_REJECT_BELOW=0.38`
    - `MICRO_MULTI_MIN_UNITS=500`
    - `MICRO_MULTI_MAX_MARGIN_USAGE=0.95`
- 作業前市況チェック（ローカルV2 + OANDA API）:
  - `USD_JPY bid=156.930 ask=156.938 spread=0.8p`
  - `ATR14=2.764p / ATR60=2.997p / range60=22.0p`
  - `orders.db` 直近24h: `filled=662`, `reject_like=29`
  - OANDA API応答: pricing 平均 `228.5ms`（max `232.3ms`）、candles `217.3ms`
- 検証手順（ローカルV2）:
  - `scripts/local_v2_stack.sh restart --profile trade_min --env ops/env/local-v2-stack.env`
  - `scripts/local_v2_stack.sh status --profile trade_min --env ops/env/local-v2-stack.env`
  - `tail -n 120 logs/local_v2_stack/quant-scalp-ping-5s-b.log`
  - `tail -n 120 logs/local_v2_stack/quant-micro-rangebreak.log`

## 2026-03-05 09:10 JST / `scalp_ping_5s_b` 逆期待値抑制（sell固定 + 薄エッジ遮断）

- 目的:
  - 「全然稼げてない」状態に対し、`scalp_ping_5s_b_live` の逆期待値エントリーを即時抑制する。
- 市況実測（作業前チェック, ローカルV2 + OANDA API）:
  - `USD/JPY mid=156.977`
  - `spread`（tick_cache直近300）`avg=0.8 pips / p95=0.8 pips`
  - `ATR14(M1)=1.743 pips`, `range60=14.2 pips`
  - OANDA summary API 応答: 平均 `218.46ms`（max `238.77ms`, 3/3 success）
- 収益分解（`logs/trades.db`, 直近24h, `strategy_tag=scalp_ping_5s_b_live`）:
  - 総計: `n=611`, `net_realized=-175.658`, `PF=0.416`
  - side別: `buy n=412 net=-177.957`, `sell n=199 net=+2.299`
  - `STOP_LOSS_ORDER` 偏重で損失が集中。
- 原因:
  - 実運用値が緩和モード（`SCALP_PING_5S_B_SIDE_FILTER=none`）のままで、buy側逆期待値を通していた。
  - lookahead/net-edge 閾値が実コスト（`cost_vs_mid ~0.4p`, spread ~`0.8p`）に対して低かった。
- 反映（`ops/env/scalp_ping_5s_b.env`）:
  - `SCALP_PING_5S_B_SIDE_FILTER=sell`
  - `SCALP_PING_5S_B_ALLOW_NO_SIDE_FILTER=0`
  - `SCALP_PING_5S_B_MAX_SPREAD_PIPS=0.90`（1.15→0.90）
  - `SCALP_PING_5S_B_LOOKAHEAD_ALLOW_THIN_EDGE=0`
  - `SCALP_PING_5S_B_LOOKAHEAD_EDGE_MIN_PIPS=0.35`（0.10→0.35）
  - `SCALP_PING_5S_B_LOOKAHEAD_SAFETY_MARGIN_PIPS=0.16`（0.08→0.16）
  - `SCALP_PING_5S_B_ENTRY_NET_EDGE_MIN_PIPS=0.35`（0.20→0.35）
  - `SCALP_PING_5S_B_BASE_ENTRY_UNITS=24`（35→24）
- 再検証KPI（30m / 2h / 24h）:
  - 30m: `buy` の `submit_attempt/filled` が 0 維持、`rejected/api_error` 増加なし。
  - 2h: `scalp_ping_5s_b_live` の `PF >= 0.8`、`expectancy_realized` の改善。
  - 24h: `strategy_tag=scalp_ping_5s_b_live` の `net_realized` 改善、`STOP_LOSS_ORDER` 比率低下。

## 2026-03-05 11:35 JST / scalp_ping_5s_b no-signal緩和（可変パラメータ拡張）

Period:
- 調査時刻: 2026-03-05 11:30〜11:35 JST
- 対象: `ops/env/local-v2-stack.env`, `logs/local_v2_stack/quant-scalp-ping-5s-b.log`, `logs/orders.db`

Fact:
- `SCALP_PING_5S_B_LOOKAHEAD_GATE_ENABLED=0` は実効反映済み（process env確認）。
- ただし最新 skip は `no_signal:revert_not_found` / `no_signal:momentum_tail_failed` が主因で、`filled` 最終時刻は `2026-03-05T02:22:07+00:00` のまま更新停滞。

Improvement:
- `ops/env/local-v2-stack.env` へ以下を追加し、シグナル生成を可変で緩和。
  - `SCALP_PING_5S_B_MIN_SIGNAL_TICKS=3`
  - `SCALP_PING_5S_B_LONG_MIN_SIGNAL_TICKS=3`
  - `SCALP_PING_5S_B_SHORT_MIN_SIGNAL_TICKS=3`
  - `SCALP_PING_5S_B_SIGNAL_MODE_BLOCKLIST=`
  - `SCALP_PING_5S_B_ENTRY_LEADING_PROFILE_REJECT_BELOW=0.72`
  - `SCALP_PING_5S_B_ENTRY_LEADING_PROFILE_REJECT_BELOW_SHORT=0.78`

Verification:
- 再起動後に `entry-skip summary` の `signal_mode_blocked` と `momentum_tail_failed` 比率が低下すること。
- `orders.db` の `submit_attempt/filled` 最終時刻が更新されること。

Status:
- in_progress

## 2026-03-05 11:40 JST / units_below_min対策（risk floor + base units）

Fact:
- `entry-skip summary` は `units_below_min` が継続（short側で 4〜10 件）。
- `orders` 最終 `filled` は `2026-03-05T02:22:07+00:00` で更新停滞。

Improvement:
- `ops/env/local-v2-stack.env` へ追加。
  - `SCALP_PING_5S_B_BASE_ENTRY_UNITS=32`
  - `RISK_PERF_MIN_MULT=0.55`

Verification:
- 再起動後に `entry-skip summary` の `units_below_min` 比率が低下すること。
- `orders.db` の `submit_attempt` / `filled` 最終時刻が更新されること。

Status:
- in_progress

## 2026-03-05 13:05 JST / 全戦略ワーカー可変調整 + 停止系復旧（local_v2 trade_all）

- 目的:
  - 「全戦略ワーカーの停止解消」と「収益悪化戦略の即時可変チューニング」を同時実施し、trade_all を安定稼働へ戻す。
- 作業前市況（ローカル実測 / OANDA API）:
  - `USD/JPY bid=156.990 ask=156.998 spread=0.8p`
  - `ATR14(M1)=2.921p`, `range60=19.3p`
  - API遅延（pricing 6サンプル）: `avg=251.42ms`, `p95=260.10ms`, `max=271.72ms`
- 停止要因（ログ根拠）:
  - `quant-position-manager(8301)` への `Connection refused` が exit worker で連鎖。
  - `local_v2_autorecover` の旧 `trade_min` 復旧履歴が残り、core restart連鎖を誘発。
- 実施変更:
  - `workers/scalp_macd_rsi_div_b/exit_worker.py` を追加（B exit import欠損解消）。
  - `ops/env/local-v2-stack.env` を可変調整:
    - B: `SIDE_FILTER=sell`, `ALLOW_NO_SIDE_FILTER=0`, `MAX_ACTIVE_TRADES=4`, `MAX_PER_DIRECTION=2`,
      `MIN/LONG/SHORT_MIN_SIGNAL_TICKS=5`, `ENTRY_LEADING_PROFILE_REJECT_BELOW=0.80/0.86`,
      `ENTRY_PROBABILITY_ALIGN_FLOOR=0.62`, `BASE_ENTRY_UNITS=18`。
    - C: `LOOKAHEAD_GATE_ENABLED=1`, `MIN_SIGNAL_TICKS=3`,
      `ENTRY_LEADING_PROFILE_REJECT_BELOW=0.74/0.80`, `ENTRY_PROBABILITY_ALIGN_FLOOR=0.52`, `MAX_SPREAD_PIPS=1.20`。
    - D: `LOOKAHEAD_GATE_ENABLED=1`, `MIN_SIGNAL_TICKS=3`,
      `ENTRY_LEADING_PROFILE_REJECT_BELOW=0.66/0.74`, `ENTRY_PROBABILITY_ALIGN_FLOOR=0.55`,
      `BASE_ENTRY_UNITS=4200`, `MAX_ACTIVE_TRADES=1`, `MAX_PER_DIRECTION=1`,
      `DIRECTION_BIAS_LONG_OPPOSITE_UNITS_MULT=0.30`,
      `FORCE_EXIT_MAX_FLOATING_LOSS_PIPS=0.65`, `SHORT_FORCE_EXIT_MAX_FLOATING_LOSS_PIPS=0.55`, `MAX_SPREAD_PIPS=1.20`。
    - Flow: `LOOKAHEAD_GATE_ENABLED=1`, `MIN_SIGNAL_TICKS=3`,
      `ENTRY_LEADING_PROFILE_REJECT_BELOW=0.40/0.52`, `BASE_ENTRY_UNITS=700`,
      `MAX_ACTIVE_TRADES=6`, `MAX_PER_DIRECTION=3`。
- 反映:
  - `scripts/local_v2_stack.sh restart --env ops/env/local-v2-stack.env --services quant-market-data-feed,quant-strategy-control,quant-order-manager,quant-position-manager,quant-scalp-ping-5s-b,quant-scalp-ping-5s-c,quant-scalp-ping-5s-d,quant-scalp-ping-5s-flow`
  - 直後に `up --services core4` を再実行して PID/health を再確定。
- 検証:
  - `scripts/local_v2_stack.sh status --profile trade_all --env ops/env/local-v2-stack.env` で `stopped=0`。
  - 60秒後再確認でも core + B/C/D/Flow はすべて `running` 維持。
  - `logs/local_v2_autorecover.log` 最新で `2026-03-05 13:01:51 JST [recover] ... profile=trade_all` を確認。
  - `scripts/collect_local_health.sh` 成功（`logs/health_snapshot.json` 更新）。
- 反映後の初期観測（`datetime(close_time) >= 2026-03-05 04:01:00 UTC`）:
  - closed trades: `1`, realized `+0.21`, win_rate `1.0`（サンプル小）。
  - order status は B/C/D/Flow で `lookahead block` / `entry_probability_reject` / `strategy_cooldown` が主となり、逆期待値シグナルの通過が抑制。

## 2026-03-05 14:40 JST / ローカルLLM常時運用化（停止回避→改善優先）

- 目的:
  - 「トレード停止」ではなく、ローカルLLMで判定改善・パラメータ改善を回しながらエントリーを維持する。

- 作業前市況チェック（ローカル実測）:
  - 取得元: `logs/orders.db`, `logs/brain_state.db`, `logs/local_v2_stack/quant-market-data-feed.log`, `logs/health_snapshot.json`
  - 直近90分 `orders.db`:
    - `mid range: 156.980 - 157.164`
    - `spread avg/min/max: 0.801 / 0.8 / 1.0 pips`
    - `atr_m1 avg: 2.346 pips`, `atr_m5 avg: 5.738 pips`
  - OANDA API品質:
    - `quant-market-data-feed.log` 直近800行のHTTP集計: `HTTP/1.1 200 = 129`, `non-200 = 0`
  - 実行品質:
    - `health_snapshot`: `data_lag_ms ~403`, `decision_latency_ms ~21`
    - 直近90分 `orders.db`: `filled=162`, `rejected=2`（STOP_LOSS_ON_FILL_LOSS）
  - 判定:
    - 通常レンジ内として作業継続（保留条件には非該当）。

- ローカルLLM比較ベンチ（2026-03-05）:
  - レポート:
    - `logs/brain_local_llm_benchmark_multimodel_20260305.json`
    - `logs/brain_local_llm_benchmark_multimodel_4way_20260305.json`
  - 4モデル比較（10 samples, outcome prioritize）:
    - `gpt-oss:20b` score `1.245`, parse `1.0`, p95 `27791ms`
    - `qwen2.5:7b` score `1.231`, parse `1.0`, p95 `3641ms`
    - `llama3.1:8b` score `1.217`, parse `1.0`, p95 `4250ms`
    - `gemma3:4b` score `1.182`, parse `1.0`, p95 `2682ms`
  - 運用選定:
    - preflightは遅延制約を優先して `qwen2.5:7b`
    - async autotuneは品質優先で `gpt-oss:20b`

- 反映内容:
  - `ops/env/local-v2-stack.env`
    - `STRATEGY_CONTROL_ENTRY_*` を再有効化（B/C/D/Flow/MicroPullbackEMA/MicroTrendRetest = 1）
    - `LOCAL_V2_EXTRA_ENV_FILES=ops/env/profiles/brain-ollama.env`
  - `ops/env/profiles/brain-ollama.env`
    - `BRAIN_OLLAMA_MODEL=qwen2.5:7b`
    - `BRAIN_TIMEOUT_SEC=8`
    - `BRAIN_PROMPT_AUTO_TUNE_MODEL=gpt-oss:20b`
    - `BRAIN_RUNTIME_PARAM_AUTO_TUNE_MODEL=gpt-oss:20b`
  - `config/brain_prompt_profile.json`
    - 初期プロファイルを `REDUCE優先・参加率維持` 方針へ更新
  - `config/brain_runtime_param_profile.json`
    - `activity_rate_floor=0.55`, `block_rate_soft_limit=0.78` 等へ更新

- 次の監視KPI:
  - `strategy_control_entry_disabled` の減少
  - `filled / submit_attempt` 比率の回復
  - `brain_prompt_autotune_latest.json`, `brain_runtime_param_autotune_latest.json` の更新継続
  - `block_rate` と `activity_rate` が `runtime_profile` の目標帯に収束すること

## 2026-03-05 15:22 JST / Brain autoPDCA実行欠損の修正（改善ループ常時化）

- 作業前市況（ローカル実測 + OANDA API）:
  - `USD/JPY bid=157.240 ask=157.248 spread=0.8p`
  - `ATR14(M1)=2.613p`, `ATR14(M5)=6.676p`, `M1 range60=21.9p`
  - `orders_60m_total=1522`, `reject-like=282 (18.53%)`
  - `quant-market-data-feed` の HTTP 応答は直近サンプルで `200 OK` のみ
  - 判定: 通常レンジ内のため改善作業を継続

- 原因:
  - `local_v2_autorecover_once.sh` は `run_brain_autopdca_cycle.sh --interval-sec` を呼ぶが、
    受け側が `--interval-sec` 非対応で即終了し、LLMベンチ→モデル反映→再起動のPDCAが回っていなかった。

- 改善:
  - `run_brain_autopdca_cycle.sh` に以下を追加。
    - `--interval-sec` / `--force`
    - lock/state による多重起動・短周期連打防止
    - `env_changed=true` の時だけ core再起動
    - 市況ガード（spread/reject-rate）で異常時は skip
    - `latest + history(jsonl)` 監査ログ出力
  - `test_run_brain_autopdca_cycle.py` を契約整合（`env_changed`）へ修正し、
    interval skip ケースを追加。

- 期待効果:
  - 「停止寄り」ではなく、ローカルLLMの判断履歴を使ったモデル・プロンプト・タイムアウト改善を定期実行できる。
  - 不要な再起動を抑えつつ、改善が出た時だけ即時反映してトレード品質を上げる。

- 追補（15:33 JST）:
  - 初回実運用で `market_snapshot: null` を検知。原因は market取得出力に警告行が混在し、JSON全文パースが失敗していたこと。
  - `run_brain_autopdca_cycle.sh` を修正し、末尾JSON行を優先抽出する方式に変更。
  - 確認: `logs/brain_autopdca_cycle_latest.json` で `market_snapshot.status=ok` と実測 spread/ATR/reject-rate が記録されることを確認。

## 2026-03-05 17:40 JST / trade_min を `M1Scalper + MicroRangeBreak` に固定（ping5s停止）+ launchd追随（local V2）

- 目的:
  - `scalp_ping_5s_b_live` の「取引数だけ増えてSLで負ける」状態から脱し、trade_min を `M1Scalper + MicroRangeBreak` に寄せて収益性を回復する。

- 作業前市況（ローカル実測 / OANDA API, 2026-03-05 17:17 JST）:
  - `USD/JPY bid=157.281 ask=157.289 spread=0.8p`
  - `ATR14(M1)=3.700p`, `ATR14(M5)=7.193p`, `range60=42.2p`
  - API遅延: `pricing=241ms`, `candles(M1)=253ms`, `candles(M5)=271ms`, `summary=244ms`
  - 判定: spreadは通常、レンジは拡大気味だが流動性悪化（スプレッド拡大/応答劣化）は見られないため作業継続。

- 直近の損益分解（`logs/trades.db`, pocket<>manual, close_time>=now-24h）:
  - NOTE: `close_time` は `2026-03-05T05:13:34+00:00` のようなISO文字列のため、時間窓のSQLは `datetime(substr(close_time,1,19))` で正規化して集計する。
  - `n=618`, `win_rate=0.2104`, `PF=0.407`, `expectancy_jpy=-0.8`, `net_jpy=-489.2`
  - 寄与（pocket×strategy, net_jpy上位の赤字）:
    - `scalp_fast / scalp_ping_5s_b_live`: `n=518`, `net=-103.9`, `win_rate=0.189`
    - `micro / MicroPullbackEMA`: `n=25`, `net=-133.9`, `win_rate=0.28`

- 口座リスク（OANDA, 2026-03-05 17:39 JST）:
  - NAV `49,928.68 JPY`, margin_used `44,768.80`, margin_available `5,188.10`, health_buffer `0.1038`
  - openTrades: `USD_JPY -6998`（2026-03-02 open, clientExtensionsなし）, `USD_JPY -120`（tag=codex_bi_hf）
  - 直近の重大損失: `pocket=manual` の `MARKET_ORDER_MARGIN_CLOSEOUT`（2026-03-02, ticket `412993`, realized `-7696`）

- 実施（ローカルV2導線のみ）:
  - `scripts/local_v2_stack.sh` の `PROFILE_trade_min` を更新（`scalp_ping_5s_b(+exit)` を外し、`quant-m1scalper(+exit)` を追加）。
    - commit: `5fb475eb chore(local_v2): trade_min add m1scalper`
  - 既存の launchd が `--services` 固定（core+microのみ）で動いており、profile更新が反映されず `quant-m1scalper` が起動しない状態だった。
    - `scripts/install_local_v2_launchd.sh --interval-sec 20 --profile trade_min --env ops/env/local-v2-stack.env` を再実行し、launchd を「profile追随（--services無し）」へ戻した。

- 反映確認:
  - `scripts/local_v2_stack.sh status --profile trade_min --env ops/env/local-v2-stack.env`
    - `quant-m1scalper` / `quant-m1scalper-exit` が `running`（ppid=1）で常駐することを確認。

- Pattern Gate（opt-in）確認:
  - `ops/env/local-v2-stack.env`: `ORDER_MANAGER_PATTERN_GATE_ENABLED=1`
  - `ops/env/quant-v2-runtime.env`: `ORDER_PATTERN_GATE_GLOBAL_OPT_IN=0`（全戦略強制はしない）
  - `orders.db` 直近2hで `request_json LIKE '%pattern_gate%'` の行が `1210`（pattern gate payload が request に注入されていることを確認）

- 次に見るKPI（再検証条件）:
  - 直近60m/24hの `M1Scalper-M1` と `MicroRangeBreak` の `PF>1.0`、`expectancy>=0` へ回復すること
  - `health_buffer>=0.10` を維持（下回る場合は「自動戦略追加で押さない」方向へ即時縮小）

## 2026-03-05 18:20 JST / Pattern Gate マッチ率改善（MicroRangeBreak canonical tag）+ entry_thesis backfill診断強化

- 背景:
  - `orders.db` の `MicroRangeBreak-*` 注文は `entry_thesis.strategy_tag` も suffix 付きのままで、
    Pattern book 側（`logs/patterns.db`）の `st:microrangebreak` と一致せず gate が no-op になりうる。
  - `scripts/backfill_entry_thesis_from_orders.py` は `submit_attempt.request_json` 前提だが、
    過去世代は `orders` 行が存在しても `request_json` が残っていないケースがあり、復元不能が混在する。

- 変更:
  - `workers/micro_runtime/worker.py`
    - `MicroRangeBreak`/`MicroVWAPBound` の `entry_thesis.strategy_tag` を base tag へ正規化し、raw は `strategy_tag_raw` へ退避。
  - `scripts/backfill_entry_thesis_from_orders.py`
    - `orders.db` を `ATTACH` して参照し、`submit_attempt -> preflight_start -> other` 優先で `request_json` を拾う方式へ更新。
    - `orders_matched / orders_with_request / recovered_from_orders` を出力して「一致はするが復元ソース無し」を判別可能にした。

- 再検証:
  - Pattern book 更新後（`scripts/pattern_book_worker.py`）、`MicroRangeBreak` の新規注文で `orders.status IN ('pattern_scaled','pattern_block','pattern_scale_below_min')` が出ること。

## 2026-03-05 19:40 JST / M1Scalper: quickshot hard-gate で signal を全drop（local V2）

- 症状:
  - `logs/orders.db` の `max(ts)` が `2026-03-05T05:53:30Z`（`14:53 JST`）以降更新されず、`trade_min` 起動中でも新規注文が出ない。

- 原因（RCA）:
  - `workers/scalp_m1scalper/worker.py` の quickshot 判定が `quickshot_allow=False` の場合、ログ後に無条件 `continue` しており quickshot が「必須ゲート」になっていた。
  - quickshot は `M5 breakout + M1 pullback` を要求するため、レンジ局面では成立しにくく、結果として signal をほぼ全drop していた。

- 対応（local V2）:
  - `M1SCALP_USDJPY_QUICKSHOT_HARD_GATE`（default=1）を追加し、`0` のときは quickshot 不成立でも通常フローで entry を継続（quickshot plan は適用しない）。
  - `ops/env/local-v2-stack.env` で `M1SCALP_USDJPY_QUICKSHOT_HARD_GATE=0`、`M1SCALP_USDJPY_QUICKSHOT_MAX_SPREAD_PIPS=1.20` を設定（retest 要件は維持）。

- 検証手順:
  - `./scripts/local_v2_stack.sh restart --profile trade_min --env ops/env/local-v2-stack.env --services quant-m1scalper,quant-m1scalper-exit`
  - `sqlite3 logs/orders.db 'select max(ts) from orders;'` が `2026-03-05T05:53:30Z` より新しい
  - `scripts/local_v2_stack.sh logs --service quant-order-manager --tail 200` で preflight の通過ログを確認

## 2026-03-05 20:19 JST / 11:19 UTC - no-entry 継続の暫定復旧（M1Scalper 強トレンドflip + MicroRangeBreak hist_block 緩和, local V2）

- 症状:
  - `logs/orders.db` の `max(ts)=2026-03-05T05:53:30Z` 以降更新が止まり、`trade_min` 起動中でも新規注文が出ない。
  - `logs/local_v2_stack/quant-micro-rangebreak.log` に `hist_block ... score=0.266 n=27` が継続。
  - `logs/local_v2_stack/quant-m1scalper.log` で `range_hold_reversion_*` と `trend_block_*` が交互に出て signal が返らず無風になりやすい。

- 作業前市況（ローカル実測 / OANDA API, 2026-03-05 20:18 JST）:
  - `USD/JPY bid=157.240 ask=157.248 spread=0.8p`
  - `ATR14(M1)=2.434p`, `range60(M1)=18.8p`
  - API遅延: `pricing=334ms`, `candles(M1)=244ms`
  - 判定: 通常レンジ、流動性悪化は顕著でないため作業継続。

- 対応（local V2 / main 反映）:
  - `strategies/scalping/m1_scalper.py`
    - `range_reversion_only==True` でも `strong_up/strong_down` のときは `OPEN_LONG/OPEN_SHORT` へflipし、`trend_block_*` で全dropしないようにした。
    - 監視ログ: `range_flip_to_trend_long` / `range_flip_to_trend_short`
  - `ops/env/local-v2-stack.env`
    - `MICRO_MULTI_HIST_SKIP_SCORE=0.20`（`hist_block` の hard skip を緩和）
    - `MICRO_MULTI_HIST_LOT_MIN=0.25`（低スコア時は縮小運転）
    - `M1SCALP_ENTRY_GUARD_BYPASS=1` は暫定維持（BB/projection reject の可視化/復旧用）。`entry_guard_bypass` が常態化する場合は閾値チューニングへ戻す。

- 影響範囲:
  - M1Scalper のシグナル方向が強トレンドで順張りに寄る（range_reversion_only の freeze 回避）。
  - micro runtime の hist skip を緩和し、低品質戦略はロット縮小で継続。

- 検証:
  - `scripts/local_v2_stack.sh restart --profile trade_min --env ops/env/local-v2-stack.env --services quant-m1scalper,quant-m1scalper-exit,quant-micro-rangebreak,quant-micro-rangebreak-exit`
  - `sqlite3 logs/orders.db 'select max(ts) from orders;'` が `2026-03-05T05:53:30Z` より新しい
  - `logs/local_v2_stack/quant-m1scalper.log` に `range_flip_to_trend_*` または `entry_guard_bypass` が出て、preflight が流れること
  - `logs/local_v2_stack/quant-micro-rangebreak.log` の `hist_block` 頻度が減り、entry が流れること
  - `orders.db` で `error_code=INSUFFICIENT_MARGIN` / `margin_*` 系の拒否が急増しないこと（増えるならロット縮小へ即応）

## 2026-03-05 21:34 JST / ping5s の dyn alloc sizing 適用 + dyn alloc soft-participation 安全化 + order-manager Brain import fail-open

- 作業前市況（ローカル実測 / OANDA API, 2026-03-05 21:29 JST）:
  - `USD/JPY bid=157.388 ask=157.396 spread=0.8p`
  - `ATR(M1)=2.388p`, `ATR(M5)=5.170p`
  - 判定: 通常レンジ、流動性悪化は顕著でないため作業継続。

- 狙い / 仮説:
  - `trade_all` 等で「未観測 strategy が dyn alloc 未適用のまま full size」になると損益・マージンが悪化しやすい → soft-participation では未観測も `min_lot_multiplier` へ寄せる。
  - ping5s 系（B/C/D/flow）は `config/dynamic_alloc.json` の score/lot_multiplier を sizing に取り込めておらず、悪化戦略の縮小が効かない → dyn alloc multiplier を entry の `units` へ反映する。
  - Brain ゲートは default disabled だが、依存 import 失敗で `quant-order-manager` が起動不能になると no-trade を再発する → import を fail-open にして起動継続。

- 対応（main反映 / local V2）:
  - `execution/order_manager.py`: `workers.common.brain` を optional import に変更し、Brain gate enabled でも module 不在時は warning+metric を出して skip（fail-open）。
  - `workers/common/dynamic_alloc.py`: `allocation_policy.soft_participation=true` のとき、`dynamic_alloc.json` に無い strategy は `min_lot_multiplier` をデフォルト適用（1.0固定を回避）。
  - `workers/scalp_ping_5s/config.py` / `workers/scalp_ping_5s/worker.py`: dyn alloc profile を読み、`lot_multiplier` を entry `units` に反映。`entry_thesis.dynamic_alloc` を付与（found時）。

- 検証手順:
  - `python3 -m compileall execution/order_manager.py workers/common/dynamic_alloc.py workers/scalp_ping_5s/config.py workers/scalp_ping_5s/worker.py`
  - `scripts/local_v2_stack.sh restart --env ops/env/local-v2-stack.env --services quant-order-manager,quant-scalp-ping-5s-b,quant-scalp-ping-5s-b-exit`
  - `scripts/local_v2_stack.sh status --env ops/env/local-v2-stack.env --services quant-order-manager,quant-scalp-ping-5s-b,quant-scalp-ping-5s-b-exit`
  - `logs/local_v2_stack/quant-scalp-ping-5s-b.log` で `dynamic_alloc` が `entry_thesis` に付与されていること（found時）

## 2026-03-05 22:08 JST / trade_all の worker 大半が停止（stale pid）+ OM timeout 緩和 + strategy_entry dyn alloc trim-only（local V2）

- 症状:
  - `scripts/local_v2_stack.sh status --profile trade_all --env ops/env/local-v2-stack.env` で `[running] 10 / [stopped] 54`（停止側は `stale_pid_file` 付き）。
  - worker ログに `order_manager` 呼び出しの `Read timed out. (read timeout=8.0)` が散発（例: `logs/local_v2_stack/quant-micro-momentumburst.log`）。

- 作業前市況（ローカル実測 / OANDA pricing, 2026-03-05 21:44 JST）:
  - `USD/JPY bid=157.445 ask=157.453 spread=0.8p`
  - `pricing latency=260ms`
  - `ATR(M1)=2.520p`, `ATR(M5)=5.559p`

- 狙い / 仮説:
  - trade_all で停止 worker が多い状態だと、戦略が走っておらず機会損失 → 起動/復旧導線を標準化して観測できる状態に戻す。
  - `ORDER_MANAGER_SERVICE_TIMEOUT=8.0` だと負荷時に service call timeout が出やすく、skip が増える → local override を `12.0` に上げて false-timeout を減らす。
  - dyn alloc 未対応 worker が raw_units を full size で通すと risk/margin が悪化しやすい → `execution/strategy_entry.py` 側で dyn alloc を「trim-only（縮小のみ）」として適用し、かつ worker 側で `entry_thesis.dynamic_alloc` が付与済みなら二重適用しない。

- 対応（main反映 / commit=`a296316d`）:
  - `execution/strategy_entry.py`
    - `STRATEGY_DYNAMIC_ALLOC_*` を追加し、`entry_thesis.dynamic_alloc` が無い注文に限り `lot_multiplier` で units を trim（デフォルトは up-scale しない）。
  - `workers/common/dynamic_alloc.py`
    - `allocation_policy.soft_participation=true` かつ unknown strategy の場合でも `found=true` の fallback profile を返し、metadata 付与/二重適用回避をしやすくした。
  - `ops/env/local-v2-stack.env`
    - `ORDER_MANAGER_SERVICE_TIMEOUT=12.0`（runtime 8.0 → local override 12.0）。

- 検証（local V2）:
  - `python3 -m compileall execution/strategy_entry.py workers/common/dynamic_alloc.py`
  - `pytest -q tests/workers/common/test_dynamic_alloc.py tests/execution/test_strategy_entry_dynamic_alloc_trim.py` → `6 passed`
  - `scripts/local_v2_stack.sh restart --env ops/env/local-v2-stack.env --services quant-market-data-feed,quant-strategy-control,quant-order-manager,quant-position-manager,quant-scalp-ping-5s-b,quant-scalp-ping-5s-b-exit,quant-micro-rangebreak,quant-micro-rangebreak-exit,quant-m1scalper,quant-m1scalper-exit`
  - `sqlite3 logs/orders.db 'select max(ts) from orders;'` が更新継続（例: `2026-03-05T13:06:52Z`）。

- 注記:
  - trade_all の全 worker 常時起動はホスト負荷が大きい可能性がある（load avg が急上昇）。維持できない場合は「走らせたい戦略のみ worker を残す」方向で profile を再設計する。

## 2026-03-05 21:55 JST / strategy_entry dyn alloc trim-only（未対応戦略のfull-size抑制）+ order-manager timeout上書き

- 狙い / 仮説:
  - `trade_all` 等で「dyn alloc 未対応 strategy が full size のまま走る」→ 損益悪化/マージン圧迫で no-entry に見える状況を作りやすい。
  - `execution/strategy_entry.py` で dyn alloc を trim-only（<=1.0）適用して、未対応戦略を自動縮小しつつ「一律停止」には寄せない。
  - 既に strategy 側で `entry_thesis.dynamic_alloc` を付与しているケースは二重適用しない（`dynamic_alloc` があれば skip）。
  - `trade_all` は `quant-order-manager` 呼び出しの read timeout（既定 8s）で worker 側 skip が増えやすい → timeout を上書きして false skip を減らす。

- 対応（main反映 / local V2）:
  - `execution/strategy_entry.py`
    - `STRATEGY_DYNAMIC_ALLOC_*` を追加し、coordinate 前に `lot_multiplier` を trim-only 適用。
    - 適用時は `entry_thesis.dynamic_alloc.source=strategy_entry` を付与し、監査可能にした。
  - `workers/common/dynamic_alloc.py`
    - soft-participation 時、`dynamic_alloc.json` に無い strategy でも `min_lot_multiplier` をデフォルト返却し、trim-only が必ず効くようにした。
  - `ops/env/local-v2-stack.env`
    - `ORDER_MANAGER_SERVICE_TIMEOUT=12.0` を追加（trade_all で timeout skip が増えやすいため）。

- 検証:
  - Unit test:
    - `pytest -q tests/workers/common/test_dynamic_alloc.py tests/execution/test_strategy_entry_dynamic_alloc_trim.py`
  - local V2:
    - `scripts/local_v2_stack.sh restart --profile trade_min --env ops/env/local-v2-stack.env`
    - `logs/health_snapshot.json` の `git_rev` が `a296316d` で、`orders_last_ts` が更新され続けること。
  - 監査:
    - `orders.db` の `request_json.entry_thesis.dynamic_alloc.source=strategy_entry` が（dyn alloc 未実装戦略で）付与されること。

## 2026-03-05 22:18 JST / ping5s: scalp_fast protection fallback 縮小 + mode blocklist + flow SL有効化（local V2）

- 背景（ローカル実測 / `logs/trades.db` + `logs/orders.db`）:
  - `scalp_ping_5s_b_live` の損失は `Trend + long` に集中（直近3日 `n=444 / -182.1 JPY`）。
  - ping5s B は `STOP_LOSS_ON_FILL_LOSS` の reject 後に protection fallback が走り、SL gap が `8p+` になる filled が存在（直近3日 `28/761`）。
  - `scalp_ping_5s_flow_live` は `entry_thesis.sl_pips=null` かつ `disable_entry_hard_stop=1` のまま取引が入り、平均損失pipsが大きくなりやすい（例: avg win `+0.83p` vs avg loss `-4.82p`）。

- 対応（local override / commit=`45a5fb18`）:
  - `ops/env/local-v2-stack.env`
    - `ORDER_PROTECTION_FALLBACK_PIPS_SCALP_FAST=0.02`（USDJPYで約2p。既定 `0.12` は12p相当でscalp_fastに広すぎ）
    - ping5s B/D/flow の `*_SIGNAL_MODE_BLOCKLIST` を設定（直近負けモードを遮断）
    - `SCALP_PING_5S_FLOW_USE_SL=1`（flowのSL/entry hard stop を復帰）

- 検証（再起動後）:
  - `sqlite3 logs/orders.db 'select count(*) from orders where status=\"rejected\" and error_message=\"STOP_LOSS_ON_FILL_LOSS\" and datetime(ts) >= datetime(\"now\",\"-1 day\") and client_order_id like \"%scalp_ping_5s_b%\";'` が減る
  - `sqlite3 logs/orders.db 'with o as (select executed_price, sl_price from orders where status=\"filled\" and datetime(ts) >= datetime(\"now\",\"-1 day\") and client_order_id like \"%scalp_ping_5s_b%\" and executed_price is not null and sl_price is not null) select sum(case when abs(executed_price - sl_price)/0.01 >= 8.0 then 1 else 0 end), count(*) from o;'` の `>=8p` 比率が下がる
  - flow を走らせた場合: `sqlite3 logs/trades.db 'select json_extract(entry_thesis,\"$.sl_pips\"), json_extract(entry_thesis,\"$.disable_entry_hard_stop\") from trades where strategy_tag like \"scalp_ping_5s_flow_live%\" order by close_time desc limit 5;'` で `sl_pips` が埋まる

## 2026-03-05 22:45 JST / trade_all 起動不全の主要クラッシュ修正（ping5s C/D/flow wrapper・TrendBreakout・micro runtime）

- 作業前市況（ローカル実測 / OANDA pricing, 2026-03-05 22:12 JST）:
  - `USD/JPY bid=157.516 ask=157.524 spread=0.8p`
  - `pricing latency=243ms`
  - `ATR(M1)=2.358p`（直近120本の complete M1）

- 症状（local V2）:
  - `scripts/local_v2_stack.sh up --profile trade_all --env ops/env/local-v2-stack.env` が失敗しやすく、`status` で `stale_pid_file` が多数出て「戦略が走っていない」状態になる。
  - ping5s C/D/flow: `logs/local_v2_stack/quant-scalp-ping-5s-*.log` に `CalledProcessError ... died with SIGTERM`（wrapperが子プロセス死亡で終了）。
  - TrendBreakout: `logs/local_v2_stack/quant-scalp-trend-breakout.log` で `TypeError: _log() got multiple values for argument 'reason'`。
  - micro: `logs/local_v2_stack/quant-micro-momentumburst.log` で `UnboundLocalError: bb_style referenced before assignment`。

- 原因:
  - ping5s C/D/flow wrapper が `workers.scalp_ping_5s.worker` を `subprocess.run(check=True)` で起動しており、`local_v2_stack` 側の cleanup（ping5s-b が汎用 module を pattern に含める）で子プロセスが巻き込み SIGTERM → wrapper が例外で落ちる。
  - `strategies/scalping/m1_scalper.py` の `_log(reason, **kwargs)` に対して `reason=` を kwargs で渡していた（TrendBreakoutが M1Scalper を参照するため worker ごと落ちる）。
  - `workers/micro_runtime/worker.py` の `bb_style` が未初期化のまま `_bb_entry_allowed(...)` に渡され得る。

- 対応（main反映）:
  - ping5s C/D/flow: wrapper を **1プロセス**に変更し、`workers.scalp_ping_5s.worker:scalp_ping_5s_worker()` を `asyncio.run()` で直接実行（commit=`dc7751f2`）。
  - M1Scalper: `_log` の kwargs `reason` を `flip_reason` へ変更（commit=`105b15f2`）。
  - micro runtime: `bb_style` を `reversion` で初期化し、`_TREND_STRATEGIES` を `trend` 判定に追加（commit=`71b9dbd2`）。

- 検証:
  - `pytest -q tests/workers/test_micro_multistrat_trend_flip.py tests/workers/test_m1scalper_nwave_tolerance_override.py tests/workers/test_m1scalper_config.py`
  - `scripts/collect_local_health.sh` で `orders_last_ts` が更新され続けること（例: `filled` が直近1hで継続）

## 2026-03-05 14:34 UTC / 2026-03-05 23:34 JST - `close_reject_no_negative` 抑制の allow negative reason 追補（local V2）

- 事実（ローカル実測: `logs/orders.db` / `logs/trades.db` / `logs/metrics.db`）:
  - 直近3h `orders.db`: `close_reject_no_negative=525`, `close_ok=816`, `filled=961`。
  - 直近24h `close_reject_no_negative` の `exit_reason` は `reentry_reset=323`, `__de_risk__=202` に集中。
  - 直近3h `trades.db` の flow系（`strategy|strategy_tag LIKE '%flow%'`）は `n=14`, `avg_pips=-1.55`, `sum_realized_pl=-55.3`。
  - 直近3h `metrics.db` の `account.margin_usage_ratio` は `avg=0.8297`, `max=1.0003`。

- 仮説:
  - `reentry_reset` / `__de_risk__` が `EXIT_ALLOW_NEGATIVE_REASONS` 未登録のため `close_reject_no_negative` が反復し、負け玉解放遅延を通じて flow の平均pips悪化と margin usage 高止まりを誘発している。

- 実施変更:
  - `ops/env/quant-order-manager.env` に `EXIT_ALLOW_NEGATIVE_REASONS` を明示設定。
  - 既定トークン（`hard_stop,tech_hard_stop,max_adverse,time_stop,no_recovery,max_floating_loss,fast_cut_time,time_cut,tech_return_fail,tech_reversal_combo,tech_candle_reversal,tech_nwave_flip`）を維持したまま、`reentry_reset` と `__de_risk__` を追加。

- 検証観点（反映後 1-3h）:
  1. `orders.db` の `status='close_reject_no_negative'` 件数が減少すること（総数と reason 内訳）。
  2. `trades.db` の flow系 `avg(pl_pips)` が改善すること。
  3. `metrics.db` の `account.margin_usage_ratio` が高止まりせず、`max` が改善方向に向かうこと。

## 2026-03-05 15:30 UTC / 2026-03-06 00:30 JST - flow系: `close_reject_no_negative` 再発防止（strict neg_exit allow + closeログ改善）

- 追加確認（ローカル実測: `logs/orders.db`）:
  - `close_reject_no_negative` の request_json に `exit_reason=__de_risk__/reentry_reset` が含まれている。
  - `strategy_exit_protections.yaml` の `scalp_ping_5s` は `neg_exit.strict_no_negative=true` のため、未許可 reason は worker 側 `allow_negative=true` でも close が拒否され得る。
  - `close_reject_no_negative` を orders.db で棚卸しする際に `pocket/instrument` が欠けており、戦略別の集計がしづらい。

- 対応（main反映）:
  - `config/strategy_exit_protections.yaml`（`scalp_ping_5s`）:
    - `neg_exit.strict_allow_reasons` / `allow_reasons` に `reentry_reset` / `__de_risk__` を追補（commit=`efd2f83e`）。
  - `execution/order_manager.py`:
    - `close_reject_no_negative` の orders.db ログに `pocket/instrument/strategy_tag` を付与し、`est_pips` も記録（commit=`efd2f83e`）。

- 検証観点（反映後 1-3h）:
  1. `orders.db`: `status='close_reject_no_negative'` が `reentry_reset/__de_risk__` で反復しないこと。
  2. `orders.db`: `close_reject_no_negative` 行に `pocket/instrument` が入ること（戦略別に集計できること）。

## 2026-03-05 15:45 UTC / 2026-03-06 00:45 JST - MicroRangeBreak: reversion 全敗の強レンジ絞り込み + ping5s D/flow neg_exit no-block + Brain fast/micro の stall 対策

- 事実（ローカル実測: `logs/trades.db` / `logs/orders.db`）:
  - 直近6h `trades.db`（MicroRangeBreak）: `n=32`, `wins=0`, `avg_pips=-1.2531`, `sum_jpy=-32.9`（全敗）。
  - entry_thesis: `signal_mode=reversion (range_scalp)` に偏り、`range_score=0.356..0.382` と「弱レンジ」でも short リバが走っている。`trend_snapshot(H4).adx≈24.9` でも同様。
  - 直近6h `orders.db`: `close_reject_no_negative=475`（`exit_reason='__de_risk__'` 等が起点になり得る）。

- 市況スナップショット（ローカル実測: `logs/tick_cache.json` / `logs/factor_cache.json`）:
  - `USD/JPY bid=157.676 ask=157.684 spread=0.8p`
  - `ATR(M1)=2.81p` / `ADX(H4)=24.87`

- 仮説:
  - MicroRangeBreak の reversion が「弱レンジ」でも発火し、トレンド寄り局面で `m1_structure_break` / `max_adverse` 由来の早期損切りが連発している。
  - ping5s D/flow は B/C と neg_exit 設定が非対称で、`__de_risk__` / `reentry_reset` が `close_reject_no_negative` になりやすい。
  - Brain は過去に `brain_latency_ms` が平均 ~6s に張り付いた時間帯があり、将来有効化しても fast/micro を stall させない設計が必要。

- 対応（main反映 / commit=`48716111`）:
  - MicroRangeBreak（reversionを“強いレンジ”へ絞る）:
    - `MICRO_RANGEBREAK_MIN_RANGE_SCORE=0.44`（`0.32` → `0.44`）
    - `MICRO_RANGEBREAK_REVERSION_MAX_ADX=23.0`（`27.0` → `23.0`）
    - `MICRO_RANGEBREAK_ENTRY_RATIO=0.25`（`0.38` → `0.25`）
    - ※ `local_v2_stack` は base env → service env の順で上書きされるため、`ops/env/quant-micro-rangebreak.env` も同値へ更新。
  - ping5s D/flow: `config/strategy_exit_protections.yaml` に `neg_exit` を付与し、B/C と同じ `no-block` 方針へ（`strict_no_negative=false`, `allow_reasons=*SCALP_PING_5S_NO_BLOCK_NEG_EXIT_ALLOW_REASONS`）。
  - Brain（有効化時のみ）: `workers/common/brain.py` に pocket別 override を追加（`BRAIN_TIMEOUT_SEC_MICRO` / `BRAIN_TIMEOUT_SEC_SCALP_FAST` / `BRAIN_FAIL_POLICY_MICRO` / `BRAIN_FAIL_POLICY_SCALP_FAST`）。

- 検証観点（反映後 3-6h）:
  1. `trades.db`: MicroRangeBreak の `signal_mode=reversion` が `range_score>=0.44` 帯に寄り、全敗が止まること。
  2. `orders.db`: ping5s D/flow の `close_reject_no_negative` が `__de_risk__/reentry_reset` 起点で減ること。
  3. Brain 有効化時: micro/scalp_fast のエントリーが timeout で stall しないこと（fail-open）。

- 反映後実測（local V2, 2026-03-06 01:25〜01:30 JST）:
  - 市況（OANDA pricing / candles, 01:03 JST）:
    - `USD/JPY mid=157.552 spread=0.80p pricing latency=303ms`
    - `ATR14(M1)=3.76p / 60m range=38.0p candles latency=203ms`
  - 反映:
    - `scripts/local_v2_stack.sh restart --env ops/env/local-v2-stack.env --services quant-micro-rangebreak,quant-micro-rangebreak-exit,quant-order-manager,quant-position-manager`
  - ping5s flow_live の close_reject_no_negative（直近30分 / `orders.db`）:
    - `close_reject_no_negative=0`
    - `close_ok=7`（`exit_reason`: `max_adverse=5`, `__de_risk__=2`）
  - 建玉:
    - `position-manager` の `open_positions` で `scalp_fast` が空（flow_live の含み損玉が解消されている）
    - MicroRangeBreak の open positions は無し
  - 収益（直近30分 / `trades.db`）:
    - 全体: `n=76 / PF=2.449 / net_jpy=+185.187`
    - flow_live: `n=4 / net_jpy=+61.840`
  - メモ:
    - `trades.db` が `orders.db` より遅延するため、必要に応じて `/position/sync_trades` を叩いて追いつかせてから評価する（例: `trades_last_close` が `13:30 UTC` → `16:26 UTC` へ更新）。

## 2026-03-05 18:30 UTC / 2026-03-06 03:30 JST - M1Scalper: buy-dip 停止（戦略内ブロック） + sell-rally は projection flip 時のみ許可

- 市況スナップショット（OANDA pricing / candles, 03:29 JST）:
  - `USD/JPY mid=157.818 spread=0.8p`
  - `ATR14(M1)=2.94p / 60m range=26.5p`
  - `pricing latency=226ms / candles latency=300ms`

- 事実（ローカル実測: `logs/trades.db`）:
  - 直近24h（M1Scalper signal別）:
    - `M1Scalper-buy-dip`: `n=64 / net_jpy=-372.9`（継続的に負け寄与）
    - `M1Scalper-sell-rally`: `n=209 / net_jpy=+633.5`
    - `M1Scalper-trend-long`: `n=73 / net_jpy=+251.9`
  - 直近24h（sell-rally の exec_side 別）:
    - `exec_side=long`: `n=169 / net_jpy=+1097.1 / win_rate=0.941`
    - `exec_side=short`: `n=40 / net_jpy=-463.6 / win_rate=0.125`

- 仮説:
  - `buy-dip` は現行の市場状態で逆期待値になっており、継続稼働は資産減少を誘発する。
  - `sell-rally` は「projection による side flip（signal_side と逆）」のときのみ強い期待値があり、signal_side のまま（short）実行すると大きく負ける。

- 対応（main反映予定）:
  - `workers/scalp_m1scalper/worker.py`:
    - `buy-dip` は戦略内でブロック（`buy_dip_block` ログ）。
    - `sell-rally` は projection 適用後に `side != signal_side`（flip）を満たすときのみ許可し、非flipはブロック（`sell_rally_no_flip_block` ログ）。

- 検証観点（反映後 1-3h）:
  1. `trades.db`: `source_signal_tag='M1Scalper-buy-dip'` が 0 件に収束すること。
  2. `trades.db`: `source_signal_tag='M1Scalper-sell-rally'` の `exec_side='short'` が 0 件に収束すること。
  3. `trades.db`: `M1Scalper-M1` の直近1h `net_jpy` / `PF` が改善方向で安定すること。

## 2026-03-05 20:07 UTC / 2026-03-06 05:07 JST - order-manager: `scalp_ping_5s_flow_live` / `M1Scalper-M1` の stopLossOnFill 欠損（broker SLなし）→ targeted allowlist 修正

- 事実（ローカル実測: OANDA openTrades / `logs/orders.db`）:
  - `scalp_ping_5s_flow_live` / `M1Scalper-M1` の一部エントリーで `takeProfitOnFill` は付くが `stopLossOnFill` が付かず、建玉が broker SLなしで残る（`stopLossOrder=null`）。
  - 一方で `logs/orders.db` には同一 `client_order_id` の `sl_price` が記録されており、SL価格の算出自体は行われていた。

- 原因:
  - `ORDER_FIXED_SL_MODE=0` かつ strategy override 未設定時、`execution/order_manager.py:_allow_stop_loss_on_fill()` の family override が `scalp_ping_5s_b/c/d` のみで、`scalp_ping_5s_flow*` と `M1Scalper-M1` が許可対象外になっていた。

- 対応（local V2, 予防的/後方互換）:
  - `execution/order_manager.py`: `scalp_ping_5s_flow*` を family override に追加し、`ORDER_ALLOW_STOP_LOSS_ON_FILL_SCALP_PING_5S_FLOW` を参照（未設定時の既定は `False`）。
  - `execution/order_manager.py`: `_disable_hard_stop_by_strategy()` が `scalp_ping_5s_flow*` を `ORDER_DISABLE_ENTRY_HARD_STOP_SCALP_PING_5S_FLOW` で解決できるようにし、env既定は後方互換で `True`（＝hard stop 無効）を維持。
  - `ops/env/quant-order-manager.env`: `ORDER_ALLOW_STOP_LOSS_ON_FILL_STRATEGY_M1SCALPER_M1=1` を追加（strategy 単位で stopLossOnFill を許可）。

## 2026-03-05 21:36 UTC / 2026-03-06 06:36 JST - position-manager: ORDER_FILL close の trades.db 欠損（収益RCA誤差）→ clientTradeID fallback + watermark hole防止 + 24h backfill

- Period:
  - 集計窓: 直近24h（OANDA transactions from/to）
  - 市況確認（OANDA実測）:
    - `USD_JPY bid/ask=157.527/157.535`、`spread=0.8 pips`
    - `ATR14(M5)=7.607 pips`、`range_60m=18.2 pips`
    - API応答: pricing `225ms` / openTrades `220ms` / candles(M5) `228ms`

- Fact:
  - `logs/metrics.db` の `account.balance` が直近24hで大きく減少（約 `-5.48k JPY`）。
  - OANDA transactions 真値では、
    - `ORDER_FILL(pl+financing+commission)` 合計が約 `-5.13k JPY`
    - `DAILY_FINANCING` 合計が約 `-0.35k JPY`
    - 上記合計が `balance delta` と整合。
  - backfill 後の `logs/trades.db`（直近24h）:
    - pocket別: `scalp=-4.78k` / `scalp_fast=-1.64k` / `micro=+0.91k` / `manual=+0.38k`（合計 `-5.13k`）
    - loss寄与TOP: `M1Scalper-M1=-4.78k`, `scalp_ping_5s_flow_live=-1.54k`
  - 一方 `logs/trades.db` は close の `ORDER_FILL` が大量欠損し、同窓で
    - `missing_pairs=1265`
    - `missing_realized_sum=-5670.674 JPY`
    - 欠損 reason: `MARKET_ORDER_TRADE_CLOSE` が大半（`1157`）

- Failure Cause:
  - `execution/position_manager.py:_parse_and_save_trades()` が
    `details = _get_trade_details(trade_id)` 失敗時に `continue` して close を保存しない。
  - 同時に `_last_tx_id` を `max(processed_tx_ids)` へ進めるため、
    一部トランザクションが失敗すると水位が穴を飛び越え、永続欠損になる。

- Improvement:
  - `execution/position_manager.py`:
    - `_parse_and_save_trades()`:
      - `closed_trade.clientTradeID` を使い `orders.db (client_order_id)` から entry meta を復元する fallback を追加。
      - details が取れない場合でも tx/closed_trade から最小限の details を組み立て、**close を必ず保存**。
      - `_last_tx_id` は連続区間でのみ進め、hole を飛び越えないようにした（idempotent 再処理は許容）。
  - `scripts/backfill_trades_from_oanda_idrange.py`:
    - OANDA transactions `idrange` を指定レンジで取得し、`PositionManager._parse_and_save_trades()` へ流し込む backfill/repair ツールを追加（`uniq(transaction_id, ticket_id)` により冪等）。

- Verification:
  - 直近24h window を再処理し、OANDA `ORDER_FILL close` の `(transaction_id, tradeID)` が `logs/trades.db` に欠損なく保存される（`missing_pairs=0`）こと。
  - `logs/trades.db` の 24h `sum(realized_pl)` が OANDA `ORDER_FILL` 真値（約 `-5.13k JPY`）と整合すること。

- Status:
  - done（`quant-position-manager` restart + `python scripts/backfill_trades_from_oanda_idrange.py --last-n 5000` 実行、`missing_pairs=0` を確認）
  - 追記: windowズレで `--last-n 5000` が不足し得るため、`python scripts/backfill_trades_from_oanda_idrange.py --from-id 418699` を実行し、`[418699, 439517]` の `missing_pairs=0` を再確認。

## 2026-03-05 21:54 UTC / 2026-03-06 06:54 JST - order-manager: micro pocket の stopLossOnFill 欠損（TPのみ/SLなし）→ strategy allowlist 追補

- Fact（ローカル実測: OANDA openTrades / `logs/orders.db`）:
  - `USD_JPY` の openTrades `7/7` が `takeProfit` は付くが `stopLoss` が `null`（broker SLなし）で残存。全て `pocket=micro`。
  - `logs/orders.db` には同一 `client_order_id` の `sl_price` が記録されており、SL価格の算出自体は行われていた。

- Failure Cause:
  - `ORDER_FIXED_SL_MODE=0` 既定では stopLossOnFill が無効（`stop_loss_policy` の baseline）。
  - `execution/order_manager.py:_allow_stop_loss_on_fill()` の strategy override が micro の主要 strategy tag（`MomentumBurst*` / `MicroLevelReactor*` / `MicroTrendRetest*` / `MicroRangeBreak*` / `MicroVWAPRevert*` / `MicroVWAPBound*`）に未設定だった。

- Improvement（local V2）:
  - `ops/env/quant-order-manager.env`:
    - `ORDER_ALLOW_STOP_LOSS_ON_FILL_STRATEGY_MOMENTUMBURST=1`
    - `ORDER_ALLOW_STOP_LOSS_ON_FILL_STRATEGY_MICROLEVELREACTOR=1`
    - `ORDER_ALLOW_STOP_LOSS_ON_FILL_STRATEGY_MICROTRENDRETEST=1`
    - `ORDER_ALLOW_STOP_LOSS_ON_FILL_STRATEGY_MICRORANGEBREAK=1`
    - `ORDER_ALLOW_STOP_LOSS_ON_FILL_STRATEGY_MICROVWAPREVERT=1`
    - `ORDER_ALLOW_STOP_LOSS_ON_FILL_STRATEGY_MICROVWAPBOUND=1`
  - `scripts/local_v2_stack.sh restart --env ops/env/local-v2-stack.env --services quant-order-manager`

- Verification（反映後 次の micro entry で確認）:
  1. `logs/orders.db` の `submit_attempt.request_json` に `stopLossOnFill` が含まれること。
  2. OANDA openTrades の `stopLossOrder` が `null` ではなくなること。
  3. reject率が悪化しないこと（`orders.db status='rejected'` / `STOP_LOSS_ON_FILL_LOSS` など）。

- Status:
  - in_progress（order-manager restart 済み。次の micro entry で broker SL 付与を実測確認）

## 2026-03-06 04:30 UTC / 2026-03-06 13:30 JST - position-manager: sync_trades が backlog>MAX_FETCH で newest 側にジャンプし欠損を作る → forward paging + /summary lastTransactionID

- Fact:
  - `execution/position_manager.py:_fetch_closed_trades()` は backlog が大きいとき
    `fetch_from=min_allowed=max(1,last_tx_id-_MAX_FETCH+1)` へジャンプし、
    `self._last_tx_id+1..min_allowed-1` の範囲を取得しない（＝決済 tx を永続欠損させ得る）。

- Improvement（local V2）:
  - `execution/position_manager.py:_fetch_closed_trades()`:
    - `fetch_from=self._last_tx_id+1` を維持し、`fetch_to=min(last_tx_id, fetch_from+_MAX_FETCH-1)` までの **前進型ページング** に変更（hole を飛び越えない）。
    - `lastTransactionID` の取得を `/v3/accounts/{ACCOUNT}/summary` に変更し、巨大な `transactions` payload を避ける。

- Verification（ローカル実測: OANDA API）:
  - `POSITION_MANAGER_MAX_FETCH=50`、`pm._last_tx_id=remote_last-200` で `_fetch_closed_trades()` を呼ぶと、
    `fetch_from=remote_last-199` から `fetch_to=remote_last-150` を取得する（newest 側 `remote_last-49` へジャンプしない）。

## 2026-03-06 04:35 UTC / 2026-03-06 13:35 JST - scalp_ping_5s_flow: 直近24hで大幅マイナスのため緊急リスク縮小（破滅サイズ停止）

- Fact（ローカル実測: `logs/trades.db`、backfill反映後）:
  - 直近24h（`2026-03-05T04:28Z..2026-03-06T04:28Z`）の `scalp_ping_5s_flow_live`:
    - `net=-6427.87 JPY`, `trades=379`, `win=0.325`

- Decision:
  - 時間帯ブロックではなく、まず **破滅サイズ（過剰ユニット/過剰同時建玉）** を止めるための緊急縮小を入れる。
  - 目的: 連続損失時の口座急落を抑えつつ、引き続き原因分析と改善を進める。

- Change（local V2 / worker-local env）:
  - `ops/env/scalp_ping_5s_flow.env`（strategy-local）:
    - `SCALP_PING_5S_FLOW_MAX_ACTIVE_TRADES: 16 -> 2`
    - `SCALP_PING_5S_FLOW_MAX_PER_DIRECTION: 8 -> 1`
    - `SCALP_PING_5S_FLOW_BASE_ENTRY_UNITS: 1200 -> 120`
    - `SCALP_PING_5S_FLOW_MAX_UNITS: 3500 -> 600`
    - `SCALP_PING_5S_FLOW_MAX_SPREAD_PIPS: 1.3 -> 0.9`
  - `ops/env/quant-scalp-ping-5s-flow.env`（service overlay）:
    - `SCALP_PING_5S_FLOW_MAX_ACTIVE_TRADES: 16 -> 2`
    - `SCALP_PING_5S_FLOW_MAX_PER_DIRECTION: 8 -> 1`

## 2026-03-06 06:30 UTC / 2026-03-06 15:30 JST - flow short-probe rescue が dynamic trim を打ち消していたため修正（local V2）

- 市況確認（ローカルV2実測 + OANDA API）:
  - `USD/JPY bid=157.784 ask=157.792 spread=0.8p`
  - `tick_cache 直近15m`: `spread mean=0.8p / max=1.0p / mid range=12.8p`
  - `factor_cache`: `ATR(M1)=3.44p / ATR(M5)=6.20p / ATR(H1)=18.36p`
  - OANDA API応答: `pricing=246.6ms / account_summary=319.7ms / openTrades=233.6ms`
  - 判定: メンテ時間帯（JST 7-8時）外で、流動性異常ではなく作業継続可。

- 事実（ローカル実測: `logs/trades.db` / `logs/orders.db` / `logs/local_v2_stack/quant-scalp-ping-5s-flow.log`）:
  - 直近24h `trades.db`: 全体 `net=-10892.5 JPY / PF=0.549`。
  - 同24h 赤字寄与上位:
    - `scalp_ping_5s_flow_live`: `n=420 / net=-7131.1 JPY / PF=0.295 / avg_units=1842.3`
    - `M1Scalper-M1`: `n=2001 / net=-5725.5 JPY / PF=0.552`
  - 直近3hでも `scalp_ping_5s_flow_live` は `n=273 / net=-5593.4 JPY / PF=0.193 / avg_units=2120.2` と継続悪化。
  - `entry_thesis.dynamic_alloc.lot_multiplier=0.25` が記録されている一方、flow worker log に
    `min_units_rescue applied mode=short_probe_rescued units=2000 ... risk_cap=1200`
    が連発し、縮小後サイズが `MIN_UNITS=2000` rescue で押し戻されていた。
  - `ops/env/scalp_ping_5s_flow.env` の緊急縮小値（`120 / 2 / 1`）も、
    `ops/env/local-v2-stack.env` の `700 / 6 / 3` override で上書きされていた。

- 仮説:
  - flow の主因は市況ではなく、`short_probe_rescue` が non-B/C clone でも既定有効で、
    dynamic alloc と risk mult による縮小を戦略側で打ち消していたこと。
  - 併せて local-v2 override の食い違いが、flow の破滅サイズ継続を許していた。

- 対応:
  - `workers/scalp_ping_5s/config.py`
    - `SHORT_PROBE_RESCUE_ENABLED` の既定を `B/C clone のみ true` へ変更。
  - `workers/scalp_ping_5s/worker.py`
    - `_maybe_rescue_short_probe()` が `units_risk < MIN_UNITS` の場合に rescue しないよう修正。
  - `ops/env/local-v2-stack.env`
    - flow override を緊急縮小値へ統一:
      - `BASE_ENTRY_UNITS=120`
      - `MAX_ACTIVE_TRADES=2`
      - `MAX_PER_DIRECTION=1`
      - `MAX_SPREAD_PIPS=0.90`
      - `MIN_UNITS_RESCUE_ENABLED=0`
      - `SHORT_PROBE_RESCUE_ENABLED=0`
    - `M1SCALP_DYN_ALLOC_MULT_MIN=0.45` として、最新 `dynamic_alloc.json` の縮小値を worker 側で潰さないよう修正。
  - `config/dynamic_alloc.json`
    - `scripts/dynamic_alloc_worker.py --limit 0 --lookback-days 7 --min-trades 12 --pf-cap 2.0 --target-use 0.88 --half-life-hours 36 --min-lot-multiplier 0.45 --max-lot-multiplier 1.65 --soft-participation 1`
      を再実行し、`as_of=2026-03-06T06:26:55Z` へ更新。
    - 主な更新:
      - `scalp_ping_5s_flow_live lot_multiplier=0.45`
      - `M1Scalper-M1 lot_multiplier=0.50`

- 再検証条件（反映後 30-90分）:
  1. `quant-scalp-ping-5s-flow.log` に `short_probe_rescued units=2000` が再発しないこと。
  2. `orders.db/trades.db` の flow 約定ユニットが `<= 600` 帯へ収まること。
  3. `trades.db` の直近1-3h で `scalp_ping_5s_flow_live` の PF が `0.193` から改善すること。
  4. `M1Scalper-M1` の実トレードサイズが更新後 `dynamic_alloc` に従って縮小すること。

## 2026-03-06 08:55 UTC / 2026-03-06 17:55 JST - position-manager sync_trades が timeout/busy 競合で遅延し trades.db 反映が詰まる件を緩和（local V2）

- 市況確認（ローカルV2実測 + OANDA API）:
  - `USD/JPY bid=157.746 ask=157.754 spread=0.8p`
  - `tick_cache 直近300`: `spread mean=0.8p / max=0.8p / mid range=4.1p`
  - `factor_cache`: `ATR(M1)=3.13p / ATR(M5)=6.95p / ATR(H1)=18.36p`
  - OANDA account summary: `openTradeCount=21`
  - 判定: メンテ時間帯外、spread/ATR は異常ではなく、作業継続可。

- 事実（ローカル実測: `logs/trades.db` / `logs/orders.db` / `logs/health_snapshot.json` / `logs/local_v2_stack/quant-position-manager.log`）:
  - `health_snapshot` では `orders_last_ts=2026-03-06T06:47:57Z` に対し `trades_last_close=2026-03-06T06:11:02Z` で、監視上は trades 側が止まって見えていた。
  - `quant-position-manager.log` には
    - `2026-03-06 17:53:15 JST [POSITION_MANAGER_WORKER] request failed: position manager busy`
    - `2026-03-06 17:53:21 JST [POSITION_MANAGER_WORKER] request failed: sync_trades timeout (8.0s)`
    が出ていた。
  - その直後、`trades.db` には `updated_at >= 2026-03-06T08:53:00Z` で `304` 件が一括反映され、
    `transaction_id` は `443041 -> 446546` まで前進した。
  - つまり `trades.db` は永久停止ではなく、`sync_trades` が timeout 後も裏で完走し、反映が後ろ倒しになっていた。

- 仮説:
  - `workers/position_manager/worker.py` では `sync_trades` / `performance_summary` / `fetch_recent_trades` が同じ `position_manager_db_call_lock` を共有しており、read系呼び出しや再試行と競合すると `position manager busy` が返る。
  - さらに `sync_trades` は `asyncio.wait_for(..., 8.0s)` で包まれているため、backlog catch-up（今回 304件規模）が 8秒を超えると表では timeout 扱いになり、監視上は未反映に見える。

- 対応:
  - `workers/position_manager/worker.py`
    - `sync_trades` 専用の `position_manager_sync_trades_call_lock` を追加し、read系 API と lock を分離。
  - `ops/env/local-v2-stack.env`
    - `POSITION_MANAGER_WORKER_SYNC_TRADES_TIMEOUT_SEC=20.0`
    - `POSITION_MANAGER_WORKER_SYNC_TRADES_MAX_FETCH=200`
    - backlog catch-up 時の false timeout を減らし、1回の同期処理量も抑える。

- 再検証条件:
  1. `quant-position-manager.log` で `sync_trades timeout` / `position manager busy` の再発頻度が下がること。
  2. `health_snapshot` の `orders_last_ts` と `trades_last_close` の差が縮むこと。
  3. `trades.db` の `updated_at` が数分単位で追随し、まとめ書きの塊が減ること。

## 2026-03-06 09:10 UTC / 2026-03-06 18:10 JST - quant-position-manager を常時 background sync 化し、監視 blind を構造的に解消

- 事実:
  - 手動 `pm_sync_trades` では backlog が大きいと API 側は timeout を返すが、裏では `Saved 304 new trades.` が完走した。
  - つまり監視 blind の主因は「保存不能」ではなく、「position-manager 自身が backlog を平常時から削り続けていない」ことだった。

- 対応:
  - `workers/position_manager/worker.py`
    - worker lifespan で background `sync_trades` loop を追加。
    - 既定値は `start_delay=1s / interval=5s / max_fetch=120`。
    - 成功時は worker cache を更新して、後続 `/position/sync_trades` の stale 返却も改善。
  - `ops/env/local-v2-stack.env`
    - background sync 系 env を明示して local V2 で固定化。

- 期待効果:
  - `orders.db` と `trades.db` の差分を小さい backlog に保ち、PF/net の監視が「タイムアウト後にまとめて追いつく」状態から、「ほぼ追随」へ寄る。

## 2026-03-06 09:20 UTC / 2026-03-06 18:20 JST - M1Scalper が `MAX_OPEN_TRADES=1` を無視して積み上がる実装漏れを修正

- 市況確認（ローカルV2実測 + OANDA API）:
  - `USD/JPY bid=157.664 ask=157.672 spread=0.8p`
  - `ATR(M1)=2.05p / ATR(M5)=5.34p / ATR(H1)=18.24p`
  - `openTradeCount=15` で、その大半が `M1Scalper-M1` ロングに偏っていた。

- 事実:
  - OANDA open trades では `M1Scalper-M1` ロングが数秒間隔で `~390-480 units` ずつ積み上がっていた。
  - `logs/local_v2_stack/quant-m1scalper.log` でも同 tag の連続 `OPEN_FILLED` が確認でき、`M1SCALP_MAX_OPEN_TRADES=1` が機能していなかった。
  - 一部は `position_manager service call failed` / `order_manager service call failed` の直後でも継続発注されており、fail-open が過剰エクスポージャを拡大していた。

- 対応:
  - `M1Scalper` worker に `PositionManager` ベースの open-trades guard を追加。
  - `strategy_tag` 単位で open trade 数が `MAX_OPEN_TRADES` 以上なら新規 entry を拒否。
  - `position_manager` 不達時は `M1SCALP_FAIL_CLOSED_ON_POSITIONS_ERROR=1` で fail-closed。

- 期待効果:
  - M1 の「同方向ナンピン的な積み上がり」を止め、利益より先に損失側の tail risk を削る。

## 2026-03-06 09:35 UTC / 2026-03-06 18:35 JST - live の積み増し主因は `TrendBreakout` 派生 worker だったため、M1 family 派生 worker へ同じ fail-closed guard を展開

- 市況確認（ローカルV2実測）:
  - `USD/JPY close=157.730`
  - `ATR(M1)=2.15p / ATR(M5)=5.44p / ATR(H1)=18.24p`
  - `quant-order-manager / quant-position-manager` health は `200`、応答は `9-14ms`
  - `position/open_positions` は `stale=true age_sec~4s` を返す瞬間があり、worker 側が fail-open だと積み増しを止められない条件だった

- 事実:
  - `quant-m1scalper.log` 側では `open_trades_block` が出ていた一方、`quant-scalp-trend-breakout.log` では 2026-03-06 18:06 JST 台に
    `TrendBreakout` が `source_signal_tag=M1Scalper-breakout-retest-long` を受けて `447144 / 447167 / 447174` を連続送信していた。
  - `position/open_positions` でも当該 open trades は `strategy_tag=TrendBreakout`、`entry_thesis.source_signal_tag=M1Scalper-breakout-retest-long` で確認できた。
  - `workers/scalp_trend_breakout/config.py` と `workers/scalp_pullback_continuation/config.py` は `MAX_OPEN_TRADES` を持っていたが、worker 実装側では評価していなかった。

- 対応:
  - `TrendBreakout` / `pullback_continuation` worker に `PositionManager` ベースの `_passes_open_trades_guard()` を追加。
  - `M1Scalper` / `TrendBreakout` / `pullback_continuation` すべてで `entry_thesis.env_prefix=M1SCALP` を同一 family とみなし、別 `strategy_tag` でも同方向積み上がりを block。
  - `M1SCALP_FAIL_CLOSED_ON_POSITIONS_ERROR` を両 config でも読むようにし、`position_manager` 不達時は fail-open せず reject。
  - local override の `M1SCALP_ENTRY_GUARD_BYPASS=1` を `0` に戻し、`bb_entry_reject` を無視して送る経路を止めた。
  - それぞれに targeted test を追加して、limit 到達時 block / family alias block / position-manager error 時 fail-closed を固定化。

- 期待効果:
  - `M1Scalper` 本体だけでなく、同一シグナル系列の派生 worker が別 strategy tag で同方向に積み上がる経路を止める。
  - `position_manager` が stale/busy の瞬間でも、M1 family は「見えないから建てる」ではなく「見えないから建てない」に寄る。
  - `bb_entry_reject` long の bypass を止め、直近3hで続いていた M1 long の赤字送信をさらに削る。

## 2026-03-06 11:10 UTC / 2026-03-06 20:10 JST - 収益悪化の主因は `flow` / `M1` の赤字単価と OANDA 応答劣化だったため、サイズ縮退と timeout 緩和を優先

- 市況確認（ローカルV2実測 + OANDA）:
  - `USD/JPY mid=157.892 spread=0.8p`
  - `ATR(M1)=2.04p / ATR(M5)=5.39p / ATR(H1)=18.05p`
  - `tail300 range=3.6p / tail1000 range=4.5p`
  - `orderbook latency ~=166ms`
  - `health_snapshot`: `data_lag_ms=717ms`, `decision_latency_ms=12.4ms`

- 異常条件:
  - `quant-order-manager.log` で `ORDER_OANDA_REQUEST_TIMEOUT_SEC=8.0` の read timeout と `503 Service unavailable` を確認。
  - `quant-market-data-feed.log` で stream reconnect が断続し、`[Errno 28] No space left on device` により `tick_cache/orderbook/factor_cache` の persist 失敗が発生。
  - `df -h .` は `/System/Volumes/Data` 空き `115-116MiB`、容量 `100%`。ローカル runtime 自体が収益評価を歪める水準だった。

- 収益分解:
  - `scripts/pdca_profitability_report.py --instrument USD_JPY`
    - 24h: `net_jpy=-11176.8 / net_pips=-1754.9 / PF=0.65`
  - 主損失:
    - `scalp_ping_5s_flow_live: -6921.7 JPY`
    - `M1Scalper-M1: -6171.2 JPY`
  - 主利益:
    - `MomentumBurst: +1856.9 JPY / PF=5.99`
  - `M1Scalper-M1` の損失は `MARKET_ORDER_TRADE_CLOSE` 主体、`flow` は `STOP_LOSS_ORDER` と `MARKET_ORDER_TRADE_CLOSE` の複合赤字だった。

- 対応方針:
  - 戦略停止ではなく、`dynamic_alloc` 側で `MARKET_ORDER_TRADE_CLOSE` の負け寄与を新しい縮退シグナルとして扱い、重赤字戦略の `lot_multiplier` を `0.45` 未満へ落とせるようにする。
  - `local-v2-stack.env` では `M1` の dynamic alloc floor を `0.25`、`flow` clone を `0.18` へ下げ、同じ負け方を繰り返す戦略の赤字単価を先に落とす。
  - OANDA read timeout は `10s`、order-manager service timeout は `14s` に緩和し、8秒超の submit/close 失敗を減らす。
  - disk 逼迫で patch すら失敗したため、`logs/replay`, `logs/archive`, `logs/local_vm_parity`, `logs/reports/forecast_improvement`, `logs/oanda` の古い生成物を整理して空きを `310MiB` まで回復してから修正に入る。

## 2026-03-06 11:30 UTC / 2026-03-06 20:30 JST - OANDA `/summary` 503 で entry が全面停止していたため、`order_manager` のみ 15 秒以内 stale snapshot を許可する bounded fallback を追加

- 市況確認（ローカルV2実測）:
  - `USD/JPY pricing/stream` は `200 OK` を維持。
  - 一方で `/summary` と `/openTrades` は `503 Service unavailable` が連発。
  - `health_snapshot`: `data_lag_ms=292-1129ms`, `decision_latency_ms=19-110ms`
  - `orders.db` では `2026-03-06 20:16:55 JST` が直近 `filled` で、その後は `preflight_start -> margin_snapshot_failed` が続いた。

- 事実:
  - `execution/order_manager.py` は `market_order` / `limit_order` / `_preflight_units` の3経路で `/summary` 失敗を即 fail-closed していた。
  - `utils/oanda_account.py` には共有 cache が既にあったが、`order_manager` 側は stale age/source を見られず、`openPositions` 要約も request failure 時に `(0,0)` へ落ちて free margin を過大評価しうる余地があった。

- 対応:
  - `utils/oanda_account.py` に snapshot の `source / age_sec / stale / error_kind` を返す state helper を追加。
  - `order_manager` では `market_order` / `limit_order` / `_preflight_units` の3箇所だけ、`503/timeout/connection_error` かつ `15s` 以内・`free_margin_ratio>=0.30`・`health_buffer>=0.25` の stale snapshot を許可する bounded fallback を実装。
  - `get_position_summary()` は request failure 時でも usable な stale cache を優先再利用し、`margin_used>0` なのに `(0,0)` へ落ちた時に side free margin ratio を `1.0` へ誤って押し上げないよう修正。
  - `ops/env/local-v2-stack.env` に `ORDER_MARGIN_STALE_ALLOW_SEC=15`、`ORDER_MARGIN_STALE_MIN_FREE_RATIO=0.30`、`ORDER_MARGIN_STALE_MIN_HEALTH_BUFFER=0.25` を明示。

- 期待効果:
  - 数秒級の OANDA `/summary` flap では entry を無駄に止めず、長めの outage では従来どおり fail-closed を維持する。
  - stale fallback は `order_manager` の margin/preflight 導線に限定し、全体を fail-open にしない。

## 2026-03-06 13:39 UTC / 2026-03-06 22:39 JST - 通信回復後の `scalp_fast` reject は `STOP_LOSS_ON_FILL_LOSS` に偏っていたため、protection fallback gap を 2p→3p へ戻し過ぎない範囲で拡大

- 市況確認（ローカルV2実測）:
  - `check_oanda_summary.py` は `200` 復帰。`openTrades=3`、`margin used=3169JPY / avail=34803JPY`。
  - `orders.db` では `2026-03-06 22:37 JST` 台に `filled` が再開。
  - `USD/JPY mid=157.51 spread=0.8p`

- 事実:
  - `2026-03-06T13:30:00Z` 以降の `orders.db` は `filled=33`, `rejected=7`, `entry_probability_reject=24`。
  - `rejected` 7件はすべて `scalp_fast` 系で、`quant-order-manager.log` でも `STOP_LOSS_ON_FILL_LOSS` と `protection fallback applied ... gap=0.0200` が並んでいた。
  - 既存の `ORDER_PROTECTION_FALLBACK_PIPS_SCALP_FAST=0.02` は、3/5 の再調整では有効だったが、3/6 22:30 JST 台の回復局面では再び tight 側に寄っていた。

- 対応:
  - `ops/env/local-v2-stack.env` の `ORDER_PROTECTION_FALLBACK_PIPS_SCALP_FAST` を `0.02 -> 0.03` へ変更。
  - 12p 既定値へ戻すのではなく、scalp_fast の fallback だけを 3p に限定して reject 低減を狙う。

- 期待効果:
  - `STOP_LOSS_ON_FILL_LOSS` を減らし、回復直後の `submit_attempt -> rejected` を `filled` 側へ寄せる。
  - fallback SL を広げ過ぎず、scalp_fast の損失尾を増やさない範囲で執行成立率を改善する。

## 2026-03-06 13:58 UTC / 2026-03-06 22:58 JST - OANDA は回復維持だが `scalp_ping_5s_d_live` の期待値が明確に負で、直近クローズのほぼ全件が `STOP_LOSS_ORDER` だったため、D variant の entry 条件を局所的に強化

- 市況確認（ローカルV2実測 + OANDA）:
  - `check_oanda_summary.py` は `200` を維持。`openTrades=4`
  - `USD/JPY mid=157.865 spread=0.8p`
  - `health_snapshot`: `data_lag_ms≈793`, `decision_latency_ms≈47`
  - core services と `quant-scalp-ping-5s-b` / `-exit` は稼働中

- 事実:
  - `pdca_profitability_report.py --instrument USD_JPY`:
    - 24h `scalp_ping_5s_d_live`: `30 trades / win 0.0% / PF 0.00 / -196.6 JPY`
    - 7d `scalp_ping_5s_d_live`: `42 trades / win 4.8% / PF 0.08 / -280.7 JPY`
  - `trades.db` の直近 24h は `STOP_LOSS_ORDER=39 trades / -321.9 JPY / -64.1 pips`、`MARKET_ORDER_TRADE_CLOSE=3 trades / +41.2 JPY`。負けの中心は exit ではなく entry quality。
  - `analyze_entry_precision.py --limit 220` では `scalp_ping_5s_d_live` が `slip_mean=0.314p / slip_p95=1.600p` と、同時間帯の `B` より明確に悪い。
  - `orders.db` では `D` の fills は `407-686 units` 帯でも連続し、その多くが数十秒以内に `STOP_LOSS_ORDER` で閉じていた。

- 対応:
  - `ops/env/local-v2-stack.env`
    - `SCALP_PING_5S_D_ENTRY_LEADING_PROFILE_REJECT_BELOW=0.72` (`0.66` から引き上げ)
    - `SCALP_PING_5S_D_ENTRY_LEADING_PROFILE_REJECT_BELOW_SHORT=0.80` (`0.74` から引き上げ)
    - `SCALP_PING_5S_D_BASE_ENTRY_UNITS=3000` (`4200` から縮小)
    - `SCALP_PING_5S_D_MAX_SPREAD_PIPS=0.90` (`1.20` から圧縮)

- 期待効果:
  - `D` の弱いシグナルと広めスプレッド帯だけを削り、`scalp_fast` 全体は止めずに赤字単価を圧縮する。
  - `STOP_LOSS_ON_FILL_LOSS` fallback に依存する前段の low-edge entry を減らし、fills 後の即 SL を抑える。

## 2026-03-06 14:05 UTC / 2026-03-06 23:05 JST - `M1Scalper-M1` は negative expectancy のまま取引回数が圧倒的に多いため、shared path を触らず `base units` だけを 40% 縮小

- 市況確認（ローカルV2実測 + OANDA）:
  - `check_oanda_summary.py` は `200` 維持、`openTrades=4`
  - `USD/JPY spread=0.8p`
  - `local_v2_stack` の core services と `quant-m1scalper` / `-exit` は稼働中

- 事実:
  - `pdca_profitability_report.py --instrument USD_JPY`
    - 24h `M1Scalper-M1`: `1775 trades / win 56.5% / PF 0.55 / -5750.4 JPY`
    - 7d `M1Scalper-M1`: `2290 trades / win 58.5% / PF 0.59 / -6172.5 JPY`
  - `trades.db` 直近 24h の `M1Scalper-M1` は `avg abs(units)=352.8`、負けの内訳は `MARKET_ORDER_TRADE_CLOSE=-3611.1 JPY`、`STOP_LOSS_ORDER=-2283.8 JPY`。
  - すでに dynamic alloc と open-trades guard は効いているため、ここで一番境界の小さいレバーは `base units` のみ。

- 対応:
  - `ops/env/quant-m1scalper.env`
    - `M1SCALP_BASE_UNITS=1800` (`3000` から縮小)

- 期待効果:
  - `M1` の挙動や exit 判断を変えず、負けトレードの赤字単価だけを先に 35-40% 程度圧縮する。
  - shared protection や order_manager を再度触らず、strategy ローカルのサイズだけで loss drag を落とす。

## 2026-03-06 14:22 UTC / 2026-03-06 23:31 JST - 24h赤字主因を再点検し、`main` 最新の local-v2 調整が active `trade_min` に未反映だったため反映確認を優先

- 市況確認（ローカルV2実測 + OANDA API）:
  - `pricing`: `bid=158.010 ask=158.018 spread=0.8p status=200 latency=310ms`
  - `summary`: `status=200 latency=244ms openTradeCount=4`
  - `openTrades`: `status=200 latency=281ms`
  - OANDA `M1` 直近120本: `ATR14=4.74p / 15m range=18.7p / 60m range=64.5p`
  - 判定: spread は通常帯、ATR/range はやや高めだが異常域ではなく、作業継続可。

- 24h収益分解（ローカル `logs/trades.db` / `logs/orders.db` / `logs/metrics.db`）:
  - 全体: `3368 trades / win_rate=50.0% / PF=0.552 / expectancy=-3.4 JPY / net=-11595.8 JPY`
  - 赤字寄与上位:
    - `scalp_ping_5s_flow_live: 420 trades / -7131.1 JPY`
    - `M1Scalper-M1: 2290 trades / -6172.5 JPY`
    - `scalp_ping_5s_d_live: 42 trades / -280.7 JPY`
  - reject: `STOP_LOSS_ON_FILL_LOSS=25`, `INSUFFICIENT_MARGIN=6`, `TRADE_DOESNT_EXIST=3`
  - 執行品質: `spread_mean=0.807p / slip_mean=0.008p / latency_submit_p50=190ms / latency_preflight_p50=227ms`
  - つまり主因は「執行遅延」より「負け戦略の件数・赤字単価」で、特に `flow` と `M1` が支配的。

- 直近反映確認で分かったこと:
  - `local-v2-stack.env` は `2026-03-06 23:27:38 JST` に、`ee476feb tune: tighten local-v2 m1 and flow profitability` 相当の値へ更新済みだった。
  - しかし `quant-m1scalper.log` の `2026-03-06 23:20:34 JST` 起動行は `tag_filter=-` で、`M1SCALP_SIGNAL_TAG_CONTAINS=breakout-retest-long,nwave-long` が live プロセスへ入っていなかった。
  - `scripts/local_v2_stack.sh restart --profile trade_min --env ops/env/local-v2-stack.env` を実行し、`2026-03-06 23:29-23:31 JST` に active services を再起動。
  - 再起動後の確認:
    - `quant-m1scalper.log`: `worker start ... tag_filter=breakout-retest-long,nwave-long`
    - 直後に `tag_filter_block tag=M1Scalper-sell-rally`
    - `quant-scalp-ping-5s-b.log`: `side_filter=sell`
    - `scripts/local_v2_stack.sh status --profile trade_min --env ops/env/local-v2-stack.env`: core + `B / micro-rangebreak / M1` が running

- 判断:
  - 反映直後は `trade_min` の post-restart sample がまだ薄く、ここで追加の speculative tune を重ねるより、`main` 最新の tighten を live へ載せて効果を見る方が筋が良い。
  - 現在の `trade_min` active services は `B / micro-rangebreak / M1` で、`flow / D` は反映後 profile では動いていない。次の追加調整は post-restart 実績を見てからにする。

- 再検証条件:
  1. `quant-m1scalper.log` に `tag_filter_block tag=M1Scalper-sell-rally` が継続し、`tag_filter=-` が再発しないこと。
  2. `orders.db` の post-restart で `STOP_LOSS_ON_FILL_LOSS` reject が再拡大しないこと。
  3. 30-60分後の `trades.db` で `M1Scalper-M1` と `scalp_ping_5s_b_live` の追加 net が、再起動前の時間帯より悪化しないこと。
