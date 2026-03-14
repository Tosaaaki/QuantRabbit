# QuantRabbit 開発履歴 日次 Annex

- この annex は [REPO_HISTORY_MINUTES.md](./REPO_HISTORY_MINUTES.md) の詳細版で、active date ごとの raw ledger です。
- 各行の `主な領域` は当日の変更ファイルの top-level path 上位 3 件です。
- `代表変更` は merge commit を除いた先頭 2 件を並べ、残件数は `+N more` で表記しています。

## 2025-06

| Date | Commits | 主な領域 | 代表変更 |
| --- | ---: | --- | --- |
| 2025-06-21 | 1 | .gitignore, README.md | Initial commit |
| 2025-06-23 | 14 | analysis, infra, market_data | Initial import at repo root / Add initial requirements for core, Google Cloud SDKs, HTTP/RSS, and development tools / +11 more |
| 2025-06-26 | 4 | cloudrun, Dockerfile, README.md | feat: add Dockerfile and news summarizer script for Cloud Run integration / feat: add fetch_news_runner.py and requirements.txt for news fetching functionality / +1 more |
| 2025-06-30 | 1 | cloudrun, startup-trader.sh | feat: update Dockerfile and fetch_news_runner.py for news fetching functionality; add startup-trader.sh |

## 2025-07

| Date | Commits | 主な領域 | 代表変更 |
| --- | ---: | --- | --- |
| 2025-07-02 | 3 | market_data, analysis, execution | feat: Enhance trading system with new strategies and improvements / Refactor: Remove OpenAI API key from config files and add to .gitignore / +1 more |
| 2025-07-13 | 7 | infra, analysis, execution | feat: Integrate Google Cloud Secret Manager for sensitive configurations / Ignore .terraform directory / +3 more |
| 2025-07-14 | 16 | analysis, strategies, .gitignore | Clean up macOS artifacts and unused script / style: format code and add lint CI / +6 more |
| 2025-07-15 | 15 | analysis, market_data, strategies | fix: import pandas_ta / Add mock tick stream fallback / +7 more |
| 2025-07-16 | 1 | market_data, indicators | fix: truncate OANDA API timestamps to microseconds for historical candle fetch |
| 2025-07-26 | 1 | analysis, cloudrun, strategies | feat: Fix news summarization and trade program issues, and add Dockerfile for news fetching |

## 2025-08

| Date | Commits | 主な領域 | 代表変更 |
| --- | ---: | --- | --- |
| 2025-08-03 | 1 | analysis, infra, execution | fix(news): ニュースパイプラインの修復とリファクタリング |
| 2025-08-08 | 2 | cloudrun, infra, Dockerfile.news_summarizer_runner | fix: update Dockerfile and fetch_news_runner.py for improved functionality and error handling |
| 2025-08-21 | 3 | cloudrun, Dockerfile.news_summarizer_runner, infra | Fix: Resolve news pipeline issues and remove VM / ニュース要約でイベント時刻とインパクトを保持 |

## 2025-10

| Date | Commits | 主な領域 | 代表変更 |
| --- | ---: | --- | --- |
| 2025-10-22 | 14 | execution, scripts, analytics | chore: 現状のシステム状態を GitHub main に反映 / chore: ローカル変更を同期 (git update) / +12 more |
| 2025-10-23 | 31 | scripts, templates, execution | risk: default per-trade risk to 1% via env; persist stage state across restarts; document risk_pct in env.example / Boost lot allocation for macro/micro stages / +29 more |
| 2025-10-24 | 34 | execution, main.py, analysis | Add heuristic fallback for GPT decider / Fix timezone handling for open trades / +32 more |
| 2025-10-25 | 26 | main.py, advisors, analysis | Enable hedging aware entry gating / Increase GPT decision timeout / +24 more |
| 2025-10-27 | 6 | advisors, execution, strategies | Fix stage planner to use GPT weights / Disable drawdown guard by default / +4 more |
| 2025-10-29 | 9 | execution, scripts, analysis | マクロ配分の下限とロットブーストを導入 / chore: ignore local cache and vm artifacts / +7 more |
| 2025-10-30 | 35 | workers, analysis, main.py | Adjust macro staging and reinforce pocket allocations / Add tick buffer and scalp timeout exits / +33 more |
| 2025-10-31 | 16 | workers, main.py, execution | refactor: switch GPT decider to chat completions / chore: reduce news payload for GPT / +13 more |

## 2025-11

| Date | Commits | 主な領域 | 代表変更 |
| --- | ---: | --- | --- |
| 2025-11-01 | 1 | workers, main.py | Rename manual spike worker to mirror spike |
| 2025-11-02 | 3 | .gcloud, workers, scripts | chore: sync latest changes / feat: wire macro state gating into main loop / +1 more |
| 2025-11-03 | 5 | workers, scripts, execution | [ops] Cap macro allocation at 30%: hard cap for lot (GLOBAL_MACRO_WEIGHT_CAP=0.30) and DD pocket cap (POCKET_MAX_RATIOS.macro=0.3); doc updated / [hotfix] Fix NameError: init GLOBAL_MACRO_WEIGHT_CAP after helper funcs; keep default 0.30 and env override / +3 more |
| 2025-11-04 | 11 | workers, scripts, docs | [Task:T-20251103-001] Add Trend H1 worker and OANDA fetch script / Limit pullback S5 trading hours and align replay / +9 more |
| 2025-11-05 | 5 | workers, main.py, scripts | Allow disabling drawdown guard via env / Enable worker loops and adjust unit sizing / +3 more |
| 2025-11-06 | 12 | workers, scripts, execution | mirror-s5: tighten SELL gating; disable dynamic SL/TP widening for scalp mirror-s5; cap scalp dynamic SL [ops: deploy] / feat(macro): H4-led entries with MTF confluence; pattern-aware macro exits (retest + structure kill-line); enforce min 1k units and cap new entries at 10k units / +10 more |
| 2025-11-07 | 7 | workers, execution, analysis | [Task:T-20251105-003][Task:T-20251105-004] Macro/Scalp plan executors: shared helpers (order_ids, stage_rules, managed_positions, pocket_limits), PocketPlan/PlanCursor, macro/scalp core workers + main delegation gating; default to delegate+enable for test mode / [Task:T-20251105-003] plan_bus: optional publish logging via PLAN_BUS_LOG=1 to verify plan flow on VM / +5 more |
| 2025-11-10 | 4 | workers, main.py, execution | fix(workers): sync market_order return handling / fix(core): derive SL/TP from latest quotes / +2 more |
| 2025-11-11 | 7 | workers, docs, main.py | chore(tasks): add macro snapshot + entry coverage task / fix(macro+sync): auto-refresh snapshot and allow BQ-less pipeline / +5 more |
| 2025-11-18 | 5 | strategies, workers, execution | [Task:T-20251117-002][Task:T-20251117-003] Micro pocket adaptive gating & logging / [fix] guard manual/hold blockers / +3 more |
| 2025-11-20 | 1 | execution, main.py | Improve macro exits and volatility guards |
| 2025-11-21 | 7 | main.py, strategies, workers | Make pullback scalp ATR-driven / Boost margin usage and allow concurrent strategies / +5 more |
| 2025-11-24 | 6 | workers, execution, docs | [Task:T-20251123-001] Add surge guard tuning and replay helpers / [Task:T-20251123-001] Make entry protections spread-aware / +4 more |
| 2025-11-25 | 17 | workers, main.py, execution | [Task:T-20251123-001] Unstick entries with stage reset and API fallback / Relax spread guard defaults to allow entries / +15 more |
| 2025-11-26 | 11 | workers, execution, main.py | Enable scalp exits + loosen profit snatch thresholds / Disable manual sentinel blocking and auto-release / +9 more |
| 2025-11-27 | 23 | execution, analysis, main.py | Use Secret Manager for OpenAI key fallback / Add idle self-refresh on trade inactivity / +20 more |
| 2025-11-28 | 14 | main.py, workers, execution | [Hotfix] StageTracker cluster cooldowns and recent profiles / [Hotfix] Fix factor warmup NameError and timezone decay / +12 more |
| 2025-11-30 | 1 | workers, execution, main.py | Stabilize worker lifecycle and SL control |

## 2025-12

| Date | Commits | 主な領域 | 代表変更 |
| --- | ---: | --- | --- |
| 2025-12-02 | 2 | workers, analysis, execution | Improve momentum burst gating and allow cooldown disable / Disable news strategies and harden fast scalp ATR |
| 2025-12-03 | 2 | strategies, analysis, execution | Relax scalp/macro gating and lower scalp lot floor / Tighten GPT decider and low-vol handling |
| 2025-12-09 | 13 | strategies, main.py, execution | Adjust scalp ATR and range gates for low-vol sessions / [Task:T-20251123-001][Task:T-20251208-001] exit/alloc tuning and log maintenance / +11 more |
| 2025-12-10 | 8 | scripts, analysis, execution | Enable fast/pullback_runner scalps and add orphan guard for micro / Add level-map pipeline and optional lookup in main / +6 more |
| 2025-12-11 | 3 | execution, AGENTS.md, analysis | Add directional exposure guard and virtual SLTP logging / Fix partial reduction scaling helper / +1 more |
| 2025-12-12 | 48 | workers, execution, main.py | Guard M1Scalper against strong trend and fix cooldown datetime bug / Make value-cut stricter to avoid premature exits / +46 more |
| 2025-12-13 | 19 | execution, main.py, strategies | Add ichimoku cloud and cluster distances / Apply tech overlays across strategies / +17 more |
| 2025-12-14 | 2 | execution, main.py, utils | tighten margin cap enforcement and fresh snapshots / relax pulsebreak vol threshold in low ATR |
| 2025-12-15 | 18 | main.py, execution, workers | Fix mirror spike workers entrypoint / Log account/exposure metrics for hazard tuning / +16 more |
| 2025-12-16 | 11 | main.py, workers, execution | Clamp L3 reduce-only and macro hedge tighten / Fix reduce_only assignment crash / +9 more |
| 2025-12-17 | 1 | analysis, execution, strategies | Refine GPT bias flow and risk guards |
| 2025-12-18 | 3 | analysis, workers, execution | Make GPT decider fall back to local ranking and relax exits/margin / Use GPT only for biases and broaden local strategy ranking / +1 more |
| 2025-12-19 | 8 | main.py, execution, analysis | Tune GPT bias parsing, relax spread guard and exposure caps / Set min cap floor in risk_guard to avoid over-shrinking / +6 more |
| 2025-12-20 | 1 | main.py | Increase risk caps to use more margin |
| 2025-12-22 | 12 | main.py, execution, systemd | Raise minimum order units to ensure meaningful size / Add dynamic allocation loader and scoring worker / +10 more |
| 2025-12-23 | 24 | workers, systemd, execution | Fix MAX_MARGIN_USAGE import to prevent logic loop crash / Loosen macro loss cap and fix ATR default in range guard / +22 more |
| 2025-12-24 | 11 | workers, TODO_EXIT_ALIGNMENT.md, execution | Align exits with entries: disable auto exit, add H1Momentum structure-based exit / Align TrendMA exits with entries: structure break + meta-aware thresholds / +9 more |
| 2025-12-25 | 3 | TODO_EXIT_ALIGNMENT.md, systemd, workers | Add dedicated exit worker for M1Scalper / Document full technical list and pattern coverage ToDo / +1 more |
| 2025-12-26 | 31 | workers, systemd, main.py | Add per-strategy tech scoring and exit TP scaling / Fix vwap magnet worker loss cooldown and vwap gap / +29 more |
| 2025-12-27 | 2 | workers, systemd, docs | Remove core workers and add per-strategy exits / Add exit services for single micro strategies |
| 2025-12-29 | 55 | workers, execution, strategies | chore: sync local changes / chore: sync exit manager / +53 more |
| 2025-12-30 | 18 | workers, main.py, execution | Run logic_loop and GPT worker in main / Prioritize macro/micro signals and cap scalp slots / +16 more |
| 2025-12-31 | 21 | main.py, strategies, execution | tune: relax range guard thresholds / tune: use net margin estimate for usage guard / +19 more |

## 2026-01

| Date | Commits | 主な領域 | 代表変更 |
| --- | ---: | --- | --- |
| 2026-01-01 | 6 | main.py, execution, tests | Lower scalp absolute lot floor to 0.001 / Drop scalp absolute lot floor to 0.0005 / +4 more |
| 2026-01-02 | 16 | main.py, strategies, tests | Feed spread monitor with latest ticks / Loosen low-vol gates for longs and scalps / +14 more |
| 2026-01-03 | 1 | main.py, strategies | Loosen entry waits and raise stage floors |
| 2026-01-05 | 8 | strategies, main.py, workers | Fix entry plan logging without sl/tp price / Pass strategy_tag to market_order to avoid reject / +6 more |
| 2026-01-06 | 2 | execution, main.py | Scale down units when hitting margin cap / Add GPT_DISABLED flag and skip GPT evaluation when set |
| 2026-01-07 | 2 | scripts | Add log maintenance script for rotation/archive / Add optional GCS upload in log maintenance |
| 2026-01-08 | 1 | workers, execution, analysis | Add MR guard tracking and soft TP exits |
| 2026-01-09 | 11 | workers, main.py, execution | Expand micro exit coverage for MR overlays / Add M1Scalper max-hold and adverse exits / +9 more |
| 2026-01-12 | 8 | workers, execution, strategies | Add section-axis exits across workers / Add orders.db WAL checkpoint guard / +6 more |
| 2026-01-13 | 15 | workers, execution, analysis | Add technique-based entry/exit gating and optimize GPT payload / Tune strategy-specific technique policies / +13 more |
| 2026-01-14 | 31 | main.py, workers, execution | Soften entry guard with multi-factor checks / Loosen high-zone countertrend gating / +29 more |
| 2026-01-15 | 4 | workers, execution, analysis | Use technical override for short entry guard and enrich exit MTF / Gate Donchian trend-failure exits with technical signal / +2 more |
| 2026-01-16 | 1 | config, execution | Enable MTF entry guard confirmation |
| 2026-01-17 | 4 | config, execution, analytics | Relax micro MTF chart gate / Add entry guard gates and reports / +2 more |
| 2026-01-18 | 10 | workers, docs, execution | Add reentry open-stack limits / Allow distance-based reentry override / +8 more |
| 2026-01-19 | 6 | workers, scripts, systemd | Fix TechniquePolicy slots for exit overrides / Log exit reasons on trade closes / +4 more |
| 2026-01-20 | 15 | workers, execution, systemd | Tune range entries and min RR guard / Require fresh realtime metrics and add BQ job services / +13 more |
| 2026-01-21 | 36 | scripts, AGENTS.md, systemd | Relax scalp entry gating and reentry blocks / Add stream idle and data lag watchdogs / +34 more |
| 2026-01-22 | 29 | workers, systemd, docs | Migrate qr units to quant services / Relax reentry distance and onepip latency gate / +27 more |
| 2026-01-23 | 70 | config, workers, main.py | Add profit giveback guard for scalp entries / Relax scalp tech/reentry gates / +68 more |
| 2026-01-24 | 40 | config, workers, scripts | Fix ops policy JSON extraction / Hoist misplaced ops policy fields / +38 more |
| 2026-01-25 | 20 | workers, systemd, apps | Add VM snapshot API for Cloud Run / Lazy import PositionManager for Cloud Run / +18 more |
| 2026-01-26 | 8 | workers, systemd, strategies | Include open positions in full lite snapshot / Add high-volatility confidence guards for macro / +6 more |
| 2026-01-27 | 17 | workers, execution, config | Disable policy gating on VM and raise sizing caps / Add reentry caps, adverse drift cap, and strategy-aware risk scaling / +15 more |
| 2026-01-28 | 14 | workers, config, strategies | Harden exits and boost range winners / Allow mtf_reversal negative exits / +12 more |
| 2026-01-29 | 13 | workers, systemd, config | Add volatility spike rider worker / Load /etc/quantrabbit.env for vol spike rider services / +11 more |
| 2026-01-30 | 52 | workers, templates, execution | Widen mirror_spike_tight SL using real spread and hard_stop / Regime-aware perf guard with safer caching / +50 more |
| 2026-01-31 | 17 | workers, systemd, execution | Remove LLM dependencies and keep local-only decisions / Remove legacy LLM references from AGENTS / +15 more |

## 2026-02

| Date | Commits | 主な領域 | 代表変更 |
| --- | ---: | --- | --- |
| 2026-02-01 | 6 | config, workers, scripts | Add daily maintenance window and stabilize tick stream / Increase scalp cadence and sizing defaults / +4 more |
| 2026-02-02 | 3 | systemd, workers, scripts | Enhance scalp precision sizing and replay controls / Load env for scalp precision services / +1 more |
| 2026-02-03 | 16 | workers, scripts, systemd | Select freshest UI snapshot / Smooth dashboard refresh / +14 more |
| 2026-02-04 | 1 | execution, workers, config | Add per-pocket hard SL and loss-cut for new scalp trades |
| 2026-02-05 | 19 | workers, systemd, scripts | Fix UI snapshot publish fallback / stabilize: disable fast_scalp and enable perf guard / +17 more |
| 2026-02-06 | 20 | .gcloud, systemd, workers | Enable pro_stop in exit workers and add ATR tuning / Add TickImbalance partial take and BE protection / +17 more |
| 2026-02-10 | 1 | analysis, requirements.txt, scripts | forecast: add sklearn multi-horizon bundle + trainer |
| 2026-02-11 | 92 | workers, systemd, tests | Sync main with VM branch (reentry-wrb-both) / order: add entry-quality gates (regime + microstructure) / +88 more |
| 2026-02-12 | 70 | workers, tests, ops | order: apply entry loss-cap sizing for new entries only / scalp_ping_5s: enforce new-entry-only time stop / +68 more |
| 2026-02-13 | 58 | workers, ops, systemd | Fix retry on LOSING_TAKE_PROFIT / Add scalp_ping_5s MAX env overrides / +56 more |
| 2026-02-14 | 32 | systemd, docs, workers | fix(order): allow stopLossOnFill when pocket SL enabled / fix(execution): disable 5s hard stop by strategy tag / +30 more |
| 2026-02-15 | 15 | workers, docs, execution | feat(execution): route strategy entries through pre-coordination wrapper / fix: remove order manager intent coordination blackboard / +13 more |
| 2026-02-16 | 58 | workers, ops, docs | fix(scalp-5s): add resilient open-position fetch and open-market retry logs / fix(scalp-5s): tolerate missing BLOCK_HOURS_JST in config / +56 more |
| 2026-02-17 | 67 | workers, docs, ops | fix(env): relax scalp 5s b entry guard thresholds / chore: tune m1scalper entry and env guard settings / +65 more |
| 2026-02-18 | 53 | workers, docs, ops | feat(forecast): add tf-specific breakout adaptive weight map / fix(ops): harden scalp_fast against margin closeout / +51 more |
| 2026-02-19 | 21 | docs, workers, ops | fix: restore scalp_ping_5s_b stop-loss enforcement / fix: enforce scalp ping b protected entry baseline / +19 more |
| 2026-02-20 | 28 | docs, scripts, ops | scalp_ping_5s_b: speed up direction flip and rebalance sizing / scalp_ping_5s_b: add side-metrics direction flip / +26 more |
| 2026-02-21 | 19 | docs, ops, systemd | Tune reentry hour blocks for ping5s-b and M1Scalper / Extend M1Scalper hour blocks for loss windows / +17 more |
| 2026-02-22 | 12 | docs, workers, ops | tune(forecast): adopt candD 5m10m acceleration profile / tune(forecast): refine candD with 1m mae boost / +10 more |
| 2026-02-23 | 2 | docs, ops | tune forecast dynamic meta rnd056 / tune forecast dynamic meta rnd087 |
| 2026-02-24 | 32 | docs, ops, tests | tune: switch ping5s-d replay gate to 3x2 WFO / tune: harden dynamic allocation for polluted strategy tags / +30 more |
| 2026-02-25 | 49 | docs, ops, workers | tune: relax ping5s-b hard block and speed hourly reduce / feat: narrow scalp ping d to high-yield session / +47 more |
| 2026-02-26 | 101 | docs, ops, tests | feat(forecast): add hourly improvement audit worker / ops: make ping5s-d canary apply and restart automatically / +99 more |
| 2026-02-27 | 80 | docs, ops, workers | Pin scalp ping b/c side filters back to sell / tune: de-risk scalp ping b with tighter entry floors / +78 more |
| 2026-02-28 | 28 | docs, ops, execution | tune(ping-c): relax failfast block thresholds round13 / fix: prevent scalp extrema reversal stranded losses / +26 more |

## 2026-03

| Date | Commits | 主な領域 | 代表変更 |
| --- | ---: | --- | --- |
| 2026-03-01 | 11 | docs, scripts, ops | chore(ops): retune scalp b/c gates and enforce v2 runtime route / docs: require multi-agent task execution in AGENTS / +9 more |
| 2026-03-02 | 6 | ops, docs, AGENTS.md | chore: relax scalp_ping_5s_b entry gates to unblock live entries / docs: require market check before execution / +4 more |
| 2026-03-03 | 12 | ops, docs, scripts | fix(order): unblock entries while preserving manual position / fix(order): lower min-unit floors under manual exposure / +10 more |
| 2026-03-04 | 14 | docs, workers, scripts | tune(all): comprehensive strategy audit - tighten risk, fix bugs, unify exits / tune scalp_ping_5s_b entry quality and risk throttles / +12 more |
| 2026-03-05 | 61 | workers, docs, ops | ops: add resilient local-v2 autorecover via launchd / feat(local): harden auto-recovery and tighten scalp_ping_5s_b gating / +59 more |
| 2026-03-06 | 30 | docs, ops, workers | fix(exit): allow ping5s derisk/reentry closes / docs: record ping5s close reject fix / +28 more |
| 2026-03-07 | 46 | docs, scripts, tests | docs: align trade_min profitability notes / tune: relax micro level reactor gate / +44 more |
| 2026-03-09 | 61 | docs, tests, ops | fix: tighten local reversal guards / fix: harden local loser entry paths / +59 more |
| 2026-03-10 | 45 | docs, tests, ops | fix(micro): tighten momentum and level loser clusters / fix: tighten micro loser cluster guards / +43 more |
| 2026-03-11 | 35 | docs, tests, workers | docs: formalize trade findings change diary / fix: restore boosted lane feedback coverage / +33 more |
| 2026-03-12 | 53 | docs, tests, workers | fix(feedback): repair local strategy coverage / fix(trading): block shallow extrema reversal shorts / +51 more |
| 2026-03-13 | 40 | docs, tests, workers | fix: tighten extrema reversal short setup-pressure guard / fix(local-v2): tighten precision lowvol and extrema guards / +38 more |
| 2026-03-14 | 4 | docs, scripts, AGENTS.md | fix(trading): tighten momentumburst transition long chase / feat(ops): add anti-loop improvement guardrails / +2 more |
