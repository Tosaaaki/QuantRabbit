# OPS Local Runbook

## ローカル運用方針（現行）
- 本番運用はローカルV2導線（`scripts/local_v2_stack.sh`）のみ。
- VM/GCP/Cloud Run は現行運用に存在しない前提で、関連コマンドは実行対象外。
- 旧VM/GCP資料は履歴アーカイブとしてのみ保持する。


本書は「ローカル開発/ローカル実売買運用」に限定した手順書。  
現行運用は VM 非依存（ローカル専用）を前提とする。

ハブ自動投稿の監視・起動・単発投稿・停止の詳細手順は [AGENT_COLLAB_HUB.md](./AGENT_COLLAB_HUB.md) の「運用手順」を参照する。

## 0. 運用原則
- 現行はローカルPDCAを優先する。
- ローカル検証でも作業前にUSD/JPYの市況（価格/スプレッド/ATR/API応答）を確認する。
- VM の稼働監視/操作はデフォルトで行わない（`QR_LOCAL_ONLY=1`）。
- ローカル運用タスクでは `scripts/vm.sh` / `deploy_to_vm.sh` / `gcloud compute *` を実行しない。
- ローカル実売買は自己責任で行い、手動ポジションへの干渉条件を常に確認する。
- local-v2 の仕組みを追加・削除したタスクでは、`docs/CURRENT_MECHANISMS.md` も同じ変更で更新する。
  - 削除・停止した仕組みは単純削除せず、同ファイルの `Archive` セクションへ日付と理由つきで移す。

## 1. ローカルV2スタック（worker群）
- 制御スクリプト: `scripts/local_v2_stack.sh`
- 既定上書きenv: `ops/env/local-v2-stack.env`
- メモリ圧迫時の既定:
  - `ORDER_MANAGER_SERVICE_WORKERS=1`（uvicorn多重worker抑制）
  - `OMP/OPENBLAS/MKL/NUMEXPR/VECLIB/BLIS` の各スレッドを `1` に固定

### 排他運用（必須）
- `local_v2_stack.sh` と `local_vm_parity_stack.sh` は排他運用とする（同時運転禁止）。
- `qr-local-parity` screen セッション、または `scripts/local_vm_parity_supervisor.py` が稼働中の場合、
  `local_v2_stack.sh` の `up/down/restart` は既定で拒否される。
- parity 停止は先に `scripts/local_vm_parity_stack.sh stop` を実行する。
- `status/logs` は parity 稼働中でも実行可能。
- 競合を理解した上で意図的に実行する場合のみ `--force-conflict` を付ける。
- `--force-conflict` で parity 併走を行う場合、`local_v2_stack.sh` は conflict-safe mode となり、
  モジュール名ベースの一括 kill/cleanup を行わない（誤って parity 側 worker を落とさないため）。

### parity 併走の sidecar ポート（必要時のみ）
- プロファイル: `ops/env/local-v2-sidecar-ports.env`
  - `ORDER_MANAGER_SERVICE_PORT=18300`
  - `POSITION_MANAGER_SERVICE_PORT=18301`
  - `ORDER_MANAGER_SERVICE_URL=http://127.0.0.1:18300`
  - `POSITION_MANAGER_SERVICE_URL=http://127.0.0.1:18301`
- 例（parity 稼働中に API 系だけ sidecar で上げる）:
```bash
scripts/local_v2_stack.sh up --services "quant-order-manager,quant-position-manager" \
  --env ops/env/local-v2-sidecar-ports.env --force-conflict
```
- 停止:
```bash
scripts/local_v2_stack.sh down --services "quant-order-manager,quant-position-manager" \
  --env ops/env/local-v2-sidecar-ports.env --force-conflict
```
- 確認:
```bash
lsof -nP -iTCP:18300 -sTCP:LISTEN
lsof -nP -iTCP:18301 -sTCP:LISTEN
```

### 起動
```bash
scripts/local_v2_stack.sh up --profile trade_min --env ops/env/local-v2-stack.env
```

### 状態確認
```bash
scripts/local_v2_stack.sh status --profile trade_min --env ops/env/local-v2-stack.env
```

### ヘルススナップショット確認
```bash
scripts/collect_local_health.sh
```

- `logs/health_snapshot.json` の `mechanism_integrity.ok` を見る。
- `mechanism_integrity.missing_mechanisms` が空でない場合は、
  `strategy_feedback`, `dynamic_alloc`, `pattern_book`, `forecast_runtime/service`,
  `blackboard(entry_intent_board)` のどこが欠けたかを優先確認する。
- `collect_local_health.sh` は snapshot 鮮度に加えて
  `mechanism_integrity=yes|no missing=...` を標準出力へ出す。

### ログ確認
```bash
scripts/local_v2_stack.sh logs --service quant-order-manager --tail 200 --follow
```

### 停止
```bash
scripts/local_v2_stack.sh down --profile trade_min --env ops/env/local-v2-stack.env
```

### ログ/ PID の保存場所
- ログ: `logs/local_v2_stack/*.log`
- PID: `logs/local_v2_stack/pids/*.pid`
- `local_v2_stack.sh` は worker を親シェルと別セッションで起動する。
  - `scalp_ping_5s_*` clone wrapper のような thin launcher でも、`up/restart` コマンド終了時に巻き込まれず継続稼働することを前提とする。

### 自動復帰（推奨）
ローカルPCの再起動・ログイン後・ネット復帰後に `local_v2_stack` を自動で回復させるには、
macOS `launchd` の LaunchAgent を使う。

まず手元で watchdog を常駐起動（launchd を使わない運用）:
```bash
scripts/local_v2_stack.sh watchdog --daemon \
  --profile trade_min \
  --env ops/env/local-v2-stack.env \
  --interval-sec 10
```

状態確認 / 停止:
```bash
scripts/local_v2_stack.sh watchdog-status
scripts/local_v2_stack.sh watchdog-stop
```

インストール:
```bash
scripts/install_local_v2_launchd.sh \
  --profile trade_min \
  --env ops/env/local-v2-stack.env \
  --interval-sec 10 \
  --resume-gap-sec 90
```

状態確認:
```bash
scripts/status_local_v2_launchd.sh
```

アンインストール:
```bash
scripts/uninstall_local_v2_launchd.sh
```

補足:
- 自動復帰本体: `scripts/local_v2_autorecover_once.sh`
- watchdog ループ本体: `scripts/local_v2_watchdog.sh`
- 監視ログ: `logs/local_v2_autorecover.log`
- watchdog daemon ログ: `logs/local_v2_stack/watchdog.log`
- network down→up 復帰時は `quant-market-data-feed` を自動再起動（既定ON）して再接続を強制する。
- `local_vm_parity` 競合時は既存ガードに従って自動復帰をスキップする。
- `launchd` は `~/Documents` 配下の実ファイル読み取りで `Operation not permitted` になる場合がある。
  現行はリポジトリ実体を `/Users/tossaki/App/QuantRabbit` に置き、
  `/Users/tossaki/Documents/App/QuantRabbit` は互換用シンボリックリンクとして運用する。
- `scripts/local_v2_stack.sh` / `scripts/local_v2_autorecover_once.sh` /
  `scripts/install_local_v2_launchd.sh` は repo root を物理パスへ正規化して扱う。
  `scripts/status_local_v2_launchd.sh` が symlink plist を警告した場合は、
  `scripts/install_local_v2_launchd.sh --profile trade_min --env ops/env/local-v2-stack.env`
  を再実行して plist を再生成する。
- `local_v2_autorecover_once.sh` はロック異常終了時の stale lock を自動除去して再開し、sleep/wake 相当のポーリングギャップと network down→up をログ記録する。
- `local_v2_autorecover_once.sh` は健全時/復旧時に `scripts/run_local_feedback_cycle.py` を非同期起動し、
  `dynamic_alloc / pattern_book / trade_counterfactual / replay_quality_gate / trade_findings_draft` を
  ローカルでも interval 管理付きで再計算する（既定ON）。
  `strategy_feedback` は `local_v2_stack` 管理サービス側で loop 更新するため、
  cycle 側は既定OFF。必要時だけ `LOCAL_FEEDBACK_CYCLE_STRATEGY_FEEDBACK_ENABLED=1`
  で明示有効化する。
  `replay_quality_gate` は market-open 中を既定skipしつつ、
  closed帯で `trade_counterfactual -> worker_reentry.yaml` の auto-improve を反映する。
  replay入力が不足する closed帯では soft-skip として扱い、
  `local_feedback_cycle_latest.json` 全体を false positive の `error` にしない。
  - 全体ON/OFF: `QR_LOCAL_V2_FEEDBACK_CYCLE_ENABLED=1|0`
  - 各jobは `LOCAL_FEEDBACK_CYCLE_<JOB>_{ENABLED,INTERVAL_SEC,TIMEOUT_SEC,CMD,ENV_FILES,OUTPUTS}` で上書きできる。
  - `participation_allocator` job の override key は
    `LOCAL_FEEDBACK_CYCLE_PARTICIPATION_ALLOCATOR_*` を正とする。
    `CMD` に空白を含む場合は env 値全体を必ずクォートする。
  - 最新実行結果は `logs/local_feedback_cycle_latest.json` と
    `logs/local_feedback_cycle_history.jsonl`、個別stdout/stderrは
    `logs/local_feedback_cycle/*.log` を正とする。
  - `trade_findings_draft` は `logs/trade_findings_draft_latest.json` /
    `logs/trade_findings_draft_latest.md` /
    `logs/trade_findings_draft_history.jsonl` を更新し、
    review-only draft を生成する。`docs/TRADE_FINDINGS.md` へは自動追記せず、
    反映はレビュー後に手動で行う。whiteboard 通知は
    `TRADE_FINDINGS_DRAFT_WHITEBOARD_ENABLED=1` のときだけ有効で、
    同一 fingerprint の draft では `logs/agent_whiteboard.db` へ重複通知しない。
- `local_v2_autorecover_once.sh` は健全時/復旧時に `scripts/run_brain_autopdca_cycle.sh` を非同期起動する（既定ON）。
  - `QR_LOCAL_V2_BRAIN_AUTOPDCA_ENABLED=1|0` で有効/無効。
  - `QR_LOCAL_V2_BRAIN_AUTOPDCA_INTERVAL_SEC`（互換: `QR_LOCAL_V2_BRAIN_PDCA_INTERVAL_SEC`）で実行間隔を指定。
- `run_brain_autopdca_cycle.sh` は `--interval-sec` とロックで多重実行を抑止し、`env_changed=true` の時だけ `quant-order-manager,quant-strategy-control` を再起動する。
  出力は `logs/brain_autopdca_cycle_latest.json` / `logs/brain_autopdca_cycle_history.jsonl`。

## 1.1 ローカルVMパリティスタック（V2 + 予測/分析/黒板）
- 制御スクリプト: `scripts/local_vm_parity_stack.sh`
- 既定上書きenv: `ops/env/local-v2-full.env`
- supervisor: `scripts/local_vm_parity_supervisor.py`

### 起動
```bash
scripts/local_vm_parity_stack.sh start
```

### 状態確認
```bash
scripts/local_vm_parity_stack.sh status
```

### ログ確認
```bash
scripts/local_vm_parity_stack.sh logs
scripts/local_vm_parity_stack.sh logs quant-order-manager
scripts/local_vm_parity_stack.sh logs status
```

### 停止
```bash
scripts/local_vm_parity_stack.sh stop
```

### 補足
- `start` 時に競合プロセス（`qr-local-v2` / `qr-local-lane` / 旧 supervisor）を停止する。
- `codex_long_autotrade.py`（QuantRabbitLocalLane）の残留プロセスも停止して多重ローカル売買を防ぐ。
- 8300/8301/8302 の LISTEN 残留プロセスは自動掃除してから起動する。
- parity 稼働中に `local_v2_stack.sh` を使う必要がある場合は、先に `scripts/local_vm_parity_stack.sh stop` で parity を停止する。
- `quant-bq-sync` はローカル運用向けに `--disable-bq --disable-gcs` で動作する。
- `run_health_snapshot` は `HEALTH_UPLOAD_DISABLE=1` でローカル保存のみ実行する。
- `local_v2_stack.sh` は `--env` の上書き値を最後に適用するため、`quant-*.env` 側の既定値（例: `ORDER_MANAGER_SERVICE_WORKERS=6`）をローカル用に安全に差し替えできる。

## 2. ローカルLLMレーン（実売買）
- レポジトリ: `/Users/tossaki/Documents/App/QuantRabbitLocalLane`
- 実行スクリプト: `/Users/tossaki/Documents/App/QuantRabbitLocalLane/bot/codex_long_autotrade.py`
- 実行ログ: `/Users/tossaki/Documents/App/QuantRabbitLocalLane/state/runner_stdout.log`

### 稼働確認
```bash
pgrep -af 'codex_long_autotrade.py'
```

### screen セッション確認
```bash
screen -ls | rg qr-local-lane
```

## UIダッシュボード（コマンドなし確認）
- 実装: `apps/runtime_ui.py`
- URL: `http://127.0.0.1:8092`
- 機能:
  - ローカル専用モード表示（`QR_LOCAL_ONLY=1`）
  - Local parity（main）状態
  - Local V2 worker群状態
  - ローカルLLMレーン稼働状態
  - 主要ログの最新行
  - Local parity / Local V2 の Start / Restart / Stop ボタン

### 起動 / 停止 / 状態
```bash
scripts/runtime_ui.sh start
scripts/runtime_ui.sh status
scripts/runtime_ui.sh stop
```

### ローカル専用モード（既定）
`runtime_ui.sh` は既定で `QR_LOCAL_ONLY=1` を渡して起動する。  
必要時のみ `QR_LOCAL_ONLY=0 scripts/runtime_ui.sh restart` で VM 表示を有効化する。

### ログ確認
```bash
scripts/runtime_ui.sh logs
```

### ワンクリック起動（macOS）
`scripts/open_runtime_ui.command` をダブルクリックすると UI を起動してブラウザを開く。

## 3. ローカルLLM設定（任意）
- 現行既定は `ops/env/local-v2-stack.env` の
  `LOCAL_V2_EXTRA_ENV_FILES=ops/env/profiles/brain-ollama-safe.env`。
  `trade_min` の manual restart / watchdog / launchd 復旧でも safe canary が追随する。
- 使う前に `python3 scripts/prepare_local_brain_canary.py` で safe canary readiness を更新/確認する。
- safe canary profile:
  - `ops/env/profiles/brain-ollama-safe.env`
  - micro pocket のみ
  - `MomentumBurst, MicroLevelReactor, MicroRangeBreak, MicroTrendRetest` に限定
  - `brain_gate_mode=shadow`
  - `ORDER_MANAGER_SERVICE_WORKERS=1`
  - `fail-open` / `sample_rate=0.35` / `ttl=15s` / `timeout_cap=4s` / auto-tune off
  - `BRAIN_OLLAMA_KEEP_ALIVE=-1` で qwen を常駐させ、最初の live micro decision が cold start timeout で `no_llm` になりにくいようにする
- safe canary の反映:
```bash
python3 scripts/prepare_local_brain_canary.py
scripts/local_v2_stack.sh restart --profile trade_min \
  --env ops/env/local-v2-stack.env \
  --services quant-order-manager,quant-strategy-control
```
- aggressive profile は `ops/env/profiles/brain-ollama.env` を使うが、全 pocket + auto-tune のため、
  offline ベンチや限定検証以外では既定にしない。
- readiness レポート出力:
  - `logs/brain_canary_readiness_latest.json`
  - `logs/brain_model_selection_safe_latest.json`
  - `checks.quality_gate_ok=true` を満たしてから使う

## 4. （任意）旧VMログ/DBのローカル参照先
- GCSミラー一式:
  - `remote_logs_current/vm_gcs_mirror_*`
- 抽出済みデータ:
  - `remote_logs_current/vm_latest_core_*`

### 取得例
```bash
gcloud storage cp --recursive \
  gs://quantrabbit-logs/qr-logs/fx-trader-vm \
  remote_logs_current/vm_gcs_mirror_$(date +%Y%m%dT%H%M%S)/
```

## 5. 典型的なローカルPDCAループ
1. 市況確認（spread/ATR/API応答）。
2. `local_v2_stack.sh` で起動。
3. `logs/local_v2_stack/*.log` とローカルLLMレーンログを監視。
4. `docs/TRADE_FINDINGS.md` へ改善/敗因を追記。
5. 必要時のみ旧VM由来データを `remote_logs_current/` で参照。

### 5.1 低稼働時の高速PDCA
- 発火条件:
  - 市況が通常帯
    （spread/ATR/API 応答に異常なし）
    なのに
    `fills_15m=0`
    または
    `fills_30m<=1`
  - 直近 30 分の active lane が 1 本以下
- 15 分以内に確認する順序:
  1. `logs/orders.db` で recent `preflight_start / filled / entry_probability_reject / strategy_cooldown`
     の時系列と strategy 別件数を確認する。
  2. `logs/local_v2_stack/quant-order-manager.log`
     で
     `OPEN_SKIP`
     の dominant reason を確認する。
  3. active / expected-active worker のログを見て、
     `lookahead_block`,
     `no_signal:revert_not_found`,
     `cluster cooldown`,
     `strategy_cooldown:loss_streak`
     のどれが主因かを 1 つに絞る。
  4. `logs/health_snapshot.json`
     で
     `mechanism_integrity`
     の freshness と coverage gap が無いことを確認する。
- 判断ルール:
  - `lookahead_block` /
    `entry_probability_reject`
    が主因:
    shared gate を広く緩めず、
    該当 worker の signal / probability / size 設計を strategy-local に見直す。
  - `cluster cooldown` /
    `strategy_cooldown:loss_streak`
    が主因:
    「entry が少ないから即 cooldown を殺す」ではなく、
    recent loser burst の lane を特定して、
    cooldown 条件か loss cluster 条件の妥当性を点検する。
    `logs/stage_state.db`
    の
    `pocket_loss_window`
    を見て、
    `strategy_tag`
    の breadth が 1 本しか無く、
    損失が contained
    なら
    pocket-wide cooldown ではなく
    culprit strategy 側の改善を優先する。
  - `no_signal:revert_not_found`
    が主因:
    signal 生成側の revert / setup 検出不足を優先し、
    後段 gate ではなく worker 内条件を調べる。
  - `margin` /
    `reject` /
    `perf_block`
    が主因:
    execution / risk path を優先確認する。
- 実行ルール:
  - 1 回の PDCA で広域に複数 lane を同時に緩めない。
  - 変更は 1 本の dominant bottleneck に対する
    strategy-local
    の 1 変更を原則とする。
  - 観測と判断は
    `docs/TRADE_FINDINGS.md`
    に必ず記録する。
