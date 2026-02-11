# 自動学習（オンラインチューニング/自動ブロック）実装計画

更新日時: 2026-02-11 JST

このファイルは「自動学習」と呼んでいる一連の仕組み（オンラインチューナ・自動ブロック・自動配分）をまとめて、
実装/運用の優先順位と、1PR=1ファイルの切り方で管理するためのタスクボードです。

## 対象スコープ（現状の実装）
- Online tuner: `scripts/run_online_tuner.py` / `autotune/online_tuner.py`
  - Exit（lowvol）・micro gate・quiet_low_vol の配分を小刻みに提案/適用
  - systemd: `systemd/quant-online-tuner.timer` → `scripts/run_online_tuner_live.sh`
- Policy cycle（LLM無しのヒューリスティック）: `scripts/run_policy_cycle.py` / `analytics/policy_generator.py`
  - Pocket の allow_new 等を「自動ブロック」するオーバーレイ生成
  - systemd: `systemd/quant-policy-cycle.timer`
- Worker 内の自動ガード
  - PF/勝率悪化によるブロック/縮小: `workers/common/perf_guard.py`
  - 再入場や時間帯ブロック: `execution/stage_tracker.py` / `execution/reentry_gate.py`（戦略ごとの gate）
- （任意）Scalp の自己調整: `autotune/scalp_trainer.py`（`SCALP_AUTOTUNE_ENABLED=1`）
- （任意）Continuous backtest + autotune: `scripts/continuous_backtest.py`（`systemd/quant-autotune.timer`）

## 非交渉ルール（要点）
- ニュース連動は使わない（`news_fetcher` 等は無効のまま）。
- LLM は `workers/common/brain.py` の任意ゲート用途のみ（メイン判定はローカル）。
- 本番確認は必ず VM（systemd/journalctl/DB）または OANDA API を根拠にする。

## 最優先の問題（今すぐ直す）
1. **自動学習が tracked file を書き換えて deploy を壊す**
   - `scripts/vm.sh deploy` は VM 上で `git pull --ff-only` を実行しており、tracked file が書き換わっていると pull が失敗し、しかも `|| true` で握りつぶされる。
   - 現状、以下が tracked のまま書き換わり得る:
     - `config/tuning_overrides.yaml`, `config/tuning_overlay.yaml`, `config/tuning_history/*`
     - `configs/scalp_active_params.json`
2. **“効いているか/止まっているか”が VM 上で一目で分からない**
   - チューナ/ポリシーの最終適用時刻・差分・ガード判定が散在し、停止と劣化の切り分けが遅い。
3. **データ不足/偏りでノブがドリフトし得る**
   - 直近 N 分の trades が少ない/特定 pocket のみ/理由分類が欠落、などで「意味のない微調整」が連続する余地がある。

## 実装方針（設計の再整理）
- ライブで触って良いのは “ゆっくり動くノブ” のみ（既存方針を維持）。
- 変更は「提案→ガード→適用→履歴→ロールバック」を一貫して機械化する。
- runtime state は **git 管理外**（`logs/` など）へ集約し、deploy と独立させる。
- 競合する調整（例: 同じ値を policy と tuner が触る）は禁止し、責務を分ける。

## フェーズ別タスク（1PR=1ファイルで切る）

### Phase 0: デプロイ破壊の根絶（最短で収束）
- [ ] `utils/tuning_loader.py`: 参照先の優先順位を `logs/` 系の runtime state → `config/` の presets に変更（後方互換: config の既存ファイルも読める）。  
  受け入れ条件: VM 上で tracked file に触れずに tuning が反映される構成が取れる。
- [ ] `scripts/run_online_tuner_live.sh`: 出力先デフォルトを `logs/tuning/*.yaml` に変更（環境変数で上書き可）。  
  受け入れ条件: `git status` が汚れないまま 10 分周期で走る。
- [ ] `analytics/policy_apply.py`: tuning の出力先デフォルトを `logs/tuning/*.yaml` に変更（policy 自体は `logs/policy_*.json` のまま）。  
  受け入れ条件: `systemd/quant-policy-cycle.timer` が走っても deploy を阻害しない。
- [ ] `autotune/scalp_trainer.py`: `configs/scalp_active_params.json` への直接書き込みを廃止し、runtime state（例: `logs/tuning/scalp_active_params.json`）へ移動。  
  受け入れ条件: Scalp 自己調整を再導入しても git を汚さない。

### Phase 1: “暴れない”オンライン学習にする（安全弁）
- [ ] `autotune/online_tuner.py`: データ不足/偏り時の freeze（min trades, pocket/strategy のカバレッジ、reason 欠落）を追加。  
  受け入れ条件: trades が少ない時間帯に micro_share 等が連続で同方向へ寄らない。
- [ ] `autotune/online_tuner.py`: LKG（last known good）と自動ロールバックを追加（EV/PF/勝率の下振れ検知で 1 つ前に戻す）。  
  受け入れ条件: 劣化時に自動で shadow に戻るか、前回値へ復帰できる。

### Phase 2: “何が効いているか”を可視化する（運用の短縮）
- [ ] `scripts/run_online_tuner_live.sh`: run 毎に “採用/不採用理由” と “差分” を構造化ログで残す（journalctl で追える）。  
  受け入れ条件: 直近 24h の変更履歴と理由が VM のログだけで追跡できる。
- [ ] `docs/ONLINE_TUNER.md`: 現行の systemd 構成（timer/service、出力先、disable 方法）を最新に更新。  
  受け入れ条件: 新規運用者がこのドキュメントだけで停止/再開/確認できる。

### Phase 3: 調整の競合排除（責務の固定）
- [ ] `docs/RANGE_MODE.md`: “どの仕組みがどのノブを触るか” を明文化（policy と tuner の責務分離）。  
  受け入れ条件: 重複変更のレビュー指摘が機械的にできる。

## VM ロールアウト手順（共通）
1. ブランチで実装 → `git commit` → `git push`
2. VM へ反映: `scripts/vm.sh ... deploy -i -t`
3. VM で確認（例）:
   - `systemctl status quant-online-tuner.timer`
   - `journalctl -u quant-online-tuner.service -n 200 --no-pager`
   - `git -C ~/QuantRabbit status --porcelain`（dirty が無いこと）

## 参考（過去のメモ）
- 2025-11 時点の “config/ に tuning を書く” 方針は、deploy の `git pull` と競合するため廃止する（この計画で置き換え）。
