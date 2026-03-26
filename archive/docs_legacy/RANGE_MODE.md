# Range Mode & Online Tuning

## 1. レンジモード強化（2025-10）

### 判定
- `analysis/range_guard.detect_range_mode` が M1 の `ADX<=24`, `BBW<=0.24`, `ATR<=7` を主に見つつ圧縮/ボラ比の複合スコア（0.66 以上）や `compression_trigger` で `range_mode` を返す。
- `metrics.composite` と `reason` をログ。

### エントリー制御
- `range_mode=True` 中は macro 新規を抑制。
- 許可戦略を BB 逆張り（`BB_RSI` など）に限定。
- `focus_tag` を `micro` へ縮退、`weight_macro` 上限 0.15。

### 利確/損切り
- レンジ中は各 `exit_worker` が TP/トレイル/lock をタイトに。
- 目安 1.5〜2.0 pips の RR≒1:1、fast_scalp/micro/macro で閾値別設定。
- 共通 `exit_manager` は使用しない。

### 分割利確
- `execution/order_manager.plan_partial_reductions` はレンジ中にしきい値を引き下げ。
- macro 16/22, micro 10/16, scalp 6/10 pips。

### ステージ/再入場
- `execution/stage_tracker` が方向別クールダウンを管理。
- 連続 3 敗で 15 分ブロック。
- 勝敗に応じてロット係数縮小（マーチン禁止）。

## 2. オンライン自動チューニング
- 5〜15 分間隔で `scripts/run_online_tuner.py` を呼び、Exit 感度や入口ゲート・quiet_low_vol 配分を小幅調整（ホットパスは対象外）。
- 生成物は既定で `logs/tuning/` 配下（`tuning_overrides.yaml`, `tuning_overlay.yaml`, `history/`）に書く（deploy の `git pull` と競合させないため）。
- ランタイムの読込は `utils/tuning_loader.py` が `logs/tuning/` → 旧 `config/` → `config/tuning_presets.yaml` の順で解決（`TUNING_*_PATH` を明示した場合はそのパスのみ）。
- ToDo/検証タスクは `docs/autotune_taskboard.md` に集約し、完了後は同ファイルでアーカイブ。
- 詳細手順は `docs/ONLINE_TUNER.md` を参照。
