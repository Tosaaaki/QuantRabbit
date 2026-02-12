# ワーカー別 戻り待ち/再エントリー設計（ドラフト）

## 目的
- ワーカー別に「戻り待ちが有利/不利」を最終確定し、保有時間・勝率・平均損益の差異を根拠化する。
- ワーカー別の再エントリー条件（クールダウン/同方向再入場/低ボラ時間帯ブロック）を統一フォーマットで運用に落とす。
- entry_thesis のフラグを確実にログへ残し、成功パターン差異の精度を上げる。

## 前提
- 判定単位は pocket ではなくワーカー（strategy_tag の base）を主キーにする。
- `trades.db` と M1 ローソクを用いて MFE/MAE/BE を算出する。
- `entry_thesis` の flag を正規化して保存し、後段分析に再利用する。

## 1. 解析で出す指標（ワーカー別）
### 1-1. ベース統計
- 勝率 / PF / 平均pips / 平均保有時間
- 保有時間分布（p50/p75/p90）を勝ち/負けで分ける

### 1-2. 戻り待ち関連
- `BE_hit_hold`: 保有中に建値へ戻った割合
- `BE_hit_post_15/30/60`: 決済後 N 分以内に建値へ戻った割合
- `avg_BE_time`: 建値到達までの平均分数
- `MFE/MAE`: 最大含み益/含み損（pips）

### 1-3. 戻り待ち有利/不利の判定基準（初期案）
- **有利 (favor)**  
  - `BE_hit_post_30 >= 0.35` かつ `avg_BE_time <= 60m`  
  - `avg_MAE <= max(8p, avg_MFE * 1.4)`
- **不利 (avoid)**  
  - `BE_hit_post_60 <= 0.20` または `avg_MAE >= avg_MFE * 2.0`  
  - 勝ちの保有時間 p50 < 20m かつ 負け p50 > 60m（ロス長期化の疑い）
- **中立 (neutral)**: 上記以外
- サンプル数 `< 30` は `neutral` 固定（要保留）

## 2. 再エントリー条件（ワーカー別）
### 2-1. ルール軸
- **クールダウン**: 勝ち/負けで秒数を分離（例: win 60s / loss 180s）
- **同方向再入場の閾値**:  
  - `long` は `price <= last_close - reentry_pips`  
  - `short` は `price >= last_close + reentry_pips`
- **時間帯フィルタ**: 極端にボラがない時間帯のみブロック（基本は時間帯フィルタなし）
- **同方向オープン抑制**: 同一ワーカーの同方向オープン数や含み損が閾値を超えたら新規を停止
- **距離が十分離れた場合の例外**: `stack_reentry_pips` を超える逆行距離なら、soft cap を一時的に許可（hard cap は維持）

### 2-2. 設定フォーマット（案）
- `config/worker_reentry.yaml`
  - `defaults`: 全体デフォルト
  - `strategies`: strategy_tag base ごとに上書き

```yaml
defaults:
  cooldown_win_sec: 60
  cooldown_loss_sec: 180
  same_dir_reentry_pips: 1.8
  allow_jst_hours: []
  block_jst_hours: []
  return_wait_bias: neutral
  max_open_trades: 0
  max_open_adverse_pips: 0.0
  max_open_trades_hard: 0
  stack_reentry_pips: 0.0

strategies:
  TrendMA:
    cooldown_loss_sec: 420
    same_dir_reentry_pips: 3.0
    block_jst_hours: [4,5,6]
    max_open_trades: 3
    max_open_adverse_pips: 40.0
    max_open_trades_hard: 6
    stack_reentry_pips: 25.0
    return_wait_bias: favor
  M1Scalper:
    cooldown_loss_sec: 60
    same_dir_reentry_pips: 1.2
    block_jst_hours: [4,5,6]
    max_open_trades: 2
    max_open_adverse_pips: 30.0
    return_wait_bias: avoid
```

### 2-3. 実装ポイント（案）
- `logs/stage_state.db` に `strategy_reentry_state` テーブルを追加
  - `strategy`, `direction`, `last_close_time`, `last_close_price`, `last_result`
- `execution/reentry_gate.py` を追加し、`order_manager` で最終送信前にチェック
  - `cooldown` / `same_dir_reentry_pips` / `block_jst_hours` を判定（`allow_jst_hours` があれば優先）
  - `block_jst_hours` はデフォルトで全体共通（global）運用
  - `return_wait_bias` を使い、favor は待ち/距離を強め、avoid は緩める
  - ブロック時は `orders.db` に `status="reentry_block"` を記録
- `stage_tracker.update_loss_streaks()` 内で `strategy_reentry_state` を更新

## 3. entry_thesis フラグのログ整備
### 3-1. 追加したいフラグ（統一キー）
- `trend_bias` / `trend_score`  
- `range_snapshot` / `entry_mean` / `reversion_failure` / `profile`  
- `size_factor_hint` / `pattern_tag` / `pattern_meta`

### 3-2. 正規化方針
- `entry_thesis.flags = [<trueなキー>...]` を併記し、SQLで集計しやすくする
- `strategy_tag` の base で統一（`-long/-short` の suffix は別名保持）
## 4. 分析フロー（案）
1) `analytics/worker_return_wait_report.py` で日次集計  
2) `--out-json` で差異のJSON保存、`--out-yaml` で `worker_reentry.yaml` の叩き台を生成  
   - `cooldown_win/loss` と `same_dir_reentry_pips` は保有時間/MFE/MAEから推奨値を算出  
3) `block_jst_hours` は低ボラ時間帯を自動抽出（min trades + MFE/MAE閾値）  
4) デフォルトは global、必要なら `--block-hours-scope per_strategy` で個別化  
5) `config/worker_reentry.yaml` へ反映（必要なら自動適用）  

### 4-1. ワンショット実行（VM/ローカル）
- `scripts/run_worker_reentry_pipeline.sh --days 14 --min-trades 30`
- 時間帯抽出の閾値は `--block-hour-trades` / `--block-hour-mfe-max` / `--block-hour-mae-max` / `--block-hour-top` で調整可能
- `--block-hour-window` で抽出対象の時間帯を JST で限定可能（例: `3-6` や `3,4,5`）
- `--block-hours-scope`（default: global）で時間帯ブロックの粒度を切替
- `--apply` で `config/worker_reentry.yaml` を自動更新
- `--require-block-hours` を併用すると、`defaults.block_jst_hours` が空のままなら失敗させる

## 5. 直近の実装優先度
1) entry_thesis フラグの正規化と保存  
2) ワーカー別の戻り待ち判定レポート  
3) reentry_gate + worker_reentry.yaml の導入
