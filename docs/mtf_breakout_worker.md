# MTF‑Breakout Worker

H1 でトレンド方向を確認しつつ、M5 のプルバック後ブレイクで発火するワーカーのスケルトンです。リプレイ環境やドライランに載せやすいよう、`DataFeed`/`Broker` の最小インタフェースだけに依存しています。

## 特徴
- 上位足 (`timeframe_trend`=H1 既定) のトレンド強度を評価し、閾値 (`edge_threshold`) 以上でのみシグナルを許可。
- 下位足 (`timeframe_entry`=M5 既定) の直近レンジ高値/安値を計測し、最低 `pullback_min_bars` 本のプルバック経過後にブレイクした場合のみエントリー。
- `place_orders=False` でシグナル出力のみ。問題なければ True にして `Broker.send()` を呼び出す運用へ拡張可能。

## 使い方（最小例）
```python
from workers.mtf_breakout import MtfBreakoutWorker
from workers.mtf_breakout.config import DEFAULT_CONFIG

cfg = {**DEFAULT_CONFIG, "universe": ["USD_JPY"], "place_orders": False}
worker = MtfBreakoutWorker(cfg, broker=None, datafeed=datafeed, logger=logger)
intents = worker.run_once()
for intent in intents:
    print(intent)
```

## コンフィグ項目
- `breakout_lookback`: M5 でレンジを測る本数（デフォルト 20）
- `pullback_min_bars`: レンジ高値/安値からの最小プルバック本数
- `cooldown_bars`: シグナル発生後のクールダウン
- `edge_threshold`: 上位足トレンド強度の最低値（0〜1）
- `budget_bps`: 擬似サイズ計算に利用する BPS 値（`_mk_order` 内で使用）
- `filters`: 出来高ランクやスプレッド制限など、必要に応じて拡張可能なプレースホルダ

## 次のステップ
1. `place_orders=False` のまま `run_once()` の戻り値をログに流し、シグナルの頻度・質感を確認
2. リプレイ (`scripts/replay_workers.py` 等) にフックして挙動を検証
3. 問題なければ `place_orders=True` に切替え、実ブローカー層と ExitManager を統合する
