# Trader Tools — プロトレーダーの自作ツール置き場

このディレクトリには、Claudeが自分で作った分析ツールを置く。
scalp-trader や macro-intel が必要に応じて作成・改善する。

## ルール
- 各スクリプトは独立して `python scripts/trader_tools/xxx.py` で実行可能にする
- 出力は stdout (JSON推奨) または `logs/` に書き出す
- 既存モジュール (`indicators/`, `analysis/`) を import して活用してよい
- while True ループ禁止。ワンショット実行のみ。

## 既存分析基盤の使い方

### テクニカル指標 (70+)
```python
from indicators.factor_cache import all_factors, refresh_cache_from_disk
refresh_cache_from_disk()
factors = all_factors()  # Dict[str, Dict[str, float]] — M1,M5,H1,H4,D1
# factors['M1']['rsi'], factors['H1']['adx'], etc.
```

### レジーム判定
```python
from analysis.market_regime import classify_regime
regime = classify_regime(factors_m1, factors_m5)
# regime.regime: MarketRegime enum
# regime.is_tradeable(), regime.favors_long(), etc.
```

### 戦略フィードバック
```python
import json
with open('logs/strategy_feedback.json') as f:
    fb = json.load(f)
# fb['strategies']['StrategyName']['strategy_params']['win_rate']
```

### マーケットコンテキスト
```python
import json
with open('logs/market_context_latest.json') as f:
    ctx = json.load(f)
# ctx['dollar']['dxy'], ctx['rates']['us10y'], ctx['risk']['vix']
```

## 作成例
- `confluence_scanner.py` — 複数テクニカルのコンフルエンス度を数値化
- `regime_history.py` — レジーム変化履歴をトラッキング
- `pair_correlation.py` — ペア間相関をリアルタイム計算
- `entry_timing.py` — 最適エントリータイミングのパターン分析
