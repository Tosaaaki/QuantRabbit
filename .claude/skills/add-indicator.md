---
name: add-indicator
description: "factor_cache.pyに新しいテクニカル指標を追加。コード生成→テスト→登録まで。"
trigger: "Use when the user says '指標追加', 'add indicator', '新しいインジケーター', or asks to add a technical indicator to the system."
---

# 指標追加スキル

## 使い方

- 「Keltner Channelを追加して」
- 「Williams %Rを入れて」
- 「カスタム指標: 20EMA乖離率」

## 実行手順

### Step 1: 既存構造の確認

1. `indicators/calc_core.py` の IndicatorEngine を確認
2. `indicators/factor_cache.py` のキャッシュ構造を確認
3. 既存指標一覧を把握

### Step 2: 実装

1. `indicators/calc_core.py` に計算ロジックを追加
   - pandas/numpyベース
   - 既存パターンに合わせた関数シグネチャ
2. `indicators/factor_cache.py` にキャッシュ登録
3. shared_state.jsonのtechnicals_summaryに出力追加

### Step 3: テスト

```python
# サンプルデータで計算結果を検証
python3 -c "
from indicators.calc_core import IndicatorEngine
# テスト実行
"
```

### Step 4: 報告

```
✅ 指標追加完了: Keltner Channel

追加ファイル:
- indicators/calc_core.py: keltner_channel() メソッド追加
- indicators/factor_cache.py: KC_upper, KC_lower をキャッシュに追加

利用方法:
- shared_state.json の technicals_summary に KC_upper/KC_lower が出力される
- scalp-trader等のプロンプトで参照可能
```
