# WFO Overfit Report

`analytics/wfo_overfit_report.py` は、trades 履歴から WFO 頑健性を定期点検する軽量レポートです。

## 1. 目的
- 運用に近い時系列分割で train/test を繰り返し、崩れやすい戦略選抜を検知する。
- PBO の完全実装ではなく、`PBO-lite`（train最良戦略がtestで下位半分に落ちる確率）を使う。
- 併せて、戦略ごとの Sharpe に対する DSR 近似を算出する。

## 2. 実行
```bash
scripts/run_wfo_overfit_report.sh
```

出力:
- `logs/reports/wfo_overfit/latest.json`
- `logs/reports/wfo_overfit/latest.md`

## 3. 読み方
- `pbo_lite`: 高いほど過学習的な選抜リスクが高い。
- `selected_positive_rate`: 各 test window で train 選抜戦略が正の成績だった比率。
- `median_test_percentile`: train 選抜戦略の test 内順位（0〜1）。
- `strategy_stats[].dsr`: 複数戦略比較を加味した Sharpe 有意性の近似。

## 4. 注意
- これは online health 指標であり、論文準拠の CSCV/PBO を完全再現するものではない。
- 判断は必ず VM の本番 `trades.db` を使って行う。

