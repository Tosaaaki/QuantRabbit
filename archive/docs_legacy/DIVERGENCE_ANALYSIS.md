Divergence 成績分析クエリ

前提
- divergence は `entry_thesis.divergence` に保存される（JSON, trades.db の entry_thesis）。
- この項目は 2026-01-30 06:45 UTC のデプロイ以降に記録されます。
- sqlite の JSON1 拡張が有効な環境で `json_extract` / `json_type` が使えます。

共通フィルタ例
- 直近30日: `close_time >= datetime('now','-30 days')`
- divergence あり: `json_type(entry_thesis,'$.divergence') IS NOT NULL`

1) divergence 有無の比較
```
SELECT
  CASE WHEN json_type(entry_thesis,'$.divergence') IS NOT NULL THEN 'div_on' ELSE 'div_off' END AS div_flag,
  COUNT(*) AS trades,
  ROUND(SUM(pl_pips), 2) AS sum_pips,
  ROUND(AVG(pl_pips), 2) AS avg_pips,
  ROUND(AVG(CASE WHEN pl_pips > 0 THEN 1.0 ELSE 0.0 END), 3) AS win_rate,
  ROUND(AVG(realized_pl), 0) AS avg_jpy
FROM trades
WHERE close_time >= datetime('now','-30 days')
GROUP BY div_flag
ORDER BY div_flag;
```

2) divergence スコアの符号別（順張り/逆張りの当たり）
```
WITH t AS (
  SELECT
    pl_pips,
    realized_pl,
    units,
    json_extract(entry_thesis,'$.divergence.score') AS div_score
  FROM trades
  WHERE close_time >= datetime('now','-30 days')
    AND json_type(entry_thesis,'$.divergence') IS NOT NULL
)
SELECT
  CASE
    WHEN div_score IS NULL OR div_score = 0 THEN 'neutral'
    WHEN (div_score > 0 AND units > 0) OR (div_score < 0 AND units < 0) THEN 'aligned'
    ELSE 'counter'
  END AS div_align,
  COUNT(*) AS trades,
  ROUND(SUM(pl_pips), 2) AS sum_pips,
  ROUND(AVG(pl_pips), 2) AS avg_pips,
  ROUND(AVG(CASE WHEN pl_pips > 0 THEN 1.0 ELSE 0.0 END), 3) AS win_rate
FROM t
GROUP BY div_align
ORDER BY div_align;
```

3) divergence スコア強度別
```
WITH t AS (
  SELECT
    pl_pips,
    realized_pl,
    ABS(COALESCE(json_extract(entry_thesis,'$.divergence.score'), 0.0)) AS abs_score
  FROM trades
  WHERE close_time >= datetime('now','-30 days')
    AND json_type(entry_thesis,'$.divergence') IS NOT NULL
)
SELECT
  CASE
    WHEN abs_score >= 0.60 THEN '>=0.60'
    WHEN abs_score >= 0.40 THEN '0.40-0.59'
    WHEN abs_score >= 0.20 THEN '0.20-0.39'
    WHEN abs_score > 0.00 THEN '0.01-0.19'
    ELSE '0.00'
  END AS score_bucket,
  COUNT(*) AS trades,
  ROUND(AVG(pl_pips), 2) AS avg_pips,
  ROUND(AVG(CASE WHEN pl_pips > 0 THEN 1.0 ELSE 0.0 END), 3) AS win_rate
FROM t
GROUP BY score_bucket
ORDER BY score_bucket;
```

4) RSI / MACD の divergence 種別別（bull/bear）
```
WITH t AS (
  SELECT
    pl_pips,
    json_extract(entry_thesis,'$.divergence.rsi.kind') AS rsi_kind,
    json_extract(entry_thesis,'$.divergence.macd.kind') AS macd_kind
  FROM trades
  WHERE close_time >= datetime('now','-30 days')
    AND json_type(entry_thesis,'$.divergence') IS NOT NULL
)
SELECT
  CASE
    WHEN rsi_kind > 0 THEN 'rsi_bull'
    WHEN rsi_kind < 0 THEN 'rsi_bear'
    ELSE 'rsi_none'
  END AS rsi_bucket,
  CASE
    WHEN macd_kind > 0 THEN 'macd_bull'
    WHEN macd_kind < 0 THEN 'macd_bear'
    ELSE 'macd_none'
  END AS macd_bucket,
  COUNT(*) AS trades,
  ROUND(AVG(pl_pips), 2) AS avg_pips,
  ROUND(AVG(CASE WHEN pl_pips > 0 THEN 1.0 ELSE 0.0 END), 3) AS win_rate
FROM t
GROUP BY rsi_bucket, macd_bucket
ORDER BY rsi_bucket, macd_bucket;
```

5) strategy 別（divergence ありのみ）
```
SELECT
  COALESCE(strategy_tag, strategy, 'unknown') AS strategy,
  COUNT(*) AS trades,
  ROUND(SUM(pl_pips), 2) AS sum_pips,
  ROUND(AVG(pl_pips), 2) AS avg_pips,
  ROUND(AVG(CASE WHEN pl_pips > 0 THEN 1.0 ELSE 0.0 END), 3) AS win_rate
FROM trades
WHERE close_time >= datetime('now','-30 days')
  AND json_type(entry_thesis,'$.divergence') IS NOT NULL
GROUP BY strategy
ORDER BY sum_pips DESC
LIMIT 30;
```

6) pocket 別
```
SELECT
  pocket,
  COUNT(*) AS trades,
  ROUND(SUM(pl_pips), 2) AS sum_pips,
  ROUND(AVG(pl_pips), 2) AS avg_pips,
  ROUND(AVG(CASE WHEN pl_pips > 0 THEN 1.0 ELSE 0.0 END), 3) AS win_rate
FROM trades
WHERE close_time >= datetime('now','-30 days')
  AND json_type(entry_thesis,'$.divergence') IS NOT NULL
GROUP BY pocket
ORDER BY sum_pips DESC;
```

VM で実行する場合（例）
```
scripts/vm.sh -p quantrabbit -z asia-northeast1-a -m fx-trader-vm \
  sql -f /home/tossaki/QuantRabbit/logs/trades.db -t \
  -q "SELECT COUNT(*) FROM trades WHERE json_type(entry_thesis,'$.divergence') IS NOT NULL;"
```

ローカルで実行する場合（例）
```
sqlite3 logs/trades.db "SELECT COUNT(*) FROM trades WHERE json_type(entry_thesis,'$.divergence') IS NOT NULL;"
```
