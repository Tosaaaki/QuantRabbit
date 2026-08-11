# Integrated Profit Cube V1

QuantRabbitの既存システムを、旧V1損益ではなくV2費用後会計へ再結合するread-only研究です。

## 統合したもの

- V2実現損益、日次financing、partial reduction
- causal forecast
- completed-bar price action
- pair / side / regime empirical-Bayes sizing
- price-action Ridge sizing
- inventory / technical cap
- concurrent-position throttle
- 7 exit armsと5 hedge armsの証拠状態
- xarray sparse cube、SALib TRAIN sensitivity、pymoo TRAIN Pareto、MAPIE TRAIN-only conformal interval

出口と両建ては、path / cost / account margin / unwindが完全でない限り損益を生成しません。欠測は0ではなくnullです。

## 実行

```bash
research/integrated_profit_cube/2026-08-11/.adapter_env/bin/python \
  research/integrated_profit_cube/2026-08-11/run_integrated_fusion.py

python3 research/integrated_profit_cube/2026-08-11/verify_independent_oracle.py

python3 -m unittest \
  research/integrated_profit_cube/2026-08-11/test_integrated_profit_cube.py -v
```

`.adapter_env`はgit対象外です。既存のSHA固定wheelhouseからofflineで作成しています。

## 判定

V2の64日VALIDATION基準線は費用後 `+11,706.0523円 / PF 1.44693` です。

TRAINだけで決めた統合候補は、75% cohort-margin cap下の64日VALIDATIONでbaseline比 `+879.6026円`、PF `1.3729 → 1.4789`、realized DD `9,876.0991 → 7,724.0006円` でした。ただしpaired one-sided 95% LCBは `-19.9172円/episode`、account margin / netting / external inventoryは未証明です。したがって、点推定は改善していますが本番採用はできません。

exit / hedgeの改善は、現在のstrict pathと費用・margin・unwind coverageでは `NOT_EVALUABLE` です。これは不採用ではなく、識別に必要な証拠が不足している状態です。
