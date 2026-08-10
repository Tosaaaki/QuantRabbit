# Monthly 3x Growth Engine V1

費用後プラスの `ALL_TRADES` を捨てず、予測を `TRADE/SKIP` ではなくサイズ配分へ接続する研究専用エンジンです。20万円から30暦日で60万円という目標を、機会数・1取引edge・資本効率へ分解します。

正本は `preregister_v1.json` です。結果後の閾値選択、holdout、live/Paper/order/deployは使いません。

実行:

```bash
python3 research/monthly_3x_growth_engine/2026-08-10/run_growth_engine.py
python3 research/monthly_3x_growth_engine/2026-08-10/run_strategy_expansion.py
python3 research/monthly_3x_growth_engine/2026-08-10/run_strategy_exit_expansion.py
python3 research/monthly_3x_growth_engine/2026-08-10/build_profit_reason_ledger.py
python3 -m unittest \
  research/monthly_3x_growth_engine/2026-08-10/test_growth_engine.py \
  research/monthly_3x_growth_engine/2026-08-10/test_strategy_expansion.py
python3 research/monthly_3x_growth_engine/2026-08-10/verify_independent_oracle.py
```

結論と次の利益経路は `verdict_v1.md`、機械可読な根拠は `profit_reason_ledger_v1.json` を参照してください。
