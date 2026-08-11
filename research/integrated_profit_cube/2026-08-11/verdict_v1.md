# Verdict — INTEGRATED_PROFIT_CUBE_V1

## 結論

`BASELINE_POSITIVE_INTEGRATED_IMPROVEMENT_NOT_YET_ADMISSIBLE`

QuantRabbitは「何をやってもマイナス」ではありません。修正済みV2会計の64日VALIDATIONは費用後でプラスです。問題は、各systemの出力が旧V1損益や集計表に分散し、最終decision consumerへ一貫して接続されていなかったことです。

今回、decision_idを主キーに、予測、価格行動、pair/side/regime、inventory、technical cap、concurrency、exit、hedgeを同一cubeへ載せました。TRAINだけで選んだ候補は64日VALIDATIONの点推定を改善し、DDも縮小しました。

ただし、改善の不確実性下限は負で、account-level margin truthはありません。出口・両建てもstrict pathが少なく、変更後の完全な費用後損益を識別できません。このため本番採用・注文許可には進めません。

## 次の独立境界

1. order/fill/protection/exitをtick順で確定できるforward ledger
2. decision-time fee/financing schedule
3. account margin available/used/rate、netting、manual/external inventory
4. partial fill/depthとdual-leg unwind
5. 同一contractの変更eventを最低30件、exposure clusterを最低20件

この5点が揃えば、今回のcubeを変更せずにBE/trail/hedgeをactual-after-costへ昇格できます。
