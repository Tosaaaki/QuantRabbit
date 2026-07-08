# Trader Goal Loop Orchestrator

- Status: `NEXT_WORK_SELECTED`
- Generated at UTC: `2026-07-08T07:07:18.989612+00:00`
- Read only: `True`
- Live side effects: `[]`
- Live permission allowed: `False`
- requires_operator_approval_for_this_report: `False`
- requires_operator_review_before_scout_or_routing: `True`
- Current phase: `SCOUT_BLOCKED_OPERATOR_REVIEW`
- Selected next work type: `OPERATOR_REVIEW_REPORT`
- Selection reason: scout status is SCOUT_BLOCKED_OPERATOR_REVIEW; the next artifact must package SCOUT approval/rejection evidence only, not live permission.
- Four x progress hypothesis: EUR_USD|SHORT|BREAKOUT_FAILURE の attached-TP HARVEST 証拠を proof floor まで進め、market-close leak と month-scale negative を隠さず除外/修復できれば、rolling 30d funding-adjusted equity 4x に近づく 正の期待値レーンを増やせる。現在の不足サンプルは 0。
- Root improvement target: EUR_USD|SHORT|BREAKOUT_FAILURE を、発注許可ではなく read-only の SCOUT 判断材料と期待値改善実験で live-grade HARVEST 候補へ近づける。
- Expected edge improvement: TP proof 20勝 / 0 TP負け、期待値 643.2912 JPY、proof gap 0 sample、max_loss_jpy_cap 418.0 を起点に、追加証拠で HARVEST の薄い正期待値を補強し、market-close leak / negative expectancy / month-scale replay negative を隠さず NO_TRADE 除外へ回す。

## Key State

- Payoff verdict: `MIXED_HARVEST_PRIMARY` / stale=`False`
- HARVEST closest: `EUR_USD|SHORT|BREAKOUT_FAILURE` / live promotion allowed=`False`
- Scout status: `SCOUT_BLOCKED_OPERATOR_REVIEW` / allowed=`False`
- Proof queue count: `2`
- Can create live permission count: `0`
- Normal routing allowed: `False`
- Guardian clear: `False`

## Approval Boundary

- このreport生成自体は承認不要。ただしSCOUT/normal routing前にはoperator review必須。
- `requires_operator_approval_for_this_report` is report-generation approval only.
- `requires_operator_review_before_scout_or_routing` is the separate SCOUT/normal-routing gate.

## Repeat Guard

- Repeat allowed: `True`
- Key blocker: `SCOUT_BLOCKED_OPERATOR_REVIEW:NORMAL_ROUTING_FALSE`

## Success Condition Evaluation

- Status: `MET`

## Next Allowed Commands

- `python3 -m json.tool data/trader_goal_loop_orchestrator.json >/dev/null`
- `PYTHONPATH=src python3 -m unittest tests.test_trader_goal_loop_orchestrator -v`

## Safety Boundary

- This artifact is not live permission.
- Order send, order cancel, position close, and launchd load/reload remain prohibited here.
- Negative expectancy, month-scale replay negatives, and proof queue emptiness are surfaced as blockers.
