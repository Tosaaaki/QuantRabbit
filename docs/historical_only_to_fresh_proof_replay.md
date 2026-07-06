# Historical-Only To Fresh Proof Replay

- Generated: `2026-07-06T09:04:10Z`
- Fresh replay: `logs/reports/forecast_improvement/oanda_history_replay_validate_latest.json`
- Price truth: `PRICE_TRUTH_OK`
- Adoption level: `PAIR_LOCAL_RANK_ONLY`

| lane | class | daily % | fresh S5 status | forecast | geometry | margin | permission |
|---|---|---:|---|---|---|---|---|
| `range_trader:GBP_USD:LONG:RANGE_ROTATION` | `HISTORICAL_ONLY` | 8.1576 | `EVIDENCE_GAP_UNDER_SAMPLED` | `True` | `False` | `False` | `False` |
| `trend_trader:AUD_JPY:SHORT:TREND_CONTINUATION` | `HISTORICAL_ONLY` | 7.741 | `EVIDENCE_GAP_UNDER_SAMPLED` | `True` | `False` | `False` | `False` |

## 744h Replay Boundary

- {'generated_at_utc': '2026-07-06T01:58:38.527953+00:00', 'status': 'OK', 'lookback_basis': 'execution_timing_audit current local artifact', 'loss_closes_audited': 8, 'historical_pre_repair_loss_closes_profit_capture_missed': 1, 'historical_pre_repair_loss_closes_repair_replay_triggered': 1, 'loss_close_repair_replay_delta_jpy': 466.2, 'post_repair_live_evidence_loss_closes_audited': 5, 'post_repair_live_evidence_loss_closes_profit_capture_missed': 0, 'permission_boundary': 'System-level 744h timing replay is diagnostic; it is not a lane-specific live permission receipt.'}
