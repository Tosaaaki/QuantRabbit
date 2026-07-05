# Historical-Only To Fresh Proof Replay

- Generated: `2026-07-05T18:15:59Z`
- Fresh replay: `logs/reports/forecast_improvement/oanda_history_replay_validate_latest.json`
- Price truth: `PRICE_TRUTH_OK`
- Adoption level: `PAIR_LOCAL_RANK_ONLY`

| lane | class | daily % | fresh S5 status | forecast | geometry | margin | permission |
|---|---|---:|---|---|---|---|---|
| `range_trader:GBP_USD:LONG:RANGE_ROTATION` | `HISTORICAL_ONLY` | 8.357 | `EVIDENCE_GAP_UNDER_SAMPLED` | `False` | `False` | `False` | `False` |
| `trend_trader:AUD_JPY:SHORT:TREND_CONTINUATION` | `HISTORICAL_ONLY` | 7.9303 | `EVIDENCE_GAP_UNDER_SAMPLED` | `False` | `False` | `False` | `False` |

## 744h Replay Boundary

- {'generated_at_utc': '2026-07-05T17:59:08.900081+00:00', 'status': 'OK', 'lookback_basis': 'execution_timing_audit current local artifact', 'loss_closes_audited': 39, 'historical_pre_repair_loss_closes_profit_capture_missed': 14, 'historical_pre_repair_loss_closes_repair_replay_triggered': 13, 'loss_close_repair_replay_delta_jpy': 18771.7372, 'post_repair_live_evidence_loss_closes_audited': 5, 'post_repair_live_evidence_loss_closes_profit_capture_missed': 0, 'permission_boundary': 'System-level 744h timing replay is diagnostic; it is not a lane-specific live permission receipt.'}
