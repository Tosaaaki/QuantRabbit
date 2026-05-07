# Gap Report: CLOSE Decision Not Executed

**Generated**: 2026-05-07T00:19:00Z  
**Cycle**: 2026-05-07T00:03:43Z  
**Severity**: CRITICAL (blocks daily target campaign)

## Summary

The trader decided to CLOSE two contradicted EUR_USD SHORT positions (trades 470279, 470266) to lock +113 JPY profit and free 2/4 portfolio slots. GPT verification ACCEPTED the decision. However, PositionProtectionGateway executed NO_ACTION—all positions remained HOLD_PROTECTED. The portfolio remains at 4/4 capacity, blocking all 30 lanes with PORTFOLIO_POSITION_LIMIT.

## Impact

- **Daily target progress**: 0.21% (43 JPY / 20,834 JPY)
- **Portfolio capacity**: 4/4 positions (at limit)
- **LIVE_READY lanes**: 0 (all 30 lanes blocked by PORTFOLIO_POSITION_LIMIT)
- **Expected profit locked**: +113 JPY (not realized)
- **Expected freed capacity**: 2 position slots (not freed)
- **Campaign status**: STUCK—cannot add new entries until positions close

## Decision vs Execution Mismatch

### Trader Decision (`codex_trader_decision_response.json` at 23:59:30 UTC)
```json
{
  "action": "CLOSE",
  "close_trade_ids": ["470279", "470266"],
  "status": "DECISION_COMPLETE",
  "confidence": "HIGH",
  "thesis": "Market contradiction + portfolio capacity pressure + profitable exit opportunity",
  "reason_for_action": "CHART_DIRECTION_CONFLICT cited: EUR_USD SHORT conflicts with current direction bias LONG. Close to lock +113 JPY profit and free 2/4 slots."
}
```

### GPT Verification (`gpt_trader_decision_report.md` at 00:03:43 UTC)
```
Status: ACCEPTED
Action: CLOSE
Verification Issues: none
```

### Position Execution (`position_execution.json` at 00:03:43 UTC)
```json
{
  "status": "NO_ACTION",
  "send_requested": true,
  "sent": false,
  "actions": [
    {"trade_id": "470279", "pair": "EUR_USD", "management_action": "HOLD_PROTECTED", "sent": false},
    {"trade_id": "470266", "pair": "EUR_USD", "management_action": "HOLD_PROTECTED", "sent": false}
  ]
}
```

**Expected**: Trades 470279, 470266 closed via PositionProtectionGateway  
**Actual**: Both trades HOLD_PROTECTED, no close requests generated

## Root Cause

**Architectural gap**: The `close_trade_ids` field from the trader decision receipt is not connected to the PositionManager/PositionProtectionGateway execution path.

### Code Flow Analysis

1. **PositionManager** (`position_manager.py:147-149`):
   ```python
   elif opposite_score is not None and same_score is not None \
        and opposite_score >= same_score + 20 \
        and position.unrealized_pl_jpy < 0:  # <-- BLOCKS profitable closes
       reasons.append(f"opposite thesis score {opposite_score:.1f} materially exceeds same-direction {same_score:.1f}")
       action = ACTION_REVIEW_EXIT
   ```
   - Sets `ACTION_REVIEW_EXIT` ONLY for contradicted positions with **negative P&L**
   - EUR_USD SHORT positions are contradicted BUT have **positive P&L** (+34, +79 JPY)
   - Result: action = HOLD_PROTECTED (default path)

2. **PositionProtectionGateway** (`position_execution.py:149-153`):
   ```python
   if managed.action == ACTION_HOLD_PROTECTED:
       return action  # <-- No close request generated
   if managed.action == ACTION_REVIEW_EXIT:
       action["request"] = {"type": "CLOSE", "trade_id": position.trade_id, "units": "ALL"}
       return action
   ```
   - Generates close requests ONLY when `managed.action == ACTION_REVIEW_EXIT`
   - Since PositionManager returned HOLD_PROTECTED, no close requests were created

3. **Missing Link**:
   - The trader decision's `close_trade_ids` field is never read by PositionManager
   - PositionManager only considers its own logic (contradiction + negative P&L)
   - There is no code path from `decision["close_trade_ids"]` → `ACTION_REVIEW_EXIT`

### Why This Matters

The current architecture assumes PositionManager autonomy: it decides what to do with positions based on scores and P&L. But the trader decision system (TraderBrain → GPT verification) can have **strategic reasons** to close positions that PositionManager's heuristics don't capture:

- **Portfolio capacity pressure** (this case: need to free slots for new entries)
- **Campaign timing** (lock profit before event risk)
- **Discretionary judgment** (operator sees market shift PositionManager doesn't)

The `close_trade_ids` field exists in the decision schema but is ignored by the execution path—a **dead field**.

## Contract Violations

Per AGENT_CONTRACT §10:
> **Contradicted trader-owned positions can be closed.**

The decision correctly identified contradicted positions and chose to close them. The verification accepted the decision. But the execution path has an additional undocumented gate: "contradicted positions with negative P&L only."

Per AGENT_CONTRACT §2:
> The operator (Codex or Claude) is the discretionary decision layer.

The operator's explicit close decision was overridden by PositionManager's profit-filter logic.

## Fix Options

### Option A: Route close_trade_ids to PositionManager (Recommended)
Modify PositionManager to accept `close_trade_ids` from the decision receipt:
- If a position's trade_id is in `close_trade_ids`, set `action = ACTION_REVIEW_EXIT`
- Bypasses the profit filter when the operator explicitly chooses to close

**Pros**: Preserves PositionManager's protective logic for automatic exits while honoring explicit close decisions  
**Cons**: Requires passing decision through PositionManager constructor or run() method

### Option B: Make profit filter configurable
Change the hardcoded `position.unrealized_pl_jpy < 0` gate to accept profitable contradicted closes when:
- Operator decision explicitly lists the trade_id, OR
- Portfolio capacity is at limit, OR
- Campaign pressure requires freeing slots

**Pros**: More flexible, handles multiple use cases  
**Cons**: More complex logic, harder to audit

### Option C: Separate close gateway for operator decisions
Create a second execution path for operator-driven closes that bypasses PositionManager:
- `autotrade-cycle --use-gpt-trader` with `action=CLOSE` → direct to PositionProtectionGateway
- PositionManager path remains unchanged for automatic protection

**Pros**: Clean separation of operator vs automatic actions  
**Cons**: Two execution paths to maintain

## Immediate Workaround

Manual intervention required:
1. Operator manually closes trades 470279, 470266 via OANDA UI or API
2. Next cycle detects freed capacity and can proceed with new entries

OR

1. Modify PositionManager line 147 to remove `and position.unrealized_pl_jpy < 0` temporarily
2. Re-run the cycle
3. Restore the profit filter after close executes

## Regression Test Requirements

Before merging a fix:
1. **Failing test**: Trader decision with `action=CLOSE` and `close_trade_ids=[...]` → execution must close those trades
2. **Passing test**: CLOSE decision for profitable contradicted positions → gateway executes close
3. **Passing test**: CLOSE decision for positions not in close_trade_ids → gateway ignores them
4. **Passing test**: PositionManager automatic ACTION_REVIEW_EXIT still works for losing contradicted positions

## Files Modified (for Option A fix)

- `src/quant_rabbit/strategy/position_manager.py`: Accept `close_trade_ids` parameter, set ACTION_REVIEW_EXIT for listed trades
- `src/quant_rabbit/automation.py`: Pass `close_trade_ids` from decision receipt to PositionManager
- `tests/test_position_manager.py`: Add regression tests for close_trade_ids handling

## Campaign Recovery

Once this gap is fixed and the close executes:
- Portfolio: 2/4 positions (GBP_USD LONG, AUD_JPY LONG)
- Realized P&L: +113 JPY additional progress
- Available lanes: Need to resolve STALE_QUOTE (refresh broker snapshot) and other deterministic blockers
- Expected LIVE_READY lanes: EUR_USD SHORT methods (if chart bias shifts back) or other pairs

The critical blocker (PORTFOLIO_POSITION_LIMIT) will clear, allowing the campaign to resume.
