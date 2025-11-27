# Session Strategy Roadmap (London Momentum / NY VWAP)

_Last updated: 2025-11-11_

## 1. London Momentum (LMO)
### Objective
Capture 07:00–11:30 UTC directional runs when macro bias aligns with early London order flow.

### Status
- Worker scaffold implemented (`workers/london_momentum`).
- Configurable via `LONDON_MOMENTUM_ENABLED` env (default False).
- Uses H1 EMA20/EMA50 gap + M5 momentum and ATR>5p to gate entries.

### Next Steps
1. **Tick replay validation**: Add worker to `scripts/replay_workers.py` to test with Oct tick logs (P1).
2. **Pocket affinity**: When `macro_bias >= 0.4`, keep using macro pocket; otherwise fallback to micro. Requires hooking into `analysis.macro_state`.
3. **Performance metrics**: instrument log metrics (entries, win rate, average hold) and send to `analytics/realtime_metrics_client`.
4. **Session gating**: integrate event calendar to skip high-impact releases (CPI/NFP) automatically.

### Env Flags
```ini
LONDON_MOMENTUM_ENABLED=true
LONDON_MOMENTUM_POCKET=macro
LONDON_MOMENTUM_START_UTC=06:45
LONDON_MOMENTUM_END_UTC=11:30
LONDON_MOMENTUM_RISK_PCT=0.015
```

---

## 2. NY VWAP Reversion (pending)
### Requirements
- Time window 12:30–20:00 UTC.
- Signal: price deviates ≥12p from 90m VWAP + RSI extreme.
- Spread <=1.6p, event skip, risk_pct ≈0.012.

### Tasks
1. Build worker similar to LMO (planned after LMO soak).
2. Provide helper to compute rolling VWAP from latest M1 candles.
3. Replay/test on early Nov data.

---

## 3. Macro / Micro Integration
- Auto-refresh fix deployed (11/10). Need to monitor `[MACRO] snapshot refreshed` logs and confirm macro trades resume (currently 0 entries since 11/6).
- Once macro live, enable London Momentum via env and monitor risk (lot) impact.

