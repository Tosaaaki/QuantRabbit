# Vol‑Squeeze Breakout Worker
- Identify 'squeeze' via Bollinger Band width percentile within a local window.
- When squeezed, trigger on Keltner (EMA ± k*ATR) breakout in either direction.
- Use ATR‑based stops/targets via `ExitManager` if available.
