# Vol‑Squeeze Breakout Worker
- Identify 'squeeze' via Bollinger Band width percentile within a local window.
- When squeezed, trigger on Keltner (EMA ± k*ATR) breakout in either direction.
- 実運用では各戦略専用の exit_worker で SL/TP/トレイルを実装する（本リポの ExitManager はスタブで自動EXITなし）。
