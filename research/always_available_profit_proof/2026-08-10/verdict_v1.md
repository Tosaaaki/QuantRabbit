# Verdict: conditional profit exists; the answer is always available

## What is proved

- The exact `EUR_USD|SHORT|BREAKOUT_FAILURE|LIMIT|HARVEST` vehicle has four distinct frozen OANDA/legacy receipts, four wins, zero losses, and realized after-bid/ask Net **+3,255.0938 JPY**.
- Expectancy is **+813.77345 JPY/trade**; the deterministic bootstrap 95% lower bound is **+292.49165 JPY/trade**.
- Normalized to 1,000 units, expectancy is **+150.31417 JPY** and its lower bound is **+112.45846 JPY**.
- The forward decision function is total and deterministic: every request returns exactly one `TRADE` or `WAIT`. Exhaustive gate combinations and leakage controls are tested.
- Body and wick are not interchangeable. A completed body close confirms failure; side-correct bid/ask wick touch controls LIMIT, TP, and SL execution.

## What is not proved

“Profit can be taken at every time” is not true and is not encoded. The engine returns `WAIT` when the edge or execution evidence is absent. This is how the profitable vehicle is kept separate from the ten historical market-close losses that used a different exit vehicle.

The current forward answer remains `WAIT` for two explicit reasons only: four independent exact receipts are below the frozen 20-sample floor, and four active days are below the 10-day floor. The older 228-sample forecast replay is supporting evidence only because 96.05% of its samples occurred on one day; it is not counted as 228 independent trades.

## How it becomes continuously usable

The same decision function already changes to `TRADE` when all preregistered decision-time gates pass. The remaining work is acquisition, not another parameter search: record 16 more independent exact-vehicle decisions across at least six additional active days, with causal bid/ask, fillability, financing, margin, and unwind evidence. No outcome field is consumed by the decision.

This proof is research-only. Holdout, live, Paper, broker order, and deploy paths were not used.
