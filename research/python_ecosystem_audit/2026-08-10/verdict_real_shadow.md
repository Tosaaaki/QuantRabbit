# Real-cohort OSS adapter shadow verdict

Decision: **retain xarray 2026.7.0, SALib 1.5.2, pymoo 0.6.2 and MAPIE 1.5.0 as research-only adapters. Keep DoWhy and River on HOLD. Do not change the ALL_TRADES strategy baseline.**

This is adapter admission, not strategy admission. The four adapters changed no financial cell and contributed **0 JPY** of incremental profit. Holdout remained unread; live, Paper, broker order and deploy were untouched.

## Frozen real evidence

- Input: 251 OANDA `ACTUAL_AFTER_COST` executed labels, with actual spread/fill slippage implicit, financing explicit, and opportunity cost still missing.
- Chronology: the preregistered 16/32/64-day windows, chronological 60/40 split, close-time purge and one-hour embargo. The respective TRAIN/VALIDATION counts were 13/12, 43/31 and 145/101.
- Feature boundary: Dukascopy bid/ask tick-derived completed M5 features were feature-only. They were present for 141/251 episodes and were never substituted for an OANDA fill.
- The frozen financial report was reproduced within `1e-9` for net, retention and paired LCB. The 64-day VALIDATION ALL_TRADES baseline remained **+15,144.4802 JPY**, PF **1.58525**, max DD **6,794.7768 JPY** and margin-evidence coverage only **14.85%**.

## Adapter results

- **xarray — ADOPT_RESEARCH_ADAPTER.** It reconstructed the 10-axis named cube with populated-value max error **0**, retained **414** null long-table values, and returned NaN for a known absent coordinate. Median measured runtime was about **70 ms** with **35.0 MB** peak Python allocation. This improves organization and auditability only.
- **SALib — ADOPT_RESEARCH_ADAPTER, but no feature promotion.** Only the 64-day TRAIN had enough price-feature rows (114; 31 missing rows stayed excluded). The SALib delta rank and the custom absolute-Spearman rank had agreement **0.3393**; the frozen TRAIN ranking versus VALIDATION ranking was **-0.0893**. The leading TRAIN sensitivity therefore did not remain stable. Median runtime was about **4.00 s**, peak Python allocation **1.15 MB**.
- **pymoo — ADOPT_RESEARCH_ADAPTER.** Its diagnostic Pareto fronts exactly matched the independent dominance oracle in all three windows. With preregistered margin/fill/unwind constraints, every real constrained front was empty because margin evidence was incomplete. On the 64-day diagnostic bootstrap, only ALL_TRADES and its exact fallback duplicate B had front inclusion 1.0; price-action HGB was 0.55. Median runtime was about **0.94 s**, peak Python allocation **0.94 MB**.
- **MAPIE — ADOPT_RESEARCH_ADAPTER, but no decision rule.** Only the 64-day cohort supported a purged inner TRAIN split: 62 fit, 46 conformal and 22 feature-complete VALIDATION rows. MAPIE/manual bound error was **0**. Nominal 90% coverage measured **86.36%**, with mean interval width **9,488.78 JPY** and mean lower bound **-4,507.23 JPY**. It quantifies wide uncertainty and does not justify filtering trades. Median runtime was about **7.1 ms**, peak Python allocation **0.09 MB**.

## Profitability and robustness conclusion

No library reversed the existing strategy conclusion. On the same 64-day VALIDATION cohort:

- ALL_TRADES: **+15,144.4802 JPY**, paired incremental LCB 0 by definition.
- Frozen HGB: **+1,676.3043 JPY**, incremental **-13,468.1759 JPY**, paired LCB **-338.0989 JPY/episode**.
- Price-action HGB: **+10,437.0991 JPY**, incremental **-4,707.3811 JPY**, paired LCB **-142.0678 JPY/episode**.
- The TRAIN-only cost-aware rule fell back exactly to ALL_TRADES and added **0 JPY**; it is not a value-adding model.

The adapter result is robust enough for research use because all four were deterministic, wheel hashes and SBOM matched the fixed checkpoint, xarray/MAPIE/pymoo matched independent oracles, and the stdlib readback passed **33/33** checks. Strategy adoption remains blocked by incomplete decision-time margin conversion, incomplete all-entry counterfactual coverage, unstable sensitivity ranks and wide conformal uncertainty.

Reproduce with:

```bash
research/historical_learning_admission/.venv/bin/python \
  research/python_ecosystem_audit/2026-08-10/run_real_shadow.py
python3 research/python_ecosystem_audit/2026-08-10/verify_real_shadow.py
```

Rollback is bounded per adapter: remove its ignored `.adapter_envs/<candidate>` directory and restore it from the recorded wheel hashes. No runtime dependency lock or production configuration was changed.
