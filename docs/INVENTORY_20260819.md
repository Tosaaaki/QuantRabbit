# QuantRabbit inventory — 2026-08-19

Everything that was built, what each thing measured, and what the account
actually did. Ground truth is the OANDA transaction stream (read-only GET,
transactions 470222 → 473327), not any research artifact.

## 0. One line

Across ~2,100 commits, 72 runtime modules, 34 research programmes and 955 Codex
sessions since May, **the only component that has ever produced positive money
is the operator's manual entries** — and what took it back was not the entries
but **forced liquidation**: 17 margin closeouts, −193,151 JPY, against
+170,769 JPY on the other 38 closes. Right now the account sits at **93% margin
utilisation** (`marginCloseoutPercent 0.93`; OANDA closes at 1.00) with a
35,000-unit USD_JPY short carrying −14,700 unrealised and no stop.

## 1. Realised P&L, whole record (JPY)

| period | engine | closes | win | realised | note |
|---|---|---|---|---|---|
| 2026-03 → 04 | Claude discretionary trader (v8) | 301 | 34–60% | **−21,609** | legacy log; Apr 253 closes at 34% |
| 2026-05-06 → 07-16 | `failure_trader` | 143 | 58% | −15,671 | PF 0.73 |
| | `range_trader` | 56 | 64% | −13,777 | PF 0.41 |
| | `trend_trader` | 39 | 54% | −4,440 | PF 0.65 |
| | **bots subtotal** | 238 | | **−33,888** | |
| | manual (MARKET) | 19 | 89% | **+11,405** | PF 2.25 |
| 2026-07-16 → 08-18 | manual, non-forced closes | 38 | 73–100% | **+170,769** | 18 TP hits +145,444 |
| | manual, **margin closeouts** | 17 | 6% | **−193,151** | 07-31 ×6, 08-07 ×9, 08-10 ×2 |
| 2026-07-16 → today | bots | 0 | | 0 | live dead since 07-22 |
| 2026-06-10 → 06-23 | DecaBot (account -003) | 2 | | ≈0 | idle since 06-23 |

Bots lifetime: **−55.5k over ~540 closes**, never a positive month.
Manual lifetime: **+182k gross on 57 closes, −193k in forced liquidation.**

The three closeout days: 07-30 realised **+143,494** (TP +111,650 on a
50,000-unit USD_JPY long); 07-31 **−132,884** (six closeouts on 50,000-unit
positions); 08-10 **−76,200** (two closeouts, −45,720 and −30,480). Median
manual size 2,000 units; the closeouts happened at 40,000–50,000.

## 2. What was built, and what it measured

### Runtime (`src/quant_rabbit`, 72 modules)

| component | verdict | evidence |
|---|---|---|
| `failure_trader` / `range_trader` / `trend_trader` lanes | **losing, all three** | table above; PF 0.41–0.73 |
| fast_bot / episode / learning-shadow stack | shadow only, never live | 47-day scoreboard: capital inversely allocated to edge; M1Scalper +307.5 pips at 11 JPY/pip; pullback_s5 −37,661 in 3 days |
| regime classifier | **fail** | MIXED stuck (state-machine bug); fixed version −0.49p; entry-side conditioning ceiling +0.8p |
| position guardian / disaster stop | **not loaded** | `launchctl` shows no QR services; plist present, service absent |
| thesis ledger / horizon expiry | plugged a −78k exit leak in June | 12h+ dumps 22/22 losers → THESIS_EXPIRED gate |
| AI regime supervision (`qr-trader` automation) | **PAUSED** since 07-31 | order authority NONE by contract; nothing to supervise |
| self-improvement watch / hole audit / news digest | **PAUSED** since 07-31 | |
| execution ledger (1.29 GB) | working | 39,552 execution events, 222k verification observations |
| spread calibration (`config/oanda_spread_calibration_v1.json`) | **expired 08-13T15:00Z** | import-time constant → `OandaReadOnlyClient` unconstructable repo-wide; 131 of 202 test files fail collection |

### Research (34 programmes in `203e`, 25 here) — survivors and closures

| line | result | status |
|---|---|---|
| `mom_break@2880` × EFF20 quiet-side gate | +2.93 pips/day, passed 3-stage independent test | **only survivor of the L001–L028 sweep**; never taken forward |
| exact-S5 causal weekend gate | TRAIN +1,089p / VAL +970p, no significance | one lock; Jul20–Aug3 forward test was pending; M5 acquisition blocked on validator |
| `system_v1` (26-pair carry top-8, no SL) | TEST efficiency 0.423 | positive but month-3× needs 2.0; carry decaying (USD_JPY 2025 +2.8%); swap table was 2.6× overstated |
| triangular arbitrage | 100% OOS convergence | **108× too small** to matter |
| non-price input: calendar | effect exists | book does not improve |
| structural stops vs plain distance | no difference | closed |
| TP retention | recovers 99% of fills | 1% tail inseparable from dips |
| maker inventory (hold / invert) | both lose to the same faster flow | closed |
| hedging (same-pair / cross-pair) | same-pair = stop with 2× cost; cross-pair kills DD **and** profit | closed |
| frame sweep (60k cells) | conditions never cross the frame; TEST −0.21, p=0.76 vs null | closed |
| bot scoreboard 47 days | strategy-level STOP is 4× more valuable than GO | led to `QR_AI_STRATEGY_ALLOCATION_V1` (unimplemented) |
| contextual candidate engine (08-13) | gross **−0.03 pips, 49.2% > 0**; feasible-cell CI [−0.80, +0.23] | **closed 08-19**; corpus was 10.5 h re-scored for 6 days |
| operator method reproduction | `NOT_EVALUABLE`: **0 contemporaneous decision labels** | recorder now exists (`tools/capture_decision.py`); 0 capsules so far |

Closed hypotheses share one shape: an entry-side signal measured against a
cost floor of 0.8–1.7 pips (majors) to 8.3 pips (GBP_NZD) with a leverage cap
of 25×. Efficiency = (monthly rate − 1) / max DD is leverage-invariant; best
measured 0.423, sustained estimate 0.02; monthly 2× needs 4.0.

## 3. Where the money actually went

Manual, non-forced: **+182k on 57 closes**, ~90% win, TP-driven, mostly
USD_JPY at 2,000 units.

Forced: **−193k on 17 closes**, all at 40,000–50,000 units, all `noSL`, all
under the SL-free doctrine. With no stop and full leverage the broker's margin
call *is* the stop — placed at the worst price by construction, on the day
after the largest win.

The bots never had this problem because they never made enough to lose it.

## 4. Present exposure (read 2026-08-19 03:00Z)

| | |
|---|---|
| NAV | 247,596 |
| margin used | 230,471 (**93%**) |
| margin available | 17,272 |
| `marginCloseoutPercent` | **0.93** (closeout at 1.00) |
| open | USD_JPY −35,000 @ 158.932, uPL −14,700, TP set, **no SL** |
| | EUR_USD −1,000 @ 1.15754, uPL −144, TP set, no SL |
| position guardian | not loaded |

This is the same configuration as 07-31 and 08-10.

## 5. What this inventory rules in and out

Out: every bot lane that has traded live (all negative), pair selection as a
rescue for the contextual engine (gross was zero), regime switching, hedging,
frame widening, structural stops, and — as a target — monthly 2× (needs
efficiency ~10× the best ever measured).

In, with evidence: the operator's entries. Not because they are proven — 57
closes, one instrument, six weeks — but because they are the only line in
the whole record with a positive gross, and the loss on that line is not from
being wrong about direction. It is from size.

Never tested: the operator's method as a rule (0 labels — see §2), longer
horizons, non-price inputs beyond the calendar.

## 6. The recommendation, and its cost

Stop searching for a signal. The next hundred hours of research on the entry
side buys, at best, another `mom_break@2880`: +2.93 pips/day against a cost
floor that eats most of it.

The one change with a measured effect on the realised line is a **hard
size/margin cap on manual entries** — the account already contains the
counterfactual: +170,769 with it, −22,382 without it. This is not a signal, a
bot, or a strategy. It is a rule about the second-largest number in the ledger.

Two ways to enforce it, in order of intrusiveness:

1. **Broker-side leverage/margin ceiling** on the sub-account — no code, no
   agent, no automation to pause. Enforced by OANDA before any order fills.
2. A read-only monitor that reports `marginCloseoutPercent` and distance to
   1.00 on every manual open — an alarm, not an actor.

Both are decisions about the operator's own account. Neither is something the
runtime should do unasked, and neither is a trade.

## Sources

- OANDA transactions 470222–473327 (execution ledger to 07-16; live GET after)
- `data/legacy_history.db` `live_trade_events` (Mar–Apr 2026)
- `~/.codex/automations/qr-*/automation.toml` (status, updated_at)
- memory: `project_scalp_carry_verdict_20260809`, `project_frame_sweep_20260807`,
  `project_search_state_20260806`, `project_bot_scoreboard_20260805`,
  `project_regime_verdict_20260805`, `project_hedge_verdict_20260809`,
  `project_exact_s5_causal_weekend_gate_20260718`, `project_exit_leak_repair_20260612`
- this session: `docs/contextual_candidate_close_20260819.md`,
  `docs/DECISION_CAPSULE_RECORDER.md`
