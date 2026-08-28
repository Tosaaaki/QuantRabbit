# FX Session Break Response Surface V5 — frozen preregistration

Candidate: `FX_SESSION_BREAK_RESPONSE_SURFACE_V5`. This is an offline,
paper-only experiment. Network, credential, launchd, broker mutation, external
orders, and Git actions are all outside the runner's authority.

## Frozen data boundary and splits

The runner verifies immutable OANDA M5 BID/ASK dataset seal
`721904751fc1d590a64c7cefd0a533e7df314f043b10783c116d2a82793f14fb`
and reads only byte prefixes ending before `2025-08-28T04:05:00Z`.
Calibration is 2024-08-28→2024-11-28, discovery is
2024-11-28→2025-05-28, and locked internal validation is
2025-05-28→2025-08-28. Post-boundary development and holdout values and labels
must decode zero times. The winner is selected on discovery before validation
events are decoded.

## Frozen structural equations

All OHLC values are BID/ASK midpoints and `epsilon=1e-12`. For reference rail
`[L,U]`, `R=log(U/L)>0`; break side `b=+1` for upper and `b=-1` for lower.
Event returns are `r_k=log(C_k/C_(k-1))`, starting from event open, and
`PE=abs(sum r)/(sum abs r+epsilon)`.

- ACCEPT: `D=abs(log(C_event/O_event))/(R+epsilon)`. Upper settle is
  `clip((C_event-U)/(H_event-U+epsilon),0,1)` and lower settle is
  `clip((L-C_event)/(L-L_event+epsilon),0,1)`. Persist is the fraction of the
  last six completed closes beyond the broken rail. `G=(PE*settle*persist)^(1/3)`.
  Structure additionally requires the last two completed closes beyond exactly
  one pierced rail; direction follows the break.
- REJECT: `D=abs(log(X_break_extreme/O_event))/(R+epsilon)`, using event high
  for upper and event low for lower. Upper settle is
  `clip((H_event-C_event)/(H_event-U+epsilon),0,1)` and lower settle is
  `clip((C_event-L_event)/(L-L_event+epsilon),0,1)`. Reverse is
  `clip(-b*sum(last3 returns)/(sum(abs(last3 returns))+epsilon),0,1)` and
  `G=((1-PE)*settle*reverse)^(1/3)`. Structure requires exactly one rail pierce,
  final close strictly back inside, and last-three return opposite the break;
  direction fades the break.

Both rails touched, ambiguous structure, or any required timestamp gap emits
no signal. Displacement thresholds are the mode/session calibration Q50/Q67
floored at one; geometry thresholds are mode/session Q50/Q67.

USD breadth uses `q=-1` for EUR_USD/AUD_USD and `q=+1` for USD_JPY,
`u_i=q_i*log(C_event_i/O_event_i)/(R_i+epsilon)`, and
`B=abs(sum u)/sum(abs u)`. MODE_MATCHED ACCEPT requires B at or above the
session calibration Q50 and the pair break aligned with the common USD sign;
MODE_MATCHED REJECT requires B below Q50.

Activity is explicitly an OANDA price-count proxy, not traded volume or true
order flow. For each pair/session, `V=sum(event volume)`,
`M=median(calibration V)`, and `A=V/(M+epsilon)`. MODE_MATCHED ACCEPT requires
`A>=A_Q50`; MODE_MATCHED REJECT requires `A_Q25<=A<A_Q50`.

## Fixed family, execution, and evidence

Exactly 128 configs cross two sessions, two modes, D Q50/Q67, G Q50/Q67,
breadth ANY/MODE_MATCHED, activity ANY/MODE_MATCHED, and H24/H48. Only the 32
MODE_MATCHED/MODE_MATCHED configs are selectable; all others are ablations.
Signals ignore costs. Each signal has one lineage across RAW midpoint,
EXECUTABLE_BASE observed BID/ASK plus 0.3 pip per side, and ADVERSE_STRESS plus
0.9 pip per side. Entry is the first exact M5 open strictly later than the
completed event decision. Exit is the exact open H bars later. No TP or price
SL is used; terminal inventory is liquidated and marked.

Discovery uses 10,000 deterministic common USD-linked UTC-day five-day moving
block resamples and a one-sided max-T 95% FWER LCB across all 128 configs.
Ranking and density floors are frozen in `PREREGISTRATION.json`. Validation
applies the locked winner once, including a daily capacity-normalized
interaction LCB versus its exact ANY/ANY ablation. BASE and ADVERSE results are
post-classification only and never choose or rescue the winner.
