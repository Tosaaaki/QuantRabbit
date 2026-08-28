# M5 EMA state/impulse replay V1

This directory is the only write scope for the experiment. It is offline,
paper-only, and separate from the running zero-order shadow services.

Current checkpoint:

- exact prior historical evidence: 0;
- preregistration: frozen before result execution;
- historical rows decoded by this experiment: 0;
- replay/result artifacts: not created;
- live/broker/account/order/network/credential/launchd/Git authority: none.

`python3 replay_m5_ema_state_impulse.py --audit-only` validates local immutable
contracts without opening candle files. Synthetic unit tests likewise do not
decode the sealed dataset. A result run additionally requires
`INDEPENDENT_REVIEW.json`; that receipt is deliberately not self-issued by this
implementation lane.

Expected review sequence:

1. inspect the JSON preregistration, chronology, stress proxy, statistics and
   pass/fail rule;
2. inspect the runner and focused tests;
3. record an independent receipt that binds all three hashes;
4. only then invoke the explicit one-shot offline replay.
