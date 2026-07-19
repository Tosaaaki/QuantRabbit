# DOJO AI-supervised worker tuning — 2026-07-19

AI trainers read the losing worker ledgers, changed bounded strategy parameters,
and sent the candidates back through actual multi-pair DOJO replay.  Every row
is worn-history TRAIN evidence only; none is historical holdout, prospective
proof, promotion, or live authority.

The strongest batch is spike-fade tail guarding.  Adding a finite 25-pip stop
removed the weekend-gap/no-stop tail that dominated the first run.  Both fixed
intrabar paths were positive for two configurations.  The lower-leverage,
four-hour version is the safer research candidate; its pessimistic path made
JPY 17,351.15 on JPY 200,000, with 1.58% realized drawdown and 32% configured
peak margin.  This is a TRAIN survivor, not evidence that it will repeat.

The pullback trainer found a much smaller positive hypothesis: JPY 170.24 on
the pessimistic path and a 1.000912 calendar-30-day multiple.  Momentum tuning
remained negative.  The first four-type diversity run is retained as invalid
because its old fill path allowed same-bar resting orders to breach pair/global
concurrency caps.  A separate post-fix rerun enforced those caps at every fill;
all four types still lost after costs, so none survived TRAIN.

The AI trainer then selected the least-negative round-number fade and changed
one non-risk mechanism only: take profit from 2.0 ATR to 3.0 ATR.  On a separate
worn TRAIN interval the two paths lost JPY 1,719.70 and JPY 1,244.79.  The
hypothesis is rejected: wider winners did not compensate for the lower hit rate.

Capital occupancy was tested on one fixed four-pair spike-fade stream.  The
60-minute stale-position release beat both 480-minute full HOLD and 5x split
reserve on both intrabar paths.  On the pessimistic path it made JPY 166,990.89
profit from JPY 200,000, ending at JPY 366,990.89, versus JPY 90,671.91 for
full HOLD. The replay ran only from 2025-03-03 00:00 through 2025-03-14 21:59
UTC (11 days 21 hours 59 minutes), not one month. Realized drawdown fell from 10.47% to
9.30%.  The split policy had zero margin rejects and only 4.25% drawdown, but
left too much capacity idle and made JPY 43,704.45.  These 30-day figures are
mechanical compound extrapolations from that short worn-TRAIN
period, not observed monthly results or monthly proof. Drawdown is measured at
realized exit events, not continuously marked to market.

All valid runs used OANDA M1 bid/ask, OHLC and OLHC paths, declared slippage and
financing, owner isolation, margin accounting, and terminal settlement.
