#!/usr/bin/env python3
"""Build the immutable V5 preregistration before any replay is run."""
from __future__ import annotations

import itertools
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def configurations() -> list[dict]:
    rows = []
    for (
        session,
        mode,
        displacement_q,
        geometry_q,
        breadth,
        activity,
        horizon,
    ) in itertools.product(
        ("LONDON_MIDDAY", "LONDON_FIX"),
        ("ACCEPT_CONTINUATION", "REJECT_FADE"),
        ("Q50", "Q67"),
        ("Q50", "Q67"),
        ("ANY", "MODE_MATCHED"),
        ("ANY", "MODE_MATCHED"),
        (24, 48),
    ):
        config_id = (
            f"{session}__{mode}__D{displacement_q[1:]}__"
            f"G{geometry_q[1:]}__B{breadth}__A{activity}__H{horizon}"
        )
        rows.append(
            {
                "config_id": config_id,
                "session": session,
                "mode": mode,
                "displacement_quantile": displacement_q,
                "geometry_quantile": geometry_q,
                "breadth": breadth,
                "activity": activity,
                "horizon_bars": horizon,
                "selection_eligible": breadth == activity == "MODE_MATCHED",
            }
        )
    rows.sort(key=lambda row: row["config_id"])
    if len(rows) != 128 or len({row["config_id"] for row in rows}) != 128:
        raise RuntimeError("the fixed family must contain exactly 128 unique configs")
    if sum(row["selection_eligible"] for row in rows) != 32:
        raise RuntimeError("the fixed selection subset must contain exactly 32 configs")
    return rows


def build() -> dict:
    return {
        "schema": "QR_FX_SESSION_BREAK_RESPONSE_PREREGISTRATION_V1",
        "candidate_id": "FX_SESSION_BREAK_RESPONSE_SURFACE_V5",
        "status": "PREREGISTERED_RESEARCH_ONLY",
        "authority": {
            "offline_only": True,
            "network_attempts_allowed": 0,
            "credential_reads_allowed": 0,
            "broker_mutation_allowed": False,
            "external_orders_allowed": 0,
            "launchd_actions_allowed": 0,
            "git_actions_allowed": 0,
        },
        "input": {
            "provider": "OANDA_V20_LIVE_CANDLES_IMMUTABLE_CAPTURE",
            "dataset_root": "research/oanda_history_capture/2026-08-28-v1/runs/oanda-live-m5-ba-730d-20260828T040500Z-40b27f46ae63",
            "canonical_dataset_sha256": "721904751fc1d590a64c7cefd0a533e7df314f043b10783c116d2a82793f14fb",
            "manifest_sha256": "3408963dce76f6c2da5be7f766a48b4e1a91b3fbd03d7082f5369f8e1f2a4a00",
            "gap_report_sha256": "95e7f222a0579a7339db0c35e8299af9ca2c139425161638176e1eea6dbc32e7",
            "symbols": ["EUR_USD", "USD_JPY", "AUD_USD"],
            "granularity": "M5",
            "price_component": "BID_ASK",
            "decoder_exclusive_end_utc": "2025-08-28T04:05:00.000000Z",
            "discovery_prefix_contract": {
                "AUD_USD": {
                    "path": "data/AUD_USD_M5_BA.jsonl",
                    "exclusive_byte_offset": 20259438,
                    "prefix_sha256": "2c36f2888ea1419b479c5c15f00eb32cdc262554bc9758530e2641147fd37fb1",
                    "prefix_rows": 55566,
                },
                "EUR_USD": {
                    "path": "data/EUR_USD_M5_BA.jsonl",
                    "exclusive_byte_offset": 20273794,
                    "prefix_sha256": "850f0cb01a2f75e1e3320c30df1efd2700a292b73a1cc305869ec5bb5f52a9e2",
                    "prefix_rows": 55576,
                },
                "USD_JPY": {
                    "path": "data/USD_JPY_M5_BA.jsonl",
                    "exclusive_byte_offset": 20287549,
                    "prefix_sha256": "fb32e71eb8d539a96ffe4e4cdffbca3caba7054f41dfa3546589f3dd7aa23502",
                    "prefix_rows": 55560,
                },
            },
            "byte_prefix_contract": {
                "AUD_USD": {
                    "path": "data/AUD_USD_M5_BA.jsonl",
                    "exclusive_byte_offset": 27186414,
                    "prefix_sha256": "79d959900555cc3b52ac1f759437bf4998f03c383e31629598355bc9250cd894",
                    "prefix_rows": 74568,
                },
                "EUR_USD": {
                    "path": "data/EUR_USD_M5_BA.jsonl",
                    "exclusive_byte_offset": 27205812,
                    "prefix_sha256": "0d1e76cad0ed725b05da593df241fba87549710d8ac15eb7478460fc808ad7f5",
                    "prefix_rows": 74582,
                },
                "USD_JPY": {
                    "path": "data/USD_JPY_M5_BA.jsonl",
                    "exclusive_byte_offset": 27222596,
                    "prefix_sha256": "bb102f9ff123fa45c2b248a851fbe073e3c33e0f1ff4d446b97bd157a748e4e2",
                    "prefix_rows": 74562,
                },
            },
            "post_boundary_price_or_volume_decode_allowed": False,
            "post_boundary_label_computation_allowed": False,
        },
        "splits": {
            "calibration": {
                "from_utc": "2024-08-28T04:05:00.000000Z",
                "to_utc": "2024-11-28T04:05:00.000000Z",
                "use": "threshold calibration only",
            },
            "discovery": {
                "from_utc": "2024-11-28T04:05:00.000000Z",
                "to_utc": "2025-05-28T04:05:00.000000Z",
                "use": "fixed-budget selection among 32 MODE_MATCHED/MODE_MATCHED configs",
            },
            "locked_internal_validation": {
                "from_utc": "2025-05-28T04:05:00.000000Z",
                "to_utc": "2025-08-28T04:05:00.000000Z",
                "use": "one locked winner; no parameter reselection",
            },
            "opened_development": {
                "from_utc": "2025-08-28T04:05:00.000000Z",
                "to_utc": "2026-05-28T04:05:00.000000Z",
                "decode_allowed": False,
            },
            "untouched_holdout": {
                "from_utc": "2026-05-28T04:05:00.000000Z",
                "to_utc": "2026-08-28T04:05:00.000000Z",
                "decode_allowed": False,
            },
        },
        "chronology": {
            "completed_data_only": True,
            "expected_step_seconds": 300,
            "candle_timestamp_semantics": "M5 open; complete bar closes five minutes later",
            "decision": "event-window final bar close",
            "entry": "first exact executable M5 BID/ASK open strictly later than decision; the open equal to decision time is forbidden",
            "exit": "fixed exact M5 open H24 or H48 bars after entry; no TP and no price SL",
            "gap_rule": "missing required rail/event/entry/path/exit/conversion timestamp makes that signal unavailable; no synthesis or forward search",
            "both_rails_touched": "AMBIGUOUS_NO_SIGNAL",
        },
        "sessions": {
            "timezone": "Europe/London",
            "LONDON_MIDDAY": {
                "reference": "same UTC date 00:00-05:55 UTC",
                "event": "same Europe/London local date 08:00-11:55",
            },
            "LONDON_FIX": {
                "reference": "same Europe/London local date 08:00-11:55",
                "event": "same Europe/London local date 12:00-15:55",
            },
        },
        "indicator": {
            "epsilon": 1e-12,
            "midpoint": "arithmetic midpoint of BID/ASK for every OHLC field",
            "rail": "U=max midpoint high in reference; L=min midpoint low in reference; R=log(U/L)>0",
            "event_returns": "r_k=log(C_k/C_(k-1)) inside the completed event path, with the event open as C_(-1)",
            "path_efficiency": "PE=abs(sum(r_event))/(sum(abs(r_event))+epsilon)",
            "accept_displacement": "D_accept=abs(log(C_event/O_event))/(R+epsilon)",
            "reject_displacement": "D_reject=abs(log(X_break_extreme/O_event))/(R+epsilon), X=H_event for upper and L_event for lower",
            "accept_settle": "upper clip((C_event-U)/(H_event-U+epsilon),0,1); lower clip((L-C_event)/(L-L_event+epsilon),0,1)",
            "accept_persist": "count(last six completed closes beyond the broken rail)/6",
            "accept_geometry": "G_A=(PE*settle_A*persist_A)^(1/3)",
            "reject_settle": "upper clip((H_event-C_event)/(H_event-U+epsilon),0,1); lower clip((C_event-L_event)/(L-L_event+epsilon),0,1)",
            "reject_reverse": "reverse_R=clip(-b*sum(last3 returns)/(sum(abs(last3 returns))+epsilon),0,1), b=+1 upper/-1 lower",
            "reject_geometry": "G_R=((1-PE)*settle_R*reverse_R)^(1/3)",
            "accept_structure": "event pierces exactly one rail and the final two completed closes settle beyond that rail",
            "reject_structure": "event pierces exactly one rail, final close returns strictly inside the rail, and the last-three-return sum is opposite break side",
            "direction": "ACCEPT_CONTINUATION follows break side; REJECT_FADE opposes break side",
            "usd_orientation_q": {"EUR_USD": -1, "AUD_USD": -1, "USD_JPY": 1},
            "breadth_component": "u_i=q_i*log(C_event_i/O_event_i)/(R_i+epsilon)",
            "breadth": "B=abs(sum(u_i))/sum(abs(u_i)); all three exact session observations required",
            "breadth_mode_matched": "ACCEPT requires B>=session calibration Q50 and q_pair*break_side=sign(sum(u_i)); REJECT requires B<session calibration Q50",
            "volume_semantics": "OANDA_PRICE_COUNT_NOT_TRADED_VOLUME; activity proxy only, not traded volume or order flow",
            "activity": "V=sum OANDA price-count volume over event; M=median(V over valid calibration days for same pair/session); A=V/(M+epsilon)",
            "activity_mode_matched": "ACCEPT A>=pair/session A_Q50; REJECT A_Q25<=A<A_Q50",
            "cost_is_entry_gate": False,
        },
        "calibration": {
            "displacement_quantiles": [0.50, 0.67],
            "minimum_displacement_threshold": 1.0,
            "geometry_quantiles": [0.50, 0.67],
            "threshold_scope": "session x structural mode, pooled across the three preregistered pairs",
            "minimum_structural_events_per_session_mode": 24,
            "breadth_q": 0.50,
            "minimum_common_breadth_days_per_session": 45,
            "activity_quantiles": [0.25, 0.50],
            "minimum_valid_activity_days_per_pair_session": 45,
            "pooling_on_minimum_failure": False,
        },
        "family": {
            "size": 128,
            "dimensions": {
                "session": ["LONDON_MIDDAY", "LONDON_FIX"],
                "mode": ["ACCEPT_CONTINUATION", "REJECT_FADE"],
                "displacement_quantile": ["Q50", "Q67"],
                "geometry_quantile": ["Q50", "Q67"],
                "breadth": ["ANY", "MODE_MATCHED"],
                "activity": ["ANY", "MODE_MATCHED"],
                "horizon_bars": [24, 48],
            },
            "configs": configurations(),
        },
        "execution_arms": {
            "shared_signal_id_and_lineage": True,
            "RAW_SIGNAL": "mid open to mid open; no spread/fee/slippage",
            "EXECUTABLE_BASE": "same path; observed ask/bid plus 0.3 pip slippage per side",
            "ADVERSE_STRESS": "same path; observed ask/bid plus 0.9 pip slippage per side",
            "fee_pips_per_side": 0.0,
            "cost_gate": False,
            "units": 1000,
            "terminal_liquidation_and_mtm": True,
            "exact_time_jpy_conversion": True,
            "latency_sensitivity": "+5 minutes to entry and identical H-bar holding interval",
            "latency_sensitivity_bars": 1,
            "base_slippage_pips_per_side": 0.3,
            "adverse_slippage_pips_per_side": 0.9,
        },
        "selection": {
            "data": "discovery RAW_SIGNAL only",
            "eligible_configs": "the 32 MODE_MATCHED/MODE_MATCHED configs only",
            "fixed_budget": 128,
            "cluster": "UTC decision day, common USD-linked resamples across every pair/config",
            "bootstrap": "deterministic common five-day moving-block bootstrap",
            "bootstrap_resamples": 10000,
            "bootstrap_block_days": 5,
            "bootstrap_seed": 20260828,
            "multiplicity": "one-sided 95% max-T FWER simultaneous LCB across all 128 configs",
            "density_floor": {
                "trades": 96,
                "active_utc_days": 48,
                "pairs_meeting_floor": 2,
                "trades_per_qualifying_pair": 24,
            },
            "rank_order": [
                "RAW corrected LCB descending",
                "RAW mean descending",
                "worst-pair RAW mean descending",
                "positive anchored-month fraction descending",
                "N_eff descending",
                "config_id ascending",
            ],
        },
        "validation": {
            "winner_locked_before_validation_decode": True,
            "parameter_reselection": False,
            "bootstrap": "one-sided 95% percentile LCB, same deterministic common five-day moving-block design",
            "interaction_daily_capacity": "three preregistered pair slots per session-day; absent signals contribute zero and each arm is divided by 3 before differencing",
            "density_floor": {
                "trades": 48,
                "active_utc_days": 24,
                "pairs_meeting_floor": 2,
                "trades_per_qualifying_pair": 12,
            },
            "pass": [
                "RAW mean pips > 0",
                "RAW one-sided 95% LCB > 0",
                "daily capacity-normalized interaction difference LCB versus exact ANY/ANY ablation > 0",
                "at least 2 of 3 pairs have positive RAW mean",
                "at least 2 of 3 anchored validation months have positive RAW mean",
                "density floor passes",
                "no ruin, nonfinite accounting, leverage fitting, terminal inventory omission, or hard-guard breach",
            ],
            "base_and_adverse": "post-classification only; never changes winner",
        },
        "portfolio_and_reporting": {
            "initial_equity_jpy": 200000.0,
            "units_per_signal": 1000,
            "gross_leverage_observation_cap": 20.0,
            "leverage_fitting": False,
            "martingale": False,
            "price_stop_loss": False,
            "financing_model": "zero over maximum four-hour horizon; limitation disclosed",
            "break_even_round_trip_cost": "c_star_pips=RAW mean pips",
            "required": [
                "gross/net/adverse expectancy and cost drag",
                "direction accuracy, MFE/MAE, counts and N_eff",
                "pair and anchored-month stability",
                "plus-five-minute latency sensitivity",
                "gap/unpriceable/terminal liquidation counts and terminal MTM",
                "drawdown, ruin, leverage observation and hard guards",
            ],
        },
    }


def markdown(prereg: dict) -> str:
    return """# FX Session Break Response Surface V5 — frozen preregistration

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
"""


def main() -> int:
    prereg = build()
    (ROOT / "PREREGISTRATION.json").write_text(
        json.dumps(prereg, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (ROOT / "PREREGISTRATION.md").write_text(markdown(prereg), encoding="utf-8")
    print(json.dumps({"configs": len(prereg["family"]["configs"]), "status": prereg["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
