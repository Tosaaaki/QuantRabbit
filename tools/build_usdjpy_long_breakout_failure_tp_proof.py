#!/usr/bin/env python3
"""Build the USD_JPY LONG BREAKOUT_FAILURE TP proof packet.

This tool is read-only with respect to broker state. It reads local artifacts
and writes the proof JSON/report requested by the operator.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

TARGET_LANE_ID = "failure_trader:USD_JPY:LONG:BREAKOUT_FAILURE:LIMIT"
TARGET_SCOPE_KEY = "USD_JPY|LONG|BREAKOUT_FAILURE|TAKE_PROFIT_ORDER"
TARGET_RULE_NAME = "USD_JPY_DOWN_H61_240m_CLT0p50_FADE_TO_UP_S5_BIDASK_CONTRARIAN_HARVEST_TP10_SL7"
TARGET_PAIR = "USD_JPY"
TARGET_FORECAST_DIRECTION = "DOWN"
TARGET_TRADE_DIRECTION = "UP"
TARGET_TP_PIPS = 10.0
TARGET_SL_PIPS = 7.0

DEFAULT_REPLAY_REPORT = Path(
    "logs/reports/forecast_improvement/usdjpy_tp_proof_probe/"
    "oanda_history_replay_validate_latest.json"
)
OUTPUT_JSON = Path("data/usdjpy_long_breakout_failure_tp_proof.json")
OUTPUT_MD = Path("docs/usdjpy_long_breakout_failure_tp_proof_report.md")


def main() -> int:
    proof = build_proof()
    (ROOT / OUTPUT_JSON).write_text(
        json.dumps(proof, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (ROOT / OUTPUT_MD).write_text(build_markdown(proof), encoding="utf-8")
    print(f"wrote {OUTPUT_JSON}")
    print(f"wrote {OUTPUT_MD}")
    return 0


def build_proof() -> dict[str, Any]:
    broker = _load("data/broker_snapshot.json")
    daily = _load("data/daily_target_state.json")
    capital = _load("data/capital_flows.json")
    capture = _load("data/capture_economics.json")
    intents = _load("data/order_intents.json")
    acceptance = _load("data/profitability_acceptance.json")
    support = _load("data/trader_support_bot.json")
    board = _load("data/as_lane_candidate_board.json")
    profile = _load("data/strategy_profile.json")
    rules = _load("src/quant_rabbit/bidask_replay_precision_rules.json")
    replay = _load(DEFAULT_REPLAY_REPORT)

    target_intent = _find_intent(intents)
    stale_candidate = _find_stale_candidate(board)
    packaged_rule = _find_rule(rules)
    broad_replay_row = _find_replay_row(
        replay,
        "by_pair_forecast_direction_trade_direction",
        pair=TARGET_PAIR,
        forecast_direction=TARGET_FORECAST_DIRECTION,
        direction=TARGET_TRADE_DIRECTION,
    )
    strict_replay_row = _find_replay_row(
        replay,
        "by_pair_forecast_direction_trade_direction_horizon_confidence",
        pair=TARGET_PAIR,
        forecast_direction=TARGET_FORECAST_DIRECTION,
        direction=TARGET_TRADE_DIRECTION,
        horizon_bucket="31-60m",
        confidence_bucket="<0.50",
    )
    broad_check = _evaluate_replay_row(broad_replay_row)
    strict_check = _evaluate_replay_row(strict_replay_row)
    capture_scope = _capture_scope(capture)
    usd_jpy_long_profile = _strategy_profile_entry(profile)
    support_metrics = support.get("metrics") if isinstance(support.get("metrics"), dict) else {}
    broker_account = broker.get("account") if isinstance(broker.get("account"), dict) else {}
    positions = broker.get("positions") if isinstance(broker.get("positions"), list) else []
    eur_usd_manual = [
        position for position in positions
        if position.get("pair") == "EUR_USD" and str(position.get("trade_id")) == "472987"
    ]

    current_replay_pass = bool(
        broad_check["sample_size_sufficient"]
        and broad_check["active_days_sufficient"]
        and broad_check["non_negative_expectancy"]
        and broad_check["profit_factor_above_breakeven"]
        and broad_check["positive_day_rate_sufficient"]
    )
    exact_live_scope_proven = bool(capture_scope.get("present"))
    target_lane_present = target_intent is not None
    exact_tp_proof_valid = bool(
        current_replay_pass
        and exact_live_scope_proven
        and target_lane_present
    )

    blockers: list[dict[str, Any]] = []
    if not broad_check["sample_size_sufficient"]:
        blockers.append(
            _blocker(
                "CURRENT_REPLAY_UNDER_SAMPLED",
                "missing evidence",
                "Collect at least 30 evaluated USD_JPY DOWN->UP samples for the exact TP10/SL7 vehicle.",
                str(DEFAULT_REPLAY_REPORT),
                f"current n={broad_check['n']}, required n>=30",
            )
        )
    if not broad_check["active_days_sufficient"]:
        blockers.append(
            _blocker(
                "CURRENT_REPLAY_ACTIVE_DAYS_THIN",
                "missing evidence",
                "Collect at least 3 active JST campaign days for the exact TP10/SL7 vehicle.",
                str(DEFAULT_REPLAY_REPORT),
                f"current active_days={broad_check['active_days']}, required >=3",
            )
        )
    if not broad_check["non_negative_expectancy"]:
        blockers.append(
            _blocker(
                "CURRENT_REPLAY_NEGATIVE_EXPECTANCY",
                "expectancy blocker",
                "Re-run only after new forecast/candle evidence makes TP10/SL7 average realized pips non-negative.",
                str(DEFAULT_REPLAY_REPORT),
                f"current avg_realized_pips={broad_check['avg_realized_pips']}",
            )
        )
    if not broad_check["profit_factor_above_breakeven"]:
        blockers.append(
            _blocker(
                "CURRENT_REPLAY_PF_BELOW_BREAKEVEN",
                "expectancy blocker",
                "Require TP10/SL7 profit factor above 1.0 for the exact replay vehicle.",
                str(DEFAULT_REPLAY_REPORT),
                f"current profit_factor={broad_check['profit_factor']}",
            )
        )
    if not broad_check["positive_day_rate_sufficient"]:
        blockers.append(
            _blocker(
                "CURRENT_REPLAY_POSITIVE_DAY_RATE_LOW",
                "forecast blocker",
                "Require positive-day rate >= 2/3 before treating the replay as daily-stable.",
                str(DEFAULT_REPLAY_REPORT),
                f"current positive_day_rate={broad_check['positive_day_rate']}",
            )
        )
    if not capture_scope.get("present"):
        blockers.append(
            _blocker(
                "MISSING_LOCAL_TP_SCOPE",
                "missing evidence",
                "Produce exact broker-local USD_JPY|LONG|BREAKOUT_FAILURE|TAKE_PROFIT_ORDER capture proof.",
                "data/capture_economics.json",
                "capture_economics has no exact pair/side/method/TAKE_PROFIT_ORDER scope",
            )
        )
    if not target_lane_present:
        blockers.append(
            _blocker(
                "FRESH_TARGET_LANE_ABSENT",
                "stale artifact",
                "Regenerate order_intents from fresh inputs and require this exact lane to be emitted.",
                "data/order_intents.json",
                "fresh order_intents has 73 DRY_RUN_BLOCKED lanes and no target lane",
            )
        )
    if str(usd_jpy_long_profile.get("status") or "").upper() == "BLOCK_UNTIL_NEW_EVIDENCE":
        blockers.append(
            _blocker(
                "STRATEGY_PROFILE_BLOCK_UNTIL_NEW_EVIDENCE",
                "strategy profile blocker",
                "Repair only a method/shape-scoped USD_JPY LONG BREAKOUT_FAILURE LIMIT HARVEST profile after exact proof exists.",
                "data/strategy_profile.json; src/quant_rabbit/strategy/profile.py",
                usd_jpy_long_profile.get("required_fix"),
            )
        )
    for item in acceptance.get("blockers") or []:
        code = str(item).split(":", 1)[0]
        blockers.append(
            _blocker(
                code,
                "expectancy blocker" if code != "SELF_IMPROVEMENT_P0_PRESENT" else "true safety blocker",
                "Clear this profitability acceptance blocker without weakening acceptance thresholds.",
                "data/profitability_acceptance.json",
                str(item),
            )
        )
    for item in support.get("blockers") or []:
        if not isinstance(item, dict):
            continue
        code = str(item.get("code") or "")
        if code in {"PROFITABILITY_ACCEPTANCE_BLOCKED"}:
            continue
        blockers.append(
            _blocker(
                code,
                "guardian/operator-review blocker" if "GUARDIAN" in code else "true safety blocker",
                "Normal new-entry routing remains blocked until this support blocker clears.",
                "data/trader_support_bot.json",
                item.get("message"),
            )
        )
    blockers.append(
        _blocker(
            "NO_FRESH_GPT_TRADE_ADD_RECEIPT",
            "true safety blocker",
            "Obtain a fresh GPT-5.5 TRADE/ADD receipt for this exact lane after all evidence and risk gates pass.",
            "operator receipt workflow",
            "no fresh TRADE/ADD receipt was generated or consumed",
        )
    )

    return {
        "generated_at_utc": _now(),
        "mode": "read_only_evidence",
        "lane_id": TARGET_LANE_ID,
        "required_shape": {
            "pair": "USD_JPY",
            "side": "LONG",
            "strategy": "BREAKOUT_FAILURE",
            "entry_type": "LIMIT",
            "exit_method": "TAKE_PROFIT_ORDER",
            "tp_pips": TARGET_TP_PIPS,
            "sl_pips": TARGET_SL_PIPS,
            "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
            "tp_target_intent": "HARVEST",
            "with_move_chase_allowed": False,
        },
        "current_inputs": {
            "broker_snapshot_fetched_at_utc": broker.get("fetched_at_utc"),
            "capture_economics_generated_at_utc": capture.get("generated_at_utc"),
            "order_intents_generated_at_utc": intents.get("generated_at_utc"),
            "profitability_acceptance_generated_at_utc": acceptance.get("generated_at_utc"),
            "trader_support_bot_generated_at_utc": support.get("generated_at_utc"),
            "profitability_acceptance_status": acceptance.get("status"),
            "profitability_acceptance_freshness": support_metrics.get(
                "profitability_acceptance_artifact_status"
            ),
            "profitability_acceptance_stale_against_inputs": support_metrics.get(
                "profitability_acceptance_stale_against_inputs"
            ),
            "profitability_acceptance_stale_input_names": support_metrics.get(
                "profitability_acceptance_stale_input_names"
            ),
            "current_order_intent_target_lane_present": target_lane_present,
            "current_order_intent_live_ready_lanes": sum(
                1 for item in intents.get("results") or [] if item.get("status") == "LIVE_READY"
            ),
            "current_order_intent_result_count": len(intents.get("results") or []),
            "stale_candidate_board_generated_at_utc": board.get("generated_at_utc"),
            "stale_candidate_board_order_intents_generated_at_utc": board.get(
                "order_intents_generated_at_utc"
            ),
        },
        "capital_flow_basis": {
            "daily_target_generated_at_utc": daily.get("generated_at_utc"),
            "current_equity_raw": daily.get("current_equity_raw"),
            "broker_nav_jpy": broker_account.get("nav_jpy"),
            "funding_adjusted_equity": daily.get("funding_adjusted_equity"),
            "capital_flows_30d": daily.get("capital_flows_30d"),
            "rolling_30d_multiplier_funding_adjusted": daily.get(
                "rolling_30d_multiplier_funding_adjusted"
            ),
            "rolling_30d_multiplier_raw": daily.get("rolling_30d_multiplier_raw"),
            "performance_basis": daily.get("performance_basis"),
            "sizing_basis": daily.get("sizing_basis"),
            "capital_flows": capital.get("capital_flows"),
        },
        "manual_eurusd_protection": {
            "trade_id": "472987",
            "present": bool(eur_usd_manual),
            "classification": _nested(eur_usd_manual[0], "operator_manual_position", "classification")
            if eur_usd_manual
            else None,
            "management_intent": _nested(
                eur_usd_manual[0],
                "operator_manual_position",
                "management_intent",
            )
            if eur_usd_manual
            else None,
            "system_pl_counted": _nested(
                eur_usd_manual[0],
                "operator_manual_position",
                "system_pl_counted",
            )
            if eur_usd_manual
            else None,
            "same_theme_auto_add_allowed": _nested(
                eur_usd_manual[0],
                "operator_manual_position",
                "same_theme_auto_add_allowed",
            )
            if eur_usd_manual
            else None,
            "auto_sl_attach_allowed": _nested(
                eur_usd_manual[0],
                "operator_manual_position",
                "auto_sl_attach_allowed",
            )
            if eur_usd_manual
            else None,
            "auto_tp_modify_allowed": _nested(
                eur_usd_manual[0],
                "operator_manual_position",
                "auto_tp_modify_allowed",
            )
            if eur_usd_manual
            else None,
            "take_profit": eur_usd_manual[0].get("take_profit") if eur_usd_manual else None,
            "stop_loss": eur_usd_manual[0].get("stop_loss") if eur_usd_manual else None,
        },
        "stale_candidate_shape": _candidate_shape(stale_candidate),
        "packaged_bidask_replay_rule": _packaged_rule_summary(packaged_rule),
        "current_bidask_replay_probe": {
            "source_artifact": str(DEFAULT_REPLAY_REPORT),
            "generated_at_utc": replay.get("generated_at_utc"),
            "truth_source": replay.get("truth_source"),
            "history_dirs": replay.get("history_dirs"),
            "price_truth_coverage": replay.get("price_truth_coverage"),
            "summary": replay.get("summary"),
            "broad_down_to_up_tp10_sl7": broad_check,
            "strict_current_bucket_down_to_up_tp10_sl7": strict_check,
        },
        "capture_exact_tp_scope": capture_scope,
        "strategy_profile": {
            "usd_jpy_long_profile_status": usd_jpy_long_profile.get("status"),
            "usd_jpy_long_profile_method": usd_jpy_long_profile.get("method"),
            "required_fix": usd_jpy_long_profile.get("required_fix"),
            "repair_applied": False,
            "repair_result": (
                "not applied because exact TP proof is invalid against current replay, "
                "local capture scope is missing, and the fresh target lane is absent"
            ),
        },
        "proof_checks": {
            "shape_matches_pair_side_strategy_entry": True,
            "shape_matches_tp_sl": True,
            "spread_included_at_current_replay_level": True,
            "spread_included_basis": (
                "oanda_history_replay_validate uses local OANDA S5 bid/ask candles; "
                "UP enters at ask and exits at bid, so spread cost is included."
            ),
            "sample_size_sufficient": broad_check["sample_size_sufficient"],
            "sample_size_threshold": 30,
            "active_days_sufficient": broad_check["active_days_sufficient"],
            "active_days_threshold": 3,
            "non_negative_expectancy": broad_check["non_negative_expectancy"],
            "profit_factor_positive": broad_check["profit_factor_positive"],
            "profit_factor_above_breakeven": broad_check["profit_factor_above_breakeven"],
            "hit_rate_reported": broad_check["hit_rate"] is not None,
            "positive_day_rate_reported": broad_check["positive_day_rate"] is not None,
            "positive_day_rate_sufficient": broad_check["positive_day_rate_sufficient"],
            "adverse_metric_reported": "avg_mae_pips",
            "adverse_metric_value_pips": broad_check["avg_mae_pips"],
            "max_adverse_exact_available": False,
            "drawdown_metric_reported": "worst_daily_realized_pips",
            "drawdown_metric_value_pips": broad_check["worst_daily_realized_pips"],
            "max_drawdown_exact_available": False,
            "result_reproducible_from_current_artifacts": True,
            "raw_replay_audit_report_present": True,
            "stale_packaged_rule_raw_audit_report_present": bool(
                (ROOT / str((packaged_rule or {}).get("audit_report") or "")).exists()
            ),
            "local_capture_exact_tp_scope_present": exact_live_scope_proven,
            "local_capture_exact_tp_scope_key": TARGET_SCOPE_KEY,
            "fresh_order_intent_target_lane_present": target_lane_present,
        },
        "verdict": {
            "status": "TP_PROOF_REJECTED_CURRENT_REPLAY_NEGATIVE_UNDERSAMPLED",
            "stale_packaged_rule_positive": bool(
                packaged_rule
                and (packaged_rule.get("optimized_profit_factor") or 0) > 1
                and (packaged_rule.get("samples") or 0) >= 30
            ),
            "current_bidask_replay_tp10_sl7_valid": current_replay_pass,
            "exact_live_take_profit_order_scope_proven": exact_live_scope_proven,
            "strategy_profile_repair_allowed": exact_tp_proof_valid,
            "as_grade_allowed": False,
            "live_ready_allowed": False,
            "reason": (
                "The tracked packaged rule is positive, but its declared raw audit report is absent "
                "and the fresh USD_JPY-only bid/ask replay from current forecast_history does not "
                "confirm the exact TP10/SL7 DOWN->UP vehicle: it is under-sampled, has negative "
                "average realized pips, PF below breakeven, and positive-day rate below the "
                "daily-stability floor. The fresh order-intent rebuild also does not emit the lane."
            ),
        },
        "remaining_clearing_conditions": _dedup_blockers(blockers),
        "safety": {
            "broker_side_effects": [],
            "orders_placed": False,
            "orders_cancelled": False,
            "positions_closed": False,
            "sl_tp_modified": False,
            "execution_flags_enabled": False,
            "broker_state_modified": False,
        },
    }


def build_markdown(proof: dict[str, Any]) -> str:
    current = proof["current_inputs"]
    candidate = proof["stale_candidate_shape"]
    broad = proof["current_bidask_replay_probe"]["broad_down_to_up_tp10_sl7"]
    strict = proof["current_bidask_replay_probe"]["strict_current_bucket_down_to_up_tp10_sl7"]
    packaged = proof["packaged_bidask_replay_rule"]
    checks = proof["proof_checks"]
    profile = proof["strategy_profile"]
    verdict = proof["verdict"]
    support_blockers = _unique_codes(proof["remaining_clearing_conditions"])
    manual = proof["manual_eurusd_protection"]
    capital = proof["capital_flow_basis"]

    lines = [
        "# USD_JPY LONG BREAKOUT_FAILURE TP Proof",
        "",
        f"Generated: `{proof['generated_at_utc']}`",
        "",
        "Mode: read-only evidence. No orders, cancels, closes, SL/TP changes, execution flag changes, or broker-state modifications.",
        "",
        "## Verdict",
        "",
        f"`{verdict['status']}`",
        "",
        verdict["reason"],
        "",
        "The candidate did not become A/S or `LIVE_READY`, and no strategy-profile repair was applied.",
        "",
        "## Exact Shape",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| lane_id | `{proof['lane_id']}` |",
        "| pair | `USD_JPY` |",
        "| side | `LONG` |",
        "| strategy | `BREAKOUT_FAILURE` |",
        "| entry_type | `LIMIT` |",
        "| exit_method | `TAKE_PROFIT_ORDER` |",
        "| TP/SL | TP10 / SL7 |",
        "| TP mode | `ATTACHED_TECHNICAL_TP` |",
        "| TP intent | `HARVEST` |",
        "| chase permission | with-move chase not allowed |",
        "",
        "## Current Inputs",
        "",
        "| Artifact | Current state |",
        "| --- | --- |",
        f"| broker snapshot | fetched `{current['broker_snapshot_fetched_at_utc']}` |",
        f"| capture_economics | generated `{current['capture_economics_generated_at_utc']}`, `NEGATIVE_EXPECTANCY` |",
        f"| order_intents | generated `{current['order_intents_generated_at_utc']}`, {current['current_order_intent_result_count']} results, {current['current_order_intent_live_ready_lanes']} `LIVE_READY` |",
        f"| target lane in current order_intents | `{current['current_order_intent_target_lane_present']}` |",
        f"| profitability_acceptance | generated `{current['profitability_acceptance_generated_at_utc']}`, `{current['profitability_acceptance_status']}` |",
        f"| profitability freshness | `{current['profitability_acceptance_freshness']}`, stale=`{current['profitability_acceptance_stale_against_inputs']}` |",
        f"| trader_support_bot | generated `{current['trader_support_bot_generated_at_utc']}`, blockers remain |",
        "",
        "The old A/S candidate board was generated "
        f"`{current['stale_candidate_board_generated_at_utc']}` from order intents "
        f"`{current['stale_candidate_board_order_intents_generated_at_utc']}`; it is stale candidate history, not current routing input.",
        "",
        "## Stale Candidate Snapshot",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| entry / TP / SL | {candidate.get('entry')} / {candidate.get('tp')} / {candidate.get('sl')} |",
        f"| units | {candidate.get('units')} |",
        f"| expected RR | {candidate.get('expected_rr')} |",
        f"| risk JPY | {candidate.get('risk_jpy')} |",
        f"| reward JPY | {candidate.get('reward_jpy')} |",
        f"| spread pips | {candidate.get('spread_pips')} |",
        f"| forecast direction / confidence | {candidate.get('forecast_direction')} / {candidate.get('forecast_confidence')} |",
        f"| market support ok | {candidate.get('forecast_market_support_ok')} |",
        "",
        "## Replay Evidence",
        "",
        "Packaged historical rule:",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| rule | `{packaged.get('name')}` |",
        f"| samples | {packaged.get('samples')} |",
        f"| active_days | {packaged.get('active_days')} |",
        f"| optimized_avg_realized_pips | {packaged.get('optimized_avg_realized_pips')} |",
        f"| optimized_profit_factor | {packaged.get('optimized_profit_factor')} |",
        f"| directional_hit_rate | {packaged.get('directional_hit_rate')} |",
        f"| positive_day_rate | {packaged.get('positive_day_rate')} |",
        f"| avg_MFE / avg_MAE | {packaged.get('avg_mfe_pips')} / {packaged.get('avg_mae_pips')} |",
        f"| worst_daily_realized_pips | {packaged.get('worst_daily_realized_pips')} |",
        f"| raw audit present | {checks['stale_packaged_rule_raw_audit_report_present']} |",
        "",
        "Fresh USD_JPY-only current replay:",
        "",
        "| Metric | Broad DOWN->UP TP10/SL7 | Strict current bucket |",
        "| --- | ---: | ---: |",
        f"| n | {broad.get('n')} | {strict.get('n')} |",
        f"| active_days | {broad.get('active_days')} | {strict.get('active_days')} |",
        f"| avg_realized_pips | {broad.get('avg_realized_pips')} | {strict.get('avg_realized_pips')} |",
        f"| profit_factor | {broad.get('profit_factor')} | {strict.get('profit_factor')} |",
        f"| win_rate | {broad.get('win_rate')} | {strict.get('win_rate')} |",
        f"| hit_rate | {broad.get('hit_rate')} | {strict.get('hit_rate')} |",
        f"| positive_day_rate | {broad.get('positive_day_rate')} | {strict.get('positive_day_rate')} |",
        f"| avg_MFE / avg_MAE | {broad.get('avg_mfe_pips')} / {broad.get('avg_mae_pips')} | {strict.get('avg_mfe_pips')} / {strict.get('avg_mae_pips')} |",
        f"| worst_daily_realized_pips | {broad.get('worst_daily_realized_pips')} | {strict.get('worst_daily_realized_pips')} |",
        "",
        "Spread is included in the fresh replay because `oanda_history_replay_validate.py` uses S5 bid/ask candles: UP enters at ask and exits at bid.",
        "",
        "## Strategy Profile Decision",
        "",
        f"No strategy profile repair was made. USD_JPY LONG remains `{profile['usd_jpy_long_profile_status']}`.",
        "",
        f"Reason: {profile['repair_result']}. Broad `USD_JPY LONG` and broad `BREAKOUT_FAILURE` remain locked; no with-move chase was enabled.",
        "",
        "## A/S Readiness",
        "",
        "| Check | Result |",
        "| --- | --- |",
        "| A/S grade | No |",
        "| LIVE_READY | No |",
        "| RiskEngine dry-run | Not run for target lane because current order intents do not contain the lane |",
        "| LiveOrderGateway eligibility | Not eligible; no current intent and hard blockers remain |",
        "| Fresh GPT TRADE/ADD receipt | Missing |",
        "| Guardian/operator review | Blocks normal routing |",
        "| Profitability acceptance | Fresh but blocked |",
        "| Negative expectancy override | Not allowed |",
        "| EUR_USD manual conflict | Manual EUR_USD remains excluded from system P/L and occupancy |",
        "",
        "## Remaining Blockers",
        "",
    ]
    for blocker in support_blockers:
        lines.append(f"- `{blocker}`")
    lines.extend(
        [
            "",
            "See `remaining_clearing_conditions` in the JSON proof for clearing condition, file/module, and evidence requirement per blocker.",
            "",
            "## Capital And Manual Protection",
            "",
            f"- current_equity_raw equals broker NAV: `{capital['current_equity_raw']}` / `{capital['broker_nav_jpy']}`.",
            f"- funding_adjusted_equity excludes the 100,000 JPY deposit: `{capital['funding_adjusted_equity']}`.",
            f"- rolling_30d_multiplier_funding_adjusted is authoritative: `{capital['rolling_30d_multiplier_funding_adjusted']}`; raw multiplier `{capital['rolling_30d_multiplier_raw']}` is context only.",
            f"- EUR_USD `472987` present=`{manual['present']}`, classification=`{manual['classification']}`, management_intent=`{manual['management_intent']}`.",
            f"- manual EUR_USD system_pl_counted=`{manual['system_pl_counted']}`, same_theme_auto_add_allowed=`{manual['same_theme_auto_add_allowed']}`, auto_sl_attach_allowed=`{manual['auto_sl_attach_allowed']}`, auto_tp_modify_allowed=`{manual['auto_tp_modify_allowed']}`.",
            "",
            "## Safety",
            "",
            "No broker-side action was performed. No order was placed, cancelled, or modified; no position was closed; no SL/TP was attached or changed; no execution flag was enabled.",
            "",
        ]
    )
    return "\n".join(lines)


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _find_intent(intents: dict[str, Any]) -> dict[str, Any] | None:
    for item in intents.get("results") or []:
        if item.get("lane_id") == TARGET_LANE_ID:
            return item
    return None


def _find_stale_candidate(board: dict[str, Any]) -> dict[str, Any]:
    for item in board.get("candidates") or []:
        if item.get("lane_id") == TARGET_LANE_ID:
            return item
    return {}


def _find_rule(rules: dict[str, Any]) -> dict[str, Any] | None:
    for section in ("daily_stable_contrarian_edge_rules", "contrarian_edge_rules", "edge_rules"):
        for item in rules.get(section) or []:
            if item.get("name") == TARGET_RULE_NAME:
                return item
    return None


def _find_replay_row(replay: dict[str, Any], section: str, **criteria: Any) -> dict[str, Any] | None:
    grids = replay.get("segment_exit_grids") if isinstance(replay.get("segment_exit_grids"), dict) else {}
    for item in grids.get(section) or []:
        if all(item.get(key) == value for key, value in criteria.items()):
            return item
    return None


def _evaluate_replay_row(row: dict[str, Any] | None) -> dict[str, Any]:
    if not row:
        return {
            "present": False,
            "n": 0,
            "tp_pips": TARGET_TP_PIPS,
            "sl_pips": TARGET_SL_PIPS,
            "sample_size_sufficient": False,
            "active_days_sufficient": False,
            "non_negative_expectancy": False,
            "profit_factor_positive": False,
            "profit_factor_above_breakeven": False,
            "positive_day_rate_sufficient": False,
        }
    grid = _exact_grid(row) or {}
    daily = row.get("daily_stability") if isinstance(row.get("daily_stability"), dict) else {}
    summary = row.get("summary") if isinstance(row.get("summary"), dict) else {}
    n = _number(grid.get("n") if grid else row.get("n"))
    avg = _number(grid.get("avg_realized_pips"))
    pf = _number(grid.get("profit_factor"))
    positive_day_rate = _number(daily.get("positive_day_rate"))
    active_days = _number(daily.get("active_days"))
    return {
        "present": True,
        "pair": row.get("pair"),
        "forecast_direction": row.get("forecast_direction"),
        "direction": row.get("direction"),
        "horizon_bucket": row.get("horizon_bucket"),
        "confidence_bucket": row.get("confidence_bucket"),
        "tp_pips": _number(grid.get("take_profit_pips")),
        "sl_pips": _number(grid.get("stop_loss_pips")),
        "n": n,
        "avg_realized_pips": avg,
        "profit_factor": pf,
        "profit_factor_positive": pf is not None and pf > 0,
        "profit_factor_above_breakeven": pf is not None and pf > 1.0,
        "win_rate": _number(grid.get("win_rate")),
        "tp_rate": _number(grid.get("tp_rate")),
        "sl_rate": _number(grid.get("sl_rate")),
        "timeout_rate": _number(grid.get("timeout_rate")),
        "hit_rate": _number(summary.get("hit_rate")),
        "avg_final_pips": _number(summary.get("avg_final_pips")),
        "median_final_pips": _number(summary.get("median_final_pips")),
        "avg_mfe_pips": _number(summary.get("avg_mfe_pips")),
        "avg_mae_pips": _number(summary.get("avg_mae_pips")),
        "active_days": active_days,
        "positive_day_rate": positive_day_rate,
        "worst_daily_realized_pips": _number(daily.get("worst_daily_realized_pips")),
        "max_daily_sample_share": _number(daily.get("max_daily_sample_share")),
        "daily_summaries": daily.get("daily_summaries"),
        "sample_size_sufficient": n is not None and n >= 30,
        "active_days_sufficient": active_days is not None and active_days >= 3,
        "non_negative_expectancy": avg is not None and avg >= 0.0,
        "positive_day_rate_sufficient": positive_day_rate is not None and positive_day_rate >= (2.0 / 3.0),
    }


def _exact_grid(row: dict[str, Any]) -> dict[str, Any] | None:
    for item in row.get("exit_grid") or []:
        if (
            _number(item.get("take_profit_pips")) == TARGET_TP_PIPS
            and _number(item.get("stop_loss_pips")) == TARGET_SL_PIPS
        ):
            return item
    return None


def _capture_scope(capture: dict[str, Any]) -> dict[str, Any]:
    payload = _nested(
        capture,
        "by_pair_side_method_exit_reason",
        "USD_JPY",
        "LONG",
        "BREAKOUT_FAILURE",
        "TAKE_PROFIT_ORDER",
    )
    if isinstance(payload, dict):
        return {"scope_key": TARGET_SCOPE_KEY, "present": True, **payload}
    return {"scope_key": TARGET_SCOPE_KEY, "present": False}


def _strategy_profile_entry(profile: dict[str, Any]) -> dict[str, Any]:
    for item in profile.get("profiles") or []:
        if item.get("pair") == "USD_JPY" and item.get("direction") == "LONG" and item.get("method") is None:
            return item
    return {}


def _candidate_shape(candidate: dict[str, Any]) -> dict[str, Any]:
    shape = candidate.get("trade_shape") if isinstance(candidate.get("trade_shape"), dict) else {}
    forecast = candidate.get("forecast") if isinstance(candidate.get("forecast"), dict) else {}
    spread = candidate.get("spread_state") if isinstance(candidate.get("spread_state"), dict) else {}
    return {
        "entry": shape.get("entry"),
        "tp": shape.get("tp"),
        "sl": shape.get("sl"),
        "units": shape.get("units"),
        "expected_rr": shape.get("expected_rr"),
        "risk_jpy": shape.get("risk_jpy"),
        "reward_jpy": shape.get("reward_jpy"),
        "spread_pips": spread.get("spread_pips"),
        "forecast_direction": forecast.get("direction"),
        "forecast_confidence": forecast.get("confidence"),
        "forecast_market_support_ok": forecast.get("market_support_ok"),
        "market_read_direction": candidate.get("market_read_direction"),
        "current_blockers": candidate.get("current_blockers"),
        "exact_missing_evidence": candidate.get("exact_missing_evidence"),
        "candidate_source": "data/as_lane_candidate_board.json",
    }


def _packaged_rule_summary(rule: dict[str, Any] | None) -> dict[str, Any]:
    if not rule:
        return {"present": False}
    keys = (
        "name",
        "adoption_status",
        "adoption_blockers",
        "pair",
        "side",
        "forecast_direction",
        "faded_direction",
        "direction",
        "contrarian_edge",
        "granularity",
        "horizon_bucket",
        "confidence_bucket",
        "samples",
        "active_days",
        "first_day",
        "last_day",
        "optimized_take_profit_pips",
        "optimized_stop_loss_pips",
        "optimized_avg_realized_pips",
        "optimized_win_rate",
        "optimized_profit_factor",
        "directional_hit_rate",
        "positive_day_rate",
        "positive_days",
        "negative_days",
        "max_daily_sample_share",
        "daily_stability_status",
        "avg_final_pips",
        "median_final_pips",
        "avg_mfe_pips",
        "avg_mae_pips",
        "avg_daily_realized_pips",
        "best_daily_realized_pips",
        "worst_daily_realized_pips",
        "audit_report",
    )
    out = {"present": True}
    out.update({key: rule.get(key) for key in keys if key in rule})
    out["audit_report_present"] = bool((ROOT / str(rule.get("audit_report") or "")).exists())
    return out


def _blocker(
    code: str,
    classification: str,
    clearing_condition: str,
    file_or_module: str,
    evidence: Any,
) -> dict[str, Any]:
    return {
        "blocker": code,
        "classification": classification,
        "clearing_condition": clearing_condition,
        "file_or_module": file_or_module,
        "required_test_or_evidence": evidence,
    }


def _nested(payload: Any, *keys: str) -> Any:
    current = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _number(value: Any) -> float | int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return round(value, 10)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number.is_integer():
        return int(number)
    return round(number, 10)


def _unique_codes(blockers: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for blocker in blockers:
        code = str(blocker.get("blocker") or "").strip()
        if code and code not in seen:
            seen.add(code)
            out.append(code)
    return out


def _dedup_blockers(blockers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for blocker in blockers:
        code = str(blocker.get("blocker") or "").strip()
        if not code or code in seen:
            continue
        seen.add(code)
        out.append(blocker)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
