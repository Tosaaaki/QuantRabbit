from __future__ import annotations

import copy
import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from quant_rabbit.fast_bot_shock_guard import (
    CONTINUATION_CONFIRMED,
    FAILED_CONTINUATION,
    NORMAL,
    SHOCK_FREEZE,
    advance_state,
    canonical_sha,
    guard_shadow,
    load_config,
    observe_market,
    protective_stop_candidates,
    run_guard_cycle,
    seal,
    size_units_for_stop,
    structure_exit_plan,
    evaluate_structure_exit,
    validate_protective_stop,
    validate_structure_exit_plan,
)


ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 8, 28, 14, 0, tzinfo=timezone.utc)


def _config():
    return load_config(ROOT / "config" / "fast_bot_shock_guard_v1.json")


def _chart(
    *,
    pair: str = "EUR_USD",
    direction: str = "DOWN",
    pips: float = 18.0,
    gap: bool = False,
    stale: bool = False,
):
    start = NOW - timedelta(minutes=35)
    sign = 1.0 if direction == "UP" else -1.0
    factor = 100.0 if pair.endswith("JPY") else 10_000.0
    base = 150.0 if pair.endswith("JPY") else 1.1000
    rows = []
    for index in range(36):
        offset = index + (1 if gap and index == 28 else 0)
        at = start + timedelta(minutes=offset)
        if stale:
            at -= timedelta(minutes=10)
        trend_index = max(0, index - 20)
        price = base + sign * (pips / factor) * trend_index / 15.0
        rows.append(
            {
                "t": at.isoformat(),
                "o": price,
                "h": price + 0.2 / factor,
                "l": price - 0.2 / factor,
                "c": price,
                "spread_pips": 0.8,
                "complete": True,
            }
        )
    views = []
    for tf, market_direction in (("M1", direction), ("M5", direction), ("M15", direction), ("H1", direction)):
        views.append(
            {
                "granularity": tf,
                "recent_candles": rows if tf == "M1" else rows[-3:],
                "indicators": {"atr_pips": 9.0 if tf == "M5" else 5.0},
                "market_state": {"direction": market_direction},
            }
        )
    return {"generated_at_utc": NOW.isoformat(), "charts": [{"pair": pair, "views": views}]}


def _normal(pair: str = "EUR_USD"):
    return seal(
        {
            "contract": "QR_FAST_BOT_SHOCK_GUARD_STATE_V1",
            "schema_version": 1,
            "pair": pair,
            "state": NORMAL,
            "event_id": None,
            "shock_direction": None,
            "observed_at_utc": NOW.isoformat(),
            "decision_due_at_utc": None,
            "cooldown_until_utc": None,
            "last_transition_at_utc": NOW.isoformat(),
            "last_complete_m1_at_utc": None,
            "resolution": None,
            "thresholds": {},
            "evidence": {},
            "fail_closed_reason": None,
            "execution_authority": "NONE",
            "broker_mutation_allowed": False,
            "external_order_attempts": 0,
            "external_orders": 0,
        }
    )


def _direct_observation(
    direction: str,
    *,
    new_extreme: bool,
    adverse_pips: float,
    pair: str = "EUR_USD",
):
    sign = 1.0 if direction == "UP" else -1.0
    factor = 100.0 if pair.endswith("JPY") else 10_000.0
    base = 150.0 if pair.endswith("JPY") else 1.1000
    initial_high = base + 20.0 / factor
    initial_low = base - 20.0 / factor
    post = []
    for index in range(1, 6):
        if direction == "UP":
            high = initial_high + (1.0 / factor if new_extreme else -0.1 / factor)
            low = initial_high - adverse_pips / factor
        else:
            low = initial_low - (1.0 / factor if new_extreme else -0.1 / factor)
            high = initial_low + adverse_pips / factor
        post.append(
            {
                "at": (NOW + timedelta(minutes=index)).isoformat(),
                "high": high,
                "low": low,
                "close": high if direction == "UP" else low,
            }
        )
    return {
        "valid": True,
        "pair": pair,
        "latest_complete_m1_at_utc": post[-1]["at"],
        "impulse_direction": direction,
        "impulse_magnitude_pips": 20.0,
        "raw_confirmation_count": 2,
        "raw_confirmations": {"velocity": True, "prior_swing_break": True},
        "atr_pips": 8.0,
        "atr_multiple": 2.5,
        "initial_high": initial_high,
        "initial_low": initial_low,
        "timeframe_alignment": {"M1": direction, "M5": direction, "M15": direction, "H1": direction},
        "short_term_reversal": False,
        "higher_timeframe_continuation": True,
        "post_window": post,
    }


def test_mirror_symmetric_threshold_and_boundary():
    config, _ = _config()
    up = observe_market(pair_charts=_chart(direction="UP"), pair="EUR_USD", now_utc=NOW + timedelta(minutes=1), config=config)
    down = observe_market(pair_charts=_chart(direction="DOWN"), pair="EUR_USD", now_utc=NOW + timedelta(minutes=1), config=config)
    assert up["impulse_magnitude_pips"] == down["impulse_magnitude_pips"] == 18.0
    assert up["atr_multiple"] == down["atr_multiple"] == 2.0
    assert up["impulse_direction"] == "UP"
    assert down["impulse_direction"] == "DOWN"
    assert up["raw_confirmation_count"] == down["raw_confirmation_count"] >= 2
    assert up["prior_swing_break"] is down["prior_swing_break"] is True


def test_raw_detector_does_not_use_atr_as_onset_gate():
    config, config_sha = _config()
    packet = _chart(direction="UP")
    for view in packet["charts"][0]["views"]:
        if view["granularity"] == "M5":
            view["indicators"] = {}
    observation = observe_market(
        pair_charts=packet,
        pair="EUR_USD",
        now_utc=NOW + timedelta(minutes=1),
        config=config,
    )
    assert observation["valid"] is True
    assert observation["atr_pips"] is None
    assert observation["atr_multiple"] is None
    state = advance_state(
        prior=_normal(),
        observation=observation,
        now_utc=NOW + timedelta(minutes=1),
        config=config,
        config_sha256=config_sha,
    )
    assert state["state"] == SHOCK_FREEZE
    assert state["thresholds"]["atr_role"] == "AUXILIARY_NORMALIZATION_AND_UPPER_BOUND_ONLY"


def test_dedicated_shock_history_supplies_36_bars_without_changing_legacy_30():
    config, _ = _config()
    packet = _chart(direction="DOWN")
    m1 = packet["charts"][0]["views"][0]
    full = list(m1["recent_candles"])
    m1["recent_candles"] = full[-30:]
    m1["shock_guard_recent_candles"] = full
    observation = observe_market(
        pair_charts=packet,
        pair="EUR_USD",
        now_utc=NOW + timedelta(minutes=1),
        config=config,
    )
    assert len(m1["recent_candles"]) == 30
    assert len(m1["shock_guard_recent_candles"]) == 36
    assert observation["valid"] is True

    del m1["shock_guard_recent_candles"]
    legacy = observe_market(
        pair_charts=packet,
        pair="EUR_USD",
        now_utc=NOW + timedelta(minutes=1),
        config=config,
    )
    assert legacy == {
        "valid": False,
        "reason": "M1_HISTORY_INSUFFICIENT",
        "pair": "EUR_USD",
    }


def test_spread_expansion_is_observed_symmetrically_without_becoming_side_specific():
    config, _ = _config()
    for direction in ("UP", "DOWN"):
        packet = _chart(direction=direction)
        latest = packet["charts"][0]["views"][0]["recent_candles"][-1]
        latest["spread_pips"] = 1.6
        observation = observe_market(
            pair_charts=packet,
            pair="EUR_USD",
            now_utc=NOW + timedelta(minutes=1),
            config=config,
        )
        assert observation["spread_shock"] is True
        assert observation["spread_ratio"] == 2.0
        assert observation["raw_confirmations"]["spread_expansion"] is True


def test_stale_and_gap_fail_closed():
    config, config_sha = _config()
    for packet, reason in ((_chart(gap=True), "M1_GAP"), (_chart(stale=True), "M1_STALE_OR_FUTURE")):
        observation = observe_market(pair_charts=packet, pair="EUR_USD", now_utc=NOW + timedelta(minutes=1), config=config)
        assert observation["reason"] == reason
        state = advance_state(prior=_normal(), observation=observation, now_utc=NOW + timedelta(minutes=1), config=config, config_sha256=config_sha)
        assert state["state"] == SHOCK_FREEZE
        assert state["fail_closed_reason"] == reason


def test_five_minute_classification_is_mirrored_and_before_five_stays_frozen():
    config, config_sha = _config()
    for direction in ("UP", "DOWN"):
        opened = advance_state(prior=_normal(), observation=_direct_observation(direction, new_extreme=False, adverse_pips=0.0), now_utc=NOW, config=config, config_sha256=config_sha)
        before = advance_state(prior=opened, observation=_direct_observation(direction, new_extreme=False, adverse_pips=3.0), now_utc=NOW + timedelta(minutes=4, seconds=59), config=config, config_sha256=config_sha)
        assert before["state"] == SHOCK_FREEZE
        failed = advance_state(prior=opened, observation=_direct_observation(direction, new_extreme=False, adverse_pips=2.1), now_utc=NOW + timedelta(minutes=5), config=config, config_sha256=config_sha)
        assert failed["state"] == FAILED_CONTINUATION
        continued = advance_state(prior=opened, observation=_direct_observation(direction, new_extreme=True, adverse_pips=1.0), now_utc=NOW + timedelta(minutes=5), config=config, config_sha256=config_sha)
        assert continued["state"] == CONTINUATION_CONFIRMED


def test_shock_decision_records_side_relative_regime_transition_mismatch():
    config, config_sha = _config()
    for direction, side in (("DOWN", "LONG"), ("UP", "SHORT")):
        state = advance_state(
            prior=_normal(),
            observation=_direct_observation(
                direction, new_extreme=False, adverse_pips=0.0
            ),
            now_utc=NOW,
            config=config,
            config_sha256=config_sha,
        )
        shadow = {
            "contract_sha256": "source",
            "signals": [
                {
                    "signal_id": f"range-{direction.lower()}",
                    "pair": "EUR_USD",
                    "side": side,
                    "method": "RANGE_ROTATION",
                    "strategy_id": "range_rotation",
                    "entry": 1.1,
                    "take_profit_pips": 2.4,
                    "m5_atr_pips": 4.0,
                    "spread_pips": 0.8,
                }
            ],
        }
        guarded, decisions = guard_shadow(
            shadow=shadow,
            state=state,
            pair_charts=_chart(direction=direction),
            config=config,
            config_sha256=config_sha,
            now_utc=NOW,
        )
        assert guarded["signals"] == []
        assert decisions[0]["entry_allowed"] is False
        assert decisions[0]["side_relative_alignment"] == "COUNTERTREND"
        assert decisions[0]["regime_transition_mismatch"] is True
        assert decisions[0]["strategy_id"] == "range_rotation"
        assert decisions[0]["deterministic_shock_guard_precedes_llm"] is True
        assert decisions[0]["llm_order_fields_allowed"] is False


def test_protective_stop_geometry_symmetry_and_inverse_units():
    config, _ = _config()
    long = protective_stop_candidates(pair="EUR_USD", side="LONG", entry=1.1000, atr_pips=4.0, spread_pips=0.8, recent_swing_price=1.0995, observed_at_utc=NOW, config=config)
    short = protective_stop_candidates(pair="EUR_USD", side="SHORT", entry=1.1000, atr_pips=4.0, spread_pips=0.8, recent_swing_price=1.1005, observed_at_utc=NOW, config=config)
    assert [row["stop_loss_pips"] for row in long] == [row["stop_loss_pips"] for row in short]
    assert long[0]["stop_loss"] < 1.1000 < short[0]["stop_loss"]
    assert size_units_for_stop(max_loss_jpy=500.0, stop_loss_pips=8.0, pip_value_jpy_per_unit=0.01) < size_units_for_stop(max_loss_jpy=500.0, stop_loss_pips=4.0, pip_value_jpy_per_unit=0.01)
    catastrophe = next(row for row in long if row["geometry_id"] == "CONSERVATIVE_CATASTROPHE")
    assert catastrophe["server_side_catastrophic_stop"] is True
    assert catastrophe["live_candidate_eligible"] is True
    assert catastrophe["stop_loss_pips"] >= 18.0
    no_sl = next(row for row in long if row["geometry_id"] == "NO_SL_SHADOW_ONLY")
    signal = {
        "pair": "EUR_USD",
        "side": "LONG",
        "entry": 1.1,
        "stop_loss": None,
        "stop_loss_pips": None,
        "protective_stop": no_sl,
        "attached_stop_loss_required": False,
    }
    ok, reason, _ = validate_protective_stop(signal, now_utc=NOW)
    assert ok is False
    assert reason == "PROTECTIVE_STOP_NOT_CATASTROPHIC_LIVE_CANDIDATE"


def test_structure_exit_plan_and_velocity_are_mirrored():
    config, _ = _config()
    for side, sign in (("LONG", 1.0), ("SHORT", -1.0)):
        plan = structure_exit_plan(
            pair="EUR_USD", side=side, observed_at_utc=NOW, config=config
        )
        signal = {"pair": "EUR_USD", "side": side, "structure_exit_plan": plan}
        assert validate_structure_exit_plan(signal, now_utc=NOW) == (True, None)
        base = [1.1000 + sign * step * 0.00002 for step in range(7)]
        closes = base + [base[-1] - sign * 0.00020]
        highs = [value + 0.00002 for value in closes]
        lows = [value - 0.00002 for value in closes]
        result = evaluate_structure_exit(
            side=side,
            closes=closes,
            highs=highs,
            lows=lows,
            spreads_pips=[0.8] * len(closes),
            held_minutes=8,
            failed_continuation=False,
            pair="EUR_USD",
            plan=plan,
        )
        assert result["exit"] is True
        assert result["reason"] in {
            "ADVERSE_SWING_BREAK",
            "ADVERSE_VELOCITY",
            "ADVERSE_ACCELERATION",
        }
        assert result["evidence"]["atr_used_for_exit_trigger"] is False


def test_structure_exit_fails_closed_on_runtime_restart_evidence_gap_and_time_stops():
    config, _ = _config()
    plan = structure_exit_plan(
        pair="EUR_USD", side="LONG", observed_at_utc=NOW, config=config
    )
    insufficient = evaluate_structure_exit(
        side="LONG",
        closes=[1.1, 1.1001],
        highs=[1.1001, 1.1002],
        lows=[1.0999, 1.1],
        spreads_pips=[0.8, 0.8],
        held_minutes=2,
        failed_continuation=False,
        pair="EUR_USD",
        plan=plan,
    )
    assert insufficient == {
        "exit": True,
        "reason": "STRUCTURE_EVIDENCE_INSUFFICIENT",
        "fail_closed": True,
    }
    flat = [1.1] * 8
    timed = evaluate_structure_exit(
        side="LONG",
        closes=flat,
        highs=[1.1001] * 8,
        lows=[1.0999] * 8,
        spreads_pips=[0.8] * 8,
        held_minutes=int(plan["time_stop_minutes"]),
        failed_continuation=False,
        pair="EUR_USD",
        plan=plan,
    )
    assert timed["exit"] is True
    assert timed["reason"] == "TIME_STOP"


def test_guard_rejects_shock_and_emits_paper_only_drain_without_touching_manual():
    config, config_sha = _config()
    state = advance_state(prior=_normal(), observation=_direct_observation("DOWN", new_extreme=False, adverse_pips=0.0), now_utc=NOW, config=config, config_sha256=config_sha)
    signal = {
        "signal_id": "s1",
        "signal_sha256": "x",
        "pair": "EUR_USD",
        "side": "SHORT",
        "method": "TREND_CONTINUATION",
        "entry": 1.1000,
        "take_profit": 1.0990,
        "take_profit_pips": 10.0,
        "stop_loss": 1.10032,
        "stop_loss_pips": 3.2,
        "m5_atr_pips": 4.0,
        "spread_pips": 0.8,
        "attached_stop_loss_required": True,
    }
    guarded, decisions = guard_shadow(shadow={"signals": [signal]}, state=state, pair_charts=_chart(direction="DOWN"), config=config, config_sha256=config_sha, now_utc=NOW)
    assert guarded["signals"] == []
    assert decisions[0]["entry_allowed"] is False
    assert decisions[0]["drain_intent"] == {
        "scope": "BOT_OWNED_ONLY",
        "fraction": 0.5,
        "execution_scope": "PAPER_SHADOW_ONLY",
        "manual_tagless_policy": "NO_TOUCH",
    }
    assert decisions[0]["external_order_attempts"] == decisions[0]["external_orders"] == 0


def test_pair_states_are_isolated_and_missing_pair_state_fails_closed():
    config, config_sha = _config()
    eur_state = advance_state(
        prior=_normal("EUR_USD"),
        observation=_direct_observation(
            "DOWN",
            new_extreme=False,
            adverse_pips=0.0,
            pair="EUR_USD",
        ),
        now_utc=NOW,
        config=config,
        config_sha256=config_sha,
    )
    usd_state = _normal("USD_JPY")
    packet = _chart(pair="EUR_USD", direction="DOWN")
    packet["charts"].extend(_chart(pair="USD_JPY", direction="UP")["charts"])
    signals = [
        {
            "signal_id": "eur-frozen",
            "pair": "EUR_USD",
            "side": "SHORT",
            "method": "TREND_CONTINUATION",
            "entry": 1.1,
            "take_profit_pips": 10.0,
            "m5_atr_pips": 4.0,
            "spread_pips": 0.8,
        },
        {
            "signal_id": "usd-normal",
            "pair": "USD_JPY",
            "side": "LONG",
            "method": "TREND_CONTINUATION",
            "entry": 150.0,
            "take_profit_pips": 10.0,
            "m5_atr_pips": 4.0,
            "spread_pips": 0.8,
        },
    ]
    guarded, decisions = guard_shadow(
        shadow={"contract_sha256": "source", "signals": signals},
        state=eur_state,
        states={"EUR_USD": eur_state, "USD_JPY": usd_state},
        pair_charts=packet,
        config=config,
        config_sha256=config_sha,
        now_utc=NOW,
    )
    by_pair = {row["pair"]: row for row in decisions}
    assert by_pair["EUR_USD"]["entry_allowed"] is False
    assert by_pair["EUR_USD"]["state"] == SHOCK_FREEZE
    assert by_pair["USD_JPY"]["entry_allowed"] is True
    assert by_pair["USD_JPY"]["state"] == NORMAL
    assert [row["pair"] for row in guarded["signals"]] == ["USD_JPY"]

    missing_guarded, missing_decisions = guard_shadow(
        shadow={"contract_sha256": "source", "signals": [signals[1]]},
        state=eur_state,
        pair_charts=packet,
        config=config,
        config_sha256=config_sha,
        now_utc=NOW,
    )
    assert missing_guarded["signals"] == []
    assert missing_decisions[0]["rejection_reason"] == "SHOCK_GUARD_SHOCK_FREEZE"


def test_restart_restore_duplicate_event_and_invalid_state(tmp_path: Path):
    config, config_sha = _config()
    chart = _chart(direction="DOWN")
    shadow = {"contract_sha256": canonical_sha({}), "signals": []}
    kwargs = dict(pair_charts=chart, shadow=shadow, config=config, config_sha256=config_sha, state_path=tmp_path / "state.json", decision_ledger_path=tmp_path / "decisions.jsonl", scorecard_path=tmp_path / "scorecard.json", output_path=tmp_path / "guarded.json", now_utc=NOW + timedelta(minutes=1))
    first = run_guard_cycle(**kwargs)
    second = run_guard_cycle(**kwargs)
    assert first["event_id"] == second["event_id"]
    assert second["decision_ledger_appended"] == 0

    (tmp_path / "state.json").write_text("{}", encoding="utf-8")
    third = run_guard_cycle(**{**kwargs, "now_utc": NOW + timedelta(minutes=2)})
    assert third["state"] == SHOCK_FREEZE
    restored = json.loads((tmp_path / "scorecard.json").read_text())
    assert restored["restart_restore_valid"] is False


def test_run_cycle_persists_one_durable_state_per_pair(tmp_path: Path):
    config, config_sha = _config()
    packet = _chart(pair="EUR_USD", direction="DOWN")
    packet["charts"].extend(_chart(pair="USD_JPY", direction="UP")["charts"])
    result = run_guard_cycle(
        pair_charts=packet,
        shadow={"contract_sha256": canonical_sha({}), "signals": []},
        config=config,
        config_sha256=config_sha,
        state_path=tmp_path / "state.json",
        decision_ledger_path=tmp_path / "decisions.jsonl",
        scorecard_path=tmp_path / "scorecard.json",
        output_path=tmp_path / "guarded.json",
        now_utc=NOW + timedelta(minutes=1),
    )
    assert set(result["states"]) == {"EUR_USD", "USD_JPY"}
    assert Path(result["state_paths"]["EUR_USD"]).is_file()
    assert Path(result["state_paths"]["USD_JPY"]).is_file()
    scorecard = json.loads((tmp_path / "scorecard.json").read_text())
    assert set(scorecard["current_states"]) == {"EUR_USD", "USD_JPY"}
    assert scorecard["external_order_attempts"] == scorecard["external_orders"] == 0
    assert scorecard["manual_tagless_policy"] == "NO_TOUCH"
    assert scorecard["existing_tp_sl_policy"] == "NO_TOUCH"
