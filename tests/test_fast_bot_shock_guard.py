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
    validate_protective_stop,
)


ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 8, 28, 14, 0, tzinfo=timezone.utc)


def _config():
    return load_config(ROOT / "config" / "fast_bot_shock_guard_v1.json")


def _chart(*, direction: str = "DOWN", pips: float = 18.0, gap: bool = False, stale: bool = False):
    start = NOW - timedelta(minutes=15)
    sign = 1.0 if direction == "UP" else -1.0
    rows = []
    for index in range(16):
        offset = index + (1 if gap and index == 8 else 0)
        at = start + timedelta(minutes=offset)
        if stale:
            at -= timedelta(minutes=10)
        price = 1.1000 + sign * (pips / 10_000.0) * index / 15.0
        rows.append(
            {
                "t": at.isoformat(),
                "o": price,
                "h": price + 0.00002,
                "l": price - 0.00002,
                "c": price,
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
    return {"generated_at_utc": NOW.isoformat(), "charts": [{"pair": "EUR_USD", "views": views}]}


def _normal():
    return seal(
        {
            "contract": "QR_FAST_BOT_SHOCK_GUARD_STATE_V1",
            "schema_version": 1,
            "pair": "EUR_USD",
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


def _direct_observation(direction: str, *, new_extreme: bool, adverse_pips: float):
    sign = 1.0 if direction == "UP" else -1.0
    initial_high = 1.1020
    initial_low = 1.0980
    post = []
    for index in range(1, 6):
        if direction == "UP":
            high = initial_high + (0.0001 if new_extreme else -0.00001)
            low = initial_high - adverse_pips / 10_000.0
        else:
            low = initial_low - (0.0001 if new_extreme else -0.00001)
            high = initial_low + adverse_pips / 10_000.0
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
        "pair": "EUR_USD",
        "latest_complete_m1_at_utc": post[-1]["at"],
        "impulse_direction": direction,
        "impulse_magnitude_pips": 20.0,
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


def test_protective_stop_geometry_symmetry_and_inverse_units():
    config, _ = _config()
    long = protective_stop_candidates(pair="EUR_USD", side="LONG", entry=1.1000, atr_pips=4.0, spread_pips=0.8, recent_swing_price=1.0995, observed_at_utc=NOW, config=config)
    short = protective_stop_candidates(pair="EUR_USD", side="SHORT", entry=1.1000, atr_pips=4.0, spread_pips=0.8, recent_swing_price=1.1005, observed_at_utc=NOW, config=config)
    assert [row["stop_loss_pips"] for row in long] == [row["stop_loss_pips"] for row in short]
    assert long[0]["stop_loss"] < 1.1000 < short[0]["stop_loss"]
    assert size_units_for_stop(max_loss_jpy=500.0, stop_loss_pips=8.0, pip_value_jpy_per_unit=0.01) < size_units_for_stop(max_loss_jpy=500.0, stop_loss_pips=4.0, pip_value_jpy_per_unit=0.01)


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
