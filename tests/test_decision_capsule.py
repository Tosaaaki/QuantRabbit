"""Contract tests for direct operator decision capsules.

The audit failed the previous corpus on selection bias, inferred labels and
missing decision-time state. These tests pin the properties that keep this
recorder from repeating any of it.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from quant_rabbit.decision_capsule import (
    CAPSULE_TIMEFRAMES,
    ATR_PERIOD,
    CapsuleError,
    append_capsule,
    build_broker_context,
    build_capsule,
    normalize_pair,
    parse_intake,
    pip_size,
    timeframe_features,
    validate_capsule,
    verify_chain,
)

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "docs" / "schemas" / "manual_decision_capsule_v1.schema.json"
NOW = datetime(2026, 8, 19, 3, 30, tzinfo=timezone.utc)


def _candles(count: int, *, last_complete: bool = True, start: float = 150.0, step: float = 0.05) -> list[dict]:
    candles = []
    for index in range(count):
        close = start + step * index
        candles.append(
            {
                "time": (NOW - timedelta(minutes=count - index)).isoformat().replace("+00:00", "Z"),
                "complete": True if index < count - 1 else last_complete,
                "mid": {"o": f"{close - step:.5f}", "h": f"{close + 0.02:.5f}", "l": f"{close - 0.02:.5f}", "c": f"{close:.5f}"},
            }
        )
    return candles


def _timeframes(count: int = 30) -> list[dict]:
    return [timeframe_features(timeframe, _candles(count), "USD_JPY", NOW) for timeframe in CAPSULE_TIMEFRAMES]


def _broker_context() -> dict:
    return build_broker_context(
        quote_time_utc="2026-08-19T03:30:00Z", bid=150.100, ask=150.108, spread=0.8,
        nav=1_000_000.0, margin_available=900_000.0, margin_used=100_000.0,
        positions=[], orders=[], transaction_watermark="12345",
    )


def _capsule(line: str = "USDJPY skip 弱い") -> dict:
    return build_capsule(
        parse_intake(line), captured_at=NOW, decision_cutoff=NOW,
        timeframes=_timeframes(), broker_context=_broker_context(),
    )


# --------------------------------------------------------------------------- #
# Intake                                                                        #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "raw,pair", [("USDJPY skip", "USD_JPY"), ("usd_jpy skip", "USD_JPY"), ("EUR/USD skip", "EUR_USD")]
)
def test_pair_forms_normalize(raw: str, pair: str) -> None:
    assert parse_intake(raw).pair == pair


def test_long_folds_side_into_enter() -> None:
    intake = parse_intake("GBPJPY long 0.8 確信中")
    assert (intake.primary_action, intake.side, intake.confidence, intake.note) == ("ENTER", "LONG", 0.8, "確信中")


def test_skip_is_a_first_class_label() -> None:
    """SKIP is the negative case the previous corpus never had."""

    intake = parse_intake("USDJPY skip 弱い")
    assert intake.primary_action == "SKIP"
    assert intake.side is None
    assert intake.note == "弱い"


def test_confidence_is_never_inferred_from_words() -> None:
    """'確信中' is kept as text, never scored — that would fabricate a label."""

    intake = parse_intake("USDJPY skip 確信中")
    assert intake.confidence is None
    assert intake.note == "確信中"


def test_confidence_outside_unit_interval_is_rejected() -> None:
    with pytest.raises(CapsuleError):
        parse_intake("USDJPY long 1.4")


def test_unknown_action_is_rejected_not_guessed() -> None:
    with pytest.raises(CapsuleError):
        parse_intake("USDJPY maybe")


def test_bad_pair_is_rejected() -> None:
    with pytest.raises(CapsuleError):
        parse_intake("NOTAPAIR skip")


# --------------------------------------------------------------------------- #
# Features                                                                      #
# --------------------------------------------------------------------------- #


def test_forming_bar_never_enters_a_feature() -> None:
    """The last bar is incomplete, so bar_end_utc must be the one before it."""

    candles = _candles(30, last_complete=False)
    entry = timeframe_features("M5", candles, "USD_JPY", NOW)
    assert entry["complete"] is True
    assert entry["bar_end_utc"] == candles[-2]["time"]
    assert entry["candle"]["c"] == pytest.approx(float(candles[-2]["mid"]["c"]))


def test_features_are_null_when_bars_are_short_not_imputed() -> None:
    entry = timeframe_features("H4", _candles(ATR_PERIOD - 2), "USD_JPY", NOW)
    assert entry["atr"] is None
    assert entry["normalized_slope"] is None
    assert entry["momentum"] is None


def test_atr_is_reported_in_pips() -> None:
    entry = timeframe_features("M5", _candles(30), "USD_JPY", NOW)
    # True range is dominated by high - previous close = 0.05 step + 0.02 wick
    # = 0.07 price, which is 7 pips on a JPY-quote pair.
    assert entry["atr"] == pytest.approx(7.0, abs=0.1)


def test_pip_size_splits_jpy_from_the_rest() -> None:
    assert pip_size("USD_JPY") == 0.01
    assert pip_size("EUR_USD") == 0.0001


def test_normalized_slope_is_positive_on_a_rising_series() -> None:
    entry = timeframe_features("M5", _candles(30, step=0.05), "USD_JPY", NOW)
    assert entry["normalized_slope"] > 0
    assert 0 < entry["normalized_angle"] < 90


# --------------------------------------------------------------------------- #
# Capsule invariants                                                            #
# --------------------------------------------------------------------------- #


def test_capsule_validates_against_the_published_schema() -> None:
    validate_capsule(_capsule(), SCHEMA_PATH)


def test_capsule_carries_no_machine_label() -> None:
    capsule = _capsule()
    assert capsule["proxy_classifier"] is None
    assert capsule["inferred_label"] is None


def test_recorder_has_no_write_authority() -> None:
    assert _capsule()["broker_context"]["read_only"] is True


def test_capsule_id_is_content_addressed() -> None:
    assert _capsule()["capsule_id"] == _capsule()["capsule_id"]
    assert _capsule("USDJPY skip 弱い")["capsule_id"] != _capsule("USDJPY long 弱い")["capsule_id"]


def test_tampering_breaks_capsule_id_validation() -> None:
    capsule = _capsule()
    capsule["operator_evidence"]["side"] = "LONG"
    with pytest.raises(CapsuleError, match="does not match content"):
        validate_capsule(capsule, SCHEMA_PATH)


def test_inferred_label_is_rejected_by_validation() -> None:
    capsule = _capsule()
    capsule["inferred_label"] = "LONG"
    with pytest.raises(CapsuleError, match="null"):
        validate_capsule(capsule)


def test_absent_confidence_is_recorded_as_missing_with_a_reason() -> None:
    reasons = {item["field"]: item["reason"] for item in _capsule("USDJPY skip 弱い")["missing"]}
    assert "NEVER_INFERRED_FROM_WORDS" in reasons["operator_evidence.confidence"]


def test_undrawn_geometry_is_null_and_declared_missing() -> None:
    capsule = _capsule()
    geometry = capsule["market_context"]["geometry"]
    assert all(value is None for value in geometry.values())
    assert any(item["field"].startswith("market_context.geometry") for item in capsule["missing"])


def test_capsule_is_event_oversample_not_a_denominator() -> None:
    """A voiced decision never claims to speak for the other 27 pairs."""

    capsule = _capsule()
    assert capsule["population_stream"] == "EVENT_OVERSAMPLE"
    assert capsule["record_kind"] == "MANUAL_EVENT"


def test_seven_timeframes_are_mandatory_and_ordered() -> None:
    with pytest.raises(CapsuleError, match="expected 7"):
        build_capsule(
            parse_intake("USDJPY skip"), captured_at=NOW, decision_cutoff=NOW,
            timeframes=_timeframes()[:5], broker_context=_broker_context(),
        )


def test_broker_failure_still_yields_a_valid_capsule() -> None:
    """The label outlives the API. A null broker read must not lose the record."""

    empty = [timeframe_features(timeframe, [], "USD_JPY", NOW) for timeframe in CAPSULE_TIMEFRAMES]
    blank = build_broker_context(
        quote_time_utc=None, bid=None, ask=None, spread=None, nav=None,
        margin_available=None, margin_used=None, positions=None, orders=None, transaction_watermark=None,
    )
    capsule = build_capsule(
        parse_intake("USDJPY skip 弱い"), captured_at=NOW, decision_cutoff=NOW,
        timeframes=empty, broker_context=blank,
        extra_missing=[{"field": "broker_context", "reason": "BROKER_UNAVAILABLE: URLError"}],
    )
    validate_capsule(capsule, SCHEMA_PATH)
    assert capsule["operator_evidence"]["primary_action"] == "SKIP"


def test_decision_cutoff_precedes_every_bar_end() -> None:
    capsule = _capsule()
    cutoff = datetime.fromisoformat(capsule["decision_cutoff_utc"].replace("Z", "+00:00"))
    for entry in capsule["market_context"]["timeframes"]:
        assert datetime.fromisoformat(entry["bar_end_utc"].replace("Z", "+00:00")) <= cutoff


# --------------------------------------------------------------------------- #
# Append-only store                                                             #
# --------------------------------------------------------------------------- #


def test_append_builds_a_verifiable_chain(tmp_path: Path) -> None:
    for line in ("USDJPY skip 弱い", "GBPJPY long 0.8", "EURUSD short 様子見"):
        append_capsule(_capsule(line), tmp_path)
    report = verify_chain(tmp_path)
    assert report["status"] == "PASS"
    assert report["records"] == 3


def test_chain_detects_edited_history(tmp_path: Path) -> None:
    for line in ("USDJPY skip 弱い", "GBPJPY long 0.8"):
        append_capsule(_capsule(line), tmp_path)
    path = tmp_path / "capsules.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[0]["operator_evidence"]["primary_action"] = "ENTER"
    path.write_text("\n".join(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    assert verify_chain(tmp_path)["status"] == "FAIL"


def test_duplicate_intake_is_refused(tmp_path: Path) -> None:
    append_capsule(_capsule(), tmp_path)
    with pytest.raises(CapsuleError, match="duplicate"):
        append_capsule(_capsule(), tmp_path)


def test_empty_store_verifies_clean(tmp_path: Path) -> None:
    assert verify_chain(tmp_path)["status"] == "EMPTY"


def test_normalize_pair_rejects_junk() -> None:
    with pytest.raises(CapsuleError):
        normalize_pair("USDJPYX")
