from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from quant_rabbit.contextual_technical_forward import (
    OUTCOME_CONTRACT,
    SHADOW_CONTRACT,
    BidAskCandle,
    Ohlc,
    _feature_rows,
    _pair_selector_rows,
    _seal,
    build_forward_scorecard,
    decision_window,
    load_jsonl,
    load_candidate,
)


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_PATH = ROOT / "config" / "contextual_technical_240m_forward_candidate_v1.json"


def _candles(count: int = 2400) -> list[BidAskCandle]:
    start = datetime(2026, 6, 1, tzinfo=timezone.utc)
    rows: list[BidAskCandle] = []
    for index in range(count):
        timestamp = start + timedelta(minutes=5 * index)
        mid_open = 0.9000 + index * 0.000002 + math.sin(index / 31.0) * 0.0002
        mid_close = mid_open + math.sin(index / 7.0) * 0.00003
        spread = 0.00012 + (index % 3) * 0.00001
        bid_open = mid_open - spread / 2.0
        ask_open = mid_open + spread / 2.0
        bid_close = mid_close - spread / 2.0
        ask_close = mid_close + spread / 2.0
        rows.append(
            BidAskCandle(
                timestamp_utc=timestamp,
                pair="USD_CHF",
                bid=Ohlc(
                    bid_open,
                    max(bid_open, bid_close) + 0.00004,
                    min(bid_open, bid_close) - 0.00004,
                    bid_close,
                ),
                ask=Ohlc(
                    ask_open,
                    max(ask_open, ask_close) + 0.00004,
                    min(ask_open, ask_close) - 0.00004,
                    ask_close,
                ),
            )
        )
    return rows


def test_candidate_is_locked_shadow_only() -> None:
    candidate = load_candidate(CANDIDATE_PATH)
    assert candidate["live_order_enabled"] is False
    assert candidate["promotion_allowed"] is False
    assert candidate["selection_disclosure"]["historical_holdout_inspected"] is True


def test_decision_window_is_only_open_for_locked_ninety_seconds() -> None:
    candidate = load_candidate(CANDIDATE_PATH)
    opened = decision_window(
        candidate,
        as_of_utc=datetime(2026, 7, 16, 4, 0, 30, tzinfo=timezone.utc),
    )
    late = decision_window(
        candidate,
        as_of_utc=datetime(2026, 7, 16, 4, 1, 31, tzinfo=timezone.utc),
    )
    assert opened["status"] == "OPEN"
    assert opened["decision_at_utc"] == "2026-07-16T04:00:00+00:00"
    assert late["status"] == "OUTSIDE_COLLECTION_WINDOW"


def test_pure_python_runtime_features_match_research_dataframe() -> None:
    np = pytest.importorskip("numpy", reason="optional research dependency")
    pd = pytest.importorskip("pandas", reason="optional research dependency")
    from scripts.audit_market_phase_technical_selector import (
        RULES,
        _phase_and_rule_frame,
    )
    from scripts.train_causal_technical_forecaster import _pair_feature_frame

    candles = _candles()
    runtime = _feature_rows("USD_CHF", candles)
    research = _phase_and_rule_frame(
        _pair_feature_frame("USD_CHF", candles, pair_code=0, np=np, pd=pd),
        np=np,
        pd=pd,
    )
    for runtime_row in runtime[-20:]:
        timestamp = runtime_row["timestamp_utc"]
        research_row = research.loc[timestamp]
        assert runtime_row["market_phase"] == research_row["market_phase"]
        for rule in RULES:
            if pd.isna(research_row[rule]):
                assert runtime_row[rule] is None
            else:
                assert abs(float(runtime_row[rule]) - float(research_row[rule])) < 1e-10


def test_runtime_history_uses_only_exactly_resolved_horizons() -> None:
    candles = _candles(2600)
    decision = candles[-49].timestamp_utc
    history_candles = [item for item in candles if item.timestamp_utc < decision]
    entry_quote = [item for item in candles if item.timestamp_utc == decision]
    history, _current = _pair_selector_rows(
        "USD_CHF",
        history_candles,
        entry_quote,
        decision_at_utc=decision,
        selector={
            "context_lookback_days": 14,
            "horizon_min": 240,
            "schedule_interval_min": 240,
            "spread_cap_pips": 2.0,
        },
    )
    forecast_boundary = decision - timedelta(minutes=5)
    assert history
    assert all(
        datetime.fromisoformat(row["future_timestamp_utc"]) <= forecast_boundary
        for row in history
    )


def test_fixed_forward_cohort_can_pass_review_but_never_promotes_live() -> None:
    candidate = load_candidate(CANDIDATE_PATH)
    candidate_sha = "a" * 64
    signals = []
    outcomes = []
    start = datetime(2026, 7, 17, tzinfo=timezone.utc)
    for index in range(120):
        decision = start + timedelta(hours=4 * index)
        signal_sha = f"{index:064x}"
        signals.append(
            {
                "signal_sha256": signal_sha,
                "decision_at_utc": decision.isoformat(),
                "pair": "USD_CHF",
            }
        )
        outcomes.append(
            {
                "contract": OUTCOME_CONTRACT,
                "candidate_sha256": candidate_sha,
                "signal_sha256": signal_sha,
                "decision_at_utc": decision.isoformat(),
                "conservative_pips": 2.0,
                "exit_reason": "TIME_CLOSE",
            }
        )
    shadows = [
        {
            "contract": SHADOW_CONTRACT,
            "candidate_sha256": candidate_sha,
            "signals": signals,
        }
    ]
    scorecard = build_forward_scorecard(
        candidate,
        shadows,
        outcomes,
        candidate_sha256=candidate_sha,
        as_of_utc=start + timedelta(days=30),
    )
    assert scorecard["forward_evidence_passed"] is True
    assert scorecard["promotion_allowed"] is False
    assert scorecard["live_order_enabled"] is False


def test_forward_ledger_rejects_tamper_and_duplicate_identity(tmp_path: Path) -> None:
    candidate_sha = "b" * 64
    row = _seal(
        {
            "contract": SHADOW_CONTRACT,
            "schema_version": 1,
            "candidate_sha256": candidate_sha,
            "decision_id": "decision-1",
            "signals": [],
        },
        "shadow_sha256",
    )
    path = tmp_path / "shadow.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert load_jsonl(
        path,
        contract=SHADOW_CONTRACT,
        candidate_sha=candidate_sha,
    ) == [row]

    tampered = dict(row)
    tampered["signals"] = [{"pair": "USD_CHF"}]
    path.write_text(json.dumps(tampered) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="seal mismatch"):
        load_jsonl(
            path,
            contract=SHADOW_CONTRACT,
            candidate_sha=candidate_sha,
        )

    path.write_text(
        json.dumps(row) + "\n" + json.dumps(row) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicated"):
        load_jsonl(
            path,
            contract=SHADOW_CONTRACT,
            candidate_sha=candidate_sha,
        )
