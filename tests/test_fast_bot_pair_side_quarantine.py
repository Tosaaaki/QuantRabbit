from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from quant_rabbit.fast_bot import REGIME_CONTRACT, _seal, build_fast_bot_shadow
from quant_rabbit.fast_bot_pair_side_quarantine import load_jsonl, run_selection


ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "config" / "fast_bot_pair_side_quarantine_v1.json"
CUTOFF = datetime(2026, 9, 1, 15, 20, tzinfo=timezone.utc)


def _signals() -> list[dict]:
    rows = [
        {
            "pair": pair,
            "side": side,
            "method": "RANGE_ROTATION",
            "state": "GO",
            "execution_enabled": True,
            "score": 5.0,
            "m1_closed_candle_utc": CUTOFF.isoformat(),
            "m5_atr_pips": 5.0,
        }
        for pair, side in (("EUR_USD", "SHORT"), ("USD_JPY", "LONG"))
    ]
    regime = _seal(
        {
            "contract": REGIME_CONTRACT,
            "schema_version": 1,
            "generated_at_utc": CUTOFF.isoformat(),
            "rows": rows,
        }
    )
    snapshot = {
        "fetched_at_utc": CUTOFF.isoformat(),
        "quotes": {
            "EUR_USD": {"bid": 1.16, "ask": 1.16008, "timestamp_utc": CUTOFF.isoformat()},
            "USD_JPY": {"bid": 147.0, "ask": 147.008, "timestamp_utc": CUTOFF.isoformat()},
        },
    }
    return build_fast_bot_shadow(regime, broker_snapshot=snapshot, now_utc=CUTOFF)["signals"]


def test_quarantine_excludes_eurusd_short_and_keeps_control_ledger(tmp_path: Path) -> None:
    signals = _signals()
    raw = tmp_path / "raw.jsonl"
    raw.write_text("".join(json.dumps(row) + "\n" for row in signals), encoding="utf-8")
    selected = tmp_path / "selected.jsonl"
    decisions = tmp_path / "decisions.jsonl"
    output = tmp_path / "latest.json"

    result = run_selection(
        raw_signal_ledger_path=raw,
        policy_path=POLICY,
        selected_ledger_path=selected,
        decision_ledger_path=decisions,
        output_path=output,
        now_utc=datetime(2026, 9, 1, 15, 20, 20, tzinfo=timezone.utc),
    )

    assert len(load_jsonl(raw)) == 2
    assert [(row["pair"], row["side"]) for row in load_jsonl(selected)] == [("USD_JPY", "LONG")]
    rows = load_jsonl(decisions)
    blocked = next(row for row in rows if row["pair"] == "EUR_USD")
    assert blocked["status"] == "REJECTED"
    assert blocked["reasons"] == ["PRECOMMITTED_PAIR_SIDE_QUARANTINE"]
    assert result["quarantined_signals_this_run"] == 1
    assert result["execution_authority"] == "NONE"
    assert result["external_orders"] == 0

    replay = run_selection(
        raw_signal_ledger_path=raw,
        policy_path=POLICY,
        selected_ledger_path=selected,
        decision_ledger_path=decisions,
        output_path=output,
        now_utc=datetime(2026, 9, 1, 15, 20, 30, tzinfo=timezone.utc),
    )
    assert replay["decisions_appended"] == 0
    assert replay["selected_signals_appended"] == 0
    assert len(load_jsonl(selected)) == 1


def test_quarantine_never_backfills_expired_signal(tmp_path: Path) -> None:
    allowed = next(row for row in _signals() if row["pair"] == "USD_JPY")
    raw = tmp_path / "raw.jsonl"
    raw.write_text(json.dumps(allowed) + "\n", encoding="utf-8")
    selected = tmp_path / "selected.jsonl"
    decisions = tmp_path / "decisions.jsonl"

    result = run_selection(
        raw_signal_ledger_path=raw,
        policy_path=POLICY,
        selected_ledger_path=selected,
        decision_ledger_path=decisions,
        output_path=tmp_path / "latest.json",
        now_utc=datetime(2026, 9, 1, 15, 21, tzinfo=timezone.utc),
    )

    assert result["status"] == "REJECTED"
    assert not selected.exists()
    assert load_jsonl(decisions)[0]["reasons"] == ["SELECTION_WINDOW_EXPIRED"]
