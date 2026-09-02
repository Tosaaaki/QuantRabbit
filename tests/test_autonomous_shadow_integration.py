from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from quant_rabbit.autonomous_shadow_integration import run_autonomous_shadow_integration
from quant_rabbit.fast_bot import REGIME_CONTRACT, _seal, build_fast_bot_shadow
from quant_rabbit.fast_bot_knowledge import EPISODE_CONTRACT
from quant_rabbit.fast_bot_shock_guard import DECISION_CONTRACT, seal
from quant_rabbit.fast_bot_truth import OUTCOME_CONTRACT


NOW = datetime(2026, 9, 3, 3, 0, tzinfo=timezone.utc)


def _signal() -> dict:
    regime = _seal(
        {
            "contract": REGIME_CONTRACT,
            "schema_version": 1,
            "generated_at_utc": NOW.isoformat(),
            "rows": [
                {
                    "pair": "EUR_USD",
                    "side": "LONG",
                    "method": "TREND_CONTINUATION",
                    "state": "GO",
                    "execution_enabled": True,
                    "score": 6.0,
                    "m1_closed_candle_utc": NOW.isoformat(),
                    "m5_atr_pips": 5.0,
                }
            ],
        }
    )
    return build_fast_bot_shadow(
        regime,
        broker_snapshot={
            "fetched_at_utc": NOW.isoformat(),
            "quotes": {
                "EUR_USD": {
                    "bid": 1.10000,
                    "ask": 1.10008,
                    "timestamp_utc": NOW.isoformat(),
                }
            },
        },
        now_utc=NOW,
    )["signals"][0]


def _guard(signal: dict, *, allowed: bool = True) -> dict:
    return seal(
        {
            "contract": DECISION_CONTRACT,
            "schema_version": 1,
            "decision_id": f"guard:{signal['signal_id']}",
            "signal_id": signal["signal_id"],
            "signal_sha256": signal["signal_sha256"],
            "pair": signal["pair"],
            "side": signal["side"],
            "method": signal["method"],
            "strategy_id": signal["strategy_id"],
            "entry_allowed": allowed,
            "rejection_reason": None if allowed else "SHOCK_FREEZE",
            "execution_authority": "NONE",
            "broker_mutation_allowed": False,
            "external_order_attempts": 0,
            "external_orders": 0,
            "llm_order_fields_allowed": False,
        }
    )


def _outcome(signal: dict, *, filled: bool) -> dict:
    resolved = NOW + timedelta(minutes=20)
    return seal(
        {
            "contract": OUTCOME_CONTRACT,
            "schema_version": 3,
            "signal_id": signal["signal_id"],
            "signal_sha256": signal["signal_sha256"],
            "pair": signal["pair"],
            "side": signal["side"],
            "method": signal["method"],
            "resolved_at_utc": resolved.isoformat(),
            "filled": filled,
            "fill_at_utc": (NOW + timedelta(seconds=5)).isoformat() if filled else None,
            "exit_at_utc": (NOW + timedelta(minutes=15)).isoformat() if filled else None,
            "exit_reason": "TAKE_PROFIT" if filled else "UNFILLED",
            "truth_request_coverage_proved": True,
            "shadow_only": True,
            "live_permission": False,
            "broker_mutation": False,
        }
    )


def _learning(signal: dict, outcome: dict) -> dict:
    return seal(
        {
            "contract": EPISODE_CONTRACT,
            "schema_version": 1,
            "episode_id": f"episode:{signal['signal_id']}",
            "trade_id": signal["signal_id"],
            "resolved_at_utc": outcome["resolved_at_utc"],
            "raw_source_refs": {
                "signal_sha256": signal["signal_sha256"],
                "outcome_sha256": outcome["contract_sha256"],
            },
            "outcome": {"filled": outcome["filled"]},
            "execution_authority": "NONE",
            "shadow_only": True,
            "live_permission": False,
            "broker_mutation": False,
            "external_order_attempts": 0,
            "external_orders": 0,
        }
    )


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _run(tmp_path: Path, signal: dict, guard: dict, outcomes: list[dict], learning: list[dict]) -> dict:
    paths = {
        "shadow": tmp_path / "shadow.jsonl",
        "guard": tmp_path / "guard.jsonl",
        "outcome": tmp_path / "outcome.jsonl",
        "learning": tmp_path / "learning.jsonl",
    }
    for name, rows in (
        ("shadow", [signal]),
        ("guard", [guard]),
        ("outcome", outcomes),
        ("learning", learning),
    ):
        _write(paths[name], rows)
    return run_autonomous_shadow_integration(
        shadow_ledger_path=paths["shadow"],
        shock_guard_decision_ledger_path=paths["guard"],
        outcome_ledger_path=paths["outcome"],
        learning_episode_ledger_path=paths["learning"],
        state_root=tmp_path / "nervous",
        output_path=tmp_path / "state.json",
        report_path=tmp_path / "report.md",
        now_utc=NOW + timedelta(minutes=21),
    )


@pytest.mark.parametrize("filled", [True, False])
def test_resident_evidence_reaches_learning_without_live_authority(tmp_path: Path, filled: bool) -> None:
    signal = _signal()
    outcome = _outcome(signal, filled=filled)
    result = _run(tmp_path, signal, _guard(signal), [outcome], [_learning(signal, outcome)])
    rows = [
        json.loads(line)
        for line in (tmp_path / "nervous" / "episodes" / signal["signal_id"] / "synapses.jsonl").read_text().splitlines()
    ]

    assert result["status"] == "COMPLETE_THROUGH_SOURCE"
    assert result["episode_state_counts"] == {"LEARNED": 1}
    assert rows[-1]["to_state"] == "LEARNED"
    assert ("UNFILLED" in [row["to_state"] for row in rows]) is (not filled)
    assert all(row["execution_authority"] == "NONE" for row in rows)
    assert all(row["external_orders"] == 0 for row in rows)


def test_missing_outcome_waits_without_human_approval(tmp_path: Path) -> None:
    signal = _signal()
    result = _run(tmp_path, signal, _guard(signal), [], [])

    assert result["status"] == "WAITING_FOR_EVIDENCE"
    assert result["episode_state_counts"] == {"ADMITTED": 1}
    assert result["human_approval_required"] is False


def test_guard_rejection_expires_episode(tmp_path: Path) -> None:
    signal = _signal()
    result = _run(tmp_path, signal, _guard(signal, allowed=False), [], [])

    assert result["status"] == "COMPLETE_THROUGH_SOURCE"
    assert result["episode_state_counts"] == {"EXPIRED": 1}


def test_tampered_source_fails_closed_before_episode_append(tmp_path: Path) -> None:
    signal = _signal()
    signal["live_permission"] = True
    with pytest.raises(ValueError, match="invalid fast-bot shadow signal"):
        _run(tmp_path, signal, _guard(_signal()), [], [])
    assert not (tmp_path / "nervous" / "episodes").exists()
