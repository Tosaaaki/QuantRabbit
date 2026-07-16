from __future__ import annotations

import json
import tempfile
import unittest
import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

from quant_rabbit.fast_bot import (
    AI_SUPERVISION_CONTRACT,
    build_fast_bot_shadow,
    build_hierarchical_regime_contract,
    run_fast_bot_shadow,
)


NOW = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)
TF_MINUTES = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D": 1440}


def _candles(timeframe: str, *, failed_break_short: bool = False) -> list[dict]:
    minutes = TF_MINUTES[timeframe]
    rows = []
    for index in range(21):
        started = NOW - timedelta(minutes=(21 - index) * minutes)
        rows.append(
            {
                "t": started.isoformat(),
                "o": 1.105,
                "h": 1.110,
                "l": 1.100,
                "c": 1.105,
                "complete": True,
            }
        )
    if failed_break_short:
        rows[-1] = {
            "t": (NOW - timedelta(minutes=minutes)).isoformat(),
            "o": 1.109,
            "h": 1.111,
            "l": 1.104,
            "c": 1.109,
            "complete": True,
        }
    return rows


def _view(
    timeframe: str,
    *,
    direction: str,
    phase: str,
    readiness: str = "TRIGGERED",
    location: str = "MIDDLE_THIRD",
    value_zone: str = "FAIR_VALUE",
    extension: str = "BALANCED",
    failed_break_short: bool = False,
) -> dict:
    return {
        "granularity": timeframe,
        "recent_candles": _candles(timeframe, failed_break_short=failed_break_short),
        "candle_integrity": {"forecast_blocking": False},
        "indicators": {"atr_pips": 5.0},
        "market_state": {
            "direction": direction,
            "phase": phase,
            "readiness": readiness,
            "trigger": "BREAKOUT_CLOSE",
            "structure": "BREAKOUT_ACTIVE",
            "location": location,
            "value_zone": value_zone,
            "extension": extension,
            "evidence_complete": True,
        },
    }


def _inputs(*, failed_break_short: bool = False) -> tuple[dict, dict, dict]:
    fast = {
        "generated_at_utc": NOW.isoformat(),
        "charts": [
            {
                "pair": "EUR_USD",
                "views": [
                    _view("M1", direction="DOWN" if failed_break_short else "UP", phase="PRE_RANGE" if failed_break_short else "TREND"),
                    _view("M5", direction="DOWN" if failed_break_short else "UP", phase="PRE_RANGE" if failed_break_short else "TREND", failed_break_short=failed_break_short),
                    _view("M15", direction="DOWN" if failed_break_short else "UP", phase="PRE_RANGE" if failed_break_short else "TREND"),
                ],
            }
        ],
    }
    slow = {
        "generated_at_utc": NOW.isoformat(),
        "charts": [
            {
                "pair": "EUR_USD",
                "views": [
                    _view(tf, direction="DOWN" if failed_break_short else "UP", phase="PRE_RANGE" if failed_break_short and tf == "M30" else "TREND")
                    for tf in ("M30", "H1", "H4", "D")
                ],
            }
        ],
    }
    snapshot = {
        "fetched_at_utc": NOW.isoformat(),
        "quotes": {
            "EUR_USD": {
                "bid": 1.10000,
                "ask": 1.10008,
                "timestamp_utc": NOW.isoformat(),
            }
        },
        "positions": [],
        "orders": [],
    }
    return fast, slow, snapshot


def _row(contract: dict, *, side: str, method: str) -> dict:
    return next(
        item
        for item in contract["rows"]
        if item["pair"] == "EUR_USD" and item["side"] == side and item["method"] == method
    )


def _seal_contract(body: dict) -> dict:
    raw = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return {**body, "contract_sha256": hashlib.sha256(raw).hexdigest()}


class FastBotTest(unittest.TestCase):
    def test_hierarchical_trend_gate_is_bot_owned_and_go(self) -> None:
        fast, slow, snapshot = _inputs()
        contract = build_hierarchical_regime_contract(
            fast_pair_charts=fast,
            slow_pair_charts=slow,
            broker_snapshot=snapshot,
            guardian_events={"events": []},
            now_utc=NOW,
        )

        trend = _row(contract, side="LONG", method="TREND_CONTINUATION")
        self.assertEqual(trend["state"], "GO")
        self.assertTrue(trend["execution_enabled"])
        self.assertEqual(contract["entry_decision_authority"], "DETERMINISTIC_BOT")
        self.assertFalse(contract["ai_per_trade_approval_required"])
        self.assertEqual(
            contract["timeframe_roles"],
            {
                "execution": ["M1"],
                "operating": ["M5", "M15", "M30"],
                "structure": ["H1", "H4"],
                "anchor": ["D"],
            },
        )

    def test_technical_stale_event_stops_fast_entry_and_wakes_ai_only_for_change(self) -> None:
        fast, slow, snapshot = _inputs()
        contract = build_hierarchical_regime_contract(
            fast_pair_charts=fast,
            slow_pair_charts=slow,
            broker_snapshot=snapshot,
            guardian_events={
                "events": [
                    {"event_id": "stale", "event_type": "TECHNICAL_INPUT_STALE", "pair": "EUR_USD"},
                    {"event_id": "change", "event_type": "TECHNICAL_STATE_CHANGE", "pair": "EUR_USD"},
                ]
            },
            ai_supervision={"last_tuned_at_utc": (NOW - timedelta(hours=1)).isoformat()},
            now_utc=NOW,
        )

        trend = _row(contract, side="LONG", method="TREND_CONTINUATION")
        self.assertEqual(trend["state"], "STOP")
        self.assertIn("TECHNICAL_INPUT_STALE", trend["hard_blockers"])
        self.assertTrue(contract["ai_wake_required"])
        self.assertIn("GUARDIAN_EVENT:TECHNICAL_STATE_CHANGE:EUR_USD", contract["ai_wake_reasons"])

    def test_breakout_failure_binds_exact_m5_side(self) -> None:
        fast, slow, snapshot = _inputs(failed_break_short=True)
        contract = build_hierarchical_regime_contract(
            fast_pair_charts=fast,
            slow_pair_charts=slow,
            broker_snapshot=snapshot,
            guardian_events={"events": []},
            now_utc=NOW,
        )

        short = _row(contract, side="SHORT", method="BREAKOUT_FAILURE")
        long = _row(contract, side="LONG", method="BREAKOUT_FAILURE")
        self.assertEqual(short["failed_break_direction"], "SHORT")
        self.assertEqual(short["state"], "GO")
        self.assertEqual(long["state"], "STOP")
        self.assertIn("M5_FAILED_BREAK_DIRECTION_NOT_BOUND_TO_SIDE", long["hard_blockers"])

    def test_ai_regime_stop_is_pair_level_not_trade_approval(self) -> None:
        fast, slow, snapshot = _inputs()
        supervision = _seal_contract({
            "contract": AI_SUPERVISION_CONTRACT,
            "last_tuned_at_utc": NOW.isoformat(),
            "pairs": {
                "EUR_USD": {
                    "mode": "STOP",
                    "reason": "material volatility transition",
                    "expires_at_utc": (NOW + timedelta(minutes=30)).isoformat(),
                }
            },
        })
        contract = build_hierarchical_regime_contract(
            fast_pair_charts=fast,
            slow_pair_charts=slow,
            broker_snapshot=snapshot,
            guardian_events={"events": []},
            ai_supervision=supervision,
            now_utc=NOW,
        )

        trend = _row(contract, side="LONG", method="TREND_CONTINUATION")
        self.assertEqual(trend["state"], "STOP")
        self.assertIn("AI_REGIME_SUPERVISOR_STOP", trend["hard_blockers"])
        self.assertFalse(contract["ai_per_trade_approval_required"])

    def test_unsealed_ai_supervision_cannot_stop_or_reset_tuning_clock(self) -> None:
        fast, slow, snapshot = _inputs()
        contract = build_hierarchical_regime_contract(
            fast_pair_charts=fast,
            slow_pair_charts=slow,
            broker_snapshot=snapshot,
            guardian_events={"events": []},
            ai_supervision={
                "contract": AI_SUPERVISION_CONTRACT,
                "last_tuned_at_utc": NOW.isoformat(),
                "pairs": {
                    "EUR_USD": {
                        "mode": "STOP",
                        "expires_at_utc": (NOW + timedelta(minutes=30)).isoformat(),
                    }
                },
            },
            now_utc=NOW,
        )

        trend = _row(contract, side="LONG", method="TREND_CONTINUATION")
        self.assertEqual(trend["state"], "GO")
        self.assertEqual(trend["ai_supervision"]["mode"], "UNSUPERVISED")
        self.assertTrue(contract["tuning_due"])

    def test_shadow_signal_has_no_live_permission_and_ledger_dedupes(self) -> None:
        fast, slow, snapshot = _inputs()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = {
                "fast": root / "fast.json",
                "slow": root / "slow.json",
                "snapshot": root / "snapshot.json",
                "events": root / "events.json",
                "regime": root / "regime.json",
                "shadow": root / "shadow.json",
                "ledger": root / "shadow.jsonl",
                "report": root / "report.md",
            }
            for key, value in (("fast", fast), ("slow", slow), ("snapshot", snapshot), ("events", {"events": []})):
                paths[key].write_text(json.dumps(value), encoding="utf-8")
            kwargs = dict(
                fast_pair_charts_path=paths["fast"],
                slow_pair_charts_path=paths["slow"],
                broker_snapshot_path=paths["snapshot"],
                guardian_events_path=paths["events"],
                ai_supervision_path=root / "missing-ai.json",
                regime_output_path=paths["regime"],
                shadow_output_path=paths["shadow"],
                shadow_ledger_path=paths["ledger"],
                report_path=paths["report"],
                now_utc=NOW,
            )
            first = run_fast_bot_shadow(**kwargs)
            second = run_fast_bot_shadow(**kwargs)
            shadow = json.loads(paths["shadow"].read_text())
            ledger_rows = [json.loads(line) for line in paths["ledger"].read_text().splitlines()]

        self.assertEqual(first["ledger_appended"], 1)
        self.assertEqual(second["ledger_appended"], 0)
        self.assertEqual(len(ledger_rows), 1)
        self.assertFalse(shadow["live_permission"])
        self.assertFalse(shadow["broker_mutation_allowed"])
        self.assertFalse(shadow["ai_per_trade_approval_required"])
        signal = shadow["signals"][0]
        self.assertEqual(len(signal["signal_sha256"]), 64)
        self.assertFalse(signal["broker_mutation_allowed"])
        self.assertGreater(shadow["signals"][0]["take_profit"], shadow["signals"][0]["entry"])


if __name__ == "__main__":
    unittest.main()
