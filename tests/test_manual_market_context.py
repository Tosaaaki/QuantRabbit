from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from quant_rabbit.analysis.candles import Candle
from quant_rabbit.manual_market_context import build_manual_market_context_audit


def _series(start: datetime, *, minutes: int, n: int, first: float, step: float) -> list[Candle]:
    out: list[Candle] = []
    prev = first
    for index in range(n):
        close = first + step * index
        high = max(prev, close) + abs(step) * 0.7 + 0.02
        low = min(prev, close) - abs(step) * 0.7 - 0.02
        out.append(
            Candle(
                timestamp_utc=start + timedelta(minutes=minutes * index),
                open=prev,
                high=high,
                low=low,
                close=close,
                volume=1000 + index,
                complete=True,
            )
        )
        prev = close
    return out


def _manual_payload(start: datetime) -> dict:
    def ts(hours: float) -> str:
        return (start + timedelta(hours=hours)).isoformat().replace("+00:00", "Z")

    return {
        "transaction_count": 10,
        "exit_events": 3,
        "closed_trades": 3,
        "reduced_trades": 0,
        "trades": [
            {
                "trade_id": "long-1",
                "pair": "USD_JPY",
                "units": 10000,
                "open_time": ts(40),
                "close_time": ts(40.5),
                "hold_hours": 0.5,
                "open_price": 151.0,
                "realized_pl": 1000.0,
                "financing": 0.0,
                "close_reason": "MARKET_ORDER_TRADE_CLOSE",
            },
            {
                "trade_id": "long-2",
                "pair": "USD_JPY",
                "units": 10000,
                "open_time": ts(45),
                "close_time": ts(45.25),
                "hold_hours": 0.25,
                "open_price": 151.2,
                "realized_pl": 800.0,
                "financing": 0.0,
                "close_reason": "TAKE_PROFIT_ORDER",
            },
            {
                "trade_id": "short-1",
                "pair": "USD_JPY",
                "units": -10000,
                "open_time": ts(50),
                "close_time": ts(53),
                "hold_hours": 3.0,
                "open_price": 151.4,
                "realized_pl": -900.0,
                "financing": 0.0,
                "close_reason": "MARKET_ORDER_TRADE_CLOSE",
            },
        ],
    }


class ManualMarketContextAuditTest(unittest.TestCase):
    def test_extracts_trend_alignment_context_from_manual_trades(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            start = datetime(2025, 6, 1, tzinfo=timezone.utc)
            manual = root / "manual.json"
            manual.write_text(json.dumps(_manual_payload(start)))
            candles = {
                "H1": _series(start, minutes=60, n=90, first=150.0, step=0.03),
                "M5": _series(start, minutes=5, n=900, first=150.0, step=0.0025),
            }

            summary = build_manual_market_context_audit(
                manual_history_path=manual,
                output_path=root / "audit.json",
                report_path=root / "audit.md",
                candles_by_tf=candles,
            )

            self.assertEqual(summary.status, "MANUAL_MARKET_CONTEXT_PASS")
            self.assertEqual(summary.analyzed_trades, 3)
            self.assertEqual(summary.best_h1_alignment, "WITH_H1_TREND")
            self.assertEqual(summary.worst_h1_alignment, "AGAINST_H1_TREND")
            payload = json.loads((root / "audit.json").read_text())
            h1 = {row["bucket"]: row for row in payload["bounded_replay_profile"]["by_h1_alignment"]}
            self.assertEqual(h1["WITH_H1_TREND"]["net_jpy"], 1800.0)
            self.assertEqual(h1["AGAINST_H1_TREND"]["net_jpy"], -900.0)
            self.assertEqual(payload["guidance"]["basis"], "bounded_replay_lt_12h_excluding_margin_closeout")
            self.assertEqual(
                payload["guidance"]["require_extra_current_reason_when_conflicting"]["h1_alignment"],
                "AGAINST_H1_TREND",
            )
            self.assertTrue(payload["contract"]["advisory_only"])
            self.assertIn("RiskEngine", payload["contract"]["cannot_override"])

    def test_guidance_uses_bounded_profile_when_raw_tail_dominates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            start = datetime(2025, 6, 1, tzinfo=timezone.utc)
            payload = _manual_payload(start)
            payload["trades"].append(
                {
                    "trade_id": "tail-short",
                    "pair": "USD_JPY",
                    "units": -10000,
                    "open_time": (start + timedelta(hours=55)).isoformat().replace("+00:00", "Z"),
                    "close_time": (start + timedelta(hours=80)).isoformat().replace("+00:00", "Z"),
                    "hold_hours": 25.0,
                    "open_price": 151.5,
                    "realized_pl": 10000.0,
                    "financing": 0.0,
                    "close_reason": "MARKET_ORDER_POSITION_CLOSEOUT",
                }
            )
            manual = root / "manual.json"
            manual.write_text(json.dumps(payload))
            candles = {
                "H1": _series(start, minutes=60, n=100, first=150.0, step=0.03),
                "M5": _series(start, minutes=5, n=1100, first=150.0, step=0.0025),
            }

            summary = build_manual_market_context_audit(
                manual_history_path=manual,
                output_path=root / "audit.json",
                report_path=root / "audit.md",
                candles_by_tf=candles,
            )

            audit = json.loads((root / "audit.json").read_text())
            self.assertEqual(summary.status, "MANUAL_MARKET_CONTEXT_WARN")
            self.assertEqual(audit["guidance"]["prefer_when_citing_precedent"]["h1_alignment"], "WITH_H1_TREND")
            raw_h1 = {row["bucket"]: row for row in audit["technical_profile"]["by_h1_alignment"]}
            bounded_h1 = {row["bucket"]: row for row in audit["bounded_replay_profile"]["by_h1_alignment"]}
            self.assertGreater(raw_h1["AGAINST_H1_TREND"]["net_jpy"], raw_h1["WITH_H1_TREND"]["net_jpy"])
            self.assertGreater(bounded_h1["WITH_H1_TREND"]["net_jpy"], bounded_h1["AGAINST_H1_TREND"]["net_jpy"])
            self.assertTrue(any("raw H1 alignment net is dominated" in item for item in audit["warnings"]))

    def test_missing_candles_blocks_instead_of_guessing_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manual = root / "manual.json"
            manual.write_text(json.dumps(_manual_payload(datetime(2025, 6, 1, tzinfo=timezone.utc))))

            summary = build_manual_market_context_audit(
                manual_history_path=manual,
                output_path=root / "audit.json",
                report_path=root / "audit.md",
                candles_by_tf={"M5": [], "H1": []},
            )

            self.assertEqual(summary.status, "MANUAL_MARKET_CONTEXT_BLOCKED")
            payload = json.loads((root / "audit.json").read_text())
            self.assertGreater(len(payload["blockers"]), 0)
            self.assertIn("technical context could not be computed", payload["blockers"][-1])


if __name__ == "__main__":
    unittest.main()
