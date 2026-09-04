from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from quant_rabbit.ai_evidence_adapter import (
    MAX_PACKET_BYTES,
    EvidencePaths,
    build_ai_evidence_packet,
    write_ai_evidence_packet,
)


NOW = datetime(2026, 9, 4, 3, 0, tzinfo=timezone.utc)


class AIEvidenceAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.paths = EvidencePaths(
            broker_snapshot=self.root / "broker_snapshot.json",
            pair_charts=self.root / "pair_charts.json",
            market_context_matrix=self.root / "market_context_matrix.json",
            news_health=self.root / "news_health.json",
            news_snapshot=self.root / "news_items.json",
            daily_target_state=self.root / "daily_target_state.json",
            capture_economics=self.root / "capture_economics.json",
            execution_timing=self.root / "execution_timing_audit.json",
        )
        self._write_inputs()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _dump(self, path: Path, value: object) -> None:
        path.write_text(json.dumps(value), encoding="utf-8")

    def _write_inputs(self) -> None:
        stamp = NOW.isoformat()
        self._dump(
            self.paths.broker_snapshot,
            {
                "fetched_at_utc": stamp,
                "account": {
                    "fetched_at_utc": stamp,
                    "last_transaction_id": "9001",
                    "balance_jpy": 100_000,
                    "nav_jpy": 101_000,
                    "margin_available_jpy": 80_000,
                    "margin_used_jpy": 21_000,
                    "financing_jpy": -12.5,
                    "hedging_enabled": True,
                },
                "quotes": {
                    "EUR_USD": {"bid": 1.1000, "ask": 1.1002, "timestamp_utc": stamp},
                },
                "positions": [
                    {"trade_id": "s1", "owner": "trader", "pair": "EUR_USD", "side": "LONG", "units": 100, "unrealized_pl_jpy": 50},
                    {"trade_id": "m1", "owner": "operator_manual", "pair": "USD_JPY", "side": "SHORT", "units": 200, "unrealized_pl_jpy": -20},
                    {"trade_id": "u1", "owner": "unknown", "pair": "GBP_USD", "side": "LONG", "units": 300, "unrealized_pl_jpy": 10},
                ],
                "orders": [],
            },
        )
        views = []
        for timeframe in ("M1", "M5", "M15", "H1", "H4", "D"):
            candles = [
                {"t": (NOW - timedelta(minutes=index)).isoformat(), "o": 1.1, "h": 1.2, "l": 1.0, "c": 1.11, "complete": True, "blob": "x" * 20_000}
                for index in range(30, 0, -1)
            ]
            candles.append({"t": (NOW + timedelta(minutes=1)).isoformat(), "o": 9, "h": 9, "l": 9, "c": 9, "complete": True})
            views.append(
                {
                    "granularity": timeframe,
                    "regime": "TREND_UP",
                    "indicators": {"atr_pips": 8.5},
                    "regime_reading": {"state": "TREND_STRONG", "hurst": 0.6, "adx": 30, "choppiness": 35, "atr_percentile": 70, "confidence": 0.75},
                    "family_scores": {"trend_score": 0.8, "mean_rev_score": 0.1, "breakout_score": 0.6, "disagreement": 0.2},
                    "market_state": {"phase": "TREND", "direction": "UP", "volatility": "NORMAL", "momentum": "ACCELERATING", "noise": "ORDERLY", "liquidity": "CLEAR", "evidence_complete": True},
                    "structure": {
                        "last_event": {"timestamp": (NOW - timedelta(minutes=2)).isoformat(), "kind": "BOS_UP", "broken_pivot_price": 1.09, "close_confirmed": True},
                        "swings": [
                            {"timestamp": (NOW - timedelta(minutes=8 - index)).isoformat(), "side": "HIGH" if index % 2 else "LOW", "price": 1.08 + index / 1000}
                            for index in range(8)
                        ],
                    },
                    "recent_candles": candles,
                    "hierarchical_bot_regime": {"must_not": "escape"},
                }
            )
        self._dump(
            self.paths.pair_charts,
            {
                "generated_at_utc": stamp,
                "charts": [{"pair": "EUR_USD", "dominant_regime": "TREND_UP", "long_score": 0.8, "short_score": 0.2, "views": views, "confluence": {"tf_agreement_score": 1.0}}],
            },
        )
        pair_raw = self.paths.pair_charts.read_bytes()
        self._dump(
            self.paths.market_context_matrix,
            {
                "generated_at_utc": stamp,
                "pair_charts_binding": {"sha256": hashlib.sha256(pair_raw).hexdigest()},
                "pairs": {
                    "EUR_USD": {
                        "LONG": {"evidence_ref": "matrix:EUR_USD:LONG", "support_count": 2, "reject_count": 0, "warning_count": 0, "missing_count": 0, "horizon_conflict_count": 0, "strongest_support": "aligned", "supports": []},
                        "SHORT": {"evidence_ref": "matrix:EUR_USD:SHORT", "support_count": 0, "reject_count": 2, "warning_count": 0, "missing_count": 0, "horizon_conflict_count": 0, "strongest_reject": "opposed", "rejects": []},
                    }
                },
                "order_intents": [{"pair": "EUR_USD", "units": 999999}],
            },
        )
        self._dump(self.paths.news_health, {"generated_at_utc": stamp, "status": "OK", "item_count": 1})
        self._dump(self.paths.news_snapshot, {"generated_at_utc": stamp, "items": [{"published_at_utc": stamp, "title": "Central bank update", "source": "wire", "pairs": ["EUR_USD"], "topics": ["central_bank"]}]})
        self._dump(self.paths.daily_target_state, {"as_of_utc": stamp, "status": "PURSUE_TARGET", "pace_state": "ON_PACE", "current_equity_raw": 101_000, "daily_risk_budget_jpy": 1_000, "remaining_risk_budget_jpy": 800})
        self._dump(self.paths.capture_economics, {"generated_at_utc": stamp, "status": "OK", "overall": {"trades": 20, "win_rate": 0.6, "expectancy_jpy_per_trade": 12.0}, "average_slippage_pips": 0.2})
        self._dump(self.paths.execution_timing, {"generated_at_utc": stamp, "status": "OK", "summary": {"average_latency_ms": 80}})

    def test_packet_is_canonical_compact_and_omits_full_candles(self) -> None:
        packet = build_ai_evidence_packet(self.paths, now_utc=NOW)
        raw = json.dumps(packet, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        self.assertLessEqual(len(raw) + 1, MAX_PACKET_BYTES)
        self.assertNotIn("recent_candles", raw.decode())
        latest = packet["markets"]["EUR_USD"]["timeframes"]["M5"]["latest_complete_ohlc"]
        self.assertEqual(latest["c"], 1.11)
        self.assertLess(latest["c"], 9)
        body = {key: value for key, value in packet.items() if key != "packet_sha256"}
        expected = hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        self.assertEqual(packet["packet_sha256"], expected)

    def test_old_bot_and_order_intent_data_never_enter_packet(self) -> None:
        packet = build_ai_evidence_packet(self.paths, now_utc=NOW)
        raw = json.dumps(packet, sort_keys=True).lower()
        self.assertNotIn("order_intents", raw)
        self.assertNotIn("hierarchical_bot_regime", raw)
        self.assertNotIn("fast_bot", raw)
        self.assertNotIn("target_trades_per_day", raw)
        self.assertNotIn("per_trade_risk_budget_jpy", raw)

    def test_stale_and_malformed_sources_are_explicit_and_not_consumed(self) -> None:
        broker = json.loads(self.paths.broker_snapshot.read_text())
        broker["fetched_at_utc"] = (NOW - timedelta(hours=1)).isoformat()
        broker["account"]["fetched_at_utc"] = broker["fetched_at_utc"]
        self._dump(self.paths.broker_snapshot, broker)
        self.paths.news_snapshot.write_text("{bad json", encoding="utf-8")
        packet = build_ai_evidence_packet(self.paths, now_utc=NOW)
        self.assertEqual(packet["status"], "BLOCKED")
        self.assertEqual(packet["sources"]["broker_snapshot"]["status"], "STALE")
        self.assertEqual(packet["sources"]["news_snapshot"]["status"], "MALFORMED")
        self.assertEqual(packet["broker"]["quotes"], {})
        self.assertEqual(packet["news"]["items"], [])

    def test_broker_owner_classification_keeps_manual_and_unknown_no_touch(self) -> None:
        packet = build_ai_evidence_packet(self.paths, now_utc=NOW)
        exposure = packet["broker"]["exposure"]
        self.assertEqual(exposure["system_position_count"], 1)
        self.assertEqual(exposure["no_touch_position_count"], 2)
        by_id = {row["trade_id"]: row for row in exposure["positions"]}
        self.assertEqual(by_id["s1"]["mutation_policy"], "GATEWAY_VALIDATION_REQUIRED")
        self.assertEqual(by_id["m1"]["mutation_policy"], "NO_TOUCH")
        self.assertEqual(by_id["u1"]["mutation_policy"], "NO_TOUCH")

    def test_unchanged_packet_does_not_rewrite_output(self) -> None:
        output = self.root / "packet.json"
        first = write_ai_evidence_packet(self.paths, output, now_utc=NOW)
        self.assertTrue(first.written)
        fixed_ns = 1_700_000_000_000_000_000
        os.utime(output, ns=(fixed_ns, fixed_ns))
        second = write_ai_evidence_packet(self.paths, output, now_utc=NOW + timedelta(seconds=1))
        self.assertFalse(second.written)
        self.assertEqual(output.stat().st_mtime_ns, fixed_ns)
        self.assertEqual(first.packet_sha256, second.packet_sha256)


if __name__ == "__main__":
    unittest.main()
