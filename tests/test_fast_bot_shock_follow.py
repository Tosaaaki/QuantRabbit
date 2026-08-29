from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from quant_rabbit.fast_bot_shock_follow import (
    OUTCOME_CONTRACT,
    SIGNAL_CONTRACT,
    STRATEGIES,
    build_scorecard,
    build_shock_follow_shadow,
    load_config,
    market_is_closed,
    resolve_signal,
    run_incremental,
    seal,
    sealed_valid,
)
from quant_rabbit.technical_forecast_forward_outcome import S5BidAskCandle


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "fast_bot_shock_follow_v1.json"


def _candle(start: datetime, o: float, h: float, l: float, c: float) -> dict:
    return {"t": start.isoformat(), "o": o, "h": h, "l": l, "c": c, "v": 10, "complete": True}


def _packet(now: datetime, *, pullback: bool = False, m5_direction: str = "UP") -> tuple[dict, dict]:
    m1: list[dict] = []
    start = now - timedelta(minutes=17, seconds=10)
    price = 1.10000
    for index in range(14):
        opened = price + (0.00001 if index % 2 else 0.0)
        m1.append(_candle(start + timedelta(minutes=index), opened, opened + 0.00005, opened - 0.00005, opened))
    if pullback:
        anchor_start = start + timedelta(minutes=14)
        m1.extend(
            [
                _candle(anchor_start, 1.10000, 1.10022, 1.09998, 1.10020),
                _candle(anchor_start + timedelta(minutes=1), 1.10020, 1.10021, 1.10016, 1.10017),
                _candle(anchor_start + timedelta(minutes=2), 1.10017, 1.10030, 1.10017, 1.10025),
            ]
        )
    else:
        m1.append(_candle(start + timedelta(minutes=14), 1.10000, 1.10005, 1.09995, 1.10000))
        m1.extend(
            [
                _candle(start + timedelta(minutes=15), 1.10000, 1.10022, 1.09998, 1.10020),
                _candle(start + timedelta(minutes=16), 1.10020, 1.10030, 1.10019, 1.10025),
            ]
        )
    m5: list[dict] = []
    m5_start = now - timedelta(minutes=75, seconds=10)
    for index in range(14):
        m5.append(_candle(m5_start + timedelta(minutes=5 * index), 1.10000, 1.10005, 1.09995, 1.10000))
    if m5_direction == "UP":
        m5.append(_candle(m5_start + timedelta(minutes=70), 1.10000, 1.10020, 1.09999, 1.10018))
    else:
        m5.append(_candle(m5_start + timedelta(minutes=70), 1.10018, 1.10019, 1.09998, 1.10000))
    chart = {
        "pair": "EUR_USD",
        "views": [
            {
                "granularity": "M1",
                "candle_integrity": {
                    "schema": "QR_TECHNICAL_CANDLE_INTEGRITY_V2",
                    "source": "OANDA_MBA",
                    "pair": "EUR_USD",
                    "granularity": "M1",
                    "evaluation_status": "PASS",
                    "forecast_blocking": False,
                    "provenance_complete": True,
                    "coverage_complete": True,
                    "recent_clean_coverage_complete": True,
                },
                "market_state": {"direction": "UP", "evidence_complete": True},
                "recent_candles": m1,
            },
            {
                "granularity": "M5",
                "candle_integrity": {
                    "schema": "QR_TECHNICAL_CANDLE_INTEGRITY_V2",
                    "source": "OANDA_MBA",
                    "pair": "EUR_USD",
                    "granularity": "M5",
                    "evaluation_status": "PASS",
                    "forecast_blocking": False,
                    "provenance_complete": True,
                    "coverage_complete": True,
                    "recent_clean_coverage_complete": True,
                },
                "market_state": {"direction": m5_direction, "evidence_complete": True},
                "recent_candles": m5,
            },
        ],
    }
    packet = {"generated_at_utc": (now - timedelta(seconds=5)).isoformat(), "charts": [chart]}
    snapshot = {
        "fetched_at_utc": (now - timedelta(seconds=2)).isoformat(),
        "quotes": {
            "EUR_USD": {
                "bid": 1.10026,
                "ask": 1.10027,
                "timestamp_utc": (now - timedelta(seconds=2)).isoformat(),
            }
        },
    }
    return packet, snapshot


def _s5(at: datetime, bid_o: float, bid_h: float, bid_l: float, bid_c: float) -> S5BidAskCandle:
    spread = 0.00008
    return S5BidAskCandle(
        timestamp_utc=at,
        bid_o=bid_o,
        bid_h=bid_h,
        bid_l=bid_l,
        bid_c=bid_c,
        ask_o=bid_o + spread,
        ask_h=bid_h + spread,
        ask_l=bid_l + spread,
        ask_c=bid_c + spread,
    )


def _as_usdjpy(packet: dict, snapshot: dict) -> tuple[dict, dict]:
    def converted(value: float) -> float:
        return round(150.0 + (float(value) - 1.1) * 100.0, 3)

    chart = packet["charts"][0]
    chart["pair"] = "USD_JPY"
    for view in chart["views"]:
        view["candle_integrity"]["pair"] = "USD_JPY"
        for candle in view["recent_candles"]:
            for key in ("o", "h", "l", "c"):
                candle[key] = converted(candle[key])
    quote = snapshot["quotes"].pop("EUR_USD")
    quote["bid"] = converted(quote["bid"])
    quote["ask"] = converted(quote["ask"])
    snapshot["quotes"]["USD_JPY"] = quote
    return packet, snapshot


class FastBotShockFollowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config, self.config_sha = load_config(CONFIG)
        self.now = datetime(2026, 8, 31, 12, 0, 10, tzinfo=timezone.utc)

    def test_config_seals_two_stop_strategies_and_zero_authority(self) -> None:
        self.assertEqual(tuple(row["strategy_id"] for row in self.config["strategies"]), STRATEGIES)
        self.assertTrue(all(row["order_type"] == "STOP" for row in self.config["strategies"]))
        self.assertEqual(self.config["authority"]["execution_authority"], "NONE")
        self.assertFalse(self.config["authority"]["automatic_adoption_allowed"])
        self.assertFalse(self.config["evidence"]["retrospective_reinterpretation_allowed"])

    def test_breakout_uses_only_completed_m1_m5_and_freezes_stop_vehicle(self) -> None:
        packet, snapshot = _packet(self.now)
        shadow = build_shock_follow_shadow(
            pair_charts=packet,
            broker_snapshot=snapshot,
            config=self.config,
            config_sha256=self.config_sha,
            now_utc=self.now,
        )
        signals = [row for row in shadow["signals"] if row["strategy_id"] == "SHOCK_BREAKOUT_FOLLOW"]
        self.assertEqual(len(signals), 1)
        signal = signals[0]
        self.assertTrue(sealed_valid(signal, SIGNAL_CONTRACT))
        self.assertEqual(signal["order_type"], "STOP")
        self.assertEqual(signal["entry_ttl_seconds"], 90)
        self.assertEqual(signal["truth_chunk_candle_limit"], 4500)
        self.assertGreater(signal["entry"], signal["quote_ask"])
        self.assertEqual(signal["range_rotation_shock_policy"], "RETAIN_EXISTING_VOL_SHOCK_VETO")
        self.assertFalse(signal["normal_strategy_override"])
        self.assertFalse(signal["lookahead_used"])
        self.assertFalse(signal["live_permission"])

    def test_pullback_requires_bounded_retrace_and_reacceleration(self) -> None:
        packet, snapshot = _packet(self.now, pullback=True)
        shadow = build_shock_follow_shadow(
            pair_charts=packet,
            broker_snapshot=snapshot,
            config=self.config,
            config_sha256=self.config_sha,
            now_utc=self.now,
        )
        signals = [row for row in shadow["signals"] if row["strategy_id"] == "SHOCK_PULLBACK_CONTINUATION"]
        self.assertEqual(len(signals), 1)
        self.assertGreaterEqual(signals[0]["pullback_atr_ratio"], 0.15)
        self.assertLessEqual(signals[0]["pullback_atr_ratio"], 0.8)
        self.assertFalse(signals[0]["opposite_break_invalidated"])

    def test_usdjpy_has_an_independent_breakout_lane(self) -> None:
        packet, snapshot = _as_usdjpy(*_packet(self.now))
        shadow = build_shock_follow_shadow(
            pair_charts=packet,
            broker_snapshot=snapshot,
            config=self.config,
            config_sha256=self.config_sha,
            now_utc=self.now,
        )
        signals = [row for row in shadow["signals"] if row["pair"] == "USD_JPY"]
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0]["strategy_id"], "SHOCK_BREAKOUT_FOLLOW")
        self.assertEqual(signals[0]["side"], "LONG")
        self.assertGreater(signals[0]["entry"], signals[0]["quote_ask"])

    def test_direction_mismatch_spread_shock_future_candle_and_weekend_fail_closed(self) -> None:
        packet, snapshot = _packet(self.now, m5_direction="DOWN")
        shadow = build_shock_follow_shadow(
            pair_charts=packet,
            broker_snapshot=snapshot,
            config=self.config,
            config_sha256=self.config_sha,
            now_utc=self.now,
        )
        self.assertEqual(shadow["signals"], [])
        packet, snapshot = _packet(self.now)
        packet["charts"][0]["views"][0]["candle_integrity"]["forecast_blocking"] = True
        shadow = build_shock_follow_shadow(
            pair_charts=packet,
            broker_snapshot=snapshot,
            config=self.config,
            config_sha256=self.config_sha,
            now_utc=self.now,
        )
        self.assertEqual(shadow["signals"], [])
        self.assertIn("M1_M5_CANDLE_INTEGRITY_NOT_PROVEN", shadow["pair_rejections"]["EUR_USD"])
        packet, snapshot = _packet(self.now)
        snapshot["quotes"]["EUR_USD"]["ask"] = 1.10040
        shadow = build_shock_follow_shadow(
            pair_charts=packet,
            broker_snapshot=snapshot,
            config=self.config,
            config_sha256=self.config_sha,
            now_utc=self.now,
        )
        self.assertIn("SPREAD_SHOCK_OR_ATR_UNAVAILABLE", shadow["pair_rejections"]["EUR_USD"])
        packet, snapshot = _packet(self.now)
        packet["charts"][0]["views"][0]["recent_candles"][-1]["t"] = (self.now + timedelta(minutes=1)).isoformat()
        shadow = build_shock_follow_shadow(
            pair_charts=packet,
            broker_snapshot=snapshot,
            config=self.config,
            config_sha256=self.config_sha,
            now_utc=self.now,
        )
        self.assertEqual(shadow["signals"], [])
        friday = datetime(2026, 8, 28, 22, 0, tzinfo=timezone.utc)
        self.assertTrue(market_is_closed(friday))

    def test_exact_s5_stop_fill_records_slippage_mfe_mae_and_zero_orders(self) -> None:
        packet, snapshot = _packet(self.now)
        signal = next(
            row
            for row in build_shock_follow_shadow(
                pair_charts=packet,
                broker_snapshot=snapshot,
                config=self.config,
                config_sha256=self.config_sha,
                now_utc=self.now,
            )["signals"]
            if row["strategy_id"] == "SHOCK_BREAKOUT_FOLLOW"
        )
        entry = float(signal["entry"])
        at = self.now.replace(second=10)
        maturity = self.now + timedelta(seconds=signal["entry_ttl_seconds"] + signal["max_hold_seconds"])
        candles = []
        while at < maturity:
            candles.append(_s5(at, entry - 0.00012, entry - 0.00009, entry - 0.00013, entry - 0.00010))
            at += timedelta(seconds=5)
        candles[1] = _s5(candles[1].timestamp_utc, entry - 0.00007, entry + 0.00004, entry - 0.00007, entry)
        candles[2] = _s5(
            candles[2].timestamp_utc,
            entry + 0.00002,
            float(signal["take_profit"]) + 0.00001,
            entry - 0.00002,
            float(signal["take_profit"]),
        )
        outcome = resolve_signal(
            signal,
            candles,
            truth_chunk_sha256=["a" * 64],
            resolved_at_utc=self.now + timedelta(minutes=20),
        )
        self.assertTrue(sealed_valid(outcome, OUTCOME_CONTRACT))
        self.assertTrue(outcome["filled"])
        self.assertEqual(outcome["exit_reason"], "TAKE_PROFIT")
        self.assertGreaterEqual(outcome["entry_slippage_pips"], 0.0)
        self.assertGreater(outcome["mfe_pips"], 0.0)
        self.assertGreaterEqual(outcome["mae_pips"], 0.0)
        self.assertEqual(outcome["external_order_attempts"], 0)
        self.assertEqual(outcome["external_orders"], 0)
        with self.assertRaisesRegex(ValueError, "invalid S5 truth coverage"):
            resolve_signal(
                signal,
                candles,
                truth_chunk_sha256=["a" * 64, "b" * 64],
                resolved_at_utc=self.now + timedelta(minutes=20),
            )

    def test_scorecard_separates_pair_strategy_side_bucket_and_diagnostics(self) -> None:
        packet, snapshot = _packet(self.now)
        signal = next(
            row
            for row in build_shock_follow_shadow(
                pair_charts=packet,
                broker_snapshot=snapshot,
                config=self.config,
                config_sha256=self.config_sha,
                now_utc=self.now,
            )["signals"]
            if row["strategy_id"] == "SHOCK_BREAKOUT_FOLLOW"
        )
        outcome_body = {
            "contract": OUTCOME_CONTRACT,
            "signal_sha256": signal["contract_sha256"],
            "config_sha256": self.config_sha,
            "pair": "EUR_USD",
            "strategy": "SHOCK_BREAKOUT_FOLLOW",
            "side": "LONG",
            "shock_bucket": signal["shock_bucket"],
            "signal_generated_at_utc": signal["generated_at_utc"],
            "filled": True,
            "after_cost_net_pips": 1.2,
            "mfe_pips": 1.5,
            "mae_pips": 0.2,
            "entry_slippage_pips": 0.1,
            "evidence_mode": "PROSPECTIVE_FORWARD_ONLY",
        }
        outcome = seal(outcome_body)
        diagnostic = seal(
            {
                "contract": "QR_FAST_BOT_CORRECTIVE_CHALLENGER_ROW_V1",
                "arm_id": "BASELINE",
                "pair": "EUR_USD",
                "vol_shock": True,
                "filled": True,
                "after_cost_net_pips": -2.0,
                "generated_at_utc": "2026-08-28T00:00:00+00:00",
                "signal_id": "old",
            }
        )
        card = build_scorecard(
            signals=[signal],
            outcomes=[outcome],
            corrective_rows=[diagnostic],
            config_sha256=self.config_sha,
            generated_at_utc=self.now,
        )
        groups = card["prospective"]["pair_strategy_side_shock_bucket"]
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["pair"], "EUR_USD")
        self.assertEqual(groups[0]["strategy"], "SHOCK_BREAKOUT_FOLLOW")
        self.assertEqual(groups[0]["net_pips"], 1.2)
        self.assertEqual([row["pair"] for row in card["prospective"]["by_pair"]], ["EUR_USD", "USD_JPY"])
        self.assertEqual(
            [row["strategy"] for row in card["prospective"]["by_strategy"]],
            ["SHOCK_BREAKOUT_FOLLOW", "SHOCK_PULLBACK_CONTINUATION"],
        )
        diag = card["historical_diagnostic_reference"]
        self.assertFalse(diag["counts_as_forward_evidence"])
        self.assertEqual(diag["comparison"][0]["net_pips"], -2.0)
        self.assertEqual(diag["pair_arm"][0]["pair"], "EUR_USD")
        self.assertEqual(diag["pair_arm"][0]["arm_id"], "BASELINE")
        self.assertFalse(card["forward_evidence_passed"])
        self.assertFalse(card["automatic_adoption_allowed"])
        self.assertFalse(card["promotion_allowed"])
        self.assertFalse(card["live_permission"])

    def test_incremental_is_idempotent_and_does_not_create_client_without_due(self) -> None:
        packet, snapshot = _packet(self.now)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            charts = root / "charts.json"
            broker = root / "broker.json"
            signals = root / "signals.jsonl"
            outcomes = root / "outcomes.jsonl"
            corrective = root / "corrective.jsonl"
            scorecard = root / "scorecard.json"
            charts.write_text(json.dumps(packet))
            broker.write_text(json.dumps(snapshot))
            corrective.write_text("")
            first = run_incremental(
                pair_charts_path=charts,
                broker_snapshot_path=broker,
                signal_ledger_path=signals,
                outcome_ledger_path=outcomes,
                scorecard_path=scorecard,
                corrective_ledger_path=corrective,
                config_path=CONFIG,
                now=self.now,
            )
            second = run_incremental(
                pair_charts_path=charts,
                broker_snapshot_path=broker,
                signal_ledger_path=signals,
                outcome_ledger_path=outcomes,
                scorecard_path=scorecard,
                corrective_ledger_path=corrective,
                config_path=CONFIG,
                now=self.now,
            )
            self.assertEqual(first["signal_ledger_appended"], 1)
            self.assertEqual(second["signal_ledger_appended"], 0)
            self.assertFalse(first["broker_read"])
            self.assertEqual(first["external_order_attempts"], 0)
            self.assertEqual(first["external_orders"], 0)
            card = json.loads(scorecard.read_text())
            self.assertEqual(card["prospective"]["emitted_signal_count"], 1)


if __name__ == "__main__":
    unittest.main()
