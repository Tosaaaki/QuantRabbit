from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from quant_rabbit.models import MarketContext, OrderIntent, OrderType, Owner, Side, TradeMethod
from quant_rabbit.strategy.profile import StrategyProfile


class StrategyProfileTest(unittest.TestCase):
    def test_blocks_history_rejected_profile_for_live_send(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profile.json"
            path.write_text(
                json.dumps(
                    {
                        "profiles": [
                            {
                                "pair": "USD_JPY",
                                "direction": "LONG",
                                "status": "BLOCK_UNTIL_NEW_EVIDENCE",
                                "required_fix": "bad history",
                                "live_net_jpy": -1200.5,
                                "live_n": 4,
                                "pretrade_net_jpy": -300.25,
                                "pretrade_n": 3,
                                "seat_net_jpy": -2200.0,
                                "seat_win_rate_pct": 25.0,
                                "seat_pl_n": 8,
                                "top_block_reasons": ["live tape negative", "pretrade negative"],
                            },
                            {
                                "pair": "EUR_USD",
                                "direction": "LONG",
                                "status": "RISK_REPAIR_CANDIDATE",
                                "required_fix": "cap risk",
                            },
                        ]
                    }
                )
            )
            profile = StrategyProfile.load(path)
            prior = _disable_sl_free_for_test()
            try:
                blocked = profile.validate(_intent("USD_JPY"), for_live_send=True)
                repair_dry = profile.validate(_intent("EUR_USD"), for_live_send=False)
                repair_live = profile.validate(_intent("EUR_USD"), for_live_send=True)
            finally:
                _restore_env("QR_TRADER_DISABLE_SL_REPAIR", prior)

            self.assertEqual(blocked[0].code, "STRATEGY_NOT_ELIGIBLE")
            self.assertEqual(blocked[0].severity, "BLOCK")
            evidence = profile.issue_evidence(_intent("USD_JPY"))
            self.assertEqual(evidence["profile_status"], "BLOCK_UNTIL_NEW_EVIDENCE")
            self.assertEqual(evidence["required_fix"], "bad history")
            self.assertEqual(evidence["live_net_jpy"], -1200.5)
            self.assertEqual(evidence["pretrade_net_jpy"], -300.25)
            self.assertEqual(evidence["seat_win_rate_pct"], 25.0)
            self.assertEqual(evidence["top_block_reasons"], ["live tape negative", "pretrade negative"])
            self.assertEqual(repair_dry[0].code, "STRATEGY_RISK_REPAIR_REQUIRED")
            self.assertEqual(repair_dry[0].severity, "WARN")
            self.assertEqual(repair_live[0].severity, "BLOCK")

    def test_method_specific_profile_cannot_authorize_another_strategy_method(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profile.json"
            path.write_text(
                json.dumps(
                    {
                        "profiles": [
                            {
                                "pair": "EUR_USD",
                                "direction": "LONG",
                                "method": "BREAKOUT_FAILURE",
                                "status": "CANDIDATE",
                                "required_fix": "failed-breakout edge only",
                            }
                        ]
                    }
                )
            )
            profile = StrategyProfile.load(path)

            prior = _disable_sl_free_for_test()
            try:
                breakout = profile.validate(
                    _intent("EUR_USD", method=TradeMethod.BREAKOUT_FAILURE),
                    for_live_send=True,
                )
                trend = profile.validate(
                    _intent("EUR_USD", method=TradeMethod.TREND_CONTINUATION),
                    for_live_send=True,
                )
            finally:
                _restore_env("QR_TRADER_DISABLE_SL_REPAIR", prior)

            self.assertEqual(breakout, ())
            self.assertEqual(trend[0].code, "STRATEGY_METHOD_PROFILE_MISSING")
            self.assertEqual(trend[0].severity, "BLOCK")

    def test_mine_missed_edge_blocks_market_live_send_under_sl_free(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profile = StrategyProfile.load(_profile(Path(tmp), status="MINE_MISSED_EDGE"))
            prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
            os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
            try:
                issues = profile.validate(
                    _intent(
                        "EUR_USD",
                        method=TradeMethod.BREAKOUT_FAILURE,
                        order_type=OrderType.MARKET,
                    ),
                    for_live_send=True,
                )
            finally:
                _restore_env("QR_TRADER_DISABLE_SL_REPAIR", prior)

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].code, "STRATEGY_TRIGGER_RECEIPT_REQUIRED")
        self.assertEqual(issues[0].severity, "BLOCK")

    def test_mine_missed_edge_allows_pending_trigger_receipt_under_sl_free(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profile = StrategyProfile.load(_profile(Path(tmp), status="MINE_MISSED_EDGE"))
            prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
            os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
            try:
                issues = profile.validate(
                    _intent(
                        "EUR_USD",
                        method=TradeMethod.BREAKOUT_FAILURE,
                        order_type=OrderType.STOP_ENTRY,
                    ),
                    for_live_send=True,
                )
            finally:
                _restore_env("QR_TRADER_DISABLE_SL_REPAIR", prior)

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].code, "STRATEGY_TRIGGER_RECEIPT_REQUIRED")
        self.assertEqual(issues[0].severity, "WARN")

    def test_block_until_new_evidence_blocks_live_send_under_sl_free(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profile = StrategyProfile.load(_profile(Path(tmp), status="BLOCK_UNTIL_NEW_EVIDENCE"))
            prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
            os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
            try:
                issues = profile.validate(
                    _intent(
                        "EUR_USD",
                        method=TradeMethod.BREAKOUT_FAILURE,
                        order_type=OrderType.STOP_ENTRY,
                    ),
                    for_live_send=True,
                )
            finally:
                _restore_env("QR_TRADER_DISABLE_SL_REPAIR", prior)

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].code, "STRATEGY_NOT_ELIGIBLE")
        self.assertEqual(issues[0].severity, "BLOCK")

    def test_watch_only_blocks_live_send_under_sl_free(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profile = StrategyProfile.load(_profile(Path(tmp), status="WATCH_ONLY"))
            prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
            os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
            try:
                issues = profile.validate(
                    _intent(
                        "EUR_USD",
                        method=TradeMethod.BREAKOUT_FAILURE,
                        order_type=OrderType.STOP_ENTRY,
                    ),
                    for_live_send=True,
                )
            finally:
                _restore_env("QR_TRADER_DISABLE_SL_REPAIR", prior)

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].code, "STRATEGY_NOT_ELIGIBLE")
        self.assertEqual(issues[0].severity, "BLOCK")

    def test_watch_only_forecast_seed_pending_trigger_is_advisory_under_sl_free(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profile = StrategyProfile.load(_profile(Path(tmp), status="WATCH_ONLY"))
            prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
            os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
            try:
                issues = profile.validate(
                    _intent(
                        "EUR_USD",
                        method=TradeMethod.BREAKOUT_FAILURE,
                        order_type=OrderType.STOP_ENTRY,
                        metadata={"forecast_seed": True},
                    ),
                    for_live_send=True,
                )
            finally:
                _restore_env("QR_TRADER_DISABLE_SL_REPAIR", prior)

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].code, "STRATEGY_NOT_ELIGIBLE")
        self.assertEqual(issues[0].severity, "WARN")

    def test_watch_only_forecast_seed_market_still_blocks_under_sl_free(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profile = StrategyProfile.load(_profile(Path(tmp), status="WATCH_ONLY"))
            prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
            os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
            try:
                issues = profile.validate(
                    _intent(
                        "EUR_USD",
                        method=TradeMethod.BREAKOUT_FAILURE,
                        order_type=OrderType.MARKET,
                        metadata={"forecast_seed": True},
                    ),
                    for_live_send=True,
                )
            finally:
                _restore_env("QR_TRADER_DISABLE_SL_REPAIR", prior)

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].code, "STRATEGY_NOT_ELIGIBLE")
        self.assertEqual(issues[0].severity, "BLOCK")

    def test_missing_profile_blocks_live_send_under_sl_free(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profile = StrategyProfile.load(_profile(Path(tmp), status="CANDIDATE"))
            prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
            os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
            try:
                issues = profile.validate(
                    _intent(
                        "GBP_USD",
                        method=TradeMethod.BREAKOUT_FAILURE,
                        order_type=OrderType.STOP_ENTRY,
                    ),
                    for_live_send=True,
                )
            finally:
                _restore_env("QR_TRADER_DISABLE_SL_REPAIR", prior)

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].code, "STRATEGY_PROFILE_MISSING")
        self.assertEqual(issues[0].severity, "BLOCK")

    def test_high_confidence_forecast_seed_missing_profile_is_advisory_under_sl_free(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profile = StrategyProfile.load(_profile(Path(tmp), status="CANDIDATE"))
            prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
            os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
            try:
                issues = profile.validate(
                    _intent(
                        "GBP_USD",
                        method=TradeMethod.TREND_CONTINUATION,
                        order_type=OrderType.STOP_ENTRY,
                        metadata={"forecast_seed": True, "forecast_confidence": 0.72},
                    ),
                    for_live_send=True,
                )
            finally:
                _restore_env("QR_TRADER_DISABLE_SL_REPAIR", prior)

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].code, "STRATEGY_PROFILE_MISSING")
        self.assertEqual(issues[0].severity, "WARN")

    def test_low_confidence_forecast_seed_missing_profile_blocks_under_sl_free(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profile = StrategyProfile.load(_profile(Path(tmp), status="CANDIDATE"))
            prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
            os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
            try:
                issues = profile.validate(
                    _intent(
                        "GBP_USD",
                        method=TradeMethod.TREND_CONTINUATION,
                        order_type=OrderType.STOP_ENTRY,
                        metadata={
                            "forecast_seed": True,
                            "forecast_confidence": 0.5349,
                            "forecast_market_support_ok": True,
                        },
                    ),
                    for_live_send=True,
                )
            finally:
                _restore_env("QR_TRADER_DISABLE_SL_REPAIR", prior)

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].code, "STRATEGY_PROFILE_MISSING")
        self.assertEqual(issues[0].severity, "BLOCK")

    def test_forecast_seed_missing_profile_market_blocks_under_sl_free(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profile = StrategyProfile.load(_profile(Path(tmp), status="CANDIDATE"))
            prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
            os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
            try:
                issues = profile.validate(
                    _intent(
                        "GBP_USD",
                        method=TradeMethod.TREND_CONTINUATION,
                        order_type=OrderType.MARKET,
                        metadata={"forecast_seed": True, "forecast_confidence": 0.91},
                    ),
                    for_live_send=True,
                )
            finally:
                _restore_env("QR_TRADER_DISABLE_SL_REPAIR", prior)

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].code, "STRATEGY_PROFILE_MISSING")
        self.assertEqual(issues[0].severity, "BLOCK")

    def test_empty_profile_blocks_forecast_seed_live_send_under_sl_free(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "strategy.json"
            path.write_text(json.dumps({"profiles": []}))
            profile = StrategyProfile.load(path)
            prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
            os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
            try:
                issues = profile.validate(
                    _intent(
                        "EUR_USD",
                        method=TradeMethod.BREAKOUT_FAILURE,
                        order_type=OrderType.STOP_ENTRY,
                        metadata={"forecast_seed": True},
                    ),
                    for_live_send=True,
                )
            finally:
                _restore_env("QR_TRADER_DISABLE_SL_REPAIR", prior)

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].code, "STRATEGY_PROFILE_EMPTY")
        self.assertEqual(issues[0].severity, "BLOCK")

    def test_reversal_recovery_hedge_missing_profile_is_advisory_under_sl_free(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profile = StrategyProfile.load(_profile(Path(tmp), status="CANDIDATE"))
            prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
            os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
            try:
                issues = profile.validate(
                    _intent(
                        "GBP_USD",
                        method=TradeMethod.TREND_CONTINUATION,
                        order_type=OrderType.MARKET,
                        metadata={
                            "position_intent": "HEDGE",
                            "hedge_recovery": True,
                            "hedge_timing_class": "REVERSAL",
                        },
                    ),
                    for_live_send=True,
                )
            finally:
                _restore_env("QR_TRADER_DISABLE_SL_REPAIR", prior)

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].code, "STRATEGY_PROFILE_MISSING")
        self.assertEqual(issues[0].severity, "WARN")


def _profile(root: Path, *, status: str) -> Path:
    path = root / "strategy.json"
    path.write_text(
        json.dumps(
            {
                "profiles": [
                    {
                        "pair": "EUR_USD",
                        "direction": "LONG",
                        "method": "BREAKOUT_FAILURE",
                        "status": status,
                        "required_fix": "build trigger/pending-entry receipts before live execution",
                    }
                ]
            }
        )
    )
    return path


def _intent(
    pair: str,
    *,
    method: TradeMethod = TradeMethod.TREND_CONTINUATION,
    order_type: OrderType = OrderType.MARKET,
    metadata: dict | None = None,
) -> OrderIntent:
    return OrderIntent(
        pair=pair,
        side=Side.LONG,
        order_type=order_type,
        units=1000,
        entry=1.0,
        tp=1.01,
        sl=0.99,
        thesis="test",
        owner=Owner.TRADER,
        market_context=MarketContext(
            regime=f"{method.value} test",
            narrative="test",
            chart_story="test",
            method=method,
            invalidation="test",
        ),
        metadata=metadata or {},
    )


def _restore_env(key: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value


def _disable_sl_free_for_test() -> str | None:
    prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
    os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
    return prior


if __name__ == "__main__":
    unittest.main()
