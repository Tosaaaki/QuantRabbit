"""Unit tests for strategy/regime_classifier.py."""

from __future__ import annotations

import unittest

from quant_rabbit.strategy.regime_classifier import (
    REGIME_REVERSAL_RISK_PENALTY,
    RegimeSnapshot,
    classify_all,
    classify_pair,
    regime_score_modifier,
)


def _pair_chart(
    pair: str,
    *,
    sigma_24h: float | None = None,
    atr_pct_24h: float | None = None,
    tf_agreement: float | None = None,
    d_bias: str = "",
    h4_bias: str = "",
    h1_bias: str = "",
) -> dict:
    confluence: dict = {}
    if sigma_24h is not None:
        confluence["range_24h_sigma_multiple"] = sigma_24h
        confluence["range_24h_expansion_ratio"] = sigma_24h
        confluence["range_24h_expansion_upper_fence"] = 2.5
        confluence["range_24h_expansion_outlier"] = sigma_24h >= 2.5
    if atr_pct_24h is not None:
        confluence["atr_percentile_24h"] = atr_pct_24h
    if tf_agreement is not None:
        confluence["tf_agreement_score"] = tf_agreement
    chart = {"pair": pair, "confluence": confluence}
    if d_bias:
        chart["D"] = {"bias": d_bias}
    if h4_bias:
        chart["H4"] = {"bias": h4_bias}
    if h1_bias:
        chart["H1"] = {"bias": h1_bias}
    return chart


class RegimeClassifierTest(unittest.TestCase):
    def test_missing_payload_returns_unknown(self) -> None:
        snap = classify_pair({})
        self.assertEqual(snap.label, "UNKNOWN")
        self.assertEqual(snap.reversal_risk, 0.0)

    def test_stable_trend_when_all_tfs_align(self) -> None:
        chart = _pair_chart(
            "EUR_USD",
            sigma_24h=1.0, atr_pct_24h=0.5, tf_agreement=1.0,
            d_bias="UP", h4_bias="UP", h1_bias="UP",
        )
        snap = classify_pair(chart)
        self.assertEqual(snap.label, "STABLE_TREND")
        self.assertEqual(snap.direction_hint, "UP")
        self.assertEqual(snap.reversal_risk, 0.0)

    def test_reversal_risk_at_exhausted_range(self) -> None:
        chart = _pair_chart(
            "AUD_JPY",
            sigma_24h=2.8,  # above injected pair-relative upper fence
            atr_pct_24h=0.95,  # ≥ 0.90 vol expansion
            tf_agreement=0.6,
            d_bias="UP", h4_bias="UP", h1_bias="UP",
        )
        snap = classify_pair(chart)
        self.assertEqual(snap.label, "REVERSAL_RISK")
        self.assertGreater(snap.reversal_risk, 0.6)
        # Should include both exhaustion AND vol-expansion signals
        joined = "|".join(snap.signals)
        self.assertIn("range exhausted", joined)
        self.assertIn("volatility expansion", joined)

    def test_tf_disagreement_raises_reversal_risk(self) -> None:
        chart = _pair_chart(
            "GBP_USD",
            sigma_24h=1.0, atr_pct_24h=0.5,
            d_bias="DOWN", h4_bias="UP", h1_bias="UP",  # D vs H4/H1 disagree
        )
        snap = classify_pair(chart)
        # tf disagreement alone gives 0.30 → below 0.60 gate, so label NOT REVERSAL_RISK
        # but reversal_risk > 0
        self.assertGreater(snap.reversal_risk, 0.0)
        self.assertIn("TF disagreement", "|".join(snap.signals))

    def test_classify_all_keys_by_pair(self) -> None:
        payload = {
            "charts": [
                _pair_chart("EUR_USD", d_bias="UP", h4_bias="UP", h1_bias="UP"),
                _pair_chart("GBP_USD", d_bias="DOWN", h4_bias="DOWN", h1_bias="DOWN"),
            ]
        }
        result = classify_all(payload)
        self.assertEqual(set(result.keys()), {"EUR_USD", "GBP_USD"})
        self.assertEqual(result["EUR_USD"].direction_hint, "UP")
        self.assertEqual(result["GBP_USD"].direction_hint, "DOWN")


class RegimeScoreModifierTest(unittest.TestCase):
    def test_no_snapshot_returns_zero(self) -> None:
        delta, rationale = regime_score_modifier(None, "LONG")
        self.assertEqual(delta, 0.0)
        self.assertIsNone(rationale)

    def test_reversal_risk_same_side_entry_penalized(self) -> None:
        snap = RegimeSnapshot(
            pair="EUR_USD", label="REVERSAL_RISK",
            direction_hint="UP", reversal_risk=0.8, signals=("test",),
        )
        delta, rationale = regime_score_modifier(snap, "LONG")
        self.assertLess(delta, 0.0)
        # Magnitude proportional to reversal_risk × max penalty
        self.assertAlmostEqual(delta, -REGIME_REVERSAL_RISK_PENALTY * 0.8, places=1)
        self.assertIn("REVERSAL_RISK", rationale or "")

    def test_reversal_risk_counter_trend_entry_neutral(self) -> None:
        snap = RegimeSnapshot(
            pair="EUR_USD", label="REVERSAL_RISK",
            direction_hint="UP", reversal_risk=0.8, signals=(),
        )
        # SHORT into a REVERSAL_RISK UP trend → no penalty (fading reversal)
        delta, rationale = regime_score_modifier(snap, "SHORT")
        self.assertEqual(delta, 0.0)
        self.assertIn("counter-trend", rationale or "")

    def test_stable_trend_rewards_aligned_entry(self) -> None:
        snap = RegimeSnapshot(
            pair="AUD_JPY", label="STABLE_TREND",
            direction_hint="UP", reversal_risk=0.0, signals=(),
        )
        delta, _ = regime_score_modifier(snap, "LONG")
        self.assertGreater(delta, 0.0)
        delta_against, _ = regime_score_modifier(snap, "SHORT")
        self.assertLess(delta_against, 0.0)

    def test_ranging_pair_small_penalty(self) -> None:
        snap = RegimeSnapshot(
            pair="EUR_JPY", label="RANGING",
            direction_hint="NEUTRAL", reversal_risk=0.0, signals=(),
        )
        delta_long, _ = regime_score_modifier(snap, "LONG")
        delta_short, _ = regime_score_modifier(snap, "SHORT")
        self.assertLess(delta_long, 0.0)
        self.assertEqual(delta_long, delta_short)


if __name__ == "__main__":
    unittest.main()
