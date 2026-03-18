from __future__ import annotations

from workers.common.setup_context import (
    derive_live_setup_context,
    extract_setup_identity,
)


def test_derive_live_setup_context_prefers_common_fingerprint_over_conflicting_flow_label() -> (
    None
):
    fingerprint = (
        "PrecisionLowVol|short|range_compression|tight_fast|"
        "rsi:overbought|atr:low|gap:up_flat|volatility_compression"
    )
    summary = derive_live_setup_context(
        {
            "strategy_tag": "PrecisionLowVol",
            "flow_regime": "range_fade",
            "microstructure_bucket": "wide_normal",
            "setup_fingerprint": fingerprint,
        },
        units=-240,
    )

    assert isinstance(summary, dict)
    assert summary["flow_regime"] == "range_compression"
    assert summary["microstructure_bucket"] == "tight_fast"
    assert summary["setup_fingerprint"] == fingerprint


def test_derive_live_setup_context_appends_mtf_suffix_for_countertrend_macro() -> None:
    summary = derive_live_setup_context(
        {
            "strategy_tag": "PrecisionLowVol",
            "range_mode": "range",
            "range_score": 0.52,
            "range_reason": "volatility_compression",
            "technical_context": {
                "indicators": {
                    "M1": {
                        "atr_pips": 2.4,
                        "rsi": 61.0,
                        "adx": 18.0,
                        "plus_di": 22.0,
                        "minus_di": 20.0,
                        "ma10": 158.021,
                        "ma20": 158.020,
                    },
                    "H1": {
                        "atr_pips": 11.0,
                        "adx": 24.0,
                        "plus_di": 29.0,
                        "minus_di": 14.0,
                        "ma10": 158.280,
                        "ma20": 158.170,
                    },
                    "H4": {
                        "atr_pips": 24.0,
                        "adx": 28.0,
                        "plus_di": 31.0,
                        "minus_di": 12.0,
                        "ma10": 158.620,
                        "ma20": 158.360,
                    },
                },
                "ticks": {
                    "spread_pips": 0.8,
                    "tick_rate": 8.4,
                },
            },
        },
        units=-180,
    )

    assert isinstance(summary, dict)
    assert summary["flow_regime"] == "range_compression"
    assert summary["h1_flow_regime"] == "trend_long"
    assert summary["h4_flow_regime"] == "trend_long"
    assert summary["macro_flow_regime"] == "trend_long"
    assert summary["mtf_alignment"] == "countertrend"
    assert "macro:trend_long" in str(summary["setup_fingerprint"])
    assert "align:countertrend" in str(summary["setup_fingerprint"])


def test_extract_setup_identity_repairs_common_fingerprint_context_from_live_setup() -> (
    None
):
    fingerprint = (
        "DroughtRevert|long|range_compression|unknown|"
        "rsi:mid|atr:low|gap:up_flat|volatility_compression"
    )
    context = extract_setup_identity(
        {
            "flow_regime": "range_fade",
            "live_setup_context": {
                "flow_regime": "range_fade",
                "setup_fingerprint": fingerprint,
            },
        },
        units=120,
    )

    assert context == {
        "setup_fingerprint": fingerprint,
        "flow_regime": "range_compression",
        "microstructure_bucket": "unknown",
    }


def test_extract_setup_identity_preserves_custom_strategy_context_when_fingerprint_is_non_common() -> (
    None
):
    fingerprint = "RangeFader|short|sell-fade|trend_long|p0"
    context = extract_setup_identity(
        {
            "flow_regime": "trend_long",
            "microstructure_bucket": "tight_fast",
            "setup_fingerprint": fingerprint,
        },
        units=-180,
    )

    assert context == {
        "setup_fingerprint": fingerprint,
        "flow_regime": "trend_long",
        "microstructure_bucket": "tight_fast",
    }
