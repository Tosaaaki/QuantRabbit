from __future__ import annotations

from workers.common.setup_context import derive_live_setup_context, extract_setup_identity


def test_derive_live_setup_context_prefers_common_fingerprint_over_conflicting_flow_label() -> None:
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


def test_extract_setup_identity_repairs_common_fingerprint_context_from_live_setup() -> None:
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


def test_extract_setup_identity_preserves_custom_strategy_context_when_fingerprint_is_non_common() -> None:
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
