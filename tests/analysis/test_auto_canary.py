from __future__ import annotations

import json

from analysis import auto_canary


def _reset_auto_canary(monkeypatch, *, path) -> None:
    monkeypatch.setattr(auto_canary, "_PATH", path, raising=False)
    monkeypatch.setattr(auto_canary, "_CACHE", {"loaded": 0.0, "mtime": None, "payload": None}, raising=False)


def test_current_override_resolves_base_strategy_for_directional_tag(monkeypatch, tmp_path) -> None:
    path = tmp_path / "auto_canary.json"
    path.write_text(
        json.dumps(
            {
                "updated_at": "2026-03-11T00:00:00Z",
                "strategies": {
                    "MicroTrendRetest": {
                        "units_multiplier": 0.85,
                        "probability_offset": -0.03,
                    }
                },
            },
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    _reset_auto_canary(monkeypatch, path=path)
    monkeypatch.setattr(auto_canary, "_MAX_AGE_SEC", 86400.0, raising=False)

    payload = auto_canary.current_override("MicroTrendRetest-long")

    assert payload is not None
    assert payload["strategy_key"] == "MicroTrendRetest"
    assert payload["units_multiplier"] == 0.85


def test_current_override_skips_stale_payload(monkeypatch, tmp_path) -> None:
    path = tmp_path / "auto_canary.json"
    path.write_text(
        json.dumps(
            {
                "updated_at": "2026-03-01T00:00:00Z",
                "strategies": {
                    "MicroTrendRetest": {
                        "units_multiplier": 0.85,
                    }
                },
            },
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    _reset_auto_canary(monkeypatch, path=path)
    monkeypatch.setattr(auto_canary, "_MAX_AGE_SEC", 60.0, raising=False)

    assert auto_canary.current_override("MicroTrendRetest-long") is None


def test_current_override_prefers_setup_override(monkeypatch, tmp_path) -> None:
    path = tmp_path / "auto_canary.json"
    path.write_text(
        json.dumps(
            {
                "updated_at": "2026-03-11T00:00:00Z",
                "strategies": {
                    "RangeFader-sell-fade": {
                        "units_multiplier": 0.88,
                        "probability_offset": -0.02,
                        "setup_overrides": [
                            {
                                "match_dimension": "setup_fingerprint",
                                "setup_fingerprint": "RangeFader-sell-fade|short|trend_long|tight_fast|rsi:overbought|atr:mid|gap:up_extended|volatility_compression",
                                "flow_regime": "trend_long",
                                "microstructure_bucket": "tight_fast",
                                "units_multiplier": 0.79,
                                "probability_offset": -0.035,
                                "confidence": 0.82,
                                "samples": 19,
                            }
                        ],
                    }
                },
            },
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    _reset_auto_canary(monkeypatch, path=path)
    monkeypatch.setattr(auto_canary, "_MAX_AGE_SEC", 86400.0, raising=False)

    payload = auto_canary.current_override(
        "RangeFader-sell-fade",
        entry_thesis={
            "setup_fingerprint": "RangeFader-sell-fade|short|trend_long|tight_fast|rsi:overbought|atr:mid|gap:up_extended|volatility_compression",
            "live_setup_context": {
                "flow_regime": "trend_long",
                "microstructure_bucket": "tight_fast",
            },
        },
    )

    assert payload is not None
    assert payload["units_multiplier"] == 0.79
    assert payload["probability_offset"] == -0.035
    assert payload["setup_override"]["match_dimension"] == "setup_fingerprint"
