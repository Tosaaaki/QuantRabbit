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
    monkeypatch.setattr(auto_canary, "_MAX_AGE_SEC", 1800.0, raising=False)

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
