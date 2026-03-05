from __future__ import annotations

import execution.strategy_entry as strategy_entry


def test_apply_dynamic_alloc_trim_scales_down_and_rescues_min_units(monkeypatch) -> None:
    monkeypatch.setattr(strategy_entry, "_STRATEGY_DYNAMIC_ALLOC_ENABLED", True, raising=False)
    monkeypatch.setattr(strategy_entry, "_STRATEGY_DYNAMIC_ALLOC_TRIM_ONLY", True, raising=False)
    monkeypatch.setattr(strategy_entry, "_STRATEGY_DYNAMIC_ALLOC_POCKETS", {"micro"}, raising=False)

    monkeypatch.setattr(
        strategy_entry,
        "load_strategy_profile",
        lambda *_args, **_kwargs: {
            "found": True,
            "strategy_key": "MicroRangeBreak",
            "score": 0.0,
            "lot_multiplier": 0.25,
            "trades": 200,
        },
        raising=False,
    )

    thesis: dict = {}
    units, reason = strategy_entry._apply_dynamic_alloc_trim(
        strategy_tag="MicroRangeBreak",
        pocket="micro",
        units=100,
        min_units=30,
        entry_thesis=thesis,
    )
    assert reason is None
    assert units == 30
    assert isinstance(thesis.get("dynamic_alloc"), dict)
    assert thesis["dynamic_alloc"]["source"] == "strategy_entry"


def test_apply_dynamic_alloc_trim_skips_when_already_applied(monkeypatch) -> None:
    monkeypatch.setattr(strategy_entry, "_STRATEGY_DYNAMIC_ALLOC_ENABLED", True, raising=False)
    monkeypatch.setattr(strategy_entry, "_STRATEGY_DYNAMIC_ALLOC_POCKETS", {"scalp_fast"}, raising=False)

    thesis: dict = {"dynamic_alloc": {"source": "worker", "lot_multiplier": 0.5}}
    units, reason = strategy_entry._apply_dynamic_alloc_trim(
        strategy_tag="scalp_ping_5s_b_live",
        pocket="scalp_fast",
        units=-120,
        min_units=10,
        entry_thesis=thesis,
    )
    assert reason is None
    assert units == -120
