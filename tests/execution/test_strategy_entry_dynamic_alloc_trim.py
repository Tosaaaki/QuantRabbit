from __future__ import annotations

from types import SimpleNamespace

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


def test_apply_dynamic_alloc_trim_allows_headroom_boost_for_fresh_winner_lane(
    monkeypatch,
) -> None:
    monkeypatch.setattr(strategy_entry, "_STRATEGY_DYNAMIC_ALLOC_ENABLED", True, raising=False)
    monkeypatch.setattr(strategy_entry, "_STRATEGY_DYNAMIC_ALLOC_TRIM_ONLY", True, raising=False)
    monkeypatch.setattr(strategy_entry, "_STRATEGY_DYNAMIC_ALLOC_POCKETS", {"scalp"}, raising=False)
    monkeypatch.setattr(strategy_entry, "_STRATEGY_DYNAMIC_ALLOC_MULT_MAX", 1.35, raising=False)
    monkeypatch.setattr(
        strategy_entry,
        "load_strategy_profile",
        lambda *_args, **_kwargs: {
            "found": True,
            "strategy_key": "PrecisionLowVol",
            "score": 0.78,
            "lot_multiplier": 1.12,
            "trades": 18,
            "payload_stale": False,
            "target_use": 0.88,
            "pocket_cap": 0.24,
            "pocket_target_use": 0.2112,
            "allocation_mode": "soft_participation",
        },
        raising=False,
    )
    monkeypatch.setattr(
        strategy_entry,
        "get_account_snapshot",
        lambda: SimpleNamespace(
            margin_used=10.0,
            margin_available=90.0,
            nav=100.0,
            free_margin_ratio=0.90,
        ),
        raising=False,
    )

    thesis: dict = {}
    units, reason = strategy_entry._apply_dynamic_alloc_trim(
        strategy_tag="PrecisionLowVol",
        pocket="scalp",
        units=100,
        min_units=20,
        entry_thesis=thesis,
    )

    assert reason is None
    assert units == 112
    assert thesis["dynamic_alloc"]["reason"] == "headroom_boost"
    assert thesis["dynamic_alloc"]["target_use"] == 0.88
    assert thesis["dynamic_alloc"]["pocket_target_use"] == 0.2112
    assert thesis["dynamic_alloc"]["current_margin_usage"] == 0.1


def test_apply_dynamic_alloc_trim_keeps_trim_only_behavior_near_target_use(monkeypatch) -> None:
    monkeypatch.setattr(strategy_entry, "_STRATEGY_DYNAMIC_ALLOC_ENABLED", True, raising=False)
    monkeypatch.setattr(strategy_entry, "_STRATEGY_DYNAMIC_ALLOC_TRIM_ONLY", True, raising=False)
    monkeypatch.setattr(strategy_entry, "_STRATEGY_DYNAMIC_ALLOC_POCKETS", {"scalp"}, raising=False)
    monkeypatch.setattr(strategy_entry, "_STRATEGY_DYNAMIC_ALLOC_MULT_MAX", 1.35, raising=False)
    monkeypatch.setattr(
        strategy_entry,
        "load_strategy_profile",
        lambda *_args, **_kwargs: {
            "found": True,
            "strategy_key": "PrecisionLowVol",
            "score": 0.78,
            "lot_multiplier": 1.12,
            "trades": 18,
            "payload_stale": False,
            "target_use": 0.88,
        },
        raising=False,
    )
    monkeypatch.setattr(
        strategy_entry,
        "get_account_snapshot",
        lambda: SimpleNamespace(
            margin_used=82.0,
            margin_available=18.0,
            nav=100.0,
            free_margin_ratio=0.18,
        ),
        raising=False,
    )

    thesis: dict = {}
    units, reason = strategy_entry._apply_dynamic_alloc_trim(
        strategy_tag="PrecisionLowVol",
        pocket="scalp",
        units=100,
        min_units=20,
        entry_thesis=thesis,
    )

    assert reason is None
    assert units == 100
    assert "dynamic_alloc" not in thesis


def test_apply_participation_alloc_trims_units_and_boosts_probability(monkeypatch) -> None:
    monkeypatch.setattr(strategy_entry, "_STRATEGY_PARTICIPATION_ALLOC_ENABLED", True, raising=False)
    monkeypatch.setattr(strategy_entry, "_STRATEGY_PARTICIPATION_ALLOC_POCKETS", {"micro"}, raising=False)
    monkeypatch.setattr(
        strategy_entry,
        "load_participation_profile",
        lambda *_args, **_kwargs: {
            "found": True,
            "strategy_key": "MicroRangeBreak",
            "lot_multiplier": 0.8,
            "probability_boost": 0.05,
            "preflights": 120,
            "filled": 18,
            "fill_rate": 0.15,
            "hard_block_rate": 0.55,
            "quality_score": 0.61,
            "current_share": 0.16,
            "target_share": 0.09,
        },
        raising=False,
    )

    thesis: dict = {}
    units, probability, applied = strategy_entry._apply_participation_alloc(
        strategy_tag="MicroRangeBreak",
        pocket="micro",
        units=100,
        min_units=20,
        entry_probability=0.80,
        entry_thesis=thesis,
    )

    assert units == 80
    assert round(float(probability or 0.0), 4) == 0.81
    assert isinstance(applied, dict)
    assert thesis["participation_alloc"]["source"] == "strategy_entry"


def test_inject_macro_news_context_skips_stale_payload(monkeypatch) -> None:
    monkeypatch.setattr(strategy_entry, "_STRATEGY_MACRO_NEWS_CONTEXT_ENABLED", True, raising=False)
    monkeypatch.setattr(
        strategy_entry,
        "load_macro_news_context",
        lambda **_kwargs: {
            "found": True,
            "stale": True,
            "generated_at": "2026-03-10T00:00:00Z",
        },
        raising=False,
    )

    thesis: dict = {}
    next_thesis, context = strategy_entry._inject_macro_news_context(thesis)

    assert next_thesis == thesis
    assert context is None
    assert "macro_news_context" not in thesis


def test_inject_macro_news_context_attaches_summary(monkeypatch) -> None:
    monkeypatch.setattr(strategy_entry, "_STRATEGY_MACRO_NEWS_CONTEXT_ENABLED", True, raising=False)
    monkeypatch.setattr(
        strategy_entry,
        "load_macro_news_context",
        lambda **_kwargs: {
            "found": True,
            "stale": False,
            "generated_at": "2026-03-10T00:00:00Z",
            "age_sec": 120.0,
            "event_severity": "high",
            "caution_window_active": True,
            "usd_jpy_bias": "neutral",
            "sources": [{"name": "fed"}],
            "headlines": [
                {
                    "source": "fed",
                    "published_at": "2026-03-10T00:00:00Z",
                    "title": "FOMC statement",
                    "severity": "high",
                    "link": "https://example.test/fomc",
                }
            ],
        },
        raising=False,
    )

    thesis: dict = {}
    next_thesis, context = strategy_entry._inject_macro_news_context(thesis)

    assert next_thesis is thesis
    assert isinstance(context, dict)
    assert thesis["macro_caution_window_active"] is True
    assert thesis["macro_news_context"]["event_severity"] == "high"
    assert thesis["macro_news_context"]["headlines"][0]["title"] == "FOMC statement"
