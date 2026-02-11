from __future__ import annotations

from workers.micro_multistrat import worker


def _patch_trend_snapshot(monkeypatch, direction: str = "short", adx: float = 30.0) -> None:
    monkeypatch.setattr(
        worker,
        "_trend_snapshot",
        lambda *_args, **_kwargs: {
            "tf": "H4",
            "gap_pips": -20.0 if direction == "short" else 20.0,
            "direction": direction,
            "adx": adx,
        },
    )
    monkeypatch.setattr(worker.config, "TREND_FLIP_ENABLED", True)
    monkeypatch.setattr(worker.config, "TREND_FLIP_ADX_MIN", 20.0)
    monkeypatch.setattr(worker.config, "TREND_FLIP_TP_MULT", 1.12)
    monkeypatch.setattr(worker.config, "TREND_FLIP_SL_MULT", 0.95)


def test_trend_flip_blocklist_prevents_flip(monkeypatch):
    _patch_trend_snapshot(monkeypatch, direction="short", adx=35.0)
    monkeypatch.setattr(worker.config, "TREND_FLIP_STRATEGY_ALLOWLIST", frozenset())
    monkeypatch.setattr(
        worker.config,
        "TREND_FLIP_STRATEGY_BLOCKLIST",
        frozenset({"MicroLevelReactor"}),
    )

    side, tag, flip_meta, tp_mult, sl_mult, trend = worker._apply_trend_flip(
        side="long",
        signal_tag="MicroLevelReactor-bounce-lower",
        strategy_name="MicroLevelReactor",
        fac_m1={},
        fac_m5={},
        fac_h1={},
        fac_h4={},
    )

    assert trend is not None
    assert side == "long"
    assert tag == "MicroLevelReactor-bounce-lower"
    assert flip_meta is None
    assert tp_mult == 1.0
    assert sl_mult == 1.0


def test_trend_flip_allowlist_limits_targets(monkeypatch):
    _patch_trend_snapshot(monkeypatch, direction="short", adx=35.0)
    monkeypatch.setattr(
        worker.config,
        "TREND_FLIP_STRATEGY_ALLOWLIST",
        frozenset({"MicroVWAPBound"}),
    )
    monkeypatch.setattr(worker.config, "TREND_FLIP_STRATEGY_BLOCKLIST", frozenset())

    side, tag, flip_meta, tp_mult, sl_mult, _trend = worker._apply_trend_flip(
        side="long",
        signal_tag="MicroCompressionRevert-long",
        strategy_name="MicroCompressionRevert",
        fac_m1={},
        fac_m5={},
        fac_h1={},
        fac_h4={},
    )

    assert side == "long"
    assert tag == "MicroCompressionRevert-long"
    assert flip_meta is None
    assert tp_mult == 1.0
    assert sl_mult == 1.0


def test_trend_flip_still_applies_when_allowed(monkeypatch):
    _patch_trend_snapshot(monkeypatch, direction="short", adx=35.0)
    monkeypatch.setattr(worker.config, "TREND_FLIP_STRATEGY_ALLOWLIST", frozenset())
    monkeypatch.setattr(worker.config, "TREND_FLIP_STRATEGY_BLOCKLIST", frozenset())

    side, tag, flip_meta, tp_mult, sl_mult, trend = worker._apply_trend_flip(
        side="long",
        signal_tag="MicroVWAPBound-short",
        strategy_name="MicroVWAPBound",
        fac_m1={},
        fac_m5={},
        fac_h1={},
        fac_h4={},
    )

    assert trend is not None
    assert side == "short"
    assert tag.endswith("-trendflip")
    assert flip_meta == {
        "from": "long",
        "to": "short",
        "tf": "H4",
        "gap_pips": -20.0,
        "adx": 35.0,
    }
    assert tp_mult == 1.12
    assert sl_mult == 0.95

def test_strategy_cooldown_active_when_recent(monkeypatch):
    monkeypatch.setattr(worker.config, "STRATEGY_COOLDOWN_SEC", 45.0)
    worker._STRATEGY_LAST_TS.clear()
    worker._STRATEGY_LAST_TS["MicroLevelReactor"] = 100.0

    assert worker._strategy_cooldown_active("MicroLevelReactor", 120.0) is True
    assert worker._strategy_cooldown_active("MicroLevelReactor", 146.0) is False


def test_strategy_cooldown_disabled_by_default(monkeypatch):
    monkeypatch.setattr(worker.config, "STRATEGY_COOLDOWN_SEC", 0.0)
    worker._STRATEGY_LAST_TS.clear()
    worker._STRATEGY_LAST_TS["MicroLevelReactor"] = 100.0

    assert worker._strategy_cooldown_active("MicroLevelReactor", 110.0) is False


def test_mlr_strict_range_gate_blocks_without_range_context(monkeypatch):
    monkeypatch.setattr(worker.config, "MLR_STRICT_RANGE_GATE", True)
    monkeypatch.setattr(worker.config, "MLR_MIN_RANGE_SCORE", 0.62)
    monkeypatch.setattr(worker.config, "MLR_MAX_ADX", 20.0)
    monkeypatch.setattr(worker.config, "MLR_MAX_MA_GAP_PIPS", 2.2)

    ok, diag = worker._mlr_strict_range_ok(
        {"adx": 24.0, "ma10": 150.050, "ma20": 150.020},
        range_active=False,
        range_score=0.48,
    )

    assert ok is False
    assert diag["adx"] == 24.0
    assert diag["range_score"] == 0.48
    assert diag["ma_gap_pips"] > 2.2


def test_mlr_strict_range_gate_allows_strong_range_context(monkeypatch):
    monkeypatch.setattr(worker.config, "MLR_STRICT_RANGE_GATE", True)
    monkeypatch.setattr(worker.config, "MLR_MIN_RANGE_SCORE", 0.62)
    monkeypatch.setattr(worker.config, "MLR_MAX_ADX", 20.0)
    monkeypatch.setattr(worker.config, "MLR_MAX_MA_GAP_PIPS", 2.2)

    ok, diag = worker._mlr_strict_range_ok(
        {"adx": 16.0, "ma10": 150.023, "ma20": 150.010},
        range_active=True,
        range_score=0.70,
    )

    assert ok is True
    assert diag["adx"] == 16.0
    assert diag["range_active"] == 1.0
