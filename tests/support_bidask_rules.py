from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from quant_rabbit.forecast_precision import BIDASK_REPLAY_RULES_ENV


def write_nonmatching_bidask_rules(root: Path) -> Path:
    """Write a valid rule file that cannot match normal FX test fixtures."""

    path = root / "bidask_replay_nonmatching_rules.json"
    payload = {
        "schema_version": 1,
        "generated_at_utc": "2026-06-22T00:00:00Z",
        "generated_from": "unit-test-nonmatching",
        "edge_rules": [],
        "contrarian_edge_rules": [],
        "negative_rules": [
            {
                "name": "TEST_PAIR_UP_S5_BIDASK_NEGATIVE_EXPECTANCY",
                "pair": "TEST_PAIR",
                "side": "LONG",
                "direction": "UP",
                "granularity": "S5",
                "samples": 1,
                "directional_hit_rate": 0.0,
                "avg_final_pips": -1.0,
                "avg_mfe_pips": 0.0,
                "avg_mae_pips": 1.0,
                "optimized_take_profit_pips": 1.0,
                "optimized_stop_loss_pips": 1.0,
                "optimized_avg_realized_pips": -1.0,
                "optimized_win_rate": 0.0,
                "optimized_profit_factor": 0.0,
                "blocks_live_support": True,
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def write_bidask_replay_fixture_rules(root: Path) -> Path:
    """Write stable bid/ask replay fixtures for tests that exercise that layer."""

    path = root / "bidask_replay_fixture_rules.json"
    payload = {
        "schema_version": 1,
        "generated_at_utc": "2026-06-22T00:00:00Z",
        "generated_from": "unit-test-fixture",
        "edge_rules": [
            {
                "name": "EUR_USD_DOWN_S5_BIDASK_HARVEST_TP5_SL7",
                "pair": "EUR_USD",
                "side": "SHORT",
                "direction": "DOWN",
                "granularity": "S5",
                "samples": 48,
                "directional_hit_rate": 0.72,
                "avg_final_pips": 1.8,
                "avg_mfe_pips": 5.4,
                "avg_mae_pips": 3.7,
                "optimized_take_profit_pips": 5.0,
                "optimized_stop_loss_pips": 7.0,
                "optimized_avg_realized_pips": 2.1,
                "optimized_win_rate": 0.69,
                "optimized_profit_factor": 2.4,
                "daily_stability_status": "RANK_ONLY",
                "active_days": 1,
                "max_daily_sample_share": 1.0,
                "positive_day_rate": 0.50,
                "min_target_pips": 4.8,
                "max_target_pips": 5.5,
                "max_stop_pips": 7.2,
            }
        ],
        "contrarian_edge_rules": [
            {
                "name": "AUD_JPY_UP_FADE_TO_DOWN_S5_BIDASK_CONTRARIAN_HARVEST_TP10_SL7",
                "pair": "AUD_JPY",
                "side": "SHORT",
                "direction": "DOWN",
                "forecast_direction": "UP",
                "faded_direction": "UP",
                "contrarian_edge": True,
                "confidence_bucket": "0.75-0.90",
                "granularity": "S5",
                "samples": 124,
                "source_directional_hit_rate": 0.20,
                "source_avg_final_pips": -6.76,
                "directional_hit_rate": 0.76,
                "avg_final_pips": 5.8,
                "avg_mfe_pips": 12.0,
                "avg_mae_pips": 4.5,
                "optimized_take_profit_pips": 10.0,
                "optimized_stop_loss_pips": 7.0,
                "optimized_avg_realized_pips": 2.4,
                "optimized_win_rate": 0.70,
                "optimized_profit_factor": 2.5,
                "daily_stability_status": "RANK_ONLY",
                "active_days": 1,
                "max_daily_sample_share": 1.0,
                "positive_day_rate": 0.50,
                "min_target_pips": 9.8,
                "max_target_pips": 10.5,
                "max_stop_pips": 7.2,
            }
        ],
        "negative_rules": [
            {
                "name": "AUD_JPY_UP_S5_BIDASK_NEGATIVE_EXPECTANCY",
                "pair": "AUD_JPY",
                "side": "LONG",
                "direction": "UP",
                "granularity": "S5",
                "samples": 124,
                "directional_hit_rate": 0.20,
                "avg_final_pips": -6.76,
                "avg_mfe_pips": 3.8,
                "avg_mae_pips": 14.4,
                "optimized_take_profit_pips": 2.0,
                "optimized_stop_loss_pips": 2.0,
                "optimized_avg_realized_pips": -2.0,
                "optimized_win_rate": 0.0,
                "optimized_profit_factor": 0.0,
                "blocks_live_support": True,
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@contextmanager
def bidask_rules_env(path: Path) -> Iterator[None]:
    prior = os.environ.get(BIDASK_REPLAY_RULES_ENV)
    os.environ[BIDASK_REPLAY_RULES_ENV] = str(path)
    try:
        yield
    finally:
        if prior is None:
            os.environ.pop(BIDASK_REPLAY_RULES_ENV, None)
        else:
            os.environ[BIDASK_REPLAY_RULES_ENV] = prior
