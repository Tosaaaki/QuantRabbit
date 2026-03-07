from __future__ import annotations

import json
from pathlib import Path

from sim.pseudo_cfg import SimCfg
from sim.pseudo_ticks import _parse_candle_time, synth_from_candles


def test_parse_candle_time_accepts_nanosecond_precision() -> None:
    dt = _parse_candle_time("2026-03-06T09:01:00.000000000Z")

    assert dt.isoformat() == "2026-03-06T09:01:00+00:00"


def test_synth_from_candles_accepts_nanosecond_precision(tmp_path: Path) -> None:
    candles_path = tmp_path / "candles.json"
    out_path = tmp_path / "ticks.jsonl"
    payload = {
        "instrument": "USD_JPY",
        "granularity": "S5",
        "candles": [
            {
                "time": "2026-03-06T09:01:00.000000000Z",
                "mid": {"o": "157.700", "h": "157.705", "l": "157.698", "c": "157.704"},
                "volume": 12,
            },
            {
                "time": "2026-03-06T09:01:05.000000000Z",
                "mid": {"o": "157.704", "h": "157.706", "l": "157.703", "c": "157.705"},
                "volume": 10,
            },
        ],
    }
    candles_path.write_text(json.dumps(payload), encoding="utf-8")

    sim_path, density_info = synth_from_candles(str(candles_path), str(out_path), SimCfg())

    assert sim_path == out_path
    assert out_path.exists()
    assert out_path.read_text(encoding="utf-8").strip()
    assert density_info
