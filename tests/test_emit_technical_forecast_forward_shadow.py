from __future__ import annotations

import importlib.util
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from quant_rabbit.analysis.candles import Candle


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "emit_technical_forecast_forward_shadow.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("emit_technical_shadow", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EmitTechnicalForecastForwardShadowTest(unittest.TestCase):
    def test_fresh_m5_acquisition_publishes_clean_tail_only(self) -> None:
        module = _load_script()
        start = datetime(2026, 7, 15, 21, 30, tzinfo=timezone.utc)
        candles = tuple(
            Candle(
                timestamp_utc=start + timedelta(minutes=5 * index),
                open=1.0 + index * 0.0001,
                high=1.0001 + index * 0.0001,
                low=0.9999 + index * 0.0001,
                close=1.0 + index * 0.0001,
                volume=100,
                complete=True,
            )
            for index in range(30)
        )
        batch = SimpleNamespace(
            candles=candles,
            integrity={
                "recent_clean_tail_count": 18,
                "indicator_warmup_min_clean_count": 30,
            },
        )

        with patch.object(
            module,
            "fetch_technical_candles_via_client",
            return_value=batch,
        ):
            payload = module._fresh_m5_pair_charts(
                object(),
                ["EUR_USD"],
                observed_at_utc=datetime(2026, 7, 16, 0, 0, 5, tzinfo=timezone.utc),
                workers=1,
            )

        recent = payload["charts"][0]["views"][0]["recent_candles"]
        self.assertEqual(len(recent), 18)
        self.assertEqual(recent[0]["t"], candles[-18].timestamp_utc.isoformat())
        self.assertTrue(payload["fresh_m5_read_only"])

    def test_fresh_m5_acquisition_fails_closed_on_invalid_receipt_shape(self) -> None:
        module = _load_script()
        batch = SimpleNamespace(
            candles=(),
            integrity={
                "recent_clean_tail_count": True,
                "indicator_warmup_min_clean_count": 30,
            },
        )

        with patch.object(
            module,
            "fetch_technical_candles_via_client",
            return_value=batch,
        ):
            payload = module._fresh_m5_pair_charts(
                object(),
                ["EUR_USD"],
                observed_at_utc=datetime(2026, 7, 16, 0, 0, 5, tzinfo=timezone.utc),
                workers=1,
            )

        self.assertEqual(
            payload["charts"][0]["views"][0]["recent_candles"],
            [],
        )


if __name__ == "__main__":
    unittest.main()
