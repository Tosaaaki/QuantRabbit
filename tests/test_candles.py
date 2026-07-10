from __future__ import annotations

import unittest

from quant_rabbit.analysis.candles import _candles_from_payload


class CandleParsingTest(unittest.TestCase):
    def test_forming_oanda_tail_candle_is_excluded_from_strategy_series(self) -> None:
        payload = {
            "candles": [
                {
                    "time": "2026-07-10T10:00:00Z",
                    "complete": True,
                    "mid": {"o": "1.1400", "h": "1.1410", "l": "1.1390", "c": "1.1405"},
                    "volume": 100,
                },
                {
                    "time": "2026-07-10T10:05:00Z",
                    "complete": False,
                    "mid": {"o": "1.1405", "h": "1.1450", "l": "1.1400", "c": "1.1448"},
                    "volume": 2,
                },
            ]
        }

        candles = _candles_from_payload(payload)

        self.assertEqual(len(candles), 1)
        self.assertTrue(candles[0].complete)
        self.assertEqual(candles[0].close, 1.1405)

    def test_missing_complete_flag_remains_backward_compatible_as_complete(self) -> None:
        payload = {
            "candles": [
                {
                    "time": "2026-07-10T10:00:00Z",
                    "mid": {"o": "1.1400", "h": "1.1410", "l": "1.1390", "c": "1.1405"},
                }
            ]
        }

        candles = _candles_from_payload(payload)

        self.assertEqual(len(candles), 1)
        self.assertTrue(candles[0].complete)


if __name__ == "__main__":
    unittest.main()
