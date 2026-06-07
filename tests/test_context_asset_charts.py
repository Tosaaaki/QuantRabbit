from __future__ import annotations

import unittest

from quant_rabbit.analysis.context_assets import build_context_asset_charts


class _CandleClient:
    def get_json(self, path: str, params: dict | None = None) -> dict:
        if not path.endswith("/candles"):
            raise AssertionError(path)
        granularity = (params or {}).get("granularity", "M5")
        base = 4300.0 if "XAU_USD" in path else 100.0
        step = 0.1 if granularity in {"M1", "M5", "M15"} else 0.5
        return {
            "candles": [
                {
                    "time": f"2026-06-{4 + idx // 24:02d}T{idx % 24:02d}:00:00Z",
                    "mid": {
                        "o": str(base + (idx * step)),
                        "h": str(base + (idx * step) + 0.2),
                        "l": str(base + (idx * step) - 0.2),
                        "c": str(base + (idx * step) + 0.05),
                    },
                    "volume": 100 + idx,
                    "complete": True,
                }
                for idx in range(80)
            ]
        }


class ContextAssetChartsTest(unittest.TestCase):
    def test_builds_non_fx_full_chart_packet_without_trade_permission(self) -> None:
        payload = build_context_asset_charts(
            client=_CandleClient(),
            instruments=("XAU_USD",),
            timeframes=("M5", "H1"),
            count=80,
        )

        self.assertEqual(payload["role"], "NON_FX_CONTEXT_TECHNICALS_NOT_TRADE_PERMISSION")
        self.assertEqual(len(payload["charts"]), 1)
        chart = payload["charts"][0]
        self.assertEqual(chart["pair"], "XAU_USD")
        self.assertEqual(len(chart["views"]), 2)
        self.assertIn("chart_story", chart)


if __name__ == "__main__":
    unittest.main()
