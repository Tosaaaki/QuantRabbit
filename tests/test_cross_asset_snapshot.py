from __future__ import annotations

import unittest

from quant_rabbit.analysis.cross_asset import DEFAULT_CROSS_ASSET_INSTRUMENTS, build_cross_asset_snapshot


class _CandleClient:
    def get_json(self, path: str, params: dict | None = None) -> dict:
        if not path.endswith("/candles"):
            raise AssertionError(path)
        return {
            "candles": [
                {
                    "time": f"2026-06-{4 + hour // 24:02d}T{hour % 24:02d}:00:00Z",
                    "mid": {
                        "o": str(100.0 + hour),
                        "h": str(100.5 + hour),
                        "l": str(99.5 + hour),
                        "c": str(100.1 + hour),
                    },
                    "volume": 100,
                    "complete": True,
                }
                for hour in range(60)
            ]
        }


class CrossAssetSnapshotTest(unittest.TestCase):
    def test_default_instruments_do_not_include_persistent_missing_copper_feed(self) -> None:
        self.assertNotIn("COPPER_USD", DEFAULT_CROSS_ASSET_INSTRUMENTS)

    def test_jp10y_placeholder_stays_local_not_top_level_action_issue(self) -> None:
        snapshot = build_cross_asset_snapshot(
            client=_CandleClient(),
            instruments=("USB10Y_USD",),
            correlation_pairs=(),
            count=60,
        )

        self.assertNotIn("MISSING_JP10Y_FEED", "\n".join(snapshot.issues))
        self.assertEqual(snapshot.yield_spreads[0].issue.split(":", 1)[0], "MISSING_JP10Y_FEED")


if __name__ == "__main__":
    unittest.main()
