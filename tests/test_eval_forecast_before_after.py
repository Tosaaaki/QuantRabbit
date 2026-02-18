from __future__ import annotations

import pandas as pd

from scripts.eval_forecast_before_after import _to_datetime_utc


def test_to_datetime_utc_handles_mixed_precision_iso8601() -> None:
    raw = pd.Series(
        [
            "2026-01-23T18:19:59.526306+00:00",
            "2026-02-18T02:40:00+00:00",
        ]
    )
    parsed = _to_datetime_utc(raw)
    assert int(parsed.isna().sum()) == 0
    assert str(parsed.iloc[0].isoformat()) == "2026-01-23T18:19:59.526306+00:00"
    assert str(parsed.iloc[1].isoformat()) == "2026-02-18T02:40:00+00:00"
