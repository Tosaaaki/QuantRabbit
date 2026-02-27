from __future__ import annotations

from datetime import date

from scripts import import_market_brief


SAMPLE = """
以下、2026/2/27 時点の市況

* USD/JPY：156円近辺
* EUR/USD：1.180付近
* ドル指数 DXY：97.7付近
* AUD/JPY：111円付近
* EUR/JPY：184円付近
* 金利：米10年 4.00%前後、日10年 2.12%前後

| イベント                  | FF掲載時刻（Chicago） | 東京時間（JST） | ざっくり影響 |
| --------------------- | --------------: | --------------: | ---------------------------------- |
| 米 Core PPI / PPI  | 7:30am | **22:30** | USD全般 |
| Chicago PMI       | 8:45am | **23:45** | 追撃のボラ |
| Construction Spending | 9:00am | **翌0:00（2/28）** | 小さめ |
"""


def test_build_external_snapshot_from_brief() -> None:
    payload = import_market_brief._build_external_snapshot(SAMPLE)
    assert payload["pairs"]["USD_JPY"]["price"] == 156.0
    assert payload["pairs"]["EUR_USD"]["price"] == 1.18
    assert payload["pairs"]["AUD_JPY"]["price"] == 111.0
    assert payload["pairs"]["EUR_JPY"]["price"] == 184.0
    assert payload["dxy"] == 97.7
    assert payload["rates"]["US10Y"] == 4.0
    assert payload["rates"]["JP10Y"] == 2.12


def test_extract_events_from_table_with_next_day() -> None:
    events = import_market_brief._extract_events_from_table(SAMPLE, date(2026, 2, 27))
    assert len(events) == 3
    assert events[0].name.startswith("米 Core PPI")
    assert events[0].when_jst.strftime("%Y-%m-%d %H:%M") == "2026-02-27 22:30"
    assert events[2].when_jst.strftime("%Y-%m-%d %H:%M") == "2026-02-28 00:00"
