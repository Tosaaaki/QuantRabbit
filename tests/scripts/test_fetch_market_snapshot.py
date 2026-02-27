from __future__ import annotations

from datetime import datetime, timezone

from scripts import fetch_market_snapshot


def test_parse_stooq_row() -> None:
    csv_text = """Symbol,Date,Time,Open,High,Low,Close,Volume\nUSDJPY,2026-02-27,14:25:56,156.103,156.2285,155.5405,156.0215,\n"""
    row = fetch_market_snapshot._parse_stooq_row(csv_text)
    assert row["symbol"] == "USDJPY"
    assert row["open"] == 156.103
    assert row["close"] == 156.0215


def test_extract_te_bond_value_prefers_techartsmeta() -> None:
    html = '... TEChartsMeta = [{"value":3.984000000000,"symbol":"USGG10YR:IND"}]; ...'
    val = fetch_market_snapshot._extract_te_bond_value(html)
    assert val == 3.984


def test_extract_calendar_events_filters_window_and_country() -> None:
    html = """
    <table>
      <tr data-url="/united-states/ppi" data-country="united states" data-event="ppi mom">
        <td class="2026-02-27"><span class="event-1 calendar-date-3">1:30 PM</span></td>
        <td><a class='calendar-event' href='/united-states/ppi'>PPI MoM</a></td>
      </tr>
      <tr data-url="/japan/cpi" data-country="japan" data-event="cpi yoy">
        <td class="2026-02-28"><span class="event-2 calendar-date-2">3:00 AM</span></td>
        <td><a class='calendar-event' href='/japan/cpi'>CPI YoY</a></td>
      </tr>
      <tr data-url="/brazil/cpi" data-country="brazil" data-event="cpi yoy">
        <td class="2026-02-27"><span class="event-3 calendar-date-3">2:00 PM</span></td>
        <td><a class='calendar-event' href='/brazil/cpi'>CPI YoY</a></td>
      </tr>
    </table>
    """
    now_utc = datetime(2026, 2, 27, 13, 20, tzinfo=timezone.utc)
    events = fetch_market_snapshot._extract_calendar_events(
        html,
        countries={"united states", "japan"},
        now_utc=now_utc,
        before_min=60,
        after_min=900,
    )
    assert len(events) == 2
    assert events[0]["name"] == "PPI MoM"
    assert events[0]["impact"] == "high"
    assert events[1]["impact"] == "medium"
