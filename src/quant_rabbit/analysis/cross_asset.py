"""Cross-asset / inter-market snapshot.

The trader can't make a decent FX call without seeing what DXY, US Treasury
yields, equities, gold, oil, and BTC are doing — they are the macro drivers
that move every G10 pair. This module fetches OANDA CFD candles for each
proxy, computes regime stats, and emits per-pair correlations against the
core FX pairs the trader is watching.

Synthetic DXY: when OANDA does not list `DXY_USD` as an instrument, we
re-build the ICE DXY basket from the six component pairs:

    DXY ≈ 50.14348112 × EUR/USD^(-0.576) × USD/JPY^(+0.136)
                       × GBP/USD^(-0.119) × USD/CAD^(+0.091)
                       × USD/SEK^(+0.042) × USD/CHF^(+0.036)

US-JP yield spread: US10Y is fetched from OANDA's USB10Y_USD bond CFD price
(price ≈ 100 + yield component for futures-style CFDs; we report the raw
price and percent change, marking the absolute yield as MISSING because OANDA
exposes the future, not the cash yield). JP10Y has no OANDA CFD; the
US10Y-JP10Y spread row carries that limitation as local metadata so the
top-level issue list remains reserved for actionable fetch failures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import log
from statistics import mean, stdev
from typing import Iterable, Mapping, Sequence

from quant_rabbit.analysis.candles import Candle, fetch_candles_via_client
from quant_rabbit.broker.oanda import OandaReadOnlyClient
from quant_rabbit.instruments import DEFAULT_CONTEXT_ASSETS, DEFAULT_TRADER_PAIRS


# OANDA CFD instruments. Anything not tradable on the account simply errors
# during fetch; we capture that as a `MISSING_*` issue rather than crashing.
DEFAULT_CROSS_ASSET_INSTRUMENTS: tuple[str, ...] = DEFAULT_CONTEXT_ASSETS

DXY_BASKET: tuple[tuple[str, float], ...] = (
    ("EUR_USD", -0.576),
    ("USD_JPY", +0.136),
    ("GBP_USD", -0.119),
    ("USD_CAD", +0.091),
    ("USD_SEK", +0.042),
    ("USD_CHF", +0.036),
)
DXY_CONSTANT = 50.14348112

DEFAULT_CORRELATION_PAIRS = DEFAULT_TRADER_PAIRS


@dataclass(frozen=True)
class AssetReading:
    instrument: str
    last_price: float | None
    change_pct_24h: float | None  # vs ~24h ago using H1 candles count back
    change_pct_5d: float | None
    z_score_60: float | None
    realized_vol_60: float | None  # stdev of log returns × √(bars/year)
    trend_label: str | None  # "UP" / "DOWN" / "FLAT"
    fetched: bool
    issue: str | None = None  # MISSING_INSTRUMENT or fetch error reason

    def to_dict(self) -> dict[str, object]:
        return {
            "instrument": self.instrument,
            "last_price": self.last_price,
            "change_pct_24h": self.change_pct_24h,
            "change_pct_5d": self.change_pct_5d,
            "z_score_60": self.z_score_60,
            "realized_vol_60": self.realized_vol_60,
            "trend_label": self.trend_label,
            "fetched": self.fetched,
            "issue": self.issue,
        }


@dataclass(frozen=True)
class SyntheticDXY:
    last_value: float | None
    change_pct_24h: float | None
    change_pct_5d: float | None
    components_used: tuple[str, ...]
    components_missing: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "last_value": self.last_value,
            "change_pct_24h": self.change_pct_24h,
            "change_pct_5d": self.change_pct_5d,
            "components_used": list(self.components_used),
            "components_missing": list(self.components_missing),
        }


@dataclass(frozen=True)
class YieldSpread:
    name: str
    leg_a: str
    leg_b: str
    a_last: float | None
    b_last: float | None
    spread_last: float | None
    spread_change_24h: float | None
    issue: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "leg_a": self.leg_a,
            "leg_b": self.leg_b,
            "a_last": self.a_last,
            "b_last": self.b_last,
            "spread_last": self.spread_last,
            "spread_change_24h": self.spread_change_24h,
            "issue": self.issue,
        }


@dataclass(frozen=True)
class CrossAssetSnapshot:
    generated_at_utc: str
    granularity: str
    candle_count: int
    assets: tuple[AssetReading, ...]
    synthetic_dxy: SyntheticDXY | None
    yield_spreads: tuple[YieldSpread, ...]
    correlations: dict[str, dict[str, float | None]]
    issues: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        return {
            "generated_at_utc": self.generated_at_utc,
            "granularity": self.granularity,
            "candle_count": self.candle_count,
            "assets": [a.to_dict() for a in self.assets],
            "synthetic_dxy": self.synthetic_dxy.to_dict() if self.synthetic_dxy else None,
            "yield_spreads": [y.to_dict() for y in self.yield_spreads],
            "correlations": self.correlations,
            "issues": list(self.issues),
        }


def build_cross_asset_snapshot(
    *,
    client: OandaReadOnlyClient,
    instruments: Sequence[str] = DEFAULT_CROSS_ASSET_INSTRUMENTS,
    correlation_pairs: Sequence[str] = DEFAULT_CORRELATION_PAIRS,
    granularity: str = "H1",
    count: int = 200,
) -> CrossAssetSnapshot:
    """Fetch cross-asset candles, compute readings, and correlate vs FX pairs."""

    issues: list[str] = []

    # 1. Fetch each cross-asset instrument
    asset_candles: dict[str, tuple[Candle, ...]] = {}
    asset_readings: list[AssetReading] = []
    for instr in instruments:
        try:
            candles = fetch_candles_via_client(client, instr, granularity, count=count)
            asset_candles[instr] = candles
            asset_readings.append(_reading_from_candles(instr, candles))
        except Exception as exc:
            issues.append(f"MISSING_{instr}: {exc}")
            asset_readings.append(AssetReading(
                instrument=instr, last_price=None, change_pct_24h=None,
                change_pct_5d=None, z_score_60=None, realized_vol_60=None,
                trend_label=None, fetched=False, issue=f"{exc}",
            ))

    # 2. Synthetic DXY from FX components
    fx_candles: dict[str, tuple[Candle, ...]] = {}
    for pair, _ in DXY_BASKET:
        try:
            fx_candles[pair] = fetch_candles_via_client(client, pair, granularity, count=count)
        except Exception as exc:
            issues.append(f"MISSING_DXY_COMPONENT_{pair}: {exc}")
    synthetic_dxy = _build_synthetic_dxy(fx_candles)

    # 3. Yield-spread placeholders (US-JP requires JP10Y feed which OANDA lacks)
    us_jp_spread = YieldSpread(
        name="US10Y_minus_JP10Y",
        leg_a="USB10Y_USD",
        leg_b="JP10Y",
        a_last=_last_close(asset_candles.get("USB10Y_USD", ())),
        b_last=None,
        spread_last=None,
        spread_change_24h=None,
        issue="MISSING_JP10Y_FEED: OANDA exposes USB10Y_USD futures CFD but no JGB cash 10Y feed; plug in BoJ/FRED feed to compute spread",
    )
    us_us_2_10 = _build_simple_spread(
        "US10Y_minus_US2Y", "USB10Y_USD", "USB02Y_USD", asset_candles,
    )
    yield_spreads: list[YieldSpread] = [us_jp_spread, us_us_2_10]

    # 4. FX-pair correlations vs every fetched cross-asset
    fx_correlation_candles: dict[str, tuple[Candle, ...]] = dict(fx_candles)
    for pair in correlation_pairs:
        if pair in fx_correlation_candles:
            continue
        try:
            fx_correlation_candles[pair] = fetch_candles_via_client(client, pair, granularity, count=count)
        except Exception as exc:
            issues.append(f"MISSING_CORR_PAIR_{pair}: {exc}")

    correlations: dict[str, dict[str, float | None]] = {}
    for pair, p_candles in fx_correlation_candles.items():
        per_asset: dict[str, float | None] = {}
        for instr, a_candles in asset_candles.items():
            per_asset[instr] = _correlation(p_candles, a_candles, lookback=min(60, count))
        correlations[pair] = per_asset

    return CrossAssetSnapshot(
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        granularity=granularity,
        candle_count=count,
        assets=tuple(asset_readings),
        synthetic_dxy=synthetic_dxy,
        yield_spreads=tuple(yield_spreads),
        correlations=correlations,
        issues=tuple(issues),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reading_from_candles(instr: str, candles: Sequence[Candle]) -> AssetReading:
    if not candles:
        return AssetReading(
            instrument=instr, last_price=None, change_pct_24h=None,
            change_pct_5d=None, z_score_60=None, realized_vol_60=None,
            trend_label=None, fetched=False, issue="empty",
        )
    closes = [c.close for c in candles]
    last = closes[-1]
    # H1 → 24h ago = -24, 5d ago = -120
    change_24 = _pct(closes[-1], closes[-25]) if len(closes) >= 25 else None
    change_5d = _pct(closes[-1], closes[-121]) if len(closes) >= 121 else None
    z = _z_score(closes, 60)
    rv = _realized_vol(closes, 60)
    trend = _trend_label(closes)
    return AssetReading(
        instrument=instr, last_price=last, change_pct_24h=change_24,
        change_pct_5d=change_5d, z_score_60=z, realized_vol_60=rv,
        trend_label=trend, fetched=True,
    )


def _build_synthetic_dxy(fx_candles: Mapping[str, Sequence[Candle]]) -> SyntheticDXY | None:
    components_used: list[str] = []
    components_missing: list[str] = []
    series_now: list[float] = []
    series_24h: list[float] = []
    series_5d: list[float] = []
    weights_used: list[float] = []
    for pair, weight in DXY_BASKET:
        candles = fx_candles.get(pair)
        if not candles:
            components_missing.append(pair)
            continue
        components_used.append(pair)
        weights_used.append(weight)
        closes = [c.close for c in candles]
        series_now.append(closes[-1])
        if len(closes) >= 25:
            series_24h.append(closes[-25])
        if len(closes) >= 121:
            series_5d.append(closes[-121])

    if len(components_used) < 4:
        return None  # not enough basket coverage

    def _dxy(values: Sequence[float]) -> float | None:
        if len(values) != len(weights_used):
            return None
        prod = 1.0
        for v, w in zip(values, weights_used):
            if v <= 0:
                return None
            prod *= v ** w
        return DXY_CONSTANT * prod

    last = _dxy(series_now)
    prev24 = _dxy(series_24h)
    prev5d = _dxy(series_5d)
    return SyntheticDXY(
        last_value=last,
        change_pct_24h=_pct(last, prev24),
        change_pct_5d=_pct(last, prev5d),
        components_used=tuple(components_used),
        components_missing=tuple(components_missing),
    )


def _build_simple_spread(
    name: str, leg_a: str, leg_b: str, asset_candles: Mapping[str, Sequence[Candle]],
) -> YieldSpread:
    a = asset_candles.get(leg_a)
    b = asset_candles.get(leg_b)
    if not a or not b:
        return YieldSpread(name=name, leg_a=leg_a, leg_b=leg_b,
                           a_last=None, b_last=None, spread_last=None,
                           spread_change_24h=None,
                           issue=f"MISSING_LEG: a={'OK' if a else 'NA'}, b={'OK' if b else 'NA'}")
    a_last = a[-1].close
    b_last = b[-1].close
    spread_last = a_last - b_last
    a_24 = a[-25].close if len(a) >= 25 else None
    b_24 = b[-25].close if len(b) >= 25 else None
    spread_24 = (a_24 - b_24) if (a_24 is not None and b_24 is not None) else None
    spread_change = (spread_last - spread_24) if spread_24 is not None else None
    return YieldSpread(
        name=name, leg_a=leg_a, leg_b=leg_b,
        a_last=a_last, b_last=b_last, spread_last=spread_last,
        spread_change_24h=spread_change,
    )


def _correlation(a: Sequence[Candle], b: Sequence[Candle], *, lookback: int = 60) -> float | None:
    if not a or not b:
        return None
    closes_a = [c.close for c in a][-lookback:]
    closes_b = [c.close for c in b][-lookback:]
    n = min(len(closes_a), len(closes_b))
    if n < 10:
        return None
    closes_a = closes_a[-n:]
    closes_b = closes_b[-n:]
    rets_a = [log(closes_a[i] / closes_a[i - 1]) for i in range(1, n) if closes_a[i - 1] > 0 and closes_a[i] > 0]
    rets_b = [log(closes_b[i] / closes_b[i - 1]) for i in range(1, n) if closes_b[i - 1] > 0 and closes_b[i] > 0]
    m = min(len(rets_a), len(rets_b))
    if m < 5:
        return None
    rets_a = rets_a[-m:]
    rets_b = rets_b[-m:]
    mean_a = mean(rets_a)
    mean_b = mean(rets_b)
    num = sum((rets_a[i] - mean_a) * (rets_b[i] - mean_b) for i in range(m))
    den_a = sum((r - mean_a) ** 2 for r in rets_a) ** 0.5
    den_b = sum((r - mean_b) ** 2 for r in rets_b) ** 0.5
    if den_a == 0 or den_b == 0:
        return None
    return num / (den_a * den_b)


def _pct(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or b == 0:
        return None
    return (a - b) / b * 100.0


def _z_score(values: Sequence[float], period: int) -> float | None:
    if len(values) < period:
        return None
    window = values[-period:]
    m = sum(window) / period
    var = sum((v - m) ** 2 for v in window) / period
    sd = var ** 0.5
    if sd == 0:
        return 0.0
    return (window[-1] - m) / sd


def _realized_vol(values: Sequence[float], period: int) -> float | None:
    if len(values) <= period:
        return None
    rets = []
    for i in range(len(values) - period, len(values)):
        if i == 0 or values[i - 1] <= 0 or values[i] <= 0:
            continue
        rets.append(log(values[i] / values[i - 1]))
    if len(rets) < 5:
        return None
    try:
        sd = stdev(rets)
    except Exception:
        return None
    return sd * (252.0 ** 0.5)


def _trend_label(values: Sequence[float], short: int = 20, long: int = 60) -> str | None:
    if len(values) < long:
        return None
    short_ma = sum(values[-short:]) / short
    long_ma = sum(values[-long:]) / long
    if abs(short_ma - long_ma) / long_ma < 0.001:
        return "FLAT"
    return "UP" if short_ma > long_ma else "DOWN"


def _last_close(candles: Sequence[Candle]) -> float | None:
    return candles[-1].close if candles else None
