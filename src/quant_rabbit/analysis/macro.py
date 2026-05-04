"""Macro / FRED / risk-score composite layer.

Builds the macro context the trader reads each cycle: short-rate level,
yield-curve shape, credit spread, equity vol, gold, plus a single composite
``risk_score`` blending VIX, HY OAS, SPX 60d return, AUDJPY 60d return,
Copper/Gold ratio, and US2Y 60d change. Also computes a USD-credibility
regime flag (default ON / OFF / UNKNOWN) by checking the sign of the rolling
5-day beta of DXY change vs US10Y change.

Source of formulas:
  ``docs/research/04-intermarket-macro.md`` §3 (Risk-on/off composite),
  §9 (free data sources), §11 (operational recommendations).

Contract:
  ``docs/AGENT_CONTRACT.md`` §3.5 — every numeric constant in this module
  must carry an (a) what market reality / (b) why constant / (c) what would
  replace it docstring or inline comment.

No third-party deps; uses ``urllib.request`` for FRED HTTP. CI must pass an
``offline_payload`` so tests never hit the live network.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Iterable, Mapping, Sequence

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# §3.5 documented constants
# ---------------------------------------------------------------------------

# (a) The "60-day" window represents roughly one quarter of trading sessions —
#     long enough to capture a macro regime shift but short enough that the
#     trader's intraday actions remain inside the window's tail.
# (b) Constant rather than market-derived because the research §3 composite
#     formula explicitly anchors all sub-scores to a 60-day z-window so the
#     six components share a comparable horizon.
# (c) If a future research note retunes the composite (e.g. 90d for credit,
#     30d for vol), replace this constant with per-series windows surfaced
#     through ``MacroReading.risk_score_components`` rather than a single
#     scalar override.
_RISK_SCORE_WINDOW_DAYS = 60

# (a) The ±3 z-score clip caps any single component at three standard
#     deviations so a stale FRED tick or a flash-crash print cannot pin the
#     composite at an arbitrary extreme.
# (b) Constant because §3 of the research note treats the composite as the
#     equal-weighted sum of clipped sub-scores; the symmetric ±3 bound is
#     part of the formula, not a tunable risk parameter.
# (c) If the composite later moves to weighted (not equal) sub-scores, the
#     clip can move to per-component caps; until then, do not change.
_RISK_SCORE_CLIP = 3.0

# (a) The 5-day beta window matches the "post-April-2-2025 tariff regime"
#     diagnostic in research §11: a one-week rolling lens is short enough to
#     flip when USD credibility breaks but long enough to avoid noise.
# (b) Constant because the regime check is a *diagnostic* — operators want a
#     stable "ON / OFF" signal, not one that retunes per pair.
# (c) If credibility-break diagnostics get richer (e.g. 5d + 20d agreement),
#     extend the dataclass with a second field rather than mutating this.
_USD_CREDIBILITY_BETA_WINDOW = 5

# (a) FRED API base URL — operational endpoint published by St. Louis Fed.
# (b) Constant because the URL is part of FRED's public contract; if FRED
#     ever migrates the endpoint, this is the only place to update.
# (c) Replace if FRED changes its hostname or path; do *not* swap to a
#     paid mirror without updating §9 of the research note first.
_FRED_API_BASE = "https://api.stlouisfed.org/fred/series/observations"

# (a) Default request timeout for FRED HTTP — research §9 recommends free
#     daily data only, so the call is not on the hot path.
# (b) Constant because no live trading decision blocks on FRED; a long
#     timeout is acceptable on the daily refresh cycle.
# (c) If FRED is moved into the per-cycle hot path, lower this and add a
#     circuit-breaker; for now it is a daily background refresh.
_FRED_HTTP_TIMEOUT_SECONDS = 10.0

# (a) Number of years of recent observations to request from FRED. The
#     composite needs ~60 trading days of history; one calendar year is
#     comfortably more than that and stays inside FRED's free-tier limits.
# (b) Constant because a fixed observation window keeps the FRED call
#     deterministic and cheap; the composite never needs more than ~252 obs.
# (c) If a multi-year backtest is added, expose ``observation_start`` to the
#     caller (already available on ``FredClient.fetch_series``) instead of
#     widening this default.
_FRED_DEFAULT_LOOKBACK_DAYS = 365

# Environment variable holding the operator's FRED API key.
# (a) FRED requires a free API key per §9; we look it up in the environment
#     so it is never checked into the repo.
# (b) Constant because the env var name is part of the operator's setup
#     contract — changing it silently would break every running scheduled
#     task.
# (c) If FRED auth ever moves to OAuth or a different scheme, deprecate
#     this constant rather than reusing the name.
_FRED_API_KEY_ENV = "QR_FRED_API_KEY"


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MacroReading:
    """Snapshot of the macro / risk-on-off composite at a single moment."""

    fetched_at_utc: datetime
    us2y: float | None
    us10y: float | None
    t10y2y: float | None
    t10y_inflation_be: float | None  # T10YIE
    hy_oas: float | None  # BAMLH0A0HYM2
    vix: float | None  # VIXCLS
    gold: float | None  # GOLDPMGBD228NLBM
    risk_score: float | None
    risk_score_components: dict[str, float] = field(default_factory=dict)
    usd_credibility_regime: str = "UNKNOWN"
    dxy_us10y_5d_beta_sign: int | None = None
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "fetched_at_utc": self.fetched_at_utc.isoformat(),
            "us2y": self.us2y,
            "us10y": self.us10y,
            "t10y2y": self.t10y2y,
            "t10y_inflation_be": self.t10y_inflation_be,
            "hy_oas": self.hy_oas,
            "vix": self.vix,
            "gold": self.gold,
            "risk_score": self.risk_score,
            "risk_score_components": dict(self.risk_score_components),
            "usd_credibility_regime": self.usd_credibility_regime,
            "dxy_us10y_5d_beta_sign": self.dxy_us10y_5d_beta_sign,
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# FRED client
# ---------------------------------------------------------------------------


class FredClient:
    """Minimal FRED observations client (stdlib only).

    Use ``offline_payload`` injection at the ``build_macro_reading`` layer for
    tests; the live HTTP path here is exercised only in production.
    """

    def __init__(self, api_key: str, *, base_url: str = _FRED_API_BASE) -> None:
        if not api_key:
            raise ValueError("FRED API key is required")
        self._api_key = api_key
        self._base_url = base_url

    def fetch_series(
        self,
        series_id: str,
        *,
        observation_start: str | None = None,
    ) -> list[tuple[date, float]]:
        """Fetch a FRED series and return ``[(date, value), ...]`` ascending.

        Missing values (FRED encodes as ``"."``) are skipped.
        """

        params = {
            "series_id": series_id,
            "api_key": self._api_key,
            "file_type": "json",
        }
        if observation_start:
            params["observation_start"] = observation_start
        url = f"{self._base_url}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={"User-Agent": "QuantRabbit/1.0"})
        with urllib.request.urlopen(req, timeout=_FRED_HTTP_TIMEOUT_SECONDS) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        return _parse_fred_observations(payload)


def _parse_fred_observations(payload: Mapping[str, object]) -> list[tuple[date, float]]:
    obs = payload.get("observations") or []
    out: list[tuple[date, float]] = []
    for row in obs:
        if not isinstance(row, Mapping):
            continue
        raw_date = row.get("date")
        raw_value = row.get("value")
        if not isinstance(raw_date, str) or not isinstance(raw_value, str):
            continue
        if raw_value == "." or raw_value == "":
            continue
        try:
            d = date.fromisoformat(raw_date)
            v = float(raw_value)
        except (TypeError, ValueError):
            continue
        out.append((d, v))
    out.sort(key=lambda pair: pair[0])
    return out


# ---------------------------------------------------------------------------
# Statistics helpers (stdlib)
# ---------------------------------------------------------------------------


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _stdev(values: Sequence[float]) -> float | None:
    n = len(values)
    if n < 2:
        return None
    m = sum(values) / n
    var = sum((v - m) ** 2 for v in values) / (n - 1)
    return var ** 0.5 if var > 0 else 0.0


def _z_score(values: Sequence[float], window: int = _RISK_SCORE_WINDOW_DAYS) -> float | None:
    """Return the z-score of the most recent observation vs the trailing window."""

    if not values:
        return None
    sample = list(values[-window:])
    if len(sample) < 2:
        return None
    last = sample[-1]
    base = sample[:-1] if len(sample) > 1 else sample
    m = _mean(base)
    sd = _stdev(base)
    if m is None or sd is None or sd == 0:
        return None
    return (last - m) / sd


def _clip(value: float, *, low: float = -_RISK_SCORE_CLIP, high: float = _RISK_SCORE_CLIP) -> float:
    return max(low, min(high, value))


def _series_changes(values: Sequence[float]) -> list[float]:
    return [values[i] - values[i - 1] for i in range(1, len(values))]


def _beta_sign(xs: Sequence[float], ys: Sequence[float]) -> int | None:
    """Sign of the simple linear-regression slope of y on x.

    Returns +1 (positive), -1 (negative), 0 (flat / undefined within tolerance),
    or None when the inputs are too short to fit.
    """

    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx = _mean(list(xs))
    my = _mean(list(ys))
    if mx is None or my is None:
        return None
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return None
    slope = num / den
    # Tolerance is intentionally tiny — operators want "essentially zero" to
    # surface as flat, not as a forced ±1 from numerical noise.
    if abs(slope) < 1e-9:
        return 0
    return 1 if slope > 0 else -1


# ---------------------------------------------------------------------------
# Composite builders
# ---------------------------------------------------------------------------


def _last_value(series: Sequence[tuple[date, float]] | None) -> float | None:
    if not series:
        return None
    return series[-1][1]


def _values(series: Sequence[tuple[date, float]] | None) -> list[float]:
    if not series:
        return []
    return [v for _, v in series]


def _build_risk_score(
    *,
    vix_series: Sequence[float],
    hy_oas_series: Sequence[float],
    spx_60d_return: float | None,
    audjpy_60d_return: float | None,
    copper_gold_ratio_series: Sequence[float],
    us2y_series: Sequence[float],
) -> tuple[float | None, dict[str, float]]:
    """Return ``(risk_score, components)`` per research §3.

    Formula:
      risk_score = z(VIX, 60d)·-1
                 + z(HY OAS, 60d)·-1
                 + clip(spx_60d_return / spx_sd_proxy)
                 + clip(audjpy_60d_return / audjpy_sd_proxy)
                 + z(Copper/Gold, 60d)
                 + z(US2Y_change, 60d)·-1
    Each sub-score is clipped to ``±_RISK_SCORE_CLIP``.

    For the SPX / AUDJPY 60d-return inputs we accept a scalar already
    representing the trailing 60d return (the trader feeds it from chart
    data, since FRED does not provide AUDJPY). To make those scalars
    comparable to the FRED z-scores we pass them through the clip directly.
    """

    components: dict[str, float] = {}

    # Inverse signs: VIX / HY OAS / US2Y up = risk-OFF.
    z_vix = _z_score(vix_series)
    if z_vix is not None:
        components["vix_z_inv"] = _clip(-z_vix)

    z_hy = _z_score(hy_oas_series)
    if z_hy is not None:
        components["hy_oas_z_inv"] = _clip(-z_hy)

    if spx_60d_return is not None:
        # 60d returns above zero = risk-ON; clipped to the same ±3 bound.
        components["spx_60d_return"] = _clip(spx_60d_return)

    if audjpy_60d_return is not None:
        components["audjpy_60d_return"] = _clip(audjpy_60d_return)

    z_cg = _z_score(copper_gold_ratio_series)
    if z_cg is not None:
        components["copper_gold_z"] = _clip(z_cg)

    if us2y_series:
        # Composite needs the *change* in 2Y yields, not the level.
        us2y_changes = _series_changes(list(us2y_series))
        z_us2y = _z_score(us2y_changes)
        if z_us2y is not None:
            components["us2y_change_z_inv"] = _clip(-z_us2y)

    if not components:
        return None, components
    raw = sum(components.values())
    return _clip(raw), components


def _build_usd_credibility(
    dxy_recent: Sequence[float] | None,
    us10y_recent: Sequence[float] | None,
) -> tuple[str, int | None]:
    """USD-credibility regime per research §11.

    Returns one of ``"ON"`` (yield-spread default model holds), ``"OFF"``
    (post-April-2-2025 tariff regime: USD trades against rate differentials),
    or ``"UNKNOWN"``.
    """

    if not dxy_recent or not us10y_recent:
        return "UNKNOWN", None
    dxy_changes = _series_changes(list(dxy_recent[-(_USD_CREDIBILITY_BETA_WINDOW + 1):]))
    us10y_changes = _series_changes(list(us10y_recent[-(_USD_CREDIBILITY_BETA_WINDOW + 1):]))
    n = min(len(dxy_changes), len(us10y_changes))
    if n < 2:
        return "UNKNOWN", None
    sign = _beta_sign(us10y_changes[-n:], dxy_changes[-n:])
    if sign is None:
        return "UNKNOWN", None
    if sign > 0:
        return "ON", sign
    if sign < 0:
        return "OFF", sign
    return "UNKNOWN", 0


# ---------------------------------------------------------------------------
# Top-level builder
# ---------------------------------------------------------------------------


# (a) FRED series IDs used by the composite — these are the canonical free
#     daily series listed in research §9.
# (b) Constant because changing the series ID changes the meaning of the
#     composite; any swap is a research-note revision, not a runtime knob.
# (c) Replace if a series is renamed/retired by FRED, after updating §9.
_FRED_SERIES: dict[str, str] = {
    "us2y": "DGS2",
    "us10y": "DGS10",
    "t10y2y": "T10Y2Y",
    "t10y_inflation_be": "T10YIE",
    "hy_oas": "BAMLH0A0HYM2",
    "vix": "VIXCLS",
    "gold": "GOLDPMGBD228NLBM",
}


def build_macro_reading(
    *,
    fred_key: str | None = None,
    dxy_recent: Sequence[float] | None = None,
    us10y_recent: Sequence[float] | None = None,
    spx_60d_return: float | None = None,
    audjpy_60d_return: float | None = None,
    copper_gold_ratio: float | None = None,
    copper_gold_ratio_series: Sequence[float] | None = None,
    us2y_60d_change: float | None = None,
    offline_payload: Mapping[str, Sequence[tuple[date, float]]] | None = None,
    now: datetime | None = None,
) -> MacroReading:
    """Build a ``MacroReading`` either from FRED or from an offline payload.

    Tests must always pass ``offline_payload`` (no live HTTP in CI). In
    production, the caller passes ``fred_key`` and we hit FRED with stdlib.

    Per §3.5, missing key → graceful all-``None`` reading (not an exception).
    """

    fetched_at = now or datetime.now(timezone.utc)
    notes: list[str] = []

    series_map: dict[str, list[tuple[date, float]]] = {}

    if offline_payload is not None:
        for name in _FRED_SERIES:
            payload_series = offline_payload.get(name)
            series_map[name] = list(payload_series) if payload_series else []
        notes.append("offline_payload")
    else:
        key = fred_key or os.environ.get(_FRED_API_KEY_ENV)
        if not key:
            LOGGER.warning(
                "FRED key missing (env %s unset); MacroReading falls back to all-None",
                _FRED_API_KEY_ENV,
            )
            notes.append(f"fred_key_missing:{_FRED_API_KEY_ENV}")
            return MacroReading(
                fetched_at_utc=fetched_at,
                us2y=None,
                us10y=None,
                t10y2y=None,
                t10y_inflation_be=None,
                hy_oas=None,
                vix=None,
                gold=None,
                risk_score=None,
                risk_score_components={},
                usd_credibility_regime="UNKNOWN",
                dxy_us10y_5d_beta_sign=None,
                notes=tuple(notes),
            )
        client = FredClient(key)
        for name, series_id in _FRED_SERIES.items():
            try:
                series_map[name] = client.fetch_series(series_id)
            except Exception as exc:  # noqa: BLE001 — degrade loudly but don't crash
                LOGGER.warning("FRED fetch failed for %s (%s): %s", name, series_id, exc)
                series_map[name] = []
                notes.append(f"fred_fetch_failed:{series_id}")

    last = {name: _last_value(series_map.get(name)) for name in _FRED_SERIES}

    # Risk-score components.
    if copper_gold_ratio_series is None and copper_gold_ratio is not None:
        # (a) When the operator only has a single scalar Copper/Gold ratio,
        #     there is no meaningful 60d z-score for it; we record it on the
        #     reading but leave the composite component out.
        # (b) Constant rather than synthesizing a fake series so the composite
        #     remains honest about which sub-scores were observable.
        # (c) Replace by passing ``copper_gold_ratio_series`` from the caller
        #     once a daily Copper/Gold series is wired in.
        copper_gold_ratio_series = []

    risk_score, components = _build_risk_score(
        vix_series=_values(series_map.get("vix")),
        hy_oas_series=_values(series_map.get("hy_oas")),
        spx_60d_return=spx_60d_return,
        audjpy_60d_return=audjpy_60d_return,
        copper_gold_ratio_series=list(copper_gold_ratio_series or []),
        us2y_series=_values(series_map.get("us2y")),
    )

    if copper_gold_ratio is not None and "copper_gold_z" not in components:
        # Surface the raw scalar in the components map so the trader can see
        # what the operator passed even when no series-derived z exists.
        components["copper_gold_ratio_raw"] = float(copper_gold_ratio)

    if us2y_60d_change is not None and "us2y_change_z_inv" not in components:
        components["us2y_60d_change_raw"] = float(us2y_60d_change)

    regime, beta_sign = _build_usd_credibility(dxy_recent, us10y_recent)

    return MacroReading(
        fetched_at_utc=fetched_at,
        us2y=last.get("us2y"),
        us10y=last.get("us10y"),
        t10y2y=last.get("t10y2y"),
        t10y_inflation_be=last.get("t10y_inflation_be"),
        hy_oas=last.get("hy_oas"),
        vix=last.get("vix"),
        gold=last.get("gold"),
        risk_score=risk_score,
        risk_score_components=components,
        usd_credibility_regime=regime,
        dxy_us10y_5d_beta_sign=beta_sign,
        notes=tuple(notes),
    )


__all__ = [
    "MacroReading",
    "FredClient",
    "build_macro_reading",
]
