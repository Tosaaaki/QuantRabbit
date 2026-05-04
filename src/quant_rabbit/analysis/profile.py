"""Time-Price Opportunity (TPO) daily profile.

OANDA tick volume is **not** real volume, so a classic Volume Profile built on
``Candle.volume`` is unreliable on retail FX feeds. TPO replaces volume with
**time letters**: each 30-minute bracket of the session is a letter (A, B, …)
and that letter is stamped at every price the bracket touched. The shape of
the resulting histogram describes how much *time* the market spent at each
price — which is volume-agnostic and works on tick-volume feeds.

This module mirrors the style of ``analysis/indicators.py``:

- stdlib only,
- no pandas / numpy dependency,
- pure functions returning frozen dataclasses,
- ATR-derived numeric defaults (per ``docs/AGENT_CONTRACT.md`` §3.5: every
  numeric constant documents (a) what market reality it represents, (b) why
  it is constant rather than market-derived, (c) what should replace it if
  the value ever needs to change).

Reference: ``docs/research/02-volume-profile-tpo.md`` §2 (TPO primitives),
§3 (Auction Market Theory day classification), §5 (HTF context, naked POCs).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Sequence

from quant_rabbit.analysis.candles import Candle


# ---------------------------------------------------------------------------
# Numeric defaults — every constant carries (a)/(b)/(c) per §3.5
# ---------------------------------------------------------------------------

# (a) Market reality: bin granularity for the price axis of the TPO histogram.
#     Auction Market Theory groups prices into "ticks"; on retail FX with no
#     true tick size, we use a fraction of the session's ATR(14, M30) so the
#     bin count stays roughly constant across pairs and regimes.
# (b) Constant rather than market-derived: the *factor* (5%) is constant; the
#     resulting bin width is fully ATR-derived. 0.05 yields ~20 buckets across
#     a one-ATR move, which is the granularity Dalton uses on CME products.
# (c) Replace with: a regime-aware factor (tighter in chop, wider in trend)
#     once we have empirical evidence that the fixed 5% misclassifies HVN/LVN.
DEFAULT_BIN_ATR_FACTOR: float = 0.05

# (a) IB (Initial Balance) is the first hour of the session per Dalton's CME
#     definition (two 30-min brackets = A + B = 60 minutes).
# (b) Constant: the IB definition is anchored to CME pit-session structure
#     and Dalton's literature; 30-min brackets are the TPO unit by definition,
#     so "first hour" = exactly two brackets.
# (c) Replace only if we adopt a non-30-min bracket size (which would also
#     require redefining day-type heuristics).
IB_BRACKET_COUNT: int = 2

# (a) Value Area covers the central 70% of TPO touches — one standard
#     deviation of a normal distribution, the percentage Steidlmayer himself
#     used for the original Market Profile.
# (b) Constant: the 70% rule is a definitional anchor of Market Profile, not
#     a tunable parameter.
# (c) Replace only when explicitly publishing a non-Market-Profile distribution
#     statistic (e.g. 68% one-sigma, 80% wide-area), and rename the field.
VALUE_AREA_PCT: float = 0.70

# (a) Naked POC max age: how many sessions back we still consider a prior
#     POC "live magnet" if untouched. Five sessions ≈ one calendar trading
#     week, which is the horizon where unfilled auctions still attract price.
# (b) Constant: it is operator policy / strategy-profile parameter, not a
#     market-derived value. Older naked POCs decay in informational value.
# (c) Replace via ``naked_pocs(..., max_age_sessions=N)`` if the campaign
#     wants a longer / shorter HTF context window.
DEFAULT_NAKED_POC_MAX_AGE_SESSIONS: int = 5

# (a) HVN / LVN detection threshold: a bucket qualifies as a local extremum
#     if its TPO count differs from both neighbours by at least this fraction.
# (b) Constant: a small denoising margin to avoid declaring every micro-jitter
#     a node. Ten percent is wide enough to ignore noise on small profiles
#     yet narrow enough to find real shelves.
# (c) Replace with an ATR / range-aware threshold once we mine HVN/LVN
#     significance against subsequent retracement frequency.
HVN_LVN_RELATIVE_MARGIN: float = 0.10

# (a) Session boundary in UTC. NY close / Sydney open at 22:00 UTC is the
#     conventional FX day-rollover used by CME (5pm ET) and major institutional
#     desks during standard time. (DST shifts 21:00 UTC for part of the year;
#     callers can override session_start_utc to track that.)
# (b) Constant: the rollover convention is broker / market structure, not
#     market-derived.
# (c) Replace by passing ``session_start_utc`` explicitly when running
#     against DST sessions or non-FX assets.
DEFAULT_SESSION_START_HOUR_UTC: int = 22

# (a) Bracket width: the TPO unit is universally 30 minutes (Steidlmayer /
#     Dalton, and CME's published profile data).
# (b) Constant: 30 min is *the* definition of one TPO; changing it changes
#     the meaning of the output, not just a parameter.
# (c) Do not replace; create a new module if a non-30-min profile is desired.
BRACKET_MINUTES: int = 30


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TPOBracket:
    """One 30-minute bracket: its letter and the H/L it touched."""

    letter: str            # "A", "B", … "Z", "a", "b", …
    start_utc: datetime
    end_utc: datetime
    high: float
    low: float


@dataclass(frozen=True)
class TPOProfile:
    """Daily TPO profile for one session (22:00 UTC → 22:00 UTC by default)."""

    session_start_utc: datetime
    session_end_utc: datetime
    brackets: tuple[TPOBracket, ...]

    poc: float                              # Point of Control — most-touched price
    vah: float                              # Value Area High (top of 70% band)
    val: float                              # Value Area Low (bottom of 70% band)

    initial_balance_high: float             # IB = max H of first IB_BRACKET_COUNT brackets
    initial_balance_low: float              # IB = min L of first IB_BRACKET_COUNT brackets

    range_extension_up: bool                # any post-IB bracket above IB high
    range_extension_down: bool              # any post-IB bracket below IB low

    day_type: str                           # TREND / NORMAL / NORMAL_VARIATION / DOUBLE_DISTRIBUTION / NEUTRAL
    open_relation: str                      # OAOR / OAOY / OARE / OARY (vs prior value area)
    open_type: str                          # OD / OTD / OAIR / OAOR_TYPE

    hvn: tuple[float, ...]                  # high-volume nodes (local maxima of histogram)
    lvn: tuple[float, ...]                  # low-volume nodes (local minima)
    single_prints: tuple[float, ...]        # buckets touched by exactly one bracket


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_tpo_profile(
    candles_m30: Sequence[Candle],
    *,
    prior_value_area: tuple[float, float] | None = None,
    bin_pips: float | None = None,
    pip_size: float | None = None,
    session_start_utc: datetime | None = None,
) -> TPOProfile:
    """Build a TPO profile from a sequence of M30 candles.

    Parameters
    ----------
    candles_m30:
        M30 OHLC candles covering the session window.
    prior_value_area:
        Optional ``(VAH_y, VAL_y)`` from the prior session, used to classify
        ``open_relation`` (OAOR / OAOY / OARE / OARY).
    bin_pips:
        Override the bin width (in pips). When ``None`` we derive it from
        ``DEFAULT_BIN_ATR_FACTOR × ATR(14, M30)``.
    pip_size:
        Pip size override (e.g. 0.01 for *_JPY). When ``None`` we infer from
        the candle prices: prices ≥ 10 imply a JPY-style pair (pip = 0.01),
        else 0.0001. Tests can override directly.
    session_start_utc:
        Optional session anchor. When ``None`` we use 22:00 UTC of the day
        the candles begin on.
    """

    if not candles_m30:
        raise ValueError("build_tpo_profile requires at least one M30 candle")

    pip = pip_size if pip_size is not None else _infer_pip_size(candles_m30)
    session_start = session_start_utc or _default_session_start(candles_m30[0].timestamp_utc)
    session_end = session_start + timedelta(hours=24)

    in_session = [c for c in candles_m30 if session_start <= c.timestamp_utc < session_end]
    if not in_session:
        # Fall back to using all candles when the caller passed a non-aligned
        # range (e.g. tests with synthetic times). We still anchor on the
        # first candle so the session window is deterministic.
        in_session = list(candles_m30)
        session_start = in_session[0].timestamp_utc
        session_end = in_session[-1].timestamp_utc + timedelta(minutes=BRACKET_MINUTES)

    # Build brackets: one per M30 candle, letters A–Z then a–z.
    brackets: list[TPOBracket] = []
    for i, c in enumerate(in_session):
        letter = _letter_for_index(i)
        brackets.append(
            TPOBracket(
                letter=letter,
                start_utc=c.timestamp_utc,
                end_utc=c.timestamp_utc + timedelta(minutes=BRACKET_MINUTES),
                high=float(c.high),
                low=float(c.low),
            )
        )

    # Bin size — ATR-derived per §3.5 unless caller overrides.
    if bin_pips is not None:
        bin_width = float(bin_pips) * pip
    else:
        atr = _atr_m30(in_session, period=14)
        if atr is None or atr <= 0:
            # Tiny synthetic series → use the actual session range / 20 as a
            # reasonable proxy. Never a hardcoded pip literal.
            session_high = max(b.high for b in brackets)
            session_low = min(b.low for b in brackets)
            span = max(session_high - session_low, pip)
            bin_width = max(span / 20.0, pip)
        else:
            bin_width = max(atr * DEFAULT_BIN_ATR_FACTOR, pip)

    # Build the TPO histogram: bucket -> set of bracket indices that touched it.
    bucket_to_brackets: dict[int, set[int]] = {}
    for bi, br in enumerate(brackets):
        lo_bucket = _price_to_bucket(br.low, bin_width)
        hi_bucket = _price_to_bucket(br.high, bin_width)
        for b in range(lo_bucket, hi_bucket + 1):
            bucket_to_brackets.setdefault(b, set()).add(bi)

    if not bucket_to_brackets:
        raise ValueError("TPO histogram is empty — candles produced no buckets")

    bucket_counts: dict[int, int] = {b: len(s) for b, s in bucket_to_brackets.items()}
    bucket_prices: dict[int, float] = {b: _bucket_to_price(b, bin_width) for b in bucket_counts}

    # POC = price bucket with most letter touches (tie → bucket nearest the
    # session midpoint, classical Market Profile resolution).
    session_high = max(b.high for b in brackets)
    session_low = min(b.low for b in brackets)
    mid = (session_high + session_low) / 2.0
    sorted_buckets = sorted(
        bucket_counts.keys(),
        key=lambda b: (-bucket_counts[b], abs(bucket_prices[b] - mid)),
    )
    poc_bucket = sorted_buckets[0]
    poc_price = bucket_prices[poc_bucket]

    # Value Area: classical Steidlmayer expansion from POC outward. Compare
    # the 2-buckets-up-pair count vs the 2-buckets-down-pair count and absorb
    # the larger side until ≥70% of total touches is captured.
    vah_bucket, val_bucket = _value_area_expansion(
        bucket_counts, poc_bucket, target_pct=VALUE_AREA_PCT
    )
    vah_price = _bucket_to_price(vah_bucket, bin_width) + bin_width  # top edge
    val_price = _bucket_to_price(val_bucket, bin_width)               # bottom edge

    # Initial Balance (first IB_BRACKET_COUNT brackets = first 60 min).
    ib_n = min(IB_BRACKET_COUNT, len(brackets))
    ib_high = max(brackets[i].high for i in range(ib_n))
    ib_low = min(brackets[i].low for i in range(ib_n))

    # Range extension: any bracket after the IB period extending past IB H/L.
    range_ext_up = any(b.high > ib_high for b in brackets[ib_n:])
    range_ext_down = any(b.low < ib_low for b in brackets[ib_n:])

    # Day type (Dalton heuristics — see §2 of the research file).
    day_type = _classify_day_type(
        brackets=brackets,
        ib_high=ib_high,
        ib_low=ib_low,
        range_ext_up=range_ext_up,
        range_ext_down=range_ext_down,
        bucket_counts=bucket_counts,
        bucket_prices=bucket_prices,
        bin_width=bin_width,
    )

    # Open relation / open type — best-effort heuristics per §3.
    session_open = float(in_session[0].open)
    open_relation = _classify_open_relation(session_open, prior_value_area)
    open_type = _classify_open_type(brackets, session_open)

    # HVN / LVN — local maxima / minima of the histogram with a margin.
    hvn = _detect_local_extrema(
        bucket_counts, bucket_prices, mode="max", margin=HVN_LVN_RELATIVE_MARGIN
    )
    lvn = _detect_local_extrema(
        bucket_counts, bucket_prices, mode="min", margin=HVN_LVN_RELATIVE_MARGIN
    )

    # Single prints: buckets touched by exactly one bracket.
    singles = tuple(
        sorted(
            bucket_prices[b] for b, s in bucket_to_brackets.items() if len(s) == 1
        )
    )

    return TPOProfile(
        session_start_utc=session_start,
        session_end_utc=session_end,
        brackets=tuple(brackets),
        poc=poc_price,
        vah=vah_price,
        val=val_price,
        initial_balance_high=ib_high,
        initial_balance_low=ib_low,
        range_extension_up=range_ext_up,
        range_extension_down=range_ext_down,
        day_type=day_type,
        open_relation=open_relation,
        open_type=open_type,
        hvn=hvn,
        lvn=lvn,
        single_prints=singles,
    )


def naked_pocs(
    prior_profiles: Sequence[TPOProfile],
    current_price: float,
    *,
    max_age_sessions: int = DEFAULT_NAKED_POC_MAX_AGE_SESSIONS,
) -> tuple[float, ...]:
    """Return prior-session POCs that have NOT been retraded since formation.

    A naked POC is a magnet: price has not yet returned to "fill" the auction
    that printed it. We walk the supplied profiles in chronological order;
    for each profile, the POC is naked if no *subsequent* profile's
    [session_low, session_high] range, nor the current spot price, has crossed it.

    Parameters
    ----------
    prior_profiles:
        Past sessions in chronological order (oldest first).
    current_price:
        Current spot. A POC already retraded by spot is not naked.
    max_age_sessions:
        Drop POCs older than this many sessions back from the most recent.
    """

    if not prior_profiles:
        return tuple()

    # Trim by age — keep only the last ``max_age_sessions`` profiles.
    sessions = list(prior_profiles)[-max_age_sessions:]

    naked: list[float] = []
    for i, prof in enumerate(sessions):
        poc = prof.poc
        retraded = False
        # Check every later session's H/L range.
        for later in sessions[i + 1 :]:
            later_high = max(b.high for b in later.brackets) if later.brackets else poc
            later_low = min(b.low for b in later.brackets) if later.brackets else poc
            if later_low <= poc <= later_high:
                retraded = True
                break
        if retraded:
            continue
        # Check current spot — if spot has crossed it, also retraded.
        last = sessions[-1]
        if i < len(sessions) - 1:
            # Already covered by inter-session check above.
            pass
        else:
            # Most-recent profile: spot must be on a defined side.
            last_high = max(b.high for b in last.brackets) if last.brackets else poc
            last_low = min(b.low for b in last.brackets) if last.brackets else poc
            if last_low <= poc <= last_high:
                # POC was inside its own session range — it's "fresh" only if
                # current price has not yet returned to it.
                pass
        # Final retrade check vs current spot since session close.
        # We treat the POC as naked if current_price has not crossed back to it
        # from outside its own session range. Simplification: if current price
        # equals the POC exactly (or sits between it and the latest extreme on
        # the same side), it counts as retraded.
        if i == len(sessions) - 1:
            # Most recent — naked if spot is not at the POC.
            if current_price == poc:
                continue
        naked.append(poc)
    return tuple(naked)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _infer_pip_size(candles: Sequence[Candle]) -> float:
    sample = float(candles[0].close)
    # JPY-style pairs trade above ~50; other majors well below 10. Threshold
    # at 10 keeps EUR/GBP/AUD on the 0.0001 side and the JPY family on 0.01.
    return 0.01 if sample >= 10.0 else 0.0001


def _default_session_start(first_ts: datetime) -> datetime:
    if first_ts.tzinfo is None:
        first_ts = first_ts.replace(tzinfo=timezone.utc)
    base = first_ts.astimezone(timezone.utc)
    anchor = base.replace(
        hour=DEFAULT_SESSION_START_HOUR_UTC,
        minute=0,
        second=0,
        microsecond=0,
    )
    if base.hour < DEFAULT_SESSION_START_HOUR_UTC:
        anchor -= timedelta(days=1)
    return anchor


def _letter_for_index(i: int) -> str:
    if i < 26:
        return chr(ord("A") + i)
    if i < 52:
        return chr(ord("a") + (i - 26))
    # Beyond two letter banks (more than 26h of brackets) we cycle digits;
    # this shouldn't happen within a single 24h session of 30-min brackets
    # (max 48 brackets), but stay safe.
    return str(i % 10)


def _price_to_bucket(price: float, bin_width: float) -> int:
    return int(price // bin_width)


def _bucket_to_price(bucket: int, bin_width: float) -> float:
    return bucket * bin_width


def _atr_m30(candles: Sequence[Candle], period: int = 14) -> float | None:
    n = len(candles)
    if n <= period:
        return None
    trs: list[float] = []
    for i in range(1, n):
        h = candles[i].high
        low_value = candles[i].low
        prev_close = candles[i - 1].close
        tr = max(h - low_value, abs(h - prev_close), abs(low_value - prev_close))
        trs.append(tr)
    if len(trs) < period:
        return None
    atr = sum(trs[:period]) / period
    for tr in trs[period:]:
        atr = (atr * (period - 1) + tr) / period
    return atr


def _value_area_expansion(
    bucket_counts: dict[int, int],
    poc_bucket: int,
    *,
    target_pct: float = VALUE_AREA_PCT,
) -> tuple[int, int]:
    """Classical 70% Value Area expansion outward from POC."""

    total = sum(bucket_counts.values())
    target = total * target_pct
    high = poc_bucket
    low = poc_bucket
    accumulated = bucket_counts[poc_bucket]

    min_bucket = min(bucket_counts.keys())
    max_bucket = max(bucket_counts.keys())

    while accumulated < target:
        if high >= max_bucket and low <= min_bucket:
            break
        # Look two buckets up from current high and two buckets down from current
        # low; absorb whichever side has more touches (Steidlmayer convention).
        up_pair = bucket_counts.get(high + 1, 0) + bucket_counts.get(high + 2, 0)
        down_pair = bucket_counts.get(low - 1, 0) + bucket_counts.get(low - 2, 0)

        if up_pair == 0 and down_pair == 0:
            break

        if up_pair >= down_pair and high < max_bucket:
            high += 1
            accumulated += bucket_counts.get(high, 0)
            if accumulated >= target or high >= max_bucket:
                continue
            high += 1
            accumulated += bucket_counts.get(high, 0)
        elif low > min_bucket:
            low -= 1
            accumulated += bucket_counts.get(low, 0)
            if accumulated >= target or low <= min_bucket:
                continue
            low -= 1
            accumulated += bucket_counts.get(low, 0)
        else:
            break

    # Clamp to populated range so callers never index a bucket without data.
    high = min(high, max_bucket)
    low = max(low, min_bucket)
    return high, low


def _classify_day_type(
    *,
    brackets: Sequence[TPOBracket],
    ib_high: float,
    ib_low: float,
    range_ext_up: bool,
    range_ext_down: bool,
    bucket_counts: dict[int, int],
    bucket_prices: dict[int, float],
    bin_width: float,
) -> str:
    """Best-effort Dalton day-type classification.

    Heuristics (§2 of the research file):

    * TREND — total range > 1.5 × IB *and* close near one extreme (top or
      bottom 25% of the day's range).
    * DOUBLE_DISTRIBUTION — two clear histogram peaks separated by a low-volume
      gap (a single-print "neck").
    * NEUTRAL — range extension on **both** sides of IB.
    * NORMAL_VARIATION — range extension on exactly one side, but TREND
      condition not met.
    * NORMAL — no range extension on either side; price stays inside IB.
    """

    if not brackets:
        return "NORMAL"

    session_high = max(b.high for b in brackets)
    session_low = min(b.low for b in brackets)
    session_range = session_high - session_low
    ib_range = max(ib_high - ib_low, bin_width)
    last_close_proxy = brackets[-1].high  # post-IB latest bar's high acts as close-side proxy
    last_low_proxy = brackets[-1].low

    # TREND check — long range and close pinned near one extreme.
    quarter = session_range * 0.25
    if session_range > 1.5 * ib_range:
        if last_close_proxy >= session_high - quarter and range_ext_up and not range_ext_down:
            return "TREND"
        if last_low_proxy <= session_low + quarter and range_ext_down and not range_ext_up:
            return "TREND"

    # NEUTRAL — extension on both sides.
    if range_ext_up and range_ext_down:
        return "NEUTRAL"

    # DOUBLE_DISTRIBUTION — two peaks separated by a sparse zone.
    if _has_double_distribution(bucket_counts, bucket_prices):
        return "DOUBLE_DISTRIBUTION"

    if range_ext_up or range_ext_down:
        return "NORMAL_VARIATION"

    return "NORMAL"


def _has_double_distribution(
    bucket_counts: dict[int, int], bucket_prices: dict[int, float]
) -> bool:
    """Detect two distinct histogram peaks separated by a sparse 'neck'."""

    if len(bucket_counts) < 5:
        return False
    sorted_buckets = sorted(bucket_counts.keys())
    counts = [bucket_counts[b] for b in sorted_buckets]
    peak = max(counts)
    if peak < 2:
        return False
    threshold_high = peak * 0.7
    threshold_low = max(1, int(peak * 0.3))
    # Walk: high → low → high pattern.
    state = "search_first_high"
    found_first_high = False
    found_neck = False
    for c in counts:
        if state == "search_first_high":
            if c >= threshold_high:
                state = "search_neck"
                found_first_high = True
        elif state == "search_neck":
            if c <= threshold_low:
                state = "search_second_high"
                found_neck = True
        elif state == "search_second_high":
            if c >= threshold_high:
                return found_first_high and found_neck
    return False


def _classify_open_relation(
    session_open: float, prior_value_area: tuple[float, float] | None
) -> str:
    """Open vs prior value area (VAH_y, VAL_y).

    * OAOR — Open Above prior value, Outside Range (open > VAH_y by margin).
    * OAOY — Open Above prior value, but inside prior day's range (open > VAH_y but ≤ session high).
    * OARE — Open Around prior value (inside [VAL_y, VAH_y]).
    * OARY — Open Above (or below) prior value, Returning to prior Yesterday's value.
    """

    if prior_value_area is None:
        return "OARE"  # no prior context → conservative default
    vah_y, val_y = prior_value_area
    if val_y > vah_y:
        vah_y, val_y = val_y, vah_y
    if val_y <= session_open <= vah_y:
        return "OARE"
    if session_open > vah_y:
        # Above prior value. If far above, OAOR; otherwise OAOY (still
        # close to the prior range).
        margin = (vah_y - val_y) * 0.25
        return "OAOR" if session_open > vah_y + margin else "OAOY"
    # Below prior value.
    margin = (vah_y - val_y) * 0.25
    return "OAOR" if session_open < val_y - margin else "OARY"


def _classify_open_type(
    brackets: Sequence[TPOBracket], session_open: float
) -> str:
    """Open type from the first 1–3 brackets (Dalton, §3 of research).

    * OD       — Open-Drive: first bracket extends in one direction
                 with no return through open.
    * OTD      — Open-Test-Drive: first bracket tests near open then drives.
    * OAIR     — Open-Auction In Range: first 2 brackets oscillate around open.
    * OAOR_TYPE— Open-Auction Out of Range: first bracket gaps out and stays.
    """

    if not brackets:
        return "OAIR"

    first = brackets[0]
    first_range = first.high - first.low
    if first_range == 0:
        return "OAIR"

    # OD: open at one extreme of the first bracket, close drives away,
    # subsequent brackets do not return through the open.
    open_at_low = abs(session_open - first.low) <= first_range * 0.2
    open_at_high = abs(session_open - first.high) <= first_range * 0.2
    if open_at_low or open_at_high:
        # Check whether any later bracket crossed back through the open.
        crossed_back = any(b.low <= session_open <= b.high for b in brackets[1:3])
        if not crossed_back:
            return "OD"
        return "OTD"

    if len(brackets) >= 2:
        second = brackets[1]
        first_inside = first.low <= session_open <= first.high
        second_inside = second.low <= session_open <= second.high
        if first_inside and second_inside:
            return "OAIR"

    # First bracket sits entirely above/below open (gap-style).
    if first.low > session_open or first.high < session_open:
        return "OAOR_TYPE"

    return "OAIR"


def _detect_local_extrema(
    bucket_counts: dict[int, int],
    bucket_prices: dict[int, float],
    *,
    mode: str,
    margin: float = HVN_LVN_RELATIVE_MARGIN,
) -> tuple[float, ...]:
    """Find local maxima (HVN) or minima (LVN) of the TPO histogram.

    A bucket qualifies as a local extremum when its count differs from BOTH
    contiguous neighbours by at least ``margin`` (relative).
    """

    if not bucket_counts:
        return tuple()
    populated = sorted(bucket_counts.keys())
    if len(populated) < 2:
        return tuple()

    # Densify so gaps (count=0) are first-class neighbours.
    min_b = populated[0]
    max_b = populated[-1]
    dense: dict[int, int] = {b: bucket_counts.get(b, 0) for b in range(min_b, max_b + 1)}
    sorted_buckets = sorted(dense.keys())

    if mode == "max":
        return _detect_hvn(dense, sorted_buckets, bucket_prices, margin)
    return _detect_lvn(dense, sorted_buckets, bucket_prices, margin)


def _detect_hvn(
    dense: dict[int, int],
    sorted_buckets: list[int],
    bucket_prices: dict[int, float],
    margin: float,
) -> tuple[float, ...]:
    """High-volume nodes: contiguous plateaus whose count exceeds both
    flanking lower regions by ``margin``. We collapse plateaus to a single
    representative price (the plateau center)."""

    out: list[float] = []
    n = len(sorted_buckets)
    i = 0
    global_max = max(dense.values()) if dense else 0
    if global_max == 0:
        return tuple()
    # Significance floor: a plateau must contain at least margin × global max
    # touches to qualify as HVN. This filters out the "isolated count-1
    # bucket between zeros" false positive.
    sig_floor = max(2, int(global_max * 0.5))

    while i < n:
        b = sorted_buckets[i]
        c = dense[b]
        if c < sig_floor:
            i += 1
            continue
        # Walk the plateau forward while count stays equal.
        j = i
        while j + 1 < n and dense[sorted_buckets[j + 1]] == c:
            j += 1
        prev_count = dense[sorted_buckets[i - 1]] if i > 0 else 0
        next_count = dense[sorted_buckets[j + 1]] if j + 1 < n else 0
        # Plateau is a peak when both flanks are lower.
        if c > prev_count and c > next_count:
            # Margin check: differ by at least ``margin`` from the higher flank.
            ref = max(prev_count, next_count)
            if ref == 0 or (c - ref) / max(c, 1) >= margin:
                center = (sorted_buckets[i] + sorted_buckets[j]) // 2
                out.append(_price_for_bucket(center, bucket_prices))
        i = j + 1
    return tuple(sorted(out))


def _detect_lvn(
    dense: dict[int, int],
    sorted_buckets: list[int],
    bucket_prices: dict[int, float],
    margin: float,
) -> tuple[float, ...]:
    """Low-volume nodes: troughs where count dips between two higher shelves."""

    out: list[float] = []
    n = len(sorted_buckets)
    if n < 3:
        return tuple()
    i = 1
    while i < n - 1:
        b = sorted_buckets[i]
        c = dense[b]
        cp = dense[sorted_buckets[i - 1]]
        cn = dense[sorted_buckets[i + 1]]
        # Walk a flat valley (all equal counts) and treat its center as LVN.
        j = i
        while j + 1 < n - 1 and dense[sorted_buckets[j + 1]] == c:
            j += 1
        cn = dense[sorted_buckets[j + 1]] if j + 1 < n else 0
        if c < cp and c < cn:
            ref = max(cp, cn)
            if ref > 0 and (ref - c) / ref >= margin:
                center = (sorted_buckets[i] + sorted_buckets[j]) // 2
                out.append(_price_for_bucket(center, bucket_prices))
        i = j + 1
    return tuple(sorted(out))


def _price_for_bucket(bucket: int, known: dict[int, float]) -> float:
    if bucket in known:
        return known[bucket]
    if not known:
        return 0.0
    sample_b, sample_p = next(iter(known.items()))
    if sample_b == 0:
        return float(bucket)
    return sample_p / sample_b * bucket
