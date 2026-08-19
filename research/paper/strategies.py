"""Causal strategy registry for the paper engine.

Each strategy is a pure function of precomputed causal features that returns
`(index, side)` pairs. Nothing here may look at a bar later than `i`.

Adding a strategy means adding one function and one `@strategy` decoration. The
engine scores it the same way as every other, so a new idea cannot be adopted on
a story -- it has to clear the same gate as the rest.

Seeded with the five families the previous session measured (so the new engine
reproduces known numbers) plus eight new ones spanning different mechanisms:
breakout, mean reversion, session, volatility regime, and multi-timeframe.
"""

REGISTRY = {}


def strategy(name):
    def deco(fn):
        REGISTRY[name] = fn
        return fn
    return deco


# --------------------------------------------------------------------------
# previously measured families -- kept so the engine can be checked against
# research/regime numbers rather than trusted
# --------------------------------------------------------------------------

@strategy("fade_range")
def fade_range(f, i):
    if f["eff"][i] < 0.15 and f["loc6h"][i] < 0.10:
        return +1
    if f["eff"][i] < 0.15 and f["loc6h"][i] > 0.90:
        return -1


@strategy("fade_stretch")
def fade_stretch(f, i):
    if f["z20"][i] < -2.5:
        return +1
    if f["z20"][i] > 2.5:
        return -1


@strategy("mom_break")
def mom_break(f, i):
    if f["c"][i] >= f["hi6h"][i] - 1e-12 and f["mom60"][i] > 1.0:
        return +1
    if f["c"][i] <= f["lo6h"][i] + 1e-12 and f["mom60"][i] < -1.0:
        return -1


@strategy("pullback")
def pullback(f, i):
    if f["c"][i] > f["e120"][i] and -1.5 < f["z20"][i] < -0.5 and f["mom60"][i] > 0:
        return +1
    if f["c"][i] < f["e120"][i] and 0.5 < f["z20"][i] < 1.5 and f["mom60"][i] < 0:
        return -1


@strategy("mom_trend")
def mom_trend(f, i):
    if f["eff"][i] > 0.35 and f["mom60"][i] > 2.0 and f["c"][i] > f["e120"][i]:
        return +1
    if f["eff"][i] > 0.35 and f["mom60"][i] < -2.0 and f["c"][i] < f["e120"][i]:
        return -1


# --------------------------------------------------------------------------
# new candidates
# --------------------------------------------------------------------------

@strategy("donchian_24h")
def donchian_24h(f, i):
    """Classic channel break on a longer window than mom_break's 6h."""
    if f["c"][i] >= f["hi24h"][i] - 1e-12:
        return +1
    if f["c"][i] <= f["lo24h"][i] + 1e-12:
        return -1


@strategy("range_rail")
def range_rail(f, i):
    """Fade the 24h rail, but only when the day has NOT been trending -- the
    range-rotation shape AGENT_CONTRACT §5 allows when width is executable."""
    if f["eff"][i] > 0.20 or f["atr"][i] <= 0:
        return None
    width = (f["hi24h"][i] - f["lo24h"][i]) / f["atr"][i]
    if width < 8:
        return None
    if f["loc24h"][i] < 0.05:
        return +1
    if f["loc24h"][i] > 0.95:
        return -1


@strategy("ema_cross_trend")
def ema_cross_trend(f, i):
    """Fast EMA crossing slow, taken only in the direction of the 24h EMA."""
    if f["e20"][i - 1] <= f["e120"][i - 1] and f["e20"][i] > f["e120"][i] \
            and f["c"][i] > f["e1440"][i]:
        return +1
    if f["e20"][i - 1] >= f["e120"][i - 1] and f["e20"][i] < f["e120"][i] \
            and f["c"][i] < f["e1440"][i]:
        return -1


@strategy("vol_expansion")
def vol_expansion(f, i):
    """Enter with the move when short ATR expands sharply over long ATR."""
    if f["atr_slow"][i] <= 0:
        return None
    if f["atr"][i] / f["atr_slow"][i] < 1.8:
        return None
    if f["mom60"][i] > 1.5:
        return +1
    if f["mom60"][i] < -1.5:
        return -1


@strategy("vol_squeeze_break")
def vol_squeeze_break(f, i):
    """The opposite regime: compressed volatility, then the first push out."""
    if f["atr_slow"][i] <= 0 or f["atr"][i] / f["atr_slow"][i] > 0.7:
        return None
    if f["c"][i] >= f["hi6h"][i] - 1e-12:
        return +1
    if f["c"][i] <= f["lo6h"][i] + 1e-12:
        return -1


@strategy("tokyo_open_break")
def tokyo_open_break(f, i):
    """Session shape: first break of the overnight range after the Tokyo open."""
    if not (0 <= f["hour"][i] <= 2):
        return None
    if f["c"][i] >= f["hi6h"][i] - 1e-12:
        return +1
    if f["c"][i] <= f["lo6h"][i] + 1e-12:
        return -1


@strategy("london_reversion")
def london_reversion(f, i):
    """Fade an overextended move into the London open."""
    if not (7 <= f["hour"][i] <= 9):
        return None
    if f["z20"][i] > 2.0:
        return -1
    if f["z20"][i] < -2.0:
        return +1


@strategy("mtf_aligned_pullback")
def mtf_aligned_pullback(f, i):
    """pullback, but requiring the 24h EMA to agree as well -- the multi-
    timeframe version, to test whether the extra filter is worth anything."""
    if f["c"][i] > f["e120"][i] > f["e1440"][i] and -1.5 < f["z20"][i] < -0.5:
        return +1
    if f["c"][i] < f["e120"][i] < f["e1440"][i] and 0.5 < f["z20"][i] < 1.5:
        return -1
