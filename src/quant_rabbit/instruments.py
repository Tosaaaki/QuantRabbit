from __future__ import annotations


G8_CURRENCIES: tuple[str, ...] = ("USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD")

# Default watchlist for the 5-minute trader cycle.
#
# Market reality: the old 7-pair list covered mostly USD majors and JPY crosses,
# while the currency-strength layer already reasons over the full G8 28-pair
# matrix. A 5-minute campaign that needs many small, bounded attempts should
# watch the same G8 universe it uses for strength and relative-value context.
# This is a coverage universe, not permission to trade every pair; missing
# OANDA instruments, stale quotes, spreads, calendar windows, profile status,
# and risk validation still block individual lanes.
DEFAULT_TRADER_PAIRS: tuple[str, ...] = (
    "EUR_USD", "GBP_USD", "AUD_USD", "NZD_USD", "USD_JPY", "USD_CAD", "USD_CHF",
    "EUR_GBP", "EUR_JPY", "EUR_AUD", "EUR_CAD", "EUR_CHF", "EUR_NZD",
    "GBP_JPY", "GBP_AUD", "GBP_CAD", "GBP_CHF", "GBP_NZD",
    "AUD_JPY", "AUD_CAD", "AUD_CHF", "AUD_NZD",
    "CAD_JPY", "CAD_CHF",
    "CHF_JPY",
    "NZD_JPY", "NZD_CAD", "NZD_CHF",
)

DEFAULT_TRADER_PAIRS_ARG = ",".join(DEFAULT_TRADER_PAIRS)

# Non-FX market context instruments. These are monitored with the same
# multi-timeframe technical stack as FX, but they are not automatically tradeable:
# broker account instruments must explicitly list them before any order path can
# treat them as candidates.
DEFAULT_CONTEXT_ASSETS: tuple[str, ...] = (
    # Equity indices
    "SPX500_USD", "NAS100_USD", "US30_USD", "JP225_USD", "DE30_EUR", "UK100_GBP",
    # Bonds
    "USB02Y_USD", "USB05Y_USD", "USB10Y_USD", "USB30Y_USD",
    # Commodities
    "XAU_USD", "XAG_USD", "BCO_USD", "WTICO_USD", "NATGAS_USD",
    # Crypto
    "BTC_USD", "ETH_USD",
)

DEFAULT_CONTEXT_ASSETS_ARG = ",".join(DEFAULT_CONTEXT_ASSETS)

# Broker-spec spread baselines used by RiskEngine's current-spread cap.
#
# These are named instrument specs, not strategy geometry. They describe the
# normal spread class for liquid G8 FX pairs so the §9 spread gate can compare
# live spread with a broker-facing baseline. Geometry and sizing still come
# from ATR, live spread, conversion quotes, and equity. Replace this table with
# persisted broker median spreads once the flow snapshot writes a reliable
# per-pair rolling baseline to the broker snapshot.
NORMAL_SPREAD_PIPS: dict[str, float] = {
    "EUR_USD": 0.5,
    "GBP_USD": 0.9,
    "AUD_USD": 0.5,
    "NZD_USD": 0.8,
    "USD_JPY": 0.4,
    "USD_CAD": 0.7,
    "USD_CHF": 0.7,
    "EUR_GBP": 0.7,
    "EUR_JPY": 0.8,
    "EUR_AUD": 1.2,
    "EUR_CAD": 1.4,
    "EUR_CHF": 0.9,
    "EUR_NZD": 1.8,
    "GBP_JPY": 1.5,
    "GBP_AUD": 1.8,
    "GBP_CAD": 2.0,
    "GBP_CHF": 1.8,
    "GBP_NZD": 2.5,
    "AUD_JPY": 0.8,
    "AUD_CAD": 1.1,
    "AUD_CHF": 1.2,
    "AUD_NZD": 1.4,
    "CAD_JPY": 1.0,
    "CAD_CHF": 1.4,
    "CHF_JPY": 1.2,
    "NZD_JPY": 1.1,
    "NZD_CAD": 1.5,
    "NZD_CHF": 1.5,
}


def instrument_pip_factor(pair: str) -> int:
    """Return the broker price precision used for one pip.

    JPY-quoted FX pairs conventionally quote one pip as 0.01; the other G8
    FX pairs quote one pip as 0.0001. This is a broker-spec convention, not a
    trading threshold.
    """

    return 100 if pair.upper().endswith("_JPY") else 10000
