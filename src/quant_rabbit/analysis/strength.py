"""G8 currency-strength meter from a 28-pair matrix.

Compute each currency's relative strength by averaging its percentage change
across all pairs it participates in over the chosen lookback. Pairs where the
currency is the base count positively; where it is the quote, negatively.

Output ranks USD / EUR / GBP / JPY / AUD / CAD / CHF / NZD by score and flags
the strongest×weakest cross as the prime trade-direction candidate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable, Mapping, Sequence

from quant_rabbit.analysis.candles import Candle, fetch_candles_via_client
from quant_rabbit.broker.oanda import OandaReadOnlyClient


G8_CURRENCIES: tuple[str, ...] = ("USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD")

# 28 unique pairs across G8 (n*(n-1)/2)
DEFAULT_PAIR_UNIVERSE: tuple[str, ...] = (
    "EUR_USD", "GBP_USD", "AUD_USD", "NZD_USD", "USD_JPY", "USD_CAD", "USD_CHF",
    "EUR_GBP", "EUR_JPY", "EUR_AUD", "EUR_CAD", "EUR_CHF", "EUR_NZD",
    "GBP_JPY", "GBP_AUD", "GBP_CAD", "GBP_CHF", "GBP_NZD",
    "AUD_JPY", "AUD_CAD", "AUD_CHF", "AUD_NZD",
    "CAD_JPY", "CAD_CHF",
    "CHF_JPY",
    "NZD_JPY", "NZD_CAD", "NZD_CHF",
)


@dataclass(frozen=True)
class CurrencyScore:
    currency: str
    score_pct: float
    rank: int  # 1 = strongest

    def to_dict(self) -> dict[str, object]:
        return {"currency": self.currency, "score_pct": self.score_pct, "rank": self.rank}


@dataclass(frozen=True)
class StrengthSnapshot:
    generated_at_utc: str
    granularity: str
    lookback_bars: int
    pairs_used: tuple[str, ...]
    pairs_missing: tuple[str, ...]
    scores: tuple[CurrencyScore, ...]
    strongest_pair_suggestion: str | None  # e.g. "USD_JPY:LONG" if USD strongest, JPY weakest
    issues: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        return {
            "generated_at_utc": self.generated_at_utc,
            "granularity": self.granularity,
            "lookback_bars": self.lookback_bars,
            "pairs_used": list(self.pairs_used),
            "pairs_missing": list(self.pairs_missing),
            "scores": [s.to_dict() for s in self.scores],
            "strongest_pair_suggestion": self.strongest_pair_suggestion,
            "issues": list(self.issues),
        }


def build_strength_snapshot(
    *,
    client: OandaReadOnlyClient,
    pairs: Sequence[str] = DEFAULT_PAIR_UNIVERSE,
    granularity: str = "H1",
    lookback_bars: int = 24,
    fetch_count: int = 50,
) -> StrengthSnapshot:
    """Fetch candles for each pair and rank G8 currencies by aggregate change."""

    if lookback_bars >= fetch_count:
        fetch_count = lookback_bars + 5

    closes_by_pair: dict[str, tuple[float, ...]] = {}
    issues: list[str] = []
    pairs_missing: list[str] = []
    for pair in pairs:
        try:
            candles = fetch_candles_via_client(client, pair, granularity, count=fetch_count)
            closes_by_pair[pair] = tuple(c.close for c in candles)
        except Exception as exc:
            issues.append(f"MISSING_PAIR_{pair}: {exc}")
            pairs_missing.append(pair)

    sums: dict[str, float] = {c: 0.0 for c in G8_CURRENCIES}
    counts: dict[str, int] = {c: 0 for c in G8_CURRENCIES}
    pairs_used: list[str] = []
    for pair, closes in closes_by_pair.items():
        if len(closes) <= lookback_bars:
            continue
        pct = (closes[-1] - closes[-lookback_bars - 1]) / closes[-lookback_bars - 1] * 100.0 if closes[-lookback_bars - 1] else 0.0
        base, _, quote = pair.upper().partition("_")
        if base in sums and quote in sums:
            sums[base] += pct
            counts[base] += 1
            sums[quote] -= pct
            counts[quote] += 1
            pairs_used.append(pair)

    raw_scores = []
    for cur in G8_CURRENCIES:
        avg = (sums[cur] / counts[cur]) if counts[cur] else 0.0
        raw_scores.append((cur, avg))
    raw_scores.sort(key=lambda kv: kv[1], reverse=True)
    scores: list[CurrencyScore] = []
    for rank, (cur, sc) in enumerate(raw_scores, start=1):
        scores.append(CurrencyScore(currency=cur, score_pct=sc, rank=rank))

    suggestion: str | None = None
    if len(scores) >= 2:
        strongest = scores[0].currency
        weakest = scores[-1].currency
        candidates = (f"{strongest}_{weakest}", f"{weakest}_{strongest}")
        for cand in candidates:
            if cand in DEFAULT_PAIR_UNIVERSE or cand in pairs:
                if cand.startswith(strongest + "_"):
                    suggestion = f"{cand}:LONG"
                else:
                    suggestion = f"{cand}:SHORT"
                break

    return StrengthSnapshot(
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        granularity=granularity,
        lookback_bars=lookback_bars,
        pairs_used=tuple(pairs_used),
        pairs_missing=tuple(pairs_missing),
        scores=tuple(scores),
        strongest_pair_suggestion=suggestion,
        issues=tuple(issues),
    )
