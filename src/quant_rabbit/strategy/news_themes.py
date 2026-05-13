"""Parse `logs/news_digest.md` into per-(pair, direction) score biases.

The `qr-news-digest` Claude Desktop routine writes a curated trader-
perspective digest each hour. This module reads that digest and
translates its themes into deterministic, bounded score modifiers per
(pair, direction) that trader_brain applies during _score_lane.

The translation is **rule-based**, not LLM-generated, so the trader
cycle stays deterministic and the parsing is unit-testable. Three
classes of signal are extracted:

1. **Currency-strength themes** (e.g. "USD strong post-CPI"): scan
   for "{CCY} strong|rally|bid|surge" or "{CCY} weak|sell|drop" and
   apply ±NEWS_CURRENCY_STRONG_BIAS to all pairs containing that
   currency, accounting for base/quote orientation.
2. **Risk-on/off themes**: "risk-off" flips currencies on the
   risk-asset axis (JPY/CHF strong, AUD/NZD weak); "risk-on" flips
   the opposite way.
3. **Explicit pair guidance** parsed from the `## 💱 Pair-Specific
   Notes` section ("**EUR_USD**: Bearish below 1.18" → EUR_USD LONG
   -NEWS_EXPLICIT_PAIR_BIAS). These dominate currency-level bias when
   present.

Modifiers are CLAMPED to ±NEWS_MAX_TOTAL_BIAS so even when multiple
themes pile up on one pair, the news layer cannot single-handedly
veto an otherwise strong technical setup. The trader's price-action
read still gets to win or lose on its own merits.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple


NEWS_CURRENCY_STRONG_BIAS = float(os.environ.get("QR_NEWS_CURRENCY_BIAS", "8.0"))
NEWS_RISK_TONE_BIAS = float(os.environ.get("QR_NEWS_RISK_TONE_BIAS", "6.0"))
NEWS_EXPLICIT_PAIR_BIAS = float(os.environ.get("QR_NEWS_EXPLICIT_PAIR_BIAS", "15.0"))
NEWS_MAX_TOTAL_BIAS = float(os.environ.get("QR_NEWS_MAX_TOTAL_BIAS", "25.0"))


# Currencies the trader runs on. Order matters only for display.
KNOWN_CURRENCIES = ("USD", "EUR", "GBP", "JPY", "AUD", "NZD", "CAD", "CHF")

# Pairs to score. Add new pairs here; theme translation auto-adapts
# because base/quote are split below.
KNOWN_PAIRS = (
    "EUR_USD", "GBP_USD", "AUD_USD", "NZD_USD", "USD_JPY", "USD_CAD",
    "USD_CHF", "EUR_JPY", "GBP_JPY", "AUD_JPY", "NZD_JPY", "CHF_JPY",
    "EUR_GBP", "EUR_CHF", "GBP_CHF",
)

# Theme phrases. Each phrase contributes ±1 to the currency's score
# inside `_currency_strength_score`. Keep patterns case-insensitive.
_CCY_STRENGTH_POSITIVE = re.compile(
    r"\b(strong|stronger|strength|"
    r"rally|rallies|rallying|rallied|"
    r"bid|surge|surges|surging|surged|"
    r"spike|spikes|spiking|spiked|"
    r"firm|firms|firming|firmed|"
    r"bullish|bull|breakout|breaks?\s+out|"
    r"climb|climbs|climbing|climbed|"
    r"gain|gains|gaining|gained|"
    r"jump|jumps|jumping|jumped|"
    r"reaccelerate[ds]?|reaccelerating|"
    r"higher|outperform)\b",
    re.IGNORECASE,
)
_CCY_STRENGTH_NEGATIVE = re.compile(
    r"\b(weak|weaker|weakness|"
    r"sell|sells|selling|sold|"
    r"drop|drops|dropping|dropped|"
    r"slide|slides|sliding|slid|"
    r"soft|softer|softening|softened|"
    r"bearish|bear|breakdown|breaks?\s+down|"
    r"tumble|tumbles|tumbling|tumbled|"
    r"fall|falls|falling|fell|"
    r"slump|slumps|slumping|slumped|"
    r"falter|falters|faltering|faltered|"
    r"lower|underperform|"
    r"intervention|intervene)\b",
    re.IGNORECASE,
)
_RISK_OFF_RE = re.compile(r"\brisk[\s-]*off\b", re.IGNORECASE)
_RISK_ON_RE = re.compile(r"\brisk[\s-]*on\b", re.IGNORECASE)
# "Short XXX/YYY" or "Long XXX/YYY" with underscore or slash separator.
_EXPLICIT_SHORT_RE = re.compile(
    r"\bshort\s+([A-Z]{3})[_/]([A-Z]{3})\b", re.IGNORECASE
)
_EXPLICIT_LONG_RE = re.compile(
    r"\blong\s+([A-Z]{3})[_/]([A-Z]{3})\b", re.IGNORECASE
)
# Bias headline. Captures both:
#   "**EUR_USD**: Bearish below 1.18"      (single pair)
#   "**GBP_USD, USD_JPY** — Asia open..."  (multi-pair, comma list)
# The first capture is a comma-separated list of pair tickers; the
# second is body text (200 chars) we'll scan for bull/bear keywords.
_PAIR_NOTE_RE = re.compile(
    r"\*\*([A-Z]{3}_[A-Z]{3}(?:\s*,\s*[A-Z]{3}_[A-Z]{3})*)\*\*\s*[:—\-]?\s*(.{0,250})",
    re.IGNORECASE | re.DOTALL,
)
_BEARISH_RE = re.compile(r"\b(bearish|short|downside|breakdown|sell)\b", re.IGNORECASE)
_BULLISH_RE = re.compile(r"\b(bullish|long|upside|breakout|buy)\b", re.IGNORECASE)


@dataclass(frozen=True)
class NewsThemes:
    biases: Dict[Tuple[str, str], float]  # (pair, direction) -> total delta
    detected_themes: tuple[str, ...]
    source_path: Optional[Path]

    def for_pair(self, pair: str, direction: str) -> tuple[float, str | None]:
        key = (pair, direction.upper())
        delta = self.biases.get(key, 0.0)
        if delta == 0.0:
            return 0.0, None
        sign = "+" if delta >= 0 else ""
        themes_short = ", ".join(self.detected_themes[:3])
        rationale = f"news themes ({themes_short}) → {pair}:{direction} {sign}{delta:.1f}"
        return delta, rationale


def _split_pair(pair: str) -> tuple[str, str] | None:
    parts = pair.split("_")
    if len(parts) != 2:
        return None
    return parts[0], parts[1]


def _currency_strength_signals(text: str) -> Dict[str, int]:
    """Per-currency net signal count: +1 per positive phrase near the
    ticker, -1 per negative phrase near the ticker. Window is ±80 chars.

    Also detects long-form mentions ("US Dollar rally" → USD+1,
    "Yen weakness" → JPY-1) via explicit name-to-code mappings so a
    digest that talks about "Dollar" or "Yen" without the ticker still
    contributes.
    """
    out: Dict[str, int] = {ccy: 0 for ccy in KNOWN_CURRENCIES}

    # Long-form currency aliases (case-insensitive whole-word).
    long_form = {
        "USD": (r"\b(US\s+Dollar|Dollar\s+Index|DXY|Greenback)\b",),
        "EUR": (r"\b(Euro)\b",),
        "GBP": (r"\b(Sterling|Pound|Cable)\b",),
        "JPY": (r"\b(Yen)\b",),
        "AUD": (r"\b(Aussie)\b",),
        "NZD": (r"\b(Kiwi)\b",),
        "CAD": (r"\b(Loonie|Canadian\s+Dollar)\b",),
        "CHF": (r"\b(Swissie|Swiss\s+Franc)\b",),
    }

    for ccy in KNOWN_CURRENCIES:
        # Build a single regex that matches either the 3-letter code
        # alone OR any long-form alias for this currency.
        patterns = [rf"\b{ccy}\b"] + list(long_form.get(ccy, ()))
        combined = "|".join(patterns)
        for m in re.finditer(combined, text, re.IGNORECASE):
            start = max(0, m.start() - 30)
            end = min(len(text), m.end() + 30)
            window = text[start:end]
            # Cross-contamination guard: if the window contains MORE
            # mentions of a DIFFERENT currency than of this one, the
            # sentiment word is more likely about that other currency.
            this_count = len(re.findall(combined, window, re.IGNORECASE))
            other_ccy_count = sum(
                len(re.findall(rf"\b{c}\b", window, re.IGNORECASE))
                for c in KNOWN_CURRENCIES if c != ccy
            )
            if other_ccy_count > this_count:
                continue
            if _CCY_STRENGTH_POSITIVE.search(window):
                out[ccy] += 1
            if _CCY_STRENGTH_NEGATIVE.search(window):
                out[ccy] -= 1
    return out


def _apply_currency_score(
    biases: Dict[Tuple[str, str], float],
    currency: str,
    strength: int,
) -> None:
    """Translate a per-currency signal count into per-pair biases.

    Magnitude scales with the absolute signal count (more mentions of
    "USD rallies" = stronger bias) but saturates via `tanh` so a single
    high-count outlier can't dominate. `NEWS_CURRENCY_STRONG_BIAS` is
    the saturation ceiling per pair side.

    Strong currency on BASE side → that pair LONG +bias, SHORT -bias.
    Strong currency on QUOTE side → that pair LONG -bias, SHORT +bias.
    (And vice versa for weak.)

    Magnitude formula `tanh(strength / 3) × ceiling` saturates near
    strength=6 (≈0.91× ceiling) and treats strength=1 as ~0.32× ceiling
    so a single fleeting mention doesn't move the score much.
    """
    if strength == 0:
        return
    import math
    sign = 1 if strength > 0 else -1
    magnitude = NEWS_CURRENCY_STRONG_BIAS * math.tanh(abs(strength) / 3.0)
    for pair in KNOWN_PAIRS:
        split = _split_pair(pair)
        if not split:
            continue
        base, quote = split
        if currency == base:
            biases[(pair, "LONG")] = biases.get((pair, "LONG"), 0.0) + sign * magnitude
            biases[(pair, "SHORT")] = biases.get((pair, "SHORT"), 0.0) - sign * magnitude
        elif currency == quote:
            biases[(pair, "LONG")] = biases.get((pair, "LONG"), 0.0) - sign * magnitude
            biases[(pair, "SHORT")] = biases.get((pair, "SHORT"), 0.0) + sign * magnitude


def _apply_risk_tone(biases: Dict[Tuple[str, str], float], risk_off: bool) -> None:
    """Risk-off → JPY/CHF strong, AUD/NZD weak. Risk-on → opposite.

    Magnitude is `NEWS_RISK_TONE_BIAS` per currency leg; smaller than
    a direct currency-strength signal because risk tone is a meta-theme.
    """
    sign = 1 if risk_off else -1
    safe_havens = ("JPY", "CHF")
    risk_assets = ("AUD", "NZD")
    for ccy in safe_havens:
        _apply_currency_score(biases, ccy, sign * (NEWS_RISK_TONE_BIAS // NEWS_CURRENCY_STRONG_BIAS or 1))
    for ccy in risk_assets:
        _apply_currency_score(biases, ccy, -sign * (NEWS_RISK_TONE_BIAS // NEWS_CURRENCY_STRONG_BIAS or 1))


def _parse_explicit_pair_notes(text: str, biases: Dict[Tuple[str, str], float]) -> None:
    """Read pair-bold headlines (any section) and convert bullish/bearish
    prose into pair-level biases. Handles both:
      `**EUR_USD**: Bearish below 1.18` (single pair)
      `**GBP_USD, USD_JPY** — Asia open...` (multi-pair comma list)
    Multi-pair entries split the bias across all listed pairs (each
    pair gets the SAME bias direction; magnitude unchanged because the
    body sentence applies to all of them).
    """
    for m in _PAIR_NOTE_RE.finditer(text):
        pair_list_raw = m.group(1)
        body = m.group(2)
        # Strip and split the comma-separated pair list.
        pairs = [p.strip().upper() for p in pair_list_raw.split(",") if p.strip()]
        bearish = bool(_BEARISH_RE.search(body))
        bullish = bool(_BULLISH_RE.search(body))
        if bearish and not bullish:
            for pair in pairs:
                biases[(pair, "LONG")] = biases.get((pair, "LONG"), 0.0) - NEWS_EXPLICIT_PAIR_BIAS
                biases[(pair, "SHORT")] = biases.get((pair, "SHORT"), 0.0) + NEWS_EXPLICIT_PAIR_BIAS
        elif bullish and not bearish:
            for pair in pairs:
                biases[(pair, "LONG")] = biases.get((pair, "LONG"), 0.0) + NEWS_EXPLICIT_PAIR_BIAS
                biases[(pair, "SHORT")] = biases.get((pair, "SHORT"), 0.0) - NEWS_EXPLICIT_PAIR_BIAS


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def parse_news_themes(digest_path: Path) -> NewsThemes:
    """Parse the news digest into per-(pair, direction) biases."""
    if not digest_path.exists():
        return NewsThemes(biases={}, detected_themes=(), source_path=None)
    try:
        text = digest_path.read_text(encoding="utf-8")
    except OSError:
        return NewsThemes(biases={}, detected_themes=(), source_path=None)

    biases: Dict[Tuple[str, str], float] = {}
    themes: list[str] = []

    # Currency strength signals first.
    strength = _currency_strength_signals(text)
    for ccy in KNOWN_CURRENCIES:
        if strength[ccy] != 0:
            _apply_currency_score(biases, ccy, strength[ccy])
            themes.append(f"{ccy}{'+' if strength[ccy] > 0 else '-'}{abs(strength[ccy])}")

    # Risk tone (single overall theme).
    if _RISK_OFF_RE.search(text):
        _apply_risk_tone(biases, risk_off=True)
        themes.append("risk-off")
    elif _RISK_ON_RE.search(text):
        _apply_risk_tone(biases, risk_off=False)
        themes.append("risk-on")

    # Explicit pair notes (these are typically the strongest signal —
    # the digest writer has already done the analysis).
    _parse_explicit_pair_notes(text, biases)

    # Clamp per (pair, direction) to bounded range.
    biases = {k: _clamp(v, -NEWS_MAX_TOTAL_BIAS, NEWS_MAX_TOTAL_BIAS) for k, v in biases.items()}

    return NewsThemes(
        biases=biases,
        detected_themes=tuple(themes),
        source_path=digest_path,
    )
