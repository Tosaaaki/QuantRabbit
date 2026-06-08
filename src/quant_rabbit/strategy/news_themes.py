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

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

from quant_rabbit.instruments import DEFAULT_TRADER_PAIRS


NEWS_CURRENCY_STRONG_BIAS = float(os.environ.get("QR_NEWS_CURRENCY_BIAS", "8.0"))
NEWS_RISK_TONE_BIAS = float(os.environ.get("QR_NEWS_RISK_TONE_BIAS", "6.0"))
NEWS_EXPLICIT_PAIR_BIAS = float(os.environ.get("QR_NEWS_EXPLICIT_PAIR_BIAS", "15.0"))
NEWS_MAX_TOTAL_BIAS = float(os.environ.get("QR_NEWS_MAX_TOTAL_BIAS", "25.0"))


# Currencies the trader runs on. Order matters only for display.
KNOWN_CURRENCIES = ("USD", "EUR", "GBP", "JPY", "AUD", "NZD", "CAD", "CHF")

# Pairs to score. Keep this aligned with the trader's discovery universe;
# missing pairs silently remove macro/news context from candidate mining.
KNOWN_PAIRS = DEFAULT_TRADER_PAIRS

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
_JPY_INTERVENTION_RE = re.compile(r"\b(intervention|intervene|rate[\s-]*check)\b", re.IGNORECASE)
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
_PAIR_TOKEN = r"[A-Z]{3}[_/][A-Z]{3}"
_PAIR_LIST_SEPARATOR = r"(?:,|、|\s+/\s+|\s+\band\b\s+)"
_PAIR_NOTE_RES = (
    re.compile(
        rf"\*\*({_PAIR_TOKEN}(?:\s*{_PAIR_LIST_SEPARATOR}\s*{_PAIR_TOKEN})*)\*\*\s*[:—\-]?\s*(.{{0,250}})",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        rf"(?m)^\s*[-*]\s*({_PAIR_TOKEN}(?:\s*{_PAIR_LIST_SEPARATOR}\s*{_PAIR_TOKEN})*)\s*[:—\-]\s*(.{{0,250}})",
        re.IGNORECASE,
    ),
)
_BEARISH_RE = re.compile(
    r"\b(bearish|short|downside|breakdown|breaks?\s+below|sell|sold|"
    r"lower|lows?|fragile|vulnerable|falls?|drops?|tumbles?|weak(?:er|ness)?|"
    r"pullbacks?\s+can\s+be\s+sold)\b",
    re.IGNORECASE,
)
_BULLISH_RE = re.compile(
    r"\b(bullish|long|upside|breakout|breaks?\s+above|buy|higher|highs?|"
    r"firm|strong(?:er)?|support(?:ed)?|bid)\b",
    re.IGNORECASE,
)
_UPSIDE_CAP_RE = re.compile(r"\b(cap(?:s|ped|ping)?\s+upside|upside\s+(?:is\s+)?cap(?:ped|s)?)\b", re.IGNORECASE)
_DIGEST_JST_STAMP_RE = re.compile(r"FX News Digest\s+—\s+(\d{4}-\d{2}-\d{2})\s+(\d{2}):(\d{2})\s+JST")
_PRE_EVENT_LANGUAGE_RE = re.compile(
    r"\b(ahead\s+of|pre[-\s]?release|next\s+\d+(?:-\d+)?\s+hours?|"
    r"tonight\s+is\s+the\s+main\s+volatility\s+event|lands?\s+at\s+the\s+same\s+time)\b",
    re.IGNORECASE,
)
_HIGH_IMPACT_TOKENS = {"HIGH", "VERY_HIGH", "RED"}


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
            jpy_intervention = ccy == "JPY" and bool(_JPY_INTERVENTION_RE.search(window))
            if jpy_intervention:
                out[ccy] += 1
            if _CCY_STRENGTH_POSITIVE.search(window):
                out[ccy] += 1
            if _CCY_STRENGTH_NEGATIVE.search(window) and not jpy_intervention:
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
    _apply_currency_bias(biases, currency, sign=sign, magnitude=magnitude)


def _apply_currency_bias(
    biases: Dict[Tuple[str, str], float],
    currency: str,
    *,
    sign: int,
    magnitude: float,
) -> None:
    if magnitude <= 0:
        return
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
        _apply_currency_bias(biases, ccy, sign=sign, magnitude=NEWS_RISK_TONE_BIAS)
    for ccy in risk_assets:
        _apply_currency_bias(biases, ccy, sign=-sign, magnitude=NEWS_RISK_TONE_BIAS)


def _parse_explicit_pair_notes(text: str, biases: Dict[Tuple[str, str], float]) -> None:
    """Read pair-bold headlines (any section) and convert bullish/bearish
    prose into pair-level biases. Handles both:
      `**EUR_USD**: Bearish below 1.18` (single pair)
      `**GBP_USD, USD_JPY** — Asia open...` (multi-pair comma list)
    Multi-pair entries split the bias across all listed pairs (each
    pair gets the SAME bias direction; magnitude unchanged because the
    body sentence applies to all of them).
    """
    for pair_list_raw, body in _iter_pair_notes(text):
        pairs = _parse_pair_list(pair_list_raw)
        upside_capped = bool(_UPSIDE_CAP_RE.search(body))
        bearish = bool(_BEARISH_RE.search(body)) or upside_capped
        bullish = bool(_BULLISH_RE.search(body)) and not upside_capped
        if bearish and not bullish:
            for pair in pairs:
                biases[(pair, "LONG")] = biases.get((pair, "LONG"), 0.0) - NEWS_EXPLICIT_PAIR_BIAS
                biases[(pair, "SHORT")] = biases.get((pair, "SHORT"), 0.0) + NEWS_EXPLICIT_PAIR_BIAS
        elif bullish and not bearish:
            for pair in pairs:
                biases[(pair, "LONG")] = biases.get((pair, "LONG"), 0.0) + NEWS_EXPLICIT_PAIR_BIAS
                biases[(pair, "SHORT")] = biases.get((pair, "SHORT"), 0.0) - NEWS_EXPLICIT_PAIR_BIAS


def _iter_pair_notes(text: str) -> list[tuple[str, str]]:
    notes: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for pattern in _PAIR_NOTE_RES:
        for match in pattern.finditer(text):
            item = (match.group(1), match.group(2))
            if item in seen:
                continue
            seen.add(item)
            notes.append(item)
    return notes


def _parse_pair_list(value: str) -> list[str]:
    pairs: list[str] = []
    for match in re.finditer(_PAIR_TOKEN, value.upper()):
        pair = match.group(0).replace("/", "_")
        if pair in KNOWN_PAIRS and pair not in pairs:
            pairs.append(pair)
    return pairs


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def parse_news_themes(
    digest_path: Path,
    *,
    calendar_path: Path | None = None,
    news_items_path: Path | None = None,
    now_utc: datetime | None = None,
) -> NewsThemes:
    """Parse the news digest into per-(pair, direction) biases."""
    source_path: Path | None = None
    digest_text = ""
    themes: list[str] = []
    if digest_path.exists():
        source_path = digest_path
        try:
            loaded = digest_path.read_text(encoding="utf-8")
        except OSError:
            loaded = ""
        if _pre_event_digest_is_stale(loaded, digest_path, calendar_path=calendar_path, now_utc=now_utc):
            themes.append("stale_pre_event_digest")
        else:
            digest_text = loaded
    news_items_text, news_item_theme = _load_news_items_theme_text(
        news_items_path,
        calendar_path=calendar_path,
        now_utc=now_utc,
    )
    if news_items_text and source_path is None:
        source_path = news_items_path
    if news_item_theme:
        themes.append(news_item_theme)

    text = "\n".join(part for part in (digest_text, news_items_text) if part.strip())
    if not text.strip():
        return NewsThemes(biases={}, detected_themes=tuple(themes), source_path=source_path)

    biases: Dict[Tuple[str, str], float] = {}

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
        source_path=source_path,
    )


def _load_news_items_theme_text(
    news_items_path: Path | None,
    *,
    calendar_path: Path | None,
    now_utc: datetime | None,
) -> tuple[str, str | None]:
    if news_items_path is None or not news_items_path.exists():
        return "", None
    try:
        payload = json.loads(news_items_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "", None
    items = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        return "", None
    now = now_utc or datetime.now(timezone.utc)
    lookback_hours = _optional_float(payload.get("lookback_hours"))
    cutoff = now - timedelta(hours=lookback_hours) if lookback_hours and lookback_hours > 0 else None
    lines: list[str] = []
    used = 0
    for item in items:
        if not isinstance(item, dict):
            continue
        published = _parse_utc(item.get("published_at_utc"))
        if published is not None:
            if published > now:
                continue
            if cutoff is not None and published < cutoff:
                continue
        title = str(item.get("title") or "")
        summary = str(item.get("summary") or "")
        categories = " ".join(str(v) for v in item.get("categories") or [] if v)
        currencies = " ".join(str(v) for v in item.get("currencies") or [] if v)
        pairs = " ".join(str(v) for v in item.get("pairs") or [] if v)
        topics = " ".join(str(v) for v in item.get("topics") or [] if v)
        text = " ".join(part for part in (title, summary, categories, currencies, pairs, topics) if part).strip()
        if not text:
            continue
        if _news_item_is_stale_pre_event(text, published, calendar_path=calendar_path, now_utc=now):
            continue
        lines.append(text)
        used += 1
    theme = f"news_items:{used}" if used else None
    return "\n".join(lines), theme


def _optional_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _news_item_is_stale_pre_event(
    text: str,
    published_at: datetime | None,
    *,
    calendar_path: Path | None,
    now_utc: datetime,
) -> bool:
    if not text.strip() or calendar_path is None or not calendar_path.exists():
        return False
    if not _PRE_EVENT_LANGUAGE_RE.search(text):
        return False
    try:
        payload = json.loads(calendar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    text_upper = text.upper()
    for event in payload.get("events") or payload.get("calendar") or []:
        if not isinstance(event, dict):
            continue
        impact = str(event.get("impact") or event.get("importance") or "").upper()
        if impact not in _HIGH_IMPACT_TOKENS:
            continue
        ts = _parse_utc(event.get("timestamp_utc") or event.get("time_utc") or event.get("time") or event.get("timestamp"))
        if ts is None or ts > now_utc:
            continue
        title = str(event.get("title") or event.get("event") or "")
        currency = str(event.get("currency") or event.get("country") or "").upper()
        if not _event_mentioned(text_upper, title=title, currency=currency):
            continue
        if published_at is None or published_at <= ts or now_utc - ts <= timedelta(hours=6):
            return True
    return False


def _pre_event_digest_is_stale(
    text: str,
    digest_path: Path,
    *,
    calendar_path: Path | None,
    now_utc: datetime | None,
) -> bool:
    """Suppress pre-release news bias after the named high-impact event fires.

    The hourly curated digest is advisory. Once a scheduled high-impact event
    occurs after that digest was written, its pre-event USD/GBP/etc. tone is
    stale and chart/broker truth must rebuild the market read.
    """
    if not text.strip() or calendar_path is None or not calendar_path.exists():
        return False
    digest_ts = _digest_timestamp_utc(text, digest_path)
    now = now_utc or datetime.now(timezone.utc)
    try:
        payload = json.loads(calendar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False

    events = payload.get("events") or payload.get("calendar") or []
    if isinstance(events, list):
        text_upper = text.upper()
        for event in events:
            if not isinstance(event, dict):
                continue
            impact = str(event.get("impact") or event.get("importance") or "").upper()
            if impact not in _HIGH_IMPACT_TOKENS:
                continue
            ts = _parse_utc(event.get("timestamp_utc") or event.get("time_utc") or event.get("time") or event.get("timestamp"))
            if ts is None or ts > now:
                continue
            title = str(event.get("title") or event.get("event") or "")
            currency = str(event.get("currency") or event.get("country") or "").upper()
            mentioned = _event_mentioned(text_upper, title=title, currency=currency)
            if mentioned and digest_ts <= ts:
                return True
            if mentioned and _PRE_EVENT_LANGUAGE_RE.search(text) and now - ts <= timedelta(hours=6):
                return True

    issues = payload.get("issues") or []
    if (
        any("MISSING_FOREX_FACTORY_FEED" in str(issue) for issue in issues)
        and _PRE_EVENT_LANGUAGE_RE.search(text)
    ):
        return True
    return False


def _digest_timestamp_utc(text: str, digest_path: Path) -> datetime:
    match = _DIGEST_JST_STAMP_RE.search(text)
    if match:
        date_part, hour, minute = match.groups()
        try:
            ts = datetime.fromisoformat(f"{date_part}T{hour}:{minute}:00").replace(
                tzinfo=timezone(timedelta(hours=9))
            )
            return ts.astimezone(timezone.utc)
        except ValueError:
            pass
    try:
        return datetime.fromtimestamp(digest_path.stat().st_mtime, tz=timezone.utc)
    except OSError:
        return datetime.now(timezone.utc)


def _parse_utc(value: object) -> datetime | None:
    if not value:
        return None
    try:
        ts = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _event_mentioned(text_upper: str, *, title: str, currency: str) -> bool:
    title_upper = title.upper()
    event_aliases = {
        "NON-FARM": ("NFP", "NON-FARM", "PAYROLL"),
        "EMPLOYMENT": ("EMPLOYMENT", "JOBS", "LABOR", "LABOUR"),
        "UNEMPLOYMENT": ("UNEMPLOYMENT", "JOBS", "LABOR", "LABOUR"),
        "EARNINGS": ("EARNINGS", "WAGES"),
        "CPI": ("CPI", "INFLATION"),
        "FOMC": ("FOMC", "FED"),
        "RATE": ("RATE", "CENTRAL BANK"),
        "PMI": ("PMI",),
        "GDP": ("GDP",),
    }
    aliases: set[str] = set()
    for token, values in event_aliases.items():
        if token in title_upper:
            aliases.update(values)
    words = [w for w in re.findall(r"[A-Z]{4,}", title_upper) if w not in {"CHANGE", "RATE"}]
    aliases.update(words[:3])
    if aliases and any(alias in text_upper for alias in aliases):
        return True
    return bool(currency and currency in text_upper and _PRE_EVENT_LANGUAGE_RE.search(text_upper))
