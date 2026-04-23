#!/usr/bin/env python3
"""
News Fetcher — Periodically fetch FX news from 3 sources and cache as JSON

Sources:
1. Finnhub: Market news (forex) + economic calendar
2. Alpha Vantage: News sentiment (by currency)
3. Forex Factory: Economic calendar (HTML scraping)

Output: logs/news_cache.json
Run: Hourly news pipeline fetch. Trader sessions only read the cache.

Usage:
    python3 tools/news_fetcher.py              # Fetch all sources
    python3 tools/news_fetcher.py --summary     # Display cache summary
    python3 tools/news_fetcher.py --calendar    # Calendar only
    python3 tools/news_fetcher.py --headlines   # Headlines only
    python3 tools/news_fetcher.py --sentiment   # Sentiment only
"""
import json
import os
import ssl
import sys
import time
import urllib.request
import urllib.parse
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE_FILE = ROOT / "logs" / "news_cache.json"
FX_CURRENCIES = ["USD", "JPY", "EUR", "GBP", "AUD"]
FX_PAIRS = ["USD_JPY", "EUR_USD", "GBP_USD", "AUD_USD", "EUR_JPY", "GBP_JPY", "AUD_JPY"]

# SSL context (workaround for SSL verification issues in some environments)
SSL_CTX = ssl.create_default_context()


def load_config():
    cfg = {}
    env_path = ROOT / "config" / "env.toml"
    if not env_path.exists():
        return cfg
    for line in open(env_path):
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            cfg[k.strip()] = v.strip().strip('"')
    return cfg


def api_get(url, headers=None, timeout=10):
    """urllib GET request. Returns None on failure."""
    try:
        req = urllib.request.Request(url, headers=headers or {})
        resp = urllib.request.urlopen(req, timeout=timeout, context=SSL_CTX)
        return json.loads(resp.read())
    except Exception as e:
        print(f"  [WARN] API error: {url[:80]}... → {e}", file=sys.stderr)
        return None


def load_existing_cache():
    """Load existing cache. Used for merging."""
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text())
        except Exception:
            pass
    return {}


# ========== SOURCE 1: Finnhub ==========

def fetch_finnhub_news(token: str, max_items: int = 30) -> list[dict]:
    """Finnhub /news?category=forex → FX-related headlines"""
    url = f"https://finnhub.io/api/v1/news?category=forex&token={token}"
    data = api_get(url)
    if not data:
        return []

    results = []
    for item in data[:max_items]:
        # Estimate related currencies
        related = detect_currencies(item.get("headline", "") + " " + item.get("summary", ""))
        results.append({
            "time": datetime.fromtimestamp(item.get("datetime", 0), tz=timezone.utc).isoformat(),
            "headline": item.get("headline", ""),
            "summary": (item.get("summary", "") or "")[:200],
            "source": item.get("source", ""),
            "url": item.get("url", ""),
            "related_currencies": related,
            "category": item.get("category", "forex"),
        })
    return results


def fetch_finnhub_calendar(token: str) -> list[dict]:
    """Finnhub /calendar/economic → Economic calendar"""
    # Today through 3 days ahead
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    end = (datetime.now(timezone.utc) + timedelta(days=3)).strftime("%Y-%m-%d")

    url = f"https://finnhub.io/api/v1/calendar/economic?from={today}&to={end}&token={token}"
    data = api_get(url)
    if not data:
        return []

    events = data.get("economicCalendar", [])
    results = []
    for ev in events:
        country = ev.get("country", "")
        # FX-related countries only
        if country not in ["US", "JP", "EU", "GB", "AU", "DE", "FR", "CA", "NZ", "CH"]:
            continue

        impact = estimate_impact(ev.get("event", ""), ev.get("impact", ""))
        results.append({
            "time": ev.get("time", ""),
            "country": country,
            "event": ev.get("event", ""),
            "impact": impact,
            "forecast": ev.get("estimate", ""),
            "previous": ev.get("prev", ""),
            "actual": ev.get("actual", ""),
            "unit": ev.get("unit", ""),
        })

    # Sort by impact (high > medium > low)
    impact_order = {"high": 0, "medium": 1, "low": 2}
    results.sort(key=lambda x: (impact_order.get(x["impact"], 3), x["time"]))
    return results


# ========== SOURCE 2: Alpha Vantage ==========

def fetch_alphavantage_sentiment(token: str) -> dict:
    """Alpha Vantage NEWS_SENTIMENT → Sentiment by currency"""
    # Use topics filter only (FOREX: tickers not supported on free tier)
    url = (
        f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT"
        f"&topics=economy_fiscal,economy_monetary,finance"
        f"&sort=LATEST&limit=50&apikey={token}"
    )
    data = api_get(url, timeout=15)
    if not data or "feed" not in data:
        return {"scores": {}, "articles": []}

    # Aggregate scores by currency
    currency_scores: dict[str, list[float]] = {c: [] for c in FX_CURRENCIES}
    articles = []

    for item in data.get("feed", [])[:50]:
        headline = item.get("title", "")
        sentiment = float(item.get("overall_sentiment_score", 0))
        time_str = item.get("time_published", "")

        # Sentiment per currency
        for ticker_data in item.get("ticker_sentiment", []):
            ticker = ticker_data.get("ticker", "")
            score = float(ticker_data.get("ticker_sentiment_score", 0))
            for ccy in FX_CURRENCIES:
                if ccy in ticker:
                    currency_scores[ccy].append(score)

        # Article list (with related currencies)
        related = detect_currencies(headline)
        if related:
            articles.append({
                "time": format_av_time(time_str),
                "headline": headline,
                "sentiment": sentiment,
                "sentiment_label": item.get("overall_sentiment_label", ""),
                "source": item.get("source", ""),
                "related_currencies": related,
            })

    # Average scores
    scores = {}
    for ccy, vals in currency_scores.items():
        if vals:
            avg = sum(vals) / len(vals)
            scores[ccy] = {
                "score": round(avg, 4),
                "count": len(vals),
                "label": sentiment_label(avg),
            }

    return {"scores": scores, "articles": articles[:20]}


def format_av_time(time_str: str) -> str:
    """Alpha Vantage time '20260330T120000' → ISO format"""
    try:
        dt = datetime.strptime(time_str[:15], "%Y%m%dT%H%M%S")
        return dt.replace(tzinfo=timezone.utc).isoformat()
    except Exception:
        return time_str


def sentiment_label(score: float) -> str:
    if score >= 0.15:
        return "Bullish"
    elif score > 0.05:
        return "Somewhat_Bullish"
    elif score > -0.05:
        return "Neutral"
    elif score > -0.15:
        return "Somewhat_Bearish"
    else:
        return "Bearish"


# ========== SOURCE 3: Forex Factory Calendar (RSS) ==========

def fetch_forexfactory_calendar() -> list[dict]:
    """Fetch events from Forex Factory RSS/XML (fallback)"""
    url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
    data = api_get(url, timeout=10)
    if not data:
        return []

    results = []
    for ev in data:
        country = ev.get("country", "")
        if country not in ["USD", "JPY", "EUR", "GBP", "AUD", "CAD", "NZD", "CHF"]:
            continue

        results.append({
            "time": ev.get("date", ""),
            "country": country,
            "event": ev.get("title", ""),
            "impact": ev.get("impact", "Low").lower(),
            "forecast": ev.get("forecast", ""),
            "previous": ev.get("previous", ""),
            "actual": ev.get("actual", ""),
        })
    return results


# ========== Utilities ==========

def detect_currencies(text: str) -> list[str]:
    """Detect related currencies from text"""
    text_upper = text.upper()
    found = []

    # Currency name matching
    currency_keywords = {
        "USD": ["USD", "DOLLAR", "FED ", "FEDERAL RESERVE", "US ECONOMY", "FOMC",
                "NFP", "NONFARM", "TREASURY", "WALL STREET"],
        "JPY": ["JPY", "YEN", "BOJ", "BANK OF JAPAN", "日銀", "円", "JAPAN"],
        "EUR": ["EUR", "EURO", "ECB", "EUROPEAN", "EUROZONE", "LAGARDE"],
        "GBP": ["GBP", "POUND", "STERLING", "BOE", "BANK OF ENGLAND", "UK "],
        "AUD": ["AUD", "AUSSIE", "RBA", "AUSTRALIA", "RESERVE BANK OF AUSTRALIA"],
    }

    for ccy, keywords in currency_keywords.items():
        if any(kw in text_upper for kw in keywords):
            found.append(ccy)
    return found


def estimate_impact(event_name: str, raw_impact: str) -> str:
    """Estimate impact from event name"""
    if raw_impact:
        raw = raw_impact.lower()
        if raw in ("high", "medium", "low"):
            return raw

    event_upper = event_name.upper()
    high_keywords = ["NFP", "NONFARM", "CPI", "INTEREST RATE", "FOMC", "GDP ",
                     "BOJ", "BOE", "ECB", "RBA", "EMPLOYMENT", "RETAIL SALES",
                     "PMI", "INFLATION"]
    medium_keywords = ["TRADE BALANCE", "HOUSING", "CONFIDENCE", "PPI",
                       "MANUFACTURING", "INDUSTRIAL"]

    if any(kw in event_upper for kw in high_keywords):
        return "high"
    elif any(kw in event_upper for kw in medium_keywords):
        return "medium"
    return "low"


# ========== Main ==========

def fetch_all(cfg: dict) -> dict:
    """Fetch from all sources and write to cache"""
    now = datetime.now(timezone.utc).isoformat()
    cache = load_existing_cache()

    finnhub_token = cfg.get("finnhub_token", "")
    av_token = cfg.get("alphavantage_token", "")

    # 1. Economic calendar (Finnhub + FF fallback)
    print("[1/3] Economic Calendar...")
    calendar = []
    if finnhub_token:
        calendar = fetch_finnhub_calendar(finnhub_token)
        print(f"  Finnhub calendar: {len(calendar)} events")
    if not calendar:
        calendar = fetch_forexfactory_calendar()
        print(f"  FF calendar (fallback): {len(calendar)} events")

    # 2. Headline news (Finnhub)
    print("[2/3] Headlines...")
    headlines = []
    if finnhub_token:
        headlines = fetch_finnhub_news(finnhub_token)
        print(f"  Finnhub headlines: {len(headlines)} articles")

    # 3. Sentiment (Alpha Vantage)
    print("[3/3] Sentiment...")
    sentiment = {"scores": {}, "articles": []}
    if av_token:
        sentiment = fetch_alphavantage_sentiment(av_token)
        print(f"  AV sentiment: {len(sentiment.get('scores', {}))} currencies, "
              f"{len(sentiment.get('articles', []))} articles")

    # Merge with existing headlines (deduplicate, keep only within last 24h)
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
    old_headlines = [h for h in cache.get("headlines", []) if h.get("time", "") > cutoff]
    old_headings = {h.get("headline", "") for h in old_headlines}
    new_headlines = [h for h in headlines if h.get("headline", "") not in old_headings]
    merged_headlines = sorted(
        old_headlines + new_headlines,
        key=lambda x: x.get("time", ""),
        reverse=True
    )[:60]  # Keep up to 60 items

    result = {
        "fetched_at": now,
        "sources": {
            "finnhub": bool(finnhub_token),
            "alphavantage": bool(av_token),
            "forexfactory": not bool(finnhub_token) or not calendar,
        },
        "calendar": calendar,
        "headlines": merged_headlines,
        "sentiment": sentiment,
    }

    # Write cache
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n→ Saved to {CACHE_FILE} ({len(json.dumps(result))} bytes)")
    return result


JST = timezone(timedelta(hours=9))


def _parse_event_time_utc(time_str: str | None) -> datetime | None:
    if not time_str:
        return None
    raw = str(time_str).strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _to_jst(time_str: str) -> str:
    """Convert ISO time string to JST display (HH:MM JST). Returns original[:16] on failure."""
    dt = _parse_event_time_utc(time_str)
    if dt is None:
        return ""
    return dt.astimezone(JST).strftime("%m/%d %H:%M JST")


def _event_countdown(time_str: str, now: datetime) -> str:
    """Calculate countdown string for an event. Returns '' if unparseable or past."""
    ev_dt = _parse_event_time_utc(time_str)
    if ev_dt is None:
        return ""
    diff = ev_dt - now
    minutes = diff.total_seconds() / 60
    if minutes < 0:
        return " [RELEASED]"
    if minutes < 60:
        return f" [in {int(minutes)}min]"
    if minutes < 1440:
        h = int(minutes // 60)
        m = int(minutes % 60)
        return f" [in {h}h{m:02d}m]"
    return f" [in {int(minutes // 1440)}d]"


def print_summary():
    """Display cache summary (for trader session)"""
    if not CACHE_FILE.exists():
        print("NEWS: No cache. Run python3 tools/news_fetcher.py to fetch.")
        return

    cache = json.loads(CACHE_FILE.read_text())
    fetched = cache.get("fetched_at", "unknown")

    # Freshness check
    try:
        fetched_dt = datetime.fromisoformat(fetched)
        age_min = (datetime.now(timezone.utc) - fetched_dt).total_seconds() / 60
        age_str = f"{age_min:.0f}m ago"
        if age_min > 60:
            age_str = f"⚠️ {age_min/60:.1f}h ago (stale)"
    except Exception:
        age_str = "unknown"

    print(f"📰 NEWS ({age_str})")

    # Calendar: today's high impact events with countdown
    now = datetime.now(timezone.utc)
    today_jst = now.astimezone(JST).date()
    high_events = []
    for event in cache.get("calendar", []):
        if event.get("impact") != "high":
            continue
        ev_dt = _parse_event_time_utc(event.get("time"))
        if ev_dt is None:
            continue
        if ev_dt.astimezone(JST).date() == today_jst:
            high_events.append((ev_dt, event))
    high_events.sort(key=lambda item: item[0])
    if high_events:
        print(f"🔴 HIGH IMPACT TODAY:")
        for _, ev in high_events[:5]:
            actual = f" → {ev['actual']}" if ev.get("actual") else ""
            forecast = f" (forecast:{ev['forecast']})" if ev.get("forecast") else ""
            countdown = _event_countdown(ev.get("time", ""), now)
            print(f"  {_to_jst(ev.get('time',''))} {ev['country']} {ev['event']}{forecast}{actual}{countdown}")
    else:
        upcoming_high = []
        for event in cache.get("calendar", []):
            if event.get("impact") != "high":
                continue
            ev_dt = _parse_event_time_utc(event.get("time"))
            if ev_dt is None or ev_dt < now:
                continue
            upcoming_high.append((ev_dt, event))
        upcoming_high.sort(key=lambda item: item[0])
        if upcoming_high:
            _, next_event = upcoming_high[0]
            countdown = _event_countdown(next_event.get("time", ""), now)
            print(f"📅 Next HIGH IMPACT: {next_event['country']} {next_event['event']} ({_to_jst(next_event.get('time',''))}){countdown}")
        else:
            print("📅 No upcoming high impact events")

    # Headlines: latest 5
    headlines = cache.get("headlines", [])
    if headlines:
        print(f"📢 Latest headlines ({len(headlines)} total, top 5):")
        for h in headlines[:5]:
            currencies = ",".join(h.get("related_currencies", []))
            time_jst = _to_jst(h.get("time", ""))
            print(f"  [{currencies}] {h['headline'][:80]} ({time_jst})")

    # Sentiment
    scores = cache.get("sentiment", {}).get("scores", {})
    if scores:
        sorted_scores = sorted(scores.items(), key=lambda x: x[1].get("score", 0), reverse=True)
        score_line = " | ".join(
            f"{ccy}:{data['score']:+.3f}({data['label']})" for ccy, data in sorted_scores
        )
        print(f"💭 Sentiment: {score_line}")


def main():
    args = sys.argv[1:]

    if "--summary" in args:
        print_summary()
        return

    cfg = load_config()

    if "--calendar" in args:
        token = cfg.get("finnhub_token", "")
        if token:
            events = fetch_finnhub_calendar(token)
        else:
            events = fetch_forexfactory_calendar()
        for e in events:
            print(json.dumps(e, ensure_ascii=False))
        return

    if "--headlines" in args:
        token = cfg.get("finnhub_token", "")
        if not token:
            print("finnhub_token not set in config/env.toml")
            return
        headlines = fetch_finnhub_news(token)
        for h in headlines:
            print(json.dumps(h, ensure_ascii=False))
        return

    if "--sentiment" in args:
        token = cfg.get("alphavantage_token", "")
        if not token:
            print("alphavantage_token not set in config/env.toml")
            return
        sentiment = fetch_alphavantage_sentiment(token)
        print(json.dumps(sentiment, indent=2, ensure_ascii=False))
        return

    # --if-stale N: re-fetch only if cache is older than N minutes
    if "--if-stale" in args:
        idx = args.index("--if-stale")
        stale_min = int(args[idx + 1]) if idx + 1 < len(args) else 15
        if CACHE_FILE.exists():
            try:
                cache = json.loads(CACHE_FILE.read_text())
                fetched_dt = datetime.fromisoformat(cache["fetched_at"])
                age = (datetime.now(timezone.utc) - fetched_dt).total_seconds() / 60
                if age < stale_min:
                    print(f"[news_fetcher: cache fresh ({age:.0f}m < {stale_min}m), skip]")
                    return
            except Exception:
                pass  # Cache is corrupted → re-fetch

    # Default: fetch all
    if not cfg.get("finnhub_token") and not cfg.get("alphavantage_token"):
        print("⚠️ API keys not set. Add the following to config/env.toml:")
        print('  finnhub_token = "YOUR_KEY"        # https://finnhub.io/register')
        print('  alphavantage_token = "YOUR_KEY"    # https://www.alphavantage.co/support/#api-key')
        print("\nRunning with Forex Factory calendar only...\n")

    t0 = time.time()
    fetch_all(cfg)
    print(f"[news_fetcher: {time.time()-t0:.1f}s]")


if __name__ == "__main__":
    main()
