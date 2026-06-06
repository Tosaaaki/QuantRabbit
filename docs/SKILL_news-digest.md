# qr-news-digest — FX News Digest

Purpose: collect FX-relevant news via WebSearch and API parser, then write a
trader-perspective summary to `logs/news_digest.md`. The trader session reads
this file at the start of each cycle to build macro context into trade theses.
This task is observation only: do not run `autotrade-cycle`, place orders, call
OANDA write gateways, print secrets, or write tracked `docs/*_report.md` files.

## Working Directory

`/Users/tossaki/App/QuantRabbit-live`

## Steps

1. WebSearch x3 — collect news from three angles:
   - Breaking FX news: `"forex" OR "FX" OR "USD JPY EUR GBP" site:reuters.com OR site:bloomberg.com OR site:forexlive.com latest`
   - Central bank actions: `"Fed" OR "BOJ" OR "ECB" OR "RBA" interest rate policy statement 2026`
   - Economic calendar events today: `economic calendar today "NFP" OR "CPI" OR "GDP" OR "FOMC" OR "BOJ" 2026`
   - For every high/medium event in the next 48h, explicitly search and
     summarize the pre-release evidence stack:
     - Labor: `ADP employment`, `initial jobless claims`, `JOLTS job openings`,
       `ISM employment`, `Challenger layoffs`, `average hourly earnings`,
       current `NFP consensus`
     - Inflation: `CPI`, `core CPI`, `PCE`, `PPI`, `wages`, `oil`, inflation
       expectations, consensus vs prior
     - Central banks: `FOMC/ECB/BOJ/BOE/RBA/BOC/RBNZ/SNB`, rate decision,
       meeting minutes, speeches, `hawkish/dovish`, yields, intervention risk
     - Growth/consumption: `GDP`, `PMI`, `ISM`, `retail sales`, consumer
       confidence/sentiment, industrial production, durable goods
     - Trade/commodity/risk: trade balance, current account, oil/WTI/Brent,
       tariffs, sanctions, geopolitics, risk-on/off
     The trader uses this as directional nowcast evidence only; it does not
     override spread, RR, chart, or gateway checks.

2. Run the API parser. This may fail if a future provider requires keys; continue
   with WebSearch results if it fails.

```bash
cd /Users/tossaki/App/QuantRabbit-live && PYTHONPATH=src python3 tools/news_fetcher.py
```

3. Also refresh the deterministic RSS artifacts. This writes `data/news_items.json`,
   `logs/news_digest.md`, and `logs/news_flow_log.md`; overwrite the digest in
   step 4 with the trader-perspective WebSearch summary.

```bash
PYTHONPATH=src python3 -m quant_rabbit.cli news-snapshot
```

4. Write `logs/news_digest.md` manually from a pro trader's perspective. Keep it
   under 60 lines total. Prioritize what changes trade thesis, entries, avoid
   zones, and what the market is pricing in. Do not summarize everything.

```markdown
# FX News Digest — YYYY-MM-DD HH:MM JST

## 🔴 High Impact (act on this)
[Events/news that could move markets in the next 1-4 hours]

## 🟡 Watch List
[Developing stories, scheduled releases today]

## 📅 Economic Calendar Today (JST)
[Key releases with consensus vs prior]

## 🧭 Pre-Event Nowcast
[For every high/medium event in 48h: whether leading evidence leans stronger/weaker than consensus, currency impact, and sources]

## 🏦 Central Bank Tracker
[Latest from Fed/BOJ/ECB/RBA — stance, recent statements]

## 💱 Pair-Specific Notes
[USD_JPY / EUR_USD / GBP_USD / AUD_USD — any pair-specific drivers]

## ⚠️ Risk Events (48h)
[Upcoming scheduled risks]

---
Updated: YYYY-MM-DD HH:MM JST | Sources: WebSearch + news_fetcher.py
```

5. Refresh the market-story profile without writing tracked reports:

```bash
PYTHONPATH=src python3 -m quant_rabbit.cli mine-market-stories \
  --news-dir logs \
  --profile data/market_story_profile.json \
  --report data/market_story_report.md
```

6. Audit the full news path. This must return no `BLOCK` before reporting
   success: the check verifies freshness, required WebSearch digest sections,
   structured source diversity, qr-news-digest automation state, and that
   `data/market_story_profile.json` is newer than the news artifacts. Report
   any `WARN` lines instead of hiding them.

```bash
PYTHONPATH=src python3 -m quant_rabbit.cli news-health --verify-fetch --strict
```

## Notes

- If WebSearch returns no useful results, write a minimal digest noting low news
  flow.
- JST = UTC+9. Always display times in JST.
- Keep it concise; the trader reads this in 10 seconds.
- Use browser inspection only as fallback for source pages or feeds. Do not copy
  full article bodies; use short sourced summaries and links.
