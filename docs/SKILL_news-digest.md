# QuantRabbit News Digest Playbook

Purpose: refresh the ignored news artifacts that feed the market-story miner.
This task is observation only. It must not run `autotrade-cycle`, place orders,
call OANDA write gateways, or print secrets.

## Steps

1. Work in `/Users/tossaki/App/QuantRabbit`.
2. Fetch and normalize public news feeds:

```bash
PYTHONPATH=src python3 -m quant_rabbit.cli news-snapshot
```

3. Refresh the market-story profile without writing tracked reports:

```bash
PYTHONPATH=src python3 -m quant_rabbit.cli mine-market-stories \
  --news-dir logs \
  --profile data/market_story_profile.json \
  --report data/market_story_report.md
```

4. If `news-snapshot` reports `MISSING_*` or `STALE_*` issues, use browser/web
   fallback only to inspect source pages or feeds such as MarketPulse. Write
   short, sourced headline summaries to `logs/news_digest.md`; do not copy full
   article bodies. Then rerun step 3.

## Outputs

- `data/news_items.json`
- `logs/news_digest.md`
- `logs/news_flow_log.md`
- `data/market_story_profile.json`
- `data/market_story_report.md`

All outputs above are ignored runtime artifacts. Do not write
`docs/market_story_report.md` from this task.

## Report

Return a concise status with item count, issues, source URLs used, and the
artifact paths above.
