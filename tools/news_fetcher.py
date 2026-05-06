#!/usr/bin/env python3
"""Fetch structured news cache for the hourly FX digest task.

This script is intentionally read-only with respect to trading. It collects the
configured public feed snapshot and writes compact source metadata to
``logs/news_cache.json`` for the scheduled Codex task to use while composing
``logs/news_digest.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quant_rabbit.analysis.news import DEFAULT_LOOKBACK_HOURS, DEFAULT_MAX_ITEMS, build_news_snapshot


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="news_fetcher.py")
    parser.add_argument("--output", type=Path, default=ROOT / "logs" / "news_cache.json")
    parser.add_argument("--lookback-hours", type=int, default=DEFAULT_LOOKBACK_HOURS)
    parser.add_argument("--max-items", type=int, default=DEFAULT_MAX_ITEMS)
    parser.add_argument("--no-fetch", action="store_true", help="Skip network fetch for tests.")
    args = parser.parse_args(argv)

    try:
        snapshot = build_news_snapshot(
            lookback_hours=args.lookback_hours,
            max_items=args.max_items,
            fetch=not args.no_fetch,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(snapshot.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        print(
            json.dumps(
                {
                    "output_path": str(args.output),
                    "items": len(snapshot.items),
                    "issues": list(snapshot.issues),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except Exception as exc:  # pragma: no cover - defensive scheduled-task boundary
        print(json.dumps({"error": str(exc), "output_path": str(args.output)}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
