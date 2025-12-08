#!/usr/bin/env python3
"""
Summarize strategy skip reasons from logs that contain
`[STRAT_SKIP_DETAIL] <Strategy> reason=<reason> ...`.

Usage:
    python scripts/skip_reason_summary.py --log path/to/journal.log
    journalctl -u quantrabbit.service --since "2 hours ago" \
        | python scripts/skip_reason_summary.py
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from typing import Iterable, TextIO


PATTERN = re.compile(r"\[STRAT_SKIP_DETAIL\]\s+(\w+)\s+reason=([^\s]+)")


def iter_lines(handles: Iterable[TextIO]) -> Iterable[str]:
    for fh in handles:
        for line in fh:
            yield line.rstrip("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize strategy skip reasons")
    parser.add_argument(
        "--log",
        action="append",
        dest="logs",
        help="Log file to parse (reads stdin if omitted)",
    )
    args = parser.parse_args()

    handles = []
    if args.logs:
        for path in args.logs:
            try:
                handles.append(open(path, "r", encoding="utf-8", errors="ignore"))
            except OSError as exc:
                sys.stderr.write(f"warn: failed to open {path}: {exc}\n")
    else:
        handles.append(sys.stdin)

    counts: dict[str, Counter] = defaultdict(Counter)
    for line in iter_lines(handles):
        m = PATTERN.search(line)
        if not m:
            continue
        strat, reason = m.group(1), m.group(2)
        counts[strat][reason] += 1

    for h in handles:
        if h is not sys.stdin:
            h.close()

    if not counts:
        print("no skip entries found")
        return

    for strat, counter in sorted(counts.items()):
        total = sum(counter.values())
        print(f"{strat}: total={total}")
        for reason, cnt in counter.most_common():
            pct = (cnt / total) * 100
            print(f"  {reason:<28} {cnt:5d} ({pct:4.1f}%)")


if __name__ == "__main__":
    main()
