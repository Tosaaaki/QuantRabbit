#!/usr/bin/env python3
"""Screen (pair, horizon) cells for arithmetic feasibility before any research runs.

    tools/feasibility_gate.py path/to/evaluation_rows.jsonl
    tools/feasibility_gate.py rows.jsonl --min-ceiling 1.0 --json out.json

Reads only the long/short executable returns, so a cell can never be admitted
because a strategy happened to win in it. Exits non-zero when nothing is
feasible, so it can sit in front of a research run as a gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quant_rabbit.feasibility import SIGNAL_COLUMNS, screen  # noqa: E402

KEEP = ("pair", "horizon_minutes", "long_executable_return_pips", "short_executable_return_pips")


def read_rows(path: Path):
    """Yield records reduced to the four admissible columns.

    The projection happens here, at the boundary, so a corpus that carries
    signal columns can still be screened without those columns ever reaching
    the screen.
    """

    with path.open(encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                yield {key: record.get(key) for key in KEEP}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("rows", type=Path, help="JSONL with long/short executable returns per bar")
    parser.add_argument("--min-ceiling", type=float, default=0.0, help="pips a perfect predictor must clear (default 0)")
    parser.add_argument("--min-rows", type=int, default=30, help="minimum bars per cell (default 30)")
    parser.add_argument("--json", type=Path, default=None, help="also write the full report here")
    args = parser.parse_args(argv)

    if not args.rows.exists():
        print(f"no such file: {args.rows}", file=sys.stderr)
        return 2

    report = screen(read_rows(args.rows), min_ceiling_pips=args.min_ceiling, min_rows=args.min_rows)
    if not report["cells"]:
        print("no cell had enough price-true rows to screen", file=sys.stderr)
        return 2

    print(f"rows used {report['rows_used']}  cells {report['cells']}  "
          f"admitted {report['admitted']}  closed {report['closed']}  "
          f"(ceiling >= {args.min_ceiling} pips)\n")
    print(f"{'pair':10} {'horizon':>7} {'n':>5} {'|move|':>7} {'cost':>6} {'ceiling':>8} {'±se':>6}  verdict")
    for cell in report["cells_detail"]:
        print(
            f"{cell['pair']:10} {cell['horizon_minutes']:>7} {cell['rows']:>5} "
            f"{cell['mean_abs_move_pips']:>7.2f} {cell['mean_cost_pips']:>6.2f} "
            f"{cell['mean_oracle_pips']:>8.2f} {cell['oracle_stderr_pips']:>6.2f}  {cell['verdict']}"
        )
    print(f"\n{report['interpretation']}")

    if args.json:
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json}")

    # Non-zero when the gate admits nothing, so a research run can depend on it.
    return 0 if report["admitted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
