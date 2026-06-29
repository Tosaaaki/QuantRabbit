# Trade Case Studies

This directory is the Git-backed record for detailed trade judgment history.
Use it for cases that should be reviewable by another trader, engineer, or
strategy reviewer.

## Storage Rule

- Git is the source of truth for detailed case studies: timeline, broker truth,
  operator thesis, bot thesis, outcome, and repair requests.
- Notion should mirror the summary only: title, tags, short lesson, and a link
  to the Git file.
- OANDA transaction truth remains in `data/execution_ledger.db`; this directory
  is for human-readable judgment and improvement context.

## Case Template

Each case should include:

- UTC/JST timeline.
- Instruments, units, entry, TP, SL, realized/unrealized P/L.
- Operator thesis before the result was known.
- Bot/system thesis before the result was known.
- What happened next.
- What the system missed.
- Concrete code or policy improvement requests.

