#!/usr/bin/env python3
"""
news_flow_append.py — Hourly news flow snapshot

Reads logs/news_digest.md and extracts:
  - HOT:   Top high-impact headline (first bullet under High Impact)
  - THEME: Narrative summary (first watch-list item or top theme)
  - WATCH: Top watch-list item

Appends a timestamped entry to logs/news_flow_log.md
Trims log to keep the most recent 48 entries (48h of hourly snapshots)
"""

import re
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

# JST = UTC+9
JST = timezone(timedelta(hours=9))

BASE_DIR = Path(__file__).parent.parent
DIGEST_PATH = BASE_DIR / "logs" / "news_digest.md"
FLOW_LOG_PATH = BASE_DIR / "logs" / "news_flow_log.md"
MAX_ENTRIES = 48

ENTRY_SEPARATOR = "---ENTRY---"


def read_digest(path: Path) -> str:
    if not path.exists():
        print(f"[ERROR] Digest not found: {path}", file=sys.stderr)
        sys.exit(1)
    return path.read_text(encoding="utf-8")


def _find_section(text: str, keyword: str) -> str:
    """Return the body of the first ## section whose header line contains keyword.
    Uses MULTILINE so ^ anchors to each line start, avoiding cross-line greedy matches."""
    # Match the section header line, then capture body until next ## header or end
    pattern = rf"^##[^\n]*{re.escape(keyword)}[^\n]*\n(.*?)(?=\n^##|\Z)"
    m = re.search(pattern, text, re.DOTALL | re.MULTILINE | re.IGNORECASE)
    if m:
        return m.group(1)
    return ""


def extract_hot(text: str) -> str:
    """Extract first bold headline from High Impact section."""
    section = _find_section(text, "High Impact")
    if not section:
        return "N/A"
    bold = re.search(r"\*\*(.+?)\*\*", section)
    if bold:
        return bold.group(1).strip()
    for line in section.splitlines():
        line = line.strip().lstrip("-• ")
        if line:
            return line[:120]
    return "N/A"


def extract_theme(text: str) -> str:
    """Derive macro narrative theme from High Impact content (first 2 bold items)."""
    section = _find_section(text, "High Impact")
    if not section:
        return "N/A"
    bolds = re.findall(r"\*\*(.+?)\*\*", section)
    if bolds:
        return " / ".join(b.strip() for b in bolds[:2])
    return "N/A"


def extract_watch(text: str) -> str:
    """Extract first item from Watch List section."""
    section = _find_section(text, "Watch")
    if not section:
        return "N/A"
    # First bullet line with bold: - **PAIR**: detail
    line_m = re.search(r"\*\*(.+?)\*\*[:\s]*(.*)", section)
    if line_m:
        label = line_m.group(1).strip()
        detail = line_m.group(2).strip()[:80]
        return f"{label}: {detail}" if detail else label
    # Fallback: first non-empty bullet line
    for line in section.splitlines():
        line = line.strip().lstrip("-•* ")
        if line:
            return line[:120]
    return "N/A"


def load_existing_entries(path: Path) -> list[str]:
    if not path.exists():
        return []
    content = path.read_text(encoding="utf-8")
    # Split by separator, filter empties
    parts = content.split(ENTRY_SEPARATOR)
    return [p.strip() for p in parts if p.strip()]


def build_entry(now: datetime, hot: str, theme: str, watch: str) -> str:
    ts = now.strftime("%Y-%m-%d %H:%M JST")
    return f"### {ts}\n- HOT: {hot}\n- THEME: {theme}\n- WATCH: {watch}"


def write_log(path: Path, entries: list[str]) -> None:
    content = f"\n{ENTRY_SEPARATOR}\n".join(entries) + f"\n{ENTRY_SEPARATOR}\n"
    path.write_text(content, encoding="utf-8")


def main():
    now = datetime.now(JST)
    digest_text = read_digest(DIGEST_PATH)

    hot = extract_hot(digest_text)
    theme = extract_theme(digest_text)
    watch = extract_watch(digest_text)

    new_entry = build_entry(now, hot, theme, watch)

    entries = load_existing_entries(FLOW_LOG_PATH)
    entries.append(new_entry)

    # Trim to MAX_ENTRIES (keep newest)
    if len(entries) > MAX_ENTRIES:
        entries = entries[-MAX_ENTRIES:]

    write_log(FLOW_LOG_PATH, entries)

    print(f"[OK] Appended entry for {now.strftime('%Y-%m-%d %H:%M JST')}")
    print(f"     HOT:   {hot}")
    print(f"     THEME: {theme}")
    print(f"     WATCH: {watch}")
    print(f"     Total entries in log: {len(entries)}")


if __name__ == "__main__":
    main()
