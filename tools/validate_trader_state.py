#!/usr/bin/env python3
"""
Validate trader state handoff before SESSION_END.

The goal is not to force specific market opinions. The goal is to block
prose-only flat books such as:
  - "armed mentally only"
  - "retest only"
  - "none placed"
  - flat-book summaries with zero real order receipts

Usage:
  python3 tools/validate_trader_state.py
  python3 tools/validate_trader_state.py /path/to/state.md
"""
from __future__ import annotations

import sys
from pathlib import Path

from record_s_hunt_ledger import STATE_PATH, _extract_section, _parse_simple_line, build_entry


REQUIRED_HORIZONS = ("Short-term S", "Medium-term S", "Long-term S")
BAD_PROSE = (
    "armed mentally",
    "retest only",
    "breakout only",
    "none placed",
)


def _has_id(value: str | None) -> bool:
    return bool(value and "id=" in value.lower())


def _is_dead(value: str | None) -> bool:
    return bool(value and "dead" in value.lower())


def _contains_bad_prose(value: str | None) -> str | None:
    if not value:
        return None
    lowered = value.lower()
    for phrase in BAD_PROSE:
        if phrase in lowered:
            return phrase
    return None


def validate_state(state_path: Path) -> list[str]:
    entry = build_entry(state_path)
    if entry is None:
        return [f"Could not parse state handoff from {state_path}"]

    text = state_path.read_text()
    capital_section = _extract_section(text, "Capital Deployment")
    flat_status = _parse_simple_line(capital_section, "Flat-book status") or ""
    live_now = _parse_simple_line(capital_section, "LIVE NOW") or ""
    reload_line = _parse_simple_line(capital_section, "RELOAD") or ""
    second_line = _parse_simple_line(capital_section, "SECOND SHOT / OTHER SIDE") or ""

    errors: list[str] = []
    horizons_by_name = {h.get("horizon"): h for h in entry.get("horizons", [])}
    horizon_deployment = entry.get("horizon_deployment", {})

    deployed_horizons = 0
    for horizon_name in REQUIRED_HORIZONS:
        horizon = horizons_by_name.get(horizon_name)
        if not horizon:
            errors.append(f"Missing `{horizon_name}` block in `## S Hunt`.")
            continue

        orderability = (horizon.get("orderability") or "").strip()
        deployment_result = (horizon.get("deployment_result") or "").strip()
        summary_line = (horizon_deployment.get(horizon_name.replace(" S", "")) or "").strip()

        if not orderability:
            errors.append(f"`{horizon_name}` is missing `Orderability:`.")
        if not deployment_result:
            errors.append(f"`{horizon_name}` is missing `Deployment result:`.")

        bad_phrase = _contains_bad_prose(deployment_result)
        if bad_phrase:
            errors.append(f"`{horizon_name}` deployment result still uses prose-only closure (`{bad_phrase}`).")

        if orderability:
            lowered = orderability.lower()
            if "still pass" in lowered:
                if not _is_dead(deployment_result):
                    errors.append(f"`{horizon_name}` is `STILL PASS` but does not close as a dead thesis.")
            else:
                if not _has_id(deployment_result):
                    errors.append(f"`{horizon_name}` is live/orderable but has no real `id=` in `Deployment result`.")

        if _has_id(deployment_result):
            deployed_horizons += 1

        if not summary_line:
            errors.append(f"`Capital Deployment` is missing the `{horizon_name.replace(' S', '')}` summary line.")
        else:
            bad_phrase = _contains_bad_prose(summary_line)
            if bad_phrase:
                errors.append(f"`Capital Deployment` `{horizon_name.replace(' S', '')}` line still uses prose-only closure (`{bad_phrase}`).")

            if _has_id(deployment_result) and not _has_id(summary_line):
                errors.append(f"`Capital Deployment` `{horizon_name.replace(' S', '')}` line drifted away from the real order receipt.")
            if _is_dead(deployment_result) and "dead" not in summary_line.lower():
                errors.append(f"`Capital Deployment` `{horizon_name.replace(' S', '')}` line must also close as dead thesis.")

    capital_has_id = any(_has_id(value) for value in (live_now, reload_line, second_line))
    is_flat = "flat" in flat_status.lower()
    if is_flat and deployed_horizons == 0 and not capital_has_id:
        errors.append("Flat book has zero real order IDs across `S Hunt` and `Capital Deployment`.")
    if is_flat and "acceptable flat book" in flat_status.lower() and deployed_horizons == 0 and not capital_has_id:
        errors.append("`acceptable flat book` is invalid when every horizon is still prose-only.")

    return errors


def main(argv: list[str]) -> int:
    state_path = Path(argv[1]).expanduser().resolve() if len(argv) > 1 else STATE_PATH
    errors = validate_state(state_path)
    if errors:
        print("STATE_VALIDATION_FAILED")
        for error in errors:
            print(f"- {error}")
        return 1
    print("STATE_VALIDATION_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
