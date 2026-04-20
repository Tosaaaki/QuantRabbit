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

import re
import sys
from pathlib import Path

from record_s_hunt_ledger import STATE_PATH, _extract_section, _parse_simple_line, build_entry


REQUIRED_HORIZONS = ("Short-term S", "Medium-term S", "Long-term S")
REQUIRED_EXCAVATION_PAIRS = (
    "USD_JPY",
    "EUR_USD",
    "GBP_USD",
    "AUD_USD",
    "EUR_JPY",
    "GBP_JPY",
    "AUD_JPY",
)
EXCAVATION_REQUIRED_FROM = "2026-04-18"
PLACEHOLDER_TOKEN = "___"
PAIR_DIRECTION_RE = re.compile(r"\b([A-Z]{3}_[A-Z]{3})\s+(LONG|SHORT)\b")
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


def _extract_pipe_field(line: str, label: str) -> str | None:
    for part in line.split("|"):
        stripped = part.strip()
        for candidate in (stripped, stripped.split(":", 1)[1].strip() if ":" in stripped else None):
            if not candidate:
                continue
            if candidate.lower().startswith(label.lower()):
                value = candidate[len(label):].strip(" :")
                return value or None
    return None


def _extract_block_field(text: str, label: str, stop_labels: tuple[str, ...]) -> str | None:
    stop_expr = "|".join(re.escape(item) for item in stop_labels)
    pattern = rf"{re.escape(label)}\s*(.*?)(?=(?:{stop_expr})|$)"
    match = re.search(pattern, text, re.S)
    return match.group(1).strip(" .") if match else None


def _validate_s_excavation(text: str, session_date: str | None) -> list[str]:
    if not session_date or session_date < EXCAVATION_REQUIRED_FROM:
        return []

    section = _extract_section(text, "S Excavation Matrix")
    if not section:
        return [f"Missing `## S Excavation Matrix` for {EXCAVATION_REQUIRED_FROM}+ state handoffs."]

    lines = [line.strip() for line in section.splitlines() if line.strip()]
    errors: list[str] = []
    for pair in REQUIRED_EXCAVATION_PAIRS:
        pair_line = next((line for line in lines if line.startswith(f"{pair}:")), None)
        if not pair_line:
            errors.append(f"`S Excavation Matrix` is missing the `{pair}` line.")
            continue
        if PLACEHOLDER_TOKEN in pair_line:
            errors.append(f"`S Excavation Matrix` `{pair}` line still contains placeholder blanks.")
        lowered = pair_line.lower()
        best_expression = _extract_pipe_field(pair_line, "Best expression")
        why_not_s_now = _extract_pipe_field(pair_line, "Why not S now")
        upgrade_only_if = _extract_pipe_field(pair_line, "Upgrade to S only if")
        dead_if = _extract_pipe_field(pair_line, "Dead if")
        if "best expression" not in lowered:
            errors.append(f"`S Excavation Matrix` `{pair}` line is missing `Best expression`.")
        elif not best_expression:
            errors.append(f"`S Excavation Matrix` `{pair}` line must fill `Best expression`.")
        if "why not s now" not in lowered:
            errors.append(f"`S Excavation Matrix` `{pair}` line is missing `Why not S now`.")
        elif not why_not_s_now:
            errors.append(f"`S Excavation Matrix` `{pair}` line must fill `Why not S now`.")
        if "upgrade to s" not in lowered:
            errors.append(f"`S Excavation Matrix` `{pair}` line is missing `Upgrade to S`.")
        elif not upgrade_only_if:
            errors.append(f"`S Excavation Matrix` `{pair}` line must fill `Upgrade to S only if`.")
        if "dead if" not in lowered:
            errors.append(f"`S Excavation Matrix` `{pair}` line is missing `Dead if`.")
        elif not dead_if:
            errors.append(f"`S Excavation Matrix` `{pair}` line must fill `Dead if`.")

    for idx in range(1, 4):
        podium_line = next((line for line in lines if line.lower().startswith(f"podium #{idx}:")), None)
        if not podium_line:
            errors.append(f"`S Excavation Matrix` is missing `Podium #{idx}`.")
            continue
        if PLACEHOLDER_TOKEN in podium_line:
            errors.append(f"`S Excavation Matrix` `Podium #{idx}` still contains placeholder blanks.")
        if not PAIR_DIRECTION_RE.search(podium_line):
            errors.append(f"`S Excavation Matrix` `Podium #{idx}` must name a concrete `PAIR LONG/SHORT` seat.")
        reason = _extract_pipe_field(podium_line, "Closest-to-S because")
        blocker = _extract_pipe_field(podium_line, "Still blocked by")
        upgrade_action = _extract_pipe_field(podium_line, "If it upgrades")
        if "closest-to-s because" not in podium_line.lower():
            errors.append(f"`S Excavation Matrix` `Podium #{idx}` is missing `Closest-to-S because`.")
        elif not reason:
            errors.append(f"`S Excavation Matrix` `Podium #{idx}` must fill `Closest-to-S because`.")
        if "still blocked by" not in podium_line.lower():
            errors.append(f"`S Excavation Matrix` `Podium #{idx}` is missing `Still blocked by`.")
        elif not blocker:
            errors.append(f"`S Excavation Matrix` `Podium #{idx}` must fill `Still blocked by`.")
        if not upgrade_action:
            errors.append(f"`S Excavation Matrix` `Podium #{idx}` must fill `If it upgrades`.")
        elif upgrade_action.upper() not in {"MARKET", "LIMIT", "STOP-ENTRY"}:
            errors.append(f"`S Excavation Matrix` `Podium #{idx}` must name the upgrade action (`MARKET / LIMIT / STOP-ENTRY`).")

    return errors


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
    errors.extend(_validate_s_excavation(text, entry.get("session_date")))
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
        raw_text = (horizon.get("raw") or "").strip()
        promotion_proof = _extract_block_field(
            raw_text,
            "Promotion proof:",
            ("MTF chain:", "Payout path:", "Orderability:", "If not live:", "Deployment result:"),
        )

        if not orderability:
            errors.append(f"`{horizon_name}` is missing `Orderability:`.")
        if not deployment_result:
            errors.append(f"`{horizon_name}` is missing `Deployment result:`.")
        if not promotion_proof:
            errors.append(f"`{horizon_name}` is missing `Promotion proof:`.")

        bad_phrase = _contains_bad_prose(deployment_result)
        if bad_phrase:
            errors.append(f"`{horizon_name}` deployment result still uses prose-only closure (`{bad_phrase}`).")

        if orderability:
            lowered = orderability.lower()
            if "still pass" in lowered:
                if not _is_dead(deployment_result):
                    errors.append(f"`{horizon_name}` is `STILL PASS` but does not close as a dead thesis.")
                if "no seat cleared promotion gate" not in (promotion_proof or "").lower():
                    errors.append(
                        f"`{horizon_name}` is `STILL PASS` so `Promotion proof` must say `none — no seat cleared promotion gate`."
                    )
                if "no seat cleared promotion gate" not in deployment_result.lower():
                    errors.append(
                        f"`{horizon_name}` is `STILL PASS` so `Deployment result` must close as `dead thesis because no seat cleared promotion gate: ...`."
                    )
            else:
                if not _has_id(deployment_result):
                    errors.append(f"`{horizon_name}` is live/orderable but has no real `id=` in `Deployment result`.")
                lowered_proof = (promotion_proof or "").lower()
                if lowered_proof.startswith("none") or "no seat cleared promotion gate" in lowered_proof:
                    errors.append(f"`{horizon_name}` is promoted live/orderable but `Promotion proof` still says no seat cleared.")
                if "blocker was" not in lowered_proof or "cleared by" not in lowered_proof:
                    errors.append(
                        f"`{horizon_name}` `Promotion proof` must state `blocker was ... -> cleared by ...`."
                    )

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
