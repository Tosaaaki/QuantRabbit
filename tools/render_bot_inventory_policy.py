#!/usr/bin/env python3
"""Render bot inventory policy markdown into machine-readable JSON."""

from __future__ import annotations

import argparse
import json
from datetime import timedelta
from pathlib import Path

from bot_policy import (
    POLICY_JSON,
    POLICY_MD,
    clamp_float,
    default_policy,
    format_policy_time,
    parse_policy_time,
    utc_now,
)


def write_default(ttl_min: int) -> int:
    policy = default_policy(utc_now(), ttl_min=ttl_min)
    policy["global_status"] = "REDUCE_ONLY"
    policy["notes"] = "Bootstrap policy. No new local-bot entries until trader or backup inventory-director writes the first live policy."
    lines = [
        "# Bot Inventory Policy",
        "",
        f"Updated: {policy['generated_at_human']}",
        f"Expires: {policy['expires_at_human']}",
        f"Global Status: {policy['global_status']}",
        f"Projected Margin Cap: {policy['projected_margin_cap']:.2f}",
        f"Panic Margin Cap: {policy['panic_margin_cap']:.2f}",
        f"Release Margin Cap: {policy['release_margin_cap']:.2f}",
        f"Max Pending Age Min: {policy['max_pending_age_min']}",
        f"Target Active Worker Pairs: {policy['target_active_worker_pairs']}",
        f"Notes: {policy['notes']}",
        "",
        "| Pair | Mode | Market | Pending | MaxPending | Ownership | Tempo | EntryBias | Note |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for pair in (
        "USD_JPY", "EUR_USD", "GBP_USD", "AUD_USD", "EUR_JPY", "GBP_JPY", "AUD_JPY",
        "NZD_USD", "USD_CAD", "USD_CHF", "EUR_GBP",
        "NZD_JPY", "CAD_JPY",
        "EUR_CHF", "AUD_NZD", "AUD_CAD",
    ):
        lines.append(
            f"| {pair} | BOTH | YES | KEEP | 1 | TRADER_ONLY | BALANCED | BALANCED | default active |"
        )
    POLICY_MD.parent.mkdir(parents=True, exist_ok=True)
    POLICY_MD.write_text("\n".join(lines) + "\n")
    POLICY_JSON.write_text(json.dumps(policy, indent=2) + "\n")
    print(f"WROTE default policy: {POLICY_MD}")
    print(f"WROTE default policy json: {POLICY_JSON}")
    return 0


def parse_markdown(path: Path) -> dict:
    text = path.read_text()
    header: dict[str, str] = {}
    rows: list[list[str]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("|") and line.endswith("|"):
            cols = [cell.strip() for cell in line.strip("|").split("|")]
            if cols and cols[0] != "---" and cols[0] != "Pair":
                rows.append(cols)
            continue
        if ":" in line and not line.startswith("#"):
            key, value = line.split(":", 1)
            header[key.strip().lower()] = value.strip()

    now = utc_now()
    updated = parse_policy_time(header.get("updated")) or now
    expires = parse_policy_time(header.get("expires")) or (updated + timedelta(minutes=30))

    policy = default_policy(updated, ttl_min=max(1, int((expires - updated).total_seconds() / 60)))
    policy["generated_at_human"] = format_policy_time(updated)
    policy["generated_at"] = updated.strftime("%Y-%m-%dT%H:%M:%SZ")
    policy["expires_at_human"] = format_policy_time(expires)
    policy["expires_at"] = expires.strftime("%Y-%m-%dT%H:%M:%SZ")
    policy["global_status"] = header.get("global status", policy["global_status"]).upper()
    policy["projected_margin_cap"] = clamp_float(
        header.get("projected margin cap"), policy["projected_margin_cap"], 0.50, 0.90
    )
    policy["panic_margin_cap"] = clamp_float(
        header.get("panic margin cap"), policy["panic_margin_cap"], 0.75, 0.95
    )
    policy["release_margin_cap"] = clamp_float(
        header.get("release margin cap"), policy["release_margin_cap"], 0.60, 0.90
    )
    policy["max_pending_age_min"] = int(
        clamp_float(header.get("max pending age min"), policy["max_pending_age_min"], 5, 120)
    )
    policy["target_active_worker_pairs"] = int(
        clamp_float(
            header.get("target active worker pairs"),
            policy["target_active_worker_pairs"],
            0,
            7,
        )
    )
    policy["notes"] = header.get("notes", policy["notes"])
    pairs: dict[str, dict] = {}
    for cols in rows:
        if len(cols) < 6:
            continue
        pair = cols[0].strip().upper()
        if not pair:
            continue
        ownership = "TRADER_ONLY"
        tempo = "BALANCED"
        entry_bias = "BALANCED"
        note = cols[5].strip()
        if len(cols) >= 9:
            ownership = cols[5].strip().upper() or "TRADER_ONLY"
            tempo = cols[6].strip().upper() or "BALANCED"
            entry_bias = cols[7].strip().upper() or "BALANCED"
            note = cols[8].strip()
        elif len(cols) >= 8:
            ownership = cols[5].strip().upper() or "TRADER_ONLY"
            tempo = cols[6].strip().upper() or "BALANCED"
            note = cols[7].strip()
        elif len(cols) >= 7:
            ownership = cols[5].strip().upper() or "TRADER_ONLY"
            note = cols[6].strip()
        pairs[pair] = {
            "mode": cols[1].strip().upper(),
            "allow_market": cols[2].strip().upper() in {"YES", "Y", "TRUE"},
            "pending": cols[3].strip().upper(),
            "max_pending": int(cols[4].strip() or 0),
            "ownership": ownership,
            "tempo": tempo,
            "entry_bias": entry_bias,
            "note": note,
        }
    policy["pairs"] = pairs
    return policy


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-md", type=Path, default=POLICY_MD)
    parser.add_argument("--to-json", type=Path, default=POLICY_JSON)
    parser.add_argument("--write-default", action="store_true")
    parser.add_argument("--ttl-min", type=int, default=30)
    args = parser.parse_args()

    if args.write_default:
        return write_default(args.ttl_min)

    policy = parse_markdown(args.from_md)
    args.to_json.parent.mkdir(parents=True, exist_ok=True)
    args.to_json.write_text(json.dumps(policy, indent=2) + "\n")
    print(f"RENDERED policy json: {args.to_json}")
    print(
        f"status={policy['global_status']} projected_cap={policy['projected_margin_cap']:.2f} "
        f"pairs={len(policy['pairs'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
