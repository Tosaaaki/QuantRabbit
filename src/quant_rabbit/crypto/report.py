from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def scan_markdown(
    scan: dict[str, Any],
    metrics: dict[str, Any] | None = None,
    canary: dict[str, Any] | None = None,
) -> str:
    counts = scan["counts"]
    guardian = scan["guardian"]
    lines = [
        "# QuantRabbit Crypto bitbank canary",
        "",
        f"- Observed: {scan['observed_at_utc']}",
        f"- Mode: {scan['mode']}",
        f"- Guardian: {guardian['state']}",
        f"- JPY pairs: {counts['discovered_jpy_pairs']}",
        f"- Eligible: {counts['eligible_pairs']}",
        f"- Candidates: {counts['candidate_pairs']}",
        "- Live/private mutation: disabled",
        "",
        "## Candidates",
        "",
    ]
    if not scan["candidates"]:
        lines.append("- No candidate cleared the conservative safety buffer.")
    for item in scan["candidates"]:
        lines.append(
            f"- `{item['pair']}` net={item['net_edge_bps']}bps "
            f"spread={item['spread_bps']}bps depth25={item['depth_25bps_jpy']}JPY"
        )
    lines.extend(["", "## Rejections", ""])
    for item in scan["pairs"][:20]:
        if not item["candidate"]:
            reasons = ", ".join(item["reasons"]) or "NOT_SHORTLISTED"
            lines.append(f"- `{item['pair']}`: {reasons}")
    if metrics is not None:
        lines.extend(
            [
                "",
                "## Paper KPI",
                "",
                f"- Net PnL: {metrics['net_pnl_jpy']} JPY",
                f"- Max DD: {metrics['max_drawdown_jpy']} JPY",
                f"- Fills: {metrics['trade_count']}",
                f"- Fees: {metrics['fees_jpy']} JPY",
                f"- Spread cost: {metrics['spread_cost_jpy']} JPY",
                f"- Slippage cost: {metrics['slippage_cost_jpy']} JPY",
                f"- Discipline violations: {metrics['discipline_violations']}",
            ]
        )
    if canary is not None:
        stream = canary["public_stream"]
        integrity = canary["ledger_integrity"]
        keychain = canary["private_rest"]["keychain"]
        services = ", ".join(
            f"`{entry['service']}`" for entry in keychain["entries"]
        )
        present = " / ".join(
            f"`{str(entry['present']).lower()}`"
            for entry in keychain["entries"]
        )
        lines.extend(
            [
                "",
                "## Canary evidence",
                "",
                f"- REST cycles: {len(canary['cycles'])}",
                f"- Public Stream: {'PASS' if stream['ok'] else 'FAIL'}",
                f"- Public Stream messages: {stream['message_count']}",
                f"- Ledger events: {integrity['event_count']}",
                f"- Ledger hash chain: {'valid' if integrity['valid'] else 'invalid'}",
                "- Private REST: BLOCKED (rotated Keychain credential absent)",
                f"- Keychain services: {services}",
                f"- Keychain account: `{keychain['account']}`",
                f"- Keychain present: {present}",
                "",
                "No market cleared the positive-momentum and net-edge buffer in "
                "this window, so zero virtual orders is the expected disciplined "
                "result rather than an execution failure.",
            ]
        )
    return "\n".join(lines) + "\n"


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
