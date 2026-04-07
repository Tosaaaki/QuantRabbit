#!/usr/bin/env python3
"""
Quality Audit — cross-checks trader decisions against rules and S-conviction data.

Designed to run as a separate scheduled task (Sonnet, every 30 min).
Reads state.md, live_trade_log, S-conviction scan, OANDA positions/margin,
and flags inconsistencies.

Output: logs/quality_audit.md (overwritten each run). Only writes when issues found.
Exit code: 0 = clean, 1 = issues found (for task to decide whether to Slack).
"""
import json
import re
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV_PYTHON = str(ROOT / ".venv" / "bin" / "python")
PAIRS = ["USD_JPY", "EUR_USD", "GBP_USD", "AUD_USD", "EUR_JPY", "GBP_JPY", "AUD_JPY"]

# Normal spreads per pair (pip)
NORMAL_SPREAD = {
    "USD_JPY": 1.2, "EUR_USD": 1.2, "GBP_USD": 1.8,
    "AUD_USD": 1.8, "EUR_JPY": 2.5, "GBP_JPY": 3.5, "AUD_JPY": 2.0,
}


def load_config():
    cfg = {}
    for line in open(ROOT / "config" / "env.toml"):
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            cfg[k.strip()] = v.strip().strip('"')
    return cfg


def oanda_api(path, token, acct):
    url = f"https://api-fxtrade.oanda.com{path}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    return json.loads(urllib.request.urlopen(req, timeout=10).read())


def run_script(args, timeout=30):
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=timeout, cwd=str(ROOT))
        return r.stdout.strip()
    except Exception as e:
        return f"(error: {e})"


# ── Check 1: S-candidates missed ──

def check_s_candidates_missed(state_text: str) -> list[str]:
    """Run S-conviction scan and compare with state.md Pass reasons."""
    issues = []

    # Run S-scan
    scan_out = run_script([VENV_PYTHON, "tools/s_conviction_scan.py"])
    if "no S-candidates" in scan_out or "error" in scan_out:
        return []

    # Parse S-candidates: "🎯 PAIR DIR Recipe: details"
    s_candidates = []
    for line in scan_out.split("\n"):
        if line.startswith("🎯"):
            m = re.match(r"🎯 (\w+) (LONG|SHORT) (\S+):", line)
            if m:
                s_candidates.append({
                    "pair": m.group(1),
                    "direction": m.group(2),
                    "recipe": m.group(3),
                    "full": line,
                })

    if not s_candidates:
        return []

    # Parse state.md for held pairs and Pass reasons
    held_pairs = set()
    pass_reasons = {}  # pair -> reason text

    # Find held pairs
    for m in re.finditer(r"### (\w+_\w+) (?:LONG|SHORT) \(id=", state_text):
        held_pairs.add(m.group(1))

    # Find Tier 2 scan entries with "Pass" or pass-like reasons
    for line in state_text.split("\n"):
        for pair in PAIRS:
            if pair in line and ("Pass" in line or "pass" in line or "Skip" in line):
                pass_reasons[pair] = line.strip()

    # Cross-check: S-candidate exists but pair is not held and has a Pass reason
    for sc in s_candidates:
        pair = sc["pair"]
        if pair in held_pairs:
            continue  # Already holding — check sizing instead
        if pair in pass_reasons:
            reason = pass_reasons[pair]
            issues.append(
                f"**S-CANDIDATE MISSED**: {sc['full']}\n"
                f"  Trader Pass reason: {reason}\n"
                f"  → Is this Pass reason valid? S-scan says enter."
            )
        elif pair not in held_pairs:
            # Not held, no explicit Pass — just not entered
            issues.append(
                f"**S-CANDIDATE NOT ENTERED**: {sc['full']}\n"
                f"  No position held. No Pass reason found in state.md."
            )

    return issues


# ── Check 2: Sizing discipline ──

def check_sizing(trades: list, nav: float) -> list[str]:
    """Check if recent entries match conviction sizing rules."""
    issues = []
    log_path = ROOT / "logs" / "live_trade_log.txt"
    if not log_path.exists():
        return []

    lines = log_path.read_text().strip().split("\n")
    # Look at last 20 entries from today
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    recent_entries = []
    for line in lines[-30:]:
        if "ENTRY" in line and today in line:
            recent_entries.append(line)

    for entry in recent_entries:
        # Parse conviction and units
        conv_match = re.search(r"pretrade=(\w+)\((\w)(\d*)\)", entry)
        units_match = re.search(r"(\d+)u", entry)
        if not conv_match or not units_match:
            continue

        conviction_letter = conv_match.group(2) if conv_match.group(2) in "SABC" else None
        if not conviction_letter:
            # Try format like "pretrade=S(9)"
            conv_match2 = re.search(r"pretrade=([SABC])\((\d+)\)", entry)
            if conv_match2:
                conviction_letter = conv_match2.group(1)
        if not conviction_letter:
            continue

        units = int(units_match.group(1))

        # Expected sizing
        expected_pct = {"S": 0.30, "A": 0.15, "B": 0.05, "C": 0.02}
        pct = expected_pct.get(conviction_letter, 0.05)
        # Rough expected units (assume avg price ~150 for JPY pairs)
        expected_margin = nav * pct
        # Don't need exact units — just check if drastically undersized
        min_expected = expected_margin * 0.4  # allow 60% tolerance

        # Extract pair for price estimation
        pair_match = re.search(r"(USD_JPY|EUR_USD|GBP_USD|AUD_USD|EUR_JPY|GBP_JPY|AUD_JPY)", entry)
        if not pair_match:
            continue
        pair = pair_match.group(1)
        # Estimate margin per unit
        price_est = 150 if "JPY" in pair else 1.2
        leverage = 25
        margin_per_unit = price_est / leverage
        actual_margin = units * margin_per_unit
        actual_pct = actual_margin / nav if nav > 0 else 0

        if conviction_letter == "S" and actual_pct < 0.15:
            issues.append(
                f"**UNDERSIZED S-ENTRY**: {entry[:120]}...\n"
                f"  Conviction=S → expected ~30% NAV (~{int(expected_margin)}JPY margin).\n"
                f"  Actual: {units}u = ~{actual_pct*100:.0f}% NAV ({int(actual_margin)}JPY margin).\n"
                f"  → Double-discount? Historical WR fear? S = S-size."
            )
        elif conviction_letter == "A" and actual_pct < 0.07:
            issues.append(
                f"**UNDERSIZED A-ENTRY**: {entry[:120]}...\n"
                f"  Conviction=A → expected ~15% NAV. Actual: {units}u = ~{actual_pct*100:.0f}% NAV."
            )

    # Check minimum 2000u rule
    for entry in recent_entries:
        units_match = re.search(r"(\d+)u", entry)
        if units_match and int(units_match.group(1)) < 2000:
            issues.append(
                f"**BELOW MINIMUM 2000u**: {entry[:120]}...\n"
                f"  Units={units_match.group(1)}u. Minimum is 2000u. Below that, spread eats the profit."
            )

    return issues


# ── Check 3: Margin utilization ──

def check_margin(nav: float, margin_used: float, s_candidate_count: int, position_count: int) -> list[str]:
    """Flag idle margin when opportunities exist."""
    issues = []
    margin_pct = (margin_used / nav * 100) if nav > 0 else 0
    idle_pct = 100 - margin_pct

    if s_candidate_count >= 3 and idle_pct > 60 and position_count <= 1:
        issues.append(
            f"**MARGIN IDLE WITH S-CANDIDATES**: {s_candidate_count} S-candidates detected.\n"
            f"  Margin: {margin_pct:.0f}% used, {idle_pct:.0f}% idle. {position_count} position(s).\n"
            f"  → {s_candidate_count} S-setups × 30% NAV each. Why is 60%+ sitting idle?"
        )
    elif s_candidate_count >= 1 and idle_pct > 80 and position_count == 0:
        issues.append(
            f"**FLAT WITH S-CANDIDATES**: {s_candidate_count} S-candidate(s) but 0 positions.\n"
            f"  → Market is moving. S-conviction scan found setups. Enter or explain."
        )

    return issues


# ── Check 4: Rule misapplication ──

def check_rule_misapplication(state_text: str) -> list[str]:
    """Detect known patterns of rule misapplication."""
    issues = []

    # Circuit breaker applied to wrong direction
    # Pattern: "circuit breaker" mentioned for a pair, but S-candidate is opposite direction
    cb_pairs = {}
    for line in state_text.split("\n"):
        cb_match = re.search(r"(\w+_\w+).*circuit.?breaker", line, re.IGNORECASE)
        if cb_match:
            pair = cb_match.group(1)
            # Try to find which direction the breaker is for
            if "SHORT" in line.upper() and "LONG" not in line.split("circuit")[0].upper():
                cb_pairs[pair] = "SHORT"
            elif "LONG" in line.upper():
                cb_pairs[pair] = "LONG"
            else:
                cb_pairs[pair] = "UNKNOWN"

    # Check if circuit breaker is blocking the opposite direction
    scan_out = run_script([VENV_PYTHON, "tools/s_conviction_scan.py"])
    for pair, blocked_dir in cb_pairs.items():
        if blocked_dir == "UNKNOWN":
            continue
        opposite = "LONG" if blocked_dir == "SHORT" else "SHORT"
        # Check if S-scan found opposite direction for this pair
        pattern = f"🎯 {pair} {opposite}"
        if pattern in scan_out:
            # Check if state.md blocks it
            for line in state_text.split("\n"):
                if pair in line and "circuit" in line.lower() and ("Pass" in line or "pass" in line):
                    issues.append(
                        f"**CIRCUIT BREAKER MISAPPLIED**: {pair} {blocked_dir} circuit breaker\n"
                        f"  is blocking {opposite} entry. S-scan: {pattern} detected.\n"
                        f"  Rule: circuit breaker is DIRECTION-ONLY. {blocked_dir} losses don't affect {opposite}."
                    )

    # Spread excuse on normal spread
    for line in state_text.split("\n"):
        spread_match = re.search(
            r"(\w+_\w+).*[Ss]pread[=:]?\s*([\d.]+)\s*pip.*(?:wide|⚠|too)", line
        )
        if spread_match:
            pair = spread_match.group(1)
            spread_val = float(spread_match.group(2))
            normal = NORMAL_SPREAD.get(pair, 5.0)
            if spread_val <= normal:
                issues.append(
                    f"**NORMAL SPREAD CALLED WIDE**: {pair} Sp={spread_val}pip flagged as wide.\n"
                    f"  Normal range for {pair}: up to {normal}pip.\n"
                    f"  → This is the cost of doing business, not a reason to pass."
                )

    # "Thin market" as blanket entry blocker
    thin_count = 0
    thin_lines = []
    for line in state_text.split("\n"):
        if re.search(r"thin|holiday|Easter|liquidity", line, re.IGNORECASE):
            if re.search(r"Pass|no entry|no new|skip", line, re.IGNORECASE):
                thin_count += 1
                thin_lines.append(line.strip()[:100])

    if thin_count >= 3:
        issues.append(
            f"**THIN MARKET BLANKET BLOCK**: {thin_count} pairs blocked by thin market/holiday.\n"
            f"  Rule: thin market affects SL design, NOT entry decisions.\n"
            f"  Examples: {'; '.join(thin_lines[:2])}"
        )

    return issues


# ── Check 5: Pass reason quality ──

def check_pass_reasons(state_text: str) -> list[str]:
    """Check if Pass reasons in state.md are substantive."""
    issues = []

    # Find Tier 2 scan lines
    vague_passes = []
    for line in state_text.split("\n"):
        if "Pass" in line or "pass" in line:
            # Check for vague reasons
            vague_patterns = [
                r"Pass\s*$",  # Just "Pass" with nothing after
                r"Pass\s*\(",  # Pass (reason) where reason is very short
                r"not now",
                r"waiting for.*confirmation",
            ]
            for vp in vague_patterns:
                if re.search(vp, line, re.IGNORECASE):
                    vague_passes.append(line.strip()[:120])
                    break

    if vague_passes:
        issues.append(
            f"**VAGUE PASS REASONS** ({len(vague_passes)}):\n"
            + "\n".join(f"  - {p}" for p in vague_passes[:3])
            + "\n  → Every Pass needs a specific, falsifiable reason."
        )

    return issues


# ── Check 6: Consecutive same-direction entries (directional bias) ──

def check_directional_bias(trades: list) -> list[str]:
    """Check if all open positions are in the same direction."""
    issues = []
    if len(trades) < 2:
        return []

    directions = set()
    pairs = []
    for t in trades:
        units = int(t.get("currentUnits", 0))
        if units > 0:
            directions.add("LONG")
        else:
            directions.add("SHORT")
        pairs.append(t["instrument"])

    if len(directions) == 1 and len(trades) >= 3:
        d = list(directions)[0]
        issues.append(
            f"**ALL POSITIONS {d}**: {len(trades)} positions, all {d}.\n"
            f"  Pairs: {', '.join(pairs)}\n"
            f"  → Not diversification, it's a bet. Where's the counter-trade?"
        )

    return issues


def main():
    t0 = time.time()
    all_issues = []

    # Load config and OANDA data
    try:
        cfg = load_config()
        token = cfg["oanda_token"]
        acct = cfg["oanda_account_id"]
    except Exception as e:
        print(f"Config error: {e}")
        sys.exit(0)

    # Fetch OANDA state
    try:
        summary = oanda_api(f"/v3/accounts/{acct}/summary", token, acct).get("account", {})
        nav = float(summary.get("NAV", 0))
        margin_used = float(summary.get("marginUsed", 0))
    except Exception as e:
        print(f"OANDA summary error: {e}")
        nav, margin_used = 0, 0

    try:
        trades_resp = oanda_api(f"/v3/accounts/{acct}/openTrades", token, acct)
        trades = trades_resp.get("trades", [])
    except Exception:
        trades = []

    # Read state.md
    state_path = ROOT / "collab_trade" / "state.md"
    state_text = state_path.read_text() if state_path.exists() else ""

    # Check state.md freshness
    if state_path.exists():
        age_min = (time.time() - state_path.stat().st_mtime) / 60
        if age_min > 10:
            all_issues.append(
                f"**STATE.MD STALE**: Last updated {age_min:.0f} minutes ago.\n"
                f"  → Trader may not be running. Check lock file and cron."
            )

    # Count S-candidates
    scan_out = run_script([VENV_PYTHON, "tools/s_conviction_scan.py"])
    s_count = scan_out.count("🎯")

    # Run all checks
    all_issues.extend(check_s_candidates_missed(state_text))
    all_issues.extend(check_sizing(trades, nav))
    all_issues.extend(check_margin(nav, margin_used, s_count, len(trades)))
    all_issues.extend(check_rule_misapplication(state_text))
    all_issues.extend(check_pass_reasons(state_text))
    all_issues.extend(check_directional_bias(trades))

    elapsed = time.time() - t0

    if not all_issues:
        print(f"CLEAN — no issues ({elapsed:.1f}s)")
        # Remove stale audit file if exists
        audit_path = ROOT / "logs" / "quality_audit.md"
        if audit_path.exists():
            audit_path.unlink()
        sys.exit(0)

    # Write audit report
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    report = f"# Quality Audit — {now}\n\n"
    report += f"NAV: {nav:.0f} JPY | Margin: {margin_used:.0f}/{nav:.0f} ({margin_used/nav*100:.0f}%) | "
    report += f"Positions: {len(trades)} | S-candidates: {s_count}\n\n"
    report += f"## Issues ({len(all_issues)})\n\n"
    for i, issue in enumerate(all_issues, 1):
        report += f"### {i}. {issue.split(chr(10))[0]}\n\n"
        rest = "\n".join(issue.split("\n")[1:])
        if rest.strip():
            report += f"{rest}\n\n"

    report += f"\n---\n[audit: {elapsed:.1f}s]\n"

    audit_path = ROOT / "logs" / "quality_audit.md"
    audit_path.write_text(report)

    # Print summary for the task
    print(f"ISSUES FOUND: {len(all_issues)} ({elapsed:.1f}s)")
    for issue in all_issues:
        first_line = issue.split("\n")[0]
        print(f"  • {first_line}")

    sys.exit(1)


if __name__ == "__main__":
    main()
