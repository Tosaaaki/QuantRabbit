#!/usr/bin/env python3
"""
Quality Audit v2 — Fact-gathering for discretionary auditing.

Design principle: Script presents VERIFIED FACTS. Sonnet-auditor makes judgments.
Ground truth = OANDA API (not state.md regex parsing).

Output:
  - logs/quality_audit.md  (human-readable, for trader + auditor)
  - logs/quality_audit.json (machine-readable, for daily-review)
  - logs/audit_history.jsonl (append-only scanner facts; the final audit prompt appends narrative picks)
  - stdout: summary for the task runner

Exit code: 0 = no findings, 1 = findings present (auditor decides severity)
"""
import json
import re
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from config_loader import get_oanda_config
from technicals_json import load_technicals_timeframes

ROOT = Path(__file__).resolve().parent.parent
VENV_PYTHON = str(ROOT / ".venv" / "bin" / "python")
PAIRS = ["USD_JPY", "EUR_USD", "GBP_USD", "AUD_USD", "EUR_JPY", "GBP_JPY", "AUD_JPY"]

NORMAL_SPREAD = {
    "USD_JPY": 1.2, "EUR_USD": 1.2, "GBP_USD": 1.8,
    "AUD_USD": 1.8, "EUR_JPY": 2.5, "GBP_JPY": 3.5, "AUD_JPY": 2.0,
}

PIP_MULT = {
    "USD_JPY": 100, "EUR_JPY": 100, "GBP_JPY": 100, "AUD_JPY": 100,
    "EUR_USD": 10000, "GBP_USD": 10000, "AUD_USD": 10000,
}


def load_config():
    return get_oanda_config()


def oanda_api(path, cfg):
    url = f"{cfg['oanda_base_url']}{path}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {cfg['oanda_token']}"})
    return json.loads(urllib.request.urlopen(req, timeout=10).read())


def run_script(args, timeout=30):
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=timeout, cwd=str(ROOT))
        return r.stdout.strip()
    except Exception as e:
        return f"(error: {e})"


def load_technicals(pair: str) -> dict:
    f = ROOT / f"logs/technicals_{pair}.json"
    if not f.exists():
        return {}
    try:
        return load_technicals_timeframes(f)
    except Exception:
        return {}


# ──────────────────────────────────────────────
#  OANDA ground-truth helpers
# ──────────────────────────────────────────────

def build_held_pairs(trades: list) -> dict:
    """Build held pairs dict from OANDA trades. Returns {pair: {direction, units, upl, id, price}}."""
    held = {}
    for t in trades:
        pair = t["instrument"]
        units = int(t.get("currentUnits", 0))
        held[pair] = {
            "direction": "LONG" if units > 0 else "SHORT",
            "units": abs(units),
            "upl_jpy": float(t.get("unrealizedPL", 0)),
            "id": t.get("id", ""),
            "entry_price": float(t.get("price", 0)),
        }
    return held


def get_trade_orders(trade: dict, token: str, acct: str) -> dict:
    """Fetch dependent orders (TP/SL/trailing) for a trade."""
    orders = {"has_tp": False, "has_sl": False, "has_trailing": False,
              "tp_price": None, "sl_price": None, "trailing_distance": None}
    if "takeProfitOrder" in trade and trade["takeProfitOrder"]:
        orders["has_tp"] = True
        orders["tp_price"] = float(trade["takeProfitOrder"].get("price", 0))
    if "stopLossOrder" in trade and trade["stopLossOrder"]:
        orders["has_sl"] = True
        orders["sl_price"] = float(trade["stopLossOrder"].get("price", 0))
    if "trailingStopLossOrder" in trade and trade["trailingStopLossOrder"]:
        orders["has_trailing"] = True
        orders["trailing_distance"] = float(trade["trailingStopLossOrder"].get("distance", 0))
    return orders


# ──────────────────────────────────────────────
#  S-scan parsing (runs ONCE, result passed around)
# ──────────────────────────────────────────────

def parse_s_scan(scan_output: str) -> list[dict]:
    """Parse S-conviction scan output into structured data."""
    candidates = []
    for line in scan_output.split("\n"):
        if not line.startswith("🎯"):
            continue
        m = re.match(r"🎯 (\w+) (LONG|SHORT) (\S+): (.+)", line)
        if m:
            details = m.group(4)
            # Extract @price tag if present (appended by s_conviction_scan.py)
            price = 0.0
            price_m = re.search(r"@([\d.]+)$", details)
            if price_m:
                price = float(price_m.group(1))
                details = details[:price_m.start()].rstrip()
            candidates.append({
                "pair": m.group(1),
                "direction": m.group(2),
                "recipe": m.group(3),
                "details": details,
                "price": price,
                "full": line,
            })
    return candidates


def unique_s_candidates(candidates: list[dict]) -> list[dict]:
    """Keep exact-duplicate scanner lines out without collapsing distinct same-pair seats."""
    unique: list[dict] = []
    seen: set[tuple[str, str, str, str, float]] = set()
    for candidate in candidates:
        key = (
            str(candidate.get("pair") or ""),
            str(candidate.get("direction") or ""),
            str(candidate.get("recipe") or ""),
            str(candidate.get("details") or ""),
            float(candidate.get("price") or 0.0),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


# ──────────────────────────────────────────────
#  Fact checks (no judgments — just data)
# ──────────────────────────────────────────────

def gather_s_scan_facts(s_candidates: list, held: dict, margin_pct: float) -> list[dict]:
    """Cross-reference S-scan with held positions. Returns fact entries."""
    facts = []
    for sc in s_candidates:
        pair = sc["pair"]
        entry = {
            "pair": pair,
            "direction": sc["direction"],
            "recipe": sc["recipe"],
            "details": sc["details"],
            "price": sc.get("price", 0),
        }
        if pair in held:
            if held[pair]["direction"] == sc["direction"]:
                entry["status"] = "ALREADY_HELD"
                entry["held_units"] = held[pair]["units"]
                entry["held_upl"] = held[pair]["upl_jpy"]
            else:
                entry["status"] = "HELD_OPPOSITE"
                entry["held_direction"] = held[pair]["direction"]
                entry["held_units"] = held[pair]["units"]
        else:
            entry["status"] = "NOT_HELD"
            if margin_pct >= 90:
                entry["margin_note"] = f"margin-blocked at {margin_pct:.0f}%"
            elif margin_pct >= 85:
                entry["margin_note"] = f"margin tight at {margin_pct:.0f}%"
            else:
                entry["margin_note"] = f"margin available ({100 - margin_pct:.0f}% idle)"
        facts.append(entry)
    return facts


def gather_exit_quality(trades: list, state_text: str, token: str, acct: str) -> list[dict]:
    """Check exit quality: peak drawdown, BE SL, ATR×1.0 stall."""
    findings = []

    for t in trades:
        pair = t["instrument"]
        units = int(t.get("currentUnits", 0))
        side = "LONG" if units > 0 else "SHORT"
        upl = float(t.get("unrealizedPL", 0))
        entry_price = float(t.get("price", 0))

        # --- Peak drawdown ---
        peak_jpy = None
        # Parse from state.md: "Peak: +588 JPY" or "peak: +727 JPY" or "Peak: +3,200 JPY"
        in_section = False
        for line in state_text.split("\n"):
            if pair in line and "###" in line:
                in_section = True
            elif in_section:
                if line.startswith("##") or line.startswith("---"):
                    break
                peak_m = re.search(r"[Pp]eak[:\s]+\+?([\d,]+)\s*(?:JPY|円|yen)", line, re.IGNORECASE)
                if peak_m:
                    peak_jpy = float(peak_m.group(1).replace(",", ""))

        if peak_jpy is not None and peak_jpy > 50 and upl < peak_jpy:
            drawdown_pct = ((peak_jpy - upl) / peak_jpy * 100) if peak_jpy > 0 else 0
            # If UPL went negative, drawdown exceeds 100% — mark as profit_lost
            severity = "WATCH" if drawdown_pct >= 60 else "DATA"
            findings.append({
                "pair": pair, "type": "peak_drawdown", "severity": severity,
                "peak_jpy": round(peak_jpy), "current_jpy": round(upl),
                "drawdown_pct": round(min(drawdown_pct, 100)),
                "profit_lost": round(peak_jpy - upl),
                "went_negative": upl < 0,
            })

        # --- BE SL detection ---
        orders = get_trade_orders(t, token, acct)
        if orders["has_sl"] and orders["sl_price"] and entry_price > 0:
            pip_m = PIP_MULT.get(pair, 100)
            sl_dist_pips = abs(orders["sl_price"] - entry_price) * pip_m
            if sl_dist_pips < 2.0 and upl > 0:  # SL within 2 pips of entry AND in profit
                findings.append({
                    "pair": pair, "type": "be_sl", "severity": "WATCH",
                    "entry_price": entry_price, "sl_price": orders["sl_price"],
                    "sl_dist_pips": round(sl_dist_pips, 1),
                    "current_upl": round(upl),
                    "half_tp_value": round(upl / 2),
                })

        # --- ATR×1.0 reached, no recent action ---
        tfs = load_technicals(pair)
        h1 = tfs.get("H1", {})
        atr = h1.get("atr", 0)
        if atr > 0 and entry_price > 0:
            pip_m = PIP_MULT.get(pair, 100)
            current_price = h1.get("close", entry_price)
            if side == "LONG":
                move_pips = (current_price - entry_price) * pip_m
            else:
                move_pips = (entry_price - current_price) * pip_m
            atr_pips = atr * pip_m
            atr_ratio = move_pips / atr_pips if atr_pips > 0 else 0

            if atr_ratio >= 1.0 and upl > 0:
                # Check if recent close/half_tp in live_trade_log
                log_path = ROOT / "logs" / "live_trade_log.txt"
                recent_action = False
                if log_path.exists():
                    for log_line in log_path.read_text().strip().split("\n")[-20:]:
                        if pair in log_line and any(w in log_line for w in ["HALF_TP", "CLOSE", "profit_check"]):
                            recent_action = True
                            break

                if not recent_action:
                    findings.append({
                        "pair": pair, "type": "atr_reached_no_action", "severity": "WATCH",
                        "atr_ratio": round(atr_ratio, 2), "upl_jpy": round(upl),
                        "atr_pips": round(atr_pips, 1), "move_pips": round(move_pips, 1),
                    })

    return findings


def _is_logged_trade_receipt_line(line: str) -> bool:
    """Return true when a log line records a real trade receipt, not only an order."""
    if not line:
        return False
    if any(token in line for token in ("ORDER_REJECT", "CANCEL_ORDER", "MODIFY ")):
        return False
    if "ENTRY_ORDER" in line:
        return False
    if any(token in line for token in ("LIMIT_FILL", "STOP_FILL", "via LIMIT fill", "via STOP-ENTRY fill")):
        return True
    if re.search(r"\bENTRY\s+[A-Z]{3}_[A-Z]{3}\s+(LONG|SHORT)\b", line):
        return True
    if re.search(r"\]\s+[A-Z]{3}_[A-Z]{3}\s+(LONG|SHORT)\s+\d+u\s+@", line):
        return True
    return False


def load_logged_trade_ids() -> set:
    """Load trade IDs that appear in live_trade_log.txt as real fill receipts."""
    log_path = ROOT / "logs" / "live_trade_log.txt"
    if not log_path.exists():
        return set()
    ids = set()
    for line in log_path.read_text().split("\n"):
        if not _is_logged_trade_receipt_line(line):
            continue
        for m in re.finditer(r"id=(\d+)", line):
            ids.add(m.group(1))
    return ids


def gather_position_facts(trades: list, token: str, acct: str) -> list[dict]:
    """Build position summary with order status + manual position detection."""
    logged_ids = load_logged_trade_ids()
    positions = []
    for t in trades:
        pair = t["instrument"]
        units = int(t.get("currentUnits", 0))
        trade_id = t.get("id", "")
        orders = get_trade_orders(t, token, acct)
        is_manual = trade_id not in logged_ids
        positions.append({
            "pair": pair,
            "direction": "LONG" if units > 0 else "SHORT",
            "units": abs(units),
            "upl_jpy": round(float(t.get("unrealizedPL", 0))),
            "entry_price": float(t.get("price", 0)),
            "has_tp": orders["has_tp"],
            "has_sl": orders["has_sl"],
            "has_trailing": orders["has_trailing"],
            "sl_is_be": False,  # set below
            "trade_id": trade_id,
            "is_manual": is_manual,
        })
        # Check BE
        if orders["has_sl"] and orders["sl_price"]:
            pip_m = PIP_MULT.get(pair, 100)
            sl_dist = abs(orders["sl_price"] - float(t.get("price", 0))) * pip_m
            if sl_dist < 2.0:
                positions[-1]["sl_is_be"] = True
    return positions


def gather_directional_mix(trades: list) -> dict:
    """Directional balance check."""
    long_count = sum(1 for t in trades if int(t.get("currentUnits", 0)) > 0)
    short_count = len(trades) - long_count
    long_pairs = [t["instrument"] for t in trades if int(t.get("currentUnits", 0)) > 0]
    short_pairs = [t["instrument"] for t in trades if int(t.get("currentUnits", 0)) < 0]
    if len(trades) >= 3 and (long_count == 0 or short_count == 0):
        status = f"ALL_{'LONG' if long_count > 0 else 'SHORT'}"
        severity = "WATCH"
    else:
        status = "mixed"
        severity = "DATA"
    return {
        "long": long_count, "short": short_count,
        "long_pairs": long_pairs, "short_pairs": short_pairs,
        "status": status, "severity": severity,
    }


def gather_sizing_facts(nav: float) -> list[dict]:
    """Check today's entries against conviction sizing rules."""
    findings = []
    log_path = ROOT / "logs" / "live_trade_log.txt"
    if not log_path.exists():
        return []

    lines = log_path.read_text().strip().split("\n")
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    for line in lines[-30:]:
        if "ENTRY" not in line or today not in line:
            continue

        units_match = re.search(r"(\d+)u", line)
        if not units_match:
            continue
        units = int(units_match.group(1))

        # Parse conviction letter
        conviction = None
        m = re.search(r"pretrade=([SABC])\(\d+\)", line)
        if m:
            conviction = m.group(1)
        else:
            m = re.search(r"pretrade=\w+\(([SABC])\d*\)", line)
            if m:
                conviction = m.group(1)
        if not conviction:
            # Try conviction= field
            m = re.search(r"conviction=([SABC])", line)
            if m:
                conviction = m.group(1)
        if not conviction:
            continue

        # Estimate margin %
        pair_match = re.search(r"(USD_JPY|EUR_USD|GBP_USD|AUD_USD|EUR_JPY|GBP_JPY|AUD_JPY)", line)
        if not pair_match:
            continue
        pair = pair_match.group(1)
        leverage = 25
        if "JPY" in pair:
            margin_per_unit = 150 / leverage
        else:
            base_jpy = {"EUR_USD": 170, "GBP_USD": 195, "AUD_USD": 100}
            margin_per_unit = base_jpy.get(pair, 155) / leverage
        actual_pct = (units * margin_per_unit / nav) if nav > 0 else 0

        expected_pct = {"S": 0.30, "A": 0.15, "B": 0.05, "C": 0.02}
        exp = expected_pct.get(conviction, 0.05)

        if conviction == "S" and actual_pct < 0.15:
            findings.append({
                "type": "undersized", "severity": "WATCH", "conviction": conviction,
                "pair": pair, "units": units,
                "actual_pct": round(actual_pct * 100, 1),
                "expected_pct": round(exp * 100),
                "entry_line": line[:140],
            })
        elif conviction == "A" and actual_pct < 0.07:
            findings.append({
                "type": "undersized", "severity": "DATA", "conviction": conviction,
                "pair": pair, "units": units,
                "actual_pct": round(actual_pct * 100, 1),
                "expected_pct": round(exp * 100),
                "entry_line": line[:140],
            })

        if units < 2000:
            findings.append({
                "type": "below_minimum", "severity": "DATA",
                "pair": pair, "units": units,
                "entry_line": line[:140],
            })

    return findings


def gather_rule_flags(state_text: str, s_candidates: list) -> list[dict]:
    """Detect rule misapplication patterns from state.md text."""
    flags = []

    # Circuit breaker blocking opposite direction
    cb_pairs = {}
    for line in state_text.split("\n"):
        cb_match = re.search(r"(\w+_\w+).*circuit.?breaker", line, re.IGNORECASE)
        if cb_match:
            pair = cb_match.group(1)
            if "SHORT" in line.upper() and "LONG" not in line.split("circuit")[0].upper():
                cb_pairs[pair] = "SHORT"
            elif "LONG" in line.upper():
                cb_pairs[pair] = "LONG"

    for pair, blocked_dir in cb_pairs.items():
        opposite = "LONG" if blocked_dir == "SHORT" else "SHORT"
        for sc in s_candidates:
            if sc["pair"] == pair and sc["direction"] == opposite:
                flags.append({
                    "type": "circuit_breaker_misapplied", "severity": "WATCH",
                    "pair": pair, "blocked_dir": blocked_dir, "s_direction": opposite,
                    "s_recipe": sc["recipe"],
                })

    # Normal spread called wide
    for line in state_text.split("\n"):
        spread_match = re.search(
            r"(\w+_\w+).*[Ss]pread[=:]?\s*([\d.]+)\s*pip.*(?:wide|⚠|too)", line
        )
        if spread_match:
            pair = spread_match.group(1)
            spread_val = float(spread_match.group(2))
            normal = NORMAL_SPREAD.get(pair, 5.0)
            if spread_val <= normal:
                flags.append({
                    "type": "spread_excuse", "severity": "DATA",
                    "pair": pair, "spread": spread_val, "normal": normal,
                })

    # Thin market blanket block
    thin_count = 0
    for line in state_text.split("\n"):
        if re.search(r"thin|holiday|Easter|liquidity", line, re.IGNORECASE):
            if re.search(r"Pass|no entry|no new|skip", line, re.IGNORECASE):
                thin_count += 1
    if thin_count >= 3:
        flags.append({
            "type": "thin_market_blanket", "severity": "DATA",
            "count": thin_count,
        })

    return flags


def parse_state_positions(state_text: str) -> list[dict]:
    """Parse held positions from state.md across current and legacy formats."""
    positions = []
    seen = set()
    lines = state_text.splitlines()

    # Current canonical format: open positions live under "## Positions (Current)".
    in_positions = False
    for raw_line in lines:
        line = raw_line.strip()
        if line.startswith("## "):
            in_positions = line.lower().startswith("## positions")
            continue
        if not in_positions or not line:
            continue
        upper = line.upper()
        if "LIMIT" in upper or "PENDING" in upper:
            continue
        matches = list(re.finditer(r"\b([A-Z]{3}_[A-Z]{3})\s+(LONG|SHORT)\b", line))
        for m in matches:
            before = line[:m.start()].lower()
            after = line[m.end():].lower()
            local_context = after[:120]
            if "closed" in local_context or "not live" in local_context:
                continue
            if "id=" not in local_context and "trade id=" not in local_context and not re.match(
                r"\s+[\d,]+u\b", after
            ):
                continue
            key = (m.group(1), m.group(2))
            if key in seen:
                continue
            seen.add(key)
            positions.append({
                "pair": m.group(1),
                "direction": m.group(2),
                "line": line,
                "source": "positions_current",
            })

    if positions:
        return positions

    # Legacy state.md format used position headers like "### EUR_USD LONG ..."
    for raw_line in lines:
        line = raw_line.strip()
        if not line.startswith("###") or "LIMIT" in line.upper():
            continue
        m = re.match(r"^###\s+(?:Position:\s+)?([A-Z]{3}_[A-Z]{3})\s+(LONG|SHORT)\b", line)
        if not m:
            continue
        key = (m.group(1), m.group(2))
        if key in seen:
            continue
        seen.add(key)
        positions.append({
            "pair": m.group(1),
            "direction": m.group(2),
            "line": line,
            "source": "legacy_header",
        })

    return positions


def self_check(positions: list[dict], state_text: str) -> dict:
    """Verify OANDA ground truth vs state.md parsed positions."""
    oanda_pairs = {p["pair"] for p in positions}
    parsed_state_positions = parse_state_positions(state_text)
    parsed_pairs = {p["pair"] for p in parsed_state_positions}

    oanda_by_key = {}
    for p in positions:
        key = (p["pair"], p["direction"])
        entry = oanda_by_key.setdefault(key, {
            "pair": p["pair"],
            "direction": p["direction"],
            "trade_ids": [],
            "units": 0,
            "upl_jpy": 0,
            "is_manual": False,
        })
        entry["trade_ids"].append(str(p.get("trade_id", "")))
        entry["units"] += int(p.get("units", 0))
        entry["upl_jpy"] += int(p.get("upl_jpy", 0))
        entry["is_manual"] = entry["is_manual"] or bool(p.get("is_manual"))

    state_by_key = {}
    for p in parsed_state_positions:
        key = (p["pair"], p["direction"])
        state_by_key[key] = {
            "pair": p["pair"],
            "direction": p["direction"],
            "line": p["line"],
            "source": p["source"],
        }

    oanda_keys = set(oanda_by_key)
    state_keys = set(state_by_key)
    live_not_in_state = [oanda_by_key[key] for key in sorted(oanda_keys - state_keys)]
    state_not_live = [state_by_key[key] for key in sorted(state_keys - oanda_keys)]
    match = oanda_keys == state_keys

    note_parts = []
    if live_not_in_state:
        live_desc = []
        for p in live_not_in_state:
            manual_note = " [MANUAL/NO_LOG]" if p["is_manual"] else ""
            ids = ",".join(p["trade_ids"])
            live_desc.append(
                f"{p['pair']} {p['direction']} id={ids}{manual_note}"
            )
        note_parts.append(f"OANDA only: {', '.join(live_desc)}")
    if state_not_live:
        state_desc = [f"{p['pair']} {p['direction']}" for p in state_not_live]
        note_parts.append(f"state.md only: {', '.join(state_desc)}")

    result = {
        "held_positions_match": match,
        "held_pairs_match": match,
        "oanda_pairs": sorted(oanda_pairs),
        "parsed_pairs": sorted(parsed_pairs),
        "oanda_positions": [f"{pair} {direction}" for pair, direction in sorted(oanda_keys)],
        "parsed_positions": [f"{pair} {direction}" for pair, direction in sorted(state_keys)],
        "live_not_in_state": live_not_in_state,
        "state_not_live": state_not_live,
    }
    if not match:
        result["note"] = " | ".join(note_parts) if note_parts else (
            f"OANDA has {sorted(oanda_pairs)}, state.md parsed {sorted(parsed_pairs)}"
        )
    return result


# ──────────────────────────────────────────────
#  Output formatters
# ──────────────────────────────────────────────

def build_json_report(data: dict) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)


def build_markdown_report(data: dict) -> str:
    """Human-readable report for trader + auditor. Facts, not accusations."""
    ts = data["timestamp"]
    acct = data["account"]
    mkt_state = data.get("market_state", "OPEN")
    mkt_reason = data.get("market_reason", "")

    lines = [
        f"# Quality Audit — {ts}",
        "",
    ]

    # Market state banner — prevents panic actions on stale/distorted data
    if mkt_state in ("CLOSED", "ROLLOVER"):
        lines.append(f"## ⛔ MARKET {mkt_state}")
        lines.append(f"{mkt_reason}")
        lines.append("**All findings below are INFORMATIONAL ONLY. DO NOT act on them until market reopens.**")
        lines.append("")

    lines.append(
        f"NAV: {acct['nav']:.0f} JPY | Margin: {acct['margin_pct']:.0f}% | "
        f"Positions: {acct['positions']} | S-candidates: {len(data['s_scan'])}"
    )
    lines.append("")

    # Self-check warning
    sc = data["self_check"]
    if not sc["held_pairs_match"]:
        lines.append(f"⚠ **SELF-CHECK**: {sc['note']}")
        lines.append("")
        if sc["live_not_in_state"] or sc["state_not_live"]:
            lines.append("## ⚠ Self-Check Drift")
            lines.append("")
            for p in sc["live_not_in_state"]:
                manual_note = " **[MANUAL/NO_LOG]**" if p.get("is_manual") else ""
                ids = ",".join(p.get("trade_ids", []))
                lines.append(
                    f"- OANDA only: **{p['pair']} {p['direction']} {p['units']}u** "
                    f"id={ids} | UPL: {p['upl_jpy']:+.0f} JPY{manual_note}"
                )
            for p in sc["state_not_live"]:
                lines.append(
                    f"- state.md only: **{p['pair']} {p['direction']}** "
                    f"| line: `{p['line']}`"
                )
            lines.append("")

    # Manual positions (user-entered, not in trade log)
    manual_positions = [p for p in data["positions"] if p.get("is_manual")]
    if manual_positions:
        lines.append("## ⚠ Manual Positions (not in trade log)")
        lines.append("")
        for p in manual_positions:
            lines.append(
                f"- **{p['pair']} {p['direction']} {p['units']}u** id={p.get('trade_id','')} "
                f"@{p['entry_price']} | UPL: {p['upl_jpy']:+.0f} JPY"
            )
            lines.append(
                f"  Entered via OANDA directly (no ENTRY/LIMIT_FILL in live_trade_log.txt). "
                f"No pretrade_check, no Slack notification, no conviction record."
            )
        lines.append("")

    # Positions
    if data["positions"]:
        lines.append("## Positions (OANDA verified)")
        lines.append("")
        for p in data["positions"]:
            protection = []
            if p["has_tp"]:
                protection.append("TP")
            if p["has_sl"]:
                protection.append("BE-SL" if p["sl_is_be"] else "SL")
            if p["has_trailing"]:
                protection.append("Trail")
            prot_str = "+".join(protection) if protection else "NO PROTECTION"
            manual_tag = " **[MANUAL]**" if p.get("is_manual") else ""
            lines.append(
                f"- {p['pair']} {p['direction']} {p['units']}u | "
                f"UPL: {p['upl_jpy']:+.0f} JPY | {prot_str}{manual_tag}"
            )
        lines.append("")

    # Directional mix
    dm = data["directional_mix"]
    if dm["severity"] != "DATA":
        lines.append(f"## Directional Mix: {dm['status']}")
        lines.append(f"LONG({dm['long']}): {', '.join(dm['long_pairs'])} / "
                      f"SHORT({dm['short']}): {', '.join(dm['short_pairs'])}")
        lines.append("")

    # Exit quality
    if data["exit_quality"]:
        lines.append("## Exit Quality")
        lines.append("")
        for eq in data["exit_quality"]:
            if eq["type"] == "peak_drawdown":
                if eq.get("went_negative"):
                    lines.append(
                        f"- {eq['pair']}: Peak {eq['peak_jpy']:+d} JPY → Now {eq['current_jpy']:+d} JPY "
                        f"(gave back all profit + now in loss. Lost {eq['profit_lost']} JPY from peak)"
                    )
                else:
                    lines.append(
                        f"- {eq['pair']}: Peak {eq['peak_jpy']:+d} JPY → Now {eq['current_jpy']:+d} JPY "
                        f"(drawdown {eq['drawdown_pct']}%, lost {eq['profit_lost']} JPY from peak)"
                    )
            elif eq["type"] == "be_sl":
                lines.append(
                    f"- {eq['pair']}: SL at entry ({eq['sl_price']}, {eq['sl_dist_pips']}pip from entry). "
                    f"Current UPL: {eq['current_upl']:+d} JPY. "
                    f"HALF_TP would lock in ~{eq['half_tp_value']:+d} JPY."
                )
            elif eq["type"] == "atr_reached_no_action":
                lines.append(
                    f"- {eq['pair']}: ATR×{eq['atr_ratio']:.1f} reached "
                    f"({eq['move_pips']:.0f}/{eq['atr_pips']:.0f}pip). "
                    f"UPL: {eq['upl_jpy']:+d} JPY. No HALF_TP/CLOSE in recent log."
                )
        lines.append("")

    # S-scan facts
    not_held = [s for s in data["s_scan"] if s["status"] == "NOT_HELD"]
    already = [s for s in data["s_scan"] if s["status"] == "ALREADY_HELD"]
    opposite = [s for s in data["s_scan"] if s["status"] == "HELD_OPPOSITE"]

    if not_held or already or opposite:
        lines.append("## S-Scan Results")
        lines.append("")
        if already:
            for s in already:
                lines.append(f"- ✅ {s['pair']} {s['direction']} {s['recipe']}: ALREADY HELD")
        if not_held:
            for s in not_held:
                margin_note = s.get("margin_note", "")
                lines.append(
                    f"- 📊 {s['pair']} {s['direction']} {s['recipe']}: "
                    f"NOT HELD ({margin_note})"
                )
                lines.append(f"  Details: {s['details']}")
                # Add the thinking-prompt for trader
                lines.append(f"  If I would enter: ___ / If I would not: ___")
        if opposite:
            for s in opposite:
                lines.append(
                    f"- ↕ {s['pair']} {s['direction']} {s['recipe']}: "
                    f"HELD {s['held_direction']} ({s['held_units']}u)"
                )
        lines.append("")

    # Sizing
    if data["sizing_flags"]:
        lines.append("## Sizing")
        lines.append("")
        for sf in data["sizing_flags"]:
            if sf["type"] == "undersized":
                lines.append(
                    f"- {sf['pair']} {sf['conviction']}-conviction: "
                    f"{sf['units']}u = {sf['actual_pct']}% NAV "
                    f"(expected ~{sf['expected_pct']}%)"
                )
            elif sf["type"] == "below_minimum":
                lines.append(f"- {sf['pair']}: {sf['units']}u below 2000u minimum")
        lines.append("")

    # Rule flags
    if data["rule_flags"]:
        lines.append("## Rule Observations")
        lines.append("")
        for rf in data["rule_flags"]:
            if rf["type"] == "circuit_breaker_misapplied":
                lines.append(
                    f"- {rf['pair']}: Circuit breaker ({rf['blocked_dir']}) "
                    f"vs S-scan {rf['s_direction']} {rf['s_recipe']}"
                )
            elif rf["type"] == "spread_excuse":
                lines.append(
                    f"- {rf['pair']}: Spread {rf['spread']}pip flagged as wide "
                    f"(normal: {rf['normal']}pip)"
                )
            elif rf["type"] == "thin_market_blanket":
                lines.append(
                    f"- {rf['count']} pairs blocked by thin market / holiday"
                )
        lines.append("")

    # State freshness
    if data.get("state_age_min") and data["state_age_min"] > 10:
        lines.append(f"## State.md: last updated {data['state_age_min']:.0f} minutes ago")
        lines.append("")

    lines.append(f"---")
    lines.append(f"[audit v2: {data['elapsed']:.1f}s]")
    return "\n".join(lines)


def extract_auditor_view(text: str) -> str:
    """Return the trailing Auditor's View block from quality_audit.md if present."""
    if not text:
        return ""
    match = re.search(r"^## Auditor's View — .*$", text, re.M)
    if not match:
        return ""
    return text[match.start():].strip()


def merge_previous_auditor_view(md_report: str, previous_text: str) -> str:
    """Keep the last Auditor's View available for the next cycle's follow-up step.

    Step 2 of the quality-audit playbook reads `logs/quality_audit.md` to compare the
    current cycle with the previous audit predictions. If the fact-gathering script
    overwrites the file with facts only, the auditor loses the previous narrative and
    the follow-up loop breaks. Preserve the latest Auditor's View until the current
    cycle writes a fresh one.
    """
    report = md_report.rstrip()
    prior_view = extract_auditor_view(previous_text)
    if not prior_view:
        return report + "\n"

    first_line = prior_view.splitlines()[0].strip()
    preserved_ts = first_line.replace("## Auditor's View — ", "", 1)
    merged = [
        report,
        "",
        "---",
        "",
        f"_Previous Auditor's View preserved for follow-up ({preserved_ts}) until this cycle writes a new one._",
        "",
        prior_view,
    ]
    return "\n".join(merged).rstrip() + "\n"


def append_audit_history(data: dict):
    """Append S-scan results with prices to audit_history.jsonl for outcome tracking."""
    history_path = ROOT / "logs" / "audit_history.jsonl"

    # Rotation: keep last 5000 lines (~6 months at 30min intervals)
    MAX_LINES = 5000
    if history_path.exists() and history_path.stat().st_size > 2_000_000:  # 2MB safety
        lines = history_path.read_text().strip().split("\n")
        if len(lines) > MAX_LINES:
            history_path.write_text("\n".join(lines[-MAX_LINES:]) + "\n")

    entry = {
        "timestamp": data["timestamp"],
        "source": "s_scan",
        "s_scan": [],
        "positions": len(data["positions"]),
        "margin_pct": data["account"]["margin_pct"],
    }
    for sc in data["s_scan"]:
        # Use price parsed from scan output (@price tag) — more accurate than stale cache
        price = sc.get("price", 0)
        if not price:
            tfs = load_technicals(sc["pair"])
            m5 = tfs.get("M5", {})
            price = m5.get("close", 0)
        entry["s_scan"].append({
            "pair": sc["pair"],
            "direction": sc["direction"],
            "recipe": sc["recipe"],
            "status": sc["status"],
            "price_at_detection": price,
        })
    with open(history_path, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ──────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────

def main():
    t0 = time.time()

    # Load config
    try:
        cfg = load_config()
        token = cfg["oanda_token"]
        acct = cfg["oanda_account_id"]
    except Exception as e:
        print(f"Config error: {e}")
        sys.exit(0)

    # Fetch OANDA state
    try:
        summary = oanda_api(f"/v3/accounts/{acct}/summary", cfg).get("account", {})
        nav = float(summary.get("NAV", 0))
        margin_used = float(summary.get("marginUsed", 0))
    except Exception as e:
        print(f"OANDA summary error: {e}")
        nav, margin_used = 0, 0

    try:
        trades_resp = oanda_api(f"/v3/accounts/{acct}/openTrades", cfg)
        trades = trades_resp.get("trades", [])
    except Exception:
        trades = []

    margin_pct = (margin_used / nav * 100) if nav > 0 else 0

    # Read state.md
    state_path = ROOT / "collab_trade" / "state.md"
    state_text = state_path.read_text() if state_path.exists() else ""
    state_age_min = None
    if state_path.exists():
        state_age_min = (time.time() - state_path.stat().st_mtime) / 60

    # Build OANDA ground truth
    held = build_held_pairs(trades)

    # Run S-scan ONCE
    scan_output = run_script([VENV_PYTHON, "tools/s_conviction_scan.py"])
    raw_candidates = parse_s_scan(scan_output)
    s_candidates = unique_s_candidates(raw_candidates)

    # Gather all facts
    s_scan_facts = gather_s_scan_facts(s_candidates, held, margin_pct)
    exit_quality = gather_exit_quality(trades, state_text, token, acct)
    positions = gather_position_facts(trades, token, acct)
    directional_mix = gather_directional_mix(trades)
    sizing_flags = gather_sizing_facts(nav)
    rule_flags = gather_rule_flags(state_text, s_candidates)
    sc = self_check(positions, state_text)

    elapsed = time.time() - t0
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Market state (time-based: OPEN/ROLLOVER/CLOSED)
    from market_state import get_market_state
    mkt_state, mkt_reason = get_market_state()

    # Assemble report data
    data = {
        "timestamp": now,
        "market_state": mkt_state,
        "market_reason": mkt_reason,
        "account": {
            "nav": nav, "margin_used": margin_used,
            "margin_pct": round(margin_pct, 1),
            "positions": len(trades),
        },
        "s_scan": s_scan_facts,
        "s_scan_inventory_note": (
            f"{len(raw_candidates)} raw seats scanned → {len(s_candidates)} kept "
            "(exact-duplicate filter only; no pair+direction compression)"
        ),
        "exit_quality": exit_quality,
        "positions": positions,
        "directional_mix": directional_mix,
        "sizing_flags": sizing_flags,
        "rule_flags": rule_flags,
        "self_check": sc,
        "state_age_min": state_age_min,
        "elapsed": round(elapsed, 1),
    }

    # Count actionable findings (for exit code)
    manual_count = len([p for p in positions if p.get("is_manual")])
    has_findings = bool(
        [s for s in s_scan_facts if s["status"] == "NOT_HELD"]
        or exit_quality
        or directional_mix["severity"] != "DATA"
        or sizing_flags
        or rule_flags
        or manual_count
        or (state_age_min and state_age_min > 10)
        or not sc["held_pairs_match"]
    )

    # Write outputs
    md_report = build_markdown_report(data)
    json_report = build_json_report(data)

    audit_md = ROOT / "logs" / "quality_audit.md"
    audit_json = ROOT / "logs" / "quality_audit.json"
    previous_audit_text = audit_md.read_text() if audit_md.exists() else ""
    md_output = merge_previous_auditor_view(md_report, previous_audit_text)

    audit_md.write_text(md_output)
    audit_json.write_text(json_report)

    # Always append to history (for outcome tracking)
    if s_scan_facts:
        append_audit_history(data)

    # Print summary
    not_held_count = len([s for s in s_scan_facts if s["status"] == "NOT_HELD"])
    exit_count = len(exit_quality)
    rule_count = len(rule_flags)
    sizing_count = len(sizing_flags)

    # When market is CLOSED/ROLLOVER, findings are informational only — exit 0 (no alert)
    if mkt_state in ("CLOSED", "ROLLOVER"):
        label = f"MARKET {mkt_state}"
        if has_findings:
            print(f"{label} — {not_held_count+exit_count+rule_count+sizing_count} finding(s) recorded as informational ({elapsed:.1f}s)")
        else:
            print(f"{label} — CLEAN ({elapsed:.1f}s)")
        sys.exit(0)

    if has_findings:
        parts = []
        if manual_count:
            parts.append(f"manual:{manual_count}")
        if not_held_count:
            parts.append(f"s-scan:{not_held_count}")
        if exit_count:
            parts.append(f"exit:{exit_count}")
        if directional_mix["severity"] != "DATA":
            parts.append(f"direction:{directional_mix['status']}")
        if sizing_count:
            parts.append(f"sizing:{sizing_count}")
        if rule_count:
            parts.append(f"rules:{rule_count}")
        if not sc["held_pairs_match"]:
            parts.append(
                f"self-check:live{len(sc['live_not_in_state'])}/state{len(sc['state_not_live'])}"
            )
        if state_age_min and state_age_min > 10:
            parts.append(f"stale:{state_age_min:.0f}min")
        print(f"FINDINGS: {' | '.join(parts)} ({elapsed:.1f}s)")
        sys.exit(1)
    else:
        print(f"CLEAN ({elapsed:.1f}s)")
        sys.exit(0)


if __name__ == "__main__":
    main()
