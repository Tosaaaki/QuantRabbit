#!/usr/bin/env python3
"""Build a research-only decision-time execution evidence ledger.

The decision boundary and the observed execution/outcome boundary are kept
separate on purpose.  No broker or runtime modules are imported.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sqlite3
import statistics
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
EPISODES = REPO / "research/historical_learning_admission/all_entry_episodes_v1.jsonl"
PRIOR_REPORT = REPO / "research/system_utilization_rca/2026-08-10/utilization_report_v1.json"
PRIOR_FUSED = REPO / "research/system_utilization_rca/2026-08-10/fused_decisions_v1.jsonl"
REAL_PAYLOAD = REPO / "research/python_ecosystem_audit/2026-08-10/real_shadow_payload.json"
EXECUTION_DB = REPO / "data/execution_ledger.db"
SEED = 20_260_810
BOOTSTRAPS = 4_000
S5_RE = re.compile(r"_S5_BA_(\d{8}T\d{6}Z)_(\d{8}T\d{6}Z)\.jsonl\.gz$")
WINDOWS = ("INITIAL_16D", "DOUBLE_32D", "QUADRUPLE_64D")
REQUIRED_STAGES = (
    "pricing",
    "candidate_order",
    "fillability",
    "slippage_fee_financing",
    "margin_exposure_concurrency",
    "exit_unwind",
)


def parse_time(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    if "." in normalized:
        prefix, suffix = normalized.split(".", 1)
        offset_at = max(suffix.find("+"), suffix.find("-"))
        if offset_at >= 0:
            fraction, offset = suffix[:offset_at], suffix[offset_at:]
            normalized = f"{prefix}.{fraction[:6].ljust(6, '0')}{offset}"
    return datetime.fromisoformat(normalized).astimezone(timezone.utc)


def iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def logical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def bind(row: dict[str, Any]) -> dict[str, Any]:
    bound = dict(row)
    bound.pop("output_sha", None)
    bound["output_sha"] = logical_sha(bound)
    return bound


def evidence(kind: str, coverage: bool, value: Any, provenance: list[str], cutoff: str, reason: str | None = None) -> dict[str, Any]:
    return {
        "evidence_kind": kind,
        "coverage": coverage,
        "value": value,
        "provenance": provenance,
        "causal_cutoff": cutoff,
        "reason_code": reason,
    }


def load_events() -> tuple[list[dict[str, Any]], str]:
    uri = f"file:{EXECUTION_DB}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    rows = [dict(row) for row in conn.execute(
        """SELECT event_uid, ts_utc, event_type, order_id, trade_id, pair, side,
                  units, price, tp, sl, realized_pl_jpy, financing_jpy,
                  exit_reason, raw_json
             FROM execution_events
            ORDER BY ts_utc, event_uid"""
    )]
    conn.close()
    for row in rows:
        row["raw"] = json.loads(row.pop("raw_json"))
    source_digest = logical_sha([
        {key: row[key] for key in row if key != "raw"} | {"raw_sha": logical_sha(row["raw"])}
        for row in rows
    ])
    return rows, source_digest


def load_splits() -> dict[str, dict[str, str]]:
    payload = json.loads(REAL_PAYLOAD.read_text(encoding="utf-8"))
    result: dict[str, dict[str, str]] = defaultdict(dict)
    for row in payload["episode_records"]:
        if row["method"] == "ALL_TRADES":
            result[row["episode_id"]][row["window"]] = row["split"]
    return result


def discover_s5_files(episodes: list[dict[str, Any]]) -> tuple[dict[str, Path], dict[str, list[str]]]:
    files: list[tuple[str, datetime, datetime, Path]] = []
    roots = [REPO / "logs/replay/oanda_history", REPO / "logs/replay/oanda_prediction_truth"]
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*_S5_BA_*.jsonl.gz"):
            match = S5_RE.search(path.name)
            if not match or ".partial" in path.name:
                continue
            pair = path.name.split("_S5_BA_", 1)[0]
            start = datetime.strptime(match.group(1), "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
            end = datetime.strptime(match.group(2), "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
            files.append((pair, start, end, path))
    selected: dict[str, Path] = {}
    reasons: dict[str, list[str]] = defaultdict(list)
    for episode in episodes:
        decision = parse_time(episode["feature_at_utc"])
        candidates = [item for item in files if item[0] == episode["pair"] and item[1] <= decision <= item[2] + timedelta(seconds=5)]
        if not candidates:
            reasons[episode["episode_id"]].append("NO_OANDA_S5_FILE_CONTAINING_DECISION")
            continue
        candidates.sort(key=lambda item: ((item[2] - item[1]).total_seconds(), str(item[3])))
        selected[episode["episode_id"]] = candidates[0][3]
    return selected, reasons


def scan_selected_s5(selected: dict[str, Path], episodes: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_file: dict[Path, list[tuple[str, datetime]]] = defaultdict(list)
    for episode_id, path in selected.items():
        by_file[path].append((episode_id, parse_time(episodes[episode_id]["feature_at_utc"])))
    snapshots: dict[str, dict[str, Any]] = {}
    file_sha_cache: dict[Path, str] = {}
    for path, targets in sorted(by_file.items(), key=lambda item: str(item[0])):
        targets.sort(key=lambda item: item[1])
        history: deque[dict[str, Any]] = deque(maxlen=12)
        target_index = 0
        prior_bar_time: datetime | None = None
        file_sha_cache[path] = sha256(path)
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                bar_time = parse_time(row["time"])
                bar_end = bar_time + timedelta(seconds=5)
                while target_index < len(targets) and targets[target_index][1] < bar_end:
                    episode_id, decision = targets[target_index]
                    snapshots[episode_id] = make_s5_snapshot(history, decision, path, file_sha_cache[path])
                    target_index += 1
                if target_index >= len(targets):
                    break
                if not row.get("complete") or row.get("price") != "BA" or row.get("granularity") != "S5":
                    continue
                if prior_bar_time is not None and bar_time <= prior_bar_time:
                    continue
                prior_bar_time = bar_time
                history.append(row)
        while target_index < len(targets):
            episode_id, decision = targets[target_index]
            snapshots[episode_id] = make_s5_snapshot(history, decision, path, file_sha_cache[path])
            target_index += 1
    return snapshots


def make_s5_snapshot(history: deque[dict[str, Any]], decision: datetime, path: Path, source_sha: str) -> dict[str, Any]:
    if not history:
        return {"coverage": False, "reason": "NO_COMPLETE_CAUSAL_S5_BAR", "path": str(path.relative_to(REPO)), "source_sha": source_sha}
    last = history[-1]
    watermark = parse_time(last["time"]) + timedelta(seconds=5)
    age = (decision - watermark).total_seconds()
    spreads = [float(row["ask"]["c"]) - float(row["bid"]["c"]) for row in history]
    bid = float(last["bid"]["c"])
    ask = float(last["ask"]["c"])
    coverage = age >= 0.0 and age <= 15.0 and len(spreads) >= 6 and bid <= ask
    reason = None
    if age < 0:
        reason = "FUTURE_S5_WATERMARK_REJECTED"
    elif age > 15.0:
        reason = "STALE_CAUSAL_S5"
    elif len(spreads) < 6:
        reason = "INSUFFICIENT_SPREAD_BASELINE"
    elif bid > ask:
        reason = "BID_GT_ASK"
    return {
        "coverage": coverage,
        "reason": reason,
        "path": str(path.relative_to(REPO)),
        "source_sha": source_sha,
        "watermark": iso(watermark),
        "watermark_age_seconds": age,
        "bid": bid,
        "ask": ask,
        "mid": (bid + ask) / 2.0,
        "spread": ask - bid,
        "normal_spread_baseline": statistics.median(spreads) if len(spreads) >= 6 else None,
        "baseline_count": len(spreads),
    }


def raw_full_price(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if not row:
        return None
    full = row["raw"].get("fullPrice")
    if not isinstance(full, dict) or not full.get("timestamp"):
        return None
    bids = [{"price": float(level["price"]), "liquidity": int(level["liquidity"])} for level in full.get("bids", [])]
    asks = [{"price": float(level["price"]), "liquidity": int(level["liquidity"])} for level in full.get("asks", [])]
    if not bids or not asks:
        return None
    return {
        "timestamp": full["timestamp"],
        "bid": bids[0]["price"],
        "ask": asks[0]["price"],
        "bids": bids,
        "asks": asks,
        "closeout_bid": float(full["closeoutBid"]) if full.get("closeoutBid") is not None else None,
        "closeout_ask": float(full["closeoutAsk"]) if full.get("closeoutAsk") is not None else None,
    }


def depth_bound(levels: list[dict[str, Any]], units: int, side: str) -> tuple[bool, float | None, int]:
    remaining = abs(units)
    filled = 0
    worst = None
    for level in levels:
        take = min(remaining, int(level["liquidity"]))
        if take:
            filled += take
            remaining -= take
            worst = float(level["price"])
        if remaining == 0:
            break
    if remaining:
        return False, None, filled
    top = float(levels[0]["price"])
    bound = max(0.0, (worst - top) if side == "LONG" else (top - worst))
    return True, bound, filled


def position_snapshots(events: list[dict[str, Any]], episodes: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    relevant = [row for row in events if row["event_type"] in {"ORDER_FILLED", "TRADE_CLOSED", "TRADE_REDUCED"}]
    relevant.sort(key=lambda row: (parse_time(row["ts_utc"]), row["event_uid"]))
    decisions = sorted(((parse_time(row["feature_at_utc"]), row["episode_id"]) for row in episodes), key=lambda item: item[0])
    open_trades: dict[str, dict[str, Any]] = {}
    snapshots: dict[str, dict[str, Any]] = {}
    cursor = 0
    unknown_prefix = bool(relevant and relevant[0]["event_type"] == "TRADE_CLOSED")
    for decision, episode_id in decisions:
        while cursor < len(relevant) and parse_time(relevant[cursor]["ts_utc"]) < decision:
            row = relevant[cursor]
            raw = row["raw"]
            if row["event_type"] == "ORDER_FILLED" and raw.get("tradeOpened"):
                trade_id = str(raw["tradeOpened"]["tradeID"])
                open_trades[trade_id] = {
                    "trade_id": trade_id,
                    "pair": row["pair"],
                    "side": row["side"],
                    "units": abs(int(row["units"] or 0)),
                    "opened_at_utc": row["ts_utc"],
                }
            elif row["event_type"] in {"TRADE_CLOSED", "TRADE_REDUCED"}:
                for closed in raw.get("tradesClosed", []):
                    open_trades.pop(str(closed.get("tradeID")), None)
                for reduced in raw.get("tradeReduced", []) if isinstance(raw.get("tradeReduced"), list) else []:
                    trade_id = str(reduced.get("tradeID"))
                    if trade_id in open_trades:
                        open_trades[trade_id]["units"] = abs(int(float(reduced.get("units", open_trades[trade_id]["units"]))))
            cursor += 1
        values = sorted(open_trades.values(), key=lambda row: row["trade_id"])
        currency_abs: Counter[str] = Counter()
        for item in values:
            base, quote = item["pair"].split("_", 1)
            signed = item["units"] if item["side"] == "LONG" else -item["units"]
            currency_abs[base] += signed
            currency_abs[quote] -= signed
        snapshots[episode_id] = {
            "existing_positions": values,
            "concurrency": len(values),
            "currency_net_units_proxy": dict(sorted(currency_abs.items())),
            "prefix_complete": not unknown_prefix,
        }
    return snapshots


def build_row(
    episode: dict[str, Any], accepted: dict[str, Any] | None, fill: dict[str, Any] | None,
    close: dict[str, Any] | None, s5: dict[str, Any] | None, portfolio: dict[str, Any],
    splits: dict[str, str], source_sha: str,
) -> dict[str, Any]:
    decision = episode["feature_at_utc"]
    decision_dt = parse_time(decision)
    exact_order = bool(accepted and accepted["event_uid"] == episode["episode_id"] and accepted["order_id"] == episode["order_id"] and accepted["ts_utc"] == decision)
    accepted_raw = accepted["raw"] if accepted else {}
    candidate_value = None
    if accepted:
        candidate_value = {
            "order_type": accepted_raw.get("type"),
            "entry_price": float(accepted_raw["price"]) if accepted_raw.get("price") is not None else None,
            "units_signed": int(accepted_raw.get("units", accepted["units"] or 0)),
            "time_in_force": accepted_raw.get("timeInForce"),
            "partial_fill_policy": accepted_raw.get("partialFill"),
            "tp": accepted["tp"],
            "sl": accepted["sl"],
        }
    candidate = evidence(
        "ACTUAL" if exact_order else "MISSING", exact_order, candidate_value,
        [accepted["event_uid"]] if accepted else [], decision,
        None if exact_order else "EXACT_ORDER_ACCEPTED_JOIN_MISSING",
    )

    if s5:
        price_value = {key: s5.get(key) for key in ("bid", "ask", "mid", "spread", "normal_spread_baseline", "watermark", "watermark_age_seconds", "baseline_count")}
        price_value["pricing_source"] = "OANDA_HISTORICAL_S5_BA"
        price_value["executable_entry_side_price"] = s5.get("ask") if episode["side"] == "LONG" else s5.get("bid")
        pricing = evidence(
            "RECONSTRUCTED_CAUSAL" if s5.get("coverage") else "MISSING",
            bool(s5.get("coverage")), price_value,
            [f"{s5.get('path')}#{s5.get('source_sha')}"] if s5.get("path") else [], decision, s5.get("reason"),
        )
    else:
        pricing = evidence("MISSING", False, None, [], decision, "NO_OANDA_S5_SOURCE")

    full = raw_full_price(fill)
    causal_full = bool(full and parse_time(full["timestamp"]) <= decision_dt and fill and parse_time(fill["ts_utc"]) <= decision_dt)
    fillability_value = None
    fillability_ok = False
    slip_bound = None
    if causal_full and candidate_value and candidate_value["order_type"] == "MARKET_ORDER":
        levels = full["asks"] if episode["side"] == "LONG" else full["bids"]
        fillability_ok, slip_bound, depth_units = depth_bound(levels, abs(int(episode["units"])), episode["side"])
        fillability_value = {
            "window_start_utc": full["timestamp"],
            "window_end_utc": decision,
            "top_bid": full["bid"],
            "top_ask": full["ask"],
            "available_depth_units": depth_units,
            "gap": False,
            "partial": not fillability_ok,
            "no_fill": not fillability_ok,
        }
    fillability = evidence(
        "RECONSTRUCTED_CAUSAL" if fillability_ok else "MISSING", fillability_ok, fillability_value,
        [fill["event_uid"] + ":fullPrice"] if causal_full and fill else [], decision,
        None if fillability_ok else ("NON_MARKET_ORDER_FUTURE_TRIGGER_REQUIRED" if candidate_value and candidate_value["order_type"] != "MARKET_ORDER" else "CAUSAL_DEPTH_SNAPSHOT_MISSING"),
    )
    slippage = evidence(
        "CONSERVATIVE_BOUND" if slip_bound is not None else "MISSING", slip_bound is not None,
        {"model": "causal_depth_worst_level_minus_top", "bound_price_units": slip_bound} if slip_bound is not None else None,
        fillability["provenance"], decision, None if slip_bound is not None else "CAUSAL_SLIPPAGE_BOUND_MISSING",
    )
    fee = evidence("MISSING", False, None, [], decision, "DECISION_TIME_FEE_SCHEDULE_NOT_ARCHIVED")
    financing = evidence("MISSING", False, None, [], decision, "DECISION_TIME_FINANCING_SCHEDULE_NOT_ARCHIVED")
    costs_coverage = all(item["coverage"] for item in (slippage, fee, financing))

    portfolio_value = {
        **portfolio,
        "margin_available": None,
        "margin_used": None,
        "margin_rate": None,
        "candidate_initial_margin_required": None,
    }
    portfolio_evidence = evidence(
        "MISSING", False, portfolio_value,
        ["data/execution_ledger.db:causal_event_prefix"], decision,
        "DECISION_TIME_MARGIN_AVAILABLE_USED_RATE_MISSING",
    )

    exit_policy_ok = exact_order and accepted and accepted["tp"] is not None and accepted["sl"] is not None
    exit_policy = evidence(
        "ACTUAL" if exit_policy_ok else "MISSING", bool(exit_policy_ok),
        {
            "tp": accepted["tp"] if accepted else None,
            "sl": accepted["sl"] if accepted else None,
            "exit_horizon": None,
            "dual_leg_ordering": "NOT_APPLICABLE_BASELINE_SINGLE_LEG",
        },
        [accepted["event_uid"]] if accepted else [], decision,
        None if exit_policy_ok else "DECISION_TIME_EXIT_POLICY_INCOMPLETE",
    )
    unwind = evidence("MISSING", False, None, [], decision, "DECISION_TIME_EXECUTABLE_UNWIND_EVIDENCE_MISSING")
    exit_unwind_coverage = exit_policy["coverage"] and unwind["coverage"]

    observed_fill = None
    if fill:
        fill_full = raw_full_price(fill)
        observed_fill = {
            "evaluation_only": True,
            "event_uid": fill["event_uid"],
            "fill_at_utc": fill["ts_utc"],
            "fill_price": fill["price"],
            "full_price_watermark": fill_full["timestamp"] if fill_full else None,
            "bid": fill_full["bid"] if fill_full else None,
            "ask": fill_full["ask"] if fill_full else None,
            "commission": float(fill["raw"].get("commission", 0.0)) if fill["raw"].get("commission") is not None else None,
            "half_spread_cost_jpy": float(fill["raw"].get("halfSpreadCost", 0.0)) if fill["raw"].get("halfSpreadCost") is not None else None,
            "initial_margin_required_jpy": float(fill["raw"].get("tradeOpened", {}).get("initialMarginRequired")) if fill["raw"].get("tradeOpened", {}).get("initialMarginRequired") is not None else None,
            "delay_seconds": (parse_time(fill["ts_utc"]) - decision_dt).total_seconds(),
        }
    observed_close = None
    if close:
        close_full = raw_full_price(close)
        observed_close = {
            "evaluation_only": True,
            "event_uid": close["event_uid"],
            "close_at_utc": close["ts_utc"],
            "exit_price": close["price"],
            "exit_bid": close_full["bid"] if close_full else None,
            "exit_ask": close_full["ask"] if close_full else None,
            "terminal_reason": close["exit_reason"],
            "realized_pl_jpy": close["realized_pl_jpy"],
            "financing_jpy": close["financing_jpy"],
        }

    stages = {
        "pricing": pricing["coverage"],
        "candidate_order": candidate["coverage"],
        "fillability": fillability["coverage"],
        "slippage_fee_financing": costs_coverage,
        "margin_exposure_concurrency": portfolio_evidence["coverage"],
        "exit_unwind": exit_unwind_coverage,
    }
    row = {
        "decision_id": episode["episode_id"],
        "decision_time": decision,
        "pair": episode["pair"],
        "side": episode["side"],
        "horizon": episode.get("forecast_horizon_min"),
        "source_sha": source_sha,
        "splits": {window: splits.get(window, "OUTSIDE_OR_EMBARGO") for window in WINDOWS},
        "pricing": pricing,
        "candidate_order": candidate,
        "fillability": fillability,
        "costs": {"slippage": slippage, "fee": fee, "financing": financing, "coverage": costs_coverage},
        "portfolio_margin": portfolio_evidence,
        "exit_unwind": {"policy": exit_policy, "unwind_validity": unwind, "coverage": exit_unwind_coverage},
        "stage_coverage": stages,
        "strict_eligible": all(stages.values()),
        "strict_ineligibility_reasons": [stage for stage, ok in stages.items() if not ok],
        "observed_execution": {"evaluation_only": True, "fill": observed_fill, "close": observed_close},
    }
    return bind(row)


def profit_factor(values: list[float]) -> float | None:
    gain = sum(value for value in values if value > 0)
    loss = -sum(value for value in values if value < 0)
    return gain / loss if loss else None


def max_drawdown(values: list[float]) -> float:
    equity = peak = worst = 0.0
    for value in values:
        equity += value
        peak = max(peak, equity)
        worst = max(worst, peak - equity)
    return worst


def bootstrap_lcb(deltas: list[float], seed_offset: int) -> float | None:
    if not deltas:
        return None
    rng = random.Random(SEED + seed_offset)
    means = sorted(statistics.fmean(rng.choice(deltas) for _ in deltas) for _ in range(BOOTSTRAPS))
    return means[int(0.025 * (len(means) - 1))]


def metrics(rows: list[dict[str, Any]], selected: dict[str, bool], seed_offset: int) -> dict[str, Any]:
    if not rows:
        return {
            "available": 0,
            "selected": 0,
            "decisions_changed_vs_all_trades": 0,
            "after_cost_net_jpy": None,
            "all_trades_net_jpy": None,
            "incremental_net_jpy": None,
            "paired_bootstrap_lcb_jpy": None,
            "profit_factor": None,
            "max_drawdown_jpy": None,
            "all_trades_max_drawdown_jpy": None,
            "margin_coverage": None,
            "fill_validity": None,
            "unwind_validity": None,
            "sample_coverage": None,
        }
    actual = [float(row["net_jpy"]) for row in rows]
    applied = [value if selected.get(row["episode_id"], False) else 0.0 for row, value in zip(rows, actual)]
    deltas = [candidate - baseline for candidate, baseline in zip(applied, actual)]
    selected_count = sum(bool(selected.get(row["episode_id"], False)) for row in rows)
    return {
        "available": len(rows),
        "selected": selected_count,
        "decisions_changed_vs_all_trades": len(rows) - selected_count,
        "after_cost_net_jpy": sum(applied),
        "all_trades_net_jpy": sum(actual),
        "incremental_net_jpy": sum(deltas),
        "paired_bootstrap_lcb_jpy": bootstrap_lcb(deltas, seed_offset),
        "profit_factor": profit_factor(applied),
        "max_drawdown_jpy": max_drawdown(applied),
        "all_trades_max_drawdown_jpy": max_drawdown(actual),
        "margin_coverage": 1.0 if selected_count else None,
        "fill_validity": 1.0 if selected_count else None,
        "unwind_validity": 1.0 if selected_count else None,
        "sample_coverage": selected_count / len(rows) if rows else None,
    }


def evaluate(episodes: list[dict[str, Any]], ledger: dict[str, dict[str, Any]], splits: dict[str, dict[str, str]]) -> dict[str, Any]:
    prior = json.loads(PRIOR_REPORT.read_text(encoding="utf-8"))
    output: dict[str, Any] = {}
    for index, window in enumerate(WINDOWS):
        validation = [row for row in episodes if splits.get(row["episode_id"], {}).get(window) == "VALIDATION"]
        strict = [row for row in validation if ledger[row["episode_id"]]["strict_eligible"]]
        candidates = prior["fusion"][window]["candidates"]
        single_predictions = candidates["single_statistical"].get("predictions", {})
        fused_predictions = candidates["calibrated_weighted_vote"].get("predictions", {})
        single_selected = {row["episode_id"]: single_predictions.get(row["episode_id"], {}).get("lower", -math.inf) > 0 for row in strict}
        fused_selected = {row["episode_id"]: fused_predictions.get(row["episode_id"], {}).get("lower", -math.inf) > 0 for row in strict}
        all_selected = {row["episode_id"]: True for row in strict}
        edge_positive_all = [episode_id for episode_id, pred in fused_predictions.items() if pred.get("lower", -math.inf) > 0]
        output[window] = {
            "validation_total": len(validation),
            "strict_eligible": len(strict),
            "fusion_prediction_coverage": len(fused_predictions),
            "fusion_edge_positive_before_execution_gate": len(edge_positive_all),
            "fusion_edge_positive_ids": edge_positive_all,
            "edge_positive_blocked_by_execution_evidence": sum(not ledger[episode_id]["strict_eligible"] for episode_id in edge_positive_all),
            "all_trades_same_eligible_cohort": metrics(strict, all_selected, index * 100 + 1),
            "single_statistical_same_eligible_cohort": metrics(strict, single_selected, index * 100 + 2),
            "fusion_same_eligible_cohort": metrics(strict, fused_selected, index * 100 + 3),
            "status": "NOT_EVALUABLE_NO_STRICT_ELIGIBLE_EPISODES" if not strict else "EVALUATED",
        }
    return output


def coverage_report(episodes: list[dict[str, Any]], ledger: dict[str, dict[str, Any]], splits: dict[str, dict[str, str]], evaluation: dict[str, Any]) -> dict[str, Any]:
    overall = {stage: sum(row["stage_coverage"][stage] for row in ledger.values()) for stage in REQUIRED_STAGES}
    by_window: dict[str, Any] = {}
    for window in WINDOWS:
        by_window[window] = {}
        for split in ("TRAIN", "VALIDATION", "OUTSIDE_OR_EMBARGO"):
            ids = [row["episode_id"] for row in episodes if splits.get(row["episode_id"], {}).get(window, "OUTSIDE_OR_EMBARGO") == split]
            by_window[window][split] = {
                "episodes": len(ids),
                "strict_eligible": sum(ledger[episode_id]["strict_eligible"] for episode_id in ids),
                "stages": {stage: sum(ledger[episode_id]["stage_coverage"][stage] for episode_id in ids) for stage in REQUIRED_STAGES},
            }
    reason_counts = Counter(reason for row in ledger.values() for reason in row["strict_ineligibility_reasons"])
    full_net = sum(float(row["net_jpy"]) for row in episodes)
    pricing_order = [row for row in episodes if ledger[row["episode_id"]]["stage_coverage"]["pricing"] and ledger[row["episode_id"]]["stage_coverage"]["candidate_order"]]
    pricing_order_net = sum(float(row["net_jpy"]) for row in pricing_order)
    return {
        "contract": "DECISION_TIME_EXECUTION_EVIDENCE_LEDGER_V1",
        "episode_count": len(episodes),
        "overall_stage_coverage": overall,
        "strict_eligible": sum(row["strict_eligible"] for row in ledger.values()),
        "strict_ineligibility_reason_counts": dict(reason_counts),
        "by_window_split": by_window,
        "observed_execution_boundary": {
            "fills": sum(row["observed_execution"]["fill"] is not None for row in ledger.values()),
            "closes": sum(row["observed_execution"]["close"] is not None for row in ledger.values()),
            "never_used_as_decision_input": True,
        },
        "coverage_selection_bias_diagnostic": {
            "full_count": len(episodes),
            "full_net_jpy": full_net,
            "pricing_and_order_count": len(pricing_order),
            "pricing_and_order_net_jpy": pricing_order_net,
            "count_fraction": len(pricing_order) / len(episodes),
            "net_per_episode_full": full_net / len(episodes),
            "net_per_episode_pricing_and_order": pricing_order_net / len(pricing_order) if pricing_order else None,
            "warning": "outcomes are used only to diagnose selection bias, never to admit evidence or tune rules",
        },
        "evaluation": evaluation,
        "holdout_read": False,
        "live_paper_broker_order_deploy_touched": False,
    }


def rerun_decisions(episodes: list[dict[str, Any]], ledger: dict[str, dict[str, Any]], splits: dict[str, dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prior_report = json.loads(PRIOR_REPORT.read_text(encoding="utf-8"))
    predictions = prior_report["fusion"]["QUADRUPLE_64D"]["candidates"]["calibrated_weighted_vote"].get("predictions", {})
    old = {row["decision_id"]: row for row in read_jsonl(PRIOR_FUSED)}
    rows = []
    for episode in episodes:
        episode_id = episode["episode_id"]
        prediction = predictions.get(episode_id)
        split = splits.get(episode_id, {}).get("QUADRUPLE_64D", "OUTSIDE_OR_EMBARGO")
        admissible = ledger[episode_id]["strict_eligible"]
        edge_positive = bool(prediction and prediction.get("lower", -math.inf) > 0)
        if split != "VALIDATION":
            action, constraint, reason = "WAIT", "OUTSIDE_FROZEN_VALIDATION", "no evaluation action outside frozen validation"
        elif prediction is None:
            action, constraint, reason = "WAIT", "EDGE_FAMILY_COVERAGE", "two independent edge families are unavailable"
        elif not edge_positive:
            action, constraint, reason = "SKIP", "TRAIN_FIXED_EDGE_LCB_NON_POSITIVE", "expected-after-cost lower bound is not positive"
        elif not admissible:
            action, constraint, reason = "WAIT", "EXECUTION_EVIDENCE_INCOMPLETE", ",".join(ledger[episode_id]["strict_ineligibility_reasons"])
        else:
            action, constraint, reason = "TRADE", None, None
        row = bind({
            "decision_id": episode_id,
            "decision_time": episode["feature_at_utc"],
            "action": action,
            "pair": episode["pair"],
            "side": episode["side"],
            "horizon": episode.get("forecast_horizon_min"),
            "entry_zone": {"order_type": ledger[episode_id]["candidate_order"]["value"].get("order_type") if ledger[episode_id]["candidate_order"]["value"] else None, "entry_price": ledger[episode_id]["candidate_order"]["value"].get("entry_price") if ledger[episode_id]["candidate_order"]["value"] else None},
            "target_or_path": episode.get("tp"),
            "invalidation": episode.get("sl"),
            "exit_or_unwind_policy": ledger[episode_id]["exit_unwind"]["policy"]["value"],
            "size_cap": abs(int(episode["units"])) if action == "TRADE" else 0,
            "confidence": None,
            "prediction_interval": [prediction["lower"], prediction["upper"]] if prediction else None,
            "expected_after_cost": prediction["point"] if prediction else None,
            "worst_case_dd_margin": {"dd": None, "margin_available": None, "margin_used": None, "margin_rate": None},
            "supporting_families": ["technical", "statistical_ml"] if prediction else [],
            "dissenting_families": ["execution", "portfolio", "risk_exit"] if not admissible else [],
            "decisive_constraint": constraint,
            "abstain_reason": reason,
            "evidence_ledger_sha": ledger[episode_id]["output_sha"],
            "input_lineage": ["frozen FULL_INFERENCE_ENSEMBLE_V1 prediction", "DECISION_TIME_EXECUTION_EVIDENCE_LEDGER_V1"],
            "holdout_read": False,
        })
        rows.append(row)
    changed_rows = [row for row in rows if old.get(f"fused:{row['decision_id']}", {}).get("action") != row["action"]]
    nonvalidation_corrections = sum(
        splits.get(row["decision_id"], {}).get("QUADRUPLE_64D") != "VALIDATION"
        and old.get(f"fused:{row['decision_id']}", {}).get("action") == "SKIP"
        and row["action"] == "WAIT"
        for row in changed_rows
    )
    evidence_driven_changes = sum(
        splits.get(row["decision_id"], {}).get("QUADRUPLE_64D") == "VALIDATION"
        for row in changed_rows
    )
    summary = {
        "counts": dict(Counter(row["action"] for row in rows)),
        "validation_counts": dict(Counter(row["action"] for row in rows if splits.get(row["decision_id"], {}).get("QUADRUPLE_64D") == "VALIDATION")),
        "action_changes_vs_prior_fused": len(changed_rows),
        "evidence_driven_action_changes": evidence_driven_changes,
        "prior_nonvalidation_false_skip_corrections": nonvalidation_corrections,
        "change_explanation": "the prior renderer marked two-family non-validation rows SKIP despite having no validation prediction; the rerun makes them WAIT and does not use this correction as profitability evidence",
        "trade_count": sum(row["action"] == "TRADE" for row in rows),
    }
    return rows, summary


def forward_contract() -> dict[str, Any]:
    return {
        "contract": "DECISION_TIME_EXECUTION_EVIDENCE_FORWARD_ACQUISITION_V1",
        "mode": "append-only research evidence capture before any candidate order",
        "key": "decision_id",
        "required_pre_decision_receipts": [
            "OANDA pricing snapshot with bid/ask/depth/source timestamp",
            "normal-spread baseline source window and watermark",
            "candidate order intent with order type/entry/units/TP/SL",
            "versioned slippage bound, fee schedule, and financing schedule",
            "account margin available/used/rate and candidate margin requirement",
            "open trades, pair/theme/currency exposure, and concurrency",
            "explicit exit/unwind policy including dual-leg ordering when applicable",
            "input SHA set and causal cutoff"
        ],
        "required_post_decision_receipts_evaluation_only": [
            "accept/reject/partial/no-fill",
            "fill bid/ask/fullVWAP/slippage/fees/initial margin",
            "all exit or dual-unwind legs in broker transaction order",
            "realized financing and after-cost outcome"
        ],
        "fail_closed": [
            "missing margin is never zero or sufficient",
            "post-decision receipts never backfill pre-decision inputs",
            "Dukascopy remains feature-only",
            "no mid fill, interpolation, M1 substitution, or hindsight exit"
        ],
        "activation": "future research cohorts only; no retroactive admission of the frozen 251",
        "live_permission": False,
    }


def main() -> None:
    episodes = [row for row in read_jsonl(EPISODES) if row.get("label_status") == "ACTUAL_AFTER_COST"]
    if len(episodes) != 251 or len({row["episode_id"] for row in episodes}) != 251:
        raise RuntimeError("frozen cohort must contain exactly 251 unique actual-after-cost episodes")
    source_sha = sha256(EPISODES)
    by_id = {row["episode_id"]: row for row in episodes}
    events, event_source_sha = load_events()
    accepted = {row["event_uid"]: row for row in events if row["event_type"] == "ORDER_ACCEPTED"}
    fills_by_order = {str(row["order_id"]): row for row in events if row["event_type"] == "ORDER_FILLED" and row.get("order_id")}
    closes_by_trade = {str(row["trade_id"]): row for row in events if row["event_type"] == "TRADE_CLOSED" and row.get("trade_id")}
    splits = load_splits()
    selected, discovery_reasons = discover_s5_files(episodes)
    snapshots = scan_selected_s5(selected, by_id)
    portfolios = position_snapshots(events, episodes)
    rows = []
    for episode in episodes:
        episode_id = episode["episode_id"]
        snapshot = snapshots.get(episode_id)
        if snapshot is None and discovery_reasons.get(episode_id):
            snapshot = {"coverage": False, "reason": discovery_reasons[episode_id][0]}
        rows.append(build_row(
            episode,
            accepted.get(episode_id),
            fills_by_order.get(str(episode["order_id"])),
            closes_by_trade.get(str(episode["trade_id"])),
            snapshot,
            portfolios[episode_id],
            splits.get(episode_id, {}),
            source_sha,
        ))
    ledger = {row["decision_id"]: row for row in rows}
    evaluation = evaluate(episodes, ledger, splits)
    report = coverage_report(episodes, ledger, splits, evaluation)
    rerun, rerun_summary = rerun_decisions(episodes, ledger, splits)
    report["fused_decision_rerun"] = rerun_summary
    report["causal_diagnosis"] = {
        "edge": "no TRAIN-fixed weighted-fusion lower bound is positive in the 64-day validation prediction cohort" if evaluation["QUADRUPLE_64D"]["fusion_edge_positive_before_execution_gate"] == 0 else "some edge-positive candidates exist before evidence gating",
        "pipeline": "zero episodes have complete decision-time cost, margin, and executable-unwind evidence",
        "separation": "edge insufficiency is established only on modeled two-family rows; pipeline insufficiency prevents strict executable evaluation on the full validation cohort",
    }
    write_jsonl(HERE / "evidence_ledger_v1.jsonl", rows)
    write_json(HERE / "coverage_report_v1.json", report)
    write_jsonl(HERE / "fused_decisions_rerun_v1.jsonl", rerun)
    write_json(HERE / "forward_acquisition_contract_v1.json", forward_contract())
    manifest = {
        "contract": "DECISION_TIME_EXECUTION_EVIDENCE_LEDGER_V1",
        "source_sha256": {
            str(EPISODES.relative_to(REPO)): source_sha,
            "data/execution_ledger.db:logical_event_snapshot": event_source_sha,
            str(PRIOR_REPORT.relative_to(REPO)): sha256(PRIOR_REPORT),
            str(PRIOR_FUSED.relative_to(REPO)): sha256(PRIOR_FUSED),
            str(REAL_PAYLOAD.relative_to(REPO)): sha256(REAL_PAYLOAD),
        },
        "outputs": {},
        "holdout_read": False,
        "live_paper_broker_order_deploy_touched": False,
    }
    for name in ("preregister_v1.json", "evidence_ledger_v1.jsonl", "coverage_report_v1.json", "fused_decisions_rerun_v1.jsonl", "forward_acquisition_contract_v1.json"):
        manifest["outputs"][name] = sha256(HERE / name)
    write_json(HERE / "run_manifest_v1.json", manifest)
    print(json.dumps({
        "episodes": len(rows),
        "strict_eligible": report["strict_eligible"],
        "stage_coverage": report["overall_stage_coverage"],
        "rerun": rerun_summary,
        "verdict": "PIPELINE_INSUFFICIENT_AND_MODELED_EDGE_INSUFFICIENT",
    }, sort_keys=True))


if __name__ == "__main__":
    main()
