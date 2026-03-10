#!/usr/bin/env python3
"""Mine recent loser clusters from closed-trade entry context."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sqlite3
import sys
import tempfile
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.strategy_tags import resolve_strategy_tag


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_json_loads(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return {}
    text = raw.strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        delete=False,
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    ) as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
        tmp_path = Path(fh.name)
    tmp_path.replace(path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False))
        fh.write("\n")


def _bin(value: float | None, bounds: list[float], labels: list[str]) -> str:
    if value is None or math.isnan(value):
        return "unknown"
    for bound, label in zip(bounds, labels):
        if value <= bound:
            return label
    return labels[-1]


def _extract_nested_number(payload: dict[str, Any], *path: str) -> float | None:
    node: Any = payload
    for key in path:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    try:
        value = float(node)
    except Exception:
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def _feature_snapshot(entry_thesis: dict[str, Any]) -> dict[str, Any]:
    technical_context = entry_thesis.get("technical_context") if isinstance(entry_thesis.get("technical_context"), dict) else {}
    ticks = technical_context.get("ticks") if isinstance(technical_context.get("ticks"), dict) else {}
    factors = technical_context.get("factors") if isinstance(technical_context.get("factors"), dict) else {}
    m1 = factors.get("M1") if isinstance(factors.get("M1"), dict) else {}
    side = str(entry_thesis.get("side") or "").strip().lower()
    rsi = _extract_nested_number(entry_thesis, "rsi")
    if rsi is None:
        rsi = _extract_nested_number(m1, "rsi")
    adx = _extract_nested_number(entry_thesis, "adx")
    if adx is None:
        adx = _extract_nested_number(m1, "adx")
    ma_gap_pips = _extract_nested_number(entry_thesis, "ma_gap_pips")
    range_score = _extract_nested_number(entry_thesis, "range_score")
    chop_score = _extract_nested_number(entry_thesis, "chop_score")
    spread_pips = _extract_nested_number(entry_thesis, "spread_pips")
    if spread_pips is None:
        spread_pips = _extract_nested_number(ticks, "spread_pips")
    return {
        "side": side or "unknown",
        "rsi_bin": _bin(rsi, [35.0, 45.0, 55.0, 65.0], ["rsi_00_le_35", "rsi_01_35_45", "rsi_02_45_55", "rsi_03_55_65", "rsi_04_gt_65"]),
        "adx_bin": _bin(adx, [15.0, 25.0, 35.0], ["adx_00_le_15", "adx_01_15_25", "adx_02_25_35", "adx_03_gt_35"]),
        "ma_gap_bin": _bin(ma_gap_pips, [-1.0, 0.0, 1.0, 3.0], ["magap_00_le_-1", "magap_01_-1_0", "magap_02_0_1", "magap_03_1_3", "magap_04_gt_3"]),
        "range_bin": _bin(range_score, [0.25, 0.45, 0.65], ["range_00_le_025", "range_01_025_045", "range_02_045_065", "range_03_gt_065"]),
        "chop_bin": _bin(chop_score, [0.35, 0.55, 0.75], ["chop_00_le_035", "chop_01_035_055", "chop_02_055_075", "chop_03_gt_075"]),
        "spread_bin": _bin(spread_pips, [0.40, 0.80, 1.20], ["spread_00_le_040", "spread_01_040_080", "spread_02_080_120", "spread_03_gt_120"]),
    }


def _cluster_key(features: dict[str, Any]) -> str:
    return "|".join(
        [
            str(features.get("side") or "unknown"),
            str(features.get("rsi_bin") or "unknown"),
            str(features.get("adx_bin") or "unknown"),
            str(features.get("ma_gap_bin") or "unknown"),
            str(features.get("range_bin") or "unknown"),
            str(features.get("chop_bin") or "unknown"),
            str(features.get("spread_bin") or "unknown"),
        ]
    )


def _fetch_trade_rows(trades_db: Path, *, lookback_days: int) -> list[dict[str, Any]]:
    if not trades_db.exists():
        return []
    con = sqlite3.connect(f"file:{trades_db}?mode=ro", uri=True, timeout=8.0, isolation_level=None)
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            """
            SELECT
              ticket_id,
              COALESCE(NULLIF(strategy_tag, ''), strategy) AS strategy_key,
              pocket,
              units,
              pl_pips,
              realized_pl,
              entry_thesis
            FROM trades
            WHERE close_time IS NOT NULL
              AND julianday(close_time) >= julianday('now', ?)
            """,
            (f"-{max(1, int(lookback_days))} day",),
        ).fetchall()
    finally:
        con.close()
    return [dict(row) for row in rows]


def build_loser_clusters(rows: list[dict[str, Any]], *, min_cluster_size: int, top_k: int) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        strategy_tag = resolve_strategy_tag(str(row.get("strategy_key") or "").strip()) or str(row.get("strategy_key") or "").strip()
        if not strategy_tag:
            continue
        entry_thesis = _safe_json_loads(row.get("entry_thesis"))
        features = _feature_snapshot(entry_thesis)
        if features["side"] == "unknown":
            features["side"] = "long" if _safe_float(row.get("units"), 0.0) >= 0.0 else "short"
        groups[(strategy_tag, _cluster_key(features))].append(
            {
                "strategy_tag": strategy_tag,
                "pocket": str(row.get("pocket") or "").strip() or "unknown",
                "pl_pips": _safe_float(row.get("pl_pips"), 0.0),
                "realized_pl": _safe_float(row.get("realized_pl"), 0.0),
                "features": features,
            }
        )

    per_strategy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    all_clusters: list[dict[str, Any]] = []
    for (strategy_tag, cluster_key), samples in groups.items():
        if len(samples) < max(1, min_cluster_size):
            continue
        pl_values = [float(sample["pl_pips"]) for sample in samples]
        realized_values = [float(sample["realized_pl"]) for sample in samples]
        losses = sum(1 for value in pl_values if value < 0.0)
        win_rate = sum(1 for value in pl_values if value > 0.0) / max(1, len(pl_values))
        avg_pips = sum(pl_values) / float(len(pl_values))
        avg_realized = sum(realized_values) / float(len(realized_values))
        loss_rate = losses / float(len(pl_values))
        sample_scale = min(1.0, len(samples) / float(max(min_cluster_size, 6)))
        severity = (
            0.45 * loss_rate
            + 0.35 * min(1.0, max(0.0, -avg_pips) / 2.5)
            + 0.20 * min(1.0, max(0.0, -avg_realized) / 3.0)
        ) * sample_scale
        severity = _clamp(severity, 0.0, 1.0)
        units_multiplier = 1.0 - min(0.22, severity * 0.22)
        probability_offset = -min(0.035, severity * 0.035)
        cluster = {
            "strategy_tag": strategy_tag,
            "cluster_key": cluster_key,
            "samples": len(samples),
            "loss_rate": round(loss_rate, 6),
            "win_rate": round(win_rate, 6),
            "avg_pips": round(avg_pips, 6),
            "avg_realized_jpy": round(avg_realized, 6),
            "severity": round(severity, 6),
            "features": dict(samples[0]["features"]),
            "suggestion": {
                "action": "tighten_canary",
                "units_multiplier": round(_clamp(units_multiplier, 0.78, 1.0), 4),
                "probability_offset": round(_clamp(probability_offset, -0.04, 0.0), 4),
            },
        }
        per_strategy[strategy_tag].append(cluster)
        all_clusters.append(cluster)

    for clusters in per_strategy.values():
        clusters.sort(key=lambda item: (-float(item["severity"]), -int(item["samples"]), item["cluster_key"]))
    all_clusters.sort(key=lambda item: (-float(item["severity"]), -int(item["samples"]), item["cluster_key"]))

    strategy_summary: dict[str, Any] = {}
    for strategy_tag, clusters in sorted(per_strategy.items()):
        worst = clusters[0]
        strategy_summary[strategy_tag] = {
            "cluster_count": len(clusters),
            "worst_severity": worst["severity"],
            "suggestion": worst["suggestion"],
            "clusters": clusters[: max(1, int(top_k))],
        }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "strategies": strategy_summary,
        "top_clusters": all_clusters[: max(1, int(top_k))],
    }


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def main() -> None:
    ap = argparse.ArgumentParser(description="Mine loser clusters from recent trades")
    ap.add_argument("--trades-db", default="logs/trades.db")
    ap.add_argument("--output", default="logs/loser_cluster_latest.json")
    ap.add_argument("--history", default="logs/loser_cluster_history.jsonl")
    ap.add_argument("--lookback-days", type=int, default=7)
    ap.add_argument("--min-cluster-size", type=int, default=4)
    ap.add_argument("--top-k", type=int, default=8)
    args = ap.parse_args()

    rows = _fetch_trade_rows(Path(args.trades_db).resolve(), lookback_days=max(1, int(args.lookback_days)))
    payload = build_loser_clusters(rows, min_cluster_size=max(1, int(args.min_cluster_size)), top_k=max(1, int(args.top_k)))
    payload["lookback_days"] = max(1, int(args.lookback_days))
    payload["min_cluster_size"] = max(1, int(args.min_cluster_size))
    _write_json_atomic(Path(args.output).resolve(), payload)
    _append_jsonl(Path(args.history).resolve(), payload)
    print(
        f"[loser-cluster-worker] wrote {Path(args.output).resolve()} "
        f"strategies={len(payload['strategies'])} clusters={len(payload['top_clusters'])}"
    )


if __name__ == "__main__":
    main()
