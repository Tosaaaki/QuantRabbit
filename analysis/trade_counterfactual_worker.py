#!/usr/bin/env python3
"""Counterfactual trade review worker.

This worker analyzes recent closed trades and produces "what should have been
done" suggestions with conservative certainty scoring.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import sqlite3
import tempfile
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.getenv(name, str(default))))
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return float(default)


def _resolve_path(value: str) -> Path:
    path = Path(str(value).strip())
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


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


def _parse_iso(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _to_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if math.isnan(out):
        return None
    return out


def _normalize_probability(value: Any) -> float | None:
    out = _to_float(value)
    if out is None:
        return None
    if out > 1.0:
        if out <= 100.0:
            out = out / 100.0
        else:
            return None
    if out < 0.0:
        return 0.0
    if out > 1.0:
        return 1.0
    return out


def _extract_entry_probability(thesis: dict[str, Any]) -> float | None:
    for key in ("entry_probability", "entry_probability_fused", "confidence"):
        if key not in thesis:
            continue
        out = _normalize_probability(thesis.get(key))
        if out is not None:
            return out
    return None


def _spread_bin(spread_pips: float | None) -> str:
    if spread_pips is None:
        return "unknown"
    if spread_pips <= 0.40:
        return "s00_le_0.40"
    if spread_pips <= 0.80:
        return "s01_0.40_0.80"
    if spread_pips <= 1.20:
        return "s02_0.80_1.20"
    return "s03_gt_1.20"


def _prob_bin(prob: float | None) -> str:
    if prob is None:
        return "unknown"
    if prob < 0.50:
        return "p00_lt_0.50"
    if prob < 0.60:
        return "p01_0.50_0.60"
    if prob < 0.70:
        return "p02_0.60_0.70"
    return "p03_ge_0.70"


def _mean(values: list[float]) -> float:
    return sum(values) / float(len(values))


def _sample_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    m = _mean(values)
    var = sum((v - m) ** 2 for v in values) / float(len(values) - 1)
    return math.sqrt(max(var, 0.0))


def _mean_ci95(values: list[float]) -> tuple[float, float, float]:
    m = _mean(values)
    std = _sample_std(values)
    se = std / math.sqrt(max(1.0, float(len(values))))
    delta = 1.96 * se
    return m, m - delta, m + delta


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
        temp_path = Path(fh.name)
    temp_path.replace(path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False))
        fh.write("\n")


@dataclass(frozen=True)
class ReviewConfig:
    trades_db: Path
    orders_db: Path
    out_path: Path
    history_path: Path
    strategy_like: str
    lookback_days: int
    min_samples: int
    fold_count: int
    min_fold_samples: int
    min_fold_consistency: float
    block_lb_pips: float
    boost_lb_pips: float
    reduce_factor: float
    boost_factor: float
    jst_offset_hours: int
    top_k: int


@dataclass(frozen=True)
class TradeSample:
    ticket_id: str
    client_order_id: str
    strategy_tag: str
    side: str
    hour_jst: int
    day_jst: str
    pl_pips: float
    entry_probability: float | None
    spread_pips: float | None

    @property
    def fold_day_key(self) -> str:
        return self.day_jst


def _load_trade_rows(cfg: ReviewConfig) -> list[TradeSample]:
    if not cfg.trades_db.exists():
        return []

    since = datetime.now(timezone.utc) - timedelta(days=max(1, cfg.lookback_days))
    rows: list[TradeSample] = []

    con = sqlite3.connect(f"file:{cfg.trades_db}?mode=ro", uri=True, timeout=8.0, isolation_level=None)
    con.row_factory = sqlite3.Row
    try:
        sql = """
        SELECT
            ticket_id,
            COALESCE(client_order_id, '') AS client_order_id,
            COALESCE(strategy_tag, strategy, '') AS strategy_tag,
            units,
            pl_pips,
            entry_time,
            close_time,
            entry_thesis
        FROM trades
        WHERE close_time IS NOT NULL
          AND close_time >= :since
          AND strategy_tag LIKE :strategy_like
        ORDER BY close_time ASC
        """
        raw_rows = con.execute(
            sql,
            {
                "since": since.isoformat(),
                "strategy_like": cfg.strategy_like,
            },
        ).fetchall()
    finally:
        con.close()

    for row in raw_rows:
        entry_ts = _parse_iso(row["entry_time"])
        close_ts = _parse_iso(row["close_time"])
        if entry_ts is None and close_ts is None:
            continue
        ref_ts = entry_ts or close_ts
        if ref_ts is None:
            continue
        jst_ts = ref_ts + timedelta(hours=cfg.jst_offset_hours)

        units = int(row["units"] or 0)
        if units == 0:
            continue
        pl_pips = _to_float(row["pl_pips"])
        if pl_pips is None:
            continue

        side = "long" if units > 0 else "short"
        strategy_tag = str(row["strategy_tag"] or "").strip() or "unknown"
        ticket_id = str(row["ticket_id"] or "").strip() or f"row-{len(rows)}"
        client_order_id = str(row["client_order_id"] or "").strip()
        thesis = _safe_json_loads(row["entry_thesis"])
        entry_probability = _extract_entry_probability(thesis)

        rows.append(
            TradeSample(
                ticket_id=ticket_id,
                client_order_id=client_order_id,
                strategy_tag=strategy_tag,
                side=side,
                hour_jst=int(jst_ts.hour),
                day_jst=jst_ts.date().isoformat(),
                pl_pips=float(pl_pips),
                entry_probability=entry_probability,
                spread_pips=None,
            )
        )
    return rows


def _load_spread_map(cfg: ReviewConfig, client_ids: list[str]) -> dict[str, float]:
    if not cfg.orders_db.exists() or not client_ids:
        return {}

    out: dict[str, tuple[str, float]] = {}
    con = sqlite3.connect(f"file:{cfg.orders_db}?mode=ro", uri=True, timeout=8.0, isolation_level=None)
    con.row_factory = sqlite3.Row
    try:
        chunk_size = 400
        for idx in range(0, len(client_ids), chunk_size):
            chunk = client_ids[idx : idx + chunk_size]
            placeholders = ",".join(["?"] * len(chunk))
            sql = f"""
            SELECT client_order_id, ts, status, request_json
            FROM orders
            WHERE client_order_id IN ({placeholders})
              AND status = 'filled'
            """
            rows = con.execute(sql, chunk).fetchall()
            for row in rows:
                cid = str(row["client_order_id"] or "").strip()
                if not cid:
                    continue
                ts = str(row["ts"] or "")
                payload = _safe_json_loads(row["request_json"])
                quote = payload.get("quote") if isinstance(payload.get("quote"), dict) else {}
                spread = _to_float(quote.get("spread_pips"))
                if spread is None:
                    bid = _to_float(quote.get("bid"))
                    ask = _to_float(quote.get("ask"))
                    if bid is not None and ask is not None:
                        spread = (ask - bid) / 0.01
                if spread is None:
                    continue
                prev = out.get(cid)
                # Keep latest filled row by ts for deterministic mapping.
                if prev is None or ts > prev[0]:
                    out[cid] = (ts, float(spread))
    finally:
        con.close()
    return {cid: spread for cid, (_, spread) in out.items()}


def _inject_spread(samples: list[TradeSample], spread_map: dict[str, float]) -> list[TradeSample]:
    updated: list[TradeSample] = []
    for sample in samples:
        spread = spread_map.get(sample.client_order_id) if sample.client_order_id else None
        updated.append(
            TradeSample(
                ticket_id=sample.ticket_id,
                client_order_id=sample.client_order_id,
                strategy_tag=sample.strategy_tag,
                side=sample.side,
                hour_jst=sample.hour_jst,
                day_jst=sample.day_jst,
                pl_pips=sample.pl_pips,
                entry_probability=sample.entry_probability,
                spread_pips=spread,
            )
        )
    return updated


def _fold_index(day: str, day_to_index: dict[str, int], day_count: int, fold_count: int) -> int:
    if day_count <= 1:
        return 0
    idx = int(day_to_index.get(day, 0))
    ratio = idx / float(max(1, day_count))
    out = int(ratio * fold_count)
    if out >= fold_count:
        out = fold_count - 1
    return max(0, out)


def _make_feature_map(sample: TradeSample) -> dict[str, str]:
    spread = _spread_bin(sample.spread_pips)
    prob = _prob_bin(sample.entry_probability)
    hour = f"h{sample.hour_jst:02d}"
    side = sample.side
    return {
        "side": side,
        "hour": hour,
        "side_hour": f"{side}|{hour}",
        "spread_bin": spread,
        "prob_bin": prob,
        "side_spread": f"{side}|{spread}",
        "side_prob": f"{side}|{prob}",
    }


def _certainty(
    *,
    n: int,
    min_samples: int,
    fold_consistency: float,
    lower_bound: float,
    pivot: float,
) -> float:
    support = min(1.0, float(n) / float(max(1, min_samples * 2)))
    effect = min(1.0, abs(lower_bound) / max(0.05, abs(pivot)))
    raw = support * max(0.0, min(1.0, fold_consistency)) * effect
    return round(max(0.0, min(1.0, raw)), 4)


def _build_recommendations(samples: list[TradeSample], cfg: ReviewConfig) -> list[dict[str, Any]]:
    if not samples:
        return []

    days = sorted({s.fold_day_key for s in samples})
    day_to_index = {d: i for i, d in enumerate(days)}
    day_count = len(days)
    fold_count = max(2, cfg.fold_count)

    by_feature: dict[tuple[str, str], list[TradeSample]] = defaultdict(list)
    for sample in samples:
        fmap = _make_feature_map(sample)
        for name, bucket in fmap.items():
            if bucket == "unknown":
                continue
            by_feature[(name, bucket)].append(sample)

    recs: list[dict[str, Any]] = []
    for (feature_name, bucket), rows in by_feature.items():
        n = len(rows)
        if n < cfg.min_samples:
            continue

        pips = [r.pl_pips for r in rows]
        mean_pips, lb_pips, ub_pips = _mean_ci95(pips)
        wins = sum(1 for v in pips if v > 0.0)
        win_rate = wins / float(n)

        fold_values: dict[int, list[float]] = defaultdict(list)
        for row in rows:
            fold = _fold_index(row.fold_day_key, day_to_index, day_count, fold_count)
            fold_values[fold].append(row.pl_pips)
        fold_means = [sum(vs) / float(len(vs)) for vs in fold_values.values() if len(vs) >= cfg.min_fold_samples]

        if fold_means:
            pos = sum(1 for v in fold_means if v >= 0.0)
            neg = len(fold_means) - pos
            fold_consistency = max(pos, neg) / float(len(fold_means))
        else:
            fold_consistency = 0.0

        if fold_consistency < cfg.min_fold_consistency:
            continue

        sum_pips = sum(pips)
        action = "hold"
        expected_uplift = 0.0
        pivot = cfg.block_lb_pips

        if lb_pips <= cfg.block_lb_pips:
            action = "block"
            expected_uplift = -sum_pips
            pivot = cfg.block_lb_pips
        elif mean_pips < 0.0:
            action = "reduce"
            expected_uplift = -(sum_pips * cfg.reduce_factor)
            pivot = cfg.block_lb_pips
        elif lb_pips >= cfg.boost_lb_pips:
            action = "boost"
            expected_uplift = sum_pips * cfg.boost_factor
            pivot = cfg.boost_lb_pips
        else:
            continue

        if expected_uplift <= 0.0:
            continue

        certainty = _certainty(
            n=n,
            min_samples=cfg.min_samples,
            fold_consistency=fold_consistency,
            lower_bound=lb_pips if action != "boost" else lb_pips,
            pivot=pivot,
        )
        recs.append(
            {
                "feature": feature_name,
                "bucket": bucket,
                "action": action,
                "trades": n,
                "sum_pips": round(sum_pips, 4),
                "mean_pips": round(mean_pips, 4),
                "lb95_pips": round(lb_pips, 4),
                "ub95_pips": round(ub_pips, 4),
                "win_rate": round(win_rate, 4),
                "fold_consistency": round(fold_consistency, 4),
                "expected_uplift_pips": round(expected_uplift, 4),
                "certainty": certainty,
            }
        )

    recs.sort(
        key=lambda r: (
            float(r.get("expected_uplift_pips", 0.0)),
            float(r.get("certainty", 0.0)),
            int(r.get("trades", 0)),
        ),
        reverse=True,
    )
    return recs[: max(1, cfg.top_k)]


def _extract_policy_hints(recs: list[dict[str, Any]]) -> dict[str, Any]:
    block_hours: list[int] = []
    side_mode: dict[str, str] = {}
    for rec in recs:
        action = str(rec.get("action") or "")
        certainty = float(rec.get("certainty") or 0.0)
        if action not in {"block", "reduce", "boost"} or certainty < 0.65:
            continue

        feature = str(rec.get("feature") or "")
        bucket = str(rec.get("bucket") or "")
        if feature == "hour" and bucket.startswith("h") and len(bucket) == 3 and action == "block":
            try:
                block_hours.append(int(bucket[1:3]))
            except Exception:
                pass
        if feature == "side" and bucket in {"long", "short"}:
            side_mode[bucket] = action

    block_hours = sorted(set(v for v in block_hours if 0 <= v <= 23))
    return {
        "block_jst_hours": block_hours,
        "side_actions": side_mode,
    }


def build_report(cfg: ReviewConfig) -> dict[str, Any]:
    samples = _load_trade_rows(cfg)
    client_ids = [s.client_order_id for s in samples if s.client_order_id]
    spread_map = _load_spread_map(cfg, client_ids)
    samples = _inject_spread(samples, spread_map)
    recs = _build_recommendations(samples, cfg)
    hints = _extract_policy_hints(recs)

    total_trades = len(samples)
    mean_pips = _mean([s.pl_pips for s in samples]) if samples else 0.0
    sum_pips = sum(s.pl_pips for s in samples)
    win_rate = (
        sum(1 for s in samples if s.pl_pips > 0.0) / float(total_trades)
        if total_trades > 0
        else 0.0
    )
    with_spread = sum(1 for s in samples if s.spread_pips is not None)
    with_prob = sum(1 for s in samples if s.entry_probability is not None)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "strategy_like": cfg.strategy_like,
        "lookback_days": cfg.lookback_days,
        "summary": {
            "trades": total_trades,
            "sum_pips": round(sum_pips, 4),
            "mean_pips": round(mean_pips, 4),
            "win_rate": round(win_rate, 4),
            "coverage": {
                "with_spread_ratio": round(with_spread / float(total_trades), 4) if total_trades else 0.0,
                "with_entry_probability_ratio": round(with_prob / float(total_trades), 4) if total_trades else 0.0,
            },
        },
        "thresholds": {
            "min_samples": cfg.min_samples,
            "fold_count": cfg.fold_count,
            "min_fold_samples": cfg.min_fold_samples,
            "min_fold_consistency": cfg.min_fold_consistency,
            "block_lb_pips": cfg.block_lb_pips,
            "boost_lb_pips": cfg.boost_lb_pips,
            "reduce_factor": cfg.reduce_factor,
            "boost_factor": cfg.boost_factor,
        },
        "policy_hints": hints,
        "recommendations": recs,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Counterfactual trade review worker")
    ap.add_argument(
        "--trades-db",
        default=os.getenv("COUNTERFACTUAL_TRADES_DB", "logs/trades.db"),
    )
    ap.add_argument(
        "--orders-db",
        default=os.getenv("COUNTERFACTUAL_ORDERS_DB", "logs/orders.db"),
    )
    ap.add_argument(
        "--out-path",
        default=os.getenv("COUNTERFACTUAL_OUT_PATH", "logs/trade_counterfactual_latest.json"),
    )
    ap.add_argument(
        "--history-path",
        default=os.getenv("COUNTERFACTUAL_HISTORY_PATH", "logs/trade_counterfactual_history.jsonl"),
    )
    ap.add_argument(
        "--strategy-like",
        default=os.getenv("COUNTERFACTUAL_STRATEGY_LIKE", "scalp_ping_5s_b_live%"),
    )
    ap.add_argument(
        "--lookback-days",
        type=int,
        default=_env_int("COUNTERFACTUAL_LOOKBACK_DAYS", 14),
    )
    ap.add_argument(
        "--min-samples",
        type=int,
        default=_env_int("COUNTERFACTUAL_MIN_SAMPLES", 80),
    )
    ap.add_argument(
        "--fold-count",
        type=int,
        default=_env_int("COUNTERFACTUAL_FOLD_COUNT", 5),
    )
    ap.add_argument(
        "--min-fold-samples",
        type=int,
        default=_env_int("COUNTERFACTUAL_MIN_FOLD_SAMPLES", 8),
    )
    ap.add_argument(
        "--min-fold-consistency",
        type=float,
        default=_env_float("COUNTERFACTUAL_MIN_FOLD_CONSISTENCY", 0.60),
    )
    ap.add_argument(
        "--block-lb-pips",
        type=float,
        default=_env_float("COUNTERFACTUAL_BLOCK_LB_PIPS", -0.25),
    )
    ap.add_argument(
        "--boost-lb-pips",
        type=float,
        default=_env_float("COUNTERFACTUAL_BOOST_LB_PIPS", 0.20),
    )
    ap.add_argument(
        "--reduce-factor",
        type=float,
        default=_env_float("COUNTERFACTUAL_REDUCE_FACTOR", 0.50),
    )
    ap.add_argument(
        "--boost-factor",
        type=float,
        default=_env_float("COUNTERFACTUAL_BOOST_FACTOR", 0.30),
    )
    ap.add_argument(
        "--jst-offset-hours",
        type=int,
        default=_env_int("COUNTERFACTUAL_JST_OFFSET_HOURS", 9),
    )
    ap.add_argument(
        "--top-k",
        type=int,
        default=_env_int("COUNTERFACTUAL_TOP_K", 24),
    )
    return ap.parse_args()


def _build_config(args: argparse.Namespace) -> ReviewConfig:
    return ReviewConfig(
        trades_db=_resolve_path(args.trades_db),
        orders_db=_resolve_path(args.orders_db),
        out_path=_resolve_path(args.out_path),
        history_path=_resolve_path(args.history_path),
        strategy_like=str(args.strategy_like).strip() or "%",
        lookback_days=max(1, int(args.lookback_days)),
        min_samples=max(10, int(args.min_samples)),
        fold_count=max(2, int(args.fold_count)),
        min_fold_samples=max(2, int(args.min_fold_samples)),
        min_fold_consistency=max(0.0, min(1.0, float(args.min_fold_consistency))),
        block_lb_pips=float(args.block_lb_pips),
        boost_lb_pips=float(args.boost_lb_pips),
        reduce_factor=max(0.0, min(1.0, float(args.reduce_factor))),
        boost_factor=max(0.0, min(2.0, float(args.boost_factor))),
        jst_offset_hours=int(args.jst_offset_hours),
        top_k=max(1, int(args.top_k)),
    )


def run_once(cfg: ReviewConfig) -> dict[str, Any]:
    report = build_report(cfg)
    _write_json_atomic(cfg.out_path, report)
    _append_jsonl(cfg.history_path, report)
    return report


def main() -> int:
    args = parse_args()
    cfg = _build_config(args)
    report = run_once(cfg)
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    recs = report.get("recommendations") if isinstance(report.get("recommendations"), list) else []
    print(
        "[trade-counterfactual-worker] "
        f"trades={summary.get('trades', 0)} "
        f"mean_pips={summary.get('mean_pips', 0.0)} "
        f"recs={len(recs)} "
        f"out={cfg.out_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
