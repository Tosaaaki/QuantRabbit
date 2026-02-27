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
import glob
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import tempfile
from typing import Any

from utils.market_hours import is_market_open

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


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _parse_csv(value: Any) -> tuple[str, ...]:
    text = str(value or "").strip()
    if not text:
        return ()
    return tuple(part.strip() for part in text.split(",") if part.strip())


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


def _like_to_regex(pattern: str) -> re.Pattern[str]:
    raw = str(pattern or "%").strip()
    if not raw:
        raw = "%"
    escaped = re.escape(raw)
    escaped = escaped.replace("%", ".*").replace("_", ".")
    return re.compile(f"^{escaped}$")


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


def _clamp(value: float, low: float, high: float) -> float:
    if value < low:
        return low
    if value > high:
        return high
    return value


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
    oos_enabled: bool
    oos_min_folds: int
    oos_min_action_match_ratio: float
    oos_min_positive_ratio: float
    oos_min_lb_uplift_pips: float
    replay_json_globs: tuple[str, ...] = ()
    include_live_trades: bool = True
    stuck_hold_sec: float = 120.0
    stuck_loss_pips: float = -0.30
    stuck_reasons: tuple[str, ...] = ("time_stop", "no_recovery", "max_floating_loss", "end_of_replay")
    block_stuck_rate: float = 0.45
    reduce_stuck_rate: float = 0.30
    boost_stuck_rate: float = 0.10
    pattern_prior_enabled: bool = True
    pattern_book_path: Path = (REPO_ROOT / "config" / "pattern_book_deep.json").resolve()
    pattern_prior_weight: float = 0.35
    noise_spread_weight: float = 0.65
    noise_stuck_weight: float = 0.50
    noise_oos_weight: float = 0.45
    noise_min_spread_coverage: float = 0.40
    reentry_hint_min_confidence: float = 0.70
    reentry_hint_min_lcb_uplift_pips: float = 0.20


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
    reason: str = "unknown"
    hold_sec: float | None = None
    source: str = "live"

    @property
    def fold_day_key(self) -> str:
        return self.day_jst


def _normalize_reason(value: Any) -> str:
    raw = str(value or "").strip().lower()
    return raw if raw else "unknown"


def _hold_bin(hold_sec: float | None) -> str:
    if hold_sec is None:
        return "unknown"
    hold = max(0.0, float(hold_sec))
    if hold <= 30.0:
        return "h00_le_30s"
    if hold <= 90.0:
        return "h01_30_90s"
    if hold <= 180.0:
        return "h02_90_180s"
    return "h03_gt_180s"


def _is_stuck(sample: TradeSample, cfg: ReviewConfig) -> bool:
    reason_hit = sample.reason in cfg.stuck_reasons
    hold_hit = sample.hold_sec is not None and float(sample.hold_sec) >= float(cfg.stuck_hold_sec)
    loss_hit = float(sample.pl_pips) <= float(cfg.stuck_loss_pips)
    if reason_hit and loss_hit:
        return True
    if hold_hit and (reason_hit or loss_hit):
        return True
    return False


def _load_live_trade_rows(cfg: ReviewConfig) -> list[TradeSample]:
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
                reason="unknown",
                hold_sec=(
                    max(0.0, (close_ts - entry_ts).total_seconds())
                    if entry_ts is not None and close_ts is not None
                    else None
                ),
                source="live",
            )
        )
    return rows


def _load_replay_trade_rows(cfg: ReviewConfig) -> list[TradeSample]:
    if not cfg.replay_json_globs:
        return []

    matcher = _like_to_regex(cfg.strategy_like)
    since = datetime.now(timezone.utc) - timedelta(days=max(1, cfg.lookback_days))
    files: set[Path] = set()
    for pattern in cfg.replay_json_globs:
        query = pattern if os.path.isabs(pattern) else str(REPO_ROOT / pattern)
        for raw in glob.glob(query):
            path = Path(raw).resolve()
            if path.is_file():
                files.add(path)

    rows: list[TradeSample] = []
    for replay_path in sorted(files):
        try:
            payload = json.loads(replay_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        trades = payload.get("trades")
        if not isinstance(trades, list):
            continue

        for idx, item in enumerate(trades):
            if not isinstance(item, dict):
                continue
            strategy_tag = str(
                item.get("strategy_tag") or item.get("strategy") or item.get("tag") or ""
            ).strip() or "unknown"
            if not matcher.match(strategy_tag):
                continue

            entry_ts = _parse_iso(item.get("entry_time") or item.get("open_time"))
            close_ts = _parse_iso(item.get("exit_time") or item.get("close_time"))
            ref_ts = entry_ts or close_ts
            if ref_ts is None or ref_ts < since:
                continue
            jst_ts = ref_ts + timedelta(hours=cfg.jst_offset_hours)

            units_raw = _to_float(item.get("units"))
            if units_raw is None or units_raw == 0.0:
                continue
            side = "long" if units_raw > 0 else "short"

            pl_pips = _to_float(item.get("pnl_pips"))
            if pl_pips is None:
                pl_pips = _to_float(item.get("pl_pips"))
            if pl_pips is None:
                continue

            hold_sec = _to_float(item.get("hold_sec"))
            if hold_sec is None and entry_ts is not None and close_ts is not None:
                hold_sec = max(0.0, (close_ts - entry_ts).total_seconds())

            rows.append(
                TradeSample(
                    ticket_id=str(item.get("trade_id") or item.get("ticket_id") or f"replay-{idx}"),
                    client_order_id=str(item.get("client_order_id") or "").strip(),
                    strategy_tag=strategy_tag,
                    side=side,
                    hour_jst=int(jst_ts.hour),
                    day_jst=jst_ts.date().isoformat(),
                    pl_pips=float(pl_pips),
                    entry_probability=_normalize_probability(item.get("entry_probability")),
                    spread_pips=_to_float(item.get("spread_pips")),
                    reason=_normalize_reason(item.get("reason")),
                    hold_sec=hold_sec,
                    source="replay",
                )
            )
    return rows


def _load_trade_rows(cfg: ReviewConfig) -> list[TradeSample]:
    rows: list[TradeSample] = []
    if cfg.include_live_trades:
        rows.extend(_load_live_trade_rows(cfg))
    rows.extend(_load_replay_trade_rows(cfg))
    rows.sort(key=lambda row: (row.day_jst, row.hour_jst, row.ticket_id))
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
                reason=sample.reason,
                hold_sec=sample.hold_sec,
                source=sample.source,
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


def _make_feature_map(sample: TradeSample, cfg: ReviewConfig) -> dict[str, str]:
    spread = _spread_bin(sample.spread_pips)
    prob = _prob_bin(sample.entry_probability)
    hour = f"h{sample.hour_jst:02d}"
    side = sample.side
    reason = _normalize_reason(sample.reason)
    hold = _hold_bin(sample.hold_sec)
    stuck = "stuck" if _is_stuck(sample, cfg) else "normal"
    return {
        "side": side,
        "hour": hour,
        "reason": reason,
        "hold_bin": hold,
        "stuck": stuck,
        "side_hour": f"{side}|{hour}",
        "side_reason": f"{side}|{reason}",
        "hour_reason": f"{hour}|{reason}",
        "side_stuck": f"{side}|{stuck}",
        "spread_bin": spread,
        "prob_bin": prob,
        "side_spread": f"{side}|{spread}",
        "side_prob": f"{side}|{prob}",
        "hour_spread": f"{hour}|{spread}",
        "hour_prob": f"{hour}|{prob}",
    }


def _extract_pattern_strategy_token(row: dict[str, Any]) -> str:
    strategy = str(row.get("strategy_tag") or "").strip()
    if strategy:
        return strategy
    pattern_id = str(row.get("pattern_id") or "")
    if "st:" not in pattern_id:
        return ""
    for token in pattern_id.split("|"):
        if token.startswith("st:"):
            return token[3:]
    return ""


def _load_pattern_prior(cfg: ReviewConfig) -> tuple[dict[str, float], dict[str, Any]]:
    empty_meta = {
        "enabled": bool(cfg.pattern_prior_enabled),
        "path": str(cfg.pattern_book_path),
        "matched_rows": 0,
        "used_rows": 0,
    }
    if not cfg.pattern_prior_enabled:
        return {}, empty_meta
    if not cfg.pattern_book_path.exists():
        empty_meta["missing"] = True
        return {}, empty_meta

    try:
        payload = json.loads(cfg.pattern_book_path.read_text(encoding="utf-8"))
    except Exception:
        empty_meta["parse_error"] = True
        return {}, empty_meta
    if not isinstance(payload, dict):
        empty_meta["parse_error"] = True
        return {}, empty_meta

    matcher = _like_to_regex(cfg.strategy_like.lower())
    rows: list[dict[str, Any]] = []
    for key in ("top_robust", "top_weak"):
        raw = payload.get(key)
        if isinstance(raw, list):
            rows.extend(item for item in raw if isinstance(item, dict))

    scores: dict[str, list[float]] = defaultdict(list)
    used_rows = 0
    matched_rows = 0
    for row in rows:
        strategy = _extract_pattern_strategy_token(row)
        if not strategy or not matcher.match(strategy.lower()):
            continue
        matched_rows += 1
        side = str(row.get("direction") or "unknown").strip().lower()
        if side not in {"long", "short"}:
            side = "unknown"

        robust_score = _to_float(row.get("robust_score"))
        trades = int(_to_float(row.get("trades")) or 0)
        if robust_score is None:
            continue
        quality = str(row.get("quality") or "").strip().lower()
        score = _clamp(float(robust_score) / 4.0, -1.0, 1.0)
        if quality == "avoid" and score > 0.0:
            score = -score
        if quality in {"robust", "candidate"} and score < 0.0:
            score = abs(score)
        if quality in {"learn_only", "neutral"}:
            score *= 0.5
        weight = _clamp(math.log10(max(2, trades + 1)) / 2.0, 0.15, 1.0)
        weighted_score = score * weight
        scores[side].append(weighted_score)
        scores["__overall__"].append(weighted_score)
        used_rows += 1

    prior = {
        side: round(_clamp(_mean(values), -1.0, 1.0), 6)
        for side, values in scores.items()
        if values
    }
    meta = {
        "enabled": bool(cfg.pattern_prior_enabled),
        "path": str(cfg.pattern_book_path),
        "matched_rows": matched_rows,
        "used_rows": used_rows,
        "prior": prior,
    }
    return prior, meta


def _candidate_noise_metrics(
    *,
    rows: list[TradeSample],
    cfg: ReviewConfig,
    expected_uplift: float,
    stuck_rate: float,
    oos_action_match_ratio: float,
    oos_positive_ratio: float,
) -> dict[str, float]:
    n = len(rows)
    spread_values = [float(row.spread_pips) for row in rows if row.spread_pips is not None]
    spread_coverage = (len(spread_values) / float(n)) if n > 0 else 0.0
    mean_spread = _mean(spread_values) if spread_values else 0.0
    spread_excess = max(0.0, mean_spread - 0.40)
    spread_penalty = spread_excess * float(n) * max(0.0, cfg.noise_spread_weight)
    coverage_penalty = (
        max(0.0, cfg.noise_min_spread_coverage - spread_coverage)
        * abs(expected_uplift)
        * max(0.0, cfg.noise_spread_weight)
    )
    stuck_penalty = (
        max(0.0, stuck_rate) * float(n) * max(0.0, cfg.noise_stuck_weight)
    )
    oos_conf = _clamp(0.5 * oos_action_match_ratio + 0.5 * oos_positive_ratio, 0.0, 1.0)
    oos_penalty = (1.0 - oos_conf) * abs(expected_uplift) * max(0.0, cfg.noise_oos_weight)
    total_penalty = spread_penalty + coverage_penalty + stuck_penalty + oos_penalty
    return {
        "spread_coverage_ratio": round(spread_coverage, 6),
        "spread_mean_pips": round(mean_spread, 6),
        "noise_penalty_pips": round(total_penalty, 6),
        "noise_penalty_spread_pips": round(spread_penalty, 6),
        "noise_penalty_coverage_pips": round(coverage_penalty, 6),
        "noise_penalty_stuck_pips": round(stuck_penalty, 6),
        "noise_penalty_oos_pips": round(oos_penalty, 6),
    }


def _certainty(
    *,
    n: int,
    min_samples: int,
    fold_consistency: float,
    lower_bound: float,
    pivot: float,
    oos_action_match_ratio: float = 1.0,
    oos_positive_ratio: float = 1.0,
    oos_lb_uplift_pips: float = 0.0,
) -> float:
    support = min(1.0, float(n) / float(max(1, min_samples * 2)))
    effect = min(1.0, abs(lower_bound) / max(0.05, abs(pivot)))
    oos_effect = min(1.0, max(0.0, oos_lb_uplift_pips) / 0.20)
    raw = (
        support
        * max(0.0, min(1.0, fold_consistency))
        * effect
        * max(0.0, min(1.0, oos_action_match_ratio))
        * max(0.0, min(1.0, oos_positive_ratio))
        * oos_effect
    )
    return round(max(0.0, min(1.0, raw)), 4)


def _infer_action(
    rows: list[TradeSample], cfg: ReviewConfig
) -> tuple[str, float, float, float, float] | None:
    n = len(rows)
    if n < cfg.min_samples:
        return None
    pips = [r.pl_pips for r in rows]
    mean_pips, lb_pips, _ = _mean_ci95(pips)
    sum_pips = sum(pips)
    stuck_rate = sum(1 for row in rows if _is_stuck(row, cfg)) / float(max(1, n))

    if stuck_rate >= cfg.block_stuck_rate and (lb_pips <= cfg.block_lb_pips or mean_pips <= 0.0):
        action = "block"
        expected_uplift = -sum_pips
    elif stuck_rate >= cfg.reduce_stuck_rate and mean_pips <= 0.05:
        action = "reduce"
        expected_uplift = -(sum_pips * cfg.reduce_factor)
    elif lb_pips <= cfg.block_lb_pips:
        action = "block"
        expected_uplift = -sum_pips
    elif mean_pips < 0.0:
        action = "reduce"
        expected_uplift = -(sum_pips * cfg.reduce_factor)
    elif stuck_rate <= cfg.boost_stuck_rate and lb_pips >= cfg.boost_lb_pips:
        action = "boost"
        expected_uplift = sum_pips * cfg.boost_factor
    else:
        return None

    if expected_uplift <= 0.0:
        return None
    return action, mean_pips, lb_pips, expected_uplift, stuck_rate


def _simulate_uplift(action: str, rows: list[TradeSample], cfg: ReviewConfig) -> float:
    sum_pips = sum(r.pl_pips for r in rows)
    if action == "block":
        return -sum_pips
    if action == "reduce":
        return -(sum_pips * cfg.reduce_factor)
    if action == "boost":
        return sum_pips * cfg.boost_factor
    return 0.0


def _evaluate_oos(
    *,
    feature_name: str,
    bucket: str,
    all_samples: list[TradeSample],
    cfg: ReviewConfig,
    day_to_index: dict[str, int],
    day_count: int,
    fold_count: int,
    candidate_action: str,
) -> dict[str, float]:
    fold_to_rows: dict[int, list[TradeSample]] = defaultdict(list)
    for sample in all_samples:
        fmap = _make_feature_map(sample, cfg)
        if fmap.get(feature_name) != bucket:
            continue
        fold = _fold_index(sample.fold_day_key, day_to_index, day_count, fold_count)
        fold_to_rows[fold].append(sample)

    total_eval_folds = 0
    matched_uplifts: list[float] = []
    for fold in range(fold_count):
        test_rows = fold_to_rows.get(fold, [])
        if len(test_rows) < cfg.min_fold_samples:
            continue
        train_rows: list[TradeSample] = []
        for other_fold, rows in fold_to_rows.items():
            if other_fold == fold:
                continue
            train_rows.extend(rows)
        if len(train_rows) < cfg.min_samples:
            continue

        total_eval_folds += 1
        inferred = _infer_action(train_rows, cfg)
        if inferred is None:
            continue
        inferred_action = inferred[0]
        if inferred_action != candidate_action:
            continue

        uplift = _simulate_uplift(candidate_action, test_rows, cfg)
        matched_uplifts.append(float(uplift))

    action_match_ratio = (
        float(len(matched_uplifts)) / float(total_eval_folds)
        if total_eval_folds > 0
        else 0.0
    )
    positive_ratio = (
        sum(1 for u in matched_uplifts if u > 0.0) / float(len(matched_uplifts))
        if matched_uplifts
        else 0.0
    )
    mean_uplift = _mean(matched_uplifts) if matched_uplifts else 0.0
    if len(matched_uplifts) >= 2:
        _, lb95_uplift, _ = _mean_ci95(matched_uplifts)
    elif matched_uplifts:
        lb95_uplift = matched_uplifts[0]
    else:
        lb95_uplift = 0.0

    return {
        "oos_eval_folds": float(total_eval_folds),
        "oos_action_match_ratio": float(action_match_ratio),
        "oos_positive_ratio": float(positive_ratio),
        "oos_mean_uplift_pips": float(mean_uplift),
        "oos_lb95_uplift_pips": float(lb95_uplift),
    }


def _build_recommendations(
    samples: list[TradeSample],
    cfg: ReviewConfig,
    *,
    pattern_prior: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    if not samples:
        return []
    prior = pattern_prior or {}

    days = sorted({s.fold_day_key for s in samples})
    day_to_index = {d: i for i, d in enumerate(days)}
    day_count = len(days)
    fold_count = max(2, cfg.fold_count)

    by_feature: dict[tuple[str, str], list[TradeSample]] = defaultdict(list)
    for sample in samples:
        fmap = _make_feature_map(sample, cfg)
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

        inferred = _infer_action(rows, cfg)
        if inferred is None:
            continue
        action, _, _, expected_uplift, stuck_rate = inferred
        sum_pips = sum(pips)
        pivot = cfg.boost_lb_pips if action == "boost" else cfg.block_lb_pips

        oos = {
            "oos_eval_folds": float(fold_count),
            "oos_action_match_ratio": 1.0,
            "oos_positive_ratio": 1.0,
            "oos_mean_uplift_pips": expected_uplift,
            "oos_lb95_uplift_pips": expected_uplift,
        }
        if cfg.oos_enabled:
            oos = _evaluate_oos(
                feature_name=feature_name,
                bucket=bucket,
                all_samples=samples,
                cfg=cfg,
                day_to_index=day_to_index,
                day_count=day_count,
                fold_count=fold_count,
                candidate_action=action,
            )
            if int(oos["oos_eval_folds"]) < cfg.oos_min_folds:
                continue
            if float(oos["oos_action_match_ratio"]) < cfg.oos_min_action_match_ratio:
                continue
            if float(oos["oos_positive_ratio"]) < cfg.oos_min_positive_ratio:
                continue
            if float(oos["oos_lb95_uplift_pips"]) < cfg.oos_min_lb_uplift_pips:
                continue

        certainty = _certainty(
            n=n,
            min_samples=cfg.min_samples,
            fold_consistency=fold_consistency,
            lower_bound=lb_pips if action != "boost" else lb_pips,
            pivot=pivot,
            oos_action_match_ratio=float(oos["oos_action_match_ratio"]),
            oos_positive_ratio=float(oos["oos_positive_ratio"]),
            oos_lb_uplift_pips=float(oos["oos_lb95_uplift_pips"]),
        )
        noise = _candidate_noise_metrics(
            rows=rows,
            cfg=cfg,
            expected_uplift=expected_uplift,
            stuck_rate=stuck_rate,
            oos_action_match_ratio=float(oos["oos_action_match_ratio"]),
            oos_positive_ratio=float(oos["oos_positive_ratio"]),
        )
        noise_adjusted_uplift = expected_uplift - float(noise["noise_penalty_pips"])
        noise_lcb_uplift = min(noise_adjusted_uplift, float(oos["oos_lb95_uplift_pips"]))

        long_count = sum(1 for row in rows if row.side == "long")
        short_count = sum(1 for row in rows if row.side == "short")
        if feature_name == "side" and bucket in {"long", "short"}:
            prior_side = bucket
        elif long_count > short_count:
            prior_side = "long"
        elif short_count > long_count:
            prior_side = "short"
        else:
            prior_side = "unknown"
        pattern_prior_score = float(prior.get(prior_side, prior.get("__overall__", 0.0)))
        signed_prior = pattern_prior_score if action == "boost" else -pattern_prior_score
        pattern_multiplier = 1.0 + _clamp(
            signed_prior * max(0.0, cfg.pattern_prior_weight),
            -0.35,
            0.35,
        )
        pattern_adjusted_lcb = noise_lcb_uplift * pattern_multiplier
        quality_score = pattern_adjusted_lcb * max(0.0, certainty)

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
                "stuck_rate": round(stuck_rate, 4),
                "expected_uplift_pips": round(expected_uplift, 4),
                "certainty": certainty,
                "oos_eval_folds": int(oos["oos_eval_folds"]),
                "oos_action_match_ratio": round(float(oos["oos_action_match_ratio"]), 4),
                "oos_positive_ratio": round(float(oos["oos_positive_ratio"]), 4),
                "oos_mean_uplift_pips": round(float(oos["oos_mean_uplift_pips"]), 4),
                "oos_lb95_uplift_pips": round(float(oos["oos_lb95_uplift_pips"]), 4),
                "noise_adjusted_uplift_pips": round(noise_adjusted_uplift, 4),
                "noise_lcb_uplift_pips": round(noise_lcb_uplift, 4),
                "quality_score": round(quality_score, 6),
                "pattern_prior_side": prior_side,
                "pattern_prior_score": round(pattern_prior_score, 6),
                "pattern_prior_multiplier": round(pattern_multiplier, 6),
                "pattern_adjusted_lcb_uplift_pips": round(pattern_adjusted_lcb, 4),
                "noise": noise,
            }
        )

    recs.sort(
        key=lambda r: (
            float(r.get("quality_score", 0.0)),
            float(r.get("noise_lcb_uplift_pips", 0.0)),
            float(r.get("certainty", 0.0)),
            int(r.get("trades", 0)),
        ),
        reverse=True,
    )
    return recs[: max(1, cfg.top_k)]


def _extract_policy_hints(recs: list[dict[str, Any]], cfg: ReviewConfig) -> dict[str, Any]:
    block_hours: list[int] = []
    block_reasons: list[str] = []
    reduce_reasons: list[str] = []
    side_mode: dict[str, str] = {}
    actionable: list[dict[str, Any]] = []
    for rec in recs:
        action = str(rec.get("action") or "")
        certainty = float(rec.get("certainty") or 0.0)
        noise_lcb = float(rec.get("noise_lcb_uplift_pips") or 0.0)
        if action not in {"block", "reduce", "boost"}:
            continue
        if certainty < max(0.50, cfg.reentry_hint_min_confidence * 0.85):
            continue

        feature = str(rec.get("feature") or "")
        bucket = str(rec.get("bucket") or "")
        if feature == "hour" and bucket.startswith("h") and len(bucket) == 3 and action == "block":
            try:
                block_hours.append(int(bucket[1:3]))
            except Exception:
                pass
        if feature == "reason" and bucket != "unknown":
            if action == "block":
                block_reasons.append(bucket)
            if action == "reduce":
                reduce_reasons.append(bucket)
        if feature == "side" and bucket in {"long", "short"}:
            side_mode[bucket] = action
        if (
            certainty >= cfg.reentry_hint_min_confidence
            and noise_lcb >= cfg.reentry_hint_min_lcb_uplift_pips
        ):
            actionable.append(rec)

    block_hours = sorted(set(v for v in block_hours if 0 <= v <= 23))
    tighten_score = 0.0
    loosen_score = 0.0
    tighten_lcb = 0.0
    loosen_lcb = 0.0
    max_certainty = 0.0
    for rec in actionable:
        action = str(rec.get("action") or "")
        score = max(0.0, float(rec.get("quality_score") or 0.0))
        lcb = max(0.0, float(rec.get("noise_lcb_uplift_pips") or 0.0))
        certainty = max(0.0, float(rec.get("certainty") or 0.0))
        max_certainty = max(max_certainty, certainty)
        if action in {"block", "reduce"}:
            tighten_score += score
            tighten_lcb += lcb
        elif action == "boost":
            loosen_score += score
            loosen_lcb += lcb

    mode = "neutral"
    confidence = 0.0
    uplift_lcb = 0.0
    cooldown_loss_mult = 1.0
    cooldown_win_mult = 1.0
    same_dir_reentry_pips_mult = 1.0
    return_wait_bias = "neutral"

    total_score = tighten_score + loosen_score
    if total_score > 0.0:
        net_score = tighten_score - loosen_score
        severity = _clamp(abs(net_score) / total_score, 0.0, 1.0)
        confidence = _clamp(0.6 * severity + 0.4 * max_certainty, 0.0, 1.0)
        if net_score > 0.0:
            mode = "tighten"
            uplift_lcb = max(0.0, tighten_lcb - loosen_lcb)
            cooldown_loss_mult = 1.0 + 0.55 * severity
            cooldown_win_mult = 1.0 + 0.25 * severity
            same_dir_reentry_pips_mult = 1.0 + 0.35 * severity
            return_wait_bias = "avoid"
        elif net_score < 0.0:
            mode = "loosen"
            uplift_lcb = max(0.0, loosen_lcb - tighten_lcb)
            cooldown_loss_mult = 1.0 - 0.28 * severity
            cooldown_win_mult = 1.0 - 0.16 * severity
            same_dir_reentry_pips_mult = 1.0 - 0.22 * severity
            return_wait_bias = "favor"

    top_candidates = [
        {
            "feature": str(rec.get("feature") or ""),
            "bucket": str(rec.get("bucket") or ""),
            "action": str(rec.get("action") or ""),
            "quality_score": round(float(rec.get("quality_score") or 0.0), 6),
            "certainty": round(float(rec.get("certainty") or 0.0), 4),
            "noise_lcb_uplift_pips": round(float(rec.get("noise_lcb_uplift_pips") or 0.0), 4),
        }
        for rec in actionable[:5]
    ]
    return {
        "block_jst_hours": block_hours,
        "block_reasons": sorted(set(block_reasons)),
        "reduce_reasons": sorted(set(reduce_reasons)),
        "side_actions": side_mode,
        "reentry_overrides": {
            "mode": mode,
            "confidence": round(confidence, 4),
            "lcb_uplift_pips": round(uplift_lcb, 4),
            "cooldown_loss_mult": round(_clamp(cooldown_loss_mult, 0.60, 1.80), 4),
            "cooldown_win_mult": round(_clamp(cooldown_win_mult, 0.70, 1.50), 4),
            "same_dir_reentry_pips_mult": round(
                _clamp(same_dir_reentry_pips_mult, 0.70, 1.60), 4
            ),
            "return_wait_bias": return_wait_bias,
            "source": "counterfactual_noise_lcb_pattern_prior",
        },
        "top_reentry_candidates": top_candidates,
    }


def build_report(cfg: ReviewConfig) -> dict[str, Any]:
    samples = _load_trade_rows(cfg)
    client_ids = [s.client_order_id for s in samples if s.client_order_id]
    spread_map = _load_spread_map(cfg, client_ids)
    samples = _inject_spread(samples, spread_map)
    pattern_prior, pattern_meta = _load_pattern_prior(cfg)
    recs = _build_recommendations(samples, cfg, pattern_prior=pattern_prior)
    hints = _extract_policy_hints(recs, cfg)

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
    stuck_trades = sum(1 for s in samples if _is_stuck(s, cfg))
    reason_counts: dict[str, int] = defaultdict(int)
    source_counts: dict[str, int] = defaultdict(int)
    for sample in samples:
        reason_counts[_normalize_reason(sample.reason)] += 1
        source_counts[str(sample.source or "unknown")] += 1
    top_reasons = [
        {"reason": reason, "trades": count}
        for reason, count in sorted(reason_counts.items(), key=lambda item: (-item[1], item[0]))[:8]
    ]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "strategy_like": cfg.strategy_like,
        "lookback_days": cfg.lookback_days,
        "summary": {
            "trades": total_trades,
            "sum_pips": round(sum_pips, 4),
            "mean_pips": round(mean_pips, 4),
            "win_rate": round(win_rate, 4),
            "stuck_trades": stuck_trades,
            "stuck_trade_ratio": round(stuck_trades / float(total_trades), 4) if total_trades else 0.0,
            "coverage": {
                "with_spread_ratio": round(with_spread / float(total_trades), 4) if total_trades else 0.0,
                "with_entry_probability_ratio": round(with_prob / float(total_trades), 4) if total_trades else 0.0,
                "sources": dict(source_counts),
            },
            "top_close_reasons": top_reasons,
            "pattern_prior": pattern_meta,
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
            "oos_enabled": cfg.oos_enabled,
            "oos_min_folds": cfg.oos_min_folds,
            "oos_min_action_match_ratio": cfg.oos_min_action_match_ratio,
            "oos_min_positive_ratio": cfg.oos_min_positive_ratio,
            "oos_min_lb_uplift_pips": cfg.oos_min_lb_uplift_pips,
            "include_live_trades": cfg.include_live_trades,
            "replay_json_globs": list(cfg.replay_json_globs),
            "stuck_hold_sec": cfg.stuck_hold_sec,
            "stuck_loss_pips": cfg.stuck_loss_pips,
            "stuck_reasons": list(cfg.stuck_reasons),
            "block_stuck_rate": cfg.block_stuck_rate,
            "reduce_stuck_rate": cfg.reduce_stuck_rate,
            "boost_stuck_rate": cfg.boost_stuck_rate,
            "pattern_prior_enabled": cfg.pattern_prior_enabled,
            "pattern_book_path": str(cfg.pattern_book_path),
            "pattern_prior_weight": cfg.pattern_prior_weight,
            "noise_spread_weight": cfg.noise_spread_weight,
            "noise_stuck_weight": cfg.noise_stuck_weight,
            "noise_oos_weight": cfg.noise_oos_weight,
            "noise_min_spread_coverage": cfg.noise_min_spread_coverage,
            "reentry_hint_min_confidence": cfg.reentry_hint_min_confidence,
            "reentry_hint_min_lcb_uplift_pips": cfg.reentry_hint_min_lcb_uplift_pips,
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
        "--replay-json-globs",
        default=os.getenv("COUNTERFACTUAL_REPLAY_JSON_GLOBS", ""),
        help="Comma-separated replay JSON globs (ex: tmp/replay_quality_gate/*/runs/*/replay_exit_workers.json).",
    )
    ap.add_argument(
        "--include-live-trades",
        type=int,
        default=_env_int("COUNTERFACTUAL_INCLUDE_LIVE_TRADES", 1),
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
    ap.add_argument(
        "--oos-enabled",
        type=int,
        default=_env_int("COUNTERFACTUAL_OOS_ENABLED", 1),
    )
    ap.add_argument(
        "--oos-min-folds",
        type=int,
        default=_env_int("COUNTERFACTUAL_OOS_MIN_FOLDS", 3),
    )
    ap.add_argument(
        "--oos-min-action-match-ratio",
        type=float,
        default=_env_float("COUNTERFACTUAL_OOS_MIN_ACTION_MATCH_RATIO", 0.60),
    )
    ap.add_argument(
        "--oos-min-positive-ratio",
        type=float,
        default=_env_float("COUNTERFACTUAL_OOS_MIN_POSITIVE_RATIO", 0.60),
    )
    ap.add_argument(
        "--oos-min-lb-uplift-pips",
        type=float,
        default=_env_float("COUNTERFACTUAL_OOS_MIN_LB_UPLIFT_PIPS", 0.0),
    )
    ap.add_argument(
        "--stuck-hold-sec",
        type=float,
        default=_env_float("COUNTERFACTUAL_STUCK_HOLD_SEC", 120.0),
    )
    ap.add_argument(
        "--stuck-loss-pips",
        type=float,
        default=_env_float("COUNTERFACTUAL_STUCK_LOSS_PIPS", -0.30),
    )
    ap.add_argument(
        "--stuck-reasons",
        default=os.getenv(
            "COUNTERFACTUAL_STUCK_REASONS",
            "time_stop,no_recovery,max_floating_loss,end_of_replay",
        ),
    )
    ap.add_argument(
        "--block-stuck-rate",
        type=float,
        default=_env_float("COUNTERFACTUAL_BLOCK_STUCK_RATE", 0.45),
    )
    ap.add_argument(
        "--reduce-stuck-rate",
        type=float,
        default=_env_float("COUNTERFACTUAL_REDUCE_STUCK_RATE", 0.30),
    )
    ap.add_argument(
        "--boost-stuck-rate",
        type=float,
        default=_env_float("COUNTERFACTUAL_BOOST_STUCK_RATE", 0.10),
    )
    ap.add_argument(
        "--pattern-prior-enabled",
        type=int,
        choices=(0, 1),
        default=_env_int("COUNTERFACTUAL_PATTERN_PRIOR_ENABLED", 1),
    )
    ap.add_argument(
        "--pattern-book-path",
        default=os.getenv("COUNTERFACTUAL_PATTERN_BOOK_PATH", "config/pattern_book_deep.json"),
    )
    ap.add_argument(
        "--pattern-prior-weight",
        type=float,
        default=_env_float("COUNTERFACTUAL_PATTERN_PRIOR_WEIGHT", 0.35),
    )
    ap.add_argument(
        "--noise-spread-weight",
        type=float,
        default=_env_float("COUNTERFACTUAL_NOISE_SPREAD_WEIGHT", 0.65),
    )
    ap.add_argument(
        "--noise-stuck-weight",
        type=float,
        default=_env_float("COUNTERFACTUAL_NOISE_STUCK_WEIGHT", 0.50),
    )
    ap.add_argument(
        "--noise-oos-weight",
        type=float,
        default=_env_float("COUNTERFACTUAL_NOISE_OOS_WEIGHT", 0.45),
    )
    ap.add_argument(
        "--noise-min-spread-coverage",
        type=float,
        default=_env_float("COUNTERFACTUAL_NOISE_MIN_SPREAD_COVERAGE", 0.40),
    )
    ap.add_argument(
        "--reentry-hint-min-confidence",
        type=float,
        default=_env_float("COUNTERFACTUAL_REENTRY_HINT_MIN_CONFIDENCE", 0.70),
    )
    ap.add_argument(
        "--reentry-hint-min-lcb-uplift-pips",
        type=float,
        default=_env_float("COUNTERFACTUAL_REENTRY_HINT_MIN_LCB_UPLIFT_PIPS", 0.20),
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
        oos_enabled=bool(int(args.oos_enabled)),
        oos_min_folds=max(1, int(args.oos_min_folds)),
        oos_min_action_match_ratio=max(0.0, min(1.0, float(args.oos_min_action_match_ratio))),
        oos_min_positive_ratio=max(0.0, min(1.0, float(args.oos_min_positive_ratio))),
        oos_min_lb_uplift_pips=float(args.oos_min_lb_uplift_pips),
        replay_json_globs=_parse_csv(args.replay_json_globs),
        include_live_trades=bool(int(args.include_live_trades)),
        stuck_hold_sec=max(10.0, float(args.stuck_hold_sec)),
        stuck_loss_pips=float(args.stuck_loss_pips),
        stuck_reasons=tuple(_normalize_reason(v) for v in _parse_csv(args.stuck_reasons)),
        block_stuck_rate=max(0.0, min(1.0, float(args.block_stuck_rate))),
        reduce_stuck_rate=max(0.0, min(1.0, float(args.reduce_stuck_rate))),
        boost_stuck_rate=max(0.0, min(1.0, float(args.boost_stuck_rate))),
        pattern_prior_enabled=bool(int(args.pattern_prior_enabled)),
        pattern_book_path=_resolve_path(args.pattern_book_path),
        pattern_prior_weight=max(0.0, min(1.0, float(args.pattern_prior_weight))),
        noise_spread_weight=max(0.0, min(2.0, float(args.noise_spread_weight))),
        noise_stuck_weight=max(0.0, min(2.0, float(args.noise_stuck_weight))),
        noise_oos_weight=max(0.0, min(2.0, float(args.noise_oos_weight))),
        noise_min_spread_coverage=max(0.0, min(1.0, float(args.noise_min_spread_coverage))),
        reentry_hint_min_confidence=max(0.0, min(1.0, float(args.reentry_hint_min_confidence))),
        reentry_hint_min_lcb_uplift_pips=float(args.reentry_hint_min_lcb_uplift_pips),
    )


def run_once(cfg: ReviewConfig) -> dict[str, Any]:
    report = build_report(cfg)
    _write_json_atomic(cfg.out_path, report)
    _append_jsonl(cfg.history_path, report)
    return report


def main() -> int:
    if _env_bool("COUNTERFACTUAL_SKIP_WHEN_MARKET_OPEN", False) and is_market_open():
        print("[trade-counterfactual-worker] skipped: market_open")
        return 0
    args = parse_args()
    cfg = _build_config(args)
    report = run_once(cfg)
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    recs = report.get("recommendations") if isinstance(report.get("recommendations"), list) else []
    coverage = summary.get("coverage") if isinstance(summary.get("coverage"), dict) else {}
    sources = coverage.get("sources") if isinstance(coverage.get("sources"), dict) else {}
    print(
        "[trade-counterfactual-worker] "
        f"trades={summary.get('trades', 0)} "
        f"mean_pips={summary.get('mean_pips', 0.0)} "
        f"stuck_ratio={summary.get('stuck_trade_ratio', 0.0)} "
        f"replay={sources.get('replay', 0)} "
        f"recs={len(recs)} "
        f"out={cfg.out_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
