#!/usr/bin/env python3
"""Benchmark local LLM Brain decision quality using local SQLite data.

This tool replays historical local entry contexts against one or more local
LLM variants (Ollama chat JSON) and writes a machine-readable JSON report.

Key metrics:
- parse pass/fail and reasons
- ALLOW/REDUCE/BLOCK action mix
- latency distribution
- optional outcome alignment against realized trades (trades.db)
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sqlite3
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import requests

# Allow running as `python3 scripts/...py` without manual PYTHONPATH.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_BRAIN_DB = Path("logs/brain_state.db")
DEFAULT_ORDERS_DB = Path("logs/orders.db")
DEFAULT_TRADES_DB = Path("logs/trades.db")
DEFAULT_OUTPUT = Path("logs/brain_local_llm_benchmark_latest.json")


Action = str


@dataclass(frozen=True)
class Sample:
    sample_id: str
    ts: str
    source: str
    strategy_tag: str
    pocket: str
    side: str
    units: int
    sl_price: Optional[float]
    tp_price: Optional[float]
    confidence: Optional[float]
    client_order_id: Optional[str]
    context: dict[str, Any]
    realized_pl: Optional[float] = None
    pl_pips: Optional[float] = None


@dataclass(frozen=True)
class VariantSpec:
    name: str
    model: str
    url: str
    temperature: float
    max_tokens: int
    timeout_sec: float
    prompt_template: Optional[str] = None
    prompt_prefix: str = ""
    prompt_suffix: str = ""


@dataclass(frozen=True)
class CallOutcome:
    payload: Optional[dict[str, Any]]
    fail_reason: str = ""
    raw_content: str = ""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark local LLM Brain quality against local DB contexts.")
    parser.add_argument("--source", choices=("auto", "brain", "orders"), default="auto")
    parser.add_argument("--brain-db", type=Path, default=DEFAULT_BRAIN_DB)
    parser.add_argument("--orders-db", type=Path, default=DEFAULT_ORDERS_DB)
    parser.add_argument("--trades-db", type=Path, default=DEFAULT_TRADES_DB)
    parser.add_argument("--lookback-hours", type=float, default=24.0)
    parser.add_argument("--max-samples", type=int, default=120)
    parser.add_argument("--sample-mode", choices=("recent", "random"), default="recent")
    parser.add_argument(
        "--outcome-sample-policy",
        choices=("any", "prioritize", "require"),
        default="prioritize",
        help="How to treat realized-outcome samples during selection.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        help=(
            "Variant JSON. Can be repeated. "
            "Keys: name, model, url, temperature, max_tokens, timeout_sec, "
            "prompt_template_path, prompt_prefix, prompt_suffix"
        ),
    )
    parser.add_argument("--default-model", default="gpt-oss:20b")
    parser.add_argument("--default-url", default="http://127.0.0.1:11434/api/chat")
    parser.add_argument("--default-temperature", type=float, default=0.2)
    parser.add_argument("--default-max-tokens", type=int, default=2048)
    parser.add_argument("--default-timeout-sec", type=float, default=70.0)
    parser.add_argument(
        "--ranking-min-outcome-samples",
        type=int,
        default=20,
        help="Minimum scored-trade count before outcome score affects ranking.",
    )
    parser.add_argument(
        "--disable-alignment",
        action="store_true",
        help="Disable realized trade alignment scoring.",
    )
    parser.add_argument(
        "--include-sample-details",
        action="store_true",
        help="Include per-sample variant decisions in report (can be large).",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _read_num(payload: dict[str, Any], key: str, default: float) -> float:
    value = payload.get(key)
    if value is None:
        return float(default)
    try:
        return float(value)
    except Exception:
        return float(default)


def _read_int(payload: dict[str, Any], key: str, default: int) -> int:
    value = payload.get(key)
    if value is None:
        return int(default)
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _safe_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except Exception:
        return None
    if not math.isfinite(number):
        return None
    return number


def _extract_json_payload(text: str) -> Optional[dict[str, Any]]:
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        payload = json.loads(raw)
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(raw[start : end + 1])
    except Exception:
        return None
    if isinstance(payload, dict):
        return payload
    return None


def _call_ollama_chat_json_verbose(
    prompt: str,
    *,
    model: str,
    url: str,
    timeout_sec: float,
    temperature: float = 0.2,
    max_tokens: int = 256,
) -> CallOutcome:
    def _single_call(num_predict: int) -> tuple[Optional[dict[str, Any]], str, str, bool]:
        req_payload = {
            "model": model,
            "stream": False,
            "think": False,
            "messages": [{"role": "user", "content": prompt}],
            "options": {
                "temperature": max(0.0, min(float(temperature), 1.0)),
                "num_predict": max(64, int(num_predict)),
            },
        }
        try:
            resp = requests.post(url, json=req_payload, timeout=max(1.0, float(timeout_sec)))
        except requests.Timeout:
            return None, "http_timeout", "", False
        except Exception:
            return None, "http_exception", "", False

        if resp.status_code >= 400:
            return None, f"http_{int(resp.status_code)}", "", False
        try:
            body = resp.json()
        except Exception:
            text = str(resp.text or "")
            return None, "invalid_http_json", text[:240], False
        if not isinstance(body, dict):
            return None, "invalid_http_json_body", "", False
        message = body.get("message")
        if not isinstance(message, dict):
            return None, "missing_message", "", False
        content = str(message.get("content") or "")
        parsed = _extract_json_payload(content)
        if parsed is not None:
            return parsed, "", content[:240], False
        thinking = str(message.get("thinking") or "")
        parsed = _extract_json_payload(thinking)
        if parsed is not None:
            return parsed, "", thinking[:240], False
        done_reason = str(body.get("done_reason") or "").strip().lower()
        should_retry = bool(not content and done_reason == "length" and thinking)
        return None, "invalid_model_json", content[:240], should_retry

    requested_tokens = max(64, int(max_tokens))
    parsed, reason, raw, retry = _single_call(requested_tokens)
    if parsed is not None:
        return CallOutcome(payload=parsed, fail_reason="", raw_content=raw)
    if not retry:
        return CallOutcome(payload=None, fail_reason=reason, raw_content=raw)
    boosted_tokens = min(max(requested_tokens * 2, 2048), 4096)
    parsed_retry, reason_retry, raw_retry, _ = _single_call(boosted_tokens)
    if parsed_retry is not None:
        return CallOutcome(payload=parsed_retry, fail_reason="", raw_content=raw_retry)
    return CallOutcome(payload=None, fail_reason=reason_retry or reason, raw_content=raw_retry or raw)


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _parse_iso(value: str) -> Optional[datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except Exception:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _normalized_confidence(entry_thesis: dict[str, Any], context_confidence: Any = None) -> Optional[float]:
    cands: list[Any] = [
        entry_thesis.get("entry_probability"),
        entry_thesis.get("entry_probability_raw"),
        context_confidence,
        entry_thesis.get("confidence"),
    ]
    for cand in cands:
        value = _safe_float(cand)
        if value is None:
            continue
        if value > 1.0 and value <= 100.0:
            return max(0.0, min(1.0, value / 100.0))
        if 0.0 <= value <= 1.0:
            return value
    return None


def _build_context(
    *,
    ts: str,
    strategy_tag: str,
    pocket: str,
    side: str,
    units: int,
    sl_price: Optional[float],
    tp_price: Optional[float],
    entry_thesis: Optional[dict[str, Any]],
    meta: Optional[dict[str, Any]],
    confidence: Optional[float],
) -> dict[str, Any]:
    return {
        "ts": ts,
        "strategy_tag": strategy_tag,
        "pocket": pocket,
        "side": side,
        "units": int(units),
        "sl_price": sl_price,
        "tp_price": tp_price,
        "confidence": confidence,
        "memory": "",
        "entry_thesis": entry_thesis or {},
        "meta": meta or {},
        "runtime_param_profile_version": "bench",
    }


def _load_trade_outcomes(trades_db: Path) -> dict[str, dict[str, float]]:
    outcomes: dict[str, dict[str, float]] = {}
    if not trades_db.exists():
        return outcomes
    con = sqlite3.connect(trades_db)
    try:
        rows = con.execute(
            """
            SELECT client_order_id,
                   SUM(COALESCE(realized_pl, 0.0)) AS realized_pl,
                   AVG(COALESCE(pl_pips, 0.0)) AS pl_pips,
                   COUNT(*) AS trade_rows
            FROM trades
            WHERE close_time IS NOT NULL
              AND client_order_id IS NOT NULL
              AND TRIM(client_order_id) <> ''
            GROUP BY client_order_id
            """
        ).fetchall()
    finally:
        con.close()

    for client_order_id, realized_pl, pl_pips, trade_rows in rows:
        key = str(client_order_id or "").strip()
        if not key:
            continue
        outcomes[key] = {
            "realized_pl": float(realized_pl or 0.0),
            "pl_pips": float(pl_pips or 0.0),
            "trade_rows": float(trade_rows or 0.0),
        }
    return outcomes


def _load_samples_from_brain(
    brain_db: Path,
    *,
    lookback_hours: float,
    max_samples: int,
    outcomes: dict[str, dict[str, float]],
) -> list[Sample]:
    if not brain_db.exists():
        return []

    cutoff_epoch = time.time() - (max(1.0, float(lookback_hours)) * 3600.0)
    con = sqlite3.connect(brain_db)
    try:
        rows = con.execute(
            """
            SELECT id, ts, strategy_tag, pocket, side, units, sl_price, tp_price,
                   confidence, client_order_id, context_json
            FROM brain_decisions
            WHERE ts_epoch >= ?
            ORDER BY ts_epoch DESC
            LIMIT ?
            """,
            (float(cutoff_epoch), int(max_samples)),
        ).fetchall()
    finally:
        con.close()

    samples: list[Sample] = []
    for row in rows:
        (
            row_id,
            ts,
            strategy_tag,
            pocket,
            side,
            units,
            sl_price,
            tp_price,
            confidence,
            client_order_id,
            context_json,
        ) = row
        context: dict[str, Any] = {}
        if context_json:
            try:
                parsed = json.loads(str(context_json))
                if isinstance(parsed, dict):
                    context = parsed
            except Exception:
                context = {}
        if not context:
            entry_thesis: dict[str, Any] = {}
            meta: dict[str, Any] = {}
            context = _build_context(
                ts=str(ts or ""),
                strategy_tag=str(strategy_tag or ""),
                pocket=str(pocket or ""),
                side=str(side or ""),
                units=_to_int(units),
                sl_price=_safe_float(sl_price),
                tp_price=_safe_float(tp_price),
                entry_thesis=entry_thesis,
                meta=meta,
                confidence=_safe_float(confidence),
            )

        cid = str(client_order_id or "").strip() or None
        outcome = outcomes.get(cid or "") if cid else None
        samples.append(
            Sample(
                sample_id=f"brain_decision:{int(row_id)}",
                ts=str(ts or ""),
                source="brain_decisions",
                strategy_tag=str(strategy_tag or context.get("strategy_tag") or ""),
                pocket=str(pocket or context.get("pocket") or ""),
                side=str(side or context.get("side") or ""),
                units=_to_int(units),
                sl_price=_safe_float(sl_price),
                tp_price=_safe_float(tp_price),
                confidence=_safe_float(confidence),
                client_order_id=cid,
                context=context,
                realized_pl=(float(outcome["realized_pl"]) if outcome else None),
                pl_pips=(float(outcome["pl_pips"]) if outcome else None),
            )
        )
    return samples


def _load_samples_from_orders(
    orders_db: Path,
    *,
    lookback_hours: float,
    max_samples: int,
    outcomes: dict[str, dict[str, float]],
) -> list[Sample]:
    if not orders_db.exists():
        return []

    cutoff_dt = datetime.now(timezone.utc) - timedelta(hours=max(1.0, float(lookback_hours)))
    fetch_limit = max(200, max_samples * 8)

    con = sqlite3.connect(orders_db)
    try:
        rows = con.execute(
            """
            SELECT id, ts, pocket, side, units, sl_price, tp_price, client_order_id, request_json
            FROM orders
            WHERE status = 'preflight_start'
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(fetch_limit),),
        ).fetchall()
    finally:
        con.close()

    samples: list[Sample] = []
    for row in rows:
        row_id, ts, pocket, side, units, sl_price, tp_price, client_order_id, request_json = row
        ts_text = str(ts or "")
        ts_dt = _parse_iso(ts_text)
        if ts_dt is None or ts_dt < cutoff_dt:
            continue

        payload: dict[str, Any] = {}
        if request_json:
            try:
                parsed = json.loads(str(request_json))
                if isinstance(parsed, dict):
                    payload = parsed
            except Exception:
                payload = {}

        entry_thesis = payload.get("entry_thesis") if isinstance(payload.get("entry_thesis"), dict) else {}
        meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
        strategy_tag = str(
            payload.get("strategy_tag")
            or entry_thesis.get("strategy_tag")
            or meta.get("strategy_tag")
            or ""
        )
        pocket_text = str(pocket or payload.get("pocket") or meta.get("pocket") or "")
        side_text = str(side or payload.get("side") or "")
        confidence = _normalized_confidence(entry_thesis, payload.get("confidence"))

        context = _build_context(
            ts=ts_text,
            strategy_tag=strategy_tag,
            pocket=pocket_text,
            side=side_text,
            units=_to_int(units),
            sl_price=_safe_float(sl_price),
            tp_price=_safe_float(tp_price),
            entry_thesis=entry_thesis,
            meta=meta,
            confidence=confidence,
        )

        cid = str(client_order_id or "").strip() or None
        outcome = outcomes.get(cid or "") if cid else None
        samples.append(
            Sample(
                sample_id=f"orders_preflight:{int(row_id)}",
                ts=ts_text,
                source="orders_preflight",
                strategy_tag=strategy_tag,
                pocket=pocket_text,
                side=side_text,
                units=_to_int(units),
                sl_price=_safe_float(sl_price),
                tp_price=_safe_float(tp_price),
                confidence=confidence,
                client_order_id=cid,
                context=context,
                realized_pl=(float(outcome["realized_pl"]) if outcome else None),
                pl_pips=(float(outcome["pl_pips"]) if outcome else None),
            )
        )
        if len(samples) >= max_samples:
            break
    return samples


def _latency_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "count": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "p95": 0.0,
            "max": 0.0,
            "min": 0.0,
        }
    ordered = sorted(values)
    p95_index = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * 0.95)))
    return {
        "count": float(len(values)),
        "mean": round(sum(values) / len(values), 3),
        "median": round(statistics.median(ordered), 3),
        "p95": round(float(ordered[p95_index]), 3),
        "max": round(max(values), 3),
        "min": round(min(values), 3),
    }


def _load_prompt_template(template_path: str) -> str:
    path = Path(template_path)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path.read_text(encoding="utf-8")


def _build_default_prompt(context: dict[str, Any]) -> str:
    # Reuse production prompt builder for high-fidelity benchmark baseline.
    try:
        from workers.common import brain as brain_module

        prompt = brain_module._build_prompt(context)  # type: ignore[attr-defined]
        if isinstance(prompt, str) and prompt.strip():
            return prompt
    except Exception:
        pass

    return (
        "You are the decision brain for an automated USD/JPY trading worker. "
        "Decide whether to allow, reduce, or block a single entry candidate. "
        "Respond JSON only with keys action, scale, reason, memory_update.\n"
        f"Context:\n{json.dumps(context, ensure_ascii=True)}"
    )


def _render_prompt(sample: Sample, variant: VariantSpec) -> str:
    context_json = json.dumps(sample.context, ensure_ascii=True, sort_keys=True)
    if variant.prompt_template:
        prompt = variant.prompt_template.format(
            context_json=context_json,
            context=sample.context,
            strategy_tag=sample.strategy_tag,
            pocket=sample.pocket,
            side=sample.side,
            units=sample.units,
        )
    else:
        prompt = _build_default_prompt(sample.context)

    prefix = variant.prompt_prefix.strip()
    suffix = variant.prompt_suffix.strip()
    if prefix:
        prompt = f"{prefix}\n\n{prompt}"
    if suffix:
        prompt = f"{prompt}\n\n{suffix}"
    return prompt


def _normalize_action(raw_action: Any) -> Optional[Action]:
    action = str(raw_action or "").strip().upper()
    if action in {"ALLOW", "REDUCE", "BLOCK"}:
        return action
    return None


def _normalize_decision(payload: Optional[dict[str, Any]]) -> tuple[Optional[dict[str, Any]], str]:
    if payload is None:
        return None, "no_payload"
    if not isinstance(payload, dict):
        return None, "payload_not_dict"

    action = _normalize_action(payload.get("action"))
    if action is None:
        return None, "invalid_action"

    scale = _safe_float(payload.get("scale"))
    if action == "BLOCK":
        normalized_scale = 0.0
    else:
        if scale is None:
            return None, "invalid_scale"
        normalized_scale = max(0.0, min(scale, 1.0))
        if normalized_scale <= 0.0:
            return None, "invalid_scale"

    reason = str(payload.get("reason") or "").strip() or "llm_decision"
    normalized = {
        "action": action,
        "scale": normalized_scale,
        "reason": reason,
    }
    return normalized, ""


def _alignment_score(action: str, realized_pl: float) -> float:
    # 1.0 = best aligned with observed outcome, 0.0 = worst aligned.
    if abs(realized_pl) < 1e-9:
        return 0.5
    if realized_pl > 0.0:
        if action == "ALLOW":
            return 1.0
        if action == "REDUCE":
            return 0.6
        return 0.0
    # realized_pl < 0.0
    if action == "BLOCK":
        return 1.0
    if action == "REDUCE":
        return 0.6
    return 0.0


def _evaluate_variant(
    *,
    samples: list[Sample],
    variant: VariantSpec,
    enable_alignment: bool,
    include_sample_details: bool,
    caller: Callable[..., CallOutcome] = _call_ollama_chat_json_verbose,
) -> dict[str, Any]:
    action_counts: dict[str, int] = {"ALLOW": 0, "REDUCE": 0, "BLOCK": 0}
    fail_reasons: dict[str, int] = {}
    fail_examples: dict[str, str] = {}
    latencies_ms: list[float] = []
    pass_latencies_ms: list[float] = []

    detail_rows: list[dict[str, Any]] = []
    parse_pass = 0
    parse_fail = 0

    align_scores: list[float] = []
    align_hard_hits = 0
    align_scored = 0
    align_pos = 0
    align_neg = 0
    align_neutral = 0
    action_realized_stats: dict[str, dict[str, float]] = {
        "ALLOW": {"n": 0.0, "sum_realized_pl": 0.0, "wins": 0.0},
        "REDUCE": {"n": 0.0, "sum_realized_pl": 0.0, "wins": 0.0},
        "BLOCK": {"n": 0.0, "sum_realized_pl": 0.0, "wins": 0.0},
    }

    for sample in samples:
        prompt = _render_prompt(sample, variant)
        start = time.monotonic()
        try:
            call_result = caller(
                prompt,
                model=variant.model,
                url=variant.url,
                timeout_sec=variant.timeout_sec,
                temperature=variant.temperature,
                max_tokens=variant.max_tokens,
            )
        except Exception:
            call_result = CallOutcome(payload=None, fail_reason="caller_exception")
        latency_ms = max(0.0, (time.monotonic() - start) * 1000.0)
        latencies_ms.append(latency_ms)
        normalized, fail_reason = _normalize_decision(call_result.payload)
        if normalized is None:
            if fail_reason == "no_payload" and call_result.fail_reason:
                fail_reason = call_result.fail_reason
            parse_fail += 1
            fail_reasons[fail_reason] = fail_reasons.get(fail_reason, 0) + 1
            if fail_reason not in fail_examples and call_result.raw_content:
                fail_examples[fail_reason] = str(call_result.raw_content)[:240]
            action = "PARSE_FAIL"
            scale = None
            reason = fail_reason
        else:
            parse_pass += 1
            pass_latencies_ms.append(latency_ms)
            action = str(normalized["action"])
            scale = float(normalized["scale"])
            reason = str(normalized["reason"])
            action_counts[action] += 1

            if enable_alignment and sample.realized_pl is not None:
                realized = float(sample.realized_pl)
                score = _alignment_score(action, realized)
                align_scores.append(score)
                align_scored += 1

                if abs(realized) < 1e-9:
                    align_neutral += 1
                    align_hard_hits += 1
                elif realized > 0.0:
                    align_pos += 1
                    if action in {"ALLOW", "REDUCE"}:
                        align_hard_hits += 1
                else:
                    align_neg += 1
                    if action in {"BLOCK", "REDUCE"}:
                        align_hard_hits += 1

                stats = action_realized_stats[action]
                stats["n"] += 1.0
                stats["sum_realized_pl"] += realized
                if realized > 0.0:
                    stats["wins"] += 1.0

        if include_sample_details:
            detail_rows.append(
                {
                    "sample_id": sample.sample_id,
                    "source": sample.source,
                    "ts": sample.ts,
                    "strategy_tag": sample.strategy_tag,
                    "pocket": sample.pocket,
                    "side": sample.side,
                    "units": sample.units,
                    "client_order_id": sample.client_order_id,
                    "latency_ms": round(latency_ms, 3),
                    "parse_ok": normalized is not None,
                    "action": action,
                    "scale": scale,
                    "reason": reason,
                    "realized_pl": sample.realized_pl,
                    "pl_pips": sample.pl_pips,
                }
            )

    total = len(samples)
    action_mix = {
        "allow": round(action_counts["ALLOW"] / total, 4) if total else 0.0,
        "reduce": round(action_counts["REDUCE"] / total, 4) if total else 0.0,
        "block": round(action_counts["BLOCK"] / total, 4) if total else 0.0,
    }

    outcome_alignment: dict[str, Any] = {
        "enabled": bool(enable_alignment),
        "scored_trades": int(align_scored),
        "positive_trades": int(align_pos),
        "negative_trades": int(align_neg),
        "neutral_trades": int(align_neutral),
        "score_mean": round(sum(align_scores) / align_scored, 4) if align_scored else None,
        "score_median": round(statistics.median(align_scores), 4) if align_scored else None,
        "hard_alignment_rate": round(align_hard_hits / align_scored, 4) if align_scored else None,
        "action_realized_stats": {},
    }
    for action_key, stats in action_realized_stats.items():
        n = int(stats["n"])
        if n <= 0:
            continue
        outcome_alignment["action_realized_stats"][action_key] = {
            "n": n,
            "avg_realized_pl": round(stats["sum_realized_pl"] / n, 6),
            "win_rate": round(stats["wins"] / n, 4),
        }

    result: dict[str, Any] = {
        "variant": {
            "name": variant.name,
            "model": variant.model,
            "url": variant.url,
            "temperature": variant.temperature,
            "max_tokens": variant.max_tokens,
            "timeout_sec": variant.timeout_sec,
            "has_prompt_template": bool(variant.prompt_template),
            "has_prompt_prefix": bool(variant.prompt_prefix.strip()),
            "has_prompt_suffix": bool(variant.prompt_suffix.strip()),
        },
        "sample_count": total,
        "parse": {
            "pass": parse_pass,
            "fail": parse_fail,
            "pass_rate": round(parse_pass / total, 4) if total else 0.0,
            "fail_reasons": fail_reasons,
            "fail_examples": fail_examples,
        },
        "actions": action_counts,
        "action_mix": action_mix,
        "latency_ms": _latency_stats(latencies_ms),
        "latency_ms_parse_pass": _latency_stats(pass_latencies_ms),
        "outcome_alignment": outcome_alignment,
    }
    if include_sample_details:
        result["sample_details"] = detail_rows
    return result


def _parse_variant_specs(args: argparse.Namespace) -> list[VariantSpec]:
    specs: list[VariantSpec] = []
    for idx, raw in enumerate(args.variant, start=1):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"--variant #{idx} invalid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise SystemExit(f"--variant #{idx} must decode to JSON object")

        name = str(payload.get("name") or f"variant_{idx}").strip() or f"variant_{idx}"
        model = str(payload.get("model") or args.default_model).strip()
        url = str(payload.get("url") or args.default_url).strip()
        temperature = _read_num(payload, "temperature", float(args.default_temperature))
        max_tokens = _read_int(payload, "max_tokens", int(args.default_max_tokens))
        timeout_sec = _read_num(payload, "timeout_sec", float(args.default_timeout_sec))

        prompt_template = None
        prompt_template_path = payload.get("prompt_template_path")
        if prompt_template_path:
            prompt_template = _load_prompt_template(str(prompt_template_path))

        specs.append(
            VariantSpec(
                name=name,
                model=model,
                url=url,
                temperature=max(0.0, min(1.0, temperature)),
                max_tokens=max(64, max_tokens),
                timeout_sec=max(1.0, timeout_sec),
                prompt_template=prompt_template,
                prompt_prefix=str(payload.get("prompt_prefix") or ""),
                prompt_suffix=str(payload.get("prompt_suffix") or ""),
            )
        )

    if specs:
        return specs

    return [
        VariantSpec(
            name="default",
            model=str(args.default_model),
            url=str(args.default_url),
            temperature=max(0.0, min(1.0, float(args.default_temperature))),
            max_tokens=max(64, int(args.default_max_tokens)),
            timeout_sec=max(1.0, float(args.default_timeout_sec)),
        )
    ]


def _select_samples(
    *,
    source: str,
    brain_db: Path,
    orders_db: Path,
    trades_db: Path,
    lookback_hours: float,
    max_samples: int,
    sample_mode: str,
    seed: int,
    outcome_sample_policy: str,
) -> tuple[str, list[Sample], dict[str, Any]]:
    outcomes = _load_trade_outcomes(trades_db)

    load_meta: dict[str, Any] = {
        "brain_db_exists": brain_db.exists(),
        "orders_db_exists": orders_db.exists(),
        "trades_db_exists": trades_db.exists(),
        "trade_outcome_keys": len(outcomes),
    }

    selected_source = source
    samples: list[Sample] = []
    candidate_max_samples = int(max(1, max_samples))
    if outcome_sample_policy in {"prioritize", "require"}:
        candidate_max_samples = max(candidate_max_samples, int(max_samples) * 8)

    brain_samples: list[Sample] = []
    if source in {"auto", "brain"}:
        brain_samples = _load_samples_from_brain(
            brain_db,
            lookback_hours=lookback_hours,
            max_samples=candidate_max_samples,
            outcomes=outcomes,
        )
        load_meta["brain_samples"] = len(brain_samples)
        if source == "brain":
            selected_source = "brain"
            samples = brain_samples

    order_samples: list[Sample] = []
    should_load_orders = source == "orders" or source == "auto" or not samples
    if should_load_orders and source in {"auto", "orders"}:
        order_samples = _load_samples_from_orders(
            orders_db,
            lookback_hours=lookback_hours,
            max_samples=candidate_max_samples,
            outcomes=outcomes,
        )
        load_meta["order_samples"] = len(order_samples)
        if source == "orders":
            selected_source = "orders"
            samples = order_samples
    elif source == "auto":
        load_meta["order_samples"] = 0

    if source == "auto":
        brain_outcomes = sum(1 for sample in brain_samples if sample.realized_pl is not None)
        order_outcomes = sum(1 for sample in order_samples if sample.realized_pl is not None)
        load_meta["brain_outcome_samples"] = int(brain_outcomes)
        load_meta["order_outcome_samples"] = int(order_outcomes)
        if outcome_sample_policy in {"prioritize", "require"}:
            if order_outcomes > brain_outcomes:
                selected_source = "orders"
                samples = order_samples
                load_meta["auto_source_reason"] = "orders_higher_outcome_coverage"
            elif brain_outcomes > order_outcomes:
                selected_source = "brain"
                samples = brain_samples
                load_meta["auto_source_reason"] = "brain_higher_outcome_coverage"
            else:
                if len(order_samples) > len(brain_samples):
                    selected_source = "orders"
                    samples = order_samples
                    load_meta["auto_source_reason"] = "orders_more_candidates_tie"
                else:
                    selected_source = "brain"
                    samples = brain_samples
                    load_meta["auto_source_reason"] = "brain_more_candidates_or_tie"
        else:
            if brain_samples:
                selected_source = "brain"
                samples = brain_samples
                load_meta["auto_source_reason"] = "brain_default"
            else:
                selected_source = "orders"
                samples = order_samples
                load_meta["auto_source_reason"] = "orders_fallback"

    if sample_mode == "random" and len(samples) > 1:
        rng = random.Random(seed)
        rng.shuffle(samples)

    outcome_samples = [sample for sample in samples if sample.realized_pl is not None]
    non_outcome_samples = [sample for sample in samples if sample.realized_pl is None]
    outcome_insufficient_reason = ""
    requested_samples = max(0, int(max_samples))

    if outcome_sample_policy == "prioritize":
        samples = outcome_samples + non_outcome_samples
        if len(outcome_samples) < requested_samples:
            outcome_insufficient_reason = (
                f"insufficient_realized_outcome_samples:{len(outcome_samples)}/{requested_samples}"
            )
    elif outcome_sample_policy == "require":
        samples = outcome_samples
        if len(outcome_samples) == 0:
            outcome_insufficient_reason = "outcome_required_but_no_realized_outcome_samples"
        elif len(outcome_samples) < requested_samples:
            outcome_insufficient_reason = (
                f"required_realized_outcomes_below_requested:{len(outcome_samples)}/{requested_samples}"
            )
    else:
        if len(outcome_samples) == 0:
            outcome_insufficient_reason = "no_realized_outcome_samples"

    samples = samples[:requested_samples]
    selected_outcomes = sum(1 for sample in samples if sample.realized_pl is not None)
    selected_total = len(samples)
    outcome_policy_meta: dict[str, Any] = {
        "policy": str(outcome_sample_policy),
        "candidate_total_samples": int(len(outcome_samples) + len(non_outcome_samples)),
        "candidate_with_trade_outcome": int(len(outcome_samples)),
        "candidate_without_trade_outcome": int(len(non_outcome_samples)),
        "selected_total_samples": int(selected_total),
        "selected_with_trade_outcome": int(selected_outcomes),
        "selected_without_trade_outcome": int(selected_total - selected_outcomes),
        "selected_coverage_ratio": (
            round(selected_outcomes / selected_total, 4) if selected_total > 0 else 0.0
        ),
    }
    if outcome_insufficient_reason:
        outcome_policy_meta["insufficient_data_reason"] = outcome_insufficient_reason
    load_meta["outcome_policy"] = outcome_policy_meta
    return selected_source, samples, load_meta


def _rank_variants(
    results: list[dict[str, Any]],
    *,
    min_outcome_samples: int,
) -> list[dict[str, Any]]:
    ranking_rows: list[dict[str, Any]] = []
    required_scored = max(1, int(min_outcome_samples))
    for item in results:
        variant = item.get("variant", {})
        parse_info = item.get("parse", {})
        latency_info = item.get("latency_ms_parse_pass", {})
        align = item.get("outcome_alignment", {})

        score = float(parse_info.get("pass_rate") or 0.0)
        align_score = align.get("score_mean")
        scored_trades = int(align.get("scored_trades") or 0)
        sample_count = max(1, int(item.get("sample_count") or 0))
        alignment_coverage = round(min(1.0, scored_trades / sample_count), 4)
        outcome_score_used = False
        outcome_score_reason = ""
        if isinstance(align_score, (int, float)) and scored_trades >= required_scored:
            score += 0.35 * float(align_score) * float(alignment_coverage)
            outcome_score_used = True
            outcome_score_reason = "used"
        elif not isinstance(align_score, (int, float)):
            outcome_score_reason = "outcome_score_unavailable"
        else:
            outcome_score_reason = f"insufficient_scored_trades:{scored_trades}<{required_scored}"

        p95 = float(latency_info.get("p95") or 0.0)
        ranking_rows.append(
            {
                "name": str(variant.get("name") or ""),
                "model": str(variant.get("model") or ""),
                "score": round(score, 6),
                "parse_pass_rate": float(parse_info.get("pass_rate") or 0.0),
                "alignment_score_mean": align_score,
                "alignment_coverage": alignment_coverage,
                "outcome_scored_trades": scored_trades,
                "outcome_score": (float(align_score) * float(alignment_coverage) if outcome_score_used else None),
                "outcome_score_used": outcome_score_used,
                "outcome_score_reason": outcome_score_reason,
                "latency_p95_ms": p95,
            }
        )

    ranking_rows.sort(
        key=lambda row: (
            -float(row.get("score") or 0.0),
            -float(row.get("parse_pass_rate") or 0.0),
            float(row.get("latency_p95_ms") or 0.0),
        )
    )
    for rank, row in enumerate(ranking_rows, start=1):
        row["rank"] = rank
    return ranking_rows


def main() -> int:
    args = _parse_args()
    variants = _parse_variant_specs(args)

    selected_source, samples, load_meta = _select_samples(
        source=str(args.source),
        brain_db=Path(args.brain_db),
        orders_db=Path(args.orders_db),
        trades_db=Path(args.trades_db),
        lookback_hours=float(args.lookback_hours),
        max_samples=int(args.max_samples),
        sample_mode=str(args.sample_mode),
        seed=int(args.seed),
        outcome_sample_policy=str(args.outcome_sample_policy),
    )

    if not samples:
        outcome_meta = load_meta.get("outcome_policy") if isinstance(load_meta.get("outcome_policy"), dict) else {}
        report = {
            "generated_at": _iso_now(),
            "status": "no_samples",
            "selected_source": selected_source,
            "inputs": {
                "source": args.source,
                "brain_db": str(args.brain_db),
                "orders_db": str(args.orders_db),
                "trades_db": str(args.trades_db),
                "lookback_hours": float(args.lookback_hours),
                "max_samples": int(args.max_samples),
                "sample_mode": args.sample_mode,
                "outcome_sample_policy": str(args.outcome_sample_policy),
                "ranking_min_outcome_samples": int(args.ranking_min_outcome_samples),
            },
            "load_meta": load_meta,
            "sample_summary": {
                "total_samples": 0,
                "outcome_coverage": outcome_meta,
                "insufficient_data_reason": (
                    str(outcome_meta.get("insufficient_data_reason"))
                    if outcome_meta.get("insufficient_data_reason")
                    else "no_samples_after_source_selection"
                ),
            },
            "variants": [
                {
                    "name": spec.name,
                    "model": spec.model,
                    "url": spec.url,
                }
                for spec in variants
            ],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
        return 0

    started = time.monotonic()
    variant_results: list[dict[str, Any]] = []
    for spec in variants:
        variant_results.append(
            _evaluate_variant(
                samples=samples,
                variant=spec,
                enable_alignment=not bool(args.disable_alignment),
                include_sample_details=bool(args.include_sample_details),
            )
        )

    elapsed_sec = max(0.0, time.monotonic() - started)
    report = {
        "generated_at": _iso_now(),
        "status": "ok",
        "selected_source": selected_source,
        "inputs": {
            "source": args.source,
            "brain_db": str(args.brain_db),
            "orders_db": str(args.orders_db),
            "trades_db": str(args.trades_db),
            "lookback_hours": float(args.lookback_hours),
            "max_samples": int(args.max_samples),
            "sample_mode": args.sample_mode,
            "seed": int(args.seed),
            "outcome_sample_policy": str(args.outcome_sample_policy),
            "ranking_min_outcome_samples": int(args.ranking_min_outcome_samples),
            "disable_alignment": bool(args.disable_alignment),
            "include_sample_details": bool(args.include_sample_details),
        },
        "load_meta": load_meta,
        "sample_summary": {
            "total_samples": len(samples),
            "source_counts": {
                "brain_decisions": sum(1 for s in samples if s.source == "brain_decisions"),
                "orders_preflight": sum(1 for s in samples if s.source == "orders_preflight"),
            },
            "with_trade_outcome": sum(1 for s in samples if s.realized_pl is not None),
            "outcome_coverage": {},
            "strategy_counts": {},
            "pocket_counts": {},
        },
        "variants": variant_results,
        "ranking": _rank_variants(
            variant_results,
            min_outcome_samples=int(args.ranking_min_outcome_samples),
        ),
        "ranking_meta": {
            "outcome_weight": 0.35,
            "min_outcome_samples": max(1, int(args.ranking_min_outcome_samples)),
        },
        "runtime": {
            "elapsed_sec": round(elapsed_sec, 3),
            "variant_count": len(variants),
        },
    }

    strategy_counts: dict[str, int] = {}
    pocket_counts: dict[str, int] = {}
    for sample in samples:
        strategy_counts[sample.strategy_tag] = strategy_counts.get(sample.strategy_tag, 0) + 1
        pocket_counts[sample.pocket] = pocket_counts.get(sample.pocket, 0) + 1
    report["sample_summary"]["strategy_counts"] = dict(
        sorted(strategy_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    )
    report["sample_summary"]["pocket_counts"] = dict(
        sorted(pocket_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    )
    outcome_meta = load_meta.get("outcome_policy") if isinstance(load_meta.get("outcome_policy"), dict) else {}
    report["sample_summary"]["outcome_coverage"] = outcome_meta
    report["sample_summary"]["insufficient_data_reason"] = (
        str(outcome_meta.get("insufficient_data_reason"))
        if outcome_meta.get("insufficient_data_reason")
        else None
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    summary = {
        "status": report["status"],
        "output": str(args.output),
        "samples": report["sample_summary"]["total_samples"],
        "selected_source": report["selected_source"],
        "top_variant": (report["ranking"][0] if report["ranking"] else None),
    }
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
