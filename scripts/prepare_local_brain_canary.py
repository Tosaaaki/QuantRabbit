#!/usr/bin/env python3
"""Prepare and assess local Brain/Ollama canary readiness for local-v2."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

import requests

# Allow running as `python3 scripts/...py` without manual PYTHONPATH.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.secrets import get_secret

DEFAULT_BENCHMARK = Path("logs/brain_local_llm_benchmark_latest.json")
DEFAULT_SELECTION_OUTPUT = Path("logs/brain_model_selection_safe_latest.json")
DEFAULT_ENV_PROFILE = Path("ops/env/profiles/brain-ollama-safe.env")
DEFAULT_OUTPUT = Path("logs/brain_canary_readiness_latest.json")
APPLY_SCRIPT = PROJECT_ROOT / "scripts" / "apply_brain_model_selection.py"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare local Brain/Ollama canary readiness for Monday open.")
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--selection-output", type=Path, default=DEFAULT_SELECTION_OUTPUT)
    parser.add_argument("--env-profile", type=Path, default=DEFAULT_ENV_PROFILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-benchmark-age-hours", type=float, default=72.0)
    parser.add_argument("--max-selection-age-hours", type=float, default=72.0)
    parser.add_argument("--min-parse-pass-rate", type=float, default=0.90)
    parser.add_argument("--max-preflight-latency-ms", type=float, default=4000.0)
    parser.add_argument("--timeout-cap-sec", type=float, default=4.0)
    parser.add_argument("--ollama-timeout-sec", type=float, default=4.0)
    parser.add_argument("--warmup-timeout-sec", type=float, default=12.0)
    parser.add_argument("--max-spread-pips", type=float, default=1.2)
    parser.add_argument("--max-tick-age-sec", type=float, default=10.0)
    parser.add_argument("--skip-selection-sync", action="store_true")
    parser.add_argument("--skip-ollama", action="store_true")
    parser.add_argument("--skip-market", action="store_true")
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _parse_env_file(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    if not path.exists():
        return result
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in raw_line:
            continue
        key, value = raw_line.split("=", 1)
        result[key.strip()] = value.strip()
    return result


def _bool_env(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _float_env(value: str | None) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_iso(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _age_hours(value: Any, *, now: datetime) -> float | None:
    dt = _parse_iso(value)
    if dt is None:
        return None
    age_sec = max(0.0, (now - dt).total_seconds())
    return round(age_sec / 3600.0, 3)


def _resolve_tags_url(chat_url: str) -> str:
    parsed = urlparse(str(chat_url or "").strip() or "http://127.0.0.1:11434/api/chat")
    path = parsed.path or "/api/chat"
    if path.endswith("/chat"):
        path = path[: -len("/chat")] + "/tags"
    else:
        path = "/api/tags"
    return urlunparse((parsed.scheme, parsed.netloc, path, "", "", ""))


def _collect_required_models(env_values: dict[str, str], selection_payload: dict[str, Any]) -> list[str]:
    required: list[str] = []

    def add(value: str | None) -> None:
        model = str(value or "").strip()
        if model and model not in required:
            required.append(model)

    add(env_values.get("BRAIN_OLLAMA_MODEL") or selection_payload.get("preflight_model"))
    if _bool_env(env_values.get("BRAIN_PROMPT_AUTO_TUNE_ENABLED")):
        add(env_values.get("BRAIN_PROMPT_AUTO_TUNE_MODEL") or selection_payload.get("autotune_model"))
    if _bool_env(env_values.get("BRAIN_RUNTIME_PARAM_AUTO_TUNE_ENABLED")):
        add(env_values.get("BRAIN_RUNTIME_PARAM_AUTO_TUNE_MODEL") or selection_payload.get("autotune_model"))
    return required


def _find_benchmark_row(benchmark_payload: dict[str, Any], model: str | None) -> dict[str, Any]:
    model_key = str(model or "").strip()
    if not model_key:
        return {}
    ranking = benchmark_payload.get("ranking")
    if not isinstance(ranking, list):
        return {}
    for item in ranking:
        if not isinstance(item, dict):
            continue
        if str(item.get("model") or "").strip() == model_key:
            return item
    return {}


def _profile_safety_issues(env_values: dict[str, str], *, timeout_cap_sec: float) -> list[str]:
    issues: list[str] = []
    pocket_allowlist = {
        item.strip().lower()
        for item in str(env_values.get("BRAIN_POCKET_ALLOWLIST") or "").split(",")
        if item.strip()
    }
    if pocket_allowlist != {"micro"}:
        issues.append("pocket_allowlist_micro_only")
    fail_policy = str(env_values.get("BRAIN_FAIL_POLICY") or "").strip().lower()
    if fail_policy != "allow":
        issues.append("fail_policy_not_allow")
    sample_rate = _float_env(env_values.get("BRAIN_SAMPLE_RATE"))
    if sample_rate is None or sample_rate <= 0.0 or sample_rate > 0.5:
        issues.append("sample_rate_out_of_range")
    if _bool_env(env_values.get("BRAIN_PROMPT_AUTO_TUNE_ENABLED")):
        issues.append("prompt_autotune_enabled")
    if _bool_env(env_values.get("BRAIN_RUNTIME_PARAM_AUTO_TUNE_ENABLED")):
        issues.append("runtime_autotune_enabled")
    gate_mode = str(env_values.get("ORDER_MANAGER_BRAIN_GATE_MODE") or "").strip().lower()
    if gate_mode != "shadow":
        issues.append("brain_gate_mode_not_shadow")
    workers = _float_env(env_values.get("ORDER_MANAGER_SERVICE_WORKERS"))
    if workers is None or int(workers) != 1:
        issues.append("order_manager_workers_not_1")
    timeout_sec = _float_env(env_values.get("BRAIN_TIMEOUT_SEC"))
    if timeout_cap_sec > 0.0 and (timeout_sec is None or timeout_sec > timeout_cap_sec):
        issues.append("timeout_above_cap")
    return issues


def _run_selection_sync(args: argparse.Namespace) -> tuple[dict[str, Any], list[str]]:
    if args.skip_selection_sync:
        return {"skipped": True, "env_changed": False}, []

    cmd = [
        sys.executable,
        str(APPLY_SCRIPT),
        "--benchmark",
        str(args.benchmark),
        "--env-profile",
        str(args.env_profile),
        "--output",
        str(args.selection_output),
        "--timeout-cap-sec",
        str(float(args.timeout_cap_sec)),
    ]
    if args.dry_run:
        cmd.append("--dry-run")

    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        text=True,
        capture_output=True,
    )
    stdout_lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    payload: dict[str, Any] = {}
    for line in reversed(stdout_lines):
        try:
            parsed = json.loads(line)
        except Exception:
            continue
        if isinstance(parsed, dict):
            payload = parsed
            break
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "apply_brain_model_selection failed")
    if not payload:
        payload = _load_json(args.selection_output)
    return payload, stdout_lines


def _fetch_ollama_status(chat_url: str, required_models: list[str], timeout_sec: float) -> dict[str, Any]:
    tags_url = _resolve_tags_url(chat_url)
    started = time.monotonic()
    try:
        resp = requests.get(tags_url, timeout=max(1.0, float(timeout_sec)))
        resp.raise_for_status()
        body = resp.json()
    except Exception as exc:
        return {
            "status": "error",
            "url": tags_url,
            "error": str(exc),
            "reachable": False,
            "latency_ms": round((time.monotonic() - started) * 1000.0, 1),
            "required_models": required_models,
            "installed_models": [],
            "missing_models": list(required_models),
        }

    models = body.get("models") if isinstance(body, dict) else None
    installed: list[str] = []
    for item in models or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or item.get("model") or "").strip()
        if name and name not in installed:
            installed.append(name)
    missing = [model for model in required_models if model not in installed]
    return {
        "status": "ok",
        "url": tags_url,
        "reachable": True,
        "latency_ms": round((time.monotonic() - started) * 1000.0, 1),
        "required_models": required_models,
        "installed_models": installed,
        "missing_models": missing,
    }


def _warmup_model(chat_url: str, model: str, timeout_sec: float) -> dict[str, Any]:
    payload = {
        "model": model,
        "stream": False,
        "think": False,
        "messages": [{"role": "user", "content": 'Return only {"ok":true}'}],
        "options": {"temperature": 0.0, "num_predict": 64},
    }
    started = time.monotonic()
    try:
        resp = requests.post(chat_url, json=payload, timeout=max(1.0, float(timeout_sec)))
        resp.raise_for_status()
        body = resp.json()
    except Exception as exc:
        return {
            "model": model,
            "status": "error",
            "error": str(exc),
            "latency_ms": round((time.monotonic() - started) * 1000.0, 1),
        }

    message = body.get("message") if isinstance(body, dict) else None
    ok = isinstance(message, dict)
    return {
        "model": model,
        "status": "ok" if ok else "error",
        "latency_ms": round((time.monotonic() - started) * 1000.0, 1),
        "error": "" if ok else "missing_message",
    }


def _true_range(high: float, low: float, prev_close: float | None) -> float:
    if prev_close is None:
        return high - low
    return max(high - low, abs(high - prev_close), abs(low - prev_close))


def _atr(values: list[float], period: int = 14) -> float | None:
    if len(values) < period:
        return None
    seed = sum(values[:period]) / float(period)
    current = seed
    for item in values[period:]:
        current = ((period - 1) * current + item) / float(period)
    return current


def _fetch_market_snapshot(max_tick_age_sec: float, max_spread_pips: float) -> dict[str, Any]:
    instrument = "USD_JPY"
    now_utc = datetime.now(timezone.utc)
    try:
        practice = str(get_secret("oanda_practice")).strip().lower() in {"1", "true", "yes"}
        api_base = "https://api-fxpractice.oanda.com" if practice else "https://api-fxtrade.oanda.com"
        account_id = get_secret("oanda_account_id")
        token = get_secret("oanda_token")
    except Exception as exc:
        return {"status": "error", "error": f"secrets_unavailable: {exc}"}

    headers = {"Authorization": f"Bearer {token}"}
    try:
        pricing = requests.get(
            f"{api_base}/v3/accounts/{account_id}/pricing",
            headers=headers,
            params={"instruments": instrument},
            timeout=12.0,
        )
        pricing.raise_for_status()
        pricing_body = pricing.json()
        price = pricing_body["prices"][0]
        bid = float(price["bids"][0]["price"])
        ask = float(price["asks"][0]["price"])
        spread_pips = (ask - bid) * 100.0
        tick_time = _parse_iso(price.get("time"))
        tick_age_sec = (now_utc - tick_time).total_seconds() if tick_time is not None else math.inf

        candles = requests.get(
            f"{api_base}/v3/instruments/{instrument}/candles",
            headers=headers,
            params={
                "granularity": "M1",
                "price": "M",
                "from": (now_utc - timedelta(minutes=180)).isoformat(),
                "count": 180,
            },
            timeout=12.0,
        )
        candles.raise_for_status()
        m1_rows = [row for row in candles.json().get("candles", []) if row.get("complete")]
    except Exception as exc:
        return {"status": "error", "error": str(exc)}

    closes: list[float] = []
    tr_values: list[float] = []
    prev_close: float | None = None
    for row in m1_rows:
        high = float(row["mid"]["h"])
        low = float(row["mid"]["l"])
        close = float(row["mid"]["c"])
        closes.append(close)
        tr_values.append(_true_range(high, low, prev_close))
        prev_close = close

    recent = closes[-6:] if len(closes) >= 6 else closes
    atr14 = _atr(tr_values, 14)
    market_ok = bool(spread_pips <= float(max_spread_pips) and tick_age_sec <= float(max_tick_age_sec))
    return {
        "status": "ok",
        "instrument": instrument,
        "bid": round(bid, 3),
        "ask": round(ask, 3),
        "spread_pips": round(spread_pips, 3),
        "tick_age_sec": round(tick_age_sec, 1) if math.isfinite(tick_age_sec) else None,
        "recent_range_pips_6m": round((max(recent) - min(recent)) * 100.0, 3) if recent else None,
        "atr_proxy_pips": round(atr14 * 100.0, 3) if atr14 is not None else None,
        "ok": market_ok,
    }


def main() -> int:
    args = _parse_args()
    now_utc = datetime.now(timezone.utc)

    benchmark_payload = _load_json(args.benchmark)
    benchmark_age_hours = _age_hours(benchmark_payload.get("generated_at"), now=now_utc)

    try:
        selection_sync, sync_stdout = _run_selection_sync(args)
        sync_error = ""
    except Exception as exc:
        selection_sync = {}
        sync_stdout = []
        sync_error = str(exc)

    selection_payload = selection_sync if selection_sync else _load_json(args.selection_output)
    selection_age_hours = _age_hours(selection_payload.get("generated_at"), now=now_utc)

    env_values = _parse_env_file(args.env_profile)
    profile_safety_issues = _profile_safety_issues(env_values, timeout_cap_sec=float(args.timeout_cap_sec))
    required_models = _collect_required_models(env_values, selection_payload)
    preflight_model = env_values.get("BRAIN_OLLAMA_MODEL") or selection_payload.get("preflight_model")
    benchmark_row = _find_benchmark_row(benchmark_payload, preflight_model)

    if args.skip_ollama:
        ollama_status = {
            "status": "skipped",
            "reachable": True,
            "required_models": required_models,
            "installed_models": [],
            "missing_models": [],
        }
    else:
        ollama_status = _fetch_ollama_status(
            env_values.get("BRAIN_OLLAMA_URL", "http://127.0.0.1:11434/api/chat"),
            required_models,
            timeout_sec=float(args.ollama_timeout_sec),
        )

    warmups: list[dict[str, Any]] = []
    if args.warmup and not args.skip_ollama and ollama_status.get("reachable") and not ollama_status.get("missing_models"):
        chat_url = env_values.get("BRAIN_OLLAMA_URL", "http://127.0.0.1:11434/api/chat")
        for model in required_models:
            warmups.append(_warmup_model(chat_url, model, timeout_sec=float(args.warmup_timeout_sec)))

    if args.skip_market:
        market_status = {"status": "skipped", "ok": True}
    else:
        market_status = _fetch_market_snapshot(
            max_tick_age_sec=float(args.max_tick_age_sec),
            max_spread_pips=float(args.max_spread_pips),
        )

    benchmark_fresh = benchmark_age_hours is not None and benchmark_age_hours <= float(args.max_benchmark_age_hours)
    selection_fresh = selection_age_hours is not None and selection_age_hours <= float(args.max_selection_age_hours)
    parse_pass_rate = benchmark_row.get("parse_pass_rate")
    latency_p95_ms = benchmark_row.get("latency_p95_ms")
    quality_gate_ok = False
    try:
        quality_gate_ok = (
            float(parse_pass_rate) >= float(args.min_parse_pass_rate)
            and float(latency_p95_ms) <= float(args.max_preflight_latency_ms)
        )
    except Exception:
        quality_gate_ok = False
    profile_enabled = _bool_env(env_values.get("BRAIN_ENABLED")) and _bool_env(
        env_values.get("ORDER_MANAGER_BRAIN_GATE_ENABLED")
    )
    ollama_ready = bool(ollama_status.get("reachable")) and not ollama_status.get("missing_models")
    if warmups:
        ollama_ready = ollama_ready and all(item.get("status") == "ok" for item in warmups)
    market_ready = bool(market_status.get("ok"))

    checks = {
        "profile_exists": args.env_profile.exists(),
        "profile_enabled": profile_enabled,
        "profile_safe": not profile_safety_issues,
        "selection_sync_ok": not sync_error,
        "benchmark_fresh": benchmark_fresh,
        "selection_fresh": selection_fresh,
        "quality_gate_ok": quality_gate_ok,
        "ollama_ready": True if args.skip_ollama else ollama_ready,
        "market_ready": True if args.skip_market else market_ready,
    }
    blockers = [key for key, ok in checks.items() if not ok]

    payload = {
        "generated_at": now_utc.isoformat(timespec="seconds"),
        "benchmark": {
            "path": str(args.benchmark),
            "generated_at": benchmark_payload.get("generated_at"),
            "age_hours": benchmark_age_hours,
            "max_age_hours": float(args.max_benchmark_age_hours),
            "selected_row": benchmark_row,
            "min_parse_pass_rate": float(args.min_parse_pass_rate),
            "max_preflight_latency_ms": float(args.max_preflight_latency_ms),
        },
        "selection": {
            "path": str(args.selection_output),
            "generated_at": selection_payload.get("generated_at"),
            "age_hours": selection_age_hours,
            "max_age_hours": float(args.max_selection_age_hours),
            "sync": selection_sync,
            "sync_stdout_tail": sync_stdout[-5:],
            "sync_error": sync_error,
        },
        "profile": {
            "path": str(args.env_profile),
            "exists": args.env_profile.exists(),
            "brain_enabled": env_values.get("BRAIN_ENABLED"),
            "order_manager_brain_gate_enabled": env_values.get("ORDER_MANAGER_BRAIN_GATE_ENABLED"),
            "brain_gate_mode": env_values.get("ORDER_MANAGER_BRAIN_GATE_MODE"),
            "ollama_model": env_values.get("BRAIN_OLLAMA_MODEL"),
            "pocket_allowlist": env_values.get("BRAIN_POCKET_ALLOWLIST"),
            "strategy_allowlist": env_values.get("BRAIN_STRATEGY_ALLOWLIST"),
            "sample_rate": env_values.get("BRAIN_SAMPLE_RATE"),
            "ttl_sec": env_values.get("BRAIN_TTL_SEC"),
            "timeout_sec": env_values.get("BRAIN_TIMEOUT_SEC"),
            "timeout_cap_sec": float(args.timeout_cap_sec),
            "safety_issues": profile_safety_issues,
        },
        "required_models": required_models,
        "ollama": {
            **ollama_status,
            "warmups": warmups,
        },
        "market": {
            **market_status,
            "max_spread_pips": float(args.max_spread_pips),
            "max_tick_age_sec": float(args.max_tick_age_sec),
        },
        "checks": checks,
        "ready": {
            "enable_recommended": not blockers,
            "blockers": blockers,
        },
        "recommended_commands": {
            "enable_canary": (
                "scripts/local_v2_stack.sh restart --profile trade_min "
                f"--env ops/env/local-v2-stack.env,{args.env_profile} "
                "--services quant-order-manager,quant-strategy-control"
            ),
            "status": (
                "scripts/local_v2_stack.sh status --profile trade_min "
                "--env ops/env/local-v2-stack.env"
            ),
        },
    }

    if not args.dry_run:
        _write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
