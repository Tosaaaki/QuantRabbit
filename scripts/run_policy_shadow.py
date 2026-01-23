#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from analytics.policy_generator import (
    DEFAULT_OPENAI_MODEL,
    build_policy_prompt,
    call_openai,
    parse_policy_diff,
    summarize_policy_rows,
)
from analytics.policy_ledger import PolicyLedger
from analytics.policy_mart import PolicyMartClient
from analytics.policy_diff import normalize_policy_diff
from utils.secrets import get_secret


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate GPT shadow policy_diff (no apply).")
    ap.add_argument("--project", default=None)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--table", default=None)
    ap.add_argument("--lookback-days", type=int, default=14)
    ap.add_argument("--min-trades", type=int, default=0)
    ap.add_argument("--output", default="logs/policy_diff_shadow.json")
    ap.add_argument("--model", default=DEFAULT_OPENAI_MODEL)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--no-ledger", action="store_true")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    if os.getenv("POLICY_SHADOW_ENABLED", "0").strip().lower() not in {"1", "true", "yes", "on"}:
        logging.info("[POLICY_SHADOW] disabled (POLICY_SHADOW_ENABLED=0)")
        return

    client = PolicyMartClient(
        project_id=args.project,
        dataset_id=args.dataset or os.getenv("BQ_DATASET", "quantrabbit"),
        trades_table=args.table or os.getenv("BQ_TRADES_TABLE", "trades_raw"),
    )
    rows = client.fetch_rows(lookback_days=args.lookback_days, min_trades=args.min_trades)
    summary = summarize_policy_rows(rows)
    prompt = build_policy_prompt(summary)

    try:
        api_key = get_secret("openai_api_key")
    except Exception:
        api_key = None
    text = call_openai(
        prompt,
        api_key=api_key,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    if not text:
        payload = normalize_policy_diff(
            {"no_change": True, "reason": "shadow_failed", "source": "gpt_shadow"},
            source="gpt_shadow",
        )
    else:
        parsed = parse_policy_diff(text, source="gpt_shadow")
        payload = parsed if parsed else normalize_policy_diff(
            {"no_change": True, "reason": "shadow_invalid", "source": "gpt_shadow"},
            source="gpt_shadow",
        )

    out_path = Path(args.output)
    _write_json(out_path, payload)
    logging.info("[POLICY_SHADOW] diff written: %s", out_path)

    if not args.no_ledger:
        ledger = PolicyLedger(project_id=args.project)
        ledger.record(payload, status="shadow", summary=summary)


if __name__ == "__main__":
    main()
