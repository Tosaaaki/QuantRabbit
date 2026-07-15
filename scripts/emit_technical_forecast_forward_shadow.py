#!/usr/bin/env python3
"""Emit the frozen technical forecast candidate into a read-only ledger."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quant_rabbit.broker.oanda import OandaReadOnlyClient  # noqa: E402
from quant_rabbit.analysis.candles import (  # noqa: E402
    TECHNICAL_CANDLE_INDICATOR_WARMUP_MIN_CLEAN_COUNT,
    fetch_technical_candles_via_client,
)
from quant_rabbit.instruments import DEFAULT_TRADER_PAIRS  # noqa: E402
from quant_rabbit.paths import DEFAULT_PAIR_CHARTS  # noqa: E402
from quant_rabbit.technical_forecast_forward_shadow import (  # noqa: E402
    append_shadow_once,
    build_forward_shadow,
    forward_collection_window,
    load_forward_candidate,
    write_shadow_atomic,
)


def main() -> int:
    args = _parse_args()
    candidate = load_forward_candidate(args.candidate)
    observed = datetime.now(timezone.utc)
    window = forward_collection_window(candidate, observed)
    if window["open"] is not True:
        print(
            json.dumps(
                {
                    **window,
                    "shadow_only": True,
                    "broker_read": False,
                    "broker_mutation": False,
                    "output_unchanged": True,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    candidate_sha256 = _file_sha256(args.candidate)
    if _decision_already_recorded(
        args.ledger,
        candidate_sha256=candidate_sha256,
        terminal_m5_timestamp_utc=str(window["terminal_m5_timestamp_utc"]),
    ):
        print(
            json.dumps(
                {
                    **window,
                    "status": "ALREADY_EMITTED",
                    "shadow_only": True,
                    "broker_read": False,
                    "broker_mutation": False,
                    "output_unchanged": True,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    client = OandaReadOnlyClient()
    if args.fresh_m5:
        pairs = sorted(DEFAULT_TRADER_PAIRS)
        pair_charts = _fresh_m5_pair_charts(
            client,
            pairs,
            observed_at_utc=observed,
            workers=args.fresh_m5_workers,
        )
    else:
        pair_charts = _load_object(args.pair_charts)
        pairs = sorted(
            {
                str(chart.get("pair") or "").upper()
                for chart in pair_charts.get("charts") or []
                if isinstance(chart, dict) and chart.get("pair")
            }
        )
    quotes = client.quotes(pairs)
    shadow = build_forward_shadow(
        candidate,
        pair_charts,
        quotes,
        candidate_sha256=candidate_sha256,
        observed_at_utc=observed,
    )
    write_shadow_atomic(args.output, shadow)
    appended = append_shadow_once(args.ledger, shadow)
    print(
        json.dumps(
            {
                "status": shadow.get("status"),
                "decision_id": shadow.get("decision_id"),
                "selected_pair_count": shadow.get("selected_pair_count", 0),
                "ledger_appended": appended,
                "shadow_only": True,
                "broker_mutation": False,
                "output": str(args.output.resolve()),
                "ledger": str(args.ledger.resolve()),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        type=Path,
        default=ROOT / "config" / "technical_forecast_forward_candidate_v1.json",
    )
    parser.add_argument("--pair-charts", type=Path, default=DEFAULT_PAIR_CHARTS)
    parser.add_argument("--fresh-m5", action="store_true")
    parser.add_argument("--fresh-m5-workers", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "technical_forecast_forward_shadow.json",
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        default=ROOT / "data" / "technical_forecast_forward_shadow_ledger.jsonl",
    )
    return parser.parse_args()


def _fresh_m5_pair_charts(
    client: OandaReadOnlyClient,
    pairs: list[str],
    *,
    observed_at_utc: datetime,
    workers: int,
) -> dict[str, Any]:
    if not 1 <= workers <= 28:
        raise ValueError("fresh M5 workers must be inside 1..28")

    def fetch(pair: str) -> dict[str, Any]:
        try:
            batch = fetch_technical_candles_via_client(
                client,
                pair,
                "M5",
                count=30,
            )
            clean_tail_count = batch.integrity.get("recent_clean_tail_count")
            if (
                clean_tail_count.__class__ is not int
                or not 0 <= clean_tail_count <= len(batch.candles)
                or batch.integrity.get("indicator_warmup_min_clean_count")
                != TECHNICAL_CANDLE_INDICATOR_WARMUP_MIN_CLEAN_COUNT
            ):
                candles = ()
            else:
                candles = (
                    batch.candles[-clean_tail_count:]
                    if clean_tail_count > 0
                    else ()
                )
            recent = [
                {
                    "t": candle.timestamp_utc.isoformat(),
                    "o": candle.open,
                    "h": candle.high,
                    "l": candle.low,
                    "c": candle.close,
                    "v": candle.volume,
                    "complete": candle.complete,
                }
                for candle in candles[-30:]
            ]
            return {
                "pair": pair,
                "views": [
                    {
                        "granularity": "M5",
                        "recent_candles": recent,
                        "candle_integrity": batch.integrity,
                    }
                ],
            }
        except Exception as exc:  # pragma: no cover - live network boundary
            return {
                "pair": pair,
                "views": [{"granularity": "M5", "recent_candles": []}],
                "fresh_m5_fetch_error": str(exc)[:256],
            }

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        charts = list(executor.map(fetch, pairs))
    return {
        "generated_at_utc": observed_at_utc.isoformat(),
        "pairs_requested": len(pairs),
        "charts": charts,
        "fresh_m5_read_only": True,
    }


def _load_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON object: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be an object: {path}")
    return payload


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _decision_already_recorded(
    path: Path,
    *,
    candidate_sha256: str,
    terminal_m5_timestamp_utc: str,
) -> bool:
    if not path.exists():
        return False
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    continue
                if (
                    row.get("status") == "EMITTED"
                    and row.get("candidate_sha256") == candidate_sha256
                    and row.get("terminal_m5_timestamp_utc")
                    == terminal_m5_timestamp_utc
                ):
                    return True
    except (OSError, json.JSONDecodeError):
        return False
    return False


if __name__ == "__main__":
    raise SystemExit(main())
