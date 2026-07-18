#!/usr/bin/env python3
"""THE BOT: one executable cycle of the composed shadow trading brain.

This is the runnable form of everything built tonight.  One invocation:
market panel -> market-conditions read (the board) -> per-candidate
measured regime -> family routing -> composed brain (inventory, read
handoff reserved for CODEX_AI_TRADER, metacognition, gates, sizing) ->
sealed cycle artifact.  Configuration comes from the sealed tuning
contract, never from code edits.

Modes:
  --demo          synthetic panel and inputs; proves the loop end to end
  --inputs FILE   real inputs JSON (Codex points this at live feeds)

The bot emits shadow intents only: order_authority NONE, live permission
false.  Wiring it to LiveOrderGateway is Codex's step behind the
operator's approval, exactly as the runbook orders.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from quant_rabbit.market_conditions_reader import read_market_conditions
from quant_rabbit.regime_classifier_shadow import classify_regime
from quant_rabbit.regime_family_router import build_family_catalog
from quant_rabbit.worker_arsenal import WorkerArsenal, WorkerSpec
from quant_rabbit.shadow_trading_brain import (
    LAYER2_SHADOW_SOURCE,
    run_shadow_brain_cycle,
)

UTC = timezone.utc


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_contract(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    body = {k: v for k, v in value.items() if k != "tuning_contract_sha256"}
    if value.get("tuning_contract_sha256") != _canonical_sha(body):
        raise ValueError("tuning contract digest is invalid")
    return value


def _default_family_catalog() -> dict[str, Any]:
    # Cells from the sealed all-weather attribution: survivor is eligible
    # where it measured profitable; RANGE_HIGH / TREND_LOW stay uncovered
    # until lane F earns them.
    return build_family_catalog(
        [
            {
                "family_id": "S5_SURVIVOR",
                "regime_affinity": ["TREND", "SQUEEZE", "RANGE"],
                "vol_affinity": ["LOW", "HIGH"],
                "promotion_state": "VALIDATION_REPLICATED",
            }
        ]
    )


def _demo_inputs(now: datetime) -> dict[str, Any]:
    """Synthetic but internally consistent inputs proving the whole loop."""

    start = now - timedelta(minutes=200)
    panel: dict[str, list[dict[str, Any]]] = {}
    for index, pair in enumerate(("EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD")):
        drift = 0.0004 * (1 if index % 2 == 0 else -1)
        base = 1.10 + index * 0.2
        closes = [base + i * drift * (1.4 if i % 2 else 1.0) for i in range(140)]
        pip = 0.01 if pair.endswith("JPY") else 0.0001
        half_spread = 0.6 * pip
        panel[pair] = [
            {"time": start + timedelta(minutes=i), "close": c,
             "bid_close": c - half_spread, "ask_close": c + half_spread}
            for i, c in enumerate(closes)
        ]
    sha = "a" * 64
    return {
        "cycle_id": now.strftime("%Y%m%dT%H%M%SZ") + "-DEMO",
        "panel": panel,
        "evidence_packet_sha256": sha,
        "proprietary_indicator_sha256": sha,
        "broker_positions": [
            {"position_id": "p1", "pair": "EUR_USD", "side": "LONG", "nav_exposure_fraction": 0.2}
        ],
        "ledger_open_positions": [
            {"position_id": "p1", "lane_id": "S5_SURVIVOR", "thesis_state": "STILL_VALID"}
        ],
        "manual_no_touch_ids": [],
        "nav_account_currency": 291_030.0,
        "broker_snapshot_sha256": sha,
        "ledger_tip_sha256": "b" * 64,
        "market_read": {
            "read_source": LAYER2_SHADOW_SOURCE,
            "declared_regime": "TREND",
            "pair_reads": [
                {
                    "pair": "GBP_USD",
                    "action": "GO",
                    "side": "LONG",
                    "narrative_sha256": sha,
                    "predicted_direction": "LONG",
                    "conviction_conditions": [
                        ["REGIME_ALIGNED", True],
                        ["SESSION_FAVORABLE", True],
                        ["NO_EVENT_WINDOW", True],
                    ],
                }
            ],
        },
        "candidates": [
            {
                "pair": "GBP_USD",
                "side": "LONG",
                "hold_minutes": 180,
                "family_id": "S5_SURVIVOR",
                "nav_exposure_fraction": 0.2,
            }
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--as-of")
    args = parser.parse_args()
    if bool(args.demo) == bool(args.inputs):
        raise ValueError("exactly one of --demo or --inputs is required")
    contract = _load_contract(args.contract)
    now = (
        datetime.fromisoformat(args.as_of)
        if args.as_of
        else datetime(2026, 7, 15, 9, 0, tzinfo=UTC)
    )
    if now.tzinfo is None:
        raise ValueError("--as-of must be timezone-aware")

    if args.demo:
        inputs = _demo_inputs(now)
    else:
        raw = json.loads(args.inputs.read_text(encoding="utf-8"))
        for pair, candles in raw["panel"].items():
            for candle in candles:
                candle["time"] = datetime.fromisoformat(candle["time"])
        inputs = raw

    conditions = read_market_conditions(inputs["panel"], as_of_utc=now)

    catalog = _default_family_catalog()
    measured_by_pair: dict[str, Any] = {}
    for candidate in inputs["candidates"]:
        pair = str(candidate["pair"]).upper()
        if pair in inputs["panel"] and pair not in measured_by_pair:
            measured_by_pair[pair] = classify_regime(
                inputs["panel"][pair], as_of_utc=now
            )

    # ---- worker arsenal arming (W40) ----------------------------------
    arsenal_path = Path("research/data/worker_arsenal_contract_v1.json")
    arm_decisions_out: list[dict[str, Any]] = []
    arsenal_contract_sha = None
    if arsenal_path.exists():
        arsenal_doc = json.loads(arsenal_path.read_text(encoding="utf-8"))
        check = {k: v for k, v in arsenal_doc.items() if k != "contract_sha256"}
        if _canonical_sha(check) != arsenal_doc.get("contract_sha256"):
            raise ValueError("worker arsenal contract digest invalid")
        arsenal_contract_sha = arsenal_doc["contract_sha256"]
        # Declared runtime protections per entry style (documented mapping;
        # numeric protections move into contract v2).
        style_defaults = {
            "MARKET": {"max_concurrent": 3, "time_stop_minutes": 60, "per_position_leverage": 4.3},
            "LIMIT_PASSIVE": {"max_concurrent": 5, "time_stop_minutes": 240, "per_position_leverage": 0.9},
        }
        arsenal = WorkerArsenal()
        for w in arsenal_doc["workers"]:
            d = style_defaults[w["entry_style"]]
            arsenal.register(WorkerSpec(
                worker_id=w["worker_id"], cell=w["cell"],
                pairs=tuple(w["pairs"]), max_spread_pips=float(w["max_spread_pips"]),
                entry_style=w["entry_style"], kill_switch="per contract",
                **d,
            ))
        arsenal.ai_heartbeat(now)
        # spreads measured from the panel's last candle (ask - bid at close)
        spreads: dict[str, float] = {}
        for pair, candles in inputs["panel"].items():
            last = candles[-1]
            bid = last.get("bid_close")
            ask = last.get("ask_close")
            if bid is not None and ask is not None:
                pip = 0.01 if pair.endswith("JPY") else 0.0001
                spreads[pair] = round((float(ask) - float(bid)) / pip, 2)
        # Measure every panel pair (worker habitats are wider than the
        # candidate list) and arm each worker-pair on ITS pair's cell.
        measured_all: dict[str, Any] = dict(measured_by_pair)
        for pair, candles in inputs["panel"].items():
            if pair not in measured_all:
                try:
                    measured_all[pair] = classify_regime(candles, as_of_utc=now)
                except Exception:
                    continue  # unmeasured pair stays unmeasured -> disarmed
        for pair, measured in measured_all.items():
            cell = f"{measured['regime']}_{measured['vol_state']}"
            if measured["regime"] in ("UNCLEAR", "EVENT"):
                cell = measured["regime"]
            for d in arsenal.arm_cycle(now=now, measured_cell=cell,
                                       spreads_pips=spreads):
                if d.pair == pair:
                    arm_decisions_out.append({
                        "worker_id": d.worker_id, "pair": d.pair,
                        "measured_cell": cell, "armed": d.armed,
                        "reason": d.reason,
                    })

    first_pair = str(inputs["candidates"][0]["pair"]).upper() if inputs["candidates"] else None
    cycle = run_shadow_brain_cycle(
        cycle_id=inputs["cycle_id"],
        decision_utc=now,
        evidence_packet_sha256=inputs["evidence_packet_sha256"],
        proprietary_indicator_sha256=conditions["snapshot_sha256"],
        broker_positions=inputs["broker_positions"],
        ledger_open_positions=inputs["ledger_open_positions"],
        manual_no_touch_ids=inputs["manual_no_touch_ids"],
        nav_account_currency=float(inputs["nav_account_currency"]),
        broker_snapshot_sha256=inputs["broker_snapshot_sha256"],
        ledger_tip_sha256=inputs["ledger_tip_sha256"],
        market_read=inputs["market_read"],
        supervision_scorecard=inputs.get("supervision_scorecard"),
        candidates=inputs["candidates"],
        measured_regime=measured_by_pair.get(first_pair),
        family_catalog=catalog,
    )

    body: dict[str, Any] = {
        "contract": "QR_SHADOW_BOT_CYCLE_RUN_V1",
        "schema_version": 1,
        "tuning_contract_sha256": contract["tuning_contract_sha256"],
        "market_conditions": {
            "board_reading": conditions["board_reading"],
            "dominant_regime": conditions["dominant_regime"],
            "trend_breadth": conditions["trend_breadth"],
            "high_vol_share": conditions["high_vol_share"],
            "dominant_theme": conditions["dominant_theme"],
            "snapshot_sha256": conditions["snapshot_sha256"],
        },
        "brain_cycle": cycle,
        "worker_arsenal": {
            "contract_sha256": arsenal_contract_sha,
            "arm_decisions": arm_decisions_out,
        },
        "mode": "DEMO" if args.demo else "INPUTS",
        "live_wiring_owner": "CODEX",
        "shadow_only": True,
        "order_authority": "NONE",
        "live_permission": False,
    }
    sealed = {**body, "run_sha256": _canonical_sha(body)}
    payload = json.dumps(sealed, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n"
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{args.output.name}.", suffix=".tmp", dir=args.output.parent
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_name, args.output)
    print(
        json.dumps(
            {
                "status": "BOT_CYCLE_SEALED",
                "board_reading": conditions["board_reading"],
                "dominant_theme": conditions["dominant_theme"],
                "admitted": cycle["admitted_candidate_count"],
                "armed_workers": sorted({d["worker_id"] for d in arm_decisions_out if d["armed"]}),
                "candidates": [
                    {"pair": r["pair"], "admitted": r["admitted"], "reasons": r["refusal_reasons"], "risk": r["shadow_risk_fraction"]}
                    for r in cycle["candidate_rows"]
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
