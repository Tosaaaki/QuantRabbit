"""Portfolio inventory reconciliation (weakness ledger W24).

Accurate stocktaking means three truths agree every cycle: what the broker
actually holds, what the append-only ledger claims, and which lane owns
each position.  This module seals that three-way match: every broker
position must be either lane-owned (with a thesis state) or explicitly
manual NO_TOUCH; every ledger-open position must exist at the broker; net
per-currency exposure is recomputed from broker truth.  Any mismatch flags
the inventory UNRECONCILED — a fail-closed state a wired admission gate
must treat as no-new-entries.  Measurement only; no broker mutation.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from quant_rabbit.currency_exposure_guard import net_currency_exposure

CONTRACT = "QR_PORTFOLIO_INVENTORY_RECONCILIATION_V1"
THESIS_STATES = frozenset({"STILL_VALID", "WEAKENED", "BROKEN", "EXPIRED"})


class InventoryReconciliationError(ValueError):
    """Raised when reconciliation inputs are malformed."""


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _position_key(row: Mapping[str, Any]) -> str:
    key = str(row.get("position_id") or "").strip()
    if not key:
        raise InventoryReconciliationError("position_id is required")
    return key


def build_portfolio_inventory(
    *,
    broker_positions: Sequence[Mapping[str, Any]],
    ledger_open_positions: Sequence[Mapping[str, Any]],
    manual_no_touch_ids: Sequence[str],
    nav_account_currency: float,
    as_of_utc: datetime,
    broker_snapshot_sha256: str,
    ledger_tip_sha256: str,
) -> dict[str, Any]:
    """Seal a three-way inventory match; any mismatch is UNRECONCILED."""

    if as_of_utc.tzinfo is None:
        raise InventoryReconciliationError("inventory clock must be timezone-aware")
    if not isinstance(nav_account_currency, (int, float)) or nav_account_currency <= 0:
        raise InventoryReconciliationError("nav must be positive")
    manual_ids = {str(item) for item in manual_no_touch_ids}
    ledger_by_id: dict[str, Mapping[str, Any]] = {}
    for row in ledger_open_positions:
        key = _position_key(row)
        if key in ledger_by_id:
            raise InventoryReconciliationError(f"duplicate ledger position: {key}")
        thesis = str(row.get("thesis_state") or "")
        if thesis not in THESIS_STATES:
            raise InventoryReconciliationError(
                f"ledger position {key} thesis state is invalid"
            )
        if not str(row.get("lane_id") or "").strip():
            raise InventoryReconciliationError(
                f"ledger position {key} has no owning lane"
            )
        ledger_by_id[key] = row

    rows: list[dict[str, Any]] = []
    untracked: list[str] = []
    seen_broker: set[str] = set()
    for row in broker_positions:
        key = _position_key(row)
        if key in seen_broker:
            raise InventoryReconciliationError(f"duplicate broker position: {key}")
        seen_broker.add(key)
        if key in manual_ids:
            ownership = "MANUAL_NO_TOUCH"
            lane = None
            thesis = None
        elif key in ledger_by_id:
            ownership = "LANE_OWNED"
            lane = str(ledger_by_id[key]["lane_id"])
            thesis = str(ledger_by_id[key]["thesis_state"])
        else:
            ownership = "UNTRACKED_AT_BROKER"
            lane = None
            thesis = None
            untracked.append(key)
        rows.append(
            {
                "position_id": key,
                "pair": str(row.get("pair")),
                "side": str(row.get("side")),
                "nav_exposure_fraction": float(row.get("nav_exposure_fraction")),
                "ownership": ownership,
                "lane_id": lane,
                "thesis_state": thesis,
            }
        )
    orphaned = sorted(set(ledger_by_id) - seen_broker)
    tradeable = [
        row for row in rows if row["ownership"] != "MANUAL_NO_TOUCH"
    ]
    exposure = net_currency_exposure(tradeable) if tradeable else {}
    reconciled = not untracked and not orphaned
    broken_theses = sorted(
        row["position_id"]
        for row in rows
        if row["thesis_state"] in {"BROKEN", "EXPIRED"}
    )
    body: dict[str, Any] = {
        "contract": CONTRACT,
        "schema_version": 1,
        "as_of_utc": as_of_utc.astimezone(timezone.utc).isoformat(),
        "broker_snapshot_sha256": str(broker_snapshot_sha256),
        "ledger_tip_sha256": str(ledger_tip_sha256),
        "position_rows": rows,
        "position_count": len(rows),
        "lane_owned_count": sum(r["ownership"] == "LANE_OWNED" for r in rows),
        "manual_no_touch_count": sum(
            r["ownership"] == "MANUAL_NO_TOUCH" for r in rows
        ),
        "untracked_at_broker": sorted(untracked),
        "orphaned_in_ledger": orphaned,
        "net_currency_exposure_excl_manual": exposure,
        "positions_requiring_thesis_review": broken_theses,
        "reconciled": reconciled,
        "status": "RECONCILED" if reconciled else "UNRECONCILED_FAIL_CLOSED",
        "new_entries_admissible": reconciled,
        "manual_positions_counted_in_trader_pnl": False,
        "order_authority": "NONE",
        "live_permission": False,
        "broker_mutation_allowed": False,
    }
    return {**body, "inventory_sha256": _canonical_sha(body)}
