#!/usr/bin/env python3
"""GET-only progressive-live mode/sizing preflight with a durable ledger.

This executable has no Gateway and no broker write client.  It records why the
runtime remains SHADOW_ONLY/FREEZE_NEW or, after explicit release sealing, the
broker-minimum THROTTLED_LIVE lot that may proceed to the separate promotion
validator.  It never describes readiness as an order attempt.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

from quant_rabbit.broker.oanda import OandaReadOnlyClient
from quant_rabbit.fast_bot_promotion import (
    FORWARD_ADMISSION_CONTRACT,
    RISK_CONTRACT,
    build_sizing_receipt,
)
from quant_rabbit.inventory_controller import InventoryController, broker_position_identity
from quant_rabbit.trade_readiness import (
    ExplicitRiskLimits,
    RuntimeMode,
    SignalSizingInput,
    estimate_account_stress_mcp,
    screen_trade_readiness,
    size_signal_for_runtime_mode,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATE_ROOT = (
    Path.home() / ".codex" / "state" / "quantrabbit" / "progressive-live-preflight-v1"
)
MODE_LEDGER_CONTRACT = "QR_PROGRESSIVE_LIVE_MODE_LEDGER_V1"
PREFLIGHT_CONTRACT = "QR_PROGRESSIVE_LIVE_PREFLIGHT_V1"


class PreflightBlocked(RuntimeError):
    """The preflight cannot prove a zero-authority or admission invariant."""


def canonical_sha(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreflightBlocked(f"JSON_INVALID:{path.name}") from exc
    if not isinstance(value, dict):
        raise PreflightBlocked(f"JSON_OBJECT_REQUIRED:{path.name}")
    return value


def load_sealer_module() -> Any:
    path = REPO_ROOT / "tools" / "seal_progressive_live_risk_contract.py"
    spec = importlib.util.spec_from_file_location("qr_progressive_live_sealer", path)
    if spec is None or spec.loader is None:
        raise PreflightBlocked("SEALER_IMPORT_FAILED")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sealed(value: Mapping[str, Any], *, contract: str, seal_key: str) -> bool:
    if value.get("contract") != contract:
        return False
    body = {key: item for key, item in value.items() if key != seal_key}
    return value.get(seal_key) == canonical_sha(body)


def verify_release_receipt(
    release: Mapping[str, Any],
    *,
    approval_packet_sha256: str,
    software_manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    body = {
        key: value
        for key, value in release.items()
        if key != "release_receipt_sha256"
    }
    if (
        release.get("contract") != "QR_PROGRESSIVE_LIVE_RELEASE_RECEIPT_V1"
        or release.get("status") != "SEALED_AWAITING_FRESH_ACCOUNT_GATE"
        or release.get("release_receipt_sha256") != canonical_sha(body)
        or release.get("approval_packet_sha256") != approval_packet_sha256
        or release.get("live_permission") is not False
        or release.get("broker_mutation_allowed") is not False
        or release.get("software_manifest") != software_manifest
    ):
        raise PreflightBlocked("RELEASE_RECEIPT_INVALID_OR_DRIFTED")
    admission = release.get("forward_admission")
    risk = release.get("risk_contract")
    if not isinstance(admission, Mapping) or not _sealed(
        admission,
        contract=FORWARD_ADMISSION_CONTRACT,
        seal_key="admission_sha256",
    ):
        raise PreflightBlocked("PROGRESSIVE_ADMISSION_SEAL_INVALID")
    if not isinstance(risk, Mapping) or not _sealed(
        risk,
        contract=RISK_CONTRACT,
        seal_key="risk_contract_sha256",
    ):
        raise PreflightBlocked("RISK_CONTRACT_SEAL_INVALID")
    if (
        risk.get("accepted_by_user") is not True
        or risk.get("acceptance_source") != "EXPLICIT_USER_DECISION"
        or admission.get("admission_mode") != "PROGRESSIVE_MICRO_LIVE"
        or admission.get("fixed_sample_wait_required_for_micro_live") is not False
        or admission.get("micro_live_only") is not True
    ):
        raise PreflightBlocked("RELEASE_AUTHORIZATION_INVALID")
    return dict(admission), dict(risk)


def _number(value: object, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise PreflightBlocked(f"ACCOUNT_VALUE_INVALID:{name}") from exc
    if not math.isfinite(parsed):
        raise PreflightBlocked(f"ACCOUNT_VALUE_INVALID:{name}")
    return parsed


def _instrument_map(rows: tuple[dict, ...]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("name") or ""): dict(row)
        for row in rows
        if row.get("name") in {"EUR_USD", "USD_JPY"}
    }


def signal_sizing_input(
    signal: Mapping[str, Any],
    *,
    quotes: Mapping[str, Any],
    instruments: Mapping[str, Mapping[str, Any]],
    stress_pips: float,
) -> SignalSizingInput:
    pair = str(signal.get("pair") or "")
    side = str(signal.get("side") or "")
    quote = quotes.get(pair)
    instrument = instruments.get(pair)
    if quote is None or not isinstance(instrument, Mapping) or side not in {"LONG", "SHORT"}:
        raise PreflightBlocked("SIGNAL_MARKET_MODEL_MISSING")
    usd_jpy = quotes.get("USD_JPY")
    if usd_jpy is None:
        raise PreflightBlocked("USD_JPY_CONVERSION_MISSING")
    usd_to_jpy = float(usd_jpy.mid)
    base, quote_currency = pair.split("_", 1)
    base_to_jpy = (
        float(quote.mid) * usd_to_jpy
        if base == "EUR"
        else usd_to_jpy
        if base == "USD"
        else None
    )
    quote_to_jpy = 1.0 if quote_currency == "JPY" else usd_to_jpy
    if base_to_jpy is None:
        raise PreflightBlocked("BASE_CURRENCY_CONVERSION_MISSING")
    margin_rate = _number(instrument.get("marginRate"), "marginRate")
    minimum = int(float(instrument.get("minimumTradeSize") or 0))
    maximum = int(float(instrument.get("maximumOrderUnits") or 0))
    pip_location = int(instrument.get("pipLocation"))
    pip_size = 10.0 ** pip_location
    entry = _number(signal.get("entry"), "entry")
    stop_loss = _number(signal.get("stop_loss"), "stop_loss")
    stop_pips = abs(entry - stop_loss) / pip_size
    pip_value_jpy_per_unit = pip_size * quote_to_jpy
    planned_loss_per_unit = stop_pips * pip_value_jpy_per_unit
    margin_per_unit = base_to_jpy * margin_rate
    stress_loss_per_unit = stress_pips * pip_value_jpy_per_unit
    signed = 1.0 if side == "LONG" else -1.0
    factor_delta = {
        base: signed * base_to_jpy,
        quote_currency: -signed * float(quote.mid) * quote_to_jpy,
    }
    if minimum <= 0 or maximum < minimum or planned_loss_per_unit <= 0.0:
        raise PreflightBlocked("INSTRUMENT_OR_STOP_MODEL_INVALID")
    return SignalSizingInput(
        requested_units=maximum,
        broker_minimum_units=minimum,
        margin_jpy_per_unit=margin_per_unit,
        closeout_margin_jpy_per_unit=margin_per_unit,
        stress_closeout_margin_jpy_per_unit=margin_per_unit + stress_loss_per_unit,
        loss_jpy_per_unit=planned_loss_per_unit,
        factor_delta_jpy_per_unit=factor_delta,
    )


def _limits(candidate: Mapping[str, Any], *, admission_sha: str | None, risk_sha: str | None) -> ExplicitRiskLimits:
    return ExplicitRiskLimits(
        max_loss_per_order_jpy=_number(candidate.get("max_loss_per_order_jpy"), "max_loss_per_order_jpy"),
        stop_drawdown_jpy=_number(candidate.get("stop_drawdown_jpy"), "stop_drawdown_jpy"),
        minimum_margin_buffer_jpy=_number(candidate.get("minimum_margin_buffer_jpy"), "minimum_margin_buffer_jpy"),
        max_post_entry_current_mcp=_number(candidate.get("max_post_entry_current_mcp"), "max_post_entry_current_mcp"),
        max_post_entry_stress_mcp=_number(candidate.get("max_post_entry_stress_mcp"), "max_post_entry_stress_mcp"),
        max_currency_factor_nav_multiple=_number(candidate.get("max_currency_factor_nav_multiple"), "max_currency_factor_nav_multiple"),
        max_bot_positions=int(candidate.get("max_bot_positions") or 0),
        mode_hysteresis_mcp=_number(candidate.get("mode_hysteresis_mcp"), "mode_hysteresis_mcp"),
        forward_proof_sha256=admission_sha,
        risk_contract_sha256=risk_sha,
    )


def _signal_current(signal: Mapping[str, Any], now: datetime) -> bool:
    try:
        quote_at = datetime.fromisoformat(
            str(signal.get("quote_timestamp_utc") or "").replace("Z", "+00:00")
        ).astimezone(timezone.utc)
        ttl = int(signal.get("entry_ttl_seconds") or 0)
    except (ValueError, TypeError):
        return False
    return (
        signal.get("shadow_only") is True
        and signal.get("live_permission") is False
        and ttl > 0
        and quote_at <= now <= quote_at + timedelta(seconds=ttl)
    )


def _inventory_readback(
    path: Path | None,
    *,
    campaign_id: str | None,
    now: datetime,
    nav_jpy: float,
    initialize_if_missing: bool,
) -> dict[str, Any]:
    if path is None or campaign_id is None:
        return {
            "exists": False,
            "initialized_from_fresh_nav": False,
            "profit_lock_configured": False,
            "inventory_state": "RUNNING",
            "cooldown_elapsed": False,
            "campaign_drawdown_jpy": 0.0,
            "cycle_start_nav_jpy": None,
            "cycle_peak_nav_jpy": None,
        }
    existed = path.is_file()
    if not existed and not initialize_if_missing:
        return {
            "exists": False,
            "initialized_from_fresh_nav": False,
            "profit_lock_configured": False,
            "inventory_state": "RUNNING",
            "cooldown_elapsed": False,
            "campaign_drawdown_jpy": 0.0,
            "cycle_start_nav_jpy": None,
            "cycle_peak_nav_jpy": None,
        }
    controller = InventoryController.open(path, campaign_id=campaign_id, now_utc=now)
    initialized = False
    if controller.cycle_start_nav_jpy is None:
        if not initialize_if_missing:
            raise PreflightBlocked("INVENTORY_PROFIT_LOCK_BASELINE_MISSING")
        controller.configure_profit_lock(cycle_start_nav_jpy=nav_jpy, now_utc=now)
        initialized = True
    peak = float(controller.cycle_peak_nav_jpy or controller.cycle_start_nav_jpy or nav_jpy)
    return {
        "exists": True,
        "initialized_from_fresh_nav": initialized,
        "profit_lock_configured": controller.cycle_start_nav_jpy is not None,
        "inventory_state": controller.state.value,
        "cooldown_elapsed": controller.cooldown_elapsed(now),
        "campaign_drawdown_jpy": max(0.0, peak - nav_jpy),
        "cycle_start_nav_jpy": controller.cycle_start_nav_jpy,
        "cycle_peak_nav_jpy": controller.cycle_peak_nav_jpy,
    }


def append_event_once(path: Path, event: Mapping[str, Any]) -> bool:
    event_id = str(event.get("event_id") or "")
    if not event_id:
        raise PreflightBlocked("MODE_EVENT_ID_REQUIRED")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    prior = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise PreflightBlocked("MODE_LEDGER_CORRUPT") from exc
                if prior.get("event_id") == event_id:
                    return False
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(event), ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    return True


def _snapshot_binding(snapshot: Any, raw_account: Mapping[str, Any]) -> dict[str, Any]:
    account = raw_account.get("account") if isinstance(raw_account.get("account"), Mapping) else raw_account
    return {
        "fetched_at_utc": snapshot.fetched_at_utc.isoformat(),
        "account": {
            key: account.get(key)
            for key in (
                "NAV",
                "marginUsed",
                "marginAvailable",
                "marginCloseoutNAV",
                "marginCloseoutMarginUsed",
                "marginCloseoutPercent",
                "lastTransactionID",
            )
        },
        "positions": [
            {
                "trade_id": position.trade_id,
                "pair": position.pair,
                "side": position.side.value,
                "units": position.units,
                "owner": position.owner.value,
            }
            for position in snapshot.positions
        ],
        "orders": [
            {
                "order_id": order.order_id,
                "type": order.order_type,
                "trade_id": order.trade_id,
                "owner": order.owner.value,
            }
            for order in snapshot.orders
        ],
        "quotes": {
            pair: {
                "bid": quote.bid,
                "ask": quote.ask,
                "timestamp_utc": quote.timestamp_utc.isoformat(),
            }
            for pair, quote in snapshot.quotes.items()
        },
    }


def run_preflight(
    *,
    env_file: Path,
    approval_packet_path: Path,
    expected_packet_sha256: str,
    resident_status_path: Path,
    release_receipt_path: Path | None,
    inventory_state_path: Path | None,
    state_root: Path,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    now = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)
    sealer = load_sealer_module()
    packet = sealer.verify_approval_packet(
        read_json(approval_packet_path),
        expected_packet_sha256=expected_packet_sha256,
    )
    manifest = sealer.software_manifest()
    resident_status = read_json(resident_status_path)
    resident = sealer.verify_resident_shadow(resident_status)

    admission: dict[str, Any] | None = None
    risk: dict[str, Any] | None = None
    release_sha: str | None = None
    if release_receipt_path is not None and release_receipt_path.is_file():
        release = read_json(release_receipt_path)
        admission, risk = verify_release_receipt(
            release,
            approval_packet_sha256=expected_packet_sha256,
            software_manifest=manifest,
        )
        release_sha = str(release.get("release_receipt_sha256") or "")

    client = OandaReadOnlyClient(env_file=env_file)
    snapshot = client.snapshot(tuple(packet["initial_pairs"]))
    raw_account = client.get_json(f"/v3/accounts/{client.account_id}/summary")
    instruments = _instrument_map(client.account_instruments())
    # Network reads occur after the function starts; use a post-read clock so
    # fresh broker quotes are never misclassified as future timestamps.
    if now_utc is None:
        now = datetime.now(timezone.utc)
    candidate = risk if risk is not None else packet["candidate_limits"]
    limits = _limits(
        candidate,
        admission_sha=(admission or {}).get("admission_sha256"),
        risk_sha=(risk or {}).get("risk_contract_sha256"),
    )
    screen = screen_trade_readiness(
        snapshot=snapshot,
        raw_account=raw_account,
        limits=limits,
        software_ready=True,
        now_utc=now,
    )
    account = raw_account.get("account") if isinstance(raw_account.get("account"), Mapping) else raw_account
    nav = _number(account.get("NAV"), "NAV")
    margin_available = _number(account.get("marginAvailable"), "marginAvailable")
    current_mcp = _number(account.get("marginCloseoutPercent"), "marginCloseoutPercent")
    stress_mcp = estimate_account_stress_mcp(
        snapshot=snapshot,
        raw_account=raw_account,
        stress_pips=float(candidate["stress_pips"]),
    )
    if stress_mcp is None:
        raise PreflightBlocked("STRESS_MCP_UNAVAILABLE")
    system_positions = [
        position
        for position in snapshot.positions
        if broker_position_identity(position) is not None
    ]
    campaign_id = str((risk or {}).get("live_campaign_id") or "") or None
    inventory = _inventory_readback(
        inventory_state_path,
        campaign_id=campaign_id,
        now=now,
        nav_jpy=nav,
        initialize_if_missing=release_sha is not None and not system_positions,
    )
    previous_state = read_json(state_root / "state.json") if (state_root / "state.json").is_file() else {}
    previous_mode_raw = str(previous_state.get("mode") or RuntimeMode.SHADOW_ONLY.value)
    try:
        previous_mode = RuntimeMode(previous_mode_raw)
    except ValueError:
        previous_mode = RuntimeMode.SHADOW_ONLY

    shadow_output_path = (resident_status.get("last_shadow_result") or {}).get("shadow_output")
    shadow = read_json(Path(shadow_output_path)) if shadow_output_path and Path(shadow_output_path).is_file() else {}
    signals = [
        dict(signal)
        for signal in shadow.get("signals", [])
        if isinstance(signal, Mapping) and _signal_current(signal, now)
    ]
    signal_receipts: list[dict[str, Any]] = []
    mode = RuntimeMode.SHADOW_ONLY.value
    reason = "RISK_CONTRACT_UNACCEPTED"
    if system_positions and release_sha is None:
        mode = RuntimeMode.FREEZE_NEW.value
        reason = "UNACCEPTED_RISK_WITH_BOT_INVENTORY"
    elif release_sha is not None and not inventory["exists"]:
        reason = "LIVE_INVENTORY_STATE_MISSING"
    elif release_sha is not None and not signals:
        reason = "NO_FRESH_GO_SIGNAL"
    elif release_sha is not None:
        for signal in signals:
            pair = str(signal.get("pair") or "")
            quote_row = screen["quotes"].get(pair, {})
            sizing_input = signal_sizing_input(
                signal,
                quotes=snapshot.quotes,
                instruments=instruments,
                stress_pips=float(risk["stress_pips"]),
            )
            if screen["account"]["pending_entry_count"] > 0:
                blocked_mode = (
                    RuntimeMode.FREEZE_NEW.value
                    if system_positions
                    else RuntimeMode.SHADOW_ONLY.value
                )
                mode_receipt = {
                    "mode": blocked_mode,
                    "transition_reason": "PENDING_ENTRY_ORDER_PRESENT",
                    "requested_units": sizing_input.requested_units,
                    "calculated_units": 0,
                    "broker_minimum_units": sizing_input.broker_minimum_units,
                    "post_entry_current_mcp": None,
                    "post_entry_stress_mcp": None,
                    "safe_unit_capacity": 0,
                    "planned_loss_jpy": 0.0,
                    "post_entry_margin_available_jpy": margin_available,
                    "post_entry_max_currency_factor_nav_multiple": None,
                    "mutation_allowed": False,
                }
            elif quote_row.get("fresh") is not True or quote_row.get("spread_ok") is not True:
                blocked_mode = (
                    RuntimeMode.FREEZE_NEW.value
                    if system_positions
                    else RuntimeMode.SHADOW_ONLY.value
                )
                mode_receipt = {
                    "mode": blocked_mode,
                    "transition_reason": "QUOTE_FRESHNESS_OR_SPREAD_GATE_FAILED",
                    "requested_units": sizing_input.requested_units,
                    "calculated_units": 0,
                    "broker_minimum_units": sizing_input.broker_minimum_units,
                    "post_entry_current_mcp": None,
                    "post_entry_stress_mcp": None,
                    "safe_unit_capacity": 0,
                    "planned_loss_jpy": 0.0,
                    "post_entry_margin_available_jpy": margin_available,
                    "post_entry_max_currency_factor_nav_multiple": None,
                    "mutation_allowed": False,
                }
            else:
                mode_receipt = size_signal_for_runtime_mode(
                    previous_mode=previous_mode,
                    inventory_state=str(inventory["inventory_state"]),
                    has_bot_inventory=bool(system_positions),
                    nav_jpy=nav,
                    margin_available_jpy=margin_available,
                    current_mcp=current_mcp,
                    stress_baseline_mcp=stress_mcp,
                    campaign_drawdown_jpy=float(inventory["campaign_drawdown_jpy"]),
                    current_bot_position_count=len(system_positions),
                    cooldown_elapsed=bool(inventory["cooldown_elapsed"]),
                    factor_exposure_jpy=screen["currency_factor_jpy"],
                    limits=limits,
                    software_ready=True,
                    signal=sizing_input,
                )
            quote_age = float(quote_row.get("quote_age_seconds") or math.inf)
            account_age = max(
                0.0,
                (now - snapshot.fetched_at_utc.astimezone(timezone.utc)).total_seconds(),
            )
            sizing = build_sizing_receipt(
                mode_receipt=mode_receipt,
                signal_sha256=str(signal.get("signal_sha256") or ""),
                forward_admission_sha256=str(admission["admission_sha256"]),
                risk_contract_sha256=str(risk["risk_contract_sha256"]),
                software_version_sha256=str(manifest["software_version_sha256"]),
                account_snapshot_sha256=canonical_sha(_snapshot_binding(snapshot, raw_account)),
                quote_snapshot_sha256=canonical_sha({pair: quote_row}),
                campaign_drawdown_jpy=float(inventory["campaign_drawdown_jpy"]),
                account_snapshot_age_seconds=account_age,
                quote_age_seconds=quote_age,
                calculated_at_utc=now,
                spread_gate_passed=quote_row.get("fresh") is True and quote_row.get("spread_ok") is True,
            )
            signal_receipts.append(
                {
                    "signal_id": signal.get("signal_id"),
                    "signal_sha256": signal.get("signal_sha256"),
                    "pair": pair,
                    "strategy_id": signal.get("strategy_id"),
                    "mode_receipt": mode_receipt,
                    "sizing_receipt": sizing,
                }
            )
        if signal_receipts:
            selected = max(
                signal_receipts,
                key=lambda item: int(item["mode_receipt"].get("calculated_units") or 0),
            )
            mode = str(selected["mode_receipt"].get("mode") or RuntimeMode.SHADOW_ONLY.value)
            reason = str(selected["mode_receipt"].get("transition_reason") or "MODE_UNRESOLVED")

    snapshot_sha = canonical_sha(_snapshot_binding(snapshot, raw_account))
    event_body = {
        "contract": MODE_LEDGER_CONTRACT,
        "evaluated_at_utc": now.isoformat(),
        "software_commit": manifest["commit"],
        "software_version_sha256": manifest["software_version_sha256"],
        "approval_packet_sha256": expected_packet_sha256,
        "release_receipt_sha256": release_sha,
        "resident_source_commit": resident["source_commit"],
        "resident_source_bundle_sha256": resident["source_bundle_sha256"],
        "account_snapshot_sha256": snapshot_sha,
        "mode": mode,
        "transition_reason": reason,
        "signal_count": len(signals),
        "signal_receipts": signal_receipts,
        "account": screen["account"],
        "currency_factor_jpy": screen["currency_factor_jpy"],
        "screen_blockers": screen["blockers"],
        "inventory": inventory,
        "manual_tagless_policy": "NO_TOUCH",
        "existing_tp_sl_policy": "NO_TOUCH",
        "broker_http_methods_used": ["GET"],
        "broker_mutation_performed": False,
        "external_order_attempts": 0,
        "external_orders": 0,
    }
    event_id = f"qrplm:{canonical_sha(event_body)}"
    event = {**event_body, "event_id": event_id}
    ledger = state_root / "mode_ledger.jsonl"
    appended = append_event_once(ledger, event)
    state = {
        "contract": PREFLIGHT_CONTRACT,
        "updated_at_utc": now.isoformat(),
        "mode": mode,
        "transition_reason": reason,
        "last_event_id": event_id,
        "ledger_path": str(ledger),
        "ledger_appended": int(appended),
        "signal_count": len(signals),
        "calculated_units": max(
            (int(item["mode_receipt"].get("calculated_units") or 0) for item in signal_receipts),
            default=0,
        ),
        "safe_unit_capacity": max(
            (int(item["mode_receipt"].get("safe_unit_capacity") or 0) for item in signal_receipts),
            default=0,
        ),
        "needs_user_decision": release_sha is None,
        "waiting_external_state": release_sha is not None and mode != RuntimeMode.THROTTLED_LIVE.value,
        "promotion_ready": release_sha is not None and mode == RuntimeMode.THROTTLED_LIVE.value,
        "live_permission": False,
        "broker_mutation_allowed": False,
        "external_order_attempts": 0,
        "external_orders": 0,
    }
    state_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    temp = state_root / f".state.{os.getpid()}.tmp"
    temp.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    os.chmod(temp, 0o600)
    os.replace(temp, state_root / "state.json")
    return state


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--approval-packet", type=Path, required=True)
    parser.add_argument("--expected-packet-sha256", required=True)
    parser.add_argument("--resident-status", type=Path, required=True)
    parser.add_argument("--release-receipt", type=Path)
    parser.add_argument("--inventory-state", type=Path)
    parser.add_argument("--state-root", type=Path, default=DEFAULT_STATE_ROOT)
    args = parser.parse_args()
    result = run_preflight(
        env_file=args.env_file,
        approval_packet_path=args.approval_packet,
        expected_packet_sha256=args.expected_packet_sha256,
        resident_status_path=args.resident_status,
        release_receipt_path=args.release_receipt,
        inventory_state_path=args.inventory_state,
        state_root=args.state_root,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
