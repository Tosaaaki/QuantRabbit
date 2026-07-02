from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from quant_rabbit.instruments import instrument_pip_factor
from quant_rabbit.models import AccountSummary, BrokerPosition, BrokerSnapshot, OrderIntent, Owner, Quote, Side
from quant_rabbit.paths import DEFAULT_GUARDIAN_RECEIPT_OPERATOR_REVIEW, DEFAULT_OPERATOR_MANUAL_POSITIONS


OPERATOR_MANUAL = "OPERATOR_MANUAL"
OPERATOR_ALPHA_CANDIDATE = "OPERATOR_ALPHA_CANDIDATE"
OPERATOR_MANUAL_POSITION_PACKET = "OPERATOR_MANUAL_POSITION"
JPY_FRESH_ADD_BLOCK_CODE = "OPERATOR_MANUAL_JPY_EXPOSURE_ACTIVE"
SAME_THEME_ADD_BLOCK_CODE = "OPERATOR_MANUAL_SAME_THEME_ADD_BLOCKED"
OPERATOR_MANUAL_AUTH_METADATA_KEY = "operator_authorized_manual_overlap"

# Broker/account policy mirror of RiskEngine's OANDA Japan retail FX margin.
# The value represents 25:1 retail FX leverage. It is a broker/account constant,
# not market geometry; replace with broker instrument marginRate once the
# account-instruments adapter feeds runtime snapshots.
OANDA_JP_RETAIL_FX_MARGIN_RATE = 0.04


def load_operator_manual_confirmations(
    path: Path = DEFAULT_OPERATOR_MANUAL_POSITIONS,
    *,
    operator_review_path: Path = DEFAULT_GUARDIAN_RECEIPT_OPERATOR_REVIEW,
) -> list[dict[str, Any]]:
    rows = _load_confirmation_rows(path)
    if path == DEFAULT_OPERATOR_MANUAL_POSITIONS or operator_review_path != DEFAULT_GUARDIAN_RECEIPT_OPERATOR_REVIEW:
        rows.extend(_load_operator_review_confirmations(operator_review_path))
    return rows


def _load_confirmation_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    rows = payload.get("confirmations") if isinstance(payload, dict) else None
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, dict)]
    if isinstance(payload, dict):
        return [payload]
    return []


def _load_operator_review_confirmations(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    rows = payload.get("operator_position_reviews") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return []
    confirmations: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("operator_decision") or "") != "OPERATOR_CONFIRMED_MANUAL_OWNED":
            continue
        if str(row.get("owner") or "").upper() != OPERATOR_MANUAL:
            continue
        normalized = dict(row)
        normalized.setdefault("operator_confirmed", True)
        normalized.setdefault("owner_confirmed", True)
        normalized.setdefault("classification", OPERATOR_MANUAL)
        normalized.setdefault("alpha_classification", OPERATOR_ALPHA_CANDIDATE)
        confirmations.append(normalized)
    return confirmations


def classify_operator_manual_snapshot(
    snapshot: BrokerSnapshot,
    *,
    confirmations: list[dict[str, Any]] | None = None,
) -> BrokerSnapshot:
    rows = confirmations if confirmations is not None else load_operator_manual_confirmations()
    if not rows or not snapshot.positions:
        return snapshot
    confirmed_rows = _confirmed_operator_manual_rows(snapshot.positions, rows)
    if not confirmed_rows:
        return snapshot
    positions = tuple(_classify_position(position, confirmed_rows) for position in snapshot.positions)
    if positions == snapshot.positions:
        return snapshot
    return replace(snapshot, positions=positions)


def is_operator_manual_position(position: BrokerPosition | dict[str, Any]) -> bool:
    owner = _owner_value(position)
    if owner == Owner.OPERATOR_MANUAL.value:
        return True
    raw = _raw_value(position)
    packet = raw.get("operator_manual_position") if isinstance(raw, dict) else None
    if isinstance(packet, dict) and packet.get("packet_type") == OPERATOR_MANUAL_POSITION_PACKET:
        return True
    if isinstance(position, dict):
        top_packet = position.get("operator_manual_position")
        return (
            isinstance(top_packet, dict)
            and top_packet.get("packet_type") == OPERATOR_MANUAL_POSITION_PACKET
        )
    return False


def is_operator_managed_manual_owner(owner: Owner | str) -> bool:
    value = owner.value if isinstance(owner, Owner) else str(owner or "").lower()
    return value in {Owner.MANUAL.value, Owner.UNKNOWN.value, Owner.OPERATOR_MANUAL.value}


def operator_manual_position_packets(snapshot: BrokerSnapshot) -> list[dict[str, Any]]:
    positions = [position for position in snapshot.positions if is_operator_manual_position(position)]
    if not positions:
        return []
    grouped: dict[tuple[str, str], list[BrokerPosition]] = {}
    for position in positions:
        grouped.setdefault((position.pair, position.side.value), []).append(position)
    packets: list[dict[str, Any]] = []
    for (pair, side), group in sorted(grouped.items()):
        units = sum(abs(int(position.units)) for position in group)
        if units <= 0:
            continue
        avg_entry = sum(position.entry_price * abs(int(position.units)) for position in group) / units
        upl = sum(float(position.unrealized_pl_jpy or 0.0) for position in group)
        seed = _position_operator_packet(group[0])
        quote = snapshot.quotes.get(pair)
        pip_value = _pip_value_jpy(pair=pair, units=units, snapshot=snapshot)
        margin_used = _estimated_margin_jpy(pair=pair, units=units, entry_price=avg_entry, snapshot=snapshot)
        margin_pressure = _margin_pressure(snapshot, estimated_manual_margin_jpy=margin_used)
        thesis_state = major_figure_fade_thesis_state(
            side=Side(side),
            quote=quote,
            unrealized_pl_jpy=upl,
            major_figure=_maybe_float(seed.get("major_figure")),
            accepted_break=bool(seed.get("accepted_break_above_major_figure")),
            wick_above=bool(seed.get("wick_above_major_figure")),
        )
        if margin_pressure.get("state") == "EMERGENCY":
            thesis_state = {
                "state": "EMERGENCY",
                "exact_invalidation_evidence": (
                    "account margin pressure is emergency; this is a protection state, "
                    "not proof that the 162.00 thesis is invalidated"
                ),
            }
        packet = {
            "packet_type": OPERATOR_MANUAL_POSITION_PACKET,
            "classification": seed.get("classification") or OPERATOR_MANUAL,
            "alpha_classification": seed.get("alpha_classification") or OPERATOR_ALPHA_CANDIDATE,
            "pair": pair,
            "side": side,
            "units": units,
            "avg_entry": round(avg_entry, 5),
            "unrealized_pl_jpy": round(upl, 4),
            "pip_value_jpy_per_pip": round(pip_value, 4) if pip_value is not None else None,
            "estimated_margin_used_jpy": round(margin_used, 4) if margin_used is not None else None,
            "thesis": seed.get("thesis") or "operator manual thesis",
            "invalidation": seed.get("invalidation")
            or "accepted trade beyond operator invalidation; red P/L alone is not invalidation",
            "harvest_trigger": seed.get("harvest_trigger") or "operator-managed profit harvest trigger",
            "harvest_zone": seed.get("harvest_zone") or seed.get("harvest_trigger"),
            "major_figure": seed.get("major_figure"),
            "thesis_state": thesis_state["state"],
            "exact_invalidation_evidence": thesis_state["exact_invalidation_evidence"],
            "margin_pressure": margin_pressure,
            "management_rule": (
                "observe, TP-assist, and report only; no SL, loss-side close, "
                "or averaging unless operator explicitly asks"
            ),
            "blocks_fresh_jpy_adds": True,
            "trade_ids": [position.trade_id for position in group],
        }
        for key in (
            "operator_decision",
            "management_intent",
            "reason",
            "operator_confirmation_source",
            "no_live_side_effects",
            "system_pl_counted",
            "same_theme_auto_add_allowed",
            "loss_side_auto_close_allowed",
            "auto_sl_attach_allowed",
            "auto_tp_modify_allowed",
        ):
            if key in seed:
                packet[key] = seed[key]
        packets.append(packet)
    return packets


def operator_manual_position_packets_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    positions = tuple(_position_from_payload(item) for item in payload.get("positions", []) or [] if isinstance(item, dict))
    positions = tuple(position for position in positions if position is not None)
    quotes = {
        str(pair): quote
        for pair, quote in (
            _quote_from_payload(str(pair), item)
            for pair, item in (payload.get("quotes") or {}).items()
            if isinstance(item, dict)
        )
        if quote is not None
    }
    home_conversions = payload.get("home_conversions") if isinstance(payload.get("home_conversions"), dict) else {}
    return operator_manual_position_packets(
        BrokerSnapshot(
            fetched_at_utc=_datetime_from_payload(payload.get("fetched_at_utc")),
            positions=positions,
            quotes=quotes,
            account=_account_from_payload(payload.get("account")),
            home_conversions={
                str(key): float(value)
                for key, value in home_conversions.items()
                if _maybe_float(value) is not None
            },
        )
    )


def operator_manual_jpy_add_block_issue(intent: OrderIntent, snapshot: BrokerSnapshot) -> dict[str, str] | None:
    if not _is_jpy_add_guard_pair(intent.pair):
        return None
    if bool((intent.metadata or {}).get(OPERATOR_MANUAL_AUTH_METADATA_KEY)):
        return None
    packets = operator_manual_position_packets(snapshot)
    active = [
        packet
        for packet in packets
        if packet.get("pair") == "USD_JPY" and packet.get("side") == Side.SHORT.value
    ]
    if not active:
        return None
    exposure = active[0]
    return {
        "code": JPY_FRESH_ADD_BLOCK_CODE,
        "message": (
            f"operator_manual USD_JPY SHORT exposure is active "
            f"({exposure.get('units')}u, thesis_state={exposure.get('thesis_state')}); "
            f"fresh {intent.pair} bot adds require intent.metadata['{OPERATOR_MANUAL_AUTH_METADATA_KEY}']=true"
        ),
    }


def operator_manual_same_theme_add_block_issue(intent: OrderIntent, snapshot: BrokerSnapshot) -> dict[str, str] | None:
    if bool((intent.metadata or {}).get(OPERATOR_MANUAL_AUTH_METADATA_KEY)):
        return None
    for position in snapshot.positions:
        if position.pair != intent.pair or position.side != intent.side:
            continue
        if not is_operator_manual_position(position):
            continue
        packet = _position_operator_packet(position)
        if packet.get("same_theme_auto_add_allowed") is not False:
            continue
        return {
            "code": SAME_THEME_ADD_BLOCK_CODE,
            "message": (
                f"operator_manual {position.pair} {position.side.value} trade_id={position.trade_id} "
                "sets same_theme_auto_add_allowed=false; same-theme bot adds require "
                f"intent.metadata['{OPERATOR_MANUAL_AUTH_METADATA_KEY}']=true"
            ),
        }
    return None


def major_figure_fade_thesis_state(
    *,
    side: Side,
    quote: Quote | None = None,
    unrealized_pl_jpy: float = 0.0,
    major_figure: float | None = None,
    accepted_break: bool = False,
    wick_above: bool = False,
) -> dict[str, str]:
    if major_figure is None:
        return {
            "state": "ALIVE",
            "exact_invalidation_evidence": "no major-figure invalidation configured; red P/L alone is ignored",
        }
    if side == Side.SHORT:
        if accepted_break:
            return {
                "state": "INVALIDATED",
                "exact_invalidation_evidence": f"accepted trade above {major_figure:.2f} major figure",
            }
        ask = quote.ask if quote is not None else None
        if ask is not None and ask >= major_figure:
            return {
                "state": "WOUNDED",
                "exact_invalidation_evidence": (
                    f"ask {ask:.5f} is at/above {major_figure:.2f}, but no accepted break is confirmed"
                ),
            }
        if wick_above:
            return {
                "state": "WOUNDED",
                "exact_invalidation_evidence": (
                    f"wick/stop-run above {major_figure:.2f} observed; accepted break still absent"
                ),
            }
    return {
        "state": "ALIVE",
        "exact_invalidation_evidence": (
            f"no accepted trade beyond {major_figure:.2f}; "
            f"red P/L alone is not invalidation (unrealized P/L {unrealized_pl_jpy:.1f} JPY)"
        ),
    }


def _classify_position(
    position: BrokerPosition,
    confirmed_rows: dict[str, dict[str, Any]],
) -> BrokerPosition:
    row = confirmed_rows.get(position.trade_id)
    if row is None:
        return position
    packet = _packet_from_confirmation(row, position)
    raw = dict(position.raw or {})
    raw["operator_manual_position"] = packet
    return replace(position, owner=Owner.OPERATOR_MANUAL, raw=raw)


def _confirmed_operator_manual_rows(
    positions: tuple[BrokerPosition, ...],
    confirmations: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    confirmed: dict[str, dict[str, Any]] = {}
    for row in confirmations:
        for position in _positions_confirmed_by_row(row, positions):
            confirmed.setdefault(position.trade_id, row)
    return confirmed


def _confirmation_matches_position(
    row: dict[str, Any],
    position: BrokerPosition,
    all_positions: tuple[BrokerPosition, ...],
) -> bool:
    return any(
        candidate.trade_id == position.trade_id
        for candidate in _positions_confirmed_by_row(row, all_positions)
    )


def _positions_confirmed_by_row(
    row: dict[str, Any],
    all_positions: tuple[BrokerPosition, ...],
) -> tuple[BrokerPosition, ...]:
    if not bool(row.get("operator_confirmed") or row.get("owner_confirmed")):
        return ()
    pair = str(row.get("pair") or "").upper()
    side = str(row.get("side") or "").upper()
    trade_id = str(row.get("trade_id") or "").strip()
    eligible = tuple(
        candidate
        for candidate in all_positions
        if candidate.pair == pair
        and candidate.side.value == side
        and (not trade_id or candidate.trade_id == trade_id)
        and candidate.owner in {Owner.UNKNOWN, Owner.MANUAL, Owner.OPERATOR_MANUAL}
        and not _has_system_lane_or_gateway_receipt(candidate)
    )
    if not eligible:
        return ()
    expected_units = _maybe_int(row.get("units"))
    if expected_units is None:
        return eligible
    aggregate_units = sum(abs(int(candidate.units)) for candidate in eligible)
    if aggregate_units == expected_units:
        return eligible
    if aggregate_units < expected_units:
        return ()
    return _oldest_prefix_for_units(eligible, expected_units)


def _oldest_prefix_for_units(
    positions: tuple[BrokerPosition, ...],
    expected_units: int,
) -> tuple[BrokerPosition, ...]:
    selected: list[BrokerPosition] = []
    total = 0
    for position in sorted(positions, key=_position_open_sort_key):
        units = abs(int(position.units))
        if total + units > expected_units:
            return ()
        selected.append(position)
        total += units
        if total == expected_units:
            return tuple(selected)
    return ()


def _position_open_sort_key(position: BrokerPosition) -> tuple[datetime, str]:
    raw = position.raw if isinstance(position.raw, dict) else {}
    return _datetime_from_payload(raw.get("openTime")), position.trade_id


def _packet_from_confirmation(row: dict[str, Any], position: BrokerPosition) -> dict[str, Any]:
    packet = {
        "packet_type": OPERATOR_MANUAL_POSITION_PACKET,
        "classification": OPERATOR_MANUAL,
        "alpha_classification": row.get("alpha_classification") or OPERATOR_ALPHA_CANDIDATE,
        "trade_id": position.trade_id,
        "pair": position.pair,
        "side": position.side.value,
        "units": abs(int(position.units)),
        "avg_entry": position.entry_price,
        "unrealized_pl_jpy": position.unrealized_pl_jpy,
        "thesis": row.get("thesis") or "operator manual thesis",
        "invalidation": row.get("invalidation") or "",
        "harvest_trigger": row.get("harvest_trigger") or "",
        "harvest_zone": row.get("harvest_zone") or row.get("harvest_trigger") or "",
        "major_figure": _maybe_float(row.get("major_figure")),
        "accepted_break_above_major_figure": bool(row.get("accepted_break_above_major_figure")),
        "wick_above_major_figure": bool(row.get("wick_above_major_figure")),
        "operator_confirmed": True,
        "system_lane_absent": True,
    }
    for key in (
        "operator_decision",
        "management_intent",
        "reason",
        "operator_confirmation_source",
        "no_live_side_effects",
        "system_pl_counted",
        "same_theme_auto_add_allowed",
        "loss_side_auto_close_allowed",
        "auto_sl_attach_allowed",
        "auto_tp_modify_allowed",
    ):
        if key in row:
            packet[key] = row[key]
    return packet


def _has_system_lane_or_gateway_receipt(position: BrokerPosition) -> bool:
    if position.owner == Owner.TRADER:
        return True
    raw = position.raw if isinstance(position.raw, dict) else {}
    for key in ("clientExtensions", "tradeClientExtensions"):
        ext = raw.get(key) if isinstance(raw.get(key), dict) else {}
        tag = str(ext.get("tag") or "").strip().lower()
        if tag == Owner.TRADER.value:
            return True
        comment = str(ext.get("comment") or "").lower()
        cid = str(ext.get("id") or "").lower()
        if "lane" in comment or "gateway" in comment or "trader" in comment:
            return True
        if "lane" in cid or "gateway" in cid or "trader" in cid:
            return True
    return False


def _position_operator_packet(position: BrokerPosition) -> dict[str, Any]:
    raw = position.raw if isinstance(position.raw, dict) else {}
    packet = raw.get("operator_manual_position") if isinstance(raw.get("operator_manual_position"), dict) else {}
    return dict(packet)


def _raw_value(position: BrokerPosition | dict[str, Any]) -> dict[str, Any]:
    if isinstance(position, BrokerPosition):
        return position.raw if isinstance(position.raw, dict) else {}
    raw = position.get("raw") if isinstance(position, dict) else {}
    return raw if isinstance(raw, dict) else {}


def _owner_value(position: BrokerPosition | dict[str, Any]) -> str:
    if isinstance(position, BrokerPosition):
        return position.owner.value
    return str(position.get("owner") or "").lower() if isinstance(position, dict) else ""


def _position_from_payload(item: dict[str, Any]) -> BrokerPosition | None:
    try:
        raw = item.get("raw") if isinstance(item.get("raw"), dict) else {}
        if "operator_manual_position" not in raw and isinstance(item.get("operator_manual_position"), dict):
            raw = {**raw, "operator_manual_position": item["operator_manual_position"]}
        return BrokerPosition(
            trade_id=str(item["trade_id"]),
            pair=str(item["pair"]),
            side=Side.parse(str(item["side"])),
            units=int(item["units"]),
            entry_price=float(item["entry_price"]),
            unrealized_pl_jpy=float(item.get("unrealized_pl_jpy") or 0.0),
            take_profit=float(item["take_profit"]) if item.get("take_profit") is not None else None,
            stop_loss=float(item["stop_loss"]) if item.get("stop_loss") is not None else None,
            owner=_owner_from_payload(item.get("owner")),
            raw=raw,
        )
    except (KeyError, TypeError, ValueError):
        return None


def _owner_from_payload(value: object) -> Owner:
    try:
        return Owner(str(value or Owner.UNKNOWN.value))
    except ValueError:
        return Owner.UNKNOWN


def _quote_from_payload(pair: str, item: dict[str, Any]) -> tuple[str, Quote | None]:
    try:
        return pair, Quote(
            pair=pair,
            bid=float(item["bid"]),
            ask=float(item["ask"]),
            timestamp_utc=_datetime_from_payload(item.get("timestamp_utc")),
        )
    except (KeyError, TypeError, ValueError):
        return pair, None


def _account_from_payload(value: object) -> AccountSummary | None:
    if not isinstance(value, dict):
        return None
    try:
        return AccountSummary(
            nav_jpy=float(value.get("nav_jpy") or 0.0),
            balance_jpy=float(value.get("balance_jpy") or 0.0),
            unrealized_pl_jpy=float(value.get("unrealized_pl_jpy") or 0.0),
            margin_used_jpy=float(value.get("margin_used_jpy") or 0.0),
            margin_available_jpy=float(value.get("margin_available_jpy") or 0.0),
            pl_jpy=float(value.get("pl_jpy") or 0.0),
            financing_jpy=float(value.get("financing_jpy") or 0.0),
            last_transaction_id=value.get("last_transaction_id"),
            hedging_enabled=bool(value.get("hedging_enabled")),
            fetched_at_utc=_datetime_from_payload(value.get("fetched_at_utc")),
        )
    except (TypeError, ValueError):
        return None


def _datetime_from_payload(value: object) -> datetime:
    text = str(value or "").strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text) if text else datetime.now(timezone.utc)
    except ValueError:
        normalized = _trim_iso_fraction_to_microseconds(text)
        try:
            parsed = datetime.fromisoformat(normalized) if normalized else datetime.now(timezone.utc)
        except ValueError:
            parsed = datetime.now(timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _trim_iso_fraction_to_microseconds(text: str) -> str:
    dot = text.find(".")
    if dot < 0:
        return text
    tz_start = len(text)
    for index in range(dot + 1, len(text)):
        if text[index] in "+-Z":
            tz_start = index
            break
    fraction = text[dot + 1 : tz_start]
    if len(fraction) <= 6 or not fraction.isdigit():
        return text
    return f"{text[: dot + 1]}{fraction[:6]}{text[tz_start:]}"


def _pip_value_jpy(*, pair: str, units: int, snapshot: BrokerSnapshot) -> float | None:
    quote_to_jpy = _quote_to_jpy(pair, snapshot)
    if quote_to_jpy is None:
        return None
    return (abs(int(units)) / instrument_pip_factor(pair)) * quote_to_jpy


def _estimated_margin_jpy(
    *,
    pair: str,
    units: int,
    entry_price: float,
    snapshot: BrokerSnapshot,
) -> float | None:
    quote_to_jpy = _quote_to_jpy(pair, snapshot)
    if quote_to_jpy is None:
        return None
    return max(0.0, abs(int(units)) * abs(float(entry_price)) * quote_to_jpy * OANDA_JP_RETAIL_FX_MARGIN_RATE)


def _quote_to_jpy(pair: str, snapshot: BrokerSnapshot) -> float | None:
    quote_currency = pair.split("_")[-1].upper() if "_" in pair else ""
    if quote_currency == "JPY":
        return 1.0
    home = snapshot.home_conversions.get(quote_currency)
    if home and home > 0:
        return float(home)
    conversion = snapshot.quotes.get(f"{quote_currency}_JPY")
    if conversion is not None:
        return max(float(conversion.bid), float(conversion.ask))
    return None


def _margin_pressure(
    snapshot: BrokerSnapshot,
    *,
    estimated_manual_margin_jpy: float | None,
) -> dict[str, Any]:
    account = snapshot.account
    if account is None:
        return {
            "state": "UNKNOWN",
            "estimated_manual_margin_jpy": estimated_manual_margin_jpy,
            "reason": "account margin snapshot missing",
        }
    utilization = (
        (account.margin_used_jpy / account.nav_jpy) * 100.0
        if account.nav_jpy
        else None
    )
    if account.margin_available_jpy <= 0:
        state = "EMERGENCY"
    elif utilization is None:
        state = "UNKNOWN"
    elif utilization >= 90.0:
        state = "PRESSURED"
    else:
        state = "OK"
    return {
        "state": state,
        "account_margin_used_jpy": round(account.margin_used_jpy, 4),
        "account_margin_available_jpy": round(account.margin_available_jpy, 4),
        "account_nav_jpy": round(account.nav_jpy, 4),
        "account_margin_utilization_pct": round(utilization, 4) if utilization is not None else None,
        "estimated_manual_margin_jpy": round(estimated_manual_margin_jpy, 4)
        if estimated_manual_margin_jpy is not None
        else None,
    }


def _is_jpy_add_guard_pair(pair: str) -> bool:
    pair = str(pair or "").upper()
    return pair == "USD_JPY" or pair.endswith("_JPY")


def _maybe_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _maybe_int(value: object) -> int | None:
    try:
        if value is None:
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None
