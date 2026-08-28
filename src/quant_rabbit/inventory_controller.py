from __future__ import annotations

import fcntl
import json
import math
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from quant_rabbit.instruments import instrument_pip_factor
from quant_rabbit.models import BrokerPosition, Owner, Side


INVENTORY_CONTRACT = "QR_FAST_BOT_INVENTORY_V1"
OWNER_TAG = "fast_bot"
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,31}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class InventoryState(str, Enum):
    RUNNING = "RUNNING"
    FREEZE_NEW = "FREEZE_NEW"
    DRAINING = "DRAINING"
    FLAT = "FLAT"
    STOPPED = "STOPPED"


@dataclass(frozen=True, slots=True)
class LotIdentity:
    campaign_id: str
    strategy_id: str
    lot_id: str
    owner_tag: str = OWNER_TAG

    def __post_init__(self) -> None:
        for label, value in (
            ("campaign_id", self.campaign_id),
            ("strategy_id", self.strategy_id),
            ("lot_id", self.lot_id),
        ):
            if _IDENTIFIER_RE.fullmatch(value) is None:
                raise ValueError(f"{label} is not a bounded canonical identifier")
        if self.owner_tag != OWNER_TAG:
            raise ValueError("owner_tag must remain fast_bot")

    def to_metadata(self) -> dict[str, str]:
        return asdict(self)

    def broker_client_id(self) -> str:
        value = (
            f"qr-fb|c={self.campaign_id}|s={self.strategy_id}|l={self.lot_id}"
        )
        if len(value) > 128:
            raise ValueError("bot ownership identity exceeds broker client id limit")
        return value

    @classmethod
    def from_metadata(cls, value: Mapping[str, Any]) -> "LotIdentity":
        return cls(
            campaign_id=str(value.get("campaign_id") or ""),
            strategy_id=str(value.get("strategy_id") or ""),
            lot_id=str(value.get("lot_id") or ""),
            owner_tag=str(value.get("owner_tag") or ""),
        )

    @classmethod
    def from_broker_client_id(cls, value: object) -> "LotIdentity":
        text = str(value or "")
        if not text.startswith("qr-fb|"):
            raise ValueError("broker client id is not a fast-bot identity")
        tokens = {
            name: item
            for token in text.split("|")[1:]
            if "=" in token
            for name, item in [token.split("=", 1)]
        }
        return cls(
            campaign_id=str(tokens.get("c") or ""),
            strategy_id=str(tokens.get("s") or ""),
            lot_id=str(tokens.get("l") or ""),
        )


@dataclass(slots=True)
class InventoryLot:
    identity: LotIdentity
    pair: str
    side: str
    original_units: int
    remaining_units: int
    entry_price: float
    opened_at_utc: str
    last_mark_at_utc: str
    current_unrealized_pips: float = 0.0
    mfe_pips: float = 0.0
    mae_pips: float = 0.0
    peak_unrealized_pips: float = 0.0
    giveback_pips: float = 0.0
    realized_after_cost_jpy: float = 0.0
    last_progress_at_utc: str = ""
    estimated_margin_relief_jpy: float = 0.0
    estimated_close_loss_and_cost_jpy: float = 0.0
    currency_factor: str = ""
    reduction_started: bool = False

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["identity"] = self.identity.to_metadata()
        return payload

    @classmethod
    def from_payload(cls, value: Mapping[str, Any]) -> "InventoryLot":
        return cls(
            identity=LotIdentity.from_metadata(_mapping(value.get("identity"))),
            pair=str(value.get("pair") or ""),
            side=str(value.get("side") or ""),
            original_units=_positive_int(value.get("original_units"), "original_units"),
            remaining_units=_nonnegative_int(value.get("remaining_units"), "remaining_units"),
            entry_price=_positive_float(value.get("entry_price"), "entry_price"),
            opened_at_utc=_utc_text(value.get("opened_at_utc")),
            last_mark_at_utc=_utc_text(value.get("last_mark_at_utc")),
            current_unrealized_pips=_finite_float(value.get("current_unrealized_pips")),
            mfe_pips=_finite_float(value.get("mfe_pips")),
            mae_pips=_finite_float(value.get("mae_pips")),
            peak_unrealized_pips=_finite_float(value.get("peak_unrealized_pips")),
            giveback_pips=_finite_float(value.get("giveback_pips")),
            realized_after_cost_jpy=_finite_float(value.get("realized_after_cost_jpy")),
            last_progress_at_utc=_utc_text(value.get("last_progress_at_utc")),
            estimated_margin_relief_jpy=_finite_float(
                value.get("estimated_margin_relief_jpy", 0.0)
            ),
            estimated_close_loss_and_cost_jpy=_finite_float(
                value.get("estimated_close_loss_and_cost_jpy", 0.0)
            ),
            currency_factor=str(value.get("currency_factor") or ""),
            reduction_started=bool(value.get("reduction_started") or False),
        )


@dataclass(frozen=True, slots=True)
class UnwindAction:
    action: str
    lot_id: str | None = None
    units: int | None = None
    pending_order_id: str | None = None
    reason: str = ""


@dataclass(slots=True)
class InventoryController:
    state_path: Path
    campaign_id: str
    state: InventoryState = InventoryState.RUNNING
    revision: int = 0
    cooldown_until_utc: str | None = None
    stop_reason: str | None = None
    pending_entry_ids: list[str] = field(default_factory=list)
    lots: dict[str, InventoryLot] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    applied_receipt_ids: list[str] = field(default_factory=list)
    applied_event_dedupe_keys: list[str] = field(default_factory=list)
    supervision_regime: str | None = None
    allowed_strategy_ids: list[str] = field(default_factory=list)
    supervision_risk_budget_cap_jpy: float = 0.0
    supervision_max_positions_cap: int = 0
    supervision_expires_at_utc: str | None = None
    cycle_start_nav_jpy: float | None = None
    cycle_peak_nav_jpy: float | None = None
    cycle_count: int = 0
    cycle_retained_return: float | None = None
    cycle_giveback_jpy: float = 0.0
    cycle_execution_cost_jpy: float = 0.0
    profit_lock_triggered: bool = False
    profit_floor_breached: bool = False
    profit_lock_reduction_fraction: float = 0.5
    _persisted_revision: int = field(default=0, repr=False)

    @classmethod
    def open(
        cls,
        state_path: Path,
        *,
        campaign_id: str,
        now_utc: datetime,
    ) -> "InventoryController":
        LotIdentity(campaign_id=campaign_id, strategy_id="bootstrap", lot_id="bootstrap")
        if state_path.exists():
            controller = cls._load(state_path)
            if controller.campaign_id != campaign_id:
                if controller.state is not InventoryState.STOPPED:
                    raise RuntimeError("an unfinished campaign already owns inventory state")
                if not controller.cooldown_elapsed(now_utc):
                    raise RuntimeError("durable post-stop cooldown is still active")
                controller = cls(
                    state_path=state_path,
                    campaign_id=campaign_id,
                    revision=controller.revision,
                    events=list(controller.events),
                    cycle_count=controller.cycle_count,
                    cycle_retained_return=controller.cycle_retained_return,
                    cycle_giveback_jpy=controller.cycle_giveback_jpy,
                    cycle_execution_cost_jpy=controller.cycle_execution_cost_jpy,
                    _persisted_revision=controller.revision,
                )
                controller._record("CAMPAIGN_STARTED", now_utc)
                controller._persist()
            return controller
        controller = cls(state_path=state_path, campaign_id=campaign_id)
        controller._record("CAMPAIGN_STARTED", now_utc)
        controller._persist()
        return controller

    @classmethod
    def _load(cls, state_path: Path) -> "InventoryController":
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        if payload.get("contract") != INVENTORY_CONTRACT:
            raise RuntimeError("inventory state contract is invalid")
        lots = {
            str(item["identity"]["lot_id"]): InventoryLot.from_payload(item)
            for item in payload.get("lots", [])
            if isinstance(item, Mapping) and isinstance(item.get("identity"), Mapping)
        }
        controller = cls(
            state_path=state_path,
            campaign_id=str(payload.get("campaign_id") or ""),
            state=InventoryState(str(payload.get("state") or "")),
            revision=_nonnegative_int(payload.get("revision"), "revision"),
            cooldown_until_utc=(
                _utc_text(payload.get("cooldown_until_utc"))
                if payload.get("cooldown_until_utc")
                else None
            ),
            stop_reason=str(payload.get("stop_reason") or "") or None,
            pending_entry_ids=[str(item) for item in payload.get("pending_entry_ids", [])],
            lots=lots,
            events=[dict(item) for item in payload.get("events", []) if isinstance(item, Mapping)],
            applied_receipt_ids=[str(item) for item in payload.get("applied_receipt_ids", [])],
            applied_event_dedupe_keys=[
                str(item) for item in payload.get("applied_event_dedupe_keys", [])
            ],
            supervision_regime=str(payload.get("supervision_regime") or "") or None,
            allowed_strategy_ids=[
                str(item) for item in payload.get("allowed_strategy_ids", [])
            ],
            supervision_risk_budget_cap_jpy=max(
                0.0, _finite_float(payload.get("supervision_risk_budget_cap_jpy", 0.0))
            ),
            supervision_max_positions_cap=_nonnegative_int(
                payload.get("supervision_max_positions_cap", 0),
                "supervision_max_positions_cap",
            ),
            supervision_expires_at_utc=(
                _utc_text(payload.get("supervision_expires_at_utc"))
                if payload.get("supervision_expires_at_utc")
                else None
            ),
            cycle_start_nav_jpy=(
                _positive_float(payload.get("cycle_start_nav_jpy"), "cycle_start_nav_jpy")
                if payload.get("cycle_start_nav_jpy") is not None
                else None
            ),
            cycle_peak_nav_jpy=(
                _positive_float(payload.get("cycle_peak_nav_jpy"), "cycle_peak_nav_jpy")
                if payload.get("cycle_peak_nav_jpy") is not None
                else None
            ),
            cycle_count=_nonnegative_int(payload.get("cycle_count", 0), "cycle_count"),
            cycle_retained_return=(
                _finite_float(payload.get("cycle_retained_return"))
                if payload.get("cycle_retained_return") is not None
                else None
            ),
            cycle_giveback_jpy=max(0.0, _finite_float(payload.get("cycle_giveback_jpy", 0.0))),
            cycle_execution_cost_jpy=max(
                0.0, _finite_float(payload.get("cycle_execution_cost_jpy", 0.0))
            ),
            profit_lock_triggered=payload.get("profit_lock_triggered") is True,
            profit_floor_breached=payload.get("profit_floor_breached") is True,
            profit_lock_reduction_fraction=min(
                1.0,
                max(0.5, _finite_float(payload.get("profit_lock_reduction_fraction", 0.5))),
            ),
            _persisted_revision=_nonnegative_int(payload.get("revision"), "revision"),
        )
        if any(lot.identity.campaign_id != controller.campaign_id for lot in lots.values()):
            raise RuntimeError("inventory lot is bound to another campaign")
        if controller.state in {InventoryState.FLAT, InventoryState.STOPPED} and any(
            lot.remaining_units > 0 for lot in lots.values()
        ):
            raise RuntimeError("flat/stopped inventory state contains remaining units")
        return controller

    def can_enter(self, now_utc: datetime) -> bool:
        return self.state is InventoryState.RUNNING and self.cooldown_elapsed(now_utc)

    def cooldown_elapsed(self, now_utc: datetime) -> bool:
        now = _aware_utc(now_utc)
        return self.cooldown_until_utc is None or now >= _parse_utc(self.cooldown_until_utc)

    def configure_profit_lock(self, *, cycle_start_nav_jpy: float, now_utc: datetime) -> None:
        """Bind one campaign cycle to its immutable NAV baseline.

        The +5% floor is relative to this baseline, never relative to peak NAV.
        Rebinding an active cycle is forbidden.
        """

        nav = _positive_float(cycle_start_nav_jpy, "cycle_start_nav_jpy")
        if self.state is not InventoryState.RUNNING:
            raise RuntimeError("profit-lock cycle can start only while RUNNING")
        if self.cycle_start_nav_jpy is not None:
            if not math.isclose(self.cycle_start_nav_jpy, nav, rel_tol=0.0, abs_tol=1e-6):
                raise RuntimeError("cycle_start_NAV is immutable for the active campaign")
            return
        self.cycle_start_nav_jpy = nav
        self.cycle_peak_nav_jpy = nav
        self.cycle_count += 1
        self.cycle_retained_return = None
        self.cycle_giveback_jpy = 0.0
        self.cycle_execution_cost_jpy = 0.0
        self.profit_lock_triggered = False
        self.profit_floor_breached = False
        self.profit_lock_reduction_fraction = 0.5
        self._record(
            "PROFIT_LOCK_CYCLE_STARTED",
            now_utc,
            cycle_count=self.cycle_count,
            cycle_start_nav_jpy=nav,
            target_nav_jpy=round(nav * 1.10, 6),
            retained_floor_nav_jpy=round(nav * 1.05, 6),
        )
        self._persist()

    def evaluate_profit_lock(
        self,
        *,
        current_nav_jpy: float,
        now_utc: datetime,
        hard_limit_reason: str | None = None,
    ) -> str:
        """Advance the deterministic +10% freeze / +5% retained-floor policy."""

        if self.cycle_start_nav_jpy is None:
            raise RuntimeError("cycle_start_NAV is not configured")
        nav = _positive_float(current_nav_jpy, "current_nav_jpy")
        start = self.cycle_start_nav_jpy
        peak = max(float(self.cycle_peak_nav_jpy or start), nav)
        self.cycle_peak_nav_jpy = peak
        self.cycle_giveback_jpy = round(max(0.0, peak - nav), 6)
        self.cycle_retained_return = round((nav / start) - 1.0, 8)
        target = round(start * 1.10, 6)
        retained_floor = round(start * 1.05, 6)
        action = "NO_CHANGE"
        if hard_limit_reason and self.state is InventoryState.RUNNING:
            self.state = InventoryState.FREEZE_NEW
            self.stop_reason = f"HARD_LIMIT:{str(hard_limit_reason).strip()}"
            self._record("FREEZE_NEW", now_utc, reason=self.stop_reason)
            self.state = (
                InventoryState.DRAINING
                if self.pending_entry_ids or any(lot.remaining_units > 0 for lot in self.lots.values())
                else InventoryState.FLAT
            )
            self._record(self.state.value, now_utc, reason=self.stop_reason)
            action = f"HARD_LIMIT_{self.state.value}"
        if self.state is InventoryState.RUNNING and nav >= target:
            self.state = InventoryState.FREEZE_NEW
            self.stop_reason = "CYCLE_START_NAV_PLUS_10_PERCENT"
            self.profit_lock_triggered = True
            self._record(
                "PROFIT_LOCK_TARGET_REACHED",
                now_utc,
                cycle_count=self.cycle_count,
                cycle_start_nav_jpy=start,
                current_nav_jpy=nav,
                retained_floor_nav_jpy=round(retained_floor, 6),
            )
            self.state = (
                InventoryState.DRAINING
                if self.pending_entry_ids or any(lot.remaining_units > 0 for lot in self.lots.values())
                else InventoryState.FLAT
            )
            self._record(self.state.value, now_utc, reason=self.stop_reason)
            action = self.state.value
        if self.state is InventoryState.DRAINING and self.profit_lock_triggered:
            retained = self.cycle_retained_return
            if nav <= retained_floor:
                self.profit_floor_breached = True
                self.profit_lock_reduction_fraction = 1.0
                action = "FORCE_FLAT_AT_CYCLE_START_PLUS_5_PERCENT"
            elif retained <= 0.06:
                self.profit_lock_reduction_fraction = 1.0
                action = "DRAIN_100_PERCENT_NEAR_RETAINED_FLOOR"
            elif retained <= 0.075:
                self.profit_lock_reduction_fraction = 0.75
                action = "DRAIN_75_PERCENT_APPROACHING_RETAINED_FLOOR"
            else:
                self.profit_lock_reduction_fraction = 0.5
                action = "DRAIN_50_PERCENT_AFTER_TARGET"
            self._record(
                "PROFIT_LOCK_DRAIN_EVALUATED",
                now_utc,
                cycle_count=self.cycle_count,
                current_nav_jpy=nav,
                retained_return=round(retained, 8),
                giveback_jpy=self.cycle_giveback_jpy,
                reduction_fraction=self.profit_lock_reduction_fraction,
                action=action,
            )
        self._persist()
        return action

    def register_pending_entry(self, order_id: str, *, now_utc: datetime) -> None:
        if not self.can_enter(now_utc):
            raise RuntimeError("new entries are frozen")
        normalized = str(order_id or "").strip()
        if not normalized or normalized in self.pending_entry_ids:
            raise ValueError("pending entry id must be non-empty and unique")
        self.pending_entry_ids.append(normalized)
        self._record("PENDING_ENTRY_REGISTERED", now_utc, order_id=normalized)
        self._persist()

    def register_fill(
        self,
        *,
        identity: LotIdentity,
        pair: str,
        side: Side,
        units: int,
        entry_price: float,
        now_utc: datetime,
    ) -> None:
        if not self.can_enter(now_utc):
            raise RuntimeError("new fills are blocked outside RUNNING")
        if identity.campaign_id != self.campaign_id:
            raise ValueError("lot identity is bound to another campaign")
        if identity.lot_id in self.lots:
            raise ValueError("duplicate lot_id")
        if any(
            item.identity.strategy_id == identity.strategy_id
            and item.pair == str(pair).upper()
            and item.reduction_started
            for item in self.lots.values()
        ):
            raise RuntimeError("reduced strategy/pair inventory cannot be re-added")
        units = _positive_int(units, "units")
        opened = _format_utc(now_utc)
        self.lots[identity.lot_id] = InventoryLot(
            identity=identity,
            pair=str(pair).upper(),
            side=side.value,
            original_units=units,
            remaining_units=units,
            entry_price=_positive_float(entry_price, "entry_price"),
            opened_at_utc=opened,
            last_mark_at_utc=opened,
            last_progress_at_utc=opened,
        )
        self._record("LOT_FILLED", now_utc, lot_id=identity.lot_id, units=units)
        self._persist()

    def update_unwind_economics(
        self,
        lot_id: str,
        *,
        estimated_margin_relief_jpy: float,
        estimated_close_loss_and_cost_jpy: float,
        currency_factor: str,
        now_utc: datetime,
    ) -> None:
        lot = self._lot(lot_id)
        lot.estimated_margin_relief_jpy = max(
            0.0, _finite_float(estimated_margin_relief_jpy)
        )
        lot.estimated_close_loss_and_cost_jpy = max(
            0.0, _finite_float(estimated_close_loss_and_cost_jpy)
        )
        lot.currency_factor = str(currency_factor or "").upper()
        self._record("LOT_UNWIND_ECONOMICS_UPDATED", now_utc, lot_id=lot_id)
        self._persist()

    def apply_supervision_receipt(
        self,
        *,
        event: Mapping[str, Any],
        receipt: Mapping[str, Any] | None,
        now_utc: datetime,
    ) -> str:
        """Apply one bounded LLM supervisory receipt, never an order instruction."""

        if receipt is None:
            if self.state is InventoryState.RUNNING:
                self.freeze_new(reason="LLM_NO_RESPONSE", now_utc=now_utc)
            return "FREEZE_NEW_NO_RESPONSE"
        receipt_id = str(receipt.get("receipt_id") or "")
        dedupe_key = str(receipt.get("dedupe_key") or "")
        event_id = str(event.get("event_id") or "")
        if (
            not receipt_id
            or receipt_id in self.applied_receipt_ids
            or dedupe_key in self.applied_event_dedupe_keys
        ):
            return "DUPLICATE_IGNORED"
        try:
            generated_at = _parse_utc(receipt.get("generated_at_utc"))
            expires_at = _parse_utc(receipt.get("expires_at_utc"))
        except (TypeError, ValueError):
            if self.state is InventoryState.RUNNING:
                self.freeze_new(reason="LLM_RECEIPT_INVALID_OR_EXPIRED", now_utc=now_utc)
            return "FREEZE_NEW_INVALID_RECEIPT"
        now = _aware_utc(now_utc)
        binding_valid = (
            str(receipt.get("event_id") or "") == event_id
            and dedupe_key == str(event.get("dedupe_key") or "")
            and generated_at <= now <= expires_at
            and _SHA256_RE.fullmatch(
                str(receipt.get("feature_snapshot_sha256") or "")
            )
            is not None
        )
        if not binding_valid:
            if self.state is InventoryState.RUNNING:
                self.freeze_new(reason="LLM_RECEIPT_INVALID_OR_EXPIRED", now_utc=now_utc)
            return "FREEZE_NEW_INVALID_RECEIPT"
        decision = str(receipt.get("decision") or "").upper()
        if decision not in {"ALLOW", "FREEZE_NEW", "UNWIND"}:
            if self.state is InventoryState.RUNNING:
                self.freeze_new(reason="LLM_RECEIPT_INVALID_DECISION", now_utc=now_utc)
            return "FREEZE_NEW_INVALID_DECISION"
        regime = str(receipt.get("regime") or "").upper()
        raw_strategies = receipt.get("allowed_strategy_ids")
        try:
            risk_budget_cap = _finite_float(receipt.get("risk_budget_cap_jpy", 0.0))
            max_positions_cap = _nonnegative_int(
                receipt.get("max_positions_cap", 0), "max_positions_cap"
            )
        except (TypeError, ValueError):
            if self.state is InventoryState.RUNNING:
                self.freeze_new(reason="LLM_RECEIPT_INVALID_STRUCTURE", now_utc=now_utc)
            return "FREEZE_NEW_INVALID_STRUCTURE"
        structured_valid = bool(
            _IDENTIFIER_RE.fullmatch(regime)
            and isinstance(raw_strategies, list)
            and all(
                isinstance(item, str) and _IDENTIFIER_RE.fullmatch(item)
                for item in raw_strategies
            )
            and len(set(raw_strategies)) == len(raw_strategies)
            and (
                (
                    decision == "ALLOW"
                    and raw_strategies
                    and risk_budget_cap > 0.0
                    and max_positions_cap > 0
                )
                or (
                    decision in {"FREEZE_NEW", "UNWIND"}
                    and not raw_strategies
                    and risk_budget_cap == 0.0
                    and max_positions_cap == 0
                )
            )
        )
        if not structured_valid:
            if self.state is InventoryState.RUNNING:
                self.freeze_new(reason="LLM_RECEIPT_INVALID_STRUCTURE", now_utc=now_utc)
            return "FREEZE_NEW_INVALID_STRUCTURE"
        self.applied_receipt_ids.append(receipt_id)
        self.applied_event_dedupe_keys.append(dedupe_key)
        self.applied_receipt_ids = self.applied_receipt_ids[-256:]
        self.applied_event_dedupe_keys = self.applied_event_dedupe_keys[-256:]
        self._record("LLM_SUPERVISION_APPLIED", now_utc, receipt_id=receipt_id, decision=decision)
        self.supervision_regime = regime
        self.allowed_strategy_ids = list(raw_strategies)
        self.supervision_risk_budget_cap_jpy = risk_budget_cap
        self.supervision_max_positions_cap = max_positions_cap
        self.supervision_expires_at_utc = _format_utc(expires_at)
        if decision in {"FREEZE_NEW", "UNWIND"} and self.state is InventoryState.RUNNING:
            self.state = InventoryState.FREEZE_NEW
            self.stop_reason = f"LLM_{decision}"
            self._record("FREEZE_NEW", now_utc, reason=self.stop_reason)
        if decision == "UNWIND" and self.state is InventoryState.FREEZE_NEW:
            self.state = (
                InventoryState.DRAINING
                if self.pending_entry_ids
                or any(lot.remaining_units > 0 for lot in self.lots.values())
                else InventoryState.FLAT
            )
            self._record(self.state.value, now_utc)
        self._persist()
        return f"APPLIED_{decision}"

    def mark_lot(self, lot_id: str, *, executable_price: float, now_utc: datetime) -> None:
        lot = self._lot(lot_id)
        price = _positive_float(executable_price, "executable_price")
        direction = 1.0 if lot.side == Side.LONG.value else -1.0
        unrealized = (price - lot.entry_price) * instrument_pip_factor(lot.pair) * direction
        prior_peak = lot.peak_unrealized_pips
        lot.current_unrealized_pips = round(unrealized, 6)
        lot.mfe_pips = round(max(lot.mfe_pips, unrealized), 6)
        lot.mae_pips = round(min(lot.mae_pips, unrealized), 6)
        lot.peak_unrealized_pips = round(max(lot.peak_unrealized_pips, unrealized), 6)
        lot.giveback_pips = round(max(0.0, lot.peak_unrealized_pips - unrealized), 6)
        lot.last_mark_at_utc = _format_utc(now_utc)
        if lot.peak_unrealized_pips > prior_peak:
            lot.last_progress_at_utc = lot.last_mark_at_utc
        self._record("LOT_MARKED", now_utc, lot_id=lot_id)
        self._persist()

    def freeze_new(self, *, reason: str, now_utc: datetime) -> None:
        if self.state is not InventoryState.RUNNING:
            raise RuntimeError("FREEZE_NEW requires RUNNING")
        self.state = InventoryState.FREEZE_NEW
        self.stop_reason = str(reason or "").strip() or "UNSPECIFIED_STOP"
        self._record("FREEZE_NEW", now_utc, reason=self.stop_reason)
        self._persist()

    def begin_draining(self, *, now_utc: datetime) -> None:
        if self.state is not InventoryState.FREEZE_NEW:
            raise RuntimeError("DRAINING requires FREEZE_NEW")
        self.state = (
            InventoryState.DRAINING
            if self.pending_entry_ids or any(lot.remaining_units > 0 for lot in self.lots.values())
            else InventoryState.FLAT
        )
        self._record(self.state.value, now_utc)
        self._persist()

    def unwind_actions(
        self,
        *,
        now_utc: datetime,
        terminal_deadline_utc: datetime,
    ) -> tuple[UnwindAction, ...]:
        if self.state is not InventoryState.DRAINING:
            return ()
        actions: list[UnwindAction] = [
            UnwindAction(
                action="CANCEL_PENDING_ENTRY",
                pending_order_id=order_id,
                reason="DRAINING_CANCEL_SYSTEM_PENDING",
            )
            for order_id in self.pending_entry_ids
        ]
        hard_terminal = _aware_utc(now_utc) >= _aware_utc(terminal_deadline_utc)
        for lot in sorted(
            (item for item in self.lots.values() if item.remaining_units > 0),
            key=_unwind_priority,
        ):
            force_profit_floor = self.profit_floor_breached
            fraction = self.profit_lock_reduction_fraction if self.profit_lock_triggered else 0.5
            units = (
                lot.remaining_units
                if hard_terminal or force_profit_floor
                else max(1, math.ceil(lot.remaining_units * fraction))
            )
            reason = (
                "HARD_TERMINAL_ALL_REMAINING"
                if hard_terminal
                else "PROFIT_FLOOR_FORCE_FLAT_ALL_REMAINING"
                if force_profit_floor
                else "PARTIAL_SCALE_OUT_ALL_OWNED_LOTS"
            )
            actions.append(
                UnwindAction(
                    action="REDUCE_BOT_LOT",
                    lot_id=lot.identity.lot_id,
                    units=units,
                    reason=reason,
                )
            )
        return tuple(actions)

    def record_pending_cancel(self, order_id: str, *, now_utc: datetime) -> None:
        if self.state is not InventoryState.DRAINING:
            raise RuntimeError("pending cancellation is only valid while DRAINING")
        self.pending_entry_ids.remove(order_id)
        self._record("PENDING_ENTRY_CANCELLED", now_utc, order_id=order_id)
        self._advance_flat_if_empty(now_utc)
        self._persist()

    def record_unwind_fill(
        self,
        lot_id: str,
        *,
        units: int,
        realized_after_cost_jpy: float,
        execution_cost_jpy: float = 0.0,
        now_utc: datetime,
    ) -> None:
        if self.state is not InventoryState.DRAINING:
            raise RuntimeError("unwind fills are only valid while DRAINING")
        lot = self._lot(lot_id)
        units = _positive_int(units, "units")
        if units > lot.remaining_units:
            raise ValueError("unwind fill exceeds bot-owned remaining units")
        lot.remaining_units -= units
        lot.reduction_started = True
        lot.realized_after_cost_jpy = round(
            lot.realized_after_cost_jpy
            + _finite_float(realized_after_cost_jpy),
            6,
        )
        cost = max(0.0, _finite_float(execution_cost_jpy))
        self.cycle_execution_cost_jpy = round(self.cycle_execution_cost_jpy + cost, 6)
        self._record(
            "LOT_REDUCED",
            now_utc,
            lot_id=lot_id,
            units=units,
            realized_after_cost_jpy=_finite_float(realized_after_cost_jpy),
            execution_cost_jpy=cost,
            cycle_cost_jpy=self.cycle_execution_cost_jpy,
        )
        self._advance_flat_if_empty(now_utc)
        self._persist()

    def stop(self, *, now_utc: datetime, cooldown: timedelta) -> None:
        if self.state is not InventoryState.FLAT:
            raise RuntimeError("STOPPED requires verified FLAT inventory")
        if cooldown.total_seconds() <= 0:
            raise ValueError("cooldown must be positive")
        self.state = InventoryState.STOPPED
        self.cooldown_until_utc = _format_utc(_aware_utc(now_utc) + cooldown)
        self._record("STOPPED", now_utc, cooldown_until_utc=self.cooldown_until_utc)
        self._persist()

    def _advance_flat_if_empty(self, now_utc: datetime) -> None:
        if not self.pending_entry_ids and not any(
            lot.remaining_units > 0 for lot in self.lots.values()
        ):
            self.state = InventoryState.FLAT
            self._record(
                "FLAT",
                now_utc,
                cycle_count=self.cycle_count,
                cycle_retained_return=self.cycle_retained_return,
                cycle_giveback_jpy=self.cycle_giveback_jpy,
                cycle_execution_cost_jpy=self.cycle_execution_cost_jpy,
            )

    def _lot(self, lot_id: str) -> InventoryLot:
        try:
            return self.lots[lot_id]
        except KeyError as exc:
            raise KeyError("unknown bot-owned lot_id") from exc

    def _record(self, event_type: str, now_utc: datetime, **details: Any) -> None:
        self.revision += 1
        self.events.append(
            {
                "revision": self.revision,
                "event_type": event_type,
                "at_utc": _format_utc(now_utc),
                "details": details,
            }
        )
        self.events = self.events[-256:]

    def _persist(self) -> None:
        payload = {
            "contract": INVENTORY_CONTRACT,
            "campaign_id": self.campaign_id,
            "state": self.state.value,
            "revision": self.revision,
            "cooldown_until_utc": self.cooldown_until_utc,
            "stop_reason": self.stop_reason,
            "pending_entry_ids": list(self.pending_entry_ids),
            "lots": [
                item.to_payload()
                for item in sorted(self.lots.values(), key=lambda lot: lot.identity.lot_id)
            ],
            "events": list(self.events),
            "applied_receipt_ids": list(self.applied_receipt_ids),
            "applied_event_dedupe_keys": list(self.applied_event_dedupe_keys),
            "supervision_regime": self.supervision_regime,
            "allowed_strategy_ids": list(self.allowed_strategy_ids),
            "supervision_risk_budget_cap_jpy": self.supervision_risk_budget_cap_jpy,
            "supervision_max_positions_cap": self.supervision_max_positions_cap,
            "supervision_expires_at_utc": self.supervision_expires_at_utc,
            "cycle_start_nav_jpy": self.cycle_start_nav_jpy,
            "cycle_peak_nav_jpy": self.cycle_peak_nav_jpy,
            "cycle_count": self.cycle_count,
            "cycle_retained_return": self.cycle_retained_return,
            "cycle_giveback_jpy": self.cycle_giveback_jpy,
            "cycle_execution_cost_jpy": self.cycle_execution_cost_jpy,
            "profit_lock_triggered": self.profit_lock_triggered,
            "profit_floor_breached": self.profit_floor_breached,
            "profit_lock_reduction_fraction": self.profit_lock_reduction_fraction,
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.state_path.with_name(f".{self.state_path.name}.lock")
        descriptor = os.open(
            lock_path,
            os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        with os.fdopen(descriptor, "a+") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            if self.state_path.exists():
                current = json.loads(self.state_path.read_text(encoding="utf-8"))
                if current.get("contract") != INVENTORY_CONTRACT:
                    raise RuntimeError("inventory state contract changed under lock")
                current_revision = _nonnegative_int(
                    current.get("revision"), "revision"
                )
                if current_revision != self._persisted_revision:
                    raise RuntimeError(
                        "inventory state changed concurrently; reopen before retry"
                    )
            elif self._persisted_revision != 0:
                raise RuntimeError("inventory state disappeared under lock")
            _write_json_atomic(self.state_path, payload)
            self._persisted_revision = self.revision


def broker_position_identity(position: BrokerPosition) -> LotIdentity | None:
    """Return bot ownership only when broker owner and all four tags agree.

    Tagless/manual positions intentionally return ``None`` and remain NO_TOUCH.
    """

    if position.owner is not Owner.TRADER:
        return None
    raw = position.raw if isinstance(position.raw, Mapping) else {}
    for key in ("tradeClientExtensions", "clientExtensions"):
        extension = raw.get(key)
        if not isinstance(extension, Mapping):
            continue
        comment = str(extension.get("comment") or "")
        tokens = {
            name: value
            for token in comment.split()
            if "=" in token
            for name, value in [token.split("=", 1)]
        }
        try:
            identity = LotIdentity.from_broker_client_id(extension.get("id"))
        except ValueError:
            identity = None
        if identity is not None and tokens.get("owner") == OWNER_TAG:
            return identity
        try:
            return LotIdentity.from_metadata(tokens)
        except ValueError:
            continue
    return None


def _unwind_priority(lot: InventoryLot) -> tuple[float, float, str, str]:
    # First harvest profitable/near-entry lots.  Among losing lots, maximize
    # stressed margin relief per yen of realized loss+cost; oldest breaks ties.
    harvest_bucket = 0.0 if lot.current_unrealized_pips >= 0.0 else 1.0
    denominator = max(1.0, lot.estimated_close_loss_and_cost_jpy)
    relief_efficiency = lot.estimated_margin_relief_jpy / denominator
    return (
        harvest_bucket,
        -relief_efficiency,
        lot.opened_at_utc,
        lot.identity.lot_id,
    )


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(
        temp,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass


def _mapping(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("expected object")
    return value


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("datetime must be timezone-aware")
    return value.astimezone(timezone.utc)


def _format_utc(value: datetime) -> str:
    return _aware_utc(value).isoformat().replace("+00:00", "Z")


def _parse_utc(value: object) -> datetime:
    return _aware_utc(datetime.fromisoformat(_utc_text(value).replace("Z", "+00:00")))


def _utc_text(value: object) -> str:
    text = str(value or "")
    if not text:
        raise ValueError("UTC timestamp is required")
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    return _format_utc(parsed)


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a positive integer")
    parsed = int(value)
    if parsed <= 0 or parsed != float(value):
        raise ValueError(f"{label} must be a positive integer")
    return parsed


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a non-negative integer")
    parsed = int(value)
    if parsed < 0 or parsed != float(value):
        raise ValueError(f"{label} must be a non-negative integer")
    return parsed


def _positive_float(value: object, label: str) -> float:
    parsed = _finite_float(value)
    if parsed <= 0:
        raise ValueError(f"{label} must be positive")
    return parsed


def _finite_float(value: object) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("numeric value must be finite")
    return parsed
