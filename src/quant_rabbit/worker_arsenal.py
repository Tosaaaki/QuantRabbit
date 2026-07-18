"""Worker arsenal: the eye-hand-mechanic runtime state machine.

Adjudicated design (W37-W39, 2026-07-18/19):

  EYE    layer-2 discretionary read declares the measured cell each cycle;
         workers never self-arm.  hands > eye: arming alone cannot create
         profit, so every worker here starts SHADOW-only.
  HANDS  each worker is a declared spec bound to exactly one regime cell,
         with a spread-whitelisted habitat and mandatory protections.
  CAGE   inventory discretion is delegated to an AI judge (blind exit
         pilot: +0.6 pips/position over the blunt time-stop), but the
         mechanical fallback NEVER leaves: if the AI heartbeat goes stale
         (dead-man switch, the 2026-06-09 36h-silence lesson), the arsenal
         disarms everything and inventory reverts to the mechanical
         time-stop cut.

Fail-closed rules: unknown cells arm nothing; missing spread measurements
arm nothing; the bleed cell RANGE_HIGH and UNCLEAR arm nothing by
declaration; live promotion requires an operator-approval artifact plus a
sealed prospective cost-positive proof digest — this module cannot grant
it by itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

UTC = timezone.utc

# Cells a worker may bind to.  RANGE_HIGH and UNCLEAR are deliberately
# absent: measured bleed cell and no-information cell arm nothing, ever.
ARMABLE_CELLS = frozenset(
    {
        "TREND_LOW",
        "TREND_HIGH",
        "RANGE_LOW",
        "SQUEEZE_LOW",
        "SQUEEZE_HIGH",
    }
)
NEVER_ARM_CELLS = frozenset({"RANGE_HIGH", "UNCLEAR", "EVENT"})

DEADMAN_TTL = timedelta(minutes=20)


class WorkerArsenalError(ValueError):
    """Raised on any contract violation; callers must fail closed."""


@dataclass(frozen=True)
class WorkerSpec:
    """One declared hand.  Immutable; tuning replaces the spec, never
    mutates it in place, so every change is auditable."""

    worker_id: str
    cell: str
    pairs: tuple[str, ...]
    max_spread_pips: float
    entry_style: str  # "LIMIT_PASSIVE" or "MARKET"
    max_concurrent: int
    time_stop_minutes: int
    kill_switch: str  # human-readable declared kill rule
    per_position_leverage: float

    def __post_init__(self) -> None:
        if self.cell in NEVER_ARM_CELLS:
            raise WorkerArsenalError(
                f"NEVER_ARM_CELL_BINDING: {self.worker_id} -> {self.cell}"
            )
        if self.cell not in ARMABLE_CELLS:
            raise WorkerArsenalError(
                f"UNKNOWN_CELL_BINDING: {self.worker_id} -> {self.cell}"
            )
        if not self.pairs:
            raise WorkerArsenalError(f"EMPTY_HABITAT: {self.worker_id}")
        if self.max_spread_pips <= 0:
            raise WorkerArsenalError(f"INVALID_SPREAD_CAP: {self.worker_id}")
        if self.entry_style not in {"LIMIT_PASSIVE", "MARKET"}:
            raise WorkerArsenalError(f"INVALID_ENTRY_STYLE: {self.worker_id}")
        if self.max_concurrent < 1 or self.time_stop_minutes < 1:
            raise WorkerArsenalError(f"INVALID_PROTECTIONS: {self.worker_id}")
        if self.per_position_leverage <= 0:
            raise WorkerArsenalError(f"INVALID_LEVERAGE: {self.worker_id}")


@dataclass(frozen=True)
class ArmDecision:
    worker_id: str
    pair: str
    armed: bool
    reason: str


@dataclass
class WorkerArsenal:
    """Cycle-level arming and inventory-judgment routing."""

    specs: dict[str, WorkerSpec] = field(default_factory=dict)
    _last_ai_heartbeat: Optional[datetime] = None
    _live_grants: dict[str, str] = field(default_factory=dict)

    def register(self, spec: WorkerSpec) -> None:
        if spec.worker_id in self.specs:
            raise WorkerArsenalError(f"DUPLICATE_WORKER: {spec.worker_id}")
        self.specs[spec.worker_id] = spec

    # ---- dead-man switch -------------------------------------------------
    def ai_heartbeat(self, stamp: datetime) -> None:
        if stamp.tzinfo is None:
            raise WorkerArsenalError("NAIVE_HEARTBEAT_TIMESTAMP")
        if self._last_ai_heartbeat and stamp < self._last_ai_heartbeat:
            raise WorkerArsenalError("HEARTBEAT_CLOCK_REGRESSION")
        self._last_ai_heartbeat = stamp

    def ai_alive(self, now: datetime) -> bool:
        if now.tzinfo is None:
            raise WorkerArsenalError("NAIVE_NOW_TIMESTAMP")
        if self._last_ai_heartbeat is None:
            return False
        return now - self._last_ai_heartbeat <= DEADMAN_TTL

    # ---- arming ----------------------------------------------------------
    def arm_cycle(
        self,
        *,
        now: datetime,
        measured_cell: str,
        spreads_pips: dict[str, float],
    ) -> list[ArmDecision]:
        """One eye-cycle: return the arm decision for every (worker, pair).

        Workers arm only when ALL hold: the AI heartbeat is fresh (the eye
        is alive), the measured cell equals the worker's cell, the pair has
        a fresh spread measurement, and that spread is inside the habitat
        cap.  Anything else disarms with an explicit reason.
        """

        decisions: list[ArmDecision] = []
        eye_alive = self.ai_alive(now)
        for spec in self.specs.values():
            for pair in spec.pairs:
                if not eye_alive:
                    decisions.append(ArmDecision(
                        spec.worker_id, pair, False, "DEADMAN_EYE_STALE"))
                    continue
                if measured_cell in NEVER_ARM_CELLS:
                    decisions.append(ArmDecision(
                        spec.worker_id, pair, False,
                        f"NEVER_ARM_CELL:{measured_cell}"))
                    continue
                if measured_cell != spec.cell:
                    decisions.append(ArmDecision(
                        spec.worker_id, pair, False,
                        f"CELL_MISMATCH:{measured_cell}"))
                    continue
                spread = spreads_pips.get(pair)
                if spread is None:
                    decisions.append(ArmDecision(
                        spec.worker_id, pair, False, "SPREAD_UNMEASURED"))
                    continue
                if spread > spec.max_spread_pips:
                    decisions.append(ArmDecision(
                        spec.worker_id, pair, False,
                        f"SPREAD_OVER_CAP:{spread}"))
                    continue
                decisions.append(ArmDecision(spec.worker_id, pair, True, "ARMED"))
        return decisions

    # ---- inventory discretion -------------------------------------------
    def inventory_action(
        self,
        *,
        now: datetime,
        worker_id: str,
        minutes_held: float,
        ai_decision: Optional[str],
    ) -> str:
        """Route one open position's exit decision.

        The AI judge may answer HOLD or CUT while its heartbeat is fresh.
        A stale heartbeat, or no answer, falls back to the mechanical
        time-stop (the adjudicated floor: never worse than the blunt cut).
        HOLD never extends past 4x the time-stop — a hard ceiling so a
        wrong AI cannot recreate the 12/09 overnight strand.
        """

        spec = self.specs.get(worker_id)
        if spec is None:
            raise WorkerArsenalError(f"UNKNOWN_WORKER: {worker_id}")
        if minutes_held < 0:
            raise WorkerArsenalError("NEGATIVE_HOLD")
        hard_ceiling = spec.time_stop_minutes * 4
        if minutes_held >= hard_ceiling:
            return "CUT_HARD_CEILING"
        if not self.ai_alive(now) or ai_decision is None:
            return (
                "CUT_MECHANICAL_TIME_STOP"
                if minutes_held >= spec.time_stop_minutes
                else "HOLD_MECHANICAL"
            )
        if ai_decision not in {"HOLD", "CUT"}:
            raise WorkerArsenalError(f"INVALID_AI_DECISION: {ai_decision}")
        return "CUT_AI" if ai_decision == "CUT" else "HOLD_AI"

    # ---- live promotion (fail closed) -----------------------------------
    def grant_live(
        self,
        worker_id: str,
        *,
        operator_approval_sha256: str,
        prospective_proof_sha256: str,
    ) -> None:
        if worker_id not in self.specs:
            raise WorkerArsenalError(f"UNKNOWN_WORKER: {worker_id}")
        if len(operator_approval_sha256) != 64 or len(prospective_proof_sha256) != 64:
            raise WorkerArsenalError("INVALID_PROMOTION_DIGESTS")
        self._live_grants[worker_id] = prospective_proof_sha256

    def is_live(self, worker_id: str) -> bool:
        return worker_id in self._live_grants
