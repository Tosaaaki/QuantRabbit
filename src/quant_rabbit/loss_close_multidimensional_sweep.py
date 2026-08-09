"""Pre-holdout multidimensional sweep and plateau-selection primitives.

The sweep is deliberately separated from order/runtime code.  TRAIN chooses a
connected parameter region; VALIDATION may only confirm or reject that frozen
region.  TEST/HOLDOUT input is invalid.  A single best cell can never survive.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from statistics import median
from typing import Any, Mapping, Sequence

from quant_rabbit.loss_close_price_action_shadow import PriceActionFeatureSpec


MULTIDIMENSIONAL_SWEEP_CONTRACT = "loss_close_multidimensional_sweep_v1"
STAGE_1 = "GEOMETRY"
STAGE_2 = "LOCAL_STRUCTURE"
STAGE_3 = "LOCAL_TOLERANCE"
_SPLITS = ("TRAIN", "VALIDATION")
_CONFIG_FIELDS = (
    "frames_seconds",
    "structure_bars",
    "regime_bars",
    "breakout_bars",
    "acceptance_bars",
    "attack_tolerance_ratio",
)
_ARM_FIELDS = (
    "mean_net_jpy",
    "max_drawdown_jpy",
    "ruin_floor_breach_count",
    "margin_closeout_breach_count",
    "incomplete_unwind_count",
    "unresolved_fill_order_count",
)


@dataclass(frozen=True)
class SweepContract:
    """Frozen anti-overfit and pre-holdout acceptance rules."""

    min_events_per_split: int = 30
    min_plateau_cells: int = 3
    min_centre_neighbours: int = 2
    plateau_relative_floor: float = 0.80
    min_increment_jpy: float = 0.0
    max_unwind_seconds: int = 3600
    embargo_seconds: int = 3600


@dataclass(frozen=True)
class SweepGridPoint:
    stage: str
    feature_spec: PriceActionFeatureSpec

    @property
    def config(self) -> dict[str, Any]:
        return {
            "frames_seconds": list(self.feature_spec.frames_seconds),
            "structure_bars": self.feature_spec.structure_bars,
            "regime_bars": self.feature_spec.regime_bars,
            "breakout_bars": self.feature_spec.breakout_bars,
            "acceptance_bars": self.feature_spec.acceptance_bars,
            "attack_tolerance_ratio": self.feature_spec.attack_tolerance_ratio,
        }

    @property
    def config_id(self) -> str:
        encoded = json.dumps(
            self.config, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        return hashlib.sha256(encoded).hexdigest()[:16]


def build_stage1_price_action_grid() -> tuple[SweepGridPoint, ...]:
    """Return a bounded geometry grid instead of a full Cartesian search.

    Window choices are coupled into coherent geometries.  This keeps Stage 1
    at 27 cells (3 frame sets x 3 structure/regime pairs x 3 break/accept
    pairs), with tolerance frozen.  Later stages may refine only TRAIN
    plateaus returned by this stage.
    """

    frames = ((60,), (300,), (60, 300))
    structures = ((6, 18), (12, 24), (24, 48))
    breaks = ((4, 2), (8, 2), (12, 3))
    return tuple(
        SweepGridPoint(
            stage=STAGE_1,
            feature_spec=PriceActionFeatureSpec(
                frames_seconds=frame,
                structure_bars=structure,
                regime_bars=regime,
                breakout_bars=breakout,
                acceptance_bars=acceptance,
                attack_tolerance_ratio=0.08,
            ),
        )
        for frame in frames
        for structure, regime in structures
        for breakout, acceptance in breaks
    )


def build_local_refinement_grid(
    centres: Sequence[PriceActionFeatureSpec], *, stage: str
) -> tuple[SweepGridPoint, ...]:
    """Build one-axis-at-a-time neighbours around TRAIN plateau centres."""

    if stage not in {STAGE_2, STAGE_3}:
        raise ValueError("local refinement stage must be STAGE_2 or STAGE_3")
    points: dict[str, SweepGridPoint] = {}
    for centre in centres:
        candidates = [centre]
        if stage == STAGE_2:
            structure_values = {
                max(4, centre.structure_bars // 2),
                centre.structure_bars,
                centre.structure_bars * 2,
            }
            for structure in sorted(structure_values):
                candidates.append(
                    _replace_spec(
                        centre,
                        structure_bars=structure,
                        regime_bars=max(centre.regime_bars, structure),
                    )
                )
            for breakout in sorted(
                {max(2, centre.breakout_bars // 2), centre.breakout_bars, centre.breakout_bars * 2}
            ):
                candidates.append(_replace_spec(centre, breakout_bars=breakout))
            for acceptance in sorted(
                {2, centre.acceptance_bars, min(4, centre.acceptance_bars + 1)}
            ):
                candidates.append(_replace_spec(centre, acceptance_bars=acceptance))
        else:
            candidates.extend(
                _replace_spec(centre, attack_tolerance_ratio=value)
                for value in (0.04, 0.08, 0.12)
            )
        for spec in candidates:
            if spec.regime_bars < spec.breakout_bars + spec.acceptance_bars:
                continue
            point = SweepGridPoint(stage=stage, feature_spec=spec)
            points[point.config_id] = point
    return tuple(points[key] for key in sorted(points))


def evaluate_multidimensional_plateau(
    rows: Sequence[Mapping[str, Any]],
    *,
    contract: SweepContract = SweepContract(),
    holdout_used: bool = False,
) -> dict[str, Any]:
    """Select on TRAIN and confirm the frozen connected region on VALIDATION.

    Each row is one aggregate config/split cell.  Its three arms must use the
    same split cohort and cost model.  The price-action arm must beat both the
    same-frame inventory and one/two-candle controls without worsening risk,
    fill resolution, or unwind completion.
    """

    blockers = _contract_issues(contract, holdout_used=holdout_used)
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        blockers.append("ROWS_NOT_SEQUENCE")
        return _result(blockers=blockers)
    try:
        frozen = tuple(dict(row) for row in rows)
    except Exception:
        return _result(blockers=blockers + ["ROWS_SNAPSHOT_UNREADABLE"])
    if not frozen:
        blockers.append("ROWS_EMPTY")

    cleaned: dict[tuple[str, str], dict[str, Any]] = {}
    split_cohorts: dict[str, tuple[str, str, int]] = {}
    for index, row in enumerate(frozen):
        split = row.get("split")
        if split not in _SPLITS:
            blockers.append(f"FORBIDDEN_OR_INVALID_SPLIT:{index}")
            continue
        config = _clean_config(row.get("config"))
        if config is None:
            blockers.append(f"INVALID_CONFIG:{index}")
            continue
        config_id = _config_id(config)
        if row.get("config_id") != config_id:
            blockers.append(f"CONFIG_ID_MISMATCH:{index}")
        cohort_sha = row.get("cohort_sha256")
        cost_sha = row.get("cost_model_sha256")
        event_count = row.get("event_count")
        if not _sha256(cohort_sha):
            blockers.append(f"INVALID_COHORT_SHA256:{index}")
        if not _sha256(cost_sha):
            blockers.append(f"INVALID_COST_MODEL_SHA256:{index}")
        if event_count.__class__ is not int or event_count < contract.min_events_per_split:
            blockers.append(f"INSUFFICIENT_EVENTS:{index}")
        cohort_key = (str(cohort_sha), str(cost_sha), int(event_count or 0))
        if split in split_cohorts and split_cohorts[split] != cohort_key:
            blockers.append(f"SPLIT_COHORT_OR_COST_MISMATCH:{split}")
        else:
            split_cohorts[split] = cohort_key
        arms = _clean_arms(row.get("arms"))
        if arms is None:
            blockers.append(f"INVALID_ARMS:{index}")
            continue
        key = (config_id, split)
        if key in cleaned:
            blockers.append(f"DUPLICATE_CONFIG_SPLIT:{index}")
        cleaned[key] = {
            "config_id": config_id,
            "config": config,
            "split": split,
            "event_count": event_count,
            "arms": arms,
        }
    if set(split_cohorts) != set(_SPLITS):
        blockers.append("TRAIN_AND_VALIDATION_REQUIRED")
    elif len({value[1] for value in split_cohorts.values()}) != 1:
        blockers.append("COST_MODEL_MISMATCH_BETWEEN_SPLITS")
    train_ids = {key[0] for key in cleaned if key[1] == "TRAIN"}
    validation_ids = {key[0] for key in cleaned if key[1] == "VALIDATION"}
    if train_ids != validation_ids:
        blockers.append("CONFIG_SET_MISMATCH_BETWEEN_SPLITS")
    if blockers:
        return _result(blockers=list(dict.fromkeys(blockers)))

    assert train_ids
    train_scores = {
        config_id: _cell_score(cleaned[(config_id, "TRAIN")])
        for config_id in train_ids
    }
    positive_scores = [
        score["increment_jpy"]
        for score in train_scores.values()
        if score["risk_ok"] and score["increment_jpy"] > contract.min_increment_jpy
    ]
    if not positive_scores:
        return _result(
            blockers=[],
            status="REJECTED_NO_TRAIN_INCREMENT",
            payload=_payload_no_survivor(train_scores),
        )
    best = max(positive_scores)
    relative_floor = max(contract.min_increment_jpy, best * contract.plateau_relative_floor)
    eligible = {
        config_id
        for config_id, score in train_scores.items()
        if score["risk_ok"] and score["increment_jpy"] >= relative_floor
    }
    configs = {config_id: cleaned[(config_id, "TRAIN")]["config"] for config_id in train_ids}
    train_components = _plateau_components(
        eligible,
        configs,
        min_cells=contract.min_plateau_cells,
        min_centre_neighbours=contract.min_centre_neighbours,
    )
    if not train_components:
        return _result(
            blockers=[],
            status="REJECTED_ISOLATED_TRAIN_PEAK",
            payload={
                **_payload_no_survivor(train_scores),
                "train_relative_floor_jpy": relative_floor,
            },
        )

    # Component choice uses TRAIN only.  VALIDATION is not consulted here.
    selected = max(
        train_components,
        key=lambda component: (
            median(train_scores[x]["increment_jpy"] for x in component),
            min(train_scores[x]["increment_jpy"] for x in component),
            len(component),
            tuple(sorted(component)),
        ),
    )
    validation_scores = {
        config_id: _cell_score(cleaned[(config_id, "VALIDATION")])
        for config_id in selected
    }
    validation_eligible = {
        config_id
        for config_id, score in validation_scores.items()
        if score["risk_ok"] and score["increment_jpy"] > contract.min_increment_jpy
    }
    validation_components = _plateau_components(
        validation_eligible,
        configs,
        min_cells=contract.min_plateau_cells,
        min_centre_neighbours=contract.min_centre_neighbours,
    )
    surviving = [component for component in validation_components if component <= selected]
    survives = bool(surviving)
    centre = _component_centre(selected, configs, train_scores)
    return _result(
        blockers=[],
        status=(
            "PRE_HOLDOUT_PLATEAU_SURVIVES_VALIDATION"
            if survives
            else "REJECTED_ON_VALIDATION"
        ),
        payload={
            "train_relative_floor_jpy": relative_floor,
            "selected_train_plateau_config_ids": sorted(selected),
            "selected_train_plateau_size": len(selected),
            "representative_config_id": centre,
            "representative_config": configs[centre],
            "train_scores": {key: train_scores[key] for key in sorted(selected)},
            "validation_scores": {
                key: validation_scores[key] for key in sorted(selected)
            },
            "validation_surviving_plateau_config_ids": (
                sorted(max(surviving, key=len)) if surviving else []
            ),
            "hypothesis_survives_pre_holdout": survives,
            "holdout_unlock_allowed": False,
            "selection_used_validation": False,
            "single_best_cell_adoption_allowed": False,
        },
    )


def _replace_spec(spec: PriceActionFeatureSpec, **updates: Any) -> PriceActionFeatureSpec:
    values = {
        "frames_seconds": spec.frames_seconds,
        "structure_bars": spec.structure_bars,
        "regime_bars": spec.regime_bars,
        "breakout_bars": spec.breakout_bars,
        "acceptance_bars": spec.acceptance_bars,
        "attack_tolerance_ratio": spec.attack_tolerance_ratio,
    }
    values.update(updates)
    return PriceActionFeatureSpec(**values)


def _clean_config(value: object) -> dict[str, Any] | None:
    if not isinstance(value, Mapping) or set(value) != set(_CONFIG_FIELDS):
        return None
    try:
        frames = tuple(value["frames_seconds"])
        spec = PriceActionFeatureSpec(
            frames_seconds=frames,
            structure_bars=value["structure_bars"],
            regime_bars=value["regime_bars"],
            breakout_bars=value["breakout_bars"],
            acceptance_bars=value["acceptance_bars"],
            attack_tolerance_ratio=value["attack_tolerance_ratio"],
        )
    except Exception:
        return None
    if (
        not frames
        or any(item.__class__ is not int or item < 60 or item % 5 for item in frames)
        or len(set(frames)) != len(frames)
        or any(
            getattr(spec, name).__class__ is not int or getattr(spec, name) < 2
            for name in ("structure_bars", "regime_bars", "breakout_bars", "acceptance_bars")
        )
        or spec.structure_bars < 4
        or spec.regime_bars < spec.structure_bars
        or spec.regime_bars < spec.breakout_bars + spec.acceptance_bars
        or spec.attack_tolerance_ratio.__class__ is not float
        or not math.isfinite(spec.attack_tolerance_ratio)
        or not 0.0 < spec.attack_tolerance_ratio <= 0.5
    ):
        return None
    return {
        "frames_seconds": list(frames),
        "structure_bars": spec.structure_bars,
        "regime_bars": spec.regime_bars,
        "breakout_bars": spec.breakout_bars,
        "acceptance_bars": spec.acceptance_bars,
        "attack_tolerance_ratio": spec.attack_tolerance_ratio,
    }


def _clean_arms(value: object) -> dict[str, dict[str, float]] | None:
    if not isinstance(value, Mapping) or set(value) != {
        "INVENTORY_ONLY",
        "CANDLE_1_2",
        "PRICE_ACTION_MULTI_BAR",
    }:
        return None
    out: dict[str, dict[str, float]] = {}
    for name, raw in value.items():
        if not isinstance(raw, Mapping) or set(raw) != set(_ARM_FIELDS):
            return None
        clean: dict[str, float] = {}
        for field in _ARM_FIELDS:
            item = raw[field]
            if item.__class__ not in {int, float} or not math.isfinite(float(item)):
                return None
            if field != "mean_net_jpy" and float(item) < 0.0:
                return None
            clean[field] = float(item)
        out[str(name)] = clean
    return out


def _cell_score(row: Mapping[str, Any]) -> dict[str, Any]:
    arms = row["arms"]
    pa = arms["PRICE_ACTION_MULTI_BAR"]
    controls = (arms["INVENTORY_ONLY"], arms["CANDLE_1_2"])
    increment = min(pa["mean_net_jpy"] - item["mean_net_jpy"] for item in controls)
    risk_ok = (
        pa["max_drawdown_jpy"] <= min(item["max_drawdown_jpy"] for item in controls)
        and pa["ruin_floor_breach_count"] <= min(
            item["ruin_floor_breach_count"] for item in controls
        )
        and pa["margin_closeout_breach_count"] <= min(
            item["margin_closeout_breach_count"] for item in controls
        )
        and pa["incomplete_unwind_count"] == 0.0
        and pa["unresolved_fill_order_count"] == 0.0
    )
    return {"increment_jpy": increment, "risk_ok": risk_ok}


def _plateau_components(
    ids: set[str],
    configs: Mapping[str, Mapping[str, Any]],
    *,
    min_cells: int,
    min_centre_neighbours: int,
) -> list[set[str]]:
    neighbours = {
        key: {other for other in ids - {key} if _adjacent(configs[key], configs[other], configs)}
        for key in ids
    }
    components: list[set[str]] = []
    remaining = set(ids)
    while remaining:
        root = next(iter(remaining))
        component = {root}
        stack = [root]
        while stack:
            current = stack.pop()
            for other in neighbours[current] - component:
                component.add(other)
                stack.append(other)
        remaining -= component
        has_centre = (
            max(len(neighbours[x] & component) for x in component)
            >= min_centre_neighbours
        )
        if len(component) >= min_cells and has_centre:
            components.append(component)
    return components


def _adjacent(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    configs: Mapping[str, Mapping[str, Any]],
) -> bool:
    changed = [field for field in _CONFIG_FIELDS if left[field] != right[field]]
    if len(changed) != 1:
        return False
    field = changed[0]
    raw_values = {_sortable(config[field]) for config in configs.values()}
    values = (
        sorted(raw_values, key=repr)
        if all(isinstance(value, tuple) for value in raw_values)
        else sorted(raw_values)
    )
    return abs(values.index(_sortable(left[field])) - values.index(_sortable(right[field]))) == 1


def _sortable(value: Any) -> Any:
    return tuple(value) if isinstance(value, list) else value


def _component_centre(
    component: set[str],
    configs: Mapping[str, Mapping[str, Any]],
    scores: Mapping[str, Mapping[str, Any]],
) -> str:
    degrees = {
        key: sum(_adjacent(configs[key], configs[other], configs) for other in component - {key})
        for key in component
    }
    return max(
        component,
        key=lambda key: (degrees[key], scores[key]["increment_jpy"], key),
    )


def _config_id(config: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        config, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def _sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(x in "0123456789abcdef" for x in value)
    )


def _contract_issues(contract: SweepContract, *, holdout_used: bool) -> list[str]:
    issues: list[str] = []
    if holdout_used is not False:
        issues.append("HOLDOUT_USE_FORBIDDEN")
    if contract.__class__ is not SweepContract:
        return issues + ["INVALID_SWEEP_CONTRACT"]
    for name in (
        "min_events_per_split",
        "min_plateau_cells",
        "min_centre_neighbours",
        "max_unwind_seconds",
        "embargo_seconds",
    ):
        value = getattr(contract, name)
        if value.__class__ is not int or value < 1:
            issues.append(f"INVALID_CONTRACT_INTEGER:{name}")
    if contract.embargo_seconds < contract.max_unwind_seconds:
        issues.append("EMBARGO_SHORTER_THAN_MAX_UNWIND")
    if (
        contract.plateau_relative_floor.__class__ is not float
        or not 0.0 < contract.plateau_relative_floor <= 1.0
    ):
        issues.append("INVALID_PLATEAU_RELATIVE_FLOOR")
    if contract.min_increment_jpy.__class__ is not float or contract.min_increment_jpy < 0.0:
        issues.append("INVALID_MIN_INCREMENT_JPY")
    return issues


def _payload_no_survivor(scores: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "train_scores": {key: scores[key] for key in sorted(scores)},
        "selected_train_plateau_config_ids": [],
        "hypothesis_survives_pre_holdout": False,
        "holdout_unlock_allowed": False,
        "selection_used_validation": False,
        "single_best_cell_adoption_allowed": False,
    }


def _result(
    *, blockers: Sequence[str], status: str = "BLOCKED", payload: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "contract": MULTIDIMENSIONAL_SWEEP_CONTRACT,
        "status": status,
        "blockers": list(blockers),
        "read_only": True,
        "paper_permission_allowed": False,
        "live_permission_allowed": False,
        "broker_order_allowed": False,
        "deployment_allowed": False,
        "holdout_used": False,
        "always_profit_claim_allowed": False,
        "statistical_claim_allowed": False,
    }
    if payload:
        result.update(payload)
    return result
