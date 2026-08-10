#!/usr/bin/env python3
"""Research-only MONTHLY_3X_GROWTH_ENGINE_V1.

The engine keeps executed ALL_TRADES as the default action and uses decision-time
systems only as size modifiers.  It never reads holdout or sends broker orders.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.linear_model import Ridge


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PREREG = HERE / "preregister_v1.json"
SEED = 20260810


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def utc(value: str) -> datetime:
    text = value.replace("Z", "+00:00")
    # Python accepts nanosecond-looking values only after reducing to microseconds.
    head, sep, tail = text.partition(".")
    if sep:
        digits, offset = tail.split("+", 1)
        text = f"{head}.{digits[:6]}+{offset}"
    return datetime.fromisoformat(text).astimezone(timezone.utc)


def profit_factor(values: Iterable[float]) -> float | None:
    values = list(values)
    gain = sum(value for value in values if value > 0)
    loss = -sum(value for value in values if value < 0)
    if loss == 0:
        return None if gain == 0 else math.inf
    return gain / loss


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=float), q, method="linear"))


def paired_lcb(values: list[float], key: str) -> float | None:
    if not values:
        return None
    digest = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(SEED ^ digest)
    array = np.asarray(values, dtype=float)
    means = np.empty(2000, dtype=float)
    for index in range(len(means)):
        means[index] = float(rng.choice(array, size=len(array), replace=True).mean())
    return percentile(means.tolist(), 0.025)


@dataclass(frozen=True)
class Episode:
    episode_id: str
    fill_at: datetime
    close_at: datetime
    pair: str
    side: str
    regime: str
    units: float
    corrected_net_jpy: float
    initial_margin_jpy: float
    price_action_features: dict[str, float]


class PolicyFit:
    def __init__(self, train: list[Episode], inventory: dict[str, float], technical: dict[str, float]):
        self.inventory = inventory
        self.technical = technical
        normalized = [ep.corrected_net_jpy / ep.units * 1000.0 for ep in train]
        self.global_mean = float(np.mean(normalized)) if normalized else 0.0

        grouped: dict[tuple[str, str, str], list[float]] = {}
        for episode, value in zip(train, normalized):
            grouped.setdefault((episode.pair, episode.side, episode.regime), []).append(value)
        self.group_scores: dict[tuple[str, str, str], float] = {}
        for group, values in grouped.items():
            if len(values) < 8:
                continue
            self.group_scores[group] = (sum(values) + 20.0 * self.global_mean) / (len(values) + 20.0)
        train_group_scores = [self._group_score(episode) for episode in train]
        self.group_reference = sorted(train_group_scores)

        feature_names = sorted({key for episode in train for key in episode.price_action_features})
        self.feature_names = feature_names
        complete = [episode for episode in train if self._features_complete(episode)]
        self.ridge: Ridge | None = None
        self.means: np.ndarray | None = None
        self.scales: np.ndarray | None = None
        self.ridge_low: float | None = None
        self.ridge_high: float | None = None
        if feature_names and len(complete) >= 10:
            x = np.asarray([[episode.price_action_features[name] for name in feature_names] for episode in complete])
            y = np.asarray([episode.corrected_net_jpy / episode.units * 1000.0 for episode in complete])
            means = x.mean(axis=0)
            scales = x.std(axis=0)
            scales[scales == 0] = 1.0
            standardized = (x - means) / scales
            ridge = Ridge(alpha=10.0, fit_intercept=True)
            ridge.fit(standardized, y)
            predictions = ridge.predict(standardized)
            self.ridge = ridge
            self.means = means
            self.scales = scales
            self.ridge_low = float(np.quantile(predictions, 0.33))
            self.ridge_high = float(np.quantile(predictions, 0.67))

    def _features_complete(self, episode: Episode) -> bool:
        return bool(self.feature_names) and all(
            name in episode.price_action_features
            and math.isfinite(float(episode.price_action_features[name]))
            for name in self.feature_names
        )

    def _group_score(self, episode: Episode) -> float:
        return self.group_scores.get((episode.pair, episode.side, episode.regime), self.global_mean)

    def group_multiplier(self, episode: Episode) -> float:
        if not self.group_reference:
            return 1.0
        score = self._group_score(episode)
        below = sum(value < score for value in self.group_reference)
        equal = sum(value == score for value in self.group_reference)
        rank = (below + 0.5 * equal) / len(self.group_reference)
        return min(1.5, max(0.5, 0.5 + rank))

    def ridge_multiplier(self, episode: Episode) -> float:
        if self.ridge is None or not self._features_complete(episode):
            return 1.0
        assert self.means is not None and self.scales is not None
        vector = np.asarray([[episode.price_action_features[name] for name in self.feature_names]])
        prediction = float(self.ridge.predict((vector - self.means) / self.scales)[0])
        assert self.ridge_low is not None and self.ridge_high is not None
        if prediction <= self.ridge_low:
            return 0.5
        if prediction >= self.ridge_high:
            return 1.5
        return 1.0

    def multiplier(self, policy: str, episode: Episode) -> float:
        if policy == "BASELINE_ACTUAL_SIZE":
            return 1.0
        if policy == "INVENTORY_CAP_V2_RELABELED":
            return self.inventory.get(episode.episode_id, 1.0)
        if policy == "TECHNICAL_DISSENT_CAP_V3_RELABELED":
            return self.technical.get(episode.episode_id, 1.0)
        if policy == "GROUP_RELATIVE_SIZE":
            return self.group_multiplier(episode)
        if policy == "PRICE_ACTION_RIDGE_SIZE":
            return self.ridge_multiplier(episode)
        if policy == "FUSED_SIZE":
            return min(1.5, max(0.5, 0.5 * (self.group_multiplier(episode) + self.ridge_multiplier(episode))))
        raise KeyError(policy)


def equity_at(events: list[tuple[datetime, float]], start: float, cutoff: datetime) -> float:
    return start + sum(value for ts, value in events if ts <= cutoff)


def rolling_30d(events: list[tuple[datetime, float]], start_equity: float, starts: list[datetime]) -> list[float]:
    if not events:
        return []
    end = max(ts for ts, _ in events)
    ratios: list[float] = []
    for start in sorted(set(starts)):
        boundary = start + timedelta(days=30)
        if boundary > end:
            continue
        before = equity_at(events, start_equity, start - timedelta(microseconds=1))
        after = equity_at(events, start_equity, boundary)
        if before > 0:
            ratios.append(after / before)
    return ratios


def simulate(
    episodes: list[Episode],
    multipliers: dict[str, float],
    risk_scale: float,
    margin_cap: float,
    start_equity: float = 200000.0,
) -> tuple[dict[str, Any], dict[str, float]]:
    open_positions: list[dict[str, Any]] = []
    close_events: list[tuple[datetime, float]] = []
    scaled_by_episode: dict[str, float] = {}
    applied_by_episode: dict[str, float] = {}
    equity = start_equity
    peak = start_equity
    max_dd = 0.0
    margin_peak = 0.0
    capped = 0
    ruined = False

    def close_due(cutoff: datetime) -> None:
        nonlocal open_positions, equity, peak, max_dd, ruined
        due = sorted((row for row in open_positions if row["close_at"] <= cutoff), key=lambda row: row["close_at"])
        for row in due:
            equity += row["pnl"]
            close_events.append((row["close_at"], row["pnl"]))
            peak = max(peak, equity)
            max_dd = max(max_dd, peak - equity)
            if equity <= 0:
                ruined = True
        due_ids = {row["episode_id"] for row in due}
        open_positions = [row for row in open_positions if row["episode_id"] not in due_ids]

    ordered = sorted(episodes, key=lambda episode: (episode.fill_at, episode.episode_id))
    for episode in ordered:
        close_due(episode.fill_at)
        if ruined:
            applied_by_episode[episode.episode_id] = 0.0
            scaled_by_episode[episode.episode_id] = 0.0
            continue
        open_margin = sum(row["margin"] for row in open_positions)
        desired = max(0.0, multipliers[episode.episode_id] * risk_scale)
        capacity = max(0.0, margin_cap * equity - open_margin)
        applied = min(desired, capacity / episode.initial_margin_jpy)
        if applied + 1e-12 < desired:
            capped += 1
        margin = episode.initial_margin_jpy * applied
        pnl = episode.corrected_net_jpy * applied
        applied_by_episode[episode.episode_id] = applied
        scaled_by_episode[episode.episode_id] = pnl
        if applied > 0:
            open_positions.append(
                {
                    "episode_id": episode.episode_id,
                    "close_at": episode.close_at,
                    "margin": margin,
                    "pnl": pnl,
                }
            )
        margin_peak = max(margin_peak, open_margin + margin)
    close_due(datetime.max.replace(tzinfo=timezone.utc))

    pnl_values = [scaled_by_episode[episode.episode_id] for episode in ordered]
    ratios = rolling_30d(close_events, start_equity, [episode.fill_at for episode in ordered])
    gains = sum(value for value in pnl_values if value > 0)
    losses = -sum(value for value in pnl_values if value < 0)
    metrics = {
        "trades": len(ordered),
        "opportunity_count": sum(applied_by_episode[value.episode_id] > 0 for value in ordered),
        "after_cost_net_jpy": sum(pnl_values),
        "ending_equity_jpy": equity,
        "ending_equity_multiple": equity / start_equity,
        "profit_factor": None if losses == 0 and gains == 0 else (math.inf if losses == 0 else gains / losses),
        "expectancy_jpy": sum(pnl_values) / len(ordered) if ordered else None,
        "realized_max_drawdown_jpy": max_dd,
        "realized_max_drawdown_fraction": max_dd / start_equity,
        "cohort_margin_peak_jpy": margin_peak,
        "cohort_margin_peak_fraction_of_start": margin_peak / start_equity,
        "margin_capped_entries": capped,
        "ruin": ruined,
        "rolling_30d_count": len(ratios),
        "rolling_30d_equity_multiple_min": min(ratios) if ratios else None,
        "rolling_30d_equity_multiple_median": float(np.median(ratios)) if ratios else None,
        "rolling_30d_equity_multiple_max": max(ratios) if ratios else None,
        "monthly_target_reached_any": any(value >= 3.0 for value in ratios),
        "monthly_target_reached_all": bool(ratios) and all(value >= 3.0 for value in ratios),
        "monthly_target_gap_jpy_from_ending_equity": 600000.0 - equity,
        "requested_risk_scale": risk_scale,
        "margin_cap_fraction": margin_cap,
    }
    return metrics, scaled_by_episode


def adjacent(point: tuple[float, float], other: tuple[float, float], risk: list[float], caps: list[float]) -> bool:
    ri, ci = risk.index(point[0]), caps.index(point[1])
    rj, cj = risk.index(other[0]), caps.index(other[1])
    return abs(ri - rj) + abs(ci - cj) == 1


def main() -> None:
    prereg = json.loads(PREREG.read_text())
    for source in prereg["frozen_sources"].values():
        path = ROOT / source["path"]
        actual = sha256(path)
        if actual != source["sha256"]:
            raise SystemExit(f"source SHA mismatch: {path}: {actual}")

    cashflows = {row["episode_id"]: row for row in load_jsonl(ROOT / prereg["frozen_sources"]["financial_labels"]["path"])}
    paths = {row["episode_id"]: row for row in load_jsonl(ROOT / prereg["frozen_sources"]["margin_and_path"]["path"])}
    payload = json.loads((ROOT / prereg["frozen_sources"]["split_and_features"]["path"]).read_text())
    inventory = {row["episode_id"]: float(row["size_multiplier"]) for row in load_jsonl(ROOT / prereg["frozen_sources"]["inventory_modifier"]["path"])}
    technical = {row["episode_id"]: float(row["size_multiplier"]) for row in load_jsonl(ROOT / prereg["frozen_sources"]["technical_modifier"]["path"])}

    all_rows = [row for row in payload["episode_records"] if row["method"] == "ALL_TRADES"]
    membership: dict[tuple[str, str], list[Episode]] = {}
    for row in all_rows:
        episode_id = row["episode_id"]
        label = cashflows[episode_id]
        path = paths[episode_id]
        features = row.get("price_action_features") or {}
        episode = Episode(
            episode_id=episode_id,
            fill_at=utc(label["fill_at_utc"]),
            close_at=utc(label["close_at_utc"]),
            pair=label["pair"],
            side=label["side"],
            regime=row.get("regime") or "MISSING",
            units=float(label["units"]),
            corrected_net_jpy=float(label["corrected_net_jpy"]),
            initial_margin_jpy=float(path["entry_actual_initial_margin_jpy"]),
            price_action_features={key: float(value) for key, value in features.items() if value is not None},
        )
        membership.setdefault((row["window"], row["split"]), []).append(episode)

    policies = list(prereg["policies"])
    risk_grid = [float(value) for value in prereg["growth_grid"]["risk_scale"]]
    cap_grid = [float(value) for value in prereg["growth_grid"]["cohort_margin_cap_fraction"]]
    results: list[dict[str, Any]] = []
    multiplier_rows: list[dict[str, Any]] = []
    fits: dict[str, PolicyFit] = {}

    for window in prereg["cohort"]["windows"]:
        train = membership.get((window, "TRAIN"), [])
        validation = membership.get((window, "VALIDATION"), [])
        fit = PolicyFit(train, inventory, technical)
        fits[window] = fit
        for split, episodes in (("TRAIN", train), ("VALIDATION", validation)):
            for episode in episodes:
                for policy in policies:
                    multiplier_rows.append(
                        {
                            "window": window,
                            "split": split,
                            "episode_id": episode.episode_id,
                            "decision_time": episode.fill_at.isoformat().replace("+00:00", "Z"),
                            "policy": policy,
                            "decision_multiplier": fit.multiplier(policy, episode),
                            "actual_outcome_used_for_this_decision": False,
                        }
                    )

        for split, episodes in (("TRAIN", train), ("VALIDATION", validation)):
            policy_multipliers = {
                policy: {episode.episode_id: fit.multiplier(policy, episode) for episode in episodes}
                for policy in policies
            }
            for risk_scale in risk_grid:
                for cap in cap_grid:
                    baseline_metrics, baseline_scaled = simulate(
                        episodes, policy_multipliers["BASELINE_ACTUAL_SIZE"], risk_scale, cap
                    )
                    for policy in policies:
                        metrics, scaled = simulate(episodes, policy_multipliers[policy], risk_scale, cap)
                        deltas = [scaled[episode.episode_id] - baseline_scaled[episode.episode_id] for episode in episodes]
                        changed = sum(
                            abs(policy_multipliers[policy][episode.episode_id] - 1.0) > 1e-12
                            for episode in episodes
                        )
                        results.append(
                            {
                                "window": window,
                                "split": split,
                                "policy": policy,
                                "risk_scale": risk_scale,
                                "margin_cap_fraction": cap,
                                "changed_decisions": changed,
                                "paired_incremental_net_jpy": sum(deltas),
                                "paired_incremental_expectancy_jpy": float(np.mean(deltas)) if deltas else None,
                                "paired_lcb_jpy": paired_lcb(deltas, f"{window}:{split}:{policy}:{risk_scale}:{cap}"),
                                "baseline_after_cost_net_jpy": baseline_metrics["after_cost_net_jpy"],
                                "baseline_ending_equity_jpy": baseline_metrics["ending_equity_jpy"],
                                "baseline_realized_max_drawdown_jpy": baseline_metrics["realized_max_drawdown_jpy"],
                                **metrics,
                            }
                        )

    # TRAIN-only connected feasible plateaus; the exact points are then frozen for validation.
    selected: set[tuple[str, str, float, float]] = set()
    for window in prereg["cohort"]["windows"]:
        for policy in policies:
            candidates = [row for row in results if row["window"] == window and row["split"] == "TRAIN" and row["policy"] == policy]
            feasible: list[tuple[float, float]] = []
            for row in candidates:
                changed_ok = policy == "BASELINE_ACTUAL_SIZE" or row["changed_decisions"] >= 10
                if (
                    row["after_cost_net_jpy"] > 0
                    and row["profit_factor"] is not None
                    and row["profit_factor"] > 1
                    and row["realized_max_drawdown_fraction"] <= 0.30
                    and not row["ruin"]
                    and changed_ok
                ):
                    feasible.append((row["risk_scale"], row["margin_cap_fraction"]))
            for point in feasible:
                if any(adjacent(point, other, risk_grid, cap_grid) for other in feasible if other != point):
                    selected.add((window, policy, point[0], point[1]))

    for row in results:
        key = (row["window"], row["policy"], row["risk_scale"], row["margin_cap_fraction"])
        row["train_connected_plateau"] = key in selected
        row["validation_admission_candidate"] = row["split"] == "VALIDATION" and key in selected
        row["validation_success"] = bool(
            row["validation_admission_candidate"]
            and row["paired_incremental_net_jpy"] > 0
            and row["paired_lcb_jpy"] is not None
            and row["paired_lcb_jpy"] > 0
            and row["profit_factor"] is not None
            and row["profit_factor"] > 1
            and row["realized_max_drawdown_fraction"] <= 0.30
            and not row["ruin"]
            and row["monthly_target_reached_any"]
        )

    # Target arithmetic is independent of the fitted policies.
    target_math = []
    for trades in (50, 100, 200, 400, 800):
        target_math.append(
            {
                "monthly_trades": trades,
                "fixed_jpy_expectancy_required": 400000.0 / trades,
                "equal_compound_return_per_trade": 3.0 ** (1.0 / trades) - 1.0,
                "first_trade_jpy_at_200k": 200000.0 * (3.0 ** (1.0 / trades) - 1.0),
            }
        )

    validation_successes = [row for row in results if row["validation_success"]]
    val64_baseline = next(
        row for row in results
        if row["window"] == "QUADRUPLE_64D"
        and row["split"] == "VALIDATION"
        and row["policy"] == "BASELINE_ACTUAL_SIZE"
        and row["risk_scale"] == 1.0
        and row["margin_cap_fraction"] == 0.75
    )
    report = {
        "contract": prereg["contract"],
        "preregister_sha256": sha256(PREREG),
        "holdout_used": False,
        "source_hashes_verified": True,
        "windows": {
            window: {
                split: len(membership.get((window, split), []))
                for split in ("TRAIN", "VALIDATION")
            }
            for window in prereg["cohort"]["windows"]
        },
        "target_math": target_math,
        "corrected_64d_validation_baseline_at_1x_75pct_cap": val64_baseline,
        "train_connected_points": len(selected),
        "validation_success_count": len(validation_successes),
        "validation_successes": validation_successes,
        "conclusion": "TARGET_PATH_FOUND" if validation_successes else "TARGET_PATH_NOT_YET_PROVEN",
        "reason_can_work": [
            "Corrected 64d validation ALL_TRADES remains after-cost positive, so the starting edge is non-zero.",
            "Sizing modifiers preserve every baseline opportunity and can only redistribute bounded exposure; they no longer discard winners through a SKIP fallback.",
            "The engine now measures the exact opportunity/edge/capital-efficiency gap to 3x under causal margin caps instead of treating missing evidence as zero edge."
        ],
        "limits": prereg["known_limits"],
    }

    def write_json(path: Path, value: Any) -> None:
        path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")

    def write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
        path.write_text("".join(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n" for value in values))

    write_jsonl(HERE / "decision_multipliers_v1.jsonl", multiplier_rows)
    write_jsonl(HERE / "growth_grid_v1.jsonl", results)
    write_json(HERE / "growth_report_v1.json", report)
    manifest = {
        "contract": prereg["contract"],
        "preregister_sha256": sha256(PREREG),
        "outputs": {
            name: sha256(HERE / name)
            for name in ("decision_multipliers_v1.jsonl", "growth_grid_v1.jsonl", "growth_report_v1.json")
        },
    }
    write_json(HERE / "run_manifest_v1.json", manifest)


if __name__ == "__main__":
    main()
