from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Iterator

from quant_rabbit.paths import (
    DEFAULT_EXECUTION_LEDGER_DB,
    DEFAULT_HISTORY_DB,
    DEFAULT_OUTCOME_MART,
    DEFAULT_OUTCOME_MART_REPORT,
)


# These percentiles are report descriptors only. They summarize archive outcome
# tails for review and are not production risk gates or strategy thresholds.
POSITIVE_TAIL_PERCENTILE = 0.75
NEGATIVE_TAIL_PERCENTILE = 0.25

# Report length only; the JSON output keeps every bucket so downstream tools do
# not depend on a display limit.
REPORT_ROW_LIMIT = 16
REPORT_MIN_CONDITION_OUTCOMES = 5
CONTEXT_FEATURE_TEXT_LIMIT = 160

# Offline validation sample size only. This is the minimum number of prior
# comparable outcomes before the report describes whether a historical
# condition edge would have pointed in the right direction; it is not a live
# trading gate and must not block or authorize a lane.
VALIDATION_MIN_PRIOR_OUTCOMES = 5

_UTC = timezone.utc


@dataclass(frozen=True)
class OutcomeMartSummary:
    output_path: Path
    report_path: Path
    status: str
    archive_outcomes: int
    execution_ledger_outcomes: int
    story_observations: int
    condition_edges: int
    condition_rollups: int
    validated_condition_outcomes: int
    condition_directional_hit_rate_pct: float | None
    method_edges: int
    setup_buckets: int
    context_feature_edges: int
    context_feature_outcomes: int
    context_feature_coverage_pct: float


@dataclass(frozen=True)
class OutcomeRow:
    source: str
    pair: str
    direction: str
    method: str
    order_type: str
    session_bucket: str
    regime: str
    pl_jpy: float
    observed_at_utc: datetime | None = None
    context_evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StoryObservation:
    pair: str
    direction: str
    method: str
    order_type: str
    session_bucket: str
    regime: str


@dataclass
class _Bucket:
    pair: str
    direction: str
    method: str = "ALL"
    order_type: str = "ALL"
    session_bucket: str = "ALL"
    regime: str = "ALL"
    values: list[float] = field(default_factory=list)
    archive_outcomes: int = 0
    execution_ledger_outcomes: int = 0
    story_observations: int = 0

    @property
    def key(self) -> str:
        return ":".join((self.pair, self.direction, self.method, self.order_type, self.session_bucket, self.regime))

    def add_outcome(self, value: float, *, source: str) -> None:
        self.values.append(value)
        if source == "execution_ledger":
            self.execution_ledger_outcomes += 1
        else:
            self.archive_outcomes += 1

    def add_observation(self) -> None:
        self.story_observations += 1

    def to_dict(self) -> dict[str, Any]:
        values = sorted(self.values)
        wins = [value for value in values if value > 0]
        losses = [value for value in values if value < 0]
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        outcome_n = len(values)
        return {
            "key": self.key,
            "pair": self.pair,
            "direction": self.direction,
            "method": self.method,
            "order_type": self.order_type,
            "session_bucket": self.session_bucket,
            "regime": self.regime,
            "outcome_n": outcome_n,
            "archive_outcome_n": self.archive_outcomes,
            "execution_ledger_outcome_n": self.execution_ledger_outcomes,
            "story_observation_n": self.story_observations,
            "win_n": len(wins),
            "loss_n": len(losses),
            "win_rate_pct": _round((len(wins) / outcome_n) * 100.0) if outcome_n else None,
            "net_jpy": _round(sum(values)) if values else 0.0,
            "avg_jpy": _round(sum(values) / outcome_n) if outcome_n else None,
            "median_jpy": _round(float(median(values))) if values else None,
            "best_jpy": _round(max(values)) if values else None,
            "worst_jpy": _round(min(values)) if values else None,
            "positive_tail_jpy": _round(_percentile(wins, POSITIVE_TAIL_PERCENTILE)) if wins else None,
            "negative_tail_jpy": _round(_percentile(losses, NEGATIVE_TAIL_PERCENTILE)) if losses else None,
            "profit_factor": _round(gross_profit / gross_loss) if gross_loss > 0 else (None if gross_profit == 0 else "INF"),
            "evidence_state": _evidence_state(values),
        }


@dataclass
class _ValidationBucket:
    values: list[float] = field(default_factory=list)

    @property
    def outcome_n(self) -> int:
        return len(self.values)

    @property
    def net_jpy(self) -> float:
        return sum(self.values)

    def add_outcome(self, value: float) -> None:
        self.values.append(value)


@dataclass
class _ValidationMatchStats:
    key: tuple[str, str, str, str]
    predicted_sign: int
    values: list[float] = field(default_factory=list)
    hits: int = 0

    def add_match(self, value: float) -> None:
        actual_sign = 1 if value > 0 else -1
        if actual_sign == self.predicted_sign:
            self.hits += 1
        self.values.append(value)

    def to_dict(self) -> dict[str, Any]:
        wins = [value for value in self.values if value > 0]
        outcome_n = len(self.values)
        method, order_type, session_bucket, regime = self.key
        return {
            "key": ":".join(("ALL", "ALL", method, order_type, session_bucket, regime)),
            "method": method,
            "order_type": order_type,
            "session_bucket": session_bucket,
            "regime": regime,
            "predicted_edge": "POSITIVE" if self.predicted_sign > 0 else "NEGATIVE",
            "outcomes": outcome_n,
            "directional_hit_outcomes": self.hits,
            "directional_hit_rate_pct": _round((self.hits / outcome_n) * 100.0) if outcome_n else None,
            "actual_net_jpy": _round(sum(self.values)) if self.values else 0.0,
            "actual_win_rate_pct": _round((len(wins) / outcome_n) * 100.0) if outcome_n else None,
            "match_scope": "rollup" if "ALL" in self.key else "exact",
        }


@dataclass
class _ContextFeatureBucket:
    feature_type: str
    feature: str
    values: list[float] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.feature_type}:{self.feature}"

    def add_outcome(self, value: float) -> None:
        self.values.append(value)

    def to_dict(self) -> dict[str, Any]:
        values = sorted(self.values)
        wins = [value for value in values if value > 0]
        losses = [value for value in values if value < 0]
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        outcome_n = len(values)
        return {
            "key": self.key,
            "feature_type": self.feature_type,
            "feature": self.feature,
            "outcome_n": outcome_n,
            "win_n": len(wins),
            "loss_n": len(losses),
            "win_rate_pct": _round((len(wins) / outcome_n) * 100.0) if outcome_n else None,
            "net_jpy": _round(sum(values)) if values else 0.0,
            "avg_jpy": _round(sum(values) / outcome_n) if outcome_n else None,
            "best_jpy": _round(max(values)) if values else None,
            "worst_jpy": _round(min(values)) if values else None,
            "profit_factor": _round(gross_profit / gross_loss) if gross_loss > 0 else (None if gross_profit == 0 else "INF"),
        }


class OutcomeMartBuilder:
    """Build a read-only outcome mart from imported archive and execution truth.

    The mart is an offline feature packet. It never grants live permission,
    changes risk budgets, or blocks lanes; it gives the operator and advisory
    ranking code a condition-aware view of historical outcomes.
    """

    def __init__(
        self,
        *,
        db_path: Path = DEFAULT_HISTORY_DB,
        execution_ledger_db_path: Path = DEFAULT_EXECUTION_LEDGER_DB,
        output_path: Path = DEFAULT_OUTCOME_MART,
        report_path: Path = DEFAULT_OUTCOME_MART_REPORT,
        entry_thesis_ledger_path: Path | None = None,
    ) -> None:
        self.db_path = db_path
        self.execution_ledger_db_path = execution_ledger_db_path
        self.output_path = output_path
        self.report_path = report_path
        self.entry_thesis_ledger_path = entry_thesis_ledger_path

    def run(self) -> OutcomeMartSummary:
        if not self.db_path.exists():
            raise FileNotFoundError(f"legacy history DB not found: {self.db_path}")

        archive_outcomes = tuple(_archive_outcome_rows(self.db_path))
        story_observations = tuple(_story_observation_rows(self.db_path))
        execution_outcomes = tuple(
            _execution_ledger_rows(
                self.execution_ledger_db_path,
                entry_thesis_ledger_path=self.entry_thesis_ledger_path,
            )
        )
        context_feature_edges = _context_feature_edges(execution_outcomes)
        context_feature_outcomes = sum(1 for row in execution_outcomes if tuple(_context_features(row.context_evidence)))
        context_feature_coverage_pct = (
            _round((context_feature_outcomes / len(execution_outcomes)) * 100.0)
            if execution_outcomes
            else 0.0
        )

        method_edges: dict[tuple[str, str, str], _Bucket] = {}
        pair_edges: dict[tuple[str, str], _Bucket] = {}
        condition_edges: dict[tuple[str, str, str, str], _Bucket] = {}
        condition_rollups: dict[tuple[str, str, str, str], _Bucket] = {}
        setup_buckets: dict[tuple[str, str, str, str, str, str], _Bucket] = {}

        def method_bucket(pair: str, direction: str, method: str) -> _Bucket:
            key = (pair, direction, method)
            if key not in method_edges:
                method_edges[key] = _Bucket(pair=pair, direction=direction, method=method)
            return method_edges[key]

        def pair_bucket(pair: str, direction: str) -> _Bucket:
            key = (pair, direction)
            if key not in pair_edges:
                pair_edges[key] = _Bucket(pair=pair, direction=direction)
            return pair_edges[key]

        def condition_bucket(item: OutcomeRow | StoryObservation) -> _Bucket:
            key = (item.method, item.order_type, item.session_bucket, item.regime)
            if key not in condition_edges:
                condition_edges[key] = _Bucket(
                    pair="ALL",
                    direction="ALL",
                    method=item.method,
                    order_type=item.order_type,
                    session_bucket=item.session_bucket,
                    regime=item.regime,
                )
            return condition_edges[key]

        def condition_rollup_bucket(method: str, order_type: str, session_bucket: str, regime: str) -> _Bucket:
            key = (method, order_type, session_bucket, regime)
            if key not in condition_rollups:
                condition_rollups[key] = _Bucket(
                    pair="ALL",
                    direction="ALL",
                    method=method,
                    order_type=order_type,
                    session_bucket=session_bucket,
                    regime=regime,
                )
            return condition_rollups[key]

        def setup_bucket(item: OutcomeRow | StoryObservation) -> _Bucket:
            key = (item.pair, item.direction, item.method, item.order_type, item.session_bucket, item.regime)
            if key not in setup_buckets:
                setup_buckets[key] = _Bucket(
                    pair=item.pair,
                    direction=item.direction,
                    method=item.method,
                    order_type=item.order_type,
                    session_bucket=item.session_bucket,
                    regime=item.regime,
                )
            return setup_buckets[key]

        for row in (*archive_outcomes, *execution_outcomes):
            condition_bucket(row).add_outcome(row.pl_jpy, source=row.source)
            for key in _condition_rollup_keys(row):
                condition_rollup_bucket(*key).add_outcome(row.pl_jpy, source=row.source)
            method_bucket(row.pair, row.direction, row.method).add_outcome(row.pl_jpy, source=row.source)
            pair_bucket(row.pair, row.direction).add_outcome(row.pl_jpy, source=row.source)
            setup_bucket(row).add_outcome(row.pl_jpy, source=row.source)

        for observation in story_observations:
            condition_bucket(observation).add_observation()
            for key in _condition_rollup_keys(observation):
                condition_rollup_bucket(*key).add_observation()
            method_bucket(observation.pair, observation.direction, observation.method).add_observation()
            pair_bucket(observation.pair, observation.direction).add_observation()
            setup_bucket(observation).add_observation()

        generated_at = datetime.now(timezone.utc).isoformat()
        condition_validation = _walk_forward_condition_validation((*archive_outcomes, *execution_outcomes))
        payload = {
            "generated_at_utc": generated_at,
            "read_only": True,
            "live_permission": False,
            "history_db": str(self.db_path),
            "execution_ledger_db": str(self.execution_ledger_db_path) if self.execution_ledger_db_path.exists() else None,
            "source_coverage": {
                "archive_outcomes": len(archive_outcomes),
                "execution_ledger_outcomes": len(execution_outcomes),
                "story_observations": len(story_observations),
                "context_feature_outcomes": context_feature_outcomes,
                "context_feature_coverage_pct": context_feature_coverage_pct,
            },
            "condition_edges": _sorted_bucket_dicts(condition_edges.values()),
            "condition_rollups": _sorted_bucket_dicts(condition_rollups.values()),
            "condition_validation": condition_validation,
            "context_feature_edges": _sorted_context_feature_dicts(context_feature_edges.values()),
            "pair_direction_edges": _sorted_bucket_dicts(pair_edges.values()),
            "method_edges": _sorted_bucket_dicts(method_edges.values()),
            "setup_buckets": _sorted_bucket_dicts(setup_buckets.values()),
            "contract": {
                "purpose": "read-only archive condition/outcome features for lane ranking and review",
                "does_not": [
                    "grant live permission",
                    "resize risk",
                    "bypass RiskEngine",
                    "override broker truth",
                ],
            },
        }
        status = "OUTCOME_MART_READY" if archive_outcomes or execution_outcomes else "NO_OUTCOMES"
        self._write_output(payload)
        self._write_report(payload, status=status)
        return OutcomeMartSummary(
            output_path=self.output_path,
            report_path=self.report_path,
            status=status,
            archive_outcomes=len(archive_outcomes),
            execution_ledger_outcomes=len(execution_outcomes),
            story_observations=len(story_observations),
            condition_edges=len(condition_edges),
            condition_rollups=len(condition_rollups),
            validated_condition_outcomes=int(condition_validation["validated_outcomes"]),
            condition_directional_hit_rate_pct=condition_validation["directional_hit_rate_pct"],
            method_edges=len(method_edges),
            setup_buckets=len(setup_buckets),
            context_feature_edges=len(context_feature_edges),
            context_feature_outcomes=context_feature_outcomes,
            context_feature_coverage_pct=context_feature_coverage_pct,
        )

    def _write_output(self, payload: dict[str, Any]) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    def _write_report(self, payload: dict[str, Any], *, status: str) -> None:
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        coverage = payload["source_coverage"]
        lines = [
            "# Outcome Mart Report",
            "",
            f"- Generated at UTC: `{payload['generated_at_utc']}`",
            f"- Status: `{status}`",
            f"- History DB: `{payload['history_db']}`",
            f"- Execution ledger DB: `{payload['execution_ledger_db']}`",
            f"- Archive outcomes: `{coverage['archive_outcomes']}`",
            f"- Execution ledger outcomes: `{coverage['execution_ledger_outcomes']}`",
            f"- Story observations: `{coverage['story_observations']}`",
            f"- Context-feature outcomes: `{coverage['context_feature_outcomes']}` (`{coverage['context_feature_coverage_pct']:.1f}%` of execution outcomes)",
            "",
            f"## Winning Conditions (>= {REPORT_MIN_CONDITION_OUTCOMES} outcomes)",
            "",
            "| Condition | Outcomes | Observations | Net JPY | Avg JPY | Win % | Worst | Best |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        winning_conditions = [
            item
            for item in payload["condition_edges"]
            if (item.get("net_jpy") or 0) > 0 and item["outcome_n"] >= REPORT_MIN_CONDITION_OUTCOMES
        ]
        for item in winning_conditions[:REPORT_ROW_LIMIT]:
            lines.append(_edge_table_row(item))
        if not winning_conditions:
            lines.append("| none | 0 | 0 | 0 |  |  |  |  |")

        negative_conditions = [
            item
            for item in payload["condition_edges"]
            if (item.get("net_jpy") or 0) < 0 and item["outcome_n"] >= REPORT_MIN_CONDITION_OUTCOMES
        ]
        lines.extend(
            [
                "",
                f"## Losing Conditions (>= {REPORT_MIN_CONDITION_OUTCOMES} outcomes)",
                "",
                "| Condition | Outcomes | Observations | Net JPY | Avg JPY | Win % | Worst | Best |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for item in sorted(negative_conditions, key=lambda value: (value["net_jpy"], -value["outcome_n"]))[:REPORT_ROW_LIMIT]:
            lines.append(_edge_table_row(item))
        if not negative_conditions:
            lines.append("| none | 0 | 0 | 0 |  |  |  |  |")

        validation = payload["condition_validation"]
        positive_validation = validation["predicted_positive"]
        negative_validation = validation["predicted_negative"]
        lines.extend(
            [
                "",
                "## Walk-Forward Condition Validation",
                "",
                "- Uses only prior outcomes at each historical point; current/future rows are not visible to the edge being tested.",
                f"- Prior sample minimum: `{validation['min_prior_outcomes']}` outcomes",
                f"- Eligible outcomes: `{validation['eligible_outcomes']}`",
                f"- Validated outcomes: `{validation['validated_outcomes']}` (`{validation['coverage_pct']:.1f}%` coverage)",
                f"- Directional hit rate: `{_format_optional_pct(validation['directional_hit_rate_pct'])}`",
                (
                    f"- Positive prior edge: `{positive_validation['outcomes']}` outcomes, "
                    f"actual net `{positive_validation['actual_net_jpy']:.1f}` JPY, "
                    f"actual win `{_format_optional_pct(positive_validation['actual_win_rate_pct'])}`"
                ),
                (
                    f"- Negative prior edge: `{negative_validation['outcomes']}` outcomes, "
                    f"actual net `{negative_validation['actual_net_jpy']:.1f}` JPY, "
                    f"actual win `{_format_optional_pct(negative_validation['actual_win_rate_pct'])}`"
                ),
                f"- Exact condition matches: `{validation['exact_match_outcomes']}`",
                f"- Rolled-up condition matches: `{validation['rollup_match_outcomes']}`",
            ]
        )
        matched_edges = validation["matched_edges"]
        validated_positive_edges = [
            item
            for item in matched_edges
            if item["predicted_edge"] == "POSITIVE" and item["actual_net_jpy"] > 0 and item["outcomes"] >= REPORT_MIN_CONDITION_OUTCOMES
        ]
        false_positive_edges = [
            item
            for item in matched_edges
            if item["predicted_edge"] == "POSITIVE" and item["actual_net_jpy"] < 0 and item["outcomes"] >= REPORT_MIN_CONDITION_OUTCOMES
        ]
        validated_negative_edges = [
            item
            for item in matched_edges
            if item["predicted_edge"] == "NEGATIVE" and item["actual_net_jpy"] < 0 and item["outcomes"] >= REPORT_MIN_CONDITION_OUTCOMES
        ]
        for title, rows in (
            ("Validated Positive Conditions", validated_positive_edges),
            ("False Positive Conditions", sorted(false_positive_edges, key=lambda item: (item["actual_net_jpy"], -item["outcomes"]))),
            ("Validated Negative Conditions", sorted(validated_negative_edges, key=lambda item: (item["actual_net_jpy"], -item["outcomes"]))),
        ):
            lines.extend(
                [
                    "",
                    f"### {title}",
                    "",
                    "| Condition | Predicted | Outcomes | Actual Net JPY | Hit % | Actual Win % | Scope |",
                    "|---|---|---:|---:|---:|---:|---|",
                ]
            )
            for item in rows[:REPORT_ROW_LIMIT]:
                lines.append(_validation_table_row(item))
            if not rows:
                lines.append("| none |  | 0 | 0 |  |  |  |")

        winning_rollups = [
            item
            for item in payload["condition_rollups"]
            if (item.get("net_jpy") or 0) > 0 and item["outcome_n"] >= REPORT_MIN_CONDITION_OUTCOMES
        ]
        losing_rollups = [
            item
            for item in payload["condition_rollups"]
            if (item.get("net_jpy") or 0) < 0 and item["outcome_n"] >= REPORT_MIN_CONDITION_OUTCOMES
        ]
        lines.extend(
            [
                "",
                f"## Winning Condition Rollups (>= {REPORT_MIN_CONDITION_OUTCOMES} outcomes)",
                "",
                "| Condition | Outcomes | Observations | Net JPY | Avg JPY | Win % | Worst | Best |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for item in winning_rollups[:REPORT_ROW_LIMIT]:
            lines.append(_edge_table_row(item))
        if not winning_rollups:
            lines.append("| none | 0 | 0 | 0 |  |  |  |  |")

        lines.extend(
            [
                "",
                f"## Losing Condition Rollups (>= {REPORT_MIN_CONDITION_OUTCOMES} outcomes)",
                "",
                "| Condition | Outcomes | Observations | Net JPY | Avg JPY | Win % | Worst | Best |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for item in sorted(losing_rollups, key=lambda value: (value["net_jpy"], -value["outcome_n"]))[:REPORT_ROW_LIMIT]:
            lines.append(_edge_table_row(item))
        if not losing_rollups:
            lines.append("| none | 0 | 0 | 0 |  |  |  |  |")

        lines.extend(
            [
                "",
                "## Pair/Method Drilldown",
                "",
                "| Key | Outcomes | Observations | Net JPY | Avg JPY | Win % | Worst | Best |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for item in payload["method_edges"][:REPORT_ROW_LIMIT]:
            lines.append(_edge_table_row(item))
        if not payload["method_edges"]:
            lines.append("| none | 0 | 0 | 0 |  |  |  |  |")

        lines.extend(
            [
                "",
                "## Context Feature Edges",
                "",
                "| Feature | Outcomes | Net JPY | Avg JPY | Win % | Worst | Best |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for item in payload["context_feature_edges"][:REPORT_ROW_LIMIT]:
            lines.append(_context_feature_table_row(item))
        if not payload["context_feature_edges"]:
            lines.append("| none | 0 | 0 |  |  |  |  |")

        lines.extend(
            [
                "",
                "## Contract",
                "",
                "- This mart is read-only archive condition evidence for ranking and review.",
                "- It never places, stages, resizes, or authorizes broker orders.",
                "- Current broker truth, RiskEngine, strategy-profile validation, and gateways remain authoritative.",
                "- Story observations without P/L increase coverage counts only; they do not create expectancy.",
                "- Pair/method drilldown is secondary; the primary question is which conditions paid or failed.",
                "- Context-feature edges are post-trade attribution only; they do not grant live permission or bypass current receipts.",
            ]
        )
        self.report_path.write_text("\n".join(lines) + "\n")


def _archive_outcome_rows(db_path: Path) -> Iterator[OutcomeRow]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        for row in conn.execute(
            """
            SELECT source_table, session_date, pair, direction, pl, execution_style, allocation_band, thesis, raw_json
            FROM legacy_records
            WHERE pair IS NOT NULL AND direction IS NOT NULL AND pl IS NOT NULL
            """
        ):
            raw = _load_json(row["raw_json"])
            pair = _norm(row["pair"])
            direction = _norm(row["direction"])
            if not pair or not direction:
                continue
            value = _float(row["pl"])
            if value is None:
                continue
            yield OutcomeRow(
                source=str(row["source_table"] or "archive"),
                pair=pair,
                direction=direction,
                method=_method_family(raw, fallback_text=row["thesis"]),
                order_type=_order_type(raw, row["execution_style"]),
                session_bucket=_session_bucket(raw),
                regime=_regime(raw),
                pl_jpy=value,
                observed_at_utc=_observed_at_utc(raw, row["session_date"]),
            )


def _story_observation_rows(db_path: Path) -> Iterator[StoryObservation]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        for row in conn.execute("SELECT source_name, raw_json FROM jsonl_events WHERE source_name='s_hunt_ledger'"):
            payload = _load_json(row["raw_json"])
            for item in _story_items(payload):
                pair = _norm(item.get("pair"))
                direction = _norm(item.get("direction"))
                if not pair or not direction:
                    continue
                yield StoryObservation(
                    pair=pair,
                    direction=direction,
                    method=_method_family(item, fallback_text=item.get("raw")),
                    order_type=_order_type(item, item.get("orderability") or item.get("upgrade_action")),
                    session_bucket=_session_bucket(payload),
                    regime=_regime(item),
                )


def _execution_ledger_rows(
    db_path: Path,
    *,
    entry_thesis_ledger_path: Path | None = None,
) -> Iterator[OutcomeRow]:
    if not db_path.exists():
        return
    thesis_context = _load_entry_thesis_context(entry_thesis_ledger_path or db_path.parent / "entry_thesis_ledger.jsonl")
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='execution_events'"
        ).fetchone()
        if not exists:
            return
        for row in conn.execute(
            """
            WITH entries AS (
                SELECT
                    trade_id,
                    CASE
                        WHEN MAX(units) > 0 THEN 'LONG'
                        WHEN MIN(units) < 0 THEN 'SHORT'
                        ELSE NULL
                    END AS position_side
                FROM execution_events
                WHERE event_type = 'ORDER_FILLED'
                  AND trade_id IS NOT NULL
                  AND trade_id != ''
                  AND units IS NOT NULL
                GROUP BY trade_id
            ),
            gateway_contexts AS (
                SELECT trade_id, MAX(raw_json) AS gateway_raw_json
                FROM execution_events
                WHERE event_type = 'GATEWAY_ORDER_SENT'
                  AND trade_id IS NOT NULL
                  AND trade_id != ''
                GROUP BY trade_id
            )
            SELECT
                e.ts_utc,
                e.lane_id,
                e.trade_id,
                e.pair,
                e.side AS close_side,
                e.realized_pl_jpy,
                e.raw_json,
                entries.position_side,
                gateway_contexts.gateway_raw_json
            FROM execution_events e
            LEFT JOIN entries ON entries.trade_id = e.trade_id
            LEFT JOIN gateway_contexts ON gateway_contexts.trade_id = e.trade_id
            WHERE e.event_type IN ('TRADE_CLOSED', 'TRADE_REDUCED')
              AND e.pair IS NOT NULL
              AND e.side IS NOT NULL
              AND e.realized_pl_jpy IS NOT NULL
            """
        ):
            raw = _load_json(row["raw_json"])
            pair = _norm(row["pair"])
            direction = _norm(row["position_side"]) or _opposite_close_side(_norm(row["close_side"]))
            value = _float(row["realized_pl_jpy"])
            if not pair or not direction or value is None:
                continue
            method = _method_from_lane_id(row["lane_id"]) or _method_family(raw)
            trade_id = str(row["trade_id"] or "")
            context_evidence = (
                thesis_context.get(trade_id)
                or _context_evidence_from_raw(_load_json(row["gateway_raw_json"]))
                or _context_evidence_from_raw(raw)
            )
            yield OutcomeRow(
                source="execution_ledger",
                pair=pair,
                direction=direction,
                method=method,
                order_type=_order_type(raw, None),
                session_bucket=_session_bucket({"created_at": row["ts_utc"]}),
                regime=_regime(raw),
                pl_jpy=value,
                observed_at_utc=_parse_datetime(row["ts_utc"]),
                context_evidence=context_evidence,
            )


def _story_items(payload: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for key in ("horizons", "s_excavation_pairs", "s_excavation_podium"):
        values = payload.get(key)
        if isinstance(values, list):
            for item in values:
                if isinstance(item, dict):
                    yield item


def _load_entry_thesis_context(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    contexts: dict[str, dict[str, Any]] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return contexts
    for line in lines:
        payload = _load_json(line)
        trade_id = str(payload.get("trade_id") or "")
        context = _context_dict(payload.get("context_evidence"))
        if trade_id and context:
            contexts[trade_id] = context
    return contexts


def _context_evidence_from_raw(payload: dict[str, Any]) -> dict[str, Any]:
    context = _context_dict(payload.get("context_evidence"))
    if context:
        return context
    record = payload.get("entry_thesis_record")
    if isinstance(record, dict):
        for key in ("thesis", "pending"):
            thesis = record.get(key)
            if isinstance(thesis, dict):
                context = _context_dict(thesis.get("context_evidence"))
                if context:
                    return context
    return {}


def _context_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _context_feature_edges(rows: Iterable[OutcomeRow]) -> dict[tuple[str, str], _ContextFeatureBucket]:
    buckets: dict[tuple[str, str], _ContextFeatureBucket] = {}
    for row in rows:
        for feature_type, feature in _context_features(row.context_evidence):
            key = (feature_type, feature)
            bucket = buckets.setdefault(key, _ContextFeatureBucket(feature_type=feature_type, feature=feature))
            bucket.add_outcome(row.pl_jpy)
    return buckets


def _context_features(evidence: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    if not isinstance(evidence, dict) or not evidence:
        return ()
    features: list[tuple[str, str]] = []

    matrix_ref = _context_feature_text(evidence.get("market_context_matrix_ref"))
    if matrix_ref:
        features.append(("matrix_ref", matrix_ref))

    for key, feature_type in (
        ("matrix_support_layers", "matrix_support_layer"),
        ("matrix_reject_layers", "matrix_reject_layer"),
        ("matrix_warning_layers", "matrix_warning_layer"),
    ):
        for value in _as_context_texts(evidence.get(key)):
            features.append((feature_type, value))

    for ref in _as_context_texts(evidence.get("context_asset_refs")):
        if ref.startswith("context_asset:"):
            features.append(("context_asset_ref", ref))
        elif ref.startswith("cross:"):
            features.append(("cross_asset_ref", ref))

    for symbol in _as_context_texts(evidence.get("context_asset_symbols")):
        features.append(("context_asset", symbol))

    for key in ("news_context", "forecast_news_context"):
        for value in _as_context_texts(evidence.get(key)):
            features.append(("news_context", _news_feature(value)))

    return tuple(dict.fromkeys(features))


def _as_context_texts(value: Any) -> list[str]:
    raw_items = value if isinstance(value, list) else [value]
    out: list[str] = []
    for item in raw_items:
        text = _context_feature_text(item)
        if text and text not in out:
            out.append(text)
    return out


def _context_feature_text(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return text[:CONTEXT_FEATURE_TEXT_LIMIT]


def _news_feature(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9_]+", "_", text.lower())
    for token in (
        "news_theme_followthrough",
        "macro_event",
        "us_employment",
        "calendar",
        "event_risk",
        "catalyst",
    ):
        if token in normalized:
            return token
    return text


def _sorted_bucket_dicts(values: Iterable[_Bucket]) -> list[dict[str, Any]]:
    rows = [bucket.to_dict() for bucket in values]
    return sorted(rows, key=lambda item: (item["net_jpy"], item["outcome_n"], item["story_observation_n"], item["key"]), reverse=True)


def _sorted_validation_match_dicts(values: Iterable[_ValidationMatchStats]) -> list[dict[str, Any]]:
    rows = [stats.to_dict() for stats in values]
    return sorted(rows, key=lambda item: (abs(item["actual_net_jpy"]), item["outcomes"], item["key"]), reverse=True)


def _sorted_context_feature_dicts(values: Iterable[_ContextFeatureBucket]) -> list[dict[str, Any]]:
    rows = [bucket.to_dict() for bucket in values]
    return sorted(rows, key=lambda item: (item["net_jpy"], item["outcome_n"], item["key"]), reverse=True)


def _edge_table_row(item: dict[str, Any]) -> str:
    avg = "" if item["avg_jpy"] is None else f"{item['avg_jpy']:.1f}"
    win_rate = "" if item["win_rate_pct"] is None else f"{item['win_rate_pct']:.1f}"
    worst = "" if item["worst_jpy"] is None else f"{item['worst_jpy']:.1f}"
    best = "" if item["best_jpy"] is None else f"{item['best_jpy']:.1f}"
    return (
        f"| `{item['key']}` | {item['outcome_n']} | {item['story_observation_n']} | "
        f"{item['net_jpy']:.1f} | {avg} | {win_rate} | {worst} | {best} |"
    )


def _validation_table_row(item: dict[str, Any]) -> str:
    hit_rate = _format_optional_pct(item["directional_hit_rate_pct"])
    win_rate = _format_optional_pct(item["actual_win_rate_pct"])
    return (
        f"| `{item['key']}` | {item['predicted_edge']} | {item['outcomes']} | "
        f"{item['actual_net_jpy']:.1f} | {hit_rate} | {win_rate} | {item['match_scope']} |"
    )


def _context_feature_table_row(item: dict[str, Any]) -> str:
    avg = "" if item["avg_jpy"] is None else f"{item['avg_jpy']:.1f}"
    win_rate = "" if item["win_rate_pct"] is None else f"{item['win_rate_pct']:.1f}"
    worst = "" if item["worst_jpy"] is None else f"{item['worst_jpy']:.1f}"
    best = "" if item["best_jpy"] is None else f"{item['best_jpy']:.1f}"
    return (
        f"| `{item['key']}` | {item['outcome_n']} | {item['net_jpy']:.1f} | "
        f"{avg} | {win_rate} | {worst} | {best} |"
    )


def _format_optional_pct(value: object) -> str:
    return "n/a" if value is None else f"{float(value):.1f}%"


def _condition_rollup_keys(item: OutcomeRow | StoryObservation) -> tuple[tuple[str, str, str, str], ...]:
    keys = (
        (item.method, item.order_type, item.session_bucket, "ALL"),
        (item.method, item.order_type, "ALL", item.regime),
        (item.method, item.order_type, "ALL", "ALL"),
        (item.method, "ALL", item.session_bucket, item.regime),
        (item.method, "ALL", item.session_bucket, "ALL"),
        (item.method, "ALL", "ALL", item.regime),
        (item.method, "ALL", "ALL", "ALL"),
    )
    return tuple(dict.fromkeys(keys))


def _condition_lookup_keys(row: OutcomeRow) -> tuple[tuple[str, str, str, str], ...]:
    keys = (
        (row.method, row.order_type, row.session_bucket, row.regime),
        *_condition_rollup_keys(row),
        (row.method, row.order_type, row.session_bucket, "UNSPECIFIED"),
        (row.method, row.order_type, "UNSPECIFIED", row.regime),
        (row.method, row.order_type, "UNSPECIFIED", "UNSPECIFIED"),
        (row.method, "UNSPECIFIED", row.session_bucket, row.regime),
        (row.method, "UNSPECIFIED", row.session_bucket, "UNSPECIFIED"),
        (row.method, "UNSPECIFIED", "UNSPECIFIED", row.regime),
        (row.method, "UNSPECIFIED", "UNSPECIFIED", "UNSPECIFIED"),
    )
    return tuple(dict.fromkeys(keys))


def _condition_training_keys(row: OutcomeRow) -> tuple[tuple[str, str, str, str], ...]:
    keys = ((row.method, row.order_type, row.session_bucket, row.regime), *_condition_rollup_keys(row))
    return tuple(dict.fromkeys(keys))


def _walk_forward_condition_validation(rows: Iterable[OutcomeRow]) -> dict[str, Any]:
    ordered = sorted(
        (row for row in rows if row.observed_at_utc is not None and row.pl_jpy != 0),
        key=lambda row: (row.observed_at_utc or datetime.min.replace(tzinfo=_UTC), row.source, row.pair, row.direction),
    )
    prior: dict[tuple[str, str, str, str], _ValidationBucket] = {}
    validated_outcomes = 0
    hit_outcomes = 0
    exact_match_outcomes = 0
    rollup_match_outcomes = 0
    positive_values: list[float] = []
    negative_values: list[float] = []
    matched_stats: dict[tuple[tuple[str, str, str, str], int], _ValidationMatchStats] = {}

    index = 0
    while index < len(ordered):
        timestamp = ordered[index].observed_at_utc
        batch: list[OutcomeRow] = []
        while index < len(ordered) and ordered[index].observed_at_utc == timestamp:
            batch.append(ordered[index])
            index += 1

        for row in batch:
            matched_key: tuple[str, str, str, str] | None = None
            matched_bucket: _ValidationBucket | None = None
            for key in _condition_lookup_keys(row):
                bucket = prior.get(key)
                if bucket is None or bucket.outcome_n < VALIDATION_MIN_PRIOR_OUTCOMES or bucket.net_jpy == 0:
                    continue
                matched_key = key
                matched_bucket = bucket
                break

            if matched_key is not None and matched_bucket is not None:
                predicted_sign = 1 if matched_bucket.net_jpy > 0 else -1
                actual_sign = 1 if row.pl_jpy > 0 else -1
                validated_outcomes += 1
                if predicted_sign == actual_sign:
                    hit_outcomes += 1
                if matched_key == (row.method, row.order_type, row.session_bucket, row.regime):
                    exact_match_outcomes += 1
                else:
                    rollup_match_outcomes += 1
                if predicted_sign > 0:
                    positive_values.append(row.pl_jpy)
                else:
                    negative_values.append(row.pl_jpy)
                stat_key = (matched_key, predicted_sign)
                stats = matched_stats.setdefault(
                    stat_key,
                    _ValidationMatchStats(key=matched_key, predicted_sign=predicted_sign),
                )
                stats.add_match(row.pl_jpy)

        for row in batch:
            for key in _condition_training_keys(row):
                bucket = prior.setdefault(key, _ValidationBucket())
                bucket.add_outcome(row.pl_jpy)

    positive_wins = [value for value in positive_values if value > 0]
    negative_wins = [value for value in negative_values if value > 0]
    coverage_pct = _round((validated_outcomes / len(ordered)) * 100.0) if ordered else 0.0
    hit_rate = _round((hit_outcomes / validated_outcomes) * 100.0) if validated_outcomes else None
    return {
        "status": "CONDITION_WALK_FORWARD_READY" if validated_outcomes else "INSUFFICIENT_PRIOR_CONDITION_HISTORY",
        "min_prior_outcomes": VALIDATION_MIN_PRIOR_OUTCOMES,
        "eligible_outcomes": len(ordered),
        "validated_outcomes": validated_outcomes,
        "coverage_pct": coverage_pct,
        "directional_hit_outcomes": hit_outcomes,
        "directional_hit_rate_pct": hit_rate,
        "exact_match_outcomes": exact_match_outcomes,
        "rollup_match_outcomes": rollup_match_outcomes,
        "predicted_positive": {
            "outcomes": len(positive_values),
            "actual_net_jpy": _round(sum(positive_values)) if positive_values else 0.0,
            "actual_win_rate_pct": _round((len(positive_wins) / len(positive_values)) * 100.0)
            if positive_values
            else None,
        },
        "predicted_negative": {
            "outcomes": len(negative_values),
            "actual_net_jpy": _round(sum(negative_values)) if negative_values else 0.0,
            "actual_win_rate_pct": _round((len(negative_wins) / len(negative_values)) * 100.0)
            if negative_values
            else None,
        },
        "matched_edges": _sorted_validation_match_dicts(matched_stats.values()),
    }


def _method_family(payload: dict[str, Any], fallback_text: object | None = None) -> str:
    lane_method = _method_from_lane_id(payload.get("lane_id"))
    if lane_method:
        return lane_method
    text = _payload_text(
        payload,
        (
            "method",
            "trade_method",
            "setup_method",
            "thesis_family",
            "thesis_vehicle",
            "thesis_structure",
            "setup_type",
            "type",
            "entry_type",
            "reason",
            "orderability",
            "upgrade_action",
            "thesis",
            "thesis_key",
            "thesis_market",
            "thesis_trigger",
            "notes",
            "why",
            "trigger",
            "payout_path",
            "invalidation",
            "deployment_status",
            "deployment_result",
            "execution_note",
            "lesson_from_review",
            "collapse_note",
            "live_tape_bias",
            "live_tape_mode",
            "live_tape_bucket",
            "mtf_chain",
        ),
        fallback_text=fallback_text,
    )
    normalized = text.replace("-", "_").replace("/", "_").replace(" ", "_")
    if any(
        token in normalized
        for token in (
            "breakout_failure",
            "failed_breakout",
            "false_break",
            "fakeout",
            "failure",
            "failed",
            "touch_and_fail",
            "retest_failure",
            "rejection",
            "reject",
            "trap",
            "sweep",
            "stop_run",
        )
    ):
        return "BREAKOUT_FAILURE"
    if any(
        token in normalized
        for token in (
            "range",
            "mean",
            "box",
            "floor",
            "ceiling",
            "rotation",
            "rail",
            "band",
            "lid",
            "fade",
            "reversal",
            "absorption",
            "counter_reversal",
            "swing_probe",
        )
    ):
        return "RANGE_ROTATION"
    if any(
        token in normalized
        for token in (
            "trend",
            "continuation",
            "breakout",
            "shelf",
            "retest",
            "pullback",
            "runner",
            "momentum",
            "impulse",
            "swing",
            "ema_pack",
            "direct_usd",
            "usd_strength",
            "h1_bear",
            "h1_bull",
            "bull_walk",
            "bear_squeeze",
        )
    ):
        return "TREND_CONTINUATION"
    return "UNSPECIFIED"


def _method_from_lane_id(value: object) -> str | None:
    if not value:
        return None
    parts = str(value).split(":")
    for part in reversed(parts):
        token = _norm(part)
        if token in {"TREND_CONTINUATION", "RANGE_ROTATION", "BREAKOUT_FAILURE"}:
            return token
    return None


def _order_type(payload: dict[str, Any], fallback: object | None) -> str:
    values = []
    for key in ("order_type", "entry_type", "setup_type", "reason", "orderability", "upgrade_action"):
        value = payload.get(key)
        if value:
            values.append(str(value))
    if fallback:
        values.append(str(fallback))
    text = " ".join(values).lower()
    if "limit" in text:
        return "LIMIT"
    if "stop" in text:
        return "STOP_ENTRY"
    if "market" in text or "enter now" in text:
        return "MARKET"
    return "UNSPECIFIED"


def _session_bucket(payload: dict[str, Any]) -> str:
    raw_bucket = payload.get("session_bucket")
    if raw_bucket:
        return _session_family(raw_bucket)
    hour = _hour_from_payload(payload)
    if hour is None:
        return "UNSPECIFIED"
    return _utc_session_bucket(hour)


def _hour_from_payload(payload: dict[str, Any]) -> int | None:
    direct = _int(payload.get("session_hour"))
    if direct is not None:
        return direct % 24
    for key in ("created_at", "updated_at", "timestamp_utc", "state_last_updated"):
        value = payload.get(key)
        if not value:
            continue
        parsed = _parse_datetime(value)
        if parsed is None:
            continue
        return parsed.hour
    return None


def _utc_session_bucket(hour: int) -> str:
    # Descriptive UTC groupings for archive analysis only. They approximate
    # common FX liquidity windows and must not become live spread/risk gates.
    if 0 <= hour < 7:
        return "ASIA"
    if 7 <= hour < 13:
        return "LONDON"
    if 13 <= hour < 21:
        return "NY"
    return "ROLLOVER"


def _regime(payload: dict[str, Any]) -> str:
    for key in ("regime", "regime_snapshot", "live_tape_state", "h1_trend", "m5_trend"):
        value = payload.get(key)
        if value:
            return _regime_family(value)
    text = _payload_text(
        payload,
        (
            "thesis_family",
            "thesis_structure",
            "thesis_key",
            "thesis_market",
            "live_tape_bias",
            "live_tape_mode",
            "live_tape_bucket",
            "mtf_chain",
            "why",
            "notes",
            "trigger",
            "payout_path",
            "invalidation",
        ),
    )
    if text:
        regime = _regime_family(text)
        if regime in {"SQUEEZE", "RANGE", "QUIET", "TRENDING", "TRANSITION"}:
            return regime
    return "UNSPECIFIED"


def _session_family(value: object) -> str:
    text = _norm(value).replace("-", "_")
    if not text:
        return "UNSPECIFIED"
    if "LONDON" in text:
        return "LONDON"
    if text in {"NY", "NEWYORK", "NEW_YORK"} or text.startswith("NY_") or "SILVER_BULLET" in text:
        return "NY"
    if text in {"TOKYO", "ASIA"} or "TOKYO" in text or "ASIA" in text:
        return "ASIA"
    if "ROLLOVER" in text or "OFF_HOURS" in text:
        return "ROLLOVER"
    return text


def _regime_family(value: object) -> str:
    text = _norm(value).replace("-", "_")
    if not text:
        return "UNSPECIFIED"
    if "SQUEEZE" in text or "BREAKOUT_PENDING" in text:
        return "SQUEEZE"
    if "RANGE" in text or "ROTATION" in text or "MEAN_REVERT" in text:
        return "RANGE"
    if "QUIET" in text or "STABLE" in text or "THIN_LIQUIDITY" in text:
        return "QUIET"
    if "TREND" in text or "BULL" in text or "BEAR" in text or "IMPULSE" in text:
        return "TRENDING"
    if "TRANSITION" in text or "FRICTION" in text or "HEADLINE" in text:
        return "TRANSITION"
    return text


def _evidence_state(values: list[float]) -> str:
    if not values:
        return "OBSERVED_ONLY"
    net = sum(values)
    if net > 0:
        return "POSITIVE_ARCHIVE_EDGE"
    if net < 0:
        return "NEGATIVE_ARCHIVE_EDGE"
    return "MIXED_ARCHIVE_EDGE"


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    pct = max(0.0, min(1.0, pct))
    idx = round((len(values) - 1) * pct)
    return values[int(idx)]


def _payload_text(
    payload: dict[str, Any],
    keys: Iterable[str],
    *,
    fallback_text: object | None = None,
) -> str:
    parts = []
    for key in keys:
        value = payload.get(key)
        if value:
            parts.append(str(value))
    if fallback_text:
        parts.append(str(fallback_text))
    return " ".join(parts).lower()


def _observed_at_utc(payload: dict[str, Any], session_date: object | None = None) -> datetime | None:
    session_day = _parse_datetime(session_date)
    for key in (
        "closed_at",
        "exit_time",
        "timestamp_utc",
        "state_last_updated",
        "created_at",
        "updated_at",
    ):
        parsed = _parse_datetime(payload.get(key))
        if parsed is None:
            continue
        if session_day is not None and parsed.date() != session_day.date():
            continue
        return parsed
    return session_day


def _parse_datetime(value: object) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = text.replace("Z", "+00:00").replace(" UTC", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=_UTC)
    return parsed.astimezone(_UTC)


def _load_json(text: object) -> dict[str, Any]:
    if not text:
        return {}
    try:
        payload = json.loads(str(text))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _norm(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip().upper().replace("/", "_").replace(" ", "_") or ""


def _opposite_close_side(value: str) -> str:
    if value == "LONG":
        return "SHORT"
    if value == "SHORT":
        return "LONG"
    return ""


def _float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _round(value: float) -> float:
    return round(value, 4)
