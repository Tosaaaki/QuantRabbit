from __future__ import annotations

import json
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


@dataclass(frozen=True)
class OutcomeMartSummary:
    output_path: Path
    report_path: Path
    status: str
    archive_outcomes: int
    execution_ledger_outcomes: int
    story_observations: int
    condition_edges: int
    method_edges: int
    setup_buckets: int


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
    ) -> None:
        self.db_path = db_path
        self.execution_ledger_db_path = execution_ledger_db_path
        self.output_path = output_path
        self.report_path = report_path

    def run(self) -> OutcomeMartSummary:
        if not self.db_path.exists():
            raise FileNotFoundError(f"legacy history DB not found: {self.db_path}")

        archive_outcomes = tuple(_archive_outcome_rows(self.db_path))
        story_observations = tuple(_story_observation_rows(self.db_path))
        execution_outcomes = tuple(_execution_ledger_rows(self.execution_ledger_db_path))

        method_edges: dict[tuple[str, str, str], _Bucket] = {}
        pair_edges: dict[tuple[str, str], _Bucket] = {}
        condition_edges: dict[tuple[str, str, str, str], _Bucket] = {}
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
            method_bucket(row.pair, row.direction, row.method).add_outcome(row.pl_jpy, source=row.source)
            pair_bucket(row.pair, row.direction).add_outcome(row.pl_jpy, source=row.source)
            setup_bucket(row).add_outcome(row.pl_jpy, source=row.source)

        for observation in story_observations:
            condition_bucket(observation).add_observation()
            method_bucket(observation.pair, observation.direction, observation.method).add_observation()
            pair_bucket(observation.pair, observation.direction).add_observation()
            setup_bucket(observation).add_observation()

        generated_at = datetime.now(timezone.utc).isoformat()
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
            },
            "condition_edges": _sorted_bucket_dicts(condition_edges.values()),
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
            method_edges=len(method_edges),
            setup_buckets=len(setup_buckets),
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
                "## Contract",
                "",
                "- This mart is read-only archive condition evidence for ranking and review.",
                "- It never places, stages, resizes, or authorizes broker orders.",
                "- Current broker truth, RiskEngine, strategy-profile validation, and gateways remain authoritative.",
                "- Story observations without P/L increase coverage counts only; they do not create expectancy.",
                "- Pair/method drilldown is secondary; the primary question is which conditions paid or failed.",
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


def _execution_ledger_rows(db_path: Path) -> Iterator[OutcomeRow]:
    if not db_path.exists():
        return
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='execution_events'"
        ).fetchone()
        if not exists:
            return
        for row in conn.execute(
            """
            SELECT ts_utc, lane_id, pair, side, realized_pl_jpy, raw_json
            FROM execution_events
            WHERE pair IS NOT NULL AND side IS NOT NULL AND realized_pl_jpy IS NOT NULL
            """
        ):
            raw = _load_json(row["raw_json"])
            pair = _norm(row["pair"])
            direction = _norm(row["side"])
            value = _float(row["realized_pl_jpy"])
            if not pair or not direction or value is None:
                continue
            method = _method_from_lane_id(row["lane_id"]) or _method_family(raw)
            yield OutcomeRow(
                source="execution_ledger",
                pair=pair,
                direction=direction,
                method=method,
                order_type=_order_type(raw, None),
                session_bucket=_session_bucket({"created_at": row["ts_utc"]}),
                regime=_regime(raw),
                pl_jpy=value,
            )


def _story_items(payload: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for key in ("horizons", "s_excavation_pairs", "s_excavation_podium"):
        values = payload.get(key)
        if isinstance(values, list):
            for item in values:
                if isinstance(item, dict):
                    yield item


def _sorted_bucket_dicts(values: Iterable[_Bucket]) -> list[dict[str, Any]]:
    rows = [bucket.to_dict() for bucket in values]
    return sorted(rows, key=lambda item: (item["net_jpy"], item["outcome_n"], item["story_observation_n"], item["key"]), reverse=True)


def _edge_table_row(item: dict[str, Any]) -> str:
    avg = "" if item["avg_jpy"] is None else f"{item['avg_jpy']:.1f}"
    win_rate = "" if item["win_rate_pct"] is None else f"{item['win_rate_pct']:.1f}"
    worst = "" if item["worst_jpy"] is None else f"{item['worst_jpy']:.1f}"
    best = "" if item["best_jpy"] is None else f"{item['best_jpy']:.1f}"
    return (
        f"| `{item['key']}` | {item['outcome_n']} | {item['story_observation_n']} | "
        f"{item['net_jpy']:.1f} | {avg} | {win_rate} | {worst} | {best} |"
    )


def _method_family(payload: dict[str, Any], fallback_text: object | None = None) -> str:
    lane_method = _method_from_lane_id(payload.get("lane_id"))
    if lane_method:
        return lane_method
    parts = []
    for key in (
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
    ):
        value = payload.get(key)
        if value:
            parts.append(str(value))
    if fallback_text:
        parts.append(str(fallback_text))
    text = " ".join(parts).lower()
    if any(token in text for token in ("range", "mean", "box", "floor", "ceiling", "rotation")):
        return "RANGE_ROTATION"
    if any(token in text for token in ("breakout failure", "failed breakout", "failure", "failed", "fade", "rejection")):
        return "BREAKOUT_FAILURE"
    if any(token in text for token in ("trend", "continuation", "breakout", "shelf", "retest")):
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
        return _norm(raw_bucket)
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
        text = str(value).replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
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
            return _norm(value)
    return "UNSPECIFIED"


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
