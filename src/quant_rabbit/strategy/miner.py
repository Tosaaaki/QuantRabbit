from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quant_rabbit.paths import DEFAULT_HISTORY_DB, DEFAULT_STRATEGY_PROFILE, DEFAULT_STRATEGY_REPORT


@dataclass
class PairDirectionProfile:
    pair: str
    direction: str
    pretrade_n: int = 0
    pretrade_net_jpy: float = 0.0
    pretrade_avg_jpy: float = 0.0
    live_n: int = 0
    live_net_jpy: float = 0.0
    live_avg_jpy: float = 0.0
    live_worst_jpy: float | None = None
    positive_evidence_n: int = 0
    positive_best_jpy: float = 0.0
    positive_tail_jpy: float = 0.0
    target_reward_risk: float = 1.5
    seat_discovered: int = 0
    seat_orderable: int = 0
    seat_deployed: int = 0
    seat_captured: int = 0
    seat_missed: int = 0
    seat_directionally_correct: int = 0
    order_blocked: int = 0
    top_block_reasons: list[str] = field(default_factory=list)
    status: str = "WATCH_ONLY"
    required_fix: str = ""

    @property
    def key(self) -> str:
        return f"{self.pair} {self.direction}"


@dataclass(frozen=True)
class StrategyMiningSummary:
    db_path: Path
    report_path: Path
    profile_path: Path
    profiles: int
    blocked: int
    candidates: int
    risk_repair_candidates: int
    mined_missed_edges: int


class StrategyMiner:
    def __init__(
        self,
        db_path: Path = DEFAULT_HISTORY_DB,
        report_path: Path = DEFAULT_STRATEGY_REPORT,
        profile_path: Path = DEFAULT_STRATEGY_PROFILE,
    ) -> None:
        self.db_path = db_path
        self.report_path = report_path
        self.profile_path = profile_path

    def run(self) -> StrategyMiningSummary:
        if not self.db_path.exists():
            raise FileNotFoundError(f"legacy history DB not found: {self.db_path}")
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            profiles = self._build_profiles(conn)
            coverage = self._coverage(conn)
            generated_at = datetime.now(timezone.utc).isoformat()
            self._write_profile(profiles, coverage, generated_at)
            self._write_report(profiles, coverage, generated_at)
        return StrategyMiningSummary(
            db_path=self.db_path,
            report_path=self.report_path,
            profile_path=self.profile_path,
            profiles=len(profiles),
            blocked=sum(1 for item in profiles if item.status == "BLOCK_UNTIL_NEW_EVIDENCE"),
            candidates=sum(1 for item in profiles if item.status == "CANDIDATE"),
            risk_repair_candidates=sum(1 for item in profiles if item.status == "RISK_REPAIR_CANDIDATE"),
            mined_missed_edges=sum(1 for item in profiles if item.status == "MINE_MISSED_EDGE"),
        )

    def _build_profiles(self, conn: sqlite3.Connection) -> list[PairDirectionProfile]:
        profiles: dict[tuple[str, str], PairDirectionProfile] = {}

        def profile(pair: str, direction: str) -> PairDirectionProfile:
            key = (pair, direction)
            if key not in profiles:
                profiles[key] = PairDirectionProfile(pair=pair, direction=direction)
            return profiles[key]

        for row in conn.execute(
            """
            SELECT pair, direction, COUNT(*) n, SUM(pl) net_jpy, AVG(pl) avg_jpy
            FROM legacy_records
            WHERE source_table='pretrade_outcomes' AND pl IS NOT NULL
              AND pair IS NOT NULL AND direction IS NOT NULL
            GROUP BY pair, direction
            """
        ):
            item = profile(row["pair"], row["direction"])
            item.pretrade_n = int(row["n"])
            item.pretrade_net_jpy = float(row["net_jpy"] or 0.0)
            item.pretrade_avg_jpy = float(row["avg_jpy"] or 0.0)

        for row in conn.execute(
            """
            SELECT pair, direction, COUNT(*) n, SUM(pl_jpy) net_jpy, AVG(pl_jpy) avg_jpy, MIN(pl_jpy) worst_jpy
            FROM live_trade_events
            WHERE pl_jpy IS NOT NULL AND pair IS NOT NULL AND direction IS NOT NULL
            GROUP BY pair, direction
            """
        ):
            item = profile(row["pair"], row["direction"])
            item.live_n = int(row["n"])
            item.live_net_jpy = float(row["net_jpy"] or 0.0)
            item.live_avg_jpy = float(row["avg_jpy"] or 0.0)
            item.live_worst_jpy = float(row["worst_jpy"]) if row["worst_jpy"] is not None else None

        for row in conn.execute(
            "SELECT pair, direction, raw_json FROM legacy_records WHERE source_table='seat_outcomes'"
        ):
            pair = row["pair"]
            direction = row["direction"]
            if not pair or not direction:
                continue
            payload = _load_json(row["raw_json"])
            item = profile(pair, direction)
            item.seat_discovered += int(bool(payload.get("discovered")))
            item.seat_orderable += int(bool(payload.get("orderable")))
            item.seat_deployed += int(bool(payload.get("deployed")))
            item.seat_captured += int(bool(payload.get("captured")))
            item.seat_missed += int(bool(payload.get("missed")))
            item.seat_directionally_correct += int(bool(payload.get("directionally_correct")))

        block_reasons: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
        for row in conn.execute("SELECT event_type, pair, direction, raw_json FROM jsonl_events"):
            if row["event_type"] != "order_blocked":
                continue
            payload = _load_json(row["raw_json"])
            pair = row["pair"] or payload.get("pair")
            direction = row["direction"] or payload.get("direction")
            if not pair or not direction:
                continue
            reason = str(payload.get("reason") or "unknown block")
            item = profile(str(pair), str(direction))
            item.order_blocked += 1
            block_reasons[(str(pair), str(direction))][reason] += 1

        for row in conn.execute(
            """
            SELECT pair, direction, reason, COUNT(*) n
            FROM live_trade_events
            WHERE action='ORDER_REJECT' AND pair IS NOT NULL AND direction IS NOT NULL
            GROUP BY pair, direction, reason
            """
        ):
            item = profile(row["pair"], row["direction"])
            item.order_blocked += int(row["n"])
            block_reasons[(row["pair"], row["direction"])][str(row["reason"] or "unknown reject")] += int(row["n"])

        for key, reasons in block_reasons.items():
            profiles[key].top_block_reasons = [f"{count}x {reason}" for reason, count in reasons.most_common(3)]

        for key, values in _positive_evidence(conn).items():
            item = profile(*key)
            positives = sorted(values)
            item.positive_evidence_n = len(positives)
            item.positive_best_jpy = _round(max(positives)) if positives else 0.0
            item.positive_tail_jpy = _round(_percentile(positives, 0.90)) if positives else 0.0
            item.target_reward_risk = _target_reward_risk(item.positive_best_jpy, item.positive_tail_jpy)

        for item in profiles.values():
            self._classify(item)
        return sorted(profiles.values(), key=_profile_sort_key)

    def _classify(self, item: PairDirectionProfile) -> None:
        worst_loss = item.live_worst_jpy if item.live_worst_jpy is not None else 0.0
        positive_pretrade = item.pretrade_n >= 5 and item.pretrade_net_jpy > 0
        positive_live = item.live_n > 0 and item.live_net_jpy > 0
        missed_pressure = item.seat_missed >= 2 and item.seat_directionally_correct > item.seat_captured
        if worst_loss <= -500.0 and positive_pretrade and positive_live:
            item.status = "RISK_REPAIR_CANDIDATE"
            item.required_fix = "edge exists but old sizing broke the loss cap; require <=500 JPY dry-run receipt before live use"
            return
        if missed_pressure and not (item.live_n >= 3 and item.live_net_jpy < 0 and item.pretrade_net_jpy <= 0):
            item.status = "MINE_MISSED_EDGE"
            item.required_fix = "missed seats paid more often than captured; build trigger/pending-entry receipts before live execution"
            if worst_loss <= -500.0:
                item.required_fix += "; every receipt must be risk-resized under the 500 JPY cap"
            return
        if worst_loss <= -500.0:
            item.status = "BLOCK_UNTIL_NEW_EVIDENCE"
            item.required_fix = "historical live loss exceeded the 500 JPY cap; only risk-resized dry-run receipts can reopen it"
            return
        if item.live_n >= 3 and item.live_net_jpy < 0 and item.pretrade_net_jpy <= 0:
            item.status = "BLOCK_UNTIL_NEW_EVIDENCE"
            item.required_fix = "both live execution and pretrade feedback are negative; require a new vehicle or market-structure proof"
            return
        if positive_pretrade and item.live_net_jpy >= 0:
            item.status = "CANDIDATE"
            item.required_fix = "eligible for dry-run order-intent generation, still behind risk gateway"
            return
        item.status = "WATCH_ONLY"
        item.required_fix = "insufficient or mixed evidence; can be observed but not promoted to live execution"

    def _coverage(self, conn: sqlite3.Connection) -> dict[str, Any]:
        legacy_rows = {
            row["source_table"]: row["n"]
            for row in conn.execute(
                "SELECT source_table, COUNT(*) n FROM legacy_records GROUP BY source_table ORDER BY source_table"
            )
        }
        live_actions = {
            row["action"]: row["n"]
            for row in conn.execute(
                "SELECT COALESCE(NULLIF(action, ''), '(empty)') action, COUNT(*) n FROM live_trade_events GROUP BY action"
            )
        }
        source_files = conn.execute("SELECT COUNT(*) FROM source_files").fetchone()[0]
        jsonl_sources = {
            row["source_name"]: row["n"]
            for row in conn.execute("SELECT source_name, COUNT(*) n FROM jsonl_events GROUP BY source_name")
        }
        return {
            "source_files": source_files,
            "legacy_rows": legacy_rows,
            "live_actions": live_actions,
            "jsonl_sources": jsonl_sources,
        }

    def _write_profile(
        self,
        profiles: list[PairDirectionProfile],
        coverage: dict[str, Any],
        generated_at: str,
    ) -> None:
        self.profile_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at_utc": generated_at,
            "history_db": str(self.db_path),
            "coverage": coverage,
            "profiles": [asdict(item) for item in profiles],
            "system_contract": {
                "live_execution": "disabled until RiskEngine and strategy profile both pass",
                "loss_cap_jpy": 500,
                "minimum_reward_risk": 1.2,
                "blocked_status": "BLOCK_UNTIL_NEW_EVIDENCE",
                "risk_repair_status": "RISK_REPAIR_CANDIDATE",
            },
        }
        self.profile_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    def _write_report(
        self,
        profiles: list[PairDirectionProfile],
        coverage: dict[str, Any],
        generated_at: str,
    ) -> None:
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        candidates = [item for item in profiles if item.status == "CANDIDATE"]
        risk_repair = [item for item in profiles if item.status == "RISK_REPAIR_CANDIDATE"]
        blocked = [item for item in profiles if item.status == "BLOCK_UNTIL_NEW_EVIDENCE"]
        missed = [item for item in profiles if item.status == "MINE_MISSED_EDGE"]

        lines = [
            "# Strategy Mining Report",
            "",
            f"- Generated at UTC: `{generated_at}`",
            f"- History DB: `{self.db_path}`",
            f"- Strategy profile JSON: `{self.profile_path}`",
            "",
            "## Evidence Coverage",
            "",
            f"- Source files indexed: `{coverage['source_files']}`",
        ]
        for table, count in coverage["legacy_rows"].items():
            lines.append(f"- `{table}` rows: `{count}`")
        for source, count in coverage["jsonl_sources"].items():
            lines.append(f"- `{source}` events: `{count}`")

        lines.extend(["", "## Candidate Edges", ""])
        for item in candidates[:12]:
            lines.append(_profile_line(item))
        if not candidates:
            lines.append("- None. No pair/direction passed the evidence gate.")

        lines.extend(["", "## Risk-Repair Candidates", ""])
        for item in risk_repair[:12]:
            lines.append(_profile_line(item))
        if not risk_repair:
            lines.append("- None.")

        lines.extend(["", "## Mine Missed Edges Before Live Use", ""])
        for item in missed[:12]:
            lines.append(_profile_line(item))
        if not missed:
            lines.append("- None.")

        lines.extend(["", "## Blocked Until New Evidence", ""])
        for item in blocked[:16]:
            lines.append(_profile_line(item))
            for reason in item.top_block_reasons:
                lines.append(f"  - block reason: {reason}")
        if not blocked:
            lines.append("- None.")

        lines.extend(
            [
                "",
                "## Generated System Rules",
                "",
                "- A strategy candidate is not an order; it becomes an order intent only after current tape supplies entry, TP, SL, and thesis.",
                "- Any pair/direction with a historical live loss worse than -500 JPY needs fresh risk-resized dry-run receipts before live use, even when expectancy is positive.",
                "- Missed directional seats are not chased at market; they require trigger or pending-entry receipts.",
                "- Mixed or weak evidence remains watch-only even if the latest prompt wants action.",
            ]
        )
        self.report_path.write_text("\n".join(lines) + "\n")


def _profile_sort_key(item: PairDirectionProfile) -> tuple[int, float, float, str]:
    rank = {
        "CANDIDATE": 0,
        "RISK_REPAIR_CANDIDATE": 1,
        "MINE_MISSED_EDGE": 2,
        "WATCH_ONLY": 3,
        "BLOCK_UNTIL_NEW_EVIDENCE": 4,
    }.get(item.status, 9)
    return (rank, -item.pretrade_net_jpy, item.live_net_jpy, item.key)


def _profile_line(item: PairDirectionProfile) -> str:
    worst = "n/a" if item.live_worst_jpy is None else f"{item.live_worst_jpy:.1f}"
    return (
        f"- `{item.key}` status=`{item.status}` pretrade n={item.pretrade_n} "
        f"net={item.pretrade_net_jpy:.1f} avg={item.pretrade_avg_jpy:.1f}; "
        f"live n={item.live_n} net={item.live_net_jpy:.1f} worst={worst}; "
        f"seats missed/captured={item.seat_missed}/{item.seat_captured}; "
        f"fix: {item.required_fix}"
    )


def _load_json(text: str) -> dict[str, Any]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _positive_evidence(conn: sqlite3.Connection) -> dict[tuple[str, str], list[float]]:
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in conn.execute(
        """
        SELECT pair, direction, pl
        FROM legacy_records
        WHERE pair IS NOT NULL AND direction IS NOT NULL AND pl > 0
          AND source_table IN ('trades', 'pretrade_outcomes', 'seat_outcomes')
        """
    ):
        values[(str(row["pair"]), str(row["direction"]))].append(float(row["pl"]))
    for row in conn.execute(
        """
        SELECT pair, direction, pl_jpy
        FROM live_trade_events
        WHERE pair IS NOT NULL AND direction IS NOT NULL AND pl_jpy > 0
        """
    ):
        values[(str(row["pair"]), str(row["direction"]))].append(float(row["pl_jpy"]))
    return values


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    pct = max(0.0, min(1.0, pct))
    idx = round((len(values) - 1) * pct)
    return values[int(idx)]


def _target_reward_risk(best_jpy: float, tail_jpy: float) -> float:
    evidence_jpy = max(tail_jpy, best_jpy)
    return _round(min(8.0, max(1.5, evidence_jpy / 500.0)))


def _round(value: float) -> float:
    return round(value, 4)
