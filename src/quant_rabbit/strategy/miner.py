from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quant_rabbit.paths import DEFAULT_DAILY_TARGET_STATE, DEFAULT_HISTORY_DB, DEFAULT_STRATEGY_PROFILE, DEFAULT_STRATEGY_REPORT
from quant_rabbit.risk import RiskPolicy


@dataclass
class PairDirectionProfile:
    pair: str
    direction: str
    method: str | None = None
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
    seat_pl_n: int = 0
    seat_net_jpy: float = 0.0
    seat_avg_jpy: float = 0.0
    seat_win_rate_pct: float = 0.0
    order_blocked: int = 0
    top_block_reasons: list[str] = field(default_factory=list)
    status: str = "WATCH_ONLY"
    required_fix: str = ""
    receipt_promotion: dict[str, Any] | None = None

    @property
    def key(self) -> str:
        method = f" {self.method}" if self.method else ""
        return f"{self.pair} {self.direction}{method}"


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


@dataclass(frozen=True)
class ResolvedStrategyLossCap:
    loss_cap_jpy: float
    source: str


class StrategyMiner:
    def __init__(
        self,
        db_path: Path = DEFAULT_HISTORY_DB,
        report_path: Path = DEFAULT_STRATEGY_REPORT,
        profile_path: Path = DEFAULT_STRATEGY_PROFILE,
        loss_cap_jpy: float | None = None,
        target_state_path: Path = DEFAULT_DAILY_TARGET_STATE,
    ) -> None:
        self.db_path = db_path
        self.report_path = report_path
        self.profile_path = profile_path
        self.loss_cap_jpy = loss_cap_jpy
        self.target_state_path = target_state_path

    def run(self) -> StrategyMiningSummary:
        if not self.db_path.exists():
            raise FileNotFoundError(f"legacy history DB not found: {self.db_path}")
        cap = _resolve_strategy_loss_cap(self.loss_cap_jpy, self.target_state_path)
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            profiles = self._build_profiles(conn, cap.loss_cap_jpy)
            profiles = self._with_preserved_receipt_promotions(profiles)
            coverage = self._coverage(conn)
            if not profiles:
                raise ValueError(
                    "mine-strategy produced zero profiles; refusing to write an empty strategy profile. "
                    "Run import-legacy and verify legacy_history.db contains pretrade/live evidence."
                )
            generated_at = datetime.now(timezone.utc).isoformat()
            self._write_profile(profiles, coverage, generated_at, cap)
            self._write_report(profiles, coverage, generated_at, cap)
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

    def _build_profiles(self, conn: sqlite3.Connection, loss_cap_jpy: float) -> list[PairDirectionProfile]:
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
            "SELECT pair, direction, pl, raw_json FROM legacy_records WHERE source_table='seat_outcomes'"
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
            if row["pl"] is not None:
                seat_pl = float(row["pl"])
                item.seat_pl_n += 1
                item.seat_net_jpy = _round(item.seat_net_jpy + seat_pl)
                item.seat_win_rate_pct = _round(
                    (((item.seat_win_rate_pct / 100.0) * (item.seat_pl_n - 1)) + (1 if seat_pl > 0 else 0))
                    / item.seat_pl_n
                    * 100.0
                )
                item.seat_avg_jpy = _round(item.seat_net_jpy / item.seat_pl_n)

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
            item.target_reward_risk = _target_reward_risk(item.positive_best_jpy, item.positive_tail_jpy, loss_cap_jpy)

        for item in profiles.values():
            self._classify(item, loss_cap_jpy)
        return sorted(profiles.values(), key=_profile_sort_key)

    def _with_preserved_receipt_promotions(
        self,
        profiles: list[PairDirectionProfile],
    ) -> list[PairDirectionProfile]:
        """Carry method-scoped receipt promotions across mine-strategy refreshes.

        `promote-receipts` converts a concrete dry-run receipt into a
        method-scoped CANDIDATE. A later `mine-strategy` pass rebuilds the base
        pair/direction profile from archive evidence, so without this merge the
        long-term memory forgets the promoted method every refresh.
        """
        preserved = _load_preservable_receipt_profiles(self.profile_path)
        if not preserved:
            return profiles

        base_by_pair_direction = {(item.pair, item.direction): item for item in profiles if item.method is None}
        seen = {(item.pair, item.direction, item.method) for item in profiles}
        merged = list(profiles)
        for item in preserved:
            base = base_by_pair_direction.get((item.pair, item.direction))
            if base is None or base.status not in {"CANDIDATE", "RISK_REPAIR_CANDIDATE", "MINE_MISSED_EDGE"}:
                continue
            if _is_missed_edge_promotion(item) and base.seat_pl_n > 0 and base.seat_net_jpy <= 0:
                continue
            key = (item.pair, item.direction, item.method)
            if key in seen:
                continue
            merged.append(
                replace(
                    base,
                    method=item.method,
                    status="CANDIDATE",
                    required_fix=item.required_fix,
                    receipt_promotion=item.receipt_promotion,
                )
            )
            seen.add(key)
        return sorted(merged, key=_profile_sort_key)

    def _classify(self, item: PairDirectionProfile, loss_cap_jpy: float) -> None:
        worst_loss = item.live_worst_jpy if item.live_worst_jpy is not None else 0.0
        positive_pretrade = item.pretrade_n >= 5 and item.pretrade_net_jpy > 0
        positive_live = item.live_n > 0 and item.live_net_jpy > 0
        missed_pressure = item.seat_missed >= 2 and item.seat_directionally_correct > item.seat_captured
        missed_pressure_profitable = missed_pressure and (item.seat_pl_n == 0 or item.seat_net_jpy > 0)
        missed_pressure_negative = missed_pressure and item.seat_pl_n > 0 and item.seat_net_jpy <= 0
        cap_text = f"{loss_cap_jpy:.0f} JPY"
        if worst_loss <= -loss_cap_jpy and positive_pretrade and positive_live:
            item.status = "RISK_REPAIR_CANDIDATE"
            item.required_fix = f"edge exists but old sizing broke the loss cap; require <={cap_text} dry-run receipt before live use"
            return
        if missed_pressure_profitable and not (item.live_n >= 3 and item.live_net_jpy < 0 and item.pretrade_net_jpy <= 0):
            item.status = "MINE_MISSED_EDGE"
            item.required_fix = "missed seats paid more often than captured; build trigger/pending-entry receipts before live execution"
            if worst_loss <= -loss_cap_jpy:
                item.required_fix += f"; every receipt must be risk-resized under the {cap_text} cap"
            return
        if worst_loss <= -loss_cap_jpy:
            item.status = "BLOCK_UNTIL_NEW_EVIDENCE"
            item.required_fix = f"historical live loss exceeded the {cap_text} cap; only risk-resized dry-run receipts can reopen it"
            return
        if item.live_n >= 3 and item.live_net_jpy < 0 and item.pretrade_net_jpy <= 0:
            item.status = "BLOCK_UNTIL_NEW_EVIDENCE"
            item.required_fix = "both live execution and pretrade feedback are negative; require a new vehicle or market-structure proof"
            return
        if positive_pretrade and item.live_net_jpy >= 0:
            item.status = "CANDIDATE"
            item.required_fix = "eligible for dry-run order-intent generation, still behind risk gateway"
            if missed_pressure_negative:
                item.required_fix += "; missed-seat promotion disabled because realized seat net is negative"
            return
        if missed_pressure_negative:
            item.status = "WATCH_ONLY"
            item.required_fix = (
                "missed seats were directionally correct, but realized seat net is negative; "
                "repair discovery filters before mining this edge"
            )
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
        cap: ResolvedStrategyLossCap,
    ) -> None:
        self.profile_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at_utc": generated_at,
            "history_db": str(self.db_path),
            "coverage": coverage,
            "profiles": [asdict(item) for item in profiles],
            "system_contract": {
                "live_execution": "disabled until RiskEngine and strategy profile both pass",
                "loss_cap_jpy": cap.loss_cap_jpy,
                "loss_cap_source": cap.source,
                "minimum_reward_risk": 1.2,
                "blocked_status": "BLOCK_UNTIL_NEW_EVIDENCE",
                "risk_repair_status": "RISK_REPAIR_CANDIDATE",
            },
        }
        promotions = [
            item.receipt_promotion
            for item in profiles
            if item.receipt_promotion is not None
        ]
        if promotions:
            payload["receipt_promotions"] = promotions
            promoted_at = [
                str(item.get("promoted_at_utc") or "")
                for item in promotions
                if isinstance(item, dict) and item.get("promoted_at_utc")
            ]
            if promoted_at:
                payload["last_receipt_promotion_at_utc"] = max(promoted_at)
        self.profile_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    def _write_report(
        self,
        profiles: list[PairDirectionProfile],
        coverage: dict[str, Any],
        generated_at: str,
        cap: ResolvedStrategyLossCap,
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
            f"- Per-trade loss cap: `{cap.loss_cap_jpy:.0f} JPY` (`{cap.source}`)",
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
                f"- Any pair/direction with a historical live loss worse than -{cap.loss_cap_jpy:.0f} JPY needs fresh risk-resized dry-run receipts before live use, even when expectancy is positive.",
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


def _load_preservable_receipt_profiles(path: Path) -> list[PairDirectionProfile]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, dict):
        return []
    profiles = payload.get("profiles")
    if not isinstance(profiles, list):
        return []

    preserved: list[PairDirectionProfile] = []
    for item in profiles:
        if not isinstance(item, dict) or str(item.get("status") or "") != "CANDIDATE":
            continue
        promotion = item.get("receipt_promotion")
        if not isinstance(promotion, dict):
            continue
        pair = str(item.get("pair") or "").strip()
        direction = str(item.get("direction") or "").strip().upper()
        method = str(item.get("method") or promotion.get("method") or "").strip().upper()
        if not pair or not direction or not method:
            continue
        required_fix = str(item.get("required_fix") or "").strip()
        if not required_fix:
            lane_id = str(promotion.get("lane_id") or "")
            reason = str(promotion.get("reason") or "receipt promotion")
            required_fix = f"promoted by {reason}; source lane {lane_id}".strip()
        preserved.append(
            PairDirectionProfile(
                pair=pair,
                direction=direction,
                method=method,
                status="CANDIDATE",
                required_fix=required_fix,
                receipt_promotion=dict(promotion),
            )
        )
    return preserved


def _is_missed_edge_promotion(item: PairDirectionProfile) -> bool:
    promotion = item.receipt_promotion if isinstance(item.receipt_promotion, dict) else {}
    from_status = str(promotion.get("from_status") or "").upper()
    reason = str(promotion.get("reason") or item.required_fix or "").upper()
    return from_status == "MINE_MISSED_EDGE" or "MISSED_EDGE" in reason or "MISSED EDGE" in reason


def _profile_line(item: PairDirectionProfile) -> str:
    worst = "n/a" if item.live_worst_jpy is None else f"{item.live_worst_jpy:.1f}"
    return (
        f"- `{item.key}` status=`{item.status}` pretrade n={item.pretrade_n} "
        f"net={item.pretrade_net_jpy:.1f} avg={item.pretrade_avg_jpy:.1f}; "
        f"live n={item.live_n} net={item.live_net_jpy:.1f} worst={worst}; "
        f"seats missed/captured={item.seat_missed}/{item.seat_captured} "
        f"net={item.seat_net_jpy:.1f} n={item.seat_pl_n} win={item.seat_win_rate_pct:.1f}%; "
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


def _resolve_strategy_loss_cap(explicit_loss_cap_jpy: float | None, target_state_path: Path) -> ResolvedStrategyLossCap:
    if explicit_loss_cap_jpy is not None:
        if explicit_loss_cap_jpy <= 0:
            raise ValueError("mine-strategy loss cap must be positive")
        return ResolvedStrategyLossCap(round(float(explicit_loss_cap_jpy), 4), "explicit loss_cap_jpy")
    state_cap = _loss_cap_from_target_state(target_state_path)
    if state_cap is not None:
        return ResolvedStrategyLossCap(state_cap, f"daily target state {target_state_path}")
    policy_cap = RiskPolicy().max_loss_jpy
    if policy_cap is None or policy_cap <= 0:
        raise ValueError("mine-strategy cannot derive a positive loss cap")
    return ResolvedStrategyLossCap(round(float(policy_cap), 4), "RiskPolicy.max_loss_jpy library default")


def _loss_cap_from_target_state(path: Path) -> float | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    raw = payload.get("per_trade_risk_budget_jpy") if isinstance(payload, dict) else None
    try:
        value = float(raw) if raw is not None else 0.0
    except (TypeError, ValueError):
        return None
    return round(value, 4) if value > 0 else None


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    pct = max(0.0, min(1.0, pct))
    idx = round((len(values) - 1) * pct)
    return values[int(idx)]


def _target_reward_risk(best_jpy: float, tail_jpy: float, loss_cap_jpy: float) -> float:
    evidence_jpy = max(tail_jpy, best_jpy)
    return _round(min(8.0, max(1.5, evidence_jpy / loss_cap_jpy)))


def _round(value: float) -> float:
    return round(value, 4)
