from __future__ import annotations

import fcntl
import hashlib
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable

from .ledger import CryptoLedger


def _utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(
        timezone.utc
    )


def _decimal(value: object) -> Decimal:
    return Decimal(str(value or "0"))


def _operation_id(kind: str, *parts: str) -> str:
    raw = "|".join(("crypto-paper-improvement", kind, *parts))
    return hashlib.sha256(raw.encode()).hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            operation_id = str(row.get("operation_id", ""))
            if not operation_id or operation_id in seen:
                raise RuntimeError(
                    f"invalid improvement outbox line {line_number}"
                )
            seen.add(operation_id)
            rows.append(row)
    return rows


def _append_once(path: Path, row: dict[str, Any]) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if any(
            existing["operation_id"] == row["operation_id"]
            for existing in _load_jsonl(path)
        ):
            return False
        encoded = json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        with path.open("a", encoding="utf-8") as handle:
            handle.write(encoded + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    return True


def _profit_factor(trades: list[dict[str, Any]]) -> str | None:
    profit = sum(
        (
            max(Decimal("0"), _decimal(row["net_pnl_jpy"]))
            for row in trades
        ),
        Decimal("0"),
    )
    loss = sum(
        (
            abs(min(Decimal("0"), _decimal(row["net_pnl_jpy"])))
            for row in trades
        ),
        Decimal("0"),
    )
    if not trades or loss == 0:
        return None
    return str(profit / loss)


def _drawdown(trades: list[dict[str, Any]]) -> Decimal:
    equity = Decimal("0")
    peak = Decimal("0")
    maximum = Decimal("0")
    for row in sorted(
        trades, key=lambda item: str(item["closed_at_utc"])
    ):
        equity += _decimal(row["net_pnl_jpy"])
        peak = max(peak, equity)
        maximum = max(maximum, peak - equity)
    return maximum


def _contribution(
    trades: list[dict[str, Any]], field: str
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Decimal]] = defaultdict(list)
    for row in trades:
        if field == "jst_hour":
            key = _utc(str(row["closed_at_utc"])).astimezone(
                timezone(timedelta(hours=9))
            ).strftime("%H")
        else:
            key = str(row.get(field) or "UNKNOWN")
        grouped[key].append(_decimal(row["net_pnl_jpy"]))
    return {
        key: {
            "trades": len(values),
            "net_pnl_jpy": str(sum(values, Decimal("0"))),
            "expectancy_jpy": str(
                sum(values, Decimal("0")) / len(values)
            ),
            "wins": sum(value > 0 for value in values),
        }
        for key, values in sorted(grouped.items())
    }


class CryptoImprovementEvaluator:
    """Evidence-first RCA and one-category Paper experiment planner."""

    def __init__(self, runtime_root: Path) -> None:
        self.runtime_root = runtime_root
        self.output_root = runtime_root / "improvement"
        self.evaluations_path = self.output_root / "evaluations.jsonl"
        self.experiments_path = self.output_root / "experiments.jsonl"

    def run_once(
        self,
        now: datetime | None = None,
        *,
        trailing_minutes: int | None = None,
    ) -> dict[str, Any]:
        now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        if trailing_minutes is None:
            end = now.replace(minute=0, second=0, microsecond=0)
            start = end - timedelta(hours=1)
            window_kind = "COMPLETED_UTC_HOUR"
        else:
            end = now.replace(second=0, microsecond=0)
            start = end - timedelta(minutes=max(1, trailing_minutes))
            window_kind = "TRAILING_MANUAL_WINDOW"
        evaluations: list[dict[str, Any]] = []
        experiments: list[dict[str, Any]] = []
        evaluation_added = 0
        experiment_added = 0
        for mode in ("spot", "margin"):
            evaluation = self.evaluate(
                mode, start, end, window_kind=window_kind
            )
            experiment = self._experiment(evaluation)
            evaluation_added += int(
                _append_once(self.evaluations_path, evaluation)
            )
            experiment_added += int(
                _append_once(self.experiments_path, experiment)
            )
            evaluations.append(evaluation)
            experiments.append(experiment)
        return {
            "schema": "QR_CRYPTO_IMPROVEMENT_RUN_V1",
            "generated_at_utc": now.isoformat(),
            "window_start_utc": start.isoformat(),
            "window_end_utc": end.isoformat(),
            "window_kind": window_kind,
            "evaluation_added": evaluation_added,
            "experiment_added": experiment_added,
            "evaluations_path": str(self.evaluations_path),
            "experiments_path": str(self.experiments_path),
            "evaluations": evaluations,
            "experiments": experiments,
            "authority": "NONE",
            "live_mutation": False,
        }

    def evaluate(
        self,
        mode: str,
        start: datetime,
        end: datetime,
        *,
        window_kind: str,
    ) -> dict[str, Any]:
        mode_root = self.runtime_root / mode
        ledger = CryptoLedger(mode_root / "ledger.db")
        events = [
            row
            for row in ledger.events()
            if start <= _utc(str(row["created_at_utc"])) < end
        ]
        epochs = [
            row["payload"]
            for row in events
            if row["event_type"] == "FAST_EPOCH_SUMMARY"
        ]
        epochs.extend(
            self._current_epoch_fallback(mode_root, start, end, epochs)
        )
        trades = [
            row
            for row in _load_jsonl(mode_root / "trade_outbox.jsonl")
            if start <= _utc(str(row["closed_at_utc"])) < end
        ]
        waits = Counter()
        actions = Counter()
        regimes = Counter()
        siblings = Counter()
        total_events = 0
        elapsed_sec = 0.0
        decision_p95: list[float] = []
        exchange_p95: list[float] = []
        near_threshold = 0
        prediction_candidates = 0
        prediction_duplicates = 0
        gross_edge_p50: list[float] = []
        expected_cost_p50: list[float] = []
        guardian_states = Counter()
        for epoch in epochs:
            runtime = epoch.get("runtime") or {}
            decisions = epoch.get("decisions") or {}
            diagnostics = epoch.get("decision_diagnostics") or {}
            latency = epoch.get("latency") or {}
            total_events += int(runtime.get("events_processed") or 0)
            elapsed_sec += float(runtime.get("elapsed_sec") or 0)
            waits.update(decisions.get("reasons") or {})
            actions.update(decisions.get("actions") or {})
            regimes.update(diagnostics.get("market_regimes") or {})
            siblings.update(
                diagnostics.get("shadow_sibling_candidates") or {}
            )
            near_threshold += int(
                diagnostics.get("near_threshold_waits") or 0
            )
            prediction_candidates += int(
                diagnostics.get("prediction_candidate_count") or 0
            )
            prediction_duplicates += int(
                diagnostics.get("prediction_duplicate_count") or 0
            )
            for source, target in (
                (diagnostics.get("gross_edge_bps_p50"), gross_edge_p50),
                (
                    diagnostics.get("expected_cost_bps_p50"),
                    expected_cost_p50,
                ),
                (latency.get("decision_us_p95"), decision_p95),
                (
                    latency.get("exchange_to_receive_ms_p95"),
                    exchange_p95,
                ),
            ):
                if source is not None:
                    target.append(float(source))
            guardian_states[
                str((epoch.get("guardian") or {}).get("state", "UNKNOWN"))
            ] += 1
        window_sec = max(1.0, (end - start).total_seconds())
        stale_count = waits["STALE_STREAM_DATA"]
        data_issue_count = (
            stale_count
            + waits["BOOK_NOT_READY"]
            + waits["NO_STREAM_EVENTS"]
            + waits["FUTURE_STREAM_DATA"]
        )
        net_pnl = sum(
            (_decimal(row["net_pnl_jpy"]) for row in trades),
            Decimal("0"),
        )
        expectancy = net_pnl / len(trades) if trades else None
        costs = {
            field: str(
                sum(
                    (_decimal(row[field]) for row in trades),
                    Decimal("0"),
                )
            )
            for field in (
                "fees_jpy",
                "spread_cost_jpy",
                "adverse_cost_jpy",
                "funding_interest_jpy",
            )
        }
        state = self._state(mode_root)
        performance = {
            "completed_trades": len(trades),
            "wins": sum(_decimal(row["net_pnl_jpy"]) > 0 for row in trades),
            "profit_factor_after_cost": _profit_factor(trades),
            "expectancy_jpy": (
                str(expectancy) if expectancy is not None else None
            ),
            "net_pnl_jpy": str(net_pnl),
            "max_drawdown_jpy": str(_drawdown(trades)),
            "equity_jpy": (state.get("metrics") or {}).get("equity_jpy"),
        }
        prediction = {
            "layer": "ENTRY_CANDIDATE_TO_COMPLETED_TRADE",
            "candidate_count": prediction_candidates,
            "duplicate_count": prediction_duplicates,
            "duplicate_rate": (
                prediction_duplicates / prediction_candidates
                if prediction_candidates
                else None
            ),
            "resolved_count": len(trades),
            "resolution_rate": (
                min(1.0, len(trades) / prediction_candidates)
                if prediction_candidates
                else None
            ),
            "expired_pending": max(
                0, prediction_candidates - len(trades)
            ),
        }
        evaluation = {
            "schema": "QR_CRYPTO_IMPROVEMENT_EVALUATION_V1",
            "operation_id": _operation_id(
                "evaluation",
                mode,
                start.isoformat(),
                end.isoformat(),
                "baseline",
            ),
            "mode": mode.upper(),
            "window_kind": window_kind,
            "window_start_utc": start.isoformat(),
            "window_end_utc": end.isoformat(),
            "window_start_jst": start.astimezone(
                timezone(timedelta(hours=9))
            ).isoformat(),
            "window_end_jst": end.astimezone(
                timezone(timedelta(hours=9))
            ).isoformat(),
            "source": {
                "ledger_path": str(ledger.path),
                "ledger_integrity": ledger.verify(),
                "trade_outbox_path": str(
                    mode_root / "trade_outbox.jsonl"
                ),
                "epoch_summaries": len(epochs),
            },
            "availability_and_freshness": {
                "events": total_events,
                "elapsed_sec": elapsed_sec,
                "window_coverage_ratio": min(1.0, elapsed_sec / window_sec),
                "events_per_sec": (
                    total_events / elapsed_sec if elapsed_sec else 0
                ),
                "data_issue_events": data_issue_count,
                "stale_events": stale_count,
                "data_issue_rate": (
                    data_issue_count / total_events if total_events else None
                ),
                "heartbeat_at_utc": state.get("heartbeat_at_utc"),
                "guardian_states": dict(guardian_states),
            },
            "performance": performance,
            "contribution": {
                "pair": _contribution(trades, "pair"),
                "side": _contribution(trades, "side"),
                "strategy": _contribution(trades, "strategy"),
                "regime": _contribution(trades, "regime"),
                "jst_hour": _contribution(trades, "jst_hour"),
            },
            "execution_cost": {
                **costs,
                "decision_latency_us_p95_max": max(
                    decision_p95, default=None
                ),
                "exchange_to_receive_ms_p95_max": max(
                    exchange_p95, default=None
                ),
                "gross_edge_bps_p50_mean": (
                    sum(gross_edge_p50) / len(gross_edge_p50)
                    if gross_edge_p50
                    else None
                ),
                "expected_cost_bps_p50_mean": (
                    sum(expected_cost_p50) / len(expected_cost_p50)
                    if expected_cost_p50
                    else None
                ),
            },
            "wait_reject_opportunity": {
                "actions": dict(actions),
                "wait_reasons": dict(waits),
                "near_threshold_waits": near_threshold,
                "net_edge_below_buffer": waits[
                    "NET_EDGE_BELOW_BUFFER"
                ],
                "short_disabled": waits["SHORT_DISABLED"],
                "shadow_sibling_candidates": dict(siblings),
            },
            "risk_guards": {
                "margin_guard_events": sum(
                    row["event_type"] == "MARGIN_GUARD"
                    for row in events
                ),
                "guardian_non_green_epochs": sum(
                    count
                    for name, count in guardian_states.items()
                    if name != "GREEN"
                ),
                "discipline_violations": (
                    state.get("metrics") or {}
                ).get("discipline_violations", 0),
                "authority": "NONE",
                "live_mutation": False,
            },
            "prediction_quality": prediction,
            "baseline": {
                "strategy": "FAST_MICROSTRUCTURE",
                "preserved": True,
            },
            "causality": {
                "future_data_used": False,
                "event_time_order_preserved": True,
            },
        }
        evaluation["root_causes_top3"] = self._root_causes(evaluation)
        evaluation["adoption_gate"] = self._adoption_gate(evaluation)
        return evaluation

    def _current_epoch_fallback(
        self,
        mode_root: Path,
        start: datetime,
        end: datetime,
        epochs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        known = {str(row.get("run_id")) for row in epochs}
        result: list[dict[str, Any]] = []
        for filename in ("latest_epoch.json", "state.json"):
            path = mode_root / filename
            if not path.exists():
                continue
            row = json.loads(path.read_text(encoding="utf-8"))
            run_id = str(row.get("run_id") or "")
            timestamp = str(
                row.get("completed_at_utc")
                or row.get("heartbeat_at_utc")
                or row.get("started_at_utc")
                or ""
            )
            if not run_id or run_id in known or not timestamp:
                continue
            observed = _utc(timestamp)
            if not (start <= observed < end):
                continue
            if filename == "state.json":
                row = {
                    "run_id": run_id,
                    "runtime": {
                        "events_processed": row.get(
                            "events_processed", 0
                        ),
                        "elapsed_sec": 0,
                    },
                    "decisions": {
                        "actions": row.get("actions", {}),
                        "reasons": row.get("reasons", {}),
                    },
                    "decision_diagnostics": {},
                    "latency": {},
                    "guardian": row.get("guardian", {}),
                    "metrics": row.get("metrics", {}),
                }
            result.append(row)
            known.add(run_id)
        return result

    @staticmethod
    def _state(mode_root: Path) -> dict[str, Any]:
        path = mode_root / "state.json"
        return (
            json.loads(path.read_text(encoding="utf-8"))
            if path.exists()
            else {}
        )

    def _root_causes(
        self, evaluation: dict[str, Any]
    ) -> list[dict[str, Any]]:
        availability = evaluation["availability_and_freshness"]
        performance = evaluation["performance"]
        opportunity = evaluation["wait_reject_opportunity"]
        cost = evaluation["execution_cost"]
        prediction = evaluation["prediction_quality"]
        events = int(availability["events"])
        causes: list[dict[str, Any]] = []

        def add(
            code: str,
            category: str,
            impact: float,
            confidence: float,
            evidence: dict[str, Any],
        ) -> None:
            causes.append(
                {
                    "code": code,
                    "category": category,
                    "impact_score": round(min(100.0, impact), 3),
                    "confidence": round(min(1.0, confidence), 3),
                    "rank_score": round(
                        min(100.0, impact) * min(1.0, confidence), 3
                    ),
                    "evidence": evidence,
                }
            )

        issue_rate = availability.get("data_issue_rate")
        if not events or issue_rate is None or issue_rate > 0.05:
            add(
                "DATA_INSUFFICIENT",
                "data_quality",
                100 if not events else issue_rate * 100,
                0.95 if not events else 0.85,
                {
                    "events": events,
                    "data_issue_rate": issue_rate,
                    "epoch_summaries": evaluation["source"][
                        "epoch_summaries"
                    ],
                },
            )
        trades = int(performance["completed_trades"])
        net_block = int(opportunity["net_edge_below_buffer"])
        gross = cost.get("gross_edge_bps_p50_mean")
        expected = cost.get("expected_cost_bps_p50_mean")
        if trades == 0 and expected and gross is not None and gross < expected:
            add(
                "STRATEGY_EDGE_BELOW_COST",
                "strategy_family",
                max(50.0, 100.0 * net_block / max(1, events)),
                0.92,
                {
                    "completed_trades": 0,
                    "gross_edge_bps_p50": gross,
                    "expected_cost_bps_p50": expected,
                    "net_edge_below_buffer": net_block,
                },
            )
        near = int(opportunity["near_threshold_waits"])
        if trades == 0 and near:
            add(
                "ENTRY_THRESHOLD_EXCESSIVE_CANDIDATE",
                "entry_threshold",
                100.0 * near / max(1, events),
                0.65,
                {
                    "near_threshold_waits": near,
                    "events": events,
                    "requires_unseen_window_test": True,
                },
            )
        mismatch = sum(
            int(opportunity["wait_reasons"].get(name, 0))
            for name in (
                "ADVERSE_MOMENTUM_BLOCK",
                "IMBALANCE_BELOW_ENTRY",
                "WIDE_SPREAD",
                "SHORT_DISABLED",
            )
        )
        sibling_count = sum(
            int(value)
            for value in opportunity[
                "shadow_sibling_candidates"
            ].values()
        )
        if trades == 0 and (mismatch or sibling_count):
            add(
                "MARKET_OR_STRATEGY_COVERAGE_MISMATCH",
                "market_strategy_fit",
                100.0 * max(mismatch, sibling_count) / max(1, events),
                0.6,
                {
                    "blocked_events": mismatch,
                    "sibling_observations": sibling_count,
                    "candidate_siblings": opportunity[
                        "shadow_sibling_candidates"
                    ],
                },
            )
        pf = performance["profit_factor_after_cost"]
        if trades and (pf is None or Decimal(pf) <= 1):
            add(
                "NEGATIVE_AFTER_COST_EXPECTANCY",
                "execution_or_strategy",
                min(100.0, abs(float(performance["net_pnl_jpy"])) + 50),
                0.9,
                {
                    "profit_factor": pf,
                    "expectancy_jpy": performance["expectancy_jpy"],
                    "net_pnl_jpy": performance["net_pnl_jpy"],
                },
            )
        if prediction["duplicate_count"]:
            add(
                "PREDICTION_DUPLICATION",
                "prediction_quality",
                prediction["duplicate_rate"] * 100,
                0.95,
                prediction,
            )
        if not causes:
            add(
                "EVIDENCE_WINDOW_TOO_SMALL",
                "sample_size",
                50,
                0.9,
                {"completed_trades": trades, "events": events},
            )
        return sorted(
            causes,
            key=lambda row: (-row["rank_score"], row["code"]),
        )[:3]

    @staticmethod
    def _adoption_gate(
        evaluation: dict[str, Any]
    ) -> dict[str, Any]:
        performance = evaluation["performance"]
        pf = performance["profit_factor_after_cost"]
        expectancy = performance["expectancy_jpy"]
        return {
            "eligible_now": bool(
                int(performance["completed_trades"]) > 0
                and pf is not None
                and Decimal(pf) > 1
                and expectancy is not None
                and Decimal(expectancy) > 0
            ),
            "required": {
                "profit_factor_after_cost": ">1",
                "expectancy_jpy": ">0",
                "max_drawdown": "NON_WORSE_THAN_BASELINE",
                "reproducibility": "AT_LEAST_3_UNSEEN_WINDOWS",
            },
            "live_promotion_allowed": False,
        }

    def _experiment(
        self, evaluation: dict[str, Any]
    ) -> dict[str, Any]:
        cause = evaluation["root_causes_top3"][0]
        category = str(cause["category"])
        siblings = {
            "RANGE_LIQUID": "RANGE_MAKER_REVERSION",
            "TREND_UP": "BREAKOUT_CONFIRMATION_LONG",
            "TREND_DOWN": "BREAKOUT_CONFIRMATION_SHORT",
            "DATA_UNREADY": "NO_STRATEGY_CHANGE",
        }
        observed = evaluation["wait_reject_opportunity"][
            "shadow_sibling_candidates"
        ]
        if category in {"strategy_family", "market_strategy_fit"}:
            selected = max(
                observed,
                key=lambda key: int(observed[key]),
                default="RANGE_MAKER_REVERSION",
            )
            changed_category = "strategy_family"
        elif category == "entry_threshold":
            selected = "ENTRY_THRESHOLD_CALIBRATION_V1"
            changed_category = "entry_threshold"
        elif category == "data_quality":
            selected = "STREAM_FRESHNESS_RECOVERY_V1"
            changed_category = "data_quality"
        elif category == "prediction_quality":
            selected = "PREDICTION_DEDUPE_V1"
            changed_category = "prediction_quality"
        else:
            selected = "MAKER_EXIT_COST_VARIANT_V1"
            changed_category = "execution_model"
        experiment_id = _operation_id(
            "experiment",
            evaluation["mode"],
            evaluation["window_start_utc"],
            cause["code"],
            selected,
        )
        return {
            "schema": "QR_CRYPTO_SHADOW_EXPERIMENT_V1",
            "operation_id": experiment_id,
            "experiment_id": experiment_id[:24],
            "mode": evaluation["mode"],
            "status": "SCHEDULED_NEXT_UNSEEN_MARKET_WINDOW",
            "discovery_evaluation_id": evaluation["operation_id"],
            "root_cause": cause,
            "baseline": {
                "strategy": "FAST_MICROSTRUCTURE",
                "preserved": True,
                "ledger_immutable": True,
            },
            "variant": {
                "name": selected,
                "changed_category": changed_category,
                "only_one_category_changed": True,
                "paper_only": True,
                "authority": "NONE",
                "decision_observation_only_until_isolated_ledger": True,
            },
            "regime_sibling_candidates": siblings,
            "protocol": {
                "future_data_allowed": False,
                "comparison_window": "NEXT_UNSEEN_REAL_MARKET_WINDOW",
                "minimum_completed_trades_per_lane": 30,
                "minimum_independent_windows": 3,
                "same_pair_universe_and_cost_model": True,
                "baseline_continues_unchanged": True,
            },
            "adoption_conditions": {
                "profit_factor_after_cost": ">1",
                "expectancy_jpy": ">0",
                "max_drawdown": "NON_WORSE_THAN_BASELINE",
                "reproducibility": "PASS_3_UNSEEN_WINDOWS",
                "live_order_promotion": "FORBIDDEN",
            },
        }


def improvements_for_period(
    runtime_root: Path, period_start: datetime, period_end: datetime
) -> list[dict[str, Any]]:
    path = runtime_root / "improvement" / "evaluations.jsonl"
    return [
        row
        for row in _load_jsonl(path)
        if _utc(str(row["window_start_utc"])) >= period_start
        and _utc(str(row["window_end_utc"])) <= period_end
    ]
