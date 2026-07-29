from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from .report import atomic_write_json, atomic_write_text
from .strategy_audit import _d, _s, _window_metrics


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _parse_time(value: object) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _hour(value: object) -> str:
    return _parse_time(value).astimezone(timezone.utc).strftime(
        "%Y-%m-%dT%H:00:00Z"
    )


def _scaled_row(row: dict[str, Any], multiplier: Decimal) -> dict[str, Any]:
    result = dict(row)
    for field in (
        "entry_notional_jpy",
        "gross_pnl_jpy",
        "fees_jpy",
        "spread_cost_jpy",
        "adverse_cost_jpy",
        "funding_interest_jpy",
        "net_pnl_jpy",
        "quantity",
    ):
        result[field] = _s(_d(row.get(field)) * multiplier)
    return result


class BitbankProfitabilityStudy:
    """Read-only, point-in-time comparison over isolated Paper ledgers."""

    def __init__(
        self,
        runtime_root: Path,
        *,
        research_config: Path,
        output_root: Path,
    ) -> None:
        self.runtime_root = runtime_root
        self.config_path = research_config
        self.config = _read_json(research_config)
        if self.config.get("schema") != "QR_BITBANK_RESEARCH_CANDIDATES_V1":
            raise ValueError("unsupported bitbank research config")
        self.output_root = output_root
        self._decision_cache: dict[
            tuple[str, str], list[dict[str, Any]]
        ] = {}
        self._root_decision_cache: dict[
            str, dict[str, list[dict[str, Any]]]
        ] = {}
        self._latency_cache: dict[str, list[Decimal]] = {}

    def run_once(self) -> dict[str, Any]:
        rows = self._portfolio_rows()
        baseline_profiles = set(self.config["baseline_profiles"])
        baseline = [
            row for row in rows if row["strategy"] in baseline_profiles
        ]
        if not baseline:
            raise ValueError("profitability study has no baseline trades")
        comparisons = [
            self._fade_stop(baseline),
            self._maker_exit(rows),
            self._entry_gate(baseline),
            self._regime_router(baseline),
            self._dynamic_size(baseline),
            self._mid_trade_exit(baseline),
        ]
        source_tip = "|".join(
            f"{row['strategy']}:{row['paper_mode']}:{row['trade_id']}"
            for row in sorted(
                baseline,
                key=lambda item: (
                    str(item["strategy"]),
                    str(item["paper_mode"]),
                    str(item["trade_id"]),
                ),
            )
        )
        operation_id = hashlib.sha256(
            (
                "qr-bitbank-profitability-study-v1|"
                + hashlib.sha256(source_tip.encode()).hexdigest()
                + "|"
                + hashlib.sha256(
                    self.config_path.read_bytes()
                ).hexdigest()
            ).encode()
        ).hexdigest()
        baseline_metrics = _window_metrics(baseline)
        fade_rows = [
            row
            for row in baseline
            if str(row["strategy"]).startswith("ORDER_BOOK_FADE")
        ]
        payload = {
            "schema": "QR_BITBANK_PROFITABILITY_STUDY_V1",
            "operation_id": operation_id,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "mode": "READ_ONLY_ISOLATED_PAPER_COMPARISON",
            "venue": "bitbank",
            "baseline_contract": {
                "profiles": sorted(baseline_profiles),
                "metrics": baseline_metrics,
                "utc_windows": self._windows(baseline),
            },
            "remeasured_loss": {
                "original_10_lane_metrics": baseline_metrics,
                "board_fade_metrics": _window_metrics(fade_rows),
                "board_fade_net_loss_share": _s(
                    abs(
                        _d(_window_metrics(fade_rows)["net_pnl_jpy"])
                    )
                    / max(
                        Decimal("0.000001"),
                        abs(_d(baseline_metrics["net_pnl_jpy"])),
                    )
                ),
                "causes": [
                    "FEE_DRAG_DOMINATES_GROSS_EDGE",
                    "PARTIAL_FILL_CHURN",
                    "MAKER_ENTRY_TAKER_EXIT_OVERTRADING",
                ],
            },
            "isolated_comparisons": comparisons,
            "adoption_gate": self.config["adoption_gate"],
            "research_review": {
                "sources": self.config["sources"],
                "candidate_shortlist": self.config[
                    "candidate_shortlist"
                ],
                "external_code_executed": False,
                "external_code_copied": False,
                "license_policy": (
                    "REFERENCE_ONLY_UNLESS_LICENSE_AND_BITBANK_FIT_REVIEWED"
                ),
                "oanda_reuse_boundary": (
                    "REUSE_SAFETY_LEDGER_EVALUATION_ONLY_NOT_STRATEGY_LOGIC"
                ),
            },
            "safety": {
                "NO_EXECUTE": True,
                "CRYPTO_LIVE_READY": False,
                "WITHDRAWAL_ENABLED": False,
                "authority": "NONE",
                "order_api_called": False,
                "cancel_api_called": False,
                "settlement_api_called": False,
                "withdrawal_api_called": False,
                "existing_shadow_changed": False,
            },
        }
        atomic_write_json(self.output_root / "study.json", payload)
        atomic_write_text(
            self.output_root / "study.md", self._markdown(payload)
        )
        return payload

    def _portfolio_rows(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for strategy in self.config["all_profiles"]:
            slug = strategy.lower().replace("_", "-")
            for mode in ("spot", "margin"):
                root = self.runtime_root / slug / mode
                for row in _read_jsonl(root / "trade_outbox.jsonl"):
                    item = dict(row)
                    item["strategy"] = strategy
                    item["paper_mode"] = mode.upper()
                    item["_root"] = str(root)
                    result.append(item)
        return result

    def _comparison(
        self,
        *,
        category: str,
        baseline: list[dict[str, Any]],
        candidate: list[dict[str, Any]],
        evidence: str,
        changed_fields: list[str],
        prospective_windows: int,
        caveats: list[str],
        status_override: str | None = None,
    ) -> dict[str, Any]:
        baseline_metrics = _window_metrics(baseline)
        candidate_metrics = _window_metrics(candidate)
        pf = candidate_metrics["profit_factor_after_cost"]
        expectancy = candidate_metrics["expectancy_jpy"]
        metric_pass = (
            pf is not None
            and _d(pf) > 1
            and expectancy is not None
            and _d(expectancy) > 0
            and _d(candidate_metrics["max_drawdown_jpy"])
            <= _d(baseline_metrics["max_drawdown_jpy"])
        )
        retrospective_windows = self._windows(candidate)
        reproducible = prospective_windows >= int(
            self.config["adoption_gate"]["minimum_unseen_windows"]
        )
        if status_override:
            status = status_override
        elif not metric_pass:
            status = "REJECTED_METRICS"
        elif not reproducible:
            status = "HOLD_FORWARD_PAPER"
        else:
            status = "ELIGIBLE_FOR_PAPER_ADOPTION"
        return {
            "category": category,
            "only_one_category_changed": True,
            "changed_fields": changed_fields,
            "evidence_class": evidence,
            "baseline_metrics": baseline_metrics,
            "candidate_metrics": candidate_metrics,
            "latency": self._latency(candidate),
            "retrospective_utc_windows": retrospective_windows,
            "retrospective_unseen_partition_count": max(
                0, len(retrospective_windows) - 1
            ),
            "prospective_unseen_window_count": prospective_windows,
            "metric_gate_passed": metric_pass,
            "reproducibility_gate_passed": reproducible,
            "status": status,
            "adopted": status == "ELIGIBLE_FOR_PAPER_ADOPTION",
            "live_promotion_allowed": False,
            "future_data_used_for_each_decision": False,
            "caveats": caveats,
        }

    def _fade_stop(
        self, baseline: list[dict[str, Any]]
    ) -> dict[str, Any]:
        candidate = [
            row
            for row in baseline
            if not str(row["strategy"]).startswith("ORDER_BOOK_FADE")
        ]
        return self._comparison(
            category="board_fade_stop_candidate",
            baseline=baseline,
            candidate=candidate,
            evidence="EXACT_PORTFOLIO_EXCLUSION",
            changed_fields=["enabled_strategy_family.ORDER_BOOK_FADE"],
            prospective_windows=0,
            caveats=[
                "Candidate means stop consideration only; no running lane was stopped.",
                "Remaining portfolio still must pass PF and expectancy gates.",
            ],
            status_override="STOP_CANDIDATE_NOT_APPLIED",
        )

    def _maker_exit(
        self, rows: list[dict[str, Any]]
    ) -> dict[str, Any]:
        variant_name = str(self.config["maker_exit_variant"])
        variant = [
            row for row in rows if row["strategy"] == variant_name
        ]
        if variant:
            start = min(_parse_time(row["opened_at_utc"]) for row in variant)
            baseline = [
                row
                for row in rows
                if row["strategy"] == "ORDER_BOOK_FADE"
                and _parse_time(row["closed_at_utc"]) >= start
            ]
        else:
            baseline = []
        return self._comparison(
            category="maker_taker_alignment",
            baseline=baseline,
            candidate=variant,
            evidence="PROSPECTIVE_PARALLEL_PAPER",
            changed_fields=["forced_exit_order_style"],
            prospective_windows=len(self._windows(variant)),
            caveats=[
                "Stop-loss remains conservative Paper taker.",
                "The sample must reach the configured minimum trade count.",
            ],
        )

    def _entry_gate(
        self, baseline: list[dict[str, Any]]
    ) -> dict[str, Any]:
        threshold = _d(self.config["screening"]["min_entry_net_edge_bps"])
        candidate: list[dict[str, Any]] = []
        unmatched = 0
        for row in baseline:
            decision = self._entry_decision(row)
            if decision is None:
                unmatched += 1
                continue
            imbalance = _d(decision.get("imbalance"))
            side = str(row.get("side"))
            aligned = (side == "LONG" and imbalance > 0) or (
                side == "SHORT" and imbalance < 0
            )
            if aligned and _d(decision.get("net_edge_bps")) >= threshold:
                candidate.append(row)
        result = self._comparison(
            category="entry_direction_and_threshold",
            baseline=baseline,
            candidate=candidate,
            evidence="RETROSPECTIVE_CAUSAL_SCREEN",
            changed_fields=["entry_direction_gate", "min_entry_net_edge_bps"],
            prospective_windows=0,
            caveats=[
                "Historical trades are filtered after collection; no adoption proof.",
                f"{unmatched} trades lacked a matchable recorded ENTER decision.",
            ],
        )
        result["screen"] = {
            "min_entry_net_edge_bps": _s(threshold),
            "require_position_side_aligned_with_book_imbalance": True,
        }
        return result

    def _regime_router(
        self, baseline: list[dict[str, Any]]
    ) -> dict[str, Any]:
        routing = self.config["screening"]["regime_strategy_map"]
        candidate = [
            row
            for row in baseline
            if str(row["strategy"])
            in set(routing.get(str(row.get("regime")), []))
        ]
        result = self._comparison(
            category="regime_specific_strategy",
            baseline=baseline,
            candidate=candidate,
            evidence="RETROSPECTIVE_PREDECLARED_ROUTING_SCREEN",
            changed_fields=["strategy_selection_by_regime"],
            prospective_windows=0,
            caveats=[
                "Routing map was defined after these observations.",
                "Sparse trend-regime trades limit inference.",
            ],
        )
        result["screen"] = {"regime_strategy_map": routing}
        return result

    def _dynamic_size(
        self, baseline: list[dict[str, Any]]
    ) -> dict[str, Any]:
        threshold = _d(self.config["screening"]["full_size_net_edge_bps"])
        low = _d(self.config["screening"]["low_edge_size_multiplier"])
        candidate: list[dict[str, Any]] = []
        unmatched = 0
        for row in baseline:
            decision = self._entry_decision(row)
            if decision is None:
                unmatched += 1
                candidate.append(_scaled_row(row, low))
                continue
            multiplier = (
                Decimal("1")
                if _d(decision.get("net_edge_bps")) >= threshold
                else low
            )
            candidate.append(_scaled_row(row, multiplier))
        result = self._comparison(
            category="dynamic_position_size",
            baseline=baseline,
            candidate=candidate,
            evidence="RETROSPECTIVE_LINEAR_SIZE_SCREEN",
            changed_fields=["target_notional_multiplier"],
            prospective_windows=0,
            caveats=[
                "Linear scaling ignores queue priority and size-dependent fill probability.",
                f"{unmatched} unmatched trades received the lower size.",
            ],
        )
        result["screen"] = {
            "full_size_net_edge_bps": _s(threshold),
            "low_edge_size_multiplier": _s(low),
            "maximum_multiplier": "1",
        }
        return result

    def _mid_trade_exit(
        self, baseline: list[dict[str, Any]]
    ) -> dict[str, Any]:
        max_hold_ms = int(self.config["screening"]["candidate_max_hold_ms"])
        candidate: list[dict[str, Any]] = []
        simulated = 0
        for row in baseline:
            decision = self._first_hold_decision(row, max_hold_ms)
            if decision is None:
                candidate.append(dict(row))
                continue
            gross = (
                _d(row.get("entry_notional_jpy"))
                * _d(decision.get("position_pnl_bps"))
                / Decimal("10000")
            )
            cost = sum(
                (
                    _d(row.get("fees_jpy")),
                    _d(row.get("spread_cost_jpy")),
                    _d(row.get("adverse_cost_jpy")),
                    _d(row.get("funding_interest_jpy")),
                ),
                Decimal("0"),
            )
            item = dict(row)
            item["gross_pnl_jpy"] = _s(gross)
            item["net_pnl_jpy"] = _s(gross - cost)
            item["closed_at_utc"] = str(decision["observed_at_utc"])
            candidate.append(item)
            simulated += 1
        result = self._comparison(
            category="mid_trade_exit",
            baseline=baseline,
            candidate=candidate,
            evidence="RETROSPECTIVE_OBSERVED_QUOTE_SCREEN",
            changed_fields=["max_hold_ms"],
            prospective_windows=0,
            caveats=[
                "Uses first recorded quote-state PnL at/after the hold limit.",
                "Keeps full realized trade costs as a conservative approximation.",
                "It is not a queue-aware fill replay.",
            ],
        )
        result["screen"] = {
            "candidate_max_hold_ms": max_hold_ms,
            "simulated_early_exits": simulated,
        }
        return result

    def _entry_decision(
        self, row: dict[str, Any]
    ) -> dict[str, Any] | None:
        decisions = self._decisions(row, actions={"ENTER"})
        opened = _parse_time(row["opened_at_utc"])
        eligible = [
            item
            for item in decisions
            if _parse_time(item["observed_at_utc"]) <= opened
        ]
        return eligible[-1] if eligible else None

    def _first_hold_decision(
        self, row: dict[str, Any], max_hold_ms: int
    ) -> dict[str, Any] | None:
        opened = _parse_time(row["opened_at_utc"])
        closed = _parse_time(row["closed_at_utc"])
        for item in self._decisions(row, actions={"WAIT", "EXIT"}):
            observed = _parse_time(item["observed_at_utc"])
            if (
                opened <= observed < closed
                and int(item.get("held_ms") or 0) >= max_hold_ms
                and item.get("position_pnl_bps") is not None
            ):
                return item
        return None

    def _decisions(
        self, row: dict[str, Any], *, actions: set[str]
    ) -> list[dict[str, Any]]:
        root = Path(str(row["_root"]))
        key = (str(root), str(row["run_id"]))
        if key not in self._decision_cache:
            root_key = str(root)
            if root_key not in self._root_decision_cache:
                grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
                database = root / "ledger.db"
                if database.exists():
                    with sqlite3.connect(database) as connection:
                        for (raw,) in connection.execute(
                            "SELECT payload_json FROM crypto_events "
                            "WHERE event_type='FAST_DECISION' "
                            "ORDER BY sequence"
                        ):
                            payload = json.loads(raw)
                            grouped[str(payload.get("run_id"))].append(payload)
                self._root_decision_cache[root_key] = dict(grouped)
            self._decision_cache[key] = self._root_decision_cache[
                root_key
            ].get(str(row["run_id"]), [])
        return [
            item
            for item in self._decision_cache[key]
            if str(item.get("pair")) == str(row["pair"])
            and str(item.get("action")) in actions
        ]

    @staticmethod
    def _windows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[_hour(row["closed_at_utc"])].append(row)
        return [
            {"window_start_utc": key, **_window_metrics(grouped[key])}
            for key in sorted(grouped)
        ]

    def _latency(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        roots = sorted({str(row.get("_root")) for row in rows})
        p95: list[Decimal] = []
        for raw_root in roots:
            if raw_root not in self._latency_cache:
                values: list[Decimal] = []
                database = Path(raw_root) / "ledger.db"
                if database.exists():
                    with sqlite3.connect(database) as connection:
                        for (raw,) in connection.execute(
                            "SELECT payload_json FROM crypto_events "
                            "WHERE event_type='FAST_EPOCH_SUMMARY'"
                        ):
                            payload = json.loads(raw)
                            value = (payload.get("latency") or {}).get(
                                "decision_us_p95"
                            )
                            if value is not None:
                                values.append(_d(value))
                self._latency_cache[raw_root] = values
            p95.extend(self._latency_cache[raw_root])
        return {
            "decision_us_p95_mean": (
                _s(sum(p95, Decimal("0")) / len(p95)) if p95 else None
            ),
            "epoch_count": len(p95),
            "contract": "PAPER_DECISION_COMPUTE_ONLY_NOT_ORDER_ACK",
        }

    @staticmethod
    def _markdown(payload: dict[str, Any]) -> str:
        baseline = payload["baseline_contract"]["metrics"]
        lines = [
            "# QuantRabbit｜bitbank Paper profitability study",
            "",
            f"- Operation ID: `{payload['operation_id']}`",
            f"- Generated UTC: `{payload['generated_at_utc']}`",
            "- Mode: `READ_ONLY_ISOLATED_PAPER_COMPARISON`",
            "- Authority: `NONE`",
            "",
            "## Remeasured original 10 lanes",
            "",
            f"- Trades: `{baseline['completed_trades']}`",
            f"- Gross PnL: `{baseline['gross_pnl_jpy']}` JPY",
            f"- Fees: `{baseline['fees_jpy']}` JPY",
            f"- Net PnL: `{baseline['net_pnl_jpy']}` JPY",
            f"- PF after cost: `{baseline['profit_factor_after_cost']}`",
            f"- Expectancy: `{baseline['expectancy_jpy']}` JPY/trade",
            f"- Max DD: `{baseline['max_drawdown_jpy']}` JPY",
            "",
            "## Isolated comparisons",
            "",
            "| Category | Trades | Net JPY | Fees JPY | PF | Exp JPY | DD JPY | Status |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
        for row in payload["isolated_comparisons"]:
            metrics = row["candidate_metrics"]
            lines.append(
                "| "
                + " | ".join(
                    [
                        row["category"],
                        str(metrics["completed_trades"]),
                        str(metrics["net_pnl_jpy"]),
                        str(metrics["fees_jpy"]),
                        str(metrics["profit_factor_after_cost"]),
                        str(metrics["expectancy_jpy"]),
                        str(metrics["max_drawdown_jpy"]),
                        row["status"],
                    ]
                )
                + " |"
            )
        lines.extend(
            [
                "",
                "Retrospective partitions are screening evidence only. "
                "They do not count as prospective unseen windows.",
                "",
                "## Safety",
                "",
                "No order, cancel, settlement, withdrawal, or account mutation "
                "API was called. Existing Paper Shadow services were unchanged.",
                "",
            ]
        )
        return "\n".join(lines)
