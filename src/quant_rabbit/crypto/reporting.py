from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Protocol

from .outbox import TRADE_COLUMNS
from .report import atomic_write_json


class SheetsSink(Protocol):
    def append_trade(
        self, operation_id: str, row: dict[str, Any]
    ) -> None: ...

    def append_summary(
        self, operation_id: str, row: dict[str, Any]
    ) -> None: ...

    def readback(self, operation_id: str) -> bool: ...


class SlackSink(Protocol):
    def post_summary(
        self, operation_id: str, row: dict[str, Any]
    ) -> str: ...

    def readback(self, operation_id: str, permalink: str) -> bool: ...


class DeliveryStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with sqlite3.connect(path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS deliveries(
                    operation_id TEXT NOT NULL,
                    target TEXT NOT NULL,
                    delivered_at_utc TEXT NOT NULL,
                    receipt TEXT,
                    PRIMARY KEY(operation_id, target)
                )
                """
            )

    def delivered(self, operation_id: str, target: str) -> bool:
        with sqlite3.connect(self.path) as conn:
            row = conn.execute(
                "SELECT 1 FROM deliveries "
                "WHERE operation_id=? AND target=?",
                (operation_id, target),
            ).fetchone()
        return row is not None

    def mark(
        self, operation_id: str, target: str, receipt: str | None = None
    ) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                "INSERT OR IGNORE INTO deliveries VALUES (?, ?, ?, ?)",
                (
                    operation_id,
                    target,
                    datetime.now(timezone.utc).isoformat(),
                    receipt,
                ),
            )


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
                    f"invalid operation in {path} line {line_number}"
                )
            seen.add(operation_id)
            rows.append(row)
    return rows


def _append_jsonl_once(path: Path, row: dict[str, Any]) -> bool:
    operation_id = str(row["operation_id"])
    if any(
        str(existing["operation_id"]) == operation_id
        for existing in _load_jsonl(path)
    ):
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
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


class PaperShadowReportingWriter:
    """Separate trade-ledger and aggregate delivery process."""

    def __init__(
        self,
        runtime_root: Path,
        *,
        sheets: SheetsSink | None = None,
        slack: SlackSink | None = None,
    ) -> None:
        self.runtime_root = runtime_root
        self.sheets = sheets
        self.slack = slack
        self.store = DeliveryStore(runtime_root / "reporting-deliveries.db")

    def run_once(
        self, now: datetime | None = None
    ) -> dict[str, Any]:
        now = now or datetime.now(timezone.utc)
        trades = self._trade_rows()
        sheets_delivered = 0
        sheets_pending = 0
        for row in trades:
            operation_id = str(row["operation_id"])
            if self.store.delivered(operation_id, "sheets_trade"):
                continue
            if self.sheets is None:
                sheets_pending += 1
                continue
            try:
                already_present = self.sheets.readback(operation_id)
                if not already_present:
                    self.sheets.append_trade(operation_id, row)
                verified = self.sheets.readback(operation_id)
            except Exception:
                verified = False
            if not verified:
                sheets_pending += 1
                continue
            self.store.mark(operation_id, "sheets_trade")
            sheets_delivered += 1

        summaries = [
            self._summary(trades, now, "hour"),
            self._summary(trades, now, "day"),
        ]
        summary_path = self.runtime_root / "summary_outbox.jsonl"
        local_summary_added = 0
        sheets_summary_pending = 0
        slack_pending = 0
        for summary in summaries:
            local_summary_added += int(
                _append_jsonl_once(summary_path, summary)
            )
            operation_id = str(summary["operation_id"])
            if not self.store.delivered(operation_id, "sheets_summary"):
                if self.sheets is None:
                    sheets_summary_pending += 1
                else:
                    try:
                        self.sheets.append_summary(operation_id, summary)
                        verified = self.sheets.readback(operation_id)
                    except Exception:
                        verified = False
                    if verified:
                        self.store.mark(operation_id, "sheets_summary")
                    else:
                        sheets_summary_pending += 1
            if not self.store.delivered(operation_id, "slack_summary"):
                if self.slack is None:
                    slack_pending += 1
                else:
                    try:
                        permalink = self.slack.post_summary(
                            operation_id, summary
                        )
                        verified = self.slack.readback(
                            operation_id, permalink
                        )
                    except Exception:
                        permalink = ""
                        verified = False
                    if verified:
                        self.store.mark(
                            operation_id,
                            "slack_summary",
                            permalink,
                        )
                    else:
                        slack_pending += 1

        result = {
            "schema": "QR_CRYPTO_PAPER_REPORTING_STATE_V1",
            "generated_at_utc": now.isoformat(),
            "trade_rows": len(trades),
            "sheets_trade_delivered": sheets_delivered,
            "sheets_trade_pending": sheets_pending,
            "local_summary_added": local_summary_added,
            "sheets_summary_pending": sheets_summary_pending,
            "slack_summary_pending": slack_pending,
            "sheets_status": (
                "CONNECTED" if self.sheets is not None else "BLOCKED_NO_CONNECTOR"
            ),
            "slack_status": (
                "CONNECTED" if self.slack is not None else "BLOCKED_NO_CONNECTOR"
            ),
            "trade_sheet": "SEPARATE_TRADE_LEDGER",
            "summary_sheet": "SEPARATE_SUMMARY_LEDGER",
            "per_trade_slack_posts": False,
        }
        atomic_write_json(self.runtime_root / "reporting_state.json", result)
        return result

    def _trade_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for mode in ("spot", "margin"):
            path = self.runtime_root / mode / "trade_outbox.jsonl"
            for row in _load_jsonl(path):
                missing = [
                    column for column in TRADE_COLUMNS if column not in row
                ]
                if missing:
                    raise RuntimeError(
                        f"trade row missing columns: {missing}"
                    )
                operation_id = str(row["operation_id"])
                if operation_id in seen:
                    raise RuntimeError("duplicate cross-mode operation id")
                seen.add(operation_id)
                rows.append(row)
        return sorted(
            rows,
            key=lambda row: (
                str(row["closed_at_utc"]),
                str(row["operation_id"]),
            ),
        )

    def _summary(
        self,
        trades: list[dict[str, Any]],
        now: datetime,
        period: str,
    ) -> dict[str, Any]:
        period_key = (
            now.strftime("%Y-%m-%dT%H:00:00Z")
            if period == "hour"
            else now.strftime("%Y-%m-%d")
        )
        selected = [
            row
            for row in trades
            if str(row["closed_at_utc"]).startswith(
                period_key[:13] if period == "hour" else period_key
            )
        ]
        reasons = Counter(str(row["exit_reason"]) for row in selected)
        service_states = self._service_states()
        operation_id = hashlib.sha256(
            f"crypto-paper-summary|{period}|{period_key}".encode()
        ).hexdigest()
        return {
            "operation_id": operation_id,
            "period": period,
            "period_key": period_key,
            "generated_at_utc": now.isoformat(),
            "completed_trades": len(selected),
            "spot_trades": sum(
                row["paper_mode"] == "SPOT" for row in selected
            ),
            "margin_trades": sum(
                row["paper_mode"] == "MARGIN" for row in selected
            ),
            "gross_pnl_jpy": str(
                sum(
                    (
                        Decimal(str(row["gross_pnl_jpy"]))
                        for row in selected
                    ),
                    Decimal("0"),
                )
            ),
            "net_pnl_jpy": str(
                sum(
                    (
                        Decimal(str(row["net_pnl_jpy"]))
                        for row in selected
                    ),
                    Decimal("0"),
                )
            ),
            "fees_jpy": str(
                sum(
                    (Decimal(str(row["fees_jpy"])) for row in selected),
                    Decimal("0"),
                )
            ),
            "spread_cost_jpy": str(
                sum(
                    (
                        Decimal(str(row["spread_cost_jpy"]))
                        for row in selected
                    ),
                    Decimal("0"),
                )
            ),
            "adverse_cost_jpy": str(
                sum(
                    (
                        Decimal(str(row["adverse_cost_jpy"]))
                        for row in selected
                    ),
                    Decimal("0"),
                )
            ),
            "funding_interest_jpy": str(
                sum(
                    (
                        Decimal(str(row["funding_interest_jpy"]))
                        for row in selected
                    ),
                    Decimal("0"),
                )
            ),
            "exit_reasons": dict(reasons),
            "service_states": service_states,
            "per_trade_slack_posts": False,
        }

    def _service_states(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for mode in ("spot", "margin"):
            path = self.runtime_root / mode / "state.json"
            if not path.exists():
                result[mode] = {"status": "NOT_STARTED"}
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            metrics = payload.get("metrics", {})
            result[mode] = {
                "status": payload.get("status"),
                "run_id": payload.get("run_id"),
                "event_count": payload.get("events_processed", 0),
                "decisions": payload.get("actions", {}),
                "fills": payload.get("fills", 0),
                "profit_factor": metrics.get("profit_factor"),
                "max_drawdown_jpy": metrics.get("max_drawdown_jpy"),
                "equity_jpy": metrics.get("equity_jpy"),
                "open_positions": metrics.get("open_position_count", 0),
                "reject_skip_reasons": payload.get("reasons", {}),
                "guardian": payload.get("guardian", {}),
            }
        return result
