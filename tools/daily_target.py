#!/usr/bin/env python3
"""Read-only UTC daily target engine for operator session starts.

This tool does not stage, send, cancel, or close OANDA orders. It reads the
local broker snapshot and execution ledger, persists the first seen NAV for the
UTC trading day, and reports progress against the base +5% target plus the
explicitly gated +10% extension target.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from quant_rabbit.capital_flows import funding_adjusted_equity, summarize_capital_flows

DEFAULT_SNAPSHOT_PATH = REPO_ROOT / "data" / "broker_snapshot.json"
DEFAULT_EXECUTION_LEDGER_DB = REPO_ROOT / "data" / "execution_ledger.db"
DEFAULT_DAY_START_DIR = REPO_ROOT / "logs" / "day_start_nav"
DEFAULT_CAPITAL_FLOWS_PATH = REPO_ROOT / "data" / "capital_flows.json"

# Contract-defined operating thresholds, not tuned market constants:
# +5% is the base daily target, +10% is an extension target only when the
# favorable-market extension gate is explicit, +2% starts ATTACK mode, and
# -1.5% is the damage-control boundary requested by the operator.
BUILD_TO_ATTACK_PCT = 2.0
BASE_TARGET_PCT = 5.0
EXTENSION_TARGET_PCT = 10.0
DAMAGE_CONTROL_PCT = -1.5
ROLLING_30D_POLICY = "ROLLING_30D_4X"
ROLLING_30D_CALENDAR_DAYS = 30
ROLLING_30D_ACTIVE_DAYS = 22
ROLLING_30D_TARGET_MULTIPLIER = 4.0
ROLLING_30D_ON_PACE_TOLERANCE = 0.98
ROLLING_30D_DANGER_DAILY_RETURN_PCT = 10.0

CLOSE_EVENT_TYPES = ("TRADE_CLOSED", "TRADE_REDUCED")


@dataclass(frozen=True)
class DailyTargetMetrics:
    trading_day_utc: str
    day_start_utc: str
    day_end_utc: str
    generated_at_utc: str
    snapshot_fetched_at_utc: str | None
    day_start_nav: float
    current_nav: float
    realized_pl_today: float
    unrealized_pl: float
    total_day_progress_yen: float
    total_day_progress_pct: float
    rolling_30d_policy: str
    rolling_30d_start_utc: str
    rolling_30d_end_utc: str
    rolling_30d_start_equity: float
    current_equity_raw: float
    capital_flows_30d: float
    capital_flow_count_30d: int
    funding_adjusted_equity: float
    current_equity: float
    rolling_30d_multiplier_raw: float
    rolling_30d_multiplier_funding_adjusted: float
    current_30d_multiplier: float
    remaining_to_4x_raw: float
    remaining_to_4x_funding_adjusted: float
    remaining_to_4x: float
    required_calendar_daily_return_raw: float | None
    required_active_day_return_raw: float | None
    required_calendar_daily_return_funding_adjusted: float | None
    required_active_day_return_funding_adjusted: float | None
    required_calendar_daily_return: float | None
    required_active_day_return: float | None
    performance_basis: str
    sizing_basis: str
    pace_state: str
    capital_flow_issues: tuple[str, ...]
    base_target_yen: float
    extension_target_yen: float
    remaining_to_5pct_yen: float
    remaining_to_10pct_yen: float
    margin_used: float
    margin_pct: float
    mode: str
    extension_gate: bool
    day_start_nav_source: str
    day_start_nav_path: str
    issues: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def utc_day_bounds(now_utc: datetime | None = None) -> tuple[date, datetime, datetime]:
    now = _normalize_utc(now_utc)
    trading_day = now.date()
    start = datetime.combine(trading_day, time.min, tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    return trading_day, start, end


def compute_daily_target(
    *,
    snapshot_path: Path = DEFAULT_SNAPSHOT_PATH,
    execution_ledger_db: Path = DEFAULT_EXECUTION_LEDGER_DB,
    day_start_dir: Path = DEFAULT_DAY_START_DIR,
    capital_flows_path: Path = DEFAULT_CAPITAL_FLOWS_PATH,
    now_utc: datetime | None = None,
    dry_run: bool = False,
    extension_gate: bool = False,
) -> DailyTargetMetrics:
    now = _normalize_utc(now_utc)
    trading_day, day_start, day_end = utc_day_bounds(now)
    snapshot = _load_json(snapshot_path)
    account = snapshot.get("account")
    if not isinstance(account, dict):
        raise ValueError(f"broker snapshot lacks account summary: {snapshot_path}")

    current_nav = _required_float(account, "nav_jpy", context="broker snapshot account")
    unrealized_pl = _float(account.get("unrealized_pl_jpy"), default=0.0)
    margin_used = _float(account.get("margin_used_jpy"), default=0.0)
    snapshot_fetched_at = _optional_str(snapshot.get("fetched_at_utc") or account.get("fetched_at_utc"))

    issues: list[str] = []
    if snapshot_fetched_at:
        fetched = _parse_datetime(snapshot_fetched_at)
        if fetched is not None and fetched.date() != trading_day:
            issues.append("SNAPSHOT_NOT_FROM_CURRENT_UTC_TRADING_DAY")

    day_start_nav, day_start_source, day_start_path, day_start_issues = _resolve_day_start_nav(
        trading_day=trading_day,
        day_start=day_start,
        current_nav=current_nav,
        snapshot_fetched_at=snapshot_fetched_at,
        account=account,
        day_start_dir=day_start_dir,
        dry_run=dry_run,
    )
    issues.extend(day_start_issues)

    realized_pl_today, realized_issues = _realized_pl_today(
        execution_ledger_db=execution_ledger_db,
        trading_day=trading_day,
    )
    issues.extend(realized_issues)

    progress_yen = round(current_nav - day_start_nav, 4)
    progress_pct = round((progress_yen / day_start_nav) * 100.0, 4) if day_start_nav > 0 else 0.0
    rolling = _rolling_30d_policy(
        current_equity=current_nav,
        day_start_dir=day_start_dir,
        capital_flows_path=capital_flows_path,
        now=now,
        dry_run=dry_run,
    )
    base_target = round(day_start_nav * (BASE_TARGET_PCT / 100.0), 4)
    extension_target = round(day_start_nav * (EXTENSION_TARGET_PCT / 100.0), 4)
    margin_pct = round((margin_used / current_nav) * 100.0, 4) if current_nav > 0 else 0.0
    mode = _mode(progress_pct=progress_pct, extension_gate=extension_gate)

    return DailyTargetMetrics(
        trading_day_utc=trading_day.isoformat(),
        day_start_utc=day_start.isoformat(),
        day_end_utc=day_end.isoformat(),
        generated_at_utc=now.isoformat(),
        snapshot_fetched_at_utc=snapshot_fetched_at,
        day_start_nav=round(day_start_nav, 4),
        current_nav=round(current_nav, 4),
        realized_pl_today=round(realized_pl_today, 4),
        unrealized_pl=round(unrealized_pl, 4),
        total_day_progress_yen=progress_yen,
        total_day_progress_pct=progress_pct,
        rolling_30d_policy=ROLLING_30D_POLICY,
        rolling_30d_start_utc=rolling["rolling_30d_start_utc"],
        rolling_30d_end_utc=rolling["rolling_30d_end_utc"],
        rolling_30d_start_equity=rolling["rolling_30d_start_equity"],
        current_equity_raw=rolling["current_equity_raw"],
        capital_flows_30d=rolling["capital_flows_30d"],
        capital_flow_count_30d=rolling["capital_flow_count_30d"],
        funding_adjusted_equity=rolling["funding_adjusted_equity"],
        current_equity=rolling["current_equity"],
        rolling_30d_multiplier_raw=rolling["rolling_30d_multiplier_raw"],
        rolling_30d_multiplier_funding_adjusted=rolling["rolling_30d_multiplier_funding_adjusted"],
        current_30d_multiplier=rolling["current_30d_multiplier"],
        remaining_to_4x_raw=rolling["remaining_to_4x_raw"],
        remaining_to_4x_funding_adjusted=rolling["remaining_to_4x_funding_adjusted"],
        remaining_to_4x=rolling["remaining_to_4x"],
        required_calendar_daily_return_raw=rolling["required_calendar_daily_return_raw"],
        required_active_day_return_raw=rolling["required_active_day_return_raw"],
        required_calendar_daily_return_funding_adjusted=rolling[
            "required_calendar_daily_return_funding_adjusted"
        ],
        required_active_day_return_funding_adjusted=rolling[
            "required_active_day_return_funding_adjusted"
        ],
        required_calendar_daily_return=rolling["required_calendar_daily_return"],
        required_active_day_return=rolling["required_active_day_return"],
        performance_basis=rolling["performance_basis"],
        sizing_basis=rolling["sizing_basis"],
        pace_state=rolling["pace_state"],
        capital_flow_issues=rolling["capital_flow_issues"],
        base_target_yen=base_target,
        extension_target_yen=extension_target,
        remaining_to_5pct_yen=round(max(0.0, base_target - progress_yen), 4),
        remaining_to_10pct_yen=round(max(0.0, extension_target - progress_yen), 4),
        margin_used=round(margin_used, 4),
        margin_pct=margin_pct,
        mode=mode,
        extension_gate=extension_gate,
        day_start_nav_source=day_start_source,
        day_start_nav_path=str(day_start_path),
        issues=tuple(issues),
    )


def format_daily_target_block(metrics: DailyTargetMetrics) -> str:
    lines = [
        "## DAILY TARGET ENGINE",
        "Trading day basis: UTC 00:00",
        f"Day-start NAV: {_format_jpy(metrics.day_start_nav)}",
        f"Current NAV: {_format_jpy(metrics.current_nav)}",
        f"Realized P/L today: {_format_signed_jpy(metrics.realized_pl_today)}",
        f"Unrealized P/L: {_format_signed_jpy(metrics.unrealized_pl)}",
        "Total day progress: "
        f"{_format_signed_jpy(metrics.total_day_progress_yen)} ({metrics.total_day_progress_pct:+.2f}%)",
        "## ROLLING 30D 4X POLICY",
        f"Rolling 30d start equity: {_format_jpy(metrics.rolling_30d_start_equity)}",
        f"current_equity_raw: {_format_jpy(metrics.current_equity_raw)}",
        f"capital_flows_30d: {_format_signed_jpy(metrics.capital_flows_30d)}",
        f"funding_adjusted_equity: {_format_jpy(metrics.funding_adjusted_equity)}",
        f"rolling_30d_multiplier_raw: {metrics.rolling_30d_multiplier_raw:.4f}x",
        f"rolling_30d_multiplier_funding_adjusted: {metrics.rolling_30d_multiplier_funding_adjusted:.4f}x",
        f"remaining_to_4x_raw: {_format_jpy(metrics.remaining_to_4x_raw)}",
        f"remaining_to_4x_funding_adjusted: {_format_jpy(metrics.remaining_to_4x_funding_adjusted)}",
        f"Current 30d multiplier: {metrics.current_30d_multiplier:.4f}x (funding-adjusted)",
        f"Remaining to 4x: {_format_jpy(metrics.remaining_to_4x)} (funding-adjusted)",
        "required_calendar_daily_return_raw: "
        + (
            f"{metrics.required_calendar_daily_return_raw:.4f}%"
            if metrics.required_calendar_daily_return_raw is not None
            else "n/a"
        ),
        "required_active_day_return_raw: "
        + (
            f"{metrics.required_active_day_return_raw:.4f}%"
            if metrics.required_active_day_return_raw is not None
            else "n/a"
        ),
        "required_calendar_daily_return_funding_adjusted: "
        + (
            f"{metrics.required_calendar_daily_return_funding_adjusted:.4f}%"
            if metrics.required_calendar_daily_return_funding_adjusted is not None
            else "n/a"
        ),
        "required_active_day_return_funding_adjusted: "
        + (
            f"{metrics.required_active_day_return_funding_adjusted:.4f}%"
            if metrics.required_active_day_return_funding_adjusted is not None
            else "n/a"
        ),
        "Required calendar daily return: "
        + (
            f"{metrics.required_calendar_daily_return:.4f}%"
            if metrics.required_calendar_daily_return is not None
            else "n/a"
        ),
        "Required active-day return: "
        + (
            f"{metrics.required_active_day_return:.4f}%"
            if metrics.required_active_day_return is not None
            else "n/a"
        ),
        f"performance_basis: {metrics.performance_basis}",
        f"sizing_basis: {metrics.sizing_basis}",
        f"Pace state: {metrics.pace_state}",
        "## DAILY PACE MARKER",
        f"Base target +5%: {_format_jpy(metrics.base_target_yen)}",
        f"Remaining to +5%: {_format_jpy(metrics.remaining_to_5pct_yen)}",
        f"Extension target +10%: {_format_jpy(metrics.extension_target_yen)}",
        f"Remaining to +10%: {_format_jpy(metrics.remaining_to_10pct_yen)}",
        f"Mode: {metrics.mode}",
    ]
    if metrics.issues:
        lines.append("Issues: " + ", ".join(metrics.issues))
    if metrics.capital_flow_issues:
        lines.append("Capital flow issues: " + ", ".join(metrics.capital_flow_issues))
    return "\n".join(lines)


def _resolve_day_start_nav(
    *,
    trading_day: date,
    day_start: datetime,
    current_nav: float,
    snapshot_fetched_at: str | None,
    account: dict[str, Any],
    day_start_dir: Path,
    dry_run: bool,
) -> tuple[float, str, Path, list[str]]:
    path = day_start_dir / f"{trading_day.isoformat()}.json"
    if path.exists():
        payload = _load_json(path)
        return (
            _required_float(payload, "day_start_nav", context=str(path)),
            str(payload.get("source") or "logs/day_start_nav"),
            path,
            [],
        )

    issues: list[str] = []
    nav = current_nav
    source = "first_seen_current_nav"
    record = {
        "trading_day_utc": trading_day.isoformat(),
        "day_start_utc": day_start.isoformat(),
        "captured_at_utc": _normalize_utc().isoformat(),
        "day_start_nav": round(nav, 4),
        "source": source,
        "snapshot_fetched_at_utc": snapshot_fetched_at,
        "account_last_transaction_id": str(account.get("last_transaction_id") or ""),
    }
    if dry_run:
        issues.append("DAY_START_NAV_DRY_RUN_NOT_PERSISTED")
        return nav, "dry_run_first_seen_current_nav", path, issues

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return nav, source, path, issues


def _rolling_30d_policy(
    *,
    current_equity: float,
    day_start_dir: Path,
    capital_flows_path: Path,
    now: datetime,
    dry_run: bool,
) -> dict[str, Any]:
    path = day_start_dir / "rolling_30d_4x.json"
    start_time: datetime | None = None
    start_equity: float | None = None
    if path.exists():
        payload = _load_json(path)
        start_time = _parse_datetime(str(payload.get("rolling_30d_start_utc") or ""))
        try:
            start_equity = _required_float(payload, "rolling_30d_start_equity", context=str(path))
        except ValueError:
            start_equity = None
    if (
        start_time is None
        or start_equity is None
        or start_equity <= 0
        or start_time > now
        or (now - start_time) >= timedelta(days=ROLLING_30D_CALENDAR_DAYS)
    ):
        start_time = now
        start_equity = current_equity
        if not dry_run:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "rolling_30d_policy": ROLLING_30D_POLICY,
                        "rolling_30d_start_utc": start_time.isoformat(),
                        "rolling_30d_start_equity": round(start_equity, 4),
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )

    end_time = start_time + timedelta(days=ROLLING_30D_CALENDAR_DAYS)
    elapsed_days = max(0.0, (now - start_time).total_seconds() / 86400.0)
    remaining_calendar = max(0.0, ROLLING_30D_CALENDAR_DAYS - elapsed_days)
    remaining_active = remaining_calendar * (ROLLING_30D_ACTIVE_DAYS / ROLLING_30D_CALENDAR_DAYS)
    target_equity = start_equity * ROLLING_30D_TARGET_MULTIPLIER
    capital_flows = summarize_capital_flows(
        capital_flows_path,
        start_utc=start_time,
        end_utc=now,
    )
    current_equity_raw = round(current_equity, 4)
    adjusted_equity = funding_adjusted_equity(current_equity_raw, capital_flows.net_amount_jpy)
    multiplier_raw = current_equity_raw / start_equity if start_equity > 0 else 0.0
    multiplier_adjusted = adjusted_equity / start_equity if start_equity > 0 else 0.0
    remaining_to_4x_raw = max(0.0, target_equity - current_equity_raw)
    remaining_to_4x_adjusted = max(0.0, target_equity - adjusted_equity)
    required_calendar_raw = _required_compound_return_pct(
        current_value=current_equity_raw,
        target_value=target_equity,
        remaining_periods=remaining_calendar,
    )
    required_active_raw = _required_compound_return_pct(
        current_value=current_equity_raw,
        target_value=target_equity,
        remaining_periods=remaining_active,
    )
    required_calendar_adjusted = _required_compound_return_pct(
        current_value=adjusted_equity,
        target_value=target_equity,
        remaining_periods=remaining_calendar,
    )
    required_active_adjusted = _required_compound_return_pct(
        current_value=adjusted_equity,
        target_value=target_equity,
        remaining_periods=remaining_active,
    )
    expected_multiplier = ROLLING_30D_TARGET_MULTIPLIER ** (
        min(elapsed_days, ROLLING_30D_CALENDAR_DAYS) / ROLLING_30D_CALENDAR_DAYS
    )
    return {
        "rolling_30d_start_utc": start_time.isoformat(),
        "rolling_30d_end_utc": end_time.isoformat(),
        "rolling_30d_start_equity": round(start_equity, 4),
        "current_equity_raw": current_equity_raw,
        "capital_flows_30d": capital_flows.net_amount_jpy,
        "capital_flow_count_30d": capital_flows.count,
        "funding_adjusted_equity": adjusted_equity,
        "current_equity": adjusted_equity,
        "rolling_30d_multiplier_raw": round(multiplier_raw, 6),
        "rolling_30d_multiplier_funding_adjusted": round(multiplier_adjusted, 6),
        "current_30d_multiplier": round(multiplier_adjusted, 6),
        "remaining_to_4x_raw": round(remaining_to_4x_raw, 4),
        "remaining_to_4x_funding_adjusted": round(remaining_to_4x_adjusted, 4),
        "remaining_to_4x": round(remaining_to_4x_adjusted, 4),
        "required_calendar_daily_return_raw": required_calendar_raw,
        "required_active_day_return_raw": required_active_raw,
        "required_calendar_daily_return_funding_adjusted": required_calendar_adjusted,
        "required_active_day_return_funding_adjusted": required_active_adjusted,
        "required_calendar_daily_return": required_calendar_adjusted,
        "required_active_day_return": required_active_adjusted,
        "performance_basis": "funding_adjusted",
        "sizing_basis": "raw_nav",
        "pace_state": _rolling_pace_state(
            current_multiplier=multiplier_adjusted,
            expected_multiplier=expected_multiplier,
            required_calendar_daily_return=required_calendar_adjusted,
            remaining_to_4x=remaining_to_4x_adjusted,
        ),
        "capital_flow_issues": capital_flows.issues,
    }


def _required_compound_return_pct(
    *,
    current_value: float,
    target_value: float,
    remaining_periods: float,
) -> float | None:
    if current_value <= 0 or target_value <= 0:
        return None
    if current_value >= target_value:
        return 0.0
    if remaining_periods <= 0:
        return None
    return round(((target_value / current_value) ** (1.0 / remaining_periods) - 1.0) * 100.0, 6)


def _rolling_pace_state(
    *,
    current_multiplier: float,
    expected_multiplier: float,
    required_calendar_daily_return: float | None,
    remaining_to_4x: float,
) -> str:
    if remaining_to_4x <= 0 or current_multiplier >= expected_multiplier:
        return "AHEAD"
    if current_multiplier >= expected_multiplier * ROLLING_30D_ON_PACE_TOLERANCE:
        return "ON_PACE"
    if (
        required_calendar_daily_return is not None
        and required_calendar_daily_return > ROLLING_30D_DANGER_DAILY_RETURN_PCT
    ):
        return "DANGER"
    return "BEHIND"


def _realized_pl_today(*, execution_ledger_db: Path, trading_day: date) -> tuple[float, list[str]]:
    if not execution_ledger_db.exists():
        return 0.0, ["EXECUTION_LEDGER_MISSING"]
    placeholders = ",".join("?" for _ in CLOSE_EVENT_TYPES)
    query = f"""
        SELECT COALESCE(SUM(COALESCE(realized_pl_jpy, 0.0)), 0.0)
        FROM execution_events
        WHERE event_type IN ({placeholders})
          AND substr(ts_utc, 1, 10) = ?
    """
    try:
        with sqlite3.connect(f"file:{execution_ledger_db}?mode=ro", uri=True) as conn:
            row = conn.execute(query, (*CLOSE_EVENT_TYPES, trading_day.isoformat())).fetchone()
    except sqlite3.Error as exc:
        return 0.0, [f"EXECUTION_LEDGER_READ_ERROR:{exc}"]
    return float((row or [0.0])[0] or 0.0), []


def _mode(*, progress_pct: float, extension_gate: bool) -> str:
    if progress_pct <= DAMAGE_CONTROL_PCT:
        return "DAMAGE_CONTROL"
    if progress_pct < BUILD_TO_ATTACK_PCT:
        return "BUILD"
    if progress_pct < BASE_TARGET_PCT:
        return "ATTACK"
    if extension_gate:
        return "EXTEND"
    return "PROTECT"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise ValueError(f"missing JSON file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON file: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file is not an object: {path}")
    return payload


def _required_float(payload: dict[str, Any], key: str, *, context: str) -> float:
    value = payload.get(key)
    if value is None or value == "":
        raise ValueError(f"{context} lacks required numeric field {key}")
    return float(value)


def _float(value: object, *, default: float) -> float:
    if value is None or value == "":
        return default
    return float(value)


def _optional_str(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _parse_datetime(value: str) -> datetime | None:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return _normalize_utc(parsed)


def _normalize_utc(value: datetime | None = None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _format_jpy(value: float) -> str:
    return f"{value:,.0f} JPY"


def _format_signed_jpy(value: float) -> str:
    return f"{value:+,.0f} JPY"


def _extension_gate_from_env() -> bool:
    raw = os.environ.get("QR_DAILY_TARGET_EXTENSION_GATE", "").strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Print the UTC daily target engine block.")
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT_PATH)
    parser.add_argument("--execution-ledger-db", type=Path, default=DEFAULT_EXECUTION_LEDGER_DB)
    parser.add_argument("--day-start-dir", type=Path, default=DEFAULT_DAY_START_DIR)
    parser.add_argument("--capital-flows", type=Path, default=DEFAULT_CAPITAL_FLOWS_PATH)
    parser.add_argument("--dry-run", action="store_true", help="Do not persist a missing day-start NAV record.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON instead of the session block.")
    parser.add_argument(
        "--extension-gate",
        choices=("yes", "no"),
        default=None,
        help="Explicit favorable-market extension gate. Defaults to QR_DAILY_TARGET_EXTENSION_GATE or no.",
    )
    args = parser.parse_args()

    extension_gate = _extension_gate_from_env() if args.extension_gate is None else args.extension_gate == "yes"
    metrics = compute_daily_target(
        snapshot_path=args.snapshot,
        execution_ledger_db=args.execution_ledger_db,
        day_start_dir=args.day_start_dir,
        capital_flows_path=args.capital_flows,
        dry_run=args.dry_run,
        extension_gate=extension_gate,
    )
    if args.json:
        print(json.dumps(metrics.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(format_daily_target_block(metrics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
