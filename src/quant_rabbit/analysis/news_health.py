"""Health checks for the live news-digest pipeline.

This module audits whether the out-of-band news routine is producing fresh,
usable trader evidence and whether the derived market-story profile has caught
up with that evidence. It is observation-only: no broker calls, no order
permission, and no strategy score changes.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - kept for old local Python.
    tomllib = None  # type: ignore[assignment]

from quant_rabbit.analysis.news import build_news_snapshot


JST = timezone(timedelta(hours=9))

# (a) Maximum tolerated age for the hourly curated digest during the open FX
#     week.
# (b) Constant because qr-news-digest is scheduled hourly; 150 minutes allows
#     one missed run plus clock/network jitter while still preventing a prior
#     session narrative from steering live entries.
# (c) Replace with scheduler-derived next-run metadata if Codex exposes it.
ACTIVE_NEWS_MAX_AGE_MINUTES = 150

# (a) Minimum fresh structured RSS/API items expected during the open FX week.
# (b) Constant because the configured public sources normally provide dozens of
#     FX/macro items per day; fewer than this means the trader is probably
#     reading a thin or broken feed rather than a true cross-market packet.
# (c) Replace with a source-specific quorum once licensed feeds are added.
ACTIVE_MIN_STRUCTURED_ITEMS = 8

# (a) Minimum source diversity for the structured news packet.
# (b) Constant because one public feed can be editorially narrow or delayed; two
#     independent sources are the minimum cross-check before calling a digest
#     broad enough for macro nowcasts.
# (c) Replace with explicit source weights if provider SLAs become available.
ACTIVE_MIN_SOURCE_COUNT = 2

REQUIRED_DIGEST_SECTIONS = (
    "High Impact",
    "Watch List",
    "Economic Calendar Today",
    "Pre-Event Nowcast",
    "Central Bank Tracker",
    "Pair-Specific Notes",
    "Risk Events",
)

HIGH_MEDIUM_IMPACTS = {"HIGH", "MEDIUM", "VERY_HIGH", "RED", "ORANGE"}
DIGEST_JST_STAMP_RE = re.compile(r"FX News Digest\s+[-—]\s+(\d{4}-\d{2}-\d{2})\s+(\d{2}):(\d{2})\s+JST")


@dataclass(frozen=True)
class NewsHealthCheck:
    name: str
    status: str
    severity: str
    message: str
    evidence: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "severity": self.severity,
            "message": self.message,
            "evidence": self.evidence,
        }


def build_news_health(
    *,
    news_items_path: Path,
    digest_path: Path,
    flow_log_path: Path,
    market_story_profile_path: Path,
    calendar_path: Path | None = None,
    automation_path: Path | None = None,
    weekend_state_path: Path | None = None,
    now_utc: datetime | None = None,
    verify_fetch: bool = False,
) -> dict[str, Any]:
    now = _aware_utc(now_utc or datetime.now(timezone.utc))
    checks: list[NewsHealthCheck] = []
    market_window = _market_window(now, weekend_state_path=weekend_state_path)
    active_market = market_window == "ACTIVE"

    news_payload = _load_json(news_items_path)
    digest_text = _read_text(digest_path)
    flow_text = _read_text(flow_log_path)
    profile_payload = _load_json(market_story_profile_path)
    calendar_payload = _load_json(calendar_path) if calendar_path is not None else None

    checks.append(
        _file_check(
            "news_items_file",
            news_items_path,
            news_payload is not None,
            active_market=active_market,
            block_when_missing=False,
        )
    )
    checks.append(
        _file_check(
            "news_digest_file",
            digest_path,
            digest_text is not None,
            active_market=active_market,
            block_when_missing=True,
        )
    )
    checks.append(
        _file_check(
            "news_flow_file",
            flow_log_path,
            flow_text is not None,
            active_market=active_market,
            block_when_missing=False,
        )
    )
    checks.append(
        _file_check(
            "market_story_profile_file",
            market_story_profile_path,
            profile_payload is not None,
            active_market=active_market,
            block_when_missing=True,
        )
    )

    checks.extend(_news_items_checks(news_payload, news_items_path, now, active_market=active_market))
    checks.extend(
        _digest_checks(
            digest_text,
            digest_path,
            now,
            active_market=active_market,
            structured_news_fallback_ok=_structured_news_fallback_ok(
                news_payload,
                now,
                active_market=active_market,
            ),
        )
    )
    checks.extend(_flow_checks(flow_text, flow_log_path, now, active_market=active_market))
    checks.extend(
        _market_story_checks(
            profile_payload,
            market_story_profile_path,
            (news_items_path, digest_path, flow_log_path),
            active_market=active_market,
        )
    )
    checks.extend(_calendar_checks(calendar_payload, now, digest_text, active_market=active_market))
    checks.append(_automation_check(automation_path, market_window=market_window, active_market=active_market))
    if verify_fetch:
        checks.append(_verify_fetch_check(active_market=active_market))

    status = _overall_status(checks)
    issues = [f"{check.severity}:{check.name}:{check.message}" for check in checks if check.status != "OK"]
    return {
        "generated_at_utc": now.isoformat(),
        "schema_version": 1,
        "status": status,
        "market_window": market_window,
        "active_news_max_age_minutes": ACTIVE_NEWS_MAX_AGE_MINUTES,
        "checks": [check.to_dict() for check in checks],
        "issues": issues,
    }


def write_news_health_report(payload: dict[str, Any], report_path: Path) -> None:
    lines = [
        "# News Health",
        "",
        f"- Generated at UTC: `{payload.get('generated_at_utc')}`",
        f"- Status: `{payload.get('status')}`",
        f"- Market window: `{payload.get('market_window')}`",
        "",
        "## Checks",
        "",
        "| Check | Status | Severity | Message |",
        "|---|---|---|---|",
    ]
    for check in payload.get("checks") or []:
        if not isinstance(check, dict):
            continue
        lines.append(
            "| `{}` | `{}` | `{}` | {} |".format(
                check.get("name"),
                check.get("status"),
                check.get("severity"),
                str(check.get("message") or "").replace("|", "/"),
            )
        )
    issues = [str(item) for item in payload.get("issues") or [] if str(item).strip()]
    if issues:
        lines.extend(["", "## Issues", ""])
        lines.extend(f"- {issue}" for issue in issues)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n")


def _news_items_checks(
    payload: dict[str, Any] | None,
    path: Path,
    now: datetime,
    *,
    active_market: bool,
) -> list[NewsHealthCheck]:
    if payload is None:
        return []
    checks: list[NewsHealthCheck] = []
    generated_at = _parse_utc(payload.get("generated_at_utc"))
    age_minutes = _age_minutes(now, generated_at)
    if active_market and (age_minutes is None or age_minutes > ACTIVE_NEWS_MAX_AGE_MINUTES):
        checks.append(
            _check(
                "news_items_freshness",
                "BLOCK",
                f"structured news is stale or missing timestamp ({age_minutes} minutes old)",
                path=str(path),
                generated_at_utc=payload.get("generated_at_utc"),
            )
        )
    else:
        checks.append(
            _ok(
                "news_items_freshness",
                "structured news timestamp is usable",
                path=str(path),
                generated_at_utc=payload.get("generated_at_utc"),
                age_minutes=age_minutes,
            )
        )

    items = payload.get("items") if isinstance(payload.get("items"), list) else []
    fresh_items = _fresh_news_items(items, now)
    sources = {str(item.get("source") or "").strip() for item in fresh_items if isinstance(item, dict)}
    sources.discard("")
    if active_market and len(fresh_items) < ACTIVE_MIN_STRUCTURED_ITEMS:
        checks.append(
            _check(
                "news_items_count",
                "WARN",
                f"only {len(fresh_items)} fresh structured news items",
                fresh_items=len(fresh_items),
                retained_items=len(items),
            )
        )
    else:
        checks.append(_ok("news_items_count", "fresh structured item count is usable", fresh_items=len(fresh_items)))

    if active_market and len(sources) < ACTIVE_MIN_SOURCE_COUNT:
        checks.append(
            _check(
                "news_source_diversity",
                "WARN",
                f"only {len(sources)} fresh structured news source(s)",
                sources=sorted(sources),
            )
        )
    else:
        checks.append(_ok("news_source_diversity", "structured news has source diversity", sources=sorted(sources)))

    issues = [str(issue) for issue in payload.get("issues") or [] if str(issue).strip()]
    blocking_feed_issue = any(
        issue.startswith(("MISSING_FRESH_NEWS_FEED", "STALE_NEWS_FEED")) for issue in issues
    )
    if active_market and blocking_feed_issue:
        checks.append(_check("news_feed_issues", "WARN", "structured feed is stale; rely on curated WebSearch digest", issues=issues))
    elif issues:
        checks.append(_check("news_feed_issues", "WARN", "news feed packet has source issue(s)", issues=issues))
    else:
        checks.append(_ok("news_feed_issues", "news feed packet has no source issues"))
    return checks


def _digest_checks(
    text: str | None,
    path: Path,
    now: datetime,
    *,
    active_market: bool,
    structured_news_fallback_ok: bool = False,
) -> list[NewsHealthCheck]:
    if text is None:
        return []
    checks: list[NewsHealthCheck] = []
    digest_ts = _digest_timestamp(text, path)
    age_minutes = _age_minutes(now, digest_ts)
    if active_market and (age_minutes is None or age_minutes > ACTIVE_NEWS_MAX_AGE_MINUTES):
        checks.append(
            _check(
                "news_digest_freshness",
                "BLOCK",
                f"curated digest is stale or missing timestamp ({age_minutes} minutes old)",
                path=str(path),
                digest_timestamp_utc=digest_ts.isoformat() if digest_ts else None,
            )
        )
    else:
        checks.append(
            _ok(
                "news_digest_freshness",
                "curated digest timestamp is usable",
                path=str(path),
                digest_timestamp_utc=digest_ts.isoformat() if digest_ts else None,
                age_minutes=age_minutes,
            )
        )

    missing_sections = [section for section in REQUIRED_DIGEST_SECTIONS if section.lower() not in text.lower()]
    if missing_sections:
        severity = "BLOCK" if active_market else "WARN"
        checks.append(
            _check(
                "news_digest_sections",
                severity,
                "curated digest is missing required section(s)",
                missing_sections=missing_sections,
            )
        )
    else:
        checks.append(_ok("news_digest_sections", "curated digest contains all required sections"))

    if "WebSearch" not in text:
        severity = "BLOCK" if active_market and not structured_news_fallback_ok else "WARN"
        checks.append(
            _check(
                "news_digest_websearch",
                severity,
                (
                    "digest does not cite WebSearch; using fresh structured public RSS fallback"
                    if structured_news_fallback_ok
                    else "digest does not cite WebSearch; raw RSS alone is not enough"
                ),
                path=str(path),
            )
        )
    else:
        checks.append(_ok("news_digest_websearch", "digest cites WebSearch-curated research"))
    return checks


def _structured_news_fallback_ok(
    payload: dict[str, Any] | None,
    now: datetime,
    *,
    active_market: bool,
) -> bool:
    if not active_market or payload is None:
        return False
    generated_at = _parse_utc(payload.get("generated_at_utc"))
    age_minutes = _age_minutes(now, generated_at)
    if age_minutes is None or age_minutes > ACTIVE_NEWS_MAX_AGE_MINUTES:
        return False
    items = payload.get("items") if isinstance(payload.get("items"), list) else []
    fresh_items = _fresh_news_items(items, now)
    sources = {str(item.get("source") or "").strip() for item in fresh_items if isinstance(item, dict)}
    sources.discard("")
    return len(fresh_items) >= ACTIVE_MIN_STRUCTURED_ITEMS and len(sources) >= ACTIVE_MIN_SOURCE_COUNT


def _flow_checks(
    text: str | None,
    path: Path,
    now: datetime,
    *,
    active_market: bool,
) -> list[NewsHealthCheck]:
    if text is None:
        return []
    mtime = _path_mtime(path)
    age_minutes = _age_minutes(now, mtime)
    if active_market and (age_minutes is None or age_minutes > ACTIVE_NEWS_MAX_AGE_MINUTES):
        return [
            _check(
                "news_flow_freshness",
                "WARN",
                f"news flow log is stale or missing mtime ({age_minutes} minutes old)",
                path=str(path),
            )
        ]
    return [_ok("news_flow_freshness", "news flow log timestamp is usable", path=str(path), age_minutes=age_minutes)]


def _market_story_checks(
    payload: dict[str, Any] | None,
    profile_path: Path,
    news_paths: Iterable[Path],
    *,
    active_market: bool,
) -> list[NewsHealthCheck]:
    if payload is None:
        return []
    existing_news = [path for path in news_paths if path.exists()]
    profile_mtime = _path_mtime(profile_path)
    latest_news_mtime = max((_path_mtime(path) for path in existing_news), default=None)
    if active_market and latest_news_mtime is not None and (
        profile_mtime is None or profile_mtime < latest_news_mtime
    ):
        return [
            _check(
                "market_story_news_sync",
                "BLOCK",
                "market_story_profile is older than news artifacts",
                profile_mtime_utc=profile_mtime.isoformat() if profile_mtime else None,
                latest_news_mtime_utc=latest_news_mtime.isoformat(),
            )
        ]
    return [
        _ok(
            "market_story_news_sync",
            "market_story_profile is synchronized with news artifacts",
            profile_mtime_utc=profile_mtime.isoformat() if profile_mtime else None,
            latest_news_mtime_utc=latest_news_mtime.isoformat() if latest_news_mtime else None,
        )
    ]


def _calendar_checks(
    payload: dict[str, Any] | None,
    now: datetime,
    digest_text: str | None,
    *,
    active_market: bool,
) -> list[NewsHealthCheck]:
    if payload is None:
        return [_check("calendar_context", "WARN", "calendar artifact missing; event nowcast coverage cannot be audited")]
    generated_at = _parse_utc(payload.get("generated_at_utc"))
    age_hours = None if generated_at is None else round((now - generated_at).total_seconds() / 3600.0, 2)
    events = _upcoming_high_medium_events(payload, now)
    age_minutes = _age_minutes(now, generated_at)
    if active_market and (age_minutes is None or age_minutes > ACTIVE_NEWS_MAX_AGE_MINUTES):
        checks: list[NewsHealthCheck] = [
            _check(
                "calendar_context",
                "BLOCK",
                "calendar artifact is stale; pre-event news coverage cannot be trusted",
                generated_at_utc=payload.get("generated_at_utc"),
                age_hours=age_hours,
            )
        ]
    else:
        checks = [
            _ok(
                "calendar_context",
                "calendar artifact read for news-health event coverage",
                generated_at_utc=payload.get("generated_at_utc"),
                age_hours=age_hours,
                upcoming_high_medium_events=len(events),
            )
        ]
    if events and digest_text is not None and "Pre-Event Nowcast".lower() not in digest_text.lower():
        severity = "BLOCK" if active_market else "WARN"
        checks.append(
            _check(
                "pre_event_nowcast_section",
                severity,
                "high/medium event(s) in next 48h but digest lacks Pre-Event Nowcast",
                events=events[:8],
            )
        )
    else:
        checks.append(
            _ok(
                "pre_event_nowcast_section",
                "digest pre-event nowcast coverage is structurally present",
                events=events[:8],
            )
        )
    return checks


def _automation_check(
    automation_path: Path | None,
    *,
    market_window: str,
    active_market: bool,
) -> NewsHealthCheck:
    if automation_path is None:
        return _check("news_automation_state", "WARN", "automation path not supplied")
    payload = _load_toml(automation_path)
    if payload is None:
        return _check("news_automation_state", "WARN", "automation file unreadable", path=str(automation_path))
    status = str(payload.get("status") or "").upper()
    rrule = str(payload.get("rrule") or "")
    cwd = payload.get("cwds")
    if active_market and status != "ACTIVE":
        return _check(
            "news_automation_state",
            "BLOCK",
            f"qr-news-digest automation is {status or 'UNKNOWN'} during active market",
            path=str(automation_path),
            rrule=rrule,
            cwds=cwd,
        )
    if market_window == "WEEKEND_PAUSED" and status == "PAUSED":
        return _ok(
            "news_automation_state",
            "qr-news-digest is paused by weekend guard",
            path=str(automation_path),
            rrule=rrule,
            cwds=cwd,
        )
    return _ok(
        "news_automation_state",
        "qr-news-digest automation state is usable",
        path=str(automation_path),
        status=status,
        rrule=rrule,
        cwds=cwd,
    )


def _verify_fetch_check(*, active_market: bool) -> NewsHealthCheck:
    try:
        snapshot = build_news_snapshot()
    except Exception as exc:  # pragma: no cover - scheduled-task boundary.
        severity = "BLOCK" if active_market else "WARN"
        return _check("news_fetch_probe", severity, f"live news fetch probe failed: {exc}")
    issues = list(snapshot.issues)
    if active_market and (not snapshot.items or issues):
        return _check(
            "news_fetch_probe",
            "WARN",
            "live news fetch probe returned a thin or imperfect packet; curated WebSearch digest remains primary",
            items=len(snapshot.items),
            issues=issues,
        )
    if issues:
        return _check("news_fetch_probe", "WARN", "live news fetch probe has issue(s)", items=len(snapshot.items), issues=issues)
    return _ok("news_fetch_probe", "live news fetch probe succeeded", items=len(snapshot.items))


def _market_window(now_utc: datetime, *, weekend_state_path: Path | None) -> str:
    jst = now_utc.astimezone(JST)
    if _inside_weekend_pause_window(jst):
        state = _load_json(weekend_state_path) if weekend_state_path is not None else None
        if isinstance(state, dict) and state.get("mode") == "paused":
            return "WEEKEND_PAUSED"
        return "WEEKEND_CLOSED"
    return "ACTIVE"


def _inside_weekend_pause_window(jst: datetime) -> bool:
    weekday = jst.weekday()  # Monday=0
    if weekday == 5:
        return jst.time() >= time(6, 0)
    if weekday == 6:
        return True
    if weekday == 0:
        return jst.time() < time(7, 0)
    return False


def _fresh_news_items(items: list[Any], now: datetime) -> list[dict[str, Any]]:
    fresh: list[dict[str, Any]] = []
    cutoff = now - timedelta(hours=24)
    for item in items:
        if not isinstance(item, dict):
            continue
        published = _parse_utc(item.get("published_at_utc"))
        if published is None or published > now or published < cutoff:
            continue
        fresh.append(item)
    return fresh


def _upcoming_high_medium_events(payload: dict[str, Any], now: datetime) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    until = now + timedelta(hours=48)
    for event in payload.get("events") or payload.get("calendar") or []:
        if not isinstance(event, dict):
            continue
        impact = str(event.get("impact") or event.get("importance") or "").upper()
        if impact not in HIGH_MEDIUM_IMPACTS:
            continue
        ts = _parse_utc(event.get("timestamp_utc") or event.get("time_utc") or event.get("time") or event.get("timestamp"))
        if ts is None or ts < now or ts > until:
            continue
        events.append(
            {
                "timestamp_utc": ts.isoformat(),
                "currency": event.get("currency") or event.get("country"),
                "impact": impact,
                "title": event.get("title") or event.get("event"),
                "forecast": event.get("forecast"),
                "previous": event.get("previous"),
            }
        )
    return sorted(events, key=lambda item: str(item.get("timestamp_utc")))


def _file_check(
    name: str,
    path: Path,
    exists: bool,
    *,
    active_market: bool,
    block_when_missing: bool,
) -> NewsHealthCheck:
    if exists:
        return _ok(name, "artifact is present", path=str(path))
    severity = "BLOCK" if active_market and block_when_missing else "WARN"
    return _check(name, severity, "artifact is missing or unreadable", path=str(path))


def _overall_status(checks: Iterable[NewsHealthCheck]) -> str:
    statuses = [check for check in checks if check.status != "OK"]
    if any(check.severity == "BLOCK" for check in statuses):
        return "BLOCK"
    if statuses:
        return "WARN"
    return "OK"


def _check(name: str, severity: str, message: str, **evidence: Any) -> NewsHealthCheck:
    return NewsHealthCheck(name=name, status="ISSUE", severity=severity, message=message, evidence=evidence)


def _ok(name: str, message: str, **evidence: Any) -> NewsHealthCheck:
    return NewsHealthCheck(name=name, status="OK", severity="INFO", message=message, evidence=evidence)


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _load_toml(path: Path) -> dict[str, Any] | None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    if tomllib is not None:
        try:
            return tomllib.loads(text)
        except tomllib.TOMLDecodeError:
            return None
    return _load_simple_toml(text)


def _load_simple_toml(text: str) -> dict[str, Any] | None:
    payload: dict[str, Any] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            return None
        key, raw_value = line.split("=", 1)
        key = key.strip()
        value = raw_value.strip()
        if not key:
            return None
        if value.startswith('"') and value.endswith('"'):
            payload[key] = value[1:-1]
        elif value.startswith("[") and value.endswith("]"):
            items: list[str] = []
            inner = value[1:-1].strip()
            if inner:
                for part in inner.split(","):
                    item = part.strip()
                    if not (item.startswith('"') and item.endswith('"')):
                        return None
                    items.append(item[1:-1])
            payload[key] = items
        else:
            try:
                payload[key] = int(value)
            except ValueError:
                return None
    return payload


def _path_mtime(path: Path) -> datetime | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except OSError:
        return None


def _digest_timestamp(text: str, path: Path) -> datetime | None:
    match = DIGEST_JST_STAMP_RE.search(text)
    if match:
        try:
            parsed = datetime.fromisoformat(f"{match.group(1)}T{match.group(2)}:{match.group(3)}:00+09:00")
        except ValueError:
            parsed = None
        if parsed is not None:
            return parsed.astimezone(timezone.utc)
    return _path_mtime(path)


def _parse_utc(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _age_minutes(now: datetime, then: datetime | None) -> float | None:
    if then is None or then > now:
        return None
    return round((now - then).total_seconds() / 60.0, 1)
