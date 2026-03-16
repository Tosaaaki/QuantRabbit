#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from generate_repo_history_lane_index import build_repo_history_lane_payload  # noqa: E402
from trade_findings_review import _entry_field, _parse_findings  # noqa: E402


DEFAULT_FINDINGS_PATH = REPO_ROOT / "docs" / "TRADE_FINDINGS.md"
DEFAULT_ARTIFACT_PATH = REPO_ROOT / "logs" / "change_preflight_latest.json"
DEFAULT_OUT_JSON = REPO_ROOT / "logs" / "improvement_gate_latest.json"
DEFAULT_OUT_MD = REPO_ROOT / "logs" / "improvement_gate_latest.md"
UNRESOLVED_VERDICTS = {"pending", "mixed"}
UNRESOLVED_STATUS = {"open", "in_progress"}
BADISH_VERDICTS = {"bad", "pending", "mixed"}
NOT_FIRED_VALUES = {"", "0", "none", "false", "no"}
MARKET_HOLD_WARNING_PREFIXES = ("tick_stale", "spread_wide", "data_lag_high", "market_closed")
ADVANCED_IDEA_PATTERNS = (
    re.compile(r"\bkalman\b"),
    re.compile(r"\bhmm\b"),
    re.compile(r"\brl\b"),
    re.compile(r"reinforcement learning"),
    re.compile(r"\bllm\b"),
    re.compile(r"feature engineering"),
    re.compile(r"generalized residual"),
    re.compile(r"\bpca residual\b"),
)
BASELINE_EVIDENCE_PATTERNS = (
    re.compile(r"\bbaseline\b"),
    re.compile(r"\breplay\b"),
    re.compile(r"walk[- ]forward"),
    re.compile(r"\bvalidation\b"),
    re.compile(r"\bcost\b"),
    re.compile(r"\bslippage\b"),
    re.compile(r"\bnet\b"),
    re.compile(r"\bsplit\b"),
    re.compile(r"\boos\b"),
)


@dataclass(frozen=True)
class Candidate:
    raw: str
    strategy: str
    surface: str
    primary_loss_driver: str
    idea: str


@dataclass(frozen=True)
class MatchSummary:
    heading: str
    hypothesis_key: str
    verdict: str
    status: str
    mechanism_fired: str
    primary_loss_driver: str
    why_not_same_as_last_time: str
    promotion_gate: str
    escalation_trigger: str
    score: int


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gate improvement proposals against open/pending TRADE_FINDINGS lanes."
    )
    parser.add_argument("--query", required=True)
    parser.add_argument(
        "--candidates",
        required=True,
        help=(
            "Candidate specs separated by '||'. "
            "Each spec must be 'strategy::surface::primary_loss_driver::idea'. "
            "The final ::idea segment is optional."
        ),
    )
    parser.add_argument("--path", default=str(DEFAULT_FINDINGS_PATH))
    parser.add_argument("--artifact", default=str(DEFAULT_ARTIFACT_PATH))
    parser.add_argument("--out-json", default=str(DEFAULT_OUT_JSON))
    parser.add_argument("--out-md", default=str(DEFAULT_OUT_MD))
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _compact(text: str, max_chars: int = 180) -> str:
    text = " ".join(str(text).split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _normalize(text: str) -> str:
    text = " ".join(str(text).split())
    if not text:
        return ""
    match = re.search(r"`([^`]+)`", text)
    if match:
        return match.group(1).strip()
    return text.lstrip("- ").strip()


def _blob(text: str) -> str:
    return " ".join(str(text).split()).lower()


def _tokenize_surface(text: str) -> list[str]:
    return [token.strip().lower() for token in text.split() if token.strip()]


def _strategy_slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def _strategy_related(candidate_strategy: str, lane_strategy: str) -> bool:
    candidate_slug = _strategy_slug(candidate_strategy)
    lane_slug = _strategy_slug(lane_strategy)
    if not candidate_slug or not lane_slug:
        return False
    return (
        candidate_slug == lane_slug
        or lane_slug.startswith(candidate_slug)
        or candidate_slug.startswith(lane_slug)
    )


def _parse_candidates(raw: str) -> list[Candidate]:
    items = [item.strip() for item in raw.split("||") if item.strip()]
    if not items:
        raise ValueError("at least one candidate is required")
    parsed: list[Candidate] = []
    for item in items:
        parts = [part.strip() for part in item.split("::")]
        if len(parts) < 3:
            raise ValueError(
                "candidate must be 'strategy::surface::primary_loss_driver::idea'"
            )
        strategy = parts[0]
        surface = parts[1]
        primary_loss_driver = parts[2]
        idea = "::".join(parts[3:]).strip() if len(parts) > 3 else ""
        if not strategy or not surface or not primary_loss_driver:
            raise ValueError(
                "candidate fields strategy/surface/primary_loss_driver must be non-empty"
            )
        parsed.append(
            Candidate(
                raw=item,
                strategy=strategy,
                surface=surface,
                primary_loss_driver=primary_loss_driver,
                idea=idea,
            )
        )
    return parsed


def _entry_blob(entry: Any) -> str:
    return _blob(f"{entry.heading}\n{entry.raw_text}")


def _entry_primary_blob(entry: Any) -> str:
    parts = [
        entry.heading,
        _entry_field(entry, "Hypothesis Key"),
        _entry_field(entry, "Change"),
        _entry_field(entry, "Improvement"),
    ]
    return _blob("\n".join(part for part in parts if part))


def _verdict(entry: Any) -> str:
    return _normalize(_entry_field(entry, "Verdict")).lower()


def _status(entry: Any) -> str:
    return _normalize(_entry_field(entry, "Status")).lower()


def _mechanism_fired(entry: Any) -> str:
    return _normalize(_entry_field(entry, "Mechanism Fired")).lower()


def _is_unresolved(entry: Any) -> bool:
    return _verdict(entry) in UNRESOLVED_VERDICTS or _status(entry) in UNRESOLVED_STATUS


def _is_badish(entry: Any) -> bool:
    return _verdict(entry) in BADISH_VERDICTS or _status(entry) in UNRESOLVED_STATUS


def _score_candidate_match(entry: Any, candidate: Candidate) -> int:
    haystack = _entry_blob(entry)
    primary_haystack = _entry_primary_blob(entry)
    strategy = candidate.strategy.lower()
    driver = candidate.primary_loss_driver.lower()
    surface = candidate.surface.lower()
    if strategy not in primary_haystack:
        return 0

    score = 5
    driver_hit = driver in haystack
    surface_phrase_hit = surface in haystack
    surface_token_hits = sum(
        1 for token in _tokenize_surface(candidate.surface) if token in haystack
    )
    idea_token_hits = sum(
        1 for token in _tokenize_surface(candidate.idea) if token in haystack
    )
    if driver_hit:
        score += 4
    if surface_phrase_hit:
        score += 4
    score += surface_token_hits
    score += min(2, idea_token_hits)

    if not driver_hit and not surface_phrase_hit and surface_token_hits == 0:
        return 0
    return score


def _family_match(entry: Any, candidate: Candidate) -> bool:
    haystack = _entry_primary_blob(entry)
    return (
        candidate.strategy.lower() in haystack
        and candidate.primary_loss_driver.lower() in haystack
    )


def _strategy_match(entry: Any, candidate: Candidate) -> bool:
    haystack = _entry_primary_blob(entry)
    return candidate.strategy.lower() in haystack


def _build_match_summary(entry: Any, score: int) -> MatchSummary:
    return MatchSummary(
        heading=entry.heading,
        hypothesis_key=_compact(_normalize(_entry_field(entry, "Hypothesis Key")), 120),
        verdict=_compact(_entry_field(entry, "Verdict"), 64),
        status=_compact(_entry_field(entry, "Status"), 64),
        mechanism_fired=_compact(_entry_field(entry, "Mechanism Fired"), 64),
        primary_loss_driver=_compact(_entry_field(entry, "Primary Loss Driver"), 120),
        why_not_same_as_last_time=_compact(
            _entry_field(entry, "Why Not Same As Last Time"), 180
        ),
        promotion_gate=_compact(_entry_field(entry, "Promotion Gate"), 180),
        escalation_trigger=_compact(_entry_field(entry, "Escalation Trigger"), 180),
        score=score,
    )


def _latest_entry_by_hypothesis(entries: list[Any]) -> dict[str, Any]:
    latest: dict[str, Any] = {}
    for entry in entries:
        key = _normalize(_entry_field(entry, "Hypothesis Key"))
        if key:
            latest[key] = entry
    return latest


def _lane_matches_candidate_strategy(lane: dict[str, Any], candidate: Candidate) -> bool:
    strategies = lane.get("strategies")
    if not isinstance(strategies, list):
        return False
    return any(
        _strategy_related(candidate.strategy, str(strategy))
        for strategy in strategies
        if str(strategy).strip()
    )


def _lane_matches_candidate_family(lane: dict[str, Any], candidate: Candidate) -> bool:
    if not _lane_matches_candidate_strategy(lane, candidate):
        return False
    lane_driver = _blob(str(lane.get("primary_loss_driver") or ""))
    return lane_driver == _blob(candidate.primary_loss_driver)


def _lane_summary(
    lane: dict[str, Any], latest_entry_by_hypothesis: dict[str, Any]
) -> dict[str, Any]:
    hypothesis_key = _normalize(str(lane.get("hypothesis_key") or ""))
    entry = latest_entry_by_hypothesis.get(hypothesis_key)
    mechanism_fired = _mechanism_fired(entry) if entry is not None else ""
    return {
        "hypothesis_key": hypothesis_key,
        "strategies": lane.get("strategies") or [],
        "family_key": str(lane.get("family_key") or ""),
        "primary_loss_driver": str(lane.get("primary_loss_driver") or ""),
        "latest_heading": str(lane.get("latest_heading") or ""),
        "latest_verdict": str(lane.get("latest_verdict") or ""),
        "latest_status": str(lane.get("latest_status") or ""),
        "repeat_risk": str(lane.get("repeat_risk") or ""),
        "history_commit_count": int(lane.get("history_commit_count") or 0),
        "mechanism_fired": mechanism_fired,
    }


def _market_status(artifact: dict[str, Any] | None) -> tuple[str, list[str]]:
    if not artifact:
        return ("unknown", ["missing_change_preflight_artifact"])
    warnings = [
        str(item)
        for item in artifact.get("warnings", [])
        if isinstance(item, str) and item.strip()
    ]
    hold_reasons = [
        warning
        for warning in warnings
        if warning.startswith(MARKET_HOLD_WARNING_PREFIXES)
    ]
    market = artifact.get("market") if isinstance(artifact.get("market"), dict) else {}
    market_open = market.get("market_open")
    seconds_until_open = market.get("seconds_until_open")
    if market_open is False:
        if isinstance(seconds_until_open, (int, float)):
            return ("market_hold", [f"market_closed:{seconds_until_open:.1f}s_to_open"])
        return ("market_hold", ["market_closed"])
    tick_age_sec = market.get("tick_age_sec")
    spread_pips = market.get("spread_pips")
    data_lag_ms = market.get("data_lag_ms")
    if isinstance(tick_age_sec, (int, float)) and tick_age_sec > 300:
        hold_reasons.append(f"tick_stale:{tick_age_sec:.1f}s")
    if isinstance(spread_pips, (int, float)) and spread_pips > 1.2:
        hold_reasons.append(f"spread_wide:{spread_pips:.2f}p")
    if isinstance(data_lag_ms, (int, float)) and data_lag_ms > 1500:
        hold_reasons.append(f"data_lag_high:{data_lag_ms:.1f}ms")
    if hold_reasons:
        return ("market_hold", sorted(set(hold_reasons)))
    return ("normal", [])


def _candidate_complexity_signals(candidate: Candidate) -> list[str]:
    text = _blob(" ".join([candidate.surface, candidate.idea]))
    signals: list[str] = []
    for pattern in ADVANCED_IDEA_PATTERNS:
        match = pattern.search(text)
        if match:
            token = match.group(0).strip().lower()
            if token not in signals:
                signals.append(token)
    return signals


def _candidate_has_baseline_evidence(candidate: Candidate) -> bool:
    text = _blob(" ".join([candidate.surface, candidate.idea]))
    return any(pattern.search(text) for pattern in BASELINE_EVIDENCE_PATTERNS)


def _entry_strategies(entry: Any) -> list[str]:
    matches = re.findall(r"`([^`]+)`", entry.heading)
    return [match.strip() for match in matches if match.strip()]


def _fallback_lane_payload(entries: list[Any], limit: int) -> dict[str, Any]:
    current_open: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    family_counts: dict[str, int] = {}

    for entry in entries:
        strategies = _entry_strategies(entry) or ["trade_findings_lane"]
        family_key = (
            f"{' + '.join(strategies)} :: "
            f"{_normalize(_entry_field(entry, 'Primary Loss Driver'))}"
        )
        family_counts[family_key] = family_counts.get(family_key, 0) + 1

    for entry in reversed(entries):
        if not _is_unresolved(entry):
            continue
        hypothesis_key = _normalize(_entry_field(entry, "Hypothesis Key"))
        if hypothesis_key and hypothesis_key in seen_keys:
            continue
        if hypothesis_key:
            seen_keys.add(hypothesis_key)
        strategies = _entry_strategies(entry) or ["trade_findings_lane"]
        primary_loss_driver = _normalize(_entry_field(entry, "Primary Loss Driver"))
        family_key = f"{' + '.join(strategies)} :: {primary_loss_driver}"
        family_entries = family_counts.get(family_key, 1)
        current_open.append(
            {
                "hypothesis_key": hypothesis_key,
                "strategies": strategies,
                "family_key": family_key,
                "primary_loss_driver": primary_loss_driver,
                "latest_heading": entry.heading,
                "latest_verdict": _verdict(entry),
                "latest_status": _status(entry),
                "history_commit_count": 0,
                "repeat_risk": "high" if family_entries >= 2 else "low",
            }
        )

    recommended_focus = current_open[0] if current_open else None
    if isinstance(recommended_focus, dict):
        recommended_focus = {
            **recommended_focus,
            "repeat_risk_reasons": [],
            "selection_reason": "fallback unresolved trading lane first",
            "next_action": _compact(_entry_field(entries[-1], "Next Action"), 160)
            if entries
            else "",
        }

    return {
        "current_open_trading_lanes": current_open[: max(limit, 1)],
        "recommended_single_focus_lane": recommended_focus,
    }


def build_improvement_gate_payload(
    *,
    findings_path: Path,
    artifact_path: Path,
    query: str,
    candidates_raw: str,
    limit: int,
) -> dict[str, Any]:
    candidates = _parse_candidates(candidates_raw)
    entries = _parse_findings(findings_path)
    latest_entry_by_hypothesis = _latest_entry_by_hypothesis(entries)
    artifact = _load_json(artifact_path)
    try:
        lane_payload = build_repo_history_lane_payload(findings_path, max(limit, 20))
    except ValueError:
        lane_payload = _fallback_lane_payload(entries, max(limit, 20))
    current_open_trading = (
        lane_payload.get("current_open_trading_lanes")
        if isinstance(lane_payload.get("current_open_trading_lanes"), list)
        else []
    )
    recommended_focus = (
        lane_payload.get("recommended_single_focus_lane")
        if isinstance(lane_payload.get("recommended_single_focus_lane"), dict)
        else None
    )
    market_status, market_reasons = _market_status(artifact)

    candidate_results: list[dict[str, Any]] = []
    blocked = False

    for candidate in candidates:
        complexity_signals = _candidate_complexity_signals(candidate)
        has_baseline_evidence = _candidate_has_baseline_evidence(candidate)
        scored_matches: list[tuple[int, Any]] = []
        family_entries: list[Any] = []
        strategy_entries: list[Any] = []
        for entry in entries:
            score = _score_candidate_match(entry, candidate)
            if score > 0:
                scored_matches.append((score, entry))
            if _family_match(entry, candidate):
                family_entries.append(entry)
            if _strategy_match(entry, candidate):
                strategy_entries.append(entry)
        scored_matches.sort(key=lambda item: (-item[0], item[1].order))
        matched_entries = [_build_match_summary(entry, score) for score, entry in scored_matches]
        unresolved_matches = [
            entry for _, entry in scored_matches if _is_unresolved(entry)
        ]
        unresolved_summaries = [
            _build_match_summary(entry, score)
            for score, entry in scored_matches
            if _is_unresolved(entry)
        ]
        family_badish_count = sum(1 for entry in family_entries if _is_badish(entry))
        family_unresolved_count = sum(1 for entry in family_entries if _is_unresolved(entry))
        strategy_badish_count = sum(1 for entry in strategy_entries if _is_badish(entry))
        strategy_unresolved_count = sum(
            1 for entry in strategy_entries if _is_unresolved(entry)
        )
        mechanism_not_fired_pending = any(
            _mechanism_fired(entry) in NOT_FIRED_VALUES for entry in unresolved_matches
        )
        same_strategy_open_lanes = [
            _lane_summary(lane, latest_entry_by_hypothesis)
            for lane in current_open_trading
            if isinstance(lane, dict) and _lane_matches_candidate_strategy(lane, candidate)
        ]
        same_family_open_lanes = [
            lane
            for lane in same_strategy_open_lanes
            if _blob(lane.get("primary_loss_driver") or "")
            == _blob(candidate.primary_loss_driver)
        ]
        stubborn_open_lanes = [
            lane
            for lane in same_strategy_open_lanes
            if _blob(lane.get("mechanism_fired") or "") in NOT_FIRED_VALUES
        ]

        reasons: list[str] = []
        if market_status == "market_hold":
            action = "market_hold_review_only"
            reasons.extend(market_reasons)
        elif unresolved_matches:
            if len(unresolved_matches) >= 2 and (
                mechanism_not_fired_pending or strategy_badish_count >= 2
            ):
                action = "escalate_family_not_tighten"
                reasons.append(
                    "multiple same-surface or same-query pending lanes are still unresolved"
                )
            else:
                action = "review_existing_pending"
                reasons.append("same-surface or same-query pending lane exists")
            if mechanism_not_fired_pending:
                reasons.append("pending lane still has Mechanism Fired=0/none")
        elif same_family_open_lanes:
            if len(same_family_open_lanes) >= 2 or family_badish_count >= 2:
                action = "escalate_family_not_tighten"
                reasons.append(
                    "same strategy/driver family already has repeated unresolved trading lanes"
                )
            else:
                action = "review_existing_pending"
                reasons.append(
                    "same strategy/driver family already has unresolved trading lane"
                )
            if any(
                _blob(lane.get("mechanism_fired") or "") in NOT_FIRED_VALUES
                for lane in same_family_open_lanes
            ):
                reasons.append("same family open lane still has Mechanism Fired=0/none")
        elif stubborn_open_lanes:
            if len(stubborn_open_lanes) >= 2 or strategy_badish_count >= 2:
                action = "escalate_family_not_tighten"
                reasons.append("same strategy has repeated unresolved lanes with Mechanism Fired=0/none")
            else:
                action = "review_existing_pending"
                reasons.append("same strategy has unresolved lane with Mechanism Fired=0/none")
            if recommended_focus and same_strategy_open_lanes:
                reasons.append(
                    "validate recommended single-focus lane before opening another same-strategy tweak"
                )
        elif same_strategy_open_lanes:
            action = "review_existing_pending"
            reasons.append("same strategy already has unresolved trading lane")
            if recommended_focus:
                reasons.append(
                    "validate recommended single-focus lane before opening another same-strategy tweak"
                )
        elif complexity_signals and not has_baseline_evidence:
            action = "baseline_before_complexity"
            reasons.append(
                "advanced-model idea detected before baseline/cost/validation evidence"
            )
            reasons.extend(f"advanced_keyword:{signal}" for signal in complexity_signals)
        elif family_badish_count >= 2 or strategy_badish_count >= 2:
            action = "escalate_family_not_tighten"
            reasons.append(
                "same strategy or strategy/driver family already has repeated bad-or-pending entries"
            )
        else:
            action = "allow_new_lane"
            reasons.append("no unresolved overlap found for this candidate")

        if action != "allow_new_lane":
            blocked = True

        candidate_results.append(
            {
                "candidate": asdict(candidate),
                "action": action,
                "reasons": reasons,
                "same_surface_matches": [
                    asdict(item) for item in matched_entries[: max(limit, 1)]
                ],
                "same_surface_unresolved": [
                    asdict(item) for item in unresolved_summaries[: max(limit, 1)]
                ],
                "complexity_signals": complexity_signals,
                "has_baseline_evidence": has_baseline_evidence,
                "same_strategy_open_lanes": same_strategy_open_lanes[: max(limit, 1)],
                "same_family_open_lanes": same_family_open_lanes[: max(limit, 1)],
                "family_summary": {
                    "family_bad_or_pending_count": family_badish_count,
                    "family_unresolved_count": family_unresolved_count,
                    "strategy_bad_or_pending_count": strategy_badish_count,
                    "strategy_unresolved_count": strategy_unresolved_count,
                },
            }
        )

    return {
        "generated_at": artifact.get("generated_at") if artifact else "",
        "query": query,
        "findings_path": str(findings_path),
        "artifact_path": str(artifact_path),
        "artifact_query": artifact.get("query") if artifact else None,
        "market_status": market_status,
        "market_reasons": market_reasons,
        "blocked": blocked,
        "recommended_single_focus_lane": recommended_focus,
        "candidates": candidate_results,
    }


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Improvement Proposal Gate",
        "",
        f"- query: `{payload['query']}`",
        f"- market_status: `{payload['market_status']}`",
        f"- blocked: `{payload['blocked']}`",
    ]
    if payload.get("market_reasons"):
        lines.append(
            "- market_reasons: "
            + ", ".join(f"`{reason}`" for reason in payload["market_reasons"])
        )
    for idx, item in enumerate(payload.get("candidates", []), start=1):
        candidate = item["candidate"]
        lines.extend(
            [
                "",
                f"## {idx}. `{candidate['strategy']}`",
                f"- surface: `{candidate['surface']}`",
                f"- primary_loss_driver: `{candidate['primary_loss_driver']}`",
                f"- idea: `{candidate['idea'] or 'n/a'}`",
                f"- action: `{item['action']}`",
                "- reasons: " + ", ".join(f"`{reason}`" for reason in item["reasons"]),
                f"- complexity_signals: `{', '.join(item['complexity_signals']) or 'none'}`",
                f"- has_baseline_evidence: `{item['has_baseline_evidence']}`",
                f"- family_bad_or_pending_count: `{item['family_summary']['family_bad_or_pending_count']}`",
                f"- family_unresolved_count: `{item['family_summary']['family_unresolved_count']}`",
                f"- strategy_bad_or_pending_count: `{item['family_summary']['strategy_bad_or_pending_count']}`",
                f"- strategy_unresolved_count: `{item['family_summary']['strategy_unresolved_count']}`",
            ]
        )
    return "\n".join(lines) + "\n"


def _print_human(payload: dict[str, Any]) -> None:
    print("IMPROVEMENT proposal gate")
    print(f"query={payload['query']}")
    print(f"market_status={payload['market_status']}")
    if payload.get("market_reasons"):
        print("market_reasons=" + ", ".join(payload["market_reasons"]))
    print(f"blocked={payload['blocked']}")
    print("")
    for idx, item in enumerate(payload.get("candidates", []), start=1):
        candidate = item["candidate"]
        print(
            f"{idx}. {candidate['strategy']} / {candidate['surface']} / "
            f"{candidate['primary_loss_driver']}"
        )
        print(f"   action: {item['action']}")
        print(f"   reasons: {', '.join(item['reasons'])}")
        print(
            "   complexity: "
            f"signals={','.join(item['complexity_signals']) or 'none'} "
            f"baseline_evidence={item['has_baseline_evidence']}"
        )
        print(
            "   family: "
            f"bad_or_pending={item['family_summary']['family_bad_or_pending_count']} "
            f"unresolved={item['family_summary']['family_unresolved_count']}"
        )
        print(
            "   strategy: "
            f"bad_or_pending={item['family_summary']['strategy_bad_or_pending_count']} "
            f"unresolved={item['family_summary']['strategy_unresolved_count']}"
        )
        if item["same_surface_unresolved"]:
            print("   unresolved_matches:")
            for match in item["same_surface_unresolved"]:
                print(
                    "   - "
                    f"{match['heading']} / key={match['hypothesis_key'] or 'n/a'} / "
                    f"verdict={match['verdict'] or 'n/a'} / "
                    f"mechanism_fired={match['mechanism_fired'] or 'n/a'}"
                )
        if item["same_strategy_open_lanes"]:
            print("   same_strategy_open_lanes:")
            for lane in item["same_strategy_open_lanes"]:
                print(
                    "   - "
                    f"{lane['hypothesis_key'] or 'n/a'} / "
                    f"driver={lane['primary_loss_driver'] or 'n/a'} / "
                    f"repeat_risk={lane['repeat_risk'] or 'n/a'} / "
                    f"mechanism_fired={lane['mechanism_fired'] or 'n/a'}"
                )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    payload = build_improvement_gate_payload(
        findings_path=Path(args.path).resolve(),
        artifact_path=Path(args.artifact).resolve(),
        query=args.query,
        candidates_raw=args.candidates,
        limit=max(args.limit, 1),
    )

    out_json = Path(args.out_json).resolve()
    out_md = Path(args.out_md).resolve()
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(_render_markdown(payload), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        _print_human(payload)
        print("")
        print(f"artifact_json={out_json}")
        print(f"artifact_md={out_md}")

    return 2 if payload.get("blocked") else 0


if __name__ == "__main__":
    raise SystemExit(main())
