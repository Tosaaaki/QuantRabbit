#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workers.common import agent_whiteboard as whiteboard


UTC = timezone.utc
JST = timezone(timedelta(hours=9))
DEFAULT_TASK = "trade_findings_draft review"


def _resolve_path(value: str | Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _safe_str(value: Any) -> str:
    return str(value or "").strip()


def _parse_dt(raw: Any) -> datetime | None:
    text = _safe_str(raw)
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _iso_utc(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).isoformat()


def _iso_jst(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(JST).isoformat()


def _now_utc() -> datetime:
    return datetime.now(UTC)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False))
        fh.write("\n")


def _relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except Exception:
        return str(path)


def _top_ranked(pdca: dict[str, Any], bucket: str) -> dict[str, Any]:
    rankings = pdca.get("trades", {}).get("24h", {}).get("rankings", {})
    section = rankings.get("by_strategy_net_jpy")
    if not isinstance(section, dict):
        return {}
    items = section.get(bucket)
    if not isinstance(items, list) or not items or not isinstance(items[0], dict):
        return {}
    return items[0]


def _recommendation_lines(counterfactual: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for item in counterfactual.get("recommendations") or []:
        if not isinstance(item, dict):
            continue
        parts = [
            _safe_str(item.get("strategy_tag") or item.get("strategy") or "generic"),
            _safe_str(item.get("action") or item.get("recommendation") or "review"),
            _safe_str(item.get("reason")),
        ]
        parts = [part for part in parts if part]
        if parts:
            out.append(" / ".join(parts))
        if len(out) >= 3:
            break
    return out


def _accepted_update_lines(replay_gate: dict[str, Any]) -> list[str]:
    auto_improve = replay_gate.get("auto_improve")
    if not isinstance(auto_improve, dict):
        return []
    out: list[str] = []
    for item in auto_improve.get("accepted_updates") or []:
        if not isinstance(item, dict):
            continue
        parts = [
            _safe_str(item.get("strategy") or item.get("worker") or item.get("strategy_tag")),
            _safe_str(item.get("field") or item.get("key")),
            _safe_str(item.get("value")),
        ]
        parts = [part for part in parts if part]
        if parts:
            out.append(" / ".join(parts))
        if len(out) >= 3:
            break
    return out


@dataclass(frozen=True)
class Config:
    health_path: Path
    pdca_path: Path
    strategy_feedback_path: Path
    trade_counterfactual_path: Path
    replay_quality_gate_path: Path
    participation_alloc_path: Path
    market_context_path: Path
    out_json: Path
    out_history: Path
    out_md: Path
    whiteboard_enabled: bool
    whiteboard_db: Path
    whiteboard_task: str
    whiteboard_author: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a TRADE_FINDINGS review draft from local artifacts.")
    parser.add_argument("--health-path", default=os.getenv("TRADE_FINDINGS_DRAFT_HEALTH_PATH", "logs/health_snapshot.json"))
    parser.add_argument("--pdca-path", default=os.getenv("TRADE_FINDINGS_DRAFT_PDCA_PATH", "logs/pdca_profitability_latest.json"))
    parser.add_argument(
        "--strategy-feedback-path",
        default=os.getenv("TRADE_FINDINGS_DRAFT_STRATEGY_FEEDBACK_PATH", "logs/strategy_feedback.json"),
    )
    parser.add_argument(
        "--trade-counterfactual-path",
        default=os.getenv("TRADE_FINDINGS_DRAFT_COUNTERFACTUAL_PATH", "logs/trade_counterfactual_latest.json"),
    )
    parser.add_argument(
        "--replay-quality-gate-path",
        default=os.getenv("TRADE_FINDINGS_DRAFT_REPLAY_PATH", "logs/replay_quality_gate_latest.json"),
    )
    parser.add_argument(
        "--participation-alloc-path",
        default=os.getenv("TRADE_FINDINGS_DRAFT_PARTICIPATION_ALLOC_PATH", "config/participation_alloc.json"),
    )
    parser.add_argument(
        "--market-context-path",
        default=os.getenv("TRADE_FINDINGS_DRAFT_MARKET_CONTEXT_PATH", "logs/market_context_latest.json"),
    )
    parser.add_argument("--out-json", default=os.getenv("TRADE_FINDINGS_DRAFT_OUT_JSON", "logs/trade_findings_draft_latest.json"))
    parser.add_argument(
        "--out-history",
        default=os.getenv("TRADE_FINDINGS_DRAFT_OUT_HISTORY", "logs/trade_findings_draft_history.jsonl"),
    )
    parser.add_argument("--out-md", default=os.getenv("TRADE_FINDINGS_DRAFT_OUT_MD", "logs/trade_findings_draft_latest.md"))
    parser.add_argument(
        "--whiteboard-db",
        default=os.getenv("TRADE_FINDINGS_DRAFT_WHITEBOARD_DB", "logs/agent_whiteboard.db"),
    )
    parser.add_argument(
        "--whiteboard-task",
        default=os.getenv("TRADE_FINDINGS_DRAFT_WHITEBOARD_TASK", DEFAULT_TASK),
    )
    parser.add_argument(
        "--whiteboard-author",
        default=os.getenv("TRADE_FINDINGS_DRAFT_WHITEBOARD_AUTHOR", os.getenv("USER", "trade-findings-draft")),
    )
    parser.add_argument(
        "--whiteboard-enabled",
        dest="whiteboard_enabled",
        action="store_true",
        default=_env_bool("TRADE_FINDINGS_DRAFT_WHITEBOARD_ENABLED", True),
    )
    parser.add_argument("--no-whiteboard", dest="whiteboard_enabled", action="store_false")
    parser.add_argument("--disable-whiteboard", dest="whiteboard_enabled", action="store_false")
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> Config:
    return Config(
        health_path=_resolve_path(args.health_path),
        pdca_path=_resolve_path(args.pdca_path),
        strategy_feedback_path=_resolve_path(args.strategy_feedback_path),
        trade_counterfactual_path=_resolve_path(args.trade_counterfactual_path),
        replay_quality_gate_path=_resolve_path(args.replay_quality_gate_path),
        participation_alloc_path=_resolve_path(args.participation_alloc_path),
        market_context_path=_resolve_path(args.market_context_path),
        out_json=_resolve_path(args.out_json),
        out_history=_resolve_path(args.out_history),
        out_md=_resolve_path(args.out_md),
        whiteboard_enabled=bool(args.whiteboard_enabled),
        whiteboard_db=_resolve_path(args.whiteboard_db),
        whiteboard_task=str(args.whiteboard_task),
        whiteboard_author=str(args.whiteboard_author),
    )


def build_report(cfg: Config) -> dict[str, Any]:
    generated_at = _now_utc()
    health = _load_json(cfg.health_path)
    pdca = _load_json(cfg.pdca_path)
    strategy_feedback = _load_json(cfg.strategy_feedback_path)
    counterfactual = _load_json(cfg.trade_counterfactual_path)
    replay_gate = _load_json(cfg.replay_quality_gate_path)
    participation_alloc = _load_json(cfg.participation_alloc_path)
    market_context = _load_json(cfg.market_context_path)

    mechanism = health.get("mechanism_integrity") if isinstance(health.get("mechanism_integrity"), dict) else {}
    missing = [str(item) for item in mechanism.get("missing_mechanisms") or [] if str(item).strip()]

    pricing = pdca.get("oanda", {}).get("pricing", {}) if isinstance(pdca.get("oanda"), dict) else {}
    account = pdca.get("oanda", {}).get("summary", {}) if isinstance(pdca.get("oanda"), dict) else {}
    trades_24h = pdca.get("trades", {}).get("24h", {}).get("overall", {}) if isinstance(pdca.get("trades"), dict) else {}
    top_loser = _top_ranked(pdca, "top_losers")
    top_winner = _top_ranked(pdca, "top_winners")
    feedback_strategies = strategy_feedback.get("strategies") if isinstance(strategy_feedback.get("strategies"), dict) else {}
    counter_summary = counterfactual.get("summary") if isinstance(counterfactual.get("summary"), dict) else {}
    accepted_updates = _accepted_update_lines(replay_gate)
    recommendation_lines = _recommendation_lines(counterfactual)
    participation_strategies = participation_alloc.get("strategies") if isinstance(participation_alloc.get("strategies"), dict) else {}

    signals: list[str] = []
    if missing:
        signals.append(f"mechanism_missing:{','.join(missing)}")
    if _safe_float(trades_24h.get("net_jpy")) < 0:
        signals.append("profitability_24h_net_negative")
    if _safe_float(trades_24h.get("pf_jpy")) < 1.0 and _safe_int(trades_24h.get("trades")) >= 20:
        signals.append("profitability_24h_pf_below_1")
    if _safe_float(top_loser.get("net_jpy")) < -1.0:
        signals.append(f"top_loser:{_safe_str(top_loser.get('strategy_tag')) or 'unknown'}")
    gate_status = _safe_str(replay_gate.get("gate_status")).lower()
    if gate_status and gate_status not in {"ok", "pass", "soft_skip", "skipped"}:
        signals.append(f"replay_quality_gate:{gate_status}")
    if accepted_updates:
        signals.append("replay_auto_improve_updates")
    if recommendation_lines:
        signals.append("counterfactual_recommendations")

    fact_lines: list[str] = []
    if pricing:
        fact_lines.append(
            "market: "
            f"USD/JPY bid={pricing.get('bid')} ask={pricing.get('ask')} spread={pricing.get('spread_pips')}p "
            f"(pricing_ok={pricing.get('meta', {}).get('ok') if isinstance(pricing.get('meta'), dict) else None})"
        )
    if health:
        fact_lines.append(
            "health: "
            f"data_lag_ms={round(_safe_float(health.get('data_lag_ms')), 1)} "
            f"decision_latency_ms={round(_safe_float(health.get('decision_latency_ms')), 1)} "
            f"mechanism_ok={bool(mechanism.get('ok'))}"
        )
    if missing:
        fact_lines.append("missing_mechanisms: " + ", ".join(missing))
    if trades_24h:
        fact_lines.append(
            "profitability_24h: "
            f"trades={_safe_int(trades_24h.get('trades'))} "
            f"net_jpy={round(_safe_float(trades_24h.get('net_jpy')), 3)} "
            f"pf_jpy={round(_safe_float(trades_24h.get('pf_jpy')), 4)} "
            f"win_rate={round(_safe_float(trades_24h.get('win_rate')) * 100.0, 2)}%"
        )
    if top_loser:
        fact_lines.append(
            f"top_loser_24h: {_safe_str(top_loser.get('strategy_tag'))} "
            f"net_jpy={round(_safe_float(top_loser.get('net_jpy')), 3)} "
            f"trades={_safe_int(top_loser.get('trades'))} "
            f"pf_jpy={round(_safe_float(top_loser.get('pf_jpy')), 4)}"
        )
    if top_winner:
        fact_lines.append(
            f"top_winner_24h: {_safe_str(top_winner.get('strategy_tag'))} "
            f"net_jpy={round(_safe_float(top_winner.get('net_jpy')), 3)} "
            f"trades={_safe_int(top_winner.get('trades'))} "
            f"pf_jpy={round(_safe_float(top_winner.get('pf_jpy')), 4)}"
        )
    if feedback_strategies:
        fact_lines.append(f"strategy_feedback: strategies={len(feedback_strategies)} updated_at={strategy_feedback.get('updated_at')}")
    if counter_summary:
        fact_lines.append(
            "counterfactual: "
            f"trades={_safe_int(counter_summary.get('trades'))} "
            f"mean_pips={round(_safe_float(counter_summary.get('mean_pips')), 3)} "
            f"stuck_trade_ratio={round(_safe_float(counter_summary.get('stuck_trade_ratio')), 4)}"
        )
    if replay_gate:
        fact_lines.append(
            "replay_quality_gate: "
            f"status={replay_gate.get('gate_status')} "
            f"failing_workers={len(replay_gate.get('failing_workers') or [])} "
            f"accepted_updates={len(accepted_updates)}"
        )
    if participation_strategies:
        fact_lines.append(f"participation_alloc: strategies={len(participation_strategies)} as_of={participation_alloc.get('as_of')}")
    if market_context:
        event_count = len(market_context.get("events") or []) if isinstance(market_context.get("events"), list) else 0
        fact_lines.append(f"market_context: generated_at={market_context.get('generated_at')} events={event_count}")
    if account:
        fact_lines.append(
            f"account: nav_jpy={account.get('nav_jpy')} margin_rate={account.get('margin_rate')} open_trade_count={account.get('open_trade_count')}"
        )

    failure_lines: list[str] = []
    if missing:
        failure_lines.append("mechanism_integrity に欠落があると、change diary の根拠が一部欠ける。")
    if _safe_float(trades_24h.get("net_jpy")) < 0:
        failure_lines.append("24h profitability が負値で、悪化側の diary 候補を優先して拾う必要がある。")
    if top_loser:
        failure_lines.append(
            f"{_safe_str(top_loser.get('strategy_tag')) or 'unknown strategy'} が直近24hの loser cluster 先頭にいる。"
        )
    if not failure_lines:
        failure_lines.append("自動判定だけで断定できる failure cause は無い。人手レビューで補う。")

    improvement_lines = accepted_updates or recommendation_lines or [
        "top loser / health gap / profitability のいずれか 1 件を選んで手動 review し、TRADE_FINDINGS 本文へ昇格させる。"
    ]

    source_times = [
        _parse_dt(health.get("generated_at")),
        _parse_dt(pdca.get("generated_at_utc") or pdca.get("generated_at_jst")),
        _parse_dt(strategy_feedback.get("updated_at")),
        _parse_dt(counterfactual.get("generated_at")),
        _parse_dt(replay_gate.get("generated_at")),
        _parse_dt(participation_alloc.get("as_of") or participation_alloc.get("generated_at")),
        _parse_dt(market_context.get("generated_at")),
        generated_at,
    ]
    snapshot_dt = max([dt for dt in source_times if dt is not None], default=generated_at)
    latest_jst = snapshot_dt.astimezone(JST)

    summary = {
        "missing_mechanisms": missing,
        "trades_24h": {
            "trades": _safe_int(trades_24h.get("trades")),
            "net_jpy": round(_safe_float(trades_24h.get("net_jpy")), 3),
            "pf_jpy": round(_safe_float(trades_24h.get("pf_jpy")), 4),
            "win_rate": round(_safe_float(trades_24h.get("win_rate")), 4),
        },
        "top_loser": {
            "strategy_tag": _safe_str(top_loser.get("strategy_tag")),
            "net_jpy": round(_safe_float(top_loser.get("net_jpy")), 3),
            "trades": _safe_int(top_loser.get("trades")),
            "pf_jpy": round(_safe_float(top_loser.get("pf_jpy")), 4),
        },
        "top_winner": {
            "strategy_tag": _safe_str(top_winner.get("strategy_tag")),
            "net_jpy": round(_safe_float(top_winner.get("net_jpy")), 3),
            "trades": _safe_int(top_winner.get("trades")),
            "pf_jpy": round(_safe_float(top_winner.get("pf_jpy")), 4),
        },
        "market": {
            "bid": pricing.get("bid"),
            "ask": pricing.get("ask"),
            "spread_pips": pricing.get("spread_pips"),
            "data_lag_ms": round(_safe_float(health.get("data_lag_ms")), 1),
            "decision_latency_ms": round(_safe_float(health.get("decision_latency_ms")), 1),
        },
        "counterfactual": {
            "trades": _safe_int(counter_summary.get("trades")),
            "mean_pips": round(_safe_float(counter_summary.get("mean_pips")), 3),
            "stuck_trade_ratio": round(_safe_float(counter_summary.get("stuck_trade_ratio")), 4),
            "recommendation_lines": recommendation_lines,
        },
        "replay_quality_gate": {
            "gate_status": replay_gate.get("gate_status"),
            "accepted_updates": len(accepted_updates),
        },
        "participation_alloc": {
            "strategies": len(participation_strategies),
            "as_of": participation_alloc.get("as_of"),
        },
        "market_context": {
            "generated_at": market_context.get("generated_at"),
            "events": len(market_context.get("events") or []) if isinstance(market_context.get("events"), list) else 0,
        },
        "actionable_signals": signals,
    }
    fingerprint = hashlib.sha1(json.dumps(summary, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:12]

    return {
        "generated_at_utc": _iso_utc(generated_at),
        "generated_at_jst": _iso_jst(generated_at),
        "fingerprint": fingerprint,
        "status": "actionable" if signals else "quiet",
        "actionable_signals": signals,
        "sources": {
            "health_snapshot": {"path": _relative_path(cfg.health_path), "exists": cfg.health_path.exists(), "generated_at": health.get("generated_at")},
            "pdca_profitability": {"path": _relative_path(cfg.pdca_path), "exists": cfg.pdca_path.exists(), "generated_at": pdca.get("generated_at_utc") or pdca.get("generated_at_jst")},
            "strategy_feedback": {"path": _relative_path(cfg.strategy_feedback_path), "exists": cfg.strategy_feedback_path.exists(), "generated_at": strategy_feedback.get("updated_at")},
            "trade_counterfactual": {"path": _relative_path(cfg.trade_counterfactual_path), "exists": cfg.trade_counterfactual_path.exists(), "generated_at": counterfactual.get("generated_at")},
            "replay_quality_gate": {"path": _relative_path(cfg.replay_quality_gate_path), "exists": cfg.replay_quality_gate_path.exists(), "generated_at": replay_gate.get("generated_at")},
            "participation_alloc": {"path": _relative_path(cfg.participation_alloc_path), "exists": cfg.participation_alloc_path.exists(), "generated_at": participation_alloc.get("as_of") or participation_alloc.get("generated_at")},
            "market_context": {"path": _relative_path(cfg.market_context_path), "exists": cfg.market_context_path.exists(), "generated_at": market_context.get("generated_at")},
        },
        "summary": summary,
        "sections": {
            "title": f"## {latest_jst.strftime('%Y-%m-%d %H:%M JST')} / auto-draft: local feedback cycle review candidate",
            "change": [
                "latest local feedback / health / profitability artifacts から TRADE_FINDINGS 向け review draft を自動生成した。",
                "この自動化は logs/trade_findings_draft_latest.{json,md} と whiteboard 通知までで止め、docs/TRADE_FINDINGS.md 本体へは直接追記しない。",
            ],
            "why": [
                "既存の analysis artifact は豊富だが、change diary に昇格する直前の下書きだけが手作業に残っていた。",
                "artifact 更新のたびに review 候補を残せば、改善履歴の抜けを減らせる。",
            ],
            "hypothesis": [
                "health / PDCA / counterfactual / replay gate を一枚に束ねれば、review から TRADE_FINDINGS 反映までを短縮できる。"
            ],
            "expected_good": [
                "good/bad/pending を決める前の facts が自動で揃う。",
                "whiteboard を 1 本の review queue として使える。",
            ],
            "expected_bad": [
                "material change が無い局面でも draft が出るとノイズになり得る。",
                "自動文の Why/Hypothesis は一般化しやすいので、最終反映は必ずレビューが必要。",
            ],
            "period": [
                f"Snapshot UTC `{_iso_utc(snapshot_dt)}`",
                f"Snapshot JST `{_iso_jst(snapshot_dt)}`",
                "profitability window は `pdca_profitability_latest.json` の `24h/7d` 集計を参照する。",
            ],
            "fact": fact_lines,
            "failure_cause": failure_lines,
            "improvement": improvement_lines,
            "verification": [
                _relative_path(cfg.health_path),
                _relative_path(cfg.pdca_path),
                _relative_path(cfg.strategy_feedback_path),
                _relative_path(cfg.trade_counterfactual_path),
                _relative_path(cfg.replay_quality_gate_path),
                _relative_path(cfg.participation_alloc_path),
                _relative_path(cfg.market_context_path),
                _relative_path(cfg.out_json),
                _relative_path(cfg.out_md),
            ],
            "verdict": ["pending"],
            "next_action": [
                "logs/trade_findings_draft_latest.md を確認し、必要な箇所だけをレビューして docs/TRADE_FINDINGS.md へ反映する。",
                "同一 fingerprint の whiteboard 通知は再投稿しない。新しい fingerprint のときだけ review note を増やす。",
            ],
            "status": ["done"],
        },
        "whiteboard": {
            "enabled": cfg.whiteboard_enabled,
            "action": "not_attempted",
            "task_id": None,
            "event_id": None,
            "db_path": _relative_path(cfg.whiteboard_db),
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    sections = report.get("sections") if isinstance(report.get("sections"), dict) else {}
    lines = [
        "<!-- auto-generated review draft; review before copying into docs/TRADE_FINDINGS.md -->",
        "",
        str(sections.get("title") or "## auto-draft"),
        "",
    ]

    def emit(label: str, items: list[str]) -> None:
        lines.append(f"- {label}:")
        for item in items:
            lines.append(f"  - {item}")

    emit("Change", list(sections.get("change") or []))
    emit("Why", list(sections.get("why") or []))
    emit("Hypothesis", list(sections.get("hypothesis") or []))
    emit("Expected Good", list(sections.get("expected_good") or []))
    emit("Expected Bad", list(sections.get("expected_bad") or []))
    emit("Period", list(sections.get("period") or []))
    emit("Fact", list(sections.get("fact") or []))
    emit("Failure Cause", list(sections.get("failure_cause") or []))
    emit("Improvement", list(sections.get("improvement") or []))
    emit("Verification", list(sections.get("verification") or []))
    emit("Verdict", list(sections.get("verdict") or []))
    emit("Next Action", list(sections.get("next_action") or []))
    emit("Status", list(sections.get("status") or []))
    lines.append("")
    return "\n".join(lines)


def _find_open_task(cfg: Config) -> whiteboard.WhiteboardTask | None:
    try:
        tasks = whiteboard.list_tasks(status="open", limit=100, db_path=cfg.whiteboard_db)
    except Exception:
        return None
    for task in tasks:
        if task.task == cfg.whiteboard_task:
            return task
    return None


def _task_has_fingerprint(task_id: int, fingerprint: str, db_path: Path) -> bool:
    try:
        events = whiteboard.list_events(task_id=task_id, limit=200, db_path=db_path)
    except Exception:
        return False
    for event in events:
        if not isinstance(event.metadata, dict):
            continue
        if event.metadata.get("kind") == "trade_findings_draft" and event.metadata.get("fingerprint") == fingerprint:
            return True
    return False


def _maybe_post_whiteboard(cfg: Config, report: dict[str, Any]) -> dict[str, Any]:
    if not cfg.whiteboard_enabled:
        return {"enabled": False, "action": "disabled", "task_id": None, "event_id": None, "db_path": _relative_path(cfg.whiteboard_db)}
    if report.get("status") != "actionable":
        return {"enabled": True, "action": "skipped_not_actionable", "task_id": None, "event_id": None, "db_path": _relative_path(cfg.whiteboard_db)}

    fingerprint = _safe_str(report.get("fingerprint"))
    task = _find_open_task(cfg)
    created_task = False
    if task is None:
        task = whiteboard.post_task(
            task=cfg.whiteboard_task,
            body="Auto-generated TRADE_FINDINGS review queue. Review the latest draft and copy only curated lines into docs/TRADE_FINDINGS.md.",
            author=cfg.whiteboard_author,
            db_path=cfg.whiteboard_db,
        )
        created_task = True
    assert task is not None
    if _task_has_fingerprint(task.id, fingerprint, cfg.whiteboard_db):
        return {"enabled": True, "action": "skipped_duplicate_fingerprint", "task_id": task.id, "event_id": None, "db_path": _relative_path(cfg.whiteboard_db)}

    note = whiteboard.post_note(
        task_id=task.id,
        body=(
            f"trade_findings_draft fingerprint={fingerprint} "
            f"status={report.get('status')} md={_relative_path(cfg.out_md)} "
            f"signals={','.join(report.get('actionable_signals') or [])}"
        ),
        author=cfg.whiteboard_author,
        metadata={
            "kind": "trade_findings_draft",
            "fingerprint": fingerprint,
            "status": report.get("status"),
            "json_path": _relative_path(cfg.out_json),
            "md_path": _relative_path(cfg.out_md),
            "actionable_signals": report.get("actionable_signals") or [],
        },
        db_path=cfg.whiteboard_db,
    )
    return {
        "enabled": True,
        "action": "created_task_and_note" if created_task else "posted_note",
        "task_id": task.id,
        "event_id": note.id,
        "db_path": _relative_path(cfg.whiteboard_db),
    }


def run_once(cfg: Config) -> dict[str, Any]:
    report = build_report(cfg)
    markdown = render_markdown(report)
    report["whiteboard"] = _maybe_post_whiteboard(cfg, report)
    previous = _load_json(cfg.out_json)
    previous_fingerprint = _safe_str(previous.get("fingerprint"))
    history_action = "appended" if previous_fingerprint != _safe_str(report.get("fingerprint")) else "skipped_duplicate_fingerprint"
    report["history"] = {"action": history_action, "path": _relative_path(cfg.out_history)}
    _write_json_atomic(cfg.out_json, report)
    if history_action == "appended":
        _append_jsonl(cfg.out_history, report)
    cfg.out_md.parent.mkdir(parents=True, exist_ok=True)
    cfg.out_md.write_text(markdown, encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    cfg = build_config(parse_args(argv))
    report = run_once(cfg)
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
