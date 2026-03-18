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
DEFAULT_WHITEBOARD_TASK = "trade_findings_draft review"


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


def _now_utc() -> datetime:
    return datetime.now(UTC)


def _iso_utc(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).isoformat()


def _iso_jst(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(JST).isoformat()


def _parse_dt(raw: Any) -> datetime | None:
    text = str(raw or "").strip()
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


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
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


def _top_ranked(
    payload: dict[str, Any], key: str, bucket: str
) -> dict[str, Any] | None:
    rankings = payload.get("trades", {}).get("24h", {}).get("rankings", {})
    section = rankings.get(key)
    if not isinstance(section, dict):
        return None
    items = section.get(bucket)
    if not isinstance(items, list) or not items:
        return None
    first = items[0]
    if not isinstance(first, dict):
        return None
    return first


def _recommendation_lines(counterfactual: dict[str, Any]) -> list[str]:
    recommendations = counterfactual.get("recommendations")
    if not isinstance(recommendations, list):
        return []
    lines: list[str] = []
    for item in recommendations[:3]:
        if not isinstance(item, dict):
            continue
        strategy = _safe_str(item.get("strategy_tag") or item.get("strategy"))
        action = _safe_str(item.get("action") or item.get("recommendation"))
        reason = _safe_str(item.get("reason"))
        parts = [
            part for part in [strategy or "generic", action or "review", reason] if part
        ]
        if parts:
            lines.append(" / ".join(parts))
    return lines


def _accepted_update_lines(replay_payload: dict[str, Any]) -> list[str]:
    auto_improve = replay_payload.get("auto_improve")
    if not isinstance(auto_improve, dict):
        return []
    updates = auto_improve.get("accepted_updates")
    if not isinstance(updates, list):
        return []
    lines: list[str] = []
    for item in updates[:3]:
        if not isinstance(item, dict):
            continue
        strategy = _safe_str(
            item.get("strategy") or item.get("worker") or item.get("strategy_tag")
        )
        field = _safe_str(item.get("field") or item.get("key"))
        value = item.get("value")
        value_text = str(value) if isinstance(value, (int, float, str, bool)) else ""
        parts = [part for part in [strategy, field, value_text] if part]
        if parts:
            lines.append(" / ".join(parts))
    return lines


def _history_has_fingerprint(path: Path, fingerprint: str) -> bool:
    if not fingerprint or not path.exists():
        return False
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if _safe_str(payload.get("fingerprint")) == fingerprint:
                    return True
    except OSError:
        return False
    return False


def _first_open_whiteboard_task(
    db_path: Path, task_name: str
) -> whiteboard.WhiteboardTask | None:
    try:
        tasks = whiteboard.list_tasks(status="open", limit=100, db_path=db_path)
    except Exception:
        return None
    for task in tasks:
        if task.task == task_name:
            return task
    return None


def _task_has_fingerprint(task_id: int, fingerprint: str, db_path: Path) -> bool:
    try:
        events = whiteboard.list_events(task_id=task_id, limit=200, db_path=db_path)
    except Exception:
        return False
    for event in events:
        metadata = event.metadata
        if not isinstance(metadata, dict):
            continue
        if (
            metadata.get("kind") == "trade_findings_draft"
            and metadata.get("fingerprint") == fingerprint
        ):
            return True
    return False


@dataclass(frozen=True)
class Config:
    health_path: Path
    pdca_path: Path
    strategy_feedback_path: Path
    trade_counterfactual_path: Path
    replay_quality_gate_path: Path
    out_json: Path
    out_history: Path
    out_md: Path
    whiteboard_enabled: bool
    whiteboard_db: Path
    whiteboard_task: str
    whiteboard_author: str


def build_config(args: argparse.Namespace) -> Config:
    return Config(
        health_path=_resolve_path(args.health_path),
        pdca_path=_resolve_path(args.pdca_path),
        strategy_feedback_path=_resolve_path(args.strategy_feedback_path),
        trade_counterfactual_path=_resolve_path(args.trade_counterfactual_path),
        replay_quality_gate_path=_resolve_path(args.replay_quality_gate_path),
        out_json=_resolve_path(args.out_json),
        out_history=_resolve_path(args.out_history),
        out_md=_resolve_path(args.out_md),
        whiteboard_enabled=bool(args.whiteboard_enabled),
        whiteboard_db=_resolve_path(args.whiteboard_db),
        whiteboard_task=str(args.whiteboard_task),
        whiteboard_author=str(args.whiteboard_author),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a TRADE_FINDINGS review-only diary draft."
    )
    parser.add_argument(
        "--health-path",
        default=os.getenv(
            "TRADE_FINDINGS_DRAFT_HEALTH_PATH", "logs/health_snapshot.json"
        ),
    )
    parser.add_argument(
        "--pdca-path",
        default=os.getenv(
            "TRADE_FINDINGS_DRAFT_PDCA_PATH", "logs/pdca_profitability_latest.json"
        ),
    )
    parser.add_argument(
        "--strategy-feedback-path",
        default=os.getenv(
            "TRADE_FINDINGS_DRAFT_STRATEGY_FEEDBACK_PATH", "logs/strategy_feedback.json"
        ),
    )
    parser.add_argument(
        "--trade-counterfactual-path",
        default=os.getenv(
            "TRADE_FINDINGS_DRAFT_COUNTERFACTUAL_PATH",
            "logs/trade_counterfactual_latest.json",
        ),
    )
    parser.add_argument(
        "--replay-quality-gate-path",
        default=os.getenv(
            "TRADE_FINDINGS_DRAFT_REPLAY_PATH", "logs/replay_quality_gate_latest.json"
        ),
    )
    parser.add_argument(
        "--out-json",
        default=os.getenv(
            "TRADE_FINDINGS_DRAFT_OUT_JSON", "logs/trade_findings_draft_latest.json"
        ),
    )
    parser.add_argument(
        "--out-history",
        default=os.getenv(
            "TRADE_FINDINGS_DRAFT_OUT_HISTORY",
            "logs/trade_findings_draft_history.jsonl",
        ),
    )
    parser.add_argument(
        "--out-md",
        default=os.getenv(
            "TRADE_FINDINGS_DRAFT_OUT_MD", "logs/trade_findings_draft_latest.md"
        ),
    )
    parser.add_argument(
        "--whiteboard-db",
        default=os.getenv(
            "TRADE_FINDINGS_DRAFT_WHITEBOARD_DB", "logs/agent_whiteboard.db"
        ),
    )
    parser.add_argument(
        "--whiteboard-task",
        default=os.getenv(
            "TRADE_FINDINGS_DRAFT_WHITEBOARD_TASK", DEFAULT_WHITEBOARD_TASK
        ),
    )
    parser.add_argument(
        "--whiteboard-author",
        default=os.getenv(
            "TRADE_FINDINGS_DRAFT_WHITEBOARD_AUTHOR",
            os.getenv("USER", "trade-findings-draft"),
        ),
    )
    parser.add_argument(
        "--whiteboard-enabled",
        dest="whiteboard_enabled",
        action="store_true",
        default=_env_bool("TRADE_FINDINGS_DRAFT_WHITEBOARD_ENABLED", False),
    )
    parser.add_argument(
        "--no-whiteboard-enabled", dest="whiteboard_enabled", action="store_false"
    )
    parser.add_argument(
        "--no-whiteboard", dest="whiteboard_enabled", action="store_false"
    )
    parser.add_argument(
        "--disable-whiteboard", dest="whiteboard_enabled", action="store_false"
    )
    return parser.parse_args(argv)


def build_report(cfg: Config) -> dict[str, Any]:
    generated_at = _now_utc()
    health = _load_json(cfg.health_path) or {}
    pdca = _load_json(cfg.pdca_path) or {}
    strategy_feedback = _load_json(cfg.strategy_feedback_path) or {}
    counterfactual = _load_json(cfg.trade_counterfactual_path) or {}
    replay_gate = _load_json(cfg.replay_quality_gate_path) or {}

    health_dt = _parse_dt(health.get("generated_at"))
    pdca_dt = _parse_dt(pdca.get("generated_at_utc") or pdca.get("generated_at_jst"))
    feedback_dt = _parse_dt(strategy_feedback.get("updated_at"))
    counterfactual_dt = _parse_dt(counterfactual.get("generated_at"))
    replay_dt = _parse_dt(replay_gate.get("generated_at"))

    mechanism = (
        health.get("mechanism_integrity")
        if isinstance(health.get("mechanism_integrity"), dict)
        else {}
    )
    missing_mechanisms = [
        str(item)
        for item in mechanism.get("missing_mechanisms") or []
        if str(item).strip()
    ]
    pricing = (
        pdca.get("oanda", {}).get("pricing", {})
        if isinstance(pdca.get("oanda"), dict)
        else {}
    )
    account_summary = (
        pdca.get("oanda", {}).get("summary", {})
        if isinstance(pdca.get("oanda"), dict)
        else {}
    )
    trades_24h = (
        pdca.get("trades", {}).get("24h", {}).get("overall", {})
        if isinstance(pdca.get("trades"), dict)
        else {}
    )
    top_loser = _top_ranked(pdca, "by_strategy_net_jpy", "top_losers") or {}
    top_winner = _top_ranked(pdca, "by_strategy_net_jpy", "top_winners") or {}
    feedback_strategies = (
        strategy_feedback.get("strategies")
        if isinstance(strategy_feedback.get("strategies"), dict)
        else {}
    )
    counter_summary = (
        counterfactual.get("summary")
        if isinstance(counterfactual.get("summary"), dict)
        else {}
    )
    gate_status = _safe_str(replay_gate.get("gate_status")).lower()
    accepted_updates = _accepted_update_lines(replay_gate)
    recommendations = _recommendation_lines(counterfactual)

    actionable_signals: list[str] = []
    if missing_mechanisms:
        actionable_signals.append(f"mechanism_missing:{','.join(missing_mechanisms)}")
    if _safe_float(trades_24h.get("net_jpy")) < 0:
        actionable_signals.append("profitability_24h_net_negative")
    if (
        _safe_float(trades_24h.get("pf_jpy")) < 1.0
        and _safe_int(trades_24h.get("trades")) >= 20
    ):
        actionable_signals.append("profitability_24h_pf_below_1")
    if _safe_float(top_loser.get("net_jpy")) < -1.0:
        actionable_signals.append(
            f"top_loser:{_safe_str(top_loser.get('strategy_tag')) or 'unknown'}"
        )
    if gate_status and gate_status not in {"ok", "pass", "soft_skip"}:
        actionable_signals.append(f"replay_quality_gate:{gate_status}")
    if accepted_updates:
        actionable_signals.append("replay_auto_improve_updates")
    if recommendations:
        actionable_signals.append("counterfactual_recommendations")

    snapshot_dt = max(
        [
            item
            for item in [
                health_dt,
                pdca_dt,
                feedback_dt,
                counterfactual_dt,
                replay_dt,
                generated_at,
            ]
            if item is not None
        ],
        default=generated_at,
    )
    latest_jst = snapshot_dt.astimezone(JST)

    fact_lines: list[str] = []
    if pricing:
        fact_lines.append(
            "market: "
            f"USD/JPY bid={pricing.get('bid')} ask={pricing.get('ask')} spread={pricing.get('spread_pips')}p "
            f"(pricing_ok={pricing.get('meta', {}).get('ok')})"
        )
    if health:
        fact_lines.append(
            "health: "
            f"data_lag_ms={round(_safe_float(health.get('data_lag_ms')), 1)} "
            f"decision_latency_ms={round(_safe_float(health.get('decision_latency_ms')), 1)} "
            f"mechanism_ok={bool(mechanism.get('ok'))}"
        )
    if missing_mechanisms:
        fact_lines.append("missing_mechanisms: " + ", ".join(missing_mechanisms))
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
            "top_loser_24h: "
            f"{_safe_str(top_loser.get('strategy_tag'))} "
            f"net_jpy={round(_safe_float(top_loser.get('net_jpy')), 3)} "
            f"trades={_safe_int(top_loser.get('trades'))} "
            f"pf_jpy={round(_safe_float(top_loser.get('pf_jpy')), 4)}"
        )
    if top_winner:
        fact_lines.append(
            "top_winner_24h: "
            f"{_safe_str(top_winner.get('strategy_tag'))} "
            f"net_jpy={round(_safe_float(top_winner.get('net_jpy')), 3)} "
            f"trades={_safe_int(top_winner.get('trades'))} "
            f"pf_jpy={round(_safe_float(top_winner.get('pf_jpy')), 4)}"
        )
    if feedback_strategies:
        fact_lines.append(
            f"strategy_feedback: strategies={len(feedback_strategies)} updated_at={strategy_feedback.get('updated_at')}"
        )
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
    if account_summary:
        fact_lines.append(
            "account: "
            f"nav_jpy={account_summary.get('nav_jpy')} "
            f"margin_rate={account_summary.get('margin_rate')} "
            f"open_trade_count={account_summary.get('open_trade_count')}"
        )

    failure_lines: list[str] = []
    if missing_mechanisms:
        failure_lines.append(
            "mechanism_integrity の欠落があるため、改善判断の根拠セットが不完全になる。"
        )
    if _safe_float(trades_24h.get("net_jpy")) < 0:
        failure_lines.append(
            "24h profitability が負値で、直近変更の悪化要因が未整理のまま残っている。"
        )
    if top_loser:
        failure_lines.append(
            f"{_safe_str(top_loser.get('strategy_tag')) or 'unknown strategy'} が直近24hの主要 loser で、"
            "strategy-local quality / exit / sizing の再点検が要る。"
        )
    if gate_status and gate_status not in {"ok", "pass", "soft_skip"}:
        failure_lines.append(
            "replay quality gate が pass しておらず、再現性監査の観点でも review 候補になっている。"
        )
    if not failure_lines:
        failure_lines.append(
            "自動判定だけでは failure cause を断定しない。最終判断はレビューで補う。"
        )

    improvement_lines: list[str] = []
    if accepted_updates:
        improvement_lines.extend(accepted_updates)
    elif recommendations:
        improvement_lines.extend(recommendations)
    else:
        improvement_lines.append(
            "top loser と replay gate の facts を見て、手動で 1 件だけ diary entry に昇格させる。"
        )

    verification_lines = [
        _relative_path(cfg.health_path),
        _relative_path(cfg.pdca_path),
        _relative_path(cfg.strategy_feedback_path),
        _relative_path(cfg.trade_counterfactual_path),
        _relative_path(cfg.replay_quality_gate_path),
        _relative_path(cfg.out_json),
        _relative_path(cfg.out_md),
    ]
    next_action_lines = [
        "logs/trade_findings_draft_latest.md をレビューし、必要な事実だけを docs/TRADE_FINDINGS.md へ手動で昇格する。",
        "same fingerprint の draft は history / whiteboard に再投稿しない。変化があるときだけ既存 review task に note を足す。",
    ]
    if accepted_updates:
        next_action_lines.append(
            "accepted auto-improve の対象 strategy は次回の health / profitability 実測で良化有無を確認する。"
        )

    summary = {
        "health_generated_at": _iso_utc(health_dt) if health_dt else None,
        "pdca_generated_at": _iso_utc(pdca_dt) if pdca_dt else None,
        "strategy_feedback_updated_at": _iso_utc(feedback_dt) if feedback_dt else None,
        "counterfactual_generated_at": (
            _iso_utc(counterfactual_dt) if counterfactual_dt else None
        ),
        "replay_generated_at": _iso_utc(replay_dt) if replay_dt else None,
        "missing_mechanisms": missing_mechanisms,
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
            "pricing_ok": (
                pricing.get("meta", {}).get("ok")
                if isinstance(pricing.get("meta"), dict)
                else None
            ),
            "data_lag_ms": (
                round(_safe_float(health.get("data_lag_ms")), 1) if health else None
            ),
            "decision_latency_ms": (
                round(_safe_float(health.get("decision_latency_ms")), 1)
                if health
                else None
            ),
        },
        "counterfactual": {
            "trades": _safe_int(counter_summary.get("trades")),
            "mean_pips": round(_safe_float(counter_summary.get("mean_pips")), 3),
            "stuck_trade_ratio": round(
                _safe_float(counter_summary.get("stuck_trade_ratio")), 4
            ),
            "recommendation_lines": recommendations,
        },
        "replay_quality_gate": {
            "gate_status": replay_gate.get("gate_status"),
            "accepted_updates": len(accepted_updates),
        },
        "actionable_signals": actionable_signals,
    }
    fingerprint = hashlib.sha1(
        json.dumps(summary, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]

    return {
        "generated_at_utc": _iso_utc(generated_at),
        "generated_at_jst": _iso_jst(generated_at),
        "fingerprint": fingerprint,
        "status": "actionable" if actionable_signals else "quiet",
        "actionable_signals": actionable_signals,
        "draft_policy": {
            "review_only": True,
            "writes_to_trade_findings": False,
        },
        "sources": {
            "health_snapshot": {
                "path": _relative_path(cfg.health_path),
                "exists": cfg.health_path.exists(),
                "generated_at": health.get("generated_at"),
            },
            "pdca_profitability": {
                "path": _relative_path(cfg.pdca_path),
                "exists": cfg.pdca_path.exists(),
                "generated_at": pdca.get("generated_at_utc")
                or pdca.get("generated_at_jst"),
            },
            "strategy_feedback": {
                "path": _relative_path(cfg.strategy_feedback_path),
                "exists": cfg.strategy_feedback_path.exists(),
                "generated_at": strategy_feedback.get("updated_at"),
            },
            "trade_counterfactual": {
                "path": _relative_path(cfg.trade_counterfactual_path),
                "exists": cfg.trade_counterfactual_path.exists(),
                "generated_at": counterfactual.get("generated_at"),
            },
            "replay_quality_gate": {
                "path": _relative_path(cfg.replay_quality_gate_path),
                "exists": cfg.replay_quality_gate_path.exists(),
                "generated_at": replay_gate.get("generated_at"),
            },
        },
        "summary": summary,
        "sections": {
            "title": f"## {latest_jst.strftime('%Y-%m-%d %H:%M JST')} / auto-draft: local feedback cycle review candidate",
            "change": [
                "latest local feedback artifacts から TRADE_FINDINGS 向け review-only draft を自動生成した。",
                "この自動化は logs/trade_findings_draft_latest.{json,md} と optional whiteboard note までで止め、docs/TRADE_FINDINGS.md 本体へは直接追記しない。",
            ],
            "why": [
                "analysis artifact は増えていたが、良化 / 悪化 / 保留を change diary 形式に束ねる最後の整形が手作業だった。",
                "facts が揃った下書きを後段で残せば、レビュー起点の取りこぼしを減らせる。",
            ],
            "hypothesis": [
                "health / profitability / strategy feedback / counterfactual / replay gate を一枚に束ねれば、TRADE_FINDINGS 反映までの手戻りを減らせる。"
            ],
            "expected_good": [
                "Change / Why / Hypothesis / Fact / Next Action を埋めた下書きが自動で残り、review が速くなる。",
                "whiteboard を 1 本の review task に寄せることで、追加メモを散らさずに扱える。",
            ],
            "expected_bad": [
                "material change が薄い局面でも draft が生成されると、review queue がノイズ化する可能性がある。",
                "自動文面をそのまま採用すると Why / Failure Cause が一般化しすぎる。",
            ],
            "period": [
                f"Snapshot UTC `{_iso_utc(snapshot_dt)}`",
                f"Snapshot JST `{_iso_jst(snapshot_dt)}`",
                "profitability window は `pdca_profitability_latest.json` の `24h` 集計を参照する。",
            ],
            "fact": fact_lines,
            "failure_cause": failure_lines,
            "improvement": improvement_lines,
            "verification": verification_lines,
            "verdict": ["pending"],
            "next_action": next_action_lines,
            "status": ["open"],
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
    sections = (
        report.get("sections") if isinstance(report.get("sections"), dict) else {}
    )
    lines: list[str] = [
        "<!-- auto-generated review draft; do not append directly to docs/TRADE_FINDINGS.md -->",
        "",
        str(sections.get("title") or "## auto-draft"),
        "",
    ]

    def emit_block(label: str, items: Any) -> None:
        lines.append(f"- {label}:")
        values = items if isinstance(items, list) else [items]
        for item in values:
            lines.append(f"  - {item}")

    emit_block("Change", sections.get("change") or [])
    emit_block("Why", sections.get("why") or [])
    emit_block("Hypothesis", sections.get("hypothesis") or [])
    emit_block("Expected Good", sections.get("expected_good") or [])
    emit_block("Expected Bad", sections.get("expected_bad") or [])
    emit_block("Period", sections.get("period") or [])
    emit_block("Fact", sections.get("fact") or [])
    emit_block("Failure Cause", sections.get("failure_cause") or [])
    emit_block("Improvement", sections.get("improvement") or [])
    emit_block("Verification", sections.get("verification") or [])
    emit_block("Verdict", sections.get("verdict") or [])
    emit_block("Next Action", sections.get("next_action") or [])
    emit_block("Status", sections.get("status") or [])
    lines.append("")
    return "\n".join(lines)


def _maybe_post_whiteboard(cfg: Config, report: dict[str, Any]) -> dict[str, Any]:
    if not cfg.whiteboard_enabled:
        return {
            "enabled": False,
            "action": "disabled",
            "task_id": None,
            "event_id": None,
            "db_path": _relative_path(cfg.whiteboard_db),
        }
    if report.get("status") != "actionable":
        return {
            "enabled": True,
            "action": "skipped_not_actionable",
            "task_id": None,
            "event_id": None,
            "db_path": _relative_path(cfg.whiteboard_db),
        }

    fingerprint = _safe_str(report.get("fingerprint"))
    task = _first_open_whiteboard_task(cfg.whiteboard_db, cfg.whiteboard_task)
    created_task = False
    if task is None:
        task = whiteboard.post_task(
            task=cfg.whiteboard_task,
            body=(
                "Auto-generated TRADE_FINDINGS review queue. "
                "Review logs/trade_findings_draft_latest.md and copy only curated lines into docs/TRADE_FINDINGS.md."
            ),
            author=cfg.whiteboard_author,
            db_path=cfg.whiteboard_db,
        )
        created_task = True

    assert task is not None
    if _task_has_fingerprint(task.id, fingerprint, cfg.whiteboard_db):
        return {
            "enabled": True,
            "action": "skipped_duplicate_fingerprint",
            "task_id": task.id,
            "event_id": None,
            "db_path": _relative_path(cfg.whiteboard_db),
        }

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
            "md_path": _relative_path(cfg.out_md),
            "json_path": _relative_path(cfg.out_json),
            "generated_at_utc": report.get("generated_at_utc"),
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
    history_appended = not _history_has_fingerprint(
        cfg.out_history, _safe_str(report.get("fingerprint"))
    )
    report["history"] = {
        "action": "appended" if history_appended else "skipped_duplicate_fingerprint",
        "path": _relative_path(cfg.out_history),
    }
    _write_json_atomic(cfg.out_json, report)
    if history_appended:
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
