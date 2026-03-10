from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from workers.common import agent_whiteboard as whiteboard


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "trade_findings_diary_draft.py"
_SPEC = importlib.util.spec_from_file_location("trade_findings_diary_draft_test", SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
trade_findings_diary_draft = importlib.util.module_from_spec(_SPEC)
sys.modules.setdefault("trade_findings_diary_draft_test", trade_findings_diary_draft)
_SPEC.loader.exec_module(trade_findings_diary_draft)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _source_paths(tmp_path: Path) -> dict[str, Path]:
    logs_dir = tmp_path / "logs"
    return {
        "health": logs_dir / "health_snapshot.json",
        "pdca": logs_dir / "pdca_profitability_latest.json",
        "feedback": logs_dir / "strategy_feedback.json",
        "counterfactual": logs_dir / "trade_counterfactual_latest.json",
        "replay": logs_dir / "replay_quality_gate_latest.json",
        "out_json": logs_dir / "trade_findings_draft_latest.json",
        "out_history": logs_dir / "trade_findings_draft_history.jsonl",
        "out_md": logs_dir / "trade_findings_draft_latest.md",
        "whiteboard_db": logs_dir / "agent_whiteboard.db",
    }


def _prepare_sources(tmp_path: Path) -> dict[str, Path]:
    paths = _source_paths(tmp_path)
    _write_json(
        paths["health"],
        {
            "generated_at": "2026-03-10T17:09:49.250377+00:00",
            "data_lag_ms": 1155.1,
            "decision_latency_ms": 14.1,
            "mechanism_integrity": {"ok": True, "missing_mechanisms": []},
        },
    )
    _write_json(
        paths["pdca"],
        {
            "generated_at_utc": "2026-03-10T17:10:00+00:00",
            "generated_at_jst": "2026-03-11T02:10:00+09:00",
            "oanda": {
                "pricing": {"bid": 157.486, "ask": 157.494, "spread_pips": 0.8, "meta": {"ok": True}},
                "summary": {"nav_jpy": 35777.42, "margin_rate": 0.04, "open_trade_count": 1},
            },
            "trades": {
                "24h": {
                    "overall": {"trades": 218, "net_jpy": -1050.433, "pf_jpy": 0.2525, "win_rate": 0.3028},
                    "rankings": {
                        "by_strategy_net_jpy": {
                            "top_losers": [
                                {"strategy_tag": "microlevelreactor", "net_jpy": -298.128, "trades": 53, "pf_jpy": 0.06}
                            ],
                            "top_winners": [
                                {"strategy_tag": "precisionlowvol", "net_jpy": 31.198, "trades": 9, "pf_jpy": 2.7203}
                            ],
                        }
                    },
                }
            },
        },
    )
    _write_json(
        paths["feedback"],
        {
            "updated_at": "2026-03-10T17:11:00+00:00",
            "strategies": {"MicroLevelReactor": {}, "PrecisionLowVol": {}},
        },
    )
    _write_json(
        paths["counterfactual"],
        {
            "generated_at": "2026-03-10T17:12:00+00:00",
            "summary": {"trades": 45, "mean_pips": -0.45, "stuck_trade_ratio": 0.21},
            "recommendations": [
                {"strategy_tag": "microlevelreactor", "action": "tighten_quality", "reason": "loss cluster"}
            ],
        },
    )
    _write_json(
        paths["replay"],
        {
            "generated_at": "2026-03-10T17:13:00+00:00",
            "gate_status": "fail",
            "failing_workers": ["microlevelreactor"],
            "auto_improve": {
                "accepted_updates": [
                    {"strategy": "microlevelreactor", "field": "quality_floor", "value": 0.62}
                ]
            },
        },
    )
    return paths


def _run_once(paths: dict[str, Path], *, whiteboard_enabled: bool = False) -> dict:
    argv = [
        "--health-path",
        str(paths["health"]),
        "--pdca-path",
        str(paths["pdca"]),
        "--strategy-feedback-path",
        str(paths["feedback"]),
        "--trade-counterfactual-path",
        str(paths["counterfactual"]),
        "--replay-quality-gate-path",
        str(paths["replay"]),
        "--out-json",
        str(paths["out_json"]),
        "--out-history",
        str(paths["out_history"]),
        "--out-md",
        str(paths["out_md"]),
        "--whiteboard-db",
        str(paths["whiteboard_db"]),
        "--whiteboard-task",
        "trade_findings_draft review",
        "--whiteboard-author",
        "pytest",
    ]
    argv.append("--whiteboard-enabled" if whiteboard_enabled else "--no-whiteboard-enabled")
    assert trade_findings_diary_draft.main(argv) == 0
    return json.loads(paths["out_json"].read_text(encoding="utf-8"))


def test_trade_findings_diary_draft_generates_required_sections(tmp_path: Path) -> None:
    paths = _prepare_sources(tmp_path)

    report = _run_once(paths, whiteboard_enabled=False)

    assert report["status"] == "actionable"
    assert report["history"]["action"] == "appended"
    assert report["draft_policy"] == {"review_only": True, "writes_to_trade_findings": False}
    assert set(report["sources"]) == {
        "health_snapshot",
        "pdca_profitability",
        "strategy_feedback",
        "trade_counterfactual",
        "replay_quality_gate",
    }

    sections = report["sections"]
    for key in [
        "change",
        "why",
        "hypothesis",
        "expected_good",
        "expected_bad",
        "period",
        "fact",
        "failure_cause",
        "improvement",
        "verification",
        "verdict",
        "next_action",
        "status",
    ]:
        assert sections[key]

    markdown = paths["out_md"].read_text(encoding="utf-8")
    assert "do not append directly to docs/TRADE_FINDINGS.md" in markdown
    assert "- Change:" in markdown
    assert "- Failure Cause:" in markdown
    assert "- Status:" in markdown
    assert "microlevelreactor" in markdown


def test_trade_findings_diary_draft_dedupes_history_and_reuses_single_whiteboard_task(tmp_path: Path) -> None:
    paths = _prepare_sources(tmp_path)

    first = _run_once(paths, whiteboard_enabled=True)
    assert first["history"]["action"] == "appended"
    assert first["whiteboard"]["action"] == "created_task_and_note"
    tasks = whiteboard.list_tasks(status="open", db_path=paths["whiteboard_db"])
    assert len(tasks) == 1
    events = whiteboard.list_events(task_id=tasks[0].id, db_path=paths["whiteboard_db"])
    assert len(events) == 1

    second = _run_once(paths, whiteboard_enabled=True)
    assert second["history"]["action"] == "skipped_duplicate_fingerprint"
    assert second["whiteboard"]["action"] == "skipped_duplicate_fingerprint"
    tasks = whiteboard.list_tasks(status="open", db_path=paths["whiteboard_db"])
    assert len(tasks) == 1
    events = whiteboard.list_events(task_id=tasks[0].id, db_path=paths["whiteboard_db"])
    assert len(events) == 1

    updated_pdca = json.loads(paths["pdca"].read_text(encoding="utf-8"))
    updated_pdca["generated_at_utc"] = "2026-03-10T17:20:00+00:00"
    updated_pdca["trades"]["24h"]["overall"]["net_jpy"] = -990.0
    _write_json(paths["pdca"], updated_pdca)

    third = _run_once(paths, whiteboard_enabled=True)
    assert third["history"]["action"] == "appended"
    assert third["whiteboard"]["action"] == "posted_note"
    tasks = whiteboard.list_tasks(status="open", db_path=paths["whiteboard_db"])
    assert len(tasks) == 1
    events = whiteboard.list_events(task_id=tasks[0].id, db_path=paths["whiteboard_db"])
    assert len(events) == 2

    restored_pdca = json.loads(paths["pdca"].read_text(encoding="utf-8"))
    restored_pdca["generated_at_utc"] = "2026-03-10T17:10:00+00:00"
    restored_pdca["trades"]["24h"]["overall"]["net_jpy"] = -1050.433
    _write_json(paths["pdca"], restored_pdca)

    fourth = _run_once(paths, whiteboard_enabled=True)
    assert fourth["history"]["action"] == "skipped_duplicate_fingerprint"
    assert fourth["whiteboard"]["action"] == "skipped_duplicate_fingerprint"
    tasks = whiteboard.list_tasks(status="open", db_path=paths["whiteboard_db"])
    assert len(tasks) == 1
    events = whiteboard.list_events(task_id=tasks[0].id, db_path=paths["whiteboard_db"])
    assert len(events) == 2

    history_lines = paths["out_history"].read_text(encoding="utf-8").strip().splitlines()
    assert len(history_lines) == 2
