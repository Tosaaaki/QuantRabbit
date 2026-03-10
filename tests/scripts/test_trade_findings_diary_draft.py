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


def _paths(tmp_path: Path) -> dict[str, Path]:
    logs_dir = tmp_path / "logs"
    config_dir = tmp_path / "config"
    return {
        "health": logs_dir / "health_snapshot.json",
        "pdca": logs_dir / "pdca_profitability_latest.json",
        "feedback": logs_dir / "strategy_feedback.json",
        "counterfactual": logs_dir / "trade_counterfactual_latest.json",
        "replay": logs_dir / "replay_quality_gate_latest.json",
        "participation": config_dir / "participation_alloc.json",
        "market_context": logs_dir / "market_context_latest.json",
        "out_json": logs_dir / "trade_findings_draft_latest.json",
        "out_history": logs_dir / "trade_findings_draft_history.jsonl",
        "out_md": logs_dir / "trade_findings_draft_latest.md",
        "whiteboard_db": logs_dir / "agent_whiteboard.db",
    }


def _prepare(tmp_path: Path) -> dict[str, Path]:
    paths = _paths(tmp_path)
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
    _write_json(paths["feedback"], {"updated_at": "2026-03-10T17:11:00+00:00", "strategies": {"MicroLevelReactor": {}, "PrecisionLowVol": {}}})
    _write_json(
        paths["counterfactual"],
        {
            "generated_at": "2026-03-10T17:12:00+00:00",
            "summary": {"trades": 45, "mean_pips": -0.45, "stuck_trade_ratio": 0.21},
            "recommendations": [{"strategy_tag": "microlevelreactor", "action": "tighten_quality", "reason": "loss cluster"}],
        },
    )
    _write_json(
        paths["replay"],
        {
            "generated_at": "2026-03-10T17:13:00+00:00",
            "gate_status": "fail",
            "failing_workers": ["microlevelreactor"],
            "auto_improve": {"accepted_updates": [{"strategy": "microlevelreactor", "field": "quality_floor", "value": 0.62}]},
        },
    )
    _write_json(paths["participation"], {"as_of": "2026-03-10T17:14:00+00:00", "strategies": {"PrecisionLowVol": {"action": "boost_participation"}}})
    _write_json(paths["market_context"], {"generated_at": "2026-03-10T17:15:00+00:00", "events": [{"headline": "range transition"}]})
    return paths


def _run(paths: dict[str, Path], *, whiteboard_enabled: bool = True) -> dict:
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
        "--participation-alloc-path",
        str(paths["participation"]),
        "--market-context-path",
        str(paths["market_context"]),
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
    if not whiteboard_enabled:
        argv.append("--no-whiteboard")
    assert trade_findings_diary_draft.main(argv) == 0
    return json.loads(paths["out_json"].read_text(encoding="utf-8"))


def test_trade_findings_diary_draft_generates_outputs_and_dedupes_history(tmp_path: Path) -> None:
    paths = _prepare(tmp_path)

    first = _run(paths, whiteboard_enabled=True)
    assert first["status"] == "actionable"
    assert first["history"]["action"] == "appended"
    markdown = paths["out_md"].read_text(encoding="utf-8")
    assert "auto-generated review draft" in markdown
    assert "- Change:" in markdown
    assert "microlevelreactor" in markdown

    history_lines = paths["out_history"].read_text(encoding="utf-8").strip().splitlines()
    assert len(history_lines) == 1
    tasks = whiteboard.list_tasks(status="open", db_path=paths["whiteboard_db"])
    assert len(tasks) == 1
    events = whiteboard.list_events(task_id=tasks[0].id, db_path=paths["whiteboard_db"])
    assert len(events) == 1

    second = _run(paths, whiteboard_enabled=True)
    assert second["history"]["action"] == "skipped_duplicate_fingerprint"
    assert second["whiteboard"]["action"] == "skipped_duplicate_fingerprint"
    history_lines = paths["out_history"].read_text(encoding="utf-8").strip().splitlines()
    assert len(history_lines) == 1
    events = whiteboard.list_events(task_id=tasks[0].id, db_path=paths["whiteboard_db"])
    assert len(events) == 1

    pdca = json.loads(paths["pdca"].read_text(encoding="utf-8"))
    pdca["generated_at_utc"] = "2026-03-10T17:20:00+00:00"
    pdca["trades"]["24h"]["overall"]["net_jpy"] = -990.0
    _write_json(paths["pdca"], pdca)

    third = _run(paths, whiteboard_enabled=True)
    assert third["history"]["action"] == "appended"
    history_lines = paths["out_history"].read_text(encoding="utf-8").strip().splitlines()
    assert len(history_lines) == 2
    events = whiteboard.list_events(task_id=tasks[0].id, db_path=paths["whiteboard_db"])
    assert len(events) == 2
