from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "replay_jpy_hour_sweep.py"
_spec = importlib.util.spec_from_file_location("replay_jpy_hour_sweep_script", _SCRIPT_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"failed to load script module: {_SCRIPT_PATH}")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


def _sample_report() -> dict:
    return {
        "worker_results": {
            "scalp_ping_5s_c_live": {
                "folds": [
                    {
                        "fold_id": 1,
                        "test_files": ["USD_JPY_ticks_20260127.jsonl"],
                        "test_metrics": {
                            "trade_count": 10.0,
                            "duration_hours": 5.0,
                            "total_jpy": 1000.0,
                            "avg_jpy": 100.0,
                            "jpy_per_hour": 200.0,
                        },
                        "gate": {
                            "checks": [
                                {"name": "test_trade_count", "ok": True, "actual": 10.0},
                                {"name": "test_jpy_per_hour", "ok": True, "actual": 200.0},
                            ]
                        },
                    },
                    {
                        "fold_id": 2,
                        "test_files": ["USD_JPY_ticks_20260128.jsonl"],
                        "test_metrics": {
                            "trade_count": 6.0,
                            "duration_hours": 3.0,
                            "total_jpy": 360.0,
                            "avg_jpy": 60.0,
                            "jpy_per_hour": 120.0,
                        },
                        "gate": {
                            "checks": [
                                {"name": "test_trade_count", "ok": True, "actual": 6.0},
                                {"name": "test_jpy_per_hour", "ok": True, "actual": 120.0},
                            ]
                        },
                    },
                ]
            }
        }
    }


def test_gate_ok_respects_overridden_jpy_per_hour_threshold() -> None:
    report = _sample_report()
    folds = report["worker_results"]["scalp_ping_5s_c_live"]["folds"]

    assert _module._gate_ok(folds[0], min_test_jpy_per_hour=150.0) is True
    assert _module._gate_ok(folds[1], min_test_jpy_per_hour=150.0) is False


def test_calc_target_feasibility_computes_required_values() -> None:
    report = _sample_report()
    folds = report["worker_results"]["scalp_ping_5s_c_live"]["folds"]
    result = _module._calc_target_feasibility(folds, target_jpy_per_hour=2000.0)
    agg = result["aggregate"]

    assert round(float(agg["total_jpy"]), 6) == 1360.0
    assert round(float(agg["total_hours"]), 6) == 8.0
    assert round(float(agg["total_trades"]), 6) == 16.0
    assert round(float(agg["achieved_jpy_per_hour"]), 6) == 170.0
    assert round(float(agg["achieved_trades_per_hour"]), 6) == 2.0
    assert round(float(agg["achieved_jpy_per_trade"]), 6) == 85.0
    assert round(float(agg["required_trades_per_hour_at_current_ev"]), 6) == round(2000.0 / 85.0, 6)
    assert round(float(agg["required_jpy_per_trade_at_current_freq"]), 6) == 1000.0


def test_script_main_outputs_json_and_md(tmp_path: Path) -> None:
    report_path = tmp_path / "quality_gate_report.json"
    report_path.write_text(json.dumps(_sample_report(), ensure_ascii=False), encoding="utf-8")
    out_json = tmp_path / "out.json"
    out_md = tmp_path / "out.md"

    import subprocess, sys

    cmd = [
        sys.executable,
        str(_SCRIPT_PATH),
        "--report",
        str(report_path),
        "--worker",
        "scalp_ping_5s_c_live",
        "--thresholds",
        "150,300",
        "--target-jpy-per-hour",
        "2000",
        "--required-pass-rate",
        "1.0",
        "--out-json",
        str(out_json),
        "--out-md",
        str(out_md),
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    assert proc.returncode == 0, proc.stderr

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["worker"] == "scalp_ping_5s_c_live"
    assert len(payload["sweep"]) == 2
    assert payload["sweep"][0]["status"] == "fail"
    assert payload["sweep"][1]["status"] == "fail"
    assert out_md.exists()
