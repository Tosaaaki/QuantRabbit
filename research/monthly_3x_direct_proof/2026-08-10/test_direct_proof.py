from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent


def load_module():
    spec = importlib.util.spec_from_file_location("direct_proof", HERE / "run_direct_proof.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_adjacent_plateau() -> None:
    module = load_module()
    assert module.adjacent_plateau({12, 24}, [12, 24, 48])
    assert module.adjacent_plateau({24, 48}, [12, 24, 48])
    assert not module.adjacent_plateau({12, 48}, [12, 24, 48])


def test_report_keeps_holdout_sealed_and_uses_full_target() -> None:
    report = json.loads((HERE / "report_v1.json").read_text())
    prereg = json.loads((HERE / "preregister_v1.json").read_text())
    assert report["holdout_used"] is False
    assert prereg["target"]["starting_equity_jpy"] == 200000
    assert prereg["target"]["ending_equity_jpy"] == 600000
    assert prereg["target"]["required_profit_jpy"] == 400000


def test_every_declared_pass_meets_all_constraints() -> None:
    report = json.loads((HERE / "report_v1.json").read_text())
    for row in report["monthly_3x_passes"]:
        assert row["projected_30d_net_jpy"] >= 400000
        assert row["projected_30d_lcb_jpy"] >= 400000
        assert row["scaled_realized_dd_jpy"] <= 80000
        assert row["validation_32d"]["validation_pass"]
        assert row["validation_64d"]["validation_pass"]


def test_signal_quality_refine_keeps_holdout_sealed() -> None:
    report = json.loads((HERE / "signal_quality_report_v2.json").read_text())
    assert report["holdout_used"] is False
    for candidate in report["stable_multiwindow_candidates"]:
        assert candidate["validation_32d"]["validation_pass"]
        assert candidate["validation_64d"]["validation_pass"]
