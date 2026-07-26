from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _summary_module():
    path = (
        Path(__file__).parents[1]
        / "scripts"
        / "summarize-dojo-paired-inventory-counterfactual.py"
    )
    spec = importlib.util.spec_from_file_location("dojo_paired_summary", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_oos_aggregate_requires_eight_unique_measured_blocks() -> None:
    module = _summary_module()
    rows = [
        {
            "block_id": f"OOS_{index:02d}",
            "status": "MEASURED_EXPERIMENTAL",
            "bot_only_net_jpy": float(index),
            "ai_managed_net_jpy": float(index + 1),
            "bot_only_max_drawdown_fraction": index / 100,
            "ai_managed_max_drawdown_fraction": index / 200,
            "bot_only_peak_margin_usage_fraction": index / 50,
            "ai_managed_peak_margin_usage_fraction": index / 75,
        }
        for index in range(1, 9)
    ]

    aggregate = module._oos_aggregate({"oos_block_rows": rows})

    assert aggregate["bot_only_net_jpy"] == 36
    assert aggregate["ai_managed_net_jpy"] == 44
    assert aggregate["bot_only_max_within_block_drawdown_fraction"] == 0.08
    assert aggregate["ai_max_within_block_drawdown_fraction"] == 0.04


def test_oos_aggregate_rejects_missing_fixed_denominator() -> None:
    module = _summary_module()

    with pytest.raises(ValueError, match="fixed denominator"):
        module._oos_aggregate({"oos_block_rows": []})
