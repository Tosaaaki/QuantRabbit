from __future__ import annotations

import ast
import datetime as dt
from pathlib import Path
from typing import Callable, Dict, Optional

import pytest


def _load_dispatch_helper():
    worker_path = (
        Path(__file__).resolve().parents[2]
        / "workers/scalp_wick_reversal_blend/worker.py"
    )
    source = worker_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(worker_path))
    selected = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {
                    "_RANGE_CTX_SIGNAL_NAMES",
                    "_M5_SIGNAL_NAMES",
                }:
                    selected.append(node)
                    break
        elif isinstance(node, ast.FunctionDef) and node.name == "_dispatch_strategy_signal":
            selected.append(node)
    module = ast.Module(body=selected, type_ignores=[])
    namespace = {
        "Callable": Callable,
        "Dict": Dict,
        "Optional": Optional,
        "datetime": dt,
    }
    exec(compile(module, str(worker_path), "exec"), namespace)
    return namespace["_dispatch_strategy_signal"]


@pytest.mark.parametrize("name", ["DroughtRevert", "PrecisionLowVol"])
def test_dispatch_strategy_signal_passes_range_ctx(name: str) -> None:
    dispatch = _load_dispatch_helper()
    marker = object()

    def fake_signal(fac_m1, range_ctx, **kwargs):
        assert fac_m1 == {"close": 150.0}
        assert range_ctx is marker
        assert kwargs == {"tag": name}
        return {"tag": name}

    signal = dispatch(
        name=name,
        fn=fake_signal,
        fac_m1={"close": 150.0},
        fac_h1={},
        fac_m5={},
        range_ctx=marker,
        now_utc=dt.datetime(2026, 3, 10, 12, 0, 0),
        kwargs={"tag": name},
    )

    assert signal == {"tag": name}
