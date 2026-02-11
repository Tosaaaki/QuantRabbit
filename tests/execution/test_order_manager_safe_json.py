from __future__ import annotations

import json
import math

from execution.order_manager import _safe_json


def _strict_loads(text: str) -> object:
    """Reject NaN/Infinity tokens to ensure output is strict JSON."""

    def _reject(token: str) -> float:  # pragma: no cover - invoked only on invalid JSON
        raise ValueError(token)

    return json.loads(text, parse_constant=_reject)


def test_safe_json_coerces_non_finite_floats_to_null() -> None:
    payload = {
        "inf": float("inf"),
        "ninf": float("-inf"),
        "nan": float("nan"),
        "ok": 1.25,
        "nested": {"x": float("inf")},
        "arr": [1, float("inf"), 2],
    }

    out = _safe_json(payload)
    obj = _strict_loads(out)

    assert isinstance(obj, dict)
    assert obj["ok"] == 1.25
    assert obj["inf"] is None
    assert obj["ninf"] is None
    assert obj["nan"] is None
    assert obj["nested"]["x"] is None
    assert obj["arr"] == [1, None, 2]


def test_safe_json_preserves_finite_floats() -> None:
    payload = {"a": 0.0, "b": -1.5, "c": math.pi}
    out = _safe_json(payload)
    obj = _strict_loads(out)
    assert obj == payload

