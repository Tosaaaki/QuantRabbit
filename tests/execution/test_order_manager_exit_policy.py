from __future__ import annotations

import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from execution.order_manager import _neg_exit_decision


def test_neg_exit_policy_explicit_reason_allows_without_global_gate() -> None:
    allowed, near_be = _neg_exit_decision(
        exit_reason="trend_failure",
        est_pips=-5.0,
        emergency_allow=False,
        reason_allow=False,
        worker_allow=False,
        neg_policy={
            "enabled": True,
            "allow_reasons": ["trend_failure"],
            "deny_reasons": [],
        },
    )
    assert allowed is True
    assert near_be is False


def test_neg_exit_policy_blocks_reason_outside_strategy_allow_list() -> None:
    allowed, near_be = _neg_exit_decision(
        exit_reason="max_adverse",
        est_pips=-8.0,
        emergency_allow=False,
        reason_allow=True,  # global allow alone should not bypass strategy deny-by-omission
        worker_allow=False,
        neg_policy={
            "enabled": True,
            "allow_reasons": ["trend_failure"],
            "deny_reasons": [],
        },
    )
    assert allowed is False
    assert near_be is False


def test_neg_exit_policy_near_be_path_still_works() -> None:
    allowed, near_be = _neg_exit_decision(
        exit_reason="near_be",
        est_pips=-0.2,
        emergency_allow=False,
        reason_allow=False,
        worker_allow=False,
        neg_policy={
            "enabled": True,
            "allow_reasons": ["near_be"],
            "deny_reasons": [],
        },
    )
    assert near_be is True
    assert allowed is True
