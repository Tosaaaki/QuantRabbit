from __future__ import annotations

import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import execution.order_manager as order_manager
import execution.strategy_entry as strategy_entry


def test_strategy_entry_infers_flow_prefix() -> None:
    assert (
        strategy_entry._infer_env_prefix_from_strategy_tag("scalp_ping_5s_flow_live")
        == "SCALP_PING_5S_FLOW"
    )


def test_order_manager_prefers_flow_prefix() -> None:
    assert (
        order_manager._infer_env_prefix_from_strategy_tag("scalp_ping_5s_flow_live")
        == "SCALP_PING_5S_FLOW"
    )


def test_order_manager_prefers_c_prefix() -> None:
    assert (
        order_manager._infer_env_prefix_from_strategy_tag("scalp_ping_5s_c_live")
        == "SCALP_PING_5S_C"
    )


def test_strategy_entry_prefers_c_prefix() -> None:
    assert (
        strategy_entry._infer_env_prefix_from_strategy_tag("scalp_ping_5s_c_live")
        == "SCALP_PING_5S_C"
    )
