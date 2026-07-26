from __future__ import annotations

import importlib.util
from datetime import datetime, timezone
from pathlib import Path

from quant_rabbit.dojo_ai_inventory import (
    DOJO_AI_INVENTORY_CONSUMER_CONTRACT,
    DOJO_AI_INVENTORY_DECISION_CONTRACT,
    DOJO_AI_INVENTORY_DECISION_ROLE,
)


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run-dojo-ai-inventory-decision.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_dojo_ai_inventory_decision",
    SCRIPT,
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load script: {SCRIPT}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_preflight_pauses_every_ai_inventory_capability_on_weekend() -> None:
    result = MODULE._market_preflight(datetime(2026, 7, 25, 12, 0, tzinfo=timezone.utc))
    assert result["status"] == "MARKET_CLOSED_AI_INVENTORY_PAUSED"
    assert result["fx_market_open"] is False
    assert result["ai_assessment_allowed"] is False
    assert result["ai_inventory_decision_allowed"] is False
    assert result["virtual_action_allowed"] is False
    assert result["virtual_broker_mutation_allowed"] is False
    assert result["external_broker_mutation_allowed"] is False
    assert result["decision_contract"] == DOJO_AI_INVENTORY_DECISION_CONTRACT
    assert result["consumer_contract"] == DOJO_AI_INVENTORY_CONSUMER_CONTRACT
    assert result["decision_role"] == DOJO_AI_INVENTORY_DECISION_ROLE
    assert result["order_authority"] == "NONE"
    assert result["live_permission"] is False


def test_preflight_allows_only_decision_writer_during_open_market() -> None:
    result = MODULE._market_preflight(datetime(2026, 7, 22, 18, 0, tzinfo=timezone.utc))
    assert result["status"] == "READY_FOR_DECISION_WRITER"
    assert result["fx_market_open"] is True
    assert result["ai_assessment_allowed"] is True
    assert result["ai_inventory_decision_allowed"] is True
    assert result["virtual_action_allowed"] is False
    assert result["virtual_broker_mutation_allowed"] is True
    assert result["external_broker_mutation_allowed"] is False
    assert result["decision_contract"] == "QR_DOJO_AI_INVENTORY_DECISION_V2"
