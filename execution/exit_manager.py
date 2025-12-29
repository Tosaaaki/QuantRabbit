"""
Legacy ExitManager stub
=======================
旧型の共通EXITロジックを完全撤去し、呼び出し互換のスタブだけを残す。
plan_closures は常に空リストを返し、自動クローズを一切発生させない。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List

LOG = logging.getLogger(__name__)


@dataclass
class ExitDecision:
    pocket: str
    units: int
    reason: str
    tag: str
    allow_reentry: bool = False


class ExitManager:
    def __init__(self, *_: Any, **__: Any) -> None:
        LOG.info("[EXIT_MANAGER] legacy manager removed; no auto exits")

    def plan_closures(self, *_: Any, **__: Any) -> List[ExitDecision]:
        """No-op: auto exits are disabled."""
        return []

    def attach(self, order: dict) -> dict:
        """Compatibility helper for addons expecting attach()."""
        order = order or {}
        order.setdefault("exit_disabled", True)
        return order

