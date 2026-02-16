"""Compatibility package for legacy micro_trendretest worker entrypoints."""

from .worker import micro_trendretest_worker
from .exit_worker import micro_trendretest_exit_worker

__all__ = ["micro_trendretest_worker", "micro_trendretest_exit_worker"]

