"""Micro multi-strategy worker (MomentumBurst/Stack/Pullback/Level/VWAP/RangeBreak/TrendMomentum)."""

__all__ = ["micro_multi_worker"]


def __getattr__(name: str):
    if name == "micro_multi_worker":
        from .worker import micro_multi_worker

        return micro_multi_worker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
