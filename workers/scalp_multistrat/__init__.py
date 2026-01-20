"""Scalp multi-strategy worker (RangeFader/PulseBreak/ImpulseRetrace)."""

__all__ = ["scalp_multi_worker"]


def __getattr__(name: str):
    if name == "scalp_multi_worker":
        from .worker import scalp_multi_worker

        return scalp_multi_worker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
