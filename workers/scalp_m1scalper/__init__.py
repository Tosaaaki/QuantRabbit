"""Scalp M1Scalper dedicated worker."""

__all__ = ["scalp_m1_worker"]


def __getattr__(name):
    if name == "scalp_m1_worker":
        from .worker import scalp_m1_worker

        return scalp_m1_worker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
