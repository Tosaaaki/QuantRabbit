"""Micro BB_RSI dedicated worker."""

__all__ = ["micro_bbrsi_worker"]


def __getattr__(name: str):
    if name == "micro_bbrsi_worker":
        from .worker import micro_bbrsi_worker

        return micro_bbrsi_worker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
