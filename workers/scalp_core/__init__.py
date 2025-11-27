from importlib import import_module

__all__ = ["scalp_core_worker"]


def __getattr__(name: str):
    if name != "scalp_core_worker":
        raise AttributeError(f"module 'workers.scalp_core' has no attribute {name!r}")
    module = import_module("workers.scalp_core.worker")
    return module.scalp_core_worker
