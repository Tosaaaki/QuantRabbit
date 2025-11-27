from importlib import import_module

__all__ = ["macro_core_worker"]


def __getattr__(name: str):
    if name != "macro_core_worker":
        raise AttributeError(f"module 'workers.macro_core' has no attribute {name!r}")
    module = import_module("workers.macro_core.worker")
    return module.macro_core_worker
