from importlib import import_module

__all__ = ["micro_core_worker"]


def __getattr__(name: str):
    if name == "micro_core_worker":
        module = import_module("workers.micro_core.worker")
        return module.micro_core_worker
    raise AttributeError(name)
