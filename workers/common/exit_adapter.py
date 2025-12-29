"""Helpers to adapt optional ExitManager usage in addon workers."""

from __future__ import annotations

import os
from typing import Any, Optional

try:  # pragma: no cover - ExitManager may be missing in lightweight runs
    from execution.exit_manager import ExitManager as _ExitManagerBase
except Exception:  # pragma: no cover
    _ExitManagerBase = None  # type: ignore[assignment]


def build_exit_manager(exit_cfg: Optional[dict[str, Any]]) -> Optional[Any]:
    """
    Attempt to construct an ExitManager instance that exposes ``attach``.

    The addon workers expect an ``attach(order_dict)`` helper to enrich orders
    with exit instructions.  The QuantRabbit runtime ships a richer
    ``execution.exit_manager.ExitManager`` used for pocket-level risk control,
    which does not provide that helper.  We therefore try to instantiate it
    defensively and only return the object if the expected helper exists.
    """

    # 共通EXITは既定で無効化（専用 exit_worker のみを使用）
    if str(os.getenv("EXIT_MANAGER_DISABLED", "1")).strip().lower() not in {"", "0", "false", "no"}:
        return None

    if not exit_cfg or _ExitManagerBase is None:
        return None

    candidate: Optional[Any] = None
    try:
        candidate = _ExitManagerBase(exit_cfg)  # type: ignore[arg-type]
    except TypeError:
        # Some implementations might be signature-based instead of dict-based.
        try:
            candidate = _ExitManagerBase(**exit_cfg)  # type: ignore[arg-type]
        except Exception:
            candidate = None
    except Exception:
        candidate = None

    if candidate is not None and hasattr(candidate, "attach"):
        return candidate

    return None
