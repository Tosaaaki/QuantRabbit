#!/usr/bin/env python3
"""Future-only paper runtime with completed M1 bot decisions.

This wrapper deliberately leaves ``run-virtual-market-session.py`` unchanged
for already-fixed controls.  It reuses that runner's paper broker, contracts,
replay path, and CLI, replacing only the live bot-decision loop.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from quant_rabbit.dojo_completed_m1_live import (  # noqa: E402
    COMPLETED_M1_SOURCE,
    make_completed_m1_run_live,
)


def _frozen_runtime() -> Any:
    path = REPO_ROOT / "scripts/run-virtual-market-session.py"
    spec = importlib.util.spec_from_file_location(
        "dojo_frozen_virtual_market_runtime",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frozen virtual runtime: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    runtime = _frozen_runtime()
    completed_m1_run_live = make_completed_m1_run_live(runtime)

    def run_live_with_error_receipt(
        args: Any,
        broker: Any,
        session_dir: Path,
        bot: Any = None,
    ) -> None:
        try:
            completed_m1_run_live(args, broker, session_dir, bot=bot)
        except Exception as exc:
            body = {
                "contract": "QR_DOJO_COMPLETED_M1_RUNTIME_ERROR_V1",
                "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:400],
                "paper_only": True,
                "order_authority": "NONE",
                "live_permission": False,
                "broker_mutation_allowed": False,
            }
            digest = hashlib.sha256(
                json.dumps(
                    body,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            broker._log(
                "SESSION_ERROR",
                {**body, "session_error_sha256": digest},
            )
            raise

    runtime.run_live = run_live_with_error_receipt

    original_contract_builder = runtime.build_session_contract

    def completed_m1_contract_builder(**kwargs: Any) -> dict[str, Any]:
        source = dict(kwargs.get("source") or {})
        if kwargs.get("feed") == "live":
            source.update(
                {
                    "kind": (
                        "live_read_only_pricing_with_completed_m1_decisions"
                    ),
                    "granularity": "M1",
                    "bot_bar_source": COMPLETED_M1_SOURCE,
                    "decision_no_lookahead": True,
                    "fills_use_executable_poll_quotes": True,
                    "seed_cutoff_policy": (
                        "fixed_window_start_exclusive_completed_m1"
                    ),
                    "seed_gap_policy": (
                        "resume_from_actual_last_seed_bar_then_seed_only"
                    ),
                    "restart_policy": (
                        "persist_cutoff_before_action_then_seed_only_restore"
                    ),
                }
            )
        kwargs["source"] = source
        return original_contract_builder(**kwargs)

    runtime.build_session_contract = completed_m1_contract_builder
    # The frozen main function binds Path(__file__) into its runtime manifest.
    # Point it at this opt-in wrapper so a future room cannot masquerade as a
    # fixed-control runtime.
    runtime.__file__ = str(Path(__file__).resolve())
    return int(runtime.main())


if __name__ == "__main__":
    raise SystemExit(main())
