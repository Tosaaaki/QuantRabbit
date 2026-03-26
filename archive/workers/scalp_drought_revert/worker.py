from __future__ import annotations

import asyncio
import logging

from workers.scalp_precision_wrapper import apply_precision_mode_env


def _apply_alt_env() -> None:
    apply_precision_mode_env(
        "SCALP_PRECISION_DROUGHT_REVERT",
        mode="drought_revert",
        fallback_log_prefix="[Scalp:DroughtRevert]",
    )


def main() -> None:
    _apply_alt_env()
    from workers.scalp_wick_reversal_blend.worker import (
        scalp_wick_reversal_blend_worker,
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )
    asyncio.run(scalp_wick_reversal_blend_worker())


if __name__ == "__main__":
    main()
