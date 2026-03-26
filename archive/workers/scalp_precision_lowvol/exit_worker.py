from __future__ import annotations

import asyncio
import logging

from workers.scalp_precision_wrapper import apply_precision_exit_env


def _apply_exit_env() -> None:
    apply_precision_exit_env(
        "SCALP_PRECISION_LOWVOL",
        exit_tags="PrecisionLowVol",
        fallback_log_prefix="[ScalpExit:PrecisionLowVol]",
    )


def main() -> None:
    _apply_exit_env()
    from workers.scalp_wick_reversal_blend.exit_worker import (
        scalp_wick_reversal_blend_exit_worker,
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )
    asyncio.run(scalp_wick_reversal_blend_exit_worker())


if __name__ == "__main__":
    main()
