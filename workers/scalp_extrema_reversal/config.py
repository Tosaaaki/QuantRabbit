from __future__ import annotations

import os

ENV_PREFIX = os.getenv(
    "SCALP_EXTREMA_REVERSAL_EXIT_ENV_PREFIX",
    os.getenv("SCALP_EXTREMA_REVERSAL_ENV_PREFIX", "SCALP_EXTREMA_REVERSAL"),
)
