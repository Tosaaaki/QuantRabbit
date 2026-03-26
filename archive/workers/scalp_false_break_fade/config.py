from __future__ import annotations

import os

ENV_PREFIX = os.getenv(
    "SCALP_FALSE_BREAK_FADE_EXIT_ENV_PREFIX",
    os.getenv("SCALP_FALSE_BREAK_FADE_ENV_PREFIX", "SCALP_FALSE_BREAK_FADE"),
)
