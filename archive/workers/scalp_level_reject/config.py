from __future__ import annotations

import os

ENV_PREFIX = os.getenv(
    "SCALP_LEVEL_REJECT_EXIT_ENV_PREFIX",
    os.getenv("SCALP_LEVEL_REJECT_ENV_PREFIX", "SCALP_LEVEL_REJECT"),
)
