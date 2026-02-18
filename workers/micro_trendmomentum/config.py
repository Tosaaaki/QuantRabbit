from __future__ import annotations

import os

ENV_PREFIX = (
    os.getenv("MICRO_MULTI_EXIT_ENV_PREFIX", os.getenv("MICRO_MULTI_ENV_PREFIX", "MICRO_MULTI")).strip()
    or "MICRO_MULTI"
)
