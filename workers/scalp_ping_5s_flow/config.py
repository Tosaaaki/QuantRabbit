from __future__ import annotations

import os

ENV_PREFIX = (
    os.getenv("SCALP_PING_5S_FLOW_EXIT_ENV_PREFIX", os.getenv("SCALP_PING_5S_FLOW_ENV_PREFIX", "SCALP_PING_5S_FLOW")).strip()
    or "SCALP_PING_5S_FLOW"
)
