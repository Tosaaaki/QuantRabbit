from __future__ import annotations

# Pytest's default import-mode prepends the tests directory to sys.path, which can
# accidentally shadow the project's top-level packages (e.g. "workers" vs
# "tests/workers"). Force the repo root to the front so imports resolve to the
# production code.

import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Unit tests should be deterministic and must not depend on the operator's
# current tuning overlays (which are frequently updated for live trading).
# Keep presets enabled (baseline), but disable overlay/overrides in tests.
os.environ.setdefault("TUNING_OVERLAY_PATH", str(ROOT / "config" / "__disabled_for_tests__.yaml"))
os.environ.setdefault("TUNING_OVERRIDES_PATH", str(ROOT / "config" / "__disabled_for_tests__.yaml"))
