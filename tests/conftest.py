from __future__ import annotations

# Pytest's default import-mode prepends the tests directory to sys.path, which can
# accidentally shadow the project's top-level packages (e.g. "workers" vs
# "tests/workers"). Force the repo root to the front so imports resolve to the
# production code.

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
