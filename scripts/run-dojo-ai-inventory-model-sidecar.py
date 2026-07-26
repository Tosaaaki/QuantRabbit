#!/usr/bin/env python3
"""Fixed paper-only entry point for the signed AI inventory model sidecar."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from quant_rabbit.dojo_ai_inventory_model_sidecar import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
