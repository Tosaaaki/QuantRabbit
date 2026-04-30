from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEGACY_ARCHIVE = Path(
    os.environ.get(
        "QR_LEGACY_ARCHIVE",
        "/Users/tossaki/App/QuantRabbit_archives/QuantRabbit_legacy_20260430T151527Z",
    )
)
DEFAULT_HISTORY_DB = ROOT / "data" / "legacy_history.db"
DEFAULT_IMPORT_REPORT = ROOT / "docs" / "legacy_import_report.md"
DEFAULT_STRATEGY_PROFILE = ROOT / "data" / "strategy_profile.json"
DEFAULT_STRATEGY_REPORT = ROOT / "docs" / "strategy_mining_report.md"
