from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_TEMPLATE = REPO_ROOT / "templates" / "autotune" / "base.html"
DASHBOARD_TEMPLATE = REPO_ROOT / "templates" / "autotune" / "dashboard.html"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_tabs_init_resets_ready_class_when_tabs_absent() -> None:
    source = _read(BASE_TEMPLATE)
    assert 'document.body.classList.remove("tabs-ready");' in source


def test_auto_refresh_reschedule_clears_previous_timers() -> None:
    source = _read(BASE_TEMPLATE)
    assert "const clearScheduledJobs = () => {" in source
    assert re.search(
        r"const schedule = \(refreshRoot, intervalSec, mode\) => \{\s*clearScheduledJobs\(\);",
        source,
    )
    assert re.search(
        r"if \(document\.hidden\) \{\s*clearScheduledJobs\(\);",
        source,
    )


def test_dashboard_chart_storage_access_is_guarded() -> None:
    source = _read(DASHBOARD_TEMPLATE)
    assert "const safeSessionGet = (key) => {" in source
    assert "const safeSessionSet = (key, value) => {" in source
    assert "const saved = safeSessionGet(storageKey);" in source
    assert 'const chartMode = safeSessionGet("qr_chart_mode");' in source
