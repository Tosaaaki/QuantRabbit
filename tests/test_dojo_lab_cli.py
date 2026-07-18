from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_lab_help_does_not_start_or_create_a_run(tmp_path: Path) -> None:
    for script_name in ("run-dojo-lab.py", "run-pair-adaptation-lab.py"):
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / script_name), "--help"],
            cwd=tmp_path,
            check=False,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout
        assert list(tmp_path.iterdir()) == []
