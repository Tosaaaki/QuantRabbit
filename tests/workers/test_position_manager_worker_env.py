from __future__ import annotations

import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from workers.position_manager import worker


def test_service_port_defaults_to_8301(monkeypatch) -> None:
    monkeypatch.delenv("POSITION_MANAGER_SERVICE_PORT", raising=False)

    assert worker._service_port() == 8301


def test_service_port_reads_env(monkeypatch) -> None:
    monkeypatch.setenv("POSITION_MANAGER_SERVICE_PORT", "9315")

    assert worker._service_port() == 9315
