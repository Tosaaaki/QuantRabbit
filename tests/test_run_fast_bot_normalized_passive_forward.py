from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_fast_bot_normalized_passive_forward",
    ROOT / "tools" / "run_fast_bot_normalized_passive_forward.py",
)
assert SPEC and SPEC.loader
resident = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(resident)


def test_resident_plist_is_independent_get_only_observer(tmp_path: Path) -> None:
    args = SimpleNamespace(
        expected_commit="a" * 40,
        expected_source_sha256="b" * 64,
        state_root=tmp_path,
        oanda_env_file=Path("/approved/.env.local"),
        interval_seconds=20.0,
    )
    plist = resident.desired_plist(args)
    assert plist["Label"] == "com.quantrabbit.normalized-passive-forward"
    assert plist["KeepAlive"] is True
    assert plist["RunAtLoad"] is True
    joined = " ".join(plist["ProgramArguments"]).lower()
    assert "normalized_passive_forward" in joined
    assert "--send" not in joined
    assert "--confirm-live" not in joined
    assert "position-execution" not in joined
    assert resident.POLICY_PATH == Path(
        "config/fast_bot_normalized_passive_forward_v2.json"
    )


def test_resident_status_hard_codes_zero_authority(tmp_path: Path) -> None:
    status = resident._base_status(
        manifest={
            "commit": "a" * 40,
            "source_bundle_sha256": "b" * 64,
        },
        root=tmp_path,
        started_at_utc="2026-09-01T12:00:00+00:00",
        restart_count=1,
        cycle_count=0,
        cycle_failures=0,
    )
    assert status["contract"] == "QR_FAST_BOT_NORMALIZED_PASSIVE_FORWARD_RESIDENT_V2"
    assert status["schema_version"] == 2
    assert status["execution_authority"] == "NONE"
    assert status["broker_http_methods_allowed"] == ["GET"]
    assert status["broker_mutation_allowed"] is False
    assert status["external_order_attempts"] == 0
    assert status["external_orders"] == 0
    assert status["gateway_invocations"] == 0
    assert status["live_permission"] is False
    assert status["promotion_allowed"] is False
