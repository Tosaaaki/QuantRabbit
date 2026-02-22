from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "replay_quality_gate.py"
_spec = importlib.util.spec_from_file_location("replay_quality_gate_script", _SCRIPT_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"failed to load script module: {_SCRIPT_PATH}")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


def _arg_value(cmd: list[str], flag: str) -> str | None:
    for idx, token in enumerate(cmd):
        if token == flag and idx + 1 < len(cmd):
            return cmd[idx + 1]
    return None


def test_load_worker_trades_main_filters_and_excludes_end_reason(tmp_path: Path) -> None:
    out_path = tmp_path / "replay_exit_workers.json"
    payload = {
        "trades": [
            {
                "strategy_tag": "TrendMA",
                "pocket": "macro",
                "source": "strategy",
                "reason": "tp_hit",
                "pnl_pips": 1.2,
            },
            {
                "strategy": "TrendMA",
                "pocket": "macro",
                "source": "strategy",
                "reason": "end_of_replay",
                "pnl_pips": 0.3,
            },
            {
                "strategy_tag": "BB_RSI",
                "pocket": "micro",
                "source": "strategy",
                "reason": "sl_hit",
                "pnl_pips": -0.8,
            },
            {
                "strategy": "MicroRangeBreak",
                "pocket": "micro",
                "source": "scalp_replay",
                "reason": "market_close",
                "pnl_pips": 0.5,
            },
        ]
    }
    out_path.write_text(json.dumps(payload), encoding="utf-8")

    workers = ["__overall__", "TrendMA", "BB_RSI", "pocket:micro", "source:strategy", "source:scalp_replay"]
    grouped = _module._load_worker_trades_main(
        out_path=out_path,
        workers=workers,
        exclude_end_of_replay=True,
    )

    assert len(grouped["__overall__"]) == 3
    assert len(grouped["TrendMA"]) == 1
    assert len(grouped["BB_RSI"]) == 1
    assert len(grouped["pocket:micro"]) == 2
    assert len(grouped["source:strategy"]) == 2
    assert len(grouped["source:scalp_replay"]) == 1


def test_build_replay_command_main_realistic_no_duplicate_flags() -> None:
    cmd = _module._build_replay_command_main(
        ticks_path=Path("/tmp/USD_JPY_ticks_20260210.jsonl"),
        out_path=Path("/tmp/out.json"),
        replay_cfg={
            "realistic": True,
            "fill_mode": "next_tick",
            "latency_ms": 250.0,
            "intraday_start_utc": "00:00:00",
            "intraday_end_utc": "06:00:00",
            "main_only": True,
        },
    )

    assert cmd.count("--fill-mode") == 1
    assert _arg_value(cmd, "--fill-mode") == "next_tick"

    # realistic defaults are kept unless explicitly overridden
    assert _arg_value(cmd, "--latency-ms") == "250.0"
    assert _arg_value(cmd, "--slip-base-pips") == "0.02"
    assert _arg_value(cmd, "--slip-spread-coef") == "0.15"
    assert _arg_value(cmd, "--slip-atr-coef") == "0.02"
    assert _arg_value(cmd, "--slip-latency-coef") == "0.0006"
    assert "--main-only" in cmd
    assert _arg_value(cmd, "--start") == "2026-02-10T00:00:00+00:00"
    assert _arg_value(cmd, "--end") == "2026-02-10T06:00:00+00:00"


def test_build_replay_command_main_explicit_start_end_override_intraday() -> None:
    cmd = _module._build_replay_command_main(
        ticks_path=Path("/tmp/USD_JPY_ticks_20260210.jsonl"),
        out_path=Path("/tmp/out.json"),
        replay_cfg={
            "start": "2026-02-10T01:00:00+00:00",
            "end": "2026-02-10T02:00:00+00:00",
            "intraday_start_utc": "00:00:00",
            "intraday_end_utc": "06:00:00",
        },
    )

    assert _arg_value(cmd, "--start") == "2026-02-10T01:00:00+00:00"
    assert _arg_value(cmd, "--end") == "2026-02-10T02:00:00+00:00"


def test_filter_tick_files_by_min_lines(tmp_path: Path) -> None:
    p1 = tmp_path / "USD_JPY_ticks_1.jsonl"
    p2 = tmp_path / "USD_JPY_ticks_2.jsonl"
    p3 = tmp_path / "USD_JPY_ticks_3.jsonl"
    p1.write_text("a\nb\nc\n", encoding="utf-8")
    p2.write_text("a\n", encoding="utf-8")
    p3.write_text("", encoding="utf-8")

    kept, dropped = _module._filter_tick_files_by_min_lines(
        [p1, p2, p3],
        min_lines=2,
    )

    assert [p.name for p in kept] == [p1.name]
    assert dropped == {p2.name: 1, p3.name: 0}


def test_collect_tick_files_deduplicates_by_basename_and_prefers_larger_file(tmp_path: Path) -> None:
    d1 = tmp_path / "replay"
    d2 = tmp_path / "archive"
    d1.mkdir(parents=True, exist_ok=True)
    d2.mkdir(parents=True, exist_ok=True)

    # Same basename in two roots; archive copy is larger and should be preferred.
    current = d1 / "USD_JPY_ticks_20260210.jsonl"
    archive = d2 / "USD_JPY_ticks_20260210.jsonl"
    other = d2 / "USD_JPY_ticks_20260211.jsonl"
    current.write_text("a\n", encoding="utf-8")
    archive.write_text("a\nb\nc\n", encoding="utf-8")
    other.write_text("a\nb\n", encoding="utf-8")

    tick_files, matched_count, duplicate_count = _module._collect_tick_files(
        [str(d1 / "USD_JPY_ticks_*.jsonl"), str(d2 / "USD_JPY_ticks_*.jsonl")]
    )

    assert matched_count == 3
    assert duplicate_count == 1
    assert [p.name for p in tick_files] == [
        "USD_JPY_ticks_20260210.jsonl",
        "USD_JPY_ticks_20260211.jsonl",
    ]
    assert tick_files[0] == archive


def test_resolve_ticks_globs_prefers_cli_then_list_then_legacy() -> None:
    assert _module._resolve_ticks_globs({}, "a/*.jsonl, b/*.jsonl") == ["a/*.jsonl", "b/*.jsonl"]
    assert _module._resolve_ticks_globs({"ticks_globs": ["x/*.jsonl", "y/*.jsonl"]}, None) == [
        "x/*.jsonl",
        "y/*.jsonl",
    ]
    assert _module._resolve_ticks_globs({"ticks_glob": "z/*.jsonl"}, None) == ["z/*.jsonl"]


def test_build_replay_env_applies_overrides_and_bool_coercion(monkeypatch) -> None:
    monkeypatch.setenv("EXISTING_KEY", "keep")
    monkeypatch.setenv("REMOVE_ME", "x")
    env, overrides = _module._build_replay_env(
        {
            "env": {
                "SCALP_REPLAY_MODE": "scalp_ping_5s_c",
                "FLAG_BOOL": True,
                "REMOVE_ME": None,
            }
        }
    )

    assert env["EXISTING_KEY"] == "keep"
    assert env["SCALP_REPLAY_MODE"] == "scalp_ping_5s_c"
    assert env["FLAG_BOOL"] == "1"
    assert "REMOVE_ME" not in env
    assert overrides == {
        "SCALP_REPLAY_MODE": "scalp_ping_5s_c",
        "FLAG_BOOL": "1",
    }


def test_build_threshold_supports_jpy_fields() -> None:
    threshold = _module._build_threshold(
        base={
            "min_test_total_jpy": 0.0,
            "min_test_jpy_per_hour": 10.0,
            "max_test_drawdown_jpy": 2500.0,
        },
        override={"min_test_total_jpy": 100.0},
    )

    assert threshold.min_test_total_jpy == 100.0
    assert threshold.min_test_jpy_per_hour == 10.0
    assert threshold.max_test_drawdown_jpy == 2500.0
