from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


REPO = Path(__file__).resolve().parents[1]


def load_tool() -> ModuleType:
    path = REPO / "tools/run-dojo-ai-model-cell.py"
    spec = importlib.util.spec_from_file_location("run_dojo_ai_model_cell", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_nested_codex_parent_and_api_keys_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    tool = load_tool()
    for name in (*tool._PARENT_CONTEXT_MARKERS, *tool._FORBIDDEN_PROVIDER_ENV):
        monkeypatch.delenv(name, raising=False)
    tool._assert_external_boundary()
    monkeypatch.setenv("CODEX_THREAD_ID", "nested")
    with pytest.raises(RuntimeError, match="external non-Codex parent"):
        tool._assert_external_boundary()
    monkeypatch.delenv("CODEX_THREAD_ID")
    monkeypatch.setenv("OPENAI_API_KEY", "must-never-be-read-or-forwarded")
    with pytest.raises(RuntimeError, match="API key variables"):
        tool._assert_external_boundary()


def test_subprocess_environment_is_exact_allowlist(tmp_path: Path) -> None:
    tool = load_tool()
    env = tool._clean_env(
        code_home=tmp_path / "codex-home",
        home=tmp_path / "home",
        tmpdir=tmp_path,
    )
    assert set(env) == {"CODEX_HOME", "HOME", "LANG", "LC_ALL", "PATH", "TMPDIR"}
    assert not set(tool._FORBIDDEN_PROVIDER_ENV).intersection(env)
    assert "CODEX_THREAD_ID" not in env


def test_command_is_one_ephemeral_no_tool_turn(tmp_path: Path) -> None:
    tool = load_tool()
    request = {
        "requested_model": "gpt-5.5",
        "reasoning_effort": "high",
        "runtime": {
            "disabled_features": [
                "shell_tool",
                "unified_exec",
                "apps",
                "plugins",
                "browser_use",
                "computer_use",
                "multi_agent",
            ]
        },
    }
    command = tool._command(
        Path("/Applications/ChatGPT.app/Contents/Resources/codex"),
        tmp_path / "empty",
        tmp_path / "schema.json",
        tmp_path / "last.json",
        request,
    )
    assert "--ephemeral" in command
    assert "--ignore-user-config" in command
    assert "--ignore-rules" in command
    assert "--strict-config" in command
    assert "--skip-git-repo-check" in command
    assert command.count("--model") == 1
    assert "resume" not in command
    assert "danger-full-access" not in command
    for feature in request["runtime"]["disabled_features"]:
        position = command.index(feature)
        assert command[position - 1] == "--disable"
