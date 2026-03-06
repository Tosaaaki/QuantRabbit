#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Iterable

from openai import OpenAI
from PIL import ImageGrab

_BUTTON_MAP = {
    "left": "left",
    "right": "right",
    "wheel": "middle",
}

_KEY_ALIASES = {
    "ALT": "alt",
    "BACKSPACE": "backspace",
    "CMD": "command",
    "CTRL": "ctrl",
    "DELETE": "delete",
    "DOWN": "down",
    "END": "end",
    "ENTER": "enter",
    "ESC": "esc",
    "HOME": "home",
    "LEFT": "left",
    "OPTION": "option",
    "PAGEDOWN": "pagedown",
    "PAGEUP": "pageup",
    "PGDN": "pagedown",
    "PGUP": "pageup",
    "RETURN": "enter",
    "RIGHT": "right",
    "SHIFT": "shift",
    "SPACE": "space",
    "SUPER": "command",
    "TAB": "tab",
    "UP": "up",
}


def _detect_environment() -> str:
    if sys.platform == "darwin":
        return "mac"
    if sys.platform.startswith("linux"):
        return "linux"
    if sys.platform.startswith("win"):
        return "windows"
    raise RuntimeError(f"unsupported platform for computer tool demo: {platform.platform()}")


def _data_url_from_png(png_bytes: bytes) -> str:
    encoded = base64.b64encode(png_bytes).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _capture_screenshot() -> tuple[bytes, int, int]:
    try:
        image = ImageGrab.grab()
    except Exception as exc:  # pragma: no cover - depends on local desktop permissions
        raise RuntimeError(
            "failed to capture the desktop. On macOS, grant Screen Recording permission and run from an interactive GUI session."
        ) from exc
    width, height = image.size
    from io import BytesIO

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue(), width, height


def _save_artifact(artifacts_dir: Path | None, name: str, data: bytes | str) -> None:
    if artifacts_dir is None:
        return
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    target = artifacts_dir / name
    if isinstance(data, bytes):
        target.write_bytes(data)
    else:
        target.write_text(data, encoding="utf-8")


def _dump_action(action: Any) -> dict[str, Any]:
    if hasattr(action, "model_dump"):
        return action.model_dump()
    if isinstance(action, dict):
        return dict(action)
    out: dict[str, Any] = {}
    for key in dir(action):
        if key.startswith("_"):
            continue
        value = getattr(action, key)
        if callable(value):
            continue
        out[key] = value
    return out


def _dump_safety_check(check: Any) -> dict[str, Any]:
    if hasattr(check, "model_dump"):
        payload = check.model_dump()
    elif isinstance(check, dict):
        payload = dict(check)
    else:
        payload = {
            "id": getattr(check, "id", None),
            "code": getattr(check, "code", None),
            "message": getattr(check, "message", None),
        }
    return {
        "id": payload.get("id"),
        "code": payload.get("code"),
        "message": payload.get("message"),
    }


def _map_key(key: str) -> str:
    upper = key.upper()
    if upper in _KEY_ALIASES:
        return _KEY_ALIASES[upper]
    return key.lower()


class DesktopExecutor:
    def __init__(self, *, live: bool, pause_seconds: float) -> None:
        self.live = live
        self.pause_seconds = pause_seconds
        self._pyautogui = None
        if live:
            try:
                import pyautogui  # type: ignore
            except ImportError as exc:  # pragma: no cover - depends on local setup
                raise RuntimeError(
                    "PyAutoGUI is required for --live. Install examples/native_computer_use_demo/requirements.txt first."
                ) from exc
            pyautogui.FAILSAFE = True
            pyautogui.PAUSE = pause_seconds
            self._pyautogui = pyautogui

    def execute(self, action: Any) -> None:
        action_type = getattr(action, "type", None)
        if not action_type:
            raise RuntimeError(f"action has no type: {action!r}")
        if not self.live:
            return
        pyautogui = self._pyautogui
        assert pyautogui is not None
        if action_type == "click":
            button = _BUTTON_MAP.get(getattr(action, "button", "left"))
            if button is None:
                raise RuntimeError(f"unsupported click button: {getattr(action, 'button', None)}")
            pyautogui.click(x=int(action.x), y=int(action.y), button=button)
            return
        if action_type == "double_click":
            pyautogui.doubleClick(x=int(action.x), y=int(action.y))
            return
        if action_type == "move":
            pyautogui.moveTo(int(action.x), int(action.y))
            return
        if action_type == "drag":
            path = list(getattr(action, "path", []) or [])
            if not path:
                raise RuntimeError("drag action has empty path")
            start = path[0]
            pyautogui.moveTo(int(start.x), int(start.y))
            for point in path[1:]:
                pyautogui.dragTo(int(point.x), int(point.y), duration=max(self.pause_seconds, 0.05), button="left")
            return
        if action_type == "scroll":
            pyautogui.moveTo(int(action.x), int(action.y))
            scroll_y = int(getattr(action, "scroll_y", 0))
            scroll_x = int(getattr(action, "scroll_x", 0))
            if scroll_y:
                pyautogui.scroll(-scroll_y)
            if scroll_x and hasattr(pyautogui, "hscroll"):
                pyautogui.hscroll(scroll_x)
            return
        if action_type == "keypress":
            keys = [_map_key(key) for key in getattr(action, "keys", [])]
            if not keys:
                raise RuntimeError("keypress action has no keys")
            if len(keys) == 1:
                pyautogui.press(keys[0])
            else:
                pyautogui.hotkey(*keys)
            return
        if action_type == "type":
            pyautogui.write(str(action.text), interval=0.01)
            return
        if action_type == "wait":
            time.sleep(self.pause_seconds)
            return
        if action_type == "screenshot":
            return
        raise RuntimeError(f"unsupported action type: {action_type}")


def _extract_computer_calls(response: Any) -> list[Any]:
    return [item for item in getattr(response, "output", []) if getattr(item, "type", None) == "computer_call"]


def _extract_actions(call: Any) -> list[Any]:
    actions = list(getattr(call, "actions", None) or [])
    if actions:
        return actions
    action = getattr(call, "action", None)
    return [action] if action is not None else []


def _print_response_summary(response: Any) -> None:
    output_text = getattr(response, "output_text", "")
    if output_text:
        print(output_text.strip())


def _pending_safety_checks(call: Any) -> list[dict[str, Any]]:
    return [_dump_safety_check(check) for check in (getattr(call, "pending_safety_checks", []) or [])]


def _acknowledge_safety_checks(checks: Iterable[dict[str, Any]], *, auto_ack: bool) -> list[dict[str, Any]]:
    payloads = [dict(check) for check in checks]
    if not payloads:
        return []
    print(json.dumps({"pending_safety_checks": payloads}, ensure_ascii=False, indent=2))
    if auto_ack:
        return payloads
    if not sys.stdin.isatty():
        raise RuntimeError(
            "pending_safety_checks were returned by the API. Re-run interactively or pass --auto-ack-safety."
        )
    answer = input("Type 'ack' to continue, anything else to abort: ").strip().lower()
    if answer != "ack":
        raise RuntimeError("aborted because pending_safety_checks were not acknowledged")
    return payloads


def _build_client(api_key: str | None) -> OpenAI:
    final_api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not final_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Export it or pass --api-key.")
    return OpenAI(api_key=final_api_key)


def main() -> int:
    parser = argparse.ArgumentParser(description="Minimal OpenAI Native Computer-Use desktop demo for macOS/Linux/Windows.")
    parser.add_argument("--task", required=True, help="Natural-language task for the computer-use model.")
    parser.add_argument("--api-key", help="Optional API key override. Defaults to OPENAI_API_KEY.")
    parser.add_argument("--model", default="gpt-5.4", help="Responses API model. Default: gpt-5.4")
    parser.add_argument(
        "--tool-type",
        default="computer",
        help="Computer tool type. Default: computer. Use computer_use_preview only as a compatibility fallback.",
    )
    parser.add_argument("--environment", default=_detect_environment(), help="computer tool environment. Default: auto-detect.")
    parser.add_argument("--max-steps", type=int, default=20, help="Maximum computer action turns.")
    parser.add_argument("--pause-seconds", type=float, default=0.35, help="Delay after each live action.")
    parser.add_argument("--live", action="store_true", help="Actually execute clicks and keypresses. Default is dry-run.")
    parser.add_argument(
        "--auto-ack-safety",
        action="store_true",
        help="Acknowledge pending safety checks automatically. Default is off.",
    )
    parser.add_argument("--artifacts-dir", type=Path, help="Optional directory for screenshots and action logs.")
    args = parser.parse_args()

    client = _build_client(args.api_key)
    executor = DesktopExecutor(live=args.live, pause_seconds=max(args.pause_seconds, 0.05))
    screenshot_bytes, width, height = _capture_screenshot()
    _save_artifact(args.artifacts_dir, "step_000_screen.png", screenshot_bytes)

    tool: dict[str, Any] = {"type": args.tool_type}
    if args.tool_type != "computer":
        tool.update(
            {
                "display_width": width,
                "display_height": height,
                "environment": args.environment,
            }
        )
    response = client.responses.create(
        model=args.model,
        tools=[tool],
        input=args.task,
    )
    _print_response_summary(response)

    for step in range(1, args.max_steps + 1):
        computer_calls = _extract_computer_calls(response)
        if not computer_calls:
            return 0

        next_input: list[dict[str, Any]] = []
        for index, call in enumerate(computer_calls, start=1):
            actions = _extract_actions(call)
            if not actions:
                continue
            pending_checks = _pending_safety_checks(call)
            acknowledged_safety_checks = _acknowledge_safety_checks(
                pending_checks,
                auto_ack=args.auto_ack_safety,
            )
            for action_index, action in enumerate(actions, start=1):
                action_payload = _dump_action(action)
                print(
                    json.dumps(
                        {
                            "step": step,
                            "call_id": call.call_id,
                            "action_index": action_index,
                            "action": action_payload,
                        },
                        ensure_ascii=False,
                    )
                )
                _save_artifact(
                    args.artifacts_dir,
                    f"step_{step:03d}_call_{index:02d}_action_{action_index:02d}.json",
                    json.dumps(
                        {
                            "call_id": call.call_id,
                            "action": action_payload,
                            "acknowledged_safety_checks": acknowledged_safety_checks,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                )
                executor.execute(action)
            screenshot_bytes, _, _ = _capture_screenshot()
            _save_artifact(args.artifacts_dir, f"step_{step:03d}_screen_{index:02d}.png", screenshot_bytes)
            next_input.append(
                {
                    "type": "computer_call_output",
                    "call_id": call.call_id,
                    "output": {
                        "type": "computer_screenshot",
                        "image_url": _data_url_from_png(screenshot_bytes),
                        "detail": "original",
                    },
                    "acknowledged_safety_checks": acknowledged_safety_checks,
                }
            )

        response = client.responses.create(
            model=args.model,
            tools=[tool],
            previous_response_id=response.id,
            input=next_input,
        )
        _print_response_summary(response)

    print(f"max steps reached without terminal response ({args.max_steps})", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
