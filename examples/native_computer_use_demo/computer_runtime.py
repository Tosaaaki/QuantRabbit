#!/usr/bin/env python3
from __future__ import annotations

import base64
import json
import platform
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable

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
    "COMMAND": "command",
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


def detect_environment() -> str:
    if sys.platform == "darwin":
        return "mac"
    if sys.platform.startswith("linux"):
        return "linux"
    if sys.platform.startswith("win"):
        return "windows"
    raise RuntimeError(f"unsupported platform for computer tool demo: {platform.platform()}")


def action_to_dict(action: Any) -> Any:
    if action is None:
        return None
    if hasattr(action, "model_dump"):
        return action.model_dump()
    if isinstance(action, dict):
        return {key: action_to_dict(value) for key, value in action.items()}
    if isinstance(action, list):
        return [action_to_dict(item) for item in action]
    return action


def extract_actions_from_call(call: Any) -> list[dict[str, Any]]:
    actions = getattr(call, "actions", None)
    if actions:
        return [action_to_dict(action) for action in actions]
    action = getattr(call, "action", None)
    if action:
        return [action_to_dict(action)]
    return []


def pending_safety_checks(call: Any) -> list[dict[str, Any]]:
    checks = getattr(call, "pending_safety_checks", None) or []
    return [action_to_dict(check) for check in checks]


def _data_url_from_png(png_bytes: bytes) -> str:
    encoded = base64.b64encode(png_bytes).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _write_artifact(target: Path, data: bytes | str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, bytes):
        target.write_bytes(data)
        return
    target.write_text(data, encoding="utf-8")


def _map_key(key: str) -> str:
    upper = key.upper()
    return _KEY_ALIASES.get(upper, key.lower())


class ComputerRuntime:
    def __init__(
        self,
        *,
        live: bool,
        pause_seconds: float,
        artifacts_dir: Path | None,
    ) -> None:
        self.live = live
        self.pause_seconds = max(0.0, float(pause_seconds))
        self.artifacts_dir = artifacts_dir
        self._pyautogui = None
        if self.live:
            try:
                import pyautogui  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    "PyAutoGUI is required for live execution. Install examples/native_computer_use_demo/requirements.txt first."
                ) from exc
            pyautogui.FAILSAFE = True
            pyautogui.PAUSE = self.pause_seconds
            self._pyautogui = pyautogui

    def save_json(self, relative_path: str, payload: Any) -> None:
        if self.artifacts_dir is None:
            return
        target = self.artifacts_dir / relative_path
        _write_artifact(target, json.dumps(payload, indent=2, ensure_ascii=False))

    def capture_screenshot_output(self, *, step_index: int, call_index: int) -> tuple[dict[str, str], int, int]:
        try:
            image = ImageGrab.grab()
        except Exception as exc:
            raise RuntimeError(
                "failed to capture the desktop. On macOS, grant Screen Recording permission and run from an interactive GUI session."
            ) from exc
        width, height = image.size
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        png_bytes = buffer.getvalue()
        if self.artifacts_dir is not None:
            rel = f"step_{step_index:02d}/call_{call_index:02d}_screen.png"
            _write_artifact(self.artifacts_dir / rel, png_bytes)
        return {"type": "computer_screenshot", "image_url": _data_url_from_png(png_bytes)}, width, height

    def execute_actions(self, actions: Iterable[dict[str, Any]]) -> None:
        for action in actions:
            self.execute_action(action)

    def execute_action(self, action: dict[str, Any]) -> None:
        action_type = str(action.get("type") or "").strip()
        if not action_type:
            raise RuntimeError(f"action has no type: {action!r}")
        if not self.live:
            return
        pyautogui = self._pyautogui
        assert pyautogui is not None
        if action_type == "screenshot":
            return
        if action_type == "click":
            pyautogui.moveTo(int(action.get("x", 0)), int(action.get("y", 0)))
            raw_button = str(action.get("button", "left"))
            button = _BUTTON_MAP.get(raw_button)
            if button is None:
                raise RuntimeError(f"unsupported click button: {raw_button}")
            pyautogui.click(button=button)
            return
        if action_type == "double_click":
            pyautogui.moveTo(int(action.get("x", 0)), int(action.get("y", 0)))
            pyautogui.doubleClick()
            return
        if action_type == "move":
            pyautogui.moveTo(int(action.get("x", 0)), int(action.get("y", 0)))
            return
        if action_type == "drag":
            path = [action_to_dict(point) for point in action.get("path", []) or []]
            if not path:
                raise RuntimeError("drag action has an empty path")
            first = path[0]
            pyautogui.moveTo(int(first.get("x", 0)), int(first.get("y", 0)))
            pyautogui.mouseDown(button="left")
            try:
                for point in path[1:]:
                    pyautogui.moveTo(
                        int(point.get("x", 0)),
                        int(point.get("y", 0)),
                        duration=max(self.pause_seconds, 0.05),
                    )
            finally:
                pyautogui.mouseUp(button="left")
            return
        if action_type == "scroll":
            pyautogui.moveTo(int(action.get("x", 0)), int(action.get("y", 0)))
            scroll_y = int(action.get("scroll_y", 0) or 0)
            scroll_x = int(action.get("scroll_x", 0) or 0)
            if scroll_y:
                pyautogui.scroll(-scroll_y)
            if scroll_x and hasattr(pyautogui, "hscroll"):
                pyautogui.hscroll(scroll_x)
            return
        if action_type == "keypress":
            keys = [_map_key(str(key)) for key in action.get("keys", []) or []]
            if not keys:
                raise RuntimeError("keypress action has no keys")
            if len(keys) == 1:
                pyautogui.press(keys[0])
            else:
                pyautogui.hotkey(*keys)
            return
        if action_type == "type":
            pyautogui.write(str(action.get("text", "")), interval=0.01)
            return
        if action_type == "wait":
            time.sleep(max(self.pause_seconds, 0.1))
            return
        raise RuntimeError(f"unsupported action type: {action_type}")
