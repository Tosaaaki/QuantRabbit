#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from openai import OpenAI

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from computer_runtime import (  # noqa: E402
    ComputerRuntime,
    action_to_dict,
    detect_environment,
    extract_actions_from_call,
    pending_safety_checks,
)


def _build_client(api_key: str | None) -> OpenAI:
    final_api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not final_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Export it or pass --api-key.")
    return OpenAI(api_key=final_api_key)


def _response_payload(response: Any) -> dict[str, Any]:
    if hasattr(response, "model_dump"):
        return response.model_dump()
    return action_to_dict(response)


def _extract_computer_calls(response: Any) -> list[Any]:
    return [
        item
        for item in getattr(response, "output", [])
        if getattr(item, "type", None) == "computer_call"
    ]


def _print_output_text(response: Any) -> None:
    output_text = getattr(response, "output_text", "")
    if output_text:
        print(output_text.strip())


def _confirm_pending_safety_checks(
    checks: list[dict[str, Any]],
    *,
    auto_approve: bool,
) -> list[dict[str, Any]] | None:
    if not checks:
        return []
    print("Pending safety checks:")
    for check in checks:
        print(json.dumps(check, indent=2, ensure_ascii=False))
    if auto_approve:
        print("Auto-approving pending safety checks.")
        return checks
    if not sys.stdin.isatty():
        return None
    answer = (
        input("Acknowledge these safety checks and continue? [y/N]: ").strip().lower()
    )
    if answer in {"y", "yes"}:
        return checks
    return None


def _build_call_output(
    *,
    call_id: str,
    screenshot_output: dict[str, str],
    acknowledged_safety_checks: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": "computer_call_output",
        "call_id": call_id,
        "output": screenshot_output,
        "status": "completed",
    }
    if acknowledged_safety_checks:
        payload["acknowledged_safety_checks"] = acknowledged_safety_checks
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Minimal local Native Computer-Use demo for macOS using the OpenAI Responses API."
    )
    parser.add_argument(
        "--instruction",
        "--task",
        dest="instruction",
        required=True,
        help="Natural-language task for the computer-use model.",
    )
    parser.add_argument(
        "--api-key", help="Optional API key override. Defaults to OPENAI_API_KEY."
    )
    parser.add_argument(
        "--model", default="gpt-5.4", help="Responses API model. Default: gpt-5.4"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=20,
        help="Maximum number of computer-action turns.",
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=0.35,
        help="Delay used for wait and drag pacing.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Actually execute mouse and keyboard actions.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=None,
        help="Optional directory for screenshots, action logs, and response payloads.",
    )
    parser.add_argument(
        "--auto-approve-safety",
        action="store_true",
        help="Automatically acknowledge pending safety checks instead of prompting.",
    )
    args = parser.parse_args()

    if detect_environment() != "mac":
        print(
            "This demo is intended for macOS local use. It may still work elsewhere, but pyautogui permissions and key mapping are tuned for macOS."
        )

    runtime = ComputerRuntime(
        live=args.live,
        pause_seconds=args.pause_seconds,
        artifacts_dir=args.artifacts_dir,
    )
    client = _build_client(args.api_key)
    tools = [{"type": "computer"}]

    response = client.responses.create(
        model=args.model,
        tools=tools,
        input=args.instruction,
    )
    runtime.save_json("response_0000_initial.json", _response_payload(response))
    _print_output_text(response)

    computer_calls = _extract_computer_calls(response)
    step_index = 0
    while computer_calls:
        step_index += 1
        if step_index > args.max_steps:
            print(
                f"Stopping after {args.max_steps} steps with computer calls still pending."
            )
            return 2

        outputs: list[dict[str, Any]] = []
        for call_index, call in enumerate(computer_calls, start=1):
            checks = pending_safety_checks(call)
            acknowledged = _confirm_pending_safety_checks(
                checks,
                auto_approve=args.auto_approve_safety,
            )
            if checks and acknowledged is None:
                print(
                    "Safety checks were not acknowledged. Stopping without sending computer_call_output."
                )
                return 3

            actions = extract_actions_from_call(call)
            if not actions:
                print(
                    f"[step {step_index} call {call_index}] no actions returned; capturing a fresh screenshot."
                )

            for action_index, action in enumerate(actions, start=1):
                runtime.save_json(
                    f"step_{step_index:02d}/call_{call_index:02d}_action_{action_index:02d}.json",
                    action,
                )
                print(
                    f"[step {step_index} call {call_index} action {action_index}] {json.dumps(action, ensure_ascii=False)}"
                )

            runtime.execute_actions(actions)
            screenshot_output, width, height = runtime.capture_screenshot_output(
                step_index=step_index,
                call_index=call_index,
            )
            if step_index == 1 and call_index == 1:
                print(f"Captured desktop screenshot at {width}x{height}.")
            outputs.append(
                _build_call_output(
                    call_id=str(getattr(call, "call_id")),
                    screenshot_output=screenshot_output,
                    acknowledged_safety_checks=acknowledged,
                )
            )

        runtime.save_json(f"step_{step_index:02d}/request.json", outputs)
        response = client.responses.create(
            model=args.model,
            tools=tools,
            previous_response_id=response.id,
            input=outputs,
        )
        runtime.save_json(
            f"response_{step_index:04d}.json", _response_payload(response)
        )
        _print_output_text(response)
        computer_calls = _extract_computer_calls(response)

    print("Computer-use loop completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
