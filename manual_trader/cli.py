"""Command-line interface for the manual trading assistant."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict
from pathlib import Path
from textwrap import wrap
from typing import Any, Dict, List

from .context import ManualContext, gather_context
from .gpt_manual import get_manual_guidance

_LOG_PATH = Path("logs/manual_sessions.jsonl")


def _format_lines(title: str, body: List[str]) -> str:
    header = f"== {title} =="
    content = "\n".join(body)
    return f"{header}\n{content}" if body else header


def _describe_context(ctx: ManualContext | None) -> List[str]:
    if not ctx:
        return []
    macro = ctx.macro
    micro = ctx.micro
    lines = [
        f"Instrument: {ctx.instrument}",
        f"Timestamp: {ctx.timestamp}",
        f"Event window active: {ctx.event_window}",
        "",
        "Macro (H4):",
        f"  Regime: {macro.regime}",
        f"  Close: {macro.factors.get('close', 0.0):.3f}",
        f"  MA20: {macro.factors.get('ma20', 0.0):.3f}",
        f"  ADX: {macro.factors.get('adx', 0.0):.2f}",
        f"  ATR: {macro.factors.get('atr', 0.0):.4f}",
        f"  Range span: {float(macro.price_snapshot.get('range_span', 0.0) or 0.0):.5f}",
        "",
        "Micro (M1):",
        f"  Regime: {micro.regime}",
        f"  Close: {micro.factors.get('close', 0.0):.3f}",
        f"  MA20: {micro.factors.get('ma20', 0.0):.3f}",
        f"  RSI: {micro.factors.get('rsi', 0.0):.2f}",
        f"  ADX: {micro.factors.get('adx', 0.0):.2f}",
        f"  ATR: {micro.factors.get('atr', 0.0):.4f}",
    ]
    return lines


def _render_guidance(data: Dict[str, Any]) -> str:
    lines = [f"Bias: {data.get('bias', 'neutral')} ({data.get('confidence', 0)}%)"]
    mv = data.get("market_view", {})
    if mv:
        lines.append("-- Market View --")
        for key in ("macro", "micro", "news"):
            text = mv.get(key)
            if text:
                wrapped = wrap(str(text), width=88)
                lines.extend([f"  {key}: {wrapped[0]}"] + [f"           {w}" for w in wrapped[1:]])
    ideas: List[Dict[str, Any]] = data.get("trade_ideas") or []
    if ideas:
        lines.append("-- Trade Ideas --")
        for idx, idea in enumerate(ideas, start=1):
            lines.append(f"  [{idx}] {idea.get('label', 'Idea')} ({idea.get('direction')})")
            fields = (
                ("style", idea.get("style")),
                ("entry", idea.get("entry_note")),
                ("stop", idea.get("stop_loss")),
                ("take", idea.get("take_profit")),
                ("risk/reward", idea.get("risk_reward")),
                ("rationale", idea.get("rationale")),
            )
            for label, value in fields:
                if not value:
                    continue
                wrapped = wrap(str(value), width=86)
                lines.append(f"     {label}: {wrapped[0]}")
                lines.extend([f"           {w}" for w in wrapped[1:]])
            conds = idea.get("conditions") or []
            if conds:
                lines.append("     checklist:")
                for cond in conds:
                    for idx2, seg in enumerate(wrap(str(cond), width=82)):
                        prefix = "       - " if idx2 == 0 else "         "
                        lines.append(f"{prefix}{seg}")
    risk = data.get("risk_notes") or []
    if risk:
        lines.append("-- Risk Notes --")
        for note in risk:
            for idx, seg in enumerate(wrap(str(note), width=88)):
                prefix = "  - " if idx == 0 else "    "
                lines.append(f"{prefix}{seg}")
    steps = data.get("next_steps") or []
    if steps:
        lines.append("-- Next Steps --")
        for idx, step in enumerate(steps, start=1):
            wrapped = wrap(str(step), width=88)
            lines.append(f"  {idx}. {wrapped[0]}")
            lines.extend([f"     {seg}" for seg in wrapped[1:]])
    return "\n".join(lines)


def _collect_manual_action() -> Dict[str, Any] | None:
    reply = input("Record manual order intention? [y/N]: ").strip().lower()
    if reply not in {"y", "yes"}:
        return None
    direction = input("Direction (long/short/none): ").strip().lower() or "none"
    entry = input("Entry plan (price/zone): ").strip()
    stop = input("Stop-loss plan: ").strip()
    take = input("Take-profit plan: ").strip()
    size = input("Position sizing / notes: ").strip()
    extra = input("Additional comments: ").strip()
    return {
        "direction": direction,
        "entry": entry,
        "stop_loss": stop,
        "take_profit": take,
        "size_note": size,
        "comment": extra,
    }


def _append_log(record: Dict[str, Any]) -> None:
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _LOG_PATH.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(record, ensure_ascii=True) + "\n")


def _context_to_dict(ctx: ManualContext) -> Dict[str, Any]:
    data = asdict(ctx)
    # Limit recent OHLC noise in logs
    data["macro"]["recent_ohlc"] = ctx.macro.recent_ohlc[-3:]
    data["micro"]["recent_ohlc"] = ctx.micro.recent_ohlc[-5:]
    return data


async def _run(args: argparse.Namespace) -> int:
    ctx = await gather_context(
        instrument=args.instrument,
        m1_count=args.m1_count,
        h4_count=args.h4_count,
    )
    guidance = await get_manual_guidance(ctx)

    print(_format_lines("Context", _describe_context(ctx)))
    print()
    print(_format_lines("Guidance", _render_guidance(guidance).splitlines()))

    manual_action = None
    if not args.non_interactive:
        try:
            manual_action = _collect_manual_action()
        except KeyboardInterrupt:
            print("\nManual input aborted.")

    if not args.no_log:
        record = {
            "context": _context_to_dict(ctx),
            "guidance": guidance,
            "manual_action": manual_action,
        }
        _append_log(record)

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manual trading assistant")
    parser.add_argument("--instrument", default="USD_JPY")
    parser.add_argument("--m1-count", type=int, default=180)
    parser.add_argument("--h4-count", type=int, default=90)
    parser.add_argument("--non-interactive", action="store_true")
    parser.add_argument("--no-log", action="store_true", help="skip writing session log")
    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
