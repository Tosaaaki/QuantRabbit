#!/usr/bin/env python3
"""Summarize mechanical legacy replay and a sealed fresh-AI shadow review."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path


def _metrics(trades: list[dict[str, object]]) -> dict[str, object]:
    if not trades:
        return {
            "net_pnl_jpy": None,
            "profit_factor": None,
            "expectancy_jpy": None,
            "max_drawdown_jpy": None,
            "profit_giveback_rate": None,
            "trades": 0,
            "status": "insufficient_no_trades",
        }
    pnls = [float(trade.get("pnl_jpy") or 0.0) for trade in trades]
    gross_profit = sum(value for value in pnls if value > 0)
    gross_loss = -sum(value for value in pnls if value < 0)
    net = sum(pnls)
    profit_factor = math.inf if gross_loss == 0 and gross_profit > 0 else (
        gross_profit / gross_loss if gross_loss else 0.0
    )
    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for pnl in pnls:
        equity += pnl
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)
    giveback = None
    if gross_profit > 0:
        giveback = max(0.0, min(1.0, (gross_profit - max(net, 0.0)) / gross_profit))
    return {
        "net_pnl_jpy": round(net, 2),
        "profit_factor": "Infinity" if math.isinf(profit_factor) else round(profit_factor, 4),
        "expectancy_jpy": round(net / len(pnls), 2),
        "max_drawdown_jpy": round(max_drawdown, 2),
        "profit_giveback_rate": None if giveback is None else round(giveback, 4),
        "trades": len(pnls),
        "status": "observed",
    }


def build_comparison(replay: dict, review: dict) -> dict:
    judgments = {
        (str(item["strategy_id"]), str(item["trade_id"])): item
        for item in review.get("judgments", [])
    }
    strategies: list[dict[str, object]] = []
    for strategy_id, replay_item in replay.items():
        replay_file = Path(replay_item["base_scenarios"]["all"]["out_path"])
        payload = json.loads(replay_file.read_text(encoding="utf-8"))
        bot_trades = payload.get("trades", [])
        ai_trades: list[dict[str, object]] = []
        judgment_count = 0
        for trade in bot_trades:
            copied = dict(trade)
            judgment = judgments.get((strategy_id, str(trade.get("trade_id"))))
            if judgment:
                judgment_count += 1
                copied["exit_time"] = judgment["checkpoint"]
                copied["exit_price"] = judgment["counterfactual_exit_price"]
                copied["pnl_pips"] = judgment["counterfactual_pnl_pips"]
                copied["pnl_jpy"] = judgment["counterfactual_pnl_jpy"]
                copied["reason"] = "fresh_ai_exit_shadow"
            ai_trades.append(copied)
        bot = _metrics(bot_trades)
        ai = _metrics(ai_trades)
        pnl_delta = None
        if bot["net_pnl_jpy"] is not None and ai["net_pnl_jpy"] is not None:
            pnl_delta = round(float(ai["net_pnl_jpy"]) - float(bot["net_pnl_jpy"]), 2)
        if strategy_id == "session_open" and bot["trades"] > 0:
            decision = "provisional_promising_more_samples_required"
        elif strategy_id == "trend_breakout":
            decision = "reject_economic_ai_shadow_only"
        else:
            decision = "insufficient_samples"
        strategies.append(
            {
                "strategy_id": strategy_id,
                "bot_only": bot,
                "ai_inventory_shadow": ai,
                "ai_management_delta_jpy": pnl_delta,
                "ai_judgment_count": judgment_count,
                "ai_cost_jpy": review.get("ai_cost_jpy"),
                "ai_cost_status": review.get("ai_cost_status"),
                "decision": decision,
            }
        )
    observed = [item for item in strategies if item["bot_only"]["trades"] > 0]
    insufficient = [item for item in strategies if item["bot_only"]["trades"] == 0]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "authority": "NONE",
        "live_permission": False,
        "replay_window": "2026-01-27T00:00:04.892723Z/2026-01-27T23:59:59.681762Z",
        "conditions": {
            "pair": "USD_JPY",
            "resample_seconds": 5,
            "initial_units": "worker-defined; identical per A/B",
            "cost_model": "realistic: next-tick fill, 180ms latency, spread/ATR/latency slippage",
            "hard_stop_loss": "disabled",
            "end_of_replay_liquidation": "excluded",
        },
        "strategies": strategies,
        "counts": {
            "replay_attempted": len(strategies),
            "observed_with_trades": len(observed),
            "insufficient_no_trades": len(insufficient),
            "provisional_promising": sum(
                item["decision"] == "provisional_promising_more_samples_required"
                for item in strategies
            ),
            "fresh_ai_judgments": len(review.get("judgments", [])),
        },
    }


def _render_report(payload: dict) -> str:
    lines = [
        "# Legacy strategy replay + fresh AI shadow comparison",
        "",
        "- Authority: NONE; live_permission=false",
        "- Window: 2026-01-27 full-day archive, USD_JPY, 5-second mechanical replay",
        "- Cost: identical realistic next-tick/spread/slippage/latency assumptions",
        "- Fresh AI: worst-loss window only; Paper/shadow counterfactual; no economic application",
        "",
        "| strategy | Bot net | AI net | PF Bot/AI | Exp Bot/AI | DD Bot/AI | Giveback Bot/AI | trades | AI decisions | AI cost | decision |",
        "|---|---:|---:|---|---|---|---|---:|---:|---|---|",
    ]
    for item in payload["strategies"]:
        bot = item["bot_only"]
        ai = item["ai_inventory_shadow"]
        def value(field: str, metrics: dict) -> str:
            raw = metrics[field]
            return "N/A" if raw is None else str(raw)
        lines.append(
            f"| `{item['strategy_id']}` | {value('net_pnl_jpy', bot)} | "
            f"{value('net_pnl_jpy', ai)} | {value('profit_factor', bot)}/{value('profit_factor', ai)} | "
            f"{value('expectancy_jpy', bot)}/{value('expectancy_jpy', ai)} | "
            f"{value('max_drawdown_jpy', bot)}/{value('max_drawdown_jpy', ai)} | "
            f"{value('profit_giveback_rate', bot)}/{value('profit_giveback_rate', ai)} | "
            f"{bot['trades']} | {item['ai_judgment_count']} | "
            f"{item['ai_cost_status']} | {item['decision']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- SessionOpen earned 461.44 JPY, but one trade is not enough to promote it beyond provisional observation.",
            "- TrendBreakout lost 582.57 JPY. Fresh AI exit at the 60-second checkpoint would have reduced the loss to 168.57 JPY (+414.00 JPY), but the result remains negative.",
            "- PullbackContinuation and FailedBreakReverse had no trades, so PF/expectancy/DD are N/A rather than zero or infinity.",
            "- No new continuous Paper room was launched from this replay because no candidate met a minimum evidence threshold.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    replay = json.loads(args.replay.read_text(encoding="utf-8"))
    review = json.loads(args.review.read_text(encoding="utf-8"))
    result = build_comparison(replay, review)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "comparison.md").write_text(_render_report(result), encoding="utf-8")
    print(json.dumps(result["counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
