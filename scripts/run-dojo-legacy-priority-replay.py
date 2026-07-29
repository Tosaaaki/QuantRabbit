#!/usr/bin/env python3
"""Recover priority legacy entry cohorts and compare causal exit policies.

The legacy archive is imported read-only in isolated child processes.  Only
pure signal helpers and factor-cache candle calculations are called; broker,
order, account, and live position paths are never invoked.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import importlib.util
import json
import os
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from quant_rabbit.dojo_legacy_exit_matrix import (  # noqa: E402
    EntrySignal,
    InventoryPolicy,
    Quote,
    ReplayArmResult,
    ReplayCosts,
    replay_exit_matrix,
)


DEFAULT_ARCHIVE = Path(
    "/Users/tossaki/App/QuantRabbit_archives/"
    "QuantRabbit_legacy_20260430T151527Z"
)
PRIORITY_STRATEGIES = (
    "scalp_ping_5s",
    "scalp_ping_5s_b",
    "scalp_ping_5s_c",
    "scalp_ping_5s_d",
    "scalp_ping_5s_flow",
    "scalp_macd_rsi_div",
    "scalp_macd_rsi_div_b",
    "scalp_wick_reversal_blend",
    "scalp_wick_reversal_pro",
    "momentum_burst",
)
ENV_FILES = {
    "scalp_ping_5s": "ops/env/quant-scalp-ping-5s.env",
    "scalp_ping_5s_b": "ops/env/scalp_ping_5s_b.env",
    "scalp_ping_5s_c": "ops/env/scalp_ping_5s_c.env",
    "scalp_ping_5s_d": "ops/env/scalp_ping_5s_d.env",
    "scalp_ping_5s_flow": "ops/env/scalp_ping_5s_flow.env",
    "scalp_macd_rsi_div": "ops/env/quant-scalp-macd-rsi-div.env",
    "scalp_macd_rsi_div_b": "ops/env/quant-scalp-macd-rsi-div-b.env",
    "scalp_wick_reversal_blend": "ops/env/quant-scalp-wick-reversal-blend.env",
    "scalp_wick_reversal_pro": "ops/env/quant-scalp-wick-reversal-pro.env",
    "momentum_burst": "ops/env/quant-micro-momentumburst.env",
}
SOURCE_PATHS = {
    "scalp_ping_5s": "workers/scalp_ping_5s/worker.py",
    "scalp_ping_5s_b": "workers/scalp_ping_5s_b/worker.py",
    "scalp_ping_5s_c": "workers/scalp_ping_5s_c/worker.py",
    "scalp_ping_5s_d": "workers/scalp_ping_5s_d/worker.py",
    "scalp_ping_5s_flow": "workers/scalp_ping_5s_flow/worker.py",
    "scalp_macd_rsi_div": "workers/scalp_macd_rsi_div/worker.py",
    "scalp_macd_rsi_div_b": "workers/scalp_macd_rsi_div_b/worker.py",
    "scalp_wick_reversal_blend": "workers/scalp_wick_reversal_pro/worker.py",
    "scalp_wick_reversal_pro": "workers/scalp_wick_reversal_pro/worker.py",
    "momentum_burst": "strategies/micro/momentum_burst.py",
}


def _read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key:
            values[key] = value
    return values


def _configure_isolated_legacy(archive_root: Path, strategy: str) -> Path:
    legacy_root = archive_root / "archive"
    for path in (legacy_root, archive_root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    os.environ["DISABLE_GCP_SECRET_MANAGER"] = "1"
    os.environ["OANDA_ACCOUNT"] = "replay-dummy"
    os.environ["OANDA_TOKEN"] = "replay-dummy"
    os.environ["OANDA_PRACTICE"] = "true"
    os.environ["DOJO_LIVE_PERMISSION"] = "false"
    os.environ["QUANTRABBIT_AUTHORITY"] = "NONE"
    env_path = legacy_root / ENV_FILES[strategy]
    os.environ["QUANTRABBIT_ENV_FILE"] = str(env_path)
    os.environ.update(_read_env_file(env_path))
    if strategy.startswith("scalp_ping_5s_"):
        suffix = strategy.removeprefix("scalp_ping_5s_").upper()
        source_prefix = f"SCALP_PING_5S_{suffix}_"
        for key, value in tuple(os.environ.items()):
            if key.startswith(source_prefix):
                os.environ[f"SCALP_PING_5S_{key[len(source_prefix):]}"] = value
    if strategy == "scalp_macd_rsi_div_b":
        source_prefix = "SCALP_MACD_RSI_DIV_B_"
        for key, value in tuple(os.environ.items()):
            if key.startswith(source_prefix):
                os.environ[f"SCALP_PRECISION_{key[len(source_prefix):]}"] = value
    return legacy_root


def _load_replay_helpers(legacy_root: Path) -> Any:
    replay_path = legacy_root / "scripts" / "replay_workers.py"
    name = f"dojo_legacy_replay_helpers_{os.getpid()}"
    spec = importlib.util.spec_from_file_location(name, replay_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load replay helpers: {replay_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_ticks(replay_helpers: Any, tick_path: Path) -> tuple[list[Any], list[Quote]]:
    legacy_ticks = replay_helpers.load_ticks(tick_path)
    quotes = [
        Quote(timestamp=tick.dt, bid=float(tick.bid), ask=float(tick.ask))
        for tick in legacy_ticks
    ]
    return legacy_ticks, quotes


def _entry(
    *,
    strategy: str,
    tick: Any,
    ordinal: int,
    side: str,
    atr_pips: float,
    tp_pips: float,
) -> EntrySignal:
    return EntrySignal(
        signal_id=f"{strategy}:{tick.dt.date().isoformat()}:{ordinal}",
        timestamp=tick.dt,
        side="long" if side.lower() in {"long", "buy", "open_long"} else "short",
        atr_pips=max(float(atr_pips), 0.1),
        take_profit_pips=max(float(tp_pips), 0.1),
    )


def _build_factor_entries(
    *,
    strategy: str,
    ticks: Sequence[Any],
    replay_helpers: Any,
) -> list[EntrySignal]:
    from analysis.range_guard import detect_range_mode
    from indicators import factor_cache

    replay_helpers._reset_factor_cache_for_replay()
    builders = {
        "M1": replay_helpers._BarBuilder(60),
        "M5": replay_helpers._BarBuilder(300),
        "H1": replay_helpers._BarBuilder(3600),
        "H4": replay_helpers._BarBuilder(14_400),
    }
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    entries: list[EntrySignal] = []
    recent_ticks: deque[dict[str, float]] = deque()
    prev_rsi: float | None = None
    long_arm_until = 0.0
    short_arm_until = 0.0
    cooldown_until = 0.0

    signal_func: Callable[[dict[str, object], Any], dict[str, object] | None]
    if strategy == "momentum_burst":
        module = importlib.import_module("strategies.micro.momentum_burst")

        def signal_func(factors: dict[str, object], range_ctx: Any) -> dict[str, object] | None:
            del range_ctx
            return module.MomentumBurstMicro.check(factors)

    elif strategy.startswith("scalp_macd_rsi_div"):
        worker = importlib.import_module("workers.scalp_macd_rsi_div.worker")
        config = importlib.reload(
            importlib.import_module("workers.scalp_macd_rsi_div.config")
        )
        cooldown_seconds = float(getattr(config, "COOLDOWN_SEC", 120.0))

        def signal_func(factors: dict[str, object], range_ctx: Any) -> dict[str, object] | None:
            nonlocal prev_rsi, long_arm_until, short_arm_until, cooldown_until
            now_epoch = float(factors["_replay_epoch"])
            rsi = float(factors.get("rsi") or 50.0)
            if rsi <= float(config.RSI_LONG_ARM):
                long_arm_until = now_epoch + max(30.0, float(config.RSI_ARM_TTL_SEC))
            if rsi >= float(config.RSI_SHORT_ARM):
                short_arm_until = now_epoch + max(30.0, float(config.RSI_ARM_TTL_SEC))
            if now_epoch < cooldown_until:
                prev_rsi = rsi
                return None
            range_score = float(getattr(range_ctx, "score", 0.0) or 0.0)
            if bool(config.REQUIRE_RANGE_ACTIVE) and not bool(
                getattr(range_ctx, "active", False)
            ):
                prev_rsi = rsi
                return None
            if range_score < float(config.RANGE_MIN_SCORE):
                prev_rsi = rsi
                return None
            if float(factors.get("adx") or 0.0) > float(config.MAX_ADX):
                prev_rsi = rsi
                return None
            side, _ = worker._signal_side(
                prev_rsi=prev_rsi,
                rsi=rsi,
                long_armed=long_arm_until > now_epoch,
                short_armed=short_arm_until > now_epoch,
                div_kind=int(float(factors.get("div_macd_kind") or 0.0)),
                div_score=float(factors.get("div_macd_score") or 0.0),
                div_strength=float(factors.get("div_macd_strength") or 0.0),
                div_age_bars=float(factors.get("div_macd_age") or 99.0),
            )
            prev_rsi = rsi
            if side is None:
                return None
            atr_pips = max(float(factors.get("atr_pips") or 0.0), 0.8)
            tp_pips, sl_pips = worker._compute_targets(atr_pips)
            cooldown_until = now_epoch + cooldown_seconds
            return {
                "action": f"OPEN_{side.upper()}",
                "atr_pips": atr_pips,
                "tp_pips": tp_pips,
                "sl_pips": sl_pips,
            }

    else:
        worker = importlib.import_module("workers.scalp_wick_reversal_pro.worker")
        worker = importlib.reload(worker)
        worker.spread_ok = lambda **_: (True, {})
        worker.projection_decision = lambda *_args, **_kwargs: (
            True,
            1.0,
            {"status": "replay_causal_passthrough"},
        )

        def signal_func(factors: dict[str, object], range_ctx: Any) -> dict[str, object] | None:
            now_epoch = float(factors["_replay_epoch"])
            window = (
                float(getattr(worker, "WICK_BLEND_TICK_WINDOW_SEC", 10.0))
                if strategy.endswith("blend")
                else float(getattr(worker, "WICK_PRO_TICK_WINDOW_SEC", 10.0))
            )
            mids = [
                float(row["mid"])
                for row in recent_ticks
                if float(row["epoch"]) >= now_epoch - window
            ]
            worker.tick_snapshot = lambda *_args, **_kwargs: (mids, {})
            if strategy.endswith("blend"):
                return worker._signal_wick_reversal_blend(
                    factors, range_ctx, tag="WickReversalBlend"
                )
            return worker._signal_wick_reversal_pro(
                factors, range_ctx, tag="WickReversalPro"
            )

    try:
        for tick in ticks:
            recent_ticks.append(
                {"epoch": float(tick.epoch), "mid": float(tick.mid)}
            )
            while recent_ticks and recent_ticks[0]["epoch"] < tick.epoch - 30.0:
                recent_ticks.popleft()
            closed_m1 = False
            for timeframe, builder in builders.items():
                closed = builder.update(tick)
                if not closed:
                    continue
                candle = {
                    "open": closed["open"],
                    "high": closed["high"],
                    "low": closed["low"],
                    "close": closed["close"],
                    "time": datetime.fromtimestamp(
                        float(closed["timestamp"]), tz=timezone.utc
                    ),
                }
                loop.run_until_complete(factor_cache.on_candle(timeframe, candle))
                if timeframe == "M1":
                    closed_m1 = True
            if not closed_m1:
                continue
            factors_by_tf = factor_cache.all_factors()
            factors = dict(factors_by_tf.get("M1") or {})
            if not factors:
                continue
            factors["candles"] = factor_cache.get_candles_snapshot("M1", limit=80)
            factors["spread_pips"] = max(
                0.0, (float(tick.ask) - float(tick.bid)) / 0.01
            )
            factors["_replay_epoch"] = float(tick.epoch)
            range_ctx = detect_range_mode(
                factors, dict(factors_by_tf.get("H4") or {})
            )
            signal = signal_func(factors, range_ctx)
            if not signal:
                continue
            action = str(signal.get("action") or "").lower()
            if action not in {"open_long", "open_short", "buy", "sell", "long", "short"}:
                continue
            entries.append(
                _entry(
                    strategy=strategy,
                    tick=tick,
                    ordinal=len(entries),
                    side=action,
                    atr_pips=float(
                        signal.get("atr_pips")
                        or factors.get("atr_pips")
                        or signal.get("sl_pips")
                        or 1.0
                    ),
                    tp_pips=float(signal.get("tp_pips") or 1.5),
                )
            )
    finally:
        loop.close()
    return entries


def _build_ping_entries(
    *,
    strategy: str,
    ticks: Sequence[Any],
) -> list[EntrySignal]:
    config = importlib.reload(importlib.import_module("workers.scalp_ping_5s.config"))
    worker = importlib.reload(importlib.import_module("workers.scalp_ping_5s.worker"))
    recent: deque[dict[str, float]] = deque()
    entries: list[EntrySignal] = []
    cooldown_until = 0.0
    window_seconds = max(float(config.WINDOW_SEC), 30.0)
    side_filter = str(getattr(config, "SIDE_FILTER", "") or "").lower()
    tp_pips = float(getattr(config, "TP_BASE_PIPS", 1.0) or 1.0)
    cooldown_seconds = float(getattr(config, "ENTRY_COOLDOWN_SEC", 2.0) or 2.0)
    loop_interval_seconds = max(
        float(getattr(config, "LOOP_INTERVAL_SEC", 0.2) or 0.2), 0.05
    )
    next_evaluation_epoch = float(ticks[0].epoch) if ticks else 0.0
    original_time = worker.time.time
    try:
        for tick in ticks:
            recent.append(
                {
                    "epoch": float(tick.epoch),
                    "bid": float(tick.bid),
                    "ask": float(tick.ask),
                    "mid": float(tick.mid),
                }
            )
            while recent and recent[0]["epoch"] < tick.epoch - window_seconds:
                recent.popleft()
            if tick.epoch < next_evaluation_epoch:
                continue
            next_evaluation_epoch = tick.epoch + loop_interval_seconds
            if tick.epoch < cooldown_until:
                continue
            worker.time.time = lambda epoch=float(tick.epoch): epoch
            spread_pips = max(0.0, (float(tick.ask) - float(tick.bid)) / 0.01)
            signal, _reason = worker._build_tick_signal(
                list(recent), spread_pips
            )
            if signal is None:
                continue
            side = str(signal.side).lower()
            if side_filter in {"buy", "long", "open_long"} and side != "long":
                continue
            if side_filter in {"sell", "short", "open_short"} and side != "short":
                continue
            atr_proxy = max(float(signal.range_pips), float(signal.instant_range_pips), 0.5)
            entries.append(
                _entry(
                    strategy=strategy,
                    tick=tick,
                    ordinal=len(entries),
                    side=side,
                    atr_pips=atr_proxy,
                    tp_pips=tp_pips,
                )
            )
            cooldown_until = tick.epoch + cooldown_seconds
    finally:
        worker.time.time = original_time
    return entries


def _finite(value: float | None) -> float | None:
    if value is None or value == float("inf") or value == float("-inf"):
        return None
    return value


def _serialize_arm(result: ReplayArmResult) -> dict[str, object]:
    metrics = result.metrics.to_dict()
    metrics["profit_factor"] = _finite(result.metrics.profit_factor)
    gains = 0.0
    losses = 0.0
    equity = 0.0
    peak = 0.0
    minimum = 0.0
    max_drawdown = 0.0
    serialized_sample: list[dict[str, object]] = []
    sample_indexes = set(range(min(10, len(result.trades))))
    sample_indexes.update(
        range(max(0, len(result.trades) - 10), len(result.trades))
    )
    for index, trade in enumerate(result.trades):
        net = float(trade.net_jpy)
        gains += max(net, 0.0)
        losses += max(-net, 0.0)
        equity += net
        peak = max(peak, equity)
        minimum = min(minimum, equity)
        max_drawdown = max(max_drawdown, peak - equity)
        if index in sample_indexes:
            serialized_sample.append(
                {
                    **asdict(trade),
                    "entry_timestamp": trade.entry_timestamp.isoformat(),
                    "exit_timestamp": trade.exit_timestamp.isoformat(),
                }
            )
    return {
        "metrics": metrics,
        "aggregation": {
            "net_jpy": round(equity, 8),
            "gains_jpy": round(gains, 8),
            "losses_jpy": round(losses, 8),
            "max_prefix_equity_jpy": round(peak, 8),
            "min_prefix_equity_jpy": round(minimum, 8),
            "max_drawdown_jpy": round(max_drawdown, 8),
        },
        "trade_sample": serialized_sample,
        "trade_sample_bounded": True,
    }


def _run_strategy(
    *,
    archive_root_text: str,
    strategy: str,
    tick_path_texts: Sequence[str],
    costs_payload: dict[str, object],
) -> dict[str, object]:
    started = time.monotonic()
    archive_root = Path(archive_root_text)
    legacy_root = _configure_isolated_legacy(archive_root, strategy)
    replay_helpers = _load_replay_helpers(legacy_root)
    source_path = legacy_root / SOURCE_PATHS[strategy]
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    costs = ReplayCosts(**costs_payload)
    windows: list[dict[str, object]] = []

    for tick_path_text in tick_path_texts:
        tick_path = Path(tick_path_text)
        ticks, quotes = _load_ticks(replay_helpers, tick_path)
        if strategy.startswith("scalp_ping_5s"):
            entries = _build_ping_entries(strategy=strategy, ticks=ticks)
        else:
            entries = _build_factor_entries(
                strategy=strategy, ticks=ticks, replay_helpers=replay_helpers
            )
        arms = replay_exit_matrix(
            quotes=quotes,
            entries=entries,
            costs=costs,
            inventory_policies=(
                InventoryPolicy(False),
                InventoryPolicy(True, checkpoint_seconds=60),
            ),
        )
        windows.append(
            {
                "tick_file": str(tick_path),
                "tick_sha256": hashlib.sha256(tick_path.read_bytes()).hexdigest(),
                "tick_count": len(ticks),
                "start": ticks[0].dt.isoformat() if ticks else None,
                "end": ticks[-1].dt.isoformat() if ticks else None,
                "entry_count": len(entries),
                "entry_sample": [
                    {
                        **asdict(entry),
                        "timestamp": entry.timestamp.isoformat(),
                    }
                    for entry in (
                        entries[:10]
                        + (entries[-10:] if len(entries) > 10 else [])
                    )
                ],
                "entry_sample_bounded": True,
                "arms": [_serialize_arm(arm) for arm in arms],
            }
        )
    return {
        "strategy": strategy,
        "source_path": str(source_path),
        "source_sha256": source_sha256,
        "env_file": str(legacy_root / ENV_FILES[strategy]),
        "adapter_status": "recovered_priority",
        "windows": windows,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def _aggregate_strategy(result: dict[str, object]) -> dict[str, object]:
    windows = list(result["windows"])
    arm_keys = [
        (policy, inventory)
        for policy in (
            "no_sl",
            "fixed_sl",
            "atr_sl",
            "volatility_trail",
            "time_stop",
        )
        for inventory in (False, True)
    ]
    aggregates: list[dict[str, object]] = []
    for policy, inventory in arm_keys:
        selected: list[dict[str, object]] = []
        for window in windows:
            selected.extend(
                arm
                for arm in window["arms"]
                if arm["metrics"]["policy"] == policy
                and arm["metrics"]["ai_inventory"] is inventory
            )
        net = round(
            sum(float(arm["aggregation"]["net_jpy"]) for arm in selected), 8
        )
        gains = sum(float(arm["aggregation"]["gains_jpy"]) for arm in selected)
        losses = sum(float(arm["aggregation"]["losses_jpy"]) for arm in selected)
        cumulative = 0.0
        global_peak = 0.0
        drawdown = 0.0
        for arm in selected:
            aggregation = arm["aggregation"]
            local_min = cumulative + float(
                aggregation["min_prefix_equity_jpy"]
            )
            drawdown = max(
                drawdown,
                float(aggregation["max_drawdown_jpy"]),
                global_peak - local_min,
            )
            global_peak = max(
                global_peak,
                cumulative + float(aggregation["max_prefix_equity_jpy"]),
            )
            cumulative += float(aggregation["net_jpy"])
        decisions = sum(
            int(arm["metrics"]["ai_decisions"])
            for arm in selected
        )
        ai_cost = round(
            sum(float(arm["metrics"]["ai_cost_jpy"]) for arm in selected), 8
        )
        trade_count = sum(int(arm["metrics"]["trades"]) for arm in selected)
        aggregates.append(
            {
                "policy": policy,
                "ai_inventory": inventory,
                "entry_cohort_size": sum(int(window["entry_count"]) for window in windows),
                "trades": trade_count,
                "net_jpy": net,
                "profit_factor": round(gains / losses, 6) if losses else None,
                "expectancy_jpy": round(net / trade_count, 8) if trade_count else None,
                "max_drawdown_jpy": round(drawdown, 8),
                "ai_decisions": decisions,
                "ai_cost_jpy": ai_cost,
                "profitable": net > 0.0,
            }
        )
    result["aggregate_arms"] = aggregates
    result["entry_cohort_size"] = sum(int(window["entry_count"]) for window in windows)
    best = max(
        aggregates,
        key=lambda item: (
            float(item["net_jpy"]),
            -float(item["max_drawdown_jpy"]),
        ),
    )
    result["best_arm"] = best
    protected_arms = [
        arm for arm in aggregates if str(arm["policy"]) != "no_sl"
    ]
    best_protected = max(
        protected_arms,
        key=lambda item: (
            float(item["net_jpy"]),
            -float(item["max_drawdown_jpy"]),
        ),
    )
    result["best_protected_arm"] = best_protected
    result["decision"] = (
        "証拠不足"
        if int(result["entry_cohort_size"]) < 30
        else (
            "採用候補"
            if (
                bool(best_protected["profitable"])
                and float(best_protected["profit_factor"] or 0.0) > 1.0
                and float(best_protected["expectancy_jpy"] or 0.0) > 0.0
            )
            else "不採用"
        )
    )
    return result


def _write_report(payload: dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "priority_replay.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# 旧・裁量戦略ワーカー高速リプレイ",
        "",
        f"- 結論: **{payload['top_line']}**",
        f"- 判定: **{payload['overall_decision']}**",
        f"- 評価済み/試行済み: **{payload['progress']['evaluated_or_attempted']} / 82**",
        f"- 残り: **{payload['progress']['remaining']}**",
        f"- authority: `{payload['safety']['authority']}` / live: `{str(payload['safety']['live']).lower()}`",
        "",
        "## 優先family結果",
        "",
        "| family | entry | best protected exit | AI | Net JPY | PF | Expectancy | DD | trades | 判定 |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for result in payload["priority_results"]:
        best = result["best_protected_arm"]
        pf = "n/a" if best["profit_factor"] is None else f"{best['profit_factor']:.3f}"
        exp = "n/a" if best["expectancy_jpy"] is None else f"{best['expectancy_jpy']:.2f}"
        lines.append(
            f"| {result['strategy']} | {result['entry_cohort_size']} | {best['policy']} | "
            f"{'ON' if best['ai_inventory'] else 'OFF'} | {best['net_jpy']:.2f} | "
            f"{pf} | {exp} | {best['max_drawdown_jpy']:.2f} | "
            f"{best['trades']} | {result['decision']} |"
        )
    lines.extend(
        [
            "",
            "## 判定上の制約",
            "",
            "- 2024–2026H1 corpusは既に反復利用済みのため、今回の5窓は `LINEAGE_UNSEEN_DIAGNOSTIC` であり、未使用holdoutとは表現しない。",
            "- AI ONは外部modelを呼ばない凍結済み因果inventory rule。model call 0、execution AI cost 0円。モデル推論費用を含む実AIの経済性は別途未確定。",
            "- spreadは記録bid/ask、slippageは全fill、financingは保有時間按分、期末open positionは実行可能sideでMTM。",
            "- financingは比較用の保守的固定debit 10円/1万通貨/日。実brokerのside別swap実績ではない。",
            "- wick blendのprojectionはarchive内の外部予測依存を除外し、past-only gateを通過したsignalに対するneutral passthrough。完全忠実性ではなくadapter診断。",
            "",
            "## 安全性",
            "",
            "- Paperは停止していない。broker/order mutationは実行していない。",
            "- archive codeはread-only import、variantはprocess隔離。",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=PRIORITY_STRATEGIES,
        default=list(PRIORITY_STRATEGIES),
    )
    parser.add_argument("--units", type=int, default=1_000)
    parser.add_argument("--slippage-pips", type=float, default=0.05)
    parser.add_argument(
        "--financing-jpy-per-10k-units-per-day", type=float, default=10.0
    )
    args = parser.parse_args()

    tick_dir = (
        args.archive_root
        / "archive"
        / "local"
        / "market_data"
        / "usdjpy_20260123_20260128"
    )
    tick_paths = sorted(tick_dir.glob("USD_JPY_ticks_*.jsonl"))
    if not tick_paths:
        raise SystemExit(f"no tick files: {tick_dir}")
    costs_payload = {
        "units": args.units,
        "slippage_pips_per_fill": args.slippage_pips,
        "financing_jpy_per_10k_units_per_day": args.financing_jpy_per_10k_units_per_day,
        "ai_cost_jpy_per_decision": 0.0,
    }
    started = time.monotonic()
    results: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(
                _run_strategy,
                archive_root_text=str(args.archive_root),
                strategy=strategy,
                tick_path_texts=[str(path) for path in tick_paths],
                costs_payload=costs_payload,
            ): strategy
            for strategy in args.strategies
        }
        for future in as_completed(futures):
            strategy = futures[future]
            try:
                results.append(_aggregate_strategy(future.result()))
            except Exception as exc:
                errors.append(
                    {"strategy": strategy, "error": f"{type(exc).__name__}: {exc}"}
                )
    results.sort(key=lambda item: str(item["strategy"]))
    positive = [result for result in results if result["decision"] == "採用候補"]
    no_sl_positive = [
        result
        for result in results
        if bool(result["best_arm"]["profitable"])
        and str(result["best_arm"]["policy"]) == "no_sl"
    ]
    newly_attempted = len(results) + len(errors)
    payload: dict[str, object] = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "top_line": (
            "保護付きで儲かっている"
            if positive
            else (
                "保護付きでは儲かっていない（no-SL対照のみ黒字）"
                if no_sl_positive
                else "儲かっていない"
            )
        ),
        "overall_decision": "採用候補あり" if positive else "採用なし",
        "priority_results": results,
        "errors": errors,
        "progress": {
            "normalized_inventory_families": 82,
            "baseline_evaluated_or_attempted": 27,
            "newly_attempted": newly_attempted,
            "evaluated_or_attempted": min(82, 27 + newly_attempted),
            "remaining": max(0, 55 - newly_attempted),
            "remaining_recoverable": max(0, 23 - newly_attempted),
            "remaining_evidence_only": 32,
        },
        "costs": costs_payload,
        "corpus_class": "LINEAGE_UNSEEN_DIAGNOSTIC",
        "parallelism": {
            "configured_workers": max(1, args.workers),
            "elapsed_seconds": round(time.monotonic() - started, 3),
        },
        "safety": {
            "live": False,
            "authority": "NONE",
            "broker_order_mutation": False,
            "paper_stopped": False,
        },
    }
    _write_report(payload, args.output_dir)
    print(json.dumps(
        {
            "output_dir": str(args.output_dir),
            "results": len(results),
            "errors": errors,
            "elapsed_seconds": payload["parallelism"]["elapsed_seconds"],
        },
        ensure_ascii=False,
    ))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
