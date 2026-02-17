#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quick forecast snapshot from live factor cache.

用途:
- VM 上で実行し、forecast_gate が見る最新の予測を即時確認する
- 各 horizon の `p_up` / `expected_pips` / trend/range 指標を一覧化
- `anchor_price` / `target_price`（現値基準 expected_pips 到達想定価格）を表示
- 天井・底・レジーム（ボラ/トレンド/レンジ）を同梱
- 5m/10m など要求 horizon が足りなくても、欠損理由と回収アクションを表示

実行例:
  python3 scripts/vm_forecast_snapshot.py \
    --env-file /home/tossaki/QuantRabbit/ops/env/quant-v2-runtime.env
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        values[key.strip()] = val.strip().strip('"').strip("'")
    return values


def _classify_state(row: dict[str, Any]) -> str:
    p_up = float(row.get("p_up", 0.5) or 0.5)
    trend = float(row.get("trend_strength", 0.5) or 0.5)
    rng = float(row.get("range_pressure", 0.5) or 0.5)

    edge = p_up - 0.5
    if p_up >= 0.60 and trend >= rng:
        return "上昇トレンド継続（まだ天井ではない）"
    if p_up <= 0.40 and trend >= rng:
        return "下落トレンド継続（まだ底ではない）"
    if p_up >= 0.58 and rng > trend:
        return "上値圧力強め（天井警戒）"
    if p_up <= 0.42 and rng > trend:
        return "下値圧力強め（底警戒）"
    if abs(edge) < 0.04 and rng > 0.60:
        return "レンジ寄り（天井/底の間）"
    if edge > 0.04:
        return "上振れ寄り"
    if edge < -0.04:
        return "下振れ寄り"
    return "中立"


def _parse_horizon_spec(horizon: str) -> dict[str, Any] | None:
    h = str(horizon or "").strip().lower()
    m = re.fullmatch(r"^(\d+)\s*([mhdw])$", h)
    if not m:
        return None
    value = int(m.group(1))
    unit = m.group(2)
    if value <= 0:
        return None
    if unit == "m":
        timeframe = "M1" if value <= 1 else "M5"
        if timeframe == "M1":
            step_bars = value
        elif value >= 10:
            step_bars = 12
        else:
            step_bars = 6
    elif unit == "h":
        timeframe = "M5"
        step_bars = max(12, value * 12)
    elif unit == "d":
        timeframe = "H1"
        step_bars = max(24, value * 24)
    else:
        timeframe = "D1"
        step_bars = max(5, value * 5)
    return {"timeframe": timeframe, "step_bars": int(step_bars)}


def _horizon_sort_key(horizon: str) -> tuple[int, int, str]:
    h = (str(horizon or "").strip().lower())
    m = re.fullmatch(r"^(\d+)\s*([mhdw])$", h)
    if not m:
        return (999, 0, h)
    value = int(m.group(1))
    unit = m.group(2)
    order = {"m": 0, "h": 1, "d": 2, "w": 3}.get(unit, 4)
    return (order, value, h)


def _latest_close(candles: list[dict[str, Any]]) -> float | None:
    for candle in reversed(candles):
        if not isinstance(candle, dict):
            continue
        for key in ("close", "mid", "price", "last", "close_price"):
            value = candle.get(key)
            try:
                price = float(value)
                if price == price:
                    return price
            except Exception:
                continue
    return None


def _format_line(horizon: str, row: dict[str, Any]) -> str:
    p_up = float(row.get("p_up", 0.5) or 0.5)
    expected = row.get("expected_pips")
    source = row.get("source") or "n/a"
    trend = row.get("trend_strength")
    rng = row.get("range_pressure")
    vol_state = row.get("volatility_state")
    trend_state = row.get("trend_state")
    range_state = row.get("range_state")
    leading = row.get("leading_indicator")
    vol_rank = row.get("volatility_rank")
    regime_score = row.get("regime_score")
    lead_strength = row.get("leading_indicator_strength")
    edge = (p_up - 0.5) * 100
    tf = row.get("timeframe") or "-"
    step = int(row.get("step_bars") or 0)
    ts = row.get("feature_ts") or "-"
    status = str(row.get("status") or ("ready" if row.get("forecast_ready") else ""))
    ready = bool(row.get("forecast_ready", row.get("p_up") is not None and row.get("status") != "insufficient_history"))
    note = ""
    if not ready:
        reason = str(row.get("reason") or "")
        detail = str(row.get("detail") or "")
        avail = row.get("available_candles")
        need = row.get("required_candles")
        remediation = str(row.get("remediation") or "")
        note = " [PENDING]"
        if reason:
            note += f" reason={reason}"
        if avail is not None and need is not None:
            note += f" have={avail} need={need}"
        if detail:
            note += f" detail={detail}"
        if remediation:
            note += f" action={remediation}"
    state = _classify_state(row)
    anchor = row.get("anchor_price")
    target = row.get("target_price")
    target_text = ""
    if anchor is not None and target is not None:
        target_text = f" anchor_price={anchor} target_price={target}"

    return (
        f"[{horizon:>3}] {state}\n"
        f"  p_up={p_up:.4f} edge={edge:+.2f}% expected_pips={expected} "
        f"source={source} tf={tf} step_bars={step} status={status}{target_text}\n"
        f"  trend_strength={trend} range_pressure={rng} as_of={ts}{note}\n"
        f"  regime: vol_state={vol_state or '-'} trend_state={trend_state or '-'} "
        f"range_state={range_state or '-'} leading={leading or '-'} "
        f"vol_rank={vol_rank if vol_rank is not None else '-'} "
        f"regime_score={regime_score if regime_score is not None else '-'} "
        f"leading_strength={lead_strength if lead_strength is not None else '-'}"
    )


def _build_pending_row(
    forecast_gate: Any,
    horizon: str,
) -> dict[str, Any] | None:
    horizon = str(horizon).strip().lower()
    if not horizon:
        return None
    meta = {}
    try:
        meta = dict(getattr(forecast_gate, "_HORIZON_META", {}))
    except Exception:
        meta = {}
    spec = meta.get(horizon)
    if not isinstance(spec, dict):
        spec = _parse_horizon_spec(horizon)
    if not isinstance(spec, dict):
        return None
    timeframe = str(spec.get("timeframe") or "").strip().upper()
    step_bars = int(spec.get("step_bars") or 0)
    if not timeframe or step_bars <= 0:
        return None
    rows_by_tf: dict[str, list[dict[str, Any]]] = {}
    try:
        rows_by_tf = forecast_gate._fetch_candles_by_tf()  # noqa: SLF001
    except Exception:
        rows_by_tf = {}
    tf_rows = rows_by_tf.get(timeframe, [])
    count = len(tf_rows)
    anchor = _latest_close(tf_rows)
    required = max(24, step_bars * 3, 50)
    row = {
        "horizon": horizon,
        "source": "technical",
        "status": "insufficient_history",
        "forecast_ready": False,
        "p_up": None,
        "expected_pips": None,
        "feature_ts": None,
        "trend_strength": 0.5,
        "range_pressure": 0.5,
        "projection_score": 0.0,
        "projection_confidence": 0.0,
        "projection_components": {},
        "timeframe": timeframe,
        "step_bars": step_bars,
        "available_candles": count,
        "required_candles": required,
        "reason": "not_generated",
        "detail": "forecast_gate did not return this horizon",
        "remediation": f"need >= {required} candles on {timeframe}; run backfill/replay for {timeframe}",
    }
    if anchor is not None and anchor == anchor:
        row["anchor_price"] = round(float(anchor), 5)
        row["target_price"] = None
    try:
        row.update(
            forecast_gate._regime_profile_from_row(  # noqa: SLF001
                row,
                p_up=float(row.get("p_up") or 0.5),
                edge=0.5,
                trend_strength=float(row.get("trend_strength") or 0.5),
                range_pressure=float(row.get("range_pressure") or 0.5),
            )
        )
    except Exception:
        row.update(
            {
                "volatility_state": None,
                "trend_state": None,
                "range_state": None,
                "volatility_rank": None,
                "regime_score": None,
                "leading_indicator": None,
                "leading_indicator_strength": None,
            }
        )
    return row


def _build_technical_fallback_row(
    forecast_gate: Any,
    horizon: str,
) -> dict[str, Any] | None:
    horizon = str(horizon).strip().lower()
    if not horizon:
        return None
    meta = {}
    try:
        meta = dict(getattr(forecast_gate, "_HORIZON_META", {}))
    except Exception:
        meta = {}
    spec = meta.get(horizon)
    if not isinstance(spec, dict):
        spec = _parse_horizon_spec(horizon)
    if not isinstance(spec, dict):
        return None
    timeframe = str(spec.get("timeframe") or "").strip().upper()
    step_bars = int(spec.get("step_bars") or 0)
    if not timeframe or step_bars <= 0:
        return None
    try:
        rows_by_tf = forecast_gate._fetch_candles_by_tf()  # noqa: SLF001
    except Exception:
        rows_by_tf = {}
    candles = rows_by_tf.get(timeframe, [])
    try:
        row = forecast_gate._technical_prediction_for_horizon(
            candles,
            horizon=horizon,
            step_bars=step_bars,
            timeframe=timeframe,
        )
    except Exception as exc:  # noqa: BLE001
        row = None
        fallback = str(exc)
    else:
        fallback = None
    if isinstance(row, dict):
        return row
    fallback_row = _build_pending_row(
        forecast_gate,
        horizon,
    )
    if not isinstance(fallback_row, dict):
        return None
    fallback_row = dict(fallback_row)
    fallback_row["detail"] = fallback or "forecast_gate did not return this horizon"
    return fallback_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show latest forecast snapshot from forecast_gate")
    parser.add_argument(
        "--env-file",
        default=str(_repo_root() / "ops" / "env" / "quant-v2-runtime.env"),
        help="path to runtime env file",
    )
    parser.add_argument(
        "--json", action="store_true", help="output raw forecast rows as JSON instead of text"
    )
    parser.add_argument("--horizon", action="append", help="horizon to show (default: all)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    env_file = Path(args.env_file)
    env_values = _load_env_file(env_file)
    for key, value in env_values.items():
        os.environ.setdefault(key, value)

    repo_root = _repo_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from workers.common import forecast_gate

    bundle = forecast_gate._load_bundle_cached()  # noqa: SLF001
    preds = forecast_gate._ensure_predictions(bundle)  # noqa: SLF001

    horizon_filter = {str(h).strip().lower() for h in args.horizon} if args.horizon else None
    selected: list[tuple[str, dict[str, Any]]] = []

    if isinstance(preds, dict):
        for horizon, row in sorted(preds.items(), key=lambda item: _horizon_sort_key(item[0])):
            if not isinstance(row, dict):
                continue
            if horizon_filter and str(horizon).strip().lower() not in horizon_filter:
                continue
            selected.append((str(horizon).strip().lower(), dict(row)))
    elif not horizon_filter:
        print("NO_PREDICTIONS")
        return 1

    if horizon_filter:
        selected_map = {str(h): r for h, r in selected}
        present = {str(h): True for h, _ in selected}
        for horizon in sorted(horizon_filter, key=_horizon_sort_key):
            if horizon in present:
                continue
            fallback = _build_pending_row(forecast_gate, horizon)
            if not isinstance(fallback, dict):
                fallback = _build_technical_fallback_row(forecast_gate, horizon)
            if not isinstance(fallback, dict):
                continue
            fallback = dict(fallback)
            fallback.setdefault("status", "insufficient_history")
            fallback.setdefault("forecast_ready", False)
            fallback.setdefault("reason", "not_generated")
            fallback["horizon"] = str(horizon)
            selected_map[str(horizon)] = fallback
        selected = sorted(selected_map.items(), key=lambda item: _horizon_sort_key(item[0]))

    if not selected:
        print("NO_MATCHING_HORIZONS")
        return 1

    if args.json:
        print(json.dumps(selected, ensure_ascii=False, indent=2))
        return 0

    print("USD/JPY forecast snapshot")
    print("source: forecast_gate | as_of: {}".format(selected[0][1].get("as_of") or "-"))
    print("")
    for horizon, row in selected:
        print(_format_line(horizon, row))
        print("")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
