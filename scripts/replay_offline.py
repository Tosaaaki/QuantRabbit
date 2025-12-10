#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts.replay_offline
~~~~~~~~~~~~~~~~~~~~~~
リプレイ用の JSON ローソクを使って ParamContext / 動的リスク / ステージ配分を
本番ロジックに近い形で検証するための簡易ハーネス。

• indicators.factor_cache を経由して因子を更新
• ParamContext → stage_overrides / _dynamic_risk_pct を評価
• 代表的なストラテジーを走査して SL 平均から lot 計算
• 各分足ごとのリスク係数・ステージ第1段の比率などを CSV に出力

使い方:
    python scripts/replay_offline.py --candles logs/candles_M1_20251020.json
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.param_context import ParamContext, ParamSnapshot
from analysis.range_guard import detect_range_mode
from execution.risk_guard import allowed_lot
from indicators import factor_cache as factor_cache_module

# main から内部関数を import（副作用抑制のため遅延 import）
main_mod = importlib.import_module("main")
STRATEGIES = main_mod.STRATEGIES
ALLOWED_RANGE_STRATEGIES = main_mod.ALLOWED_RANGE_STRATEGIES
_BASE_STAGE_RATIOS = main_mod._BASE_STAGE_RATIOS
_set_stage_plan_overrides = main_mod._set_stage_plan_overrides
_dynamic_risk_pct = main_mod._dynamic_risk_pct


def _parse_time(ts: str) -> datetime:
    ts = ts.replace("Z", "+00:00")
    if "." in ts:
        head, frac = ts.split(".", 1)
        frac = (frac + "000000")[:6]
        ts = f"{head}.{frac}+00:00"
    return datetime.fromisoformat(ts).astimezone(timezone.utc)


def _load_oanda_candles(path: Path) -> List[dict]:
    with path.open() as f:
        payload = json.load(f)
    candles = payload.get("candles", [])
    result: List[dict] = []
    for cndl in candles:
        mid = cndl.get("mid") or {}
        result.append(
            {
                "time": _parse_time(cndl["time"]),
                "open": float(mid.get("o", mid.get("open", 0.0))),
                "high": float(mid.get("h", mid.get("high", 0.0))),
                "low": float(mid.get("l", mid.get("low", 0.0))),
                "close": float(mid.get("c", mid.get("close", 0.0))),
            }
        )
    return result


def _infer_h4_path(m1_path: Path) -> Optional[Path]:
    try:
        stem = m1_path.name.replace("candles_M1_", "")
        candidate = m1_path.with_name(f"candles_H4_{stem}")
        return candidate if candidate.exists() else None
    except Exception:
        return None


def _collect_h4_history(m1_path: Path, explicit: Optional[Path]) -> List[dict]:
    if explicit and explicit.exists():
        return _load_oanda_candles(explicit)
    base_dir = m1_path.parent
    token = m1_path.stem.replace("candles_M1_", "")
    h4_files = sorted(base_dir.glob("candles_H4_*.json"))
    combined: List[dict] = []
    seen: set[str] = set()
    for path in h4_files:
        stem = path.stem.replace("candles_H4_", "")
        if stem > token:
            continue
        for candle in _load_oanda_candles(path):
            key = candle["time"].isoformat()
            if key in seen:
                continue
            seen.add(key)
            combined.append(candle)
    combined.sort(key=lambda c: c["time"])
    return combined


def _avg(values: Iterable[float]) -> float:
    data = [v for v in values if isinstance(v, (int, float))]
    if not data:
        return 0.0
    return float(statistics.mean(data))


@dataclass(slots=True)
class ReplaySample:
    ts: datetime
    risk_pct: float
    total_lot: float
    avg_sl: float
    stage_macro_first: float
    stage_micro_first: float
    stage_scalp_first: float
    vol_high_ratio: float
    volatility_state: str
    h4_fallback: bool
    story_state: str
    macro_trend: str
    micro_trend: str
    higher_trend: str
    structure_bias: float

    def to_csv_row(self) -> str:
        return ",".join(
            [
                self.ts.isoformat(),
                f"{self.risk_pct:.5f}",
                f"{self.total_lot:.4f}",
                f"{self.avg_sl:.2f}",
                f"{self.stage_macro_first:.3f}",
                f"{self.stage_micro_first:.3f}",
                f"{self.stage_scalp_first:.3f}",
                f"{self.vol_high_ratio:.3f}",
                self.volatility_state,
                "1" if self.h4_fallback else "0",
                self.story_state,
                self.macro_trend,
                self.micro_trend,
                self.higher_trend,
                f"{self.structure_bias:.2f}",
            ]
        )


@dataclass(slots=True)
class GateEvent:
    ts: datetime
    pocket: str
    action: str
    rsi: float
    atr_pips: float
    reasons: str

    def to_csv_row(self) -> str:
        return ",".join(
            [
                self.ts.isoformat(),
                self.pocket,
                self.action,
                f"{self.rsi:.2f}",
                f"{self.atr_pips:.2f}",
                self.reasons,
            ]
        )


async def _feed_candles(
    tf: str,
    candles: List[dict],
    on_candle,
) -> None:
    for c in candles:
        payload = {
            "open": c["open"],
            "high": c["high"],
            "low": c["low"],
            "close": c["close"],
            "time": c["time"],
        }
        await on_candle(tf, payload)


def _simple_ranked_strategies(range_active: bool) -> List[str]:
    base = ["TrendMA", "Donchian55", "BB_RSI", "RangeFader", "M1Scalper", "PulseBreak"]
    if range_active and "RangeFader" not in base:
        base.append("RangeFader")
    return base


def _evaluate_strategies(
    ranked: List[str], fac_m1: dict, *, range_active: bool
) -> List[dict]:
    signals: List[dict] = []
    for name in ranked:
        cls = STRATEGIES.get(name)
        if not cls:
            continue
        if range_active and cls.name not in ALLOWED_RANGE_STRATEGIES:
            continue
        raw = cls.check(fac_m1)
        if not raw:
            continue
        if not isinstance(raw, dict):
            continue
        raw = dict(raw)
        raw["strategy"] = name
        raw["pocket"] = cls.pocket
        signals.append(raw)
    return signals


def _avg_stage_first(stage_overrides: Dict[str, tuple[float, ...]], pocket: str) -> float:
    plan = stage_overrides.get(pocket)
    if not plan:
        return 0.0
    return float(plan[0]) if plan else 0.0


async def replay_day(
    m1_path: Path,
    *,
    h4_path: Optional[Path],
    equity: float,
    margin_available: float,
    margin_rate: float,
    lot_cap: Optional[float],
    start_ts: Optional[datetime] = None,
    end_ts: Optional[datetime] = None,
) -> tuple[List[ReplaySample], List[GateEvent]]:
    importlib.reload(factor_cache_module)
    on_candle = factor_cache_module.on_candle
    all_factors = factor_cache_module.all_factors

    m1_candles = _load_oanda_candles(m1_path)
    h4_candles = _collect_h4_history(m1_path, h4_path)

    if not m1_candles:
        raise RuntimeError(f"No M1 candles found in {m1_path}")

    param_ctx = ParamContext()
    chart_story = main_mod.ChartStory()
    results: List[ReplaySample] = []
    gate_events: List[GateEvent] = []

    # まず H4 を流し込んでベース指標を準備
    await _feed_candles("H4", h4_candles, on_candle)

    last_fac_h4: Optional[dict] = None

    for candle in m1_candles:
        await on_candle("M1", candle)
        factors = all_factors()
        fac_m1 = factors.get("M1")
        fac_h4 = factors.get("H4")
        if not fac_m1:
            continue
        h4_fallback = False
        if fac_h4:
            last_fac_h4 = fac_h4
        else:
            fac_h4 = last_fac_h4
            if fac_h4 is None:
                fac_h4 = {k: fac_m1.get(k) for k in fac_m1.keys()}
                h4_fallback = True
        if "close" not in fac_m1 or "close" not in fac_h4:
            continue

        ts = candle["time"]
        if start_ts and ts < start_ts:
            continue
        if end_ts and ts > end_ts:
            break
        story_state = "missing"
        story_snapshot = chart_story.update(fac_m1, fac_h4)
        if story_snapshot:
            story_state = "fresh"
        else:
            story_snapshot = chart_story.last_snapshot
            if story_snapshot:
                story_state = "reuse"
        macro_trend = story_snapshot.macro_trend if story_snapshot else ""
        micro_trend = story_snapshot.micro_trend if story_snapshot else ""
        higher_trend = story_snapshot.higher_trend if story_snapshot else ""
        structure_bias = story_snapshot.structure_bias if story_snapshot else 0.0
        range_ctx = detect_range_mode(fac_m1, fac_h4)
        ranked = _simple_ranked_strategies(range_ctx.active)
        signals = _evaluate_strategies(ranked, fac_m1, range_active=range_ctx.active)

        param_snapshot: ParamSnapshot = param_ctx.update(
            now=ts,
            fac_m1=fac_m1,
            fac_h4=fac_h4,
            spread_snapshot=None,
        )

        stage_overrides, _, _ = param_ctx.stage_overrides(
            _BASE_STAGE_RATIOS,
            range_active=range_ctx.active,
        )
        _set_stage_plan_overrides(stage_overrides)

        avg_sl = _avg(sig.get("sl_pips") for sig in signals if sig.get("sl_pips"))
        if avg_sl <= 0:
            avg_sl = 12.0

        risk_pct = _dynamic_risk_pct(
            signals,
            range_ctx.active,
            weight_macro=0.5,
            context=param_snapshot,
        )
        price = float(fac_m1.get("close") or 0.0)
        lot_total = allowed_lot(
            equity,
            sl_pips=max(1.0, avg_sl),
            margin_available=margin_available,
            price=price if price > 0 else None,
            margin_rate=margin_rate,
            risk_pct_override=risk_pct,
        )
        if lot_cap is not None:
            lot_total = min(lot_total, lot_cap)

        results.append(
            ReplaySample(
                ts=ts,
                risk_pct=risk_pct,
                total_lot=lot_total,
                avg_sl=avg_sl,
                stage_macro_first=_avg_stage_first(stage_overrides, "macro"),
                stage_micro_first=_avg_stage_first(stage_overrides, "micro"),
                stage_scalp_first=_avg_stage_first(stage_overrides, "scalp"),
                vol_high_ratio=param_snapshot.vol_high_ratio,
                volatility_state=param_snapshot.volatility_state,
                h4_fallback=h4_fallback,
                story_state=story_state,
                macro_trend=macro_trend,
                micro_trend=micro_trend,
                higher_trend=higher_trend,
                structure_bias=structure_bias,
            )
        )
        # collect gate hits for diagnostics (micro long focus)
        try:
            rsi = float(fac_m1.get("rsi", 50.0) or 0.0)
        except (TypeError, ValueError):
            rsi = 50.0
        atr_raw = fac_m1.get("atr_pips")
        if atr_raw is None:
            atr_raw = (fac_m1.get("atr") or 0.0) * 100
        try:
            atr_pips = float(atr_raw or 0.0)
        except (TypeError, ValueError):
            atr_pips = 0.0

        reasons: list[str] = []
        micro_rsi_floor = main_mod.RSI_LONG_FLOOR.get("micro")
        if micro_rsi_floor is not None and rsi < micro_rsi_floor:
            reasons.append(f"rsi<{micro_rsi_floor:.1f}")
        micro_atr_floor = main_mod.POCKET_ATR_MIN_PIPS.get("micro")
        if micro_atr_floor is not None and atr_pips < micro_atr_floor:
            reasons.append(f"atr<{micro_atr_floor:.1f}")
        if reasons:
            gate_events.append(
                GateEvent(
                    ts=ts,
                    pocket="micro",
                    action="OPEN_LONG",
                    rsi=rsi,
                    atr_pips=atr_pips,
                    reasons=";".join(reasons),
                )
            )

    return results, gate_events


def summarize(samples: List[ReplaySample], *, vol_threshold: float) -> Dict[str, Dict[str, float]]:
    if not samples:
        return {}

    def _stats(items: List[ReplaySample]) -> Dict[str, float]:
        if not items:
            return {}
        risk = [s.risk_pct for s in items]
        lots = [s.total_lot for s in items]
        macro = [s.stage_macro_first for s in items]
        micro = [s.stage_micro_first for s in items]
        scalp = [s.stage_scalp_first for s in items]
        return {
            "count": float(len(items)),
            "risk_pct_min": min(risk),
            "risk_pct_max": max(risk),
            "risk_pct_avg": _avg(risk),
            "lot_avg": _avg(lots),
            "lot_max": max(lots),
            "macro_stage_avg": _avg(macro),
            "micro_stage_avg": _avg(micro),
            "scalp_stage_avg": _avg(scalp),
        }

    high = [s for s in samples if s.vol_high_ratio >= vol_threshold]
    normal = [s for s in samples if s.vol_high_ratio < vol_threshold]
    all_stats = _stats(samples)
    high_stats = _stats(high)
    normal_stats = _stats(normal)
    out: Dict[str, Dict[str, float]] = {"all": all_stats}
    if high_stats:
        out["high_vol"] = high_stats
    if normal_stats:
        out["normal"] = normal_stats
    out["meta"] = {
        "vol_threshold": vol_threshold,
        "high_vol_ratio": len(high) / len(samples) if samples else 0.0,
        "h4_fallback_ratio": sum(1 for s in samples if s.h4_fallback) / len(samples),
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="ParamContext / risk ダイナミクスのオフラインリプレイ")
    parser.add_argument(
        "--candles",
        required=True,
        nargs="+",
        help="M1 ローソク JSON (OANDA 形式)",
    )
    parser.add_argument(
        "--h4",
        default="",
        help="H4 ローソク JSON（未指定なら M1 と同じ日付を推測）",
    )
    parser.add_argument("--equity", type=float, default=10000.0, help="口座残高の想定値")
    parser.add_argument("--margin-available", type=float, default=4000.0, help="利用可能証拠金の想定値")
    parser.add_argument("--margin-rate", type=float, default=0.02, help="証拠金率 (例: 0.02)")
    parser.add_argument("--out-dir", default="tmp", help="CSV 出力先ディレクトリ")
    parser.add_argument("--lot-cap", type=float, default=None, help="allowed_lot の出力に対する上限 (lot 単位)")
    parser.add_argument(
        "--vol-threshold",
        type=float,
        default=0.3,
        help="vol_high_ratio がこの値以上を高ボラ判定とする",
    )
    parser.add_argument(
        "--summary-json",
        default="",
        help="日次・全体サマリを書き出す JSON パス",
    )
    parser.add_argument(
        "--start",
        default="",
        help="再生の開始 UTC 時刻 (例: 2025-10-24T11:20:00Z)",
    )
    parser.add_argument(
        "--end",
        default="",
        help="再生の終了 UTC 時刻 (例: 2025-10-24T12:15:00Z)",
    )
    parser.add_argument(
        "--dump-gates",
        action="store_true",
        help="RSI/ATR ゲートヒットを CSV (tmp/replay_gate_hits.csv) に書き出す",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_day_summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    global_samples: List[ReplaySample] = []
    global_gates: List[GateEvent] = []

    start_ts = _parse_time(args.start) if args.start else None
    end_ts = _parse_time(args.end) if args.end else None

    for candle_path_str in args.candles:
        m1_path = Path(candle_path_str)
        if not m1_path.exists():
            print(f"[WARN] skip {m1_path} (not found)")
            continue
        base_name = m1_path.stem.replace("candles_M1_", "")
        explicit_h4 = Path(args.h4) if args.h4 else None
        inferred_h4 = _infer_h4_path(m1_path)
        display_h4 = explicit_h4 or inferred_h4 or "auto-history"
        print(f"[INFO] replay {base_name} (H4={display_h4})")

        samples, gate_events = asyncio.run(
            replay_day(
                m1_path,
                h4_path=explicit_h4,
                equity=args.equity,
                margin_available=args.margin_available,
                margin_rate=args.margin_rate,
                lot_cap=args.lot_cap,
                start_ts=start_ts,
                end_ts=end_ts,
            )
        )
        print(f"[INFO] generated {len(samples)} samples for {base_name}")
        if samples:
            global_samples.extend(samples)
        if gate_events:
            global_gates.extend(gate_events)
        summary = summarize(samples, vol_threshold=args.vol_threshold)
        per_day_summary[base_name] = summary

        csv_path = out_dir / f"replay_metrics_{base_name}.csv"
        with csv_path.open("w", encoding="utf-8") as fh:
            fh.write(
                "timestamp,risk_pct,lot,avg_sl,stage_macro_first,stage_micro_first,stage_scalp_first,vol_high_ratio,vol_state,h4_fallback,story_state,macro_trend,micro_trend,higher_trend,structure_bias\n"
            )
            for sample in samples:
                fh.write(sample.to_csv_row() + "\n")

        if summary:
            print(f"[SUMMARY] {base_name} -> {json.dumps(summary, ensure_ascii=False)}")
        else:
            print(f"[WARN] no samples produced for {base_name}")

    global_summary = summarize(global_samples, vol_threshold=args.vol_threshold)
    if global_summary:
        print(f"[SUMMARY] GLOBAL -> {json.dumps(global_summary, ensure_ascii=False)}")

    if args.dump_gates and global_gates:
        gate_path = out_dir / "replay_gate_hits.csv"
        with gate_path.open("w", encoding="utf-8") as fh:
            fh.write("timestamp,pocket,action,rsi,atr_pips,reasons\n")
            for event in global_gates:
                fh.write(event.to_csv_row() + "\n")
        print(f"[INFO] Gate hits exported to {gate_path}")

    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "settings": {
                "equity": args.equity,
                "margin_available": args.margin_available,
                "margin_rate": args.margin_rate,
                "lot_cap": args.lot_cap,
                "vol_threshold": args.vol_threshold,
            },
            "per_day": per_day_summary,
            "global": global_summary,
        }
        with summary_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        print(f"[INFO] summary saved to {summary_path}")


if __name__ == "__main__":
    main()
