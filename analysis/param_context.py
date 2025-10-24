"""
analysis.param_context
~~~~~~~~~~~~~~~~~~~~~~
市場環境に応じた動的パラメータ計算を担当する補助モジュール。

M1/H4 のテクニカル因子やスプレッド監視結果からローリング統計を構築し、
リスク配分やステージ配分に活用できるスナップショットを提供する。
"""

from __future__ import annotations

import bisect
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Deque, Dict, Iterable, Optional, Sequence, Tuple


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _safe_float(value: object, default: float = 0.0, *, minimum: float = 0.0) -> float:
    try:
        val = float(value) if value is not None else default
    except (TypeError, ValueError):
        return default
    if math.isnan(val) or math.isinf(val):
        return default
    if minimum is not None:
        val = max(minimum, val)
    return val


class RollingMetric:
    """単純なローリング指標。percentile rank を算出する。"""

    __slots__ = ("_values", "_maxlen", "_min_samples")

    def __init__(self, *, maxlen: int = 720, min_samples: int = 30) -> None:
        self._values: Deque[float] = deque(maxlen=maxlen)
        self._maxlen = maxlen
        self._min_samples = max(1, min_samples)

    def add(self, value: float | None) -> None:
        if value is None:
            return
        try:
            val = float(value)
        except (TypeError, ValueError):
            return
        if not math.isfinite(val):
            return
        self._values.append(val)

    def ready(self) -> bool:
        return len(self._values) >= self._min_samples

    def values(self) -> Iterable[float]:
        return tuple(self._values)

    def percentile(self, pct: float) -> float:
        if not self._values:
            return 0.0
        pct = _clamp(pct, 0.0, 100.0)
        sorted_vals = sorted(self._values)
        if len(sorted_vals) == 1:
            return sorted_vals[0]
        if pct == 100.0:
            return sorted_vals[-1]
        rank = pct / 100.0 * (len(sorted_vals) - 1)
        lower = int(math.floor(rank))
        upper = min(len(sorted_vals) - 1, lower + 1)
        frac = rank - lower
        return sorted_vals[lower] * (1.0 - frac) + sorted_vals[upper] * frac

    def percentile_rank(self, value: float | None) -> float:
        """最新値に対する百分位 (0.0-1.0)。履歴不足時は 0.5 を返す。"""
        if not self._values:
            return 0.5
        if value is None:
            value = self._values[-1]
        try:
            val = float(value)
        except (TypeError, ValueError):
            return 0.5
        if not math.isfinite(val):
            return 0.5
        sorted_vals = sorted(self._values)
        # bisect で位置を取得し、同値の中央値を使って滑らかにする
        lo = bisect.bisect_left(sorted_vals, val)
        hi = bisect.bisect_right(sorted_vals, val)
        if len(sorted_vals) == 1:
            idx = 0
        else:
            idx = (lo + hi) / 2.0
        rank = idx / max(1, len(sorted_vals) - 1)
        return _clamp(rank, 0.0, 1.0)


@dataclass(slots=True)
class ParamSnapshot:
    ts: datetime
    atr_pips: float
    atr_score: float
    adx_m1: float
    adx_m1_score: float
    adx_h4: float
    adx_h4_score: float
    vol_5m: float
    vol_score: float
    spread_pips: float
    spread_score: float
    liquidity_score: float
    volatility_state: str
    liquidity_state: str
    risk_appetite: float
    vol_high_ratio: float
    stage_bias: Dict[str, float] = field(default_factory=dict)
    notes: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, float]:
        payload: Dict[str, float] = {
            "atr_pips": round(self.atr_pips, 3),
            "atr_score": round(self.atr_score, 3),
            "adx_m1": round(self.adx_m1, 2),
            "adx_m1_score": round(self.adx_m1_score, 3),
            "adx_h4": round(self.adx_h4, 2),
            "adx_h4_score": round(self.adx_h4_score, 3),
            "vol_5m": round(self.vol_5m, 3),
            "vol_score": round(self.vol_score, 3),
            "spread_pips": round(self.spread_pips, 3),
            "spread_score": round(self.spread_score, 3),
            "liquidity_score": round(self.liquidity_score, 3),
            "risk_appetite": round(self.risk_appetite, 3),
            "vol_high_ratio": round(self.vol_high_ratio, 3),
        }
        payload.update({f"stage_bias_{k}": round(v, 3) for k, v in self.stage_bias.items()})
        payload.update(self.notes)
        return payload


class ParamContext:
    """市場パラメータのローリング推定器。"""

    def __init__(self) -> None:
        self._metrics: Dict[str, RollingMetric] = {
            "atr_pips": RollingMetric(maxlen=720, min_samples=60),
            "vol_5m": RollingMetric(maxlen=720, min_samples=60),
            "adx_m1": RollingMetric(maxlen=720, min_samples=60),
            "adx_h4": RollingMetric(maxlen=720, min_samples=60),
            "spread_pips": RollingMetric(maxlen=720, min_samples=90),
        }
        self._last_snapshot: Optional[ParamSnapshot] = None
        self._last_stage_overrides: Dict[str, Tuple[float, ...]] = {}
        self._vol_state_window: Deque[int] = deque(maxlen=720)
        self._vol_high_count: int = 0

    @staticmethod
    def _state_from_score(score: float) -> str:
        if score <= 0.35:
            return "low"
        if score >= 0.65:
            return "high"
        return "normal"

    @staticmethod
    def _liquidity_state(spread_score: float) -> str:
        if spread_score <= 0.35:
            return "tight"
        if spread_score >= 0.65:
            return "wide"
        return "normal"

    def update(
        self,
        *,
        now: datetime,
        fac_m1: Dict[str, object],
        fac_h4: Dict[str, object],
        spread_snapshot: Optional[Dict[str, object]] = None,
    ) -> ParamSnapshot:
        atr_pips = _safe_float(
            fac_m1.get("atr_pips")
            if fac_m1 is not None
            else None,
            default=(_safe_float(fac_m1.get("atr"), 0.0) if fac_m1 else 0.0) * 100.0,
        )
        atr_pips = max(0.0, atr_pips)

        vol_5m = _safe_float(fac_m1.get("vol_5m") if fac_m1 else None, default=0.0)
        vol_5m = max(0.0, vol_5m)

        adx_m1 = _safe_float(fac_m1.get("adx") if fac_m1 else None, default=0.0)
        adx_m1 = max(0.0, adx_m1)

        adx_h4 = _safe_float(fac_h4.get("adx") if fac_h4 else None, default=0.0)
        adx_h4 = max(0.0, adx_h4)

        spread_pips = 0.0
        if spread_snapshot:
            spread_pips = _safe_float(spread_snapshot.get("spread_pips"), default=0.0, minimum=0.0)

        self._metrics["atr_pips"].add(atr_pips)
        self._metrics["vol_5m"].add(vol_5m)
        self._metrics["adx_m1"].add(adx_m1)
        self._metrics["adx_h4"].add(adx_h4)
        self._metrics["spread_pips"].add(spread_pips)

        atr_score = self._metrics["atr_pips"].percentile_rank(atr_pips)
        vol_score = self._metrics["vol_5m"].percentile_rank(vol_5m)
        adx_m1_score = self._metrics["adx_m1"].percentile_rank(adx_m1)
        adx_h4_score = self._metrics["adx_h4"].percentile_rank(adx_h4)
        spread_score = self._metrics["spread_pips"].percentile_rank(spread_pips)
        liquidity_score = _clamp(1.0 - spread_score, 0.0, 1.0)

        trend_score = (adx_m1_score + adx_h4_score) / 2.0
        vol_state = self._state_from_score(atr_score)
        if self._vol_state_window.maxlen:
            removed = 0
            if len(self._vol_state_window) == self._vol_state_window.maxlen:
                removed = self._vol_state_window[0]
            flag = 1 if vol_state == "high" else 0
            self._vol_state_window.append(flag)
            self._vol_high_count += flag - removed
        window_len = len(self._vol_state_window) or 1
        vol_high_ratio = _clamp(
            (self._vol_high_count / window_len) if window_len else 0.0,
            0.0,
            1.0,
        )

        # 高ボラ＆スプレッド拡大時はリスク Appetite を抑制し、トレンドが強ければ加点
        risk_appetite = 0.5
        risk_appetite += (trend_score - 0.5) * 0.45
        risk_appetite -= (atr_score - 0.5) * 0.5
        risk_appetite -= (spread_score - 0.5) * 0.6
        risk_appetite = _clamp(risk_appetite, 0.0, 1.0)

        liquidity_penalty = max(0.0, spread_score - 0.4) * 0.45
        high_vol_penalty = max(0.0, atr_score - 0.45) * 0.55

        stage_bias = {
            "macro": _clamp(
                0.88
                + (trend_score - 0.5) * 0.35
                - high_vol_penalty
                - liquidity_penalty,
                0.6,
                1.15,
            ),
            "micro": _clamp(
                0.88
                + (0.5 - trend_score) * 0.32
                + (0.45 - atr_score) * 0.5
                - liquidity_penalty * 0.8,
                0.68,
                1.25,
            ),
            "scalp": _clamp(
                0.9
                + (atr_score - 0.5) * 0.45
                + (vol_score - 0.5) * 0.35
                - liquidity_penalty * 1.3,
                0.6,
                1.3,
            ),
        }

        if vol_high_ratio >= 0.3:
            stage_bias["macro"] = min(stage_bias["macro"], 0.65)
            stage_bias["micro"] = min(stage_bias["micro"], 0.75)
            stage_bias["scalp"] = min(stage_bias["scalp"], 0.7)

        snapshot = ParamSnapshot(
            ts=now,
            atr_pips=atr_pips,
            atr_score=_clamp(atr_score, 0.0, 1.0),
            adx_m1=adx_m1,
            adx_m1_score=_clamp(adx_m1_score, 0.0, 1.0),
            adx_h4=adx_h4,
            adx_h4_score=_clamp(adx_h4_score, 0.0, 1.0),
            vol_5m=vol_5m,
            vol_score=_clamp(vol_score, 0.0, 1.0),
            spread_pips=spread_pips,
            spread_score=_clamp(spread_score, 0.0, 1.0),
            liquidity_score=liquidity_score,
            volatility_state=vol_state,
            liquidity_state=self._liquidity_state(spread_score),
            risk_appetite=risk_appetite,
            vol_high_ratio=vol_high_ratio,
            stage_bias=stage_bias,
            notes={
                "trend_score": round(trend_score, 3),
                "vol_state_score": round(atr_score, 3),
                "spread_state_score": round(spread_score, 3),
                "vol_high_ratio": round(vol_high_ratio, 3),
            },
        )

        self._last_snapshot = snapshot
        return snapshot

    @staticmethod
    def _adjust_stage_plan(plan: Sequence[float], bias: float) -> Tuple[float, ...]:
        if not plan:
            return (1.0,)
        if len(plan) == 1:
            return (1.0,)

        bias = _clamp(bias, 0.55, 1.35)
        first = plan[0]
        base_sum = sum(plan)
        if base_sum <= 0:
            return tuple(plan)
        normalized = [max(0.0, p / base_sum) for p in plan]
        first_norm = normalized[0]
        rest_sum = sum(normalized[1:])
        if rest_sum <= 0:
            return (1.0,)

        target_first = _clamp(first_norm * bias, 0.05, 0.85)
        remaining = max(0.0, 1.0 - target_first)
        scale = remaining / rest_sum if rest_sum > 0 else 0.0
        adjusted = [target_first]
        for frac in normalized[1:]:
            adjusted.append(max(0.0, frac * scale))

        total = sum(adjusted)
        if total <= 0:
            return tuple(plan)
        adjusted = [val / total for val in adjusted]

        rounded = [round(val, 4) for val in adjusted]
        diff = round(1.0 - sum(rounded), 4)
        rounded[-1] = round(max(0.0, rounded[-1] + diff), 4)
        return tuple(rounded)

    @staticmethod
    def _limit_stages(plan: Tuple[float, ...], max_stages: int) -> Tuple[float, ...]:
        if len(plan) <= max_stages:
            return plan
        trimmed = tuple(plan[:max_stages])
        return trimmed

    def stage_overrides(
        self,
        base_plans: Dict[str, Sequence[float]],
        *,
        range_active: bool,
    ) -> Tuple[Dict[str, Tuple[float, ...]], bool, Dict[str, float]]:
        if self._last_snapshot is None:
            defaults = {k: tuple(v) for k, v in base_plans.items()}
            return defaults, False, {}

        overrides: Dict[str, Tuple[float, ...]] = {}
        applied_bias: Dict[str, float] = {}
        high_vol = self._last_snapshot.vol_high_ratio >= 0.3
        for pocket, plan in base_plans.items():
            bias = self._last_snapshot.stage_bias.get(pocket, 1.0)
            if range_active and pocket == "macro":
                bias = min(bias, 0.75)
            adjusted_plan = self._adjust_stage_plan(plan, bias)
            if high_vol:
                limit = 1 if pocket in {"macro", "micro"} else 2
                adjusted_plan = self._limit_stages(adjusted_plan, limit)
            overrides[pocket] = adjusted_plan
            applied_bias[pocket] = bias

        changed = overrides != self._last_stage_overrides
        if changed:
            self._last_stage_overrides = overrides
        return overrides, changed, applied_bias

    @property
    def last_snapshot(self) -> Optional[ParamSnapshot]:
        return self._last_snapshot
