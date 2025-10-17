"""Focus allocation helper for macro/micro/scalp pockets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Focus = Literal["micro", "macro", "hybrid", "event"]

_SCALP_DEFAULT = 0.12
_SCALP_MIN = 0.06
_SCALP_MAX = 0.2


@dataclass(frozen=True)
class FocusDecision:
    focus_tag: Focus
    weight_macro: float
    weight_micro: float
    weight_scalp: float

    def as_dict(self) -> dict:
        return {
            "focus": self.focus_tag,
            "weights": {
                "macro": self.weight_macro,
                "micro": self.weight_micro,
                "scalp": self.weight_scalp,
            },
        }


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _derive_scalp_share(
    *,
    macro_regime: str,
    micro_regime: str,
    event_soon: bool,
    strong_trend: bool,
    high_volatility: bool,
) -> float:
    share = _SCALP_DEFAULT
    if event_soon:
        share *= 0.6
    if strong_trend:
        share *= 0.75
    elif high_volatility:
        share *= 1.25

    if macro_regime in ("Range", "Mixed") and micro_regime in ("Range", "Mixed"):
        share *= 1.2
    elif micro_regime == "Breakout":
        share *= 0.85

    return _clamp(round(share, 4), _SCALP_MIN, _SCALP_MAX)


def _normalize_weights(macro: float, micro: float, scalp: float) -> tuple[float, float, float]:
    macro = max(macro, 0.0)
    micro = max(micro, 0.0)
    scalp = max(scalp, 0.0)
    total = macro + micro + scalp
    if total <= 0:
        return 0.34, 0.34, 0.32
    macro /= total
    micro /= total
    scalp /= total
    return round(macro, 4), round(micro, 4), round(scalp, 4)


def decide_focus(
    macro_regime: str,
    micro_regime: str,
    *,
    event_soon: bool = False,
    macro_pf: float | None = None,
    micro_pf: float | None = None,
    strong_trend: bool = False,
    high_volatility: bool = False,
) -> FocusDecision:
    """Return a three-pocket allocation that never drops the scalp view."""

    focus: Focus
    macro_weight: float

    if event_soon:
        focus = "event"
        macro_weight = 0.45
    elif strong_trend and macro_regime == "Trend" and micro_regime in ("Trend", "Breakout"):
        focus = "macro"
        macro_weight = 0.65
    elif macro_regime == "Trend" and micro_regime == "Trend":
        focus = "macro"
        macro_weight = 0.55
    elif macro_regime == "Trend" and micro_regime in ("Range", "Mixed"):
        focus = "hybrid"
        macro_weight = 0.45
    elif macro_regime in ("Range", "Mixed") and micro_regime == "Breakout":
        focus = "micro"
        macro_weight = 0.4
    elif macro_regime == "Range" and micro_regime == "Range":
        focus = "micro"
        macro_weight = 0.5
    elif high_volatility and micro_regime in ("Range", "Mixed"):
        focus = "hybrid"
        macro_weight = 0.6
    else:
        focus = "hybrid"
        base = 0.5
        if macro_pf is not None and micro_pf is not None:
            diff = _clamp((macro_pf - micro_pf) / 2.0, -0.2, 0.2)
            base += diff
        macro_weight = _clamp(round(base, 2), 0.3, 0.7)

    scalp_weight = _derive_scalp_share(
        macro_regime=macro_regime,
        micro_regime=micro_regime,
        event_soon=event_soon,
        strong_trend=strong_trend,
        high_volatility=high_volatility,
    )

    remainder = max(0.0, 1.0 - scalp_weight)
    macro_weight = min(macro_weight, remainder * 0.9 if focus == "event" else remainder)
    micro_weight = max(0.0, remainder - macro_weight)

    if focus != "event" and micro_weight < 0.15:
        redistribution = min(0.15 - micro_weight, macro_weight * 0.25)
        micro_weight += redistribution
        macro_weight = max(0.0, macro_weight - redistribution)

    macro_weight, micro_weight, scalp_weight = _normalize_weights(
        macro_weight,
        micro_weight,
        scalp_weight,
    )

    return FocusDecision(
        focus_tag=focus,
        weight_macro=macro_weight,
        weight_micro=micro_weight,
        weight_scalp=scalp_weight,
    )
