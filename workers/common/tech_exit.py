from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional, Tuple

from analysis.technique_engine import evaluate_exit_techniques
from indicators.factor_cache import all_factors
from utils.strategy_protection import base_strategy_tag, exit_profile_for_tag

try:  # optional in offline/backtest
    from market_data import tick_window
except Exception:  # pragma: no cover
    tick_window = None

LOG = logging.getLogger(__name__)

_LAST_TECH_EXIT_TS: dict[str, float] = {}


def _safe_float(value: object) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _latest_mid() -> Optional[float]:
    if tick_window is not None:
        try:
            ticks = tick_window.recent_ticks(seconds=2.0, limit=1)
        except Exception:
            ticks = None
        if ticks:
            try:
                return float(ticks[-1]["mid"])
            except Exception:
                pass
    try:
        return float((all_factors().get("M1") or {}).get("close"))
    except Exception:
        return None


def _trade_strategy_tag(trade: dict) -> str:
    thesis = trade.get("entry_thesis") if isinstance(trade.get("entry_thesis"), dict) else None
    if isinstance(thesis, dict):
        for key in ("strategy_tag", "strategy_tag_raw", "strategy", "tag"):
            val = thesis.get(key)
            if val:
                return str(val)
    for key in ("strategy_tag", "strategy"):
        val = trade.get(key)
        if val:
            return str(val)
    return ""


@dataclass(frozen=True, slots=True)
class TechExitParams:
    enabled: bool
    min_hold_sec: float
    min_neg_pips: float
    max_pnl_pips: float
    cooldown_sec: float


def _default_min_neg_pips(pocket: str) -> float:
    key = str(pocket or "").strip().lower()
    if key == "scalp_fast":
        return 1.3
    if key == "scalp":
        return 1.8
    if key == "micro":
        return 3.5
    if key == "macro":
        return 6.0
    return 2.0


def resolve_tech_exit(exit_profile: Optional[dict], *, pocket: str) -> TechExitParams:
    prof = exit_profile if isinstance(exit_profile, dict) else {}
    enabled = bool(prof.get("tech_exit_enabled", False))

    # Default to the worker's min-hold, unless explicitly overridden.
    min_hold_raw = prof.get("tech_exit_min_hold_sec")
    if min_hold_raw is None:
        min_hold_raw = prof.get("min_hold_sec")
    min_hold_sec = max(0.0, float(_safe_float(min_hold_raw) or 0.0))

    min_neg_raw = _safe_float(prof.get("tech_exit_min_neg_pips"))
    min_neg_pips = (
        max(0.0, float(min_neg_raw)) if min_neg_raw is not None else _default_min_neg_pips(pocket)
    )

    # Evaluate only when PnL is <= this threshold (default: <= 0, i.e. loss/flat only).
    max_pnl_raw = _safe_float(prof.get("tech_exit_max_pnl_pips"))
    max_pnl_pips = float(max_pnl_raw) if max_pnl_raw is not None else 0.0

    cooldown_raw = _safe_float(prof.get("tech_exit_cooldown_sec"))
    cooldown_sec = max(0.0, float(cooldown_raw)) if cooldown_raw is not None else 4.0

    return TechExitParams(
        enabled=enabled,
        min_hold_sec=min_hold_sec,
        min_neg_pips=min_neg_pips,
        max_pnl_pips=max_pnl_pips,
        cooldown_sec=cooldown_sec,
    )


def maybe_tech_exit(
    *,
    trade: dict,
    side: str,
    pocket: str,
    pnl_pips: Optional[float],
    hold_sec: float,
    current_price: Optional[float] = None,
    strategy_tag: Optional[str] = None,
    exit_profile: Optional[dict] = None,
) -> Tuple[bool, Optional[str], bool]:
    """Return (should_exit, reason, allow_negative).

    This is intended as a per-strategy "technical composite" loss cut:
    - gated by config/strategy_exit_protections.yaml -> exit_profile.tech_exit_*
    - uses analysis.technique_engine.evaluate_exit_techniques() for holistic signals
    """
    tag = str(strategy_tag or _trade_strategy_tag(trade) or "").strip()
    tag = base_strategy_tag(tag) or tag

    prof = exit_profile if isinstance(exit_profile, dict) else exit_profile_for_tag(tag)
    params = resolve_tech_exit(prof, pocket=pocket)
    if not params.enabled:
        return False, None, False

    if hold_sec < params.min_hold_sec:
        return False, None, False

    if pnl_pips is None:
        return False, None, False

    # Default: only evaluate for losing/flat trades.
    if pnl_pips > params.max_pnl_pips:
        return False, None, False

    if params.min_neg_pips > 0 and pnl_pips > -params.min_neg_pips:
        return False, None, False

    trade_id = str(trade.get("trade_id") or "")
    if not trade_id:
        return False, None, False

    now_mono = time.monotonic()
    last = _LAST_TECH_EXIT_TS.get(trade_id, 0.0)
    if params.cooldown_sec > 0.0 and (now_mono - last) < params.cooldown_sec:
        return False, None, False

    price = current_price
    if price is None or price <= 0:
        price = _latest_mid()
    if price is None or price <= 0:
        return False, None, False

    try:
        decision = evaluate_exit_techniques(
            trade=trade,
            current_price=float(price),
            side=str(side),
            pocket=str(pocket),
        )
    except Exception:
        LOG.exception("[tech_exit] evaluation failed trade=%s tag=%s", trade_id, tag or "-")
        return False, None, False

    if decision and getattr(decision, "should_exit", False):
        reason = getattr(decision, "reason", None)
        allow_negative = bool(getattr(decision, "allow_negative", False))
        if reason:
            _LAST_TECH_EXIT_TS[trade_id] = now_mono
            return True, str(reason), allow_negative
    return False, None, False

