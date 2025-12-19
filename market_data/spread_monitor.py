"""
market_data.spread_monitor
~~~~~~~~~~~~~~~~~~~~~~~~~~
Tick ベースで直近スプレッドを監視し、一定以上に拡大した場合は
新規エントリーをクールダウンするガードを提供する。
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple
import os

from utils.secrets import get_secret

_FALSEY = {"", "0", "false", "no"}

PIP_VALUE = 0.01  # USD/JPY で 1 pip = 0.01


def _load_float(key: str, default: float, *, minimum: Optional[float] = None) -> float:
    try:
        value = float(get_secret(key))
    except Exception:
        return default
    if minimum is not None and value < minimum:
        return default
    return value


def _load_int(key: str, default: int, *, minimum: Optional[int] = None) -> int:
    try:
        value = int(float(get_secret(key)))
    except Exception:
        return default
    if minimum is not None and value < minimum:
        return default
    return value


MAX_SPREAD_PIPS = _load_float("spread_guard_max_pips", 2.0, minimum=0.1)
RELEASE_SPREAD_PIPS = _load_float(
    "spread_guard_release_pips", max(0.7 * MAX_SPREAD_PIPS, MAX_SPREAD_PIPS - 0.4)
)
if RELEASE_SPREAD_PIPS >= MAX_SPREAD_PIPS:
    RELEASE_SPREAD_PIPS = max(MAX_SPREAD_PIPS * 0.75, MAX_SPREAD_PIPS - 0.35)

WINDOW_SECONDS = _load_float("spread_guard_window_sec", 3.5, minimum=0.5)
# サンプルを安定化させるためデフォルトをやや長めにする
WINDOW_SECONDS = max(WINDOW_SECONDS, 8.0)
COOLDOWN_SECONDS = _load_float("spread_guard_cooldown_sec", 12.0, minimum=5.0)
MAX_AGE_MS = _load_int("spread_guard_max_age_ms", 4000, minimum=1000)
MIN_HIGH_SAMPLES = _load_int("spread_guard_min_high_samples", 3, minimum=1)
RELEASE_SAMPLES = _load_int("spread_guard_release_samples", 4, minimum=1)
BASELINE_WINDOW_SECONDS = _load_float(
    "spread_guard_baseline_window_sec", 120.0, minimum=10.0
)
BASELINE_MIN_SAMPLES = _load_int(
    "spread_guard_baseline_min_samples", 30, minimum=5
)
BASELINE_BLOCK_PIPS = _load_float(
    "spread_guard_baseline_block_pips", MAX_SPREAD_PIPS, minimum=0.1
)
BASELINE_BLOCK_SECONDS = _load_float(
    "spread_guard_baseline_block_sec", 45.0, minimum=5.0
)
STALE_GRACE_SECONDS = _load_float("spread_guard_stale_grace_sec", 12.0, minimum=0.0)
# 瞬間スパイクを無視する上限（max がここ以下で pX が閾値未満ならガードしない）
SPIKE_FORGIVE_PIPS = _load_float(
    "spread_guard_spike_forgive_pips", max(MAX_SPREAD_PIPS * 2.0, MAX_SPREAD_PIPS + 0.5)
)
SPIKE_FORGIVE_PCT = _load_float("spread_guard_spike_forgive_pct", 95.0, minimum=50.0)


def _bool_from_sources(*keys: str, default: str = "0") -> bool:
    """
    Resolve a boolean-ish value from environment or config/env.toml (via get_secret).
    The first found key is used; values in {"", "0", "false", "no"} are treated as False.
    """
    for key in keys:
        env = os.getenv(key)
        if env is not None:
            return env.strip().lower() not in _FALSEY
    for key in keys:
        try:
            raw = get_secret(key)
        except Exception:
            continue
        if raw is not None:
            return str(raw).strip().lower() not in _FALSEY
    return str(default).strip().lower() not in _FALSEY


DISABLE_SPREAD_GUARD = _bool_from_sources(
    "SPREAD_GUARD_DISABLE", "SPREAD_GUARD_DISABLED", "DISABLE_SPREAD_GUARD"
)
if DISABLE_SPREAD_GUARD:
    logging.warning("[SPREAD] Guard disabled via SPREAD_GUARD_DISABLE")

# 上限を設け過去履歴が無限に伸びるのを防ぐ
_HISTORY_MAX_LEN = 180


@dataclass(slots=True)
class _Snapshot:
    monotonic_ts: float
    tick_epoch: float
    bid: float
    ask: float
    spread_pips: float


_snapshot: Optional[_Snapshot] = None
_history: Deque[Tuple[float, float]] = deque(maxlen=_HISTORY_MAX_LEN)
_baseline_history: Deque[Tuple[float, float]] = deque(maxlen=4 * _HISTORY_MAX_LEN)
_blocked_until: float = 0.0
_blocked_reason: str = ""
_last_logged_blocked: bool = False
_stale_since: Optional[float] = None


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    pct = max(0.0, min(pct, 100.0))
    if len(values) == 1:
        return values[0]
    sorted_vals = sorted(values)
    if pct == 100.0:
        return sorted_vals[-1]
    rank = pct / 100.0 * (len(sorted_vals) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_vals) - 1)
    frac = rank - lower
    return sorted_vals[lower] * (1.0 - frac) + sorted_vals[upper] * frac


def update_from_tick(tick) -> None:  # type: ignore[no-untyped-def]
    """
    Tick からスプレッド情報を更新する。tick は market_data.tick_fetcher.Tick 型想定。
    """
    global _snapshot, _blocked_until, _blocked_reason, _last_logged_blocked, _stale_since

    try:
        bid = float(tick.bid)
        ask = float(tick.ask)
    except (TypeError, AttributeError):
        return

    spread = max(0.0, ask - bid)
    spread_pips = spread / PIP_VALUE
    now = time.monotonic()
    tick_epoch = 0.0
    try:
        tick_epoch = float(tick.time.timestamp())
    except Exception:
        pass

    _snapshot = _Snapshot(
        monotonic_ts=now,
        tick_epoch=tick_epoch,
        bid=bid,
        ask=ask,
        spread_pips=spread_pips,
    )
    _stale_since = None

    _history.append((now, spread_pips))
    # 古い履歴を window 秒より前のものは削除
    while _history and now - _history[0][0] > WINDOW_SECONDS:
        _history.popleft()

    # ガード無効モード: スナップショットだけ更新して即 return
    if DISABLE_SPREAD_GUARD:
        _blocked_until = 0.0
        _blocked_reason = ""
        _last_logged_blocked = False
        return

    _baseline_history.append((now, spread_pips))
    while _baseline_history and now - _baseline_history[0][0] > BASELINE_WINDOW_SECONDS:
        _baseline_history.popleft()

    values = [val for _, val in _history]
    if not values:
        # この時点で history が空になることはないが、念のため
        return

    max_spread = max(values)
    min_spread = min(values)
    avg_spread = sum(values) / len(values)
    median_spread = _percentile(values, 50.0)
    p95_spread = _percentile(values, 95.0)

    # ガード完全無効化（エントリー阻害しない）
    triggered = False
    reason = ""

    # ベースラインが長時間広がっている場合のブロック
    baseline_values = [val for _, val in _baseline_history]
    baseline_ready = len(baseline_values) >= BASELINE_MIN_SAMPLES
    baseline_p50 = _percentile(baseline_values, 50.0) if baseline_ready else None
    if (
        baseline_ready
        and baseline_p50 is not None
        and baseline_p50 >= BASELINE_BLOCK_PIPS
    ):
        triggered = True
        reason = (
            f"baseline_p50 {baseline_p50:.2f}p >= {BASELINE_BLOCK_PIPS:.2f}p "
            f"(samples={len(baseline_values)})"
        )

    # 短期窓の中央値/95pctで判定（単発スパイクは無視）
    window_ready = len(values) >= max(MIN_HIGH_SAMPLES, RELEASE_SAMPLES)
    spike_forgive = (
        max_spread <= SPIKE_FORGIVE_PIPS
        and p95_spread < MAX_SPREAD_PIPS
        and median_spread < MAX_SPREAD_PIPS
    )
    high_count = sum(1 for val in values if val >= MAX_SPREAD_PIPS)
    if (
        not triggered
        and window_ready
        and not spike_forgive
        and (
            (median_spread >= MAX_SPREAD_PIPS and high_count >= MIN_HIGH_SAMPLES)
            or p95_spread >= MAX_SPREAD_PIPS
            or (max_spread >= MAX_SPREAD_PIPS and high_count >= MIN_HIGH_SAMPLES)
        )
    ):
        triggered = True
        reason = (
            f"spread_med {median_spread:.2f}p p95 {p95_spread:.2f}p "
            f"max {max_spread:.2f}p >= limit {MAX_SPREAD_PIPS:.2f}p "
            f"(high_samples={high_count}/{len(values)})"
        )

    if triggered and not DISABLE_SPREAD_GUARD:
        _blocked_until = max(_blocked_until, now) + COOLDOWN_SECONDS
        _blocked_reason = reason
    elif _blocked_until > now and len(values) >= RELEASE_SAMPLES:
        recent = [val for _, val in list(_history)[-RELEASE_SAMPLES:]]
        if recent and max(recent) <= RELEASE_SPREAD_PIPS:
            _blocked_until = now
            _blocked_reason = ""
    elif not triggered and _blocked_until <= now:
        _blocked_reason = ""

    blocked_now = _blocked_until > now
    if blocked_now and not _last_logged_blocked:
        logging.warning(
            "[SPREAD] Guard activated (med=%.2fp p95=%.2fp max=%.2fp avg=%.2fp samples=%d reason=%s)",
            median_spread,
            p95_spread,
            max_spread,
            avg_spread,
            len(values),
            _blocked_reason or "threshold exceeded",
        )
        _last_logged_blocked = True
    elif not blocked_now and _last_logged_blocked:
        logging.info(
            "[SPREAD] Guard cleared (med=%.2fp p95=%.2fp max=%.2fp avg=%.2fp samples=%d)",
            median_spread,
            p95_spread,
            max_spread,
            avg_spread,
            len(values),
        )
        _last_logged_blocked = False


def get_state() -> Optional[dict]:
    """
    最新スプレッドと統計情報を返す。Tick が未取得の場合は None。
    """
    global _stale_since
    if _snapshot is None:
        return None

    now = time.monotonic()
    age_ms = int(max(0.0, (now - _snapshot.monotonic_ts) * 1000))

    values = [val for _, val in _history] or [_snapshot.spread_pips]
    max_spread = max(values)
    min_spread = min(values)
    avg_spread = sum(values) / len(values)
    median_spread = _percentile(values, 50.0)
    p95_spread = _percentile(values, 95.0)
    high_count = sum(1 for val in values if val >= MAX_SPREAD_PIPS)
    stale = age_ms > MAX_AGE_MS
    stale_since = _stale_since
    if stale:
        if stale_since is None:
            stale_since = now
    else:
        stale_since = None

    _stale_since = stale_since
    stale_for = max(0.0, now - stale_since) if stale_since is not None else 0.0

    baseline_values = [val for _, val in _baseline_history]
    baseline_ready = len(baseline_values) >= BASELINE_MIN_SAMPLES
    baseline_avg = (
        sum(baseline_values) / len(baseline_values) if baseline_ready else None
    )
    baseline_p50 = _percentile(baseline_values, 50.0) if baseline_ready else None
    baseline_p95 = _percentile(baseline_values, 95.0) if baseline_ready else None

    return {
        "bid": _snapshot.bid,
        "ask": _snapshot.ask,
        "spread_pips": _snapshot.spread_pips,
        "avg_pips": avg_spread,
        "max_pips": max_spread,
        "min_pips": min_spread,
        "median_pips": median_spread,
        "p95_pips": p95_spread,
        "samples": len(values),
        "age_ms": age_ms,
        "limit_pips": MAX_SPREAD_PIPS,
        "release_pips": RELEASE_SPREAD_PIPS,
        "max_age_ms": MAX_AGE_MS,
        "stale": stale,
        "stale_for_sec": round(stale_for, 3),
        "stale_grace_sec": STALE_GRACE_SECONDS,
        "high_samples": high_count,
        "min_high_samples": MIN_HIGH_SAMPLES,
        "window_seconds": WINDOW_SECONDS,
        "baseline_window_seconds": BASELINE_WINDOW_SECONDS,
        "baseline_ready": baseline_ready,
        "baseline_samples": len(baseline_values),
        "baseline_avg_pips": baseline_avg,
        "baseline_p50_pips": baseline_p50,
        "baseline_p95_pips": baseline_p95,
    }


def is_blocked() -> Tuple[bool, int, Optional[dict], str]:
    """
    スプレッド拡大によるブロック状態を返す。
    戻り値: (blocked, remain_seconds, state, reason)
    """
    now = time.monotonic()
    remain = int(max(0.0, _blocked_until - now))
    state = get_state()
    return remain > 0, remain, state, _blocked_reason
