"""DOJO lab bot: one parameterized worker for the declared experiment grid.

Config via env var DOJO_BOT_CONFIG (JSON):
  signal        "burst" | "pullback_limit" | "range_fade_limit" |
                "session_open_range_break" | "weekend_gap_recovery"
  tp_pips       float
  sl_pips       float or null  (null = SL-free, the operator's philosophy)
  ceiling_min   int    hard exit after N minutes (the cage; always on)
  max_concurrent int
  per_pos_lev   float  NAV-proportional exposure per position
  atr_floor_pips float  minimum ATR to trade (dead-market guard)
  pull_atr      float  (pullback_limit) limit distance in ATRs below/above close
  fade_atr      float  (range_fade_limit) band distance in ATRs
  eff_max       float  (range_fade_limit) only fade when 6h efficiency below this
  session_buffer_atr float  (session_open_range_break) breakout confirmation
  session_tp_range float  (session_open_range_break) TP in opening-range widths
  session_sl_range float  (session_open_range_break) SL in opening-range widths
  weekend_gap_atr float  (weekend_gap_recovery) minimum gap in Friday ATRs
  weekend_sl_gap float  (weekend_gap_recovery) stop in original gap widths
  weekend_wait_bars int  completed Sunday M1 bars required before entry
  weekend_spread_fraction float maximum spread / remaining gap
  exit_policy   "FIXED" | "BREAKEVEN" | "ATR_TRAILING"
  be_trigger_atr float  profit required before breakeven activation
  be_offset_pips float  profit locked by the breakeven stop
  trail_trigger_atr float profit required before ATR trailing activation
  trail_distance_atr float distance from completed-bar peak/trough

All indicators are incremental (no pandas): 24h trend via a 1441-bar
ring, Wilder ATR(14), 6h efficiency ratio.  Every configured pair has an
isolated state machine under one shared owner-level concurrency cap.
"""

from __future__ import annotations

import json
import math
import os
from collections import deque
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from quant_rabbit.dojo_bot_catalog import AUTHORITY_INVARIANTS, validate_bot_config
from quant_rabbit.dojo_lab_provenance import (
    OwnedBrokerView,
    canonical_strategy_owner_id,
    owner_concurrency_caps_from_config,
)
from quant_rabbit.virtual_broker import VirtualBroker, VirtualBrokerError


_LONDON = ZoneInfo("Europe/London")
_NEW_YORK = ZoneInfo("America/New_York")
_UTC = timezone.utc
_SESSION_RANGE_END_MINUTE = 8 * 60
_SESSION_ENTRY_END_MINUTE = 11 * 60
_WEEKEND_WAIT_COMPLETED_BARS = 15
_WEEKEND_BOUNDARY_BARS = 2
_DAILY_EDGE_TOLERANCE_MINUTES = 5
_DAILY_MIN_OBSERVED_MINUTES = 23 * 60
_DAILY_MAX_OBSERVED_GAP_MINUTES = 5


def _pip(pair: str) -> float:
    return 0.01 if pair.endswith("JPY") else 0.0001


class _PairState:
    def __init__(self):
        self.day: str | None = None
        self.prev_day_high: float | None = None
        self.prev_day_low: float | None = None
        self.today_high: float | None = None
        self.today_low: float | None = None
        self.peak: float | None = None  # trailing anchor
        self.trough: float | None = None
        self.closes20: deque[float] = deque(maxlen=20)
        self.width_hist: deque[float] = deque(maxlen=720)  # 12h of 20-bar widths
        self.closes: deque[float] = deque(maxlen=1441)
        self.diffs_6h: deque[float] = deque(maxlen=360)
        self.highs3: deque[float] = deque(maxlen=3)
        self.lows3: deque[float] = deque(maxlen=3)
        self.atr: float | None = None
        self.prev_close: float | None = None
        self.daily_true_ranges: deque[float] = deque(maxlen=14)
        self.daily_atr: float | None = None
        self.daily_completed_count = 0
        self.daily_previous_close: float | None = None
        self.daily_accepted_dates: deque[str] = deque(maxlen=14)
        self.daily_weekday: int | None = None
        self.daily_first_minute: int | None = None
        self.daily_last_minute: int | None = None
        self.daily_observed_count = 0
        self.daily_max_gap_minutes = 0
        self.daily_coverage_invalid = False
        self.my_trades: dict[str, float] = {}
        self.my_orders: list[str] = []
        self.session_day: str | None = None
        self.session_range_high: float | None = None
        self.session_range_low: float | None = None
        self.session_range_count = 0
        self.session_range_last_epoch: int | None = None
        self.session_range_last_minute: int | None = None
        self.session_range_contiguous = False
        self.session_attempted_day: str | None = None
        self.last_bar_epoch: int | None = None
        self.last_bar_weekday: int | None = None
        self.last_bar_close: float | None = None
        self.recent_bar_epochs: deque[int] = deque(maxlen=_WEEKEND_BOUNDARY_BARS)
        self.weekend_friday_close: float | None = None
        self.weekend_sunday_open: float | None = None
        self.weekend_reference_atr: float | None = None
        self.weekend_bar_count = 0
        self.weekend_last_epoch: int | None = None
        self.weekend_valid = False
        self.weekend_evaluated = True
        self.trade_entry_atr: dict[str, float] = {}
        self.trade_overlay_extreme: dict[str, float] = {}
        self.trade_breakeven_done: set[str] = set()


class Bot:
    def __init__(self, broker: VirtualBroker, cfg: dict | None = None):
        cfg = dict(cfg or json.loads(os.environ["DOJO_BOT_CONFIG"]))
        if set(cfg) & set(AUTHORITY_INVARIANTS):
            owner_override = cfg.pop("strategy_owner_id", None)
            cfg = validate_bot_config(cfg)
            if owner_override is not None:
                cfg["strategy_owner_id"] = owner_override
        self.owner_id = str(
            cfg.get("strategy_owner_id")
            or canonical_strategy_owner_id(cfg, namespace="lab")
        )
        self.pairs = cfg.get("pairs", ["USD_JPY"])
        self.signal = cfg["signal"]
        self.tp_pips = float(cfg.get("tp_pips", 0) or 0)
        self.tp_atr = cfg.get("tp_atr")  # scale-free take: TP = tp_atr x ATR
        self.sl_pips = cfg.get("sl_pips")
        self.ceiling_s = int(cfg["ceiling_min"]) * 60
        self.max_concurrent, self.global_max = owner_concurrency_caps_from_config(cfg)
        self.per_pos_lev = float(cfg.get("per_pos_lev", 4.3))
        self.atr_floor = float(cfg.get("atr_floor_pips", 1.0))
        self.pull_atr = float(cfg.get("pull_atr", 0.6))
        self.fade_atr = float(cfg.get("fade_atr", 1.2))
        self.eff_max = float(cfg.get("eff_max", 0.2))
        self.session_buffer_atr = float(cfg.get("session_buffer_atr", 0.2))
        self.session_tp_range = float(cfg.get("session_tp_range", 1.5))
        self.session_sl_range = float(cfg.get("session_sl_range", 0.75))
        self.weekend_gap_atr = float(cfg.get("weekend_gap_atr", 4.0))
        self.weekend_sl_gap = float(cfg.get("weekend_sl_gap", 1.0))
        raw_weekend_wait_bars = cfg.get(
            "weekend_wait_bars", _WEEKEND_WAIT_COMPLETED_BARS
        )
        if isinstance(raw_weekend_wait_bars, bool):
            raise ValueError("weekend_wait_bars must be a positive integer")
        self.weekend_wait_bars = int(raw_weekend_wait_bars)
        if (
            isinstance(raw_weekend_wait_bars, float)
            and not raw_weekend_wait_bars.is_integer()
        ):
            raise ValueError("weekend_wait_bars must be a positive integer")
        self.weekend_spread_fraction = float(cfg.get("weekend_spread_fraction", 0.15))
        self.exit_policy = str(cfg.get("exit_policy", "FIXED")).upper()
        self.be_trigger_atr = float(cfg.get("be_trigger_atr", 1.0))
        self.be_offset_pips = float(cfg.get("be_offset_pips", 0.0))
        self.trail_trigger_atr = float(
            cfg.get("trail_trigger_atr", self.be_trigger_atr)
        )
        self.trail_distance_atr = float(cfg.get("trail_distance_atr", 2.0))
        if self.exit_policy not in {"FIXED", "BREAKEVEN", "ATR_TRAILING"}:
            raise ValueError(f"unsupported exit_policy: {self.exit_policy}")
        bounded_values = {
            "session_buffer_atr": self.session_buffer_atr,
            "session_tp_range": self.session_tp_range,
            "session_sl_range": self.session_sl_range,
            "weekend_gap_atr": self.weekend_gap_atr,
            "weekend_sl_gap": self.weekend_sl_gap,
            "weekend_spread_fraction": self.weekend_spread_fraction,
            "be_trigger_atr": self.be_trigger_atr,
            "be_offset_pips": self.be_offset_pips,
            "trail_trigger_atr": self.trail_trigger_atr,
            "trail_distance_atr": self.trail_distance_atr,
        }
        if not all(math.isfinite(value) for value in bounded_values.values()):
            raise ValueError("new-family and exit-overlay parameters must be finite")
        if (
            self.session_buffer_atr < 0
            or self.session_tp_range <= 0
            or self.session_sl_range <= 0
            or self.weekend_gap_atr <= 0
            or self.weekend_sl_gap <= 0
        ):
            raise ValueError("new-family parameters are outside their safe range")
        if self.be_trigger_atr < 0 or self.be_offset_pips < 0:
            raise ValueError("breakeven parameters must be non-negative")
        if self.trail_trigger_atr < 0 or self.trail_distance_atr <= 0:
            raise ValueError("ATR trailing parameters are outside their safe range")
        if self.weekend_wait_bars <= 0:
            raise ValueError("weekend_wait_bars must be positive")
        if not 0 < self.weekend_spread_fraction <= 1:
            raise ValueError("weekend_spread_fraction must be in (0, 1]")
        self.broker = OwnedBrokerView(
            broker,
            self.owner_id,
            max_concurrent_per_pair=self.max_concurrent,
            global_max_concurrent=self.global_max,
        )
        self.state: dict[str, _PairState] = {p: _PairState() for p in self.pairs}
        self._owner: dict[str, str] = {}  # owned trade_id -> pair (local cache)

    # ---- incremental indicators -----------------------------------------
    @staticmethod
    def _daily_coverage_is_complete(st: "_PairState") -> bool:
        """Accept a UTC D1 only when its M1 coverage proves a usable full day.

        Real quote archives occasionally omit isolated minutes, so exact 1,440-bar
        equality is too brittle.  The edge, density, and maximum-gap checks still
        reject first/last partial days, FX-short Fridays, holidays, and material
        intraday holes.
        """

        return (
            not st.daily_coverage_invalid
            and st.daily_weekday is not None
            # Friday is an intentionally shortened FX session at NY 17:00;
            # it is never a complete UTC D1 observation.
            and st.daily_weekday < 4
            and st.daily_first_minute is not None
            and st.daily_first_minute <= _DAILY_EDGE_TOLERANCE_MINUTES
            and st.daily_last_minute is not None
            and st.daily_last_minute >= 24 * 60 - 1 - _DAILY_EDGE_TOLERANCE_MINUTES
            and st.daily_observed_count >= _DAILY_MIN_OBSERVED_MINUTES
            and st.daily_max_gap_minutes <= _DAILY_MAX_OBSERVED_GAP_MINUTES
        )

    @staticmethod
    def _start_daily_coverage(st: "_PairState", weekday: int, minute: int) -> None:
        st.daily_weekday = weekday
        st.daily_first_minute = minute
        st.daily_last_minute = minute
        st.daily_observed_count = 1
        st.daily_max_gap_minutes = 0
        st.daily_coverage_invalid = False

    @staticmethod
    def _observe_daily_minute(st: "_PairState", minute: int) -> None:
        prior_minute = st.daily_last_minute
        if prior_minute is None:
            Bot._start_daily_coverage(st, st.daily_weekday or 0, minute)
            return
        if minute <= prior_minute:
            # Duplicate/out-of-order bars can otherwise inflate density and make
            # a damaged day look complete.
            st.daily_coverage_invalid = True
            return
        st.daily_observed_count += 1
        st.daily_max_gap_minutes = max(st.daily_max_gap_minutes, minute - prior_minute)
        st.daily_last_minute = minute

    def _update(self, st: "_PairState", bar: dict) -> None:
        mid_c = (bar["bid_c"] + bar["ask_c"]) / 2
        mid_h = (bar["bid_h"] + bar["ask_h"]) / 2
        mid_l = (bar["bid_l"] + bar["ask_l"]) / 2
        previous_bar_close = st.prev_close
        if previous_bar_close is not None:
            tr = max(
                mid_h - mid_l,
                abs(mid_h - previous_bar_close),
                abs(mid_l - previous_bar_close),
            )
            st.atr = tr if st.atr is None else st.atr + (tr - st.atr) / 14.0
            st.diffs_6h.append(abs(mid_c - previous_bar_close))
        import datetime as _dt

        utc_bar = _dt.datetime.fromtimestamp(bar["epoch"], _dt.timezone.utc)
        day = utc_bar.date().isoformat()
        minute = utc_bar.hour * 60 + utc_bar.minute
        if st.day != day:
            if (
                st.day is not None
                and st.today_high is not None
                and st.today_low is not None
                and previous_bar_close is not None
                and self._daily_coverage_is_complete(st)
            ):
                daily_tr = max(
                    st.today_high - st.today_low,
                    (
                        abs(st.today_high - st.daily_previous_close)
                        if st.daily_previous_close is not None
                        else 0.0
                    ),
                    (
                        abs(st.today_low - st.daily_previous_close)
                        if st.daily_previous_close is not None
                        else 0.0
                    ),
                )
                st.daily_true_ranges.append(daily_tr)
                st.daily_accepted_dates.append(st.day)
                st.daily_completed_count += 1
                if st.daily_atr is None and len(st.daily_true_ranges) == 14:
                    st.daily_atr = sum(st.daily_true_ranges) / 14.0
                elif st.daily_atr is not None:
                    st.daily_atr += (daily_tr - st.daily_atr) / 14.0
                st.daily_previous_close = previous_bar_close
            st.prev_day_high, st.prev_day_low = st.today_high, st.today_low
            st.today_high, st.today_low = mid_h, mid_l
            st.day = day
            self._start_daily_coverage(st, utc_bar.weekday(), minute)
        else:
            st.today_high = max(st.today_high or mid_h, mid_h)
            st.today_low = min(st.today_low or mid_l, mid_l)
            self._observe_daily_minute(st, minute)
        st.prev_close = mid_c
        st.closes.append(mid_c)
        st.closes20.append(mid_c)
        if len(st.closes20) == 20:
            st.width_hist.append(max(st.closes20) - min(st.closes20))
        st.highs3.append(mid_h)
        st.lows3.append(mid_l)

    @staticmethod
    def _trend(st: "_PairState") -> str | None:
        if len(st.closes) < 1441:
            return None
        return "LONG" if st.closes[-1] > st.closes[0] else "SHORT"

    @staticmethod
    def _efficiency_6h(st: "_PairState") -> float | None:
        if len(st.diffs_6h) < 360 or len(st.closes) < 361:
            return None
        path = sum(st.diffs_6h)
        if path <= 0:
            return None
        return abs(st.closes[-1] - st.closes[-361]) / path

    @staticmethod
    def _mid(bar: dict, field: str) -> float:
        return (float(bar[f"bid_{field}"]) + float(bar[f"ask_{field}"])) / 2

    def _update_session_range(self, st: "_PairState", bar: dict) -> tuple[str, int]:
        """Consume one completed bar into the London-local opening range."""

        bar_epoch = int(bar["epoch"])
        local = datetime.fromtimestamp(bar_epoch, _UTC).astimezone(_LONDON)
        local_day = local.date().isoformat()
        local_minute = local.hour * 60 + local.minute
        if st.session_day != local_day:
            st.session_day = local_day
            st.session_range_high = None
            st.session_range_low = None
            st.session_range_count = 0
            st.session_range_last_epoch = None
            st.session_range_last_minute = None
            st.session_range_contiguous = False
            st.session_attempted_day = None

        if local_minute < _SESSION_RANGE_END_MINUTE:
            mid_h = self._mid(bar, "h")
            mid_l = self._mid(bar, "l")
            if st.session_range_count == 0:
                st.session_range_contiguous = local_minute == 0
            elif st.session_range_last_epoch is None or (
                bar_epoch != st.session_range_last_epoch + 60
            ):
                st.session_range_contiguous = False
            st.session_range_high = (
                mid_h
                if st.session_range_high is None
                else max(st.session_range_high, mid_h)
            )
            st.session_range_low = (
                mid_l
                if st.session_range_low is None
                else min(st.session_range_low, mid_l)
            )
            st.session_range_count += 1
            st.session_range_last_epoch = bar_epoch
            st.session_range_last_minute = local_minute
        return local_day, local_minute

    @staticmethod
    def _session_range_is_complete(st: "_PairState") -> bool:
        return (
            st.session_range_contiguous
            and st.session_range_count == _SESSION_RANGE_END_MINUTE
            and st.session_range_last_minute == _SESSION_RANGE_END_MINUTE - 1
            and st.session_range_high is not None
            and st.session_range_low is not None
            and st.session_range_high > st.session_range_low
        )

    def _update_weekend_gap(
        self,
        st: "_PairState",
        bar: dict,
        *,
        friday_daily_atr: float | None,
        prior_epoch: int | None,
        prior_weekday: int | None,
        prior_close: float | None,
    ) -> None:
        """Track the FX weekend only at the DST-aware New York 17:00 boundary."""

        bar_epoch = int(bar["epoch"])
        utc_bar = datetime.fromtimestamp(bar_epoch, _UTC)
        new_york_bar = utc_bar.astimezone(_NEW_YORK)
        is_exact_weekend_open = (
            new_york_bar.weekday() == 6
            and new_york_bar.hour == 17
            and new_york_bar.minute == 0
            and new_york_bar.second == 0
        )
        if is_exact_weekend_open:
            st.weekend_friday_close = None
            st.weekend_sunday_open = None
            st.weekend_reference_atr = None
            st.weekend_bar_count = 0
            st.weekend_last_epoch = None
            st.weekend_valid = False
            st.weekend_evaluated = True

            friday_date = new_york_bar.date() - timedelta(days=2)
            friday_close_boundary = datetime(
                friday_date.year,
                friday_date.month,
                friday_date.day,
                17,
                tzinfo=_NEW_YORK,
            )
            expected_last_epoch = int(friday_close_boundary.timestamp()) - 60
            expected_preclose_epochs = [
                expected_last_epoch - 60 * offset
                for offset in reversed(range(_WEEKEND_BOUNDARY_BARS))
            ]
            expected_daily_dates: list[str] = []
            daily_cursor = friday_date - timedelta(days=1)
            while len(expected_daily_dates) < 14:
                if daily_cursor.weekday() < 4:
                    expected_daily_dates.append(daily_cursor.isoformat())
                daily_cursor -= timedelta(days=1)
            expected_daily_dates.reverse()
            if (
                list(st.recent_bar_epochs) == expected_preclose_epochs
                and list(st.daily_accepted_dates) == expected_daily_dates
                and prior_weekday == 4
                and prior_epoch == expected_last_epoch
                and prior_close is not None
                and friday_daily_atr is not None
                and st.daily_completed_count >= 14
            ):
                st.weekend_friday_close = prior_close
                st.weekend_sunday_open = self._mid(bar, "o")
                st.weekend_reference_atr = friday_daily_atr
                st.weekend_bar_count = 1
                st.weekend_last_epoch = bar_epoch
                st.weekend_valid = True
                st.weekend_evaluated = False
            return

        if st.weekend_evaluated or st.weekend_bar_count <= 0:
            return
        if st.weekend_last_epoch is None or bar_epoch != st.weekend_last_epoch + 60:
            st.weekend_valid = False
            st.weekend_evaluated = True
            return
        st.weekend_bar_count += 1
        st.weekend_last_epoch = bar_epoch

    def _register_trade(
        self,
        st: "_PairState",
        trade_id: str,
        pair: str,
        opened_epoch: float,
        entry_atr: float | None,
    ) -> None:
        st.my_trades[trade_id] = opened_epoch
        self._owner[trade_id] = pair
        if entry_atr is not None and entry_atr > 0:
            st.trade_entry_atr[trade_id] = entry_atr
        position = self.broker.position(trade_id)
        if position is not None:
            st.trade_overlay_extreme[trade_id] = position.entry_price

    @staticmethod
    def _position_opened_epoch(opened_ts: str) -> float | None:
        """Parse the broker-authored fill clock without trusting local time.

        Replay quotes append an intrabar phase (``#H``/``#L``/``#C``) to an
        aware ISO-8601 timestamp.  A missing timezone or non-finite epoch
        cannot safely start the hard-hold cage.
        """

        if not isinstance(opened_ts, str) or not opened_ts:
            return None
        try:
            opened_at = datetime.fromisoformat(opened_ts.split("#", 1)[0])
            if opened_at.tzinfo is None or opened_at.utcoffset() is None:
                return None
            opened_epoch = opened_at.timestamp()
        except (OverflowError, TypeError, ValueError):
            return None
        if not math.isfinite(opened_epoch):
            return None
        return opened_epoch

    def _forget_trade(self, st: "_PairState", trade_id: str) -> None:
        st.my_trades.pop(trade_id, None)
        self._owner.pop(trade_id, None)
        st.trade_entry_atr.pop(trade_id, None)
        st.trade_overlay_extreme.pop(trade_id, None)
        st.trade_breakeven_done.discard(trade_id)

    def _close_trade_and_cleanup(self, st: "_PairState", trade_id: str) -> bool:
        """Close an owned trade without losing local ownership on a failed close."""

        try:
            self.broker.close_trade(trade_id)
        except VirtualBrokerError:
            if self.broker.position(trade_id) is None:
                self._forget_trade(st, trade_id)
            return False
        self._forget_trade(st, trade_id)
        return True

    @staticmethod
    def _is_tighter_stop(side: str, candidate: float, existing: float | None) -> bool:
        if existing is None:
            return True
        return candidate > existing if side == "LONG" else candidate < existing

    def _apply_exit_overlay(
        self,
        pair: str,
        st: "_PairState",
        bar: dict,
        *,
        newly_promoted: set[str],
    ) -> None:
        """Tighten owned stops from completed bars; never widen an existing SL."""

        if self.exit_policy == "FIXED" or st.atr is None:
            return
        pip = _pip(pair)
        digits = 3 if pair.endswith("JPY") else 5
        for trade_id in list(st.my_trades):
            if trade_id in newly_promoted:
                # A resting order may have filled inside this bar.  Its earlier
                # high/low predates the fill, so the whole bar is ineligible.
                continue
            position = self.broker.position(trade_id)
            entry_atr = st.trade_entry_atr.get(trade_id)
            if position is None or entry_atr is None:
                continue
            if position.side == "LONG":
                mark = float(bar["bid_c"])
                favorable_extreme = float(bar["bid_h"])
                profit = mark - position.entry_price
                extreme = max(
                    st.trade_overlay_extreme.get(trade_id, position.entry_price),
                    favorable_extreme,
                )
            else:
                mark = float(bar["ask_c"])
                favorable_extreme = float(bar["ask_l"])
                profit = position.entry_price - mark
                extreme = min(
                    st.trade_overlay_extreme.get(trade_id, position.entry_price),
                    favorable_extreme,
                )
            st.trade_overlay_extreme[trade_id] = extreme

            candidate: float | None = None
            breakeven_triggered = False
            if self.exit_policy == "BREAKEVEN":
                if (
                    trade_id not in st.trade_breakeven_done
                    and profit >= self.be_trigger_atr * entry_atr
                ):
                    breakeven_triggered = True
                    candidate = (
                        position.entry_price + self.be_offset_pips * pip
                        if position.side == "LONG"
                        else position.entry_price - self.be_offset_pips * pip
                    )
            elif (
                self.exit_policy == "ATR_TRAILING"
                and (
                    (extreme - position.entry_price)
                    if position.side == "LONG"
                    else (position.entry_price - extreme)
                )
                >= self.trail_trigger_atr * entry_atr
            ):
                candidate = (
                    extreme - self.trail_distance_atr * st.atr
                    if position.side == "LONG"
                    else extreme + self.trail_distance_atr * st.atr
                )

            if candidate is None:
                continue
            rounded_candidate = round(candidate, digits)
            if not self._is_tighter_stop(
                position.side, rounded_candidate, position.sl_price
            ):
                if breakeven_triggered:
                    st.trade_breakeven_done.add(trade_id)
                continue
            executable_mark = position.current_price
            if executable_mark is None:
                # A stop cannot be installed safely without proving where this
                # owned position can be exited now.
                continue
            if (position.side == "LONG" and rounded_candidate >= executable_mark) or (
                position.side == "SHORT" and rounded_candidate <= executable_mark
            ):
                self._close_trade_and_cleanup(st, trade_id)
                continue
            try:
                self.broker.set_exit(
                    trade_id,
                    tp_price=position.tp_price,
                    sl_price=rounded_candidate,
                )
                if breakeven_triggered:
                    st.trade_breakeven_done.add(trade_id)
            except VirtualBrokerError:
                pass

    # ---- lifecycle -------------------------------------------------------
    def on_bar_closed(self, pair: str, bar: dict, epoch: int) -> None:
        st = self.state.get(pair)
        if st is None:
            return
        pip = _pip(pair)
        prior_h3 = max(st.highs3) if len(st.highs3) == 3 else None
        prior_l3 = min(st.lows3) if len(st.lows3) == 3 else None
        prior_atr = st.atr
        prior_epoch = st.last_bar_epoch
        prior_weekday = st.last_bar_weekday
        prior_close = st.last_bar_close
        self._update(st, bar)
        local_day, local_minute = self._update_session_range(st, bar)
        self._update_weekend_gap(
            st,
            bar,
            friday_daily_atr=st.daily_atr,
            prior_epoch=prior_epoch,
            prior_weekday=prior_weekday,
            prior_close=prior_close,
        )
        bar_epoch = int(bar["epoch"])
        st.last_bar_epoch = bar_epoch
        st.last_bar_weekday = datetime.fromtimestamp(bar_epoch, _UTC).weekday()
        st.last_bar_close = self._mid(bar, "c")
        st.recent_bar_epochs.append(bar_epoch)

        # Promote only fills descended from this hand's owned orders.  An
        # external or sibling-hand position is NO_TOUCH even on the same pair.
        newly_promoted: set[str] = set()
        for trade_id in self.broker.active_trade_ids(pair=pair):
            if trade_id not in self._owner:
                position = self.broker.position(trade_id)
                if position is None:
                    continue
                opened_epoch = self._position_opened_epoch(position.opened_ts)
                if opened_epoch is None:
                    # Unknown age must not reset or extend a mandatory cage.
                    # Register at the expiry boundary so the normal owned-close
                    # path attempts an immediate fail-closed release and keeps
                    # retry ownership intact if that close itself fails.
                    opened_epoch = float(epoch - self.ceiling_s)
                self._register_trade(st, trade_id, pair, opened_epoch, prior_atr)
                newly_promoted.add(trade_id)
        for trade_id in list(st.my_trades):
            if trade_id not in self.broker.active_trade_ids(pair=pair):
                self._forget_trade(st, trade_id)
            elif epoch - st.my_trades[trade_id] >= self.ceiling_s:
                self._close_trade_and_cleanup(st, trade_id)

        self._apply_exit_overlay(pair, st, bar, newly_promoted=newly_promoted)

        trend = self._trend(st)
        if st.atr is None:
            return
        atr_pips = st.atr / pip
        if atr_pips < self.atr_floor:
            return
        total_open = sum(len(s.my_trades) for s in self.state.values())
        if total_open >= self.global_max:
            open_n = self.max_concurrent  # treat as full
        else:
            open_n = len(st.my_trades)
        mid_c = (bar["bid_c"] + bar["ask_c"]) / 2
        try:
            executable_quote = self.broker.executable_quote(pair)
        except VirtualBrokerError:
            return

        def units_for(price: float) -> float:
            try:
                equity = self.broker.account()["equity_jpy"]
                jpy_per_unit = price * self.broker.jpy_per_quote_unit(pair)
            except VirtualBrokerError:
                return 0.0
            if jpy_per_unit <= 0:
                return 0.0
            return max(equity, 0.0) * self.per_pos_lev / jpy_per_unit

        def market_units_for(side: str) -> float:
            try:
                execution_price = self.broker.executable_market_entry_price(pair, side)
            except VirtualBrokerError:
                return 0.0
            return units_for(execution_price)

        digits = 3 if pair.endswith("JPY") else 5
        tp_pips = (float(self.tp_atr) * atr_pips) if self.tp_atr else self.tp_pips
        spread_pips = (executable_quote.ask - executable_quote.bid) / pip

        def cost_allows(target_pips: float) -> bool:
            return target_pips > 0 and spread_pips <= target_pips * 0.35

        if self.signal == "session_open_range_break":
            if not (
                _SESSION_RANGE_END_MINUTE <= local_minute < _SESSION_ENTRY_END_MINUTE
            ):
                return
            if (
                st.session_attempted_day == local_day
                or not self._session_range_is_complete(st)
            ):
                return
            range_high = float(st.session_range_high)
            range_low = float(st.session_range_low)
            range_width = range_high - range_low
            buffer = self.session_buffer_atr * st.atr
            if mid_c > range_high + buffer:
                side = "LONG"
            elif mid_c < range_low - buffer:
                side = "SHORT"
            else:
                return
            st.session_attempted_day = local_day
            dynamic_tp_pips = self.session_tp_range * range_width / pip
            dynamic_sl_pips = self.session_sl_range * range_width / pip
            if (
                open_n >= self.max_concurrent
                or not cost_allows(dynamic_tp_pips)
                or dynamic_sl_pips <= 0
            ):
                return
            units = market_units_for(side)
            if units <= 0:
                return
            try:
                tid = self.broker.market_order(
                    pair,
                    side,
                    units,
                    tp_pips=dynamic_tp_pips,
                    sl_pips=dynamic_sl_pips,
                )
                self._register_trade(st, tid, pair, epoch, st.atr)
            except VirtualBrokerError:
                pass
            return

        if self.signal == "weekend_gap_recovery":
            required_weekend_bars = max(self.weekend_wait_bars, _WEEKEND_BOUNDARY_BARS)
            if st.weekend_evaluated or st.weekend_bar_count < required_weekend_bars:
                return
            st.weekend_evaluated = True
            if not st.weekend_valid:
                return
            friday_close = st.weekend_friday_close
            sunday_open = st.weekend_sunday_open
            reference_atr = st.weekend_reference_atr
            if friday_close is None or sunday_open is None or reference_atr is None:
                return
            original_gap = sunday_open - friday_close
            if abs(original_gap) < self.weekend_gap_atr * reference_atr:
                return
            if original_gap < 0 and executable_quote.bid < friday_close:
                side = "LONG"
                target_pips = (friday_close - executable_quote.bid) / pip
            elif original_gap > 0 and executable_quote.ask > friday_close:
                side = "SHORT"
                target_pips = (executable_quote.ask - friday_close) / pip
            else:
                return
            if (
                open_n >= self.max_concurrent
                or target_pips <= 0
                or spread_pips > target_pips * self.weekend_spread_fraction
            ):
                return
            stop_distance = self.weekend_sl_gap * abs(original_gap)
            stop_price = (
                sunday_open - stop_distance
                if side == "LONG"
                else sunday_open + stop_distance
            )
            try:
                execution_price = self.broker.executable_market_entry_price(pair, side)
            except VirtualBrokerError:
                return
            target_crossed = (
                side == "LONG" and executable_quote.bid >= friday_close
            ) or (side == "SHORT" and executable_quote.ask <= friday_close)
            stop_crossed = (side == "LONG" and executable_quote.bid <= stop_price) or (
                side == "SHORT" and executable_quote.ask >= stop_price
            )
            if target_crossed or stop_crossed:
                return
            units = market_units_for(side)
            if units <= 0:
                return
            attached_tp_pips = abs(friday_close - execution_price) / pip
            attached_sl_pips = abs(execution_price - stop_price) / pip
            try:
                tid = self.broker.market_order(
                    pair,
                    side,
                    units,
                    tp_pips=attached_tp_pips,
                    sl_pips=attached_sl_pips,
                )
                position = self.broker.position(tid)
                if position is None:
                    return
                if position.tp_price != round(
                    friday_close, digits
                ) or position.sl_price != round(stop_price, digits):
                    self.broker.close_trade(tid)
                    return
                self._register_trade(st, tid, pair, epoch, st.atr)
            except VirtualBrokerError:
                if "tid" in locals() and self.broker.position(tid) is not None:
                    try:
                        self.broker.close_trade(tid)
                    except VirtualBrokerError:
                        pass
            return

        if trend is None or not cost_allows(tp_pips):
            return  # habitat gate: cost must stay well under the take

        if self.signal == "burst":
            if open_n >= self.max_concurrent or prior_h3 is None:
                return
            triggered = (trend == "LONG" and mid_c > prior_h3) or (
                trend == "SHORT" and mid_c < prior_l3
            )
            if not triggered:
                return
            units = market_units_for(trend)
            if units <= 0:
                return
            try:
                tid = self.broker.market_order(
                    pair, trend, units, tp_pips=tp_pips, sl_pips=self.sl_pips
                )
                self._register_trade(st, tid, pair, epoch, st.atr)
            except VirtualBrokerError:
                pass

        elif self.signal == "pullback_limit":
            for oid in st.my_orders:
                try:
                    self.broker.cancel_order(oid)
                except VirtualBrokerError:
                    pass
            st.my_orders = []
            if open_n >= self.max_concurrent:
                return
            dist = self.pull_atr * st.atr
            price = mid_c - dist if trend == "LONG" else mid_c + dist
            units = units_for(price)
            if units <= 0:
                return
            try:
                oid = self.broker.limit_order(
                    pair,
                    trend,
                    units,
                    price=round(price, digits),
                    tp_pips=tp_pips,
                    sl_pips=self.sl_pips,
                )
                st.my_orders = [oid]
            except VirtualBrokerError:
                pass

        elif self.signal == "compression_break":
            # SQUEEZE cell: when the 20-bar width is in its lowest quintile
            # of the last 12h, straddle LIMIT stops beyond the box; the
            # first real breakout fills one side, the other is cancelled
            # next bar.  SL-free + ceiling as configured.
            for oid in st.my_orders:
                try:
                    self.broker.cancel_order(oid)
                except VirtualBrokerError:
                    pass
            st.my_orders = []
            if open_n >= self.max_concurrent or len(st.width_hist) < 360:
                return
            width = max(st.closes20) - min(st.closes20)
            rank = sum(1 for w in st.width_hist if w < width) / len(st.width_hist)
            if rank > 0.2:
                return
            box_h = max(st.closes20)
            box_l = min(st.closes20)
            units = units_for(mid_c)
            if units <= 0:
                return
            for side, price in (("LONG", box_h + 2 * pip), ("SHORT", box_l - 2 * pip)):
                try:
                    oid = self.broker.stop_order(
                        pair,
                        side,
                        units,
                        price=round(price, digits),
                        tp_pips=tp_pips,
                        sl_pips=self.sl_pips,
                    )
                    st.my_orders.append(oid)
                except VirtualBrokerError:
                    pass

        elif self.signal == "spike_fade":
            # counter a >2.5-ATR one-bar spike with a passive fade LIMIT at
            # the spike extreme; TP scale-free; SL-free + ceiling.
            for oid in st.my_orders:
                try:
                    self.broker.cancel_order(oid)
                except VirtualBrokerError:
                    pass
            st.my_orders = []
            if open_n >= self.max_concurrent or st.prev_close is None:
                return
            bar_range = (bar["bid_h"] + bar["ask_h"]) / 2 - (
                bar["bid_l"] + bar["ask_l"]
            ) / 2
            if bar_range < 2.5 * st.atr:
                return
            up_spike = mid_c > (bar["bid_o"] + bar["ask_o"]) / 2
            units = units_for(mid_c)
            if units <= 0:
                return
            side = "SHORT" if up_spike else "LONG"
            edge = (
                (bar["bid_h"] + bar["ask_h"]) / 2
                if up_spike
                else (bar["bid_l"] + bar["ask_l"]) / 2
            )
            try:
                oid = self.broker.limit_order(
                    pair,
                    side,
                    units,
                    price=round(edge, digits),
                    tp_pips=tp_pips,
                    sl_pips=self.sl_pips,
                )
                st.my_orders = [oid]
            except VirtualBrokerError:
                pass

        elif self.signal == "prev_day_extreme_fade":
            for oid in st.my_orders:
                try:
                    self.broker.cancel_order(oid)
                except VirtualBrokerError:
                    pass
            st.my_orders = []
            if open_n >= self.max_concurrent or st.prev_day_high is None:
                return
            units = units_for(mid_c)
            if units <= 0:
                return
            for side, level in (("SHORT", st.prev_day_high), ("LONG", st.prev_day_low)):
                if abs(level - mid_c) > 40 * pip:  # only near levels
                    continue
                try:
                    oid = self.broker.limit_order(
                        pair,
                        side,
                        units,
                        price=round(level, digits),
                        tp_pips=tp_pips,
                        sl_pips=self.sl_pips,
                    )
                    st.my_orders.append(oid)
                except VirtualBrokerError:
                    pass

        elif self.signal == "round_number_fade":
            for oid in st.my_orders:
                try:
                    self.broker.cancel_order(oid)
                except VirtualBrokerError:
                    pass
            st.my_orders = []
            if open_n >= self.max_concurrent:
                return
            step = 0.50 if pair.endswith("JPY") else 0.0050
            above = (int(mid_c / step) + 1) * step
            below = int(mid_c / step) * step
            units = units_for(mid_c)
            if units <= 0:
                return
            for side, level in (("SHORT", above), ("LONG", below)):
                if abs(level - mid_c) > 25 * pip or abs(level - mid_c) < 3 * pip:
                    continue
                try:
                    oid = self.broker.limit_order(
                        pair,
                        side,
                        units,
                        price=round(level, digits),
                        tp_pips=tp_pips,
                        sl_pips=self.sl_pips,
                    )
                    st.my_orders.append(oid)
                except VirtualBrokerError:
                    pass

        elif self.signal == "daily_break_pullback":
            for oid in st.my_orders:
                try:
                    self.broker.cancel_order(oid)
                except VirtualBrokerError:
                    pass
            st.my_orders = []
            if open_n >= self.max_concurrent or st.prev_day_high is None:
                return
            units = units_for(mid_c)
            if units <= 0:
                return
            broke_up = (
                st.today_high or mid_c
            ) > st.prev_day_high and mid_c > st.prev_day_high
            broke_dn = (
                st.today_low or mid_c
            ) < st.prev_day_low and mid_c < st.prev_day_low
            if broke_up:
                try:
                    oid = self.broker.limit_order(
                        pair,
                        "LONG",
                        units,
                        price=round(st.prev_day_high, digits),
                        tp_pips=tp_pips,
                        sl_pips=self.sl_pips,
                    )
                    st.my_orders = [oid]
                except VirtualBrokerError:
                    pass
            elif broke_dn:
                try:
                    oid = self.broker.limit_order(
                        pair,
                        "SHORT",
                        units,
                        price=round(st.prev_day_low, digits),
                        tp_pips=tp_pips,
                        sl_pips=self.sl_pips,
                    )
                    st.my_orders = [oid]
                except VirtualBrokerError:
                    pass

        elif self.signal == "mean_revert_24h":
            if open_n >= self.max_concurrent or len(st.closes) < 1441:
                return
            mean = sum(st.closes) / len(st.closes)
            dev = mid_c - mean
            k = self.fade_atr * 8 * st.atr  # deep deviation in M1-ATR units
            try:
                if dev <= -k:
                    units = market_units_for("LONG")
                    if units <= 0:
                        return
                    tid = self.broker.market_order(
                        pair, "LONG", units, tp_pips=tp_pips, sl_pips=self.sl_pips
                    )
                    self._register_trade(st, tid, pair, epoch, st.atr)
                elif dev >= k:
                    units = market_units_for("SHORT")
                    if units <= 0:
                        return
                    tid = self.broker.market_order(
                        pair, "SHORT", units, tp_pips=tp_pips, sl_pips=self.sl_pips
                    )
                    self._register_trade(st, tid, pair, epoch, st.atr)
            except VirtualBrokerError:
                pass

        elif self.signal == "trailing_burst":
            # entry = burst; exit = chandelier trail (2x fade_atr ATR from peak),
            # NO fixed TP — the let-winners-run family.
            for tid in list(st.my_trades):
                pos = self.broker.position(tid)
                if pos is None:
                    continue
                if pos.side == "LONG":
                    st.peak = max(st.peak or mid_c, mid_c)
                    if mid_c <= st.peak - 2 * self.fade_atr * st.atr:
                        try:
                            self.broker.close_trade(tid)
                        except VirtualBrokerError:
                            pass
                else:
                    st.trough = min(st.trough or mid_c, mid_c)
                    if mid_c >= st.trough + 2 * self.fade_atr * st.atr:
                        try:
                            self.broker.close_trade(tid)
                        except VirtualBrokerError:
                            pass
            if open_n >= self.max_concurrent or prior_h3 is None:
                return
            triggered_l = trend == "LONG" and mid_c > prior_h3
            triggered_s = trend == "SHORT" and mid_c < prior_l3
            if not (triggered_l or triggered_s):
                return
            try:
                side = "LONG" if triggered_l else "SHORT"
                units = market_units_for(side)
                if units <= 0:
                    return
                tid = self.broker.market_order(
                    pair, side, units, tp_pips=None, sl_pips=self.sl_pips
                )
                self._register_trade(st, tid, pair, epoch, st.atr)
                st.peak = mid_c
                st.trough = mid_c
            except VirtualBrokerError:
                pass

        elif self.signal == "fade_ladder":
            # range fade + ONE bounded add-on one extra band further (nanpin-lite)
            for oid in st.my_orders:
                try:
                    self.broker.cancel_order(oid)
                except VirtualBrokerError:
                    pass
            st.my_orders = []
            eff = self._efficiency_6h(st)
            if eff is None or eff > self.eff_max:
                return
            dist = self.fade_atr * st.atr
            units = units_for(mid_c)
            if units <= 0:
                return
            layers = [1.0] if open_n == 0 else ([2.2] if open_n == 1 else [])
            for mult in layers:
                for side, price in (
                    ("LONG", mid_c - dist * mult),
                    ("SHORT", mid_c + dist * mult),
                ):
                    try:
                        oid = self.broker.limit_order(
                            pair,
                            side,
                            units,
                            price=round(price, digits),
                            tp_pips=tp_pips,
                            sl_pips=self.sl_pips,
                        )
                        st.my_orders.append(oid)
                    except VirtualBrokerError:
                        pass

        elif self.signal == "range_fade_limit":
            for oid in st.my_orders:
                try:
                    self.broker.cancel_order(oid)
                except VirtualBrokerError:
                    pass
            st.my_orders = []
            eff = self._efficiency_6h(st)
            if eff is None or eff > self.eff_max or open_n >= self.max_concurrent:
                return
            dist = self.fade_atr * st.atr
            units = units_for(mid_c)
            if units <= 0:
                return
            for side, price in (("LONG", mid_c - dist), ("SHORT", mid_c + dist)):
                try:
                    oid = self.broker.limit_order(
                        pair,
                        side,
                        units,
                        price=round(price, digits),
                        tp_pips=tp_pips,
                        sl_pips=self.sl_pips,
                    )
                    st.my_orders.append(oid)
                except VirtualBrokerError:
                    pass
