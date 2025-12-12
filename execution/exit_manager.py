"""
execution.exit_manager
~~~~~~~~~~~~~~~~~~~~~~
注文のクローズ判定を担当。
• 逆方向シグナル or 指標の劣化を検知してクローズ指示を返す
• イベント時のポケット縮退もここでハンドル
"""

from __future__ import annotations

import json
import os
import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import logging

from analysis.chart_story import ChartStorySnapshot

from analysis.ma_projection import MACrossProjection, compute_ma_projection
from analysis.mtf_utils import resample_candles_from_m1
from utils.metrics_logger import log_metric

if TYPE_CHECKING:
    from execution.stage_tracker import StageTracker


def _in_jst_window(now: datetime, start_hour: int, end_hour: int) -> bool:
    """Return True when UTC time falls within the specified JST window."""
    start = start_hour % 24
    end = end_hour % 24
    current = now if now.tzinfo else now.replace(tzinfo=timezone.utc)
    jst = current + timedelta(hours=9)
    hour = jst.hour
    if start <= end:
        return start <= hour < end
    return hour >= start or hour < end


def _session_bucket(now: datetime) -> str:
    """Rough session bucket in UTC to gate aggressiveness."""
    hour = now.hour
    if 7 <= hour < 17:
        return "london"
    if 17 <= hour < 23:
        return "ny"
    return "asia"


def _slope_from_candles(candles: List[dict], window: int = 6) -> float:
    """Return slope in pips over the window; falls back to 0.0."""
    try:
        if len(candles) < window:
            return 0.0
        closes = []
        for cndl in candles[-window:]:
            closes.append(float(cndl.get("close")))
        if len(closes) < window:
            return 0.0
        return (closes[-1] - closes[0]) / 0.01
    except Exception:
        return 0.0


def _candle_high(c: dict) -> Optional[float]:
    return c.get("high") or c.get("h") or c.get("close")


def _candle_low(c: dict) -> Optional[float]:
    return c.get("low") or c.get("l") or c.get("close")


@dataclass
class ExitDecision:
    pocket: str
    units: int
    reason: str
    tag: str
    allow_reentry: bool = False


MANAGED_POCKETS = {"macro", "micro", "scalp", "scalp_fast"}
AGENT_CLIENT_PREFIXES = tuple(
    p for p in os.getenv("AGENT_CLIENT_PREFIXES", "qr-,qs-").split(",") if p
)
if not AGENT_CLIENT_PREFIXES:
    AGENT_CLIENT_PREFIXES = ("qr-",)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "no"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(float(raw))
    except ValueError:
        return default


class ExitManager:
    def __init__(self, confidence_threshold: int = 70):
        self.confidence_threshold = confidence_threshold
        self._macro_signal_threshold = max(confidence_threshold + 10, 80)
        self._macro_trend_adx = 16
        self._macro_loss_buffer = 1.10
        self._macro_ma_gap = 3.0
        # Macro-specific stability controls
        self._macro_min_hold_minutes = 6.0  # hold a bit longer to reduce churn
        self._macro_hysteresis_pips = 1.6   # wider deadband; avoid near-flat exits
        # Pattern-aware retest handling (macro only)
        self._macro_retest_band_base = 1.2  # pips around fast MA treated as retest zone
        self._macro_retest_m5_slope = 0.06  # min M5 slope (pips/bar) aligning with position
        self._macro_retest_m10_slope = 0.04 # min M10 slope (pips/bar)
        self._macro_struct_cushion = 0.22   # ATR fraction for pivot kill-line cushion
        # Scalp-specific stability controls
        self._scalp_loss_guard = 1.15
        self._scalp_take_profit = 2.4
        self._reverse_confirmations = 2
        self._reverse_decay = timedelta(seconds=180)
        self._reverse_hits: Dict[Tuple[str, str], Dict[str, object]] = {}
        self._low_vol_hazard_hits: Dict[Tuple[str, str], int] = {}
        self._range_macro_grace_minutes = 8.0
        # Micro-specific stability controls
        # Guard against premature exits on noisy micro trades: enforce longer holds and wider grace
        self._micro_min_hold_seconds = float(os.getenv("EXIT_MICRO_MIN_HOLD_SEC", "90"))
        self._micro_min_hold_minutes = self._micro_min_hold_seconds / 60.0
        self._micro_loss_grace_pips = float(os.getenv("EXIT_MICRO_GUARD_LOSS_PIPS", "2.5"))
        self._micro_loss_hold_seconds = float(os.getenv("EXIT_MICRO_LOSS_HOLD_SEC", "90"))
        self._micro_profit_hard = float(os.getenv("EXIT_MICRO_PROFIT_TAKE_PIPS", "1.60"))
        self._micro_profit_soft = float(os.getenv("EXIT_MICRO_PROFIT_SOFT_PIPS", "1.00"))
        self._micro_profit_rsi_release_long = float(os.getenv("EXIT_MICRO_PROFIT_RSI_LONG", "53"))
        self._micro_profit_rsi_release_short = float(os.getenv("EXIT_MICRO_PROFIT_RSI_SHORT", "47"))
        self._micro_profit_ema_buffer = float(os.getenv("EXIT_MICRO_PROFIT_EMA_BUFFER", "0.0005"))
        self._micro_profit_slope_min = float(os.getenv("EXIT_MICRO_PROFIT_SLOPE_MIN", "0.05"))
        # Scalp pocket guard rails
        self._scalp_min_hold_seconds = float(os.getenv("EXIT_SCALP_MIN_HOLD_SEC", "45"))
        self._scalp_loss_grace_pips = float(os.getenv("EXIT_SCALP_GUARD_LOSS_PIPS", "2.0"))
        # Fast-cut opt-out: default OFF -> 高速カットを有効。ただしガードを強化。
        self._disable_scalp_fast_cut = str(os.getenv("EXIT_DISABLE_SCALP_FAST_CUT", "0")).strip().lower() in {
            "1",
            "true",
            "yes",
        }
        # TrendMA / volatility-specific garde rails
        self._trendma_partial_fraction = float(os.getenv("EXIT_TRENDMA_PARTIAL_FRACTION", "0.5"))
        self._trendma_partial_profit_cap = float(os.getenv("EXIT_TRENDMA_PARTIAL_PROFIT_CAP", "3.4"))
        self._vol_partial_atr_min = float(os.getenv("EXIT_VOL_PARTIAL_ATR_MIN", "1.5"))
        self._vol_partial_atr_max = float(os.getenv("EXIT_VOL_PARTIAL_ATR_MAX", "2.6"))
        self._vol_partial_fraction = float(os.getenv("EXIT_VOL_PARTIAL_FRACTION", "0.66"))
        self._vol_partial_profit_floor = float(os.getenv("EXIT_VOL_PARTIAL_PROFIT_FLOOR", "2.5"))
        # Overnight/roll spread対策: JST時間帯で自動カットを止めるためのゲート
        self._cut_disable_jst_start = _env_int("EXIT_CUT_DISABLE_JST_START", 7)
        self._cut_disable_jst_end = _env_int("EXIT_CUT_DISABLE_JST_END", 8)
        self._vol_partial_profit_cap = float(os.getenv("EXIT_VOL_PARTIAL_PROFIT_CAP", "3.0"))
        self._vol_ema_release_gap = float(os.getenv("EXIT_VOL_EMA_RELEASE_GAP", "1.0"))
        self._profit_snatch_min = float(os.getenv("EXIT_SNATCH_MIN_PROFIT_PIPS", "0.3"))
        self._profit_snatch_max = float(os.getenv("EXIT_SNATCH_MAX_PROFIT_PIPS", "0.8"))
        self._profit_snatch_hold = float(os.getenv("EXIT_SNATCH_MIN_HOLD_SEC", "70"))
        self._profit_snatch_atr_min = float(os.getenv("EXIT_SNATCH_ATR_MIN", "1.0"))
        self._profit_snatch_vol_min = float(os.getenv("EXIT_SNATCH_VOL5M_MIN", "0.8"))
        self._profit_snatch_jst_start = int(os.getenv("EXIT_SNATCH_JST_START", "0")) % 24
        self._profit_snatch_jst_end = int(os.getenv("EXIT_SNATCH_JST_END", "6")) % 24
        self._loss_guard_atr_trigger = float(os.getenv("EXIT_LOSS_GUARD_ATR_TRIGGER", "2.0"))
        self._loss_guard_vol_trigger = float(os.getenv("EXIT_LOSS_GUARD_VOL_TRIGGER", "1.5"))
        self._loss_guard_compress_ratio = float(os.getenv("EXIT_LOSS_GUARD_COMPRESS_RATIO", "0.7"))
        # MFE ベースのトレール/抑制パラメータ（未定義で落ちないように初期化）
        self._mfe_guard_base_default = float(os.getenv("EXIT_MFE_GUARD_BASE_DEFAULT", "0.8"))
        self._mfe_guard_base = {
            "macro": float(os.getenv("EXIT_MFE_GUARD_BASE_MACRO", "1.2")),
            "micro": float(os.getenv("EXIT_MFE_GUARD_BASE_MICRO", "0.9")),
            "scalp": float(os.getenv("EXIT_MFE_GUARD_BASE_SCALP", "0.7")),
        }
        self._mfe_guard_ratio = {
            "macro": float(os.getenv("EXIT_MFE_GUARD_RATIO_MACRO", "0.6")),
            "micro": float(os.getenv("EXIT_MFE_GUARD_RATIO_MICRO", "0.65")),
            "scalp": float(os.getenv("EXIT_MFE_GUARD_RATIO_SCALP", "0.7")),
        }
        # track best profit (pips) per pocket/side to avoid cutting trades that already went positive
        self._max_profit_cache: Dict[Tuple[str, str], float] = {}
        # Apply stale drawdown exits only to trades opened after this cutover (if set)
        self._dd_cutover = self._parse_cutover_env(os.getenv("EXIT_DD_CUTOVER_ISO"))
        # time-guard throttle to avoid repeated partial cuts
        self._time_guard_ts: Dict[Tuple[str, str], datetime] = {}
        # Optional: disable time-based guards to rely on technical exits only
        self._time_guard_enabled = _env_flag("EXIT_TIME_GUARD_ENABLED", False)
        # Loss clamp thresholds derived from recent MAE analysis (pips)
        self._loss_clamp_partial = {
            "micro": float(os.getenv("LOSS_CLAMP_MICRO_PARTIAL_PIPS", "10.0")),
            "scalp": float(os.getenv("LOSS_CLAMP_SCALP_PARTIAL_PIPS", "8.0")),
        }
        self._loss_clamp_full = {
            "micro": float(os.getenv("LOSS_CLAMP_MICRO_FULL_PIPS", "14.0")),
            "scalp": float(os.getenv("LOSS_CLAMP_SCALP_FULL_PIPS", "12.0")),
        }
        # MFE-based trail/partials for breakout・pullback系の尻尾切り
        self._mfe_partial_macro = float(os.getenv("EXIT_MFE_PARTIAL_MACRO", "8.0"))
        self._mfe_partial_micro = float(os.getenv("EXIT_MFE_PARTIAL_MICRO", "5.0"))
        self._mfe_trail_floor = float(os.getenv("EXIT_MFE_TRAIL_FLOOR", "3.5"))
        self._mfe_trail_gap = float(os.getenv("EXIT_MFE_TRAIL_GAP", "4.0"))
        self._agent_meta_cutover = self._parse_cutover_env(os.getenv("EXIT_AGENT_META_CUTOVER_ISO"))
        if self._agent_meta_cutover is None:
            try:
                self._agent_meta_cutover = datetime.now(timezone.utc)
            except Exception:
                self._agent_meta_cutover = None
        self._min_partial_units = int(os.getenv("EXIT_MIN_PARTIAL_UNITS", "1"))
        # H1 momentum profit lock (macro) defaults
        self._h1_lock_min_trigger = float(os.getenv("EXIT_H1_LOCK_MIN_TRIGGER", "6.0"))
        self._h1_lock_min_buffer = float(os.getenv("EXIT_H1_LOCK_MIN_BUFFER", "2.0"))
        self._h1_lock_trigger_ratio = float(os.getenv("EXIT_H1_LOCK_TRIGGER_RATIO", "0.35"))
        self._h1_lock_buffer_ratio = float(os.getenv("EXIT_H1_LOCK_BUFFER_RATIO", "0.22"))
        self._h1_lock_min_hold_minutes = float(os.getenv("EXIT_H1_LOCK_MIN_HOLD_MIN", "20.0"))
        # Low-vol / hazard controls (news removed; keep deterministic defaults)
        self._low_vol_enabled = True
        self._micro_low_vol_grace_sec = float(os.getenv("EXIT_MICRO_LOW_VOL_GRACE_SEC", "5.0"))
        self._micro_low_vol_event_budget_sec = float(os.getenv("EXIT_MICRO_LOW_VOL_EVENT_BUDGET_SEC", "4.0"))
        self._micro_low_vol_hazard_loss = float(os.getenv("EXIT_MICRO_LOW_VOL_HAZARD_LOSS", "0.4"))
        self._hazard_exit_enabled = _env_flag("EXIT_LOW_VOL_HAZARD_ENABLED", True)
        self._hazard_debounce_ticks = max(1, int(os.getenv("EXIT_LOW_VOL_HAZARD_DEBOUNCE", "2") or 2))
        self._upper_bound_max_sec = float(os.getenv("EXIT_UPPER_BOUND_MAX_SEC", "0"))
        self._timeout_soft_tp_frac = float(os.getenv("EXIT_TIMEOUT_SOFT_TP_FRAC", "0.8"))
        self._soft_tp_pips = float(os.getenv("EXIT_SOFT_TP_PIPS", "0.8"))
        # Dynamic escape (quiet/range/normal) — shrink TP/SL only, do not widen to avoid interfering with runners
        self._escape_atr_quiet = float(os.getenv("EXIT_ESCAPE_ATR_QUIET", "2.8"))
        self._escape_atr_hot = float(os.getenv("EXIT_ESCAPE_ATR_HOT", "5.2"))
        self._escape_bbw_quiet = float(os.getenv("EXIT_ESCAPE_BBW_QUIET", "0.22"))
        self._escape_bbw_hot = float(os.getenv("EXIT_ESCAPE_BBW_HOT", "0.35"))
        self._escape_vol_quiet = float(os.getenv("EXIT_ESCAPE_VOL5M_QUIET", "1.0"))
        self._escape_quiet_tp = float(os.getenv("EXIT_ESCAPE_QUIET_TP_PIPS", "1.2"))
        self._escape_quiet_draw_min = float(os.getenv("EXIT_ESCAPE_QUIET_DRAW_MIN_PIPS", "0.9"))
        self._escape_quiet_draw_ratio = float(os.getenv("EXIT_ESCAPE_QUIET_DRAW_RATIO", "0.55"))
        self._escape_momentum_cut = float(os.getenv("EXIT_ESCAPE_MOMENTUM_MAX", "0.0"))
        self._escape_min_hold = float(os.getenv("EXIT_ESCAPE_MIN_HOLD_SEC", "60"))
        self._mfe_sensitive_reasons = {
            "trend_reversal",
            "macro_trail_hit",
            "macro_atr_trail",
            "macro_trend_fade",
            "reverse_signal",
            "range_take_profit",
            "range_stop",
        }
        # 部分利確を許容する理由（それ以外はフルクローズ）
        self._partial_eligible_reasons = {
            "trend_reversal",
            "reverse_signal",
            "ma_cross_imminent",
            "ma_cross",
            "macro_trend_fade",
            "macro_trail_hit",
            "macro_atr_trail",
            "range_take_profit",
            "micro_profit_guard",
            "micro_profit_snatch",
            "micro_struct_partial",
            "nwave_partial",
            "pivot_partial",
            "candle_partial",
            "breakeven_guard",
            "mfe_trail",
            "mfe_guard",
            "micro_slope_trail",
            "macro_profit_lock",
            "advisor_takeprofit",
        }
        # 強制全決済とする理由（部分利確を無効化）
        self._force_exit_reasons = {
            "range_stop",
            "stop_loss_order",
            "event_lock",
            "micro_momentum_stop",
            "macro_loss_cap",
            "advisor_drawdown",
            "kill_switch",
            "hard_stop",
        }
        # reverse_signal連発を防ぐクールダウンとゲート
        self._reverse_cooldown_sec = float(os.getenv("EXIT_REVERSE_COOLDOWN_SEC", "120"))
        self._reverse_min_hold_sec = float(os.getenv("EXIT_REVERSE_MIN_HOLD_SEC", "120"))
        self._reverse_profit_floor = float(os.getenv("EXIT_REVERSE_PROFIT_FLOOR", "0.8"))
        self._reverse_loss_floor = float(os.getenv("EXIT_REVERSE_LOSS_FLOOR", "4.0"))
        self._reverse_mfe_ratio = float(os.getenv("EXIT_REVERSE_MFE_RATIO", "0.65"))
        self._reverse_mfe_min = float(os.getenv("EXIT_REVERSE_MFE_MIN", "2.0"))
        self._reverse_partial_frac = float(os.getenv("EXIT_REVERSE_PARTIAL_FRAC", "0.3"))
        self._reverse_bounce_buffer = float(os.getenv("EXIT_REVERSE_BOUNCE_BUFFER", "1.5"))
        self._reverse_ts: Dict[Tuple[str, str], datetime] = {}
        # Breakeven guard: lock profits before turning negative
        self._be_guard_trigger = float(os.getenv("EXIT_BE_GUARD_TRIGGER", "1.4"))
        self._be_guard_floor = float(os.getenv("EXIT_BE_GUARD_FLOOR", "0.3"))
        self._be_guard_min_loss = float(os.getenv("EXIT_BE_GUARD_MIN_LOSS", "-0.1"))
        self._be_guard_frac = float(os.getenv("EXIT_BE_GUARD_FRAC", "0.7"))
        # 最終手段としてのマイナス決済を遅らせるための部分利確設定
        self._soft_exit_floor = float(os.getenv("EXIT_SOFT_EXIT_FLOOR", "-6.0"))
        self._soft_exit_frac = float(os.getenv("EXIT_SOFT_EXIT_FRAC", "0.35"))

    def _exit_tech_context(self, fac_m1: Dict, side: str, fac_m5: Optional[Dict] = None, fac_h1: Optional[Dict] = None, fac_h4: Optional[Dict] = None) -> Dict[str, float | bool]:
        """Ichimoku/クラスタ/モメンタム/ボラをEXIT判断用にまとめる。"""
        cloud_pos = self._safe_float(fac_m1.get("ichimoku_cloud_pos"), 0.0)
        span_a_gap = self._safe_float(fac_m1.get("ichimoku_span_a_gap"), 0.0)
        span_b_gap = self._safe_float(fac_m1.get("ichimoku_span_b_gap"), 0.0)
        cluster_high = self._safe_float(fac_m1.get("cluster_high_gap"), 0.0)
        cluster_low = self._safe_float(fac_m1.get("cluster_low_gap"), 0.0)
        macd_hist = self._safe_float(fac_m1.get("macd_hist"), 0.0)
        dmi_diff = self._safe_float(fac_m1.get("plus_di"), 0.0) - self._safe_float(
            fac_m1.get("minus_di"), 0.0
        )
        stoch = self._safe_float(fac_m1.get("stoch_rsi"), 0.5)
        kc_width = self._safe_float(fac_m1.get("kc_width"), 0.0)
        don_width = self._safe_float(fac_m1.get("donchian_width"), 0.0)
        chaikin = self._safe_float(fac_m1.get("chaikin_vol"), 0.0)
        cluster_gap = cluster_high if side == "long" else cluster_low
        cloud_support = (cloud_pos > 0.2) if side == "long" else (cloud_pos < -0.2)
        in_cloud = abs(cloud_pos) < 0.1
        vol_low = (kc_width < 0.006 and don_width < 0.006) or chaikin < 0.1
        ctx = {
            "cloud_pos": cloud_pos,
            "span_a_gap": span_a_gap,
            "span_b_gap": span_b_gap,
            "cloud_support": cloud_support,
            "in_cloud": in_cloud,
            "cluster_gap": cluster_gap,
            "macd_hist": macd_hist,
            "dmi_diff": dmi_diff,
            "stoch": stoch,
            "vol_low": vol_low,
        }
        # MTF方向スコア
        ctx["mtf_m5"] = self._mtf_trend_score(fac_m5, side, adx_floor=13.0)
        ctx["mtf_h1"] = self._mtf_trend_score(fac_h1, side, adx_floor=16.0)
        ctx["mtf_h4"] = self._mtf_trend_score(fac_h4, side, adx_floor=18.0)
        return ctx

    def _mtf_trend_score(self, fac: Optional[Dict], side: str, *, adx_floor: float = 15.0) -> float:
        """
        簡易MTF方向スコア。ma10-ma20の方向とADXで評価。
        戻り余地判定やfast_cutデファーの材料に使う。
        """
        if not fac:
            return 0.0
        try:
            ma_fast = float(fac.get("ma10") or fac.get("ema12") or 0.0)
            ma_slow = float(fac.get("ma20") or fac.get("ema20") or 0.0)
            adx_val = float(fac.get("adx") or 0.0)
        except Exception:
            return 0.0
        dir_ok = (side == "long" and ma_fast > ma_slow) or (side == "short" and ma_fast < ma_slow)
        if not dir_ok:
            return -0.5
        score = 0.2
        gap = abs(ma_fast - ma_slow) / 0.01
        score += min(0.6, gap * 0.05)
        if adx_val >= adx_floor:
            score += min(0.4, (adx_val - adx_floor) * 0.02)
        return max(0.0, min(1.0, score))

    def _pattern_bias(self, story: Optional[ChartStorySnapshot], *, side: str) -> dict:
        """
        抜粋したパターンバイアスを返す。
        bias: 順行/逆行/中立 を示す簡易タグ。
        """
        if story is None:
            return {"bias": "neutral", "conf": 0.0}
        patterns = getattr(story, "pattern_summary", None) or {}
        candle = patterns.get("candlestick") if isinstance(patterns, dict) else {}
        n_wave = patterns.get("n_wave") if isinstance(patterns, dict) else {}
        c_bias = candle.get("bias")
        try:
            c_conf = float(candle.get("confidence", 0.0) or 0.0)
        except Exception:
            c_conf = 0.0
        n_bias = n_wave.get("direction") or n_wave.get("bias")
        try:
            n_conf = float(n_wave.get("confidence", 0.0) or 0.0)
        except Exception:
            n_conf = 0.0

        bias = "neutral"
        conf = 0.0
        if c_bias and c_conf >= 0.55:
            if (side == "long" and c_bias == "up") or (side == "short" and c_bias == "down"):
                bias = "with_candle"
            elif (side == "long" and c_bias == "down") or (side == "short" and c_bias == "up"):
                bias = "against_candle"
            conf = max(conf, c_conf)
        if n_bias and n_conf >= 0.55:
            if (side == "long" and n_bias == "up") or (side == "short" and n_bias == "down"):
                if bias == "against_candle":
                    bias = "mixed"
                elif bias == "with_candle":
                    bias = "with_both"
                else:
                    bias = "with_nwave"
            elif (side == "long" and n_bias == "down") or (side == "short" and n_bias == "up"):
                if bias.startswith("with_"):
                    bias = "mixed"
                else:
                    bias = "against_nwave"
            conf = max(conf, n_conf, conf)
        return {"bias": bias, "conf": conf}

    def _regime_profile(self, fac_m1: Dict, fac_h4: Dict, range_mode: bool) -> str:
        """軽量なレジームタグを返す。"""
        if range_mode:
            return "range"
        adx_m1 = self._safe_float(fac_m1.get("adx"), 0.0)
        bbw_m1 = self._safe_float(fac_m1.get("bbw"), 1.0)
        atr_pips = self._safe_float(fac_m1.get("atr_pips"), 0.0)
        adx_h4 = self._safe_float(fac_h4.get("adx"), 0.0)
        slope_h4 = abs(self._safe_float(fac_h4.get("ma20"), 0.0) - self._safe_float(fac_h4.get("ma10"), 0.0))
        if adx_m1 <= 22.0 and bbw_m1 <= 0.24 and atr_pips <= 7.0:
            return "range"
        if adx_m1 >= 26.0 and adx_h4 >= 20.0 and slope_h4 >= 0.0008:
            return "trend"
        return "mixed"

    def _time_guard_exit(
        self,
        *,
        pocket: str,
        side: str,
        open_info: Dict,
        units: int,
        profit_pips: Optional[float],
        regime: str,
        atr_pips: float,
        now: datetime,
        vwap_gap_pips: Optional[float] = None,
        close_to_vwap: bool = False,
        low_vol: bool = False,
    ) -> Optional[ExitDecision]:
        """時間と含み益/損に応じた段階的縮小を返す。"""
        if units <= 0 or profit_pips is None:
            return None
        open_trades = [tr for tr in (open_info.get("open_trades") or []) if tr.get("side") == side]
        if any(self._is_manual_trade(tr) for tr in open_trades):
            return None
        age_sec = self._youngest_trade_age_seconds(open_info, side, now)
        if age_sec is None:
            return None
        if profit_pips >= 1.2:
            return None

        # 微反転・ごく浅い含み損では時間ガードを発動させない
        small_draw = -max(1.5, atr_pips * 0.5)
        if profit_pips >= small_draw:
            return None

        # MAEセーフティ（SL代替）: 含み損が大きくなったら即時撤退
        # 早めに傷を浅く切る: 閾値をやや緩め、まずは部分クローズで被弾を抑える
        mae_floor = max(3.5, atr_pips * 1.1)
        if profit_pips <= -mae_floor:
            self._time_guard_ts[(pocket, side)] = now
            cut_units = max(self._min_partial_units, int(abs(units) * 0.6))
            signed = -cut_units if side == "long" else cut_units
            return ExitDecision(
                pocket=pocket,
                units=signed,
                reason="time_guard_mae",
                tag="time_guard",
                allow_reentry=False,
            )

        if regime == "range":
            partial_sec, full_sec = (180.0, 360.0) if pocket == "scalp" else (240.0, 540.0)
            profit_gate = 0.8
            exit_floor = -max(3.0, atr_pips * 0.7)
        elif regime == "trend":
            partial_sec, full_sec = (300.0, 480.0) if pocket == "scalp" else (360.0, 720.0)
            profit_gate = 1.4
            exit_floor = -max(4.5, atr_pips * 0.9)
        else:
            partial_sec, full_sec = (240.0, 420.0) if pocket == "scalp" else (320.0, 600.0)
            profit_gate = 1.0
            exit_floor = -max(3.5, atr_pips * 0.8)

        if pocket == "macro":
            partial_sec *= 1.4
            full_sec *= 1.5
            profit_gate += 0.4

        # 低ATR時はガードを短縮
        if atr_pips <= 2.0:
            partial_sec *= 0.75
            full_sec *= 0.85
        if low_vol:
            partial_sec *= 0.85
            full_sec *= 0.9
        if close_to_vwap or (vwap_gap_pips is not None and vwap_gap_pips <= 0.6):
            partial_sec *= 0.8
            full_sec *= 0.85
            profit_gate -= 0.1
            exit_floor = max(exit_floor, -2.2)

        key = (pocket, side)
        last_cut = self._time_guard_ts.get(key)
        if last_cut and (now - last_cut).total_seconds() < 70.0:
            return None

        if age_sec >= partial_sec and profit_pips <= profit_gate:
            frac = 0.5 if pocket in {"scalp", "micro"} else 0.33
            if close_to_vwap or low_vol:
                frac *= 0.9
            cut_units = max(1, math.ceil(abs(units) * frac))
            signed = -cut_units if side == "long" else cut_units
            self._time_guard_ts[key] = now
            return ExitDecision(
                pocket=pocket,
                units=signed,
                reason=f"time_guard_partial_{regime}",
                tag="time_guard",
                allow_reentry=True,
            )

        if age_sec >= full_sec and profit_pips <= max(0.2, exit_floor):
            self._time_guard_ts[key] = now
            signed = -abs(units) if side == "long" else abs(units)
            return ExitDecision(
                pocket=pocket,
                units=signed,
                reason=f"time_guard_exit_{regime}",
                tag="time_guard",
                allow_reentry=False,
            )
        return None

    def _vwap_revert_exit(
        self,
        *,
        pocket: str,
        side: str,
        units: int,
        profit_pips: Optional[float],
        vwap_gap_pips: Optional[float],
        regime: str,
        range_mode: bool,
        atr_pips: Optional[float],
        now: datetime,
    ) -> Optional[ExitDecision]:
        """VWAP急接近時に一部利確して往復負けを防ぐ。"""
        if units <= 0 or profit_pips is None or vwap_gap_pips is None:
            return None
        if vwap_gap_pips > 0.6:
            return None
        if profit_pips < 0.6 and pocket in {"scalp", "micro"}:
            return None
        if profit_pips < 1.0 and pocket not in {"scalp", "micro"}:
            return None
        # 静かなレンジ・低ATRほど強く利確
        frac = 0.5 if pocket in {"scalp", "micro"} else 0.35
        if range_mode or regime == "range":
            frac *= 1.1
        if atr_pips is not None and atr_pips <= 2.0:
            frac *= 1.1
        frac = max(0.25, min(0.8, frac))
        cut_units = max(1, math.ceil(abs(units) * frac))
        signed = -cut_units if side == "long" else cut_units
        return ExitDecision(
            pocket=pocket,
            units=signed,
            reason="vwap_gravity",
            tag="vwap_guard",
            allow_reentry=True,
        )

    def plan_closures(
        self,
        open_positions: Dict[str, Dict],
        signals: List[Dict],
        fac_m1: Dict,
        fac_h4: Dict,
        fac_h1: Optional[Dict] = None,
        fac_m5: Optional[Dict] = None,
        event_soon: bool = False,
        range_mode: bool = False,
        stage_state: Optional[Dict[str, Dict[str, int]]] = None,
        pocket_profiles: Optional[Dict[str, Dict[str, float]]] = None,
        now: Optional[datetime] = None,
        stage_tracker: Optional["StageTracker"] = None,
        *,
        low_vol_profile: Optional[Dict[str, float]] = None,
        low_vol_quiet: bool = False,
        news_status: str = "quiet",
    ) -> List[ExitDecision]:
        current_time = self._ensure_utc(now)
        decisions: List[ExitDecision] = []
        try:
            close_price = float(fac_m1.get("close"))
        except (TypeError, ValueError):
            close_price = None
        if close_price is None:
            # Price不明なら安全側でスキップ
            return decisions
        news_status = news_status or "quiet"  # news pipeline removed; keep flag static
        projection_m1 = compute_ma_projection(fac_m1, timeframe_minutes=1.0)
        projection_m5 = compute_ma_projection(fac_m5, timeframe_minutes=5.0) if fac_m5 else None
        projection_h1 = compute_ma_projection(fac_h1, timeframe_minutes=60.0) if fac_h1 else None
        projection_h4 = compute_ma_projection(fac_h4, timeframe_minutes=240.0)
        m1_candles = fac_m1.get("candles") or []
        atr_pips = fac_m1.get("atr_pips")
        if atr_pips is None:
            atr_pips = (fac_m1.get("atr") or 0.0) * 100.0
        atr_primary = atr_pips
        atr_m1 = atr_pips
        ema_m1 = self._safe_float(fac_m1.get("ema20"), self._safe_float(fac_m1.get("ma20")))
        ema_h4 = self._safe_float(fac_h4.get("ema20"), self._safe_float(fac_h4.get("ma20")))
        story = None
        low_vol_profile = low_vol_profile or {}
        close_price = float(fac_m1.get("close") or 0.0)
        regime_profile = self._regime_profile(fac_m1, fac_h4, range_mode)
        try:
            vwap_gap_pips = (
                abs(close_price - float(fac_m1.get("vwap"))) / 0.01
                if fac_m1.get("vwap") is not None
                else None
            )
        except Exception:
            vwap_gap_pips = None
        close_to_vwap = vwap_gap_pips is not None and vwap_gap_pips <= 0.6
        try:
            vol_5m = float(fac_m1.get("vol_5m") or 0.0)
        except Exception:
            vol_5m = 0.0
        low_vol_flag = (atr_primary or 0.0) <= 2.0 or vol_5m <= 0.8
        for pocket, info in open_positions.items():
            if pocket == "__net__" or pocket not in MANAGED_POCKETS:
                continue
            sig_pool = list(signals)
            long_units = int(info.get("long_units", 0) or 0)
            short_units = int(info.get("short_units", 0) or 0)
            avg_long = info.get("long_avg_price") or info.get("avg_price")
            avg_short = info.get("short_avg_price") or info.get("avg_price")
            long_profit = None
            short_profit = None
            if avg_long and close_price:
                long_profit = (close_price - avg_long) / 0.01
            if avg_short and close_price:
                short_profit = (avg_short - close_price) / 0.01
            # Update per-pocket/side max profit cache (for once-positive detection)
            if long_units > 0 and long_profit is not None:
                self._update_max_profit(pocket, "long", long_profit)
            if short_units > 0 and short_profit is not None:
                self._update_max_profit(pocket, "short", short_profit)
            if long_units == 0:
                self._reset_reverse_counter(pocket, "long")
                self._low_vol_hazard_hits.pop((pocket, "long"), None)
            if short_units == 0:
                self._reset_reverse_counter(pocket, "short")
                self._low_vol_hazard_hits.pop((pocket, "short"), None)
            if long_units == 0 and short_units == 0:
                continue

            # レジーム別時間ガード: 伸びないポジを段階的に縮小/撤退（既定OFF）
            if self._time_guard_enabled:
                for side, units, profit in (
                    ("long", long_units, long_profit),
                    ("short", short_units, short_profit),
                ):
                    tg_decision = self._time_guard_exit(
                        pocket=pocket,
                        side=side,
                        open_info=info,
                        units=units,
                        profit_pips=profit,
                        regime=regime_profile,
                        atr_pips=atr_primary,
                        now=current_time,
                        vwap_gap_pips=vwap_gap_pips,
                        close_to_vwap=close_to_vwap,
                        low_vol=low_vol_flag,
                    )
                    if tg_decision:
                        decisions.append(tg_decision)
                        if side == "long":
                            long_units += tg_decision.units  # negative to reduce
                        else:
                            short_units -= tg_decision.units  # positive to reduce

            # VWAP急接近で半分利確して反転負けを防ぐ
            if long_units > 0 and long_profit is not None:
                vwap_exit = self._vwap_revert_exit(
                    pocket=pocket,
                    side="long",
                    units=long_units,
                    profit_pips=long_profit,
                    vwap_gap_pips=vwap_gap_pips,
                    regime=regime_profile,
                    range_mode=range_mode,
                    atr_pips=atr_primary,
                    now=current_time,
                )
                if vwap_exit:
                    decisions.append(vwap_exit)
                    long_units += vwap_exit.units  # negative
            if short_units > 0 and short_profit is not None:
                vwap_exit = self._vwap_revert_exit(
                    pocket=pocket,
                    side="short",
                    units=short_units,
                    profit_pips=short_profit,
                    vwap_gap_pips=vwap_gap_pips,
                    regime=regime_profile,
                    range_mode=range_mode,
                    atr_pips=atr_primary,
                    now=current_time,
                )
                if vwap_exit:
                    decisions.append(vwap_exit)
                    short_units -= vwap_exit.units  # positive

            # MFEリトレースによる早期部分カット（scalp/microのみ）
            if long_units > 0 and long_profit is not None and long_profit > 0:
                mfe_exit = self._mfe_retrace_exit(
                    pocket=pocket,
                    side="long",
                    units=long_units,
                    profit_pips=long_profit,
                )
                if mfe_exit:
                    decisions.append(mfe_exit)
                    long_units += mfe_exit.units  # cut_units is negative for long
            if short_units > 0 and short_profit is not None and short_profit > 0:
                mfe_exit = self._mfe_retrace_exit(
                    pocket=pocket,
                    side="short",
                    units=short_units,
                    profit_pips=short_profit,
                )
                if mfe_exit:
                    decisions.append(mfe_exit)
                    short_units -= mfe_exit.units  # cut_units is positive for short

            if pocket == "scalp_fast":
                decisions.extend(
                    self._evaluate_scalp_fast(
                        pocket=pocket,
                        open_info=info,
                        event_soon=event_soon,
                        now=current_time,
                        story=story,
                        range_mode=range_mode,
                    )
                )
                continue

            if long_units > 0 and long_profit is not None:
                peak_reverse = self._peak_reversal_hint(
                    pocket=pocket,
                    side="long",
                    profit_pips=long_profit,
                    fac_m1=fac_m1,
                )
                if peak_reverse:
                    sig_pool.append(peak_reverse)
            if short_units > 0 and short_profit is not None:
                peak_reverse = self._peak_reversal_hint(
                    pocket=pocket,
                    side="short",
                    profit_pips=short_profit,
                    fac_m1=fac_m1,
                )
                if peak_reverse:
                    sig_pool.append(peak_reverse)

            reverse_short = self._confirm_reverse_signal(
                self._strong_signal(sig_pool, pocket, "OPEN_SHORT"),
                pocket,
                "long",
                current_time,
            )
            reverse_long = self._confirm_reverse_signal(
                self._strong_signal(sig_pool, pocket, "OPEN_LONG"),
                pocket,
                "short",
                current_time,
            )

            pocket_fac = fac_h4 if pocket == "macro" else fac_m1
            rsi = pocket_fac.get("rsi", fac_m1.get("rsi", 50.0))
            ma10 = pocket_fac.get("ma10", 0.0)
            ma20 = pocket_fac.get("ma20", 0.0)
            adx = pocket_fac.get("adx", 0.0)
            ema20 = fac_m1.get("ema20", 0.0)
            projection_primary = projection_h4 if pocket == "macro" else projection_m1
            projection_fast = projection_m1
            stage_long = ((stage_state or {}).get(pocket) or {}).get("long", 0)
            stage_short = ((stage_state or {}).get(pocket) or {}).get("short", 0)
            profile = (pocket_profiles or {}).get(pocket, {})
            stage_level = stage_long
            max_mfe_long = (
                self._max_mfe_for_side(info, "long", m1_candles, current_time)
                if long_units > 0
                else None
            )
            max_mfe_short = (
                self._max_mfe_for_side(info, "short", m1_candles, current_time)
                if short_units > 0
                else None
            )

            if long_units > 0:
                fast_cut = self._fast_cut_decision(
                    pocket=pocket,
                    side="long",
                    units=long_units,
                    open_info=info,
                    close_price=close_price,
                    atr_pips=atr_pips,
                    fac_m1=fac_m1,
                    fac_m5=fac_m5,
                    fac_h1=fac_h1,
                    now=current_time,
                )
                if fast_cut:
                    decisions.append(fast_cut)
                    continue
                if self._negative_exit_blocked(
                    pocket, info, "long", current_time, long_profit or 0.0, stage_tracker, atr_pips, fac_m1
                ):
                    continue
                decision = self._evaluate_long(
                    pocket,
                    info,
                    long_units,
                    reverse_short,
                    event_soon,
                    rsi,
                    ma10,
                    ma20,
                    adx,
                    close_price,
                    ema_m1,
                    ema_h4 if pocket == "macro" else None,
                    range_mode,
                    current_time,
                    projection_primary,
                    projection_fast,
                    atr_pips,
                    fac_m1,
                    stage_tracker,
                    max_mfe_long=max_mfe_long,
                    max_mfe_short=max_mfe_short,
                    low_vol_profile=low_vol_profile,
                    low_vol_quiet=low_vol_quiet,
                    news_status=news_status,
                    m1_candles=m1_candles,
                    profile=profile,
                )
                if decision:
                    decisions.append(decision)

            if short_units > 0:
                stage_level = stage_short
                fast_cut = self._fast_cut_decision(
                    pocket=pocket,
                    side="short",
                    units=short_units,
                    open_info=info,
                    close_price=close_price,
                    atr_pips=atr_pips,
                    fac_m1=fac_m1,
                    fac_m5=fac_m5,
                    fac_h1=fac_h1,
                    now=current_time,
                )
                if fast_cut:
                    decisions.append(fast_cut)
                    continue
                if self._negative_exit_blocked(
                    pocket, info, "short", current_time, short_profit or 0.0, stage_tracker, atr_pips, fac_m1
                ):
                    continue
                decision = self._evaluate_short(
                    pocket,
                    info,
                    short_units,
                    reverse_long,
                    event_soon,
                    rsi,
                    ma10,
                    ma20,
                    adx,
                    close_price,
                    ema_m1,
                    ema_h4 if pocket == "macro" else None,
                    range_mode,
                    current_time,
                    projection_primary,
                    projection_fast,
                    atr_pips,
                    fac_m1,
                    stage_tracker,
                    max_mfe_short=max_mfe_short,
                    max_mfe_long=max_mfe_long,
                    low_vol_profile=low_vol_profile,
                    low_vol_quiet=low_vol_quiet,
                    news_status=news_status,
                    m1_candles=m1_candles,
                    profile=profile,
                )
                if decision:
                    decisions.append(decision)

        return decisions

    def _strong_signal(
        self, signals: List[Dict], pocket: str, action: str
    ) -> Optional[Dict]:
        candidates = [
            s
            for s in signals
            if s.get("pocket") == pocket and s.get("action") == action
        ]
        if not candidates:
            return None
        best = max(candidates, key=lambda s: s.get("confidence", 0))
        threshold = self.confidence_threshold
        if pocket == "macro":
            threshold = self._macro_signal_threshold
        if best.get("confidence", 0) >= threshold:
            return best
        return None

    def _macro_loss_cap(self, atr_pips: float, matured: bool) -> float:
        atr_ref = float(atr_pips or 0.0)
        if atr_ref <= 0.0:
            atr_ref = 8.0
        value = atr_ref * (0.08 if matured else 0.06)
        return max(0.8, min(1.6, value))

    def _macro_slowdown_detected(
        self,
        *,
        profit_pips: float,
        adx: float,
        rsi: float,
        projection_fast: Optional[MACrossProjection],
        close_price: float,
        ema20: float,
        atr_pips: float,
    ) -> bool:
        if profit_pips < 1.2 or close_price is None or ema20 is None:
            return False
        buffer = max(0.8, (atr_pips or 6.0) * 0.35)
        momentum_band = (profit_pips <= max(8.0, (atr_pips or 6.0) * 1.2))
        ema_gap_pips = (close_price - ema20) / 0.01
        ema_cooling = ema_gap_pips <= buffer
        adx_fade = adx <= self._macro_trend_adx
        rsi_fade = rsi <= 58.0
        slope_fade = True
        if projection_fast is not None:
            slope_fade = projection_fast.gap_slope_pips < 0.08
        return momentum_band and ema_cooling and slope_fade and (adx_fade or rsi_fade)

    def _mfe_partial_units(
        self,
        pocket: str,
        units: int,
        profit_pips: float,
        *,
        atr_pips: Optional[float] = None,
        slope_hint: Optional[float] = None,
    ) -> int:
        """Return partial units when MFEが一定以上に達したときに利益を確定する。"""
        if units <= 0:
            return 0
        threshold = self._mfe_partial_macro if pocket == "macro" else self._mfe_partial_micro
        if atr_pips is not None:
            try:
                atr_val = float(atr_pips or 0.0)
            except Exception:
                atr_val = 0.0
            if atr_val > 0.0:
                if pocket == "macro":
                    threshold = max(threshold * 0.8, min(threshold * 1.3, atr_val * 1.1))
                else:
                    threshold = max(threshold * 0.7, min(threshold * 1.25, atr_val * 0.95))
        if slope_hint is not None:
            try:
                slope_val = float(slope_hint)
            except Exception:
                slope_val = 0.0
            if slope_val > 0.12:
                threshold *= 1.12
            elif slope_val < -0.12:
                threshold *= 0.88
        if profit_pips >= threshold:
            # take half but keep at least 1000 units
            partial = max(1000, units // 2)
            if partial < units:
                return partial
        return 0

    def _mfe_trail_hit(
        self,
        *,
        side: str,
        avg_price: Optional[float],
        close_price: Optional[float],
        profit_pips: float,
    ) -> bool:
        """Simple BE+ trail: once profit exceeds trail_gap, do not give back more than trail_floor."""
        if avg_price is None or close_price is None:
            return False
        # only arm when profit already healthy
        if profit_pips < self._mfe_trail_gap:
            return False
        cushion = max(1.0, self._mfe_trail_floor)
        give_back = profit_pips - cushion
        if give_back <= 0:
            return False
        if side == "long":
            trail_floor = avg_price + give_back * 0.01
            return close_price <= trail_floor
        trail_floor = avg_price - give_back * 0.01
        return close_price >= trail_floor

    def _fast_cut_decision(
        self,
        *,
        pocket: str,
        side: str,
        units: int,
        open_info: Dict,
        close_price: float,
        atr_pips: float,
        fac_m1: Dict,
        fac_m5: Optional[Dict],
        fac_h1: Optional[Dict],
        now: datetime,
    ) -> Optional[ExitDecision]:
        """
        ATR×時間に基づく早期クローズ。
        - 逆行が fast_cut を大きく超えた場合は即時クローズ
        - 逆行が fast_cut を超え、かつ一定時間経過でクローズ
        - ただし「一度も+域に乗っておらず若いポジ」は緩めに判定する
        """
        # スプレッド拡大時間帯（例: 07-08 JST）はカットを停止
        if _in_jst_window(now, self._cut_disable_jst_start, self._cut_disable_jst_end):
            return None
        if pocket not in {"micro", "scalp"}:
            return None
        if pocket == "scalp" and self._disable_scalp_fast_cut:
            return None
        if units <= 0:
            return None

        avg_price = open_info.get("long_avg_price") if side == "long" else open_info.get("short_avg_price")
        if avg_price is None:
            avg_price = open_info.get("avg_price")
        if avg_price is None or close_price is None:
            return None

        # 経過時間（最古トレード基準）
        open_trades = [tr for tr in (open_info.get("open_trades") or []) if tr.get("side") == side]
        if any(self._is_manual_trade(tr) for tr in open_trades):
            return None
        age_sec = None
        for tr in open_trades:
            age = self._trade_age_seconds(tr, now)
            if age is not None:
                age_sec = age if age_sec is None else min(age_sec, age)

        profit_pips = (close_price - avg_price) / 0.01 if side == "long" else (avg_price - close_price) / 0.01
        max_seen = self._max_profit_cache.get((pocket, side))
        if profit_pips >= 0:
            return None  # 順行中は早切りしない
        # 時間ではなく市況・テクニカルで判断するため、年齢ゲートは撤廃

        atr_val = float(atr_pips or 0.0)
        candles = fac_m1.get("candles") or []
        slope6 = _slope_from_candles(candles, window=6)
        slope12 = _slope_from_candles(candles, window=12) if len(candles) >= 12 else slope6
        session = _session_bucket(now if now.tzinfo else now.replace(tzinfo=timezone.utc))
        stack_units = abs(units)
        # 市況でゲートを揺らす（ATR低→広げる、高→やや狭める）
        if pocket == "scalp":
            # scalpは戻りを待つ: 基準を広げ、時間ゲートも長め
            if atr_val <= 1.6:
                fast_cut = max(7.5, atr_val * 1.4)
                time_gate = max(self._scalp_min_hold_seconds, max(95.0, atr_val * 20.0))
            elif atr_val >= 3.0:
                fast_cut = max(8.5, atr_val * 1.1)
                time_gate = max(self._scalp_min_hold_seconds, max(85.0, atr_val * 14.0))
            else:
                fast_cut = max(7.0, atr_val * 1.2)
                time_gate = max(self._scalp_min_hold_seconds, max(90.0, atr_val * 17.0))
            hard_cut = fast_cut * 1.8
        else:
            if atr_val <= 1.5:
                fast_cut = max(6.0, atr_val * 1.1)
                time_gate = max(90.0 if pocket == "micro" else 70.0, atr_val * 18.0)
            elif atr_val >= 3.5:
                fast_cut = max(6.0, atr_val * 0.8)
                time_gate = max(75.0 if pocket == "micro" else 60.0, atr_val * 12.0)
            else:
                fast_cut = max(6.0, atr_val * 0.9)
                time_gate = max(80.0 if pocket == "micro" else 65.0, atr_val * 15.0)
            hard_cut = fast_cut * 1.6
        # 低ボラはもう少し待つ
        if pocket == "scalp" and atr_val <= 1.2:
            fast_cut *= 1.1
            time_gate *= 1.2

        # 傾き・スタック・セッションで微調整
        slope_bias = 1.0
        if side == "long":
            if slope6 < -0.6 or slope12 < -0.4:
                slope_bias = 0.82
            elif slope6 > 1.2:
                slope_bias = 1.12
        else:
            if slope6 > 0.6 or slope12 > 0.4:
                slope_bias = 0.82
            elif slope6 < -1.2:
                slope_bias = 1.12
        fast_cut *= slope_bias
        hard_cut *= slope_bias
        if session == "asia":
            time_gate *= 0.9
        elif session == "ny":
            time_gate *= 1.05

        if stack_units >= 25000:
            fast_cut *= 0.85
            hard_cut *= 0.85
            time_gate *= 0.9
        elif stack_units >= 15000:
            fast_cut *= 0.92
            hard_cut *= 0.92
            time_gate *= 0.95

        thesis_fast_cut = None
        thesis_time_gate = None
        thesis_hard_mult = None
        has_fast_cut_meta = False
        has_technical_meta = False
        for tr in open_trades:
            age = self._trade_age_seconds(tr, now)
            if age is None:
                continue
            age_sec = age if age_sec is None else min(age_sec, age)
            if self._has_kill_opt_in(tr):
                has_fast_cut_meta = True
            thesis = self._parse_entry_thesis(tr)
            if thesis:
                try:
                    factors = thesis.get("factors") if isinstance(thesis.get("factors"), dict) else {}
                    fac_m1_thesis = factors.get("m1") if isinstance(factors, dict) else {}
                    atr_hint = fac_m1_thesis.get("atr_pips") or thesis.get("atr_pips")
                    rsi_hint = fac_m1_thesis.get("rsi") or thesis.get("rsi")
                    adx_hint = fac_m1_thesis.get("adx") or thesis.get("adx")
                    if atr_hint not in (None, "") and rsi_hint not in (None, "") and adx_hint not in (None, ""):
                        has_technical_meta = True
                    if thesis_fast_cut is None and thesis.get("fast_cut_pips") is not None:
                        thesis_fast_cut = float(thesis.get("fast_cut_pips"))
                    if thesis_time_gate is None and thesis.get("fast_cut_time_sec") is not None:
                        thesis_time_gate = float(thesis.get("fast_cut_time_sec"))
                    if thesis_hard_mult is None and thesis.get("fast_cut_hard_mult") is not None:
                        thesis_hard_mult = float(thesis.get("fast_cut_hard_mult"))
                except Exception:
                    pass
        if thesis_time_gate:
            time_gate = max(time_gate, thesis_time_gate)

        loss = abs(profit_pips)
        tech_ok = has_technical_meta or self._has_realtime_technicals(fac_m1)
        # fast_cut はメタ付き or テクニカルが揃ったポジのみ適用（手動・旧ポジは対象外）
        if not has_fast_cut_meta and not tech_ok:
            agent_like = False
            cutover_ok = False
            for tr in open_trades:
                cid = tr.get("client_id") or tr.get("client_order_id") or ""
                if any(cid.startswith(prefix) for prefix in AGENT_CLIENT_PREFIXES):
                    agent_like = True
                    if self._agent_meta_cutover:
                        ot = self._parse_open_time(tr.get("open_time"))
                        if ot and ot >= self._agent_meta_cutover:
                            cutover_ok = True
                    else:
                        cutover_ok = True
            if not agent_like:
                return None
            if self._agent_meta_cutover and not cutover_ok:
                return None
            has_fast_cut_meta = True
        if thesis_fast_cut and thesis_fast_cut > 0:
            fast_cut = thesis_fast_cut
        if thesis_hard_mult and thesis_hard_mult > 0:
            hard_cut = fast_cut * thesis_hard_mult

        # 一度でも+域に乗ったポジは少し緩めて待つ
        max_seen = self._max_profit_cache.get((pocket, side))
        if max_seen is not None and max_seen >= 2.5:
            fast_cut *= 1.3

        # 直近のRSIが中立帯なら1回だけ様子見
        rsi_val = None
        try:
            rsi_val = float(fac_m1.get("rsi"))
        except Exception:
            rsi_val = None
        if rsi_val is not None and 45.0 <= rsi_val <= 55.0 and loss < fast_cut * 0.8:
            return None

        # MFEリトレース＆構造・モメンタム確認: 戻しそうなら切らない（時間条件は使わない）
        rsi_val = None
        adx_val = None
        ma_fast = None
        ma_slow = None
        drawdown_ratio = None
        try:
            rsi_val = float(fac_m1.get("rsi"))
        except Exception:
            pass
        try:
            adx_val = float(fac_m1.get("adx"))
        except Exception:
            pass
        try:
            ma_fast = float(fac_m1.get("ma10"))
            ma_slow = float(fac_m1.get("ma20"))
        except Exception:
            pass
        if max_seen is not None and max_seen > 0:
            try:
                drawdown_ratio = (max_seen - profit_pips) / max_seen
            except Exception:
                drawdown_ratio = None
            # MFEが小さいときは時間を長めに取るが、ゲート時間を超えれば切る
            if max_seen < 3.0:
                return None
            if drawdown_ratio is not None and drawdown_ratio < 0.5:
                return None
        else:
            # 一度も+域に乗っていないケースはloss閾値でのみ判断（時間は使わない）
            pass

        # neutral RSI帯やADX弱いときは早切りしない
        if rsi_val is not None and 44.0 <= rsi_val <= 56.0 and drawdown_ratio is not None and drawdown_ratio < 0.9:
            return None
        if adx_val is not None and adx_val < 16.0 and drawdown_ratio is not None and drawdown_ratio < 0.9:
            return None
        # MA向きがポジ方向と整合するなら待つ
        if ma_fast is not None and ma_slow is not None:
            if side == "long" and ma_fast >= ma_slow:
                return None
            if side == "short" and ma_fast <= ma_slow:
                return None

        tech_ctx = self._exit_tech_context(fac_m1 or {}, side)
        cluster_gap = float(tech_ctx.get("cluster_gap") or 0.0)
        cloud_support = bool(tech_ctx.get("cloud_support"))
        in_cloud = bool(tech_ctx.get("in_cloud"))
        macd_hist = float(tech_ctx.get("macd_hist") or 0.0)
        dmi_diff = float(tech_ctx.get("dmi_diff") or 0.0)
        stoch = float(tech_ctx.get("stoch") or 0.5)
        mtf_m5 = float(tech_ctx.get("mtf_m5") or 0.0)
        mtf_h1 = float(tech_ctx.get("mtf_h1") or 0.0)
        mtf_h4 = float(tech_ctx.get("mtf_h4") or 0.0)
        pattern_ctx = self._pattern_bias(open_info.get("story") if isinstance(open_info, dict) else None, side=side)
        pattern_bias = pattern_ctx.get("bias")
        pattern_conf = float(pattern_ctx.get("conf") or 0.0)
        mtf_score_m5 = self._mtf_trend_score(fac_m5, side, adx_floor=13.0)
        mtf_score_h1 = self._mtf_trend_score(fac_h1, side, adx_floor=16.0)
        bounce_possible = False
        if cluster_gap > 3.0 and cloud_support:
            if side == "long":
                if macd_hist > 0 or dmi_diff > 0 or stoch <= 0.15:
                    bounce_possible = True
            else:
                if macd_hist < 0 or dmi_diff < 0 or stoch >= 0.85:
                    bounce_possible = True
        if in_cloud and tech_ctx.get("vol_low"):
            bounce_possible = True
        if pattern_bias in {"with_candle", "with_nwave", "with_both"} and pattern_conf >= 0.6:
            bounce_possible = True
        if pattern_bias in {"against_candle", "against_nwave"} and pattern_conf >= 0.6:
            bounce_possible = False
        if mtf_score_m5 >= 0.6 or mtf_score_h1 >= 0.6:
            bounce_possible = True
        if mtf_score_m5 <= 0.0 and mtf_score_h1 <= 0.0:
            bounce_possible = False
        # H4強逆行は抑制
        if mtf_h4 <= 0.0:
            bounce_possible = False

        if loss >= hard_cut:
            # ハードでもまず部分でリスク落とす（極端な損失のみ全量）
            fraction = 0.7 if pocket == "micro" else 0.6
            cut_units = max(1000, int(abs(units) * fraction))
            cut_units = -cut_units if side == "long" else cut_units
            reason = "fast_cut_hard"
            if loss >= hard_cut * 1.4:
                cut_units = -abs(units) if side == "long" else abs(units)
            return ExitDecision(
                pocket=pocket,
                units=cut_units,
                reason=reason,
                tag="fast-cut",
                allow_reentry=True,
            )
        if loss >= fast_cut:
            # 市況が味方なら様子見（雲順行＋クラスタ遠＋MACD/DMI順向 or Stoch極端）
            if bounce_possible:
                return None
            # 部分カットを基本とし、雲逆行やクラスタ近は厚めに削る
            fraction = 0.5 if pocket == "scalp" else 0.6
            if cluster_gap > 0 and cluster_gap <= 3.0:
                fraction = max(fraction, 0.7)
            cut_units = max(1000, int(abs(units) * fraction))
            cut_units = -cut_units if side == "long" else cut_units
            return ExitDecision(
                pocket=pocket,
                units=cut_units,
                reason="fast_cut_soft",
                tag="fast-cut",
                allow_reentry=True,
            )
        return None

    def _micro_adaptive_trail(
        self,
        *,
        pocket: str,
        side: str,
        units: int,
        open_info: Dict,
        profit_pips: float,
        close_price: float,
        atr_pips: float,
        fac_m1: Dict,
        now: datetime,
    ) -> Optional[ExitDecision]:
        """BE/トレールを市況とスタック量で動的に緩急。"""
        if pocket not in {"micro", "scalp"}:
            return None
        if profit_pips is None or profit_pips <= 0.0:
            return None
        try:
            avg_price = (
                open_info.get("long_avg_price") if side == "long" else open_info.get("short_avg_price")
            ) or open_info.get("avg_price")
        except Exception:
            avg_price = None
        if avg_price is None or close_price is None:
            return None

        candles = fac_m1.get("candles") or []
        slope6 = _slope_from_candles(candles, window=6)
        slope12 = _slope_from_candles(candles, window=12) if len(candles) >= 12 else slope6
        slope = (slope6 * 0.6) + (slope12 * 0.4)
        if side == "short":
            slope *= -1  # long基準に揃える
        atr_val = float(atr_pips or 0.0)
        if atr_val <= 0.0:
            try:
                atr_val = float(fac_m1.get("atr") or 0.0) * 100.0
            except Exception:
                atr_val = 0.0
        if atr_val <= 0.0:
            atr_val = 6.0

        youngest = self._youngest_trade_age_seconds(open_info, side, now) or 0.0
        stack_units = abs(units)
        session = _session_bucket(now if now.tzinfo else now.replace(tzinfo=timezone.utc))

        arm = max(0.8, min(2.4, atr_val * 0.35))
        give_back = max(0.6, min(2.6, atr_val * 0.45))
        # スキャルは fast_cut を優先し、スロープ系のBEトレールは使わない
        if pocket == "scalp":
            return None
        if pocket == "scalp":
            arm = max(0.6, arm * 0.9)
            give_back = max(0.5, give_back * 0.9)

        # 傾きが鈍いならトレールを近づけ、強いなら少し伸ばす
        if slope <= 0.15:
            arm = max(0.6, arm * 0.9)
            give_back = max(0.5, give_back * 0.8)
        elif slope >= 0.9:
            arm = min(3.2, arm * 1.25)
            give_back = min(3.0, give_back * 1.15)

        # セッションとスタック量で絞る
        if session == "asia":
            give_back = max(0.5, give_back * 0.9)
        if stack_units >= 20000:
            arm = max(0.5, arm * 0.9)
            give_back = max(0.4, give_back * 0.8)
        elif stack_units >= 15000:
            give_back = max(0.5, give_back * 0.9)

        if profit_pips < arm:
            return None

        # スタックが大きく傾き弱なら部分利確で軽くする
        if stack_units >= 20000 and slope <= 0.3 and youngest >= 40.0:
            cut_units = max(1000, int(stack_units * 0.5))
            if cut_units < stack_units:
                cut = -cut_units if side == "long" else cut_units
                return ExitDecision(
                    pocket=pocket,
                    units=cut,
                    reason="micro_slope_be_partial",
                    tag="micro-slope-be",
                    allow_reentry=True,
                )

        # BE+トレール（順行が伸びずに戻し始めたときのみ）
        cushion = profit_pips - give_back
        if cushion <= 0.2:
            return None
        if side == "long":
            trail_floor = avg_price + cushion * 0.01
            if close_price <= trail_floor:
                return ExitDecision(
                    pocket=pocket,
                    units=units,
                    reason="micro_slope_trail",
                    tag="micro-slope-be",
                    allow_reentry=True,
                )
        else:
            trail_floor = avg_price - cushion * 0.01
            if close_price >= trail_floor:
                return ExitDecision(
                    pocket=pocket,
                    units=units,
                    reason="micro_slope_trail",
                    tag="micro-slope-be",
                    allow_reentry=True,
                )
        return None

    def _loss_clamp_exit(
        self,
        *,
        pocket: str,
        side: str,
        units: int,
        profit_pips: float,
        open_info: Dict,
        atr_pips: float,
        fac_m1: Dict,
        now: datetime,
        close_price: Optional[float],
    ) -> Optional[ExitDecision]:
        """
        Clamp deep losses based on pocket-specific MAE bands, with chart/ATR/time/MFE gating.
        Partial cut at lower band, full cut at higher band. タグ必須。
        """
        if pocket not in {"micro", "scalp"}:
            return None
        if profit_pips is None or profit_pips >= 0.0:
            return None
        if _in_jst_window(now, self._cut_disable_jst_start, self._cut_disable_jst_end):
            return None
        partial = self._loss_clamp_partial.get(pocket)
        full = self._loss_clamp_full.get(pocket)
        if partial is None or full is None:
            return None

        # Opt-in required: kill_switch/fast_cutタグまたはメタ付きトレードのみ対象
        trades = [tr for tr in (open_info.get("open_trades") or []) if tr.get("side") == side and self._has_kill_opt_in(tr)]
        if not trades:
            return None

        # Chart gate: require EMA方向とMA傾きが逆行を示す
        gap_thresh = float(os.getenv("LOSS_CLAMP_EMA_GAP_PIPS", "0.6"))
        try:
            ema = float(fac_m1.get("ema20") or fac_m1.get("ma20") or 0.0)
            ma10 = float(fac_m1.get("ma10") or 0.0)
            ma20 = float(fac_m1.get("ma20") or ema or 0.0)
        except Exception:
            ema = ma10 = ma20 = 0.0
        gap_ok = False
        if close_price is not None and ema:
            ema_gap = (close_price - ema) / 0.01
            if side == "long":
                if ema_gap <= -gap_thresh and ma10 < ma20:
                    gap_ok = True
            else:
                if ema_gap >= gap_thresh and ma10 > ma20:
                    gap_ok = True
        if not gap_ok:
            return None

        # Max seen profit (MFE) to avoid cutting before giving it a chance
        max_seen = self._max_profit_cache.get((pocket, side))

        # Age scaling: newer tradesは少し緩め、長く抱えるほどタイトに
        age_min = None
        for tr in trades:
            age = self._trade_age_seconds(tr, now)
            if age is None:
                continue
            age_m = age / 60.0
            age_min = age_m if age_min is None else min(age_min, age_m)
        age_scale = 1.0
        if age_min is not None:
            if age_min < 20:
                age_scale = 1.2
            elif age_min > 180:
                age_scale = 0.9
            elif age_min > 360:
                age_scale = 0.8

        # ATR scaling: 高ボラはタイトに、低ボラはやや緩め
        atr_val = float(atr_pips or 0.0)
        if atr_val <= 0.0:
            try:
                atr_val = float(fac_m1.get("atr") or 0.0) * 100.0
            except Exception:
                atr_val = 0.0
        atr_scale = 1.0
        if atr_val > 3.5:
            atr_scale = 0.8
        elif atr_val < 1.2:
            atr_scale = 1.25

        # RSI中立帯はリバウンド余地とみてわずかに緩め
        try:
            rsi_val = float(fac_m1.get("rsi"))
        except Exception:
            rsi_val = None
        rsi_scale = 1.0
        if rsi_val is not None and 45.0 <= rsi_val <= 55.0:
            rsi_scale = 1.1

        # Require either some age or prior +MFE to avoid premature clamp
        mfe_gate = float(os.getenv("LOSS_CLAMP_MFE_GATE_PIPS", "3.0"))
        if max_seen is None:
            if age_min is None or age_min < 15.0:
                return None
            max_seen = 0.0  # ageだけで解禁

        # Retrace-based thresholds: まずはMFEに対する戻り率で判定
        retrace_partial = float(os.getenv("LOSS_CLAMP_RETRACE_PARTIAL", "0.6"))
        retrace_full = float(os.getenv("LOSS_CLAMP_RETRACE_FULL", "0.8"))

        scale = max(0.6, min(1.4, atr_scale * age_scale * rsi_scale))
        partial *= scale
        full *= scale
        # MFEがゲートを超えていれば、MFEに対する戻し割合も閾値に組み込む
        if max_seen is not None and max_seen >= mfe_gate:
            partial = max(partial, abs(max_seen) * retrace_partial)
            full = max(full, abs(max_seen) * retrace_full)
        loss = abs(profit_pips)
        if loss >= full:
            cut_units = -abs(units) if side == "long" else abs(units)
            return ExitDecision(
                pocket=pocket,
                units=cut_units,
                reason="loss_clamp_full",
                tag="loss-clamp",
                allow_reentry=True,
            )
        if loss >= partial:
            cut_units = max(1000, abs(units) // 2)
            cut_units = min(abs(units), cut_units)
            cut_units = -cut_units if side == "long" else cut_units
            return ExitDecision(
                pocket=pocket,
                units=cut_units,
                reason="loss_clamp_partial",
                tag="loss-clamp",
                allow_reentry=True,
            )
        return None

    def _stale_drawdown_exit(
        self,
        *,
        pocket: str,
        side: str,
        units: int,
        open_info: Dict,
        profit_pips: float,
        atr_pips: float,
        fac_m1: Dict,
        now: datetime,
    ) -> Optional[ExitDecision]:
        """
        Catch aging underwater trades that never armed fast_cut.
        Designed for micro/scalp pockets: widen early, tighten with age.
        """
        # stale_drawdownは新しい loss_clamp/orphan_guard へ統合
        return None

    def _orphan_guard_exit(
        self,
        *,
        pocket: str,
        side: str,
        units: int,
        open_info: Dict,
        profit_pips: float,
        atr_pips: float,
        fac_m1: Dict,
        now: datetime,
    ) -> Optional[ExitDecision]:
        """
        Trades without kill/fast_cut meta (thesis欠損/SLなし想定) を広めのソフトガードで捕捉。
        ハードSLは置かず、ATR×時間＋リトレースでマーケット決済する。
        """
        if _in_jst_window(now, self._cut_disable_jst_start, self._cut_disable_jst_end):
            return None
        if pocket not in {"micro", "scalp"}:
            return None
        if profit_pips is None or profit_pips >= 0.0:
            return None

        trades = [tr for tr in (open_info.get("open_trades") or []) if tr.get("side") == side]
        if not trades:
            return None

        eligible: list[Dict] = []
        for tr in trades:
            if self._has_kill_opt_in(tr):
                continue  # 正規の fast_cut/kill に任せる
            thesis = self._parse_entry_thesis(tr)
            has_fast_meta = thesis.get("fast_cut_pips") or thesis.get("fast_cut_time_sec") or thesis.get(
                "fast_cut_hard_mult"
            )
            if has_fast_meta:
                continue
            eligible.append(tr)
        if not eligible:
            return None

        # エージェント玉のメタ欠損のみを対象にし、cutover以前の建玉は除外
        filtered: list[Dict] = []
        for tr in eligible:
            cid = tr.get("client_id") or tr.get("client_order_id") or ""
            if not any(cid.startswith(prefix) for prefix in AGENT_CLIENT_PREFIXES):
                continue
            if self._agent_meta_cutover:
                ot = self._parse_open_time(tr.get("open_time"))
                if ot is None or ot < self._agent_meta_cutover:
                    continue
            filtered.append(tr)
        if not filtered:
            return None
        eligible = filtered

        atr_val = float(atr_pips or 0.0)
        if atr_val <= 0.0:
            try:
                atr_val = float(fac_m1.get("atr") or 0.0) * 100.0
            except Exception:
                atr_val = 0.0
        if atr_val <= 0.0:
            atr_val = 6.0

        age_sec = None
        for tr in eligible:
            age = self._trade_age_seconds(tr, now)
            if age is None:
                continue
            age_sec = age if age_sec is None else min(age_sec, age)
        if age_sec is None or age_sec < 300.0:  # 5分は待つ
            return None

        loss = abs(profit_pips)
        gate = max(6.0, atr_val * 2.5)
        max_seen = self._max_profit_cache.get((pocket, side))
        retrace_hit = False
        if max_seen is not None and max_seen >= 6.0:
            try:
                retrace_hit = loss >= max_seen * 0.7
            except Exception:
                retrace_hit = False

        if loss < gate and not retrace_hit:
            return None

        cut_units = -abs(units) if side == "long" else abs(units)
        logging.info(
            "[EXIT] orphan_guard pocket=%s side=%s loss=%.1fp gate=%.1fp retrace=%s age=%.0fs atr=%.2f",
            pocket,
            side,
            loss,
            gate,
            retrace_hit,
            age_sec,
            atr_val,
        )
        return ExitDecision(
            pocket=pocket,
            units=cut_units,
            reason="orphan_guard",
            tag="orphan-guard",
            allow_reentry=True,
        )

    def _evaluate_long(
        self,
        pocket: str,
        open_info: Dict,
        units: int,
        reverse_signal: Optional[Dict],
        event_soon: bool,
        rsi: float,
        ma10: float,
        ma20: float,
        adx: float,
        close_price: float,
        ema_fast: float,
        ema_primary: Optional[float],
        range_mode: bool,
        now: datetime,
        projection_primary: Optional[MACrossProjection],
        projection_fast: Optional[MACrossProjection],
        atr_pips: float,
        fac_m1: Dict,
        stage_tracker: Optional["StageTracker"],
        *,
        max_mfe_long: Optional[float] = None,
        max_mfe_short: Optional[float] = None,
        low_vol_profile: Optional[Dict[str, float]] = None,
        low_vol_quiet: bool = False,
        news_status: str = "quiet",
        m1_candles: Optional[List[Dict]] = None,
        profile: Optional[Dict] = None,
    ) -> Optional[ExitDecision]:

        allow_reentry = False
        reason = ""
        tag = f"{pocket}-long"
        story = None
        ema20 = ema_primary if ema_primary is not None else ema_fast
        candles = m1_candles or fac_m1.get("candles") or []
        atr_primary = atr_pips
        atr_m1 = atr_pips
        stage_level = 0
        profile = profile or {}
        avg_price = open_info.get("long_avg_price") or open_info.get("avg_price")
        try:
            vol_5m = float(fac_m1.get("vol_5m") or 0.0)
        except Exception:
            vol_5m = 0.0
        profit_pips = 0.0
        if avg_price and close_price:
            profit_pips = (close_price - avg_price) / 0.01
        orphan_guard = self._orphan_guard_exit(
            pocket=pocket,
            side="long",
            units=units,
            open_info=open_info,
            profit_pips=profit_pips,
            atr_pips=atr_pips,
            fac_m1=fac_m1,
            now=now,
        )
        if orphan_guard:
            return orphan_guard
        adaptive_exit = self._micro_adaptive_trail(
            pocket=pocket,
            side="long",
            units=units,
            open_info=open_info,
            profit_pips=profit_pips,
            close_price=close_price,
            atr_pips=atr_pips,
            fac_m1=fac_m1,
            now=now,
        )
        if adaptive_exit:
            return adaptive_exit
        drawdown_exit = self._stale_drawdown_exit(
            pocket=pocket,
            side="long",
            units=units,
            open_info=open_info,
            profit_pips=profit_pips,
            atr_pips=atr_pips,
            fac_m1=fac_m1,
            now=now,
        )
        if drawdown_exit:
            return drawdown_exit
        clamp_exit = self._loss_clamp_exit(
            pocket=pocket,
            side="long",
            units=units,
            profit_pips=profit_pips,
            open_info=open_info,
            atr_pips=atr_pips,
            fac_m1=fac_m1,
            now=now,
            close_price=close_price,
        )
        if clamp_exit:
            return clamp_exit
        neg_exit_blocked = self._negative_exit_blocked(
            pocket, open_info, "long", now, profit_pips, stage_tracker, atr_pips, fac_m1
        )
        target_bounds = self._entry_target_bounds(open_info, "long")
        if (
            reverse_signal
            and target_bounds
            and profit_pips is not None
            and profit_pips >= 0.0
            and profit_pips < target_bounds[0] * 0.75
        ):
            self._record_target_guard(
                pocket,
                "long",
                profit_pips,
                target_bounds,
                reverse_signal.get("tag") if reverse_signal else None,
            )
            reverse_signal = None

        escape_exit = self._dynamic_escape_exit(
            pocket=pocket,
            side="long",
            units=units,
            open_info=open_info,
            profit_pips=profit_pips,
            max_mfe=max_mfe_long,
            atr_pips=atr_pips,
            fac_m1=fac_m1,
            now=now,
            range_mode=range_mode,
        )
        if escape_exit:
            return escape_exit

        ma_gap_pips = 0.0
        if ma10 is not None and ma20 is not None:
            ma_gap_pips = abs(ma10 - ma20) / 0.01

        slope_support = (
            projection_fast is not None
            and projection_fast.gap_pips > 0.0
            and projection_fast.gap_slope_pips > 0.12
        )
        cross_support = (
            projection_primary is not None
            and projection_primary.gap_pips > 0.0
            and projection_primary.gap_slope_pips > 0.05
        )
        macd_cross_minutes = self._macd_cross_minutes(projection_fast, "long")

        value_cut = self._value_cut_exit(
            pocket=pocket,
            side="long",
            units=units,
            open_info=open_info,
            profit_pips=profit_pips,
            close_price=close_price,
            ema20=ema20,
            rsi=rsi,
            atr_pips=atr_pips,
            fac_m1=fac_m1,
            now=now,
            max_mfe=max_mfe_long,
            neg_exit_blocked=neg_exit_blocked,
        )
        if value_cut:
            return value_cut

        matured_macro = False
        loss_cap = None
        if pocket == "macro":
            if (
                reverse_signal
                and profit_pips >= 4.0
                and close_price is not None
                and ema20 is not None
                and close_price >= ema20 + 0.002
            ):
                if (
                    adx >= self._macro_trend_adx + 4
                    or ma_gap_pips <= self._macro_ma_gap
                    or slope_support
                    or cross_support
                ):
                    reverse_signal = None
            matured_macro = self._has_mature_trade(
                open_info, "long", now, self._macro_min_hold_minutes
            )
            loss_cap = self._macro_loss_cap(atr_pips, matured_macro)

        ema_gap_pips = None
        if close_price is not None and ema20 is not None:
            ema_gap_pips = (close_price - ema20) / 0.01

        if pocket == "macro":
            partial_units = self._trendma_partial_exit_units(
                open_info=open_info,
                side="long",
                units=units,
                profit_pips=profit_pips,
                adx=adx,
                rsi=rsi,
                projection_fast=projection_fast,
                atr_pips=atr_pips,
                loss_cap=loss_cap,
            )
            if partial_units:
                return ExitDecision(
                    pocket=pocket,
                    units=partial_units,
                    reason="trendma_partial",
                    tag="trendma-decay",
                    allow_reentry=True,
                )
            lock_reason = self._macro_profit_capture(
                open_info,
                "long",
                profit_pips,
                max_mfe_long if max_mfe_long is not None else profit_pips,
                now,
            )
            if lock_reason:
                return ExitDecision(
                    pocket=pocket,
                    units=-abs(units),
                    reason=lock_reason,
                    tag="macro-profit-lock",
                    allow_reentry=False,
                )
            if (
                close_price is not None
                and ema20 is not None
                and self._macro_slowdown_detected(
                    profit_pips=profit_pips,
                    adx=adx,
                    rsi=rsi,
                    projection_fast=projection_fast,
                    close_price=close_price,
                    ema20=ema20,
                    atr_pips=atr_pips,
                )
            ):
                partial_units = max(1000, (units // 2))
                if partial_units < units:
                    return ExitDecision(
                        pocket=pocket,
                        units=-partial_units,
                        reason="macro_slowdown",
                        tag="macro-slowdown",
                        allow_reentry=True,
                    )
            vol_partial = self._vol_partial_exit_units(
                pocket=pocket,
                side="long",
                units=units,
                profit_pips=profit_pips,
                atr_pips=atr_pips,
                ema_gap_pips=ema_gap_pips,
            )
            if vol_partial:
                return ExitDecision(
                    pocket=pocket,
                    units=vol_partial,
                    reason="macro_vol_partial",
                    tag="macro-vol-partial",
                    allow_reentry=True,
                )

        # Generic MFE-based partial/trail for breakout/pullback styles
        slope_hint = projection_fast.gap_slope_pips if projection_fast is not None else None
        mfe_partial = self._mfe_partial_units(
            pocket,
            units,
            profit_pips,
            atr_pips=atr_pips,
            slope_hint=slope_hint,
        )
        if mfe_partial:
            return ExitDecision(
                pocket=pocket,
                units=mfe_partial,
                reason="mfe_partial",
                tag=f"{pocket}-mfe-partial",
                allow_reentry=True,
            )
        if self._mfe_trail_hit(
            side="long",
            avg_price=avg_price,
            close_price=close_price,
            profit_pips=profit_pips,
        ):
            return ExitDecision(
                pocket=pocket,
                units=units,
                reason="mfe_trail",
                tag=f"{pocket}-mfe-trail",
                allow_reentry=False,
            )

        # Do not act on fresh macro trades unless conditions are clearly adverse
        if reverse_signal and pocket == "macro" and not range_mode:
            if not matured_macro:
                early_exit_ok = (
                    profit_pips <= -self._macro_loss_buffer
                    or (
                        (ma10 is not None and ma20 is not None and ma10 < ma20)
                        and adx <= (self._macro_trend_adx - 2)
                        and ma_gap_pips >= (self._macro_ma_gap + 1.0)
                    )
                )
                if not early_exit_ok:
                    reverse_signal = None

        # Night/event guard for short-lived pockets
        if event_soon and pocket in {"micro", "scalp"}:
            reason = "event_lock"
        elif reverse_signal:
            tech_ctx = self._exit_tech_context(fac_m1 or {}, "long")
            macro_skip = (
                pocket == "macro"
                and reverse_signal.get("confidence", 0) < self._macro_signal_threshold
                and profit_pips > -self._macro_loss_buffer
            )
            if not macro_skip:
                # Pattern-aware retest guard for macro: if price sits near fast MA
                # on lower TFs and M5/M10 slopes support a bounce, defer exit.
                if pocket == "macro":
                    if self._should_delay_macro_exit_for_retest(
                        side="long",
                        close_price=close_price,
                        ema20=ema20,
                        atr_pips=atr_pips,
                        fac_m1=fac_m1,
                    ):
                        return None
                # ヒステリシス: 小幅の含み損益域では逆方向シグナルだけでクローズしない
                micro_guard = (pocket == "micro" and -1.6 < profit_pips < 1.6)
                macro_guard = (
                    pocket == "macro"
                    and -self._macro_hysteresis_pips < profit_pips < self._macro_hysteresis_pips
                )
                if pocket in {"micro", "scalp"} and not range_mode:
                    # 市況に合わせたゲート: 最低ホールド、MFEリトレース、利益/損失バンド、クールダウン
                    if not self._has_mature_trade(open_info, "long", now, max(self._micro_min_hold_minutes, self._reverse_min_hold_sec / 60.0)):
                        return None
                    max_mfe = self._max_mfe_for_side(open_info, "long", candles, now)
                    if max_mfe is None or max_mfe < self._reverse_mfe_min:
                        return None
                    retrace = max(0.0, max_mfe - max(0.0, profit_pips or 0.0))
                    if retrace < self._reverse_mfe_ratio * max_mfe and profit_pips is not None and profit_pips > -self._micro_loss_grace_pips:
                        return None
                    if profit_pips is not None and (-self._reverse_loss_floor < profit_pips < self._reverse_profit_floor):
                        return None
                    # 強い順行モメンタムならデファー
                    try:
                        macd_hist = float(fac_m1.get("macd_hist") or 0.0)
                        plus_di = float(fac_m1.get("plus_di") or 0.0)
                        minus_di = float(fac_m1.get("minus_di") or 0.0)
                    except Exception:
                        macd_hist = 0.0
                        plus_di = minus_di = 0.0
                    if macd_hist > 0.0 and (plus_di - minus_di) > 0.0:
                        return None
                    # 反発余地: EMA近傍かつ傾きが順行なら見送り
                    bounce_ok = False
                    try:
                        ema_gap_pips = abs(float(close_price) - float(ema20)) / 0.01
                    except Exception:
                        ema_gap_pips = 99.0
                    try:
                        slope_fast = float(projection_fast.gap_slope_pips or 0.0) if projection_fast else 0.0
                    except Exception:
                        slope_fast = 0.0
                    buffer = max(self._reverse_bounce_buffer, (atr_pips or 0.0) * 0.3)
                    if ema_gap_pips <= buffer and slope_fast > 0.0:
                        bounce_ok = True
                    if bounce_ok:
                        return None
                    cluster_gap = float(tech_ctx.get("cluster_gap") or 0.0)
                    if cluster_gap > 0 and cluster_gap <= 2.8 and profit_pips is not None and profit_pips > self._soft_exit_floor:
                        partial_units = max(1000, int(abs(units) * max(self._reverse_partial_frac, 0.25)))
                        partial_units = min(partial_units, abs(units) - 1) if abs(units) > 1 else partial_units
                        if partial_units > 0 and partial_units < abs(units):
                            signed = -partial_units if side == "long" else partial_units
                            return ExitDecision(
                                pocket=pocket,
                                units=signed,
                                reason="cluster_partial",
                                tag=reverse_signal.get("tag", tag),
                                allow_reentry=True,
                            )
                    if tech_ctx.get("cloud_support") and cluster_gap > 4.0 and profit_pips is not None and profit_pips > -self._reverse_loss_floor:
                        return None
                    last_rev = self._reverse_ts.get((pocket, "long"))
                    if last_rev and (now - last_rev).total_seconds() < self._reverse_cooldown_sec:
                        return None
                    self._reverse_ts[(pocket, "long")] = now
                    # まず部分利確でリスク縮小し、揺れでの全撤退を防ぐ
                    partial_units = max(1000, int(abs(units) * self._reverse_partial_frac))
                    partial_units = min(partial_units, abs(units) - 1) if abs(units) > 1 else partial_units
                    if partial_units > 0 and partial_units < abs(units):
                        signed = -partial_units if side == "long" else partial_units
                        return ExitDecision(
                            pocket=pocket,
                            units=signed,
                            reason="reverse_signal_partial",
                            tag=reverse_signal.get("tag", tag),
                            allow_reentry=True,
                        )
                if micro_guard or macro_guard:
                    return None
                reason = "reverse_signal"
                tag = reverse_signal.get("tag", tag)
        elif pocket == "micro" and rsi >= 65:
            reason = "rsi_overbought"
        elif (
            pocket == "macro"
            and ma10 is not None
            and ma20 is not None
            and ma10 < ma20
            and adx <= self._macro_trend_adx
            and profit_pips <= -self._macro_loss_buffer
            and ma_gap_pips >= self._macro_ma_gap
        ):
            reason = "trend_reversal"
        elif pocket == "scalp" and close_price is not None and ema20 is not None:
            # スキャルは EMA 反転で即切りしない（fast_cut/SL/TP に任せる）
            pass
        elif (
            pocket == "micro"
            and profit_pips <= -self._micro_loss_grace_pips
            and self._micro_loss_ready(open_info, "long", now)
        ):
            reason = "micro_loss_guard"
        elif pocket == "macro" and loss_cap is not None and profit_pips <= -loss_cap:
            reason = "macro_loss_cap"
        elif (
            pocket == "macro"
            and avg_price
            and atr_pips is not None
            and profit_pips >= max(3.5, atr_pips * 0.9)
        ):
            trail_back = max(1.6, atr_pips * 0.45)
            trail_floor = avg_price + (profit_pips - trail_back) * 0.01
            if close_price is not None and close_price <= trail_floor:
                reason = "macro_atr_trail"
        elif self._should_exit_for_cross(
            pocket,
            "long",
            open_info,
            projection_primary,
            projection_fast,
            profit_pips,
            now,
            macd_cross_minutes,
            atr_pips,
        ):
            reason = "ma_cross_imminent"
        elif (
            pocket == "macro"
            and profit_pips >= max(4.2, atr_pips * 1.0)
            and close_price is not None
            and ema20 is not None
            and close_price <= ema20 - max(0.0010, (atr_pips * 0.25) / 100)
        ):
            reason = "macro_trend_fade"
        elif pocket == "micro" and self._micro_profit_exit_ready(
            side="long",
            profit_pips=profit_pips,
            rsi=rsi,
            close_price=close_price,
            ema20=ema20,
            projection_fast=projection_fast,
        ):
            reason = "micro_profit_guard"
        # レンジ中でもマクロの既存建玉を一律にクローズしない。
        # 早期利確/撤退（range_take_profit/range_stop）や逆方向シグナルのみで制御する。
        elif range_mode:
            if pocket == "macro":
                if self._has_mature_trade(open_info, "long", now, self._range_macro_grace_minutes):
                    return None
                tp = self._range_macro_take_profit
                hold = self._range_macro_hold
                stop = self._range_macro_stop
                if profit_pips >= tp:
                    reason = "range_take_profit"
                elif profit_pips > hold:
                    return None
                elif profit_pips <= -stop:
                    reason = "range_stop"
            else:
                if profit_pips >= 1.6:
                    reason = "range_take_profit"
                elif profit_pips > 0.4:
                    return None
                elif profit_pips <= -1.0:
                    reason = "range_stop"

        if not reason:
            low_reason, low_allow = self._micro_low_vol_exit_check(
                pocket,
                "long",
                open_info,
                profit_pips,
                max_mfe_long,
                atr_m1 or atr_primary,
                now,
                low_vol_profile,
                low_vol_quiet,
                news_status,
            )
            if low_reason:
                reason = low_reason
                allow_reentry = low_allow

        # MFE-based patience: if we've achieved decent favorable excursion,
        # avoid exiting on a mild pullback unless strong invalidation.
        if reason in self._mfe_sensitive_reasons and not range_mode:
            guard_threshold, guard_ratio = self._get_mfe_guard(pocket, atr_primary)
            max_mfe = self._max_mfe_for_side(open_info, "long", candles, now)
            if max_mfe is not None and max_mfe >= guard_threshold:
                retrace = max(0.0, max_mfe - max(0.0, profit_pips))
                if retrace <= guard_ratio * max_mfe and profit_pips > -self._macro_loss_buffer:
                    return None

        if reason == "trend_reversal":
            if not self._validate_trend_reversal(
                pocket,
                "long",
                story,
                close_price,
                candles,
                atr_primary=atr_primary,
                atr_m1=atr_m1,
            ):
                reason = ""
        elif reason == "macro_trail_hit":
            if not self._validate_trend_reversal(
                pocket,
                "long",
                story,
                close_price,
                candles,
                atr_primary=atr_primary,
                atr_m1=atr_m1,
                bias_only=True,
            ):
                reason = ""

        if (
            pocket == "macro"
            and reason
            and not range_mode
            and profit_pips > -self._macro_loss_buffer
        ):
            trend_supports = self._macro_trend_supports(
                "long", ma10, ma20, adx, slope_support, cross_support
            )
            mature = self._has_mature_trade(
                open_info, "long", now, self._macro_min_hold_minutes
            )
            if trend_supports and not mature and reason in {
                "reverse_signal",
                "trend_reversal",
                "ma_cross",
                "ma_cross_imminent",
            }:
                return None
            if profit_pips >= 1.6:
                reason = "range_take_profit"
            elif profit_pips > 0.4:
                return None
            elif profit_pips <= -1.0:
                reason = "range_stop"
        elif self._ema_release_ready(
            pocket=pocket,
            profit_pips=profit_pips,
            atr_pips=atr_pips,
            close_price=close_price,
            ema20=ema20,
        ):
            reason = "macro_ema_release"
        elif self._profit_snatch_ready(
            pocket=pocket,
            side="long",
            open_info=open_info,
            profit_pips=profit_pips,
            atr_pips=atr_pips,
            fac_m1=fac_m1,
            now=now,
        ):
            reason = "micro_profit_snatch"
            allow_reentry = True
        elif pocket in {"micro", "scalp"} and not reason:
            struct_partial = self._micro_struct_partial(
                pocket=pocket,
                side="long",
                units=units,
                profit_pips=profit_pips,
                close_price=close_price,
                fac_m1=fac_m1,
                atr_pips=atr_pips,
            )
            if struct_partial:
                return struct_partial
            be_guard = self._breakeven_guard(
                pocket=pocket,
                side="long",
                units=units,
                profit_pips=profit_pips,
                max_mfe=max_mfe_long,
                atr_pips=atr_pips,
                vol_5m=vol_5m,
                fac_m1=fac_m1,
            )
            if be_guard:
                return be_guard
        elif pocket == "macro" and not reason:
            partial = (
                self._nwave_partial_exit(
                    pocket=pocket,
                    side="long",
                    units=units,
                    profit_pips=profit_pips,
                    story=story,
                )
                or self._pivot_soft_partial(
                    pocket=pocket,
                    side="long",
                    units=units,
                    profit_pips=profit_pips,
                    price=close_price,
                    fac_m1=fac_m1,
                    atr_pips=atr_pips,
                )
                or self._candlestick_partial_exit(
                    pocket=pocket,
                    side="long",
                    units=units,
                    profit_pips=profit_pips,
                    story=story,
                    atr_pips=atr_pips,
                )
            )
            if partial:
                return partial

        # Structure-based kill-line: if macro and M5 pivot breaks beyond cushion, exit decisively
        if pocket == "macro" and not reason:
            kill_reason = self._structure_break_if_any(
                side="long", fac_m1=fac_m1, price=close_price, atr_pips=atr_pips
            )
            if kill_reason:
                reason = kill_reason

        if reason and pocket in {"micro", "scalp"} and reason != "event_lock":
            if neg_exit_blocked:
                return None

        # 最終手段: micro/scalp は強制理由以外ならまず部分利確で逃がす
        if (
            reason
            and pocket in {"micro", "scalp"}
            and reason not in self._force_exit_reasons
            and profit_pips is not None
            and profit_pips > self._soft_exit_floor
        ):
            partial_units = max(1000, int(abs(units) * self._soft_exit_frac))
            if partial_units >= abs(units):
                partial_units = abs(units) - 1 if abs(units) > 1 else abs(units)
            if partial_units > 0:
                signed = -partial_units if side == "long" else partial_units
                return ExitDecision(
                    pocket=pocket,
                    units=signed,
                    reason=f"{reason}_soft_partial",
                    tag="soft-partial",
                    allow_reentry=True,
                )

        if range_mode and reason == "reverse_signal":
            allow_reentry = False
        if reason and story:
            if not self._pattern_supports_exit(story, pocket, "long", reason, profit_pips):
                return None
            if not self._story_allows_exit(
                story,
                pocket,
                "long",
                reason,
                profit_pips,
                now,
                range_mode=range_mode,
            ):
                return None
        if reason == "reverse_signal":
            allow_reentry = False
        if not reason:
            return None

        close_units = self._compute_exit_units(
            pocket,
            "long",
            reason,
            units,
            stage_level,
            profile,
            range_mode=range_mode,
            profit_pips=profit_pips,
        )
        if close_units <= 0:
            return None

        self._record_exit_metric(
            pocket,
            "long",
            reason,
            profit_pips,
            story,
            range_mode,
            now,
        )

        trades = [tr for tr in (open_info.get("open_trades") or []) if tr.get("side") == "long"]
        trade = trades[0] if trades else None
        age_seconds = self._trade_age_seconds(trade, now) if trade else None
        hold_minutes = round(age_seconds / 60.0, 2) if age_seconds is not None else None
        logging.info(
            "[EXIT] pocket=%s side=long reason=%s profit=%.2fp hold=%smin close_units=%s range=%s stage=%d",
            pocket,
            reason,
            profit_pips,
            f"{hold_minutes:.2f}" if hold_minutes is not None else "n/a",
            close_units,
            range_mode,
            stage_level,
        )

        return ExitDecision(
            pocket=pocket,
            units=-abs(close_units),
            reason=reason,
            tag=tag,
            allow_reentry=allow_reentry,
        )

    def _evaluate_short(
        self,
        pocket: str,
        open_info: Dict,
        units: int,
        reverse_signal: Optional[Dict],
        event_soon: bool,
        rsi: float,
        ma10: float,
        ma20: float,
        adx: float,
        close_price: float,
        ema_fast: float,
        ema_primary: Optional[float],
        range_mode: bool,
        now: datetime,
        projection_primary: Optional[MACrossProjection],
        projection_fast: Optional[MACrossProjection],
        atr_pips: float,
        fac_m1: Dict,
        stage_tracker: Optional["StageTracker"],
        *,
        max_mfe_short: Optional[float] = None,
        max_mfe_long: Optional[float] = None,
        low_vol_profile: Optional[Dict[str, float]] = None,
        low_vol_quiet: bool = False,
        news_status: str = "quiet",
        m1_candles: Optional[List[Dict]] = None,
        profile: Optional[Dict] = None,
    ) -> Optional[ExitDecision]:

        allow_reentry = False
        reason = ""
        tag = f"{pocket}-short"
        story = None
        ema20 = ema_primary if ema_primary is not None else ema_fast
        candles = m1_candles or fac_m1.get("candles") or []
        atr_primary = atr_pips
        atr_m1 = atr_pips
        stage_level = 0
        profile = profile or {}
        avg_price = open_info.get("short_avg_price") or open_info.get("avg_price")
        try:
            vol_5m = float(fac_m1.get("vol_5m") or 0.0)
        except Exception:
            vol_5m = 0.0
        profit_pips = 0.0
        if avg_price and close_price:
            profit_pips = (avg_price - close_price) / 0.01
        orphan_guard = self._orphan_guard_exit(
            pocket=pocket,
            side="short",
            units=units,
            open_info=open_info,
            profit_pips=profit_pips,
            atr_pips=atr_pips,
            fac_m1=fac_m1,
            now=now,
        )
        if orphan_guard:
            return orphan_guard
        adaptive_exit = self._micro_adaptive_trail(
            pocket=pocket,
            side="short",
            units=units,
            open_info=open_info,
            profit_pips=profit_pips,
            close_price=close_price,
            atr_pips=atr_pips,
            fac_m1=fac_m1,
            now=now,
        )
        if adaptive_exit:
            return adaptive_exit
        drawdown_exit = self._stale_drawdown_exit(
            pocket=pocket,
            side="short",
            units=units,
            open_info=open_info,
            profit_pips=profit_pips,
            atr_pips=atr_pips,
            fac_m1=fac_m1,
            now=now,
        )
        if drawdown_exit:
            return drawdown_exit
        clamp_exit = self._loss_clamp_exit(
            pocket=pocket,
            side="short",
            units=units,
            profit_pips=profit_pips,
            open_info=open_info,
            atr_pips=atr_pips,
            fac_m1=fac_m1,
            now=now,
            close_price=close_price,
        )
        if clamp_exit:
            return clamp_exit
        neg_exit_blocked = self._negative_exit_blocked(
            pocket, open_info, "short", now, profit_pips, stage_tracker, atr_pips, fac_m1
        )
        target_bounds = self._entry_target_bounds(open_info, "short")
        if (
            reverse_signal
            and target_bounds
            and profit_pips is not None
            and profit_pips >= 0.0
            and profit_pips < target_bounds[0] * 0.75
        ):
            self._record_target_guard(
                pocket,
                "short",
                profit_pips,
                target_bounds,
                reverse_signal.get("tag") if reverse_signal else None,
            )
            reverse_signal = None

        escape_exit = self._dynamic_escape_exit(
            pocket=pocket,
            side="short",
            units=units,
            open_info=open_info,
            profit_pips=profit_pips,
            max_mfe=max_mfe_short,
            atr_pips=atr_pips,
            fac_m1=fac_m1,
            now=now,
            range_mode=range_mode,
        )
        if escape_exit:
            return escape_exit

        ma_gap_pips = 0.0
        if ma10 is not None and ma20 is not None:
            ma_gap_pips = abs(ma10 - ma20) / 0.01

        slope_support = (
            projection_fast is not None
            and projection_fast.gap_pips < 0.0
            and projection_fast.gap_slope_pips < -0.12
        )
        cross_support = (
            projection_primary is not None
            and projection_primary.gap_pips < 0.0
            and projection_primary.gap_slope_pips < -0.05
        )
        macd_cross_minutes = self._macd_cross_minutes(projection_fast, "short")

        value_cut = self._value_cut_exit(
            pocket=pocket,
            side="short",
            units=units,
            open_info=open_info,
            profit_pips=profit_pips,
            close_price=close_price,
            ema20=ema20,
            rsi=rsi,
            atr_pips=atr_pips,
            fac_m1=fac_m1,
            now=now,
            max_mfe=max_mfe_short,
            neg_exit_blocked=neg_exit_blocked,
        )
        if value_cut:
            return value_cut

        matured_macro = False
        loss_cap = None
        if pocket == "macro":
            if (
                reverse_signal
                and profit_pips >= 4.0
                and close_price is not None
                and ema20 is not None
                and close_price <= ema20 - 0.002
            ):
                if (
                    adx >= self._macro_trend_adx + 4
                    or ma_gap_pips <= self._macro_ma_gap
                    or slope_support
                    or cross_support
                ):
                    reverse_signal = None
            matured_macro = self._has_mature_trade(
                open_info, "short", now, self._macro_min_hold_minutes
            )
            loss_cap = self._macro_loss_cap(atr_pips, matured_macro)

        ema_gap_pips = None
        if close_price is not None and ema20 is not None:
            ema_gap_pips = (close_price - ema20) / 0.01

        if pocket == "macro":
            partial_units = self._trendma_partial_exit_units(
                open_info=open_info,
                side="short",
                units=units,
                profit_pips=profit_pips,
                adx=adx,
                rsi=rsi,
                projection_fast=projection_fast,
                atr_pips=atr_pips,
                loss_cap=loss_cap,
            )
            if partial_units:
                return ExitDecision(
                    pocket=pocket,
                    units=partial_units,
                    reason="trendma_partial",
                    tag="trendma-decay",
                    allow_reentry=True,
                )
            vol_partial = self._vol_partial_exit_units(
                pocket=pocket,
                side="short",
                units=units,
                profit_pips=profit_pips,
                atr_pips=atr_pips,
                ema_gap_pips=ema_gap_pips,
            )
            if vol_partial:
                return ExitDecision(
                    pocket=pocket,
                    units=vol_partial,
                    reason="macro_vol_partial",
                    tag="macro-vol-partial",
                    allow_reentry=True,
                )

        slope_hint = projection_fast.gap_slope_pips if projection_fast is not None else None
        if slope_hint is not None:
            slope_hint *= -1.0
        mfe_partial = self._mfe_partial_units(
            pocket,
            units,
            profit_pips,
            atr_pips=atr_pips,
            slope_hint=slope_hint,
        )
        if mfe_partial:
            return ExitDecision(
                pocket=pocket,
                units=mfe_partial,
                reason="mfe_partial",
                tag=f"{pocket}-mfe-partial",
                allow_reentry=True,
            )
        if self._mfe_trail_hit(
            side="short",
            avg_price=avg_price,
            close_price=close_price,
            profit_pips=profit_pips,
        ):
            return ExitDecision(
                pocket=pocket,
                units=units,
                reason="mfe_trail",
                tag=f"{pocket}-mfe-trail",
                allow_reentry=False,
            )

        if (
            pocket == "micro"
            and profit_pips <= -4.0
            and close_price is not None
            and ema_fast is not None
            and (
                close_price >= ema_fast + 0.0015
                or rsi >= 55
                or (ma10 is not None and ma20 is not None and ma10 > ma20)
            )
        ):
            reason = "micro_momentum_stop"
        elif event_soon and pocket in {"micro", "scalp"}:
            reason = "event_lock"
        elif reverse_signal:
            tech_ctx = self._exit_tech_context(fac_m1 or {}, "short")
            macro_skip = (
                pocket == "macro"
                and reverse_signal.get("confidence", 0) < self._macro_signal_threshold
                and profit_pips > -self._macro_loss_buffer
            )
            if not macro_skip:
                if pocket == "macro":
                    if self._should_delay_macro_exit_for_retest(
                        side="short",
                        close_price=close_price,
                        ema20=ema20,
                        atr_pips=atr_pips,
                        fac_m1=fac_m1,
                    ):
                        return None
                # ヒステリシス: 小幅の含み損益域では逆方向シグナルだけでクローズしない
                micro_guard = (pocket == "micro" and -1.6 < profit_pips < 1.6)
                macro_guard = (
                    pocket == "macro"
                    and -self._macro_hysteresis_pips < profit_pips < self._macro_hysteresis_pips
                )
                if pocket in {"micro", "scalp"} and not range_mode:
                    if not self._has_mature_trade(open_info, "short", now, max(self._micro_min_hold_minutes, self._reverse_min_hold_sec / 60.0)):
                        return None
                    max_mfe = self._max_mfe_for_side(open_info, "short", candles, now)
                    if max_mfe is None or max_mfe < self._reverse_mfe_min:
                        return None
                    retrace = max(0.0, max_mfe - max(0.0, profit_pips or 0.0))
                    if retrace < self._reverse_mfe_ratio * max_mfe and profit_pips is not None and profit_pips > -self._micro_loss_grace_pips:
                        return None
                    if profit_pips is not None and (-self._reverse_loss_floor < profit_pips < self._reverse_profit_floor):
                        return None
                    # 強い順行モメンタムならデファー
                    try:
                        macd_hist = float(fac_m1.get("macd_hist") or 0.0)
                        plus_di = float(fac_m1.get("plus_di") or 0.0)
                        minus_di = float(fac_m1.get("minus_di") or 0.0)
                    except Exception:
                        macd_hist = 0.0
                        plus_di = minus_di = 0.0
                    if macd_hist < 0.0 and (plus_di - minus_di) < 0.0:
                        return None
                    # パターンが順行なら逆行EXITをデファー
                    patterns = getattr(story, "pattern_summary", None) or {}
                    candle = patterns.get("candlestick") if isinstance(patterns, dict) else {}
                    c_bias = candle.get("bias")
                    try:
                        c_conf = float(candle.get("confidence", 0.0) or 0.0)
                    except Exception:
                        c_conf = 0.0
                    n_wave = patterns.get("n_wave") if isinstance(patterns, dict) else {}
                    n_bias = n_wave.get("direction") or n_wave.get("bias")
                    try:
                        n_conf = float(n_wave.get("confidence", 0.0) or 0.0)
                    except Exception:
                        n_conf = 0.0
                    if c_bias and c_conf >= 0.6:
                        if side == "long" and c_bias == "up":
                            return None
                        if side == "short" and c_bias == "down":
                            return None
                    if n_bias and n_conf >= 0.6:
                        if side == "long" and n_bias == "up":
                            return None
                        if side == "short" and n_bias == "down":
                            return None
                    # 反発余地: EMA近傍かつ傾きが順行なら見送り
                    bounce_ok = False
                    try:
                        ema_gap_pips = abs(float(close_price) - float(ema20)) / 0.01
                    except Exception:
                        ema_gap_pips = 99.0
                    try:
                        slope_fast = float(projection_fast.gap_slope_pips or 0.0) if projection_fast else 0.0
                    except Exception:
                        slope_fast = 0.0
                    buffer = max(self._reverse_bounce_buffer, (atr_pips or 0.0) * 0.3)
                    if ema_gap_pips <= buffer and slope_fast < 0.0:
                        bounce_ok = True
                    if bounce_ok:
                        return None
                    cluster_gap = float(tech_ctx.get("cluster_gap") or 0.0)
                    if cluster_gap > 0 and cluster_gap <= 2.8 and profit_pips is not None and profit_pips > self._soft_exit_floor:
                        partial_units = max(1000, int(abs(units) * max(self._reverse_partial_frac, 0.25)))
                        partial_units = min(partial_units, abs(units) - 1) if abs(units) > 1 else partial_units
                        if partial_units > 0 and partial_units < abs(units):
                            signed = -partial_units if side == "long" else partial_units
                            return ExitDecision(
                                pocket=pocket,
                                units=signed,
                                reason="cluster_partial",
                                tag=reverse_signal.get("tag", tag),
                                allow_reentry=True,
                            )
                    if tech_ctx.get("cloud_support") and cluster_gap > 4.0 and profit_pips is not None and profit_pips > -self._reverse_loss_floor:
                        return None
                    last_rev = self._reverse_ts.get((pocket, "short"))
                    if last_rev and (now - last_rev).total_seconds() < self._reverse_cooldown_sec:
                        return None
                    self._reverse_ts[(pocket, "short")] = now
                    partial_units = max(1000, int(abs(units) * self._reverse_partial_frac))
                    partial_units = min(partial_units, abs(units) - 1) if abs(units) > 1 else partial_units
                    if partial_units > 0 and partial_units < abs(units):
                        signed = -partial_units if side == "long" else partial_units
                        return ExitDecision(
                            pocket=pocket,
                            units=signed,
                            reason="reverse_signal_partial",
                            tag=reverse_signal.get("tag", tag),
                            allow_reentry=True,
                        )
                if micro_guard or macro_guard:
                    return None
                reason = "reverse_signal"
                tag = reverse_signal.get("tag", tag)
        elif pocket == "micro" and rsi <= 35:
            reason = "rsi_oversold"
        elif (
            pocket == "macro"
            and ma10 is not None
            and ma20 is not None
            and ma10 > ma20
            and adx <= self._macro_trend_adx
            and profit_pips <= -self._macro_loss_buffer
            and ma_gap_pips >= self._macro_ma_gap
        ):
            reason = "trend_reversal"
        elif (
            pocket == "micro"
            and profit_pips <= -self._micro_loss_grace_pips
            and self._micro_loss_ready(open_info, "short", now)
        ):
            reason = "micro_loss_guard"
        elif pocket == "macro" and loss_cap is not None and profit_pips <= -loss_cap:
            reason = "macro_loss_cap"
        elif pocket == "scalp" and close_price is not None and ema20 is not None:
            # スキャルは EMA 反転で即切りしない（fast_cut/SL/TP に任せる）
            pass
        elif (
            pocket == "macro"
            and avg_price
            and profit_pips >= max(6.0, atr_pips * 1.05)
        ):
            trail_back = max(2.8, atr_pips * 0.55)
            trail_ceiling = avg_price - (profit_pips - trail_back) * 0.01
            if close_price is not None and close_price >= trail_ceiling:
                reason = "macro_atr_trail"
        elif self._should_exit_for_cross(
            pocket,
            "short",
            open_info,
            projection_primary,
            projection_fast,
            profit_pips,
            now,
            macd_cross_minutes,
            atr_pips,
        ):
            reason = "ma_cross_imminent"
        elif pocket == "macro" and not range_mode:
            lock_reason = self._macro_profit_capture(
                open_info,
                "short",
                profit_pips,
                max_mfe_short,
                now,
            )
            if lock_reason:
                reason = lock_reason
                allow_reentry = False
        # レンジ中でもマクロの既存建玉を一律にクローズしない。
        # 早期利確/撤退（range_take_profit/range_stop）や逆方向シグナルのみで制御する。
        elif (
            pocket == "macro"
            and profit_pips >= max(6.5, atr_pips * 1.1)
            and close_price is not None
            and ema20 is not None
            and close_price >= ema20 + max(0.0012, (atr_pips * 0.3) / 100)
        ):
            reason = "macro_trend_fade"
        elif pocket == "micro" and self._micro_profit_exit_ready(
            side="short",
            profit_pips=profit_pips,
            rsi=rsi,
            close_price=close_price,
            ema20=ema20,
            projection_fast=projection_fast,
        ):
            reason = "micro_profit_guard"
        elif pocket in {"micro", "scalp"} and not reason:
            struct_partial = self._micro_struct_partial(
                pocket=pocket,
                side="short",
                units=units,
                profit_pips=profit_pips,
                close_price=close_price,
                fac_m1=fac_m1,
                atr_pips=atr_pips,
            )
            if struct_partial:
                return struct_partial
            be_guard = self._breakeven_guard(
                pocket=pocket,
                side="short",
                units=units,
                profit_pips=profit_pips,
                max_mfe=max_mfe_short,
                atr_pips=atr_pips,
                vol_5m=vol_5m,
                fac_m1=fac_m1,
            )
            if be_guard:
                return be_guard
        elif pocket == "macro" and not reason:
            partial = (
                self._nwave_partial_exit(
                    pocket=pocket,
                    side="short",
                    units=units,
                    profit_pips=profit_pips,
                    story=story,
                )
                or self._pivot_soft_partial(
                    pocket=pocket,
                    side="short",
                    units=units,
                    profit_pips=profit_pips,
                    price=close_price,
                    fac_m1=fac_m1,
                    atr_pips=atr_pips,
                )
                or self._candlestick_partial_exit(
                    pocket=pocket,
                    side="short",
                    units=units,
                    profit_pips=profit_pips,
                    story=story,
                    atr_pips=atr_pips,
                )
            )
            if partial:
                return partial
        elif range_mode:
            if pocket == "macro":
                if self._has_mature_trade(open_info, "short", now, self._range_macro_grace_minutes):
                    return None
                tp = self._range_macro_take_profit
                hold = self._range_macro_hold
                stop = self._range_macro_stop
                if profit_pips >= tp:
                    reason = "range_take_profit"
                elif profit_pips > hold:
                    return None
                elif profit_pips <= -stop:
                    reason = "range_stop"
            else:
                if profit_pips >= 1.6:
                    reason = "range_take_profit"
                elif profit_pips > 0.4:
                    return None
                elif profit_pips <= -1.0:
                    reason = "range_stop"

        if not reason:
            low_reason, low_allow = self._micro_low_vol_exit_check(
                pocket,
                "short",
                open_info,
                profit_pips,
                max_mfe_short,
                atr_m1 or atr_primary,
                now,
                low_vol_profile,
                low_vol_quiet,
                news_status,
            )
            if low_reason:
                reason = low_reason
                allow_reentry = low_allow

        if reason in self._mfe_sensitive_reasons and not range_mode:
            guard_threshold, guard_ratio = self._get_mfe_guard(pocket, atr_primary)
            max_mfe = self._max_mfe_for_side(open_info, "short", candles, now)
            if max_mfe is not None and max_mfe >= guard_threshold:
                retrace = max(0.0, max_mfe - max(0.0, -profit_pips))
                if retrace <= guard_ratio * max_mfe and profit_pips < self._macro_loss_buffer:
                    return None

        if reason == "trend_reversal":
            if not self._validate_trend_reversal(
                pocket,
                "short",
                story,
                close_price,
                candles,
                atr_primary=atr_primary,
                atr_m1=atr_m1,
            ):
                reason = ""
        elif reason == "macro_trail_hit":
            if not self._validate_trend_reversal(
                pocket,
                "short",
                story,
                close_price,
                candles,
                atr_primary=atr_primary,
                atr_m1=atr_m1,
                bias_only=True,
            ):
                reason = ""

        if (
            pocket == "macro"
            and reason
            and not range_mode
            and profit_pips > -self._macro_loss_buffer
        ):
            trend_supports = self._macro_trend_supports(
                "short", ma10, ma20, adx, slope_support, cross_support
            )
            mature = self._has_mature_trade(
                open_info, "short", now, self._macro_min_hold_minutes
            )
            if trend_supports and not mature and reason in {
                "reverse_signal",
                "trend_reversal",
                "ma_cross",
                "ma_cross_imminent",
            }:
                return None
            if profit_pips >= 1.6:
                reason = "range_take_profit"
            elif profit_pips > 0.4:
                return None
            elif profit_pips <= -1.0:
                reason = "range_stop"
        elif self._ema_release_ready(
            pocket=pocket,
            profit_pips=profit_pips,
            atr_pips=atr_pips,
            close_price=close_price,
            ema20=ema20,
        ):
            reason = "macro_ema_release"
        elif self._profit_snatch_ready(
            pocket=pocket,
            side="short",
            open_info=open_info,
            profit_pips=profit_pips,
            atr_pips=atr_pips,
            fac_m1=fac_m1,
            now=now,
        ):
            reason = "micro_profit_snatch"
            allow_reentry = True

        if pocket == "macro" and not reason:
            kill_reason = self._structure_break_if_any(
                side="short", fac_m1=fac_m1, price=close_price, atr_pips=atr_pips
            )
            if kill_reason:
                reason = kill_reason

        if reason and pocket in {"micro", "scalp"} and reason != "event_lock":
            if neg_exit_blocked:
                return None

        if range_mode and reason == "reverse_signal":
            allow_reentry = False
        if reason and story:
            if not self._pattern_supports_exit(story, pocket, "short", reason, profit_pips):
                return None
            if not self._story_allows_exit(
                story,
                pocket,
                "short",
                reason,
                profit_pips,
                now,
                range_mode=range_mode,
            ):
                return None
        if reason == "reverse_signal":
            allow_reentry = False
        if not reason:
            return None

        close_units = self._compute_exit_units(
            pocket,
            "short",
            reason,
            units,
            stage_level,
            profile,
            range_mode=range_mode,
            profit_pips=profit_pips,
        )
        if close_units <= 0:
            return None

        self._record_exit_metric(
            pocket,
            "short",
            reason,
            profit_pips,
            story,
            range_mode,
            now,
        )

        trades = [tr for tr in (open_info.get("open_trades") or []) if tr.get("side") == "short"]
        trade = trades[0] if trades else None
        age_seconds = self._trade_age_seconds(trade, now) if trade else None
        hold_minutes = round(age_seconds / 60.0, 2) if age_seconds is not None else None
        logging.info(
            "[EXIT] pocket=%s side=short reason=%s profit=%.2fp hold=%smin close_units=%s range=%s stage=%d",
            pocket,
            reason,
            profit_pips,
            f"{hold_minutes:.2f}" if hold_minutes is not None else "n/a",
            close_units,
            range_mode,
            stage_level,
        )

        return ExitDecision(
            pocket=pocket,
            units=abs(close_units),
            reason=reason,
            tag=tag,
            allow_reentry=allow_reentry,
        )

    # --- Pattern-aware helpers (macro) ---
    def _should_delay_macro_exit_for_retest(
        self,
        *,
        side: str,
        close_price: float,
        ema20: float,
        atr_pips: float,
        fac_m1: Dict,
    ) -> bool:
        """Return True to defer macro exit for a potential retest/bounce.

        Logic:
        - If price sits near fast MA band (M5/M10 via projection fast MA),
          and the corresponding slopes align with the position, hold.
        - Band width scales with ATR.
        """
        try:
            candles_m1 = fac_m1.get("candles") or []
            c5 = resample_candles_from_m1(candles_m1, 5)
            c10 = resample_candles_from_m1(candles_m1, 10)
            p5 = compute_ma_projection({"candles": c5}, timeframe_minutes=5.0) if len(c5) >= 30 else None
            p10 = compute_ma_projection({"candles": c10}, timeframe_minutes=10.0) if len(c10) >= 30 else None
        except Exception:
            return False

        dir_sign = 1.0 if side == "long" else -1.0
        slope_ok_5 = p5 and (p5.gap_slope_pips or 0.0) * dir_sign >= self._macro_retest_m5_slope
        slope_ok_10 = p10 and (p10.gap_slope_pips or 0.0) * dir_sign >= self._macro_retest_m10_slope
        if not (slope_ok_5 or slope_ok_10):
            return False

        band = max(self._macro_retest_band_base, (atr_pips or 0.0) * 0.25)
        # Prefer M5 fast MA proximity when available; fall back to ema20 (M1)
        near_fast_ok = False
        try:
            fast_approx = (p5.fast_ma if p5 and p5.fast_ma is not None else None)
            ref = fast_approx if fast_approx is not None else ema20
            near_fast_ok = abs(close_price - ref) / 0.01 <= band
        except Exception:
            near_fast_ok = abs(close_price - ema20) / 0.01 <= band
        return bool(near_fast_ok)

    def _structure_break_if_any(
        self,
        *,
        side: str,
        fac_m1: Dict,
        price: float,
        atr_pips: float,
    ) -> Optional[str]:
        """Detect recent M5 pivot break as decisive structure failure.

        Returns reason string when broken; otherwise None.
        """
        try:
            candles_m1 = fac_m1.get("candles") or []
            c5 = resample_candles_from_m1(candles_m1, 5)
        except Exception:
            return None
        if len(c5) < 9:
            return None

        low_key, high_key = ("low", "high")
        def _last_pivot(arr: List[Dict[str, float]], is_low: bool, width: int = 2, lookback: int = 14) -> Optional[float]:
            n = len(arr)
            start = max(2, n - lookback)
            for i in range(n - 3, start - 1, -1):
                try:
                    center = float(arr[i][low_key if is_low else high_key])
                except Exception:
                    continue
                ok = True
                for k in range(1, width + 1):
                    try:
                        left = float(arr[i - k][low_key if is_low else high_key])
                        right = float(arr[i + k][low_key if is_low else high_key])
                    except Exception:
                        ok = False
                        break
                    if is_low:
                        if not (center <= left and center <= right):
                            ok = False
                            break
                    else:
                        if not (center >= left and center >= right):
                            ok = False
                            break
                if ok:
                    return center
            return None

        cushion = max(0.6, (atr_pips or 0.0) * self._macro_struct_cushion)
        if side == "long":
            pivot = _last_pivot(c5, is_low=True)
            if pivot is not None and price <= pivot - cushion * 0.01:
                return "macro_struct_break"
        else:
            pivot = _last_pivot(c5, is_low=False)
            if pivot is not None and price >= pivot + cushion * 0.01:
                return "macro_struct_break"
        return None

    def _micro_loss_ready(
        self,
        open_info: Dict,
        side: str,
        now: datetime,
    ) -> bool:
        if self._micro_loss_hold_seconds <= 0:
            return True
        youngest = self._youngest_trade_age_seconds(open_info, side, now) or 0.0
        return youngest >= self._micro_loss_hold_seconds

    def _micro_struct_partial(
        self,
        *,
        pocket: str,
        side: str,
        units: int,
        profit_pips: Optional[float],
        close_price: Optional[float],
        fac_m1: Dict,
        atr_pips: Optional[float],
    ) -> Optional[ExitDecision]:
        """
        M1の直近高安を明確に割ったときに小さく利確して往復負けを防ぐ。
        - micro/scalpのみ
        - 利益が薄い/フラット付近でのみ発動
        - 全撤退ではなく部分利確
        """
        if pocket not in {"micro", "scalp"}:
            return None
        if profit_pips is None or profit_pips < 0.0:
            return None
        if close_price is None or units == 0:
            return None

        candles = fac_m1.get("candles") or []
        if len(candles) < 8:
            return None
        highs = [self._safe_float(c.get("high")) for c in candles[-14:]]
        lows = [self._safe_float(c.get("low")) for c in candles[-14:]]
        highs = [h for h in highs if h is not None]
        lows = [l for l in lows if l is not None]
        if not highs or not lows:
            return None
        recent_high = max(highs)
        recent_low = min(lows)
        cushion = max(0.08, min(0.25, (atr_pips or 2.0) * 0.08))
        profit_cap = max(2.8, (atr_pips or 2.0) * 1.4)

        if profit_pips > profit_cap:
            return None

        if side == "long":
            gap = (close_price - recent_low) / 0.01
            if gap > cushion:
                return None
        else:
            gap = (recent_high - close_price) / 0.01
            if gap > cushion:
                return None

        cut_units = max(1000, abs(units) // 3)
        if cut_units <= 0 or cut_units >= abs(units):
            return None

        signed = -cut_units if side == "long" else cut_units
        return ExitDecision(
            pocket=pocket,
            units=signed,
            reason="micro_struct_partial",
            tag="struct-partial",
            allow_reentry=True,
        )

    def _breakeven_guard(
        self,
        *,
        pocket: str,
        side: str,
        units: int,
        profit_pips: Optional[float],
        max_mfe: Optional[float],
        atr_pips: Optional[float],
        vol_5m: Optional[float] = None,
        fac_m1: Optional[Dict] = None,
        fac_m5: Optional[Dict] = None,
        fac_h1: Optional[Dict] = None,
        fac_h4: Optional[Dict] = None,
    ) -> Optional[ExitDecision]:
        """
        Once favorable excursion is achieved, avoid letting it slip to loss.
        Applies to micro/scalp only; prefers partial close.
        """
        if pocket not in {"micro", "scalp"}:
            return None
        if profit_pips is None or max_mfe is None:
            return None
        trigger = self._be_guard_trigger
        floor = self._be_guard_floor
        min_loss = self._be_guard_min_loss
        frac = self._be_guard_frac

        tech_ctx = self._exit_tech_context(fac_m1 or {}, side, fac_m5=fac_m5, fac_h1=fac_h1, fac_h4=fac_h4)
        cluster_gap = float(tech_ctx.get("cluster_gap") or 0.0)
        cloud_pos = float(tech_ctx.get("cloud_pos") or 0.0)
        cloud_support = bool(tech_ctx.get("cloud_support"))
        in_cloud = bool(tech_ctx.get("in_cloud"))
        mtf_m5 = float(tech_ctx.get("mtf_m5") or 0.0)
        mtf_h1 = float(tech_ctx.get("mtf_h1") or 0.0)
        mtf_h4 = float(tech_ctx.get("mtf_h4") or 0.0)

        atr_val = atr_pips or 0.0
        vol_val = vol_5m or 0.0
        if atr_val <= 2.5 or vol_val <= 0.8:
            trigger = max(1.0, trigger - 0.4)
            floor += 0.1
            frac = min(0.9, frac + 0.1)
        elif atr_val >= 6.0 or vol_val >= 3.0:
            trigger += 0.4
            frac = max(0.25, frac - 0.1)
        if cluster_gap > 0:
            if cluster_gap < 3.0:
                trigger = max(0.8, trigger - 0.4)
                floor += 0.15
                frac = min(0.95, frac + 0.12)
            elif cluster_gap > 7.0:
                trigger += 0.3
                frac = max(0.2, frac - 0.08)
        if cloud_support:
            trigger += 0.2
            frac = max(0.25, frac - 0.08)
        elif cloud_pos != 0.0:
            trigger = max(0.8, trigger - 0.2)
            frac = min(0.95, frac + 0.08)
        if in_cloud and tech_ctx.get("vol_low"):
            floor += 0.1
        mtf_dir = max(mtf_m5, mtf_h1, mtf_h4)
        if mtf_dir >= 0.6:
            trigger += 0.25
            frac = max(0.2, frac - 0.08)
        if mtf_dir <= 0.0:
            trigger = max(0.7, trigger - 0.35)
            frac = min(0.95, frac + 0.1)

        if max_mfe < trigger:
            return None
        if profit_pips > floor:
            return None
        if profit_pips < min_loss:
            return None
        frac = max(0.2, min(0.9, frac))
        if atr_val <= 1.2:
            frac *= 0.85
        pattern_ctx = self._pattern_bias(None, side=side)
        try:
            story = fac_m1.get("story")
        except Exception:
            story = None
        if story:
            pattern_ctx = self._pattern_bias(story, side=side)
        bias = pattern_ctx.get("bias")
        conf = float(pattern_ctx.get("conf") or 0.0)
        if bias in {"with_candle", "with_nwave", "with_both"} and conf >= 0.6:
            trigger += 0.2
            frac = max(0.2, frac - 0.08)
        elif bias in {"against_candle", "against_nwave"} and conf >= 0.6:
            trigger = max(0.8, trigger - 0.3)
            frac = min(0.95, frac + 0.12)
        cut_units = max(1000, int(abs(units) * frac))
        if cut_units <= 0:
            return None
        if cut_units >= abs(units):
            cut_units = abs(units)
        signed = -cut_units if side == "long" else cut_units
        return ExitDecision(
            pocket=pocket,
            units=signed,
            reason="breakeven_guard",
            tag="be-guard",
            allow_reentry=True,
        )

    def _nwave_partial_exit(
        self,
        *,
        pocket: str,
        side: str,
        units: int,
        profit_pips: Optional[float],
        story: Optional[ChartStorySnapshot],
    ) -> Optional[ExitDecision]:
        """
        H4 N波が逆行している場合に、マクロポジションを小さく利確してリスクを落とす。
        """
        if pocket != "macro" or story is None or profit_pips is None or units == 0:
            return None
        patterns = getattr(story, "pattern_summary", None) or {}
        n_wave = patterns.get("n_wave") or {}
        bias = n_wave.get("direction") or n_wave.get("bias")
        try:
            conf = float(n_wave.get("confidence", 0.0) or 0.0)
        except Exception:
            conf = 0.0
        if conf < 0.6 or bias is None:
            return None
        if side == "long" and bias != "down":
            return None
        if side == "short" and bias != "up":
            return None
        if profit_pips < 0.6 or profit_pips > 6.0:
            return None
        cut_units = max(1000, int(abs(units) * 0.33))
        if cut_units <= 0 or cut_units >= abs(units):
            return None
        signed = -cut_units if side == "long" else cut_units
        return ExitDecision(
            pocket=pocket,
            units=signed,
            reason="nwave_partial",
            tag="nwave-partial",
            allow_reentry=True,
        )

    def _pivot_soft_partial(
        self,
        *,
        pocket: str,
        side: str,
        units: int,
        profit_pips: Optional[float],
        price: float,
        fac_m1: Dict,
        atr_pips: Optional[float],
    ) -> Optional[ExitDecision]:
        """
        M5ピボット手前で先に一部利確して、キルライン到達時の全クローズを緩和する。
        マクロのみ、薄利〜中利幅で発動。
        """
        if pocket != "macro":
            return None
        if profit_pips is None or profit_pips < 0.8 or profit_pips > 6.5:
            return None
        try:
            c5 = resample_candles_from_m1(fac_m1.get("candles") or [], 5)
        except Exception:
            return None
        if len(c5) < 9:
            return None

        low_key, high_key = ("low", "high")

        def _last_pivot(arr: List[Dict[str, float]], is_low: bool, width: int = 2, lookback: int = 14) -> Optional[float]:
            n = len(arr)
            start = max(2, n - lookback)
            for i in range(n - 3, start - 1, -1):
                try:
                    center = float(arr[i][low_key if is_low else high_key])
                except Exception:
                    continue
                ok = True
                for k in range(1, width + 1):
                    try:
                        left = float(arr[i - k][low_key if is_low else high_key])
                        right = float(arr[i + k][low_key if is_low else high_key])
                    except Exception:
                        ok = False
                        break
                    if is_low:
                        if not (center <= left and center <= right):
                            ok = False
                            break
                    else:
                        if not (center >= left and center >= right):
                            ok = False
                            break
                if ok:
                    return center
            return None

        cushion = max(0.5, (atr_pips or 0.0) * self._macro_struct_cushion)
        soft_band = cushion * 0.5
        pivot = _last_pivot(c5, is_low=(side == "long"))
        if pivot is None:
            return None
        gap_pips = (price - pivot) / 0.01 if side == "long" else (pivot - price) / 0.01
        if gap_pips < 0:
            # already breached; let kill-line handle
            return None
        if gap_pips > soft_band:
            return None

        cut_units = max(1000, int(abs(units) * 0.33))
        if cut_units <= 0 or cut_units >= abs(units):
            return None
        signed = -cut_units if side == "long" else cut_units
        return ExitDecision(
            pocket=pocket,
            units=signed,
            reason="pivot_partial",
            tag="pivot-partial",
            allow_reentry=True,
        )

    def _candlestick_partial_exit(
        self,
        *,
        pocket: str,
        side: str,
        units: int,
        profit_pips: Optional[float],
        story: Optional[ChartStorySnapshot],
        atr_pips: Optional[float],
    ) -> Optional[ExitDecision]:
        """
        H1ローソク足の強い逆行パターンが出たときに、薄利〜小利幅で部分利確する。
        順行パターンでは発火しない。
        """
        if story is None or pocket not in {"macro", "micro", "scalp"}:
            return None
        patterns = getattr(story, "pattern_summary", None) or {}
        candle = patterns.get("candlestick") or {}
        bias = candle.get("bias")
        if bias is None or profit_pips is None or profit_pips < 0.4:
            return None
        try:
            conf = float(candle.get("confidence", 0.0) or 0.0)
        except Exception:
            conf = 0.0
        if conf < 0.6:
            return None
        if side == "long" and bias != "down":
            return None
        if side == "short" and bias != "up":
            return None
        profit_cap = min(2.8, max(1.4, (atr_pips or 2.0) * 1.3))
        if profit_pips > profit_cap:
            return None
        frac = 0.5 if pocket in {"micro", "scalp"} else 0.33
        cut_units = max(1000, int(abs(units) * frac))
        if cut_units <= 0 or cut_units >= abs(units):
            return None
        signed = -cut_units if side == "long" else cut_units
        return ExitDecision(
            pocket=pocket,
            units=signed,
            reason="candle_partial",
            tag="candle-partial",
            allow_reentry=True,
        )

    def _value_cut_exit(
        self,
        *,
        pocket: str,
        side: str,
        units: int,
        open_info: Dict,
        profit_pips: float,
        close_price: float,
        ema20: Optional[float],
        rsi: float,
        atr_pips: float,
        fac_m1: Dict,
        now: datetime,
        max_mfe: Optional[float],
        neg_exit_blocked: bool,
    ) -> Optional[ExitDecision]:
        """
        Dynamically close when上位足節目割れ+RSI 極端＋十分なMFE後の戻り。
        - micro/scalpのみ
        - 一定ホールド後に、pivotブレイクかつRSI極端、かつ MFEドローダウンが大きいときに発火
        - マイナスでも「価値あるカット」(MFE済み→戻し) のみに限定
        """
        if pocket not in {"micro", "scalp"}:
            return None
        if neg_exit_blocked:
            return None
        if close_price is None:
            return None
        if profit_pips is None or profit_pips < 0.0:
            return None  # プラス圏のみで発火

        candles = fac_m1.get("candles") or []
        recent = candles[-20:] if len(candles) >= 5 else candles
        if len(recent) < 5:
            return None

        highs = [float(_candle_high(c)) for c in recent if _candle_high(c) is not None]
        lows = [float(_candle_low(c)) for c in recent if _candle_low(c) is not None]
        if not highs or not lows:
            return None
        recent_high = max(highs)
        recent_low = min(lows)

        min_hold = self._scalp_min_hold_seconds if pocket == "scalp" else self._micro_min_hold_seconds
        youngest = self._youngest_trade_age_seconds(open_info, side, now) or 0.0
        if youngest < min_hold:
            return None

        # MFEロック: 一定の含み益が乗ったら部分利確＋建値近辺にロック相当の動き（部分クローズでリスク縮小）
        lock_gate = max(2.0, (atr_pips or 2.0) * 0.9)
        if max_mfe is not None and max_mfe >= lock_gate and profit_pips is not None and profit_pips >= 0.4:
            cut_units = -abs(max(units // 2, 1000)) if side == "long" else abs(max(units // 2, 1000))
            return ExitDecision(
                pocket=pocket,
                units=cut_units,
                reason="value_lock_be",
                tag="value-lock",
                allow_reentry=False,
            )

        mfe_gate = max(2.5, (atr_pips or 1.8) * 0.8)
        drawdown_gate = max(1.5, (atr_pips or 1.8) * 0.6)
        loss_gate = max(2.0, (atr_pips or 1.8) * 0.8)

        # 価格が直近高安を明確に割ったか（節目ブレイク）
        pivot_break = False
        if side == "long":
            gap = (close_price - recent_low) / 0.01
            pivot_break = gap <= 0.10  # より深いブレイクのみ反応
        else:
            gap = (recent_high - close_price) / 0.01
            pivot_break = gap <= 0.10

        rsi_extreme = (side == "long" and rsi <= 38.0) or (side == "short" and rsi >= 62.0)

        if not (pivot_break and rsi_extreme):
            return None

        drawdown_ok = False
        # 一定の含み益が乗るまでは value_cut を発動しない（プラス圏限定）
        if max_mfe is None or max_mfe < mfe_gate:
            return None

        drawdown_from_peak = max_mfe - max(profit_pips, 0.0)
        if drawdown_from_peak >= drawdown_gate or profit_pips <= -loss_gate:
            drawdown_ok = True

        if not drawdown_ok:
            return None

        cut_units = -abs(units) if side == "long" else abs(units)
        # 直後の再突入を抑制するため、価格挙動ベースの短時間ブロックを設定（強い逆行ほど長く）
        try:
            import execution.strategy_guard as strategy_guard

            momentum = 0.0
            try:
                momentum = float(fac_m1.get("momentum") or 0.0)
            except Exception:
                momentum = 0.0
            speed_factor = min(2.0, 1.0 + abs(momentum) / 0.02)
            base_sec = 45 if pocket == "scalp" else 60
            duration = int(base_sec * speed_factor)
            strategy_guard.set_block(f"{pocket}_value_cut", duration, "value_cut_cooldown")
        except Exception:
            pass

        return ExitDecision(
            pocket=pocket,
            units=cut_units,
            reason="value_cut_pivot_rsi",
            tag="value-cut",
            allow_reentry=False,  # 直後の再エントリーでドローダウンを広げない
        )

    def _micro_profit_exit_ready(
        self,
        *,
        side: str,
        profit_pips: float,
        rsi: float,
        close_price: float,
        ema20: float,
        projection_fast: Optional[MACrossProjection],
    ) -> bool:
        if profit_pips is None:
            return False

        slope = None
        if projection_fast is not None:
            try:
                slope = float(projection_fast.gap_slope_pips or 0.0)
            except Exception:
                slope = None

        # 伸ばし優先: 閾値をやや広げ、強い逆行シグナルが複合したときだけ発火
        soft_thr = self._micro_profit_soft + 0.3
        hard_thr = self._micro_profit_hard + 0.5
        # Allow winners to stretch a bit if slopeは順方向に傾いている
        if slope is not None:
            if side == "long" and slope > 0.12:
                soft_thr += 0.2
                hard_thr += 0.4
            elif side == "short" and slope < -0.12:
                soft_thr += 0.2
                hard_thr += 0.4
            elif side == "long" and slope < -0.05:
                hard_thr = max(hard_thr - 0.2, self._micro_profit_soft)
            elif side == "short" and slope > 0.05:
                hard_thr = max(hard_thr - 0.2, self._micro_profit_soft)

        hard = profit_pips >= hard_thr
        soft = profit_pips >= soft_thr
        if not (hard or soft):
            return False

        ema_trigger = False
        if close_price is not None and ema20 is not None and self._micro_profit_ema_buffer > 0:
            gap = close_price - ema20
            if side == "long" and gap <= -self._micro_profit_ema_buffer:
                ema_trigger = True
            elif side == "short" and gap >= self._micro_profit_ema_buffer:
                ema_trigger = True

        rsi_trigger = False
        try:
            rsi_val = float(rsi)
        except (TypeError, ValueError):
            rsi_val = None
        if rsi_val is not None:
            if side == "long" and rsi_val <= self._micro_profit_rsi_release_long:
                rsi_trigger = True
            elif side == "short" and rsi_val >= self._micro_profit_rsi_release_short:
                rsi_trigger = True

        slope_trigger = False
        if slope is not None and self._micro_profit_slope_min > 0:
            if side == "long" and slope <= -self._micro_profit_slope_min:
                slope_trigger = True
            elif side == "short" and slope >= self._micro_profit_slope_min:
                slope_trigger = True

        triggers = [ema_trigger, rsi_trigger, slope_trigger]
        trigger_count = sum(1 for t in triggers if t)

        # より強い逆行シグナルの組み合わせを要求
        if hard and trigger_count >= 2:
            return True
        if soft and ema_trigger and trigger_count >= 2:
            return True
        return False

    def _should_exit_for_cross(
        self,
        pocket: str,
        side: str,
        open_info: Dict,
        projection_primary: Optional[MACrossProjection],
        projection_fast: Optional[MACrossProjection],
        profit_pips: float,
        now: datetime,
        macd_cross_minutes: Optional[float],
        atr_pips: float,
    ) -> bool:
        # スキャル/マイクロは fast_cut 系の早期撤退に寄せる。
        # MA クロスは頻発しノイズになりやすいため、ここでは抑制する。
        if pocket in {"micro", "scalp"}:
            return False

        # For macro positions, prefer primary (H4) projection to avoid
        # churning exits on transient M1 flickers. For other pockets,
        # keep fast (M1) responsive.
        projection = (
            projection_primary if pocket == "macro" else (projection_fast or projection_primary)
        )
        if projection is None:
            return False

        gap = projection.gap_pips
        if side == "long" and gap < 0.0:
            return True
        if side == "short" and gap > 0.0:
            return True

        slope_source = (
            projection_primary if pocket == "macro" else (projection_fast or projection_primary)
        )
        slope = slope_source.gap_slope_pips if slope_source else 0.0
        if macd_cross_minutes is None:
            if side == "long" and slope >= 0.0:
                return False
            if side == "short" and slope <= 0.0:
                return False

        threshold = 3.5
        matured = False
        loss_guard = 0.0
        small_profit_guard = 0.0
        if pocket == "macro":
            matured = self._has_mature_trade(
                open_info, side, now, self._macro_min_hold_minutes
            )
            threshold = 7.0 if matured else 4.8
            atr_ref = float(atr_pips or 0.0)
            if atr_ref <= 0.0:
                atr_ref = 8.0
            loss_guard = atr_ref * (0.16 if matured else 0.12)
            loss_guard = max(0.9, min(1.8, loss_guard))
            small_profit_guard = max(0.5, min(1.4, loss_guard * 0.75))
        elif pocket == "scalp":
            threshold = 2.2

        candidates: List[float] = []
        if projection.projected_cross_minutes is not None:
            candidates.append(projection.projected_cross_minutes)
        if macd_cross_minutes is not None:
            candidates.append(macd_cross_minutes)
        # When no reliable cross projection is available, fall back to
        # simple loss/take-profit guards. For macro positions, only stop
        # on loss beyond the guard, or realize a small profit when it
        # exceeds the small_profit_guard. Do NOT exit simply because the
        # profit is small or slightly negative.
        if not candidates:
            if pocket == "macro":
                if profit_pips <= -loss_guard:
                    return True  # stop loss guard
                if profit_pips >= small_profit_guard:
                    return True  # small take-profit
            return False
        cross_minutes = min(candidates)

        if cross_minutes > threshold:
            if pocket == "macro" and (
                profit_pips <= -loss_guard or profit_pips >= small_profit_guard
            ):
                return True
            return False

        if pocket == "macro":
            if not matured:
                return profit_pips <= -(loss_guard * 1.1)
            if profit_pips <= -loss_guard:
                return True  # stop loss
            if profit_pips >= small_profit_guard:
                return True  # small take-profit once threshold reached
            return False

        if profit_pips >= 0.8:
            return True
        if pocket == "macro" and not matured and cross_minutes <= threshold / 2.0:
            return False
        if cross_minutes <= threshold / 2.0:
            return True
        if (
            macd_cross_minutes is not None
            and macd_cross_minutes <= threshold / 2.0
            and not (pocket == "macro" and not matured)
        ):
            return True
        return False

    @staticmethod
    def _macd_cross_minutes(
        projection: Optional[MACrossProjection],
        side: str,
    ) -> Optional[float]:
        if (
            projection is None
            or projection.macd_pips is None
            or projection.macd_slope_pips is None
        ):
            return None
        macd = projection.macd_pips
        slope = projection.macd_slope_pips
        if side == "long":
            if macd <= 0.0 and slope <= 0.0:
                return 0.0
            if macd > 0.0 and slope < 0.0 and projection.macd_cross_minutes is not None:
                return projection.macd_cross_minutes
        else:
            if macd >= 0.0 and slope >= 0.0:
                return 0.0
            if macd < 0.0 and slope > 0.0 and projection.macd_cross_minutes is not None:
                return projection.macd_cross_minutes
        return None

    @staticmethod
    def _parse_candle_time(raw: Optional[str]) -> Optional[datetime]:
        if not raw:
            return None
        t = raw.strip()
        try:
            if t.endswith("Z"):
                t = t[:-1] + "+00:00"
            if "." in t and "+" not in t:
                head, frac = t.split(".", 1)
                frac = "".join(ch for ch in frac if ch.isdigit())[:6]
                t = f"{head}.{frac}+00:00"
            elif "+" not in t:
                t = f"{t}+00:00"
            return datetime.fromisoformat(t)
        except Exception:
            try:
                base = t.split(".", 1)[0].rstrip("Z") + "+00:00"
                return datetime.fromisoformat(base)
            except Exception:
                return None

    def _max_mfe_for_side(
        self,
        open_info: Dict,
        side: str,
        m1_candles: List[Dict],
        now: datetime,
    ) -> Optional[float]:
        trades = [
            tr for tr in (open_info.get("open_trades") or []) if tr.get("side") == side
        ]
        if not trades or not m1_candles:
            return None
        # Prepare candle tuples (ts, h, l)
        candles: List[Tuple[datetime, float, float]] = []
        for c in m1_candles:
            ts = self._parse_candle_time(c.get("timestamp"))
            if not ts:
                continue
            try:
                h = float(c.get("high"))
                l = float(c.get("low"))
            except Exception:
                continue
            candles.append((ts, h, l))
        if not candles:
            return None
        max_mfe = 0.0
        for tr in trades:
            ep = tr.get("price")
            ot = self._parse_open_time(tr.get("open_time"))
            if ep is None or ot is None:
                continue
            for ts, h, l in candles:
                if ts < ot or ts > now:
                    continue
                if side == "long":
                    fav = (h - ep) / 0.01
                else:
                    fav = (ep - l) / 0.01
                if fav > max_mfe:
                    max_mfe = fav
        return round(max_mfe, 2)

    def _escape_profile(self, *, pocket: str, atr_pips: Optional[float], fac_m1: Dict, range_mode: bool) -> Dict:
        """Classify market state for escape logic; never widens profit targets."""
        try:
            bbw = float(fac_m1.get("bbw") or 0.0)
        except Exception:
            bbw = 0.0
        try:
            vol5m = float(fac_m1.get("vol_5m") or 0.0)
        except Exception:
            vol5m = 0.0
        try:
            momentum = float(fac_m1.get("momentum") or 0.0)
        except Exception:
            momentum = 0.0
        atr_val = float(atr_pips or 0.0)
        if atr_val <= 0.0:
            try:
                atr_val = float(fac_m1.get("atr") or 0.0) * 100.0
            except Exception:
                atr_val = 0.0
        quiet = range_mode or (
            (atr_val > 0 and atr_val <= self._escape_atr_quiet)
            or (bbw > 0 and bbw <= self._escape_bbw_quiet)
            or (vol5m > 0 and vol5m <= self._escape_vol_quiet)
        )
        hot = (atr_val >= self._escape_atr_hot) or (bbw >= self._escape_bbw_hot)
        state = "quiet" if quiet else ("hot" if hot else "normal")
        return {
            "state": state,
            "atr": atr_val,
            "bbw": bbw,
            "vol5m": vol5m,
            "momentum": momentum,
        }

    def _dynamic_escape_exit(
        self,
        *,
        pocket: str,
        side: str,
        units: int,
        open_info: Dict,
        profit_pips: float,
        max_mfe: Optional[float],
        atr_pips: Optional[float],
        fac_m1: Dict,
        now: datetime,
        range_mode: bool,
    ) -> Optional[ExitDecision]:
        """Shrink TP/SL dynamically in quiet/range; avoid interfering with runners."""
        if pocket not in {"micro", "scalp", "scalp_fast"}:
            return None
        profile = self._escape_profile(pocket=pocket, atr_pips=atr_pips, fac_m1=fac_m1, range_mode=range_mode)
        state = profile["state"]
        if state == "hot":
            return None  # let profit extension logic run
        if profit_pips is None:
            return None
        if profit_pips < 0.2:
            return None
        age_sec = self._youngest_trade_age_seconds(open_info, side, now) or 0.0
        if age_sec < max(self._escape_min_hold, self._scalp_min_hold_seconds if pocket.startswith("scalp") else self._micro_min_hold_seconds):
            return None
        max_seen = max_mfe if max_mfe is not None else profit_pips
        drawdown = max(0.0, max_seen - profit_pips)
        # Quiet/range: tighten aggressively; normal: only if drawdown from a small peak
        if state == "quiet":
            hit_tp = profit_pips >= self._escape_quiet_tp
            draw_gate = max(self._escape_quiet_draw_min, max_seen * self._escape_quiet_draw_ratio)
            momentum_cut = profile["momentum"] <= self._escape_momentum_cut
            if hit_tp or (max_seen >= self._escape_quiet_tp and drawdown >= draw_gate) or (momentum_cut and profit_pips >= self._escape_quiet_tp * 0.8):
                cut_units = -abs(units) if side == "long" else abs(units)
                return ExitDecision(
                    pocket=pocket,
                    units=cut_units,
                    reason=f"escape_quiet_{'tp' if hit_tp else 'draw'}",
                    tag="escape-quiet",
                    allow_reentry=False,
                )
        else:  # normal
            if max_seen <= 0.0:
                return None
            # Protect small peaks; do not trigger once profit has stretched enough (leave to other profit locks)
            if max_seen <= 3.0:
                draw_gate = max(self._escape_quiet_draw_min, max_seen * 0.6)
                if drawdown >= draw_gate and profit_pips >= 0.4:
                    cut_units = -abs(units) if side == "long" else abs(units)
                    return ExitDecision(
                        pocket=pocket,
                        units=cut_units,
                        reason="escape_normal_draw",
                        tag="escape-normal",
                        allow_reentry=False,
                    )
        return None

    def _micro_low_vol_exit_check(
        self,
        pocket: str,
        side: str,
        open_info: Dict,
        profit_pips: float,
        max_mfe: Optional[float],
        atr_hint: Optional[float],
        now: datetime,
        low_vol_profile: Optional[Dict[str, float]],
        low_vol_quiet: bool,
        news_status: str,
    ) -> tuple[Optional[str], bool]:
        if not self._low_vol_enabled:
            return None, False
        if pocket != "micro":
            return None, False
        profile = low_vol_profile or {}
        low_active = bool(profile.get("low_vol"))
        low_like = bool(profile.get("low_vol_like"))
        if not (low_active or low_like or low_vol_quiet):
            return None, False
        if news_status == "active":
            return None, False
        trades = [
            tr for tr in (open_info.get("open_trades") or []) if tr.get("side") == side
        ]
        trade = trades[0] if trades else None
        age_sec = self._trade_age_seconds(trade, now) if trade else None
        if age_sec is None:
            return None, False
        atr = max(0.6, float(atr_hint or profile.get("atr") or 2.0))
        key = (pocket, side)
        score = float(profile.get("score", 0.0) or 0.0)
        health = 0.0
        if max_mfe is not None:
            denom = max(0.45, atr)
            health = max(0.0, min(1.0, max_mfe / denom))
        budget = self._micro_low_vol_event_budget_sec
        if age_sec >= budget and health < 0.22 and profit_pips <= 0.6:
            self._low_vol_hazard_hits[key] = 0
            log_metric(
                "low_vol_exit",
                1.0,
                tags={"reason": "low_vol_event_budget", "side": side, "score": f"{score:.2f}"},
            )
            return "low_vol_event_budget", True
        grace = self._micro_low_vol_grace_sec
        key = (pocket, side)
        if age_sec <= grace and profit_pips <= -0.25 and (max_mfe is None or max_mfe <= 0.45):
            self._low_vol_hazard_hits[key] = 0
            log_metric(
                "low_vol_exit",
                1.0,
                tags={"reason": "low_vol_early_scratch", "side": side, "score": f"{score:.2f}"},
            )
            return "low_vol_early_scratch", True
        # Soft TP before hard timeout (low-vol only)
        if (
            self._upper_bound_max_sec > 0
            and age_sec >= self._timeout_soft_tp_frac * self._upper_bound_max_sec
            and profit_pips >= self._soft_tp_pips
        ):
            self._low_vol_hazard_hits[key] = 0
            log_metric(
                "low_vol_exit",
                1.0,
                tags={"reason": "soft_tp_timeout", "side": side, "score": f"{score:.2f}"},
            )
            return "soft_tp_timeout", True
        hazard_met = (
            profit_pips <= -self._micro_low_vol_hazard_loss
            and age_sec >= budget * 0.7
            and health < (0.3 if low_vol_quiet else 0.4)
        )
        if hazard_met and self._hazard_exit_enabled:
            debounce = 2 if low_vol_quiet else self._hazard_debounce_ticks
            hits = self._low_vol_hazard_hits.get(key, 0) + 1
            self._low_vol_hazard_hits[key] = hits
            if hits >= debounce:
                log_metric(
                    "low_vol_exit",
                    1.0,
                    tags={"reason": "low_vol_hazard_exit", "side": side, "score": f"{score:.2f}"},
                )
                return "low_vol_hazard_exit", True
        else:
            if key in self._low_vol_hazard_hits and self._low_vol_hazard_hits[key] != 0:
                self._low_vol_hazard_hits[key] = 0
        return None, False

    def _confirm_reverse_signal(
        self,
        signal: Optional[Dict],
        pocket: str,
        direction: str,
        now: datetime,
    ) -> Optional[Dict]:
        key = (pocket, direction)
        state = self._reverse_hits.get(key)
        if signal:
            if state:
                ts = state.get("ts")
                if isinstance(ts, datetime) and now - ts > self._reverse_decay:
                    state = None
            count = 0
            if state:
                count = int(state.get("count", 0) or 0)
            count += 1
            self._reverse_hits[key] = {"count": count, "ts": now}
            needed = self._reverse_confirmations + (1 if pocket == "macro" else 0)
            if count >= needed:
                return signal
            return None
        if state:
            ts = state.get("ts")
            if isinstance(ts, datetime) and now - ts > self._reverse_decay:
                self._reverse_hits.pop(key, None)
            else:
                self._reverse_hits[key] = {"count": 0, "ts": now}
        return None

    def _peak_reversal_hint(
        self,
        *,
        pocket: str,
        side: str,
        profit_pips: float,
        fac_m1: Dict,
    ) -> Optional[Dict]:
        """
        Detect a sharp fade after a local peak and emit a synthetic reverse signal
        (e.g., long→short) so strategies can flip quickly after topping out.
        """
        if pocket not in {"micro", "scalp"}:
            return None
        max_seen = self._max_profit_cache.get((pocket, side))
        if max_seen is None or max_seen < 5.5:
            return None
        try:
            loss_now = -float(profit_pips)
        except Exception:
            return None
        try:
            drawdown_ratio = (max_seen - profit_pips) / max_seen
        except Exception:
            drawdown_ratio = None
        if drawdown_ratio is None or drawdown_ratio < 0.7:
            return None

        candles = fac_m1.get("candles") or []
        slope6 = _slope_from_candles(candles, window=6)
        slope12 = _slope_from_candles(candles, window=12) if len(candles) >= 12 else slope6
        slope = (slope6 + slope12) / 2.0
        if side == "long" and slope > -0.35:
            return None
        if side == "short" and slope < 0.35:
            return None

        action = "OPEN_SHORT" if side == "long" else "OPEN_LONG"
        confidence = 88 if drawdown_ratio < 1.4 else 93
        return {
            "pocket": pocket,
            "action": action,
            "confidence": confidence,
            "tag": "peak_reversal",
            "reason": f"mfe_retrace_{drawdown_ratio:.2f}_loss{loss_now:.1f}p",
        }

    def _story_allows_exit(
        self,
        story: Optional[ChartStorySnapshot],
        pocket: str,
        side: str,
        reason: str,
        profit_pips: float,
        now: datetime,
        *,
        range_mode: bool,
    ) -> bool:
        if story is None:
            return True
        if reason in {"range_stop", "stop_loss_order"}:
            return True

        trend = self._story_trend(story, pocket)
        supportive = False
        if side == "long" and trend == "up":
            supportive = True
        if side == "short" and trend == "down":
            supportive = True

        if not supportive:
            return True

        if reason in {"reverse_signal", "ma_cross_imminent", "ma_cross"}:
            if profit_pips > -self._macro_loss_buffer:
                logging.info(
                    "[STORY] sustain exit defer pocket=%s side=%s reason=%s trend=%s profit=%.2f",
                    pocket,
                    side,
                    reason,
                    trend,
                    profit_pips,
                )
                log_metric(
                    "exit_story_blocked",
                    float(profit_pips),
                    tags={
                        "pocket": pocket,
                        "side": side,
                        "reason": reason,
                        "trend": trend or "",
                        "volatility": getattr(story, "volatility_state", None) or "",
                        "range_mode": str(range_mode),
                    },
                    ts=now,
                )
                return False
        return True

    def _pattern_supports_exit(
        self,
        story: Optional[ChartStorySnapshot],
        pocket: str,
        side: str,
        reason: str,
        profit_pips: float,
    ) -> bool:
        if story is None:
            return True
        patterns = getattr(story, "pattern_summary", None) or {}
        if not patterns or pocket != "macro":
            return True

        n_wave = patterns.get("n_wave") or {}
        bias = n_wave.get("direction") or n_wave.get("bias")
        confidence = float(n_wave.get("confidence", 0.0) or 0.0)
        candle = patterns.get("candlestick") or {}
        candle_bias = candle.get("bias")
        try:
            candle_conf = float(candle.get("confidence", 0.0) or 0.0)
        except Exception:
            candle_conf = 0.0
        # Reasons that are soft / pattern-driven for macro exits
        pattern_reasons = {
            "reverse_signal",
            "ma_cross",
            "ma_cross_imminent",
            "trend_reversal",
            "macro_trail_hit",
        }
        if reason not in pattern_reasons:
            return True

        # Candlestickが強く逆行する場合はEXITを許容し、順行の場合は軽くブロック
        if candle_bias and candle_conf >= 0.6:
            if side == "long":
                if candle_bias == "down":
                    return True
                if candle_bias == "up" and profit_pips > -self._macro_loss_buffer:
                    return False
            else:
                if candle_bias == "up":
                    return True
                if candle_bias == "down" and profit_pips > -self._macro_loss_buffer:
                    return False

        if side == "long":
            if bias == "up" and confidence >= 0.55 and profit_pips > -self._macro_loss_buffer:
                logging.info(
                    "[PATTERN] veto macro exit: bias=%s conf=%.2f profit=%.2f",
                    bias,
                    confidence,
                    profit_pips,
                )
                return False
            if bias == "down" and confidence >= 0.5:
                return True
        else:
            if bias == "down" and confidence >= 0.55 and profit_pips > -self._macro_loss_buffer:
                logging.info(
                    "[PATTERN] veto macro exit: bias=%s conf=%.2f profit=%.2f",
                    bias,
                    confidence,
                    profit_pips,
                )
                return False
            if bias == "up" and confidence >= 0.5:
                return True
        return True

    @staticmethod
    def _story_trend(
        story: Optional[ChartStorySnapshot],
        pocket: str,
    ) -> Optional[str]:
        if story is None:
            return None
        if pocket == "macro":
            return story.macro_trend
        if pocket == "micro":
            return story.micro_trend
        return story.higher_trend

    def _macro_profit_capture(
        self,
        open_info: Dict,
        side: str,
        profit_pips: float,
        max_mfe: Optional[float],
        now: datetime,
    ) -> Optional[str]:
        if profit_pips is None or profit_pips <= 0.0:
            return None
        if max_mfe is None or max_mfe < self._h1_lock_min_trigger:
            return None
        trades = [
            tr for tr in (open_info.get("open_trades") or []) if tr.get("side") == side
        ]
        trade = trades[0] if trades else None
        hold_seconds = self._trade_age_seconds(trade, now) if trade else None
        hold_minutes = (hold_seconds or 0.0) / 60.0 if hold_seconds is not None else 0.0
        if hold_minutes < self._h1_lock_min_hold_minutes:
            return None

        theses = []
        for tr in open_info.get("open_trades") or []:
            if tr.get("side") != side:
                continue
            thesis = tr.get("entry_thesis") or {}
            if thesis.get("strategy") == "H1Momentum":
                theses.append(thesis)
        if not theses:
            return None

        insurance_values: List[float] = []
        for thesis in theses:
            note = thesis.get("note")
            if isinstance(note, dict):
                raw = note.get("insurance_sl")
                try:
                    value = float(raw)
                except (TypeError, ValueError):
                    continue
                if value > 0.0:
                    insurance_values.append(value)
        insurance_sl = statistics.median(insurance_values) if insurance_values else 24.0
        trigger = max(self._h1_lock_min_trigger, insurance_sl * self._h1_lock_trigger_ratio)
        if max_mfe < trigger:
            return None
        buffer = max(self._h1_lock_min_buffer, insurance_sl * self._h1_lock_buffer_ratio)
        if profit_pips > max_mfe - buffer:
            return None

        log_metric(
            "macro_profit_lock",
            float(profit_pips),
            tags={
                "strategy": "H1Momentum",
                "side": side,
                "max_mfe": f"{max_mfe:.2f}",
                "insurance_sl": f"{insurance_sl:.2f}",
                "hold_min": f"{hold_minutes:.1f}",
            },
            ts=now,
        )
        return "macro_profit_lock"

    def _get_mfe_guard(
        self,
        pocket: str,
        atr_primary: Optional[float],
    ) -> Tuple[float, float]:
        base = self._mfe_guard_base.get(pocket, self._mfe_guard_base_default)
        atr = self._safe_float(atr_primary)
        if atr > 0.0:
            if pocket == "macro":
                base = max(base, min(base + 0.9, 0.9 + atr * 0.28))
            elif pocket == "micro":
                base = max(base, min(base + 0.6, 0.7 + atr * 0.22))
            elif pocket == "scalp":
                base = max(base, min(base + 0.4, 0.6 + atr * 0.18))
        ratio = self._mfe_guard_ratio.get(pocket, 0.6)
        return round(base, 2), ratio

    def _record_exit_metric(
        self,
        pocket: str,
        side: str,
        reason: str,
        profit_pips: float,
        story: Optional[ChartStorySnapshot],
        range_mode: bool,
        now: datetime,
    ) -> None:
        trend = self._story_trend(story, pocket) if story else None
        volatility = story.volatility_state if story else None
        summary_state = None
        if story and story.summary:
            summary_state = story.summary.get("H1")
        tags = {
            "pocket": pocket,
            "side": side,
            "reason": reason,
            "trend": trend or "",
            "volatility": volatility or "",
            "summary_h1": summary_state or "",
            "range_mode": str(range_mode),
        }
        log_metric(
            "exit_decision",
            float(profit_pips),
            tags=tags,
            ts=now,
        )
        log_metric(
            "exit_decision_count",
            1.0,
            tags=tags,
            ts=now,
        )

    def _apply_advisor_hint(
        self,
        hint: ExitHint,
        pocket: str,
        side: str,
        units: int,
        profit_pips: float,
        tag: str,
        story: Optional[ChartStorySnapshot],
        range_mode: bool,
        now: datetime,
    ) -> Optional[ExitDecision]:
        reason = None
        if hint.max_drawdown_pips is not None and profit_pips <= -abs(hint.max_drawdown_pips):
            reason = "advisor_drawdown"
        elif hint.min_takeprofit_pips is not None and profit_pips >= abs(hint.min_takeprofit_pips):
            reason = "advisor_takeprofit"
        if not reason:
            return None
        units_to_close = abs(units)
        signed_units = -units_to_close if side == "long" else units_to_close
        final_reason = reason
        if hint.reason:
            final_reason = f"{reason}:{hint.reason}"
        allow_reentry = hint.confidence >= 0.7
        self._record_exit_metric(
            pocket,
            side,
            final_reason,
            profit_pips,
            story,
            range_mode,
            now,
        )
        log_metric(
            "exit_advisor_trigger",
            float(profit_pips),
            tags={
                "pocket": pocket,
                "side": side,
                "reason": reason,
                "model": hint.model_used or "unknown",
            },
            ts=now,
        )
        return ExitDecision(
            pocket=pocket,
            units=signed_units,
            reason=final_reason,
            tag=tag,
            allow_reentry=allow_reentry,
        )

    def _validate_trend_reversal(
        self,
        pocket: str,
        side: str,
        story: Optional[ChartStorySnapshot],
        close_price: Optional[float],
        m1_candles: List[Dict],
        *,
        atr_primary: Optional[float],
        atr_m1: Optional[float],
        bias_only: bool = False,
    ) -> bool:
        if close_price is None:
            return False
        trend = self._story_trend(story, pocket)
        higher = getattr(story, "higher_trend", None) if story else None
        structure_bias = getattr(story, "structure_bias", 0.0) if story else 0.0
        if side == "long":
            if trend == "down" or higher == "down" or structure_bias <= -4.0:
                return True
        else:
            if trend == "up" or higher == "up" or structure_bias >= 4.0:
                return True
        if bias_only:
            return False
        atr = atr_primary or atr_m1 or 0.0
        return self._is_structural_break(side, close_price, m1_candles, atr_pips=atr)

    def _is_structural_break(
        self,
        side: str,
        close_price: float,
        candles: List[Dict],
        *,
        atr_pips: float = 0.0,
        lookback: int = 12,
    ) -> bool:
        if not candles or len(candles) < lookback + 2:
            return False
        lows: List[float] = []
        highs: List[float] = []
        for c in candles[-(lookback + 2) : -1]:
            try:
                lows.append(float(c.get("low")))
                highs.append(float(c.get("high")))
            except (TypeError, ValueError):
                continue
        if not lows or not highs:
            return False
        atr_buffer = max(0.15, (atr_pips or 0.0) * 0.28)
        buffer_price = atr_buffer * self._pip
        if side == "long":
            swing_low = min(lows)
            return close_price <= swing_low - buffer_price
        swing_high = max(highs)
        return close_price >= swing_high + buffer_price

    def _compute_exit_units(
        self,
        pocket: str,
        side: str,
        reason: str,
        total_units: int,
        stage_level: int,
        pocket_profile: Dict[str, float],
        *,
        range_mode: bool,
        profit_pips: Optional[float] = None,
    ) -> int:
        base = abs(total_units)
        if base == 0:
            return 0
        if base <= self._min_partial_units:
            return base
        # 負けているポジションでは段階的クローズを避け、即時にまとめて縮小/クローズする
        if profit_pips is not None and profit_pips <= 0.0:
            return base
        if reason not in self._partial_eligible_reasons:
            return base
        if reason in self._force_exit_reasons:
            return base
        fraction = 0.7
        if reason == "trend_reversal":
            fraction = 0.6
        elif reason in {"reverse_signal", "ma_cross_imminent", "ma_cross"}:
            fraction = 0.55
        if stage_level >= 4:
            fraction *= 0.6
        elif stage_level == 3:
            fraction *= 0.7
        elif stage_level == 2:
            fraction *= 0.8
        win_rate = pocket_profile.get("win_rate", 0.0)
        avg_loss = pocket_profile.get("avg_loss_pips", 0.0)
        if win_rate >= 0.58:
            fraction *= 0.85
        if avg_loss and avg_loss <= 3.5:
            fraction *= 0.85
        if range_mode:
            fraction = min(0.85, fraction * 1.05)
        fraction = max(0.25, min(0.75, fraction))
        units_to_close = int(round(base * fraction))
        if units_to_close <= 0:
            units_to_close = 1
        preserve_floor = max(1, int(round(base * 0.25)))
        max_close = max(1, base - preserve_floor)
        if units_to_close >= base:
            units_to_close = max_close
        elif units_to_close > max_close:
            units_to_close = max_close
        if units_to_close <= 0:
            units_to_close = max_close
        return min(base, units_to_close)

    @staticmethod
    def _safe_float(value: object, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    def _reset_reverse_counter(self, pocket: str, direction: str) -> None:
        self._reverse_hits.pop((pocket, direction), None)
        self._max_profit_cache.pop((pocket, direction), None)

    def _update_max_profit(self, pocket: str, side: str, profit_pips: float) -> None:
        try:
            val = float(profit_pips)
        except (TypeError, ValueError):
            return
        if val <= 0.0:
            return
        key = (pocket, side)
        prev = self._max_profit_cache.get(key)
        if prev is None or val > prev:
            self._max_profit_cache[key] = val

    def _mfe_retrace_exit(
        self,
        *,
        pocket: str,
        side: str,
        units: int,
        profit_pips: Optional[float],
    ) -> Optional[ExitDecision]:
        """
        Partial close when a once-positive trade has retraced deeply.
        Applies only to micro/scalp to avoid dragging a fully winning trade back to flat.
        """
        if pocket not in {"micro", "scalp"}:
            return None
        if units <= 0 or profit_pips is None or profit_pips <= 0:
            return None
        max_seen = self._max_profit_cache.get((pocket, side))
        if max_seen is None or max_seen < 2.0:
            return None
        try:
            retrace_ratio = (max_seen - profit_pips) / max_seen
        except Exception:
            return None
        if retrace_ratio < 0.5:
            return None
        partial_units = int(abs(units) * 0.35)
        partial_units = max(1000, min(abs(units) - 1, partial_units))
        if partial_units <= 0 or partial_units >= abs(units):
            return None
        cut_units = -partial_units if side == "long" else partial_units
        return ExitDecision(
            pocket=pocket,
            units=cut_units,
            reason="mfe_retrace",
            tag="mfe-retrace",
            allow_reentry=True,
        )

    def _has_mature_trade(
        self,
        open_info: Dict,
        side: str,
        now: datetime,
        threshold_minutes: float,
    ) -> bool:
        trades = open_info.get("open_trades") or []
        for tr in trades:
            if tr.get("side") != side:
                continue
            opened_at = self._parse_open_time(tr.get("open_time"))
            if opened_at is None:
                continue
            age_minutes = (now - opened_at).total_seconds() / 60.0
            if age_minutes >= threshold_minutes:
                return True
        return False

    def _youngest_trade_age_seconds(
        self,
        open_info: Dict,
        side: str,
        now: datetime,
    ) -> Optional[float]:
        trades = open_info.get("open_trades") or []
        youngest: Optional[float] = None
        for tr in trades:
            if tr.get("side") != side:
                continue
            opened_at = self._parse_open_time(tr.get("open_time"))
            if opened_at is None:
                continue
            age = (now - opened_at).total_seconds()
            if age < 0:
                continue
            if youngest is None or age < youngest:
                youngest = age
        return youngest

    def _trade_age_seconds(self, trade: Dict, now: datetime) -> Optional[float]:
        if not trade:
            return None
        opened_at = self._parse_open_time(trade.get("open_time"))
        if opened_at is None:
            return None
        age = (now - opened_at).total_seconds()
        if age < 0:
            return 0.0
        return age

    def _negative_exit_blocked(
        self,
        pocket: str,
        open_info: Dict,
        side: str,
        now: datetime,
        profit_pips: float,
        stage_tracker: Optional["StageTracker"],
        atr_pips: float,
        fac_m1: Dict,
    ) -> bool:
        if stage_tracker is None:
            return False
        if profit_pips is None or profit_pips >= 0.0:
            return False
        default_loss, default_hold = self._default_loss_hold(pocket)
        trades = [
            tr
            for tr in (open_info.get("open_trades") or [])
            if tr.get("side") == side
        ]
        if not trades:
            return False
        blocked = False
        for tr in trades:
            thesis = self._parse_entry_thesis(tr)
            strategy_tag = thesis.get("strategy_tag") or tr.get("strategy_tag")
            loss_guard, hold_req = self._trade_guard_requirements(
                thesis, pocket, default_loss, default_hold
            )
            loss_guard = self._volatility_loss_clamp(
                pocket=pocket,
                loss_guard=loss_guard,
                atr_pips=atr_pips,
                fac_m1=fac_m1,
            )
            if loss_guard <= 0.0 or hold_req <= 0.0:
                continue
            if profit_pips <= -loss_guard:
                continue
            age = self._trade_age_seconds(tr, now)
            if age is None:
                continue
            if age < hold_req:
                trade_id = tr.get("trade_id")
                logging.info(
                    "[EXIT] hold_guard block pocket=%s side=%s trade=%s age=%.1fs req=%.1fs loss_guard=%.2fp profit=%.2fp",
                    pocket,
                    side,
                    trade_id,
                    age,
                    hold_req,
                    loss_guard,
                    profit_pips,
                )
                self._record_hold_violation(
                    pocket,
                    side,
                    strategy_tag,
                    hold_req,
                    age,
                    now,
                    stage_tracker,
                )
                blocked = True
                break
        return blocked

    def _default_loss_hold(self, pocket: str) -> tuple[float, float]:
        guard_map = {
            "micro": (self._micro_loss_grace_pips, self._micro_min_hold_seconds),
            "scalp": (self._scalp_loss_grace_pips, self._scalp_min_hold_seconds),
        }
        return guard_map.get(pocket, (0.0, 0.0))

    def _trade_guard_requirements(
        self,
        thesis: Dict,
        pocket: str,
        default_loss: float,
        default_hold: float,
    ) -> tuple[float, float]:
        loss_guard = thesis.get("loss_guard_pips") or thesis.get("loss_grace_pips")
        hold_req = thesis.get("min_hold_sec") or thesis.get("min_hold_seconds")
        try:
            loss_val = float(loss_guard)
        except (TypeError, ValueError):
            loss_val = default_loss
        if loss_val <= 0.0:
            loss_val = default_loss
        try:
            hold_val = float(hold_req)
        except (TypeError, ValueError):
            hold_val = default_hold
        if hold_val <= 0.0:
            hold_val = default_hold
        return max(0.0, loss_val), max(0.0, hold_val)

    def _volatility_loss_clamp(
        self,
        *,
        pocket: str,
        loss_guard: float,
        atr_pips: float,
        fac_m1: Dict,
    ) -> float:
        if loss_guard <= 0.0 or pocket not in {"micro", "scalp"}:
            return loss_guard
        atr_val = float(atr_pips or 0.0)
        try:
            vol_val = float(fac_m1.get("vol_5m") or 0.0)
        except (TypeError, ValueError):
            vol_val = 0.0
        if atr_val < self._loss_guard_atr_trigger and (vol_val <= 0.0 or vol_val < self._loss_guard_vol_trigger):
            return loss_guard
        clamped = max(0.2, loss_guard * self._loss_guard_compress_ratio)
        if clamped < loss_guard:
            logging.info(
                "[EXIT] loss_guard clamp pocket=%s atr=%.2f vol=%.2f %.2f->%.2f",
                pocket,
                atr_val,
                vol_val,
                loss_guard,
                clamped,
            )
        return clamped

    def _record_hold_violation(
        self,
        pocket: str,
        direction: str,
        strategy_tag: Optional[str],
        required_sec: float,
        actual_sec: float,
        now: datetime,
        stage_tracker: Optional["StageTracker"],
    ) -> None:
        tags = {
            "pocket": pocket,
            "direction": direction,
        }
        if strategy_tag:
            tags["strategy"] = strategy_tag
        log_metric("exit_hold_violation", 1.0, tags=tags)
        if stage_tracker:
            stage_tracker.log_hold_violation(
                pocket,
                direction,
                required_sec=required_sec,
                actual_sec=actual_sec,
                reason="hold_guard_block",
                cooldown_seconds=self._hold_violation_cooldown(pocket),
                now=now,
            )

    @staticmethod
    def _hold_violation_cooldown(pocket: str) -> int:
        if pocket == "macro":
            return 420
        if pocket == "micro":
            return 240
        return 180

    @staticmethod
    def _flag_truthy(val: object) -> bool:
        if isinstance(val, bool):
            return val
        if isinstance(val, (int, float)):
            return val != 0
        if isinstance(val, str):
            return val.strip().lower() in {"1", "true", "yes", "on"}
        return False

    def _has_kill_opt_in(self, trade: Dict) -> bool:
        """
        Opt-in gate for stale_drawdown/fast_cut-like safety exits.
        - flags: kill_switch / kill_opt_in
        - tags/exit_tags include {kill, kill_switch, dd_kill, fast_cut}
        - fast_cut meta present (fast_cut_pips/time/hard_mult)
        """
        thesis = self._parse_entry_thesis(trade)
        if self._flag_truthy(thesis.get("kill_switch")) or self._flag_truthy(thesis.get("kill_opt_in")):
            return True
        tags = thesis.get("tags") or thesis.get("exit_tags") or []
        if isinstance(tags, str):
            tags = [tags]
        tags_norm = []
        for t in tags:
            if not isinstance(t, str):
                continue
            tnorm = t.strip().lower()
            if tnorm:
                tags_norm.append(tnorm)
        if any(t in {"kill", "kill_switch", "dd_kill", "fast_cut"} for t in tags_norm):
            return True
        # fast_cut meta implies kill opt-in
        if thesis.get("fast_cut_pips") or thesis.get("fast_cut_time_sec") or thesis.get("fast_cut_hard_mult"):
            return True
        # Fallback: client_id / strategy_tag contains known fast_cut strategies
        cid = trade.get("client_id") or trade.get("client_order_id") or trade.get("id")
        if isinstance(cid, str):
            cid_low = cid.lower()
            for key in ("impulsere", "m1scalper", "pulsebreak", "rangefader"):
                if key in cid_low:
                    return True
        strategy_tag = trade.get("strategy_tag")
        if isinstance(strategy_tag, str) and any(
            k in strategy_tag.lower() for k in ("impulsere", "m1scalper", "pulsebreak", "rangefader")
        ):
            return True
        return False

    def _is_manual_trade(self, trade: Dict) -> bool:
        """Detect manually entered/unknown trades to exclude from automated exits."""
        thesis = self._parse_entry_thesis(trade)
        if thesis.get("pocket") == "manual" or self._flag_truthy(thesis.get("manual")):
            return True
        tags = thesis.get("tags") or thesis.get("exit_tags") or []
        if isinstance(tags, str):
            tags = [tags]
        for t in tags:
            if isinstance(t, str) and t.strip().lower() == "manual":
                return True
        cid = trade.get("client_id") or trade.get("client_order_id") or ""
        if isinstance(cid, str) and cid:
            if not cid.startswith(AGENT_CLIENT_PREFIXES):
                return True
        tag = thesis.get("tag") or trade.get("tag")
        if isinstance(tag, str) and "manual" in tag.lower():
            return True
        return False

    @staticmethod
    def _has_realtime_technicals(fac_m1: Dict) -> bool:
        try:
            atr_p = float(fac_m1.get("atr_pips") or (fac_m1.get("atr") or 0.0) * 100.0)
            rsi = float(fac_m1.get("rsi"))
            adx = float(fac_m1.get("adx"))
        except Exception:
            return False
        if math.isnan(atr_p) or math.isnan(rsi) or math.isnan(adx):
            return False
        return atr_p > 0.0

    @staticmethod
    def _parse_cutover_env(raw: Optional[str]) -> Optional[datetime]:
        """
        Parse ISO8601-like string from env (e.g., 2025-12-06T11:19:00Z) for drawdown cutover.
        """
        if not raw:
            return None
        t = raw.strip()
        try:
            if t.endswith("Z"):
                t = t[:-1] + "+00:00"
            return datetime.fromisoformat(t).astimezone(timezone.utc)
        except Exception:
            return None

    def _entry_target_bounds(
        self,
        open_info: Dict,
        side: str,
    ) -> Optional[tuple[float, float]]:
        trades = open_info.get("open_trades") or []
        targets: list[float] = []
        for tr in trades:
            if tr.get("side") != side:
                continue
            thesis = self._parse_entry_thesis(tr)
            target = thesis.get("target_tp_pips") or thesis.get("tp_hint_pips")
            try:
                if target is not None:
                    targets.append(float(target))
            except (TypeError, ValueError):
                continue
        if not targets:
            return None
        targets.sort()
        return targets[0], sum(targets) / len(targets)

    def _record_target_guard(
        self,
        pocket: str,
        direction: str,
        profit_pips: float,
        target_bounds: tuple[float, float],
        signal_tag: Optional[str],
    ) -> None:
        log_metric(
            "exit_target_guard",
            1.0,
            tags={
                "pocket": pocket,
                "direction": direction,
                "signal": signal_tag or "reverse",
            },
        )
        logging.info(
            "[EXIT] target_guard pocket=%s dir=%s profit=%.2fp guard<=%.2fp",
            pocket,
            direction,
            profit_pips,
            target_bounds[0],
        )

    @staticmethod
    def _parse_entry_thesis(trade: Dict) -> Dict:
        thesis = trade.get("entry_thesis") or {}
        if isinstance(thesis, str):
            try:
                thesis = json.loads(thesis)
            except Exception:
                thesis = {}
        if not isinstance(thesis, dict):
            thesis = {}
        return thesis

    def _has_strategy(
        self,
        open_info: Dict,
        strategy_keyword: str,
        side: Optional[str] = None,
    ) -> bool:
        trades = open_info.get("open_trades") or []
        for tr in trades:
            if side and tr.get("side") != side:
                continue
            tag = tr.get("strategy_tag")
            if isinstance(tag, str) and strategy_keyword in tag:
                return True
            thesis = self._parse_entry_thesis(tr)
            strat = thesis.get("strategy_tag") or thesis.get("strategy")
            if isinstance(strat, str) and strategy_keyword in strat:
                return True
        return False

    def _trendma_partial_exit_units(
        self,
        *,
        open_info: Dict,
        side: str,
        units: int,
        profit_pips: float,
        adx: float,
        rsi: float,
        projection_fast: Optional[MACrossProjection],
        atr_pips: float,
        loss_cap: Optional[float],
    ) -> Optional[int]:
        if not self._has_strategy(open_info, "TrendMA", side):
            return None
        if profit_pips is None or profit_pips <= 0.25:
            return None
        atr_val = float(atr_pips or 0.0)
        cap = loss_cap or max(1.8, atr_val * 0.8)
        profit_ceiling = max(self._trendma_partial_profit_cap, cap * 1.25)
        if profit_pips > profit_ceiling:
            return None
        slope = None
        if projection_fast is not None:
            try:
                slope = float(projection_fast.gap_slope_pips or 0.0)
            except Exception:
                slope = None
        slope_fade = False
        if slope is not None:
            if side == "long":
                slope_fade = slope <= 0.02
            else:
                slope_fade = slope >= -0.02
        adx_fade = adx <= self._macro_trend_adx + 1.5
        if side == "long":
            rsi_fade = rsi <= 56.0
        else:
            rsi_fade = rsi >= 44.0
        if not (slope_fade or (adx_fade and rsi_fade)):
            return None
        reduce_units = max(1000, int(abs(units) * self._trendma_partial_fraction))
        if reduce_units >= abs(units):
            reduce_units = abs(units) - 1000
        if reduce_units <= 0:
            return None
        return -reduce_units if side == "long" else reduce_units

    def _vol_partial_exit_units(
        self,
        *,
        pocket: str,
        side: str,
        units: int,
        profit_pips: float,
        atr_pips: float,
        ema_gap_pips: Optional[float],
    ) -> Optional[int]:
        if pocket != "macro":
            return None
        atr_val = float(atr_pips or 0.0)
        if atr_val < self._vol_partial_atr_min:
            return None
        profit_floor = max(self._vol_partial_profit_floor, atr_val * 1.5)
        if self._vol_partial_profit_cap > 0:
            profit_floor = min(profit_floor, self._vol_partial_profit_cap)
        if profit_pips is None or profit_pips < profit_floor:
            return None
        if ema_gap_pips is not None and abs(ema_gap_pips) > atr_val * 2.5:
            return None
        fraction = self._vol_partial_fraction
        if atr_val >= self._vol_partial_atr_max:
            fraction = max(0.5, fraction * 0.85)
        reduce_units = max(1000, int(abs(units) * fraction))
        if reduce_units >= abs(units):
            reduce_units = abs(units) - 1000
        if reduce_units <= 0:
            return None
        return -reduce_units if side == "long" else reduce_units

    def _ema_release_ready(
        self,
        *,
        pocket: str,
        profit_pips: float,
        atr_pips: float,
        close_price: float,
        ema20: float,
    ) -> bool:
        if pocket != "macro":
            return False
        if profit_pips is None or profit_pips <= 0.0:
            return False
        atr_val = float(atr_pips or 0.0)
        if atr_val < self._vol_partial_atr_min:
            return False
        if close_price is None or ema20 is None:
            return False
        ema_gap = (close_price - ema20) / 0.01
        return abs(ema_gap) <= self._vol_ema_release_gap

    def _profit_snatch_ready(
        self,
        *,
        pocket: str,
        side: str,
        open_info: Dict,
        profit_pips: float,
        atr_pips: float,
        fac_m1: Dict,
        now: datetime,
    ) -> bool:
        if pocket not in {"macro", "micro"}:
            return False
        if profit_pips is None:
            return False
        if not (self._profit_snatch_min <= profit_pips <= self._profit_snatch_max):
            return False
        atr_val = float(atr_pips or 0.0)
        if atr_val < self._profit_snatch_atr_min:
            return False
        vol = fac_m1.get("vol_5m")
        try:
            vol_val = float(vol) if vol is not None else None
        except (TypeError, ValueError):
            vol_val = None
        if vol_val is not None and vol_val < self._profit_snatch_vol_min:
            return False
        if not _in_jst_window(now, self._profit_snatch_jst_start, self._profit_snatch_jst_end):
            return False
        age = self._youngest_trade_age_seconds(open_info, side, now) or 0.0
        return age >= self._profit_snatch_hold

    @staticmethod
    def _parse_open_time(value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        raw = value.strip()
        try:
            if raw.endswith("Z"):
                raw = raw[:-1] + "+00:00"
            if "." in raw:
                head, frac = raw.split(".", 1)
                frac_digits = "".join(ch for ch in frac if ch.isdigit())
                if len(frac_digits) > 6:
                    frac_digits = frac_digits[:6]
                tz_part = ""
                if "+" in raw:
                    tz_part = raw[raw.rfind("+") :]
                if not tz_part:
                    tz_part = "+00:00"
                raw = f"{head}.{frac_digits}{tz_part}"
            elif "+" not in raw:
                raw = f"{raw}+00:00"
            dt = datetime.fromisoformat(raw)
            return dt.astimezone(timezone.utc)
        except ValueError:
            try:
                trimmed = raw
                if "." in trimmed:
                    trimmed = trimmed.split(".", 1)[0]
                if not trimmed.endswith("+00:00"):
                    trimmed = trimmed.rstrip("Z") + "+00:00"
                dt = datetime.fromisoformat(trimmed)
                return dt.astimezone(timezone.utc)
            except ValueError:
                return None

    @staticmethod
    def _ensure_utc(candidate: Optional[datetime]) -> datetime:
        if candidate is None:
            return datetime.now(timezone.utc)
        if candidate.tzinfo is None:
            return candidate.replace(tzinfo=timezone.utc)
        return candidate.astimezone(timezone.utc)
