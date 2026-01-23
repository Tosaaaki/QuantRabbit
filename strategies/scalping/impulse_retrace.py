from __future__ import annotations

from typing import Dict
import logging

PIP = 0.01
MIN_ATR = 0.7
MAX_SPREAD = 1.55
MIN_DISLOCATION = 0.85  # pips away from ema20
RSI_LONG_MAX = 52
RSI_SHORT_MIN = 58
VOL_MIN = 0.3
MIN_SL_FLOOR = 1.05
SL_ATR_MULT = 1.2
TP_RATIO_MIN = 1.55
# momentum/EMAが強く順行しているときは逆張りを避ける
MOMENTUM_GUARD = 0.020  # price change per sec (approx)
SPREAD_ATR_RELAX = 0.35  # spread/atr がこの比率以下ならATR下限を緩和


def _attach_kill(signal: Dict) -> Dict:
    tags = []
    raw_tags = signal.get("exit_tags") or signal.get("tags")
    if raw_tags:
        if isinstance(raw_tags, str):
            tags = [raw_tags]
        elif isinstance(raw_tags, (list, tuple)):
            tags = list(raw_tags)
    tags = [t for t in tags if isinstance(t, str)]
    lower = [t.lower() for t in tags]
    if "kill" not in lower:
        tags.append("kill")
    if "fast_cut" not in lower:
        tags.append("fast_cut")
    signal["exit_tags"] = tags
    signal["kill_switch"] = True
    return signal


class ImpulseRetraceScalp:
    name = "ImpulseRetrace"
    pocket = "scalp"
    # 強いトレンドと逆方向のエントリーを抑止するためのガード
    TREND_BLOCK_GAP = 0.08  # ema10-ema20 (pips換算)
    TREND_BLOCK_MOMENTUM = 0.028  # 価格変化/秒目安
    TREND_WEAK_PENALTY = 0.9  # 軽い逆行なら信頼度を下げる

    @staticmethod
    def _log_skip(reason: str, **kwargs) -> None:
        extras = " ".join(f"{k}={v}" for k, v in kwargs.items() if v is not None)
        logging.info("[STRAT_SKIP_DETAIL] ImpulseRetrace reason=%s %s", reason, extras)

    @staticmethod
    def _atr_pips(fac: Dict) -> float:
        atr = fac.get("atr_pips")
        if atr is not None:
            return float(atr)
        raw = fac.get("atr")
        if raw is None:
            return 0.0
        try:
            return float(raw) * 100.0
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _atr_thresholds(fac: Dict) -> float:
        """ATRの下限をスプレッド/ボラに応じて動的に緩和する。"""
        base = MIN_ATR
        spread = fac.get("spread_pips")
        vol_5m = fac.get("vol_5m")
        try:
            spread = float(spread) if spread is not None else None
        except (TypeError, ValueError):
            spread = None
        try:
            vol_5m = float(vol_5m) if vol_5m is not None else None
        except (TypeError, ValueError):
            vol_5m = None
        atr_hint = ImpulseRetraceScalp._atr_pips(fac)
        if atr_hint > 0:
            if spread is not None:
                if spread <= 1.0 and (vol_5m is None or vol_5m >= 0.75):
                    base = min(base, 0.72)
                ratio = spread / max(atr_hint, 1e-6)
                # スプレッド負担が軽い＆最低限のボラがあるときは下限を少し緩める
                if ratio <= SPREAD_ATR_RELAX and (vol_5m is None or vol_5m >= 0.65):
                    base = min(base, max(0.62, MIN_ATR * 0.7))
            if vol_5m is not None and vol_5m >= 1.1:
                base = min(base, max(0.6, MIN_ATR * 0.66))
        return base

    @staticmethod
    def check(fac: Dict) -> Dict | None:
        close = fac.get("close")
        ema20 = fac.get("ema20") or fac.get("ma20")
        ema10 = fac.get("ema10")
        if close is None or ema20 is None:
            ImpulseRetraceScalp._log_skip("missing_price_ma", close=close, ema20=ema20)
            return None
        rsi = fac.get("rsi")
        if rsi is None:
            ImpulseRetraceScalp._log_skip("missing_rsi")
            return None
        try:
            rsi_val = float(rsi)
        except (TypeError, ValueError):
            ImpulseRetraceScalp._log_skip("rsi_parse_error", rsi=rsi)
            return None
        atr_pips = ImpulseRetraceScalp._atr_pips(fac)
        atr_floor = ImpulseRetraceScalp._atr_thresholds(fac)
        if atr_pips < atr_floor:
            ImpulseRetraceScalp._log_skip(
                "atr_low",
                atr_pips=round(atr_pips, 3),
                floor=round(atr_floor, 2),
            )
            return None
        spread = fac.get("spread_pips")
        if spread is not None:
            try:
                if float(spread) > MAX_SPREAD:
                    ImpulseRetraceScalp._log_skip(
                        "spread_high", spread=round(float(spread), 3)
                    )
                    return None
            except (TypeError, ValueError):
                pass
        vol_5m = fac.get("vol_5m")
        try:
            if vol_5m is not None and float(vol_5m) < VOL_MIN:
                ImpulseRetraceScalp._log_skip(
                    "vol_low", vol_5m=round(float(vol_5m), 3)
                )
                return None
        except (TypeError, ValueError):
            ImpulseRetraceScalp._log_skip("vol_parse_error", vol_5m=vol_5m)
            return None

        dislocation_pips = (close - ema20) / PIP
        ema_gap_pips = 0.0
        if ema10 is not None:
            ema_gap_pips = (ema10 - ema20) / PIP
        momentum = fac.get("momentum") or 0.0
        try:
            momentum = float(momentum)
        except Exception:
            momentum = 0.0
        trend_up = momentum > MOMENTUM_GUARD and ema_gap_pips > 0.05
        trend_down = momentum < -MOMENTUM_GUARD and ema_gap_pips < -0.05
        strong_up = (
            momentum > ImpulseRetraceScalp.TREND_BLOCK_MOMENTUM
            and ema_gap_pips > ImpulseRetraceScalp.TREND_BLOCK_GAP
        )
        strong_down = (
            momentum < -ImpulseRetraceScalp.TREND_BLOCK_MOMENTUM
            and ema_gap_pips < -ImpulseRetraceScalp.TREND_BLOCK_GAP
        )

        # ATRが低い環境では乖離の要求を少し緩める（ただしスプレッド優位が前提）
        dislocation_min = MIN_DISLOCATION
        if atr_pips <= max(1.1, atr_floor * 1.05):
            dislocation_min = max(0.8, MIN_DISLOCATION * 0.9)
        if vol_5m is not None and vol_5m >= 1.2:
            dislocation_min = max(0.8, dislocation_min * 0.95)

        def _build_signal(
            *,
            action: str,
            dist: float,
            rsi_value: float,
            atr_pips: float,
        ) -> Dict | None:
            if dist < dislocation_min:
                ImpulseRetraceScalp._log_skip(
                    "dislocation_small",
                    dist=round(dist, 3),
                    min_req=round(dislocation_min, 3),
                    rsi=rsi_value,
                )
                return None
            base_sl = dist * 0.7 + 0.35
            sl = max(MIN_SL_FLOOR, min(atr_pips * SL_ATR_MULT, base_sl))
            tp = max(sl * TP_RATIO_MIN, min(atr_pips * 1.9, sl + dist * 0.8))
            confidence = int(
                max(
                    48,
                    min(
                        95,
                        52
                        + (dist - MIN_DISLOCATION) * 8.0
                        + max(0.0, abs(ema_gap_pips)) * 1.2
                        + max(0.0, (atr_pips - MIN_ATR)) * 2.5,
                    ),
                )
            )
            profile = "impulse_retrace"
            # 軽い逆行トレンドの場合は信頼度を下げてロットを抑える
            if dist < dislocation_min * 1.2 and (
                (trend_up and action == "OPEN_SHORT") or (trend_down and action == "OPEN_LONG")
            ):
                confidence = int(confidence * ImpulseRetraceScalp.TREND_WEAK_PENALTY)
            min_hold = max(60.0, min(420.0, tp * 36.0))
            fast_cut = max(6.0, atr_pips * 0.9)
            fast_cut_time = max(60.0, atr_pips * 15.0)
            # 深い乖離ほどロット係数を下げる（confidence を下げることで間接的に調整）
            scale = 1.0
            if dist >= max(1.5, atr_pips * 1.5):
                scale = 0.5
            elif dist >= max(1.0, atr_pips * 1.2):
                scale = 0.7
            confidence = int(confidence * scale)
            return _attach_kill({
                "action": action,
                "sl_pips": round(sl, 2),
                "tp_pips": round(tp, 2),
                "confidence": confidence,
                "profile": profile,
                "loss_guard_pips": round(sl, 2),
                "target_tp_pips": round(tp, 2),
                "min_hold_sec": round(min_hold, 1),
                "fast_cut_pips": round(fast_cut, 2),
                "fast_cut_time_sec": int(fast_cut_time),
                "fast_cut_hard_mult": 1.6,
                "tag": f"{ImpulseRetraceScalp.name}-{action.lower()}",
            })

        if dislocation_pips <= -MIN_DISLOCATION and rsi_val <= RSI_LONG_MAX:
            # oversold spike, look for retrace long
            # 強い下向きトレンドならロングをブロック（逆張り抑制）
            if strong_down:
                ImpulseRetraceScalp._log_skip(
                    "trend_block_long",
                    momentum=round(momentum, 4),
                    ema_gap=round(ema_gap_pips, 4),
                )
                return None
            distance = abs(dislocation_pips)
            # 下向きトレンドがやや強いときは dislocation 要求を引き上げて慎重にする
            if trend_down:
                distance = max(distance, dislocation_min * 1.2)
            return _build_signal(
                action="OPEN_LONG",
                dist=distance,
                rsi_value=rsi_val,
                atr_pips=atr_pips,
            )

        if dislocation_pips >= MIN_DISLOCATION and rsi_val >= RSI_SHORT_MIN:
            # overbought spike, look for retrace short
            # 強い上向きトレンドならショートをブロック（逆張り抑制）
            if strong_up:
                ImpulseRetraceScalp._log_skip(
                    "trend_block_short",
                    momentum=round(momentum, 4),
                    ema_gap=round(ema_gap_pips, 4),
                )
                return None
            distance = abs(dislocation_pips)
            # 上向きトレンドがやや強いときは dislocation 要求を引き上げて慎重にする
            if trend_up:
                distance = max(distance, dislocation_min * 1.2)
            return _build_signal(
                action="OPEN_SHORT",
                dist=distance,
                rsi_value=rsi_val,
                atr_pips=atr_pips,
            )

        ImpulseRetraceScalp._log_skip(
            "no_extreme",
            dislocation=round(dislocation_pips, 3),
            rsi=rsi_val,
            rsi_long_max=RSI_LONG_MAX,
            rsi_short_min=RSI_SHORT_MIN,
        )
        return None
