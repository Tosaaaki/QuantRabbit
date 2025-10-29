from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
from dataclasses import dataclass
import json
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "scalp_active_params.json"
TRADES_DB_PATH = REPO_ROOT / "logs" / "trades.db"
ORDERS_DB_PATH = REPO_ROOT / "logs" / "orders.db"
STATE_PATH = REPO_ROOT / "logs" / "tuning" / "scalp_autotune_state.json"

AUTO_INTERVAL_SEC = int(os.getenv("SCALP_AUTOTUNE_INTERVAL_SEC", str(3 * 60 * 60)))
MIN_SAMPLE_SIZE = int(os.getenv("SCALP_AUTOTUNE_MIN_SAMPLES", "12"))
LOOKBACK_LIMIT = int(os.getenv("SCALP_AUTOTUNE_LOOKBACK", "90"))  # trades


@dataclass(slots=True)
class ScalpTradeSnapshot:
    trade_id: int
    side: str
    units: int
    entry_price: float
    exit_price: float
    sl_price: float | None
    tp_price: float | None
    pl_pips: float
    hold_minutes: float

    @property
    def direction(self) -> str:
        return "long" if self.units > 0 else "short"

    @property
    def is_win(self) -> bool:
        return self.pl_pips > 0.4

    @property
    def is_loss(self) -> bool:
        return self.pl_pips < -0.4

    def sl_distance(self) -> float | None:
        if self.sl_price is None or self.entry_price is None:
            return None
        delta = abs(self.entry_price - float(self.sl_price))
        return round(delta / 0.01, 4)

    def tp_distance(self) -> float | None:
        if self.tp_price is None or self.entry_price is None:
            return None
        delta = abs(self.entry_price - float(self.tp_price))
        return round(delta / 0.01, 4)


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _load_scalp_trades() -> List[ScalpTradeSnapshot]:
    if not TRADES_DB_PATH.exists() or not ORDERS_DB_PATH.exists():
        return []
    try:
        trades_conn = _connect(TRADES_DB_PATH)
        orders_conn = _connect(ORDERS_DB_PATH)
    except Exception:
        return []
    with trades_conn, orders_conn:
        trades = trades_conn.execute(
            """
            SELECT id, ticket_id, entry_time, close_time, units, entry_price, close_price, pl_pips
            FROM trades
            WHERE pocket='scalp'
            ORDER BY id DESC
            LIMIT ?
            """,
            (LOOKBACK_LIMIT,),
        ).fetchall()
        snapshots: List[ScalpTradeSnapshot] = []
        if not trades:
            return snapshots
        ticket_ids = tuple(row["ticket_id"] for row in trades if row["ticket_id"])
        if not ticket_ids:
            return snapshots
        order_map = {
            row["ticket_id"]: row
            for row in orders_conn.execute(
                f"""
                SELECT ticket_id, side, units, sl_price, tp_price, executed_price
                FROM orders
                WHERE ticket_id IN ({','.join('?' for _ in ticket_ids)}) AND status='filled'
                """,
                ticket_ids,
            ).fetchall()
        }
        for row in trades:
            ticket_id = row["ticket_id"]
            order = order_map.get(ticket_id)
            if not order:
                continue
            entry_ts = row["entry_time"]
            exit_ts = row["close_time"]
            hold_minutes = 0.0
            if entry_ts and exit_ts:
                try:
                    entry_dt = datetime.fromisoformat(entry_ts.replace("Z", "+00:00"))
                    exit_dt = datetime.fromisoformat(exit_ts.replace("Z", "+00:00"))
                    hold_minutes = max(0.0, (exit_dt - entry_dt).total_seconds() / 60.0)
                except Exception:
                    hold_minutes = 0.0
            snapshots.append(
                ScalpTradeSnapshot(
                    trade_id=row["id"],
                    side=order["side"] or "",
                    units=int(order["units"]),
                    entry_price=float(row["entry_price"]),
                    exit_price=float(row["close_price"]),
                    sl_price=float(order["sl_price"]) if order["sl_price"] else None,
                    tp_price=float(order["tp_price"]) if order["tp_price"] else None,
                    pl_pips=float(row["pl_pips"] or 0.0),
                    hold_minutes=hold_minutes,
                )
            )
        return snapshots


def _percentile(values: Iterable[float], pct: float) -> float:
    data = sorted(float(v) for v in values if v is not None)
    if not data:
        return 0.0
    if pct <= 0:
        return data[0]
    if pct >= 100:
        return data[-1]
    k = (len(data) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(data) - 1)
    if f == c:
        return data[f]
    d0 = data[f] * (c - k)
    d1 = data[c] * (k - f)
    return d0 + d1


def _load_config() -> dict:
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _write_config(data: dict) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = CONFIG_PATH.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as tmp:
        json.dump(data, tmp, ensure_ascii=False, indent=2, sort_keys=True)
    tmp_path.replace(CONFIG_PATH)


def _write_state(payload: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with STATE_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)


def run_autotune_once() -> dict | None:
    trades = _load_scalp_trades()
    if len(trades) < MIN_SAMPLE_SIZE:
        logging.info(
            "[AUTOTUNE] Skipping scalper autotune (samples=%d < required=%d)",
            len(trades),
            MIN_SAMPLE_SIZE,
        )
        return None

    wins = [t for t in trades if t.is_win]
    losses = [t for t in trades if t.is_loss]
    sl_samples = [t.sl_distance() for t in trades if t.sl_distance()]
    tp_samples = [t.tp_distance() for t in trades if t.tp_distance()]

    win_rate = len(wins) / len(trades) if trades else 0.0
    avg_win = mean(t.pl_pips for t in wins) if wins else 0.0
    avg_loss = mean(abs(t.pl_pips) for t in losses) if losses else 0.0
    hold_win = mean(t.hold_minutes for t in wins) if wins else 1.4
    hold_loss = mean(t.hold_minutes for t in losses) if losses else 1.9
    median_sl_distance = _percentile(sl_samples, 55)
    high_sl_distance = _percentile(sl_samples, 85)
    median_tp_distance = _percentile(tp_samples, 55)

    # 直近リスクリワードが悪化している場合は、既存パラメータを維持する
    config_all = _load_config()
    existing = config_all.get("M1Scalper", {})
    existing_fallback = existing.get("fallback", {})
    existing_nwave = existing.get("nwave", {})

    if win_rate < 0.35:
        logging.info(
            "[AUTOTUNE] win_rate %.3f too low; retaining existing config.", win_rate
        )
        return None

    # 産出値に安全域クランプを適用
    recommended_sl_floor = max(5.4, round(high_sl_distance + 0.5, 2)) if sl_samples else float(existing_fallback.get("sl_floor", 5.8))
    recommended_sl_floor = min(recommended_sl_floor, 7.2)

    recommended_tp_floor = max(1.4, round(median_tp_distance, 2) if tp_samples else float(existing_fallback.get("tp_floor", 1.8)))
    recommended_tp_floor = min(recommended_tp_floor, 2.2)

    recommended_sl_atr = 2.15 if win_rate >= 0.52 else 2.3
    recommended_tp_atr = 0.88 if win_rate >= 0.55 else 0.95

    recommended_momentum = 0.00118 if win_rate >= 0.6 else 0.00125
    if win_rate < 0.5:
        recommended_momentum = 0.0013
    recommended_momentum = max(0.00105, min(recommended_momentum, 0.00132))

    recommended_wick = 0.0008 if win_rate >= 0.55 else 0.00088
    recommended_wick = max(0.0007, min(recommended_wick, 0.00095))

    recommended_timeout = int(max(600, min(1500, (hold_win + hold_loss) / 2 * 60 * 2.0)))

    recommended_nwave_leg = 2.1 if win_rate >= 0.55 else 1.9
    recommended_nwave_leg = max(1.8, min(recommended_nwave_leg, 2.4))

    recommended_atr_floor = 0.98 if win_rate >= 0.5 else 1.08
    recommended_atr_floor = max(0.85, min(recommended_atr_floor, 1.2))

    config = config_all
    scalper_cfg = config.setdefault("M1Scalper", {})
    existing_nwave = scalper_cfg.get("nwave", {})
    fallback_cfg = scalper_cfg.setdefault("fallback", {})
    existing_fallback = dict(fallback_cfg)
    scalper_cfg["timeout_sec"] = recommended_timeout
    scalper_cfg["nwave"] = {
        "min_leg_pips": round(recommended_nwave_leg, 2),
        "pullback_mult": 1.65,
        "target_scale": 0.66,
        "target_floor": existing_nwave.get("target_floor", 1.6),
        "target_cap": existing_nwave.get("target_cap", 3.0),
        "tolerance_default": existing_nwave.get("tolerance_default", 0.28),
        "tolerance_tactical": existing_nwave.get(
            "tolerance_tactical", existing_nwave.get("tolerance_default", 0.28) + 0.08
        ),
        "hard_sl_floor": max(recommended_sl_floor, 6.5),
        "hard_sl_atr_mult": existing_nwave.get("hard_sl_atr_mult", 2.2),
    }
    tp_cap = float(existing_fallback.get("tp_cap", 2.3))
    tp_cap = max(1.8, min(tp_cap, 2.3))
    fallback_cfg.update(
        {
            "momentum_thresh": round(recommended_momentum, 5),
            "wick_min": round(recommended_wick, 5),
            "sl_floor": round(recommended_sl_floor, 2),
            "sl_atr_mult": round(recommended_sl_atr, 2),
            "tp_floor": round(recommended_tp_floor, 2),
            "tp_atr_mult": round(recommended_tp_atr, 2),
            "tp_cap": tp_cap,
            "atr_floor_tactical": round(recommended_atr_floor, 2),
            "atr_floor": round(
                max(1.4, min(float(existing_fallback.get("atr_floor", 1.7)), 1.9)), 2
            ),
        }
    )
    _write_config(config)

    payload = {
        "samples": len(trades),
        "win_rate": round(win_rate, 4),
        "avg_win": round(avg_win, 3),
        "avg_loss": round(avg_loss, 3),
        "hold_win": round(hold_win, 3),
        "hold_loss": round(hold_loss, 3),
        "sl_floor": fallback_cfg["sl_floor"],
        "tp_floor": fallback_cfg["tp_floor"],
        "momentum_thresh": fallback_cfg["momentum_thresh"],
        "wick_min": fallback_cfg["wick_min"],
        "timeout_sec": recommended_timeout,
        "nwave_min_leg": scalper_cfg["nwave"]["min_leg_pips"],
        "atr_floor_tactical": fallback_cfg["atr_floor_tactical"],
    }
    _write_state(payload)
    logging.info("[AUTOTUNE] Updated M1Scalper params: %s", payload)
    return payload


async def scalp_autotune_loop() -> None:
    while True:
        try:
            run_autotune_once()
        except Exception:
            logging.exception("[AUTOTUNE] Failed to update scalper parameters")
        await asyncio.sleep(AUTO_INTERVAL_SEC)


def start_background_autotune(loop: asyncio.AbstractEventLoop | None = None) -> asyncio.Task:
    event_loop = loop or asyncio.get_event_loop()
    return event_loop.create_task(scalp_autotune_loop(), name="scalp_autotune_loop")
