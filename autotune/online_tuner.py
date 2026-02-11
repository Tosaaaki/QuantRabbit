from __future__ import annotations

import glob
import os
import random
import datetime as dt
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import logging

try:
    import pandas as pd
except Exception:
    pd = None
try:
    import yaml
except Exception:
    yaml = None

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))

def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)

def _utc_iso(ts: Optional[dt.datetime] = None) -> str:
    ts = ts or _utc_now()
    # Keep timestamps stable across runs (no microseconds).
    return ts.replace(microsecond=0).isoformat().replace("+00:00", "Z")

def _parse_dt(s: str) -> Optional[dt.datetime]:
    try:
        if s.endswith("Z"):
            return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt.datetime.fromisoformat(s)
    except Exception:
        return None

def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or v == "":
        return default
    try:
        return int(v)
    except Exception:
        return default

def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or v == "":
        return default
    try:
        return float(v)
    except Exception:
        return default

def _load_yaml(path: str) -> dict:
    if yaml is None:
        raise RuntimeError("PyYAML not available. Please install pyyaml.")
    if not os.path.exists(path):
        return {}
    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}

def _dump_yaml(data: dict, path: str):
    if yaml is None:
        raise RuntimeError("PyYAML not available. Please install pyyaml.")
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, 'w') as f:
        yaml.safe_dump(data, f, sort_keys=False)
    os.replace(tmp, path)

def _log_json(level: int, payload: dict) -> None:
    try:
        msg = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    except Exception:
        msg = str(payload)
    logging.log(level, msg)

def _flatten_dict(d: dict, prefix: str = "") -> dict:
    out: dict[str, Any] = {}
    for k, v in (d or {}).items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten_dict(v, key))
        else:
            out[key] = v
    return out

def _diff_roots(before: dict, after: dict, roots: tuple[str, ...] = ("exit", "strategies", "alloc")) -> list[dict]:
    b: dict[str, Any] = {}
    a: dict[str, Any] = {}
    for r in roots:
        if isinstance(before.get(r), dict):
            b.update(_flatten_dict(before.get(r) or {}, r))
        if isinstance(after.get(r), dict):
            a.update(_flatten_dict(after.get(r) or {}, r))
    changes: list[dict] = []
    for path in sorted(set(b.keys()) | set(a.keys())):
        if b.get(path) == a.get(path):
            continue
        changes.append({"path": path, "before": b.get(path), "after": a.get(path)})
    return changes

@dataclass
class TunerBounds:
    # Exit
    upper_bound_max_sec: tuple = (3.2, 4.8)
    hazard_cost_spread_base: tuple = (0.20, 0.40)
    hazard_cost_latency_base_ms: tuple = (180, 400)
    hazard_debounce_ticks: tuple = (1, 3)
    min_grace_before_scratch_ms: tuple = (0, 400)
    scratch_requires_events: tuple = (0, 2)
    # Stage/exit cadence: keep deltas very small as these are sensitive in live
    reentry_block_s: tuple = (8, 30)

    # Micro gates
    momentum_conf_min: tuple = (0.55, 0.75)
    microvwap_z_min: tuple = (1.8, 2.6)
    volcomp_accel_pctile: tuple = (60, 85)
    bb_rsi_reentry_block_s: tuple = (8, 30)

    # Alloc
    micro_share: tuple = (0.20, 0.60)  # quiet_low_vol only

@dataclass
class OnlineTuner:
    presets_path: str
    overrides_out: str
    history_dir: str = "logs/tuning/history"
    minutes: int = 15
    bounds: TunerBounds = field(default_factory=TunerBounds)
    random_explore: float = 0.02   # 2% safe exploration
    clamp_step: float = 0.15       # max delta for weights/alloc

    # Safety gates (env overridable)
    min_trades: int = 40
    min_reason_nonempty_pct: float = 0.70
    target_regime: str = "quiet_low_vol"
    min_target_regime_pct: float = 0.40
    min_micro_trades: int = 20

    # LKG + rollback (env overridable)
    lkg_path: Optional[str] = None
    state_path: Optional[str] = None
    good_ev_min: float = 0.0
    good_pf_min: float = 1.0
    good_win_rate_min: float = 0.50
    bad_ev_max: float = -0.05
    bad_pf_max: float = 0.85
    bad_win_rate_max: float = 0.45
    rollback_after_bad_runs: int = 3
    rollback_cooldown_min: int = 60

    def __post_init__(self) -> None:
        # NOTE: read env here so systemd env files can tune safety without code changes.
        self.min_trades = _env_int("TUNER_MIN_TRADES", self.min_trades)
        self.min_micro_trades = _env_int("TUNER_MIN_MICRO_TRADES", self.min_micro_trades)
        self.rollback_after_bad_runs = _env_int("TUNER_ROLLBACK_AFTER_BAD_RUNS", self.rollback_after_bad_runs)
        self.rollback_cooldown_min = _env_int("TUNER_ROLLBACK_COOLDOWN_MIN", self.rollback_cooldown_min)

        self.min_reason_nonempty_pct = _env_float("TUNER_MIN_REASON_NONEMPTY_PCT", self.min_reason_nonempty_pct)
        self.min_target_regime_pct = _env_float("TUNER_MIN_TARGET_REGIME_PCT", self.min_target_regime_pct)
        self.good_ev_min = _env_float("TUNER_GOOD_EV_MIN", self.good_ev_min)
        self.good_pf_min = _env_float("TUNER_GOOD_PF_MIN", self.good_pf_min)
        self.good_win_rate_min = _env_float("TUNER_GOOD_WIN_RATE_MIN", self.good_win_rate_min)
        self.bad_ev_max = _env_float("TUNER_BAD_EV_MAX", self.bad_ev_max)
        self.bad_pf_max = _env_float("TUNER_BAD_PF_MAX", self.bad_pf_max)
        self.bad_win_rate_max = _env_float("TUNER_BAD_WIN_RATE_MAX", self.bad_win_rate_max)

        self.target_regime = os.getenv("TUNER_TARGET_REGIME", self.target_regime)

        overrides_dir = os.path.dirname(self.overrides_out) or "."
        self.lkg_path = (
            os.getenv("TUNER_LKG_PATH")
            or self.lkg_path
            or os.path.join(overrides_dir, "tuning_overrides.lkg.yaml")
        )
        self.state_path = (
            os.getenv("TUNER_STATE_PATH")
            or self.state_path
            or os.path.join(overrides_dir, "online_tuner_state.yaml")
        )

    def load_recent(self, logs_glob: str) -> Optional['pd.DataFrame']:
        if pd is None:
            raise RuntimeError("pandas not available. Please install pandas.")
        paths = sorted(glob.glob(logs_glob))
        if not paths:
            return None
        frames = []
        for p in paths[-10:]:  # recent up to 10 files
            try:
                if p.endswith('.jsonl'):
                    df = pd.read_json(p, lines=True)
                else:
                    df = pd.read_csv(p)
                df = self._normalize_columns(df, source=p)
                frames.append(df)
            except Exception:
                continue
        if not frames:
            return None
        df = pd.concat(frames, ignore_index=True)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            cutoff = pd.Timestamp.utcnow() - pd.Timedelta(minutes=self.minutes)
            df = df[df['timestamp'] >= cutoff]
        return df.reset_index(drop=True)

    def _normalize_columns(self, df: 'pd.DataFrame', source: str) -> 'pd.DataFrame':
        """
        Align input columns coming from live DB exports or backtests.
        Fallbacks:
        - timestamp: prefer 'timestamp', else 'close_time', else 'exit_ts', else 'entry_ts'
        - reason: prefer 'reason', else 'close_reason'
        - pips: prefer 'pips', else 'pl_pips'
        - strategy: leave as-is if present, otherwise best-effort from 'strategy_tag'
        - regime: fallback to 'pocket'
        Missing hazard/event columns are filled with 0 so downstream logic can run.
        """
        df = df.copy()
        col_map = {}
        if 'close_reason' in df.columns and 'reason' not in df.columns:
            col_map['close_reason'] = 'reason'
        if 'pl_pips' in df.columns and 'pips' not in df.columns:
            col_map['pl_pips'] = 'pips'
        if 'strategy_tag' in df.columns and 'strategy' not in df.columns:
            col_map['strategy_tag'] = 'strategy'
        df = df.rename(columns=col_map)

        # timestamp fallback
        if 'timestamp' not in df.columns:
            for alt in ('close_time', 'exit_ts', 'entry_ts'):
                if alt in df.columns:
                    df['timestamp'] = df[alt]
                    break
        if 'timestamp' not in df.columns:
            logging.warning("[tuner] missing timestamp column in %s, filling now()", source)
            df['timestamp'] = dt.datetime.utcnow().isoformat() + 'Z'

        # regime fallback
        if 'regime' not in df.columns and 'pocket' in df.columns:
            df['regime'] = df['pocket']

        # ensure all expected columns exist (fill defaults)
        defaults = {
            'reason': '',
            'pips': 0.0,
            'strategy': '',
            'regime': '',
            'hazard_ticks': 0,
            'events': 0,
            'grace_used_ms': 0,
            'scratch_hits': 0,
        }
        for col, default in defaults.items():
            if col not in df.columns:
                df[col] = default
        return df

    def compute_metrics(self, df) -> Dict[str, Any]:
        m = {}
        if df is None or len(df)==0:
            return m
        m["n_trades"] = int(len(df))

        # Reason quality: empty reasons indicate missing classification in upstream logs.
        if "reason" in df.columns:
            try:
                nonempty = (df["reason"].fillna("").astype(str).str.len() > 0).mean()
                m["reason_nonempty_pct"] = float(nonempty)
            except Exception:
                pass
        # Reasons split
        if 'reason' in df.columns:
            rs = df['reason'].value_counts(normalize=True)
            m['reasons_pct'] = rs.to_dict()
            rc = df['reason'].value_counts()
            m['reasons_cnt'] = rc.to_dict()
        # EV
        if 'pips' in df.columns:
            p = pd.to_numeric(df["pips"], errors="coerce")
            p_nonnull = p.dropna()
            if len(p_nonnull) > 0:
                m["ev"] = float(p_nonnull.mean())
                m["pips_sum"] = float(p_nonnull.sum())
                m["pips_std"] = float(p_nonnull.std(ddof=0)) if len(p_nonnull) > 1 else 0.0

                wins = p_nonnull[p_nonnull > 0]
                losses = p_nonnull[p_nonnull < 0]
                m["win_rate"] = float((p_nonnull > 0).mean())
                sum_win = float(wins.sum()) if len(wins) > 0 else 0.0
                sum_loss = float(losses.sum()) if len(losses) > 0 else 0.0
                if sum_loss < 0:
                    pf = sum_win / abs(sum_loss) if abs(sum_loss) > 0 else 999.0
                else:
                    # No losses in the window -> PF is effectively very large.
                    pf = 999.0 if sum_win > 0 else 1.0
                m["pf"] = float(min(pf, 999.0))
                m["mean_win"] = float(wins.mean()) if len(wins) > 0 else 0.0
                m["mean_loss"] = float(losses.mean()) if len(losses) > 0 else 0.0

                m["pips_nonnull_pct"] = float(len(p_nonnull) / max(1, len(p)))
        # Hazard ticks sum (if present)
        if 'hazard_ticks' in df.columns:
            m['hazard_ticks_sum'] = int(df['hazard_ticks'].sum())
        # Strategy-wise TP ratio (if present)
        if {'strategy','reason'}.issubset(df.columns):
            tp = df[df['reason'].isin(['tp','tp_hit','soft_tp_timeout'])].groupby('strategy').size()
            allc = df.groupby('strategy').size()
            srat = (tp / allc).fillna(0.0)
            m['tp_rate_by_strategy'] = srat.to_dict()
        # Regime if present
        if 'regime' in df.columns:
            m['regime_counts'] = df['regime'].value_counts().to_dict()
        # Counts by strategy
        if "strategy" in df.columns:
            try:
                m["strategy_counts"] = df["strategy"].value_counts().to_dict()
            except Exception:
                pass
        return m

    def _load_state(self) -> dict:
        if not self.state_path:
            return {"version": 1}
        try:
            payload = _load_yaml(self.state_path)
            if isinstance(payload, dict):
                payload.setdefault("version", 1)
                return payload
        except Exception:
            pass
        return {"version": 1}

    def _save_state(self, state: dict) -> None:
        if not self.state_path:
            return
        try:
            _dump_yaml(state, self.state_path)
        except Exception:
            pass

    def _regime_pct(self, metrics: Dict[str, Any], regime: str) -> float:
        counts = metrics.get("regime_counts") or {}
        if not isinstance(counts, dict) or not counts:
            return 0.0
        total = 0
        for v in counts.values():
            try:
                total += int(v)
            except Exception:
                continue
        if total <= 0:
            return 0.0
        try:
            return float(int(counts.get(regime, 0)) / total)
        except Exception:
            return 0.0

    def _quality_gate(self, df: Optional["pd.DataFrame"], metrics: Dict[str, Any]) -> tuple[bool, list[str], dict]:
        reasons: list[str] = []
        q: dict[str, Any] = {}

        n = int(metrics.get("n_trades", 0) or 0)
        q["n_trades"] = n
        if n < self.min_trades:
            reasons.append("min_trades")

        reason_nonempty = float(metrics.get("reason_nonempty_pct", 1.0) or 0.0)
        q["reason_nonempty_pct"] = reason_nonempty
        if reason_nonempty < self.min_reason_nonempty_pct:
            reasons.append("reason_missing")

        target_pct = self._regime_pct(metrics, self.target_regime)
        q["target_regime"] = self.target_regime
        q["target_regime_pct"] = target_pct
        if target_pct < self.min_target_regime_pct:
            reasons.append("target_regime_coverage")

        # Micro strategy sample size (used to gate alloc drift).
        micro_names = ["MomentumPulse", "VolCompressionBreak", "MicroVWAPRevert", "BB_RSI_Fast"]
        strat_counts = metrics.get("strategy_counts") or {}
        micro_n = 0
        if isinstance(strat_counts, dict):
            for name in micro_names:
                try:
                    micro_n += int(strat_counts.get(name, 0) or 0)
                except Exception:
                    continue
        q["micro_trades"] = micro_n
        if micro_n < self.min_micro_trades:
            q["micro_gate"] = "insufficient"
        else:
            q["micro_gate"] = "ok"

        return (len(reasons) == 0), reasons, q

    def _is_good_window(self, metrics: Dict[str, Any]) -> bool:
        n = int(metrics.get("n_trades", 0) or 0)
        if n < self.min_trades:
            return False
        ev = float(metrics.get("ev", 0.0) or 0.0)
        pf = float(metrics.get("pf", 0.0) or 0.0)
        wr = float(metrics.get("win_rate", 0.0) or 0.0)
        ok = 0
        if ev >= self.good_ev_min:
            ok += 1
        if pf >= self.good_pf_min:
            ok += 1
        if wr >= self.good_win_rate_min:
            ok += 1
        return ok >= 2

    def _is_bad_window(self, metrics: Dict[str, Any]) -> tuple[bool, list[str]]:
        n = int(metrics.get("n_trades", 0) or 0)
        if n < self.min_trades:
            return (False, [])
        ev = float(metrics.get("ev", 0.0) or 0.0)
        pf = float(metrics.get("pf", 0.0) or 0.0)
        wr = float(metrics.get("win_rate", 0.0) or 0.0)
        bad: list[str] = []
        if ev <= self.bad_ev_max:
            bad.append("bad_ev")
        if pf <= self.bad_pf_max:
            bad.append("bad_pf")
        if wr <= self.bad_win_rate_max:
            bad.append("bad_win_rate")
        return (len(bad) >= 2), bad

    def _should_rollback(self, state: dict, metrics: Dict[str, Any]) -> tuple[bool, list[str]]:
        bad, bad_reasons = self._is_bad_window(metrics)
        if not bad:
            # Reset accumulation once the window is no longer bad.
            state["bad_runs"] = 0
            return (False, [])

        bad_runs = int(state.get("bad_runs", 0) or 0)
        bad_runs += 1
        state["bad_runs"] = bad_runs
        if bad_runs < self.rollback_after_bad_runs:
            return (False, bad_reasons)

        lkg_path = self.lkg_path or ""
        if not lkg_path or not os.path.exists(lkg_path):
            return (False, bad_reasons + ["no_lkg"])

        last_rb = state.get("last_rollback_at")
        if isinstance(last_rb, str):
            last_dt = _parse_dt(last_rb)
            if last_dt is not None:
                age = _utc_now() - last_dt.astimezone(dt.timezone.utc)
                if age < dt.timedelta(minutes=self.rollback_cooldown_min):
                    return (False, bad_reasons + ["rollback_cooldown"])

        return (True, bad_reasons)

    def _maybe_bootstrap_lkg(self, state: dict, current_overrides: dict) -> None:
        lkg_path = self.lkg_path
        if not lkg_path:
            return
        if os.path.exists(lkg_path):
            return
        try:
            _dump_yaml(current_overrides or {}, lkg_path)
            state.setdefault("lkg", {})["saved_at"] = _utc_iso()
            state["lkg"]["bootstrap"] = True
        except Exception:
            pass

    def _update_lkg_if_good(self, state: dict, metrics: Dict[str, Any], current_overrides: dict) -> bool:
        if not self.lkg_path:
            return False
        if not self._is_good_window(metrics):
            return False
        try:
            _dump_yaml(current_overrides or {}, self.lkg_path)
            state.setdefault("lkg", {})["saved_at"] = _utc_iso()
            state["lkg"]["bootstrap"] = False
            state["lkg"]["metrics"] = {
                "n_trades": int(metrics.get("n_trades", 0) or 0),
                "ev": float(metrics.get("ev", 0.0) or 0.0),
                "pf": float(metrics.get("pf", 0.0) or 0.0),
                "win_rate": float(metrics.get("win_rate", 0.0) or 0.0),
            }
            return True
        except Exception:
            return False

    def _rollback_to_lkg(self, state: dict) -> bool:
        if not self.lkg_path or not os.path.exists(self.lkg_path):
            return False
        try:
            lkg = _load_yaml(self.lkg_path) or {}
            _dump_yaml(lkg, self.overrides_out)
            state["last_rollback_at"] = _utc_iso()
            state["bad_runs"] = 0
            return True
        except Exception:
            return False

    def propose_patch(self, metrics: Dict[str, Any], preset: Dict[str, Any]) -> Dict[str, Any]:
        patch = {'meta':{
            'generated_at': _utc_iso(),
            'window_min': self.minutes,
        }}
        reasons_pct = metrics.get('reasons_pct', {})
        ev = metrics.get('ev', 0.0)
        hazard_ticks_sum = metrics.get('hazard_ticks_sum', 0)

        # Exit tuning (low-vol assumed)
        hard_timeout = reasons_pct.get('hard_timeout', 0.0)
        if hard_timeout > 0.25:
            ub = preset.get('exit', {}).get('lowvol', {}).get('upper_bound_max_sec', 3.6)
            ub = max(self.bounds.upper_bound_max_sec[0], ub - 0.2)
            patch.setdefault('exit', {}).setdefault('lowvol', {})['upper_bound_max_sec'] = round(min(ub, self.bounds.upper_bound_max_sec[1]), 2)
        elif hard_timeout < 0.05:
            ub = preset.get('exit', {}).get('lowvol', {}).get('upper_bound_max_sec', 3.6)
            ub = min(self.bounds.upper_bound_max_sec[1], ub + 0.1)
            patch.setdefault('exit', {}).setdefault('lowvol', {})['upper_bound_max_sec'] = round(max(ub, self.bounds.upper_bound_max_sec[0]), 2)

        hazard_exit = reasons_pct.get('hazard_exit', 0.0)
        if hazard_exit == 0.0 and hazard_ticks_sum > 20:
            # make hazard more sensitive a bit
            hb = preset.get('exit', {}).get('lowvol', {}).get('hazard_debounce_ticks', 2)
            hb = max(self.bounds.hazard_debounce_ticks[0], hb - 1)
            patch.setdefault('exit', {}).setdefault('lowvol', {})['hazard_debounce_ticks'] = int(min(hb, self.bounds.hazard_debounce_ticks[1]))
            csb = preset.get('exit', {}).get('lowvol', {}).get('hazard_cost_spread_base', 0.25) - 0.02
            clb = preset.get('exit', {}).get('lowvol', {}).get('hazard_cost_latency_base_ms', 240) - 20
            patch['exit']['lowvol']['hazard_cost_spread_base'] = round(_clamp(csb, *self.bounds.hazard_cost_spread_base), 3)
            patch['exit']['lowvol']['hazard_cost_latency_base_ms'] = int(_clamp(clb, *self.bounds.hazard_cost_latency_base_ms))
        elif hazard_exit > 0.25:
            # too many hazard exits → loosen sensitivity a bit
            hb = preset.get('exit', {}).get('lowvol', {}).get('hazard_debounce_ticks', 2)
            hb = min(self.bounds.hazard_debounce_ticks[1], hb + 1)
            patch.setdefault('exit', {}).setdefault('lowvol', {})['hazard_debounce_ticks'] = int(max(hb, self.bounds.hazard_debounce_ticks[0]))
            csb = preset.get('exit', {}).get('lowvol', {}).get('hazard_cost_spread_base', 0.25) + 0.02
            clb = preset.get('exit', {}).get('lowvol', {}).get('hazard_cost_latency_base_ms', 240) + 20
            patch['exit']['lowvol']['hazard_cost_spread_base'] = round(_clamp(csb, *self.bounds.hazard_cost_spread_base), 3)
            patch['exit']['lowvol']['hazard_cost_latency_base_ms'] = int(_clamp(clb, *self.bounds.hazard_cost_latency_base_ms))

        scratch = reasons_pct.get('scratch', 0.0)
        if scratch > 0.55:
            mg = preset.get('exit', {}).get('lowvol', {}).get('min_grace_before_scratch_ms', 120) + 30
            sr = preset.get('exit', {}).get('lowvol', {}).get('scratch_requires_events', 1)
            patch.setdefault('exit', {}).setdefault('lowvol', {})['min_grace_before_scratch_ms'] = int(_clamp(mg, *self.bounds.min_grace_before_scratch_ms))
            patch['exit']['lowvol']['scratch_requires_events'] = int(_clamp(sr+1, *self.bounds.scratch_requires_events))
        elif scratch < 0.15:
            mg = preset.get('exit', {}).get('lowvol', {}).get('min_grace_before_scratch_ms', 120) - 20
            sr = preset.get('exit', {}).get('lowvol', {}).get('scratch_requires_events', 1)
            patch.setdefault('exit', {}).setdefault('lowvol', {})['min_grace_before_scratch_ms'] = int(_clamp(mg, *self.bounds.min_grace_before_scratch_ms))
            patch['exit']['lowvol']['scratch_requires_events'] = int(_clamp(max(sr-1, self.bounds.scratch_requires_events[0]), *self.bounds.scratch_requires_events))

        # Strategy gate tweaks based on EV
        if ev < 0.05:
            # tighten entrances a bit
            mp = preset.get('strategies', {}).get('MomentumPulse', {}).get('min_confidence', 0.65) + 0.02
            zv = preset.get('strategies', {}).get('MicroVWAPRevert', {}).get('vwap_z_min', 2.2) + 0.1
            ap = preset.get('strategies', {}).get('VolCompressionBreak', {}).get('accel_pctile', 75) + 2
            rb = preset.get('strategies', {}).get('BB_RSI_Fast', {}).get('reentry_block_s', 15) + 2
            patch.setdefault('strategies', {}).setdefault('MomentumPulse', {})['min_confidence'] = round(_clamp(mp, *self.bounds.momentum_conf_min), 2)
            patch['strategies'].setdefault('MicroVWAPRevert', {})['vwap_z_min'] = round(_clamp(zv, *self.bounds.microvwap_z_min), 2)
            patch['strategies'].setdefault('VolCompressionBreak', {})['accel_pctile'] = int(_clamp(ap, *self.bounds.volcomp_accel_pctile))
            patch['strategies'].setdefault('BB_RSI_Fast', {})['reentry_block_s'] = int(_clamp(rb, *self.bounds.bb_rsi_reentry_block_s))
        elif ev > 0.12:
            # loosen gates slightly when EV is healthy
            mp = preset.get('strategies', {}).get('MomentumPulse', {}).get('min_confidence', 0.65) - 0.01
            zv = preset.get('strategies', {}).get('MicroVWAPRevert', {}).get('vwap_z_min', 2.2) - 0.05
            ap = preset.get('strategies', {}).get('VolCompressionBreak', {}).get('accel_pctile', 75) - 1
            rb = preset.get('strategies', {}).get('BB_RSI_Fast', {}).get('reentry_block_s', 15) - 1
            patch.setdefault('strategies', {}).setdefault('MomentumPulse', {})['min_confidence'] = round(_clamp(mp, *self.bounds.momentum_conf_min), 2)
            patch['strategies'].setdefault('MicroVWAPRevert', {})['vwap_z_min'] = round(_clamp(zv, *self.bounds.microvwap_z_min), 2)
            patch['strategies'].setdefault('VolCompressionBreak', {})['accel_pctile'] = int(_clamp(ap, *self.bounds.volcomp_accel_pctile))
            patch['strategies'].setdefault('BB_RSI_Fast', {})['reentry_block_s'] = int(_clamp(rb, *self.bounds.bb_rsi_reentry_block_s))

        # Allocation tweak (quiet_low_vol only) using micro success
        tps = metrics.get('tp_rate_by_strategy', {})
        micro_names = ['MomentumPulse','VolCompressionBreak','MicroVWAPRevert','BB_RSI_Fast']
        micro_tp = [tps.get(n, 0.0) for n in micro_names]
        micro_tp_ok = (sum(micro_tp)/max(1,len(micro_tp))) if micro_tp else 0.0
        base_share = preset.get('alloc', {}).get('regime', {}).get('quiet_low_vol', {}).get('micro_share', 0.35)
        if micro_tp_ok > 0.65:
            micro_share = base_share + 0.05
        else:
            micro_share = base_share - 0.05
        patch.setdefault('alloc', {}).setdefault('regime', {}).setdefault('quiet_low_vol', {})['micro_share'] = round(_clamp(micro_share, *self.bounds.micro_share), 2)

        # Safe exploration
        if random.random() < self.random_explore:
            patch['meta']['explore'] = True
        return patch

    def run_once(self, logs_glob: str, shadow: bool=False) -> str:
        # Ensure INFO-level structured logs show up under systemd/journalctl.
        if not logging.getLogger().handlers:
            logging.basicConfig(level=logging.INFO, format="%(message)s")

        preset = _load_yaml(self.presets_path)
        recent = self.load_recent(logs_glob)
        metrics = self.compute_metrics(recent)

        state = self._load_state() if not shadow else {"version": 1}
        current_overrides = _load_yaml(self.overrides_out)

        ok, gate_reasons, quality = self._quality_gate(recent, metrics)
        summary = {
            "n_trades": int(metrics.get("n_trades", 0) or 0),
            "ev": float(metrics.get("ev", 0.0) or 0.0),
            "pf": float(metrics.get("pf", 0.0) or 0.0),
            "win_rate": float(metrics.get("win_rate", 0.0) or 0.0),
            "target_regime_pct": float(quality.get("target_regime_pct", 0.0) or 0.0),
            "reason_nonempty_pct": float(quality.get("reason_nonempty_pct", 1.0) or 0.0),
            "micro_trades": int(quality.get("micro_trades", 0) or 0),
        }

        if shadow:
            patch = self.propose_patch(metrics, preset)
            patch["meta"]["decision"] = "shadow"
            patch["meta"]["metrics"] = summary
            hist_payload = patch
            decision = {"decision": "shadow", "gate_ok": ok, "gate_reasons": gate_reasons}
        else:
            self._maybe_bootstrap_lkg(state, current_overrides)

            # Update last-run metrics in state for ops visibility.
            state["last_run_at"] = _utc_iso()
            state["last_metrics"] = summary

            if not ok:
                patch = {"meta": {"generated_at": _utc_iso(), "window_min": self.minutes}}
                patch["meta"]["decision"] = "freeze"
                patch["meta"]["freeze_reasons"] = gate_reasons
                patch["meta"]["metrics"] = summary
                hist_payload = patch
                decision = {"decision": "freeze", "gate_ok": False, "gate_reasons": gate_reasons}
                state["last_decision"] = "freeze"
                self._save_state(state)
            else:
                # LKG captures the config that produced this window's outcomes.
                lkg_updated = self._update_lkg_if_good(state, metrics, current_overrides)

                # Rollback is only considered when the window is "bad" enough, repeatedly.
                do_rb, rb_reasons = self._should_rollback(state, metrics)
                if do_rb:
                    rolled = self._rollback_to_lkg(state)
                    patch = {"meta": {"generated_at": _utc_iso(), "window_min": self.minutes}}
                    patch["meta"]["decision"] = "rollback" if rolled else "rollback_failed"
                    patch["meta"]["rollback_reasons"] = rb_reasons
                    patch["meta"]["metrics"] = summary
                    hist_payload = _load_yaml(self.overrides_out)
                    if not isinstance(hist_payload, dict):
                        hist_payload = {}
                    hist_payload.setdefault("meta", {}).update(patch["meta"])
                    decision = {
                        "decision": patch["meta"]["decision"],
                        "rollback_reasons": rb_reasons,
                        "lkg_updated": lkg_updated,
                        "changes": _diff_roots(current_overrides, hist_payload),
                    }
                    state["last_decision"] = patch["meta"]["decision"]
                    self._save_state(state)
                else:
                    is_bad, bad_reasons = self._is_bad_window(metrics)
                    if is_bad:
                        patch = {"meta": {"generated_at": _utc_iso(), "window_min": self.minutes}}
                        patch["meta"]["decision"] = "hold_bad_window"
                        patch["meta"]["bad_reasons"] = bad_reasons
                        patch["meta"]["metrics"] = summary
                        hist_payload = patch
                        decision = {"decision": "hold_bad_window", "bad_reasons": bad_reasons, "lkg_updated": lkg_updated}
                        state["last_decision"] = "hold_bad_window"
                        self._save_state(state)
                    else:
                        patch = self.propose_patch(metrics, preset)

                        # Prevent alloc drift when micro sample size is too small.
                        if int(quality.get("micro_trades", 0) or 0) < self.min_micro_trades:
                            if "alloc" in patch:
                                try:
                                    del patch["alloc"]
                                except Exception:
                                    pass
                            patch.setdefault("meta", {})["alloc_skipped"] = True
                            patch["meta"]["alloc_skip_reason"] = "min_micro_trades"

                        patch["meta"]["decision"] = "apply"
                        patch["meta"]["metrics"] = summary

                        # Merge with existing overrides (shallow merge)
                        overrides = current_overrides
                        merged = overrides.copy()
                        def deep_update(d: dict, u: dict):
                            for k,v in u.items():
                                if isinstance(v, dict):
                                    d[k] = deep_update(d.get(k, {}), v)
                                else:
                                    d[k] = v
                            return d
                        merged = deep_update(merged, patch)

                        hist_payload = merged
                        if not shadow:
                            _dump_yaml(merged, self.overrides_out)

                        state["last_apply_at"] = _utc_iso()
                        state["last_decision"] = "apply"
                        state["bad_runs"] = 0
                        self._save_state(state)
                        decision = {
                            "decision": "apply",
                            "lkg_updated": lkg_updated,
                            "alloc_skipped": patch.get("meta", {}).get("alloc_skipped", False),
                            "changes": _diff_roots(current_overrides, merged),
                        }

        ts = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        os.makedirs(self.history_dir, exist_ok=True)
        hist_path = os.path.join(self.history_dir, f"tuning_{ts}.yaml")
        _dump_yaml(hist_payload, hist_path)

        _log_json(
            logging.INFO,
            {
                "tag": "online_tuner",
                "event": "run",
                "shadow": bool(shadow),
                "presets_path": self.presets_path,
                "overrides_out": self.overrides_out,
                "history_path": hist_path,
                "quality": quality,
                "decision": decision,
                "metrics": summary,
            },
        )
        return hist_path
