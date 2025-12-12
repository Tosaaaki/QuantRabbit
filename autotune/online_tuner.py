from __future__ import annotations

import glob
import os
import random
import datetime as dt
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
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        yaml.safe_dump(data, f, sort_keys=False)

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
    history_dir: str = "config/tuning_history"
    minutes: int = 15
    bounds: TunerBounds = field(default_factory=TunerBounds)
    random_explore: float = 0.02   # 2% safe exploration
    clamp_step: float = 0.15       # max delta for weights/alloc

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
        # Reasons split
        if 'reason' in df.columns:
            rs = df['reason'].value_counts(normalize=True)
            m['reasons_pct'] = rs.to_dict()
            rc = df['reason'].value_counts()
            m['reasons_cnt'] = rc.to_dict()
        # EV
        if 'pips' in df.columns:
            m['ev'] = float(df['pips'].mean())
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
        return m

    def propose_patch(self, metrics: Dict[str, Any], preset: Dict[str, Any]) -> Dict[str, Any]:
        patch = {'meta':{
            'generated_at': dt.datetime.utcnow().isoformat()+'Z',
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
        preset = _load_yaml(self.presets_path)
        recent = self.load_recent(logs_glob)
        metrics = self.compute_metrics(recent)
        patch = self.propose_patch(metrics, preset)

        # Merge with existing overrides (shallow merge)
        overrides = _load_yaml(self.overrides_out)
        merged = overrides.copy()
        def deep_update(d: dict, u: dict):
            for k,v in u.items():
                if isinstance(v, dict):
                    d[k] = deep_update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        merged = deep_update(merged, patch)

        ts = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        os.makedirs(self.history_dir, exist_ok=True)
        hist_path = os.path.join(self.history_dir, f"tuning_{ts}.yaml")
        if not shadow:
            _dump_yaml(merged, self.overrides_out)
        _dump_yaml(merged if not shadow else patch, hist_path)
        return hist_path
