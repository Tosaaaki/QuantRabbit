"""
analysis.forecast_sklearn
~~~~~~~~~~~~~~~~~~~~~~~~
Multi-horizon probabilistic forecast utilities (USD/JPY) using scikit-learn.

The system goal is not point prediction, but stable probability + range estimates
that can be used as an execution gate / sizing modifier.

Notes
-----
- This module is offline-friendly: it trains from OHLC candles and produces a
  joblib bundle containing per-horizon models.
- Training uses a time-ordered split (train -> calibrate -> test) to avoid
  look-ahead leakage.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from pathlib import Path
from statistics import NormalDist
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PIP_USDJPY = 0.01
_NORMAL_DIST = NormalDist()
_RANGE_LOW_Q = 0.20
_RANGE_HIGH_Q = 0.80


@dataclass(frozen=True, slots=True)
class HorizonSpec:
    name: str  # "1h" | "8h" | "1d" | "1w" | "1m"
    timeframe: str  # "M5" | "H1" | "D1"
    step_bars: int
    min_move_pips: float
    calibration: str = "sigmoid"  # "sigmoid" | "isotonic" | "off"


DEFAULT_HORIZONS: tuple[HorizonSpec, ...] = (
    HorizonSpec("1h", timeframe="M5", step_bars=12, min_move_pips=0.35, calibration="sigmoid"),
    HorizonSpec("8h", timeframe="M5", step_bars=96, min_move_pips=1.0, calibration="sigmoid"),
    HorizonSpec("1d", timeframe="H1", step_bars=24, min_move_pips=2.0, calibration="sigmoid"),
    # D1 candles are trading-day bars (no weekend). Interpret as ~5D / ~21D.
    HorizonSpec("1w", timeframe="D1", step_bars=5, min_move_pips=5.0, calibration="sigmoid"),
    HorizonSpec("1m", timeframe="D1", step_bars=21, min_move_pips=10.0, calibration="sigmoid"),
)


@dataclass(slots=True)
class ForecastModel:
    horizon: HorizonSpec
    feature_names: list[str]
    pipeline: Pipeline
    calibrator: object | None
    mean_up_pips: float
    mean_down_pips: float
    trained_until: str
    n_train: int
    n_calib: int
    n_test: int

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        base = self.pipeline.predict_proba(X)[:, 1]
        prob = base
        if self.calibrator is not None:
            try:
                # IsotonicRegression
                prob = self.calibrator.predict(base)  # type: ignore[attr-defined]
            except Exception:
                # LogisticRegression calibrator
                prob = self.calibrator.predict_proba(base.reshape(-1, 1))[:, 1]  # type: ignore[attr-defined]
        prob = np.clip(prob, 0.0, 1.0)
        return np.column_stack([1.0 - prob, prob])

    def expected_pips(self, p_up: float) -> float:
        return float(p_up) * float(self.mean_up_pips) + (1.0 - float(p_up)) * float(
            self.mean_down_pips
        )


@dataclass(slots=True)
class ForecastBundle:
    instrument: str
    created_at: str
    pip_size: float
    models: dict[str, ForecastModel]  # horizon.name -> model


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _estimate_dispersion_pips(model: ForecastModel) -> float:
    spread = abs(float(model.mean_up_pips) - float(model.mean_down_pips))
    step_term = 0.22 * math.sqrt(max(1.0, float(model.horizon.step_bars)))
    move_term = max(0.20, 0.75 * float(model.horizon.min_move_pips))
    sigma = 0.42 * spread + move_term + step_term
    if not math.isfinite(sigma):
        return 0.50
    return max(0.25, float(sigma))


def _quantile_from_normal(*, mean_pips: float, sigma_pips: float, quantile: float) -> float:
    q = min(0.99, max(0.01, float(quantile)))
    sigma = max(1e-4, abs(float(sigma_pips)))
    try:
        z = _NORMAL_DIST.inv_cdf(q)
    except Exception:
        z = 0.0
    value = float(mean_pips) + float(z) * sigma
    if not math.isfinite(value):
        return float(mean_pips)
    return value


def _build_range_band(*, expected_pips: float, sigma_pips: float) -> dict[str, float]:
    low = _quantile_from_normal(mean_pips=expected_pips, sigma_pips=sigma_pips, quantile=_RANGE_LOW_Q)
    high = _quantile_from_normal(
        mean_pips=expected_pips,
        sigma_pips=sigma_pips,
        quantile=_RANGE_HIGH_Q,
    )
    if low > high:
        low, high = high, low
    return {
        "q10_pips": round(
            _quantile_from_normal(mean_pips=expected_pips, sigma_pips=sigma_pips, quantile=0.10),
            4,
        ),
        "q50_pips": round(float(expected_pips), 4),
        "q90_pips": round(
            _quantile_from_normal(mean_pips=expected_pips, sigma_pips=sigma_pips, quantile=0.90),
            4,
        ),
        "range_low_pips": round(float(low), 4),
        "range_high_pips": round(float(high), 4),
        "range_sigma_pips": round(max(0.0, float(sigma_pips)), 4),
    }


def _to_frame(candles: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for c in candles:
        if not isinstance(c, Mapping):
            continue
        ts = c.get("timestamp") or c.get("time") or c.get("ts")
        row = {
            "timestamp": ts,
            "open": c.get("open"),
            "high": c.get("high"),
            "low": c.get("low"),
            "close": c.get("close"),
        }
        rows.append(row)
    if not rows:
        return pd.DataFrame(
            columns=["open", "high", "low", "close"],
            index=pd.DatetimeIndex([], tz="UTC"),
        )
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).set_index("timestamp")
    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"]).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def resample_ohlc(
    candles: Sequence[Mapping[str, object]],
    *,
    rule: str,
    label: str = "right",
    closed: str = "right",
) -> list[dict[str, object]]:
    """
    Resample candle sequence into a higher timeframe.

    rule examples: "5min" (M5), "1H" (H1), "1D" (D1).
    """
    df = _to_frame(candles)
    if df.empty:
        return []
    agg = df.resample(rule, label=label, closed=closed).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
    )
    agg = agg.dropna(subset=["open", "high", "low", "close"])
    out: list[dict[str, object]] = []
    for ts, row in agg.iterrows():
        out.append(
            {
                "timestamp": ts.isoformat(),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
            }
        )
    return out


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def compute_feature_frame(
    candles: Sequence[Mapping[str, object]],
    *,
    pip_size: float = PIP_USDJPY,
    range_window: int = 20,
) -> pd.DataFrame:
    df = _to_frame(candles)
    if df.empty:
        return df

    close = df["close"]
    high = df["high"]
    low = df["low"]

    ret1 = (close - close.shift(1)) / pip_size
    ret3 = (close - close.shift(3)) / pip_size
    ret12 = (close - close.shift(12)) / pip_size

    ma10 = close.rolling(10).mean()
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    trend_window_short = 20
    trend_window_long = 50
    slope20 = (close - close.shift(trend_window_short - 1)) / max(1, trend_window_short - 1)
    slope50 = (close - close.shift(trend_window_long - 1)) / max(1, trend_window_long - 1)

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr14 = tr.rolling(14).mean() / pip_size

    vol20 = ret1.rolling(20).std()

    hh = high.rolling(range_window).max()
    ll = low.rolling(range_window).min()
    hh_prev = high.shift(1).rolling(range_window).max()
    ll_prev = low.shift(1).rolling(range_window).min()
    span = (hh - ll).replace(0.0, np.nan)
    range_pos = ((close - ll) / span).clip(0.0, 1.0).fillna(0.5)
    sr_span_pips = span / pip_size
    support_gap_pips = (close - ll) / pip_size
    resistance_gap_pips = (hh - close) / pip_size
    breakout_up_pips = (close - hh_prev) / pip_size
    breakout_down_pips = (ll_prev - close) / pip_size
    sr_balance = ((support_gap_pips - resistance_gap_pips) / sr_span_pips).clip(-1.0, 1.0)
    compression_ratio = vol20 / sr_span_pips.replace(0.0, np.nan)
    trend_pullback_norm = ((close - ma20) / pip_size) / atr14.replace(0.0, np.nan)
    trend_accel = (slope20 - slope50) / pip_size

    idx = df.index
    hour = idx.hour.to_numpy()
    dow = idx.dayofweek.to_numpy()
    hour_sin = np.sin(2.0 * math.pi * hour / 24.0)
    hour_cos = np.cos(2.0 * math.pi * hour / 24.0)
    dow_sin = np.sin(2.0 * math.pi * dow / 7.0)
    dow_cos = np.cos(2.0 * math.pi * dow / 7.0)

    return pd.DataFrame(
        {
            "ret_pips_1": ret1,
            "ret_pips_3": ret3,
            "ret_pips_12": ret12,
            "ma_gap_pips_10_20": (ma10 - ma20) / pip_size,
            "close_ma20_pips": (close - ma20) / pip_size,
            "close_ma50_pips": (close - ma50) / pip_size,
            "trend_slope_pips_20": slope20 / pip_size,
            "trend_slope_pips_50": slope50 / pip_size,
            "trend_accel_pips": trend_accel,
            "atr_pips_14": atr14,
            "vol_pips_20": vol20,
            "rsi_14": _rsi(close, 14),
            "range_pos": range_pos,
            "support_gap_pips_20": support_gap_pips,
            "resistance_gap_pips_20": resistance_gap_pips,
            "sr_balance_20": sr_balance,
            "breakout_up_pips_20": breakout_up_pips,
            "breakout_down_pips_20": breakout_down_pips,
            "donchian_width_pips_20": sr_span_pips,
            "range_compression_20": compression_ratio,
            "trend_pullback_norm_20": trend_pullback_norm,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "dow_sin": dow_sin,
            "dow_cos": dow_cos,
        },
        index=df.index,
    )


def build_direction_dataset(
    candles: Sequence[Mapping[str, object]],
    *,
    horizon: HorizonSpec,
    pip_size: float = PIP_USDJPY,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], pd.DatetimeIndex]:
    """
    Returns (X, y, future_pips, feature_names, timestamps).

    y:
      1 if future_pips > +min_move_pips
      0 if future_pips < -min_move_pips
      other samples are dropped (to reduce "noise labels").
    """
    base = _to_frame(candles)
    if base.empty:
        raise ValueError("empty_candles")
    feats = compute_feature_frame(candles, pip_size=pip_size)
    step = int(horizon.step_bars)
    future_pips = (base["close"].shift(-step) - base["close"]) / pip_size

    df = feats.copy()
    df["future_pips"] = future_pips
    thr = float(horizon.min_move_pips)
    df["label"] = np.where(
        df["future_pips"] > thr,
        1,
        np.where(df["future_pips"] < -thr, 0, np.nan),
    )

    feature_names = list(feats.columns)
    df = df.dropna(subset=feature_names + ["label", "future_pips"])
    X = df[feature_names].to_numpy(dtype=float)
    y = df["label"].to_numpy(dtype=int)
    fut = df["future_pips"].to_numpy(dtype=float)
    ts = df.index

    if len(X) < 50:
        raise ValueError(f"insufficient_samples:{len(X)}")
    if len(set(y.tolist())) < 2:
        raise ValueError("single_class_after_threshold")
    return X, y, fut, feature_names, ts


def _fit_calibrator(method: str, base_prob: np.ndarray, y: np.ndarray) -> object | None:
    mode = (method or "").strip().lower()
    if mode in {"", "off", "none", "0", "false"}:
        return None
    if len(set(y.tolist())) < 2:
        return None
    if mode == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(base_prob, y)
        return iso
    # sigmoid / default
    lr = LogisticRegression(solver="lbfgs")
    lr.fit(base_prob.reshape(-1, 1), y)
    return lr


def train_forecast_model(
    candles: Sequence[Mapping[str, object]],
    *,
    horizon: HorizonSpec,
    pip_size: float = PIP_USDJPY,
    train_frac: float = 0.7,
    calib_frac: float = 0.15,
    random_state: int = 42,
) -> tuple[ForecastModel, dict[str, float]]:
    X, y, fut, feature_names, ts = build_direction_dataset(
        candles, horizon=horizon, pip_size=pip_size
    )

    n = len(X)
    train_end = max(10, int(n * float(train_frac)))
    calib_end = max(train_end + 10, int(n * float(train_frac + calib_frac)))
    calib_end = min(calib_end, n - 5)

    X_train, y_train = X[:train_end], y[:train_end]
    X_cal, y_cal = X[train_end:calib_end], y[train_end:calib_end]
    X_test, y_test = X[calib_end:], y[calib_end:]

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                SGDClassifier(
                    loss="log_loss",
                    penalty="l2",
                    alpha=0.0008,
                    max_iter=2000,
                    tol=1e-3,
                    random_state=random_state,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)

    cal_base = pipeline.predict_proba(X_cal)[:, 1]
    calibrator = _fit_calibrator(horizon.calibration, cal_base, y_cal)

    def _mean_or_fallback(values: np.ndarray, fallback: float) -> float:
        if values.size <= 0:
            return float(fallback)
        try:
            val = float(np.mean(values))
        except Exception:
            return float(fallback)
        if math.isnan(val) or math.isinf(val):
            return float(fallback)
        return val

    train_fut = fut[:train_end]
    mean_all = _mean_or_fallback(train_fut, 0.0)
    mean_up = _mean_or_fallback(train_fut[y_train == 1], mean_all)
    mean_down = _mean_or_fallback(train_fut[y_train == 0], -abs(mean_all))

    model = ForecastModel(
        horizon=horizon,
        feature_names=feature_names,
        pipeline=pipeline,
        calibrator=calibrator,
        mean_up_pips=mean_up,
        mean_down_pips=mean_down,
        trained_until=str(ts[-1].to_pydatetime().isoformat()),
        n_train=len(X_train),
        n_calib=len(X_cal),
        n_test=len(X_test),
    )

    prob_test = model.predict_proba(X_test)[:, 1]
    metrics: dict[str, float] = {
        "n": float(n),
        "n_train": float(len(X_train)),
        "n_calib": float(len(X_cal)),
        "n_test": float(len(X_test)),
        "pos_rate_test": float(np.mean(y_test)),
        "logloss": float(log_loss(y_test, np.column_stack([1 - prob_test, prob_test]))),
        "brier": float(brier_score_loss(y_test, prob_test)),
    }
    try:
        if len(set(y_test.tolist())) >= 2:
            metrics["auc"] = float(roc_auc_score(y_test, prob_test))
    except Exception:
        pass
    return model, metrics


def train_bundle(
    instrument: str,
    candles_by_tf: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    horizons: Sequence[HorizonSpec] = DEFAULT_HORIZONS,
    pip_size: float = PIP_USDJPY,
) -> tuple[ForecastBundle, dict[str, dict[str, float]]]:
    models: dict[str, ForecastModel] = {}
    reports: dict[str, dict[str, float]] = {}
    for spec in horizons:
        candles = candles_by_tf.get(spec.timeframe) or []
        model, metrics = train_forecast_model(
            candles,
            horizon=spec,
            pip_size=pip_size,
        )
        models[spec.name] = model
        reports[spec.name] = metrics
    bundle = ForecastBundle(
        instrument=instrument,
        created_at=utc_now_iso(),
        pip_size=float(pip_size),
        models=models,
    )
    return bundle, reports


def save_bundle(bundle: ForecastBundle, path: str | Path) -> None:
    import joblib

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out)


def load_bundle(path: str | Path) -> ForecastBundle:
    import joblib

    bundle = joblib.load(path)
    if not isinstance(bundle, ForecastBundle):
        raise TypeError("invalid_bundle_type")
    return bundle


def predict_latest(
    bundle: ForecastBundle,
    candles_by_tf: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    as_of: str | None = None,
) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for name, model in bundle.models.items():
        candles = candles_by_tf.get(model.horizon.timeframe) or []
        feats = compute_feature_frame(candles, pip_size=bundle.pip_size)
        if feats.empty:
            raise ValueError(f"empty_features:{model.horizon.timeframe}")
        feats = feats.dropna(subset=model.feature_names)
        if feats.empty:
            raise ValueError(f"insufficient_feature_history:{model.horizon.timeframe}")
        x_last = feats[model.feature_names].to_numpy(dtype=float)[-1:, :]
        p_up = float(model.predict_proba(x_last)[0, 1])
        expected_pips = float(model.expected_pips(p_up))
        dispersion_pips = _estimate_dispersion_pips(model)
        range_band = _build_range_band(expected_pips=expected_pips, sigma_pips=dispersion_pips)
        out[name] = {
            "instrument": bundle.instrument,
            "horizon": name,
            "timeframe": model.horizon.timeframe,
            "step_bars": int(model.horizon.step_bars),
            "min_move_pips": float(model.horizon.min_move_pips),
            "p_up": round(p_up, 6),
            "expected_pips": round(expected_pips, 4),
            "dispersion_pips": round(dispersion_pips, 4),
            "trained_until": model.trained_until,
            "as_of": as_of or utc_now_iso(),
            "feature_ts": feats.index[-1].to_pydatetime().isoformat(),
            "n_train": model.n_train,
            "n_calib": model.n_calib,
            "n_test": model.n_test,
            **range_band,
        }
    return out


__all__ = [
    "HorizonSpec",
    "DEFAULT_HORIZONS",
    "ForecastModel",
    "ForecastBundle",
    "resample_ohlc",
    "compute_feature_frame",
    "build_direction_dataset",
    "train_forecast_model",
    "train_bundle",
    "save_bundle",
    "load_bundle",
    "predict_latest",
]
