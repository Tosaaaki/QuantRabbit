"""Deep statistical pattern mining for QuantRabbit trade history."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest, mannwhitneyu
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


@dataclass(slots=True)
class DeepPatternConfig:
    min_samples: int = 30
    prior_strength: int = 24
    recent_days: int = 5
    baseline_days: int = 30
    min_recent_samples: int = 8
    min_prev_samples: int = 20
    bootstrap_samples: int = 240
    cluster_min: int = 3
    cluster_max: int = 8
    cluster_min_samples: int = 20
    random_state: int = 42


def _parse_pattern_tokens(pattern_id: Any) -> dict[str, str]:
    tokens: dict[str, str] = {}
    for part in str(pattern_id or "").split("|"):
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        key = key.strip().lower()
        value = value.strip().lower()
        if key:
            tokens[key] = value
    return tokens


def _is_chase_risk(side: str, range_bucket: str) -> int:
    if side == "long" and range_bucket in {"high", "top"}:
        return 1
    if side == "short" and range_bucket in {"low", "bot"}:
        return 1
    return 0


def _df_records(df: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    if df.empty:
        return []
    clipped = df.loc[:, columns]
    return json.loads(clipped.to_json(orient="records", date_format="iso", force_ascii=False))


def _bootstrap_ci(
    values: np.ndarray,
    *,
    n_samples: int,
    seed: int,
) -> tuple[float, float]:
    if values.size < 8:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    sample_idx = rng.integers(0, values.size, size=(n_samples, values.size))
    means = values[sample_idx].mean(axis=1)
    low = float(np.quantile(means, 0.1))
    high = float(np.quantile(means, 0.9))
    return low, high


def _compute_pattern_scores(
    frame: pd.DataFrame,
    *,
    config: DeepPatternConfig,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()

    aggregate = (
        frame.groupby(
            ["pattern_id", "pocket", "strategy_tag", "direction"],
            dropna=False,
            as_index=False,
        )
        .agg(
            trades=("pl_pips", "size"),
            wins=("win", "sum"),
            avg_pips=("pl_pips", "mean"),
            std_pips=("pl_pips", "std"),
            total_pips=("pl_pips", "sum"),
            gross_profit=("pl_pips", lambda s: float(s[s > 0].sum())),
            gross_loss=("pl_pips", lambda s: float((-s[s < 0]).sum())),
            avg_hold_sec=("hold_sec", "mean"),
            spread_pips_mean=("spread_pips", "mean"),
            tp_pips_mean=("tp_pips", "mean"),
            sl_pips_mean=("sl_pips", "mean"),
            confidence_mean=("confidence", "mean"),
            chase_risk_rate=("is_chase_risk", "mean"),
            last_close_time=("close_time", "max"),
        )
    )
    aggregate["losses"] = aggregate["trades"] - aggregate["wins"]
    aggregate["win_rate"] = aggregate["wins"] / aggregate["trades"].clip(lower=1.0)
    aggregate["bayes_win_rate"] = (aggregate["wins"] + 2.0) / (aggregate["trades"] + 4.0)
    aggregate["profit_factor"] = np.where(
        aggregate["gross_loss"] <= 1e-9,
        np.where(aggregate["gross_profit"] > 0.0, 9.99, 0.0),
        aggregate["gross_profit"] / aggregate["gross_loss"].clip(lower=1e-9),
    )
    aggregate["profit_factor"] = aggregate["profit_factor"].clip(lower=0.0, upper=9.99)

    baseline = (
        frame.groupby(["pocket", "direction"], as_index=False)
        .agg(
            base_trades=("pl_pips", "size"),
            base_win_rate=("win", "mean"),
            base_avg_pips=("pl_pips", "mean"),
            base_std_pips=("pl_pips", "std"),
        )
        .copy()
    )
    aggregate = aggregate.merge(
        baseline,
        how="left",
        on=["pocket", "direction"],
    )

    global_avg = float(frame["pl_pips"].mean())
    global_std = float(frame["pl_pips"].std(ddof=1))
    if not np.isfinite(global_std) or global_std < 0.05:
        global_std = 0.4
    global_wr = float(frame["win"].mean())
    if not np.isfinite(global_wr):
        global_wr = 0.5
    prior = max(1, int(config.prior_strength))

    base_avg = aggregate["base_avg_pips"].fillna(global_avg)
    base_std = aggregate["base_std_pips"].fillna(global_std).clip(lower=max(0.05, global_std * 0.1))
    aggregate["shrink_avg_pips"] = (
        aggregate["avg_pips"] * aggregate["trades"] + base_avg * prior
    ) / (aggregate["trades"] + prior)
    std_mix = aggregate["std_pips"].fillna(base_std).fillna(global_std).clip(lower=0.05)
    denom = (std_mix / np.sqrt(aggregate["trades"] + prior)).clip(lower=1e-5)
    aggregate["z_edge"] = (aggregate["shrink_avg_pips"] - base_avg) / denom

    def _pvalue(row: pd.Series) -> float:
        trades = int(row["trades"])
        wins = int(row["wins"])
        if trades <= 0:
            return 1.0
        p0 = row.get("base_win_rate")
        if p0 is None or not np.isfinite(float(p0)):
            p0 = global_wr
        p0 = float(np.clip(float(p0), 0.05, 0.95))
        return float(binomtest(wins, trades, p=p0, alternative="two-sided").pvalue)

    aggregate["p_value"] = aggregate.apply(_pvalue, axis=1)
    aggregate["sample_weight"] = 1.0 - np.exp(-aggregate["trades"] / 40.0)
    aggregate["sig_weight"] = np.clip(1.0 - aggregate["p_value"], 0.25, 1.0)
    aggregate["chase_penalty"] = np.where(
        (aggregate["chase_risk_rate"] >= 0.99) & (aggregate["win_rate"] < 0.52),
        0.45,
        0.0,
    )
    aggregate["robust_score"] = (
        aggregate["z_edge"] * aggregate["sample_weight"] * aggregate["sig_weight"]
        - aggregate["chase_penalty"]
    ).clip(lower=-4.0, upper=4.0)
    raw_mult = 0.65 + 0.7 / (1.0 + np.exp(-aggregate["robust_score"]))
    aggregate["suggested_multiplier"] = np.where(
        aggregate["trades"] < config.min_samples,
        1.0,
        np.clip(raw_mult, 0.65, 1.35),
    )
    aggregate["is_significant"] = (
        (aggregate["trades"] >= config.min_samples)
        & (aggregate["p_value"] <= 0.10)
        & (aggregate["z_edge"].abs() >= 1.0)
    ).astype(int)

    def _quality(row: pd.Series) -> str:
        trades = int(row["trades"])
        score = float(row["robust_score"])
        pval = float(row["p_value"])
        wr = float(row["win_rate"])
        shrink = float(row["shrink_avg_pips"])
        if trades < config.min_samples:
            return "learn_only"
        if score >= 1.0 and pval <= 0.20 and wr >= 0.56 and shrink > 0.0:
            return "robust"
        if score >= 0.45 and shrink > 0.0:
            return "candidate"
        if score <= -1.0 and pval <= 0.35:
            return "avoid"
        if score <= -0.45:
            return "weak"
        return "neutral"

    aggregate["quality"] = aggregate.apply(_quality, axis=1)
    aggregate = aggregate.sort_values(
        by=["robust_score", "shrink_avg_pips", "trades"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    aggregate["score_rank"] = np.arange(1, len(aggregate) + 1, dtype=int)

    by_pattern = {
        pattern_id: group["pl_pips"].to_numpy(dtype=float)
        for pattern_id, group in frame.groupby("pattern_id", sort=False)
    }
    ci_low: list[float] = []
    ci_high: list[float] = []
    for row in aggregate.itertuples(index=False):
        if int(row.trades) < config.min_samples:
            ci_low.append(float("nan"))
            ci_high.append(float("nan"))
            continue
        values = by_pattern.get(str(row.pattern_id))
        if values is None:
            ci_low.append(float("nan"))
            ci_high.append(float("nan"))
            continue
        low, high = _bootstrap_ci(
            values,
            n_samples=max(80, int(config.bootstrap_samples)),
            seed=(hash(row.pattern_id) ^ config.random_state) & 0xFFFF_FFFF,
        )
        ci_low.append(low)
        ci_high.append(high)
    aggregate["boot_ci_low"] = ci_low
    aggregate["boot_ci_high"] = ci_high
    return aggregate


def _compute_drift(
    frame: pd.DataFrame,
    *,
    as_of: pd.Timestamp,
    config: DeepPatternConfig,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()

    baseline_from = as_of - pd.Timedelta(days=max(1, config.baseline_days))
    recent_from = as_of - pd.Timedelta(days=max(1, config.recent_days))
    window = frame.loc[
        (frame["close_dt"] >= baseline_from) & (frame["close_dt"] <= as_of)
    ].copy()
    if window.empty:
        return pd.DataFrame()

    payload: list[dict[str, Any]] = []
    for pattern_id, group in window.groupby("pattern_id", sort=False):
        recent = group.loc[group["close_dt"] >= recent_from, "pl_pips"].to_numpy(dtype=float)
        prev = group.loc[group["close_dt"] < recent_from, "pl_pips"].to_numpy(dtype=float)
        if recent.size < config.min_recent_samples or prev.size < config.min_prev_samples:
            continue
        try:
            pval = float(mannwhitneyu(recent, prev, alternative="two-sided").pvalue)
        except ValueError:
            pval = 1.0
        recent_avg = float(np.mean(recent))
        prev_avg = float(np.mean(prev))
        recent_wr = float(np.mean(recent > 0.0))
        prev_wr = float(np.mean(prev > 0.0))
        delta_avg = recent_avg - prev_avg
        delta_wr = recent_wr - prev_wr
        if pval <= 0.10 and delta_avg <= -0.08 and delta_wr <= -0.06:
            state = "deterioration"
        elif pval <= 0.10 and delta_avg >= 0.08 and delta_wr >= 0.06:
            state = "improvement"
        elif delta_avg <= -0.12:
            state = "soft_deterioration"
        elif delta_avg >= 0.12:
            state = "soft_improvement"
        else:
            state = "stable"
        payload.append(
            {
                "pattern_id": str(pattern_id),
                "recent_trades": int(recent.size),
                "prev_trades": int(prev.size),
                "recent_avg_pips": recent_avg,
                "prev_avg_pips": prev_avg,
                "delta_avg_pips": delta_avg,
                "recent_win_rate": recent_wr,
                "prev_win_rate": prev_wr,
                "delta_win_rate": delta_wr,
                "p_value": pval,
                "drift_state": state,
            }
        )
    if not payload:
        return pd.DataFrame()
    return (
        pd.DataFrame(payload)
        .sort_values(by=["delta_avg_pips", "p_value"], ascending=[True, True])
        .reset_index(drop=True)
    )


def _cluster_patterns(
    score_df: pd.DataFrame,
    *,
    config: DeepPatternConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if score_df.empty:
        return pd.DataFrame(), pd.DataFrame(), {"k": 0, "silhouette": 0.0}

    source = score_df.loc[score_df["trades"] >= max(1, config.cluster_min_samples)].copy()
    if source.empty:
        source = score_df.head(min(len(score_df), 40)).copy()
    if source.empty:
        return pd.DataFrame(), pd.DataFrame(), {"k": 0, "silhouette": 0.0}

    features = source[
        [
            "bayes_win_rate",
            "shrink_avg_pips",
            "profit_factor",
            "avg_hold_sec",
            "spread_pips_mean",
            "tp_pips_mean",
            "sl_pips_mean",
            "chase_risk_rate",
            "robust_score",
        ]
    ].copy()
    features["log_trades"] = np.log1p(source["trades"].astype(float))
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(features.median(numeric_only=True)).fillna(0.0)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    sample_size = len(source)
    best_labels = np.zeros(sample_size, dtype=int)
    best_k = 1
    best_sil = 0.0

    min_k = max(2, int(config.cluster_min))
    max_k = min(int(config.cluster_max), sample_size - 1)
    if sample_size >= 6 and min_k <= max_k:
        for k in range(min_k, max_k + 1):
            model = KMeans(n_clusters=k, random_state=config.random_state, n_init=20)
            labels = model.fit_predict(scaled)
            if len(set(labels)) < 2:
                continue
            sil = float(silhouette_score(scaled, labels))
            if sil > best_sil:
                best_sil = sil
                best_k = k
                best_labels = labels

    mapped = source.copy()
    mapped["cluster_id"] = best_labels.astype(int)
    cluster_map = mapped[["pattern_id", "cluster_id", "robust_score"]].rename(
        columns={"robust_score": "cluster_score"}
    )
    summary = (
        mapped.groupby("cluster_id", as_index=False)
        .agg(
            patterns=("pattern_id", "size"),
            mean_trades=("trades", "mean"),
            mean_win_rate=("win_rate", "mean"),
            mean_avg_pips=("avg_pips", "mean"),
            mean_pf=("profit_factor", "mean"),
            mean_score=("robust_score", "mean"),
            quality_mix=(
                "quality",
                lambda s: json.dumps(
                    {str(k): int(v) for k, v in s.value_counts().to_dict().items()},
                    ensure_ascii=False,
                ),
            ),
        )
        .sort_values(by="mean_score", ascending=False)
        .reset_index(drop=True)
    )
    return cluster_map, summary, {"k": best_k, "silhouette": round(best_sil, 4)}


def _ensure_deep_schema(con: Any) -> None:
    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS pattern_scores (
          pattern_id TEXT PRIMARY KEY,
          strategy_tag TEXT NOT NULL,
          pocket TEXT NOT NULL,
          direction TEXT NOT NULL,
          trades INTEGER NOT NULL,
          wins INTEGER NOT NULL,
          losses INTEGER NOT NULL,
          win_rate REAL NOT NULL,
          bayes_win_rate REAL NOT NULL,
          avg_pips REAL NOT NULL,
          shrink_avg_pips REAL NOT NULL,
          total_pips REAL NOT NULL,
          profit_factor REAL NOT NULL,
          avg_hold_sec REAL NOT NULL,
          spread_pips_mean REAL NOT NULL,
          tp_pips_mean REAL NOT NULL,
          sl_pips_mean REAL NOT NULL,
          confidence_mean REAL NOT NULL,
          chase_risk_rate REAL NOT NULL,
          p_value REAL NOT NULL,
          z_edge REAL NOT NULL,
          robust_score REAL NOT NULL,
          suggested_multiplier REAL NOT NULL,
          quality TEXT NOT NULL,
          is_significant INTEGER NOT NULL,
          score_rank INTEGER NOT NULL,
          boot_ci_low REAL,
          boot_ci_high REAL,
          last_close_time TEXT,
          updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_pattern_scores_quality
          ON pattern_scores(quality);

        CREATE TABLE IF NOT EXISTS pattern_drift (
          pattern_id TEXT PRIMARY KEY,
          recent_trades INTEGER NOT NULL,
          prev_trades INTEGER NOT NULL,
          recent_avg_pips REAL NOT NULL,
          prev_avg_pips REAL NOT NULL,
          delta_avg_pips REAL NOT NULL,
          recent_win_rate REAL NOT NULL,
          prev_win_rate REAL NOT NULL,
          delta_win_rate REAL NOT NULL,
          p_value REAL NOT NULL,
          drift_state TEXT NOT NULL,
          updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS pattern_clusters (
          pattern_id TEXT PRIMARY KEY,
          cluster_id INTEGER NOT NULL,
          cluster_score REAL NOT NULL,
          updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS pattern_cluster_summary (
          cluster_id INTEGER PRIMARY KEY,
          patterns INTEGER NOT NULL,
          mean_trades REAL NOT NULL,
          mean_win_rate REAL NOT NULL,
          mean_avg_pips REAL NOT NULL,
          mean_pf REAL NOT NULL,
          mean_score REAL NOT NULL,
          quality_mix TEXT NOT NULL,
          updated_at TEXT NOT NULL
        );
        """
    )


def _persist_deep_analysis(
    con: Any,
    *,
    score_df: pd.DataFrame,
    drift_df: pd.DataFrame,
    cluster_map_df: pd.DataFrame,
    cluster_summary_df: pd.DataFrame,
    as_of: str,
) -> None:
    _ensure_deep_schema(con)
    con.execute("DELETE FROM pattern_scores")
    con.execute("DELETE FROM pattern_drift")
    con.execute("DELETE FROM pattern_clusters")
    con.execute("DELETE FROM pattern_cluster_summary")

    if not score_df.empty:
        con.executemany(
            """
            INSERT INTO pattern_scores (
              pattern_id, strategy_tag, pocket, direction, trades, wins, losses,
              win_rate, bayes_win_rate, avg_pips, shrink_avg_pips, total_pips,
              profit_factor, avg_hold_sec, spread_pips_mean, tp_pips_mean,
              sl_pips_mean, confidence_mean, chase_risk_rate, p_value, z_edge,
              robust_score, suggested_multiplier, quality, is_significant,
              score_rank, boot_ci_low, boot_ci_high, last_close_time, updated_at
            ) VALUES (
              ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            [
                (
                    str(row.pattern_id),
                    str(row.strategy_tag or "unknown"),
                    str(row.pocket or "unknown"),
                    str(row.direction or "unknown"),
                    int(row.trades),
                    int(row.wins),
                    int(row.losses),
                    float(row.win_rate),
                    float(row.bayes_win_rate),
                    float(row.avg_pips),
                    float(row.shrink_avg_pips),
                    float(row.total_pips),
                    float(row.profit_factor),
                    float(row.avg_hold_sec or 0.0),
                    float(row.spread_pips_mean or 0.0),
                    float(row.tp_pips_mean or 0.0),
                    float(row.sl_pips_mean or 0.0),
                    float(row.confidence_mean or 0.0),
                    float(row.chase_risk_rate or 0.0),
                    float(row.p_value),
                    float(row.z_edge),
                    float(row.robust_score),
                    float(row.suggested_multiplier),
                    str(row.quality),
                    int(row.is_significant),
                    int(row.score_rank),
                    None if pd.isna(row.boot_ci_low) else float(row.boot_ci_low),
                    None if pd.isna(row.boot_ci_high) else float(row.boot_ci_high),
                    str(row.last_close_time or ""),
                    as_of,
                )
                for row in score_df.itertuples(index=False)
            ],
        )

    if not drift_df.empty:
        con.executemany(
            """
            INSERT INTO pattern_drift (
              pattern_id, recent_trades, prev_trades, recent_avg_pips, prev_avg_pips,
              delta_avg_pips, recent_win_rate, prev_win_rate, delta_win_rate,
              p_value, drift_state, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    str(row.pattern_id),
                    int(row.recent_trades),
                    int(row.prev_trades),
                    float(row.recent_avg_pips),
                    float(row.prev_avg_pips),
                    float(row.delta_avg_pips),
                    float(row.recent_win_rate),
                    float(row.prev_win_rate),
                    float(row.delta_win_rate),
                    float(row.p_value),
                    str(row.drift_state),
                    as_of,
                )
                for row in drift_df.itertuples(index=False)
            ],
        )

    if not cluster_map_df.empty:
        con.executemany(
            """
            INSERT INTO pattern_clusters (
              pattern_id, cluster_id, cluster_score, updated_at
            ) VALUES (?, ?, ?, ?)
            """,
            [
                (
                    str(row.pattern_id),
                    int(row.cluster_id),
                    float(row.cluster_score),
                    as_of,
                )
                for row in cluster_map_df.itertuples(index=False)
            ],
        )

    if not cluster_summary_df.empty:
        con.executemany(
            """
            INSERT INTO pattern_cluster_summary (
              cluster_id, patterns, mean_trades, mean_win_rate, mean_avg_pips,
              mean_pf, mean_score, quality_mix, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    int(row.cluster_id),
                    int(row.patterns),
                    float(row.mean_trades),
                    float(row.mean_win_rate),
                    float(row.mean_avg_pips),
                    float(row.mean_pf),
                    float(row.mean_score),
                    str(row.quality_mix),
                    as_of,
                )
                for row in cluster_summary_df.itertuples(index=False)
            ],
        )


def run_pattern_deep_analysis(
    con: Any,
    *,
    cutoff_iso: str,
    as_of: str,
    output_path: Path,
    config: DeepPatternConfig | None = None,
) -> dict[str, Any]:
    conf = config or DeepPatternConfig()
    frame = pd.read_sql_query(
        """
        SELECT
          pattern_id,
          pocket,
          strategy_tag,
          direction,
          close_time,
          hold_sec,
          pl_pips,
          signal_mode,
          mtf_gate,
          horizon_gate,
          extrema_reason,
          confidence,
          spread_pips,
          tp_pips,
          sl_pips
        FROM pattern_trade_features
        WHERE close_time >= ?
        ORDER BY close_time ASC
        """,
        con,
        params=(cutoff_iso,),
    )
    if frame.empty:
        _ensure_deep_schema(con)
        _persist_deep_analysis(
            con,
            score_df=pd.DataFrame(),
            drift_df=pd.DataFrame(),
            cluster_map_df=pd.DataFrame(),
            cluster_summary_df=pd.DataFrame(),
            as_of=as_of,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "as_of": as_of,
                    "rows_total": 0,
                    "patterns_scored": 0,
                    "drift_rows": 0,
                    "cluster_count": 0,
                    "quality_counts": {},
                    "top_robust": [],
                    "top_weak": [],
                    "drift_alerts": [],
                    "cluster_summary": [],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return {
            "rows_total": 0,
            "patterns_scored": 0,
            "drift_rows": 0,
            "cluster_count": 0,
            "quality_counts": {},
        }

    frame["close_dt"] = pd.to_datetime(frame["close_time"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["close_dt"]).copy().reset_index(drop=True)
    for col in ["pl_pips", "hold_sec", "confidence", "spread_pips", "tp_pips", "sl_pips"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["pl_pips"] = frame["pl_pips"].fillna(0.0)
    frame["hold_sec"] = frame["hold_sec"].fillna(0.0)
    frame["confidence"] = frame["confidence"].fillna(0.0)
    frame["spread_pips"] = frame["spread_pips"].fillna(0.0)
    frame["tp_pips"] = frame["tp_pips"].fillna(0.0)
    frame["sl_pips"] = frame["sl_pips"].fillna(0.0)

    token_rows = frame["pattern_id"].map(_parse_pattern_tokens)
    frame["range_bucket"] = token_rows.map(lambda t: t.get("rg", "na"))
    frame["pattern_side"] = token_rows.map(lambda t: t.get("sd", "unknown"))
    direction_norm = frame["direction"].fillna("").astype(str).str.lower()
    direction_norm = direction_norm.where(direction_norm.isin(["long", "short"]), "unknown")
    frame["direction"] = np.where(direction_norm == "unknown", frame["pattern_side"], direction_norm)
    frame["direction"] = pd.Series(frame["direction"]).astype(str).str.lower()
    frame.loc[~frame["direction"].isin(["long", "short"]), "direction"] = "unknown"
    frame["win"] = (frame["pl_pips"] > 0.0).astype(int)
    frame["is_chase_risk"] = [
        _is_chase_risk(side=str(side), range_bucket=str(rg))
        for side, rg in zip(frame["direction"], frame["range_bucket"], strict=False)
    ]

    score_df = _compute_pattern_scores(frame, config=conf)
    as_of_ts = pd.to_datetime(as_of, utc=True, errors="coerce")
    if pd.isna(as_of_ts):
        as_of_ts = pd.Timestamp.now(tz="UTC")
    drift_df = _compute_drift(frame, as_of=as_of_ts, config=conf)
    cluster_map_df, cluster_summary_df, cluster_meta = _cluster_patterns(score_df, config=conf)

    _persist_deep_analysis(
        con,
        score_df=score_df,
        drift_df=drift_df,
        cluster_map_df=cluster_map_df,
        cluster_summary_df=cluster_summary_df,
        as_of=as_of,
    )

    quality_counts = (
        score_df["quality"].value_counts().sort_index().to_dict() if not score_df.empty else {}
    )
    if drift_df.empty or "drift_state" not in drift_df.columns:
        drift_alerts_df = pd.DataFrame()
    else:
        drift_alerts_df = drift_df.loc[
            drift_df["drift_state"].isin(["deterioration", "soft_deterioration"])
        ].sort_values(by=["delta_avg_pips", "p_value"], ascending=[True, True])

    payload = {
        "as_of": as_of,
        "rows_total": int(len(frame)),
        "patterns_scored": int(len(score_df)),
        "drift_rows": int(len(drift_df)),
        "cluster_count": int(cluster_summary_df["cluster_id"].nunique() if not cluster_summary_df.empty else 0),
        "cluster_meta": cluster_meta,
        "quality_counts": {str(k): int(v) for k, v in quality_counts.items()},
        "top_robust": _df_records(
            score_df.sort_values(by=["robust_score", "trades"], ascending=[False, False]).head(30),
            [
                "pattern_id",
                "strategy_tag",
                "pocket",
                "direction",
                "trades",
                "win_rate",
                "avg_pips",
                "shrink_avg_pips",
                "profit_factor",
                "p_value",
                "robust_score",
                "suggested_multiplier",
                "quality",
                "boot_ci_low",
                "boot_ci_high",
            ],
        ),
        "top_weak": _df_records(
            score_df.sort_values(by=["robust_score", "trades"], ascending=[True, False]).head(30),
            [
                "pattern_id",
                "strategy_tag",
                "pocket",
                "direction",
                "trades",
                "win_rate",
                "avg_pips",
                "shrink_avg_pips",
                "profit_factor",
                "p_value",
                "robust_score",
                "suggested_multiplier",
                "quality",
                "chase_risk_rate",
            ],
        ),
        "drift_alerts": _df_records(
            drift_alerts_df.head(40),
            [
                "pattern_id",
                "recent_trades",
                "prev_trades",
                "recent_avg_pips",
                "prev_avg_pips",
                "delta_avg_pips",
                "recent_win_rate",
                "prev_win_rate",
                "delta_win_rate",
                "p_value",
                "drift_state",
            ],
        ),
        "cluster_summary": _df_records(
            cluster_summary_df.head(20),
            [
                "cluster_id",
                "patterns",
                "mean_trades",
                "mean_win_rate",
                "mean_avg_pips",
                "mean_pf",
                "mean_score",
                "quality_mix",
            ],
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return {
        "rows_total": int(len(frame)),
        "patterns_scored": int(len(score_df)),
        "drift_rows": int(len(drift_df)),
        "cluster_count": int(payload["cluster_count"]),
        "quality_counts": payload["quality_counts"],
    }
