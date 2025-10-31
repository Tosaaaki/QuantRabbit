from __future__ import annotations

import datetime as dt
from typing import Any, Dict, Iterable, Optional

_LAST_CACHE_KEY: Optional[tuple[str, int]] = None
_LAST_BASE_FEATURES: Dict[str, float] = {}
_LAST_LATEST_TS: Optional[dt.datetime] = None


def _parse_ts(value: Optional[str]) -> Optional[dt.datetime]:
    if not value:
        return None
    try:
        ts = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=dt.timezone.utc)
    return ts.astimezone(dt.timezone.utc)


def _collect_items(news_cache: Optional[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    if not news_cache:
        return []
    items: list[Dict[str, Any]] = []
    for horizon in ("short", "long"):
        horizon_items = news_cache.get(horizon, [])
        if isinstance(horizon_items, list):
            for item in horizon_items:
                if isinstance(item, dict):
                    items.append(item)
    return items


def build_news_features(
    news_cache: Optional[Dict[str, Any]],
    *,
    now: Optional[dt.datetime] = None,
) -> Dict[str, float]:
    """
    Convert cached news summaries into numeric features for GPT payloads.
    Results are cached until the latest news timestamp changes. The age
    metric is re-evaluated on every call so it reflects current time even
    when other statistics are cached.
    """

    global _LAST_BASE_FEATURES, _LAST_CACHE_KEY, _LAST_LATEST_TS

    if isinstance(now, dt.datetime):
        now_utc = now.astimezone(dt.timezone.utc)
    else:
        now_utc = dt.datetime.now(dt.timezone.utc)
    items = list(_collect_items(news_cache))
    latest_ts: Optional[dt.datetime] = None
    for item in items:
        ts = _parse_ts(item.get("ts") or item.get("published_at") or item.get("event_time"))
        if ts and (latest_ts is None or ts > latest_ts):
            latest_ts = ts

    cache_key = (latest_ts.isoformat() if latest_ts else "", len(items))
    if cache_key != _LAST_CACHE_KEY:
        sentiments = []
        impacts = []
        horizon_counts = {"short": 0, "long": 0}
        bias_score = []
        for item in items:
            horizon = item.get("horizon", "")
            if horizon in horizon_counts:
                horizon_counts[horizon] += 1
            raw_sent = item.get("sentiment")
            try:
                sentiments.append(float(raw_sent))
            except (TypeError, ValueError):
                pass
            raw_imp = item.get("impact")
            try:
                impacts.append(float(raw_imp))
            except (TypeError, ValueError):
                pass
            bias = item.get("pair_bias")
            if isinstance(bias, str):
                b = bias.lower()
                if "up" in b:
                    bias_score.append(1.0)
                elif "down" in b:
                    bias_score.append(-1.0)

        total = len(items)
        avg_sent = sum(sentiments) / total if sentiments else 0.0
        abs_sent = sum(abs(s) for s in sentiments) / total if sentiments else 0.0
        pos_ratio = (
            sum(1 for s in sentiments if s > 0.05) / total if total else 0.0
        )
        neg_ratio = (
            sum(1 for s in sentiments if s < -0.05) / total if total else 0.0
        )
        impact_max = max(impacts) if impacts else 0.0
        impact_mean = sum(impacts) / len(impacts) if impacts else 0.0
        bias_mean = sum(bias_score) / len(bias_score) if bias_score else 0.0

        base = {
            "news_count_total": float(total),
            "news_count_short": float(horizon_counts["short"]),
            "news_count_long": float(horizon_counts["long"]),
            "news_sentiment_mean": round(avg_sent, 4),
            "news_sentiment_abs_mean": round(abs_sent, 4),
            "news_positive_ratio": round(pos_ratio, 4),
            "news_negative_ratio": round(neg_ratio, 4),
            "news_impact_max": round(impact_max, 3),
            "news_impact_mean": round(impact_mean, 3),
            "news_bias_score": round(bias_mean, 4),
        }
        _LAST_BASE_FEATURES = base
        _LAST_CACHE_KEY = cache_key
        _LAST_LATEST_TS = latest_ts
    else:
        base = dict(_LAST_BASE_FEATURES)

    if latest_ts:
        age_minutes = max(0.0, (now_utc - latest_ts).total_seconds() / 60.0)
    elif _LAST_LATEST_TS:
        age_minutes = max(0.0, (now_utc - _LAST_LATEST_TS).total_seconds() / 60.0)
    else:
        age_minutes = 9999.0

    base["news_latest_age_minutes"] = round(age_minutes, 2)
    return dict(base)
