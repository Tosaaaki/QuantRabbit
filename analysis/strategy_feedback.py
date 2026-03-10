"""Dynamic per-strategy analysis feedback bus.

Analysis services can emit JSON updates and this module exposes them as lightweight
runtime hints for strategy entries.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Optional

from utils.strategy_tags import normalize_strategy_lookup_key
from utils.strategy_tags import resolve_strategy_tag
from utils.strategy_tags import strategy_like_matches
from workers.common.setup_context import extract_setup_identity


_FEEDBACK_ENABLED = os.getenv("STRATEGY_FEEDBACK_ENABLED", "1").strip().lower() not in {
    "0",
    "false",
    "off",
    "no",
}
_FEEDBACK_PATH = os.getenv("STRATEGY_FEEDBACK_PATH") or os.getenv(
    "STRATEGY_FEEDBACK_JSON", "logs/strategy_feedback.json"
)
if _FEEDBACK_PATH is not None and _FEEDBACK_PATH.strip().lower() in {"", "none", "off"}:
    _FEEDBACK_PATH = None

_FEEDBACK_REFRESH_SEC = float(os.getenv("STRATEGY_FEEDBACK_REFRESH_SEC", "30") or 30)
_FEEDBACK_MAX_AGE_SEC = max(
    0.0,
    float(os.getenv("STRATEGY_FEEDBACK_MAX_AGE_SEC", "1800") or 1800.0),
)

_PATH: Optional[Path] = Path(_FEEDBACK_PATH) if _FEEDBACK_PATH else None
_CACHE: dict[str, Any] = {
    "loaded": 0.0,
    "mtime": None,
    "payload": None,
}
_PARAM_CACHE: dict[str, Any] = {"loaded": 0.0, "paths": None, "payloads": []}
_COUNTERFACTUAL_ENABLED = os.getenv("STRATEGY_FEEDBACK_COUNTERFACTUAL_ENABLED", "1").strip().lower() not in {
    "0",
    "false",
    "off",
    "no",
}
_COUNTERFACTUAL_PATH_RAW = os.getenv(
    "STRATEGY_FEEDBACK_COUNTERFACTUAL_PATH",
    os.getenv("COUNTERFACTUAL_OUT_PATH", "logs/trade_counterfactual_latest.json"),
)
if _COUNTERFACTUAL_PATH_RAW and _COUNTERFACTUAL_PATH_RAW.strip().lower() in {"", "none", "off"}:
    _COUNTERFACTUAL_PATH_RAW = None
_COUNTERFACTUAL_PATH: Optional[Path] = (
    Path(_COUNTERFACTUAL_PATH_RAW) if _COUNTERFACTUAL_PATH_RAW else None
)
_COUNTERFACTUAL_REFRESH_SEC = float(
    os.getenv("STRATEGY_FEEDBACK_COUNTERFACTUAL_REFRESH_SEC", str(_FEEDBACK_REFRESH_SEC)) or _FEEDBACK_REFRESH_SEC
)
_COUNTERFACTUAL_MAX_AGE_SEC = max(
    0.0,
    float(
        os.getenv(
            "STRATEGY_FEEDBACK_COUNTERFACTUAL_MAX_AGE_SEC",
            str(_FEEDBACK_MAX_AGE_SEC),
        )
        or _FEEDBACK_MAX_AGE_SEC
    ),
)
_COUNTERFACTUAL_CACHE: dict[str, Any] = {"loaded": 0.0, "mtime": None, "payload": None}


def _parse_csv_paths(raw: Optional[str]) -> list[Path]:
    if not raw:
        return []
    paths: list[Path] = []
    for token in re.split(r"[,;\n]+", str(raw)):
        token = token.strip()
        if token:
            paths.append(Path(token))
    return paths


def _normalize_strategy_key(value: Optional[str]) -> str:
    return normalize_strategy_lookup_key(value)


def _base_strategy_tag(value: Optional[str]) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    lowered = text.lower()
    for suffix in ("-long", "-short", "_long", "_short", "/long", "/short"):
        if not lowered.endswith(suffix):
            continue
        base = text[: -len(suffix)].rstrip("-_/ ")
        if base:
            return base
    return ""


def _setup_context(entry_thesis: Optional[dict[str, Any]]) -> dict[str, str]:
    return extract_setup_identity(entry_thesis)


def _setup_match_specificity(
    override: dict[str, Any],
    *,
    setup_context: dict[str, str],
) -> int:
    if not setup_context:
        return 0
    setup_fingerprint = str(override.get("setup_fingerprint") or "").strip()
    flow_regime = str(override.get("flow_regime") or "").strip()
    microstructure_bucket = str(override.get("microstructure_bucket") or "").strip()
    if setup_fingerprint:
        return 4 if setup_context.get("setup_fingerprint") == setup_fingerprint else 0
    if flow_regime and microstructure_bucket:
        return (
            3
            if setup_context.get("flow_regime") == flow_regime
            and setup_context.get("microstructure_bucket") == microstructure_bucket
            else 0
        )
    if flow_regime:
        return 2 if setup_context.get("flow_regime") == flow_regime else 0
    if microstructure_bucket:
        return 1 if setup_context.get("microstructure_bucket") == microstructure_bucket else 0
    return 0


def _select_setup_override(
    setup_overrides: object,
    *,
    entry_thesis: Optional[dict[str, Any]],
) -> tuple[Optional[dict[str, Any]], Optional[dict[str, Any]]]:
    if not isinstance(setup_overrides, list):
        return None, None
    setup_context = _setup_context(entry_thesis)
    if not setup_context:
        return None, None

    best_override: Optional[dict[str, Any]] = None
    best_meta: Optional[dict[str, Any]] = None
    best_rank = (0, 0)
    allowed_fields = {
        "entry_probability_multiplier",
        "entry_probability_delta",
        "entry_units_multiplier",
        "entry_units_min",
        "entry_units_max",
        "sl_distance_multiplier",
        "tp_distance_multiplier",
        "notes",
    }
    for item in setup_overrides:
        if not isinstance(item, dict):
            continue
        specificity = _setup_match_specificity(item, setup_context=setup_context)
        if specificity <= 0:
            continue
        trades = _to_float(item.get("trades"), 0.0) or 0.0
        rank = (specificity, int(trades))
        if rank < best_rank:
            continue
        override_payload = {key: item.get(key) for key in allowed_fields if key in item}
        if not override_payload:
            continue
        best_rank = rank
        best_override = override_payload
        best_meta = {
            "match_dimension": str(item.get("match_dimension") or ""),
            "setup_fingerprint": str(item.get("setup_fingerprint") or "") or None,
            "flow_regime": str(item.get("flow_regime") or "") or None,
            "microstructure_bucket": str(item.get("microstructure_bucket") or "") or None,
            "trades": int(trades),
            "win_rate": _to_float(item.get("win_rate"), None),
            "profit_factor": _to_float(item.get("profit_factor"), None),
        }
    return best_override, best_meta


def _strategy_lookup_candidates(
    strategy_tag: Optional[str],
    *,
    known_keys: Optional[list[str]] = None,
) -> list[str]:
    candidates: list[str] = []

    def _append(raw: Optional[str]) -> None:
        key = _normalize_strategy_key(raw)
        if key and key not in candidates:
            candidates.append(key)

    base_tag = _base_strategy_tag(strategy_tag)
    _append(strategy_tag)
    _append(base_tag)
    for raw in (strategy_tag, base_tag):
        if not raw:
            continue
        try:
            _append(resolve_strategy_tag(raw, known_keys=known_keys))
        except Exception:
            continue
    return candidates


def _read_file_payload(path: Path) -> Optional[dict[str, Any]]:
    try:
        raw = path.read_text(encoding="utf-8")
    except Exception:
        return None
    parsed: object
    try:
        parsed = json.loads(raw)
    except Exception:
        try:
            import yaml  # type: ignore
        except Exception:
            return None
        try:
            parsed = yaml.safe_load(raw)  # type: ignore
        except Exception:
            return None
    if not isinstance(parsed, dict):
        return None
    return parsed


_PARAM_SOURCE_PATHS = _parse_csv_paths(
    os.getenv("STRATEGY_PARAMETER_PATHS")
    or os.getenv("STRATEGY_PARAMS_PATHS", "config/tuning_overlay.yaml,config/tuning_overrides.yaml")
)


def _lookup_strategy_section(
    payload: Optional[dict[str, Any]],
    strategy_key: str,
) -> Optional[dict[str, Any]]:
    if not payload:
        return None
    if strategy_key in payload:
        strategy_cfg = payload.get(strategy_key)
        if isinstance(strategy_cfg, dict):
            return strategy_cfg
    if strategy_key.upper() in payload:
        strategy_cfg = payload.get(strategy_key.upper())
        if isinstance(strategy_cfg, dict):
            return strategy_cfg
    for key, strategy_cfg in payload.items():
        if not isinstance(key, str) or not isinstance(strategy_cfg, dict):
            continue
        if _normalize_strategy_key(key) == strategy_key:
            return strategy_cfg
    return None


def _resolve_strategy_metadata(
    payload: dict[str, Any],
    strategy_key: str,
) -> dict[str, Any]:
    resolved: dict[str, Any] = {}
    for section_key in (
        "strategies",
        "strategy_profiles",
        "strategy_params",
        "strategy_parameters",
        "strategy_configs",
        "strategy_config",
    ):
        section = payload.get(section_key)
        if not isinstance(section, dict):
            continue
        matched = _lookup_strategy_section(section, strategy_key)
        if not matched:
            continue
        for mkey, mvalue in matched.items():
            if mkey == "enabled":
                resolved["enabled"] = mvalue
            elif mkey in {"status", "state"}:
                resolved[mkey] = mvalue
            else:
                resolved.setdefault("strategy_params", {})[mkey] = mvalue
    return resolved


def _merge_strategy_metadata(base: dict[str, Any], extra: Optional[dict[str, Any]]) -> dict[str, Any]:
    if not extra:
        return base
    merged = dict(base)
    if "enabled" not in merged and extra.get("enabled") is not None:
        merged["enabled"] = extra["enabled"]
    if not merged.get("status") and extra.get("status"):
        merged["status"] = extra["status"]
    strategy_params = merged.get("strategy_params")
    if not isinstance(strategy_params, dict):
        strategy_params = {}
    extra_params = extra.get("strategy_params")
    if isinstance(extra_params, dict):
        strategy_params.update(extra_params)
    if strategy_params:
        merged["strategy_params"] = strategy_params
    return merged


def _load_param_payloads() -> list[dict[str, Any]]:
    now = time.time()
    if (
        _PARAM_CACHE["payloads"]
        and _PARAM_CACHE.get("loaded", 0.0)
        and (now - float(_PARAM_CACHE["loaded"])) < _FEEDBACK_REFRESH_SEC
        and _PARAM_CACHE.get("paths") == [_p.as_posix() for _p in _PARAM_SOURCE_PATHS]
    ):
        return _PARAM_CACHE["payloads"]  # type: ignore[return-value]

    payloads: list[dict[str, Any]] = []
    for path in _PARAM_SOURCE_PATHS:
        payload = _read_file_payload(path)
        if payload:
            payloads.append(payload)
    _PARAM_CACHE.update(
        {
            "loaded": now,
            "paths": [_p.as_posix() for _p in _PARAM_SOURCE_PATHS],
            "payloads": payloads,
        }
    )
    return payloads


def _to_float(value: object, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if parsed != parsed or parsed == float("inf") or parsed == float("-inf"):
        return default
    return parsed


def _parse_iso8601_epoch(raw: object) -> Optional[float]:
    text = str(raw or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        import datetime as dt

        parsed = dt.datetime.fromisoformat(text)
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc).timestamp()


def _payload_freshness(
    payload: Optional[dict[str, Any]],
    *,
    path: Optional[Path],
    max_age_sec: float,
) -> dict[str, object]:
    age_sec: Optional[float] = None
    timestamp_source = ""
    if isinstance(payload, dict):
        for key in ("updated_at", "generated_at", "as_of", "timestamp"):
            epoch = _parse_iso8601_epoch(payload.get(key))
            if epoch is None:
                continue
            age_sec = max(0.0, time.time() - epoch)
            timestamp_source = str(key)
            break
    if age_sec is None and path is not None:
        try:
            age_sec = max(0.0, time.time() - float(path.stat().st_mtime))
            timestamp_source = "mtime"
        except OSError:
            age_sec = None
    stale = bool(max_age_sec > 0.0 and age_sec is not None and age_sec > max_age_sec)
    return {
        "age_sec": age_sec,
        "timestamp_source": timestamp_source or None,
        "stale": stale,
    }


def _to_bool(value: object, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _coerce_multiplier(value: object, default: float, min_value: float = 0.0, max_value: float = 10.0) -> float:
    parsed = _to_float(value, default=default)
    if parsed is None:
        return default
    if parsed < min_value:
        return min_value
    if parsed > max_value:
        return max_value
    return parsed


def _coerce_abs_int(value: object) -> Optional[float]:
    parsed = _to_float(value, default=None)
    if parsed is None:
        return None
    if parsed < 0:
        parsed = 0.0
    return float(abs(parsed))


def _read_payload() -> dict[str, Any]:
    if _PATH is None:
        return {}
    try:
        stat = _PATH.stat()
    except FileNotFoundError:
        return {}
    except OSError:
        return {}
    if (
        _CACHE["mtime"] is not None
        and _CACHE["payload"] is not None
        and float(_CACHE["mtime"]) == float(stat.st_mtime)
    ):
        return _CACHE["payload"]
    payload = _read_file_payload(_PATH)
    if payload is None:
        return {}
    _CACHE["mtime"] = float(stat.st_mtime)
    _CACHE["loaded"] = time.time()
    _CACHE["payload"] = payload
    return payload


def _read_counterfactual_payload() -> dict[str, Any]:
    if _COUNTERFACTUAL_PATH is None:
        return {}
    try:
        stat = _COUNTERFACTUAL_PATH.stat()
    except FileNotFoundError:
        return {}
    except OSError:
        return {}
    if (
        _COUNTERFACTUAL_CACHE["mtime"] is not None
        and _COUNTERFACTUAL_CACHE["payload"] is not None
        and float(_COUNTERFACTUAL_CACHE["mtime"]) == float(stat.st_mtime)
    ):
        return _COUNTERFACTUAL_CACHE["payload"]
    payload = _read_file_payload(_COUNTERFACTUAL_PATH)
    if payload is None:
        return {}
    _COUNTERFACTUAL_CACHE["mtime"] = float(stat.st_mtime)
    _COUNTERFACTUAL_CACHE["loaded"] = time.time()
    _COUNTERFACTUAL_CACHE["payload"] = payload
    return payload


def _refresh_if_needed(force: bool = False) -> None:
    if not _FEEDBACK_ENABLED:
        return
    now = time.time()
    if _PATH is not None:
        if force or not (_CACHE["loaded"] or 0.0) or (
            now - float(_CACHE["loaded"]) >= _FEEDBACK_REFRESH_SEC
        ):
            _read_payload()
    if _COUNTERFACTUAL_ENABLED and _COUNTERFACTUAL_PATH is not None:
        if force or not (_COUNTERFACTUAL_CACHE["loaded"] or 0.0) or (
            now - float(_COUNTERFACTUAL_CACHE["loaded"]) >= _COUNTERFACTUAL_REFRESH_SEC
        ):
            _read_counterfactual_payload()


def _apply_pocket_override(
    base: dict[str, Any],
    strategy_cfg: dict[str, Any],
    pocket: Optional[str],
) -> None:
    if not strategy_cfg:
        return
    for key in ("pockets", "pocket_profiles"):
        bucket = strategy_cfg.get(key)
        if not isinstance(bucket, dict):
            continue
        if not pocket:
            break
        pkey = _normalize_strategy_key(pocket)
        override = bucket.get(pkey)
        if isinstance(override, dict):
            base.update(override)
            return
        override = bucket.get(str(pocket).strip().lower())
        if isinstance(override, dict):
            base.update(override)
            return


def _normalize_side(value: Optional[str]) -> str:
    side = str(value or "").strip().lower()
    if side in {"buy", "long", "open_long"}:
        return "long"
    if side in {"sell", "short", "open_short"}:
        return "short"
    return ""


def _clamp(value: float, low: float, high: float) -> float:
    if value < low:
        return low
    if value > high:
        return high
    return value


def _counterfactual_overlay(strategy_tag: Optional[str], side: Optional[str]) -> dict[str, Any]:
    if not _COUNTERFACTUAL_ENABLED or _COUNTERFACTUAL_PATH is None or not strategy_tag:
        return {}
    payload = _read_counterfactual_payload()
    if not isinstance(payload, dict):
        return {}
    freshness = _payload_freshness(
        payload,
        path=_COUNTERFACTUAL_PATH,
        max_age_sec=_COUNTERFACTUAL_MAX_AGE_SEC,
    )
    if freshness.get("stale"):
        return {}
    strategy_like = str(payload.get("strategy_like") or "").strip()
    if strategy_like and not strategy_like_matches(strategy_tag, strategy_like):
        return {}
    hints = payload.get("policy_hints")
    if not isinstance(hints, dict):
        return {}
    reentry = hints.get("reentry_overrides")
    if not isinstance(reentry, dict):
        return {}

    mode = str(reentry.get("mode") or "").strip().lower()
    if mode not in {"tighten", "loosen"}:
        return {}
    confidence = _clamp(float(_to_float(reentry.get("confidence"), 0.0) or 0.0), 0.0, 1.0)
    if confidence <= 0.0:
        return {}

    cooldown_loss_mult = _clamp(float(_to_float(reentry.get("cooldown_loss_mult"), 1.0) or 1.0), 0.60, 1.80)
    cooldown_win_mult = _clamp(float(_to_float(reentry.get("cooldown_win_mult"), 1.0) or 1.0), 0.70, 1.50)
    reentry_pips_mult = _clamp(
        float(_to_float(reentry.get("same_dir_reentry_pips_mult"), 1.0) or 1.0),
        0.70,
        1.60,
    )
    lcb_uplift_pips = max(0.0, float(_to_float(reentry.get("lcb_uplift_pips"), 0.0) or 0.0))
    return_wait_bias = str(reentry.get("return_wait_bias") or "").strip().lower()
    if return_wait_bias not in {"avoid", "favor"}:
        return_wait_bias = "avoid" if mode == "tighten" else "favor"
    sl_multiplier = 1.0
    tp_multiplier = 1.0
    if mode == "tighten":
        severity = max(
            0.0,
            (cooldown_loss_mult - 1.0) / 0.80,
            (cooldown_win_mult - 1.0) / 0.50,
            (reentry_pips_mult - 1.0) / 0.60,
        )
        severity = _clamp(severity, 0.0, 1.0)
        units_multiplier = 1.0 - 0.35 * confidence * severity
        probability_multiplier = 1.0 - 0.18 * confidence * severity
        probability_delta = -0.03 * confidence * severity
        sl_multiplier = 1.0 - 0.11 * confidence * severity
        tp_multiplier = 1.0 - 0.16 * confidence * max(severity, _clamp(lcb_uplift_pips / 2.5, 0.0, 1.0))
    else:
        severity = max(
            0.0,
            (1.0 - cooldown_loss_mult) / 0.40,
            (1.0 - cooldown_win_mult) / 0.30,
            (1.0 - reentry_pips_mult) / 0.30,
        )
        severity = _clamp(severity, 0.0, 1.0)
        units_multiplier = 1.0 + 0.20 * confidence * severity
        probability_multiplier = 1.0 + 0.10 * confidence * severity
        probability_delta = 0.02 * confidence * severity
        sl_multiplier = 1.0 + 0.05 * confidence * severity
        tp_multiplier = 1.0 + 0.14 * confidence * severity + 0.08 * confidence * _clamp(lcb_uplift_pips / 2.5, 0.0, 1.0)

    if return_wait_bias == "avoid":
        sl_multiplier *= 0.98
        tp_multiplier *= 0.97
    elif return_wait_bias == "favor":
        tp_multiplier *= 1.04

    side_key = _normalize_side(side)
    side_action = ""
    side_actions = hints.get("side_actions")
    if isinstance(side_actions, dict) and side_key:
        action_raw = side_actions.get(side_key)
        if action_raw is not None:
            side_action = str(action_raw).strip().lower()
    if side_action == "block":
        units_multiplier *= 0.55
        probability_multiplier *= 0.88
        probability_delta -= 0.04 * confidence
        sl_multiplier *= 0.94
        tp_multiplier *= 0.90
    elif side_action == "reduce":
        units_multiplier *= 0.78
        probability_multiplier *= 0.92
        sl_multiplier *= 0.97
        tp_multiplier *= 0.95
    elif side_action == "boost":
        units_multiplier *= 1.08
        probability_multiplier *= 1.04
        probability_delta += 0.02 * confidence
        sl_multiplier *= 1.02
        tp_multiplier *= 1.06

    units_multiplier = _clamp(units_multiplier, 0.35, 1.35)
    probability_multiplier = _clamp(probability_multiplier, 0.55, 1.20)
    probability_delta = _clamp(probability_delta, -0.20, 0.12)
    sl_multiplier = _clamp(sl_multiplier, 0.78, 1.18)
    tp_multiplier = _clamp(tp_multiplier, 0.72, 1.28)
    return {
        "entry_units_multiplier": round(units_multiplier, 4),
        "entry_probability_multiplier": round(probability_multiplier, 4),
        "entry_probability_delta": round(probability_delta, 4),
        "sl_distance_multiplier": round(sl_multiplier, 4),
        "tp_distance_multiplier": round(tp_multiplier, 4),
        "meta": {
            "source": str(_COUNTERFACTUAL_PATH),
            "strategy_like": strategy_like,
            "mode": mode,
            "confidence": round(confidence, 4),
            "side_action": side_action,
            "lcb_uplift_pips": round(lcb_uplift_pips, 4),
            "return_wait_bias": return_wait_bias,
        },
    }


def current_advice(
    strategy_tag: Optional[str],
    *,
    pocket: Optional[str] = None,
    side: Optional[str] = None,
    entry_thesis: Optional[dict[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    if not _FEEDBACK_ENABLED or not strategy_tag:
        return None
    _refresh_if_needed()
    payload = _read_payload()
    if not isinstance(payload, dict):
        return None
    freshness = _payload_freshness(
        payload,
        path=_PATH,
        max_age_sec=_FEEDBACK_MAX_AGE_SEC,
    )
    if freshness.get("stale"):
        payload = {}
    strategies = payload.get("strategies")
    known_keys = list(strategies.keys()) if isinstance(strategies, dict) else None
    strategy_keys = _strategy_lookup_candidates(strategy_tag, known_keys=known_keys)
    if not strategy_keys:
        strategy_keys = [_normalize_strategy_key(strategy_tag)]

    # Resolve optional strategy metadata from external sources (tuning/overlay/reentry
    # profiles). Strategy-specific parameters are not directly used as hard gates;
    # they are carried as analysis metadata and used by external analysis services.
    strategy_cfg = None
    if isinstance(strategies, dict):
        for strategy_key in strategy_keys:
            strategy_cfg = _lookup_strategy_section(strategies, strategy_key)
            if strategy_cfg:
                break

    metadata: dict[str, Any] = {}
    for strategy_key in strategy_keys:
        metadata = _merge_strategy_metadata(metadata, _resolve_strategy_metadata(payload, strategy_key))
    for path_payload in _load_param_payloads():
        for strategy_key in strategy_keys:
            metadata = _merge_strategy_metadata(
                metadata,
                _resolve_strategy_metadata(path_payload, strategy_key),
            )

    counterfactual = _counterfactual_overlay(strategy_tag, side)
    if strategy_cfg is None and not metadata and not counterfactual:
        return None
    merged: dict[str, Any] = {}
    defaults = payload.get("strategy_defaults")
    if isinstance(defaults, dict):
        merged.update(defaults)
    if isinstance(strategy_cfg, dict):
        merged.update(strategy_cfg)
    if isinstance(metadata.get("strategy_params"), dict):
        merged.update(metadata.get("strategy_params") or {})
    strategy_pocket_source = None
    if isinstance(strategy_cfg, dict):
        strategy_pocket_source = strategy_cfg
    elif isinstance(metadata.get("strategy_params"), dict):
        strategy_pocket_source = metadata.get("strategy_params")
    if isinstance(strategy_pocket_source, dict):
        _apply_pocket_override(merged, strategy_pocket_source, pocket)
    setup_override = None
    setup_override_meta = None
    if isinstance(strategy_cfg, dict):
        setup_override, setup_override_meta = _select_setup_override(
            strategy_cfg.get("setup_overrides"),
            entry_thesis=entry_thesis,
        )
    if isinstance(setup_override, dict):
        merged.update(setup_override)
    if not _to_bool(merged.get("enabled"), True):
        return None
    metadata_status = str(metadata.get("status") or "").strip().lower()
    if metadata_status in {"off", "inactive", "stopped", "pause", "paused", "disabled", "stop"}:
        return None
    advice: dict[str, Any] = {}
    advice["entry_probability_multiplier"] = _coerce_multiplier(
        merged.get("entry_probability_multiplier", merged.get("probability_multiplier", 1.0)),
        default=1.0,
    )
    advice["entry_probability_delta"] = max(
        -1.0, min(1.0, _to_float(merged.get("entry_probability_delta"), 0.0) or 0.0)
    )
    advice["entry_units_multiplier"] = _coerce_multiplier(
        merged.get("entry_units_multiplier", merged.get("units_multiplier", 1.0)),
        default=1.0,
    )
    entry_units_min = _coerce_abs_int(
        merged.get("entry_units_min")
        if merged.get("entry_units_min") is not None
        else merged.get("units_abs_min")
    )
    entry_units_max = _coerce_abs_int(
        merged.get("entry_units_max")
        if merged.get("entry_units_max") is not None
        else merged.get("units_abs_max")
    )
    if entry_units_min is not None:
        advice["entry_units_min"] = entry_units_min
    if entry_units_max is not None:
        advice["entry_units_max"] = entry_units_max
    advice["sl_distance_multiplier"] = _coerce_multiplier(
        merged.get("sl_distance_multiplier", merged.get("sl_multiplier", 1.0)),
        default=1.0,
        min_value=0.05,
        max_value=5.0,
    )
    advice["tp_distance_multiplier"] = _coerce_multiplier(
        merged.get("tp_distance_multiplier", merged.get("tp_multiplier", 1.0)),
        default=1.0,
        min_value=0.05,
        max_value=5.0,
    )
    notes = merged.get("notes")
    if notes is not None:
        advice["notes"] = str(notes)
    strategy_params = metadata.get("strategy_params")
    strategy_params_dict: dict[str, Any] = {}
    configured_params = {}
    if isinstance(strategy_params, dict):
        strategy_params_dict = dict(strategy_params)
        nested_configured = strategy_params_dict.get("configured_params")
        if isinstance(nested_configured, dict):
            configured_params = nested_configured
    if counterfactual:
        advice["entry_units_multiplier"] = _coerce_multiplier(
            advice["entry_units_multiplier"] * counterfactual.get("entry_units_multiplier", 1.0),
            default=advice["entry_units_multiplier"],
            min_value=0.0,
            max_value=5.0,
        )
        advice["entry_probability_multiplier"] = _coerce_multiplier(
            advice["entry_probability_multiplier"] * counterfactual.get("entry_probability_multiplier", 1.0),
            default=advice["entry_probability_multiplier"],
            min_value=0.0,
            max_value=5.0,
        )
        advice["entry_probability_delta"] = _clamp(
            advice["entry_probability_delta"] + float(counterfactual.get("entry_probability_delta", 0.0) or 0.0),
            -1.0,
            1.0,
        )
        advice["sl_distance_multiplier"] = _coerce_multiplier(
            advice["sl_distance_multiplier"] * counterfactual.get("sl_distance_multiplier", 1.0),
            default=advice["sl_distance_multiplier"],
            min_value=0.05,
            max_value=5.0,
        )
        advice["tp_distance_multiplier"] = _coerce_multiplier(
            advice["tp_distance_multiplier"] * counterfactual.get("tp_distance_multiplier", 1.0),
            default=advice["tp_distance_multiplier"],
            min_value=0.05,
            max_value=5.0,
        )
        strategy_params_dict["counterfactual_feedback"] = dict(counterfactual.get("meta") or {})
    if strategy_params_dict:
        advice["strategy_params"] = strategy_params_dict
    has_strategy_params = (
        isinstance(advice.get("strategy_params"), dict) and bool(advice["strategy_params"])
    ) or (isinstance(configured_params, dict) and bool(configured_params))
    if isinstance(configured_params, dict) and configured_params:
        advice["configured_params"] = dict(configured_params)
    advice["_meta"] = {
        "strategy_tag": strategy_tag,
        "pocket": pocket,
        "source": str(_PATH) if _PATH else "",
        "version": payload.get("version"),
        "updated_at": payload.get("updated_at"),
        "payload_age_sec": freshness.get("age_sec"),
        "payload_timestamp_source": freshness.get("timestamp_source"),
        "payload_stale": bool(freshness.get("stale")),
    }
    if counterfactual:
        advice["_meta"]["counterfactual"] = dict(counterfactual.get("meta") or {})
    if setup_override_meta:
        advice["_meta"]["setup_override"] = dict(setup_override_meta)

    # Return None when there are no actual tuning knobs (no-op).
    if (
        advice["entry_probability_multiplier"] == 1.0
        and advice["entry_probability_delta"] == 0.0
        and advice["entry_units_multiplier"] == 1.0
        and advice["sl_distance_multiplier"] == 1.0
        and advice["tp_distance_multiplier"] == 1.0
        and "entry_units_min" not in advice
        and "entry_units_max" not in advice
        and not has_strategy_params
    ):
        return None
    return advice
