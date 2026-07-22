"""Direction-neutral anomaly admission and capital recycling for DOJO.

``room-meta-01`` does not generate alpha.  It consumes the ranking, pair, side,
and full-size units emitted by existing bots, then uses only a synchronized
exact-28 panel of completed H1 bid/ask candles plus pre-decision portfolio
exposure.  Its output alphabet is deliberately small: ``ENTER_OK``, ``HOLD``,
or ``REDUCE_SIZE`` with a multiplier in ``0, 0.25, 0.5, 1.0``.

The implementation is pure and research-only.  It never chooses or changes a
side, entry method, geometry, or candidate rank.  A held candidate consumes no
seat, so the next upstream-ranked, sufficiently uncorrelated candidate may use
the released capital.  All thresholds and lookbacks are sealed caller inputs;
there are no hidden market defaults.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any, Final

from quant_rabbit.instruments import DEFAULT_TRADER_PAIRS, G8_CURRENCIES


CONTRACT: Final = "QR_DOJO_ANOMALY_ADMISSION_ALLOCATION_V1"
POLICY_CONTRACT: Final = "QR_DOJO_ANOMALY_ADMISSION_POLICY_V1"
TRAIN_PLAN_CONTRACT: Final = "QR_DOJO_ANOMALY_ADMISSION_TRAIN_PLAN_V1"
SCHEMA_VERSION: Final = 1
ROOM_ID: Final = "room-meta-01"

FORMAL_G8_PAIRS: Final = tuple(sorted(DEFAULT_TRADER_PAIRS))
FORMAL_G8_CURRENCIES: Final = tuple(sorted(G8_CURRENCIES))

# The complete undirected G8 graph has C(8, 2)=28 pairs.  These are structural
# universe sizes, not tunable market thresholds; changing either is a new room.
FORMAL_G8_PAIR_COUNT: Final = 28
FORMAL_G8_CURRENCY_COUNT: Final = 8

# One hour is the market-data sampling contract, not a tunable strategy value.
H1_SECONDS: Final = 60 * 60

# This fixed alphabet is the operator-requested experiment surface.  Changing
# it changes the economic hypothesis and therefore requires a new contract.
SIZE_MULTIPLIERS: Final = (0.0, 0.25, 0.5, 1.0)
_POSITIVE_MULTIPLIERS_DESC: Final = (1.0, 0.5, 0.25)
ADMISSION_DECISIONS: Final = frozenset({"ENTER_OK", "HOLD", "REDUCE_SIZE"})
EXPERIMENT_ARMS: Final = (
    "BASE_BOT",
    "EXTREME_MOMENTUM_VETO",
    "REVERSAL_SHOCK_VETO",
    "VOLATILITY_CORRELATION_SIZING",
    "COMBINED_ANOMALY_ADMISSION",
    "AI_EXIT_CAPITAL_RELEASE",
)

_BAND_NAMES: Final = (
    "momentum_z",
    "reversal_shock_z",
    "volatility_ratio",
    "spread_atr_ratio",
    "correlation_concentration",
    "currency_gross_exposure_fraction",
)
_LOOKBACK_NAMES: Final = (
    "momentum_bars",
    "reversal_prior_bars",
    "volatility_short_bars",
    "volatility_long_bars",
    "atr_bars",
    "correlation_bars",
)

_AUTHORITY: Final = {
    "research_only": True,
    "historical_train_is_proof": False,
    "promotion_eligible": False,
    "live_permission": False,
    "order_authority": "NONE",
    "broker_mutation_allowed": False,
    "automatic_deployment_allowed": False,
    "direction_prediction_allowed": False,
    "direction_change_allowed": False,
}


class DojoAnomalyAdmissionError(ValueError):
    """Input, policy, or allocation evidence violates the room contract."""


def canonical_sha256(value: Any) -> str:
    """Return strict canonical-JSON SHA-256."""

    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DojoAnomalyAdmissionError("value is not strict canonical JSON") from exc
    return hashlib.sha256(encoded).hexdigest()


def _strict_clone(value: Any, *, field: str) -> Any:
    try:
        return json.loads(
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    except (TypeError, ValueError) as exc:
        raise DojoAnomalyAdmissionError(
            f"{field} must contain strict JSON values"
        ) from exc


def _exact_mapping(value: Any, expected: set[str], *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise DojoAnomalyAdmissionError(f"{field} schema mismatch")
    return value


def _sequence(value: Any, *, field: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise DojoAnomalyAdmissionError(f"{field} must be an array")
    return value


def _finite(
    value: Any,
    *,
    field: str,
    positive: bool = False,
    non_negative: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DojoAnomalyAdmissionError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise DojoAnomalyAdmissionError(f"{field} must be finite")
    if positive and result <= 0:
        raise DojoAnomalyAdmissionError(f"{field} must be positive")
    if non_negative and result < 0:
        raise DojoAnomalyAdmissionError(f"{field} must be non-negative")
    return result


def _integer(value: Any, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise DojoAnomalyAdmissionError(f"{field} must be an integer >= {minimum}")
    return value


def _identifier(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 240:
        raise DojoAnomalyAdmissionError(f"{field} must contain 1..240 characters")
    if any(ord(character) < 33 or ord(character) > 126 for character in value):
        raise DojoAnomalyAdmissionError(f"{field} must be visible ASCII")
    return value


def _sha256(value: Any, *, field: str) -> str:
    digest = _identifier(value, field=field)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise DojoAnomalyAdmissionError(f"{field} must be lowercase SHA-256")
    if digest == "0" * 64:
        raise DojoAnomalyAdmissionError(f"{field} must be non-zero")
    return digest


def _side(value: Any, *, field: str) -> str:
    if value not in {"LONG", "SHORT"}:
        raise DojoAnomalyAdmissionError(f"{field} must be LONG or SHORT")
    return str(value)


def _validate_formal_universe() -> None:
    if len(FORMAL_G8_CURRENCIES) != FORMAL_G8_CURRENCY_COUNT:
        raise DojoAnomalyAdmissionError(
            "formal G8 currency universe is no longer exact-8"
        )
    if (
        len(FORMAL_G8_PAIRS) != FORMAL_G8_PAIR_COUNT
        or len(set(FORMAL_G8_PAIRS)) != FORMAL_G8_PAIR_COUNT
    ):
        raise DojoAnomalyAdmissionError("formal G8 pair universe is no longer exact-28")
    expected = {
        f"{base}_{quote}"
        for index, base in enumerate(G8_CURRENCIES)
        for quote in G8_CURRENCIES[index + 1 :]
    }
    observed_unordered = {frozenset(pair.split("_")) for pair in FORMAL_G8_PAIRS}
    expected_unordered = {frozenset(pair.split("_")) for pair in expected}
    if observed_unordered != expected_unordered:
        raise DojoAnomalyAdmissionError("formal G8 pair graph is incomplete")


def _normalize_band(value: Any, *, field: str) -> dict[str, float]:
    row = _exact_mapping(
        value,
        {"reduce_to_half_at", "reduce_to_quarter_at", "hold_at"},
        field=field,
    )
    half = _finite(
        row["reduce_to_half_at"], field=f"{field}.reduce_to_half_at", positive=True
    )
    quarter = _finite(
        row["reduce_to_quarter_at"],
        field=f"{field}.reduce_to_quarter_at",
        positive=True,
    )
    hold = _finite(row["hold_at"], field=f"{field}.hold_at", positive=True)
    if not half < quarter < hold:
        raise DojoAnomalyAdmissionError(
            f"{field} thresholds must satisfy half < quarter < hold"
        )
    return {
        "reduce_to_half_at": half,
        "reduce_to_quarter_at": quarter,
        "hold_at": hold,
    }


def build_policy(
    *,
    policy_id: str,
    lookbacks: Mapping[str, Any],
    bands: Mapping[str, Any],
    selected_pair_abs_correlation_hold_at: float,
) -> dict[str, Any]:
    """Seal one fully explicit, trainer-proposed anomaly policy.

    Lookbacks and thresholds are required inputs because their useful values are
    empirical market hypotheses.  Supplying a silent default would hide a tuned
    choice and make replay comparisons irreproducible.
    """

    _validate_formal_universe()
    normalized_lookbacks = _exact_mapping(
        lookbacks, set(_LOOKBACK_NAMES), field="lookbacks"
    )
    parsed_lookbacks = {
        name: _integer(normalized_lookbacks[name], field=f"lookbacks.{name}", minimum=2)
        for name in _LOOKBACK_NAMES
    }
    if (
        parsed_lookbacks["volatility_short_bars"]
        >= parsed_lookbacks["volatility_long_bars"]
    ):
        raise DojoAnomalyAdmissionError(
            "volatility_short_bars must be smaller than volatility_long_bars"
        )
    normalized_bands = _exact_mapping(bands, set(_BAND_NAMES), field="bands")
    parsed_bands = {
        name: _normalize_band(normalized_bands[name], field=f"bands.{name}")
        for name in _BAND_NAMES
    }
    correlation_hold = _finite(
        selected_pair_abs_correlation_hold_at,
        field="selected_pair_abs_correlation_hold_at",
        positive=True,
    )
    if correlation_hold > 1:
        raise DojoAnomalyAdmissionError(
            "selected_pair_abs_correlation_hold_at must be <= 1"
        )
    body = {
        "contract": POLICY_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "room_id": ROOM_ID,
        "policy_id": _identifier(policy_id, field="policy_id"),
        "formal_pair_universe": list(FORMAL_G8_PAIRS),
        "lookbacks": parsed_lookbacks,
        "bands": parsed_bands,
        "selected_pair_abs_correlation_hold_at": correlation_hold,
        "size_multiplier_alphabet": list(SIZE_MULTIPLIERS),
        "direction_policy": {
            "predict_direction": False,
            "change_upstream_direction": False,
            "change_upstream_rank": False,
        },
        "authority": _strict_clone(_AUTHORITY, field="authority"),
    }
    return {**body, "policy_sha256": canonical_sha256(body)}


def validate_policy(value: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild a policy from declared inputs and reject any drift."""

    row = _strict_clone(value, field="policy")
    _exact_mapping(
        row,
        {
            "contract",
            "schema_version",
            "room_id",
            "policy_id",
            "formal_pair_universe",
            "lookbacks",
            "bands",
            "selected_pair_abs_correlation_hold_at",
            "size_multiplier_alphabet",
            "direction_policy",
            "authority",
            "policy_sha256",
        },
        field="policy",
    )
    expected = build_policy(
        policy_id=row["policy_id"],
        lookbacks=row["lookbacks"],
        bands=row["bands"],
        selected_pair_abs_correlation_hold_at=row[
            "selected_pair_abs_correlation_hold_at"
        ],
    )
    if row != expected:
        raise DojoAnomalyAdmissionError("policy differs from canonical V1")
    return row


def _normalize_panel(
    panel: Mapping[str, Any], *, decision_epoch: int, required_bars: int
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], tuple[int, ...]]:
    rows = _exact_mapping(panel, set(FORMAL_G8_PAIRS), field="completed_h1_panel")
    normalized: dict[str, list[dict[str, Any]]] = {}
    common_epochs: tuple[int, ...] | None = None
    for pair in FORMAL_G8_PAIRS:
        raw_candles = _sequence(rows[pair], field=f"completed_h1_panel.{pair}")
        if len(raw_candles) < required_bars:
            raise DojoAnomalyAdmissionError(
                f"completed_h1_panel.{pair} lacks required history"
            )
        candles: list[dict[str, Any]] = []
        epochs: list[int] = []
        for index, value in enumerate(raw_candles):
            field = f"completed_h1_panel.{pair}[{index}]"
            candle = _exact_mapping(
                value,
                {
                    "close_epoch",
                    "complete",
                    "bid_high",
                    "bid_low",
                    "bid_close",
                    "ask_high",
                    "ask_low",
                    "ask_close",
                },
                field=field,
            )
            close_epoch = _integer(
                candle["close_epoch"], field=f"{field}.close_epoch", minimum=1
            )
            if candle["complete"] is not True:
                raise DojoAnomalyAdmissionError(f"{field} is not complete")
            bid_high = _finite(
                candle["bid_high"], field=f"{field}.bid_high", positive=True
            )
            bid_low = _finite(
                candle["bid_low"], field=f"{field}.bid_low", positive=True
            )
            bid_close = _finite(
                candle["bid_close"], field=f"{field}.bid_close", positive=True
            )
            ask_high = _finite(
                candle["ask_high"], field=f"{field}.ask_high", positive=True
            )
            ask_low = _finite(
                candle["ask_low"], field=f"{field}.ask_low", positive=True
            )
            ask_close = _finite(
                candle["ask_close"], field=f"{field}.ask_close", positive=True
            )
            if not bid_low <= bid_close <= bid_high:
                raise DojoAnomalyAdmissionError(f"{field} bid OHLC is invalid")
            if not ask_low <= ask_close <= ask_high:
                raise DojoAnomalyAdmissionError(f"{field} ask OHLC is invalid")
            if ask_close < bid_close or ask_low < bid_low or ask_high < bid_high:
                raise DojoAnomalyAdmissionError(f"{field} ask must not cross below bid")
            epochs.append(close_epoch)
            candles.append(
                {
                    "close_epoch": close_epoch,
                    "complete": True,
                    "bid_high": bid_high,
                    "bid_low": bid_low,
                    "bid_close": bid_close,
                    "ask_high": ask_high,
                    "ask_low": ask_low,
                    "ask_close": ask_close,
                }
            )
        if epochs != sorted(set(epochs)):
            raise DojoAnomalyAdmissionError(
                f"{pair} close epochs must be unique and sorted"
            )
        if any(
            later - earlier != H1_SECONDS for earlier, later in zip(epochs, epochs[1:])
        ):
            raise DojoAnomalyAdmissionError(f"{pair} H1 history is discontinuous")
        if epochs[-1] != decision_epoch:
            raise DojoAnomalyAdmissionError(
                f"{pair} latest completed H1 close must equal decision_epoch"
            )
        if common_epochs is None:
            common_epochs = tuple(epochs)
        elif tuple(epochs) != common_epochs:
            raise DojoAnomalyAdmissionError("exact-28 H1 panel is not synchronized")
        normalized[pair] = candles
    if common_epochs is None:
        raise DojoAnomalyAdmissionError("exact-28 H1 panel is empty")
    return (
        _strict_clone(normalized, field="completed_h1_panel"),
        normalized,
        common_epochs,
    )


def _pip_size(pair: str) -> float:
    # JPY pairs conventionally quote one pip at 0.01; other G8 FX pairs quote
    # at 0.0001.  This is instrument representation, not a trading threshold.
    return 0.01 if pair.endswith("_JPY") else 0.0001


def _mid(candle: Mapping[str, Any], field: str) -> float:
    return (float(candle[f"bid_{field}"]) + float(candle[f"ask_{field}"])) / 2.0


def _sample_stdev(values: Sequence[float], *, field: str) -> float:
    if len(values) < 2:
        raise DojoAnomalyAdmissionError(f"{field} needs at least two observations")
    result = statistics.stdev(values)
    if not math.isfinite(result) or result <= 0:
        raise DojoAnomalyAdmissionError(f"{field} has zero or invalid variation")
    return result


def _correlation(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise DojoAnomalyAdmissionError("correlation inputs are misaligned")
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right)
    )
    left_energy = sum((value - left_mean) ** 2 for value in left)
    right_energy = sum((value - right_mean) ** 2 for value in right)
    denominator = math.sqrt(left_energy * right_energy)
    if denominator <= 0 or not math.isfinite(denominator):
        raise DojoAnomalyAdmissionError("correlation history has zero variation")
    result = numerator / denominator
    if not math.isfinite(result):
        raise DojoAnomalyAdmissionError("correlation is non-finite")
    return max(-1.0, min(1.0, result))


def _feature_rows(
    panel: Mapping[str, list[dict[str, Any]]],
    policy: Mapping[str, Any],
    active_features: Sequence[str],
) -> tuple[dict[str, dict[str, float]], dict[tuple[str, str], float]]:
    lookbacks = policy["lookbacks"]
    active = set(active_features)
    return_history: dict[str, list[float]] = {}
    features: dict[str, dict[str, float]] = {}
    for pair in FORMAL_G8_PAIRS:
        candles = panel[pair]
        closes = [_mid(candle, "close") for candle in candles]
        returns = [
            math.log(current / previous)
            for previous, current in zip(closes, closes[1:])
        ]
        return_history[pair] = returns
        features[pair] = {}
        long_sigma: float | None = None
        if active & {"momentum_z", "reversal_shock_z", "volatility_ratio"}:
            long_window = returns[-lookbacks["volatility_long_bars"] :]
            long_sigma = _sample_stdev(long_window, field=f"{pair}.long_volatility")
        if "momentum_z" in active:
            momentum_window = returns[-lookbacks["momentum_bars"] :]
            features[pair]["momentum_z"] = abs(sum(momentum_window)) / (
                long_sigma * math.sqrt(len(momentum_window))
            )
        if "reversal_shock_z" in active:
            prior_end = len(returns) - 1
            prior_start = prior_end - lookbacks["reversal_prior_bars"]
            prior_momentum = sum(returns[prior_start:prior_end])
            latest_return = returns[-1]
            features[pair]["reversal_shock_z"] = (
                abs(latest_return) / long_sigma
                if prior_momentum * latest_return < 0
                else 0.0
            )
        if "volatility_ratio" in active:
            short_sigma = _sample_stdev(
                returns[-lookbacks["volatility_short_bars"] :],
                field=f"{pair}.short_volatility",
            )
            features[pair]["volatility_ratio"] = short_sigma / long_sigma
        if "spread_atr_ratio" in active:
            true_ranges: list[float] = []
            for previous_close, candle in zip(closes, candles[1:]):
                high = _mid(candle, "high")
                low = _mid(candle, "low")
                true_ranges.append(
                    max(
                        high - low,
                        abs(high - previous_close),
                        abs(low - previous_close),
                    )
                )
            atr = statistics.fmean(true_ranges[-lookbacks["atr_bars"] :])
            if not math.isfinite(atr) or atr <= 0:
                raise DojoAnomalyAdmissionError(f"{pair}.atr is invalid")
            latest = candles[-1]
            spread = float(latest["ask_close"]) - float(latest["bid_close"])
            if spread < 0:
                raise DojoAnomalyAdmissionError(f"{pair}.spread is negative")
            pip_size = _pip_size(pair)
            features[pair]["spread_atr_ratio"] = (spread / pip_size) / (atr / pip_size)

    correlations: dict[tuple[str, str], float] = {}
    if "correlation_concentration" in active:
        correlation_bars = policy["lookbacks"]["correlation_bars"]
        for left_index, left_pair in enumerate(FORMAL_G8_PAIRS):
            for right_pair in FORMAL_G8_PAIRS[left_index + 1 :]:
                correlation = abs(
                    _correlation(
                        return_history[left_pair][-correlation_bars:],
                        return_history[right_pair][-correlation_bars:],
                    )
                )
                correlations[(left_pair, right_pair)] = correlation
        for pair in FORMAL_G8_PAIRS:
            values = [
                correlations[tuple(sorted((pair, other_pair)))]
                for other_pair in FORMAL_G8_PAIRS
                if other_pair != pair
            ]
            features[pair]["correlation_concentration"] = max(values)
    return features, correlations


def _band_cap(value: float, band: Mapping[str, float]) -> tuple[float, str | None]:
    if value >= band["hold_at"]:
        return 0.0, "HOLD"
    if value >= band["reduce_to_quarter_at"]:
        return 0.25, "REDUCE_TO_QUARTER"
    if value >= band["reduce_to_half_at"]:
        return 0.5, "REDUCE_TO_HALF"
    return 1.0, None


def _arm_feature_names(arm: str) -> tuple[str, ...]:
    if arm == "BASE_BOT":
        return ()
    if arm == "EXTREME_MOMENTUM_VETO":
        return ("momentum_z",)
    if arm == "REVERSAL_SHOCK_VETO":
        return ("reversal_shock_z",)
    if arm == "VOLATILITY_CORRELATION_SIZING":
        return (
            "volatility_ratio",
            "spread_atr_ratio",
            "correlation_concentration",
            "currency_gross_exposure_fraction",
        )
    if arm == "COMBINED_ANOMALY_ADMISSION":
        return _BAND_NAMES
    raise DojoAnomalyAdmissionError(
        "AI_EXIT_CAPITAL_RELEASE belongs to room-ai-01 and cannot be executed by room-meta-01"
    )


def _normalize_exposure(value: Mapping[str, Any]) -> dict[str, float]:
    row = _exact_mapping(
        value, set(FORMAL_G8_CURRENCIES), field="currency_gross_exposure_fractions"
    )
    return {
        currency: _finite(
            row[currency],
            field=f"currency_gross_exposure_fractions.{currency}",
            non_negative=True,
        )
        for currency in FORMAL_G8_CURRENCIES
    }


def _normalize_candidates(value: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    identities: set[str] = set()
    ranks: set[int] = set()
    for index, raw in enumerate(_sequence(value, field="candidates")):
        field = f"candidates[{index}]"
        row = _exact_mapping(
            raw,
            {
                "candidate_id",
                "priority_rank",
                "strategy_family",
                "pair",
                "side",
                "full_size_units",
                "currency_exposure_increment_at_full_size",
            },
            field=field,
        )
        candidate_id = _identifier(row["candidate_id"], field=f"{field}.candidate_id")
        rank = _integer(row["priority_rank"], field=f"{field}.priority_rank", minimum=1)
        pair = _identifier(row["pair"], field=f"{field}.pair")
        if pair not in FORMAL_G8_PAIRS:
            raise DojoAnomalyAdmissionError(f"{field}.pair is outside formal G8")
        units = _integer(
            row["full_size_units"], field=f"{field}.full_size_units", minimum=4
        )
        if candidate_id in identities or rank in ranks:
            raise DojoAnomalyAdmissionError(
                "candidate ids and priority ranks must be unique"
            )
        identities.add(candidate_id)
        ranks.add(rank)
        candidates.append(
            {
                "candidate_id": candidate_id,
                "priority_rank": rank,
                "strategy_family": _identifier(
                    row["strategy_family"], field=f"{field}.strategy_family"
                ),
                "pair": pair,
                "side": _side(row["side"], field=f"{field}.side"),
                "full_size_units": units,
                "currency_exposure_increment_at_full_size": _finite(
                    row["currency_exposure_increment_at_full_size"],
                    field=f"{field}.currency_exposure_increment_at_full_size",
                    positive=True,
                ),
            }
        )
    return sorted(candidates, key=lambda row: row["priority_rank"])


def _pair_correlation(
    correlations: Mapping[tuple[str, str], float], left: str, right: str
) -> float:
    if left == right:
        return 1.0
    return correlations[tuple(sorted((left, right)))]


def allocate_candidates(
    *,
    completed_h1_panel: Mapping[str, Any],
    decision_epoch: int,
    policy: Mapping[str, Any],
    arm: str,
    candidates: Sequence[Mapping[str, Any]],
    currency_gross_exposure_fractions: Mapping[str, Any],
    capacity_slots: int,
) -> dict[str, Any]:
    """Filter upstream bot candidates without changing their rank or direction."""

    epoch = _integer(decision_epoch, field="decision_epoch", minimum=1)
    normalized_policy = validate_policy(policy)
    if arm not in EXPERIMENT_ARMS or arm == "AI_EXIT_CAPITAL_RELEASE":
        _arm_feature_names(arm)
    active_features = _arm_feature_names(arm)
    slots = _integer(capacity_slots, field="capacity_slots", minimum=1)
    normalized_candidates = _normalize_candidates(candidates)
    exposure = _normalize_exposure(currency_gross_exposure_fractions)
    lookbacks = normalized_policy["lookbacks"]
    required_bars = max(lookbacks.values()) + 1
    panel_body, panel, common_epochs = _normalize_panel(
        completed_h1_panel, decision_epoch=epoch, required_bars=required_bars
    )
    features, correlations = _feature_rows(panel, normalized_policy, active_features)

    selected_pairs: list[str] = []
    rows: list[dict[str, Any]] = []
    for candidate in normalized_candidates:
        pair = candidate["pair"]
        reasons: list[str] = []
        cap = 1.0
        pair_features = dict(features[pair])
        for feature_name in active_features:
            if feature_name == "currency_gross_exposure_fraction":
                continue
            band = normalized_policy["bands"][feature_name]
            if arm in {"EXTREME_MOMENTUM_VETO", "REVERSAL_SHOCK_VETO"}:
                # The two single-factor ablations test exclusion only.  Partial
                # sizing belongs to the sizing and combined arms, so an
                # apparent gain cannot be attributed to the wrong mechanism.
                feature_cap, severity = (
                    (0.0, "HOLD")
                    if pair_features[feature_name] >= band["hold_at"]
                    else (1.0, None)
                )
            else:
                feature_cap, severity = _band_cap(pair_features[feature_name], band)
            cap = min(cap, feature_cap)
            if severity is not None:
                reasons.append(f"{feature_name.upper()}_{severity}")

        selected_correlation = (
            max(
                _pair_correlation(correlations, pair, selected)
                for selected in selected_pairs
            )
            if selected_pairs
            else 0.0
        )
        if (
            arm in {"VOLATILITY_CORRELATION_SIZING", "COMBINED_ANOMALY_ADMISSION"}
            and selected_correlation
            >= normalized_policy["selected_pair_abs_correlation_hold_at"]
        ):
            cap = 0.0
            reasons.append("SELECTED_PAIR_CORRELATION_HOLD")

        base_currency, quote_currency = pair.split("_")
        chosen_multiplier = 0.0
        if len(selected_pairs) >= slots:
            reasons.append("CAPACITY_FULL_HOLD")
        elif cap > 0:
            for multiplier in _POSITIVE_MULTIPLIERS_DESC:
                if multiplier > cap:
                    continue
                if "currency_gross_exposure_fraction" in active_features:
                    projected_peak = max(
                        exposure[base_currency]
                        + candidate["currency_exposure_increment_at_full_size"]
                        * multiplier,
                        exposure[quote_currency]
                        + candidate["currency_exposure_increment_at_full_size"]
                        * multiplier,
                    )
                    exposure_cap, severity = _band_cap(
                        projected_peak,
                        normalized_policy["bands"]["currency_gross_exposure_fraction"],
                    )
                    if severity is not None:
                        reasons.append(f"CURRENCY_GROSS_EXPOSURE_FRACTION_{severity}")
                    if multiplier > exposure_cap:
                        continue
                chosen_multiplier = multiplier
                break

        if chosen_multiplier == 0.0:
            decision = "HOLD"
            status = "NOT_SELECTED"
            selected_slot = None
            allocated_units = 0
            if not reasons:
                reasons.append("ANOMALY_POLICY_HOLD")
        else:
            decision = "ENTER_OK" if chosen_multiplier == 1.0 else "REDUCE_SIZE"
            status = "SELECTED"
            selected_pairs.append(pair)
            selected_slot = len(selected_pairs)
            allocated_units = math.floor(
                candidate["full_size_units"] * chosen_multiplier
            )
            increment = (
                candidate["currency_exposure_increment_at_full_size"]
                * chosen_multiplier
            )
            exposure[base_currency] += increment
            exposure[quote_currency] += increment

        rows.append(
            {
                **candidate,
                "upstream_side_preserved": candidate["side"],
                "upstream_priority_rank_preserved": candidate["priority_rank"],
                "status": status,
                "admission_decision": decision,
                "size_multiplier": chosen_multiplier,
                "allocated_units": allocated_units,
                "selected_slot": selected_slot,
                "anomaly_features": pair_features,
                "max_abs_correlation_to_selected_before_decision": selected_correlation,
                "reason_codes": sorted(set(reasons)),
            }
        )

    body = {
        "contract": CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "room_id": ROOM_ID,
        "arm": arm,
        "decision_epoch": epoch,
        "latest_completed_h1_close_epoch": common_epochs[-1],
        "formal_pair_count": len(FORMAL_G8_PAIRS),
        "completed_h1_panel_sha256": canonical_sha256(panel_body),
        "policy_sha256": normalized_policy["policy_sha256"],
        "candidate_set_sha256": canonical_sha256(normalized_candidates),
        "capacity_slots": slots,
        "selected_count": len(selected_pairs),
        "selected_candidate_ids": [
            row["candidate_id"] for row in rows if row["status"] == "SELECTED"
        ],
        "candidate_decisions": rows,
        "ending_currency_gross_exposure_fractions": exposure,
        "capital_recycle_policy": {
            "hold_consumes_capacity_slot": False,
            "next_upstream_rank_considered_after_hold": True,
            "selected_pair_correlation_gate_enabled": arm
            in {"VOLATILITY_CORRELATION_SIZING", "COMBINED_ANOMALY_ADMISSION"},
        },
        "direction_predictions_emitted": False,
        "evidence_class": "SELF_ATTESTED_UNVERIFIED_DIAGNOSTIC",
        "runner_integration_complete": False,
        "source_binding_verified": False,
        "authority": _strict_clone(_AUTHORITY, field="authority"),
    }
    return {**body, "allocation_sha256": canonical_sha256(body)}


def validate_allocation(
    value: Mapping[str, Any],
    *,
    completed_h1_panel: Mapping[str, Any],
    policy: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    currency_gross_exposure_fractions: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute an allocation from source inputs and reject output tampering."""

    row = _strict_clone(value, field="allocation")
    if not isinstance(row, Mapping):
        raise DojoAnomalyAdmissionError("allocation must be an object")
    expected = allocate_candidates(
        completed_h1_panel=completed_h1_panel,
        decision_epoch=row.get("decision_epoch"),
        policy=policy,
        arm=row.get("arm"),
        candidates=candidates,
        currency_gross_exposure_fractions=currency_gross_exposure_fractions,
        capacity_slots=row.get("capacity_slots"),
    )
    if row != expected:
        raise DojoAnomalyAdmissionError(
            "allocation differs from canonical recomputation"
        )
    return dict(row)


def build_train_plan(
    *,
    plan_id: str,
    fixed_denominator_sha256: str,
    source_binding_sha256: str,
    evaluator_binding_sha256: str,
    cost_policy_sha256: str,
    risk_policy_sha256: str,
    policy_sha256: str,
) -> dict[str, Any]:
    """Preregister requested ablations and economic metrics without running them."""

    body = {
        "contract": TRAIN_PLAN_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "room_id": ROOM_ID,
        "plan_id": _identifier(plan_id, field="plan_id"),
        "status": "PREREGISTERED_NOT_EXECUTED",
        "evidence_class": "WORN_HISTORICAL_TRAIN_ONLY",
        "bindings": {
            "fixed_denominator_sha256": _sha256(
                fixed_denominator_sha256, field="fixed_denominator_sha256"
            ),
            "source_binding_sha256": _sha256(
                source_binding_sha256, field="source_binding_sha256"
            ),
            "evaluator_binding_sha256": _sha256(
                evaluator_binding_sha256, field="evaluator_binding_sha256"
            ),
            "cost_policy_sha256": _sha256(
                cost_policy_sha256, field="cost_policy_sha256"
            ),
            "risk_policy_sha256": _sha256(
                risk_policy_sha256, field="risk_policy_sha256"
            ),
            "policy_sha256": _sha256(policy_sha256, field="policy_sha256"),
        },
        "arms": [
            {"arm": "BASE_BOT", "owner_room": "UPSTREAM_BOT_ROOM"},
            {
                "arm": "EXTREME_MOMENTUM_VETO",
                "owner_room": ROOM_ID,
            },
            {"arm": "REVERSAL_SHOCK_VETO", "owner_room": ROOM_ID},
            {
                "arm": "VOLATILITY_CORRELATION_SIZING",
                "owner_room": ROOM_ID,
            },
            {"arm": "COMBINED_ANOMALY_ADMISSION", "owner_room": ROOM_ID},
            {
                "arm": "AI_EXIT_CAPITAL_RELEASE",
                "owner_room": "room-ai-01",
                "decision_context_policy": (
                    "ONE_DECISION_ONE_FRESH_CONTEXT_NO_CROSS_DECISION_HISTORY"
                ),
            },
        ],
        "required_metrics": [
            "net_profit_after_all_costs_jpy",
            "maximum_mark_to_market_drawdown_fraction",
            "cvar_95_loss_jpy",
            "ruin_probability",
            "margin_closeout_count",
            "missed_opportunity_after_cost_jpy",
            "capital_utilization_fraction",
            "financing_jpy",
            "slippage_jpy",
            "spread_cost_jpy",
            "trade_count",
        ],
        "evaluation_requirements": {
            "exact_28_pair_completed_h1_panel_required": True,
            "both_intrabar_paths_required": True,
            "base_and_stress_cost_arms_required": True,
            "continuous_mark_to_market_required": True,
            "margin_closeout_reconstruction_required": True,
            "held_candidate_counterfactual_required": True,
            "zero_trade_is_failure": True,
            "ai_answer_key_absent_until_response_sealed": True,
        },
        "separation": {
            "room_03_generates_relative_strength_alpha": True,
            "room_meta_01_generates_direction": False,
            "room_meta_01_changes_upstream_direction": False,
            "room_ai_01_is_separate": True,
        },
        "authority": _strict_clone(_AUTHORITY, field="authority"),
    }
    return {**body, "plan_sha256": canonical_sha256(body)}


def validate_train_plan(value: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild a preregistered plan and reject arm, metric, or binding drift."""

    row = _strict_clone(value, field="train_plan")
    if not isinstance(row, Mapping):
        raise DojoAnomalyAdmissionError("train_plan must be an object")
    bindings = row.get("bindings")
    if not isinstance(bindings, Mapping):
        raise DojoAnomalyAdmissionError("train_plan.bindings must be an object")
    expected = build_train_plan(
        plan_id=row.get("plan_id"),
        fixed_denominator_sha256=bindings.get("fixed_denominator_sha256"),
        source_binding_sha256=bindings.get("source_binding_sha256"),
        evaluator_binding_sha256=bindings.get("evaluator_binding_sha256"),
        cost_policy_sha256=bindings.get("cost_policy_sha256"),
        risk_policy_sha256=bindings.get("risk_policy_sha256"),
        policy_sha256=bindings.get("policy_sha256"),
    )
    if row != expected:
        raise DojoAnomalyAdmissionError("train_plan differs from canonical V1")
    return dict(row)
