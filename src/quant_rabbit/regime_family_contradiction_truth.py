"""Read-only OANDA truth acquisition for contradiction shadow trials.

The statistical resolver in
``strategy.regime_family_contradiction_shadow`` intentionally has no network
access.  This adapter is the sole runtime bridge: it derives one frozen M1
window from each due trial before reading prices, performs only an instrument
candle ``GET``, strictly validates the bid/ask packet, and hands the resulting
local candles back to the pure resolver.

Failures are diagnostic.  They leave the trial pending and can never affect a
forecast, sizing, a risk gate, or broker-write permission.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from quant_rabbit.broker.oanda import (
    DEFAULT_OANDA_HTTP_TIMEOUT_SECONDS,
    OandaReadOnlyClient,
)
from quant_rabbit.instruments import instrument_pip_factor
from quant_rabbit.strategy.regime_family_contradiction_shadow import (
    M1BidAskCandle,
    load_regime_family_contradiction_ledger,
    persist_regime_family_contradiction_results,
    resolve_due_regime_family_contradiction_trials,
    select_independent_regime_family_contradiction_trials,
)


TRUTH_ADAPTER_CONTRACT = "QR_REGIME_FAMILY_CONTRADICTION_OANDA_M1_BA_V1"
PRODUCTION_OANDA_BASE_URL = "https://api-fxtrade.oanda.com"
# These are infrastructure bounds, not trading or statistical parameters.
DEFAULT_TRUTH_BUDGET_SECONDS = 45.0
DEFAULT_TRUTH_MAX_WORKERS = 6
MAX_DIAGNOSTIC_ERRORS = 20
MAX_ERROR_CHARS = 240

_OANDA_UTC_RE = re.compile(
    r"^(?P<seconds>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})"
    r"(?:\.(?P<fraction>\d{1,9}))?Z$"
)


class _ReadOnlyCandleClient(Protocol):
    base_url: str
    http_timeout_seconds: float

    def get_json(
        self,
        path: str,
        query: dict[str, str] | None = None,
    ) -> dict[str, Any]: ...


@dataclass(frozen=True, order=True)
class _FetchTask:
    interval_start_epoch: int
    pair: str
    interval_end_epoch: int
    trial_ids: tuple[str, ...]

    @property
    def interval_start_utc(self) -> str:
        return _epoch_utc(self.interval_start_epoch)

    @property
    def interval_end_utc(self) -> str:
        return _epoch_utc(self.interval_end_epoch)


def resolve_due_regime_family_contradiction_from_oanda(
    *,
    data_root: Path,
    client_factory: Callable[[], _ReadOnlyCandleClient] = OandaReadOnlyClient,
    clock: Callable[[], datetime] | None = None,
    monotonic_clock: Callable[[], float] = time.monotonic,
    budget_seconds: float = DEFAULT_TRUTH_BUDGET_SECONDS,
    max_workers: int = DEFAULT_TRUTH_MAX_WORKERS,
    non_spread_cost_pips: float = 0.0,
) -> dict[str, Any]:
    """Resolve due shadow rows from exact, complete OANDA M1 bid/ask truth.

    ``clock`` is sampled exactly once.  ``client_factory`` is not called until
    a valid ledger contains at least one due trial whose frozen M1 interval is
    complete as of that sample.
    """

    started = monotonic_clock()
    now_source = clock or _utc_now
    try:
        as_of = _aware_utc(now_source())
        as_of_utc = _canonical_datetime(as_of)
        as_of_epoch_ns = _datetime_epoch_ns(as_of)
        budget = _positive_finite(budget_seconds, "budget_seconds")
        workers = _exact_bounded_int(
            max_workers,
            "max_workers",
            minimum=1,
            maximum=DEFAULT_TRUTH_MAX_WORKERS,
        )
        extra_cost = _nonnegative_finite(
            non_spread_cost_pips,
            "non_spread_cost_pips",
        )
    except (TypeError, ValueError, OverflowError) as exc:
        return _base_result(
            status="CLIENT_UNAVAILABLE",
            as_of_utc=None,
            non_spread_cost_pips=0.0,
            errors=[_error("ADAPTER_CONFIGURATION", exc)],
        )

    base = _base_result(
        status="NO_LEDGER",
        as_of_utc=as_of_utc,
        non_spread_cost_pips=extra_cost,
    )
    try:
        loaded = load_regime_family_contradiction_ledger(Path(data_root))
    except Exception as exc:
        return {
            **base,
            "status": "LEDGER_INVALID",
            "errors": [_error("LEDGER_LOAD", exc)],
        }

    if loaded.get("status") == "MISSING":
        return base
    if loaded.get("status") != "VALID":
        return {
            **base,
            "status": "LEDGER_INVALID",
            "errors": [
                _error_text(
                    "LEDGER_LOAD",
                    f"unexpected ledger status {loaded.get('status')!r}",
                )
            ],
        }

    trials = loaded.get("trials")
    results = loaded.get("results")
    recorded_at = loaded.get("ledger_recorded_at_by_trial_id")
    if (
        not isinstance(trials, list)
        or not isinstance(results, list)
        or not isinstance(recorded_at, Mapping)
    ):
        return {
            **base,
            "status": "LEDGER_INVALID",
            "errors": [_error_text("LEDGER_LOAD", "validated ledger shape invalid")],
        }

    base.update(
        {
            "ledger_trial_count": len(trials),
            "already_resolved_count": len(results),
        }
    )
    try:
        selection = select_independent_regime_family_contradiction_trials(
            trials,
            as_of_utc=as_of_utc,
            require_locked_holdout=False,
            ledger_recorded_at_by_trial_id=recorded_at,
        )
    except Exception as exc:
        return {
            **base,
            "status": "LEDGER_INVALID",
            "errors": [_error("LEDGER_SELECTION", exc)],
        }
    if (
        selection.get("invalid_trials")
        or selection.get("recording_anchor_missing_count")
        or selection.get("recording_anchor_invalid_count")
        or selection.get("recording_anchor_after_due_count")
    ):
        return {
            **base,
            "status": "LEDGER_INVALID",
            "errors": [
                _error_text(
                    "LEDGER_SELECTION",
                    "trial or recording-anchor validation failed",
                )
            ],
        }

    selected = selection.get("selected_trials")
    if not isinstance(selected, list):
        return {
            **base,
            "status": "LEDGER_INVALID",
            "errors": [_error_text("LEDGER_SELECTION", "selected trials missing")],
        }
    base["selected_trial_count"] = len(selected)
    resolved_ids = {
        str(result.get("trial_id"))
        for result in results
        if isinstance(result, Mapping) and result.get("trial_id")
    }

    due_trials: list[Mapping[str, Any]] = []
    waiting_ids: list[str] = []
    grouped: dict[tuple[str, int, int], list[str]] = {}
    trial_by_id: dict[str, Mapping[str, Any]] = {}
    try:
        for trial in selected:
            if not isinstance(trial, Mapping):
                raise ValueError("selected trial is not a mapping")
            trial_id = str(trial.get("trial_id") or "")
            if not trial_id or trial_id in resolved_ids:
                continue
            due_epoch, due_ns = _strict_utc_epoch_ns(
                trial.get("evaluation_due_at_utc"),
                "evaluation_due_at_utc",
            )
            if (due_epoch, due_ns) > as_of_epoch_ns:
                continue
            pair = trial.get("pair")
            if pair.__class__ is not str:
                raise ValueError("trial pair invalid")
            start_epoch, end_epoch = _frozen_m1_window(due_epoch, due_ns)
            due_trials.append(trial)
            trial_by_id[trial_id] = trial
            if (end_epoch, 0) > as_of_epoch_ns:
                waiting_ids.append(trial_id)
                continue
            grouped.setdefault((pair, start_epoch, end_epoch), []).append(trial_id)
    except (TypeError, ValueError, OverflowError) as exc:
        return {
            **base,
            "status": "LEDGER_INVALID",
            "errors": [_error("FROZEN_WINDOW", exc)],
        }

    base.update(
        {
            "due_trial_count": len(due_trials),
            "due_pair_count": len(
                {str(trial.get("pair")) for trial in due_trials}
            ),
            "waiting_for_complete_m1_count": len(waiting_ids),
        }
    )
    if not due_trials:
        return {**base, "status": "NO_DUE"}
    if not grouped:
        return {
            **base,
            "status": "WAITING_FOR_COMPLETE_M1",
            "pending_due_without_truth": sorted(waiting_ids),
        }

    tasks = sorted(
        _FetchTask(
            interval_start_epoch=start,
            pair=pair,
            interval_end_epoch=end,
            trial_ids=tuple(sorted(trial_ids)),
        )
        for (pair, start, end), trial_ids in grouped.items()
    )
    try:
        client = client_factory()
    except Exception as exc:
        return {
            **base,
            "status": "CLIENT_UNAVAILABLE",
            "pending_due_without_truth": sorted(trial_by_id),
            "errors": [_error("CLIENT_CONSTRUCTION", exc)],
        }
    base_url = getattr(client, "base_url", None)
    if base_url != PRODUCTION_OANDA_BASE_URL:
        return {
            **base,
            "status": "CLIENT_UNAVAILABLE",
            "oanda_base_url": (
                base_url[:128] if base_url.__class__ is str else None
            ),
            "pending_due_without_truth": sorted(trial_by_id),
            "errors": [
                _error_text(
                    "CLIENT_CONFIGURATION",
                    "shadow truth requires the production OANDA read endpoint",
                )
            ],
        }
    if not callable(getattr(client, "get_json", None)):
        return {
            **base,
            "status": "CLIENT_UNAVAILABLE",
            "oanda_base_url": base_url,
            "pending_due_without_truth": sorted(trial_by_id),
            "errors": [_error_text("CLIENT_CONFIGURATION", "get_json unavailable")],
        }
    base["oanda_base_url"] = base_url

    timeout_bound = getattr(
        client,
        "http_timeout_seconds",
        DEFAULT_OANDA_HTTP_TIMEOUT_SECONDS,
    )
    try:
        timeout_bound = _positive_finite(timeout_bound, "http_timeout_seconds")
    except (TypeError, ValueError, OverflowError):
        timeout_bound = DEFAULT_OANDA_HTTP_TIMEOUT_SECONDS

    candles: list[M1BidAskCandle] = []
    errors: list[dict[str, Any]] = []
    deferred: list[_FetchTask] = []
    attempted = 0
    cursor = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        while cursor < len(tasks):
            remaining = budget - (monotonic_clock() - started)
            # Never start a new HTTP wave that cannot fit the configured
            # per-request timeout bound. Already-running calls remain bounded
            # by the read-only client's own timeout.
            if remaining < min(timeout_bound, budget):
                deferred.extend(tasks[cursor:])
                break
            wave = tasks[cursor : cursor + workers]
            cursor += len(wave)
            attempted += len(wave)
            futures = {
                task: executor.submit(_fetch_exact_m1_bid_ask, client, task)
                for task in wave
            }
            # Futures are read in frozen task order so diagnostics never depend
            # on response race order.
            for task in wave:
                try:
                    candle = futures[task].result()
                except Exception as exc:
                    errors.append(
                        _error(
                            "M1_BID_ASK_FETCH",
                            exc,
                            pair=task.pair,
                            interval_start_utc=task.interval_start_utc,
                        )
                    )
                else:
                    if candle is not None:
                        candles.append(candle)

    deferred_ids = sorted(
        trial_id for task in deferred for trial_id in task.trial_ids
    )
    base.update(
        {
            "fetch_request_count": attempted,
            "fetched_candle_count": len(candles),
            "deferred_due_count": len(deferred_ids),
        }
    )
    try:
        resolution = resolve_due_regime_family_contradiction_trials(
            due_trials,
            candles,
            as_of_utc=as_of_utc,
            non_spread_cost_pips=extra_cost,
        )
    except Exception as exc:
        return {
            **base,
            "status": "LEDGER_INVALID",
            "pending_due_without_truth": sorted(trial_by_id),
            "errors": _bounded_errors(
                errors + [_error("LOCAL_RESOLUTION", exc)]
            ),
        }
    if resolution.get("status") != "OK":
        return {
            **base,
            "status": "LEDGER_INVALID",
            "pending_due_without_truth": sorted(trial_by_id),
            "errors": _bounded_errors(
                errors
                + [
                    _error_text(
                        "LOCAL_RESOLUTION",
                        f"unexpected status {resolution.get('status')!r}",
                    )
                ]
            ),
        }

    resolved_results = resolution.get("resolved_results")
    if not isinstance(resolved_results, list):
        return {
            **base,
            "status": "LEDGER_INVALID",
            "pending_due_without_truth": sorted(trial_by_id),
            "errors": _bounded_errors(
                errors
                + [_error_text("LOCAL_RESOLUTION", "resolved results missing")]
            ),
        }
    persisted_count = 0
    persistence_failed = False
    persistence_pending_ids: set[str] = set()
    for result in sorted(
        resolved_results,
        key=lambda item: str(item.get("trial_id"))
        if isinstance(item, Mapping)
        else "",
    ):
        try:
            persisted_count += persist_regime_family_contradiction_results(
                [result],
                data_root=Path(data_root),
            )
        except Exception as exc:
            persistence_failed = True
            failed_trial_id = (
                str(result.get("trial_id"))
                if isinstance(result, Mapping)
                else ""
            )
            if failed_trial_id:
                persistence_pending_ids.add(failed_trial_id)
            errors.append(
                _error(
                    "RESULT_PERSISTENCE",
                    exc,
                    trial_id=failed_trial_id or None,
                )
            )

    if persistence_failed:
        try:
            reloaded = load_regime_family_contradiction_ledger(Path(data_root))
            persisted_ids = {
                str(item.get("trial_id"))
                for item in reloaded.get("results", [])
                if isinstance(item, Mapping)
            }
            attempted_result_ids = {
                str(item.get("trial_id"))
                for item in resolved_results
                if isinstance(item, Mapping)
            }
            persisted_count = len(attempted_result_ids & persisted_ids)
            persistence_pending_ids -= persisted_ids
        except Exception as exc:
            return {
                **base,
                "status": "LEDGER_INVALID",
                "resolved_count": len(resolved_results),
                "persisted_count": persisted_count,
                "pending_due_without_truth": sorted(trial_by_id),
                "errors": _bounded_errors(
                    errors + [_error("LEDGER_RELOAD_AFTER_PERSISTENCE", exc)]
                ),
            }

    pending = sorted(
        set(resolution.get("pending_due_without_truth") or [])
        | set(waiting_ids)
        | set(deferred_ids)
        | persistence_pending_ids
    )
    degraded = bool(errors or pending or persistence_failed)
    return {
        **base,
        "status": "PARTIAL" if degraded else "OK",
        "resolved_count": len(resolved_results),
        "persisted_count": persisted_count,
        "pending_due_without_truth": pending,
        "errors": _bounded_errors(errors),
    }


def _fetch_exact_m1_bid_ask(
    client: _ReadOnlyCandleClient,
    task: _FetchTask,
) -> M1BidAskCandle | None:
    path = f"/v3/instruments/{task.pair}/candles"
    query = {
        "from": task.interval_start_utc,
        "to": task.interval_end_utc,
        "granularity": "M1",
        "price": "BA",
        "includeFirst": "true",
        "smooth": "false",
    }
    payload = client.get_json(path, query)
    return _parse_exact_m1_bid_ask_payload(
        payload,
        task=task,
        base_url=client.base_url,
        path=path,
        query=query,
    )


def _parse_exact_m1_bid_ask_payload(
    payload: object,
    *,
    task: _FetchTask,
    base_url: str,
    path: str,
    query: Mapping[str, str],
) -> M1BidAskCandle | None:
    if payload.__class__ is not dict:
        raise ValueError("OANDA_M1_BA_PAYLOAD_NOT_OBJECT")
    if payload.get("instrument") != task.pair:
        raise ValueError("OANDA_M1_BA_INSTRUMENT_MISMATCH")
    if payload.get("granularity") != "M1":
        raise ValueError("OANDA_M1_BA_GRANULARITY_MISMATCH")
    entries = payload.get("candles")
    if entries.__class__ is not list:
        raise ValueError("OANDA_M1_BA_CANDLES_NOT_ARRAY")
    if not entries:
        return None

    decimal_places = 3 if instrument_pip_factor(task.pair) == 100 else 5
    accepted: tuple[
        tuple[tuple[str, float], ...],
        tuple[tuple[str, float], ...],
        int,
    ] | None = None
    bid_close = 0.0
    ask_close = 0.0
    for entry in entries:
        if entry.__class__ is not dict:
            raise ValueError("OANDA_M1_BA_CANDLE_NOT_OBJECT")
        opened_epoch, opened_ns = _strict_utc_epoch_ns(
            entry.get("time"),
            "candle.time",
        )
        if opened_epoch != task.interval_start_epoch or opened_ns != 0:
            raise ValueError("OANDA_M1_BA_CANDLE_OUTSIDE_FROZEN_WINDOW")
        if entry.get("complete") is not True:
            raise ValueError("OANDA_M1_BA_CANDLE_INCOMPLETE")
        bid = _strict_ohlc(entry.get("bid"), decimal_places=decimal_places)
        ask = _strict_ohlc(entry.get("ask"), decimal_places=decimal_places)
        if any(bid[key] > ask[key] for key in ("o", "h", "l", "c")):
            raise ValueError("OANDA_M1_BA_BID_ASK_ENVELOPE_INVALID")
        if not bid["c"] < ask["c"]:
            raise ValueError("OANDA_M1_BA_TERMINAL_SPREAD_INVALID")
        volume = entry.get("volume")
        if volume.__class__ is not int or volume < 0:
            raise ValueError("OANDA_M1_BA_VOLUME_INVALID")
        signature = (tuple(sorted(bid.items())), tuple(sorted(ask.items())), volume)
        if accepted is not None and accepted != signature:
            raise ValueError("OANDA_M1_BA_DUPLICATE_INTERVAL_CONFLICT")
        accepted = signature
        bid_close = bid["c"]
        ask_close = ask["c"]

    source_material = {
        "contract": TRUTH_ADAPTER_CONTRACT,
        "base_url": base_url,
        "path": path,
        "query": dict(sorted(query.items())),
        "response": payload,
    }
    try:
        source_bytes = json.dumps(
            source_material,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("OANDA_M1_BA_SOURCE_NOT_CANONICAL_JSON") from exc
    return M1BidAskCandle(
        pair=task.pair,
        timestamp_utc=task.interval_start_utc,
        bid_close=bid_close,
        ask_close=ask_close,
        complete=True,
        source_sha256=hashlib.sha256(source_bytes).hexdigest(),
    )


def _strict_ohlc(value: object, *, decimal_places: int) -> dict[str, float]:
    if value.__class__ is not dict:
        raise ValueError("OANDA_M1_BA_OHLC_NOT_OBJECT")
    pattern = re.compile(rf"(?:0|[1-9][0-9]*)\.[0-9]{{{decimal_places}}}")
    parsed: dict[str, float] = {}
    for key in ("o", "h", "l", "c"):
        raw = value.get(key)
        if (
            raw.__class__ is not str
            or raw != raw.strip()
            or pattern.fullmatch(raw) is None
        ):
            raise ValueError("OANDA_M1_BA_OHLC_DECIMAL_INVALID")
        number = float(raw)
        if not math.isfinite(number) or number <= 0.0:
            raise ValueError("OANDA_M1_BA_OHLC_NUMBER_INVALID")
        parsed[key] = number
    if not (
        parsed["l"]
        <= min(parsed["o"], parsed["c"])
        <= max(parsed["o"], parsed["c"])
        <= parsed["h"]
    ):
        raise ValueError("OANDA_M1_BA_OHLC_GEOMETRY_INVALID")
    return parsed


def _frozen_m1_window(due_epoch: int, due_ns: int) -> tuple[int, int]:
    if due_epoch % 60 == 0 and due_ns == 0:
        start = due_epoch - 60
    else:
        start = due_epoch - (due_epoch % 60)
    return start, start + 60


def _strict_utc_epoch_ns(value: object, label: str) -> tuple[int, int]:
    if value.__class__ is not str:
        raise ValueError(f"{label} invalid")
    match = _OANDA_UTC_RE.fullmatch(value)
    if match is None:
        raise ValueError(f"{label} invalid")
    try:
        parsed = datetime.strptime(
            match.group("seconds"),
            "%Y-%m-%dT%H:%M:%S",
        ).replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise ValueError(f"{label} invalid") from exc
    fraction = match.group("fraction") or ""
    nanosecond = int(fraction.ljust(9, "0")) if fraction else 0
    return int(parsed.timestamp()), nanosecond


def _datetime_epoch_ns(value: datetime) -> tuple[int, int]:
    seconds = int(value.timestamp())
    return seconds, value.microsecond * 1_000


def _aware_utc(value: datetime) -> datetime:
    if value.__class__ is not datetime or value.tzinfo is None:
        raise ValueError("clock must return an aware datetime")
    offset = value.utcoffset()
    if offset is None:
        raise ValueError("clock must return an aware datetime")
    return value.astimezone(timezone.utc)


def _canonical_datetime(value: datetime) -> str:
    text = value.astimezone(timezone.utc).isoformat(
        timespec="microseconds"
    ).replace("+00:00", "Z")
    if "." not in text:
        return text
    head, fraction = text[:-1].split(".", 1)
    fraction = fraction.rstrip("0")
    return f"{head}.{fraction}Z" if fraction else f"{head}Z"


def _epoch_utc(value: int) -> str:
    return datetime.fromtimestamp(value, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _positive_finite(value: object, label: str) -> float:
    if value.__class__ not in {int, float}:
        raise ValueError(f"{label} must be a positive finite number")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{label} must be a positive finite number")
    return parsed


def _nonnegative_finite(value: object, label: str) -> float:
    if value.__class__ not in {int, float}:
        raise ValueError(f"{label} must be a non-negative finite number")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{label} must be a non-negative finite number")
    return parsed


def _exact_bounded_int(
    value: object,
    label: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    if value.__class__ is not int or not minimum <= value <= maximum:
        raise ValueError(f"{label} must be an exact integer in [{minimum}, {maximum}]")
    return value


def _base_result(
    *,
    status: str,
    as_of_utc: str | None,
    non_spread_cost_pips: float,
    errors: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    return {
        "contract": TRUTH_ADAPTER_CONTRACT,
        "status": status,
        "as_of_utc": as_of_utc,
        "ledger_trial_count": 0,
        "selected_trial_count": 0,
        "already_resolved_count": 0,
        "due_trial_count": 0,
        "due_pair_count": 0,
        "fetch_request_count": 0,
        "fetched_candle_count": 0,
        "waiting_for_complete_m1_count": 0,
        "deferred_due_count": 0,
        "resolved_count": 0,
        "persisted_count": 0,
        "pending_due_without_truth": [],
        "non_spread_cost_pips": non_spread_cost_pips,
        "non_spread_cost_scope": (
            "CALLER_DECLARED_ONLY_NOT_AUTHENTICATED_SLIPPAGE_OR_FINANCING"
        ),
        "oanda_base_url": None,
        "errors": _bounded_errors(errors),
        "read_only": True,
        "broker_write_attempted": False,
        "live_side_effects": [],
        "proof_eligible": False,
        "source_artifact_authenticated_by_evaluator": False,
        "automatic_promotion_allowed": False,
        "live_permission": False,
        "sizing_permission": False,
        "gate_relaxation_allowed": False,
    }


def _error(
    phase: str,
    exc: BaseException,
    **context: Any,
) -> dict[str, Any]:
    return _error_text(
        phase,
        f"{type(exc).__name__}: {str(exc)[:MAX_ERROR_CHARS]}",
        **context,
    )


def _error_text(phase: str, message: str, **context: Any) -> dict[str, Any]:
    return {
        "phase": phase,
        **{key: value for key, value in context.items() if value is not None},
        "error": str(message)[:MAX_ERROR_CHARS],
    }


def _bounded_errors(
    errors: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [dict(item) for item in errors[:MAX_DIAGNOSTIC_ERRORS]]
